import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Keras 3 -> TensorFlow backend

import io
import base64
import traceback
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_file, render_template_string

import keras
from keras.utils import img_to_array
import tensorflow as tf

# ---------- CONFIG ----------
MODEL_PATH = os.environ.get("MODEL_PATH", "dog_disease_detector.h5")
IMG_SIZE   = (224, 224)    # must match training
RESCALE    = 1.0   # must match training
CLASS_NAMES = [
    "Bacterial_dermatosis",
    "Dermatitis",
    "Fungal_infections",
    "Healthy",
    "Hypersensitivity_allergic_dermatosis",
    "demodicosis",
    "ringworm",
]
TOP_K = 3
LAST_CONV_LAYER_NAME = os.environ.get("LAST_CONV_LAYER_NAME", "top_conv")  # EfficientNetB0 default
# ----------------------------

app = Flask(__name__)
model = keras.models.load_model(MODEL_PATH, compile=False)

# ---- WARM UP the model so all nested graphs are "called" ----
_dummy = tf.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32)
_ = model(_dummy, training=False)

# ---------- Helpers ----------
def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    x = img_to_array(img) * RESCALE
    x = np.expand_dims(x, axis=0)
    return img, x

def overlay_heatmap_on_pil(base_img_pil, heatmap, alpha=0.45):
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize(base_img_pil.size, resample=Image.BILINEAR)
    # Simple colorize: R=heatmap, G=0, B=(255-heatmap)//2
    heat_rgb = Image.merge(
        "RGB",
        (heatmap, Image.new("L", heatmap.size, 0), Image.eval(heatmap, lambda p: (255 - p)//2))
    )
    return Image.blend(base_img_pil.convert("RGB"), heat_rgb, alpha=alpha)

def iter_sublayers(layer):
    """Yield layer and all its nested sublayers (depth-first)."""
    yield layer
    if hasattr(layer, "layers") and isinstance(layer, keras.Model):
        for l in layer.layers:
            yield from iter_sublayers(l)

def is_leaf(layer):
    """True for non-container layers (not Model/Sequential)."""
    return not (hasattr(layer, "layers") and isinstance(layer, keras.Model))

def is_4d_output(layer):
    """True if layer has a 4D output (N,H,W,C)."""
    try:
        out_tensor = getattr(layer, "output", None)
        shp = None if out_tensor is None else out_tensor.shape
        return shp is not None and len(shp) == 4
    except Exception:
        return False

def get_layer_by_name_recursive(m, name):
    for l in iter_sublayers(m):
        if l.name == name:
            return l
    raise ValueError(f"Layer '{name}' not found (recursive).")

def get_last_conv_layer(m):
    """
    Choose the deepest **leaf** layer with a 4D output.
    Prefer names containing 'conv'. If env LAST_CONV_LAYER_NAME matches a
    valid 4D leaf layer, use that.
    """
    # Try explicit env name first
    try:
        cand = get_layer_by_name_recursive(m, LAST_CONV_LAYER_NAME)
        if is_leaf(cand) and is_4d_output(cand):
            return cand
    except Exception:
        pass

    leaves_4d = [l for l in iter_sublayers(m) if is_leaf(l) and is_4d_output(l)]
    if not leaves_4d:
        raise ValueError("No 4D leaf layers found for Grad-CAM.")
    conv_like = [l for l in leaves_4d if "conv" in l.name.lower()]
    return conv_like[-1] if conv_like else leaves_4d[-1]

def make_gradcam_heatmap(img_array, model, pred_index=None):
    # tensorize input
    x = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # locate EfficientNet base and last conv inside it
    try:
        base = model.get_layer("efficientnetb0")
    except Exception:
        base = None
        for l in model.layers:
            if isinstance(l, keras.Model) and "efficientnet" in l.name.lower():
                base = l
                break
        if base is None:
            raise ValueError("EfficientNet base not found in the model.")
    last_conv = base.get_layer(LAST_CONV_LAYER_NAME)
    print("Grad-CAM layer ->", last_conv.name)

    # ---- REBUILD HEAD: apply all layers *after* the base to base.output ----
    try:
        base_idx = [i for i, l in enumerate(model.layers) if l is base][0]
    except IndexError:
        raise RuntimeError("Could not find the EfficientNet base position in model.layers")

    y = base.output
    for l in model.layers[base_idx + 1:]:
        y = l(y)  # build the head by calling subsequent layers

    # One graph that gives conv features + final preds from base.input
    conv_and_pred = keras.Model(inputs=base.input,
                                outputs=[last_conv.output, y])

    # Forward + gradients in one tape
    with tf.GradientTape() as tape:
        conv_outputs, preds = conv_and_pred(x, training=False)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradients are None — check conv_and_pred wiring.")

    # GAP over H,W then weight channels
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    # normalize [0,1]
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-12)
    return heatmap.numpy()
# ----------------------------

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": os.path.basename(MODEL_PATH)})

@app.get("/_layers")
def list_layers():
    """List valid Grad-CAM target layers (4D leaf layers)."""
    leaves = [l for l in iter_sublayers(model) if is_leaf(l) and is_4d_output(l)]
    return jsonify([{"name": l.name,
                     "type": l.__class__.__name__,
                     "shape": str(getattr(getattr(l, "output", None), "shape", None))}
                    for l in leaves])

@app.post("/predict")
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part; send multipart/form-data with 'file'"}), 400
        f = request.files["file"]
        if not f.filename:
            return jsonify({"error": "Empty filename"}), 400

        _, img_batch = preprocess_image(f.read())
        probs = model.predict(img_batch, verbose=0)[0].astype(float)

        if len(probs) != len(CLASS_NAMES):
            return jsonify({"error": "Class count mismatch",
                            "model_logits": int(len(probs)),
                            "class_names": int(len(CLASS_NAMES))}), 500

        top_idx = probs.argsort()[::-1][:TOP_K]
        top = [{"class": CLASS_NAMES[i], "prob": round(float(probs[i]), 6)} for i in top_idx]
        full = {CLASS_NAMES[i]: round(float(probs[i]), 6) for i in range(len(CLASS_NAMES))}
        return jsonify({"predicted": top[0]["class"], "top_k": top, "probs": full})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal error", "detail": str(e)}), 500

@app.post("/gradcam")
def gradcam():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part; send multipart/form-data with 'file'"}), 400
        f = request.files["file"]
        if not f.filename:
            return jsonify({"error": "Empty filename"}), 400

        class_index = request.form.get("class_index", default=None,
                                       type=lambda v: int(v) if v not in (None, "") else None)
        base_pil, img_batch = preprocess_image(f.read())
        if class_index is None:
            preds = model.predict(img_batch, verbose=0)[0]
            class_index = int(np.argmax(preds))

        heatmap = make_gradcam_heatmap(img_batch, model, pred_index=class_index)
        overlay = overlay_heatmap_on_pil(base_pil, heatmap, alpha=0.45)

        buf = io.BytesIO()
        overlay.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal error", "detail": str(e)}), 500

@app.post("/predict_with_gradcam")
def predict_with_gradcam():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part; send multipart/form-data with 'file'"}), 400
        f = request.files["file"]
        if not f.filename:
            return jsonify({"error": "Empty filename"}), 400

        base_pil, img_batch = preprocess_image(f.read())
        probs = model.predict(img_batch, verbose=0)[0].astype(float)

        if len(probs) != len(CLASS_NAMES):
            return jsonify({"error": "Class count mismatch",
                            "model_logits": int(len(probs)),
                            "class_names": int(len(CLASS_NAMES))}), 500

        top_idx = probs.argsort()[::-1][:TOP_K]
        top = [{"class": CLASS_NAMES[i], "prob": round(float(probs[i]), 6)} for i in top_idx]
        full = {CLASS_NAMES[i]: round(float(probs[i]), 6) for i in range(len(CLASS_NAMES))}
        top_class_index = int(top_idx[0])

        heatmap = make_gradcam_heatmap(img_batch, model, pred_index=top_class_index)
        overlay = overlay_heatmap_on_pil(base_pil, heatmap, alpha=0.45)

        buf = io.BytesIO()
        overlay.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return jsonify({"predicted": top[0]["class"], "top_k": top, "probs": full,
                        "gradcam_png_base64": img_b64})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal error", "detail": str(e)}), 500

@app.get("/")
def index():
    return render_template_string("""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Dog Disease Detector</title>
  <style>
    body { font-family: system-ui, Arial, sans-serif; margin: 24px; }
    .row { display: flex; gap: 24px; align-items: flex-start; flex-wrap: wrap; }
    .card { padding: 16px; border: 1px solid #ddd; border-radius: 12px; max-width: 560px; }
    .bar { height: 10px; background:#eee; border-radius: 6px; overflow:hidden; }
    .fill { height:100%; background:#4c8bf5; width:0%; }
    img { max-width: 520px; border-radius: 12px; border:1px solid #ddd; }
    pre { background:#f8f8f8; padding:12px; border-radius:8px; overflow:auto; }
    button { padding:10px 14px; border-radius:8px; border:1px solid #ccc; cursor:pointer; }
    .muted{color:#666}
  </style>
</head>
<body>
  <h2>Dog Skin Disease — Predictor + Grad-CAM</h2>

  <div class="row">
    <div class="card">
      <form id="form" enctype="multipart/form-data">
        <input type="file" name="file" id="file" accept="image/*" required /><br><br>
        <button type="submit">Predict</button>
        <span class="muted" id="status"></span>
      </form>
      <div id="preds" style="margin-top:16px; display:none;">
        <h3>Top-K Predictions</h3>
        <div id="klist"></div>
        <details style="margin-top:10px;">
          <summary>Full probabilities (JSON)</summary>
          <pre id="probs"></pre>
        </details>
      </div>

      <div style="margin-top:16px;">
        <label class="muted">Optional: pick a class for Grad-CAM</label><br/>
        <select id="classSelect"></select>
        <button id="btnCam" style="margin-left:8px;">Make Grad-CAM</button>
      </div>
    </div>

    <div class="card">
      <h3>Input</h3>
      <img id="preview" alt="uploaded image will preview here" />
    </div>

    <div class="card">
      <h3>Grad-CAM</h3>
      <img id="cam" alt="Grad-CAM heatmap will render here" />
    </div>
  </div>

<script>
const form = document.getElementById('form');
const fileInput = document.getElementById('file');
const statusEl = document.getElementById('status');
const predsBox = document.getElementById('preds');
const probsPre = document.getElementById('probs');
const klist = document.getElementById('klist');
const preview = document.getElementById('preview');
const cam = document.getElementById('cam');
const classSelect = document.getElementById('classSelect');
const btnCam = document.getElementById('btnCam');

const CLASS_NAMES = {{ class_names | tojson }};
CLASS_NAMES.forEach((c, i)=> {
  const opt = document.createElement('option');
  opt.value = i; opt.textContent = `${i}: ${c}`;
  classSelect.appendChild(opt);
});

fileInput.addEventListener('change', ()=>{
  const f = fileInput.files[0];
  if (!f) return;
  const url = URL.createObjectURL(f);
  preview.src = url;
  cam.removeAttribute('src');
  predsBox.style.display='none';
});

form.addEventListener('submit', async (e)=>{
  e.preventDefault();
  const f = fileInput.files[0];
  if (!f) return;

  statusEl.textContent = 'Uploading & predicting...';
  cam.removeAttribute('src');

  const fd = new FormData();
  fd.append('file', f);

  const res = await fetch('/predict_with_gradcam', { method:'POST', body: fd });
  if (!res.ok) { statusEl.textContent = 'Server error: ' + res.status; return; }
  const js = await res.json();
  statusEl.textContent = '';

  klist.innerHTML = '';
  js.top_k.forEach(t => {
    const wrap = document.createElement('div'); wrap.style.margin = '6px 0';
    const label = document.createElement('div');
    label.textContent = `${t.class} — ${(t.prob*100).toFixed(2)}%`;
    const bar = document.createElement('div'); bar.className='bar';
    const fill = document.createElement('div'); fill.className='fill'; fill.style.width = (t.prob*100)+'%';
    bar.appendChild(fill); wrap.appendChild(label); wrap.appendChild(bar);
    klist.appendChild(wrap);
  });

  probsPre.textContent = JSON.stringify(js.probs, null, 2);
  predsBox.style.display = 'block';

  if (js.gradcam_png_base64) {
    cam.src = 'data:image/png;base64,' + js.gradcam_png_base64;
  }

  const topIdx = CLASS_NAMES.indexOf(js.predicted);
  if (topIdx >= 0) classSelect.value = topIdx;
});

btnCam.addEventListener('click', async ()=>{
  const f = fileInput.files[0];
  if (!f) return;
  const fd = new FormData();
  fd.append('file', f);
  fd.append('class_index', classSelect.value);
  statusEl.textContent = 'Generating Grad-CAM...';
  const res = await fetch('/gradcam', { method:'POST', body: fd });
  if (res.ok) {
    const blob = await res.blob();
    cam.src = URL.createObjectURL(blob);
    statusEl.textContent = '';
  } else {
    statusEl.textContent = 'Grad-CAM failed (' + res.status + ')';
  }
});
</script>
</body>
</html>
""", class_names=CLASS_NAMES)

if __name__ == "__main__":
    # You can also: waitress-serve --port=5000 app:app
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
