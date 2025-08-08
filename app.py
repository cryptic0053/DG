import os
os.environ["PYTHONUNBUFFERED"] = "1"

import io
import base64
import urllib.request
import traceback
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template_string, send_file

# ---------------- CONFIG ----------------
MODEL_PATH = os.environ.get("MODEL_PATH", "model.tflite")  # put model.tflite in repo, or set MODEL_URL
MODEL_URL  = os.environ.get("MODEL_URL")                   # direct link to model.tflite if not in repo
IMG_SIZE   = (224, 224)    # must match training
RESCALE    = 1.0           # must match training (your .h5 had 1.0)
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
GRADCAM_ENABLED = False  # TFLite cannot compute gradients
# ----------------------------------------

# tflite-runtime is much lighter than full TF
import tflite_runtime.interpreter as tflite

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8MB upload cap


def ensure_model_file() -> bool:
    """Ensure MODEL_PATH exists; otherwise try downloading from MODEL_URL (raw file URL)."""
    if os.path.exists(MODEL_PATH):
        return True
    if not MODEL_URL:
        print("No local model.tflite and no MODEL_URL provided.")
        return False
    try:
        os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
        print(f"Downloading TFLite model from {MODEL_URL} ...")
        with urllib.request.urlopen(MODEL_URL, timeout=120) as r, open(MODEL_PATH, "wb") as f:
            f.write(r.read())
        print("Model downloaded to", MODEL_PATH)
        return True
    except Exception as e:
        print("Model download failed:", repr(e))
        return False


if not ensure_model_file():
    # The app will still boot so you can hit /health and see the error
    print("WARNING: model file missing:", MODEL_PATH)

# ---- Load TFLite model ----
interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=int(os.environ.get("NUM_THREADS", "1")))
interpreter.allocate_tensors()
_in = interpreter.get_input_details()[0]   # assume single input
_outs = interpreter.get_output_details()
_out = _outs[0]                            # assume first output is probs/logits


# ---------- Helpers ----------
def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    x = np.asarray(img, dtype=np.float32) * RESCALE
    x = np.expand_dims(x, axis=0)  # [1, H, W, C]
    return img, x


def predict_probs(x_batch):
    xin = x_batch.astype(_in["dtype"])
    interpreter.set_tensor(_in["index"], xin)
    interpreter.invoke()
    preds = interpreter.get_tensor(_out["index"])  # [1, num_classes] usually
    preds = preds[0].astype(float)

    # If your TFLite head outputs logits, uncomment softmax below:
    # e = np.exp(preds - np.max(preds))
    # preds = e / np.sum(e)

    return preds


# ----------------------------

@app.get("/health")
def health():
    return jsonify({
        "status": "ok" if os.path.exists(MODEL_PATH) else "missing_model",
        "model_path": MODEL_PATH,
        "gradcam": GRADCAM_ENABLED
    })


@app.post("/predict")
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part; send multipart/form-data with 'file'"}), 400
        f = request.files["file"]
        if not f.filename:
            return jsonify({"error": "Empty filename"}), 400

        _, x = preprocess_image(f.read())
        probs = predict_probs(x)

        if len(probs) != len(CLASS_NAMES):
            return jsonify({"error": "Class count mismatch",
                            "model_logits": int(len(probs)),
                            "class_names": int(len(CLASS_NAMES))}), 500

        top_idx = np.argsort(probs)[::-1][:TOP_K]
        top = [{"class": CLASS_NAMES[i], "prob": round(float(probs[i]), 6)} for i in top_idx]
        full = {CLASS_NAMES[i]: round(float(probs[i]), 6) for i in range(len(CLASS_NAMES))}
        return jsonify({"predicted": top[0]["class"], "top_k": top, "probs": full})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal error", "detail": str(e)}), 500


@app.post("/predict_with_gradcam")
def predict_with_gradcam():
    # Same as /predict, but keeps the response shape your front-end expects.
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part; send multipart/form-data with 'file'"}), 400
        f = request.files["file"]
        if not f.filename:
            return jsonify({"error": "Empty filename"}), 400

        base_pil, x = preprocess_image(f.read())
        probs = predict_probs(x)

        if len(probs) != len(CLASS_NAMES):
            return jsonify({"error": "Class count mismatch",
                            "model_logits": int(len(probs)),
                            "class_names": int(len(CLASS_NAMES))}), 500

        top_idx = np.argsort(probs)[::-1][:TOP_K]
        top = [{"class": CLASS_NAMES[i], "prob": round(float(probs[i]), 6)} for i in top_idx]
        full = {CLASS_NAMES[i]: round(float(probs[i]), 6) for i in range(len(CLASS_NAMES))}

        resp = {"predicted": top[0]["class"], "top_k": top, "probs": full}
        # TFLite: no gradients → no CAM image. Keep key for backward-compat, set None.
        resp["gradcam_png_base64"] = None
        return jsonify(resp)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal error", "detail": str(e)}), 500


@app.post("/gradcam")
def gradcam():
    # Explicitly 501 so your UI can show a friendly message if you wire it.
    return jsonify({"error": "Grad-CAM is disabled on the TFLite build"}), 501


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
  <h2>Dog Skin Disease — Predictor{% if gradcam %} + Grad-CAM{% endif %}</h2>

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

      {% if gradcam %}
      <div style="margin-top:16px;">
        <label class="muted">Optional: pick a class for Grad-CAM</label><br/>
        <select id="classSelect"></select>
        <button id="btnCam" style="margin-left:8px;">Make Grad-CAM</button>
      </div>
      {% else %}
      <div class="muted" style="margin-top:16px;">
        Grad-CAM is disabled on the Render deployment (TFLite build).
      </div>
      {% endif %}
    </div>

    <div class="card">
      <h3>Input</h3>
      <img id="preview" alt="uploaded image will preview here" />
    </div>

    {% if gradcam %}
    <div class="card">
      <h3>Grad-CAM</h3>
      <img id="cam" alt="Grad-CAM heatmap will render here" />
    </div>
    {% endif %}
  </div>

<script>
const form = document.getElementById('form');
const fileInput = document.getElementById('file');
const statusEl = document.getElementById('status');
const predsBox = document.getElementById('preds');
const probsPre = document.getElementById('probs');
const klist = document.getElementById('klist');
const preview = document.getElementById('preview');
{% if gradcam %}
const cam = document.getElementById('cam');
const classSelect = document.getElementById('classSelect');
const btnCam = document.getElementById('btnCam');
const CLASS_NAMES = {{ class_names | tojson }};
CLASS_NAMES.forEach((c, i)=> {
  const opt = document.createElement('option');
  opt.value = i; opt.textContent = `${i}: ${c}`;
  classSelect.appendChild(opt);
});
{% endif %}

fileInput.addEventListener('change', ()=>{
  const f = fileInput.files[0];
  if (!f) return;
  const url = URL.createObjectURL(f);
  preview.src = url;
  predsBox.style.display='none';
  {% if gradcam %} if (cam) cam.removeAttribute('src'); {% endif %}
});

form.addEventListener('submit', async (e)=>{
  e.preventDefault();
  const f = fileInput.files[0];
  if (!f) return;

  statusEl.textContent = 'Uploading & predicting...';
  {% if gradcam %} if (cam) cam.removeAttribute('src'); {% endif %}

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
});
</script>
</body>
</html>
""", class_names=CLASS_NAMES, gradcam=GRADCAM_ENABLED)


if __name__ == "__main__":
    # local dev
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
