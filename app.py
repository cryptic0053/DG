import os
os.environ["PYTHONUNBUFFERED"] = "1"

import io
import urllib.request
import traceback
import threading
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template_string

# ---------------- CONFIG ----------------
MODEL_PATH = os.environ.get("MODEL_PATH", "model.tflite")  # commit this file or provide MODEL_URL
MODEL_URL  = os.environ.get("MODEL_URL")                   # direct URL to model.tflite (optional)
IMG_SIZE   = (224, 224)    # must match training input
RESCALE    = 1.0           # must match training preprocessing
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
GRADCAM_ENABLED = False  # TFLite can't do gradients
NUM_THREADS = int(os.environ.get("NUM_THREADS", "1"))
# ----------------------------------------

# Lightweight TF runtime
import tflite_runtime.interpreter as tflite

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8MB upload cap

# --------- Model file presence (optional download) ----------
def ensure_model_file() -> bool:
    """Ensure MODEL_PATH exists; otherwise try downloading from MODEL_URL."""
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

# Try to ensure it at boot, but don't crash if missing (health will report it)
ensure_model_file()

# --------- Lazy TFLite interpreter (thread-safe) ----------
_interpreter = None
_in_det = None
_out_det = None
_interp_lock = threading.Lock()

def get_interpreter():
    """
    Lazily create the TFLite interpreter and cache input/output details.
    Use a lock because TFLite isn't thread-safe.
    """
    global _interpreter, _in_det, _out_det
    if _interpreter is not None:
        return _interpreter, _in_det, _out_det

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}")

    with _interp_lock:
        if _interpreter is None:
            itp = tflite.Interpreter(model_path=MODEL_PATH, num_threads=NUM_THREADS)
            itp.allocate_tensors()
            _in = itp.get_input_details()[0]   # assume single input
            _out = itp.get_output_details()[0] # assume first output is logits/probs
            _interpreter, _in_det, _out_det = itp, _in, _out

    return _interpreter, _in_det, _out_det

# --------------- Helpers ----------------
def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    x = np.asarray(img, dtype=np.float32) * RESCALE
    x = np.expand_dims(x, axis=0)  # [1, H, W, C]
    return img, x

def predict_probs(x_batch):
    itp, in_det, out_det = get_interpreter()
    xin = x_batch.astype(in_det["dtype"])
    with _interp_lock:
        itp.set_tensor(in_det["index"], xin)
        itp.invoke()
        preds = itp.get_tensor(out_det["index"])  # [1, num_classes]
    preds = preds[0].astype(float)

    # If your model outputs logits, uncomment to apply softmax:
    # e = np.exp(preds - np.max(preds))
    # preds = e / np.sum(e)

    return preds
# ----------------------------------------

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
            return jsonify({
                "error": "Class count mismatch",
                "model_logits": int(len(probs)),
                "class_names": int(len(CLASS_NAMES))
            }), 500

        top_idx = np.argsort(probs)[::-1][:TOP_K]
        top = [{"class": CLASS_NAMES[i], "prob": round(float(probs[i]), 6)} for i in top_idx]
        full = {CLASS_NAMES[i]: round(float(probs[i]), 6) for i in range(len(CLASS_NAMES))}
        return jsonify({"predicted": top[0]["class"], "top_k": top, "probs": full})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal error", "detail": str(e)}), 500

@app.post("/predict_with_gradcam")
def predict_with_gradcam():
    # Same as /predict, but keeps a 'gradcam_png_base64' key (None for TFLite).
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part; send multipart/form-data with 'file'"}), 400
        f = request.files["file"]
        if not f.filename:
            return jsonify({"error": "Empty filename"}), 400

        _, x = preprocess_image(f.read())
        probs = predict_probs(x)

        if len(probs) != len(CLASS_NAMES):
            return jsonify({
                "error": "Class count mismatch",
                "model_logits": int(len(probs)),
                "class_names": int(len(CLASS_NAMES))
            }), 500

        top_idx = np.argsort(probs)[::-1][:TOP_K]
        top = [{"class": CLASS_NAMES[i], "prob": round(float(probs[i]), 6)} for i in top_idx]
        full = {CLASS_NAMES[i]: round(float(probs[i]), 6) for i in range(len(CLASS_NAMES))}

        return jsonify({
            "predicted": top[0]["class"],
            "top_k": top,
            "probs": full,
            "gradcam_png_base64": None  # not available on TFLite
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal error", "detail": str(e)}), 500

@app.post("/gradcam")
def gradcam():
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
  </div>

<script>
const form = document.getElementById('form');
const fileInput = document.getElementById('file');
const statusEl = document.getElementById('status');
const predsBox = document.getElementById('preds');
const probsPre = document.getElementById('probs');
const klist = document.getElementById('klist');
const preview = document.getElementById('preview');

fileInput.addEventListener('change', ()=>{
  const f = fileInput.files[0];
  if (!f) return;
  const url = URL.createObjectURL(f);
  preview.src = url;
  predsBox.style.display='none';
});

form.addEventListener('submit', async (e)=>{
  e.preventDefault();
  const f = fileInput.files[0];
  if (!f) return;

  statusEl.textContent = 'Uploading & predicting...';

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
    # local dev (Render will use gunicorn)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
