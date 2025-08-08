import os
os.environ["PYTHONUNBUFFERED"] = "1"

import io
import urllib.request
import traceback
import threading
import hashlib
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

# ===================== CONFIG =====================
MODEL_PATH = os.environ.get("MODEL_PATH", "model.tflite")
MODEL_URL  = os.environ.get("MODEL_URL")              # optional: remote model URL
IMG_SIZE   = (224, 224)

# Preprocess used during TRAINING:
#   "/255"         -> scale to [0,1]
#   "efficientnet" -> [-1, 1]  (x/127.5 - 1)
#   "none"         -> leave 0..255
PREPROCESS = os.environ.get("PREPROCESS", "/255").lower()

CLASS_NAMES = [
    "Bacterial_dermatosis",
    "Dermatitis",
    "Fungal_infections",
    "Healthy",
    "Hypersensitivity_allergic_dermatosis",
    "demodicosis",
    "ringworm",
]

TOP_K = int(os.environ.get("TOP_K", "3"))
NUM_THREADS = int(os.environ.get("NUM_THREADS", "1"))
# ===================================================

# ---- TFLite import that works locally & on Render ----
try:
    import tensorflow.lite as tflite  # full TF available (local dev)
    TFLITE_RUNTIME = "tensorflow"
except Exception:
    import tflite_runtime.interpreter as tflite  # Render
    TFLITE_RUNTIME = "tflite-runtime"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8MB uploads

# --------- Model file provisioning ----------
def ensure_model_file() -> bool:
    if os.path.exists(MODEL_PATH):
        return True
    if not MODEL_URL:
        print("No local model and no MODEL_URL provided.")
        return False
    try:
        os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
        print(f"Downloading model from {MODEL_URL} ...")
        with urllib.request.urlopen(MODEL_URL, timeout=120) as r, open(MODEL_PATH, "wb") as f:
            f.write(r.read())
        print("Downloaded model to", MODEL_PATH)
        return True
    except Exception as e:
        print("Download failed:", repr(e))
        return False

ensure_model_file()

# --------- TFLite interpreter (singleton) ----------
_interpreter = None
_in_det = None
_out_det = None
_interp_lock = threading.Lock()

def get_interpreter():
    """Create once, reuse. Thread-safe."""
    global _interpreter, _in_det, _out_det
    if _interpreter is not None:
        return _interpreter, _in_det, _out_det
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found: {MODEL_PATH}")
    with _interp_lock:
        if _interpreter is None:
            itp = tflite.Interpreter(model_path=MODEL_PATH, num_threads=NUM_THREADS)
            itp.allocate_tensors()
            _in  = itp.get_input_details()[0]
            _out = itp.get_output_details()[0]
            _interpreter, _in_det, _out_det = itp, _in, _out
    return _interpreter, _in_det, _out_det

# --------- Quantization helpers ----------
def _is_quantized(tensor_detail):
    qparams = tensor_detail.get("quantization_parameters", {})
    scales = qparams.get("scales", [])
    if len(scales) == 0:
        scale, _ = tensor_detail.get("quantization", (0.0, 0))
        return bool(scale)
    return any(float(s) != 0.0 for s in scales)

def _get_qparams(tensor_detail):
    q = tensor_detail.get("quantization_parameters", {})
    scales = q.get("scales", [])
    zero_points = q.get("zero_points", [])
    if len(scales) == 0:
        scale, zp = tensor_detail.get("quantization", (0.0, 0))
        return float(scale or 0.0), int(zp or 0)
    return float(scales[0]), int(zero_points[0] if zero_points else 0)

# --------- Preprocess to match TRAINING ----------
def _preprocess_array(x: np.ndarray, mode: str) -> np.ndarray:
    """x is float32 array in range 0..255"""
    if mode == "efficientnet":
        return (x / 127.5) - 1.0
    elif mode == "/255":
        return x / 255.0
    else:
        return x

def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB").resize(IMG_SIZE)
    x = np.asarray(img, dtype=np.float32)
    x = _preprocess_array(x, PREPROCESS)
    x = np.expand_dims(x, axis=0)  # (1, H, W, 3)
    return img, x

# --------- Postprocess (softmax if needed) ----------
def _maybe_softmax(v):
    v = np.asarray(v, dtype=np.float32)
    s = float(v.sum())
    if 0.98 <= s <= 1.02 and np.all(v >= -1e-6) and np.all(v <= 1.0 + 1e-6):
        return v
    e = np.exp(v - np.max(v))
    return e / np.sum(e)

# --------- Inference ----------
def predict_probs(x_batch):
    itp, in_det, out_det = get_interpreter()

    xin = x_batch.astype(np.float32)

    # If model input is quantized (e.g., uint8), convert with scale/zp
    if in_det["dtype"] == np.uint8 or _is_quantized(in_det):
        scale, zp = _get_qparams(in_det)
        if scale == 0.0:
            q = np.clip(xin, 0, 255).astype(np.uint8)
        else:
            q = np.round(xin / scale + zp)
            q = np.clip(q, 0, 255).astype(np.uint8)
        xin = q
    else:
        xin = xin.astype(in_det["dtype"])

    with _interp_lock:
        itp.set_tensor(in_det["index"], xin)
        itp.invoke()
        preds = itp.get_tensor(out_det["index"])[0]

    # Dequantize outputs if needed
    if (out_det["dtype"] == np.uint8) or _is_quantized(out_det):
        scale, zp = _get_qparams(out_det)
        preds = (preds.astype(np.float32) - zp) * (scale if scale else 1.0)
    else:
        preds = preds.astype(np.float32)

    preds = _maybe_softmax(preds)
    return preds

# --------- Response helpers ----------
def _format_response(probs):
    top_idx = np.argsort(probs)[::-1][:TOP_K]
    top = [{"label": CLASS_NAMES[i], "prob": float(round(probs[i], 6))} for i in top_idx]
    full = {CLASS_NAMES[i]: float(round(probs[i], 6)) for i in range(len(CLASS_NAMES))}
    return {
        "predicted": top[0]["label"],
        "top_k": top,
        "probs": full
    }

# ===================== ROUTES =====================

# UI at '/' (same-origin — no CORS headaches)
@app.get("/")
def ui():
    return f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Dog Skin Disease — Predictor + Grad-CAM</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  :root {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }}
  body {{ margin: 20px; background: #fafafa; color:#111; }}
  h1 {{ margin: 0 0 16px; }}
  .grid {{ display: grid; grid-template-columns: 380px 1fr; gap: 16px; align-items: start; }}
  .card {{ background:#fff; border-radius:16px; padding:16px; box-shadow:0 1px 3px rgba(0,0,0,.08); }}
  .btn {{ padding:10px 14px; border-radius:10px; border:1px solid #ccc; background:#111; color:#fff; cursor:pointer; }}
  .btn:disabled {{ opacity: .6; cursor: not-allowed; }}
  .bar {{ height:10px; border-radius:6px; background:#eee; overflow:hidden; }}
  .bar > div {{ height:100%; width:0; background:#4b8bff; transition: width .3s; }}
  select, input[type=file]{{ width:100%; padding:8px; border-radius:8px; border:1px solid #ddd; background:#fff; }}
  details {{ margin-top: 8px; }}
  pre {{ white-space: pre-wrap; }}
  .imgwrap {{ position:relative; width:100%; max-width: 720px; }}
  .imgwrap img {{ width:100%; display:block; border-radius:12px; }}
  .heat {{ position:absolute; inset:0; mix-blend-mode:multiply; opacity:.55; border-radius:12px; }}
  .row {{ display:flex; gap:8px; align-items:center; }}
  .small {{ font-size:.9em; color:#555; }}
</style>
</head>
<body>
  <h1>Dog Skin Disease — Predictor + Grad-CAM</h1>
  <div class="grid">
    <div class="card">
      <input id="file" type="file" accept="image/*" />
      <div style="height:8px"></div>
      <button id="predict" class="btn">Predict</button>
      <div style="height:16px"></div>

      <h3>Top-K Predictions</h3>
      <div id="tops"></div>
      <details>
        <summary>Full probabilities (JSON)</summary>
        <pre id="json"></pre>
      </details>

      <div style="height:16px"></div>
      <div class="row">
        <select id="classSel"></select>
        <button id="grad" class="btn">Make Grad-CAM</button>
      </div>
      <div class="small">Note: This uses an <b>occlusion heatmap</b> (no gradients) so it works with TFLite on Render. It may take ~5-20s.</div>
    </div>

    <div class="card">
      <h3>Input</h3>
      <div class="imgwrap">
        <img id="preview" alt="preview" />
        <canvas id="heat" class="heat"></canvas>
      </div>
    </div>
  </div>

<script>
const file = document.getElementById('file');
const predictBtn = document.getElementById('predict');
const gradBtn = document.getElementById('grad');
const tops = document.getElementById('tops');
const jsonOut = document.getElementById('json');
const preview = document.getElementById('preview');
const heat = document.getElementById('heat');
const classSel = document.getElementById('classSel');

const CLASS_NAMES = {CLASS_NAMES};

function barRow(label, prob) {{
  const div = document.createElement('div');
  const p = document.createElement('div'); p.className='bar'; const inner = document.createElement('div'); p.appendChild(inner);
  inner.style.width = Math.round(prob*100)+'%';
  const t = document.createElement('div'); t.textContent = `${{label}} — ${{(prob*100).toFixed(2)}}%`;
  t.style.margin = '6px 0 4px';
  div.appendChild(t); div.appendChild(p);
  return div;
}}

function setTops(top) {{
  tops.innerHTML = '';
  top.forEach(t => tops.appendChild(barRow(t.label, t.prob)));
  classSel.innerHTML = CLASS_NAMES.map((c,i)=>`<option value="${{i}}">${{i}}: ${{c}}</option>`).join('');
  const bestIdx = CLASS_NAMES.indexOf(top[0].label);
  if (bestIdx >= 0) classSel.value = String(bestIdx);
}}

async function postPredict(blob) {{
  const fd = new FormData();
  fd.append('file', blob, 'img.jpg');
  const r = await fetch('/predict', {{ method:'POST', body: fd }});
  if (!r.ok) throw new Error('predict failed');
  return await r.json();
}}

predictBtn.onclick = async () => {{
  try {{
    const f = file.files[0];
    if (!f) return alert('Pick an image first');
    heat.width = heat.height = 0; // clear heatmap
    preview.src = URL.createObjectURL(f);
    const j = await postPredict(f);
    setTops(j.top_k);
    jsonOut.textContent = JSON.stringify(j, null, 2);
  }} catch (e) {{
    alert(e);
  }}
}};

// --- Occlusion "Grad-CAM" (approx) ---
// Makes many masked copies and reads score for the selected class.
gradBtn.onclick = async () => {{
  try {{
    const f = file.files[0];
    if (!f) return alert('Pick an image first');
    const clsIdx = parseInt(classSel.value, 10) || 0;

    gradBtn.disabled = true; gradBtn.textContent = 'Working…';

    // Draw original to a canvas at model size
    const img = await createImageBitmap(f);
    const W=224, H=224;
    const base = new OffscreenCanvas(W, H);
    const bctx = base.getContext('2d');
    bctx.drawImage(img, 0, 0, W, H);

    // Get baseline score on original
    const baseBlob = await base.convertToBlob({{ type:'image/jpeg', quality:0.9 }});
    const basePred = await postPredict(baseBlob);
    const baseScore = basePred.probs[CLASS_NAMES[clsIdx]] ?? 0;

    // Grid occlusion
    const GRID = 12; // adjust 8..16 for speed/quality
    const cellW = Math.ceil(W/GRID), cellH = Math.ceil(H/GRID);
    const scores = new Float32Array(GRID*GRID);

    for (let gy=0; gy<GRID; gy++) {{
      for (let gx=0; gx<GRID; gx++) {{
        const c = new OffscreenCanvas(W,H);
        const ctx = c.getContext('2d');
        ctx.drawImage(base, 0, 0);
        // cover one cell with mean color (or gray)
        ctx.fillStyle = 'rgb(128,128,128)';
        ctx.fillRect(gx*cellW, gy*cellH, cellW, cellH);
        const blob = await c.convertToBlob({{ type:'image/jpeg', quality:0.9 }});
        const pred = await postPredict(blob);
        const sc = pred.probs[CLASS_NAMES[clsIdx]] ?? 0;
        scores[gy*GRID+gx] = baseScore - sc; // drop = importance
      }}
    }}

    // Normalize scores 0..1
    const min = Math.min(...scores), max = Math.max(...scores);
    for (let i=0;i<scores.length;i++) scores[i] = (scores[i]-min)/((max-min)||1);

    // Paint heatmap to visible canvas sized like preview
    await new Promise(res => {{
      preview.onload = res;
      if (preview.complete) res();
    }});
    heat.width = preview.clientWidth || preview.naturalWidth;
    heat.height = preview.clientHeight || preview.naturalHeight;
    const hctx = heat.getContext('2d');

    // draw upsampled heatmap
    const cellX = heat.width/GRID, cellY = heat.height/GRID;
    for (let gy=0; gy<GRID; gy++) {{
      for (let gx=0; gx<GRID; gx++) {{
        const v = scores[gy*GRID+gx];
        // simple colormap (blue→red)
        const r = Math.round(255*v);
        const g = 0;
        const b = Math.round(255*(1-v));
        hctx.fillStyle = `rgba(${{r}},${{g}},${{b}},0.55)`;
        hctx.fillRect(gx*cellX, gy*cellY, cellX, cellY);
      }}
    }}

    gradBtn.disabled = false; gradBtn.textContent = 'Make Grad-CAM';
  }} catch (e) {{
    gradBtn.disabled = false; gradBtn.textContent = 'Make Grad-CAM';
    alert(e);
  }}
}};
</script>
</body>
</html>
"""

@app.get("/healthz")
def healthz():
    try:
        get_interpreter()
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500

@app.get("/debug")
def debug():
    try:
        _, in_det, out_det = get_interpreter()

        def qinfo(d):
            s = d.get("quantization_parameters", {})
            sc = s.get("scales", [])
            zp = s.get("zero_points", [])
            if not sc:
                sc0, zp0 = d.get("quantization", (0.0, 0))
                return {"scale": float(sc0 or 0.0), "zero_point": int(zp0 or 0)}
            return {"scale": float(sc[0]), "zero_point": int(zp[0] if zp else 0)}

        return jsonify({
            "input_dtype": str(in_det["dtype"]),
            "output_dtype": str(out_det["dtype"]),
            "input_q": qinfo(in_det),
            "output_q": qinfo(out_det),
            "preprocess": PREPROCESS,
            "img_size": IMG_SIZE,
            "class_names": CLASS_NAMES,
            "model_path": MODEL_PATH,
            "tflite_runtime": TFLITE_RUNTIME,
        }), 200
    except Exception as e:
        return jsonify({"error": "debug failed", "detail": str(e)}), 500

@app.post("/predict")
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        f = request.files["file"]
        if not f.filename:
            return jsonify({"error": "Empty filename"}), 400

        _, x = preprocess_image(f.read())
        probs = predict_probs(x)
        resp = _format_response(probs)
        return jsonify(resp), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal error", "detail": str(e)}), 500

# Keep for any old callers
@app.post("/predict_with_gradcam")
def predict_with_gradcam():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        f = request.files["file"]
        if not f.filename:
            return jsonify({"error": "Empty filename"}), 400
        _, x = preprocess_image(f.read())
        probs = predict_probs(x)
        resp = _format_response(probs)
        resp["gradcam"] = None   # client now builds heatmap
        return jsonify(resp), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal error", "detail": str(e)}), 500

@app.post("/inspect")
def inspect():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        f = request.files["file"]
        raw = f.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB").resize(IMG_SIZE)
        arr = np.asarray(img, dtype=np.float32)
        arr = _preprocess_array(arr, PREPROCESS)
        arr = np.expand_dims(arr, 0).astype("float32")
        return jsonify({
            "preprocess": PREPROCESS,
            "shape": list(arr.shape),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "first5": [float(v) for v in arr.reshape(-1)[:5]],
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "inspect failed", "detail": str(e)}), 500

def _pre_x(raw, mode):
    img = Image.open(io.BytesIO(raw)).convert("RGB").resize(IMG_SIZE)
    x = np.asarray(img, dtype=np.float32)
    x = _preprocess_array(x, mode)
    return np.expand_dims(x, 0)

@app.post("/triage")
def triage():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        raw = request.files["file"].read()
        out = {}
        for mode in ["efficientnet", "/255", "none"]:
            x = _pre_x(raw, mode)
            probs = predict_probs(x)
            top_idx = np.argsort(probs)[::-1][:3]
            out[mode] = {
                "stats": {
                    "min": float(x.min()), "max": float(x.max()),
                    "mean": float(x.mean()), "std": float(x.std())
                },
                "top": [
                    {"label": CLASS_NAMES[i], "prob": float(probs[i])}
                    for i in top_idx
                ],
                "full": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
            }
        return jsonify(out)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":"triage failed","detail":str(e)}), 500

@app.get("/md5")
def md5sum():
    try:
        h = hashlib.md5(open(MODEL_PATH, "rb").read()).hexdigest()
        return jsonify({"model_path": MODEL_PATH, "md5": h})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===================== MAIN =====================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
