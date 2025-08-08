import os
os.environ["PYTHONUNBUFFERED"] = "1"

import io
import urllib.request
import traceback
import threading
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

# ===================== CONFIG =====================
MODEL_PATH = os.environ.get("MODEL_PATH", "model.tflite")
MODEL_URL  = os.environ.get("MODEL_URL")              # optional: remote model URL
IMG_SIZE   = (224, 224)

# Preprocess used during TRAINING:
#   "/255"         -> scale to [0,1]
#   "efficientnet" -> keras.applications.efficientnet.preprocess_input ([-1,1], etc.)
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

import tensorflow.lite as tflite

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
def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    x = np.asarray(img, dtype=np.float32)

    if PREPROCESS == "efficientnet":
        # EXACTLY what EfficientNetB0 expects (used during your training)
        from tensorflow.keras.applications.efficientnet import preprocess_input
        x = preprocess_input(x)  # -> [-1,1], correct mean/scale
    elif PREPROCESS == "/255":
        x = x / 255.0
    else:
        # "none": leave 0..255 float
        pass

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
            # Fallback: clamp to 0..255
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
@app.get("/")
def index():
    return "Dog Disease Detection API is running!"

@app.get("/healthz")
def healthz():
    try:
        itp, in_det, out_det = get_interpreter()
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500

@app.get("/debug")
def debug():
    try:
        itp, in_det, out_det = get_interpreter()

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

# Optional: if your UI calls this path, return same JSON (no heatmap for TFLite)
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
        resp["gradcam"] = None  # placeholder
        return jsonify(resp), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal error", "detail": str(e)}), 500
    
import base64

@app.post("/inspect")
def inspect():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        f = request.files["file"]
        raw = f.read()
        _, x = preprocess_image(raw)  # uses current PREPROCESS
        arr = x.astype("float32")
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
        return jsonify({"error":"inspect failed","detail":str(e)}), 500

def _pre_x(raw, mode):
    # make a local copy of preprocess with override
    img = Image.open(io.BytesIO(raw)).convert("RGB").resize(IMG_SIZE)
    x = np.asarray(img, dtype=np.float32)
    if mode == "efficientnet":
        from tensorflow.keras.applications.efficientnet import preprocess_input
        x = preprocess_input(x)
    elif mode == "/255":
        x = x / 255.0
    elif mode == "none":
        pass
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

# --- add to app.py, near other routes ---
import hashlib

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
