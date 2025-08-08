# rebuild_and_convert.py  -- ROBUST H5 SCANNER
import os, sys, numpy as np, tensorflow as tf, h5py
from PIL import Image
from tensorflow.keras import layers, Model

IMG_SIZE = (224, 224)
DEFAULT_NUM_CLASSES = 7
WEIGHTS_H5 = os.environ.get("WEIGHTS_H5", "dog_disease_detector_fixed.h5")
TEST_IMAGE = os.environ.get("TEST_IMAGE", "")
OUT_TFLITE = os.environ.get("OUT_TFLITE", "model.tflite")
PREPROCESS = os.environ.get("PREPROCESS", "efficientnet").lower()

CLASS_NAMES = [
    "Bacterial_dermatosis","Dermatitis","Fungal_infections",
    "Healthy","Hypersensitivity_allergic_dermatosis","demodicosis","ringworm"
]

def _find_first_kernel_shape(grp: h5py.Group):
    """
    Recursively search within an h5 group for any dataset whose name contains 'kernel'.
    Return its shape if found, else None.
    """
    out = None
    def _walk(g):
        nonlocal out
        if out is not None:
            return
        for k, v in g.items():
            if isinstance(v, h5py.Dataset):
                name = k.lower()
                if "kernel" in name:
                    out = tuple(v.shape)
                    return
            elif isinstance(v, h5py.Group):
                _walk(v)
                if out is not None:
                    return
    _walk(grp)
    return out

def infer_hidden_and_classes_from_h5(h5_path):
    """
    Determine (hidden_units_for_dense_4, num_classes_for_dense_5).
    Prefer dense_5 kernel shape; if unavailable, use dense_4 and assume classes = 7.
    """
    with h5py.File(h5_path, "r") as f:
        mw = f.get("model_weights")
        if mw is None:
            raise ValueError("No 'model_weights' group in H5; wrong file.")

        dense5_in = None
        dense5_out = None
        if "dense_5" in mw:
            kshape = _find_first_kernel_shape(mw["dense_5"])
            if kshape and len(kshape) == 2:
                dense5_in, dense5_out = int(kshape[0]), int(kshape[1])

        if dense5_in is not None and dense5_out is not None:
            return dense5_in if dense5_in != 1280 else None, dense5_out

        # fallback: look at dense_4 kernel to estimate hidden size
        dense4_units = None
        if "dense_4" in mw:
            kshape4 = _find_first_kernel_shape(mw["dense_4"])
            if kshape4 and len(kshape4) == 2:
                # dense_4 kernel is (1280, hidden_units)
                dense4_units = int(kshape4[1])

        # If we saw dense_4 but not dense_5, assume classes=DEFAULT_NUM_CLASSES
        if dense4_units is not None:
            return dense4_units, DEFAULT_NUM_CLASSES

        # Give up with a helpful error
        raise ValueError("Could not locate kernel weights for dense_5 or dense_4 in H5.")

def build_model(hidden_units, num_classes):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights=None, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="efficientnetb0"
    )
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d_2")(base.output)
    x = layers.Dropout(0.2, name="dropout_4")(x)
    if hidden_units is not None and hidden_units != 1280:
        x = layers.Dense(hidden_units, activation="relu", name="dense_4")(x)
        x = layers.Dropout(0.2, name="dropout_5")(x)
    out = layers.Dense(num_classes, activation="softmax", name="dense_5")(x)
    return Model(base.input, out)

def preprocess_img(img):
    x = np.array(img, dtype=np.float32)
    if PREPROCESS == "efficientnet":
        from tensorflow.keras.applications.efficientnet import preprocess_input
        x = preprocess_input(x)
    elif PREPROCESS == "/255":
        x = x / 255.0
    return np.expand_dims(x, 0)

def softmax_if_needed(v):
    v = np.asarray(v, np.float32)
    s = float(v.sum())
    if 0.98 <= s <= 1.02 and (v >= -1e-6).all() and (v <= 1.0+1e-6).all():
        return v
    e = np.exp(v - v.max()); return e / e.sum()

def layer_l2(model, lname):
    try:
        w, b = model.get_layer(lname).get_weights()
        return float(np.linalg.norm(w)), float(np.linalg.norm(b))
    except Exception:
        return None, None

def main():
    if not os.path.exists(WEIGHTS_H5):
        print(f"ERROR: weights file not found: {WEIGHTS_H5}")
        sys.exit(2)

    print("[1] Inferring head from H5 (robust scan)…")
    hidden_units, num_classes = infer_hidden_and_classes_from_h5(WEIGHTS_H5)
    print(f"    hidden_units (dense_4): {hidden_units}")
    print(f"    num_classes (dense_5):  {num_classes}")

    print("[2] Rebuilding model with matching layer names…")
    model = build_model(hidden_units, num_classes)

    print("[3] Loading weights by name…")
    model.load_weights(WEIGHTS_H5, by_name=True)

    k4, b4 = layer_l2(model, "dense_4")
    k5, b5 = layer_l2(model, "dense_5")
    print(f"[INFO] dense_4 L2={k4}, dense_5 L2={k5}")
    for lname in ["stem_conv", "block1a_dwconv", "block2a_dwconv"]:
        try:
            w0 = model.get_layer(lname).get_weights()[0]
            print(f"[INFO] {lname} kernel L2={float(np.linalg.norm(w0))}")
            break
        except Exception:
            pass

    if TEST_IMAGE and os.path.exists(TEST_IMAGE):
        img = Image.open(TEST_IMAGE).convert("RGB").resize(IMG_SIZE)
        x = preprocess_img(img)
        preds = model(x, training=False).numpy()[0]
        preds = softmax_if_needed(preds)
        top = np.argsort(-preds)[:3]
        print("Top-3:")
        for i in top:
            print(f"  {CLASS_NAMES[i]}: {float(preds[i]):.4f}")
    else:
        print("[INFO] TEST_IMAGE not set or missing; skipping prediction check.")

    print("[4] Converting to TFLite (float32)…")
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    tfl = conv.convert()
    open(OUT_TFLITE, "wb").write(tfl)
    import hashlib
    md5 = hashlib.md5(open(OUT_TFLITE, "rb").read()).hexdigest()
    print(f"[OK] Wrote {OUT_TFLITE} (md5: {md5})")

if __name__ == "__main__":
    main()
