import argparse, json, os, shutil, h5py, tensorflow as tf
from tensorflow import keras

def recursive_fix(obj):
    """Fix Keras H5 model_config for TF2.10 compatibility."""
    if isinstance(obj, dict):
        # batch_shape -> batch_input_shape
        if "batch_shape" in obj:
            obj["batch_input_shape"] = obj.pop("batch_shape")

        # dtype policy objects -> plain "float32"
        if "dtype" in obj and isinstance(obj["dtype"], dict):
            if obj["dtype"].get("class_name") == "DTypePolicy":
                obj["dtype"] = "float32"
            elif obj["dtype"].get("config", {}).get("name") == "float32":
                obj["dtype"] = "float32"

        # remove unknown args seen in newer Keras
        for bad in ("synchronized",):
            if bad in obj:
                obj.pop(bad)

        for k, v in list(obj.items()):
            obj[k] = recursive_fix(v)
    elif isinstance(obj, list):
        return [recursive_fix(v) for v in obj]
    return obj

def patch_h5(in_path, out_path):
    if os.path.abspath(in_path) != os.path.abspath(out_path):
        shutil.copyfile(in_path, out_path)
    with h5py.File(out_path, "r+") as f:
        raw = f.attrs.get("model_config")
        if raw is None:
            raise RuntimeError("model_config not found; not a Keras H5?")
        s = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        cfg = json.loads(s)
        fixed = recursive_fix(cfg)
        f.attrs.modify("model_config", json.dumps(fixed).encode("utf-8"))
    print("[patch] H5 model_config patched ->", out_path)

def maybe_prepend_rescale(model, img_size):
    """If the model lacks a Rescaling layer, prepend Rescaling(1/255)."""
    for l in model.layers:
        n = l.__class__.__name__.lower()
        if "rescaling" in n or "normalization" in n:
            return model
    x = keras.Input(shape=(img_size[0], img_size[1], 3), name="input")
    y = keras.layers.Rescaling(1./255., name="rescale")(x)
    y = model(y)
    print("[info] Prepending Rescaling(1/255) layer")
    return keras.Model(x, y, name=f"wrapped_{model.name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="dog_disease_detector.h5")
    ap.add_argument("--out", default="model.tflite")
    ap.add_argument("--img-size", type=int, nargs=2, default=(224,224))
    ap.add_argument("--add-rescale", action="store_true",
                    help="Prepend Rescaling(1/255) to the model.")
    args = ap.parse_args()

    fixed_h5 = os.path.splitext(args.h5)[0] + "_fixed.h5"
    patch_h5(args.h5, fixed_h5)

    print("[load] tf.keras loading", fixed_h5)
    model = keras.models.load_model(fixed_h5, compile=False)

    if args.add_rescale:
        model = maybe_prepend_rescale(model, tuple(args.img_size))

    print("[tflite] Exporting FP32 (no quantization)")
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = []  # keep weights exactly
    tfl = conv.convert()
    with open(args.out, "wb") as f:
        f.write(tfl)
    print(f"[done] wrote {args.out} ({len(tfl)/1e6:.2f} MB)")

if __name__ == "__main__":
    main()
