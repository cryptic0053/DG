import os, shutil, json, h5py, tensorflow as tf

H5_IN  = "dog_disease_detector.h5"
H5_FIX = "dog_disease_detector_fixed.h5"
OUT    = "model.tflite"

def recursive_fix(obj):
    """Recursively fix model_config for TF 2.10 compatibility."""
    if isinstance(obj, dict):
        # Fix batch_shape -> batch_input_shape
        if "batch_shape" in obj:
            obj["batch_input_shape"] = obj.pop("batch_shape")

        # Fix DTypePolicy → float32
        if "dtype" in obj and isinstance(obj["dtype"], dict):
            if obj["dtype"].get("class_name") == "DTypePolicy":
                obj["dtype"] = "float32"

        # Remove unsupported args like 'synchronized'
        if "synchronized" in obj:
            obj.pop("synchronized")

        for k, v in list(obj.items()):
            obj[k] = recursive_fix(v)

    elif isinstance(obj, list):
        return [recursive_fix(i) for i in obj]
    return obj

def patch_model_config(in_path, out_path):
    if os.path.abspath(in_path) != os.path.abspath(out_path):
        shutil.copyfile(in_path, out_path)

    with h5py.File(out_path, "r+") as f:
        raw = f.attrs.get("model_config")
        if raw is None:
            raise RuntimeError("model_config not found; not a Keras H5?")
        s = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)

        cfg = json.loads(s)
        cfg = recursive_fix(cfg)

        fixed_json = json.dumps(cfg)
        f.attrs.modify("model_config", fixed_json.encode("utf-8"))
        print("Patched model_config safely.")

def convert_to_tflite(h5_path, out_path):
    # Try loading with minimal custom_objects to bypass unknown args
    model = tf.keras.models.load_model(
        h5_path,
        compile=False,
        custom_objects={}
    )
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    tfl = conv.convert()
    with open(out_path, "wb") as f:
        f.write(tfl)
    print(f"✅ Wrote {out_path} ({len(tfl)/1e6:.2f} MB)")

if __name__ == "__main__":
    patch_model_config(H5_IN, H5_FIX)
    convert_to_tflite(H5_FIX, OUT)
