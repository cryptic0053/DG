import os, re, shutil, h5py, tensorflow as tf

H5_IN  = "dog_disease_detector.h5"
H5_FIX = "dog_disease_detector_fixed.h5"
OUT    = "model.tflite"

def patch_model_config(in_path, out_path):
    # copy original → fixed file
    if os.path.abspath(in_path) != os.path.abspath(out_path):
        shutil.copyfile(in_path, out_path)

    with h5py.File(out_path, "r+") as f:
        raw = f.attrs.get("model_config")
        if raw is None:
            raise RuntimeError("model_config not found; not a Keras H5?")
        s = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)

        # 1) batch_shape -> batch_input_shape
        s = s.replace('"batch_shape"', '"batch_input_shape"')

        # 2) Replace any DTypePolicy object with a plain "dtype":"float32"
        #    Example to replace:
        #    "dtype":{"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},"registered_name":null}
        dtype_policy_pat = r'"dtype"\s*:\s*\{\s*"module"\s*:\s*"keras"\s*,\s*"class_name"\s*:\s*"DTypePolicy"[\s\S]*?\}'
        s_new = re.sub(dtype_policy_pat, '"dtype":"float32"', s)

        if s_new != s:
            print("Patched DTypePolicy → dtype:'float32'")
            s = s_new
        else:
            print("No DTypePolicy blocks found (or already simplified).")

        f.attrs.modify("model_config", s.encode("utf-8"))
        print("Patched model_config written.")

def convert_to_tflite(h5_path, out_path):
    # Load with tf.keras 2.10 (this env) then convert
    model = tf.keras.models.load_model(h5_path, compile=False)
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]  # optional; remove if accuracy shifts
    tfl = conv.convert()
    with open(out_path, "wb") as f:
        f.write(tfl)
    print(f"✅ Wrote {out_path} ({len(tfl)/1e6:.2f} MB)")

if __name__ == "__main__":
    patch_model_config(H5_IN, H5_FIX)
    convert_to_tflite(H5_FIX, OUT)
