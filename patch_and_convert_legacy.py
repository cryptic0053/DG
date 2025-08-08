import h5py, tensorflow as tf

H5_IN = "dog_disease_detector.h5"   # your original H5 file
H5_FIX = "dog_disease_detector_fixed.h5"
OUT = "model.tflite"

# --- Patch batch_shape -> batch_input_shape in model_config ---
with h5py.File(H5_IN, "r") as f:
    cfg = f.attrs.get("model_config")
if cfg is None:
    raise RuntimeError("model_config not found in H5; not a Keras H5 file?")
s = cfg.decode("utf-8") if isinstance(cfg, (bytes, bytearray)) else str(cfg)
s2 = s.replace('"batch_shape"', '"batch_input_shape"')

if s2 != s:
    import shutil
    shutil.copyfile(H5_IN, H5_FIX)
    with h5py.File(H5_FIX, "r+") as f:
        f.attrs.modify("model_config", s2.encode("utf-8"))
    h5_path = H5_FIX
    print("Patched batch_shape -> batch_input_shape")
else:
    h5_path = H5_IN
    print("No patch needed")

# --- Load with tf.keras 2.10 ---
model = tf.keras.models.load_model(h5_path, compile=False)

# --- Convert to TFLite ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # optional
tflite_model = converter.convert()

with open(OUT, "wb") as f:
    f.write(tflite_model)

print(f"✅ Wrote {OUT} ({len(tflite_model)/1e6:.2f} MB)")
