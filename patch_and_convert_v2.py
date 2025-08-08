import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # force tf.keras 2.x inside TF 2.15

import h5py, json, shutil, tensorflow as tf

H5_IN  = "dog_disease_detector.h5"
H5_FIX = "dog_disease_detector_fixed.h5"
SM_DIR = "saved_model"
OUT    = "model.tflite"

def patch_all_batch_shape_keys(in_path, out_path):
    if in_path != out_path:
        shutil.copyfile(in_path, out_path)
    with h5py.File(out_path, "r+") as f:
        cfg = f.attrs.get("model_config")
        if cfg is None:
            raise RuntimeError("model_config not found in H5; not a Keras H5?")
        # decode -> str
        s = cfg.decode("utf-8") if isinstance(cfg, (bytes, bytearray)) else str(cfg)
        # global, JSON-safe key swap
        s2 = s.replace('"batch_shape"', '"batch_input_shape"')
        if s2 == s:
            print("No 'batch_shape' keys found to patch (maybe already OK).")
        else:
            print("Patched all 'batch_shape' → 'batch_input_shape' keys.")
        f.attrs.modify("model_config", s2.encode("utf-8"))

def convert_to_tflite(h5_path, sm_dir, out_path):
    # load with tf.keras (legacy) 2.x
    model = tf.keras.models.load_model(h5_path, compile=False)
    tf.saved_model.save(model, sm_dir)
    conv = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]  # optional, can remove
    tfl = conv.convert()
    with open(out_path, "wb") as f:
        f.write(tfl)
    print(f"Wrote {out_path} ({len(tfl)/1e6:.2f} MB)")

if __name__ == "__main__":
    patch_all_batch_shape_keys(H5_IN, H5_FIX)
    convert_to_tflite(H5_FIX, SM_DIR, OUT)
