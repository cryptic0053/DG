# export_tflite_from_rebuild.py
import tensorflow as tf
from rebuild_and_check_weights import build_exact, H5

model = build_exact()
model.load_weights(H5, by_name=True, skip_mismatch=False)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = []  # keep exact FP32 weights
tfl = converter.convert()
open("model.tflite", "wb").write(tfl)
print("✅ wrote model.tflite")
