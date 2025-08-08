import tensorflow as tf

H5_PATH = "dog_disease_detector.h5"   # your current file name
OUT_TFLITE = "model.tflite"           # what app.py expects

# Load the Keras model (TF 2.15 works with tf.keras.load_model)
model = tf.keras.models.load_model(H5_PATH, compile=False)

# Make a TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional: smaller file, faster on CPU (quantization-lite)
# Remove if it breaks accuracy or conversion.
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open(OUT_TFLITE, "wb") as f:
    f.write(tflite_model)

print(f"Wrote {OUT_TFLITE} ({len(tflite_model)/1e6:.2f} MB)")
