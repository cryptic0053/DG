import os
import tensorflow as tf
from tensorflow.keras import layers, models

H5_WEIGHTS = "dog_disease_detector.h5"   # your file
OUT_TFLITE = "model.tflite"
INPUT_SHAPE = (224, 224, 3)
N_CLASSES = 7

def build_model():
    # Front-end rescaling to match your training (1./255)
    inputs = layers.Input(shape=INPUT_SHAPE, name="input")
    x = layers.Rescaling(1./255, name="rescale")(inputs)

    # Backbone
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, input_tensor=x, weights=None
    )
    # Head
    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = layers.Dropout(0.2, name="dropout")(x)  # harmless if not used in training
    outputs = layers.Dense(N_CLASSES, activation="softmax", name="pred")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="effnetb0_dogskin")
    return model

def main():
    model = build_model()
    # Try to load weights by name; skip mismatches for layers that don't exist
    try:
        model.load_weights(H5_WEIGHTS, by_name=True, skip_mismatch=True)
        print("Loaded weights by_name=True, skip_mismatch=True")
    except Exception as e:
        print("Could not load weights directly:", e)
        # Fallback: if the H5 is a full model, try opening and saving weights-only
        try:
            m2 = tf.keras.models.load_model(H5_WEIGHTS, compile=False)
            m2.save_weights("tmp_weights.h5")
            model.load_weights("tmp_weights.h5", by_name=True, skip_mismatch=True)
            print("Loaded from tmp_weights.h5 by name")
        except Exception as e2:
            raise RuntimeError(f"Failed to recover weights: {e2}")

    # Quick sanity: check final layer units
    if model.layers[-1].output_shape[-1] != N_CLASSES:
        raise RuntimeError(f"Head units mismatch (expected {N_CLASSES}).")

    # Export to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Light-size optimizations (remove if you want exact float32 only):
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(OUT_TFLITE, "wb") as f:
        f.write(tflite_model)
    print(f"✅ Wrote {OUT_TFLITE} ({len(tflite_model)/1e6:.2f} MB)")

if __name__ == "__main__":
    main()
