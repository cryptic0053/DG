# rebuild_exact_and_export.py (TF 2.10 compatible)
import tensorflow as tf
from tensorflow.keras import layers, models

H5 = "dog_disease_detector.h5"   # your original file with weights
OUT = "model.tflite"
IMG_SIZE = (224, 224, 3)
N_CLASSES = 7

def build_exact():
    # Input + Rescaling so the server can keep RESCALE=1.0
    inp = layers.Input(shape=IMG_SIZE, name="input")
    x = layers.Rescaling(1./255.0, name="rescale")(inp)

    # EfficientNetB0 backbone — no 'name' kwarg in TF 2.10; default is "efficientnetb0"
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_tensor=x,
        weights=None,
        pooling=None,
        classifier_activation=None,
        # name="efficientnetb0",  # <- remove this line for TF 2.10
    )

    # Head with exact layer names from your H5
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d_2")(base.output)
    x = layers.Dropout(0.2, name="dropout_4")(x)
    x = layers.Dense(256, activation="relu", name="dense_4")(x)
    x = layers.Dropout(0.2, name="dropout_5")(x)
    out = layers.Dense(N_CLASSES, activation="softmax", name="dense_5")(x)

    model = models.Model(inp, out, name="dog_disease_detector_exact")
    return model

def main():
    model = build_exact()

    # Load weights by layer name from your full-model H5
    model.load_weights(H5, by_name=True, skip_mismatch=True)
    print("Loaded weights by_name with skip_mismatch=True")

    # Sanity-check: last layer should have 7 units
    assert model.layers[-1].output_shape[-1] == N_CLASSES

    # Export FP32 TFLite (no quantization)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = []  # keep exact FP32
    tfl = converter.convert()
    with open(OUT, "wb") as f:
        f.write(tfl)
    print(f"✅ Wrote {OUT} ({len(tfl)/1e6:.2f} MB)")

if __name__ == "__main__":
    main()
