import sys, traceback
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

H5 = "dog_disease_detector.h5"
IMG = (224, 224, 3)
N_CLASSES = 7

def build_exact():
    inp = layers.Input(shape=IMG, name="input")
    x = layers.Rescaling(1./255.0, name="rescale")(inp)
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_tensor=x,
        weights=None,
        pooling=None,
        classifier_activation=None,
    )
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d_2")(base.output)
    x = layers.Dropout(0.2, name="dropout_4")(x)
    x = layers.Dense(256, activation="relu", name="dense_4")(x)
    x = layers.Dropout(0.2, name="dropout_5")(x)
    out = layers.Dense(N_CLASSES, activation="softmax", name="dense_5")(x)
    return models.Model(inp, out, name="rebuild_exact")

def main():
    m = build_exact()
    m.compile()  # no-op, just to be safe

    print("[info] trying strict load (by_name=True, skip_mismatch=False)")
    try:
        m.load_weights(H5, by_name=True, skip_mismatch=False)
        print("✅ Strict load succeeded (all layer names & shapes matched).")
    except Exception as e:
        print("❌ Strict load failed. First mismatch details below:")
        traceback.print_exc(limit=2)
        # Show a few sample layer names so we can align
        print("\n[model] first 25 layer names in rebuilt model:")
        for l in m.layers[:25]:
            print("  -", l.name, [tuple(w.shape.as_list()) for w in l.weights] if l.weights else [])
        print("\n[hint] The H5 contains a group named 'model_weights/efficientnetb0/...'.")
        print("      We need the rebuilt model to have IDENTICAL layer names for those sublayers.")
        sys.exit(1)

    # Quick sanity check: run a random input to ensure outputs differ
    x = np.random.rand(1, *IMG).astype("float32")
    y = m.predict(x, verbose=0)
    print("[sanity] output sum:", float(y.sum()), " argmax:", int(np.argmax(y)))

if __name__ == "__main__":
    main()
