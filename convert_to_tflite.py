# convert_h5_to_tflite.py
import argparse
import tensorflow as tf

def try_load(h5_path):
    # 1) Try Keras 3 loader (safe_mode=False lets it ignore unknown args/configs)
    try:
        from keras.saving import load_model as k3_load_model  # Keras 3
        print("[load] keras.saving.load_model(safe_mode=False)")
        return k3_load_model(h5_path, compile=False, safe_mode=False)
    except Exception as e:
        print("[load] Keras 3 loader failed:", e)

    # 2) Fallback to tf.keras loader
    try:
        print("[load] tf.keras.models.load_model")
        return tf.keras.models.load_model(h5_path, compile=False)
    except Exception as e:
        print("[load] tf.keras loader failed:", e)
        raise RuntimeError("Both loaders failed")

def has_rescaling_layer(model):
    for l in model.layers:
        n = l.__class__.__name__.lower()
        if "rescaling" in n or "normalization" in n:
            return True
    return False

def maybe_prepend_rescale(model, img_size):
    from tensorflow import keras
    inputs = keras.Input(shape=(img_size[0], img_size[1], 3), name="input")
    x = keras.layers.Rescaling(1./255., name="rescale")(inputs)
    y = model(x)
    wrapped = keras.Model(inputs, y, name=f"wrapped_{model.name}")
    print("[info] prepended Rescaling(1/255) to the model")
    return wrapped

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="dog_disease_detector.h5")
    ap.add_argument("--out", default="model.tflite")
    ap.add_argument("--img-size", type=int, nargs=2, default=(224, 224))
    ap.add_argument("--add-rescale", action="store_true",
                    help="Prepend Rescaling(1/255) if model lacks it.")
    args = ap.parse_args()

    model = try_load(args.h5)

    if args.add_rescale and not has_rescaling_layer(model):
        model = maybe_prepend_rescale(model, args.img_size)

    print("[convert] exporting FP32 TFLite (no quantization)")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = []  # keep exact FP32
    tfl = converter.convert()
    with open(args.out, "wb") as f:
        f.write(tfl)
    print(f"[done] wrote {args.out} ({len(tfl)/1e6:.2f} MB)")

if __name__ == "__main__":
    main()
