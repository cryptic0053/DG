import tensorflow as tf
import keras
from keras import layers, Model, Input

H5 = "dog_disease_detector.h5"
OUT = "model.tflite"
IMG = (224, 224, 3)

print("[load] keras.saving.load_model(safe_mode=False)")
m = keras.saving.load_model(H5, compile=False, safe_mode=False)

# If your training graph didn't have Rescaling, prepend it so server keeps RESCALE=1.0
has_rescale = any(l.__class__.__name__.lower()=="rescaling" for l in m.layers)
if not has_rescale:
    inp = Input(shape=IMG, name="input")
    x = layers.Rescaling(1/255., name="rescale")(inp)
    out = m(x)
    m = Model(inp, out, name=f"wrapped_{m.name}")
    print("[info] prepended Rescaling(1/255)")

print("[convert] exporting FP32 TFLite")
conv = tf.lite.TFLiteConverter.from_keras_model(m)
conv.optimizations = []  # keep exact FP32 weights
tfl = conv.convert()
open(OUT, "wb").write(tfl)
print(f"[done] wrote {OUT} ({len(tfl)/1e6:.2f} MB)")
