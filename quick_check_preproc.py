import sys, numpy as np, tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.applications.efficientnet import preprocess_input

H5 = "dog_disease_detector.h5"   # your weights
TFL = "model.tflite"
IMG_SIZE = (224, 224)

def build_preproc():
    inp = layers.Input(shape=(224,224,3), name="input")
    x = layers.Lambda(preprocess_input, name="effnet_preprocess")(inp)
    base = tf.keras.applications.EfficientNetB0(include_top=False, input_tensor=x, weights=None)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d_2")(base.output)
    x = layers.Dropout(0.2, name="dropout_4")(x)
    x = layers.Dense(256, activation="relu", name="dense_4")(x)
    x = layers.Dropout(0.2, name="dropout_5")(x)
    out = layers.Dense(7, activation="softmax", name="dense_5")(x)
    return models.Model(inp, out)

def load_img(p):
    img = Image.open(p).convert("RGB").resize(IMG_SIZE)
    return np.asarray(img, np.float32)[None,...]

# Keras
km = build_preproc()
km.load_weights(H5, by_name=True, skip_mismatch=False)
def kpred(p): return km.predict(load_img(p), verbose=0)[0]

# TFLite
itp = tf.lite.Interpreter(model_path=TFL); itp.allocate_tensors()
inp = itp.get_input_details()[0]; out = itp.get_output_details()[0]
def tpred(p):
    x = load_img(p).astype(inp["dtype"])
    itp.set_tensor(inp["index"], x); itp.invoke()
    return itp.get_tensor(out["index"])[0]

def show(name, a, b):
    print(name)
    print(" img1:", np.round(a,6), " sum=", float(a.sum()), " std=", float(a.std()))
    print(" img2:", np.round(b,6), " sum=", float(b.sum()), " std=", float(b.std()))
    print(" ΔL1:", float(np.abs(a-b).sum()), " ΔLinf:", float(np.max(np.abs(a-b))))

p1, p2 = sys.argv[1], sys.argv[2]
k1, k2 = kpred(p1), kpred(p2)
t1, t2 = tpred(p1), tpred(p2)
show("KERAS(preproc)", k1, k2)
show("TFLITE(preproc)", t1, t2)
