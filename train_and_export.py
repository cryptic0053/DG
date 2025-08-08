import os, numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input

DATA_DIR = r"E:/dog_disease_data"
IMG_SIZE = (224,224)
BATCH = 32
EPOCHS_HEAD = 5
EPOCHS_FT = 15
FREEZE_UP_TO = -100   # unfreeze last 100 layers
OUT_H5 = "dog_disease_detector_refit.h5"
OUT_TFLITE = "model.tflite"

# 1) Data
train_ds = keras.preprocessing.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=1337,
    image_size=IMG_SIZE, batch_size=BATCH)
val_ds = keras.preprocessing.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=1337,
    image_size=IMG_SIZE, batch_size=BATCH)

class_names = train_ds.class_names
print("CLASS_NAMES:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
def aug():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])

def preprocess(x, y):
    x = tf.cast(x, tf.float32)
    x = preprocess_input(x)  # IMPORTANT for EfficientNet
    return x, y

train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE).cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds   = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

# Optional class weights for imbalance
def compute_class_weights(ds, n_classes):
    counts = np.zeros(n_classes, dtype=np.int64)
    for _, y in ds.unbatch():
        counts[int(y.numpy())] += 1
    total = counts.sum()
    w = total / (n_classes * np.maximum(counts, 1))
    return {i: float(w[i]) for i in range(n_classes)}
class_weights = compute_class_weights(
    keras.preprocessing.image_dataset_from_directory(
        DATA_DIR, image_size=IMG_SIZE, batch_size=1, shuffle=False),
    n_classes=len(class_names)
)
print("class_weights:", class_weights)

# 2) Model
inp = layers.Input(shape=IMG_SIZE+(3,), name="input")
x = aug()(inp)
x = layers.Lambda(preprocess_input, name="effnet_preprocess")(x)
base = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")
base.trainable = False  # warmup head
x = layers.GlobalAveragePooling2D(name="global_average_pooling2d_2")(base.output)
x = layers.Dropout(0.1, name="dropout_4")(x)
x = layers.Dense(256, activation="relu", name="dense_4")(x)
x = layers.Dropout(0.1, name="dropout_5")(x)
out = layers.Dense(len(class_names), activation="softmax", name="dense_5")(x)
model = keras.Model(inp, out)

# 3) Train head
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, class_weight=class_weights)

# 4) Fine-tune: unfreeze top of backbone
for layer in base.layers[:FREEZE_UP_TO]: layer.trainable = False
for layer in base.layers[FREEZE_UP_TO:]: layer.trainable = True

model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
cb = [
    keras.callbacks.ModelCheckpoint(OUT_H5, save_best_only=True, monitor="val_accuracy", mode="max"),
    keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
]
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FT, class_weight=class_weights, callbacks=cb)

# 5) Save final h5
model.save(OUT_H5)
print("Saved", OUT_H5)

# 6) Export TFLite with preprocess baked in (matches app)
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]  # optional
tfl = conv.convert()
with open(OUT_TFLITE, "wb") as f: f.write(tfl)
print("Saved", OUT_TFLITE)

# 7) Quick sanity
import PIL.Image as Image
def load_img(p): return np.asarray(Image.open(p).convert("RGB").resize(IMG_SIZE), np.float32)[None,...]
# Using model(inputs are raw RGB; preprocess is inside the graph)
