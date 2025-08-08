# export_with_efficientnet_preprocess.py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.efficientnet import preprocess_input
from rebuild_and_check_weights import H5

def build_preproc():
    inp = layers.Input(shape=(224,224,3), name="input")
    x = layers.Lambda(preprocess_input, name="effnet_preprocess")(inp)  # EfficientNet’s own preprocessing
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, input_tensor=x, weights=None, pooling=None, classifier_activation=None
    )
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d_2")(base.output)
    x = layers.Dropout(0.2, name="dropout_4")(x)
    x = layers.Dense(256, activation="relu", name="dense_4")(x)
    x = layers.Dropout(0.2, name="dropout_5")(x)
    out = layers.Dense(7, activation="softmax", name="dense_5")(x)
    return models.Model(inp, out, name="rebuild_effnet_preproc")

m = build_preproc()
m.load_weights(H5, by_name=True, skip_mismatch=False)

conv = tf.lite.TFLiteConverter.from_keras_model(m)
conv.optimizations = []
open("model.tflite","wb").write(conv.convert())
print("wrote model.tflite (with EfficientNet preprocess)")
