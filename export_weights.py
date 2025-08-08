import tensorflow as tf

# 1) Rebuild EXACT model the way you trained it
# If you added any extra layers (BatchNorm, extra Dense, different dropout), REPLICATE HERE.
from tensorflow.keras import layers, Model
base = tf.keras.applications.EfficientNetB0(include_top=False, weights=None, input_shape=(224,224,3))
x = layers.GlobalAveragePooling2D(name="gap")(base.output)
x = layers.Dropout(0.2, name="drop")(x)
out = layers.Dense(7, activation="softmax", name="pred")(x)
model = Model(base.input, out)

# 2) Load your trained weights/checkpoint (NOT the broken full-model H5)
# Example if you used ModelCheckpoint:
# model.load_weights("checkpoints/ckpt-final")
# or if you saved weights previously:
# model.load_weights("weights_best.h5")

# TODO: replace with your actual checkpoint path:
model.load_weights("PATH/TO/YOUR/REAL_CHECKPOINT")

# 3) Save clean weights file
model.save_weights("dog_disease_detector_fixed.h5")
print("Wrote dog_disease_detector_fixed.h5")
