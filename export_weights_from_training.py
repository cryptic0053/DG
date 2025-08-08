import tensorflow as tf
from tensorflow.keras import layers, Model

# ==== REBUILD EXACT TRAINING MODEL ====
# If you used a different EfficientNet variant, different input size,
# extra Dense/Dropout/BatchNorm layers, change this to match EXACTLY.
IMG_SIZE = (224, 224)
NUM_CLASSES = 7

base = tf.keras.applications.EfficientNetB0(
    include_top=False, weights=None, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)
x = layers.GlobalAveragePooling2D(name="gap")(base.output)
x = layers.Dropout(0.2, name="drop")(x)
out = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)
model = Model(base.input, out)

# ==== LOAD YOUR TRAINED CHECKPOINT ====
# Replace this path with the checkpoint you actually saved during training:
# Examples:
# model.load_weights("checkpoints/ckpt-final")                 # TF checkpoint
# model.load_weights("weights_best.h5")                        # weights-only H5
# model.load_weights("model_epoch_XX.h5")                      # if you saved weights
model.load_weights("F:/DG/dog_disease_detector_fixed.h5")

# ==== EXPORT CLEAN WEIGHTS ====
model.save_weights("dog_disease_detector_fixed.h5")
print("Wrote dog_disease_detector_fixed.h5")
