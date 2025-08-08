# print_class_order.py
import tensorflow as tf
ds = tf.keras.preprocessing.image_dataset_from_directory(
    "E:/dog_disease_data", image_size=(224,224), batch_size=1, shuffle=False)
print(ds.class_names)   # <- copy this into app.py as CLASS_NAMES (same order!)
