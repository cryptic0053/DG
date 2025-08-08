import numpy as np, tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input  # if you trained EfficientNet
from PIL import Image

CLASS_NAMES = [
  "Bacterial_dermatosis","Dermatitis","Fungal_infections",
  "Healthy","Hypersensitivity_allergic_dermatosis","demodicosis","ringworm"
]

img_path = r"E:\dog_disease_data\ringworm\80b5eb_...8450e5.jpg"
m = tf.keras.models.load_model("dog_disease_detector.h5")  # <-- your trained model

img = Image.open(img_path).convert("RGB").resize((224,224))
x = np.array(img, dtype=np.float32)
x = preprocess_input(x)        # or x = x/255.0  (must match training)
x = np.expand_dims(x, 0)

preds = m(x, training=False).numpy()[0]
preds = tf.nn.softmax(preds).numpy() if preds.ndim==1 and preds.max()>1.0 else preds

top = np.argsort(-preds)[:3]
for i in top:
    print(CLASS_NAMES[i], float(preds[i]))
