# Dog Skin-Disease Classifier

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Render](https://img.shields.io/badge/Render-46E3B7?style=flat-square&logo=render&logoColor=black)

A convolutional image classifier that identifies skin conditions in dogs from a
photograph, served as a web application.

The trained Keras model is also exported to **TensorFlow Lite**, so the same
classifier can run on-device rather than behind an API call.

## Model

| | |
|---|---|
| Task | Multi-class image classification |
| Format | Keras (`.h5`) with a TFLite export |
| Classes | Defined in `class_names.json` |
| Dataset | [Dog-Disease-Detection-Dataset](https://github.com/cryptic0053/Dog-Disease-Detection-Dataset) |

## Layout

```
app.py                      web application entry point
dog_disease_detector.h5     trained Keras model
class_names.json            class index -> label mapping
convert_to_tflite.py        Keras -> TFLite conversion
convert_patch_and_export.py model patching and export helper
Procfile / .render.yaml     deployment configuration
```

## Running locally

```bash
git clone https://github.com/cryptic0053/DG.git
cd DG

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python app.py
```

## Exporting to TFLite

```bash
python convert_to_tflite.py
```

Produces a quantised `.tflite` model suitable for mobile or embedded use.

## Deployment

Configured for [Render](https://render.com) via `.render.yaml` and `Procfile`.

## Disclaimer

This is a student machine-learning project, not a diagnostic tool. It is not a
substitute for examination by a qualified veterinarian.

---

[Anirban Argha](https://github.com/cryptic0053)
