# quick_check_keras_vs_tflite.py
import sys, numpy as np, tensorflow as tf
from PIL import Image
from rebuild_and_check_weights import build_exact, H5  # same file you used earlier

MODEL_TFLITE = "model.tflite"
IMG_SIZE = (224, 224)

def load_img(path):
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    x = np.asarray(img, dtype=np.float32)           # model has Rescaling(1/255) inside
    return x[None, ...]                             # [1, H, W, C]

# ----- KERAS (reference) -----
km = build_exact()
km.load_weights(H5, by_name=True, skip_mismatch=False)

def predict_keras(path):
    x = load_img(path)
    y = km.predict(x, verbose=0)[0]
    return y

# ----- TFLITE -----
def predict_tflite(path):
    interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE, num_threads=1)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    x = load_img(path).astype(inp["dtype"])         # dtype must match tflite input
    interpreter.set_tensor(inp["index"], x)
    interpreter.invoke()
    y = interpreter.get_tensor(out["index"])[0]
    return y

def show(title, a, b):
    print(title)
    print("  img1:", np.round(a, 6), " sum=", float(a.sum()), " std=", float(a.std()))
    print("  img2:", np.round(b, 6), " sum=", float(b.sum()), " std=", float(b.std()))
    print("  ΔL1:", float(np.abs(a - b).sum()), "  ΔLinf:", float(np.max(np.abs(a - b))))

def main():
    if len(sys.argv) != 3:
        print("Usage: python quick_check_keras_vs_tflite.py <img1> <img2>")
        sys.exit(2)

    p1, p2 = sys.argv[1], sys.argv[2]
    k1, k2 = predict_keras(p1), predict_keras(p2)
    t1, t2 = predict_tflite(p1), predict_tflite(p2)

    show("KERAS", k1, k2)
    show("TFLITE", t1, t2)

if __name__ == "__main__":
    main()
