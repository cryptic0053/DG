# probe_backbone.py
import numpy as np, sys
from PIL import Image
import tensorflow as tf
from rebuild_and_check_weights import build_exact, H5

IMG = (224,224)
def load_img(p):
    x = Image.open(p).convert("RGB").resize(IMG)
    return np.asarray(x, np.float32)[None,...]  # [1,H,W,C]

# build the SAME model you exported (with Rescaling(1/255) inside)
m = build_exact()
m.load_weights(H5, by_name=True, skip_mismatch=False)

# Take tensor *before* GAP (i.e., the EfficientNet output)
feat_tensor = m.get_layer("global_average_pooling2d_2").input
probe = tf.keras.Model(inputs=m.input, outputs=feat_tensor)

p1, p2 = sys.argv[1], sys.argv[2]
x1, x2 = load_img(p1), load_img(p2)
f1, f2 = probe.predict(x1, verbose=0), probe.predict(x2, verbose=0)  # shape [1, 7, 7, 1280]

def stats(name, a):
    a = a[0]
    print(f"{name}: shape={a.shape}  mean={a.mean():.6f}  std={a.std():.6f}  min={a.min():.6f}  max={a.max():.6f}")

stats("feat img1", f1)
stats("feat img2", f2)
print("ΔL1:", float(np.abs(f1-f2).sum()), " ΔLinf:", float(np.max(np.abs(f1-f2))))
