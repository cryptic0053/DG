# inspect_h5.py
import sys, os
import h5py

p = sys.argv[1] if len(sys.argv) > 1 else None
if not p or not os.path.exists(p):
    print("Usage: python inspect_h5.py <path-to-h5>")
    sys.exit(1)

with h5py.File(p, "r") as f:
    print("File:", p)
    print("Top keys:", list(f.keys()))
    if "model_weights" in f:
        grp = f["model_weights"]
        print("\n[model_weights] subgroups (layers):")
        for k in grp.keys():
            print(" -", k)
    if "layer_names" in f:
        print("\n[layer_names] count:", len(f["layer_names"]))
    if "optimizer_weights" in f:
        print("\nHas optimizer weights (likely full-model save).")
