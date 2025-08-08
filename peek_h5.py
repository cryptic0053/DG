import json, h5py, sys, re
H5 = sys.argv[1] if len(sys.argv) > 1 else "dog_disease_detector.h5"

with h5py.File(H5, "r") as f:
    cfg_raw = f.attrs.get("model_config")
    if cfg_raw is None:
        raise SystemExit("No model_config in H5 (not a full Keras model).")
    cfg = json.loads(cfg_raw.decode("utf-8") if isinstance(cfg_raw, (bytes, bytearray)) else str(cfg_raw))

    # print basic model info
    print("== model keys:", list(cfg.keys()))
    if "class_name" in cfg: print("class_name:", cfg["class_name"])
    conf = cfg.get("config", {})
    layers = conf.get("layers", [])
    print("== num layers:", len(layers))

    # first 20 layers
    print("\n-- first 20 layers:")
    for i, L in enumerate(layers[:20]):
        print(f"{i:02d} {L.get('class_name')}  name={L.get('config',{}).get('name')}")

    # search for obvious backbones
    lname = " ".join((L.get("config",{}).get("name","") or "") for L in layers).lower()
    candidates = ["efficientnet", "mobilenet", "resnet", "inception", "xception", "densenet"]
    hits = [c for c in candidates if c in lname]
    print("\n-- backbone hits (by name search):", hits or "none")

    # rescaling / normalization presence
    has_rescale = any("rescaling" in (L.get("class_name","").lower()) or
                      "rescaling" in (L.get("config",{}).get("name","").lower())
                      for L in layers)
    has_norm = any("normalization" in (L.get("class_name","").lower()) or
                   "normalization" in (L.get("config",{}).get("name","").lower())
                   for L in layers)
    print("has Rescaling layer:", has_rescale, " | has Normalization layer:", has_norm)

    # try to infer classifier units
    dense_units = []
    for L in layers[::-1]:
        if L.get("class_name") in ("Dense","Conv2D"):
            confL = L.get("config",{})
            units = confL.get("units")
            filters = confL.get("filters")
            act = confL.get("activation")
            name = confL.get("name")
            if units: dense_units.append((name, units, act))
            elif filters: dense_units.append((name, filters, act))
            if len(dense_units) >= 3: break
    print("\n-- last Dense/Conv-ish layers found (name, units/filters, activation):")
    for t in dense_units:
        print("   ", t)

    # list top-level weight group names (prefixes give away the backbone)
    if "model_weights" in f:
        print("\n-- top-level model_weights groups:")
        for k in list(f["model_weights"].keys())[:15]:
            print("   ", k)
