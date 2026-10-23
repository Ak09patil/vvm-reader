"""Reading engine for the VVM Reader app.

photo -> locate VVM (auto-centre and zoom) -> photo-quality check -> measure square and ring
-> readers (WHO rule, classical, CNN, hybrid) -> one verdict: USE, DISCARD, RE-CHECK or RETAKE.
Uses the same vvm/ package, models and thresholds as the paper.
"""
import io
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vvm import config as C          # noqa: E402
from vvm import segment as S         # noqa: E402
from vvm.classical import who_rule, ClassicalReg  # noqa: E402

TARGET_RADIUS_FRAC = 0.34   # ring radius / image side used in training (0.28-0.40)
MAX_SIDE = 1600             # work on at most this many pixels per side

# Photo-quality gate (on the 256 px working image, inside the VVM). Calibrated so normal and
# training-range synthetic images pass and extreme blur / darkness / glare are sent back.
Q_MIN_RADIUS = 30           # px: smaller than this, too far away even after zoom
Q_MIN_SHARP = 1.2           # ring-edge sharpness (see sharpness()): training images 2.0-2.6,
                            # real printed-VVM photos 2.1-3.0, blur t=1.0 ~1.1, t=1.5 ~0.9
Q_MIN_RING_Y = 0.02         # linear luminance of the ring: darker than this is too dark
Q_MAX_GLARE = 0.06          # fraction of clipped pixels inside the VVM
WHO_ONLY_MARGIN = 0.05      # |r| below this is too close to call when only the WHO rule is available

READER_ORDER = ("CNN", "Hybrid", "Classical")


# ------------------------------------------------------------------ models
def load_models():
    """Return {name: model}. Missing files or missing torch are skipped, never fatal."""
    models = {}
    p = os.path.join(ROOT, "models", "classical.pkl")
    if os.path.exists(p):
        try:
            models["Classical"] = ClassicalReg.load(p)
        except Exception:
            pass
    try:
        import torch
        from vvm.model import TinyVVMNet
        for name, label in (("cnn", "CNN"), ("hybrid", "Hybrid")):
            p = os.path.join(ROOT, "models", f"{name}.pt")
            if os.path.exists(p):
                ck = torch.load(p, map_location="cpu", weights_only=False)
                m = TinyVVMNet(hybrid=ck["hybrid"])
                m.load_state_dict(ck["state"])
                m.eval()
                models[label] = m
    except Exception:
        pass
    return models


def load_deltas():
    out = {"Classical": 0.30, "CNN": 0.015, "Hybrid": 0.015}
    p = os.path.join(ROOT, "results", "metrics", "metrics.csv")
    if os.path.exists(p):
        try:
            import pandas as pd
            m = pd.read_csv(p)
            for key, label in (("classical", "Classical"), ("cnn", "CNN"), ("hybrid", "Hybrid")):
                d = m[(m.method == key) & m.delta.notna()].delta
                if len(d):
                    out[label] = float(d.iloc[0])
        except Exception:
            pass
    return out


def run_cnn(model, crop, r):
    import torch
    x = (crop.astype(np.float32) / 255.0 - 0.5) / 0.25
    x = torch.from_numpy(x.transpose(2, 0, 1)[None].copy())
    measured = 0.0 if np.isnan(r) else 1.0
    extra = torch.tensor([[float(np.clip(np.nan_to_num(r, nan=0.0), -3, 3)), measured]], dtype=torch.float32)
    with torch.no_grad():
        return float(model(x, extra)[0].item())


# ------------------------------------------------------------------ images
def decode(data):
    """Bytes from an upload or camera -> upright RGB uint8, longest side <= MAX_SIDE."""
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    img = np.array(im)
    h, w = img.shape[:2]
    k = MAX_SIDE / max(h, w)
    if k < 1:
        img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    return img


def square_crop(img, cx, cy, side, size=C.IMG_SIZE):
    """Square window centred on (cx, cy), resized to size x size; edges replicated if needed."""
    half = side / 2
    src = np.float32([[cx - half, cy - half], [cx + half, cy - half], [cx - half, cy + half]])
    dst = np.float32([[0, 0], [size, 0], [0, size]])
    M = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)


def locate(full):
    """Find the VVM and return (work_img_256, det, how). Tries the centred photo first, then
    closer centre crops; once found, re-crops so the VVM sits centred at training scale."""
    h, w = full.shape[:2]
    m = min(h, w)
    for zoom in (1.0, 1.6, 2.5, 4.0):
        side = m / zoom
        img = square_crop(full, w / 2, h / 2, side)
        det = S.detect(img)
        if det is None:
            continue
        k = side / C.IMG_SIZE                        # working px -> full-res px
        fx = w / 2 + (det["cx"] - C.IMG_SIZE / 2) * k
        fy = h / 2 + (det["cy"] - C.IMG_SIZE / 2) * k
        R = math.sqrt(det["a"] * det["b"]) * k        # ring radius in full-res px
        new_side = R / TARGET_RADIUS_FRAC
        if new_side < 0.9 * side or abs(det["cx"] - C.IMG_SIZE / 2) > 0.12 * C.IMG_SIZE \
                or abs(det["cy"] - C.IMG_SIZE / 2) > 0.12 * C.IMG_SIZE:
            img2 = square_crop(full, fx, fy, new_side)
            det2 = S.detect(img2)
            if det2 is not None and _near_centre(det2):
                return img2, det2, "centred and zoomed"
        return img, det, "as taken" if zoom == 1.0 else f"centre zoom {zoom:g}x"
    small = square_crop(full, w / 2, h / 2, m)
    return small, None, "not found"


def _near_centre(det):
    c = C.IMG_SIZE / 2
    return abs(det["cx"] - c) < 0.2 * C.IMG_SIZE and abs(det["cy"] - c) < 0.2 * C.IMG_SIZE


# ------------------------------------------------------------------ quality
def sharpness(img, det):
    """Mean gradient magnitude in a thin band on the ring's outer edge, as a % of the
    luminance step across it. Sharp edge -> high; blurred edge -> spread out -> low."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    g = np.sqrt(gx * gx + gy * gy)
    rho = S._rho_map(img.shape, det)
    edge = (rho > 0.93) & (rho < 1.07)
    inside = (rho > 0.80) & (rho < 0.90)
    outside = (rho > 1.15) & (rho < 1.35)
    if edge.sum() < 20 or inside.sum() < 20 or outside.sum() < 20:
        return 0.0
    step = abs(float(np.median(gray[outside])) - float(np.median(gray[inside])))
    if step < 8:
        return 0.0
    return float(np.percentile(g[edge], 90)) / step   # ~4 for a crisp edge, falls as blur spreads it


def quality(img, det, feats):
    """Return (metrics dict, list of problem keys). Empty list = photo is good enough."""
    if det is None:
        return {}, ["not_found"]
    rho = S._rho_map(img.shape, det)
    disc = rho < 1.0
    r = math.sqrt(det["a"] * det["b"])
    q = {"radius_px": r,
         "sharpness": sharpness(img, det),
         "ring_Y": feats["Y_ring"] if feats else 0.0,
         "glare": float((img[disc].max(axis=1) >= 250).mean()) if disc.any() else 0.0,
         "elongation": max(det["a"], det["b"]) / max(1e-3, min(det["a"], det["b"]))}
    probs = []
    if feats is None:
        probs.append("not_found")
    if r < Q_MIN_RADIUS:
        probs.append("too_small")
    if q["sharpness"] < Q_MIN_SHARP:
        probs.append("blurry")
    if q["ring_Y"] < Q_MIN_RING_Y:
        probs.append("too_dark")
    if q["glare"] > Q_MAX_GLARE:
        probs.append("glare")
    if q["elongation"] > 1.8:
        probs.append("tilted")
    return q, probs


# ------------------------------------------------------------------ reading
def read(full_rgb, models, deltas):
    """Full pipeline on one photo. Returns a result dict used by every page."""
    img, det, how = locate(full_rgb)
    feats = S.measure(img, det) if det else None
    r = feats["log_ratio"] if feats else float("nan")
    q, probs = quality(img, det, feats)

    readings = [{"reader": "WHO rule", "s_hat": None, "stage": None,
                 "decision": "DISCARD" if who_rule([r])[0] else "USE", "flag": False, "delta": None}]
    if det is not None and feats is not None:
        crop = S.crop(img, det)
        for name in ("Classical", "CNN", "Hybrid"):
            mdl = models.get(name)
            if mdl is None:
                continue
            sh = float(mdl.predict([r])[0]) if name == "Classical" else run_cnn(mdl, crop, r)
            readings.append({"reader": name, "s_hat": sh, "stage": int(C.stage_of(sh)),
                             "decision": "DISCARD" if C.is_discard(sh) else "USE",
                             "flag": abs(sh - C.DISCARD_S) <= deltas[name], "delta": deltas[name]})

    res = {"img": img, "det": det, "feats": feats, "r": r, "quality": q, "problems": probs,
           "readings": readings, "located": how}
    res.update(decide(res))
    return res


def decide(res):
    """One verdict from all readers.
    RETAKE   : VVM not found or photo too poor to trust (the network is confidently wrong
               under heavy blur/glare, so a margin flag alone cannot catch those).
    RE-CHECK : main reader within its calibrated margin of 0.75, or main reader and WHO rule disagree.
    USE / DISCARD : main reader's decision (discard if s_hat >= 0.75)."""
    rd = res["readings"]
    if res["det"] is None or res["feats"] is None or res["problems"]:
        return {"verdict": "RETAKE", "reason": res["problems"][0] if res["problems"] else "not_found",
                "main": None, "s_hat": None}
    who = rd[0]
    main = next((x for n in READER_ORDER for x in rd if x["reader"] == n), None)
    if main is None:                                 # no trained models available
        if abs(res["r"]) < WHO_ONLY_MARGIN:
            return {"verdict": "RE-CHECK", "reason": "near_threshold", "main": who, "s_hat": None}
        return {"verdict": who["decision"], "reason": "who_only", "main": who, "s_hat": None}
    if main["flag"]:
        return {"verdict": "RE-CHECK", "reason": "near_threshold", "main": main, "s_hat": main["s_hat"]}
    if main["decision"] != who["decision"]:
        return {"verdict": "RE-CHECK", "reason": "disagree", "main": main, "s_hat": main["s_hat"]}
    return {"verdict": main["decision"], "reason": "clear", "main": main, "s_hat": main["s_hat"]}


def overlay(img, det, scale=2):
    big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR).copy()
    if det is None:
        return big
    c = (int(det["cx"] * scale), int(det["cy"] * scale))
    for k, col, th in [(1.0, (230, 57, 70), 3), (C.SQUARE_RHO, (255, 196, 0), 2),
                       (C.RING_RHO[0], (255, 196, 0), 2), (C.RING_RHO[1], (255, 196, 0), 2)]:
        axes = (max(1, int(det["a"] * k * scale)), max(1, int(det["b"] * k * scale)))
        cv2.ellipse(big, c, axes, det["angle"], 0, 360, col, th, cv2.LINE_AA)
    return big


# ------------------------------------------------------------------ synthetic + reference images
def synthetic(s, axis="none", t=1.0, heldout=False, seed=7):
    from vvm import render, augment
    rng = np.random.default_rng(seed)
    style = render.sample_style(rng, heldout=heldout)
    img, gt = render.render_clean(s, style, rng)
    if axis != "none":
        ints = {ax: C.BASELINE_T for ax in augment.ORDER}
        ints[axis] = t
        img, gt, _ = augment.apply_all(img, gt, ints, rng)
    return img


def reference_vvm(s, size=220):
    """Clean picture of a VVM at stage s, for the guide."""
    from vvm import render
    style = {"hue": 235.0, "ring_L": 42.0, "ring_C": 30.0, "sq_L0": 92.0, "sq_C0": 3.0, "q": 0.44,
             "radius_frac": 0.40, "bg": "paper", "bg_L": 97.0, "bg_hue": 0.0, "bg_C": 0.0, "heldout": False}
    img, _ = render.render_clean(s, style, np.random.default_rng(0), plain_bg=True)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
