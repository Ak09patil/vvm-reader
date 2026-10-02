"""Clean (un-degraded) synthetic VVM rendering.

Colour model (see docs/contract_stage.md):
  ring colour  = fixed reference, LCh (L_r, C_r, h)
  square colour at s:
    s <= 0.75 : linear interpolation in LCh from start colour (L0, C0, h) to ring colour
    s >  0.75 : continues darker than the ring, L = L_r - DARKER_DL * u, u=(s-0.75)/0.25
"""
import math
import numpy as np
import cv2

from . import config as C

DARKER_DL = 15.0  # how much darker than the ring the square gets at s=1 (Lab L units)


def lch_to_rgb(L, Ch, h_deg):
    """LCh (CIELab, L 0..100) -> sRGB float in [0,1]."""
    h = math.radians(h_deg)
    lab = np.array([[[L, Ch * math.cos(h), Ch * math.sin(h)]]], np.float32)
    rgb = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)[0, 0]
    return np.clip(rgb, 0, 1)


def square_lch(s, style):
    s = float(np.clip(s, 0, 1))
    Lr, Cr, h = style["ring_L"], style["ring_C"], style["hue"]
    if s <= C.DISCARD_S:
        t = s / C.DISCARD_S
        L = style["sq_L0"] + (Lr - style["sq_L0"]) * t
        Ch = style["sq_C0"] + (Cr - style["sq_C0"]) * t
    else:
        u = (s - C.DISCARD_S) / (1 - C.DISCARD_S)
        L = Lr - DARKER_DL * u
        Ch = Cr * (1 - 0.3 * u)
    return L, Ch, h


def sample_style(rng, heldout=False):
    hues = C.HELDOUT_HUES if heldout else C.TRAIN_HUES
    qlo, qhi = C.HELDOUT_Q if heldout else C.TRAIN_Q
    return {
        "hue": float(rng.choice(hues) + rng.uniform(-4, 4)),
        "ring_L": float(rng.uniform(35, 50)),
        "ring_C": float(rng.uniform(22, 38)),
        "sq_L0": float(rng.uniform(86, 95)),
        "sq_C0": float(rng.uniform(0, 8)),
        "q": float(rng.uniform(qlo, qhi)),
        "radius_frac": float(rng.uniform(0.28, 0.40)),   # ring radius / image size
        "bg": str(rng.choice(["paper", "label_print", "tinted"])),
        "bg_L": float(rng.uniform(82, 97)),
        "bg_hue": float(rng.uniform(0, 360)),
        "bg_C": float(rng.uniform(0, 4)),
        "heldout": bool(heldout),
    }


def _background(size, style, rng):
    if style["bg"] == "tinted":
        base = lch_to_rgb(style["bg_L"], 12, style["bg_hue"])
    else:
        base = lch_to_rgb(style["bg_L"], style["bg_C"], style["bg_hue"])
    img = np.ones((size, size, 3), np.float32) * base
    # low-frequency paper texture
    tex = rng.normal(0, 1, (size // 16, size // 16)).astype(np.float32)
    tex = cv2.resize(tex, (size, size), interpolation=cv2.INTER_CUBIC)
    img *= (1 + 0.015 * tex)[..., None]
    if style["bg"] == "label_print":
        # fake printed text lines near the top/bottom of the label
        ink = lch_to_rgb(25, 5, rng.uniform(0, 360))
        for band in (rng.uniform(0.04, 0.12), rng.uniform(0.86, 0.94)):
            y = int(band * size)
            x = int(rng.uniform(0.05, 0.2) * size)
            while x < size * 0.92:
                w = int(rng.uniform(0.02, 0.07) * size)
                h = int(rng.uniform(0.02, 0.035) * size)
                cv2.rectangle(img, (x, y), (x + w, y + h), ink.tolist(), -1)
                x += w + int(0.015 * size)
    return img


def render_clean(s, style, rng, size=C.IMG_SIZE, plain_bg=False):
    """Return (uint8 RGB image, gt dict with cx, cy, r in final pixels)."""
    S = size * C.RENDER_SCALE
    if plain_bg:
        img = np.ones((S, S, 3), np.float32)
        off = np.zeros(2)
    else:
        img = _background(S, style, rng)
        off = rng.uniform(-0.05, 0.05, 2) * S
    cx, cy = S / 2 + off[0], S / 2 + off[1]
    R = style["radius_frac"] * S
    ring = lch_to_rgb(style["ring_L"], style["ring_C"], style["hue"])
    sq = lch_to_rgb(*square_lch(s, style))
    shift = 4  # sub-pixel precision for cv2 drawing
    f = 2 ** shift
    cv2.circle(img, (int(cx * f), int(cy * f)), int(R * f), ring.tolist(), -1, cv2.LINE_AA, shift)
    h = style["q"] * R
    pts = np.array([[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]])
    cv2.fillPoly(img, [np.round(pts * f).astype(np.int32)], sq.tolist(), cv2.LINE_AA, shift)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    k = size / S
    return img, {"cx": cx * k, "cy": cy * k, "r": R * k}
