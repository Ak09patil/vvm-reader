"""IVP stage: locate the VVM disc, fit an ellipse, sample ring and square colours, crop.

Key design point: segmentation must NOT rely on square/ring contrast, because at the
discard point (s = 0.75) the square is identical to the ring. So we find the outer disc
(dark vs label background), fit an ellipse (handles perspective), and sample the square
and ring at fixed normalised radii inside that ellipse.
"""
import math
import numpy as np
import cv2

from . import config as C


def _candidates(gray):
    """Binary masks where the (darker) disc is foreground, from several thresholding methods."""
    n = gray.shape[0]
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    out = []
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    out.append(otsu)
    block = (n // 3) | 1
    out.append(cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 8))
    return out


def _best_ellipse(masks, shape):
    h, w = shape
    best, best_score = None, 0.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kb = (max(h, w) // 16) | 1
    k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kb, kb))
    for m in masks:
        # large opening first: cuts thin printed text / smudge bridges away from the disc
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_big)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 0.02 * h * w or len(c) < 20:
                continue
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            per = cv2.arcLength(hull, True)
            circ = 4 * math.pi * hull_area / (per * per + 1e-6)
            solidity = area / (hull_area + 1e-6)
            (cx, cy), (W, H), ang = cv2.fitEllipse(hull)
            if min(W, H) / max(W, H) < 0.4 or max(W, H) > 1.3 * max(h, w):
                continue
            # prefer round, solid, centred blobs (the target VVM is framed near the centre)
            dc = math.hypot(cx - w / 2, cy - h / 2) / (0.5 * math.hypot(w, h))
            score = circ * solidity * (1 - 0.5 * dc) * math.sqrt(area)
            if score > best_score:
                best_score, best = score, (cx, cy, W / 2, H / 2, ang)
    return best


def _hough(gray):
    n = min(gray.shape)
    c = cv2.HoughCircles(cv2.medianBlur(gray, 5), cv2.HOUGH_GRADIENT, dp=1.5, minDist=n,
                         param1=100, param2=30, minRadius=int(0.12 * n), maxRadius=int(0.6 * n))
    if c is None:
        return None
    x, y, r = c[0, 0]
    return (float(x), float(y), float(r), float(r), 0.0)


def detect(img_rgb):
    """Return dict(cx, cy, a, b, angle, method) or None. a,b are semi-axes in pixels."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    e = _best_ellipse(_candidates(gray), gray.shape)
    method = "contour"
    if e is None:
        e, method = _hough(gray), "hough"
    if e is None:
        return None
    cx, cy, a, b, ang = e
    return {"cx": cx, "cy": cy, "a": a, "b": b, "angle": ang, "method": method}


def _rho_map(shape, det):
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - det["cx"], yy - det["cy"]
    t = math.radians(det["angle"])
    u = (dx * math.cos(t) + dy * math.sin(t)) / max(det["a"], 1e-3)
    v = (-dx * math.sin(t) + dy * math.cos(t)) / max(det["b"], 1e-3)
    return np.sqrt(u * u + v * v)


def srgb_to_linear(x):
    x = x / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def measure(img_rgb, det):
    """Median colour of square region and ring annulus. Returns feature dict."""
    rho = _rho_map(img_rgb.shape, det)
    sq_m = rho < C.SQUARE_RHO
    ring_m = (rho > C.RING_RHO[0]) & (rho < C.RING_RHO[1])
    if sq_m.sum() < 10 or ring_m.sum() < 10:
        return None
    lin = srgb_to_linear(img_rgb.astype(np.float32))
    Y = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    lab = cv2.cvtColor(img_rgb.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    f = {"Y_sq": float(np.median(Y[sq_m])), "Y_ring": float(np.median(Y[ring_m]))}
    for i, ch in enumerate("Lab"):
        f[f"{ch}_sq"] = float(np.median(lab[..., i][sq_m]))
        f[f"{ch}_ring"] = float(np.median(lab[..., i][ring_m]))
    f["log_ratio"] = float(math.log((f["Y_sq"] + 1e-4) / (f["Y_ring"] + 1e-4)))
    return f


def crop(img_rgb, det, size=C.CROP_SIZE, margin=1.15):
    if det is None:
        return cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)
    half = margin * max(det["a"], det["b"])
    src = np.float32([[det["cx"] - half, det["cy"] - half], [det["cx"] + half, det["cy"] - half],
                      [det["cx"] - half, det["cy"] + half]])
    dst = np.float32([[0, 0], [size, 0], [0, size]])
    M = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(img_rgb, M, (size, size), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)


def seg_correct(det, gt_cx, gt_cy, gt_r):
    """Segmentation success: centre within 15% of radius, mean radius within 25%."""
    if det is None:
        return False
    err_c = math.hypot(det["cx"] - gt_cx, det["cy"] - gt_cy) / gt_r
    r = math.sqrt(det["a"] * det["b"])
    return err_c < 0.15 and abs(r - gt_r) / gt_r < 0.25


def load_photo(path, size=C.IMG_SIZE):
    """Real photo -> centre-cropped square RGB at IMG_SIZE (target VVM should be centred)."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    m = min(h, w)
    img = img[(h - m) // 2:(h - m) // 2 + m, (w - m) // 2:(w - m) // 2 + m]
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
