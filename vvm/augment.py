"""Six named degradation axes. Each takes intensity t in [0,1] (0 = off).

Applied in a fixed physical order:
  geometric -> illumination -> occlusion -> blur -> noise -> jpeg
Every function returns (img, params) and geometric also returns the updated ground-truth
circle (cx, cy, r) so segmentation can be scored.
"""
import math
import numpy as np
import cv2

ORDER = ("geometric", "illumination", "occlusion", "blur", "noise", "jpeg")


# (e) rotation / perspective / vial curvature ---------------------------------
def geometric(img, t, rng, gt):
    if t <= 0:
        return img, {}, gt
    n = img.shape[0]
    ang = rng.uniform(-180, 180) * t
    scale = 1 + rng.uniform(-0.15, 0.10) * t
    A = cv2.getRotationMatrix2D((gt["cx"], gt["cy"]), ang, scale)
    A = np.vstack([A, [0, 0, 1]])
    src = np.float32([[0, 0], [n, 0], [n, n], [0, n]])
    dst = src + rng.uniform(-1, 1, (4, 2)).astype(np.float32) * 0.12 * t * n
    P = cv2.getPerspectiveTransform(src, dst)
    H = P @ A
    out = cv2.warpPerspective(img, H, (n, n), borderMode=cv2.BORDER_REPLICATE)

    # cylindrical warp (label wrapped around a vial): image x = sin(theta*u_src)/sin(theta)
    theta = min(1.4, 1.2 * t * rng.uniform(0.5, 1.0))
    if theta > 1e-3:
        u = (np.arange(n, dtype=np.float32) - n / 2) / (n / 2)
        u_src = np.arcsin(np.clip(u * math.sin(theta), -1, 1)) / theta
        mapx = np.tile((u_src * n / 2 + n / 2).astype(np.float32), (n, 1))
        mapy = np.tile(np.arange(n, dtype=np.float32)[:, None], (1, n))
        out = cv2.remap(out, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # transform ground-truth circle: 32 boundary points through the same chain
    a = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    pts = np.stack([gt["cx"] + gt["r"] * np.cos(a), gt["cy"] + gt["r"] * np.sin(a)], 1)
    pts = np.vstack([[gt["cx"], gt["cy"]], pts]).astype(np.float32)
    pts = cv2.perspectiveTransform(pts[None], H)[0]
    if theta > 1e-3:
        us = (pts[:, 0] - n / 2) / (n / 2)
        pts[:, 0] = np.sin(theta * us) / math.sin(theta) * n / 2 + n / 2
    c = pts[0]
    r = float(np.mean(np.linalg.norm(pts[1:] - c, axis=1)))
    return out, {"rot_deg": ang, "scale": scale, "curv_theta": theta}, {
        "cx": float(c[0]), "cy": float(c[1]), "r": r}


# (a) illumination: colour-temperature cast + uneven shading + exposure ---------
def illumination(img, t, rng, gt=None):
    if t <= 0:
        return img, {}
    n = img.shape[0]
    f = img.astype(np.float32) / 255
    m = 0.30 * t * rng.uniform(0.5, 1.0)
    warm = bool(rng.random() < 0.5)
    gains = np.array([1 + m, 1.0, 1 - m] if warm else [1 - m, 1.0, 1 + m], np.float32)
    g = min(0.85, 0.6 * t * rng.uniform(0.5, 1.0))
    phi = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / n - 0.5
    proj = (xx * math.cos(phi) + yy * math.sin(phi)) / 0.7071
    shade = 1 - g * (0.5 + 0.5 * np.clip(proj, -1, 1))
    expo = 1 + rng.uniform(-0.30, 0.20) * t
    f = f * gains * shade[..., None] * expo
    return (np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8), {
        "cast": "warm" if warm else "cool", "cast_mag": m, "gradient": g, "exposure": expo}


# (f) smudge / partial occlusion / glare --------------------------------------
def occlusion(img, t, rng, gt):
    if t <= 0:
        return img, {}
    n = img.shape[0]
    f = img.astype(np.float32)
    cover = 0.25 * t
    n_blobs = int(rng.integers(1, 4))
    target_area = cover * math.pi * gt["r"] ** 2
    mask_total = np.zeros((n, n), np.float32)
    for _ in range(n_blobs):
        area = target_area / n_blobs
        ax = math.sqrt(area / math.pi) * rng.uniform(0.6, 1.6)
        ay = area / (math.pi * ax)
        d = rng.uniform(0, 1.0) * gt["r"]
        th = rng.uniform(0, 2 * np.pi)
        c = (int(gt["cx"] + d * math.cos(th)), int(gt["cy"] + d * math.sin(th)))
        mk = np.zeros((n, n), np.float32)
        cv2.ellipse(mk, c, (max(1, int(ax)), max(1, int(ay))), rng.uniform(0, 180), 0, 360, 1, -1)
        mk = cv2.GaussianBlur(mk, (0, 0), max(1, ax * 0.3))
        col = np.array(rng.uniform(60, 170, 3) * [1.0, 0.9, 0.75], np.float32)
        alpha = rng.uniform(0.5, 0.9)
        f = f * (1 - alpha * mk[..., None]) + col * alpha * mk[..., None]
        mask_total = np.maximum(mask_total, mk)
    glare = bool(rng.random() < 0.5)
    if glare:
        mk = np.zeros((n, n), np.float32)
        c = (int(gt["cx"] + rng.uniform(-0.6, 0.6) * gt["r"]), int(gt["cy"] + rng.uniform(-0.6, 0.6) * gt["r"]))
        ax = int(gt["r"] * rng.uniform(0.15, 0.45) * (0.5 + t))
        cv2.ellipse(mk, c, (max(1, ax), max(1, ax // 3)), rng.uniform(0, 180), 0, 360, 1, -1)
        mk = cv2.GaussianBlur(mk, (0, 0), max(1, ax * 0.4)) * 0.85 * t
        f = f * (1 - mk[..., None]) + 255 * mk[..., None]
    return np.clip(f, 0, 255).astype(np.uint8), {"cover_frac": cover, "n_blobs": n_blobs, "glare": glare}


# (c) blur: defocus or motion --------------------------------------------------
def blur(img, t, rng, gt=None):
    if t <= 0:
        return img, {}
    if rng.random() < 0.5:
        sig = 3.5 * t
        return cv2.GaussianBlur(img, (0, 0), sig), {"type": "defocus", "sigma": sig}
    L = int(round(1 + 12 * t))
    k = np.zeros((L, L), np.float32)
    k[L // 2, :] = 1
    ang = rng.uniform(0, 180)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D(((L - 1) / 2, (L - 1) / 2), ang, 1), (L, L))
    k /= max(k.sum(), 1e-6)
    return cv2.filter2D(img, -1, k), {"type": "motion", "length": L, "angle": ang}


# (b) sensor noise -------------------------------------------------------------
def noise(img, t, rng, gt=None):
    if t <= 0:
        return img, {}
    sig = 22 * t
    f = img.astype(np.float32)
    shot = rng.normal(0, 1, f.shape) * np.sqrt(np.maximum(f, 1)) * 0.6 * t   # signal-dependent
    read = rng.normal(0, sig, f.shape)                                       # signal-independent
    return np.clip(f + shot + read, 0, 255).astype(np.uint8), {"sigma": sig}


# (d) JPEG compression ---------------------------------------------------------
def jpeg(img, t, rng, gt=None):
    if t <= 0:
        return img, {}
    q = max(5, int(round(95 - 85 * t)))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), {"quality": q}


FUNCS = {"geometric": geometric, "illumination": illumination, "occlusion": occlusion,
         "blur": blur, "noise": noise, "jpeg": jpeg}


def apply_all(img, gt, intensities, rng):
    """intensities: dict axis -> t. Returns (img, gt, params)."""
    params = {}
    for ax in ORDER:
        t = float(intensities.get(ax, 0.0))
        if ax == "geometric":
            img, p, gt = geometric(img, t, rng, gt)
        else:
            img, p = FUNCS[ax](img, t, rng, gt)
        params[ax] = {"t": t, **p}
    return img, gt, params
