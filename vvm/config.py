"""Shared constants. Anything here that changes must also change docs/contract_*.md."""
import numpy as np

# ---- Exposure stage contract (docs/contract_stage.md) ----
DISCARD_S = 0.75                    # square colour == ring colour exactly at this s
STAGE_EDGES = (0.40, 0.75, 0.85)    # stage1 <0.40 <= stage2 <0.75 <= stage3 <0.85 <= stage4
STAGE_NAMES = {1: "start", 2: "darkening", 3: "discard_point", 4: "beyond_discard"}

def stage_of(s):
    """Map continuous exposure stage s in [0,1] to WHO stage 1..4 (scalars or arrays)."""
    return np.digitize(s, STAGE_EDGES) + 1

def is_discard(s):
    return np.asarray(s) >= DISCARD_S

# ---- Image geometry ----
IMG_SIZE = 256        # final synthetic image is IMG_SIZE x IMG_SIZE RGB
RENDER_SCALE = 2      # render at 2x then downsample (anti-aliasing)
CROP_SIZE = 64        # classifier input crop

# ---- Style families (train vs held-out) ----
TRAIN_HUES = (215, 235, 275, 295)   # ring hue in LCh degrees
HELDOUT_HUES = (255, 315)           # 255 = interpolation, 315 = extrapolation
TRAIN_Q = (0.40, 0.47)              # square half-side / ring radius
HELDOUT_Q = (0.48, 0.52)

# ---- Degradation axes and severity ----
AXES = ("illumination", "noise", "blur", "jpeg", "geometric", "occlusion")
LEVELS = {"low": 0.33, "med": 0.66, "high": 1.0, "extreme": 1.5}  # extreme > training max (t<=1)
BASELINE_T = 0.2      # fixed mild baseline for the five non-varied axes in the benchmark

# ---- Measurement regions (normalised radius inside fitted ellipse) ----
SQUARE_RHO = 0.28           # square sampled at rho < 0.28 (inside square for any rotation, q>=0.40)
RING_RHO = (0.78, 0.92)     # ring sampled in this annulus (outside square corners for q<=0.52)

# ---- Confidence flagging ----
TARGET_UNSAFE_RATE = 0.01   # max unsafe-error rate among unflagged readings (calibrated on val)
