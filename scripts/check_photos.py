"""Quick sanity check for real photos: does segmentation find the VVM?

  python scripts/check_photos.py path/to/folder
Writes one overlay per photo into <folder>_check/ (red = detected disc,
green = square sampling region, yellow = ring sampling annulus).
"""
import glob, os, sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vvm import config as C
from vvm import segment as S

folder = sys.argv[1] if len(sys.argv) > 1 else "data/real_print/images"
out = folder.rstrip("/") + "_check"
os.makedirs(out, exist_ok=True)
files = sorted(f for f in glob.glob(os.path.join(folder, "*"))
               if f.lower().endswith((".jpg", ".jpeg", ".png")))
if not files:
    sys.exit(f"No .jpg/.jpeg/.png files in {folder}")
bad = 0
for p in files:
    img = S.load_photo(p)
    det = S.detect(img)
    name = os.path.basename(p)
    if det is None:
        print(f"FAIL  {name}: no disc found")
        bad += 1
        continue
    size = 2 * max(det["a"], det["b"]) / img.shape[1]
    off = np.hypot(det["cx"] - img.shape[1] / 2, det["cy"] - img.shape[0] / 2) / img.shape[1]
    warn = []
    if size > 0.85:
        warn.append("too close")
    if size < 0.30:
        warn.append("too far")
    if off > 0.15:
        warn.append("not centred")
    vis = img.copy()
    for r, col, t in ((1.0, (255, 0, 0), 2), (C.SQUARE_RHO, (0, 255, 0), 1),
                      (C.RING_RHO[0], (255, 255, 0), 1), (C.RING_RHO[1], (255, 255, 0), 1)):
        cv2.ellipse(vis, ((det["cx"], det["cy"]), (2 * det["a"] * r, 2 * det["b"] * r), det["angle"]), col, t)
    cv2.imwrite(os.path.join(out, os.path.splitext(name)[0] + ".png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(("WARN  " if warn else "OK    ") + f"{name}: disc = {size:.0%} of frame width"
          + (f"  ({'; '.join(warn)})" if warn else ""))
    bad += bool(warn)
print(f"\n{len(files) - bad}/{len(files)} photos OK. Overlays in {out}/")
