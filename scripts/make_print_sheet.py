"""Printable A4 sheet of 20 synthetic VVMs with known exposure stage, for print-and-capture.

Outputs (data/print_sheet/):
  print_sheet.pdf / .png     print at 100% scale ("Actual size"), colour, matte paper
  print_sheet_labels.csv     vvm_id -> s, stage, decision   (keep this away from annotators)
At s = 0.75 the square and ring have identical RGB, so they print identically: the discard
threshold survives the printer exactly even though absolute colours shift.
"""
import os, sys
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vvm import config as C
from vvm.render import render_clean

DPI = 300
A4 = (2480, 3508)
DIAM_MM = 15                     # printed disc diameter (real VVMs are ~7-10 mm; bigger = easier)
S_VALUES = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70,
            0.72, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90, 0.95, 1.00]
STYLE = {"hue": 275.0, "ring_L": 42.0, "ring_C": 32.0, "sq_L0": 92.0, "sq_C0": 3.0, "q": 0.44,
         "radius_frac": 0.40, "bg": "paper", "bg_L": 97, "bg_hue": 0, "bg_C": 0, "heldout": False}


def main(out="data/print_sheet", seed=7):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(S_VALUES))          # shuffled so stage order isn't visible
    sheet = Image.new("RGB", A4, "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 42)
    except OSError:
        font = ImageFont.load_default(size=42)
    diam_px = int(DIAM_MM / 25.4 * DPI)
    tile = int(diam_px / (2 * STYLE["radius_frac"]))   # render tile so the disc has diam_px
    cols, rows = 4, 5
    gx, gy = A4[0] // cols, (A4[1] - 300) // rows
    labels = []
    for k, idx in enumerate(order):
        s = S_VALUES[idx]
        vid = k + 1
        img, _ = render_clean(s, STYLE, rng, size=tile, plain_bg=True)
        r, c = divmod(k, cols)
        x0 = c * gx + (gx - tile) // 2
        y0 = 200 + r * gy + (gy - tile) // 2 - 30
        sheet.paste(Image.fromarray(img), (x0, y0))
        draw.text((c * gx + gx // 2, y0 + tile + 10), f"ID {vid:02d}", fill="black", font=font, anchor="mt")
        labels.append({"vvm_id": vid, "s": s, "stage": int(C.stage_of(s)), "discard": int(C.is_discard(s))})
    draw.text((A4[0] // 2, 80), f"VVM print-and-capture sheet - print at 100% scale, colour, matte. Disc = {DIAM_MM} mm",
              fill="black", font=font, anchor="mt")
    sheet.save(os.path.join(out, "print_sheet.png"), dpi=(DPI, DPI))
    sheet.save(os.path.join(out, "print_sheet.pdf"), resolution=DPI)
    pd.DataFrame(labels).to_csv(os.path.join(out, "print_sheet_labels.csv"), index=False)
    print(f"wrote {out}/print_sheet.pdf, print_sheet.png, print_sheet_labels.csv")


if __name__ == "__main__":
    main()
