"""Turn raw print-and-capture photos into data/real_print/images + labels.csv.

Expected input layout (one folder per phone, one sub-folder per condition):
    data/real_print/raw/<phone>/<condition>/*.jpg
e.g. data/real_print/raw/iphone13/daylight/IMG_0412.jpg

Usage
  python scripts/prepare_real_print.py --assume-order
      Photos in each condition folder were shot in ID order 01..20. IDs are assigned by
      capture time (EXIF), falling back to filename order.
  python scripts/prepare_real_print.py
      No assumption: writes labels with an empty vvm_id column for you to fill in.

Either way it writes contact sheets (data/real_print/contact_sheets/) showing every photo
with the ID it was given, so you can check it against the printed "ID xx" in each photo.
"""
import argparse, os, re, shutil, sys
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

RAW = "data/real_print/raw"
OUT = "data/real_print/images"
SHEETS = "data/real_print/contact_sheets"
CONDITIONS = ["daylight", "tubelight", "warm_bulb", "shadow", "angled", "curved"]
ALIASES = {"day": "daylight", "daylight": "daylight", "natural": "daylight", "window": "daylight",
           "tube": "tubelight", "tubelight": "tubelight", "tube_light": "tubelight", "led": "tubelight",
           "white": "tubelight", "warm": "warm_bulb", "warm_bulb": "warm_bulb", "warmbulb": "warm_bulb",
           "bulb": "warm_bulb", "yellow": "warm_bulb", "shadow": "shadow", "shade": "shadow",
           "angled": "angled", "angle": "angled", "tilt": "angled", "tilted": "angled",
           "curved": "curved", "curve": "curved", "bottle": "curved"}
EXTS = (".jpg", ".jpeg", ".png")


def norm_condition(name):
    key = re.sub(r"[^a-z]+", "_", name.lower()).strip("_")
    if key in ALIASES:
        return ALIASES[key]
    for k, v in ALIASES.items():
        if k in key:
            return v
    return None


def capture_time(path):
    try:
        exif = Image.open(path).getexif()
        t = exif.get(36867) or exif.get_ifd(0x8769).get(36867) or exif.get(306)
        return str(t) if t else ""
    except Exception:
        return ""


def contact_sheet(rows, title, path):
    cols, tw = 5, 260
    th = int(tw * 1.3)
    n = len(rows)
    sheet = Image.new("RGB", (cols * tw, ((n + cols - 1) // cols) * (th + 30) + 50), "white")
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except OSError:
        font = ImageFont.load_default()
    d.text((10, 10), title, fill="black", font=font)
    for i, r in enumerate(rows):
        im = ImageOps.exif_transpose(Image.open(r["src"])).convert("RGB")
        im.thumbnail((tw - 10, th - 10))
        x, y = (i % cols) * tw, 50 + (i // cols) * (th + 30)
        sheet.paste(im, (x + 5, y))
        label = f"#{i + 1}  ->  ID {r['vvm_id']:02d}" if r["vvm_id"] else f"#{i + 1}  ->  ID ?"
        d.text((x + 8, y + th - 2), label, fill="red", font=font)
    sheet.save(path, quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assume-order", action="store_true")
    a = ap.parse_args()
    if not os.path.isdir(RAW):
        sys.exit(f"Put your photos in {RAW}/<phone>/<condition>/ first.")
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(SHEETS, exist_ok=True)
    rows, problems = [], []
    for phone in sorted(os.listdir(RAW)):
        pdir = os.path.join(RAW, phone)
        if not os.path.isdir(pdir) or phone.startswith("."):
            continue
        found = set()
        for cond_dir in sorted(os.listdir(pdir)):
            cdir = os.path.join(pdir, cond_dir)
            if not os.path.isdir(cdir) or cond_dir.startswith("."):
                continue
            cond = norm_condition(cond_dir)
            if cond is None:
                problems.append(f"{phone}/{cond_dir}: folder name not recognised as a condition")
                continue
            found.add(cond)
            files = [f for f in os.listdir(cdir) if not f.startswith(".")]
            heic = [f for f in files if f.lower().endswith(".heic")]
            if heic:
                problems.append(f"{phone}/{cond_dir}: {len(heic)} HEIC files - convert to JPEG first")
            imgs = sorted([f for f in files if f.lower().endswith(EXTS)],
                          key=lambda f: (capture_time(os.path.join(cdir, f)), f))
            if len(imgs) != 20:
                problems.append(f"{phone}/{cond_dir}: {len(imgs)} photos (expected 20)")
            group = []
            for i, f in enumerate(imgs):
                vid = i + 1 if (a.assume_order and len(imgs) == 20) else None
                ext = os.path.splitext(f)[1].lower().replace(".jpeg", ".jpg")
                new = f"{phone}_{cond}_{i + 1:02d}{ext}"
                shutil.copy2(os.path.join(cdir, f), os.path.join(OUT, new))
                r = {"filename": new, "vvm_id": vid, "condition": cond, "phone": phone,
                     "original": f"{phone}/{cond_dir}/{f}", "src": os.path.join(cdir, f)}
                rows.append(r)
                group.append(r)
            if group:
                contact_sheet(group, f"{phone} / {cond}  (check: red ID must match the printed ID in each photo)",
                              os.path.join(SHEETS, f"{phone}_{cond}.jpg"))
        missing = [c for c in CONDITIONS if c not in found]
        if missing:
            problems.append(f"{phone}: missing conditions {missing}")

    df = pd.DataFrame(rows).drop(columns="src")
    df["vvm_id"] = df.vvm_id.astype("Int64")
    path = "data/real_print/labels.csv"
    df.to_csv(path, index=False)
    print(f"{len(df)} photos copied to {OUT}/")
    print(f"labels -> {path}   contact sheets -> {SHEETS}/")
    if df.vvm_id.isna().any():
        print(f"\n{df.vvm_id.isna().sum()} photos have no vvm_id yet: open the contact sheets, read the printed ID "
              f"in each photo, and fill the vvm_id column in {path} (the #number on the sheet = row order).")
    for p in problems:
        print("  ! " + p)
    if not problems:
        print("No problems found.")


if __name__ == "__main__":
    main()
