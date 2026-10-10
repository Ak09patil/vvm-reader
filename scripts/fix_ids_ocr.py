"""Read the printed "ID xx" in every real photo (Apple Vision OCR) and fix labels.csv.

  pip install ocrmac
  python scripts/fix_ids_ocr.py

- Backs up labels.csv to labels_before_ocr.csv (only the first time).
- For each photo, finds the "ID xx" text closest to the image centre (the target VVM is
  centred, so neighbours' labels are further away) and uses it as vvm_id.
- Writes data/real_print/ocr_report.csv and lists every photo that still needs a human look:
  OCR found nothing, or an ID is duplicated / missing within a phone+condition folder.
- Redraws the contact sheets with the new IDs (orange = needs checking).
"""
import os, re, sys, tempfile
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    from ocrmac import ocrmac
except ImportError:
    sys.exit("Run:  pip install ocrmac   (macOS only)")

D = "data/real_print"
LABELS = f"{D}/labels.csv"
BACKUP = f"{D}/labels_before_ocr.csv"
IMAGES = f"{D}/images"
SHEETS = f"{D}/contact_sheets"
ID_RE = re.compile(r"(?:I|1|l|\|)\s*D\s*[:\-.]?\s*(\d{1,2})\b", re.I)
NUM_RE = re.compile(r"^\s*(\d{2})\s*$")


def read_id(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    im.thumbnail((1600, 1600))
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as t:
        im.save(t.name, quality=90)
        tmp = t.name
    try:
        ann = ocrmac.OCR(tmp).recognize()
    finally:
        os.remove(tmp)
    cands = []
    for text, conf, box in ann:
        m = ID_RE.search(text) or NUM_RE.match(text)
        if not m:
            continue
        v = int(m.group(1))
        if 1 <= v <= 20:
            x, y, w, h = box
            dist = ((x + w / 2 - 0.5) ** 2 + (y + h / 2 - 0.5) ** 2) ** 0.5
            cands.append((dist, v, float(conf), bool(ID_RE.search(text))))
    if not cands:
        return None, "", 0.0
    cands.sort(key=lambda c: (not c[3], c[0]))          # prefer real "ID xx" matches, then nearest centre
    seen = ",".join(str(c[1]) for c in cands)
    return cands[0][1], seen, cands[0][2]


def contact_sheet(rows, title, path):
    cols, tw = 5, 260
    th = int(tw * 1.3)
    sheet = Image.new("RGB", (cols * tw, ((len(rows) + cols - 1) // cols) * (th + 30) + 50), "white")
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except OSError:
        font = ImageFont.load_default()
    d.text((10, 10), title, fill="black", font=font)
    for i, r in enumerate(rows):
        im = ImageOps.exif_transpose(Image.open(os.path.join(IMAGES, r["filename"]))).convert("RGB")
        im.thumbnail((tw - 10, th - 10))
        x, y = (i % cols) * tw, 50 + (i // cols) * (th + 30)
        sheet.paste(im, (x + 5, y))
        vid = r["vvm_id"]
        label = f"{r['filename'][-6:-4]}: ID {int(vid):02d}" if pd.notna(vid) else f"{r['filename'][-6:-4]}: ID ?"
        d.text((x + 8, y + th - 2), label + ("  CHECK" if r["check"] else ""),
               fill=(230, 120, 0) if r["check"] else "red", font=font)
    sheet.save(path, quality=85)


def main():
    df = pd.read_csv(LABELS)
    if not os.path.exists(BACKUP):
        df.to_csv(BACKUP, index=False)
        print(f"backup -> {BACKUP}")
    old = df.vvm_id.astype("Int64").copy()
    ocr, seen, conf = [], [], []
    for i, r in df.iterrows():
        v, s, c = read_id(os.path.join(IMAGES, r.filename))
        ocr.append(v)
        seen.append(s)
        conf.append(round(c, 2))
        print(f"\r  OCR {i + 1}/{len(df)}", end="", flush=True)
    print()
    df["vvm_id_ocr"] = pd.array(ocr, dtype="Int64")
    df["ocr_seen"], df["ocr_conf"] = seen, conf
    df["vvm_id"] = df.vvm_id_ocr.fillna(old)

    df["check"] = df.vvm_id_ocr.isna()
    notes = []
    for (ph, cond), g in df.groupby(["phone", "condition"]):
        ids = g.vvm_id.dropna().astype(int)
        dup = ids[ids.duplicated(keep=False)]
        df.loc[dup.index, "check"] = True
        missing = sorted(set(range(1, 21)) - set(ids))
        if len(dup) or missing:
            notes.append(f"{ph}/{cond}: duplicated {sorted(set(dup))}  missing {missing}")

    changed = int((old != df.vvm_id).fillna(True).sum())
    df.drop(columns=["check"]).to_csv(LABELS, index=False)
    df[["filename", "phone", "condition", "vvm_id", "vvm_id_ocr", "ocr_seen", "ocr_conf", "check"]] \
        .to_csv(f"{D}/ocr_report.csv", index=False)
    for (ph, cond), g in df.groupby(["phone", "condition"]):
        contact_sheet(g.to_dict("records"), f"{ph} / {cond}   orange CHECK = look at this one",
                      os.path.join(SHEETS, f"{ph}_{cond}.jpg"))

    print(f"\nOCR read an ID in {df.vvm_id_ocr.notna().sum()}/{len(df)} photos; "
          f"{changed} IDs changed from the order-based guess.")
    todo = df[df.check]
    if len(todo):
        print(f"\n{len(todo)} photos need a manual look (open the contact sheets, orange labels):")
        for _, r in todo.iterrows():
            print(f"  {r.filename}: id={r.vvm_id if pd.notna(r.vvm_id) else '?'}  OCR saw [{r.ocr_seen}]")
    for n in notes:
        print("  ! " + n)
    if not len(todo) and not notes:
        print("Every folder has IDs 01-20 exactly once. Done.")


if __name__ == "__main__":
    main()
