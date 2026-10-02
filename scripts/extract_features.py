"""IVP stage for every available set: detect -> measure -> crop.

Writes, per set:
  results/features/<set>.csv   one row per image (detection + colour features + seg success)
  results/crops/<set>.npz      crops (N,64,64,3) uint8 + image_ids, for the CNN
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vvm import segment as S
from vvm.datasets import load_set, available_sets, SYNTH_SETS


def process(name, root):
    df = load_set(name, root)
    rows, crops = [], []
    t0 = time.time()
    for _, r in tqdm(df.iterrows(), total=len(df), desc=name):
        if name in SYNTH_SETS:
            img = cv2.cvtColor(cv2.imread(r.path), cv2.COLOR_BGR2RGB)
        else:
            img = S.load_photo(r.path)
        det = S.detect(img)
        feat = S.measure(img, det) if det is not None else None
        row = {"image_id": r.image_id, "detected": int(det is not None),
               "method": det["method"] if det else "none"}
        if det:
            row.update({f"det_{k}": round(v, 2) for k, v in det.items() if k != "method"})
        if "gt_r" in df.columns and not pd.isna(r.get("gt_r", np.nan)):
            row["seg_ok"] = int(S.seg_correct(det, r.gt_cx, r.gt_cy, r.gt_r))
        else:
            row["seg_ok"] = np.nan   # no geometric ground truth for real photos
        row["measured"] = int(feat is not None)
        if feat:
            row.update(feat)
        rows.append(row)
        crops.append(S.crop(img, det))
    ms = (time.time() - t0) / max(len(df), 1) * 1000
    os.makedirs("results/features", exist_ok=True)
    os.makedirs("results/crops", exist_ok=True)
    pd.DataFrame(rows).to_csv(f"results/features/{name}.csv", index=False)
    np.savez_compressed(f"results/crops/{name}.npz", crops=np.stack(crops), ids=df.image_id.values)
    out = pd.DataFrame(rows)
    seg = out.seg_ok.mean() if out.seg_ok.notna().any() else float("nan")
    print(f"  {name}: n={len(df)} detected={out.detected.mean():.3f} seg_ok={seg:.3f} ({ms:.1f} ms/img)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--sets", nargs="*", default=None)
    a = ap.parse_args()
    for name in (a.sets or available_sets(a.root)):
        process(name, a.root)


if __name__ == "__main__":
    main()
