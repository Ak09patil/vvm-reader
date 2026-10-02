"""Generate all synthetic sets. See docs/contract_schema.md for the metadata format.

Sets:
  train      3000  random degradations, train styles
  val         500  same distribution as train (used for calibration / model selection)
  benchmark  1500  clean cell + reference cell + 6 axes x 3 levels, 75 each, train styles
  style_in    300  reference-level degradation, train styles
  style_out   300  reference-level degradation, HELD-OUT styles (unseen hue / square size)
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vvm import config as C
from vvm.render import sample_style, render_clean
from vvm.augment import apply_all


def sample_s_random(rng):
    # 70% uniform over [0,1], 30% concentrated near the discard threshold
    return float(rng.uniform(0, 1) if rng.random() < 0.7 else rng.uniform(0.6, 0.9))


def stratified_s(n, rng):
    n_near = int(round(n * 0.4))
    n_uni = n - n_near
    a = (np.arange(n_uni) + rng.uniform(0, 1, n_uni)) / n_uni
    b = 0.6 + 0.3 * (np.arange(n_near) + rng.uniform(0, 1, n_near)) / n_near
    s = np.concatenate([a, b])
    rng.shuffle(s)
    return s


def random_intensities(rng):
    return {ax: (float(rng.uniform(0, 1)) if rng.random() < 0.6 else 0.0) for ax in C.AXES}


def make_one(set_name, idx, s, intens, rng, heldout, cell_axis="", cell_level=""):
    style = sample_style(rng, heldout=heldout)
    img, gt = render_clean(s, style, rng)
    img, gt, params = apply_all(img, gt, intens, rng)
    row = {
        "image_id": f"{set_name}_{idx:05d}",
        "set": set_name,
        "s": round(float(s), 5),
        "stage": int(C.stage_of(s)),
        "discard": int(C.is_discard(s)),
        "cell_axis": cell_axis,
        "cell_level": cell_level,
        "style_heldout": int(heldout),
        **{f"t_{ax}": round(float(intens.get(ax, 0.0)), 3) for ax in C.AXES},
        "gt_cx": round(gt["cx"], 2), "gt_cy": round(gt["cy"], 2), "gt_r": round(gt["r"], 2),
        "style_json": json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in style.items()}),
        "aug_json": json.dumps(params, default=lambda o: round(float(o), 4)),
    }
    return img, row


def write_set(out_dir, set_name, jobs):
    d = os.path.join(out_dir, set_name)
    os.makedirs(os.path.join(d, "images"), exist_ok=True)
    rows = []
    for img, row in tqdm(jobs, desc=set_name, total=None):
        cv2.imwrite(os.path.join(d, "images", row["image_id"] + ".png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(d, "metadata.csv"), index=False)
    print(f"  wrote {len(rows)} images -> {d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--per-cell", type=int, default=200)
    ap.add_argument("--n-style", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sets", nargs="*", default=["train", "val", "benchmark", "style_in", "style_out"])
    a = ap.parse_args()

    # independent RNG stream per set so sets never share samples
    seeds = np.random.SeedSequence(a.seed).spawn(5)
    R = [np.random.default_rng(s) for s in seeds]

    def gen_random(name, n, rng):
        for i in range(n):
            yield make_one(name, i, sample_s_random(rng), random_intensities(rng), rng, False)

    if "train" in a.sets:
        write_set(a.out, "train", gen_random("train", a.n_train, R[0]))
    if "val" in a.sets:
        write_set(a.out, "val", gen_random("val", a.n_val, R[1]))

    def gen_bench(rng):
        i = 0
        base = {ax: C.BASELINE_T for ax in C.AXES}
        cells = [("clean", "none", {ax: 0.0 for ax in C.AXES}), ("reference", "baseline", dict(base))]
        for ax in C.AXES:
            for lv, t in C.LEVELS.items():
                cells.append((ax, lv, {**base, ax: t}))
        for axis, level, intens in cells:
            for s in stratified_s(a.per_cell, rng):
                yield make_one("benchmark", i, s, intens, rng, False, axis, level)
                i += 1

    if "benchmark" in a.sets:
        write_set(a.out, "benchmark", gen_bench(R[2]))

    def gen_style(name, heldout, rng):
        base = {ax: C.BASELINE_T for ax in C.AXES}
        for i, s in enumerate(stratified_s(a.n_style, rng)):
            yield make_one(name, i, s, base, rng, heldout, "style", "heldout" if heldout else "seen")

    if "style_in" in a.sets:
        write_set(a.out, "style_in", gen_style("style_in", False, R[3]))
    if "style_out" in a.sets:
        write_set(a.out, "style_out", gen_style("style_out", True, R[4]))


if __name__ == "__main__":
    main()
