"""Score the human readers against ground truth and against the models on the SAME images.

  python scripts/score_readers.py

Reads data/human_reading/key.csv and every CSV in data/human_reading/responses/.
Writes results/metrics/human_readers.csv and prints a summary.
"""
import glob, os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

D = "data/human_reading"


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def summarise(name, g, has_stage=True):
    n = len(g)
    ok = int((g.discard_pred == g.discard).sum())
    lo, hi = wilson(ok, n)
    thr = g[g.s == 0.75]
    near = g[(g.s >= 0.70) & (g.s <= 0.80)]
    return {"reader": name, "n": n,
            "stage_acc": float((g.stage_pred == g.stage).mean()) if has_stage else np.nan,
            "decision_acc": ok / n, "ci_lo": lo, "ci_hi": hi,
            "unsafe": int(((g.discard == 1) & (g.discard_pred == 0)).sum()),
            "false_discard": int(((g.discard == 0) & (g.discard_pred == 1)).sum()),
            "acc_threshold_vvm": float((thr.discard_pred == thr.discard).mean()) if len(thr) else np.nan,
            "n_threshold_vvm": len(thr),
            "acc_near_threshold_0.70-0.80": float((near.discard_pred == near.discard).mean()) if len(near) else np.nan,
            "mean_seconds": float(g.seconds.mean()) if "seconds" in g else np.nan}


def main():
    key = pd.read_csv(f"{D}/key.csv")
    files = sorted(glob.glob(f"{D}/responses/*.csv"))
    if not files:
        sys.exit(f"No reader files in {D}/responses/")
    rows, wide = [], key[["blind_id"]].copy()
    for f in files:
        r = pd.read_csv(f)
        name = str(r.reader.iloc[0])
        r = r.drop_duplicates("blind_id", keep="last")
        g = key.merge(r[["blind_id", "stage", "seconds"]].rename(columns={"stage": "stage_pred"}), on="blind_id")
        missing = len(key) - len(g)
        if missing:
            print(f"  ! {name}: {missing} images unanswered")
        g["discard_pred"] = (g.stage_pred >= 3).astype(int)
        rows.append(summarise(f"human:{name}", g))
        wide = wide.merge(g[["blind_id", "stage_pred"]].rename(columns={"stage_pred": name}), on="blind_id", how="left")

    # models on exactly the same photographs
    p = pd.read_csv("results/predictions/real_print.csv")
    key["image_id"] = "print_" + key.filename.str.rsplit(".", n=1).str[0]
    for m, g in p.groupby("method", sort=False):
        g = key.merge(g[["image_id", "discard_pred", "stage_pred"]], on="image_id")
        rows.append(summarise(f"model:{m}", g, has_stage=m != "who_rule"))

    res = pd.DataFrame(rows)
    os.makedirs("results/metrics", exist_ok=True)
    res.to_csv("results/metrics/human_readers.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 220)
    print(res.round(3).to_string(index=False))

    names = [c for c in wide.columns if c != "blind_id"]
    if len(names) >= 2:
        a, b = names[0], names[1]
        w = wide.dropna(subset=[a, b])
        da, db = (w[a] >= 3).astype(int), (w[b] >= 3).astype(int)
        agree = {"pair": f"{a} vs {b}", "n": len(w),
                 "stage_agreement": float((w[a] == w[b]).mean()),
                 "stage_kappa": cohen_kappa_score(w[a], w[b]),
                 "stage_kappa_linear": cohen_kappa_score(w[a], w[b], weights="linear"),
                 "decision_agreement": float((da == db).mean()),
                 "decision_kappa": cohen_kappa_score(da, db)}
        pd.DataFrame([agree]).to_csv("results/metrics/human_agreement.csv", index=False, float_format="%.4f")
        print("\nInter-reader agreement:")
        for k, v in agree.items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    print("\nwrote results/metrics/human_readers.csv" + (" and human_agreement.csv" if len(names) >= 2 else ""))


if __name__ == "__main__":
    main()
