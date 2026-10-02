"""Run every method on every available set and write all result files (docs/contract_results.md).

Methods:
  who_rule   literal WHO protocol on IVP features (binary decision only)
  classical  isotonic map log-ratio -> s (fitted on train)
  cnn        TinyVVMNet on crops            (if models/cnn.pt exists)
  hybrid     TinyVVMNet + IVP features      (if models/hybrid.pt exists)
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, cohen_kappa_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vvm import config as C
from vvm.classical import who_rule, ClassicalReg
from vvm.datasets import load_set, available_sets

OUT = "results"


# ------------------------------------------------------------------ helpers
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def calibrate_delta(s_pred, s_true, target=C.TARGET_UNSAFE_RATE):
    """Smallest margin around the discard threshold s.t. unsafe rate among unflagged <= target."""
    curve = []
    for d in np.arange(0, 0.301, 0.005):
        unfl = np.abs(s_pred - C.DISCARD_S) > d
        unsafe = C.is_discard(s_true) & ~C.is_discard(s_pred) & unfl
        rate = unsafe.sum() / max(unfl.sum(), 1)
        curve.append((round(d, 3), unfl.mean(), rate))
    curve = pd.DataFrame(curve, columns=["delta", "coverage", "unsafe_error_rate_unflagged"])
    ok = curve[curve.unsafe_error_rate_unflagged <= target]
    delta = float(ok.delta.iloc[0]) if len(ok) else float(curve.delta.iloc[-1])
    return delta, curve


def metrics(g, seg):
    n = len(g)
    r = {"n": n, "seg_success_rate": seg}
    has_s = g.s_pred.notna().all()
    if has_s:
        r["stage_acc"] = float((g.stage_pred == g.stage_true).mean())
        r["stage_f1_macro"] = float(f1_score(g.stage_true, g.stage_pred, labels=[1, 2, 3, 4],
                                             average="macro", zero_division=0))
    else:
        r["stage_acc"] = r["stage_f1_macro"] = np.nan
    correct = int((g.discard_pred == g.discard_true).sum())
    r["decision_acc"] = correct / n
    r["decision_acc_ci_lo"], r["decision_acc_ci_hi"] = wilson(correct, n)
    unsafe = (g.discard_true == 1) & (g.discard_pred == 0)
    r["unsafe_error_rate"] = float(unsafe.mean())
    r["false_discard_rate"] = float(((g.discard_true == 0) & (g.discard_pred == 1)).mean())
    r["mae"] = float((g.s_pred - g.s_true).abs().mean()) if (has_s and g.s_true.notna().all()) else np.nan
    unfl = ~g.flagged.astype(bool)
    r["coverage"] = float(unfl.mean())
    r["unsafe_error_rate_unflagged"] = float((unsafe & unfl).sum() / max(unfl.sum(), 1))
    return r


# ------------------------------------------------------------------ CNN
def load_cnn(name):
    path = f"models/{name}.pt"
    if not os.path.exists(path):
        return None
    import torch
    from vvm.model import TinyVVMNet
    ck = torch.load(path, map_location="cpu")
    m = TinyVVMNet(hybrid=ck["hybrid"])
    m.load_state_dict(ck["state"])
    m.eval()
    return m


def run_cnn(model, set_name):
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train import load_arrays, predict
    ids, x, extra, _, _ = load_arrays(set_name)
    return pd.Series(predict(model, x, extra, torch.device("cpu")), index=ids)


def cpu_latency_ms(model, reps=200):
    import torch
    x = torch.randn(1, 3, C.CROP_SIZE, C.CROP_SIZE)
    e = torch.zeros(1, 2)
    with torch.no_grad():
        for _ in range(20):
            model(x, e)
        t0 = time.time()
        for _ in range(reps):
            model(x, e)
    return (time.time() - t0) / reps * 1000


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    a = ap.parse_args()
    for d in ("predictions", "metrics"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)

    sets = [s for s in available_sets(a.root) if s != "train"]
    ftr = pd.read_csv(f"{OUT}/features/train.csv").set_index("image_id")
    mtr = load_set("train", a.root).set_index("image_id")
    clf = ClassicalReg().fit(ftr.log_ratio.values, mtr.loc[ftr.index, "s"].values)
    os.makedirs("models", exist_ok=True)
    clf.save("models/classical.pkl")

    cnns = {n: m for n, m in (("cnn", load_cnn("cnn")), ("hybrid", load_cnn("hybrid"))) if m is not None}
    print("methods: who_rule, classical" + "".join(", " + k for k in cnns))

    def predict_all(set_name):
        meta = load_set(set_name, a.root).set_index("image_id")
        feats = pd.read_csv(f"{OUT}/features/{set_name}.csv").set_index("image_id").loc[meta.index]
        preds = {"who_rule": None, "classical": pd.Series(clf.predict(feats.log_ratio.values), index=meta.index)}
        for k, m in cnns.items():
            preds[k] = run_cnn(m, set_name).loc[meta.index]
        return meta, feats, preds

    # 1) calibrate flagging margin per method on val
    meta_v, _, preds_v = predict_all("val")
    deltas, curves = {}, []
    for k, p in preds_v.items():
        if p is None:
            continue
        deltas[k], cv = calibrate_delta(p.values, meta_v.s.values)
        cv.insert(0, "method", k)
        curves.append(cv)
    pd.concat(curves).to_csv(f"{OUT}/metrics/coverage_risk_val.csv", index=False)
    print("calibrated delta:", deltas)

    # 2) every set
    all_metrics = []
    for set_name in sets:
        meta, feats, preds = predict_all(set_name)
        rows = []
        for k, p in preds.items():
            df = pd.DataFrame({"image_id": meta.index, "method": k,
                               "s_true": meta.s.values, "stage_true": meta.stage.values,
                               "discard_true": meta.discard.values,
                               "group_type": meta.group_type.values, "group": meta.group.values})
            if p is None:
                df["s_pred"] = np.nan
                df["stage_pred"] = np.nan
                df["discard_pred"] = who_rule(feats.log_ratio.values)
                df["flagged"] = 0
            else:
                df["s_pred"] = np.round(p.values, 5)
                df["stage_pred"] = C.stage_of(p.values)
                df["discard_pred"] = C.is_discard(p.values).astype(int)
                df["flagged"] = (np.abs(p.values - C.DISCARD_S) <= deltas[k]).astype(int)
            rows.append(df)
        pred = pd.concat(rows)
        pred.to_csv(f"{OUT}/predictions/{set_name}.csv", index=False)

        seg = feats.seg_ok if "seg_ok" in feats else pd.Series(np.nan, index=feats.index)
        for k, g in pred.groupby("method", sort=False):
            for grp, gg in [("ALL", g)] + list(g.groupby("group")):
                sr = seg.loc[gg.image_id].mean() if seg.notna().any() else np.nan
                r = {"set": set_name, "method": k,
                     "group_type": "all" if grp == "ALL" else gg.group_type.iloc[0], "group": grp,
                     **metrics(gg, sr), "delta": deltas.get(k, np.nan)}
                all_metrics.append(r)
        if set_name == "benchmark":
            curves = []
            for k, p in preds.items():
                if p is not None:
                    _, cv = calibrate_delta(p.values, meta.s.values)
                    cv.insert(0, "method", k)
                    curves.append(cv)
            pd.concat(curves).to_csv(f"{OUT}/metrics/coverage_risk_benchmark.csv", index=False)

    M = pd.DataFrame(all_metrics)
    M.to_csv(f"{OUT}/metrics/metrics.csv", index=False, float_format="%.4f")

    # 3) dual-annotator agreement on real_web
    if "real_web" in sets:
        w = load_set("real_web", a.root)
        rows = []
        for x, y in [("annotator_a", "annotator_b"), ("annotator_a", "source_stage"), ("annotator_b", "source_stage")]:
            rows.append({"pair": f"{x}_vs_{y}", "n": len(w),
                         "agreement_rate": float((w[x] == w[y]).mean()),
                         "cohen_kappa": float(cohen_kappa_score(w[x], w[y])),
                         "cohen_kappa_linear_weighted": float(cohen_kappa_score(w[x], w[y], weights="linear"))})
        pd.DataFrame(rows).to_csv(f"{OUT}/metrics/annotator_agreement.csv", index=False, float_format="%.4f")

    # 4) summary
    summary = {"deltas": deltas, "target_unsafe_rate": C.TARGET_UNSAFE_RATE, "sets": sets, "models": {}}
    for k, m in cnns.items():
        from vvm.model import n_params
        summary["models"][k] = {"params": n_params(m), "cpu_ms_per_crop": round(cpu_latency_ms(m), 3)}
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    show = M[M.group == "ALL"][["set", "method", "n", "seg_success_rate", "stage_acc", "decision_acc",
                                 "unsafe_error_rate", "mae", "coverage", "unsafe_error_rate_unflagged"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {OUT}/metrics/metrics.csv, {OUT}/predictions/*.csv, {OUT}/summary.json")


if __name__ == "__main__":
    main()
