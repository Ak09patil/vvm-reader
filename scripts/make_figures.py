"""Paper figures from results/. Run after evaluate.py. Outputs PNG (300 dpi) in figures/."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vvm import config as C
from vvm.render import sample_style, render_clean
from vvm.augment import apply_all

os.makedirs("figures", exist_ok=True)
M = pd.read_csv("results/metrics/metrics.csv")
METHODS = [m for m in ["who_rule", "classical", "cnn", "hybrid"] if m in M.method.unique()]


def degradation_curves():
    b = M[(M.set == "benchmark") & (M.group_type == "cell")]
    xs = [C.BASELINE_T] + list(C.LEVELS.values())
    for metric, fname in [("decision_acc", "fig_degradation_accuracy.png"),
                          ("unsafe_error_rate", "fig_degradation_unsafe.png")]:
        fig, axs = plt.subplots(2, 3, figsize=(11, 6), sharey=True)
        for ax, axis in zip(axs.flat, C.AXES):
            for m in METHODS:
                ys = [b[(b.method == m) & (b.group == "reference:baseline")][metric].mean()]
                ys += [b[(b.method == m) & (b.group == f"{axis}:{lv}")][metric].mean() for lv in C.LEVELS]
                ax.plot(xs, ys, marker="o", label=m)
            ax.set_title(axis)
            ax.set_xticks(xs, ["ref"] + list(C.LEVELS))
            ax.grid(alpha=0.3)
        axs[0, 0].set_ylabel(metric)
        axs[1, 0].set_ylabel(metric)
        axs[0, 0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"figures/{fname}", dpi=300)
        plt.close(fig)


def coverage_risk():
    p = "results/metrics/coverage_risk_benchmark.csv"
    if not os.path.exists(p):
        return
    cr = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(5, 4))
    for m, g in cr.groupby("method"):
        ax.plot(g.coverage, g.unsafe_error_rate_unflagged, marker=".", label=m)
    ax.axhline(C.TARGET_UNSAFE_RATE, ls="--", c="gray", lw=1)
    ax.set_xlabel("coverage (fraction of readings not flagged)")
    ax.set_ylabel("unsafe-error rate among unflagged")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/fig_coverage_risk.png", dpi=300)
    plt.close(fig)


def real_print_bars():
    r = M[(M.set == "real_print") & (M.group_type == "condition")]
    if r.empty:
        return
    piv = r.pivot(index="group", columns="method", values="decision_acc")[METHODS]
    ax = piv.plot.bar(figsize=(8, 4), rot=30)
    ax.set_ylabel("decision accuracy")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/fig_real_print_conditions.png", dpi=300)
    plt.close()


def sample_grid():
    rng = np.random.default_rng(11)
    st = sample_style(rng)
    rows = [np.hstack([render_clean(s, st, rng)[0] for s in [0, 0.25, 0.5, 0.75, 0.85, 1.0]])]
    for axis in C.AXES:
        r = []
        for lv in [0.33, 0.66, 1.0, 0.33, 0.66, 1.0]:
            img, gt = render_clean(rng.uniform(0, 1), sample_style(rng), rng)
            r.append(apply_all(img, gt, {axis: lv}, rng)[0])
        rows.append(np.hstack(r))
    cv2.imwrite("figures/fig_samples.png", cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    degradation_curves()
    coverage_risk()
    real_print_bars()
    sample_grid()
    print("figures written to figures/")
