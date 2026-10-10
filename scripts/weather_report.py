"""Paper-ready numbers and figures from the district weather analysis (no downloads).
Reads results/weather_districts.csv + results/weather_monthly.csv.
Writes results/weather_report.txt, figures/fig_weather_map_paper.png, figures/fig_weather_monthly_paper.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

VVMS = ["VVM2", "VVM7", "VVM14", "VVM30"]
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
R = pd.read_csv("results/weather_districts.csv")
M = pd.read_csv("results/weather_monthly.csv")
pop = R.population_2011


def wq(x, w, q):
    """population-weighted quantile"""
    o = np.argsort(x.values)
    cw = np.cumsum(w.values[o]) / w.sum()
    return float(x.values[o][np.searchsorted(cw, q)])


L = [f"Districts: {len(R)}   population (Census 2011): {pop.sum()/1e6:.0f} M", ""]
L.append("Population-weighted national median: outreach sessions until a vial reaches its discard point")
piv = M.pivot(index="month", columns="vvm_type", values="pop_weighted_sessions_to_discard")[VVMS]
piv.index = [MON[m - 1] for m in piv.index]
L.append(piv.round(1).to_string())
L.append("")
for v in VVMS:
    c = R[f"{v}_sessions_to_discard_worst_month"]
    L.append(f"{v} worst-month sessions to discard, population-weighted quantiles: "
             f"p10={wq(c, pop, .10):.1f}  median={wq(c, pop, .50):.1f}  p90={wq(c, pop, .90):.1f}")
L.append("")
o = R["VVM2_outage72_discard_share"]
L.append("VVM2, 72 h cold-chain failure: share of days it would reach discard")
L.append(f"  population-weighted median {wq(o, pop, .5):.3f}, p90 {wq(o, pop, .9):.3f}, max {o.max():.3f}")
L.append(f"  population in districts where this exceeds 5% of days: {pop[o > .05].sum()/pop.sum():.1%}; "
         f"10% of days: {pop[o > .10].sum()/pop.sum():.1%}")
L.append(f"  VVM7 72 h failure ever reaches discard: {pop[R.VVM7_outage72_discard_share > 0].sum()/pop.sum():.1%} of population")
L.append("")
L.append("10 highest-risk districts (VVM2, fewest sessions to discard in worst month):")
top = R.sort_values("VVM2_sessions_to_discard_worst_month").head(10)
mm = {i + 1: m for i, m in enumerate(MON)}
for _, r in top.iterrows():
    L.append(f"  {r.district}, {r.state}: {r.VVM2_sessions_to_discard_worst_month} sessions "
             f"(worst month {mm[r.VVM2_worst_month]}), max temp {r.max_temp_c} C, 72h-failure share {r.VVM2_outage72_discard_share:.3f}")
L.append("")
st = R.assign(w=pop).groupby("state").apply(
    lambda g: pd.Series({"districts": len(g), "VVM2_worst_month_sessions_median": wq(g.VVM2_sessions_to_discard_worst_month, g.w + 1e-9, .5),
                         "VVM2_72h_share_median": wq(g.VVM2_outage72_discard_share, g.w + 1e-9, .5)}), include_groups=False)
st = st.sort_values("VVM2_worst_month_sessions_median")
st.to_csv("results/weather_by_state.csv", float_format="%.3f")
L.append("By state (population-weighted medians) -> results/weather_by_state.csv")
open("results/weather_report.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))

# ---- map figure (paper version)
fig, axs = plt.subplots(1, 2, figsize=(11, 6.2))
c = R.VVM2_sessions_to_discard_worst_month
sc = axs[0].scatter(R.Longitude, R.Latitude, c=c, s=9, cmap="viridis",
                    norm=LogNorm(vmin=max(1, c.quantile(.01)), vmax=c.quantile(.99)))
cb = fig.colorbar(sc, ax=axs[0], shrink=0.75)
t = [x for x in (2, 3, 4, 6, 10, 20, 30, 50, 100) if c.quantile(.01) <= x <= c.quantile(.99)]
cb.set_ticks(t)
cb.set_ticklabels([str(x) for x in t])
cb.minorticks_off()
cb.set_label("sessions to discard point (log scale)")
axs[0].set_title("(a) VVM2: outreach sessions until discard,\nworst month of 2024", fontsize=10)
sc = axs[1].scatter(R.Longitude, R.Latitude, c=R.VVM2_outage72_discard_share * 100, s=9, cmap="magma_r",
                    norm=Normalize(0, max(1, R.VVM2_outage72_discard_share.max() * 100)))
cb = fig.colorbar(sc, ax=axs[1], shrink=0.75)
cb.set_label("% of days in 2024")
axs[1].set_title("(b) VVM2: days on which a 72 h cold-chain\nfailure would reach the discard point", fontsize=10)
for ax in axs:
    ax.set_aspect("equal")
    ax.set_xlabel("longitude (°E)")
    ax.set_ylabel("latitude (°N)")
    ax.grid(alpha=0.2)
fig.text(0.5, 0.01, "Worst case: vial at ambient air temperature (ERA5 via Open-Meteo, 2024). "
         "Each point is one district.", ha="center", fontsize=8)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig("figures/fig_weather_map_paper.png", dpi=300)
plt.close(fig)

# ---- monthly figure (paper version)
fig, ax = plt.subplots(figsize=(7, 4))
for v in VVMS:
    ax.plot(range(1, 13), piv[v].values, marker="o", label=v)
ax.set_yscale("log")
ax.set_xticks(range(1, 13), MON)
ax.set_ylabel("outreach sessions until discard\n(population-weighted national median)")
ax.legend(title="VVM type", loc="upper left", bbox_to_anchor=(1.01, 1))
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("figures/fig_weather_monthly_paper.png", dpi=300)
plt.close(fig)
print("\nwrote results/weather_report.txt, results/weather_by_state.csv, figures/fig_weather_*_paper.png")
