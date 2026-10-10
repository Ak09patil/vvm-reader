"""Secondary analysis on real data: VVM heat-exposure risk for every Indian district, full year.

Real inputs
  - District list with population and coordinates: Census of India 2011 via the open
    IndianCities dataset (github.com/recurze/IndianCities, final_districts.csv). Districts
    with missing/invalid coordinates are re-located from OpenStreetMap district boundaries
    (Nominatim), falling back to Open-Meteo place search; the state must match. Any that
    cannot be placed are logged and excluded.
  - Hourly 2 m air temperature for each district for one full year: Open-Meteo historical
    archive API (ERA5 reanalysis), local time (Asia/Kolkata).

Kinetics: same two-point Arrhenius fit to WHO PQS VVM end-point times as weather_proxy.py.
Assumption (worst case, stated in the paper): vials sit at ambient air temperature.

Per district and VVM type we compute
  outreach   fraction of the VVM end-point budget consumed by one outreach session
             (08:00-18:00, vial outside the cold box), per day of the year
             -> sessions_to_discard = 1 / fraction (median over the year, and worst month).
             Relevant because the WHO multi-dose vial policy lets opened vials return to the
             cold chain and be reused across sessions while their VVM is still good.
  outage72   share of days on which a 72-hour cold-chain failure starting at midnight would
             push the vial past the VVM discard point.

Usage
  python scripts/weather_districts.py --limit 5          # quick test on 5 districts
  python scripts/weather_districts.py                    # all ~738 districts (cached, resumable)
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weather_proxy import END_POINT_DAYS, arrhenius_params, hours_to_end

DISTRICTS_URL = "https://raw.githubusercontent.com/recurze/IndianCities/master/final_districts.csv"
DISTRICTS_CSV = "data/india_districts.csv"
CACHE = "data/weather_cache"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
CLEAN_CSV = "data/india_districts_clean.csv"


def _valid(lat, lon):
    return 6 <= lat <= 37.5 and 68 <= lon <= 97.5          # India bounding box


GEOCODE_CACHE = "data/geocode_cache.csv"


def _names(district):
    """'Khandwa (East Nimar)' -> ['Khandwa', 'East Nimar']"""
    main = district.split("(")[0].strip()
    alt = district.split("(")[1].rstrip(")").strip() if "(" in district else None
    return [n for n in (main, alt) if n]


def _norm(x):
    return "".join(ch for ch in str(x).lower() if ch.isalnum())


def _pick(hits, names, state):
    """Choose a hit in the right state, preferring ones whose district (admin2) matches,
    then district-seat towns. Never accept an arbitrary village elsewhere in the state."""
    st = _norm(state)
    ok = [h for h in hits if h.get("country_code") == "IN" and st in _norm(h.get("admin1", ""))
          and _valid(h["latitude"], h["longitude"])]
    nn = [_norm(n) for n in names]
    for h in ok:                                         # 1) district itself matches
        if any(n and (n in _norm(h.get("admin2", "")) or _norm(h.get("admin2", "")) in n) for n in nn) \
                and _norm(h.get("admin2", "")):
            return h
    for h in ok:                                         # 2) district headquarters town
        if h.get("feature_code") in ("PPLA2", "PPLA", "ADM2"):
            return h
    return None


NOMINATIM = "https://nominatim.openstreetmap.org/search"
UA = {"User-Agent": "vvm-reader-research/1.0 (student symposium project)"}


def _nominatim(district, state):
    """OpenStreetMap district boundary (admin level ~5). Returns ("ok", (lat, lon)), ("nomatch", None) or ("error", None)."""
    for name in _names(district):
        for q in (f"{name} district, {state}, India", f"{name}, {state}, India"):
            for attempt in range(4):
                try:
                    r = requests.get(NOMINATIM, params={"q": q, "format": "jsonv2", "countrycodes": "in",
                                                        "addressdetails": 1, "limit": 10}, headers=UA, timeout=60)
                    time.sleep(1.1)                       # Nominatim policy: max 1 request / second
                    r.raise_for_status()
                    hits = r.json()
                    break
                except Exception as ex:  # noqa
                    wait = 5 * (attempt + 1)
                    print(f"    {district}: OSM network error ({type(ex).__name__}), retry in {wait}s")
                    time.sleep(wait)
            else:
                return "error", None
            st = _norm(state)
            good = [h for h in hits if st in _norm(h.get("address", {}).get("state", ""))
                    and _valid(float(h["lat"]), float(h["lon"]))]
            for h in good:                                # prefer the administrative district boundary
                if h.get("category") == "boundary" or h.get("type") == "administrative":
                    return "ok", (float(h["lat"]), float(h["lon"]))
            for h in good:                                # else its headquarters town/city
                if h.get("type") in ("city", "town"):
                    return "ok", (float(h["lat"]), float(h["lon"]))
    return "nomatch", None


def _geocode(district, state):
    """OpenStreetMap district boundary first, Open-Meteo place search as fallback."""
    st, hit = _nominatim(district, state)
    if st == "ok":
        return "ok", hit, "osm_boundary"
    st2, hit2 = _geocode_openmeteo(district, state)
    if st2 == "ok":
        return "ok", hit2, "open_meteo"
    return ("error" if "error" in (st, st2) else "nomatch"), None, None


def _geocode_openmeteo(district, state):
    """Open-Meteo geocoding. Returns ("ok", (lat, lon)), ("nomatch", None) or ("error", None)."""
    hits = []
    for name in _names(district):
        for attempt in range(5):
            try:
                r = requests.get(GEOCODE, params={"name": name, "count": 50, "countryCode": "IN"}, timeout=60)
                if r.status_code == 429:
                    time.sleep(30)
                    continue
                r.raise_for_status()
                hits += r.json().get("results", [])
                break
            except Exception as ex:  # noqa
                wait = 5 * (attempt + 1)
                print(f"    {district}: network error ({type(ex).__name__}), retry in {wait}s")
                time.sleep(wait)
        else:
            return "error", None
    h = _pick(hits, _names(district), state)
    return ("ok", (h["latitude"], h["longitude"])) if h else ("nomatch", None)


def load_districts():
    if os.path.exists(CLEAN_CSV):
        return pd.read_csv(CLEAN_CSV)
    if not os.path.exists(DISTRICTS_CSV):
        os.makedirs("data", exist_ok=True)
        r = requests.get(DISTRICTS_URL, timeout=60)
        r.raise_for_status()
        with open(DISTRICTS_CSV, "w") as f:
            f.write(r.text)
        print(f"downloaded district list -> {DISTRICTS_CSV}")
    d = pd.read_csv(DISTRICTS_CSV).dropna(subset=["Latitude", "Longitude"])
    d["Population"] = pd.to_numeric(d.Population, errors="coerce").fillna(0)   # post-2011 districts have none
    d["coord_source"] = "dataset"
    d["gkey"] = d.State + "|" + d.District

    # geocode cache: results survive re-runs, so only unresolved districts are queried again
    cache = pd.read_csv(GEOCODE_CACHE) if os.path.exists(GEOCODE_CACHE) else \
        pd.DataFrame(columns=["gkey", "status", "lat", "lon", "source"])
    done = dict(zip(cache.gkey, zip(cache.status, cache.lat, cache.lon, cache.source)))

    bad = [i for i, r in d.iterrows() if not _valid(r.Latitude, r.Longitude)]
    todo = [i for i in bad if d.at[i, "gkey"] not in done]
    print(f"{len(bad)} districts have invalid coordinates in the source file; "
          f"{len(bad) - len(todo)} already resolved, geocoding {len(todo)} ...")
    errors = 0
    for i in todo:
        status, hit, src = _geocode(d.at[i, "District"], d.at[i, "State"])
        print(f"    {d.at[i, 'District']}, {d.at[i, 'State']}: {status}" + (f" ({src})" if src else ""))
        if status == "error":
            errors += 1
            continue                                    # not cached -> retried next run
        done[d.at[i, "gkey"]] = (status, hit[0] if hit else None, hit[1] if hit else None, src)
        pd.DataFrame([(k, *v) for k, v in done.items()], columns=["gkey", "status", "lat", "lon", "source"]) \
            .to_csv(GEOCODE_CACHE, index=False)
    if errors:
        raise SystemExit(f"\n{errors} districts could not be geocoded because of network errors. "
                         f"Just run the same command again - resolved ones are cached.")

    dropped = []
    for i in bad:
        status, lat, lon, src = done[d.at[i, "gkey"]]
        if status == "ok":
            d.at[i, "Latitude"], d.at[i, "Longitude"], d.at[i, "coord_source"] = lat, lon, src
        else:
            dropped.append(i)
    if dropped:
        d.loc[dropped].to_csv("data/india_districts_dropped.csv", index=False)
        print(f"  no state-matching location found for {len(dropped)} districts "
              f"-> data/india_districts_dropped.csv (excluded)")
    d = d.drop(index=dropped).drop(columns="gkey")
    d["key"] = (d.State + "_" + d.District).str.replace(r"[^A-Za-z0-9]+", "_", regex=True)
    d = d.drop_duplicates("key").reset_index(drop=True)
    d.to_csv(CLEAN_CSV, index=False)
    print(f"  {len(d)} districts ready -> {CLEAN_CSV}")
    return d


def fetch_year(lat, lon, year, key, sleep):
    path = os.path.join(CACHE, f"{key}_{year}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["time"])
    q = {"latitude": lat, "longitude": lon, "start_date": f"{year}-01-01", "end_date": f"{year}-12-31",
         "hourly": "temperature_2m", "timezone": "Asia/Kolkata"}
    for attempt in range(12):
        try:
            r = requests.get(ARCHIVE, params=q, timeout=120)
        except requests.exceptions.RequestException as ex:
            wait = min(10 * (attempt + 1), 120)
            print(f"    network error ({type(ex).__name__}), retry in {wait}s")
            time.sleep(wait)
            continue
        if r.status_code == 429:                       # free-tier rate limit: wait and retry
            wait = 65 if attempt < 6 else 600
            print(f"    rate-limited, waiting {wait}s ...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        h = r.json()["hourly"]
        df = pd.DataFrame({"time": pd.to_datetime(h["time"]), "temp": h["temperature_2m"]}).dropna()
        os.makedirs(CACHE, exist_ok=True)
        df.to_csv(path, index=False)
        time.sleep(sleep)
        return df
    raise SystemExit("Weather download kept failing. Re-run the same command later; finished districts are cached.")


def analyse(w, vvms):
    w = w.set_index("time").sort_index()
    out = {"mean_temp_c": round(w.temp.mean(), 2), "max_temp_c": round(w.temp.max(), 2),
           "hours_ge_40c": int((w.temp >= 40).sum())}
    monthly = {}
    for v in vvms:
        A, B = arrhenius_params(v)
        rate = 1.0 / hours_to_end(w.temp.values, A, B)          # end-point fraction per hour
        r = pd.Series(rate, index=w.index)
        sess = r[(r.index.hour >= 8) & (r.index.hour < 18)].groupby(r.index[(r.index.hour >= 8) & (r.index.hour < 18)].date).sum()
        sess.index = pd.to_datetime(sess.index)
        by_month = sess.groupby(sess.index.month).median()
        out[f"{v}_session_frac_median"] = round(float(sess.median()), 5)
        out[f"{v}_sessions_to_discard_median"] = round(float(1 / sess.median()), 1)
        out[f"{v}_sessions_to_discard_worst_month"] = round(float(1 / by_month.max()), 1)
        out[f"{v}_worst_month"] = int(by_month.idxmax())
        roll = r.rolling(72).sum().shift(-71)                     # exposure of the next 72 h
        starts = roll[(roll.index.hour == 0)].dropna()
        out[f"{v}_outage72_discard_share"] = round(float((starts >= 1.0).mean()), 4)
        monthly[v] = by_month
    return out, monthly


def figures(res, monthly_tbl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs("figures", exist_ok=True)
    panels = [("VVM2_sessions_to_discard_worst_month", "VVM2: outreach sessions to discard point\n(worst month, fewer = riskier)", "viridis", True),
              ("VVM2_outage72_discard_share", "VVM2: share of days a 72 h cold-chain\nfailure would reach the discard point", "magma_r", False)]
    fig, axs = plt.subplots(1, 2, figsize=(11, 6))
    for ax, (col, title, cmap, log) in zip(axs, panels):
        if col not in res:
            ax.axis("off")
            continue
        vals = res[col].clip(lower=0.5) if log else res[col]
        sc = ax.scatter(res.Longitude, res.Latitude, c=np.log10(vals) if log else vals, s=10, cmap=cmap)
        cb = fig.colorbar(sc, ax=ax, shrink=0.7)
        if log:
            ticks = [1, 2, 5, 10, 20, 50, 100]
            cb.set_ticks(np.log10(ticks))
            cb.set_ticklabels([str(t) for t in ticks])
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig("figures/fig_weather_districts_map.png", dpi=300)
    plt.close(fig)

    if not monthly_tbl.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for v, g in monthly_tbl.groupby("vvm_type"):
            ax.plot(g.month, g.pop_weighted_sessions_to_discard, marker="o", label=v)
        ax.set_yscale("log")
        ax.set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
        ax.set_ylabel("outreach sessions to discard point\n(population-weighted median, log)")
        ax.legend()
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig("figures/fig_weather_monthly.png", dpi=300)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--vvm", nargs="+", default=["VVM2", "VVM7", "VVM14", "VVM30"], choices=list(END_POINT_DAYS))
    ap.add_argument("--limit", type=int, default=None, help="only the first N districts (testing)")
    ap.add_argument("--sleep", type=float, default=1.5, help="seconds between API calls")
    a = ap.parse_args()

    d = load_districts()
    if a.limit:
        d = d.head(a.limit)
    rows, mon = [], []
    for i, r in d.iterrows():
        print(f"[{i+1}/{len(d)}] {r.District}, {r.State}")
        w = fetch_year(r.Latitude, r.Longitude, a.year, r.key, a.sleep)
        stats, monthly = analyse(w, a.vvm)
        rows.append({"state": r.State, "district": r.District, "population_2011": r.Population,
                     "Latitude": r.Latitude, "Longitude": r.Longitude, **stats})
        for v, s in monthly.items():
            for m, frac in s.items():
                mon.append({"district": r.key, "pop": r.Population, "vvm_type": v, "month": m, "frac": frac})

    res = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    res.to_csv("results/weather_districts.csv", index=False)

    # population-weighted national monthly summary
    M = pd.DataFrame(mon)
    out = []
    for (v, m), g in M.groupby(["vvm_type", "month"]):
        g = g.sort_values("frac")
        cw = g["pop"].cumsum() / g["pop"].sum()
        med = g.frac.values[np.searchsorted(cw.values, 0.5)]
        out.append({"vvm_type": v, "month": m, "pop_weighted_median_session_frac": round(med, 5),
                    "pop_weighted_sessions_to_discard": round(1 / med, 1)})
    monthly_tbl = pd.DataFrame(out)
    monthly_tbl.to_csv("results/weather_monthly.csv", index=False)

    # national headline numbers
    pop = res.population_2011.sum()
    lines = [f"districts analysed: {len(res)}  (population covered, Census 2011: {pop/1e6:.0f} M)"]
    for v in a.vvm:
        c = f"{v}_sessions_to_discard_worst_month"
        share = res.loc[res[c] <= 10, "population_2011"].sum() / pop
        o = f"{v}_outage72_discard_share"
        share_o = res.loc[res[o] > 0, "population_2011"].sum() / pop
        lines.append(f"{v}: population share in districts where <=10 outreach sessions reach discard (worst month): "
                     f"{share:.1%}; where a 72 h failure can reach discard on some day: {share_o:.1%}")
    open("results/weather_summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    figures(res, monthly_tbl)
    print("wrote results/weather_districts.csv, weather_monthly.csv, weather_summary.txt, figures/fig_weather_*.png")


if __name__ == "__main__":
    main()
