"""Secondary analysis: weather-based heat-exposure risk triage (NOT a measurement).

For each logged trip (location + time window) we fetch hourly ambient temperature and
integrate VVM heat-exposure kinetics, assuming the WORST CASE that the vial sat at ambient
air temperature the whole time (no cold box). Output is a ranking of which vials to inspect
first, never a replacement for reading the VVM or a thermometer.

Kinetics: two-point Arrhenius fit, ln t_end(T) = A + B / T(K), using WHO PQS VVM end-point
times. Fraction consumed = sum(dt / t_end(T_i)); end point (fraction 1) = discard point, so
s_est = 0.75 * fraction, capped at 1.
!! Partner: verify END_POINT_DAYS against the current WHO PQS VVM performance specification
   (E06/IN05) and cite it. These are the commonly cited values.

Input CSV (--trips): trip_id,lat,lon,start,end   (ISO datetimes, local time)
Usage:
  python scripts/weather_proxy.py --trips data/trips.csv --vvm VVM7
  python scripts/weather_proxy.py --demo            # offline self-test, no network
Sources: Open-Meteo archive API (free, no key); recent days fall back to Open-Meteo forecast
API (past_days); Visual Crossing used only if env VISUALCROSSING_KEY is set and both fail.
"""
import argparse, math, os, sys
import numpy as np
import pandas as pd
import requests

# (temperature C, days to end point) pairs used for the two-point Arrhenius fit
END_POINT_DAYS = {
    "VVM30": [(37, 30), (25, 193)],
    "VVM14": [(37, 14), (25, 90)],
    "VVM7": [(37, 7), (25, 45)],
    "VVM2": [(37, 2), (5, 225)],
}


def arrhenius_params(vvm):
    (T1, d1), (T2, d2) = END_POINT_DAYS[vvm]
    x1, x2 = 1 / (T1 + 273.15), 1 / (T2 + 273.15)
    B = (math.log(d1) - math.log(d2)) / (x1 - x2)
    A = math.log(d1) - B * x1
    return A, B


def hours_to_end(temp_c, A, B):
    return np.exp(A + B / (np.asarray(temp_c) + 273.15)) * 24.0


def exposure(temps_c, vvm, dt_hours=1.0):
    A, B = arrhenius_params(vvm)
    frac = float(np.sum(dt_hours / hours_to_end(temps_c, A, B)))
    return frac, min(1.0, 0.75 * frac)


def fetch_open_meteo(lat, lon, start, end):
    s, e = start.date().isoformat(), end.date().isoformat()
    q = {"latitude": lat, "longitude": lon, "hourly": "temperature_2m", "timezone": "auto",
         "start_date": s, "end_date": e}
    for url in ("https://archive-api.open-meteo.com/v1/archive", "https://api.open-meteo.com/v1/forecast"):
        try:
            r = requests.get(url, params=q, timeout=30)
            r.raise_for_status()
            h = r.json()["hourly"]
            df = pd.DataFrame({"time": pd.to_datetime(h["time"]), "temp": h["temperature_2m"]}).dropna()
            if len(df):
                return df, url.split("//")[1].split("/")[0]
        except Exception as ex:  # noqa
            print(f"  {url}: {ex}")
    key = os.environ.get("VISUALCROSSING_KEY")
    if key:
        url = (f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
               f"{lat},{lon}/{s}/{e}")
        r = requests.get(url, params={"unitGroup": "metric", "include": "hours", "key": key}, timeout=30)
        r.raise_for_status()
        rows = [(f"{d['datetime']}T{h['datetime']}", h["temp"]) for d in r.json()["days"] for h in d["hours"]]
        return pd.DataFrame({"time": pd.to_datetime([a for a, _ in rows]), "temp": [b for _, b in rows]}), "visualcrossing"
    raise RuntimeError("no weather source returned data")


def triage(s_est):
    return "priority_inspect" if s_est >= 0.60 else ("inspect" if s_est >= 0.40 else "routine")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips", default="data/trips.csv")
    ap.add_argument("--vvm", default="VVM7", choices=list(END_POINT_DAYS))
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()

    if a.demo:
        for vvm in END_POINT_DAYS:
            for T in (5, 25, 37):
                print(f"{vvm} @ {T:>2} C: end point after {hours_to_end(T, *arrhenius_params(vvm))/24:8.1f} days")
        f, s = exposure(np.full(48, 37.0), "VVM7")
        print(f"demo: 48 h at 37 C on VVM7 -> fraction {f:.3f}, s_est {s:.3f}, {triage(s)}")
        return

    trips = pd.read_csv(a.trips, parse_dates=["start", "end"])
    out = []
    for _, t in trips.iterrows():
        w, src = fetch_open_meteo(t.lat, t.lon, t.start, t.end)
        w = w[(w.time >= t.start) & (w.time <= t.end)]
        frac, s = exposure(w.temp.values, a.vvm)
        out.append({"trip_id": t.trip_id, "source": src, "hours": len(w), "mean_temp_c": round(w.temp.mean(), 2),
                    "max_temp_c": w.temp.max(), "vvm_type": a.vvm, "fraction_to_end_point": round(frac, 4),
                    "s_est_worst_case": round(s, 4), "triage": triage(s)})
        print(out[-1])
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(out).to_csv("results/weather_proxy.csv", index=False)
    print("wrote results/weather_proxy.csv")


if __name__ == "__main__":
    main()
