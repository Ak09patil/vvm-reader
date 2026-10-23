"""Heat-exposure planner: how much of a vial's VVM allowance an outreach session will use.

Same kinetics as scripts/weather_proxy.py and the paper: two-point Arrhenius fit to the WHO
VVM end points, t_end(T) = exp(A + B/T); fraction used F = sum(dt / t_end(T_i)); the end point
(F = 1) is the discard point, so the stage moves by 0.75 * F. Weather: Open-Meteo (free, no key).
"""
import math
from datetime import date, datetime

import numpy as np
import pandas as pd
import requests

# (temperature C, days to end point), as in scripts/weather_proxy.py
END_POINT_DAYS = {
    "VVM2": [(37, 2), (5, 225)],
    "VVM7": [(37, 7), (25, 45)],
    "VVM14": [(37, 14), (25, 90)],
    "VVM30": [(37, 30), (25, 193)],
}
BOX_TEMP_C = 8.0      # inside a cold box / vaccine carrier with ice packs: top of the 2-8 C range
DISCARD_S = 0.75

# Offline fallback if the place search is unreachable
CITIES = {
    "Hyderabad": (17.385, 78.487), "Mumbai": (19.076, 72.878), "Delhi": (28.614, 77.209),
    "Bengaluru": (12.972, 77.594), "Chennai": (13.083, 80.271), "Kolkata": (22.573, 88.364),
    "Ahmedabad": (23.023, 72.571), "Pune": (18.520, 73.857), "Jaipur": (26.912, 75.787),
    "Lucknow": (26.847, 80.947), "Patna": (25.594, 85.138), "Bhopal": (23.260, 77.413),
    "Nagpur": (21.146, 79.088), "Raipur": (21.251, 81.630), "Ranchi": (23.344, 85.310),
    "Bhubaneswar": (20.296, 85.825), "Guwahati": (26.144, 91.736), "Chandigarh": (30.733, 76.779),
    "Thiruvananthapuram": (8.524, 76.937), "Vijayawada": (16.506, 80.648), "Warangal": (17.968, 79.594),
    "Jodhpur": (26.238, 73.024), "Bikaner": (28.022, 73.312), "Jaisalmer": (26.915, 70.908),
    "Banda": (25.476, 80.335), "Prayagraj": (25.435, 81.846), "Varanasi": (25.318, 82.974),
    "Ludhiana": (30.901, 75.857), "Srinagar": (34.084, 74.797), "Dehradun": (30.316, 78.032),
}


def arrhenius(vvm):
    (T1, d1), (T2, d2) = END_POINT_DAYS[vvm]
    x1, x2 = 1 / (T1 + 273.15), 1 / (T2 + 273.15)
    B = (math.log(d1) - math.log(d2)) / (x1 - x2)
    return math.log(d1) - B * x1, B


def hours_to_end(temp_c, vvm):
    A, B = arrhenius(vvm)
    return np.exp(A + B / (np.asarray(temp_c, float) + 273.15)) * 24.0


def reference_table(temps=(25, 30, 35, 40, 45)):
    rows = []
    for v in END_POINT_DAYS:
        h = hours_to_end(temps, v)
        rows.append({"VVM type": v, **{f"{t} °C": _fmt_duration(x) for t, x in zip(temps, h)}})
    return pd.DataFrame(rows)


def _fmt_duration(h):
    if h >= 48:
        return f"{h / 24:.0f} days"
    return f"{h:.0f} h"


# ------------------------------------------------------------------ weather
def search_places(name, n=6):
    r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                     params={"name": name, "count": n, "language": "en"}, timeout=10)
    r.raise_for_status()
    out = []
    for p in r.json().get("results", []):
        label = ", ".join(x for x in (p.get("name"), p.get("admin2"), p.get("admin1"), p.get("country")) if x)
        out.append((label, p["latitude"], p["longitude"]))
    return out


def hourly_forecast(lat, lon, day):
    """Hourly 2 m air temperature (local time) for one day, from Open-Meteo."""
    q = {"latitude": lat, "longitude": lon, "hourly": "temperature_2m", "timezone": "auto",
         "start_date": day.isoformat(), "end_date": day.isoformat()}
    last = None
    for url in ("https://api.open-meteo.com/v1/forecast", "https://archive-api.open-meteo.com/v1/archive"):
        try:
            r = requests.get(url, params=q, timeout=15)
            r.raise_for_status()
            h = r.json()["hourly"]
            break
        except Exception as ex:   # noqa
            last = ex
    else:
        raise last
    df = pd.DataFrame({"time": pd.to_datetime(h["time"]), "temp": h["temperature_2m"]}).dropna()
    if df.empty:
        raise ValueError("no temperatures returned")
    return df


def constant_day(day, temp):
    t = pd.date_range(datetime.combine(day, datetime.min.time()), periods=24, freq="h")
    return pd.DataFrame({"time": t, "temp": float(temp)})


# ------------------------------------------------------------------ projection
def project(weather, start_h, end_h, vvm, s_now, in_box):
    """Stage of the VVM hour by hour through the session.
    Returns (table, summary). Assumes the vial sits at air temperature when not in a box."""
    w = weather[(weather.time.dt.hour >= start_h) & (weather.time.dt.hour < end_h)].copy()
    w["exposed_c"] = BOX_TEMP_C if in_box else w["temp"]
    w["used"] = 1.0 / hours_to_end(w["exposed_c"], vvm)          # fraction per hour
    w["s"] = s_now + DISCARD_S * w["used"].cumsum()
    start_row = pd.DataFrame({"time": [w.time.iloc[0]], "s": [s_now]}) if len(w) else None
    table = pd.concat([start_row, w[["time", "s"]].assign(time=w.time + pd.Timedelta(hours=1))]) \
        if start_row is not None else w
    F = float(w["used"].sum())
    ds = DISCARD_S * F
    s_end = s_now + ds
    hit = w[w.s >= DISCARD_S]
    left = math.inf if ds <= 0 else max(0, math.floor((DISCARD_S - s_now) / ds))
    summary = {"hours": len(w), "max_temp": float(w.temp.max()) if len(w) else float("nan"),
               "mean_temp": float(w.temp.mean()) if len(w) else float("nan"),
               "used_pct": 100 * F, "s_end": s_end,
               "discard_at": (hit.time.iloc[0] + pd.Timedelta(hours=1)) if len(hit) else None,
               "sessions_left": left}
    summary["status"] = "over" if s_end >= DISCARD_S else ("close" if s_end >= 0.60 else "ok")
    return table, summary
