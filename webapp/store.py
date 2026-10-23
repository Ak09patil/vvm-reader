"""Vial log. Saved to logs/vial_log.csv when the app runs on your own computer.
On the public online version every visitor would share that file, so there the log is kept
only for the current browser session and the user downloads it as CSV."""
import io
import os
from datetime import datetime

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(ROOT, "logs", "vial_log.csv")
ON_CLOUD = ROOT.startswith("/mount/src") or os.environ.get("VVM_SESSION_LOG") == "1"

COLUMNS = ["time", "vaccine", "vvm_type", "batch_or_vial", "site", "checked_by",
           "verdict", "reason", "s_hat", "stage", "reader", "log_ratio", "photo_issues", "note"]


def empty():
    return pd.DataFrame(columns=COLUMNS)


def _clean(df):
    for c in COLUMNS:
        if c not in df:
            df[c] = ""
    return df[COLUMNS]


def load(state):
    if "log_df" not in state:
        df = empty()
        if not ON_CLOUD and os.path.exists(LOG_PATH):
            try:
                df = _clean(pd.read_csv(LOG_PATH, dtype=str).fillna(""))
            except Exception:
                pass
        state["log_df"] = df
    return state["log_df"]


def save(state, df):
    state["log_df"] = _clean(df)
    if not ON_CLOUD:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        state["log_df"].to_csv(LOG_PATH, index=False)


def row_from_result(res, meta, note=""):
    main = res.get("main") or {}
    s = res.get("s_hat")
    return {"time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "vaccine": meta.get("vaccine", ""), "vvm_type": meta.get("vvm_type", ""),
            "batch_or_vial": meta.get("batch", ""), "site": meta.get("site", ""),
            "checked_by": meta.get("checked_by", ""),
            "verdict": res["verdict"], "reason": res["reason"],
            "s_hat": "" if s is None else f"{s:.3f}",
            "stage": "" if main.get("stage") is None else str(main["stage"]),
            "reader": main.get("reader", ""),
            "log_ratio": "" if res["r"] != res["r"] else f"{res['r']:.3f}",
            "photo_issues": ";".join(res["problems"]), "note": note}


def add(state, rows):
    df = load(state)
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    save(state, df)


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def merge_upload(state, data):
    up = _clean(pd.read_csv(io.BytesIO(data), dtype=str).fillna(""))
    df = pd.concat([load(state), up], ignore_index=True).drop_duplicates()
    save(state, df)
    return len(up)
