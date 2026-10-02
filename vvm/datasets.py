"""Uniform loader for every evaluation set (synthetic and real).

Every loader returns a DataFrame with at least:
  image_id, path, s (NaN if unknown), stage, discard, group_type, group
and for synthetic sets gt_cx, gt_cy, gt_r (for segmentation scoring).
"""
import os
import numpy as np
import pandas as pd

from . import config as C

SYNTH_SETS = ("train", "val", "benchmark", "style_in", "style_out")
REAL_SETS = ("real_print", "real_web")


def load_set(name, root="data"):
    if name in SYNTH_SETS:
        d = os.path.join(root, "synth", name)
        df = pd.read_csv(os.path.join(d, "metadata.csv"))
        df["path"] = [os.path.join(d, "images", i + ".png") for i in df.image_id]
        if name == "benchmark":
            df["group_type"] = "cell"
            df["group"] = df.cell_axis + ":" + df.cell_level
        elif name in ("style_in", "style_out"):
            df["group_type"], df["group"] = "style", name
        else:
            df["group_type"], df["group"] = "all", name
        return df

    if name == "real_print":
        d = os.path.join(root, "real_print")
        lab = pd.read_csv(os.path.join(d, "labels.csv"))
        sheet = pd.read_csv(os.path.join(root, "print_sheet", "print_sheet_labels.csv"))
        df = lab.merge(sheet[["vvm_id", "s"]], on="vvm_id", how="left")
        if df.s.isna().any():
            raise ValueError("labels.csv has vvm_id values not on the print sheet")
        df["image_id"] = "print_" + df.filename.str.rsplit(".", n=1).str[0]
        df["path"] = [os.path.join(d, "images", f) for f in df.filename]
        df["stage"] = C.stage_of(df.s.values)
        df["discard"] = C.is_discard(df.s.values).astype(int)
        df["group_type"] = "condition"
        df["group"] = df.condition.astype(str) + "|" + df.phone.astype(str)
        return df

    if name == "real_web":
        d = os.path.join(root, "real_web")
        df = pd.read_csv(os.path.join(d, "labels.csv"))
        df["image_id"] = "web_" + df.filename.str.rsplit(".", n=1).str[0]
        df["path"] = [os.path.join(d, "images", f) for f in df.filename]
        df["s"] = np.nan
        df["stage"] = df.source_stage.astype(int)
        df["discard"] = (df.stage >= 3).astype(int)
        df["group_type"], df["group"] = "all", "real_web"
        return df

    raise KeyError(name)


def available_sets(root="data"):
    out = [s for s in SYNTH_SETS if os.path.exists(os.path.join(root, "synth", s, "metadata.csv"))]
    if os.path.exists(os.path.join(root, "real_print", "labels.csv")):
        out.append("real_print")
    if os.path.exists(os.path.join(root, "real_web", "labels.csv")):
        out.append("real_web")
    return out
