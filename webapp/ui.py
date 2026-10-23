"""Shared look and components."""
import hashlib

import numpy as np
import pandas as pd
import streamlit as st

import engine as E
from strings import t

COLORS = {"USE": "#15803D", "DISCARD": "#B91C1C", "RE-CHECK": "#B45309", "RETAKE": "#475569"}
ICONS = {"USE": "✓", "DISCARD": "✕", "RE-CHECK": "!", "RETAKE": "↺"}

CSS = """
<style>
.block-container {max-width: 760px; padding-top: 3.4rem; padding-bottom: 4rem;}
.vr-stats {display: flex; gap: 8px; margin: 0.4rem 0 1rem 0;}
.vr-stats div {flex: 1; background: #F3F6F5; border-radius: 12px; padding: 10px 12px; min-width: 0;}
.vr-stats b {display: block; font-size: 1.45rem; line-height: 1.2; color: #111827;}
.vr-stats span {font-size: 0.8rem; color: #4B5563;}
.vr-item {display: flex; gap: 12px; align-items: center; border: 1px solid #E5E7EB; border-radius: 14px;
  padding: 10px; margin-bottom: 8px;}
.vr-item img {width: 64px; height: 64px; border-radius: 10px; flex: none; object-fit: cover;}
.vr-item .vr-t {font-size: 0.86rem; color: #4B5563; margin-top: 4px; line-height: 1.35;}
.vr-strip {display: flex; gap: 10px; margin: 0.3rem 0 1rem 0;}
.vr-strip figure {flex: 1; margin: 0; text-align: center; min-width: 0;}
.vr-strip img {width: 100%; max-width: 150px; border-radius: 10px;}
.vr-strip figcaption {font-size: 0.82rem; color: #4B5563; margin-top: 4px; line-height: 1.25;}
h1 {font-weight: 700; letter-spacing: -0.02em;}
.vr-sub {color: #5B6470; font-size: 1.02rem; margin-top: -0.6rem; margin-bottom: 1.2rem;}
.vr-card {border-radius: 18px; padding: 22px 24px; color: #fff; margin: 0.4rem 0 1rem 0;}
.vr-card .vr-row {display: flex; align-items: center; gap: 14px;}
.vr-card .vr-icon {width: 46px; height: 46px; border-radius: 50%; background: rgba(255,255,255,0.22);
  display: flex; align-items: center; justify-content: center; font-size: 26px; font-weight: 700; flex: none;}
.vr-card .vr-title {font-size: 1.85rem; font-weight: 700; line-height: 1.15;}
.vr-card .vr-text {font-size: 1.02rem; margin-top: 10px; line-height: 1.45; opacity: 0.96;}
.vr-pill {display: inline-block; padding: 3px 12px; border-radius: 999px; color: #fff; font-weight: 600;
  font-size: 0.85rem;}
.vr-muted {color: #6B7280; font-size: 0.88rem;}
.vr-note {background: #F3F6F5; border-radius: 12px; padding: 12px 16px; color: #374151; font-size: 0.93rem;}
div[data-testid="stMetricValue"] {font-size: 1.7rem;}
.st-key-vr_nav {border-bottom: 1px solid #E5E7EB; padding-bottom: 6px; margin-bottom: 0.6rem; flex-wrap: wrap;}
.st-key-vr_nav a {padding: 2px 8px;}
</style>
"""


def style():
    st.markdown(CSS, unsafe_allow_html=True)


def header(title, sub=None):
    st.title(title)
    if sub:
        st.markdown(f"<div class='vr-sub'>{sub}</div>", unsafe_allow_html=True)


def card(verdict, title, text):
    st.markdown(
        f"<div class='vr-card' style='background:{COLORS[verdict]}'><div class='vr-row'>"
        f"<div class='vr-icon'>{ICONS[verdict]}</div><div class='vr-title'>{title}</div></div>"
        f"<div class='vr-text'>{text}</div></div>", unsafe_allow_html=True)


def pill(verdict, label=None):
    return f"<span class='vr-pill' style='background:{COLORS[verdict]}'>{label or verdict}</span>"


def note(text):
    st.markdown(f"<div class='vr-note'>{text}</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ cached engine
@st.cache_resource(show_spinner=False)
def models():
    return E.load_models()


@st.cache_data(show_spinner=False)
def deltas():
    return E.load_deltas()


@st.cache_data(show_spinner=False, max_entries=64)
def read_bytes(data: bytes):
    return E.read(E.decode(data), models(), deltas())


def read_array(img):
    return E.read(img, models(), deltas())


def digest(data: bytes):
    return hashlib.sha1(data).hexdigest()[:12]


# ------------------------------------------------------------------ result display
def verdict_text(res, lang, personal=False):
    v = res["verdict"]
    if v == "RETAKE":
        return t("RETAKE", lang), t("why_" + res["reason"], lang)
    if personal:
        return t("p_" + v, lang), t("p_" + v + "_sub", lang)
    if v == "RE-CHECK":
        return t(v, lang), t("RE-CHECK_" + ("disagree" if res["reason"] == "disagree" else "near_threshold"), lang)
    return t(v, lang), t(v + "_sub", lang)


def heat_bar(res, lang):
    s = res.get("s_hat")
    if s is None or res["verdict"] != "USE":
        return
    used = min(1.0, s / 0.75)
    st.markdown(f"**{t('heat_used', lang)}:** ≈ {used * 100:.0f}%")
    st.progress(used)


def details(res, truth=None):
    """The technical view: overlay, measurements, every reader. For supervisors and the demo."""
    c1, c2 = st.columns([1, 1])
    with c1:
        st.image(E.overlay(res["img"], res["det"]), width="stretch",
                 caption="Red: fitted ellipse. Gold: square (ρ < 0.28) and ring band (0.78–0.92).")
    with c2:
        if res["feats"]:
            st.metric("Square luminance Y", f"{res['feats']['Y_sq']:.4f}")
            st.metric("Ring luminance Y", f"{res['feats']['Y_ring']:.4f}")
            st.metric("r = ln(Y_sq / Y_ring)", f"{res['r']:+.3f}", help="WHO rule: discard if r ≤ 0")
        st.caption(f"VVM located: {res['located']}")
    if truth is not None:
        from vvm import config as C
        st.markdown(f"**Ground truth:** s = {truth:.2f}, stage {int(C.stage_of(truth))}, "
                    f"{'DISCARD' if C.is_discard(truth) else 'USE'}")
    rows = []
    for x in res["readings"]:
        rows.append({"Reader": x["reader"],
                     "ŝ": "" if x["s_hat"] is None else f"{x['s_hat']:.3f}",
                     "Stage": "" if x["stage"] is None else x["stage"],
                     "Decision": x["decision"],
                     "Near 0.75?": "" if x["delta"] is None else
                     (f"yes (±{x['delta']:g})" if x["flag"] else f"no (±{x['delta']:g})")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    q = res["quality"]
    if q:
        st.caption(f"Photo check: VVM radius {q['radius_px']:.0f} px, edge sharpness {q['sharpness']:.2f} "
                   f"(min {E.Q_MIN_SHARP}), ring brightness {q['ring_Y']:.3f} (min {E.Q_MIN_RING_Y}), "
                   f"glare {q['glare'] * 100:.1f}% (max {E.Q_MAX_GLARE * 100:.0f}%)"
                   + (f". Issues: {', '.join(res['problems'])}" if res["problems"] else ". Passed."))
    if not any(x["reader"] in ("CNN", "Hybrid", "Classical") for x in res["readings"]):
        st.caption("No trained models found in models/, so only the WHO rule is used.")


def _b64(img):
    import base64
    import cv2
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode()


def stats(items):
    """Small numbers in one row (stays one row on a phone)."""
    st.markdown("<div class='vr-stats'>" + "".join(f"<div><b>{v}</b><span>{k}</span></div>" for k, v in items)
                + "</div>", unsafe_allow_html=True)


def item(img, verdict, title, name, text):
    st.markdown(f"<div class='vr-item'><img src='data:image/png;base64,{_b64(img)}'><div>"
                f"{pill(verdict, title)} <span class='vr-muted'>{name}</span><div class='vr-t'>{text}</div>"
                "</div></div>", unsafe_allow_html=True)


def strip(items):
    """Row of small VVM pictures with captions that stays one row on a phone."""
    import base64
    import cv2
    figs = []
    for img, cap in items:
        ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        b64 = base64.b64encode(buf.tobytes()).decode()
        figs.append(f"<figure><img src='data:image/png;base64,{b64}'><figcaption>{cap}</figcaption></figure>")
    st.markdown("<div class='vr-strip'>" + "".join(figs) + "</div>", unsafe_allow_html=True)


def thumb(img, size=96):
    import cv2
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
