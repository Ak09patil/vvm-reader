"""VVM Reader: check whether a vaccine vial has had too much heat, from a phone photo.

Run from the repo folder:
    streamlit run webapp/app.py
Phone on the same Wi-Fi:  streamlit run webapp/app.py --server.address 0.0.0.0
then open http://<Mac IP>:8501   (Mac IP: ipconfig getifaddr en0)
"""
import os
import sys
from datetime import date, timedelta

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E   # noqa: E402
import heat as H     # noqa: E402
import store         # noqa: E402
import ui            # noqa: E402
from strings import LANGS, t  # noqa: E402

st.set_page_config(page_title="VVM Reader", page_icon=":material/vaccines:", layout="centered",
                   initial_sidebar_state="collapsed")
ui.style()
ss = st.session_state

VACCINES = ["", "BCG", "OPV", "Hepatitis B", "Pentavalent", "Rotavirus", "IPV / fIPV", "PCV",
            "Measles-Rubella (MR)", "JE", "DPT", "Td", "Other"]
VVM_TYPES = ["Not sure", "VVM2", "VVM7", "VVM14", "VVM30"]


# ------------------------------------------------------------------ helpers
def lang_picker():
    ss.setdefault("lang", "en")
    choice = st.segmented_control("Language", list(LANGS), format_func=LANGS.get, default=ss["lang"],
                                  key="lang_pick", label_visibility="collapsed")
    if choice:
        ss["lang"] = choice
    return ss["lang"]


def photo_inputs(lang, many):
    tab1, tab2 = st.tabs([t("tab_photo", lang), t("tab_camera", lang)])
    with tab1:
        up = st.file_uploader(t("upload_many" if many else "upload", lang), type=["jpg", "jpeg", "png", "webp"],
                              accept_multiple_files=many, key=f"up_{many}")
    with tab2:
        cam = st.camera_input(t("camera", lang), key=f"cam_{many}")
    items = []
    if up:
        for f in (up if many else [up]):
            items.append((f.name, f.getvalue()))
    if cam is not None:
        items.append(("camera", cam.getvalue()))
    return items


def read_safe(data):
    try:
        return ui.read_bytes(data)
    except Exception:
        st.error("This file couldn't be opened as a photo. Try a JPG or PNG.")
        return None


def save_button(res, h, meta):
    saved = ss.setdefault("saved", set())
    if h in saved:
        st.button("Saved to log", icon=":material/check:", disabled=True, key=f"sv_{h}", width="stretch")
    elif st.button("Save to vial log", icon=":material/playlist_add:", key=f"sv_{h}", width="stretch"):
        store.add(ss, [store.row_from_result(res, meta)])
        saved.add(h)
        st.rerun()


# ------------------------------------------------------------------ pages
def home():
    ui.header("VVM Reader", "Check whether a vaccine vial has had too much heat, from a phone photo.")
    ui.strip([(guide_img(s), cap) for s, cap in zip((0.15, 0.55, 0.78, 0.95),
                                                    ("Use", "Use", "Do not use", "Do not use"))])
    st.write("")
    st.markdown("#### Who is checking?")
    if st.button("I'm a health worker", icon=":material/medical_services:", width="stretch", type="primary"):
        set_mode("worker")
    st.caption("Scan vials, keep a vial log, and plan outreach sessions in the heat.")
    if st.button("I'm checking a vaccine for myself or my family", icon=":material/family_restroom:",
                 width="stretch"):
        set_mode("personal")
    st.caption("A quick check of the vial's heat indicator before a vaccination.")
    st.write("")
    ui.note("Prototype from an NMIMS IVP + AI project. It reads the WHO vaccine vial monitor (VVM) "
            "and never replaces the health worker's own check.")


def set_mode(m):
    ss["mode"] = m
    st.switch_page(PAGES["scan" if m == "worker" else "check"])


def scan():
    lang = lang_picker()
    ui.header(t("scan_title", lang), t("scan_hint", lang))
    meta = ss.setdefault("meta", {"vaccine": "", "vvm_type": "Not sure", "batch": "", "site": "", "checked_by": ""})
    with st.expander("Vial details for the log (optional)", icon=":material/edit_note:"):
        c1, c2 = st.columns(2)
        meta["vaccine"] = c1.selectbox("Vaccine", VACCINES, index=VACCINES.index(meta["vaccine"]))
        meta["vvm_type"] = c2.selectbox("VVM type", VVM_TYPES, index=VVM_TYPES.index(meta["vvm_type"]),
                                        help="Printed in the vaccine's documentation; ask your supervisor if unsure.")
        meta["batch"] = c1.text_input("Batch or vial number", meta["batch"])
        meta["site"] = c2.text_input("Session site", meta["site"])
        meta["checked_by"] = c1.text_input("Checked by", meta["checked_by"])

    items = photo_inputs(lang, many=True)
    if not items:
        ui.note("Tip: you can choose several photos at once to check a whole box. "
                "Photos are read on the spot and not stored.")
        return

    if len(items) == 1:
        data = items[0][1]
        res = read_safe(data)
        if res is None:
            return
        title, text = ui.verdict_text(res, lang)
        ui.card(res["verdict"], title, text)
        ui.heat_bar(res, lang)
        h = ui.digest(data)
        c1, c2 = st.columns(2)
        with c1:
            if res["verdict"] != "RETAKE":
                save_button(res, h, meta)
        with c2:
            if res["verdict"] == "USE" and ss.get("mode") == "worker":
                if st.button("Plan a session for this vial", icon=":material/thermostat:", width="stretch"):
                    ss["plan_s"] = res["s_hat"]
                    st.switch_page(PAGES["plan"])
        with st.expander("How it was read", icon=":material/visibility:"):
            ui.details(res)
        return

    # several photos: one row each
    results = [(name, data, read_safe(data)) for name, data in items]
    results = [x for x in results if x[2] is not None]
    counts = pd.Series([r["verdict"] for _, _, r in results]).value_counts()
    ui.stats([(lab, int(counts.get(v, 0))) for v, lab in
              (("USE", "Use"), ("DISCARD", "Do not use"), ("RE-CHECK", "Re-check"), ("RETAKE", "Retake"))])
    for name, data, res in results:
        title, text = ui.verdict_text(res, lang)
        ui.item(ui.thumb(res["img"]), res["verdict"], title, name, text)
    todo = [(ui.digest(d), r) for _, d, r in results if r["verdict"] != "RETAKE"
            and ui.digest(d) not in ss.setdefault("saved", set())]
    if todo and st.button(f"Save {len(todo)} reading(s) to vial log", icon=":material/playlist_add:",
                          type="primary", width="stretch"):
        store.add(ss, [store.row_from_result(r, meta) for _, r in todo])
        ss["saved"].update(h for h, _ in todo)
        st.rerun()
    elif not todo:
        st.caption("All readable photos are saved to the log.")


def check():
    lang = lang_picker()
    ui.header(t("check_title", lang), t("scan_hint", lang))
    ui.strip([(guide_img(0.3), "OK: square lighter than the circle"),
              (guide_img(0.95), "Not OK: square same or darker")])
    items = photo_inputs(lang, many=False)
    if items:
        res = read_safe(items[-1][1])
        if res is not None:
            title, text = ui.verdict_text(res, lang, personal=True)
            ui.card(res["verdict"], title, text)
            with st.expander("How it was read", icon=":material/visibility:"):
                ui.details(res)
    st.write("")
    ui.note("This checks only the heat indicator on the vial. It can't tell whether the vaccine was frozen "
            "or has expired, and it does not replace the health worker's check. If you are unsure, ask them "
            "to show you the vial's label.")


def log_page():
    ui.header("Vial log", "Readings you saved, with counts you can report.")
    df = store.load(ss)
    if store.ON_CLOUD:
        ui.note("On the online version the log is kept only while this page is open. "
                "Download it before closing, and load it again next time.")
        st.write("")
    if df.empty:
        st.info("No readings yet. Scan a vial and tap **Save to vial log**.")
    else:
        c1, c2 = st.columns([2, 1])
        vac = c1.selectbox("Vaccine", ["All"] + sorted(x for x in df.vaccine.unique() if x), index=0)
        today = c2.toggle("Today only", value=False)
        view = df.copy()
        if vac != "All":
            view = view[view.vaccine == vac]
        if today:
            view = view[view.time.str.startswith(date.today().isoformat())]
        counts = view.verdict.value_counts()
        ui.stats([("Checked", len(view)), ("Use", int(counts.get("USE", 0))),
                  ("Do not use", int(counts.get("DISCARD", 0))), ("Re-check", int(counts.get("RE-CHECK", 0)))])
        show = view.rename(columns={"time": "Time", "vaccine": "Vaccine", "batch_or_vial": "Batch/vial",
                                    "site": "Site", "verdict": "Verdict", "s_hat": "ŝ", "vvm_type": "VVM",
                                    "checked_by": "By", "note": "Note"})
        st.dataframe(show[["Time", "Verdict", "Vaccine", "Batch/vial", "Site", "ŝ", "VVM", "By"]].iloc[::-1],
                     hide_index=True, width="stretch")
        st.download_button("Download log (CSV)", store.to_csv_bytes(view), icon=":material/download:",
                           file_name=f"vial_log_{date.today().isoformat()}.csv", mime="text/csv",
                           width="stretch", type="primary")
        with st.expander("Summary by vaccine", icon=":material/summarize:"):
            pv = pd.crosstab(view.vaccine.replace("", "(not given)"), view.verdict)
            for v in ("USE", "DISCARD", "RE-CHECK"):
                if v not in pv:
                    pv[v] = 0
            pv = pv[["USE", "DISCARD", "RE-CHECK"]].rename(columns={"USE": "Use", "DISCARD": "Do not use",
                                                                    "RE-CHECK": "Re-check"})
            pv["Total"] = pv.sum(axis=1)
            st.dataframe(pv, width="stretch")
    with st.expander("Load a saved log", icon=":material/upload_file:"):
        f = st.file_uploader("Log CSV downloaded from this app", type=["csv"], key="log_up")
        if f is not None and st.button("Add to log", key="merge"):
            try:
                n = store.merge_upload(ss, f.getvalue())
                st.success(f"Added {n} rows.")
                st.rerun()
            except Exception:
                st.error("That file doesn't look like a vial log.")
    if not df.empty:
        with st.expander("Clear the log", icon=":material/delete:"):
            ok = st.checkbox("I have downloaded what I need. Delete all readings.")
            if st.button("Clear log", disabled=not ok):
                store.save(ss, store.empty())
                ss["saved"] = set()
                st.rerun()


@st.cache_data(ttl=3600, show_spinner=False)
def places(q):
    return H.search_places(q)


@st.cache_data(ttl=1800, show_spinner=False)
def forecast(lat, lon, day):
    return H.hourly_forecast(lat, lon, day)


def plan():
    ui.header("Session planner", "Will a vial stay usable through an outreach session in the forecast heat?")
    c1, c2 = st.columns([3, 1])
    q = c1.text_input("Place", ss.get("plan_q", "Hyderabad"), placeholder="Village, town or district")
    ss["plan_q"] = q
    loc, offline = None, False
    try:
        found = places(q) if q.strip() else []
        if found:
            label = c2.selectbox("Match", [f[0] for f in found], label_visibility="hidden")
            loc = next(f for f in found if f[0] == label)
        else:
            st.caption("No place found with that name. Try the district name.")
    except Exception:
        offline = True
        city = c2.selectbox("City", list(H.CITIES), label_visibility="hidden")
        loc = (city, *H.CITIES[city])

    c1, c2 = st.columns(2)
    day = c1.date_input("Session date", date.today(), min_value=date.today() - timedelta(days=30),
                        max_value=date.today() + timedelta(days=15), format="DD/MM/YYYY")
    hours = c2.slider("Session hours", 5, 22, (8, 18), format="%d:00")
    c1, c2 = st.columns(2)
    default_vvm = ss.get("meta", {}).get("vvm_type", "Not sure")
    vvm = c1.selectbox("VVM type", list(H.END_POINT_DAYS),
                       index=list(H.END_POINT_DAYS).index(default_vvm) if default_vvm in H.END_POINT_DAYS else 0,
                       help="VVM2 is the most heat-sensitive (for example oral polio vaccine). "
                            "If unsure, VVM2 is the cautious choice.")
    where = c2.radio("Where the vial stays", ["In the open (worst case)", "Cold box or carrier with ice packs"])
    s_from_scan = ss.get("plan_s")
    s_now = st.slider("Current VVM reading ŝ (0 = fresh, 0.75 = discard point)", 0.0, 0.74,
                      float(min(0.74, s_from_scan)) if s_from_scan is not None else 0.0, 0.01)
    if s_from_scan is not None:
        st.caption(f"Filled in from your last scan (ŝ = {s_from_scan:.2f}).")

    weather, src = None, ""
    if loc and not offline:
        try:
            with st.spinner("Getting the forecast…"):
                weather = forecast(loc[1], loc[2], day)
            src = f"Hourly forecast for {loc[0]} (Open-Meteo)"
        except Exception:
            offline = True
    if weather is None:
        st.warning("Couldn't get the forecast. Enter the expected temperature instead.")
        temp = st.slider("Expected air temperature (°C)", 15, 48, 38)
        weather, src = H.constant_day(day, temp), f"Constant {temp} °C (entered)"

    table, sm = H.project(weather, hours[0], hours[1], vvm, s_now, where.startswith("Cold"))
    if sm["hours"] == 0:
        st.info("Pick session hours to see the projection.")
        return
    if sm["status"] == "over":
        when = sm["discard_at"].strftime("%H:%M") if sm["discard_at"] is not None else "the end of the session"
        ui.card("DISCARD", "Likely to reach the discard point",
                f"At these temperatures this vial would reach the discard point around {when}. "
                "Keep it in the cold box with ice packs, and read its VVM before each use.")
    elif sm["status"] == "close":
        ui.card("RE-CHECK", "Close to the limit by the end",
                f"It would end the session at about ŝ = {sm['s_end']:.2f}. Use this vial early, keep it in "
                "the carrier between doses, and read its VVM again before the next session.")
    else:
        ui.card("USE", "Stays usable through this session",
                f"Even {'in the open' if where.startswith('In') else 'in the box'} all session, it would end at "
                f"about ŝ = {sm['s_end']:.2f}, below the 0.75 discard point.")
    left = "20+" if sm["sessions_left"] == float("inf") or sm["sessions_left"] > 20 else sm["sessions_left"]
    ui.stats([("Hottest hour", f"{sm['max_temp']:.0f} °C"), ("Allowance used", f"{sm['used_pct']:.0f}%"),
              ("Sessions like this left", left)])

    w = weather[(weather.time.dt.hour >= hours[0]) & (weather.time.dt.hour < hours[1])]
    base = alt.Chart(table).mark_line(color="#0F766E", strokeWidth=3, point=True).encode(
        x=alt.X("time:T", title=None, axis=alt.Axis(format="%H:%M")),
        y=alt.Y("s:Q", title="VVM stage ŝ", scale=alt.Scale(domain=[0, max(1.0, float(table.s.max()))])))
    rule = alt.Chart(pd.DataFrame({"y": [0.75]})).mark_rule(color="#B91C1C", strokeDash=[6, 4]).encode(y="y:Q")
    st.altair_chart((base + rule).properties(height=230, title="Projected VVM stage (red line: discard point)"),
                    width="stretch")
    tc = alt.Chart(w).mark_line(color="#D97706", strokeWidth=3).encode(
        x=alt.X("time:T", title=None, axis=alt.Axis(format="%H:%M")),
        y=alt.Y("temp:Q", title="°C", scale=alt.Scale(zero=False)))
    st.altair_chart(tc.properties(height=150, title="Air temperature"), width="stretch")
    st.caption(src)
    with st.expander("How long a VVM lasts at a steady temperature", icon=":material/table:"):
        st.dataframe(H.reference_table(), hide_index=True, width="stretch")
        st.caption("Time from fresh (ŝ = 0) to the discard point, from an Arrhenius fit to the WHO VVM end "
                   "points. Below 25 °C the fit is an extrapolation, so it is not shown.")
    ui.note("This is a planning estimate, not a measurement. It assumes the vial sits at air temperature "
            "the whole session (worst case) or at 8 °C in a box, and that the colour change adds up with heat. "
            "Always read the VVM itself before use.")


def guide():
    ui.header("Guide", "How to read a VVM, take a good photo, and what the VVM can't tell you.")
    st.markdown("#### Reading a VVM by eye")
    st.write("The VVM is a circle with a square inside. Heat slowly darkens the square. "
             "**Use the vial while the square is lighter than the circle.** Once the square matches or is "
             "darker than the circle, do not use it.")
    ui.strip([(guide_img(s), cap) for s, cap in zip(
        (0.15, 0.55, 0.75, 0.95), ("1 · Fresh: use", "2 · Darkening, still lighter: use",
                                   "3 · Same as circle: do not use", "4 · Darker: do not use"))])

    st.markdown("#### Taking a photo the app can read")
    st.markdown("- Fill most of the frame with the label, with the circle in the middle.\n"
                "- Use daylight or a tube light. Avoid direct sun or a bulb shining on the label.\n"
                "- Hold the phone straight above the label and tap the screen on the VVM to focus.\n"
                "- One VVM per photo. If the app says **Take the photo again**, follow its tip.")

    st.markdown("#### What the VVM does not tell you")
    st.markdown("- **Freezing.** Freezing damages some vaccines but does not change the VVM. "
                "Follow your programme's freeze check (shake test) if freezing is suspected.\n"
                "- **Expiry date.** Always check the date on the label too.\n"
                "- **Opened vials.** Follow your programme's open-vial policy for how long an opened "
                "vial can be kept.\n"
                "- **Damaged labels.** If the VVM is torn, smudged or missing, follow your programme's rules.")

    st.markdown("#### VVM types")
    st.write("The number is roughly how many days the VVM takes to reach its discard point at 37 °C. "
             "Lower numbers are for less heat-stable vaccines.")
    st.dataframe(H.reference_table((25, 37)), hide_index=True, width="stretch")

    st.markdown("#### How the app reads the VVM")
    st.markdown(
        "1. **Find the VVM:** Gaussian blur, Otsu and adaptive thresholds, morphological opening, the most "
        "circular solid contour, ellipse fit (Hough circle as backup). The app then re-centres and zooms.\n"
        "2. **Check the photo:** size, edge sharpness, brightness, glare, tilt. Poor photos are sent back.\n"
        "3. **Measure:** median linear luminance Y of the square (ρ < 0.28) and ring band (0.78–0.92), "
        "r = ln(Y_sq / Y_ring). WHO rule: discard if r ≤ 0.\n"
        "4. **Estimate the stage ŝ** with a small CNN (217k parameters), discard if ŝ ≥ 0.75.\n"
        "5. **Decide:** if ŝ is within δ of 0.75, or the CNN and WHO rule disagree, ask for a re-check by eye.")
    loaded = ", ".join(ui.models()) or "none (WHO rule only)"
    st.caption(f"Readers loaded: WHO rule, {loaded}.")
    ui.note("Prototype for research and teaching, not a certified medical device. Tested on synthetic VVMs "
            "and photos of printed VVMs, not yet on real vials in the field. Photos are processed and not "
            "stored; only readings you save go into the vial log.")


def lab():
    ui.header("Test lab", "Generate a VVM at a known stage, degrade it, and see how the app reads it.")
    c1, c2 = st.columns(2)
    s = c1.slider("True stage s", 0.0, 1.0, 0.70, 0.01)
    axis = c2.selectbox("Degradation", ["none", "geometric", "illumination", "occlusion", "blur", "noise", "jpeg"])
    c1, c2, c3 = st.columns([2, 1, 1])
    tt = c1.slider("Strength t (1 = training max, 1.5 = extreme)", 0.0, 1.5, 1.0, 0.05, disabled=axis == "none")
    seed = c2.number_input("Seed", 0, 9999, 7)
    held = c3.toggle("New colour style", help="Held-out ring colour and square size, never seen in training")
    img = gen(s, axis, tt, held, int(seed))
    res = ui.read_array(img)
    title, text = ui.verdict_text(res, "en")
    ui.card(res["verdict"], title, text)
    ui.details(res, truth=s)


@st.cache_data(show_spinner=False)
def gen(s, axis, tt, held, seed):
    return E.synthetic(s, axis, tt, held, seed)


@st.cache_data(show_spinner=False)
def guide_img(s):
    return E.reference_vvm(s)


# ------------------------------------------------------------------ navigation
PAGES = {
    "home": st.Page(home, title="VVM Reader", icon=":material/home:", default=True),
    "scan": st.Page(scan, title="Scan", icon=":material/photo_camera:", url_path="scan"),
    "check": st.Page(check, title="Check", icon=":material/photo_camera:", url_path="check"),
    "log": st.Page(log_page, title="Vial log", icon=":material/list_alt:", url_path="log"),
    "plan": st.Page(plan, title="Planner", icon=":material/thermostat:", url_path="planner"),
    "guide": st.Page(guide, title="Guide", icon=":material/menu_book:", url_path="guide"),
    "lab": st.Page(lab, title="Test lab", icon=":material/science:", url_path="lab"),
}

WORKER = ("scan", "log", "plan", "guide", "lab")
PERSONAL = ("check", "guide")

nav = st.navigation(list(PAGES.values()), position="hidden")
current = next(k for k, v in PAGES.items() if v.url_path == nav.url_path) if nav.url_path else "home"
if current in ("scan", "log", "plan", "lab"):
    ss["mode"] = "worker"
elif current == "check":
    ss["mode"] = "personal"

if current != "home":
    links = WORKER if ss.get("mode", "worker") == "worker" else PERSONAL
    labels = {"scan": "Scan", "check": "Check", "log": "Log", "plan": "Planner", "guide": "Guide", "lab": "Lab"}
    with st.container(horizontal=True, gap="small", key="vr_nav"):
        st.page_link(PAGES["home"], label="VVM Reader", icon=":material/vaccines:")
        for k in links:
            st.page_link(PAGES[k], label=labels[k], icon=PAGES[k].icon)
nav.run()
