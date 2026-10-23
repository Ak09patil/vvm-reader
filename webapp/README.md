# VVM Reader app

A phone-friendly app that reads a WHO vaccine vial monitor (VVM) from a photo and says
**Use**, **Do not use**, **Check again by eye** or **Take the photo again**.

| Page | For | What it does |
|---|---|---|
| Start | everyone | choose health worker or personal check |
| Scan | health worker | one photo or a whole box at once, photo-quality check, verdict, save to log |
| Log | health worker | saved readings, counts, summary by vaccine, CSV download and reload |
| Planner | health worker | hourly forecast for a place and date, how much of the VVM allowance a session uses |
| Guide | everyone | reading a VVM by eye, photo tips, what a VVM can't tell you, how the app works |
| Check | individual | simple check in plain language, no log |
| Lab | demo | synthetic VVM at a known stage and degradation, with every reader's output |

Scan and Check are in English, Hindi and Telugu.

Run locally (from the repo folder):  `streamlit run webapp/app.py`
Phone on the same Wi-Fi: add `--server.address 0.0.0.0`, then open `http://<Mac IP>:8501`.

Uses `vvm/` for image processing and `models/cnn.pt`, `models/hybrid.pt`, `models/classical.pkl`.
Readings saved on your own computer go to `logs/vial_log.csv`; on the online version the log
stays in the browser session only and is downloaded as CSV.

Prototype for research and teaching, not a certified medical device.
