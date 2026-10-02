# VVM Reader

**Robust smartphone-image reading of WHO Vaccine Vial Monitors, with a safety-oriented confidence flag.**

A Vaccine Vial Monitor (VVM) is a heat-sensitive label on vaccine vials: a filled reference circle with an inner square that darkens irreversibly with cumulative heat exposure. The WHO reading rule is simple: if the square is lighter than the circle, the vial is usable; if it is the same colour or darker, it must be discarded. In practice this visual comparison is error-prone, and the consequences of a wrong call are real: a damaged vaccine administered, or a good one wasted.

This repository contains the complete pipeline for our IIIT Vadodara student symposium submission (Track 1: AI for Healthcare, Biomedical Devices). It includes a parametric synthetic VVM generator, a segmentation and colour-measurement stage, classical and learned readers, a calibrated confidence flag, and a full robustness evaluation.

> **Status:** synthetic evaluation complete and frozen. Real-photo evaluation (print-and-capture and web reference set) in progress.

---

## Contribution, stated precisely

Reading colour-change indicators from smartphone images robustly, across lighting, devices and angles, is an established technique, already used for food freshness, fruit ripeness and diagnostic test strips. **We do not claim to invent it.** Our contribution is applying and rigorously validating it in a domain where no automated system has been evaluated: vaccine cold-chain monitoring via VVMs, where a misreading has public-health consequences. Specifically:

1. **A robustness benchmark** with six named degradation axes at four severity levels, evaluated one axis at a time against a fixed baseline. It includes a generalisation test on unseen label styles and a print-and-capture test with real cameras.
2. **A confidence flag built around the dangerous error**, calling a discard-stage vial usable, with its limits measured explicitly beyond the training range.

---

## Key results (synthetic, frozen)

Benchmark: 5,200 images (26 cells × 200).

| Method | Decision accuracy | Unsafe-error rate | MAE on s | Coverage* | Unsafe among unflagged |
|---|---|---|---|---|---|
| WHO rule (literal protocol) | 94.2% | 3.4% | – | 100% | 3.4% |
| Classical (colour ratio → s) | 94.3% | 3.2% | 0.049 | 26.5% | 1.2% |
| CNN | 97.2% | 1.8% | 0.024 | 93.9% | 0.7% |
| Hybrid (CNN + colour features) | 97.3% | 1.7% | 0.024 | 94.4% | 0.7% |

\*Coverage is the fraction of readings *not* flagged for manual re-check. The flag margin δ was calibrated on the validation set and then frozen.

**Main findings**

- **Halved unsafe errors.** The learned models roughly halve both unsafe errors and regression error compared with the WHO rule and the classical method.
- **Much less manual re-checking.** The CNN stays below 1% unsafe errors while flagging about 6% of readings. The classical method flags about 74% and still misses the 1% target.
- **Occlusion breaks classical sampling.** At extreme occlusion, classical decision accuracy falls to 79.5% with 18.5% unsafe errors. The CNN holds 93% accuracy with 4.5% unsafe errors.
- **Generalises to unseen label styles.** On ring colours and square sizes never seen in training, CNN accuracy goes from 98.3% to 97.7%, with MAE unchanged at 0.020.
- **The flag holds within the training range but not beyond it.** In-range, unsafe errors among unflagged readings stay at or below 1% in 17 of 18 condition–severity cells (maximum 1.05%). At the extreme level (t = 1.5, beyond training), the CNN becomes confidently wrong in the unsafe direction, and the flag only partly compensates (1.1–5.4%). A margin-based flag is not a substitute for input-quality checks.
- **Lightweight.** About 218K parameters, about 0.6 ms per crop on CPU.

Full per-condition numbers are in [`results/metrics/metrics.csv`](results/metrics/metrics.csv), and the figures are in [`figures/`](figures/).

---

## Pipeline

```
Synthetic generator ──► Segmentation (IVP) ──► Readers ──────────────► Confidence flag ──► Evaluation
 continuous stage s      disc detection          who_rule  (IVP)        |s − 0.75| ≤ δ       benchmark (6 axes × 4 levels)
 6 degradation axes      ellipse fit             classical (IVP)        δ calibrated on val  style generalisation
 train/held-out styles   ring/square sampling    cnn       (AI)                              print-and-capture photos
                         64×64 crop              hybrid    (AI + IVP)                        web reference set + annotator κ

Secondary: weather_proxy.py — worst-case ambient heat-exposure triage (Open-Meteo + Arrhenius kinetics)
```

**Segmentation is designed around the hardest case.** At the discard point the square is *identical* to the ring, so segmentation cannot rely on square/ring contrast. Instead, the outer disc is found against the label background and fitted with an ellipse, which handles perspective. The square and ring are then sampled at fixed normalised radii inside that ellipse.

---

## Exposure stage (the core definition)

Continuous stage `s ∈ [0, 1]`, with the discard threshold fixed at **s = 0.75**. That is where the square's colour exactly equals the ring's.

| s | WHO stage | Decision |
|---|---|---|
| 0.00–0.40 | 1, start | Use |
| 0.40–0.75 | 2, darkening | Use |
| 0.75–0.85 | 3, discard point | Discard |
| 0.85–1.00 | 4, beyond discard | Discard |

The full colour model is in [`docs/contract_stage.md`](docs/contract_stage.md).

---

## Repository structure

```
vvm-reader/
├── vvm/                     core library
│   ├── config.py            all shared constants (thresholds, axes, levels, styles)
│   ├── render.py            clean VVM rendering from s (LCh colour model)
│   ├── augment.py           6 degradation axes
│   ├── segment.py           detection, ellipse fit, colour measurement, crop
│   ├── classical.py         WHO rule + isotonic colour-ratio reader
│   ├── model.py             TinyVVMNet (multi-task CNN, optional hybrid input)
│   └── datasets.py          uniform loader for synthetic and real sets
├── scripts/
│   ├── generate_dataset.py  synthetic sets (train, val, benchmark, style_in, style_out)
│   ├── make_print_sheet.py  A4 sheet of 20 known-stage VVMs for print-and-capture
│   ├── extract_features.py  run the IVP stage on every set
│   ├── train.py             train cnn / hybrid
│   ├── evaluate.py          all methods × all sets → results/
│   ├── make_figures.py      paper figures
│   └── weather_proxy.py     secondary weather-based triage analysis
├── docs/                    contracts shared with the paper author
├── models/                  frozen trained models
├── results/                 metrics, predictions, features, summary
├── figures/                 300 dpi figures
└── run_all.sh               full synthetic pipeline in one command
```

---

## Setup (macOS)

```bash
brew install python@3.11
git clone https://github.com/<your-username>/vvm-reader.git
cd vvm-reader
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Training uses the Apple Silicon GPU (MPS) automatically when available.

## Reproduce everything

```bash
bash run_all.sh
```

Or step by step:

```bash
python scripts/generate_dataset.py               # ~9,300 synthetic images → data/synth/
python scripts/make_print_sheet.py               # data/print_sheet/
python scripts/extract_features.py               # IVP stage → results/features, results/crops
python scripts/train.py --name cnn               # → models/cnn.pt
python scripts/train.py --name hybrid --hybrid   # → models/hybrid.pt
python scripts/evaluate.py                       # → results/metrics, predictions, summary.json
python scripts/make_figures.py                   # → figures/
```

To evaluate with the frozen models instead of retraining, skip the two `train.py` lines.

## Adding real data

Place photos and labels as described in [`docs/contract_schema.md`](docs/contract_schema.md):

```
data/real_print/images/*.jpg  +  data/real_print/labels.csv   (filename,vvm_id,condition,phone)
data/real_web/images/*        +  data/real_web/labels.csv     (filename,source_stage,annotator_a,annotator_b,source_url,license)
```

Then run:

```bash
python scripts/extract_features.py --sets real_print real_web
python scripts/evaluate.py
python scripts/make_figures.py
```

Real sets and inter-annotator agreement (Cohen's κ) are picked up automatically. The print-and-capture answer key (`print_sheet_labels.csv`) is deliberately not in the repo so that annotation stays blind; `make_print_sheet.py` regenerates it identically from a fixed seed.

## Secondary analysis: weather-based triage

```bash
python scripts/weather_proxy.py --demo                         # offline check of the kinetics
python scripts/weather_proxy.py --trips data/trips.csv --vvm VVM7
```

This estimates worst-case heat exposure, assuming the vial sat at ambient air temperature, from hourly Open-Meteo data and two-point Arrhenius kinetics fitted to WHO PQS VVM end-point times. It is a triage aid for deciding which vials to inspect first. **It is not a measurement of vial temperature.**

---

## Limitations

- **Synthetic training data.** Real VVMs are only sold in bulk (minimum order 50,000 units), so the primary dataset is synthetic. The print-and-capture set tests real cameras and lighting, but printed ink is not thermochromic material.
- **Out-of-distribution behaviour.** Beyond the training degradation range, the CNN can be confidently wrong in the unsafe direction, and the margin flag does not fully catch this.
- **Segmentation success criterion.** The criterion is approximate under strong curvature and perspective warps.
- **Small web reference set.** n = 25, so its confidence intervals are wide.
- **Weather proxy.** It uses outdoor air temperature, not cold-box or vial temperature.

## Future work

- An explicit input-quality gate for out-of-distribution images.
- Validation against real VVM hardware.
- IoT temperature-logger validation (DS18B20 + ESP32).

---

## Team

- **Akanksha Patil** — pipeline, experiments, evaluation
- **Sweety Jain** — paper, literature review, real-image data, annotation

NMIMS MPSTME Hyderabad, B.Tech Computer Engineering.

## Citation

Citation details will be added after the symposium.
