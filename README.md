# VVM Reader — smartphone reading of Vaccine Vial Monitors

Synthetic-data benchmark + IVP segmentation + lightweight CNN for reading WHO Vaccine Vial Monitors from phone images, with a safety-oriented confidence flag.

## Setup (macOS)
```bash
brew install python@3.11
cd ~/Projects/vvm-reader
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Full pipeline
```bash
python scripts/generate_dataset.py          # ~5,600 synthetic images -> data/synth/
python scripts/make_print_sheet.py          # data/print_sheet/print_sheet.pdf -> send to partner
python scripts/extract_features.py          # IVP: detect, measure, crop -> results/features, results/crops
python scripts/train.py --name cnn          # CNN -> models/cnn.pt
python scripts/train.py --name hybrid --hybrid
python scripts/evaluate.py                  # all metrics -> results/
python scripts/make_figures.py              # figures/
```
Or run everything: `bash run_all.sh`.

When the partner's real data arrives, put it in `data/real_print/` and `data/real_web/` (see docs/contract_schema.md), then re-run `extract_features.py` and `evaluate.py` — real sets are picked up automatically.

## Secondary analysis
```bash
python scripts/weather_proxy.py --demo
python scripts/weather_proxy.py --trips data/trips.csv --vvm VVM7
```

## Contracts
- docs/contract_stage.md — exposure-stage definition
- docs/contract_schema.md — image + metadata schema
- docs/contract_results.md — results file formats
- docs/partner_capture_protocol.md — photo + annotation protocol
