#!/bin/bash
set -e
python scripts/generate_dataset.py
python scripts/make_print_sheet.py
python scripts/extract_features.py
python scripts/train.py --name cnn
python scripts/train.py --name hybrid --hybrid
python scripts/evaluate.py
python scripts/make_figures.py
