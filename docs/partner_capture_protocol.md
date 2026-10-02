# Partner protocol — print-and-capture + web reference set

## A. Print-and-capture (target: ~160 photos, by Day 5)
1. Print `data/print_sheet/print_sheet.pdf` in **colour**, **"Actual size" / 100 % scale** (not "fit to page"), on plain matte paper. Print 2 copies (one stays flat, one gets taped around a bottle).
2. Do **not** open `print_sheet_labels.csv` before finishing Part B labelling (keeps your annotations blind).
3. For each of the 20 VVMs (ID 01–20), photograph under these 6 conditions:
   - `daylight` near a window, no direct sun
   - `tubelight` indoor white light
   - `warm_bulb` yellow bulb only
   - `shadow` your hand/phone casting a shadow across half the VVM
   - `angled` phone tilted ~40° from straight-on
   - `curved` sheet copy 2 taped around a bottle, VVM on the curve
4. Framing: the target VVM in the **centre**, filling about **1/3 to 1/2 of the frame width**. Neighbouring VVMs partly visible is fine; the centre one is what gets read. No flash, no zoom filters, no portrait mode.
5. Use both phones if possible (20 × 6 × 2 = 240; minimum 20 × 6 × 1 = 120).
6. iPhone: set Camera → Formats → **Most Compatible** so files are JPEG, not HEIC.
7. Fill `data/real_print/labels.csv`: `filename,vvm_id,condition,phone`.

## B. Web reference set (25 images)
- Only images whose stage is **labelled by the source itself** (e.g. WHO VVM training cards/posters, UNICEF guides). Record URL + licence.
- Both of us label every image 1–4 **blind** (cover the source label), independently, before comparing.
- Fill `data/real_web/labels.csv`: `filename,source_stage,annotator_a,annotator_b,source_url,license`.

## C. Citations to verify (Days 1–2)
Murhekar et al. 2013 figures; full S-VVM reference; the FreshFusionNet authors (placeholder in plan); RipeTrack; MobiChem; ColorSensing patents; Waldhans et al. 2023; WHO PQS VVM spec E06/IN05 (end-point days used in `scripts/weather_proxy.py`); WHO procurement minimum order. Drop anything unconfirmed.
