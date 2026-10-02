# Contract 2 — Image & metadata schema

## Folder layout
```
data/synth/<set>/images/<image_id>.png      256×256 RGB PNG
data/synth/<set>/metadata.csv               one row per image
data/print_sheet/print_sheet.pdf            print this
data/print_sheet/print_sheet_labels.csv     vvm_id,s,stage,discard
data/real_print/images/<filename>.jpg       partner's photos
data/real_print/labels.csv                  partner fills (below)
data/real_web/images/<filename>             partner's web images
data/real_web/labels.csv                    partner fills (below)
```

## Synthetic sets
| set | n | content |
|---|---|---|
| train | 3000 | each axis independently on with p = 0.6, intensity t ~ U(0, 1) |
| val | 500 | same distribution as train; used for model selection and flag calibration only |
| benchmark | 5200 | 26 cells × 200: `clean` (all t = 0), `reference` (all t = 0.2), 6 axes × {low 0.33, med 0.66, high 1.0, extreme 1.5} with the other five fixed at 0.2. `extreme` lies beyond the training range (train t ≤ 1) and tests extrapolation |
| style_in | 300 | reference-level degradation, training styles |
| style_out | 300 | reference-level degradation, held-out styles |

s sampling: train/val = 70 % U(0, 1) + 30 % U(0.6, 0.9); benchmark/style cells are stratified, 60 % evenly over [0, 1] and 40 % evenly over [0.6, 0.9].

## metadata.csv columns (synthetic)
| column | type | meaning |
|---|---|---|
| image_id | str | `<set>_<5-digit index>` |
| set | str | set name |
| s | float | exposure stage (Contract 1) |
| stage | int | WHO stage 1–4 |
| discard | int | 1 if s ≥ 0.75 |
| cell_axis, cell_level | str | benchmark cell (`clean/none`, `reference/baseline`, `<axis>/<low|med|high>`, `style/seen|heldout`); empty for train/val |
| style_heldout | int | 1 if held-out style |
| t_illumination … t_occlusion | float | intensity of each axis, 0 = off |
| gt_cx, gt_cy, gt_r | float | true disc centre/radius after geometric warp (segmentation scoring) |
| style_json | json | all rendering parameters |
| aug_json | json | all degradation parameters actually drawn |

## Degradation axes (applied in this order)
| axis | what t controls (t = 1 is maximum) |
|---|---|
| geometric (e) | rotation ±180°·t, scale, perspective corner jitter 12 %·t, cylindrical vial curvature up to 1.2·t rad |
| illumination (a) | warm/cool colour cast up to 30 %·t, linear shadow gradient up to 60 %·t, exposure −30…+20 %·t |
| occlusion (f) | smudge blobs covering 25 %·t of the disc + 50 % chance of specular glare |
| blur (c) | defocus σ = 3.5·t px or motion blur length 1 + 12·t px |
| noise (b) | Gaussian read noise σ = 22·t + signal-dependent shot noise |
| jpeg (d) | quality 95 − 85·t |

## Partner files
**data/real_print/labels.csv**
```
filename,vvm_id,condition,phone
IMG_0412.jpg,7,daylight,iphone13
```
condition ∈ {daylight, tubelight, warm_bulb, shadow, angled, curved}; phone = short model name.

**data/real_web/labels.csv**
```
filename,source_stage,annotator_a,annotator_b,source_url,license
who_card_03.png,2,2,3,https://...,CC BY-NC-SA 3.0 IGO
```
source_stage = stage as labelled by the source itself; annotator_a/b = blind labels (1–4).
