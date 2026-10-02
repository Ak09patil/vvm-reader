# Contract 3 — Evaluation results files

Everything is produced by `python scripts/evaluate.py`. The partner copies numbers from these files only — never retyped by hand.

## results/metrics/metrics.csv  (main table source)
One row per (set, method, group).
| column | meaning |
|---|---|
| set | val, benchmark, style_in, style_out, real_print, real_web |
| method | who_rule, classical, cnn, hybrid |
| group_type, group | `all/ALL`; benchmark `cell/<axis>:<level>`; real_print `condition/<condition>|<phone>` |
| n | images in group |
| seg_success_rate | fraction where detected disc centre is within 15 % of radius and radius within 25 % (synthetic only) |
| stage_acc, stage_f1_macro | 4-class WHO stage (NaN for who_rule, which only outputs a decision) |
| decision_acc (+ _ci_lo, _ci_hi) | USE/DISCARD accuracy with 95 % Wilson interval |
| unsafe_error_rate | fraction with true DISCARD predicted USE (the dangerous error) |
| false_discard_rate | fraction with true USE predicted DISCARD (wastage) |
| mae | mean abs error on s (only where s is known: synthetic, real_print) |
| coverage | fraction of readings NOT flagged |
| unsafe_error_rate_unflagged | unsafe errors among unflagged readings ÷ unflagged readings |
| delta | flag margin used for this method |

## Confidence flag (method, locked)
A reading is flagged "re-check manually" when `|s_pred − 0.75| ≤ δ`. δ is the smallest margin (0.005 steps) for which the unsafe-error rate among unflagged **validation** readings is ≤ 1 %. δ is then frozen and applied unchanged to every other set. Justification for the paper: softmax confidence from a model trained on synthetic data is typically over-confident under distribution shift; a margin around the WHO decision boundary is interpretable, tied to the clinical decision, and auditable.

## Other files
| file | content |
|---|---|
| results/predictions/<set>.csv | per image × method: s_true, s_pred, stage_true/pred, discard_true/pred, flagged, group |
| results/metrics/coverage_risk_val.csv, coverage_risk_benchmark.csv | method, delta, coverage, unsafe_error_rate_unflagged (coverage–risk curve) |
| results/metrics/annotator_agreement.csv | pair, n, agreement_rate, cohen_kappa, cohen_kappa_linear_weighted |
| results/features/<set>.csv | IVP stage output: detection, ring/square colour medians, log luminance ratio, seg_ok |
| results/summary.json | frozen δ per method, model parameter counts, CPU latency per crop |
| results/weather_proxy.csv | secondary analysis (worst-case ambient exposure triage) |
| figures/*.png | 300 dpi paper figures |
