# Contract 1 — Exposure stage definition (LOCKED)

**Continuous exposure stage `s ∈ [0, 1]`.** 0 = unexposed (square lightest). Higher = more cumulative heat.

| s range | WHO stage | Name | Decision |
|---|---|---|---|
| 0.00 ≤ s < 0.40 | 1 | start | USE |
| 0.40 ≤ s < 0.75 | 2 | darkening (still lighter than ring) | USE |
| 0.75 ≤ s < 0.85 | 3 | discard point (square = ring) | DISCARD |
| 0.85 ≤ s ≤ 1.00 | 4 | beyond discard (square darker than ring) | DISCARD |

**Discard threshold:** `s = 0.75` exactly. Decision = DISCARD iff `s ≥ 0.75`.

## How colour depends on s (generator)
Colours are defined in CIE LCh (Lab lightness L, chroma C, hue h).
- Ring (static reference): `(L_r, C_r, h)`; per image L_r ∈ [35, 50], C_r ∈ [22, 38].
- Square start colour: `(L_0, C_0, h)`; L_0 ∈ [86, 95], C_0 ∈ [0, 8].
- For `s ≤ 0.75`, with `t = s / 0.75`: `L = L_0 + (L_r − L_0)·t`, `C = C_0 + (C_r − C_0)·t` → at s = 0.75 the square equals the ring exactly.
- For `s > 0.75`, with `u = (s − 0.75) / 0.25`: `L = L_r − 15·u`, `C = C_r·(1 − 0.3u)` → darker than the ring.

Lightness is monotone in s, so the WHO rule (square lighter than ring ⇔ usable) holds by construction.

## Style families (for generalisation test)
- Training styles: ring hue ∈ {215, 235, 275, 295}° (±4), square half-side / ring radius q ∈ [0.40, 0.47].
- Held-out styles (`style_out` set only): hue ∈ {255, 315}°, q ∈ [0.48, 0.52]. Never seen in training.

## Real-image mapping
- Print-and-capture: `s` is known exactly from `print_sheet_labels.csv` (nominal; printer shifts absolute colour, but square and ring at s = 0.75 share identical RGB so they still print as a match).
- Web images: only WHO stage 1–4 is known (from the source's own label) → evaluated categorically, never with MAE.

## Paper wording (methods)
"MAE on continuous s is reported only where s is known exactly (synthetic and print-and-capture). Web-sourced reference images carry only categorical WHO-stage labels and are evaluated by categorical accuracy."
