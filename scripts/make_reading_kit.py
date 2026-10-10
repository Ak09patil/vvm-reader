"""Blinded human-reader kit for the real print-and-capture photos.

  python scripts/make_reading_kit.py            # 5 photos per VVM = 100 images
  python scripts/make_reading_kit.py --per-vvm 3

Writes
  data/human_reading/reader.html   ONE self-contained file: send it to each reader
                                   (WhatsApp/email). Opens in any browser, works offline.
  data/human_reading/key.csv       the answer key. Do NOT open it, do NOT send it.

Blinding: each photo is cropped around the detected disc and everything outside the
disc margin is greyed out, so the printed "ID xx" label is never visible. Images get
random names and a random order. Photos are spread evenly over the 20 VVMs and over the
phone x condition combinations.
"""
import argparse, base64, json, os, random, sys
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vvm import segment as S

IMG = "data/real_print/images"
OUT = "data/human_reading"
SEED = 2026


def blinded_crop(path, out_px=360):
    img = S.load_photo(path, size=640)
    det = S.detect(img)
    if det is None:
        return None
    r = max(det["a"], det["b"])
    if not 0.2 < 2 * r / img.shape[1] < 0.85:          # skip mis-detections / extreme framing
        return None
    half = 1.25 * r
    pad = int(half) + 4
    big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    x0, y0 = int(det["cx"] - half) + pad, int(det["cy"] - half) + pad
    side = int(2 * half)
    crop = big[y0:y0 + side, x0:x0 + side].copy()
    # grey out everything beyond 1.18x the fitted ellipse -> printed ID label never visible
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float32)
    dx, dy = xx - (det["cx"] - (x0 - pad)), yy - (det["cy"] - (y0 - pad))
    t = np.radians(det["angle"])
    u = (dx * np.cos(t) + dy * np.sin(t)) / det["a"]
    v = (-dx * np.sin(t) + dy * np.cos(t)) / det["b"]
    crop[np.sqrt(u * u + v * v) > 1.18] = (150, 150, 150)
    return cv2.resize(crop, (out_px, out_px), interpolation=cv2.INTER_AREA)


def to_data_uri(rgb):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VVM Reading Task</title>
<style>
  :root { --bg:#d6d6d6; --card:#f2f2f2; --ink:#1d1d1d; --muted:#555; --use:#2f6f3e; --disc:#9b2c2c; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.45 -apple-system, system-ui, sans-serif; }
  main { max-width:560px; margin:0 auto; padding:16px; }
  .card { background:var(--card); border-radius:12px; padding:18px; }
  h1 { font-size:20px; margin:0 0 8px; }
  p, li { color:var(--muted); }
  input { width:100%; padding:10px; font-size:16px; border:1px solid #aaa; border-radius:8px; margin:6px 0 12px; }
  button { font:inherit; border:0; border-radius:10px; padding:12px 14px; cursor:pointer; }
  .primary { background:#1d1d1d; color:#fff; width:100%; }
  .img { display:block; width:100%; max-width:420px; aspect-ratio:1; margin:8px auto 14px; border-radius:8px; background:#969696; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
  .grid button { text-align:left; background:#fff; border:2px solid transparent; }
  .grid button:hover { border-color:#888; }
  .grid b { display:block; }
  .use b { color:var(--use); } .dis b { color:var(--disc); }
  .bar { display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; color:var(--muted); font-size:14px; }
  .ghost { background:transparent; color:var(--muted); padding:6px 8px; }
  .hidden { display:none; }
  table { border-collapse:collapse; width:100%; font-size:14px; } td { padding:4px 6px; border-bottom:1px solid #ddd; }
</style></head>
<body><main>
<div id="start" class="card">
  <h1>Vaccine vial monitor reading task</h1>
  <p>You will see __N__ photographs of a vaccine vial monitor: a coloured circle with a square inside it.
     For each one, compare the <b>inner square</b> with the <b>outer circle</b> and choose a stage:</p>
  <table>
    <tr><td><b>Stage 1</b></td><td>Square clearly lighter than the circle (use)</td></tr>
    <tr><td><b>Stage 2</b></td><td>Square lighter, but starting to darken (use)</td></tr>
    <tr><td><b>Stage 3</b></td><td>Square the <b>same</b> colour as the circle (discard)</td></tr>
    <tr><td><b>Stage 4</b></td><td>Square <b>darker</b> than the circle (discard)</td></tr>
  </table>
  <ul>
    <li>Take a few seconds per image and go with your judgement. There is no time limit.</li>
    <li>Do not zoom, edit or measure the images, and do not discuss them with anyone.</li>
    <li>Keyboard: press 1, 2, 3 or 4. You can go back to correct a mistake.</li>
  </ul>
  <label>Your name or initials</label>
  <input id="name" autocomplete="off" placeholder="e.g. SJ">
  <button class="primary" onclick="begin()">Start</button>
</div>

<div id="task" class="card hidden">
  <div class="bar"><span id="prog"></span><button class="ghost" onclick="back()">&larr; Back</button></div>
  <img id="img" class="img" alt="vaccine vial monitor">
  <div class="grid">
    <button class="use" onclick="answer(1)"><b>1</b>Clearly lighter</button>
    <button class="use" onclick="answer(2)"><b>2</b>Lighter, darkening</button>
    <button class="dis" onclick="answer(3)"><b>3</b>Same as circle</button>
    <button class="dis" onclick="answer(4)"><b>4</b>Darker than circle</button>
  </div>
</div>

<div id="done" class="card hidden">
  <h1>All done, thank you!</h1>
  <p>Download your answers and send the file back. Nothing is uploaded anywhere.</p>
  <button class="primary" onclick="download()">Download my answers (.csv)</button>
</div>
</main>
<script>
const ITEMS = __ITEMS__;
const KEY = "vvm_reading_v1";
let st = { name:"", i:0, ans:{} }, t0 = 0;
function save(){ try { localStorage.setItem(KEY, JSON.stringify(st)); } catch(e) {} }
try { const s = JSON.parse(localStorage.getItem(KEY) || "null"); if (s && s.ans) st = s; } catch(e) {}
if (st.name) document.getElementById("name").value = st.name;
function show(id){ for (const x of ["start","task","done"]) document.getElementById(x).classList.toggle("hidden", x !== id); }
function begin(){
  const n = document.getElementById("name").value.trim();
  if (!n) { alert("Please enter your name or initials."); return; }
  if (st.name && st.name !== n) st = { name:n, i:0, ans:{} };
  st.name = n; save(); render();
}
function render(){
  if (st.i >= ITEMS.length) { show("done"); return; }
  show("task");
  document.getElementById("img").src = ITEMS[st.i].src;
  document.getElementById("prog").textContent = (st.i + 1) + " / " + ITEMS.length;
  t0 = performance.now();
}
function answer(k){
  if (st.i >= ITEMS.length) return;
  st.ans[ITEMS[st.i].id] = { stage:k, sec:Math.round((performance.now() - t0) / 100) / 10 };
  st.i++; save(); render();
}
function back(){ if (st.i > 0) { st.i--; save(); render(); } }
document.addEventListener("keydown", e => {
  if (document.getElementById("task").classList.contains("hidden")) return;
  if ("1234".includes(e.key)) answer(+e.key);
  if (e.key === "Backspace" || e.key === "ArrowLeft") back();
});
function download(){
  let csv = "reader,blind_id,stage,seconds\n";
  for (const it of ITEMS) { const a = st.ans[it.id]; if (a) csv += [st.name, it.id, a.stage, a.sec].join(",") + "\n"; }
  const url = URL.createObjectURL(new Blob([csv], { type:"text/csv" }));
  const a = document.createElement("a");
  a.href = url; a.download = "responses_" + st.name.replace(/[^A-Za-z0-9]+/g, "_") + ".csv";
  document.body.appendChild(a); a.click(); a.remove();
}
if (st.name && st.i > 0) render();
</script>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-vvm", type=int, default=5)
    a = ap.parse_args()
    lab = pd.read_csv("data/real_print/labels.csv")
    key_s = pd.read_csv("data/print_sheet/print_sheet_labels.csv").set_index("vvm_id")
    combos = sorted(lab[["phone", "condition"]].drop_duplicates().itertuples(index=False, name=None))
    rng = random.Random(SEED)
    rng.shuffle(combos)

    chosen = []
    for v in range(1, 21):
        start = (a.per_vvm * (v - 1)) % len(combos)
        got = 0
        for ph, c in combos[start:] + combos[:start]:
            row = lab[(lab.vvm_id == v) & (lab.phone == ph) & (lab.condition == c)]
            if len(row) != 1:
                continue
            f = row.filename.iloc[0]
            crop = blinded_crop(os.path.join(IMG, f))
            if crop is None:
                print(f"  skipped {f} (disc not detected cleanly)")
                continue
            chosen.append((f, v, ph, c, crop))
            got += 1
            if got == a.per_vvm:
                break
        if got < a.per_vvm:
            print(f"  ! VVM {v:02d}: only {got} usable photos")
        print(f"\r  prepared VVM {v:02d}/20", end="", flush=True)
    print()

    rng.shuffle(chosen)
    os.makedirs(OUT, exist_ok=True)
    items, key = [], []
    for i, (f, v, ph, c, crop) in enumerate(chosen, 1):
        bid = f"img_{i:03d}"
        items.append({"id": bid, "src": to_data_uri(crop)})
        s = float(key_s.s[v])
        key.append({"blind_id": bid, "filename": f, "vvm_id": v, "s": s,
                    "stage": int(key_s.stage[v]), "discard": int(key_s.discard[v]), "phone": ph, "condition": c})
    pd.DataFrame(key).to_csv(f"{OUT}/key.csv", index=False)
    html = HTML.replace("__ITEMS__", json.dumps(items)).replace("__N__", str(len(items)))
    with open(f"{OUT}/reader.html", "w") as fh:
        fh.write(html)
    os.makedirs(f"{OUT}/responses", exist_ok=True)
    mb = os.path.getsize(f"{OUT}/reader.html") / 1e6
    print(f"{len(items)} blinded images -> {OUT}/reader.html ({mb:.1f} MB)")
    print(f"answer key -> {OUT}/key.csv  (do not open, do not send)")
    print(f"put the readers' returned CSV files in {OUT}/responses/")


if __name__ == "__main__":
    main()
