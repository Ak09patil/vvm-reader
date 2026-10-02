"""Train the CNN (or hybrid) on train crops, select best epoch on val MAE.

  python scripts/train.py --name cnn
  python scripts/train.py --name hybrid --hybrid
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vvm import config as C
from vvm.model import TinyVVMNet, n_params
from vvm.datasets import load_set


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")      # Apple Silicon GPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_arrays(name, root="data"):
    """Crops (N,3,64,64 float), extra IVP features (N,2), targets s (N,), stage idx (N,)."""
    z = np.load(f"results/crops/{name}.npz", allow_pickle=True)
    feats = pd.read_csv(f"results/features/{name}.csv").set_index("image_id")
    meta = load_set(name, root).set_index("image_id")
    ids = z["ids"]
    x = z["crops"].astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    x = (x - 0.5) / 0.25
    lr = feats.loc[ids, "log_ratio"].values if "log_ratio" in feats else np.full(len(ids), np.nan)
    extra = np.stack([np.clip(np.nan_to_num(lr, nan=0.0), -3, 3), (~np.isnan(lr)).astype(float)], 1)
    s = meta.loc[ids, "s"].values.astype(np.float32)
    stage = meta.loc[ids, "stage"].values.astype(np.int64) - 1
    return ids, x, extra.astype(np.float32), s, stage


def augment_batch(x):
    # label-preserving: VVM is symmetric under 90-degree rotations and flips
    k = np.random.randint(4)
    x = torch.rot90(x, k, dims=(2, 3))
    if np.random.rand() < 0.5:
        x = torch.flip(x, dims=(3,))
    return x


def predict(model, x, extra, device, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb = torch.from_numpy(x[i:i + bs]).to(device)
            eb = torch.from_numpy(extra[i:i + bs]).to(device)
            out.append(model(xb, eb)[0].cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="cnn")
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--cls-weight", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = get_device()
    print("device:", dev)

    _, xtr, etr, str_, ytr = load_arrays("train")
    _, xva, eva, sva, _ = load_arrays("val")
    model = TinyVVMNet(hybrid=a.hybrid).to(dev)
    print(f"params: {n_params(model):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

    os.makedirs("models", exist_ok=True)
    best, log = 1e9, []
    n = len(xtr)
    for ep in range(a.epochs):
        model.train()
        perm = np.random.permutation(n)
        tot = 0.0
        t0 = time.time()
        for i in range(0, n, a.bs):
            idx = perm[i:i + a.bs]
            xb = augment_batch(torch.from_numpy(xtr[idx])).to(dev)
            eb = torch.from_numpy(etr[idx]).to(dev)
            sb = torch.from_numpy(str_[idx]).to(dev)
            yb = torch.from_numpy(ytr[idx]).to(dev)
            s_pred, logits = model(xb, eb)
            loss = F.smooth_l1_loss(s_pred, sb, beta=0.05) + a.cls_weight * F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        pv = predict(model, xva, eva, dev)
        mae = float(np.mean(np.abs(pv - sva)))
        acc = float(np.mean(C.is_discard(pv) == C.is_discard(sva)))
        log.append({"epoch": ep + 1, "train_loss": tot / n, "val_mae": mae, "val_decision_acc": acc})
        flag = ""
        if mae < best:
            best = mae
            torch.save({"state": model.state_dict(), "hybrid": a.hybrid, "epoch": ep + 1, "val_mae": mae},
                       f"models/{a.name}.pt")
            flag = "  *saved"
        print(f"ep {ep+1:3d} loss {tot/n:.4f}  val MAE {mae:.4f}  val decision acc {acc:.3f}"
              f"  ({time.time()-t0:.1f}s){flag}")
    pd.DataFrame(log).to_csv(f"models/{a.name}_trainlog.csv", index=False)
    print(f"best val MAE {best:.4f} -> models/{a.name}.pt")


if __name__ == "__main__":
    main()
