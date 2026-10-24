"""MLP with embeddings for every raw column treated as categorical (+ a few numeric inputs).

Usage:
    python src/nn.py [--epochs 8] [--seed 42]
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import NUM_COLS, ROOT, SEED
from fe import RAW, Data

OUT = ROOT / "oof"


class Net(nn.Module):
    def __init__(self, cards, n_num, emb=12, hidden=(256, 128), drop=0.2):
        super().__init__()
        self.embs = nn.ModuleList(nn.Embedding(c, min(emb, c)) for c in cards)
        d_in = sum(min(emb, c) for c in cards) + n_num
        layers = []
        for h in hidden:
            layers += [nn.Linear(d_in, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(drop)]
            d_in = h
        self.mlp = nn.Sequential(*layers, nn.Linear(d_in, 1))

    def forward(self, xc, xn):
        z = torch.cat([e(xc[:, i]) for i, e in enumerate(self.embs)] + [xn], dim=1)
        return self.mlp(z).squeeze(1)


def inputs(d: Data):
    comp = d.comp().reset_index(drop=True)
    xc = np.stack([pd.factorize(comp[c].fillna(-1))[0] for c in RAW], axis=1).astype(np.int64)
    num = pd.DataFrame({c: np.log1p(comp[c].fillna(0)) for c in NUM_COLS})
    for c in NUM_COLS:
        num[f"cnt_{c}"] = np.log1p(comp[c].map(comp[c].value_counts()).fillna(0))
    num = ((num - num.mean()) / num.std()).values.astype(np.float32)
    return xc, num, xc.max(axis=0) + 1


def predict(model, xc, xn, bs=8192):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(xc), bs):
            out.append(torch.sigmoid(model(torch.from_numpy(xc[i:i + bs]), torch.from_numpy(xn[i:i + bs]))))
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--bs", type=int, default=1024)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    d = Data()
    xc, xn, cards = inputs(d)
    y = d.y.astype(np.float32)
    xc_tr, xn_tr, xc_te, xn_te = xc[: d.ntr], xn[: d.ntr], xc[d.ntr:], xn[d.ntr:]
    oof, pred = np.zeros(d.ntr), np.zeros(d.nte)
    t0 = time.time()
    for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(xc_tr, y)):
        model = Net(cards, xn.shape[1])
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        steps = args.epochs * (len(tr) // args.bs + 1)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
        lossf = nn.BCEWithLogitsLoss()
        ctr, ntr_, ytr = torch.from_numpy(xc_tr[tr]), torch.from_numpy(xn_tr[tr]), torch.from_numpy(y[tr])
        best, best_state = 0, None
        for ep in range(args.epochs):
            model.train()
            perm = torch.randperm(len(tr))
            for i in range(0, len(tr), args.bs):
                b = perm[i:i + args.bs]
                opt.zero_grad()
                lossf(model(ctr[b], ntr_[b]), ytr[b]).backward()
                opt.step()
                sched.step()
            p = predict(model, xc_tr[va], xn_tr[va])
            auc = roc_auc_score(y[va], p)
            if auc > best:
                best, best_state = auc, {n: t.clone() for n, t in model.state_dict().items()}
            print(f"fold {k} ep {ep} auc {auc:.5f} ({time.time() - t0:.0f}s)", flush=True)
        model.load_state_dict(best_state)
        oof[va] = predict(model, xc_tr[va], xn_tr[va])
        pred += predict(model, xc_te, xn_te) / 5
    name = f"nn__emb__s{args.seed}"
    print(f"{name:40s} AUC={roc_auc_score(y, oof):.5f} ({time.time() - t0:.0f}s)", flush=True)
    np.save(OUT / f"{name}_oof.npy", oof)
    np.save(OUT / f"{name}_test.npy", pred)


if __name__ == "__main__":
    main()
