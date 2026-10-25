"""Logistic-regression stack over our OOFs plus public OOF libraries (data/ext/*).

The public libraries (Chris Deotte's "124 public OOF" and "TFM OOF" datasets) hold
oof_<name>.npy / test_<name>.npy pairs for the same train/test row order.

Usage:
    python src/stack.py                 # ours + all external
    python src/stack.py --no-ext        # ours only
    python src/stack.py --C 0.05 --out sub_stack.csv
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import DATA, ROOT, SEED, load_comp

OUT = ROOT / "oof"


def logit(p):
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def to_logit(oof, test):
    """Some public files are already logits or ranks; map everything to the logit scale."""
    if oof.min() < 0 or oof.max() > 1 or test.min() < 0 or test.max() > 1:
        return oof.astype(np.float64), test.astype(np.float64)
    return logit(oof), logit(test)


def load_sets(use_ext, y, nte):
    names, oofs, tests = [], [], []
    for f in sorted(OUT.glob("*_oof.npy")):
        n = f.name[:-8]
        names.append(f"our:{n}")
        oofs.append(np.load(f))
        tests.append(np.load(OUT / f"{n}_test.npy"))
    n_ours = len(names)
    if use_ext:
        for d in sorted((DATA / "ext").iterdir()):
            for f in sorted(d.glob("oof_*.npy")):
                n = f.name[4:-4]
                t = d / f"test_{n}.npy"
                if not t.exists():
                    continue
                o, te = np.load(f, allow_pickle=False).ravel(), np.load(t, allow_pickle=False).ravel()
                if o.shape != y.shape or te.shape != (nte,) or not (np.isfinite(o).all() and np.isfinite(te).all()):
                    print("skip", d.name, n)
                    continue
                names.append(f"{d.name}:{n}")
                oofs.append(o)
                tests.append(te)
    Z, Zte = [], []
    for o, te in zip(oofs, tests):
        a, b = to_logit(o, te)
        Z.append(np.clip(a, -30, 30))
        Zte.append(np.clip(b, -30, 30))
    return names, np.column_stack(Z).astype(np.float32), np.column_stack(Zte).astype(np.float32), n_ours


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ext", action="store_true")
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--out", default="sub_stack.csv")
    args = ap.parse_args()

    _, test, sub = load_comp()
    y = np.load(OUT / "y.npy")
    names, Z, Zte, n_ours = load_sets(not args.no_ext, y, len(test))
    aucs = np.array([roc_auc_score(y, Z[:, j]) for j in range(Z.shape[1])])
    print(f"{len(names)} models ({n_ours} ours). best single: {names[aucs.argmax()]} {aucs.max():.5f}")
    for j in np.argsort(-aucs)[:8]:
        print(f"  {names[j]:50s} {aucs[j]:.5f}")

    # standardise columns so one C fits all scales
    mu, sd = Z.mean(0), Z.std(0) + 1e-9
    Zs, Ztes = (Z - mu) / sd, (Zte - mu) / sd
    oof, pred, coefs = np.zeros(len(y)), np.zeros(len(test)), []
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Zs, y):
        lr = LogisticRegression(C=args.C, max_iter=3000).fit(Zs[tr], y[tr])
        oof[va] = lr.predict_proba(Zs[va])[:, 1]
        pred += lr.predict_proba(Ztes)[:, 1] / 5
        coefs.append(lr.coef_.ravel())
    print(f"LR stack (C={args.C}) OOF AUC = {roc_auc_score(y, oof):.5f}")
    coef = np.mean(coefs, axis=0)
    print("top weights:")
    for j in np.argsort(-np.abs(coef))[:12]:
        print(f"  {names[j]:50s} {coef[j]:+.3f}")
    ours = [j for j in range(n_ours)]
    print("our models total |weight|:", round(float(np.abs(coef[ours]).sum()), 3),
          "of", round(float(np.abs(coef).sum()), 3))

    sub["satisfaction"] = pred
    path = ROOT / "submissions" / args.out
    path.parent.mkdir(exist_ok=True)
    sub.to_csv(path, index=False)
    np.save(OUT.parent / "submissions" / (Path(args.out).stem + "_oof.npy"), oof)
    print("saved", path)


if __name__ == "__main__":
    main()
