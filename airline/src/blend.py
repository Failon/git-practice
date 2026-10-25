"""Hill-climbing rank blend over saved OOF predictions, then write a submission.

Usage:
    python src/blend.py                      # all oof/*_oof.npy
    python src/blend.py lgb_base lgb_orig    # chosen models
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from common import ROOT, load_comp

OUT = ROOT / "oof"


def rank(p):
    return rankdata(p) / len(p)


def hill_climb(oofs: dict, y, steps=50):
    """Greedy forward selection with replacement (Caruana et al.)."""
    names = list(oofs)
    R = {n: rank(oofs[n]) for n in names}
    best = max(names, key=lambda n: roc_auc_score(y, R[n]))
    counts = {best: 1}
    cur = R[best].copy()
    score = roc_auc_score(y, cur)
    for _ in range(steps):
        k = sum(counts.values())
        cand = {n: roc_auc_score(y, (cur * k + R[n]) / (k + 1)) for n in names}
        n, s = max(cand.items(), key=lambda kv: kv[1])
        if s <= score + 1e-7:
            break
        cur = (cur * k + R[n]) / (k + 1)
        counts[n] = counts.get(n, 0) + 1
        score = s
    total = sum(counts.values())
    return {n: c / total for n, c in counts.items()}, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*")
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    y = np.load(OUT / "y.npy")
    names = args.models or sorted(p.name[:-8] for p in OUT.glob("*_oof.npy"))
    oofs = {n: np.load(OUT / f"{n}_oof.npy") for n in names}
    for n, p in sorted(oofs.items(), key=lambda kv: -roc_auc_score(y, kv[1])):
        print(f"{n:16s} {roc_auc_score(y, p):.5f}")

    w, score = hill_climb(oofs, y)
    print("weights:", {k: round(v, 3) for k, v in w.items()})
    print(f"hill-climb OOF AUC = {score:.5f}")

    # stacking: logistic regression on logits, scored with its own outer CV
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    Z = np.column_stack([logit(oofs[n]) for n in names])
    Zte = np.column_stack([logit(np.load(OUT / f"{n}_test.npy")) for n in names])
    lr = LogisticRegression(C=0.1, max_iter=2000)
    cv = StratifiedKFold(5, shuffle=True, random_state=7)
    stack_score = roc_auc_score(y, cross_val_predict(lr, Z, y, cv=cv, method="predict_proba")[:, 1])
    print(f"LR-stack OOF AUC = {stack_score:.5f}")

    _, _, sub = load_comp()
    if stack_score > score:
        sub["satisfaction"] = lr.fit(Z, y).predict_proba(Zte)[:, 1]
        print("using LR stack")
    else:
        sub["satisfaction"] = sum(wt * rank(np.load(OUT / f"{n}_test.npy")) for n, wt in w.items())
        print("using hill-climb blend")
    path = ROOT / "submissions" / args.out
    path.parent.mkdir(exist_ok=True)
    sub.to_csv(path, index=False)
    print("saved", path)


if __name__ == "__main__":
    main()
