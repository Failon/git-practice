"""Robustness study of the final stack: per-fold AUC of several stackers on the same folds.

Variants
  lr        : logistic regression on standardised logits (v10)
  lr_dedup  : same after dropping near-duplicate members (corr > 0.9995)
  nnls      : non-negative weights on logits (no "A minus B" constructions), L-BFGS-B
  avg       : rank-average of lr_dedup and nnls

Usage:
    python src/stack_robust.py --out sub_v11_robust.csv
"""
import argparse

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import stack
from common import ROOT, SEED, load_comp


def dedup(Z, names, thr=0.9995):
    """Greedy: keep a member unless it correlates > thr with an already kept one."""
    C = np.corrcoef(Z[::7].T)
    keep = []
    for j in range(Z.shape[1]):
        if all(C[j, k] < thr for k in keep):
            keep.append(j)
    return keep


def fit_nnls(X, y, l2=1e-3):
    """Logistic loss with w >= 0 (intercept free)."""
    n, p = X.shape

    def f(theta):
        w, b = theta[:p], theta[p]
        z = X @ w + b
        pz = 1 / (1 + np.exp(-z))
        loss = np.mean(np.logaddexp(0, z) - y * z) + l2 * w @ w
        g = (pz - y) / n
        return loss, np.append(X.T @ g + 2 * l2 * w, g.sum())

    theta0 = np.append(np.full(p, 1.0 / p), 0.0)
    bounds = [(0, None)] * p + [(None, None)]
    r = minimize(f, theta0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
    return r.x[:p], r.x[p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sub_v11_robust.csv")
    args = ap.parse_args()

    _, test, sub = load_comp()
    y = np.load(stack.OUT / "y.npy")
    names, Z, Zte, _ = stack.load_sets(True, y, len(test))
    mu, sd = Z.mean(0), Z.std(0) + 1e-9
    Z, Zte = ((Z - mu) / sd).astype(np.float64), ((Zte - mu) / sd).astype(np.float64)
    keep = dedup(Z, names)
    print(f"{len(names)} members, {len(keep)} after de-duplication")

    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Z, y))
    oofs = {k: np.zeros(len(y)) for k in ["lr", "lr_dedup", "nnls"]}
    tests = {k: np.zeros(len(Zte)) for k in oofs}
    nnz = []
    for tr, va in folds:
        for k, cols in [("lr", slice(None)), ("lr_dedup", keep)]:
            m = LogisticRegression(C=0.1, max_iter=3000).fit(Z[tr][:, cols], y[tr])
            oofs[k][va] = m.decision_function(Z[va][:, cols])
            tests[k] += m.decision_function(Zte[:, cols]) / 5
        w, b = fit_nnls(Z[tr][:, keep], y[tr])
        oofs["nnls"][va] = Z[va][:, keep] @ w + b
        tests["nnls"] += (Zte[:, keep] @ w + b) / 5
        nnz.append(int((w > 1e-4).sum()))
    r = lambda p: rankdata(p) / len(p)
    oofs["avg"] = (r(oofs["lr_dedup"]) + r(oofs["nnls"])) / 2
    tests["avg"] = (r(tests["lr_dedup"]) + r(tests["nnls"])) / 2
    print(f"nnls members with weight > 0 per fold: {nnz}")

    print(f"{'variant':10s} {'OOF':>8s}  per-fold AUC                                     std")
    for k in oofs:
        fa = [roc_auc_score(y[va], oofs[k][va]) for _, va in folds]
        print(f"{k:10s} {roc_auc_score(y, oofs[k]):.5f}  " + " ".join(f"{a:.5f}" for a in fa) + f"  {np.std(fa):.5f}")
    np.save(stack.OUT.parent / "submissions" / "robust_oofs.npy", np.column_stack([oofs[k] for k in oofs]))

    best = "avg"
    sub["satisfaction"] = tests[best]
    sub.to_csv(ROOT / "submissions" / args.out, index=False)
    print("saved", args.out, "variant", best)
    for k in tests:
        print(f"test rank-corr {k} vs lr: {np.corrcoef(r(tests[k]), r(tests['lr']))[0, 1]:.6f}")


if __name__ == "__main__":
    main()
