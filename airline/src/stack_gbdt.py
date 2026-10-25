"""Level-2 LightGBM stacker: top model logits + raw features, same 5 folds.

Lets the stack trust different models in different segments (class, travel type, ...),
which a single logistic regression cannot.

Usage:
    python src/stack_gbdt.py --top 40 --out sub_stack_gbdt.csv
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import stack
from common import ROOT, SEED, load_comp
from fe import Data, block_cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=40, help="models kept, ranked by |LR coefficient|")
    ap.add_argument("--raw", action="store_true", help="also feed raw + count features")
    ap.add_argument("--out", default="sub_stack_gbdt.csv")
    args = ap.parse_args()

    _, test, sub = load_comp()
    y = np.load(stack.OUT / "y.npy")
    names, Z, Zte, _ = stack.load_sets(True, y, len(test))

    # pick models by a full-data LR (selection only; the CV below refits everything per fold)
    mu, sd = Z.mean(0), Z.std(0) + 1e-9
    coef = LogisticRegression(C=0.1, max_iter=3000).fit((Z - mu) / sd, y).coef_.ravel()
    keep = np.argsort(-np.abs(coef))[: args.top]
    X = pd.DataFrame(Z[:, keep], columns=[f"m{j}" for j in keep])
    Xte = pd.DataFrame(Zte[:, keep], columns=X.columns)
    if args.raw:
        d = Data()
        extra = pd.concat([d.comp().reset_index(drop=True), block_cnt(d).reset_index(drop=True)], axis=1)
        X = pd.concat([X, extra.iloc[: d.ntr].reset_index(drop=True)], axis=1)
        Xte = pd.concat([Xte, extra.iloc[d.ntr:].reset_index(drop=True)], axis=1)

    params = dict(n_estimators=5000, learning_rate=0.02, num_leaves=15, max_depth=4, min_child_samples=500,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10, verbose=-1,
                  random_state=SEED)
    oof, pred = np.zeros(len(y)), np.zeros(len(Xte))
    for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y)):
        m = lgb.LGBMClassifier(**params).fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], eval_metric="auc",
                                             callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        pred += m.predict_proba(Xte)[:, 1] / 5
        print(f"  fold {k}: {roc_auc_score(y[va], oof[va]):.5f} iters={m.best_iteration_}", flush=True)
    print(f"GBDT stack (top {args.top}, raw={args.raw}) OOF AUC = {roc_auc_score(y, oof):.5f}")
    sub["satisfaction"] = pred
    sub.to_csv(ROOT / "submissions" / args.out, index=False)
    np.save(ROOT / "submissions" / (args.out[:-4] + "_oof.npy"), oof)


if __name__ == "__main__":
    main()
