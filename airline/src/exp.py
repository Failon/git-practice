"""Idea experiments on the competition data with one LightGBM setup.

Every experiment saves OOF + test predictions to oof/ so they can be blended later.

Usage:
    python src/exp.py base orig catfeat count te
"""
import argparse
import itertools
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import (CAT_COLS, NUM_COLS, RATING_COLS, ROOT, SEED, TARGET, base_encode,
                    load_comp, load_original)

warnings.filterwarnings("ignore")
OUT = ROOT / "oof"
OUT.mkdir(exist_ok=True)

LGB_PARAMS = dict(n_estimators=10000, learning_rate=0.05, num_leaves=63, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0,
                  random_state=SEED, verbose=-1)
RATING_PAIRS = list(itertools.combinations(RATING_COLS + ["Class", "Type of Travel"], 2))


def count_encode(full: pd.DataFrame, cols) -> pd.DataFrame:
    """Frequency of each value (and of rating pairs) over train+test."""
    out = pd.DataFrame(index=full.index)
    for c in cols:
        out[f"cnt_{c}"] = full[c].map(full[c].value_counts())
    for a, b in RATING_PAIRS:
        key = full[a].astype(str) + "_" + full[b].astype(str)
        out[f"cnt_{a}|{b}"] = key.map(key.value_counts())
    return out


def pair_keys(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({f"{a}|{b}": df[a] * 10 + df[b] for a, b in RATING_PAIRS}, index=df.index)


def target_encode(tr_keys, ytr, ap_keys, smooth=20, inner=5):
    """Smoothed mean-target encoding; out-of-fold inside the training part."""
    prior = ytr.mean()
    tr_out = np.zeros(tr_keys.shape)
    ap_out = np.zeros(ap_keys.shape)
    yser = pd.Series(ytr, index=tr_keys.index)
    for j, c in enumerate(tr_keys.columns):
        stats = yser.groupby(tr_keys[c]).agg(["sum", "count"])
        enc = (stats["sum"] + prior * smooth) / (stats["count"] + smooth)
        ap_out[:, j] = ap_keys[c].map(enc).fillna(prior).values
    kf = StratifiedKFold(inner, shuffle=True, random_state=SEED + 1)
    for a, b in kf.split(tr_keys, ytr):
        ya = yser.iloc[a]
        for j, c in enumerate(tr_keys.columns):
            stats = ya.groupby(tr_keys[c].iloc[a]).agg(["sum", "count"])
            enc = (stats["sum"] + prior * smooth) / (stats["count"] + smooth)
            tr_out[b, j] = tr_keys[c].iloc[b].map(enc).fillna(prior).values
    cols = [f"te_{c}" for c in tr_keys.columns]
    return (pd.DataFrame(tr_out, columns=cols, index=tr_keys.index),
            pd.DataFrame(ap_out, columns=cols, index=ap_keys.index))


def run(name, train, test, orig=None, cat_cols=None, te=False, params=None):
    params = {**LGB_PARAMS, **(params or {})}
    feats = [c for c in train.columns if c not in ("id", TARGET)]
    X, Xte, y = train[feats], test[feats], train[TARGET].values
    if cat_cols:
        X, Xte = X.copy(), Xte.copy()
        for c in cat_cols:
            dtype = pd.CategoricalDtype(sorted(set(X[c]) | set(Xte[c])))
            X[c], Xte[c] = X[c].astype(dtype), Xte[c].astype(dtype)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof, pred, iters = np.zeros(len(X)), np.zeros(len(Xte)), []
    t0 = time.time()
    for k, (tr, va) in enumerate(skf.split(X, y)):
        Xtr, ytr, Xva = X.iloc[tr], y[tr], X.iloc[va]
        Xte_k = Xte
        if orig is not None:
            Xtr = pd.concat([Xtr, orig[feats].astype(Xtr.dtypes.to_dict())], ignore_index=True)
            ytr = np.concatenate([ytr, orig[TARGET].values])
        if te:
            ktr = pair_keys(Xtr.reset_index(drop=True))
            t_tr, t_va = target_encode(ktr, ytr, pair_keys(Xva))
            _, t_te = target_encode(ktr, ytr, pair_keys(Xte), inner=2)
            Xtr = pd.concat([Xtr.reset_index(drop=True), t_tr], axis=1)
            Xva = pd.concat([Xva, t_va], axis=1)
            Xte_k = pd.concat([Xte, t_te], axis=1)
        m = lgb.LGBMClassifier(**params)
        m.fit(Xtr, ytr, eval_set=[(Xva, y[va])], eval_metric="auc",
              callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va] = m.predict_proba(Xva)[:, 1]
        pred += m.predict_proba(Xte_k)[:, 1] / 5
        iters.append(m.best_iteration_)
    auc = roc_auc_score(y, oof)
    print(f"{name:12s} AUC={auc:.5f}  iters={iters}  ({time.time() - t0:.0f}s)", flush=True)
    np.save(OUT / f"{name}_oof.npy", oof)
    np.save(OUT / f"{name}_test.npy", pred)
    return auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="+")
    args = ap.parse_args()

    train, test, _ = load_comp()
    train, test = base_encode(train), base_encode(test)
    orig = base_encode(load_original())
    np.save(OUT / "y.npy", train[TARGET].values)

    for e in args.exps:
        if e == "base":
            run("lgb_base", train, test)
        elif e == "orig":
            run("lgb_orig", train, test, orig=orig)
        elif e == "catfeat":
            run("lgb_catfeat", train, test, cat_cols=CAT_COLS + RATING_COLS)
        elif e == "count":
            full = pd.concat([train, test], ignore_index=True)
            ce = count_encode(full, RATING_COLS + NUM_COLS)
            tr2 = pd.concat([train, ce.iloc[:len(train)].reset_index(drop=True)], axis=1)
            te2 = pd.concat([test, ce.iloc[len(train):].reset_index(drop=True)], axis=1)
            run("lgb_count", tr2, te2)
        elif e == "te":
            run("lgb_te", train, test, te=True)
        elif e == "slow":
            run("lgb_slow", train, test, params=dict(learning_rate=0.02, num_leaves=31,
                                                      min_child_samples=100, colsample_bytree=0.5))
        else:
            raise SystemExit(f"unknown experiment {e}")


if __name__ == "__main__":
    main()
