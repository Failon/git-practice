"""Train one model on one feature spec with 5-fold CV; save OOF/test predictions.

Usage:
    python src/train.py lgb cnt+orig+te
    python src/train.py xgb cnt+orig2+tecore --seed 1
"""
import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

from common import CAT_COLS, RATING_COLS, ROOT, SEED
from fe import build

warnings.filterwarnings("ignore")
OUT = ROOT / "oof"
OUT.mkdir(exist_ok=True)

PARAMS = {
    "lgb": dict(n_estimators=20000, learning_rate=0.05, num_leaves=63, min_child_samples=40,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0, verbose=-1),
    "xgb": dict(n_estimators=20000, learning_rate=0.05, max_depth=7, min_child_weight=3,
                subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0, tree_method="hist",
                eval_metric="auc", early_stopping_rounds=200),
    "cat": dict(iterations=20000, learning_rate=0.08, depth=7, l2_leaf_reg=3, eval_metric="AUC",
                od_type="Iter", od_wait=200, verbose=0, thread_count=2, border_count=128),
}


def fit_predict(kind, params, seed, Xtr, ytr, Xva, yva, Xte):
    if kind == "lgb":
        import lightgbm as lgb
        cats = [c for c in Xtr.columns if c.startswith("c_")]
        if cats:
            # shared category sets across train/valid/test so codes agree
            Xtr, Xva, Xte = Xtr.copy(), Xva.copy(), Xte.copy()
            for c in cats:
                dt = pd.CategoricalDtype(sorted(set(Xtr[c]) | set(Xva[c]) | set(Xte[c])))
                Xtr[c], Xva[c], Xte[c] = Xtr[c].astype(dt), Xva[c].astype(dt), Xte[c].astype(dt)
        m = lgb.LGBMClassifier(**params, random_state=seed)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
              callbacks=[lgb.early_stopping(200, verbose=False)])
        it = m.best_iteration_
    elif kind == "xgb":
        import xgboost as xgb
        m = xgb.XGBClassifier(**params, random_state=seed)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        it = m.best_iteration
    else:
        from catboost import CatBoostClassifier
        # ratings, categoricals and Age/Distance (as route id) go in as native categorical features
        cats = CAT_COLS + RATING_COLS + ["Age", "Flight Distance"]
        Xtr, Xva, Xte = (pd.concat([X, X[cats].add_prefix("c_").astype(int)], axis=1) for X in (Xtr, Xva, Xte))
        m = CatBoostClassifier(**params, random_seed=seed, allow_writing_files=False)
        m.fit(Xtr, ytr, cat_features=[f"c_{c}" for c in cats], eval_set=(Xva, yva), use_best_model=True)
        it = m.get_best_iteration()
    return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], it


def run(kind, spec, seed=SEED, name=None, params=None, folds=5):
    t0 = time.time()
    Xtr_all, Xte, te_keys, y, _ = build(spec)
    params = {**PARAMS[kind], **(params or {})}
    skf = StratifiedKFold(folds, shuffle=True, random_state=SEED)
    oof, pred, its = np.zeros(len(y)), np.zeros(len(Xte)), []
    for tr, va in skf.split(Xtr_all, y):
        Xtr, Xva, Xte_k = Xtr_all.iloc[tr], Xtr_all.iloc[va], Xte
        if te_keys is not None:
            ktr, kte = te_keys
            enc = TargetEncoder(target_type="binary", cv=5, shuffle=True, random_state=seed)
            cols = [f"te_{c}" for c in ktr.columns]
            t_tr = enc.fit_transform(ktr.iloc[tr].astype(str), y[tr])
            t_va = enc.transform(ktr.iloc[va].astype(str))
            t_te = enc.transform(kte.astype(str))
            Xtr = pd.concat([Xtr.reset_index(drop=True), pd.DataFrame(t_tr, columns=cols)], axis=1)
            Xva = pd.concat([Xva.reset_index(drop=True), pd.DataFrame(t_va, columns=cols)], axis=1)
            Xte_k = pd.concat([Xte, pd.DataFrame(t_te, columns=cols)], axis=1)
        p_va, p_te, it = fit_predict(kind, params, seed, Xtr, y[tr], Xva, y[va], Xte_k)
        oof[va] = p_va
        pred += p_te / folds
        its.append(it)
        print(f"  fold {len(its) - 1}: AUC={roc_auc_score(y[va], p_va):.5f} iters={it} "
              f"({time.time() - t0:.0f}s)", flush=True)
    auc = roc_auc_score(y, oof)
    name = name or f"{kind}__{spec}__s{seed}"
    print(f"{name:40s} AUC={auc:.5f} nfeat={Xtr.shape[1]} iters={its} ({time.time() - t0:.0f}s)", flush=True)
    np.save(OUT / f"{name}_oof.npy", oof)
    np.save(OUT / f"{name}_test.npy", pred)
    np.save(OUT / "y.npy", y)
    return auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=list(PARAMS))
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tuned", action="store_true", help="use params/<kind>__<spec>.json from tune.py")
    ap.add_argument("--lr", type=float, default=None)
    args = ap.parse_args()
    for spec in args.specs:
        params, tag = {}, ""
        if args.tuned:
            params = json.loads((ROOT / "params" / f"{args.kind}__{spec}.json").read_text())
            tag = "tuned_"
        if args.lr:
            params["learning_rate"] = args.lr
            tag += f"lr{args.lr}_"
        run(args.kind, spec, seed=args.seed, params=params,
            name=f"{args.kind}__{tag}{spec}__s{args.seed}")


if __name__ == "__main__":
    main()
