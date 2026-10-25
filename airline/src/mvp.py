"""MVP bake-off: several model families under the same 5-fold CV.

Usage:
    python src/mvp.py                      # original data, clean + noisy proxy
    python src/mvp.py --data comp          # real competition train.csv
    python src/mvp.py --models lgbm,cat    # subset
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from common import (CAT_COLS, NUM_COLS, RATING_COLS, ROOT, SEED, TARGET, add_features,
                    add_noise, base_encode, load_comp, load_original)

warnings.filterwarnings("ignore")


def linear_pre():
    return ColumnTransformer([
        ("ohe", OneHotEncoder(handle_unknown="ignore"), CAT_COLS + RATING_COLS),
        ("num", make_pipeline(FunctionTransformer(np.log1p), StandardScaler()), NUM_COLS),
    ])


def m_logreg(Xtr, ytr, Xva, yva):
    m = make_pipeline(linear_pre(), LogisticRegression(C=1.0, max_iter=2000))
    return m.fit(Xtr, ytr).predict_proba(Xva)[:, 1]


def m_mlp(Xtr, ytr, Xva, yva):
    m = make_pipeline(linear_pre(), MLPClassifier((128, 64), alpha=1e-3, batch_size=512,
                                                   early_stopping=True, max_iter=60, random_state=SEED))
    return m.fit(Xtr, ytr).predict_proba(Xva)[:, 1]


def _lgbm(Xtr, ytr, Xva, yva):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.05, num_leaves=63, min_child_samples=40,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0,
                           random_state=SEED, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
          callbacks=[lgb.early_stopping(200, verbose=False)])
    return m.predict_proba(Xva)[:, 1]


def m_lgbm(Xtr, ytr, Xva, yva):
    return _lgbm(Xtr, ytr, Xva, yva)


def m_lgbm_fe(Xtr, ytr, Xva, yva):
    return _lgbm(add_features(Xtr), ytr, add_features(Xva), yva)


def m_xgb(Xtr, ytr, Xva, yva):
    import xgboost as xgb
    m = xgb.XGBClassifier(n_estimators=5000, learning_rate=0.05, max_depth=7, min_child_weight=3,
                          subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0, tree_method="hist",
                          eval_metric="auc", early_stopping_rounds=200, random_state=SEED)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m.predict_proba(Xva)[:, 1]


def m_cat(Xtr, ytr, Xva, yva):
    """CatBoost treating categoricals AND ratings as categorical features."""
    from catboost import CatBoostClassifier
    cats = CAT_COLS + RATING_COLS
    Xtr, Xva = Xtr.copy(), Xva.copy()
    Xtr[cats] = Xtr[cats].astype(str)
    Xva[cats] = Xva[cats].astype(str)
    m = CatBoostClassifier(iterations=3000, learning_rate=0.08, depth=6, eval_metric="AUC",
                           random_seed=SEED, verbose=0, od_type="Iter", od_wait=200, thread_count=-1)
    m.fit(Xtr, ytr, cat_features=cats, eval_set=(Xva, yva), use_best_model=True)
    return m.predict_proba(Xva)[:, 1]


def m_hgb(Xtr, ytr, Xva, yva):
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(max_iter=1000, learning_rate=0.08, max_leaf_nodes=63,
                                       categorical_features=[Xtr.columns.get_loc(c) for c in CAT_COLS],
                                       early_stopping=True, n_iter_no_change=50, random_state=SEED)
    return m.fit(Xtr, ytr).predict_proba(Xva)[:, 1]


MODELS = {"logreg": m_logreg, "mlp": m_mlp, "hgb": m_hgb, "lgbm": m_lgbm,
          "lgbm_fe": m_lgbm_fe, "xgb": m_xgb, "cat": m_cat}


def run_cv(df, models, folds, tag):
    X = base_encode(df.drop(columns=[TARGET]))
    y = df[TARGET].values
    skf = StratifiedKFold(folds, shuffle=True, random_state=SEED)
    oof = {}
    rows = []
    for name in models:
        t0 = time.time()
        pred = np.zeros(len(df))
        for tr, va in skf.split(X, y):
            pred[va] = MODELS[name](X.iloc[tr], y[tr], X.iloc[va], y[va])
        oof[name] = pred
        auc = roc_auc_score(y, pred)
        rows.append((tag, name, auc, time.time() - t0))
        print(f"[{tag}] {name:8s} AUC={auc:.5f}  ({time.time() - t0:.0f}s)", flush=True)
    # simple equal-weight rank blend of everything
    from scipy.stats import rankdata
    blend = np.mean([rankdata(p) for p in oof.values()], axis=0)
    print(f"[{tag}] rank-blend(all) AUC={roc_auc_score(y, blend):.5f}")
    out = ROOT / "oof"
    out.mkdir(exist_ok=True)
    pd.DataFrame(oof).assign(y=y).to_parquet(out / f"mvp_{tag}.parquet")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["original", "comp"], default="original")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--noise", type=float, default=0.08)
    ap.add_argument("--sample", type=int, default=0, help="subsample rows for speed")
    args = ap.parse_args()
    models = args.models.split(",")

    if args.data == "comp":
        df = load_comp()[0].drop(columns=["id"])
        if args.sample:
            df = df.sample(args.sample, random_state=SEED)
        rows = run_cv(df, models, args.folds, "comp")
    else:
        df = load_original()
        rows = run_cv(df, models, args.folds, "clean")
        rows += run_cv(add_noise(df, args.noise), models, args.folds, "noisy")

    res = pd.DataFrame(rows, columns=["data", "model", "auc", "sec"])
    print("\n" + res.pivot(index="model", columns="data", values="auc").round(5).to_string())


if __name__ == "__main__":
    main()
