"""Optuna search for LightGBM/XGBoost params on one CV fold of a feature spec.

Usage:
    python src/tune.py lgb cnt+orig2+route+te --trials 40
"""
import argparse
import json
import time
import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

from common import ROOT, SEED
from fe import build

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def fold0(spec):
    Xall, _, te_keys, y, _ = build(spec)
    tr, va = next(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xall, y))
    Xtr, Xva = Xall.iloc[tr].reset_index(drop=True), Xall.iloc[va].reset_index(drop=True)
    if te_keys is not None:
        ktr = te_keys[0]
        enc = TargetEncoder(target_type="binary", cv=5, shuffle=True, random_state=SEED)
        cols = [f"te_{c}" for c in ktr.columns]
        Xtr = pd.concat([Xtr, pd.DataFrame(enc.fit_transform(ktr.iloc[tr].astype(str), y[tr]), columns=cols)], axis=1)
        Xva = pd.concat([Xva, pd.DataFrame(enc.transform(ktr.iloc[va].astype(str)), columns=cols)], axis=1)
    return Xtr, y[tr], Xva, y[va]


def objective_lgb(trial, Xtr, ytr, Xva, yva):
    import lightgbm as lgb
    p = dict(n_estimators=5000, learning_rate=0.08, verbose=-1, random_state=SEED,
             num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
             max_depth=trial.suggest_int("max_depth", 4, 12),
             min_child_samples=trial.suggest_int("min_child_samples", 10, 400, log=True),
             subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
             colsample_bytree=trial.suggest_float("colsample_bytree", 0.1, 0.9),
             reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 50, log=True),
             reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
             min_split_gain=trial.suggest_float("min_split_gain", 0, 1),
             max_bin=trial.suggest_categorical("max_bin", [63, 127, 255, 511]))
    m = lgb.LGBMClassifier(**p).fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
                                    callbacks=[lgb.early_stopping(150, verbose=False)])
    return roc_auc_score(yva, m.predict_proba(Xva)[:, 1])


def objective_xgb(trial, Xtr, ytr, Xva, yva):
    import xgboost as xgb
    p = dict(n_estimators=5000, learning_rate=0.08, tree_method="hist", eval_metric="auc",
             early_stopping_rounds=150, random_state=SEED,
             max_depth=trial.suggest_int("max_depth", 3, 12),
             min_child_weight=trial.suggest_float("min_child_weight", 0.5, 100, log=True),
             subsample=trial.suggest_float("subsample", 0.5, 1.0),
             colsample_bytree=trial.suggest_float("colsample_bytree", 0.1, 0.9),
             reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 50, log=True),
             reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
             gamma=trial.suggest_float("gamma", 0, 2),
             max_bin=trial.suggest_categorical("max_bin", [64, 128, 256, 512]))
    m = xgb.XGBClassifier(**p).fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return roc_auc_score(yva, m.predict_proba(Xva)[:, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["lgb", "xgb"])
    ap.add_argument("spec")
    ap.add_argument("--trials", type=int, default=40)
    args = ap.parse_args()

    data = fold0(args.spec)
    obj = objective_lgb if args.kind == "lgb" else objective_xgb
    t0 = time.time()

    def cb(study, trial):
        print(f"trial {trial.number} auc={trial.value:.5f} best={study.best_value:.5f} ({time.time() - t0:.0f}s)",
              flush=True)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(lambda t: obj(t, *data), n_trials=args.trials, callbacks=[cb])
    print("best params:", study.best_params)
    out = ROOT / "params"
    out.mkdir(exist_ok=True)
    (out / f"{args.kind}__{args.spec}.json").write_text(json.dumps(study.best_params, indent=1))


if __name__ == "__main__":
    main()
