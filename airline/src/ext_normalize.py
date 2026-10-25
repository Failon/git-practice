"""Convert downloaded public OOF libraries (data/ext_raw/<lib>/...) into
data/ext/<lib>/oof_<name>.npy + test_<name>.npy (train/test in competition row order).

Only members with BOTH out-of-fold and test predictions are kept; test-only blends
(often tuned on the public leaderboard) are skipped.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATA

RAW, OUT = DATA / "ext_raw", DATA / "ext"
NTR, NTE = 699_635, 299_844


def save(lib, name, oof, test):
    oof, test = np.asarray(oof, dtype=np.float32).ravel(), np.asarray(test, dtype=np.float32).ravel()
    if oof.shape != (NTR,) or test.shape != (NTE,) or not (np.isfinite(oof).all() and np.isfinite(test).all()):
        print(f"  skip {lib}:{name} shapes {oof.shape} {test.shape}")
        return
    d = OUT / lib
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"oof_{name}.npy", oof)
    np.save(d / f"test_{name}.npy", test)


def by_id(df, col, n0):
    """Reorder an id/pred frame to competition order (train ids 0.., test ids NTR..)."""
    s = df.set_index("id")[col]
    return s.reindex(np.arange(n0, n0 + (NTR if n0 == 0 else NTE))).values


def main():
    for lib_dir in sorted(p for p in RAW.iterdir() if p.is_dir()):
        lib = re.sub(r"[^a-z0-9]+", "_", lib_dir.name.split("_", 1)[-1].lower())[:24]
        n_before = len(list((OUT / lib).glob("oof_*.npy"))) if (OUT / lib).exists() else 0
        files = {f.name: f for f in lib_dir.rglob("*") if f.is_file()}
        # 1) oof_X.npy / test_X.npy
        for fn, f in files.items():
            m = re.fullmatch(r"oof_(.+)\.npy", fn)
            if m and f"test_{m[1]}.npy" in files:
                save(lib, m[1], np.load(f, allow_pickle=False), np.load(files[f"test_{m[1]}.npy"], allow_pickle=False))
        # 2) oof_X.csv / test_X.csv with id,pred
        for fn, f in files.items():
            m = re.fullmatch(r"oof_(.+)\.csv", fn)
            if m and f"test_{m[1]}.csv" in files:
                a, b = pd.read_csv(f), pd.read_csv(files[f"test_{m[1]}.csv"])
                col = [c for c in a.columns if c != "id"][0]
                save(lib, m[1], by_id(a, col, 0), by_id(b, [c for c in b.columns if c != "id"][0], NTR))
        # 3) NN_blend_oof.csv / NN_submission.csv (najiama)
        for fn, f in files.items():
            m = re.fullmatch(r"(\d+)_blend_oof\.csv", fn)
            if m and f"{m[1]}_submission.csv" in files:
                save(lib, f"blend{m[1]}", by_id(pd.read_csv(f), "satisfaction", 0),
                     by_id(pd.read_csv(files[f"{m[1]}_submission.csv"]), "satisfaction", NTR))
        # 4) npz: either {oof,test} or {oof_X, test_X, ...}
        for fn, f in files.items():
            if not fn.endswith(".npz") or "stack" in fn:
                continue
            z = np.load(f, allow_pickle=False)
            if "oof" in z.files and "test" in z.files:
                save(lib, fn[:-4], z["oof"], z["test"])
            for k in z.files:
                if k.startswith("oof_") and f"test_{k[4:]}" in z.files:
                    save(lib, k[4:], z[k], z[f"test_{k[4:]}"])
        # 5) parquet pair with one column per model
        if "oof_predictions.parquet" in files and "test_predictions.parquet" in files:
            a = pd.read_parquet(files["oof_predictions.parquet"])
            b = pd.read_parquet(files["test_predictions.parquet"])
            for c in b.columns:
                if c != "id" and c in a.columns:
                    save(lib, c, by_id(a, c, 0), by_id(b, c, NTR))
        n = len(list((OUT / lib).glob("oof_*.npy"))) if (OUT / lib).exists() else 0
        print(f"{lib:26s} +{n - n_before} members")


if __name__ == "__main__":
    main()
