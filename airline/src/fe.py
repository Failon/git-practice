"""Composable feature blocks for S6E10.

A feature spec is a '+'-joined list of block names, e.g. "cnt+orig+route".
Blocks that need the target from the training fold (in-fold target encoding)
return key columns; train.py encodes them inside each CV fold.
"""
import itertools

import numpy as np
import pandas as pd

from common import CAT_COLS, NUM_COLS, RATING_COLS, TARGET, base_encode, load_comp, load_original

ALL_COLS = RATING_COLS + CAT_COLS + NUM_COLS
RAW = CAT_COLS + NUM_COLS + RATING_COLS

# key groups (tuples of columns) used for original-data stats and in-fold target encoding
KEYS_SINGLE = [(c,) for c in ALL_COLS]
KEYS_CORE = KEYS_SINGLE + [
    tuple(RATING_COLS), tuple(CAT_COLS), tuple(RATING_COLS + CAT_COLS),
    ("Flight Distance", "Class"), ("Flight Distance", "Type of Travel"),
    ("Age", "Class"), ("Age", "Type of Travel"), ("Age", "Customer Type"),
    ("Flight Distance", "Age"),
]
KEYS_CAT_RATING = [(c, r) for c in ["Class", "Type of Travel", "Customer Type"] for r in RATING_COLS]
KEYS_RATING_PAIRS = list(itertools.combinations(RATING_COLS, 2))
KEY_SETS = {
    "single": KEYS_SINGLE,
    "core": KEYS_CORE,
    "core2": KEYS_CORE + KEYS_CAT_RATING,
    "core3": KEYS_CORE + KEYS_CAT_RATING + KEYS_RATING_PAIRS,
}


def kname(g):
    return "|".join(g) if len(g) <= 3 else f"k{len(g)}_" + "".join(c[0] for c in g)


def make_keys(full: pd.DataFrame, groups) -> pd.DataFrame:
    """Integer id for every tuple of column values (consistent across the rows of `full`)."""
    codes = {c: pd.factorize(full[c].fillna(-1))[0].astype(np.int64) for c in ALL_COLS}
    out = {}
    for g in groups:
        key = np.zeros(len(full), dtype=np.int64)
        for c in g:
            key = key * (codes[c].max() + 1) + codes[c]
        out[kname(g)] = pd.factorize(key)[0]
    return pd.DataFrame(out, index=full.index)


class Data:
    """train/test/original, encoded, stacked in one frame so keys and counts agree."""

    def __init__(self):
        train, test, self.sub = load_comp()
        orig = load_original()
        self.y = train[TARGET].values
        self.y_orig = orig[TARGET].values
        self.ntr, self.nte = len(train), len(test)
        full = pd.concat([train.drop(columns=["id", TARGET]), test.drop(columns=["id"]),
                          orig.drop(columns=[TARGET])], ignore_index=True)
        self.full = base_encode(full)[RAW]
        self.tr = slice(0, self.ntr)
        self.te = slice(self.ntr, self.ntr + self.nte)
        self.orig = slice(self.ntr + self.nte, len(self.full))

    def comp(self):
        return self.full.iloc[: self.ntr + self.nte]


def block_cnt(d: Data):
    comp = d.comp()
    return pd.DataFrame({f"cnt_{c}": comp[c].map(comp[c].value_counts()) for c in NUM_COLS})


def block_orig(d: Data, keyset="core", smooth=10):
    """Per-key count and smoothed satisfaction rate from the original dataset."""
    keys = make_keys(d.full, KEY_SETS[keyset])
    ko, kc = keys.iloc[d.orig], keys.iloc[: d.ntr + d.nte]
    prior = d.y_orig.mean()
    out = {}
    for c in keys.columns:
        st = pd.Series(d.y_orig, index=ko.index).groupby(ko[c].values).agg(["sum", "count"])
        out[f"ocnt_{c}"] = kc[c].map(st["count"]).fillna(0).values
        out[f"ote_{c}"] = kc[c].map((st["sum"] + prior * smooth) / (st["count"] + smooth)).fillna(prior).values
    return pd.DataFrame(out)


def block_route(d: Data):
    """Treat Flight Distance as a route id: profile of passengers on the same route (train+test)."""
    comp = d.comp()
    g = comp.groupby("Flight Distance")
    out = {}
    for c in ["Age", "Class", "Type of Travel", "Customer Type"] + RATING_COLS:
        out[f"route_mean_{c}"] = g[c].transform("mean").values
    out["route_std_Age"] = g["Age"].transform("std").fillna(0).values
    return pd.DataFrame(out)


def digits(comp: pd.DataFrame) -> pd.DataFrame:
    """Decimal digits of numeric columns: synthetic generators leave artefacts there."""
    out = {}
    for c, n in [("Flight Distance", 4), ("Age", 2), ("Departure Delay in Minutes", 3),
                 ("Arrival Delay in Minutes", 3)]:
        x = comp[c].fillna(-1).astype(int)
        for k in range(n):
            out[f"dig{k}_{c}"] = (x // 10 ** k) % 10
    return pd.DataFrame(out)


def block_dig(d: Data):
    return digits(d.comp())


def te_dig(d: Data):
    """Keys: numeric values, their digits, and digits crossed with main categoricals."""
    comp = d.comp().reset_index(drop=True)
    dg = digits(comp)
    keys = make_keys(comp, [(c,) for c in NUM_COLS])
    for c in dg.columns:
        keys[f"k_{c}"] = dg[c]
    for c in ["dig0_Flight Distance", "dig1_Flight Distance", "dig0_Age"]:
        for cat in ["Class", "Type of Travel", "Customer Type"]:
            keys[f"k_{c}|{cat}"] = dg[c] * 10 + comp[cat]
    return keys


def block_ncat(d: Data):
    """Copies of numeric columns and ratings marked 'c_' -> treated as native categoricals by train.py."""
    comp = d.comp()
    return pd.DataFrame({f"c_{c}": comp[c].fillna(-1).astype(int) for c in NUM_COLS + RATING_COLS})


BLOCKS = {
    "ncat": block_ncat,
    "dig": block_dig,
    "cnt": block_cnt,
    "orig": block_orig,
    "orig2": lambda d: block_orig(d, "core2"),
    "orig3": lambda d: block_orig(d, "core3"),
    "route": block_route,
}
# in-fold target encoding key sets
TE_BLOCKS = {
    "tedig": te_dig,
    "te": lambda d: make_keys(d.comp(), [(c,) for c in NUM_COLS]),
    "tecore": lambda d: make_keys(d.comp(), KEYS_CORE),
    "tecore2": lambda d: make_keys(d.comp(), KEYS_CORE + KEYS_CAT_RATING),
}


def build(spec: str):
    """Return (X_train, X_test, te_keys_train, te_keys_test, y, data)."""
    d = Data()
    parts, te_keys = [d.comp().reset_index(drop=True)], None
    for b in filter(None, spec.split("+")):
        if b in TE_BLOCKS:
            te_keys = TE_BLOCKS[b](d).reset_index(drop=True)
        else:
            parts.append(BLOCKS[b](d).reset_index(drop=True))
    X = pd.concat(parts, axis=1)
    Xtr, Xte = X.iloc[: d.ntr].reset_index(drop=True), X.iloc[d.ntr:].reset_index(drop=True)
    if te_keys is not None:
        te_keys = (te_keys.iloc[: d.ntr].reset_index(drop=True), te_keys.iloc[d.ntr:].reset_index(drop=True))
    return Xtr, Xte, te_keys, d.y, d
