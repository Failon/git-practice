"""Shared data loading and feature engineering for Kaggle PS S6E10 (Airline Satisfaction)."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TARGET = "satisfaction"
SEED = 42

CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]
RATING_COLS = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Inflight service", "Cleanliness",
]
NUM_COLS = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]


def _encode_target(y: pd.Series) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(y):
        return (y.str.strip().str.lower() == "satisfied").astype(int)
    return y.astype(int)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    return df


def load_original() -> pd.DataFrame:
    """Original Airline Passenger Satisfaction dataset (train+test merged, ~130k rows)."""
    d = DATA / "original"
    df = pd.concat([pd.read_csv(d / "train.csv"), pd.read_csv(d / "test.csv")], ignore_index=True)
    df = _clean(df).drop(columns=["id"])
    df[TARGET] = _encode_target(df[TARGET])
    return df


def load_comp():
    """Competition files from data/comp: returns (train, test, sample_submission)."""
    d = DATA / "comp"
    train = _clean(pd.read_csv(d / "train.csv"))
    test = _clean(pd.read_csv(d / "test.csv"))
    sub = pd.read_csv(d / "sample_submission.csv")
    train[TARGET] = _encode_target(train[TARGET])
    return train, test, sub


def add_noise(df: pd.DataFrame, rate: float, seed: int = SEED) -> pd.DataFrame:
    """Mimic Playground synthetic data: flip labels and jitter some ratings.

    The original data is almost perfectly separable (AUC ~0.995) while the synthetic
    competition data tops out near 0.96, so model rankings on clean data can mislead.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    flip = rng.random(len(df)) < rate
    df.loc[flip, TARGET] = 1 - df.loc[flip, TARGET]
    for c in RATING_COLS:
        m = rng.random(len(df)) < rate
        df.loc[m, c] = np.clip(df.loc[m, c] + rng.choice([-1, 1], m.sum()), 0, 5)
    return df


def base_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Categoricals -> small ints with fixed mappings; fill missing arrival delay."""
    df = df.copy()
    df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(df["Departure Delay in Minutes"])
    df["Gender"] = (df["Gender"] == "Male").astype(int)
    df["Customer Type"] = (df["Customer Type"] == "Loyal Customer").astype(int)
    df["Type of Travel"] = (df["Type of Travel"] == "Business travel").astype(int)
    df["Class"] = df["Class"].map({"Eco": 0, "Eco Plus": 1, "Business": 2}).fillna(0).astype(int)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hand-crafted aggregates over the service ratings and delays."""
    df = df.copy()
    r = df[RATING_COLS]
    rn = r.replace(0, np.nan)
    df["rating_mean"] = rn.mean(axis=1)
    df["rating_std"] = rn.std(axis=1)
    df["rating_min"] = rn.min(axis=1)
    df["n_zero"] = (r == 0).sum(axis=1)
    df["n_low"] = ((r > 0) & (r <= 2)).sum(axis=1)
    df["n_high"] = (r >= 4).sum(axis=1)
    df["digital"] = df[["Inflight wifi service", "Ease of Online booking", "Online boarding"]].mean(axis=1)
    df["comfort"] = df[["Seat comfort", "Leg room service", "Cleanliness", "Food and drink"]].mean(axis=1)
    df["delay_max"] = df[["Departure Delay in Minutes", "Arrival Delay in Minutes"]].max(axis=1)
    df["delay_diff"] = df["Arrival Delay in Minutes"] - df["Departure Delay in Minutes"]
    df["travel_class"] = df["Type of Travel"] * 3 + df["Class"]
    df["wifi_boarding"] = df["Inflight wifi service"] * 6 + df["Online boarding"]
    return df
