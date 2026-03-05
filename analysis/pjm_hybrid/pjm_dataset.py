"""
PJM Wind Forecast — Sliding-Window Time-Series Dataset
=======================================================
Converts a CSV of hourly wind-forecast values into PyTorch-compatible
(input_window, target_horizon) pairs with engineered temporal features.

Design decisions
----------------
* **Chronological split** — no random shuffling; avoids data leakage.
* **Z-score normalisation** — fitted on train split only.
* **Cyclical encoding** — hour-of-day and day-of-week encoded as sin/cos
  so that 23:00 is close to 00:00.
* **Lag features** — past 1 h, 3 h, 6 h, 12 h, 24 h values.
* **Rolling stats** — 6 h and 24 h rolling mean & std of wind_forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional


# ── Feature engineering ────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, target_col: str = "wind_forecast") -> pd.DataFrame:
    """
    Add temporal + lag + rolling features to an hourly wind-forecast DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``interval_start_local`` (datetime-like)
        and ``target_col`` (float).
    target_col : str
        Column to forecast.

    Returns
    -------
    pd.DataFrame
        Original frame augmented with engineered features, NaN rows dropped.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["interval_start_local"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── Cyclical time encodings ──
    hour = df["timestamp"].dt.hour
    dow  = df["timestamp"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * dow / 7)

    # ── Lag features ──
    for lag in [1, 3, 6, 12, 24]:
        df[f"lag_{lag}h"] = df[target_col].shift(lag)

    # ── Rolling statistics ──
    for win in [6, 24]:
        df[f"roll_mean_{win}h"] = df[target_col].rolling(win).mean()
        df[f"roll_std_{win}h"]  = df[target_col].rolling(win).std()

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ── Column lists ────────────────────────────────────────────────────

FEATURE_COLS = [
    "wind_forecast",
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "lag_1h",   "lag_3h",  "lag_6h", "lag_12h", "lag_24h",
    "roll_mean_6h",  "roll_std_6h",
    "roll_mean_24h", "roll_std_24h",
]

TARGET_COL = "wind_forecast"


# ── Normaliser (z-score, fitted on train) ──────────────────────────

class ZScoreNormaliser:
    """Fit mean/std on train data; transform any split."""

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std:  Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "ZScoreNormaliser":
        self.mean = X.mean(axis=0)
        self.std  = X.std(axis=0) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.std + self.mean

    def inverse_transform_target(self, y: np.ndarray, target_idx: int = 0) -> np.ndarray:
        """Inverse-transform only the target column dimension."""
        return y * self.std[target_idx] + self.mean[target_idx]


# ── Sliding-window dataset ─────────────────────────────────────────

class PJMTimeSeriesDataset(Dataset):
    """
    Sliding-window dataset for PJM wind forecast.

    Each sample is a tuple ``(x, y)`` where:
    * ``x`` has shape ``(window_size, n_features)``
    * ``y`` has shape ``(horizon,)`` — the next ``horizon`` values of the target.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        window_size: int = 24,
        horizon: int = 1,
    ):
        self.features = features.astype(np.float32)
        self.targets  = targets.astype(np.float32)
        self.window_size = window_size
        self.horizon = horizon
        self.n_samples = len(features) - window_size - horizon + 1

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx : idx + self.window_size]
        y = self.targets[idx + self.window_size : idx + self.window_size + self.horizon]
        return torch.from_numpy(x), torch.from_numpy(y)


# ── Data-loading helper ────────────────────────────────────────────

def load_pjm_splits(
    csv_path: str,
    window_size: int = 24,
    horizon: int = 1,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Tuple[PJMTimeSeriesDataset, PJMTimeSeriesDataset, PJMTimeSeriesDataset, ZScoreNormaliser]:
    """
    Load a PJM wind-forecast CSV and return chronological train/val/test datasets.

    Parameters
    ----------
    csv_path : str
        Path to the merged CSV.
    window_size : int
        Number of look-back hours.
    horizon : int
        Number of hours to predict.
    train_frac, val_frac : float
        Proportions (test = 1 - train - val).

    Returns
    -------
    train_ds, val_ds, test_ds, normaliser
    """
    df = pd.read_csv(csv_path)
    df = engineer_features(df, target_col=TARGET_COL)

    # Extract numpy arrays
    features = df[FEATURE_COLS].values
    targets  = df[TARGET_COL].values

    # Chronological split boundaries
    n = len(features)
    i_train = int(n * train_frac)
    i_val   = int(n * (train_frac + val_frac))

    feat_train, feat_val, feat_test = features[:i_train], features[i_train:i_val], features[i_val:]
    tgt_train,  tgt_val,  tgt_test  = targets[:i_train],  targets[i_train:i_val],  targets[i_val:]

    # Normalise (fit on train only)
    norm = ZScoreNormaliser().fit(feat_train)
    feat_train = norm.transform(feat_train)
    feat_val   = norm.transform(feat_val)
    feat_test  = norm.transform(feat_test)

    # Also normalise targets using index 0 (wind_forecast is first feature)
    tgt_mean, tgt_std = norm.mean[0], norm.std[0]
    tgt_train = (tgt_train - tgt_mean) / tgt_std
    tgt_val   = (tgt_val   - tgt_mean) / tgt_std
    tgt_test  = (tgt_test  - tgt_mean) / tgt_std

    train_ds = PJMTimeSeriesDataset(feat_train, tgt_train, window_size, horizon)
    val_ds   = PJMTimeSeriesDataset(feat_val,   tgt_val,   window_size, horizon)
    test_ds  = PJMTimeSeriesDataset(feat_test,  tgt_test,  window_size, horizon)

    return train_ds, val_ds, test_ds, norm
