"""
PyTorch Dataset for Step 50 encoder training.

Loads 45c state tensors (stay_id × bin × features) into a format suitable
for sequential encoder training with masked reconstruction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ICUStayDataset(Dataset):
    """
    Per-stay dataset: each item is a (T, D) tensor of state features
    plus metadata (stay_id, static covariates, outcome labels).

    Parameters
    ----------
    state_df : pd.DataFrame
        Wide-format 45c state tensor with columns: stay_id, bin, <features>.
    outcomes_df : pd.DataFrame
        45d outcomes keyed by stay_id.
    stay_ids : array-like
        Subset of stay_ids to include (for train/val/test split).
    temporal_cols : list[str]
        Columns to use as temporal features (z-scored values + missingness).
    static_cols : list[str]
        Columns to use as static covariates (constant across bins).
    n_bins : int
        Expected number of bins per stay.
    absorbing_col : str
        Column in outcomes_df for absorbing_state_bin (NaN = no absorbing).
    """

    def __init__(
        self,
        state_df: pd.DataFrame,
        outcomes_df: pd.DataFrame,
        stay_ids: np.ndarray,
        temporal_cols: list[str],
        static_cols: list[str],
        n_bins: int = 12,
        absorbing_col: str = "absorbing_state_bin",
    ):
        self.n_bins = n_bins
        self.temporal_cols = temporal_cols
        self.static_cols = static_cols
        self.d_temporal = len(temporal_cols)
        self.d_static = len(static_cols)

        # Filter to requested stay_ids
        stay_set = set(stay_ids)
        mask = state_df["stay_id"].isin(stay_set)
        df = state_df.loc[mask].sort_values(["stay_id", "bin"])

        # Build numpy arrays indexed by stay
        self.stay_ids = np.array(sorted(stay_set))
        self.n_stays = len(self.stay_ids)
        sid_to_idx = {sid: i for i, sid in enumerate(self.stay_ids)}

        # Pre-allocate tensors
        self.temporal = np.zeros((self.n_stays, n_bins, self.d_temporal), dtype=np.float32)
        self.static = np.zeros((self.n_stays, self.d_static), dtype=np.float32)
        self.valid_mask = np.zeros((self.n_stays, n_bins), dtype=np.float32)

        # Fill temporal features
        temporal_vals = df[temporal_cols].values.astype(np.float32)
        bin_vals = df["bin"].values
        sid_vals = df["stay_id"].values

        for row_i in range(len(df)):
            sid = sid_vals[row_i]
            if sid not in sid_to_idx:
                continue
            stay_i = sid_to_idx[sid]
            b = int(bin_vals[row_i])
            if 0 <= b < n_bins:
                self.temporal[stay_i, b, :] = temporal_vals[row_i]
                self.valid_mask[stay_i, b] = 1.0

        # Fill static features (take first bin per stay)
        if static_cols:
            first_rows = df.drop_duplicates(subset="stay_id", keep="first")
            static_vals = first_rows.set_index("stay_id")[static_cols]
            for sid in self.stay_ids:
                if sid in static_vals.index:
                    self.static[sid_to_idx[sid]] = (
                        static_vals.loc[sid].values.astype(np.float32)
                    )

        # Absorbing state mask: bins after absorbing event should not contribute to loss
        self.absorbing_mask = np.ones((self.n_stays, n_bins), dtype=np.float32)
        if outcomes_df is not None and absorbing_col in outcomes_df.columns:
            out_indexed = outcomes_df.set_index("stay_id")
            for sid in self.stay_ids:
                if sid in out_indexed.index:
                    ab_bin = out_indexed.loc[sid, absorbing_col]
                    if pd.notna(ab_bin):
                        ab_int = int(ab_bin)
                        # Mask bins AFTER the absorbing event
                        if ab_int + 1 < n_bins:
                            self.absorbing_mask[sid_to_idx[sid], ab_int + 1:] = 0.0

        # Outcome labels
        self.mortality_28d = np.zeros(self.n_stays, dtype=np.float32)
        self.mortality_7d = np.zeros(self.n_stays, dtype=np.float32)
        if outcomes_df is not None:
            out_indexed = outcomes_df.set_index("stay_id")
            for sid in self.stay_ids:
                if sid in out_indexed.index:
                    row = out_indexed.loc[sid]
                    self.mortality_28d[sid_to_idx[sid]] = float(
                        row.get("mortality_28d", 0)
                    )
                    self.mortality_7d[sid_to_idx[sid]] = float(
                        row.get("mortality_7d", 0)
                    )

        # Replace NaN with 0 in all arrays
        np.nan_to_num(self.temporal, copy=False, nan=0.0)
        np.nan_to_num(self.static, copy=False, nan=0.0)

    def __len__(self) -> int:
        return self.n_stays

    def __getitem__(self, idx: int) -> dict:
        return {
            "stay_id": self.stay_ids[idx],
            "temporal": torch.from_numpy(self.temporal[idx]),      # (T, D_t)
            "static": torch.from_numpy(self.static[idx]),          # (D_s,)
            "valid_mask": torch.from_numpy(self.valid_mask[idx]),   # (T,)
            "absorbing_mask": torch.from_numpy(self.absorbing_mask[idx]),  # (T,)
            "mortality_28d": torch.tensor(self.mortality_28d[idx]),
            "mortality_7d": torch.tensor(self.mortality_7d[idx]),
        }


def discover_feature_columns(
    df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Auto-discover temporal and static columns from 45c state tensor schema.

    Temporal: *_z / *_zscore, *_mask, *_tsl_h, *_freq_24h, *_never_measured
    Static: 45c passthrough static covariates from 25f (categorical/dummy/scalar).

    Returns (temporal_cols, static_cols).
    """
    skip = {"stay_id", "bin", "subject_id", "hadm_id"}
    temporal_suffixes = ("_z", "_zscore", "_mask", "_tsl_h", "_freq_24h", "_never_measured")
    static_prefixes = ("static_",)
    static_exact = {
        # 45c static covariates (pre-encoding)
        "age", "gender", "admission_type", "hf_subtype", "comparator_group",
        "dominant_syndrome", "reason_for_icu", "hemo_gate_passed", "include_in_phase05",
        # Optional already-encoded/static variants
        "age_zscore", "is_female", "is_emergency", "is_surgical",
    }

    temporal_cols = []
    static_cols = []

    for col in df.columns:
        if col in skip:
            continue
        if any(col.endswith(s) for s in temporal_suffixes):
            if pd.api.types.is_numeric_dtype(df[col]):
                temporal_cols.append(col)
        elif any(col.startswith(p) for p in static_prefixes) or col in static_exact:
            if pd.api.types.is_numeric_dtype(df[col]):
                static_cols.append(col)

    return sorted(temporal_cols), sorted(static_cols)


def make_train_val_test_split(
    stay_ids: np.ndarray,
    outcomes_df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    stratify_col: str = "mortality_28d",
    seed: int = 2024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/val/test split by stay_id."""
    from sklearn.model_selection import train_test_split

    # Stable ordering for deterministic split regardless upstream dataframe ordering
    stay_ids = np.sort(np.asarray(stay_ids))

    # Get stratification labels
    out_indexed = outcomes_df.set_index("stay_id")
    labels = np.array([
        int(out_indexed.loc[sid, stratify_col]) if sid in out_indexed.index else 0
        for sid in stay_ids
    ])

    # First split: train vs (val+test)
    holdout_frac = val_frac + test_frac
    train_ids, holdout_ids, train_labels, holdout_labels = train_test_split(
        stay_ids, labels, test_size=holdout_frac, stratify=labels, random_state=seed,
    )

    # Second split: val vs test
    val_relative = val_frac / holdout_frac
    val_ids, test_ids = train_test_split(
        holdout_ids, test_size=(1 - val_relative),
        stratify=holdout_labels, random_state=seed,
    )

    return train_ids, val_ids, test_ids
