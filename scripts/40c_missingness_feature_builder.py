#!/usr/bin/env python3
"""
40c_missingness_feature_builder — Generate missingness-informative features.

Purpose
-------
For each state variable in the 2h grid, generate:
  1. _mask:              1 if value observed in this bin, 0 otherwise
  2. _time_since_last_h: hours since last observation (0 if observed)
  3. _freq_24h:          count of observations in trailing 24h (12 bins)
  4. _never_measured:     1 if never measured for this stay (static per stay)

These features are critical for sequential models — missingness in ICU
data is informative (e.g., labs ordered more frequently in sicker patients).

Protocol reference: Step 40c (C3).

Inputs
------
- data/interim/40b_grid2h_states_mimic.parquet

Outputs
-------
- data/interim/40c_missingness_features_mimic.parquet
- reports/40c_missingness_summary_mimic.txt
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path

BIN_HOURS = 2
N_BINS = 12

# State variables to generate missingness features for.
# These should match the state variables from common_variable_map_v1.yaml
# that appear as columns in the 40b grid.
STATE_VARS = [
    "map", "sbp", "dbp", "heart_rate", "spo2", "cvp", "resp_rate",
    "temperature", "fio2",
    "lactate", "creatinine", "bun", "potassium", "sodium",
    "bicarbonate", "glucose", "ph",
    "hemoglobin", "wbc", "platelet", "albumin", "bilirubin",
    "troponin", "bnp",
]


def main() -> None:
    cfg = load_config()
    interim_dir = resolve_path(cfg, "interim")
    reports_dir = resolve_path(cfg, "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("40c_missingness_feature_builder — Informative missingness features")
    print("=" * 70)

    # ── Load grid ──────────────────────────────────────────────────────
    print("\n[1/3] Loading 40b grid ...")
    grid = pd.read_parquet(interim_dir / "40b_grid2h_states_mimic.parquet")
    n_stays = grid["stay_id"].nunique()
    n_rows = len(grid)
    print(f"  Grid: {n_rows:,} rows, {n_stays:,} stays")

    # Filter to variables actually present
    available_vars = [v for v in STATE_VARS if v in grid.columns]
    missing_vars = [v for v in STATE_VARS if v not in grid.columns]
    print(f"  State vars present: {len(available_vars)}")
    if missing_vars:
        print(f"  State vars missing from grid: {missing_vars}")

    # ── Generate features ──────────────────────────────────────────────
    print("\n[2/3] Generating missingness features ...")
    grid = grid.sort_values(["stay_id", "bin"]).reset_index(drop=True)

    for var in available_vars:
        observed = grid[var].notna().astype(int)

        # 1. Mask
        mask_col = f"{var}_mask"
        grid[mask_col] = observed

        # 2. Time since last observation (hours) — vectorized
        tsl_col = f"{var}_tsl_h"
        # last_obs_bin: forward-fill the bin index where var was observed
        last_obs_bin = grid["bin"].where(grid[var].notna())
        last_obs_bin = last_obs_bin.groupby(grid["stay_id"]).ffill()
        # tsl = (current_bin - last_observed_bin) * BIN_HOURS; 0 when observed
        tsl = (grid["bin"] - last_obs_bin) * BIN_HOURS
        # Rows before any observation in the stay remain NaN (ffill produces NaN)
        grid[tsl_col] = tsl

        # 3. Frequency in trailing 24h (12 bins)
        freq_col = f"{var}_freq_24h"
        # Rolling sum of mask over 12 bins
        freq = (
            grid.groupby("stay_id")[mask_col]
            .transform(lambda x: x.rolling(N_BINS, min_periods=1).sum())
        )
        grid[freq_col] = freq.astype(int)

        # 4. Never measured (static per stay)
        nm_col = f"{var}_never_measured"
        ever_measured = grid.groupby("stay_id")[mask_col].transform("max")
        grid[nm_col] = (ever_measured == 0).astype(int)

        # Guardrail: never_measured must be constant within each stay
        nuniq = grid.groupby("stay_id")[nm_col].transform("nunique")
        if (nuniq > 1).any():
            bad_stays = grid.loc[nuniq > 1, "stay_id"].unique()
            raise ValueError(
                f"{nm_col} is not static within {len(bad_stays)} stay(s): "
                f"{bad_stays[:5].tolist()}"
            )

    # ── Summary statistics ─────────────────────────────────────────────
    print("\n[3/3] Generating summary ...")
    report_lines = [
        "=" * 72,
        "40c — Missingness Feature Summary (MIMIC-IV)",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Stays: {n_stays:,}, Rows: {n_rows:,}",
        f"State variables processed: {len(available_vars)}",
        "",
        "PER-VARIABLE MISSINGNESS RATES",
        "───────────────────────────────",
        f"  {'Variable':<20} {'%miss(bins)':<14} {'%never_meas(stays)':<20} {'mean_freq_24h':<15}",
        "  " + "─" * 67,
    ]

    for var in available_vars:
        mask_col = f"{var}_mask"
        nm_col = f"{var}_never_measured"
        freq_col = f"{var}_freq_24h"

        pct_miss = (1 - grid[mask_col].mean()) * 100
        # Never-measured rate at stay level
        stay_nm = grid.groupby("stay_id")[nm_col].first()
        pct_never = stay_nm.mean() * 100
        mean_freq = grid[freq_col].mean()

        report_lines.append(
            f"  {var:<20} {pct_miss:>10.1f}%   {pct_never:>14.1f}%      {mean_freq:>10.2f}"
        )

    report_lines.append("")

    # High-missingness flags
    report_lines.append("HIGH-MISSINGNESS VARIABLES (>50% bin-level missing)")
    report_lines.append("─────────────────────────────────────────────────────")
    for var in available_vars:
        pct = (1 - grid[f"{var}_mask"].mean()) * 100
        if pct > 50:
            report_lines.append(f"  {var}: {pct:.1f}% missing")

    report_lines.append("")
    report_lines.append("FEATURES GENERATED")
    report_lines.append("──────────────────")
    new_cols = [c for c in grid.columns if any(
        c.endswith(s) for s in ("_mask", "_tsl_h", "_freq_24h", "_never_measured")
    )]
    report_lines.append(f"  Total new columns: {len(new_cols)}")
    report_lines.append(f"  Per variable: 4 features × {len(available_vars)} vars = {4 * len(available_vars)}")

    # ── Save ───────────────────────────────────────────────────────────
    out_path = interim_dir / "40c_missingness_features_mimic.parquet"
    grid.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"  → {out_path}  ({len(grid):,} rows, {grid.shape[1]} cols)")

    report_path = reports_dir / "40c_missingness_summary_mimic.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Report: {report_path}")

    print("\n[40c] Done.")


if __name__ == "__main__":
    main()
