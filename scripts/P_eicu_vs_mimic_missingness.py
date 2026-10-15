#!/usr/bin/env python3
"""P_eicu_vs_mimic_missingness — Per-variable observation rates: MIMIC vs eICU full vs eICU Tier A.

Protocol: supplementary analysis (eICU data granularity).
Compute tier: C1 (read + groupby + summarize; ~30 s).

Purpose
-------
Produce one supplementary table answering: how complete are the bin-level
state and treatment-documentation streams in MIMIC vs the eICU full cohort
vs the eICU Tier A high-documentation subset?

For each cohort and variable we report two observation metrics:

  - obs_pct:   % of stay-bins with a non-null *raw* observation
               (LOCF / imputation NOT applied). Computed for state
               variables that can be physiologically zero/positive
               (lactate, MAP, SpO2, urine_output_ml).
  - nz_pct:    % of stay-bins with a non-zero record. For action /
               documentation variables (iv_fluid_ml, vaso_ne_eq,
               urine_output_ml) this is the bedside-relevant
               documentation-quality proxy: in eICU "no fluid record
               in this bin" most often reflects documentation
               sparsity, whereas in MIMIC it more often reflects a
               true no-fluid bin.

Tier A definition for eICU (mirrors `scripts/90b_05_evaluate_eicu_frozen_models.py`
default): stays from hospitals with hospital-level
`fluid_nz_pct >= TIER_A_THRESHOLD` (default 80).

Inputs
------
- data/interim/40b_grid2h_states_mimic.parquet
- data/interim/45b_actions_2h_mimic.parquet                     (MIMIC actions)
- data/interim/90b_eicu_grid2h_states.parquet
- data/interim/90b_eicu_actions_2h.parquet                      (eICU actions)
- data/staging/90b_eicu_cohort_v1.parquet                       (stay_id → hospitalid)
- tables/90b_04_eicu_site_heterogeneity.csv                     (per-hospital fluid_nz_pct)

Output
------
- tables/eicu_full_vs_tierA_vs_mimic_missingness.csv

CLI
---
    env/run_py.sh scripts/P_eicu_vs_mimic_missingness.py
    env/run_py.sh scripts/P_eicu_vs_mimic_missingness.py --tier-a-threshold 80
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
INTERIM_DIR = ROOT / "data" / "interim"
STAGING_DIR = ROOT / "data" / "staging"
TABLES_DIR = ROOT / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Variables to report. Each entry: (variable name in grid, metric).
#   metric = "obs"  → % bins with non-null raw value (state variables)
#   metric = "nz"   → % bins with > 0 (treatment documentation proxy)
STATE_VARS = [
    ("lactate", "obs"),
    ("map", "obs"),
    ("spo2", "obs"),
    ("creatinine", "obs"),
]
ACTION_VARS = [
    ("iv_fluid_ml", "nz"),
    ("vasopressor_ne_eq", "nz"),
    ("diuretic_furo_eq_mg", "nz"),
    ("urine_output_ml", "nz"),
]


def _read_parquet_or_die(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"[{label}] missing: {path}")
    return pd.read_parquet(path)


def _resolve_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _summary(grid: pd.DataFrame, var: str, metric: str) -> dict:
    """Return n_stays, n_bins, and observation/non-zero rate for `var` in `grid`."""
    out = {
        "n_stays": int(grid["stay_id"].nunique()),
        "n_bins": int(len(grid)),
    }
    if var not in grid.columns:
        out["pct"] = np.nan
        out["metric"] = metric
        out["available"] = False
        return out
    series = grid[var]
    if metric == "obs":
        pct = series.notna().mean() * 100.0
    elif metric == "nz":
        # Non-zero rate. Treat NaN as zero for documentation interpretation.
        pct = (series.fillna(0) > 0).mean() * 100.0
    else:
        raise ValueError(f"Unknown metric: {metric}")
    out["pct"] = float(round(pct, 2))
    out["metric"] = metric
    out["available"] = True
    return out


def _merge_state_actions(state_df: pd.DataFrame, action_df: pd.DataFrame | None) -> pd.DataFrame:
    if action_df is None:
        return state_df
    keys = ["stay_id", "bin"]
    if not all(k in state_df.columns for k in keys):
        return state_df
    if not all(k in action_df.columns for k in keys):
        return state_df
    new_cols = [c for c in action_df.columns if c not in state_df.columns or c in keys]
    return state_df.merge(action_df[new_cols], on=keys, how="left")


def _load_mimic() -> pd.DataFrame:
    """Load MIMIC bin-level grid. The 40b output already contains
    vasopressor_ne_eq, diuretic_furo_eq_mg, iv_fluid_ml, urine_output_ml
    alongside state variables, so we do not need the 45b actions parquet."""
    print("[MIMIC] loading 40b state+action grid ...")
    grid = _read_parquet_or_die(INTERIM_DIR / "40b_grid2h_states_mimic.parquet", "MIMIC grid")
    print(f"  MIMIC grid: {grid.shape[0]:,} bins, {grid['stay_id'].nunique():,} stays")
    return grid


def _load_eicu() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load eICU bin-level grid. 90b_02 places treatment-volume columns into
    the same temporal grid file."""
    print("[eICU] loading 90b state+action grid + cohort ...")
    grid = _read_parquet_or_die(
        INTERIM_DIR / "90b_eicu_grid2h_states.parquet", "eICU grid"
    )
    print(f"  eICU full grid: {grid.shape[0]:,} bins, {grid['stay_id'].nunique():,} stays")

    cohort_path = STAGING_DIR / "90b_eicu_cohort_v1.parquet"
    cohort = _read_parquet_or_die(cohort_path, "eICU cohort")
    stay_col = _resolve_column(cohort, ["stay_id", "patientunitstayid"])
    hosp_col = _resolve_column(cohort, ["hospitalid"])
    if stay_col is None or hosp_col is None:
        raise KeyError(
            f"cohort {cohort_path} must contain stay_id (or patientunitstayid) and hospitalid; "
            f"found {list(cohort.columns)}"
        )
    cohort = cohort[[stay_col, hosp_col]].rename(columns={stay_col: "stay_id", hosp_col: "hospitalid"})
    cohort = cohort.drop_duplicates("stay_id")
    return grid, cohort


def _identify_tier_a(cohort: pd.DataFrame, threshold: float) -> set[int]:
    site_path = TABLES_DIR / "90b_04_eicu_site_heterogeneity.csv"
    if not site_path.exists():
        raise FileNotFoundError(f"site heterogeneity table missing: {site_path}")
    sites = pd.read_csv(site_path)
    if "fluid_nz_pct" not in sites.columns or "hospitalid" not in sites.columns:
        raise KeyError(f"{site_path} must contain hospitalid and fluid_nz_pct")
    tier_a_hospitals = set(sites.loc[sites["fluid_nz_pct"] >= threshold, "hospitalid"])
    tier_a_stays = set(cohort.loc[cohort["hospitalid"].isin(tier_a_hospitals), "stay_id"])
    print(
        f"  Tier A (fluid_nz_pct >= {threshold:.0f}%): "
        f"{len(tier_a_hospitals)} hospitals, {len(tier_a_stays):,} stays"
    )
    return tier_a_stays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier-a-threshold",
        type=float,
        default=80.0,
        help="Hospital-level fluid_nz_pct cutoff for Tier A (default: 80)",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("eICU full vs Tier A vs MIMIC — per-variable observation rates")
    print("=" * 72)

    mimic_grid = _load_mimic()
    eicu_grid, cohort = _load_eicu()
    tier_a_stays = _identify_tier_a(cohort, args.tier_a_threshold)
    eicu_tier_a_grid = eicu_grid[eicu_grid["stay_id"].isin(tier_a_stays)].copy()
    print(f"  Tier A grid: {eicu_tier_a_grid.shape[0]:,} bins")

    print("\n[summary] computing per-variable rates ...")
    rows: list[dict] = []
    cohorts = [
        ("mimic", mimic_grid),
        ("eicu_full", eicu_grid),
        ("eicu_tier_a", eicu_tier_a_grid),
    ]
    for var, metric in STATE_VARS + ACTION_VARS:
        row: dict = {
            "variable": var,
            "metric": metric,
            "metric_label": "non-null raw observation %" if metric == "obs" else "non-zero documentation %",
        }
        for cohort_name, cohort_grid in cohorts:
            s = _summary(cohort_grid, var, metric)
            row[f"{cohort_name}_n_stays"] = s["n_stays"]
            row[f"{cohort_name}_n_bins"] = s["n_bins"]
            row[f"{cohort_name}_pct"] = s["pct"]
            row[f"{cohort_name}_available"] = s["available"]
        rows.append(row)

    out = pd.DataFrame(rows)
    out_path = TABLES_DIR / "eicu_full_vs_tierA_vs_mimic_missingness.csv"
    out.to_csv(out_path, index=False)

    print("\n" + out.to_string(index=False))
    print(f"\nSaved: {out_path}")

    # Print compact summary
    print("\n--- Summary (5 key variables) ---")
    keep = {"lactate", "map", "iv_fluid_ml", "vasopressor_ne_eq", "urine_output_ml"}
    compact = out[out["variable"].isin(keep)][
        ["variable", "metric", "mimic_pct", "eicu_full_pct", "eicu_tier_a_pct"]
    ]
    print(compact.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
