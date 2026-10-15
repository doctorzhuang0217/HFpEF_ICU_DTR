#!/usr/bin/env python3
"""
45c_build_state_tensors — Assemble modeling tensors from values, masks,
delta-t, frequency, and static covariates.

Purpose
-------
1. Drop ultra-sparse variables (BNP ≥99% missing, temperature ≥90%).
2. Apply LOCF (Last Observation Carried Forward) imputation within each stay.
3. Fill remaining NaN (pre-first-observation) with population median.
4. Compute z-score normalization parameters (mean/std from full cohort).
5. Z-score normalize state variables.
6. Merge 40b values + 40c missingness features + static covariates → tensor.

Protocol reference: Step 45c (C3-C4)

Inputs
------
- data/interim/40b_grid2h_states_mimic.parquet
- data/interim/40c_missingness_features_mimic.parquet
- data/staging/25f_primary_cohort_candidate_v1_mimic.parquet
- configs/common_variable_map_v1.yaml

Outputs
-------
- data/interim/45c_state_tensors_v1_mimic.parquet
- configs/45c_normalization_params_v1.yaml
- reports/45c_state_tensors_v1_summary.txt
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path

# ── State variables from 40c (24 total) ──────────────────────────────
STATE_VARS = [
    "map", "sbp", "dbp", "heart_rate", "spo2", "cvp", "resp_rate",
    "temperature", "fio2",
    "lactate", "creatinine", "bun", "potassium", "sodium",
    "bicarbonate", "glucose", "ph",
    "hemoglobin", "wbc", "platelet", "albumin", "bilirubin",
    "troponin", "bnp",
]

# Variables to DROP due to ultra-sparse data (>90% bin-level missing)
DROP_VARS = ["bnp", "temperature"]

# Non-state columns from 40b that are kept as-is (no imputation/normalization)
PASSTHROUGH_COLS = [
    "stay_id", "bin", "bin_start_h", "bin_end_h",
    "iv_fluid_ml", "vasopressor_ne_eq", "diuretic_furo_eq_mg",
    "fluid_balance_ml", "fluid_balance_L", "fluid_balance_L_clipped",
    "cumulative_fluid_balance_L", "urine_output_ml",
    "output_documentation_absent",
]

# Static covariates to merge from 25f cohort
STATIC_COVARIATES = [
    "age", "gender", "admission_type",
    "hf_subtype", "comparator_group",
    "dominant_syndrome", "reason_for_icu",
    "hemo_gate_passed", "include_in_phase05",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="45c — build state tensors (LOCF + z-score)")
    parser.add_argument(
        "--scheme", choices=["coarse_3", "medium_4"], default="coarse_3",
        help="Action-discretization scheme this tensor build is paired with. "
             "DEFAULT coarse_3 reproduces the frozen output exactly; medium_4 "
             "writes *_medium4 outputs (state content is action-agnostic, but a "
             "distinct artifact keeps the medium_4 chain self-contained).",
    )
    args = parser.parse_args()
    scheme = args.scheme
    suffix = "" if scheme == "coarse_3" else f"_{scheme.replace('_', '')}"

    cfg = load_config()
    interim_dir = resolve_path(cfg, "interim")
    staging_dir = resolve_path(cfg, "staging")
    configs_dir = resolve_path(cfg, "configs")
    reports_dir = resolve_path(cfg, "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"45c_build_state_tensors — LOCF + z-score + tensor assembly "
          f"(scheme={scheme})")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────
    print("\n[1/6] Loading data ...")
    grid = pd.read_parquet(interim_dir / "40b_grid2h_states_mimic.parquet")
    miss = pd.read_parquet(interim_dir / "40c_missingness_features_mimic.parquet")
    cohort = pd.read_parquet(staging_dir / "25f_primary_cohort_candidate_v1_mimic.parquet")

    n_rows = len(grid)
    n_stays = grid["stay_id"].nunique()
    print(f"  Grid: {n_rows:,} rows, {n_stays:,} stays")

    # ── Drop ultra-sparse variables ───────────────────────────────────
    print("\n[2/6] Dropping ultra-sparse variables ...")
    active_vars = [v for v in STATE_VARS if v not in DROP_VARS and v in grid.columns]
    for v in DROP_VARS:
        if v in grid.columns:
            pct_miss = grid[v].isna().mean() * 100
            print(f"  DROP: {v} ({pct_miss:.1f}% missing)")
    print(f"  Active state variables: {len(active_vars)}")

    # ── LOCF imputation ───────────────────────────────────────────────
    print("\n[3/6] LOCF imputation ...")
    grid = grid.sort_values(["stay_id", "bin"]).reset_index(drop=True)

    pre_miss = {}
    post_locf_miss = {}
    for var in active_vars:
        pre_miss[var] = grid[var].isna().mean()
        grid[var] = grid.groupby("stay_id")[var].ffill()
        post_locf_miss[var] = grid[var].isna().mean()

    locf_resolved = {v: pre_miss[v] - post_locf_miss[v] for v in active_vars}
    total_pre = np.mean(list(pre_miss.values()))
    total_post = np.mean(list(post_locf_miss.values()))
    print(f"  Mean missingness: {total_pre*100:.1f}% → {total_post*100:.1f}% (LOCF)")

    # ── Population median fill (remaining NaN = pre-first-observation) ─
    print("\n[4/6] Population median fill ...")
    pop_medians = {}
    for var in active_vars:
        med = grid[var].median()
        pop_medians[var] = float(med) if pd.notna(med) else 0.0
        remaining = grid[var].isna().sum()
        if remaining > 0:
            grid[var] = grid[var].fillna(pop_medians[var])

    final_miss = {v: grid[v].isna().sum() for v in active_vars}
    assert all(m == 0 for m in final_miss.values()), (
        f"Residual NaN after fill: {[v for v, m in final_miss.items() if m > 0]}"
    )
    print("  All state variables fully imputed.")

    # ── Z-score normalization ─────────────────────────────────────────
    print("\n[5/6] Z-score normalization ...")
    norm_params = {}
    for var in active_vars:
        mu = float(grid[var].mean())
        sigma = float(grid[var].std())
        # Protect against zero std (constant variable)
        if sigma < 1e-10:
            sigma = 1.0
            print(f"  WARNING: {var} has near-zero std, using σ=1.0")
        norm_params[var] = {"mean": mu, "std": sigma, "pop_median": pop_medians[var]}
        grid[f"{var}_z"] = (grid[var] - mu) / sigma

    # Save normalization params
    norm_cfg = {
        "step": "45c_build_state_tensors",
        "date_frozen": datetime.now().strftime("%Y-%m-%d"),
        "n_active_vars": len(active_vars),
        "dropped_vars": DROP_VARS,
        "active_vars": active_vars,
        "imputation_method": "LOCF + population_median",
        "normalization_method": "z-score",
        "params": norm_params,
    }
    norm_path = configs_dir / f"45c_normalization_params_v1{suffix}.yaml"
    with open(norm_path, "w") as f:
        yaml.dump(norm_cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  Saved: {norm_path}")

    # ── Assemble tensor ───────────────────────────────────────────────
    print("\n[6/6] Assembling state tensor ...")

    # Passthrough columns from 40b
    keep_pass = [c for c in PASSTHROUGH_COLS if c in grid.columns]

    # Raw state values (un-normalized, for reference)
    raw_state_cols = [v for v in active_vars if v in grid.columns]

    # Z-scored state columns
    z_cols = [f"{v}_z" for v in active_vars]

    # Missingness features from 40c (mask, tsl_h, freq_24h, never_measured)
    # Only for active (non-dropped) vars
    miss_feat_cols = []
    for v in active_vars:
        for feat_suffix in ("_mask", "_tsl_h", "_freq_24h", "_never_measured"):
            col = f"{v}{feat_suffix}"
            if col in miss.columns:
                miss_feat_cols.append(col)

    # Also keep missingness features for dropped vars (informative)
    for v in DROP_VARS:
        for feat_suffix in ("_mask", "_tsl_h", "_freq_24h", "_never_measured"):
            col = f"{v}{feat_suffix}"
            if col in miss.columns:
                miss_feat_cols.append(col)

    # Merge missingness features (40c has all grid columns + miss features)
    # Only take miss-specific columns + key
    miss_only = miss[["stay_id", "bin"] + miss_feat_cols].copy()
    tensor = grid[keep_pass + raw_state_cols + z_cols].merge(
        miss_only, on=["stay_id", "bin"], how="left"
    )

    # Merge static covariates from 25f
    static_avail = [c for c in STATIC_COVARIATES if c in cohort.columns]
    static_df = cohort[["stay_id"] + static_avail].drop_duplicates(subset="stay_id")
    tensor = tensor.merge(static_df, on="stay_id", how="left")

    n_tensor_cols = tensor.shape[1]
    print(f"  Tensor: {len(tensor):,} rows × {n_tensor_cols} columns")
    print(f"    Passthrough:    {len(keep_pass)}")
    print(f"    Raw state:      {len(raw_state_cols)}")
    print(f"    Z-scored state: {len(z_cols)}")
    print(f"    Miss features:  {len(miss_feat_cols)}")
    print(f"    Static covars:  {len(static_avail)}")

    # ── Save ──────────────────────────────────────────────────────────
    out_path = interim_dir / f"45c_state_tensors_v1{suffix}_mimic.parquet"
    tensor.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"  Saved: {out_path}")

    # ── Summary report ────────────────────────────────────────────────
    report_lines = [
        "=" * 72,
        "45c — State Tensor Assembly Summary (MIMIC-IV)",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Stays: {n_stays:,}, Rows: {n_rows:,}",
        f"Active state vars: {len(active_vars)}, Dropped: {DROP_VARS}",
        "",
        "IMPUTATION SUMMARY (LOCF + population median)",
        "───────────────────────────────────────────────",
        f"  {'Variable':<20} {'%pre':<10} {'%post-LOCF':<12} {'%resolved':<12} {'pop_median':<12}",
        "  " + "─" * 60,
    ]
    for var in active_vars:
        pre_pct = pre_miss[var] * 100
        post_pct = post_locf_miss[var] * 100
        resolved_pct = locf_resolved[var] * 100
        median_val = pop_medians[var]
        report_lines.append(
            f"  {var:<20} {pre_pct:>7.1f}%  {post_pct:>9.1f}%  {resolved_pct:>9.1f}%  {median_val:>10.2f}"
        )

    report_lines.extend([
        "",
        "NORMALIZATION PARAMETERS",
        "────────────────────────",
        f"  {'Variable':<20} {'mean':<12} {'std':<12}",
        "  " + "─" * 40,
    ])
    for var in active_vars:
        mu = norm_params[var]["mean"]
        sigma = norm_params[var]["std"]
        report_lines.append(f"  {var:<20} {mu:>10.4f}  {sigma:>10.4f}")

    report_lines.extend([
        "",
        "TENSOR STRUCTURE",
        "─────────────────",
        f"  Total columns: {n_tensor_cols}",
        f"  Passthrough:    {len(keep_pass)} (stay_id, bin, actions, fluid balance, ...)",
        f"  Raw state:      {len(raw_state_cols)} (un-normalized values)",
        f"  Z-scored state: {len(z_cols)} (modeling features)",
        f"  Miss features:  {len(miss_feat_cols)} (mask, tsl_h, freq_24h, never_measured)",
        f"  Static covars:  {len(static_avail)} ({', '.join(static_avail)})",
    ])

    report = "\n".join(report_lines) + "\n"
    report_path = reports_dir / f"45c_state_tensors_v1{suffix}_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {report_path}")
    print(report)
    print("[45c] Done.")


if __name__ == "__main__":
    main()
