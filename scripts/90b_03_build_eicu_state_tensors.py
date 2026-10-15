#!/usr/bin/env python3
"""90b_03 — Build eICU State Tensors with MIMIC Normalization

Protocol: v3.1, Step 90b (sub-step 3)
Compute: C2

Purpose
-------
Build eICU state tensors using FROZEN MIMIC normalization parameters.
This is the critical transportability step — we apply MIMIC z-score
params (mean/std) to eICU data, NOT re-estimated eICU params.

Pipeline mirrors MIMIC 45c but with key differences:
  1. LOCF imputation within each stay (same as MIMIC)
  2. Population median fill using MIMIC medians (NOT eICU medians)
  3. Z-score using MIMIC mean/std (NOT eICU mean/std)
  4. Merge missingness features
  5. Add static covariates from eICU cohort

This design choice ensures that the frozen MIMIC models receive
data in the same scale they were trained on.

Inputs
------
- data/interim/90b_eicu_grid2h_states.parquet
- data/interim/90b_eicu_missingness_features.parquet
- data/staging/90b_eicu_cohort_v1.parquet
- configs/45c_normalization_params_v1.yaml  (MIMIC params — FROZEN)

Outputs
-------
- data/interim/90b_eicu_state_tensors_v1.parquet
- reports/90b_03_eicu_state_tensors_summary.txt
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
INTERIM_DIR = ROOT / "data" / "interim"
STAGING_DIR = ROOT / "data" / "staging"
REPORTS_DIR = ROOT / "reports"

for d in [INTERIM_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# State variables (must match MIMIC 45c active_vars — these get _z columns)
STATE_VARS = [
    "map", "sbp", "dbp", "heart_rate", "spo2", "cvp", "resp_rate",
    "fio2", "lactate", "creatinine", "bun", "potassium", "sodium",
    "bicarbonate", "glucose", "ph", "hemoglobin", "wbc", "platelet",
    "albumin", "bilirubin", "troponin",
]

# Variables in MIMIC that have missingness features but NO z-score column
# (dropped from active vars but missingness pattern is informative)
MIMIC_ONLY_MISS_VARS = ["bnp", "temperature"]

# All vars that need missingness features (mask, tsl_h, freq_24h, never_measured)
ALL_MISS_VARS = STATE_VARS + MIMIC_ONLY_MISS_VARS

# MIMIC model statics (must match frozen model exactly)
MIMIC_STATIC_COLS = ["age", "hemo_gate_passed", "include_in_phase05"]


def main():
    print("=" * 72)
    print("90b_03 — Build eICU State Tensors (MIMIC-normalized)")
    print("=" * 72)
    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 1. Load MIMIC normalization params ───────────────────────────────
    print("\n[1] Loading MIMIC normalization params (FROZEN)...")
    norm_path = ROOT / "configs" / "45c_normalization_params_v1.yaml"
    with open(norm_path) as f:
        norm_cfg = yaml.safe_load(f)

    mimic_params = norm_cfg["params"]
    print(f"  Loaded params for {len(mimic_params)} variables")
    print("  CRITICAL: Using MIMIC mean/std — NOT re-estimated from eICU!")

    # ── 2. Load eICU grid ────────────────────────────────────────────────
    print("\n[2] Loading eICU grid data...")
    grid = pd.read_parquet(INTERIM_DIR / "90b_eicu_grid2h_states.parquet")
    # NOTE: miss_df NOT loaded — grid already contains missingness features
    # from 90b_02 inline computation. Merging miss_df would create _x/_y dupes.
    cohort = pd.read_parquet(STAGING_DIR / "90b_eicu_cohort_v1.parquet")

    n_stays = grid["stay_id"].nunique()
    n_rows = len(grid)
    print(f"  Grid: {n_rows:,} rows, {n_stays:,} stays")

    # ── 3. LOCF imputation ───────────────────────────────────────────────
    print("\n[3] LOCF imputation...")
    grid = grid.sort_values(["stay_id", "bin"]).reset_index(drop=True)

    pre_miss = {}
    post_miss = {}
    for var in STATE_VARS:
        if var not in grid.columns:
            # Variable not available in eICU → will be all NaN → all median
            grid[var] = np.nan
            pre_miss[var] = 1.0
            post_miss[var] = 1.0
            continue
        pre_miss[var] = grid[var].isna().mean()
        grid[var] = grid.groupby("stay_id")[var].ffill()
        post_miss[var] = grid[var].isna().mean()

    total_pre = np.mean(list(pre_miss.values()))
    total_post = np.mean(list(post_miss.values()))
    print(f"  Mean missingness: {total_pre*100:.1f}% → {total_post*100:.1f}% (LOCF)")

    # ── 4. Population median fill (MIMIC medians) ────────────────────────
    print("\n[4] Population median fill (MIMIC medians)...")
    for var in STATE_VARS:
        mimic_median = mimic_params.get(var, {}).get("pop_median", 0.0)
        remaining = grid[var].isna().sum()
        if remaining > 0:
            grid[var] = grid[var].fillna(mimic_median)
            print(f"    {var}: filled {remaining:,} NaN with MIMIC median={mimic_median}")

    # Verify no NaN remains
    for var in STATE_VARS:
        assert grid[var].isna().sum() == 0, f"Residual NaN in {var}!"

    # ── 5. Z-score with MIMIC params ─────────────────────────────────────
    print("\n[5] Z-score normalization (MIMIC params)...")
    z_comparison = {}
    for var in STATE_VARS:
        mimic_mean = mimic_params[var]["mean"]
        mimic_std = mimic_params[var]["std"]
        if mimic_std < 1e-10:
            mimic_std = 1.0

        # Compute z-score using MIMIC params
        grid[f"{var}_z"] = (grid[var] - mimic_mean) / mimic_std

        # Track distribution shift for reporting
        eicu_mean = grid[var].mean()
        eicu_std = grid[var].std()
        z_comparison[var] = {
            "mimic_mean": mimic_mean,
            "mimic_std": mimic_std,
            "eicu_mean": eicu_mean,
            "eicu_std": eicu_std,
            "shift_z": (eicu_mean - mimic_mean) / mimic_std,
        }

    print(f"\n  Distribution shift (eICU mean - MIMIC mean, in MIMIC SDs):")
    for var, comp in z_comparison.items():
        shift = comp["shift_z"]
        flag = " ⚠" if abs(shift) > 1.0 else ""
        print(f"    {var:20s}: Δ = {shift:+.3f} SD{flag}")

    # ── 6. Verify missingness features (already in grid from 90b_02) ─────
    print("\n[6] Verifying missingness features from grid...")
    # NOTE: grid from 90b_02 already has {var}_mask, {var}_tsl_h,
    # {var}_never_measured computed inline. We do NOT merge the separate
    # miss_df because that creates _x/_y column duplication.
    tensor = grid.copy()

    existing_miss = [c for c in tensor.columns
                     if any(c.endswith(s) for s in ("_mask", "_tsl_h", "_never_measured"))]
    print(f"  Missingness columns already in grid: {len(existing_miss)}")

    # ── 6b. Verify/add freq_24h columns ──────────────────────────────────
    print("\n[6b] Verifying freq_24h columns...")
    # STATE_VARS freq_24h should already be computed by 90b_02
    freq_from_grid = 0
    freq_added_zero = 0
    for var in ALL_MISS_VARS:
        col = f"{var}_freq_24h"
        if col in tensor.columns:
            freq_from_grid += 1
        else:
            # Not computed (bnp/temperature or missing) → set to 0
            tensor[col] = 0.0
            freq_added_zero += 1
    print(f"  freq_24h from 90b_02 grid: {freq_from_grid}")
    print(f"  freq_24h set to 0 (not in eICU): {freq_added_zero}")

    # ── 6c. Add missing missingness features for all vars ──────────────
    # Some STATE_VARS (e.g. cvp, fio2) may not be extracted from eICU raw
    # tables, so 90b_02 skipped their missingness computation. Fill them
    # as "never measured" here, same treatment as bnp/temperature.
    print("\n[6c] Adding missingness features for unextracted variables...")
    all_need_miss = ALL_MISS_VARS  # STATE_VARS + MIMIC_ONLY_MISS_VARS
    miss_added = 0
    for var in all_need_miss:
        for suffix, default in [("_mask", 0), ("_tsl_h", 0.0), ("_never_measured", 1)]:
            col = f"{var}{suffix}"
            if col not in tensor.columns:
                tensor[col] = default
                miss_added += 1
    if miss_added > 0:
        print(f"  Added {miss_added} missing missingness columns "
              f"(never-measured defaults)")
    else:
        print(f"  All missingness columns present")

    # Drop temporary helper columns from 90b_02 vectorized computation
    helper_cols = [c for c in tensor.columns if c.startswith("_obs_bin_")]
    if helper_cols:
        tensor = tensor.drop(columns=helper_cols)
        print(f"  Dropped {len(helper_cols)} helper columns")

    # ── 7. Add MIMIC-compatible static covariates ──────────────────────
    print("\n[7] Adding static covariates (MIMIC-compatible)...")

    # 7a. Age (rename from eICU's age_num)
    if "age_num" in cohort.columns:
        age_map = cohort[["patientunitstayid", "age_num"]].rename(
            columns={"patientunitstayid": "stay_id", "age_num": "age"}
        )
        if "age" not in tensor.columns:
            tensor = tensor.merge(age_map, on="stay_id", how="left")
        print(f"    age: merged from cohort.age_num")

    # 7b. hemo_gate_passed = 1 for all eICU stays (all pass by construction)
    tensor["hemo_gate_passed"] = 1
    print(f"    hemo_gate_passed: set to 1 (all eICU stays eligible)")

    # 7c. include_in_phase05 = 1 for all eICU stays
    tensor["include_in_phase05"] = 1
    print(f"    include_in_phase05: set to 1 (all eICU stays eligible)")

    # 7d. Also add hf_subtype for downstream subgroup analysis (NOT model static)
    if "hf_subtype" in cohort.columns:
        hf_map_df = cohort[["patientunitstayid", "hf_subtype"]].rename(
            columns={"patientunitstayid": "stay_id"}
        )
        if "hf_subtype" not in tensor.columns:
            tensor = tensor.merge(hf_map_df, on="stay_id", how="left")

    static_avail = [c for c in MIMIC_STATIC_COLS if c in tensor.columns]
    print(f"  MIMIC model statics: {static_avail}")

    # ── 8. Feature dimension verification ────────────────────────────────
    print("\n[8] Feature dimension verification (MIMIC compatibility)...")
    temporal_z = sorted([c for c in tensor.columns if c.endswith("_z")])
    temporal_mask = sorted([c for c in tensor.columns if c.endswith("_mask")])
    temporal_tsl = sorted([c for c in tensor.columns if c.endswith("_tsl_h")])
    temporal_freq = sorted([c for c in tensor.columns if c.endswith("_freq_24h")])
    temporal_never = sorted([c for c in tensor.columns if c.endswith("_never_measured")])

    n_temporal = len(temporal_z) + len(temporal_mask) + len(temporal_tsl) + \
                 len(temporal_freq) + len(temporal_never)
    print(f"  _z: {len(temporal_z)}, _mask: {len(temporal_mask)}, "
          f"_tsl_h: {len(temporal_tsl)}, _freq_24h: {len(temporal_freq)}, "
          f"_never_measured: {len(temporal_never)}")
    print(f"  Total temporal features: {n_temporal} (MIMIC expects 118)")
    print(f"  MIMIC model statics: {static_avail} (MIMIC expects 3)")

    if n_temporal != 118:
        print(f"  WARNING: temporal feature count {n_temporal} != 118!")
    if len(static_avail) != 3:
        print(f"  WARNING: static feature count {len(static_avail)} != 3!")

    # ── 9. Save ──────────────────────────────────────────────────────────
    print("\n[9] Saving state tensors...")
    n_cols = tensor.shape[1]
    print(f"  Tensor: {len(tensor):,} rows × {n_cols} columns")

    out_path = INTERIM_DIR / "90b_eicu_state_tensors_v1.parquet"
    tensor.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"  Saved: {out_path}")

    # ── 10. Summary report ───────────────────────────────────────────────
    report_lines = [
        "=" * 72,
        "90b_03 — eICU State Tensors (MIMIC-normalized)",
        "=" * 72,
        f"Generated: {ts}",
        "",
        "NORMALIZATION",
        "─────────────",
        "  Method: MIMIC z-score params applied to eICU data",
        "  Imputation: LOCF + MIMIC population medians",
        "",
        "IMPUTATION SUMMARY",
        "──────────────────",
        f"  {'Variable':<20} {'%pre':<10} {'%post-LOCF':<12} {'MIMIC median':<14}",
        "  " + "─" * 55,
    ]
    for var in STATE_VARS:
        pre = pre_miss.get(var, 1.0) * 100
        post = post_miss.get(var, 1.0) * 100
        med = mimic_params.get(var, {}).get("pop_median", "?")
        report_lines.append(f"  {var:<20} {pre:>7.1f}%  {post:>9.1f}%  {med}")

    report_lines.extend([
        "",
        "DISTRIBUTION SHIFT (eICU vs MIMIC)",
        "───────────────────────────────────",
        f"  {'Variable':<20} {'MIMIC μ':<12} {'eICU μ':<12} {'Δ (SD)':<10} {'Flag'}",
        "  " + "─" * 55,
    ])
    for var, comp in z_comparison.items():
        flag = "⚠ >1 SD" if abs(comp["shift_z"]) > 1.0 else ""
        report_lines.append(
            f"  {var:<20} {comp['mimic_mean']:>10.3f}  {comp['eicu_mean']:>10.3f}  "
            f"{comp['shift_z']:>+7.3f}   {flag}"
        )

    report_lines.extend([
        "",
        "TENSOR STRUCTURE",
        "────────────────",
        f"  Rows: {len(tensor):,} ({n_stays:,} stays × 12 bins)",
        f"  Columns: {n_cols}",
        f"  Temporal features: {n_temporal} (MIMIC expects 118)",
        f"  MIMIC model statics: {static_avail}",
        f"  bnp/temperature: missingness-only (no z-score, never measured in eICU)",
        f"  freq_24h: all set to 0.0 (not computed for eICU)",
        "",
        "NEXT STEPS",
        "──────────",
        "  90b_04: Build eICU actions + fit eICU behavior policy",
    ])

    report_path = REPORTS_DIR / "90b_03_eicu_state_tensors_summary.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\n  Report: {report_path}")

    total_time = time.time() - t0
    print()
    print("=" * 72)
    print(f"90b_03 COMPLETE. Time: {total_time/60:.1f} min")
    print("=" * 72)


if __name__ == "__main__":
    main()
