#!/usr/bin/env python3
"""
45e_build_data_quality_flags — Attach modeling-time quality flags.

Purpose
-------
Compute per-stay quality indicators for sensitivity analysis:
  1. extreme_sparsity: stays where ≥80% of state bins are missing across core vitals
  2. fluid_qc_flag: stays with output_documentation_absent or extreme fluid balance
  3. rrt_exposure: stays with RRT/CRRT during observation window
  4. bnp_ever_measured: whether BNP was ever measured (for sensitivity sub-analysis)
  5. sparse_action_cells: stays landing in rare action cells (<1% of total)

Protocol reference: Step 45e (C2)

Inputs
------
- data/interim/40b_grid2h_states_mimic.parquet
- data/interim/40c_missingness_features_mimic.parquet
- data/interim/45b_actions_2h_v1_mimic.parquet
- data/staging/25f_primary_cohort_candidate_v1_mimic.parquet

Outputs
-------
- data/interim/45e_data_quality_flags_v1_mimic.parquet
- reports/45e_data_quality_flags_v1_summary.txt
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path

# Core vitals for sparsity computation
CORE_VITALS = ["map", "sbp", "dbp", "heart_rate", "spo2", "resp_rate"]

# D1 fluid balance clip bounds
FB_CLIP_LOW = -10.0
FB_CLIP_HIGH = 15.0


def main() -> None:
    cfg = load_config()
    interim_dir = resolve_path(cfg, "interim")
    staging_dir = resolve_path(cfg, "staging")
    reports_dir = resolve_path(cfg, "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("45e_build_data_quality_flags — Modeling-time quality flags")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────
    print("\n[1/3] Loading data ...")
    grid = pd.read_parquet(interim_dir / "40b_grid2h_states_mimic.parquet")
    miss = pd.read_parquet(interim_dir / "40c_missingness_features_mimic.parquet")
    actions = pd.read_parquet(interim_dir / "45b_actions_2h_v1_mimic.parquet")
    cohort = pd.read_parquet(staging_dir / "25f_primary_cohort_candidate_v1_mimic.parquet")

    stays = grid["stay_id"].unique()
    n_stays = len(stays)
    print(f"  Stays: {n_stays:,}")

    flags = pd.DataFrame({"stay_id": stays})

    # ── Flag 1: Extreme sparsity ──────────────────────────────────────
    print("\n[2/3] Computing quality flags ...")
    print("  Flag 1: extreme_sparsity ...")
    # Average missingness across core vitals per stay
    mask_cols = [f"{v}_mask" for v in CORE_VITALS if f"{v}_mask" in miss.columns]
    if mask_cols:
        stay_obs_rate = (
            miss.groupby("stay_id")[mask_cols].mean().mean(axis=1)
        )
        stay_miss_rate = 1 - stay_obs_rate
        extreme_sparse = (stay_miss_rate >= 0.80).astype(int).reset_index()
        extreme_sparse.columns = ["stay_id", "extreme_sparsity"]
        flags = flags.merge(extreme_sparse, on="stay_id", how="left")
        flags["extreme_sparsity"] = flags["extreme_sparsity"].fillna(0).astype(int)
    else:
        flags["extreme_sparsity"] = 0

    # ── Flag 2: Fluid QC ─────────────────────────────────────────────
    print("  Flag 2: fluid_qc_flag ...")
    # output_documentation_absent from 40b
    if "output_documentation_absent" in grid.columns:
        oda = grid.groupby("stay_id")["output_documentation_absent"].first().reset_index()
        oda.columns = ["stay_id", "output_doc_absent"]
        flags = flags.merge(oda, on="stay_id", how="left")
        flags["output_doc_absent"] = flags["output_doc_absent"].fillna(0).astype(int)
    else:
        flags["output_doc_absent"] = 0

    # Extreme cumulative fluid balance (outside clip bounds)
    if "cumulative_fluid_balance_L" in grid.columns:
        cfb_extremes = (
            grid.groupby("stay_id")["cumulative_fluid_balance_L"]
            .agg(["min", "max"])
            .reset_index()
        )
        cfb_extremes["extreme_fb"] = (
            (cfb_extremes["min"] < FB_CLIP_LOW) |
            (cfb_extremes["max"] > FB_CLIP_HIGH)
        ).astype(int)
        flags = flags.merge(
            cfb_extremes[["stay_id", "extreme_fb"]], on="stay_id", how="left"
        )
        flags["extreme_fb"] = flags["extreme_fb"].fillna(0).astype(int)
    else:
        flags["extreme_fb"] = 0

    # Combined fluid QC flag
    flags["fluid_qc_flag"] = (
        (flags["output_doc_absent"] == 1) | (flags["extreme_fb"] == 1)
    ).astype(int)

    # ── Flag 3: RRT exposure ──────────────────────────────────────────
    print("  Flag 3: rrt_exposure ...")
    rrt_col = None
    for candidate in ["rrt_crrt_any_during_stay", "has_rrt"]:
        if candidate in cohort.columns:
            rrt_col = candidate
            break
    if rrt_col:
        rrt = (
            cohort[["stay_id", rrt_col]]
            .drop_duplicates(subset="stay_id")
            .rename(columns={rrt_col: "rrt_exposure"})
        )
        flags = flags.merge(rrt, on="stay_id", how="left")
        flags["rrt_exposure"] = flags["rrt_exposure"].fillna(0).astype(int)
    else:
        flags["rrt_exposure"] = 0

    # ── Flag 4: BNP ever measured ─────────────────────────────────────
    print("  Flag 4: bnp_ever_measured ...")
    if "bnp_never_measured" in miss.columns:
        bnp_nm = miss.groupby("stay_id")["bnp_never_measured"].first().reset_index()
        bnp_nm["bnp_ever_measured"] = (bnp_nm["bnp_never_measured"] == 0).astype(int)
        flags = flags.merge(
            bnp_nm[["stay_id", "bnp_ever_measured"]], on="stay_id", how="left"
        )
        flags["bnp_ever_measured"] = flags["bnp_ever_measured"].fillna(0).astype(int)
    else:
        flags["bnp_ever_measured"] = 0

    # ── Flag 5: Sparse action cells ───────────────────────────────────
    print("  Flag 5: sparse_action_cells ...")
    if "action_idx" in actions.columns:
        # Compute action cell frequency
        cell_freq = actions["action_idx"].value_counts(normalize=True)
        rare_cells = set(cell_freq[cell_freq < 0.01].index)

        # For each stay, check if any bin lands in a rare cell
        stay_rare = (
            actions.groupby("stay_id")["action_idx"]
            .apply(lambda x: int(x.isin(rare_cells).any()))
            .reset_index()
        )
        stay_rare.columns = ["stay_id", "has_sparse_action"]
        flags = flags.merge(stay_rare, on="stay_id", how="left")
        flags["has_sparse_action"] = flags["has_sparse_action"].fillna(0).astype(int)
    else:
        flags["has_sparse_action"] = 0

    # ── Combined exclusion recommendation ─────────────────────────────
    # Not mandatory exclusions — these are sensitivity flags
    flags["any_quality_concern"] = (
        (flags["extreme_sparsity"] == 1) |
        (flags["fluid_qc_flag"] == 1) |
        (flags["rrt_exposure"] == 1)
    ).astype(int)

    # ── Save ──────────────────────────────────────────────────────────
    print("\n[3/3] Saving ...")
    out_path = interim_dir / "45e_data_quality_flags_v1_mimic.parquet"
    flags.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"  Saved: {out_path}  ({len(flags):,} rows, {flags.shape[1]} cols)")

    # ── Summary report ────────────────────────────────────────────────
    flag_cols = [
        "extreme_sparsity", "output_doc_absent", "extreme_fb",
        "fluid_qc_flag", "rrt_exposure", "bnp_ever_measured",
        "has_sparse_action", "any_quality_concern",
    ]
    report_lines = [
        "=" * 72,
        "45e — Data Quality Flags Summary (MIMIC-IV)",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Stays: {n_stays:,}",
        "",
        "FLAG PREVALENCE",
        "────────────────",
    ]
    for col in flag_cols:
        if col in flags.columns:
            n_flagged = flags[col].sum()
            pct = n_flagged / n_stays * 100
            report_lines.append(f"  {col:<25}: {n_flagged:>6,} / {n_stays:,} ({pct:5.1f}%)")

    if "action_idx" in actions.columns:
        cell_freq = actions["action_idx"].value_counts(normalize=True)
        rare_cells = cell_freq[cell_freq < 0.01]
        report_lines.extend([
            "",
            "SPARSE ACTION CELLS (<1% frequency)",
            "─────────────────────────────────────",
            f"  Rare cells: {len(rare_cells)} / {len(cell_freq)}",
        ])
        for idx, freq in rare_cells.items():
            report_lines.append(f"  action_idx={idx}: {freq*100:.2f}%")

    report = "\n".join(report_lines) + "\n"
    report_path = reports_dir / "45e_data_quality_flags_v1_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {report_path}")
    print(report)
    print("[45e] Done.")


if __name__ == "__main__":
    main()
