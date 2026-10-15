#!/usr/bin/env python3
"""
25a_define_time_zero_landmark — Confirm T0 landmark for index cohort.

Purpose
-------
Joins 20f cohort candidates (index stays) with 10c timestamps to produce
a per-index-stay time reference table with T0 landmarks, pre-T0 windows,
decision-grid boundaries, and outcome horizons computed FROM T0 (not intime).

Most heavy lifting was done in 10c; this step scopes to the index cohort
and adds T0-relative horizons needed for Steps 25b–25f and downstream.

Inputs
------
- data/staging/20f_cohort_candidates_mimic.parquet
- data/staging/10c_timestamps_mimic.parquet

Outputs
-------
- data/staging/25a_time_zero_flags_mimic.parquet
- reports/25a_time_zero_flags_mimic_summary.txt
"""

from __future__ import annotations

import sys
import textwrap
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path


def main() -> None:
    cfg = load_config()
    staging_dir = resolve_path(cfg, "staging")
    reports_dir = resolve_path(cfg, "reports")
    staging_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    t0_primary_h = cfg["study_design"]["time_zero"]["primary_hours"]  # 6
    t0_sens_h = cfg["study_design"]["time_zero"]["sensitivity_hours"]  # 12
    grid_bin_h = cfg["study_design"]["decision_grid"]["bin_hours"]  # 2
    primary_horizon_h = cfg["study_design"]["decision_grid"]["primary_horizon_hours"]  # 24
    sens_horizons_h = cfg["study_design"]["decision_grid"]["sensitivity_horizons_hours"]  # [48, 72]

    print("=" * 70)
    print("25a_define_time_zero_landmark — T0 landmark for index cohort")
    print("=" * 70)

    # ── load inputs ──────────────────────────────────────────────────
    print("\n[1/3] Loading inputs ...")
    cohort = pd.read_parquet(staging_dir / "20f_cohort_candidates_mimic.parquet")
    idx = cohort[cohort["is_index_stay"] == True].copy()  # noqa: E712
    n_index = len(idx)
    print(f"  Index stays: {n_index:,}")

    ts = pd.read_parquet(staging_dir / "10c_timestamps_mimic.parquet")
    print(f"  Timestamps table: {len(ts):,} rows")

    # ── merge ────────────────────────────────────────────────────────
    print("\n[2/3] Building T0-relative time references ...")
    out = idx[["stay_id", "subject_id", "hadm_id", "hf_subtype", "comparator_group"]].merge(
        ts, on=["stay_id", "subject_id", "hadm_id"], how="inner",
    )
    print(f"  Merged: {len(out):,} rows")

    # T0-relative outcome horizons (from T0, not from intime)
    out["t0_plus_24h"] = out["time_zero_6h"] + pd.Timedelta(hours=primary_horizon_h)
    for h in sens_horizons_h:
        out[f"t0_plus_{h}h"] = out["time_zero_6h"] + pd.Timedelta(hours=h)

    # Decision grid boundaries from T0
    n_bins = primary_horizon_h // grid_bin_h
    for i in range(n_bins):
        start_h = i * grid_bin_h
        end_h = (i + 1) * grid_bin_h
        out[f"grid_bin_{i:02d}_start"] = out["time_zero_6h"] + pd.Timedelta(hours=start_h)
        out[f"grid_bin_{i:02d}_end"] = out["time_zero_6h"] + pd.Timedelta(hours=end_h)

    # ── write output ─────────────────────────────────────────────────
    out = out.sort_values(["subject_id", "intime"]).reset_index(drop=True)
    out_path = staging_dir / "25a_time_zero_flags_mimic.parquet"
    out.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"  → {out_path}  ({len(out):,} rows, {out.shape[1]} cols)")

    # ── summary report ───────────────────────────────────────────────
    print("\n[3/3] Generating summary report ...")

    n_t0_6h = int(out["t0_6h_eligible"].sum())
    n_t0_12h = int(out["t0_12h_eligible"].sum())

    # T0 eligibility by HF subtype
    subtype_t0 = (
        out.groupby("hf_subtype")["t0_6h_eligible"]
        .agg(["count", "sum"])
        .rename(columns={"count": "total", "sum": "t0_eligible"})
    )
    subtype_t0["pct"] = (subtype_t0["t0_eligible"] / subtype_t0["total"] * 100).round(1)
    subtype_str = subtype_t0.to_string()

    report = textwrap.dedent(f"""\
    ================================================================
    25a_define_time_zero_landmark — Summary Report
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ================================================================

    COHORT
    ──────
    Index stays             : {n_index:>10,}
    Merged with timestamps  : {len(out):>10,}

    T0 ELIGIBILITY (among index stays)
    ────────────────────────────────────
    T0-6h eligible          : {n_t0_6h:>10,}  ({n_t0_6h / len(out) * 100:.1f}%)
    T0-12h eligible         : {n_t0_12h:>10,}  ({n_t0_12h / len(out) * 100:.1f}%)

    T0 ELIGIBILITY BY HF SUBTYPE
    ──────────────────────────────
    {textwrap.indent(subtype_str, '  ')}

    CONFIGURATION
    ─────────────
    T0 primary              : ICU admission + {t0_primary_h}h
    T0 sensitivity          : ICU admission + {t0_sens_h}h
    Decision grid           : {grid_bin_h}h bins × {n_bins} bins = {primary_horizon_h}h
    Sensitivity horizons    : {sens_horizons_h}

    OUTPUT
    ──────
    {out_path}  ({len(out):,} rows, {out.shape[1]} cols)
    """)

    report_path = reports_dir / "25a_time_zero_flags_mimic_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"  → Report: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
