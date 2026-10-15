#!/usr/bin/env python3
"""
10c_normalize_timestamps_mimic — Normalize timestamps & define time landmarks.

Purpose
-------
Build a per-stay time reference table with:
  - Time-zero landmarks (T0 = intime + 6h primary, +12h sensitivity)
  - Pre-T0 window boundaries
  - Outcome-horizon endpoints (24h, 48h, 72h)
  - Reconciled death timestamp (deathtime vs dod)

All downstream longitudinal extraction steps (10d–10h) and eligibility
gates (25a, 25b) reference this table for consistent windowing.

Inputs
------
- data/staging/10a_core_icu_stays_mimic.parquet

Outputs
-------
- data/staging/10c_timestamps_mimic.parquet
- reports/10c_timestamps_mimic_summary.txt
"""

from __future__ import annotations

import sys
import textwrap
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── project imports ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path


def main() -> None:
    cfg = load_config()
    staging_dir = resolve_path(cfg, "staging")
    reports_dir = resolve_path(cfg, "reports")
    staging_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("10c_normalize_timestamps_mimic — Time landmarks & windows")
    print("=" * 70)

    # ── load 10a backbone ──────────────────────────────────────────────
    print("\n[1/4] Loading 10a backbone ...")
    cols = [
        "subject_id", "hadm_id", "stay_id",
        "intime", "outtime", "admittime", "dischtime",
        "deathtime", "dod",
        "icu_los_hours", "icu_los_ge_6h",
    ]
    stays = pd.read_parquet(
        staging_dir / "10a_core_icu_stays_mimic.parquet",
        columns=cols,
    )
    n_total = len(stays)
    print(f"  Loaded: {n_total:,} rows")

    # ── derive time landmarks ──────────────────────────────────────────
    print("\n[2/4] Deriving time landmarks ...")

    # Time-zero: primary (6h) and sensitivity (12h)
    stays["time_zero_6h"] = stays["intime"] + pd.Timedelta(hours=6)
    stays["time_zero_12h"] = stays["intime"] + pd.Timedelta(hours=12)

    # Pre-T0 window (0 to 6h post-ICU admission)
    stays["pre_t0_start"] = stays["intime"]
    stays["pre_t0_end"] = stays["time_zero_6h"]

    # Outcome horizons (from ICU admission)
    stays["horizon_24h"] = stays["intime"] + pd.Timedelta(hours=24)
    stays["horizon_48h"] = stays["intime"] + pd.Timedelta(hours=48)
    stays["horizon_72h"] = stays["intime"] + pd.Timedelta(hours=72)

    # Time from hospital admission to ICU admission (hours)
    stays["hosp_to_icu_hours"] = (
        (stays["intime"] - stays["admittime"]).dt.total_seconds() / 3600.0
    )

    # Reconciled death timestamp: prefer deathtime (has time), fallback to dod
    stays["death_timestamp"] = stays["deathtime"].fillna(
        pd.to_datetime(stays["dod"])
    )

    # Did patient die or discharge before T0?
    stays["died_before_t0_6h"] = (
        stays["death_timestamp"].notna()
        & (stays["death_timestamp"] < stays["time_zero_6h"])
    ).astype(int)

    stays["discharged_before_t0_6h"] = (
        stays["outtime"] < stays["time_zero_6h"]
    ).astype(int)

    # T0-eligible: stayed at least 6h AND did not die before T0
    stays["t0_6h_eligible"] = (
        (stays["icu_los_ge_6h"] == 1)
        & (stays["died_before_t0_6h"] == 0)
    ).astype(int)

    stays["t0_12h_eligible"] = (
        (stays["icu_los_hours"] >= 12.0)
        & (
            stays["death_timestamp"].isna()
            | (stays["death_timestamp"] >= stays["time_zero_12h"])
        )
    ).astype(int)

    # ── select output columns ─────────────────────────────────────────
    print("\n[3/4] Writing output ...")

    out_cols = [
        "subject_id", "hadm_id", "stay_id",
        # raw timestamps (carried for convenience)
        "intime", "outtime", "admittime", "dischtime",
        "death_timestamp",
        # time-zero landmarks
        "time_zero_6h", "time_zero_12h",
        # pre-T0 window
        "pre_t0_start", "pre_t0_end",
        # outcome horizons
        "horizon_24h", "horizon_48h", "horizon_72h",
        # derived intervals
        "hosp_to_icu_hours", "icu_los_hours",
        # eligibility flags
        "died_before_t0_6h", "discharged_before_t0_6h",
        "t0_6h_eligible", "t0_12h_eligible",
    ]

    ts_out = stays[out_cols].copy()
    ts_out = ts_out.sort_values(["subject_id", "intime"]).reset_index(drop=True)

    out_path = staging_dir / "10c_timestamps_mimic.parquet"
    ts_out.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"  → {out_path}  ({len(ts_out):,} rows)")

    # ── summary report ─────────────────────────────────────────────────
    print("\n[4/4] Generating summary report ...")

    n_t0_6h = int(ts_out["t0_6h_eligible"].sum())
    n_t0_12h = int(ts_out["t0_12h_eligible"].sum())
    n_died_pre_t0 = int(ts_out["died_before_t0_6h"].sum())
    n_disch_pre_t0 = int(ts_out["discharged_before_t0_6h"].sum())
    hosp_to_icu = ts_out["hosp_to_icu_hours"].describe().round(1)

    report = textwrap.dedent(f"""\
    ================================================================
    10c_normalize_timestamps_mimic — Summary Report
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ================================================================

    COHORT
    ──────
    Total ICU stays          : {n_total:>10,}

    T0 ELIGIBILITY (primary: 6h)
    ─────────────────────────────
    T0-6h eligible           : {n_t0_6h:>10,}  ({n_t0_6h / n_total * 100:.1f}%)
    Died before T0-6h        : {n_died_pre_t0:>10,}  ({n_died_pre_t0 / n_total * 100:.1f}%)
    Discharged before T0-6h  : {n_disch_pre_t0:>10,}  ({n_disch_pre_t0 / n_total * 100:.1f}%)

    T0 ELIGIBILITY (sensitivity: 12h)
    ──────────────────────────────────
    T0-12h eligible          : {n_t0_12h:>10,}  ({n_t0_12h / n_total * 100:.1f}%)

    HOSPITAL-TO-ICU INTERVAL (hours)
    ─────────────────────────────────
    {textwrap.indent(hosp_to_icu.to_string(), '  ')}

    OUTPUT
    ──────
    {out_path}  ({len(ts_out):,} rows, {ts_out.shape[1]} cols)

    SCHEMA
    ──────
    {chr(10).join('  - ' + c for c in out_cols)}
    """)

    report_path = reports_dir / "10c_timestamps_mimic_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"  → Report: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
