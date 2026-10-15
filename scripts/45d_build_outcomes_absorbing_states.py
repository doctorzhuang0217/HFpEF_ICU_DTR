#!/usr/bin/env python3
"""
45d_build_outcomes_absorbing_states — Construct outcomes with absorbing-state
markers for death/discharge and censoring flags for transfer.

Purpose
-------
Build per-stay outcome table with:
  - mortality_28d, mortality_7d (from 10a days_to_death_from_icu_admit)
  - icu_los_days (length of ICU stay for discharge timing)
  - early_discharge (discharged alive before end of 24h observation window)
  - absorbing_state_bin: first bin where death or ICU discharge occurs (NaN=survived full window)
  - absorbing_type: "death" | "discharge" | "censored" | "observed"
  - rrt_any_during_stay (from 25e)

Protocol reference: Step 45d (C2)

Inputs
------
- data/staging/25f_primary_cohort_candidate_v1_mimic.parquet
- data/staging/10a_core_icu_stays_mimic.parquet (for days_to_death, icu LOS)

Outputs
-------
- data/interim/45d_outcomes_absorb_v1_mimic.parquet
- reports/45d_outcomes_absorb_v1_summary.txt
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path

T0_HOURS = 6
BIN_HOURS = 2
N_BINS = 12
HORIZON_HOURS = 24  # post-T0


def main() -> None:
    cfg = load_config()
    interim_dir = resolve_path(cfg, "interim")
    staging_dir = resolve_path(cfg, "staging")
    reports_dir = resolve_path(cfg, "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("45d_build_outcomes_absorbing_states — Outcomes + absorbing states")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────
    print("\n[1/4] Loading data ...")
    cohort = pd.read_parquet(staging_dir / "25f_primary_cohort_candidate_v1_mimic.parquet")
    core = pd.read_parquet(staging_dir / "10a_core_icu_stays_mimic.parquet")

    # Keep gate-passed stays only (matching 40b grid population)
    cohort = cohort[cohort["hemo_gate_passed"] == True].copy()
    n_stays = cohort["stay_id"].nunique()
    print(f"  Gate-passed stays: {n_stays:,}")

    # ── Build outcome table ───────────────────────────────────────────
    print("\n[2/4] Building outcome variables ...")

    # Merge core ICU data for LOS and mortality timing
    core_cols = ["stay_id"]
    if "days_to_death_from_icu_admit" in core.columns:
        core_cols.append("days_to_death_from_icu_admit")
    # 10a outputs icu_los_hours (hours) and los (fractional days from MIMIC);
    # convert to days downstream.
    if "icu_los_hours" in core.columns:
        core_cols.append("icu_los_hours")
    elif "los" in core.columns:
        core_cols.append("los")

    out = cohort[["stay_id"]].drop_duplicates().copy()
    out = out.merge(core[core_cols].drop_duplicates(subset="stay_id"),
                    on="stay_id", how="left")

    # ── mortality_28d ─────────────────────────────────────────────────
    # From cohort if already computed in 10a/20f
    if "mortality_28d" in cohort.columns:
        mort28 = cohort[["stay_id", "mortality_28d"]].drop_duplicates(subset="stay_id")
        out = out.merge(mort28, on="stay_id", how="left")
    elif "days_to_death_from_icu_admit" in out.columns:
        dtd = out["days_to_death_from_icu_admit"]
        out["mortality_28d"] = ((dtd >= 0) & (dtd <= 28)).astype(int)
    else:
        # Fallback: use hospital_mortality
        if "hospital_mortality" in cohort.columns:
            hm = cohort[["stay_id", "hospital_mortality"]].drop_duplicates(subset="stay_id")
            out = out.merge(hm, on="stay_id", how="left")
            out["mortality_28d"] = out["hospital_mortality"].fillna(0).astype(int)

    out["mortality_28d"] = out["mortality_28d"].fillna(0).astype(int)

    # ── mortality_7d ──────────────────────────────────────────────────
    if "days_to_death_from_icu_admit" in out.columns:
        dtd = out["days_to_death_from_icu_admit"]
        out["mortality_7d"] = ((dtd >= 0) & (dtd <= 7)).astype(int)
    else:
        out["mortality_7d"] = 0
    out["mortality_7d"] = out["mortality_7d"].fillna(0).astype(int)

    # ── ICU LOS ───────────────────────────────────────────────────────
    if "icu_los_hours" in out.columns:
        out["icu_los_days"] = out["icu_los_hours"].astype(float) / 24.0
    elif "los" in out.columns:
        out["icu_los_days"] = out["los"].astype(float)
    else:
        out["icu_los_days"] = np.nan
        print("  WARNING: no ICU LOS column found — discharge absorbing states will be NaN")

    # ── Absorbing state determination ─────────────────────────────────
    print("\n[3/4] Computing absorbing states ...")
    # Observation window: T0 to T0 + 24h = 6h to 30h post-admission
    # In days: T0 = 0.25d, end = 1.25d
    t0_days = T0_HOURS / 24.0
    end_window_days = (T0_HOURS + HORIZON_HOURS) / 24.0

    absorb_type = []
    absorb_bin = []

    for _, row in out.iterrows():
        dtd = row.get("days_to_death_from_icu_admit", np.nan)
        los = row.get("icu_los_days", np.nan)

        # Death within observation window
        if pd.notna(dtd) and t0_days <= dtd <= end_window_days:
            # Which bin? (dtd - t0_days) in days → hours → bin index
            hours_from_t0 = (dtd - t0_days) * 24
            bin_idx = min(int(hours_from_t0 / BIN_HOURS), N_BINS - 1)
            absorb_type.append("death")
            absorb_bin.append(bin_idx)
        # Discharge alive within window
        elif pd.notna(los) and los < end_window_days and (pd.isna(dtd) or dtd > end_window_days):
            if los >= t0_days:
                hours_from_t0 = (los - t0_days) * 24
                bin_idx = min(int(hours_from_t0 / BIN_HOURS), N_BINS - 1)
                absorb_type.append("discharge")
                absorb_bin.append(bin_idx)
            else:
                # Should not happen (discharged before T0 = excluded)
                absorb_type.append("censored")
                absorb_bin.append(np.nan)
        else:
            # Survived full observation window in ICU
            absorb_type.append("observed")
            absorb_bin.append(np.nan)

    out["absorbing_type"] = absorb_type
    out["absorbing_state_bin"] = absorb_bin
    out["absorbing_state_bin"] = pd.to_numeric(out["absorbing_state_bin"], errors="coerce")

    # Early discharge flag
    out["early_discharge"] = (out["absorbing_type"] == "discharge").astype(int)

    # ── RRT flag from 25e (if present in cohort) ──────────────────────
    for rrt_col in ["rrt_crrt_any_during_stay", "has_rrt"]:
        if rrt_col in cohort.columns:
            rrt = cohort[["stay_id", rrt_col]].drop_duplicates(subset="stay_id")
            out = out.merge(rrt, on="stay_id", how="left")
            out[rrt_col] = out[rrt_col].fillna(0).astype(int)

    # ── Clean up intermediate columns ─────────────────────────────────
    drop_cols = ["icu_los_hours", "los", "days_to_death_from_icu_admit", "hospital_mortality"]
    for c in drop_cols:
        if c in out.columns and c not in ("mortality_28d", "mortality_7d"):
            out = out.drop(columns=[c], errors="ignore")

    # ── Save ──────────────────────────────────────────────────────────
    out_path = interim_dir / "45d_outcomes_absorb_v1_mimic.parquet"
    out.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"  Saved: {out_path}  ({len(out):,} rows, {out.shape[1]} cols)")

    # ── Summary report ────────────────────────────────────────────────
    print("\n[4/4] Generating summary ...")
    absorb_dist = out["absorbing_type"].value_counts()
    mort28_rate = out["mortality_28d"].mean() * 100
    mort7_rate = out["mortality_7d"].mean() * 100
    early_disc_rate = out["early_discharge"].mean() * 100

    report_lines = [
        "=" * 72,
        "45d — Outcomes & Absorbing States Summary (MIMIC-IV)",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Stays: {n_stays:,}",
        "",
        "OUTCOME RATES",
        "──────────────",
        f"  28-day mortality: {out['mortality_28d'].sum():,} / {n_stays:,} ({mort28_rate:.1f}%)",
        f"   7-day mortality: {out['mortality_7d'].sum():,} / {n_stays:,} ({mort7_rate:.1f}%)",
        f"  Early discharge:  {out['early_discharge'].sum():,} / {n_stays:,} ({early_disc_rate:.1f}%)",
        "",
        "ABSORBING STATE DISTRIBUTION",
        "─────────────────────────────",
    ]
    for atype, cnt in absorb_dist.items():
        pct = cnt / n_stays * 100
        report_lines.append(f"  {atype:<12}: {cnt:>8,} ({pct:5.1f}%)")

    if out["absorbing_state_bin"].notna().any():
        bin_dist = out.loc[out["absorbing_state_bin"].notna(), "absorbing_state_bin"]
        report_lines.extend([
            "",
            "ABSORBING STATE BIN DISTRIBUTION (among absorbed)",
            "──────────────────────────────────────────────────",
            f"  Mean bin: {bin_dist.mean():.1f}",
            f"  Median bin: {bin_dist.median():.0f}",
        ])

    # ICU LOS summary
    if "icu_los_days" in out.columns and out["icu_los_days"].notna().any():
        los = out["icu_los_days"]
        report_lines.extend([
            "",
            "ICU LENGTH OF STAY",
            "───────────────────",
            f"  Mean:   {los.mean():.1f} days",
            f"  Median: {los.median():.1f} days",
            f"  P25-P75: {los.quantile(0.25):.1f} - {los.quantile(0.75):.1f} days",
        ])

    report = "\n".join(report_lines) + "\n"
    report_path = reports_dir / "45d_outcomes_absorb_v1_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {report_path}")
    print(report)
    print("[45d] Done.")


if __name__ == "__main__":
    main()
