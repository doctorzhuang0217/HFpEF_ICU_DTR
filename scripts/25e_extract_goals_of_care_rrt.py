#!/usr/bin/env python3
"""
25e_extract_goals_of_care_rrt — Goals-of-care and RRT/CRRT flags.

Purpose
-------
Attach pre-T0 goals-of-care signals (DNR/DNI, comfort care) and RRT/CRRT
process flags for each index stay. These are time-varying sensitivity
covariates, NOT exclusions (hard exclusions already applied in 20e).

Patients with comfort-care conversion post-T0 are flagged but NOT excluded
from the primary analysis; they feed sensitivity analyses (85f, 85h).

Inputs
------
- data/staging/25a_time_zero_flags_mimic.parquet
- data/staging/10h_vent_indicators_mimic.parquet
- data/staging/10h_procedures_mimic.parquet

Outputs
-------
- data/staging/25e_goals_of_care_rrt_flags_mimic.parquet
- reports/25e_goals_of_care_rrt_mimic_summary.txt
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

    print("=" * 70)
    print("25e_extract_goals_of_care_rrt — Goals-of-care & RRT flags")
    print("=" * 70)

    # ── load inputs ──────────────────────────────────────────────────
    print("\n[1/4] Loading inputs ...")
    cohort = pd.read_parquet(
        staging_dir / "25a_time_zero_flags_mimic.parquet",
        columns=["stay_id", "subject_id", "hadm_id", "hf_subtype",
                 "comparator_group", "intime", "time_zero_6h"],
    )
    stay_ids = set(cohort["stay_id"])
    n_index = len(cohort)
    print(f"  Index stays: {n_index:,}")

    # ── code status from 10h ─────────────────────────────────────────
    print("\n[2/4] Extracting code-status flags ...")
    vent_ind = pd.read_parquet(staging_dir / "10h_vent_indicators_mimic.parquet")
    code_status = vent_ind[
        (vent_ind["indicator_type"] == "code_status")
        & vent_ind["stay_id"].isin(stay_ids)
    ].copy()

    # Join T0 timestamp
    code_status = code_status.merge(
        cohort[["stay_id", "time_zero_6h"]], on="stay_id", how="inner"
    )

    # Pre-T0 code status
    pre_t0_cs = code_status[code_status["charttime"] < code_status["time_zero_6h"]].copy()

    # Patterns
    comfort_pattern = r"(?i)comfort|CMO|withdraw|palliat"
    dnr_pattern = r"(?i)DNR|do not resuscitate"
    dni_pattern = r"(?i)DNI|do not intubate"

    comfort_stays = set()
    dnr_stays = set()
    dni_stays = set()
    if len(pre_t0_cs) > 0:
        comfort_stays = set(
            pre_t0_cs.loc[pre_t0_cs["value"].str.contains(comfort_pattern, na=False), "stay_id"].unique()
        )
        dnr_stays = set(
            pre_t0_cs.loc[pre_t0_cs["value"].str.contains(dnr_pattern, na=False), "stay_id"].unique()
        )
        dni_stays = set(
            pre_t0_cs.loc[pre_t0_cs["value"].str.contains(dni_pattern, na=False), "stay_id"].unique()
        )

    print(f"  DNR pre-T0:          {len(dnr_stays):,}")
    print(f"  DNI pre-T0:          {len(dni_stays):,}")
    print(f"  Comfort care pre-T0: {len(comfort_stays):,}")

    # ── RRT/CRRT from 10h procedures ─────────────────────────────────
    print("\n[3/4] Extracting RRT/CRRT flags ...")
    procs = pd.read_parquet(staging_dir / "10h_procedures_mimic.parquet")
    # 10h uses procedure_type == "rrt_crrt"
    rrt = procs[
        (procs["procedure_type"] == "rrt_crrt")
        & procs["stay_id"].isin(stay_ids)
    ].copy()

    # Join T0 for pre/post classification
    rrt = rrt.merge(
        cohort[["stay_id", "intime", "time_zero_6h"]], on="stay_id", how="inner"
    )

    # Determine time column (procedureevents uses starttime)
    time_col = "starttime" if "starttime" in rrt.columns else "charttime"
    if time_col in rrt.columns and len(rrt) > 0:
        rrt["hours_from_intime"] = (
            (rrt[time_col] - rrt["intime"]).dt.total_seconds() / 3600.0
        )
        rrt_pre_t0 = rrt[rrt[time_col] < rrt["time_zero_6h"]]
    else:
        rrt_pre_t0 = pd.DataFrame(columns=rrt.columns)

    rrt_pre_t0_stays = set(rrt_pre_t0["stay_id"].unique()) if len(rrt_pre_t0) > 0 else set()
    rrt_any_stays = set(rrt["stay_id"].unique()) if len(rrt) > 0 else set()
    print(f"  RRT/CRRT any during stay:  {len(rrt_any_stays):,}")
    print(f"  RRT/CRRT initiated pre-T0: {len(rrt_pre_t0_stays):,}")

    # ── build output ─────────────────────────────────────────────────
    print("\n[4/4] Building output flags ...")
    out = cohort[["stay_id", "subject_id", "hadm_id",
                  "hf_subtype", "comparator_group"]].copy()

    out["dnr_flag_pre_t0"] = out["stay_id"].isin(dnr_stays)
    out["dni_flag_pre_t0"] = out["stay_id"].isin(dni_stays)
    out["comfort_care_pre_t0"] = out["stay_id"].isin(comfort_stays)

    # Code status summary
    out["code_status_pre_t0"] = "full"
    out.loc[out["dnr_flag_pre_t0"], "code_status_pre_t0"] = "dnr"
    out.loc[out["dni_flag_pre_t0"], "code_status_pre_t0"] = "dni"
    out.loc[out["comfort_care_pre_t0"], "code_status_pre_t0"] = "comfort"

    out["rrt_crrt_initiated_pre_t0"] = out["stay_id"].isin(rrt_pre_t0_stays)
    out["rrt_crrt_any_during_stay"] = out["stay_id"].isin(rrt_any_stays)

    # ── write output ─────────────────────────────────────────────────
    out = out.sort_values("stay_id").reset_index(drop=True)
    out_path = staging_dir / "25e_goals_of_care_rrt_flags_mimic.parquet"
    out.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"\n  → {out_path}  ({len(out):,} rows, {out.shape[1]} cols)")

    # ── summary report ───────────────────────────────────────────────
    cs_dist = out["code_status_pre_t0"].value_counts()

    report = textwrap.dedent(f"""\
    ================================================================
    25e_extract_goals_of_care_rrt — Summary Report
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ================================================================

    CODE STATUS PRE-T0
    ───────────────────
    {textwrap.indent(cs_dist.to_string(), '  ')}

    DNR pre-T0              : {out['dnr_flag_pre_t0'].sum():>10,}  ({out['dnr_flag_pre_t0'].mean() * 100:.1f}%)
    DNI pre-T0              : {out['dni_flag_pre_t0'].sum():>10,}  ({out['dni_flag_pre_t0'].mean() * 100:.1f}%)
    Comfort care pre-T0     : {out['comfort_care_pre_t0'].sum():>10,}  ({out['comfort_care_pre_t0'].mean() * 100:.1f}%)

    RRT/CRRT
    ─────────
    Any during stay         : {out['rrt_crrt_any_during_stay'].sum():>10,}  ({out['rrt_crrt_any_during_stay'].mean() * 100:.1f}%)
    Initiated pre-T0        : {out['rrt_crrt_initiated_pre_t0'].sum():>10,}  ({out['rrt_crrt_initiated_pre_t0'].mean() * 100:.1f}%)

    OUTPUT
    ──────
    {out_path}  ({len(out):,} rows, {out.shape[1]} cols)
    """)

    report_path = reports_dir / "25e_goals_of_care_rrt_mimic_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"  → Report: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
