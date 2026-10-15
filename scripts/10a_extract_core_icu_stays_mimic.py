#!/usr/bin/env python3
"""
10a_extract_core_icu_stays_mimic — Extract validated ICU-stay backbone.

Purpose
-------
Take the merged stays table from 01_build_stays_mimic, add outcome-relevant
fields (dod for 28-day mortality), run data-quality validation, and export
the canonical ICU backbone for all downstream steps.

This step does NOT apply clinical eligibility gates (HFpEF ascertainment,
T0 filtering, exclusion criteria). Those come in Steps 20x–25x.

Inputs
------
- data/staging/01_stays_mimic.parquet  (from 01_build_stays_mimic)
- data/raw/mimiciv/hosp/patients.csv.gz  (for dod field)

Outputs
-------
- data/staging/10a_core_icu_stays_mimic.parquet
- reports/10a_core_icu_stays_mimic_summary.txt
"""

from __future__ import annotations

import sys
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── project imports ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, dataset_entrypoint, resolve_path


def main() -> None:
    cfg = load_config()
    mimic_root = dataset_entrypoint(cfg, "mimic")
    staging_dir = resolve_path(cfg, "staging")
    reports_dir = resolve_path(cfg, "reports")
    staging_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("10a_extract_core_icu_stays_mimic — Validated ICU-stay backbone")
    print("=" * 70)

    # ── load 01 output ─────────────────────────────────────────────────
    print("\n[1/5] Loading 01_stays_mimic ...")
    input_path = staging_dir / "01_stays_mimic.parquet"
    stays = pd.read_parquet(input_path)
    n_input = len(stays)
    print(f"  Loaded: {n_input:,} rows")

    # ── add dod for 28-day mortality ───────────────────────────────────
    print("\n[2/5] Adding dod (date of death) for outcome calculation ...")
    patients = pd.read_csv(mimic_root / "hosp" / "patients.csv.gz",
                           usecols=["subject_id", "dod"])
    patients["dod"] = pd.to_datetime(patients["dod"], errors="coerce")
    stays = stays.merge(patients, on="subject_id", how="left")

    # MIMIC dod is date-level; compare against ICU admission date to avoid
    # same-day deaths being misclassified as pre-ICU deaths.
    intime_date = stays["intime"].dt.floor("D")
    stays["days_to_death_from_icu_admit"] = (
        stays["dod"] - intime_date
    ).dt.total_seconds() / 86400.0

    # 28-day mortality: death within 28 days of ICU admission date
    stays["mortality_28d"] = (
        stays["days_to_death_from_icu_admit"].between(0, 28, inclusive="both")
    ).astype(int)

    # 90-day mortality (secondary endpoint)
    stays["mortality_90d"] = (
        stays["days_to_death_from_icu_admit"].between(0, 90, inclusive="both")
    ).astype(int)

    print(f"  28-day mortality: {stays['mortality_28d'].sum():,} / {n_input:,} "
          f"({stays['mortality_28d'].mean() * 100:.1f}%)")
    print(f"  90-day mortality: {stays['mortality_90d'].sum():,} / {n_input:,} "
          f"({stays['mortality_90d'].mean() * 100:.1f}%)")

    # ── data quality validation ────────────────────────────────────────
    print("\n[3/5] Running data-quality checks ...")
    qc_issues = []

    # Check 1: null identifiers
    for col in ["subject_id", "hadm_id", "stay_id"]:
        n_null = stays[col].isna().sum()
        if n_null > 0:
            qc_issues.append(f"  FAIL: {col} has {n_null:,} null values")
        else:
            print(f"  OK: {col} — no nulls")

    # Check 2: duplicate stay_ids
    n_dup = stays["stay_id"].duplicated().sum()
    if n_dup > 0:
        qc_issues.append(f"  FAIL: {n_dup:,} duplicate stay_id values")
    else:
        print(f"  OK: stay_id — no duplicates ({stays['stay_id'].nunique():,} unique)")

    # Check 3: timestamp order (intime < outtime)
    bad_ts = stays["intime"] >= stays["outtime"]
    n_bad_ts = bad_ts.sum()
    if n_bad_ts > 0:
        qc_issues.append(f"  WARN: {n_bad_ts:,} rows with intime >= outtime")
    else:
        print(f"  OK: intime < outtime for all rows")

    # Check 4: hospital admission before ICU admission
    bad_admit = stays["admittime"] > stays["intime"]
    n_bad_admit = bad_admit.sum()
    if n_bad_admit > 0:
        # This can happen legitimately (e.g., direct ICU admission recorded slightly before admin)
        print(f"  INFO: {n_bad_admit:,} rows with admittime > intime "
              f"(may be legitimate; flagging only)")
    else:
        print(f"  OK: admittime <= intime for all rows")

    # Check 5: negative or extreme age
    n_age_neg = (stays["age"] < 0).sum()
    n_age_extreme = (stays["age"] > 120).sum()
    if n_age_neg > 0:
        qc_issues.append(f"  FAIL: {n_age_neg:,} rows with age < 0")
    if n_age_extreme > 0:
        print(f"  INFO: {n_age_extreme:,} rows with age > 120 (MIMIC age-shifting)")
    print(f"  OK: age range [{stays['age'].min()}, {stays['age'].max()}]")

    # Check 6: negative ICU LOS
    n_neg_los = (stays["icu_los_hours"] < 0).sum()
    if n_neg_los > 0:
        qc_issues.append(f"  FAIL: {n_neg_los:,} rows with negative icu_los_hours")
    else:
        print(f"  OK: no negative icu_los_hours")

    # Check 7: mortality consistency (hospital_mortality vs dod)
    hosp_mort_no_dod = (
        (stays["hospital_mortality"] == 1) & stays["dod"].isna()
    ).sum()
    if hosp_mort_no_dod > 0:
        print(f"  INFO: {hosp_mort_no_dod:,} rows with hospital_mortality=1 but dod missing "
              f"(deathtime vs dod source mismatch is expected)")

    # Check 8: endpoint consistency (90-day should be a superset of 28-day)
    n_bad_horizon = ((stays["mortality_28d"] == 1) & (stays["mortality_90d"] == 0)).sum()
    if n_bad_horizon > 0:
        qc_issues.append(
            f"  FAIL: {n_bad_horizon:,} rows with mortality_28d=1 but mortality_90d=0"
        )
    else:
        print("  OK: mortality horizon consistency (28d ⊆ 90d)")

    # Check 9: deaths recorded before ICU-admit date
    n_pre_icu_dod = (stays["days_to_death_from_icu_admit"] < 0).sum()
    if n_pre_icu_dod > 0:
        print(f"  INFO: {n_pre_icu_dod:,} rows with dod before ICU admit date")
    else:
        print("  OK: no dod before ICU admit date")

    if qc_issues:
        print("\n  QC ISSUES FOUND:")
        for issue in qc_issues:
            print(f"    {issue}")
    else:
        print("\n  All QC checks passed.")

    # ── flag rows but do NOT drop (leave for downstream gates) ─────────
    stays["qc_valid_timestamps"] = (
        (~bad_ts) & stays["intime"].notna() & stays["outtime"].notna()
    ).astype(int)
    stays["qc_valid_ids"] = (
        stays["subject_id"].notna()
        & stays["hadm_id"].notna()
        & stays["stay_id"].notna()
    ).astype(int)

    # ── select and order columns ───────────────────────────────────────
    print("\n[4/5] Selecting output columns ...")

    out_cols = [
        # identifiers
        "subject_id", "hadm_id", "stay_id",
        # demographics
        "age", "gender",
        # care unit
        "first_careunit", "last_careunit",
        # timestamps
        "intime", "outtime", "admittime", "dischtime", "deathtime",
        # date of death (for 28d/90d mortality)
        "dod",
        # LOS
        "los", "icu_los_hours", "hosp_los_hours",
        # outcomes
        "hospital_mortality", "mortality_28d", "mortality_90d",
        "days_to_death_from_icu_admit",
        # stay sequence
        "icu_stay_seq", "icu_stay_seq_hadm", "is_index_stay",
        # convenience flags
        "icu_los_ge_6h",
        # QC flags
        "qc_valid_timestamps", "qc_valid_ids",
        # admission context
        "admission_type", "admission_location", "discharge_location",
        "insurance", "marital_status", "race",
    ]

    available = [c for c in out_cols if c in stays.columns]
    missing_cols = [c for c in out_cols if c not in stays.columns]
    if missing_cols:
        print(f"  Warning — columns not found (skipped): {missing_cols}")

    stays_out = stays[available].copy()
    stays_out = stays_out.sort_values(["subject_id", "intime"]).reset_index(drop=True)

    # ── write output ───────────────────────────────────────────────────
    out_path = staging_dir / "10a_core_icu_stays_mimic.parquet"
    stays_out.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"\n  → Written: {out_path}  ({len(stays_out):,} rows)")

    # ── summary report ─────────────────────────────────────────────────
    print("\n[5/5] Generating summary report ...")

    n_total = len(stays_out)
    n_patients = stays_out["subject_id"].nunique()
    n_admissions = stays_out["hadm_id"].nunique()
    n_index = int(stays_out["is_index_stay"].sum())
    n_ge6h = int(stays_out["icu_los_ge_6h"].sum())
    n_index_ge6h = int(
        stays_out.loc[
            (stays_out["is_index_stay"] == 1) & (stays_out["icu_los_ge_6h"] == 1)
        ].shape[0]
    )
    n_hosp_deaths = int(stays_out["hospital_mortality"].sum())
    n_28d_deaths = int(stays_out["mortality_28d"].sum())
    n_90d_deaths = int(stays_out["mortality_90d"].sum())
    n_qc_good = (
        ((stays_out["qc_valid_timestamps"] == 1)
         & (stays_out["qc_valid_ids"] == 1)).sum()
        if "qc_valid_timestamps" in stays_out.columns else n_total
    )

    age_desc = stays_out["age"].describe().round(1)
    los_desc = stays_out["icu_los_hours"].describe().round(1)

    # Care unit distribution
    careunit_counts = stays_out["first_careunit"].value_counts().to_string()

    # Gender distribution
    gender_counts = stays_out["gender"].value_counts().to_string()

    # Admission type distribution
    admit_type_counts = stays_out["admission_type"].value_counts().head(10).to_string()

    report = textwrap.dedent(f"""\
    ================================================================
    10a_extract_core_icu_stays_mimic — Summary Report
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Input: {input_path}
    Output: {out_path}
    ================================================================

    COHORT OVERVIEW
    ───────────────
    Total ICU stays        : {n_total:>10,}
    Unique patients        : {n_patients:>10,}
    Unique admissions      : {n_admissions:>10,}
    Index ICU stays        : {n_index:>10,}
    ICU LOS >= 6h          : {n_ge6h:>10,}
    Index stays with >= 6h : {n_index_ge6h:>10,}
    QC-valid rows          : {n_qc_good:>10,}

    MORTALITY
    ─────────
    Hospital mortality     : {n_hosp_deaths:>10,}  ({n_hosp_deaths / n_total * 100:.1f}%)
    28-day mortality        : {n_28d_deaths:>10,}  ({n_28d_deaths / n_total * 100:.1f}%)
    90-day mortality        : {n_90d_deaths:>10,}  ({n_90d_deaths / n_total * 100:.1f}%)

    AGE DISTRIBUTION
    ─────────────────
    {textwrap.indent(age_desc.to_string(), '  ')}

    ICU LOS (HOURS) DISTRIBUTION
    ─────────────────────────────
    {textwrap.indent(los_desc.to_string(), '  ')}

    GENDER
    ──────
    {textwrap.indent(gender_counts, '  ')}

    FIRST CARE UNIT
    ────────────────
    {textwrap.indent(careunit_counts, '  ')}

    ADMISSION TYPE (top 10)
    ────────────────────────
    {textwrap.indent(admit_type_counts, '  ')}

    DATA QUALITY
    ─────────────
    QC issues: {len(qc_issues)} found
    {chr(10).join('  ' + i for i in qc_issues) if qc_issues else '  None'}

    OUTPUT SCHEMA ({len(available)} columns)
    ──────────────
    {chr(10).join('  - ' + c for c in available)}
    """)

    report_path = reports_dir / "10a_core_icu_stays_mimic_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"  → Report: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
