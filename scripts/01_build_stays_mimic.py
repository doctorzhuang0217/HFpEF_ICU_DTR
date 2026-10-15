#!/usr/bin/env python3
"""
01_build_stays_mimic — Build MIMIC-IV ICU stays base table.

Purpose
-------
Load MIMIC-IV icustays, patients, and admissions tables.  Merge them into a
single stays-level table with demographics, timestamps, LOS, index-stay flag,
and hospital mortality.  This is the foundational table that all downstream
MIMIC pipeline steps build upon.

Inputs
------
- data/raw/mimiciv/icu/icustays.csv.gz
- data/raw/mimiciv/hosp/patients.csv.gz
- data/raw/mimiciv/hosp/admissions.csv.gz

Outputs
-------
- data/staging/01_stays_mimic.parquet
- reports/01_stays_mimic_summary.txt

Notes
-----
- This step does NOT apply HFpEF ascertainment, T0 filtering, or the
  hemodynamic-management gate.  Those come in later steps.
- "Index ICU stay" = the first ICU stay per subject_id (by intime).
  The protocol requires index ICU stays only for the primary analysis.
- Age is computed from anchor_age + (admittime.year - anchor_year).
"""

from __future__ import annotations

import sys
import textwrap
from datetime import datetime
from pathlib import Path

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
    print("01_build_stays_mimic — Build MIMIC-IV ICU stays base table")
    print("=" * 70)

    # ── load raw tables ───────────────────────────────────────────────────
    print("\n[1/5] Loading raw tables ...")

    icustays = pd.read_csv(
        mimic_root / "icu" / "icustays.csv.gz",
        parse_dates=["intime", "outtime"],
    )
    print(f"  icustays : {len(icustays):>8,} rows")

    patients = pd.read_csv(mimic_root / "hosp" / "patients.csv.gz")
    print(f"  patients : {len(patients):>8,} rows")

    admissions = pd.read_csv(
        mimic_root / "hosp" / "admissions.csv.gz",
        parse_dates=["admittime", "dischtime", "deathtime"],
    )
    print(f"  admissions: {len(admissions):>8,} rows")

    # ── merge ─────────────────────────────────────────────────────────────
    print("\n[2/5] Merging tables ...")

    stays = icustays.merge(patients, on="subject_id", how="left")
    stays = stays.merge(admissions, on=["subject_id", "hadm_id"], how="left")
    print(f"  merged stays: {len(stays):>8,} rows")

    # ── compute derived columns ───────────────────────────────────────────
    print("\n[3/5] Computing derived columns ...")

    # Age at hospital admission
    # MIMIC-IV: age = anchor_age + (year_of_admittime - anchor_year)
    stays["age"] = (
        stays["anchor_age"]
        + stays["admittime"].dt.year
        - stays["anchor_year"]
    )

    # ICU LOS in hours (more useful than days for our 2h-grid design)
    stays["icu_los_hours"] = (
        (stays["outtime"] - stays["intime"]).dt.total_seconds() / 3600.0
    )

    # Hospital LOS in hours
    stays["hosp_los_hours"] = (
        (stays["dischtime"] - stays["admittime"]).dt.total_seconds() / 3600.0
    )

    # Hospital mortality flag
    stays["hospital_mortality"] = stays["deathtime"].notna().astype(int)

    # Index ICU stay: first ICU stay per subject (by intime)
    stays = stays.sort_values(["subject_id", "intime"])
    stays["icu_stay_seq"] = stays.groupby("subject_id").cumcount() + 1
    stays["is_index_stay"] = (stays["icu_stay_seq"] == 1).astype(int)

    # ICU stay sequence within the same hospital admission
    stays["icu_stay_seq_hadm"] = (
        stays.groupby("hadm_id").cumcount() + 1
    )

    # Useful for downstream: can the stay potentially reach T0 = 6h?
    stays["icu_los_ge_6h"] = (stays["icu_los_hours"] >= 6.0).astype(int)

    # ── select and order columns ──────────────────────────────────────────
    print("\n[4/5] Selecting output columns ...")

    out_cols = [
        # identifiers
        "subject_id",
        "hadm_id",
        "stay_id",
        # demographics
        "age",
        "gender",
        # care unit
        "first_careunit",
        "last_careunit",
        # timestamps
        "intime",
        "outtime",
        "admittime",
        "dischtime",
        "deathtime",
        # LOS
        "los",           # original MIMIC column (days, float)
        "icu_los_hours",
        "hosp_los_hours",
        # outcomes
        "hospital_mortality",
        # stay sequence
        "icu_stay_seq",
        "icu_stay_seq_hadm",
        "is_index_stay",
        # convenience flags
        "icu_los_ge_6h",
        # admission context
        "admission_type",
        "admission_location",
        "discharge_location",
        "insurance",
        "marital_status",
        "race",
    ]

    # Keep only columns that actually exist (MIMIC-IV versions may vary)
    available = [c for c in out_cols if c in stays.columns]
    missing = [c for c in out_cols if c not in stays.columns]
    if missing:
        print(f"  Warning — columns not found (skipped): {missing}")

    stays_out = stays[available].copy()
    stays_out = stays_out.sort_values(["subject_id", "intime"]).reset_index(
        drop=True
    )

    # ── write output ──────────────────────────────────────────────────────
    out_path = staging_dir / "01_stays_mimic.parquet"
    stays_out.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"\n  → Written: {out_path}  ({len(stays_out):,} rows)")

    # ── summary report ────────────────────────────────────────────────────
    print("\n[5/5] Generating summary report ...")

    n_total = len(stays_out)
    n_patients = stays_out["subject_id"].nunique()
    n_admissions = stays_out["hadm_id"].nunique()
    n_index = stays_out["is_index_stay"].sum()
    n_ge6h = stays_out["icu_los_ge_6h"].sum()
    n_index_ge6h = stays_out.loc[
        (stays_out["is_index_stay"] == 1) & (stays_out["icu_los_ge_6h"] == 1)
    ].shape[0]
    n_deaths = stays_out["hospital_mortality"].sum()

    age_desc = stays_out["age"].describe()
    los_desc = stays_out["icu_los_hours"].describe()

    # Care unit distribution (top 10)
    careunit_counts = (
        stays_out["first_careunit"]
        .value_counts()
        .head(10)
        .to_string()
    )

    report = textwrap.dedent(f"""\
    ================================================================
    01_build_stays_mimic — Summary Report
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ================================================================

    Total ICU stays        : {n_total:>10,}
    Unique patients        : {n_patients:>10,}
    Unique admissions      : {n_admissions:>10,}
    Index ICU stays        : {n_index:>10,}
    ICU LOS >= 6h          : {n_ge6h:>10,}
    Index stays with >= 6h : {n_index_ge6h:>10,}
    Hospital deaths        : {n_deaths:>10,}
    Hospital mortality rate : {n_deaths / n_total * 100:>9.1f}%

    Age distribution:
    {textwrap.indent(age_desc.to_string(), '  ')}

    ICU LOS (hours) distribution:
    {textwrap.indent(los_desc.to_string(), '  ')}

    Top 10 first care units:
    {textwrap.indent(careunit_counts, '  ')}

    Output: {out_path}
    """)

    report_path = reports_dir / "01_stays_mimic_summary.txt"
    report_path.write_text(report)
    print(report)
    print(f"  → Report: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
