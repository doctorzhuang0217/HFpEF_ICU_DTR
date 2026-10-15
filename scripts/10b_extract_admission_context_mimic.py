#!/usr/bin/env python3
"""
10b_extract_admission_context_mimic — Extract ICD diagnosis & procedure codes.

Purpose
-------
Extract ICD diagnosis codes and procedure codes linked to each hospital
admission.  This is the prerequisite for:
  - 20a: HF candidate flag construction (ICD-based HF identification)
  - 20b: EF typing (OMR-based ejection fraction)
  - 20e: hard exclusion flags (ECMO, LVAD, transplant via procedure codes)

Step 10a already carries demographics + admission context (age, gender,
admission_type, race, etc.).  This step adds the *diagnostic and procedural*
layer that 10a deliberately omitted.

Inputs
------
- data/staging/10a_core_icu_stays_mimic.parquet  (stay backbone, for hadm_id list)
- data/raw/mimiciv/hosp/diagnoses_icd.csv.gz
- data/raw/mimiciv/hosp/d_icd_diagnoses.csv.gz
- data/raw/mimiciv/hosp/procedures_icd.csv.gz
- data/raw/mimiciv/hosp/d_icd_procedures.csv.gz

Outputs
-------
- data/staging/10b_diagnoses_mimic.parquet   (long: one row per diagnosis per hadm)
- data/staging/10b_procedures_mimic.parquet   (long: one row per procedure per hadm)
- reports/10b_admission_context_mimic_summary.txt
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
    print("10b_extract_admission_context_mimic — ICD diagnoses & procedures")
    print("=" * 70)

    # ── load 10a backbone (for hadm_id scope) ──────────────────────────
    print("\n[1/5] Loading 10a backbone ...")
    stays = pd.read_parquet(
        staging_dir / "10a_core_icu_stays_mimic.parquet",
        columns=["subject_id", "hadm_id", "stay_id"],
    )
    cohort_hadm_ids = set(stays["hadm_id"].unique())
    print(f"  Cohort: {len(cohort_hadm_ids):,} unique hadm_ids")

    # ── diagnoses ──────────────────────────────────────────────────────
    print("\n[2/5] Loading diagnoses_icd ...")
    dx = pd.read_csv(mimic_root / "hosp" / "diagnoses_icd.csv.gz")
    n_dx_total = len(dx)
    print(f"  Raw diagnoses: {n_dx_total:,} rows")

    # Scope to cohort admissions
    dx = dx[dx["hadm_id"].isin(cohort_hadm_ids)].copy()
    print(f"  After cohort filter: {len(dx):,} rows")

    # Attach descriptions
    d_dx = pd.read_csv(mimic_root / "hosp" / "d_icd_diagnoses.csv.gz")
    dx = dx.merge(
        d_dx[["icd_code", "icd_version", "long_title"]],
        on=["icd_code", "icd_version"],
        how="left",
    )
    dx.rename(columns={"long_title": "icd_title"}, inplace=True)

    # Sort for deterministic output
    dx = dx.sort_values(
        ["subject_id", "hadm_id", "seq_num"]
    ).reset_index(drop=True)

    # ── procedures ─────────────────────────────────────────────────────
    print("\n[3/5] Loading procedures_icd ...")
    px = pd.read_csv(mimic_root / "hosp" / "procedures_icd.csv.gz")
    n_px_total = len(px)
    print(f"  Raw procedures: {n_px_total:,} rows")

    px = px[px["hadm_id"].isin(cohort_hadm_ids)].copy()
    print(f"  After cohort filter: {len(px):,} rows")

    # Parse chartdate
    px["chartdate"] = pd.to_datetime(px["chartdate"], errors="coerce")

    # Attach descriptions
    d_px = pd.read_csv(mimic_root / "hosp" / "d_icd_procedures.csv.gz")
    px = px.merge(
        d_px[["icd_code", "icd_version", "long_title"]],
        on=["icd_code", "icd_version"],
        how="left",
    )
    px.rename(columns={"long_title": "icd_title"}, inplace=True)

    px = px.sort_values(
        ["subject_id", "hadm_id", "seq_num"]
    ).reset_index(drop=True)

    # ── write outputs ──────────────────────────────────────────────────
    print("\n[4/5] Writing outputs ...")

    dx_path = staging_dir / "10b_diagnoses_mimic.parquet"
    dx.to_parquet(dx_path, index=False, engine="pyarrow")
    print(f"  → {dx_path}  ({len(dx):,} rows)")

    px_path = staging_dir / "10b_procedures_mimic.parquet"
    px.to_parquet(px_path, index=False, engine="pyarrow")
    print(f"  → {px_path}  ({len(px):,} rows)")

    # ── summary report ─────────────────────────────────────────────────
    print("\n[5/5] Generating summary report ...")

    n_dx_hadm = dx["hadm_id"].nunique()
    n_px_hadm = px["hadm_id"].nunique()
    hadm_no_dx = cohort_hadm_ids - set(dx["hadm_id"].unique())
    hadm_no_px = cohort_hadm_ids - set(px["hadm_id"].unique())

    # ICD version breakdown
    dx_v9 = (dx["icd_version"] == 9).sum()
    dx_v10 = (dx["icd_version"] == 10).sum()
    px_v9 = (px["icd_version"] == 9).sum()
    px_v10 = (px["icd_version"] == 10).sum()

    # Diagnosis count per admission
    dx_per_hadm = dx.groupby("hadm_id").size()
    px_per_hadm = px.groupby("hadm_id").size()

    # Quick HF-related ICD scan (preview for 20a)
    hf_icd10 = dx[
        (dx["icd_version"] == 10) & dx["icd_code"].str.startswith("I50")
    ]
    hf_icd9 = dx[
        (dx["icd_version"] == 9) & dx["icd_code"].str.startswith("428")
    ]
    n_hadm_hf = len(
        set(hf_icd10["hadm_id"].unique()) | set(hf_icd9["hadm_id"].unique())
    )

    report = textwrap.dedent(f"""\
    ================================================================
    10b_extract_admission_context_mimic — Summary Report
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ================================================================

    DIAGNOSES
    ─────────
    Total raw rows       : {n_dx_total:>12,}
    Cohort-filtered rows : {len(dx):>12,}
    Unique hadm_id       : {n_dx_hadm:>12,}
    Admissions w/o dx    : {len(hadm_no_dx):>12,}
    ICD-9 codes          : {dx_v9:>12,}
    ICD-10 codes         : {dx_v10:>12,}
    Dx per admission     : median {dx_per_hadm.median():.0f},
                           mean {dx_per_hadm.mean():.1f},
                           max {dx_per_hadm.max():,}

    PROCEDURES
    ──────────
    Total raw rows       : {n_px_total:>12,}
    Cohort-filtered rows : {len(px):>12,}
    Unique hadm_id       : {n_px_hadm:>12,}
    Admissions w/o px    : {len(hadm_no_px):>12,}
    ICD-9 codes          : {px_v9:>12,}
    ICD-10 codes         : {px_v10:>12,}
    Px per admission     : median {px_per_hadm.median():.0f},
                           mean {px_per_hadm.mean():.1f},
                           max {px_per_hadm.max():,}

    HF DIAGNOSIS PREVIEW (for Step 20a)
    ────────────────────────────────────
    Admissions with any HF ICD code (I50.x / 428.x): {n_hadm_hf:,}
    HF prevalence among ICU admissions: {n_hadm_hf / len(cohort_hadm_ids) * 100:.1f}%
    (Note: this is a rough preview; 20a applies full ascertainment logic)

    OUTPUTS
    ───────
    {dx_path}  ({len(dx):,} rows, {dx.shape[1]} cols)
    {px_path}  ({len(px):,} rows, {px.shape[1]} cols)
    """)

    report_path = reports_dir / "10b_admission_context_mimic_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"  → Report: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
