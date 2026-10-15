#!/usr/bin/env python3
"""90b_00 — Explore eICU-CRD Schema for External Validation Planning

Purpose
-------
Diagnostic script to explore eICU table structures and map them to
MIMIC-IV variables needed for Step 90b external validation.

This script does NOT build any datasets — it only reports:
  1. Available tables and their schemas
  2. Variable mapping feasibility (22 state vars + treatments)
  3. HF diagnosis coverage (ICD codes)
  4. Sample size estimates for 29e downgrade assessment
  5. Treatment data completeness (fluid, vasopressor, diuretic)

Run this FIRST, review the output, then proceed with 90b pipeline scripts.

Compute: C1 (lightweight schema inspection)

Inputs
------
- data/raw/eicu/2.0/*.csv.gz

Outputs
-------
- reports/90b_00_eicu_schema_exploration.txt
- tables/90b_00_eicu_table_inventory.csv
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Project root ──
ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))

# eICU data root
EICU_ROOT = ROOT / "data" / "raw" / "eicu" / "2.0"


def peek_table(path: Path, n_rows: int = 5000) -> dict:
    """Read first n_rows of a CSV table and return schema info."""
    try:
        df = pd.read_csv(path, nrows=n_rows)
        info = {
            "name": path.stem.replace(".csv", ""),
            "path": str(path.relative_to(ROOT)),
            "n_cols": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "n_rows_sampled": len(df),
            "sample_nulls": {c: int(df[c].isna().sum()) for c in df.columns},
        }
        # For numeric columns, report basic stats
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            info["numeric_summary"] = {
                c: {
                    "min": float(df[c].min()) if pd.notna(df[c].min()) else None,
                    "max": float(df[c].max()) if pd.notna(df[c].max()) else None,
                    "mean": float(df[c].mean()) if pd.notna(df[c].mean()) else None,
                }
                for c in num_cols[:20]  # Limit to first 20
            }
        return info
    except Exception as e:
        return {"name": path.stem, "error": str(e)}


def count_full_table(path: Path) -> int:
    """Count total rows in a CSV table (streaming, memory-efficient)."""
    try:
        count = 0
        for chunk in pd.read_csv(path, chunksize=100_000, usecols=[0]):
            count += len(chunk)
        return count
    except Exception:
        return -1


def main():
    print("=" * 72)
    print("90b_00 — Explore eICU-CRD Schema for External Validation")
    print("=" * 72)
    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "=" * 72,
        "90b_00 — eICU-CRD Schema Exploration Report",
        "=" * 72,
        f"Generated: {ts}",
        "",
    ]

    # ── 0. Check data availability ──────────────────────────────────────
    print("\n[0] Checking eICU data availability...")
    if not EICU_ROOT.exists():
        msg = f"ERROR: eICU data root not found at {EICU_ROOT}"
        print(f"  {msg}")
        lines.append(msg)
        lines.append("  Please ensure the symlink data/raw/eicu → eICU data directory exists.")
        _save_report(lines)
        return

    # Find all CSV files
    csv_files = sorted(EICU_ROOT.glob("*.csv.gz"))
    if not csv_files:
        csv_files = sorted(EICU_ROOT.glob("*.csv"))
    print(f"  Found {len(csv_files)} table files in {EICU_ROOT}")

    # ── 1. Table inventory ───────────────────────────────────────────────
    print("\n[1] Building table inventory...")
    inventory = []
    for f in csv_files:
        print(f"  Scanning: {f.name}...", end=" ", flush=True)
        info = peek_table(f)
        n_total = count_full_table(f)
        info["n_rows_total"] = n_total
        inventory.append(info)
        print(f"{info.get('n_cols', '?')} cols, {n_total:,} rows")

    # Save inventory CSV
    inv_rows = []
    for info in inventory:
        inv_rows.append({
            "table": info.get("name", "?"),
            "n_cols": info.get("n_cols", 0),
            "n_rows": info.get("n_rows_total", -1),
            "columns_preview": ", ".join(info.get("columns", [])[:10]),
        })
    inv_df = pd.DataFrame(inv_rows)
    tables_dir = ROOT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    inv_df.to_csv(tables_dir / "90b_00_eicu_table_inventory.csv", index=False)

    lines.extend([
        "TABLE INVENTORY",
        "───────────────",
        f"  Data root: {EICU_ROOT}",
        f"  Tables found: {len(csv_files)}",
        "",
    ])
    for info in inventory:
        lines.append(f"  {info.get('name', '?'):30s}  "
                     f"cols={info.get('n_cols', '?'):>4}  "
                     f"rows={info.get('n_rows_total', '?'):>12,}")
    lines.append("")

    # ── 2. Detailed column listing per table ─────────────────────────────
    print("\n[2] Detailed column listing...")
    lines.append("DETAILED SCHEMAS")
    lines.append("────────────────")

    for info in inventory:
        tname = info.get("name", "?")
        cols = info.get("columns", [])
        lines.append(f"\n  TABLE: {tname}")
        lines.append(f"  {'─' * 60}")
        for c in cols:
            dtype = info.get("dtypes", {}).get(c, "?")
            nulls = info.get("sample_nulls", {}).get(c, "?")
            lines.append(f"    {c:40s}  {dtype:10s}  nulls={nulls}")
    lines.append("")

    # ── 3. HF diagnosis mapping ──────────────────────────────────────────
    print("\n[3] Exploring HF diagnosis coverage...")
    lines.append("HF DIAGNOSIS MAPPING")
    lines.append("────────────────────")

    # Look for diagnosis table
    diag_table = _find_table(inventory, "diagnosis")
    if diag_table:
        diag_path = _get_table_path(diag_table)
        if diag_path:
            print(f"  Reading diagnosis table: {diag_path.name}")
            diag_df = pd.read_csv(diag_path)
            lines.append(f"  Diagnosis table: {len(diag_df):,} rows")
            lines.append(f"  Columns: {list(diag_df.columns)}")

            # Check ICD code column names
            icd_col = None
            for candidate in ["icd9code", "diagnosisstring", "icd_code",
                               "diagnosis", "activeupondischarge"]:
                if candidate in diag_df.columns:
                    icd_col = candidate
                    break

            if icd_col:
                lines.append(f"  ICD column: {icd_col}")

                # Search for HF-related codes
                hf_patterns = [
                    "428",      # ICD-9 HF
                    "I50",      # ICD-10 HF
                    "heart failure",
                    "congestive",
                    "cardiomyopathy",
                    "diastolic",
                    "systolic",
                    "HFpEF",
                    "HFrEF",
                ]
                for pat in hf_patterns:
                    mask = diag_df[icd_col].astype(str).str.contains(
                        pat, case=False, na=False
                    )
                    n_match = mask.sum()
                    n_patients = diag_df.loc[mask, "patientunitstayid"].nunique() if "patientunitstayid" in diag_df.columns else "?"
                    lines.append(f"    Pattern '{pat}': {n_match:,} rows, "
                                 f"{n_patients} unique stays")

                # Show sample HF diagnoses
                hf_mask = diag_df[icd_col].astype(str).str.contains(
                    "428|I50|heart fail|congestive", case=False, na=False
                )
                if hf_mask.any():
                    sample = diag_df.loc[hf_mask, icd_col].value_counts().head(20)
                    lines.append("\n  Top 20 HF-related diagnosis values:")
                    for val, cnt in sample.items():
                        lines.append(f"    {val:60s}  n={cnt:,}")
            else:
                lines.append("  WARNING: No ICD code column found!")
                lines.append(f"  Available columns: {list(diag_df.columns)}")
    else:
        lines.append("  WARNING: No diagnosis table found!")
    lines.append("")

    # ── 4. Vitals mapping ────────────────────────────────────────────────
    print("\n[4] Exploring vitals coverage...")
    lines.append("VITALS MAPPING")
    lines.append("──────────────")

    # MIMIC state vars we need to map
    target_vars = [
        "map", "sbp", "dbp", "heart_rate", "spo2", "cvp", "resp_rate",
        "fio2", "lactate", "creatinine", "bun", "potassium", "sodium",
        "bicarbonate", "glucose", "ph", "hemoglobin", "wbc", "platelet",
        "albumin", "bilirubin", "troponin",
    ]

    # Check vitalPeriodic
    vital_periodic = _find_table(inventory, "vitalperiodic")
    if vital_periodic:
        vp_path = _get_table_path(vital_periodic)
        if vp_path:
            vp_df = pd.read_csv(vp_path, nrows=10000)
            lines.append(f"\n  vitalPeriodic: {vital_periodic.get('n_rows_total', '?'):,} rows")
            lines.append(f"  Columns: {list(vp_df.columns)}")
            for c in vp_df.select_dtypes(include=[np.number]).columns:
                if c == "patientunitstayid" or c.endswith("offset"):
                    continue
                nonnull = vp_df[c].notna().sum()
                pct = nonnull / len(vp_df) * 100
                lines.append(f"    {c:30s}  {nonnull:>6}/{len(vp_df)} ({pct:.1f}%)")

    # Check vitalAperiodic
    vital_aperiodic = _find_table(inventory, "vitalaperiodic")
    if vital_aperiodic:
        va_path = _get_table_path(vital_aperiodic)
        if va_path:
            va_df = pd.read_csv(va_path, nrows=10000)
            lines.append(f"\n  vitalAperiodic: {vital_aperiodic.get('n_rows_total', '?'):,} rows")
            lines.append(f"  Columns: {list(va_df.columns)}")
            # Check what vital types exist
            for c in ["noninvasivesystolic", "noninvasivediastolic", "noninvasivemean",
                       "paop", "cardiacoutput", "cardiacinput", "svr", "svri",
                       "pvr", "pvri"]:
                if c in va_df.columns:
                    nonnull = va_df[c].notna().sum()
                    pct = nonnull / len(va_df) * 100
                    lines.append(f"    {c:30s}  {nonnull:>6}/{len(va_df)} ({pct:.1f}%)")

    # Check lab table
    lab_table = _find_table(inventory, "lab")
    if lab_table:
        lab_path = _get_table_path(lab_table)
        if lab_path:
            lab_df = pd.read_csv(lab_path, nrows=50000)
            lines.append(f"\n  lab: {lab_table.get('n_rows_total', '?'):,} rows")
            lines.append(f"  Columns: {list(lab_df.columns)}")
            # Check labname distribution
            if "labname" in lab_df.columns:
                lab_counts = lab_df["labname"].value_counts().head(30)
                lines.append("\n  Top 30 lab names:")
                for val, cnt in lab_counts.items():
                    lines.append(f"    {val:40s}  n={cnt:,}")

                # Map to our target vars
                lab_mapping = {
                    "lactate": ["lactate"],
                    "creatinine": ["creatinine"],
                    "bun": ["BUN"],
                    "potassium": ["potassium"],
                    "sodium": ["sodium"],
                    "bicarbonate": ["bicarbonate", "HCO3"],
                    "glucose": ["glucose", "bedside glucose"],
                    "ph": ["pH"],
                    "hemoglobin": ["Hgb", "hemoglobin"],
                    "wbc": ["WBC x 1000", "WBC"],
                    "platelet": ["platelets x 1000", "platelet"],
                    "albumin": ["albumin"],
                    "bilirubin": ["total bilirubin", "bilirubin"],
                    "troponin": ["troponin - I", "troponin - T", "troponin"],
                }
                lines.append("\n  Variable mapping feasibility:")
                all_labnames = lab_df["labname"].unique()
                for target, candidates in lab_mapping.items():
                    found = [c for c in candidates
                             if any(c.lower() == ln.lower() for ln in all_labnames)]
                    if found:
                        lines.append(f"    {target:20s} → {found}  ✓")
                    else:
                        # Fuzzy search
                        fuzzy = [ln for ln in all_labnames
                                 if target.lower() in str(ln).lower()][:3]
                        if fuzzy:
                            lines.append(f"    {target:20s} → FUZZY: {fuzzy}")
                        else:
                            lines.append(f"    {target:20s} → NOT FOUND  ✗")

    lines.append("")

    # ── 5. Treatment data ────────────────────────────────────────────────
    print("\n[5] Exploring treatment data...")
    lines.append("TREATMENT DATA")
    lines.append("──────────────")

    # Infusion drug table (vasopressors + fluid)
    infusion_table = _find_table(inventory, "infusiondrug")
    if infusion_table:
        inf_path = _get_table_path(infusion_table)
        if inf_path:
            inf_df = pd.read_csv(inf_path, nrows=50000)
            lines.append(f"\n  infusionDrug: {infusion_table.get('n_rows_total', '?'):,} rows")
            lines.append(f"  Columns: {list(inf_df.columns)}")
            if "drugname" in inf_df.columns:
                drug_counts = inf_df["drugname"].value_counts().head(30)
                lines.append("\n  Top 30 drug names:")
                for val, cnt in drug_counts.items():
                    lines.append(f"    {val:50s}  n={cnt:,}")

                # Check for vasopressors
                vaso_patterns = ["norepinephrine", "vasopressin", "epinephrine",
                                 "dopamine", "phenylephrine", "dobutamine",
                                 "milrinone", "levophed"]
                lines.append("\n  Vasopressor search:")
                all_drugs = inf_df["drugname"].unique()
                for vp in vaso_patterns:
                    matches = [d for d in all_drugs
                               if vp.lower() in str(d).lower()]
                    if matches:
                        n = inf_df[inf_df["drugname"].isin(matches)].shape[0]
                        lines.append(f"    {vp:20s} → {matches[:3]}  (n={n:,})")

    # Intake/output table (fluids, urine)
    io_table = _find_table(inventory, "intakeoutput")
    if io_table:
        io_path = _get_table_path(io_table)
        if io_path:
            io_df = pd.read_csv(io_path, nrows=50000)
            lines.append(f"\n  intakeOutput: {io_table.get('n_rows_total', '?'):,} rows")
            lines.append(f"  Columns: {list(io_df.columns)}")
            if "celllabel" in io_df.columns:
                cell_counts = io_df["celllabel"].value_counts().head(30)
                lines.append("\n  Top 30 cell labels:")
                for val, cnt in cell_counts.items():
                    lines.append(f"    {val:50s}  n={cnt:,}")

    # Medication table (diuretics)
    med_table = _find_table(inventory, "medication")
    if med_table:
        med_path = _get_table_path(med_table)
        if med_path:
            med_df = pd.read_csv(med_path, nrows=50000)
            lines.append(f"\n  medication: {med_table.get('n_rows_total', '?'):,} rows")
            lines.append(f"  Columns: {list(med_df.columns)}")
            if "drugname" in med_df.columns:
                # Search for diuretics
                diuretic_patterns = ["furosemide", "lasix", "bumetanide",
                                     "torsemide", "hydrochlorothiazide",
                                     "spironolactone", "metolazone"]
                lines.append("\n  Diuretic search:")
                all_meds = med_df["drugname"].unique()
                for dp in diuretic_patterns:
                    matches = [d for d in all_meds
                               if dp.lower() in str(d).lower()]
                    if matches:
                        n = med_df[med_df["drugname"].isin(matches)].shape[0]
                        lines.append(f"    {dp:25s} → {matches[:3]}  (n={n:,})")

    lines.append("")

    # ── 6. Patient demographics + ICU stay structure ─────────────────────
    print("\n[6] Exploring patient demographics...")
    lines.append("PATIENT DEMOGRAPHICS")
    lines.append("────────────────────")

    patient_table = _find_table(inventory, "patient")
    if patient_table:
        pt_path = _get_table_path(patient_table)
        if pt_path:
            pt_df = pd.read_csv(pt_path)
            lines.append(f"  patient: {len(pt_df):,} rows (ICU stays)")
            lines.append(f"  Columns: {list(pt_df.columns)}")
            # Case-insensitive column lookup for robustness
            col_map = {c.lower(): c for c in pt_df.columns}
            uid_col = col_map.get("uniquepid")
            hosp_col = col_map.get("hospitalid")
            if uid_col:
                lines.append(f"  Unique patients: {pt_df[uid_col].nunique():,}")
            else:
                lines.append("  Unique patients: [column 'uniquepid' not found]")
            if hosp_col:
                lines.append(f"  Hospitals: {pt_df[hosp_col].nunique():,}")
            else:
                lines.append("  Hospitals: [column 'hospitalid' not found]")

            # Age distribution
            if "age" in pt_df.columns:
                # eICU age: "> 89" for patients > 89
                pt_df["age_num"] = pd.to_numeric(pt_df["age"].replace("> 89", 90), errors="coerce")
                lines.append(f"  Age: mean={pt_df['age_num'].mean():.1f}, "
                             f"median={pt_df['age_num'].median():.1f}")

            # Gender
            if "gender" in pt_df.columns:
                lines.append(f"  Gender: {pt_df['gender'].value_counts().to_dict()}")

            # Unit type
            if "unittype" in pt_df.columns:
                lines.append(f"  Unit types: {pt_df['unittype'].value_counts().head(10).to_dict()}")

            # Discharge status (mortality proxy)
            if "unitdischargestatus" in pt_df.columns:
                lines.append(f"  Unit discharge status: "
                             f"{pt_df['unitdischargestatus'].value_counts().to_dict()}")
            if "hospitaldischargestatus" in pt_df.columns:
                lines.append(f"  Hospital discharge status: "
                             f"{pt_df['hospitaldischargestatus'].value_counts().to_dict()}")

            # ICU LOS distribution
            if "unitdischargeoffset" in pt_df.columns:
                los_hours = pt_df["unitdischargeoffset"] / 60
                lines.append(f"  ICU LOS (hours): mean={los_hours.mean():.1f}, "
                             f"median={los_hours.median():.1f}, "
                             f"P95={los_hours.quantile(0.95):.1f}")
                lines.append(f"  ICU LOS ≥ 24h: {(los_hours >= 24).sum():,} "
                             f"({(los_hours >= 24).mean() * 100:.1f}%)")
                lines.append(f"  ICU LOS ≥ 30h (T0+24h horizon): "
                             f"{(los_hours >= 30).sum():,} "
                             f"({(los_hours >= 30).mean() * 100:.1f}%)")

    lines.append("")

    # ── 7. APACHE / structured severity data ─────────────────────────────
    print("\n[7] Exploring APACHE data...")
    lines.append("APACHE / SEVERITY DATA")
    lines.append("──────────────────────")

    apache_table = _find_table(inventory, "apacheapsvar")
    if apache_table:
        ap_path = _get_table_path(apache_table)
        if ap_path:
            ap_df = pd.read_csv(ap_path, nrows=10000)
            lines.append(f"  apacheApsVar: {apache_table.get('n_rows_total', '?'):,} rows")
            lines.append(f"  Columns: {list(ap_df.columns)}")
            # Check for EF or cardiac-related fields
            cardiac_cols = [c for c in ap_df.columns
                           if any(k in c.lower() for k in
                                  ["cardiac", "heart", "ef", "ejection"])]
            if cardiac_cols:
                lines.append(f"  Cardiac-related columns: {cardiac_cols}")
            else:
                lines.append("  No cardiac-specific columns found in apacheApsVar")

    apache_pred_table = _find_table(inventory, "apachepredvar")
    if apache_pred_table:
        apd_path = _get_table_path(apache_pred_table)
        if apd_path:
            apd_df = pd.read_csv(apd_path, nrows=10000)
            lines.append(f"\n  apachePredVar: {apache_pred_table.get('n_rows_total', '?'):,} rows")
            lines.append(f"  Columns: {list(apd_df.columns)}")

    apache_patient_table = _find_table(inventory, "apachepatientresult")
    if apache_patient_table:
        apr_path = _get_table_path(apache_patient_table)
        if apr_path:
            apr_df = pd.read_csv(apr_path, nrows=10000)
            lines.append(f"\n  apachePatientResult: {apache_patient_table.get('n_rows_total', '?'):,} rows")
            lines.append(f"  Columns: {list(apr_df.columns)}")
            if "apachescore" in apr_df.columns:
                lines.append(f"  APACHE score: mean={apr_df['apachescore'].mean():.1f}")
            if "actualhospitalmortality" in apr_df.columns:
                lines.append(f"  Hospital mortality: "
                             f"{apr_df['actualhospitalmortality'].value_counts().to_dict()}")

    lines.append("")

    # ── 8. 29e Downgrade assessment (preliminary) ────────────────────────
    print("\n[8] Preliminary 29e downgrade assessment...")
    lines.append("29e DOWNGRADE ASSESSMENT (PRELIMINARY)")
    lines.append("──────────────────────────────────────")
    lines.append("  Note: This is based on first-pass data exploration.")
    lines.append("  Actual numbers will be refined after full cohort build.")
    lines.append("")

    # Count HF patients
    if diag_table and diag_path:
        diag_full = pd.read_csv(diag_path)
        icd_col_found = None
        for candidate in ["icd9code", "diagnosisstring"]:
            if candidate in diag_full.columns:
                icd_col_found = candidate
                break

        if icd_col_found:
            # HF diagnosis (ICD-9: 428.x, ICD-10: I50.x)
            hf_mask = diag_full[icd_col_found].astype(str).str.contains(
                "428|I50", case=False, na=False
            )
            hf_stays = diag_full.loc[hf_mask, "patientunitstayid"].unique()
            lines.append(f"  HF diagnosis stays (ICD 428/I50): {len(hf_stays):,}")

            # Systolic HF (to exclude for broader-compatible)
            systolic_mask = diag_full[icd_col_found].astype(str).str.contains(
                "428\\.2|I50\\.2|systolic", case=False, na=False
            )
            systolic_stays = diag_full.loc[systolic_mask, "patientunitstayid"].unique()
            lines.append(f"  Systolic HF stays (to exclude): {len(systolic_stays):,}")

            # Diastolic HF proxy
            diastolic_mask = diag_full[icd_col_found].astype(str).str.contains(
                "428\\.3|I50\\.3|diastolic", case=False, na=False
            )
            diastolic_stays = diag_full.loc[diastolic_mask, "patientunitstayid"].unique()
            lines.append(f"  Diastolic HF stays (proxy): {len(diastolic_stays):,}")

            # Broader-compatible HFpEF: HF + NOT systolic
            hfpef_broader = set(hf_stays) - set(systolic_stays)
            lines.append(f"  Broader-compatible HFpEF (HF - systolic): {len(hfpef_broader):,}")

            # 29e thresholds
            lines.append("")
            lines.append("  29e Trigger 1 (sample size):")
            if len(hfpef_broader) >= 500:
                lines.append(f"    {len(hfpef_broader):,} ≥ 500 → FULL VALIDATION ELIGIBLE")
            elif len(hfpef_broader) >= 200:
                lines.append(f"    {len(hfpef_broader):,} ∈ [200, 500) → SENSITIVITY-ONLY")
            else:
                lines.append(f"    {len(hfpef_broader):,} < 200 → DESCRIPTIVE ONLY")

            lines.append("")
            lines.append("  29e Trigger 2 (ascertainment):")
            lines.append("    Grade: BROADER-COMPATIBLE (HF dx + NOT known HFrEF)")
            lines.append("    → Full validation with ascertainment sensitivity required")

    lines.append("")

    # ── 9. Feature mapping summary ───────────────────────────────────────
    print("\n[9] Feature mapping summary...")
    lines.append("FEATURE MAPPING SUMMARY")
    lines.append("───────────────────────")
    lines.append("  MIMIC target variable  →  eICU source table  →  Status")
    lines.append("  " + "─" * 60)

    mapping_plan = [
        ("map",         "vitalPeriodic.systemicmean",    "CHECK"),
        ("sbp",         "vitalPeriodic.systemicsystolic", "CHECK"),
        ("dbp",         "vitalPeriodic.systemicdiastolic","CHECK"),
        ("heart_rate",  "vitalPeriodic.heartrate",       "CHECK"),
        ("spo2",        "vitalPeriodic.sao2",            "CHECK"),
        ("cvp",         "vitalAperiodic.cvp or nurseCharting", "CHECK"),
        ("resp_rate",   "vitalPeriodic.respiration",     "CHECK"),
        ("fio2",        "respiratoryCare or nurseCharting", "CHECK"),
        ("lactate",     "lab.labname='lactate'",         "CHECK"),
        ("creatinine",  "lab.labname='creatinine'",      "CHECK"),
        ("bun",         "lab.labname='BUN'",             "CHECK"),
        ("potassium",   "lab.labname='potassium'",       "CHECK"),
        ("sodium",      "lab.labname='sodium'",          "CHECK"),
        ("bicarbonate", "lab.labname='bicarbonate'",     "CHECK"),
        ("glucose",     "lab.labname='glucose'",         "CHECK"),
        ("ph",          "lab.labname='pH'",              "CHECK"),
        ("hemoglobin",  "lab.labname='Hgb'",             "CHECK"),
        ("wbc",         "lab.labname='WBC x 1000'",      "CHECK"),
        ("platelet",    "lab.labname='platelets x 1000'", "CHECK"),
        ("albumin",     "lab.labname='albumin'",         "CHECK"),
        ("bilirubin",   "lab.labname='total bilirubin'", "CHECK"),
        ("troponin",    "lab.labname='troponin - I'",    "CHECK"),
        ("iv_fluid",    "intakeOutput (IV labels)",      "CHECK"),
        ("vasopressor", "infusionDrug (NE/vasopressin)", "CHECK"),
        ("diuretic",    "medication (furosemide/bumetanide)", "CHECK"),
        ("urine_output","intakeOutput (urine labels)",   "CHECK"),
        ("mortality_28d","patient.hospitaldischargestatus + offset", "CHECK"),
    ]
    for target, source, status in mapping_plan:
        lines.append(f"  {target:20s} → {source:40s}  [{status}]")
    lines.append("")
    lines.append("  NOTE: Actual mapping verification requires matching column")
    lines.append("  names against the inventory above. Status will be updated")
    lines.append("  after manual review of this report.")

    lines.append("")

    # ── 10. Recommendations ──────────────────────────────────────────────
    lines.extend([
        "RECOMMENDATIONS",
        "────────────────",
        "  1. Review variable mapping above against actual eICU column names",
        "  2. Verify HF diagnosis ICD patterns (eICU may use diagnosisstring instead of ICD codes)",
        "  3. Confirm 28-day mortality can be derived (hospital discharge status + offset)",
        "  4. Check vasopressor NE-equivalent conversion feasibility",
        "  5. Check IV fluid volume recording completeness",
        "  6. Assess diuretic furosemide-equivalent conversion",
        "  7. After review, proceed with 90b_01_build_eicu_cohort.py",
    ])

    # ── Save report ──────────────────────────────────────────────────────
    _save_report(lines)

    total_time = time.time() - t0
    print()
    print("=" * 72)
    print(f"90b_00 COMPLETE. Time: {total_time:.1f}s")
    print("=" * 72)


def _find_table(inventory: list[dict], name_pattern: str) -> dict | None:
    """Find a table in inventory by name pattern (case-insensitive).

    Preference order:
      1) exact table name match
      2) fallback substring match

    This avoids accidental collisions such as `patient` matching
    `apachePatientResult` before the true `patient` table.
    """
    pattern = name_pattern.lower()

    # First pass: exact match
    for info in inventory:
        tname = info.get("name", "").lower()
        if tname == pattern:
            return info

    # Second pass: substring fallback
    for info in inventory:
        tname = info.get("name", "").lower()
        if pattern in tname:
            return info
    return None


def _get_table_path(info: dict) -> Path | None:
    """Get full path to a table from its inventory info."""
    rel_path = info.get("path")
    if rel_path:
        return ROOT / rel_path

    # Try to reconstruct
    name = info.get("name", "")
    for ext in [".csv.gz", ".csv"]:
        p = EICU_ROOT / f"{name}{ext}"
        if p.exists():
            return p
    return None


def _save_report(lines: list[str]):
    """Save report to file."""
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "90b_00_eicu_schema_exploration.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Report saved: {report_path}")


if __name__ == "__main__":
    main()
