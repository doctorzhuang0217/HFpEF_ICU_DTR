#!/usr/bin/env python3
"""90b_02 — Extract eICU Temporal Data (Vitals + Labs + Treatments)

Protocol: v3.1, Step 90b (sub-step 2)
Compute: C3

Purpose
-------
Extract and grid eICU temporal data into 2-hour bins matching the MIMIC
pipeline. This mirrors MIMIC Steps 10d/10e/10f/10g + 40b/40c.

Extracts 22 state variables + 3 treatment dimensions + urine output + fluid balance.
Maps eICU-specific column names to MIMIC-harmonized variable names.

eICU table mapping:
  Vitals:      vitalPeriodic (HR, BP, SpO2, RR) + vitalAperiodic (CVP, non-invasive BP)
               + nurseCharting fallback (CVP, FiO2) [P1-1]
  Labs:        lab + customLab (all 14 lab variables; customLab recovers lactate from sites
               where standard lab interface was unavailable)
  Vasopressors: infusionDrug (NE, vasopressin, etc. → NE-equivalent)
  Fluids:      intakeOutput (IV intake volumes)
  Diuretics:   infusionDrug + medication (furosemide, bumetanide → furosemide-equivalent)
  Urine:       intakeOutput (urine output labels)

Inputs
------
- data/staging/90b_eicu_cohort_v1.parquet
- data/raw/eicu/2.0/*.csv.gz

Outputs
-------
- data/interim/90b_eicu_grid2h_states.parquet
- data/interim/90b_eicu_missingness_features.parquet
- reports/90b_02_eicu_temporal_extraction_summary.txt
"""

from __future__ import annotations

import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Project root ──
ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
EICU_ROOT = ROOT / "data" / "raw" / "eicu" / "2.0"
STAGING_DIR = ROOT / "data" / "staging"
INTERIM_DIR = ROOT / "data" / "interim"
REPORTS_DIR = ROOT / "reports"

for d in [INTERIM_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Grid parameters (must match MIMIC) ───────────────────────────────────
T0_HOURS = 6         # Time zero: ICU admission + 6h
BIN_HOURS = 2        # 2-hour decision bins
N_BINS = 12          # 12 bins × 2h = 24h horizon
HORIZON_HOURS = T0_HOURS + N_BINS * BIN_HOURS  # 30h from admission

# Time window in minutes for early filtering (avoids OOM on large tables)
T0_MIN = T0_HOURS * 60          # 360 min
HORIZON_MIN = HORIZON_HOURS * 60  # 1800 min

# ── State variables (22 active, matching MIMIC 45c) ─────────────────────
STATE_VARS = [
    "map", "sbp", "dbp", "heart_rate", "spo2", "cvp", "resp_rate",
    "fio2", "lactate", "creatinine", "bun", "potassium", "sodium",
    "bicarbonate", "glucose", "ph", "hemoglobin", "wbc", "platelet",
    "albumin", "bilirubin", "troponin",
]

# ── eICU lab name → MIMIC variable name mapping ─────────────────────────
# NOTE: These are common eICU lab names. The 90b_00 exploration report
# should be used to verify/update these mappings.
LAB_NAME_MAP = {
    # Exact matches (case-insensitive search)
    "lactate": "lactate",
    "creatinine": "creatinine",
    "BUN": "bun",
    "potassium": "potassium",
    "sodium": "sodium",
    "bicarbonate": "bicarbonate",
    "HCO3": "bicarbonate",
    "glucose": "glucose",
    "bedside glucose": "glucose",
    "pH": "ph",
    "Hgb": "hemoglobin",
    "hemoglobin": "hemoglobin",
    "WBC x 1000": "wbc",
    "WBC": "wbc",
    "platelets x 1000": "platelet",
    "albumin": "albumin",
    "total bilirubin": "bilirubin",
    "direct bilirubin": None,  # Skip, we want total
    "troponin - I": "troponin",
    "troponin - T": "troponin",
    "troponin I": "troponin",
}

# ── Vasopressor NE-equivalent conversion factors ─────────────────────────
# Based on Surviving Sepsis Campaign 2021 dose equivalence
VASO_NE_EQUIV = {
    "norepinephrine": 1.0,
    "levophed": 1.0,
    "epinephrine": 1.0,
    "vasopressin": 2.5,  # 0.04 U/min ≈ 0.1 mcg/kg/min NE
    "phenylephrine": 0.1,
    "dopamine": 0.01,    # Approximate, dose-dependent
    "dobutamine": 0.0,   # Inotrope, not vasopressor
}

# ── Diuretic furosemide-equivalent conversion factors ────────────────────
DIURETIC_FURO_EQUIV = {
    "furosemide": 1.0,
    "lasix": 1.0,
    "bumetanide": 40.0,   # 1 mg bumetanide ≈ 40 mg furosemide
    "bumex": 40.0,
    "torsemide": 2.0,     # 20 mg torsemide ≈ 40 mg furosemide (consistent with action_space_v1.yaml)
}


def parse_mixed_numeric(series: pd.Series) -> pd.Series:
    """Parse mixed-type numeric strings (e.g., '5 mcg/min', '1,000')."""
    cleaned = series.astype(str).str.replace(",", "", regex=False)
    extracted = cleaned.str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def extract_vitals(cohort_ids: set) -> pd.DataFrame:
    """Extract vitals from vitalPeriodic + vitalAperiodic."""
    print("  [Vitals] Loading vitalPeriodic...")
    records = []

    # ── vitalPeriodic ──
    vp_path = EICU_ROOT / "vitalPeriodic.csv.gz"
    if not vp_path.exists():
        vp_path = EICU_ROOT / "vitalperiodic.csv.gz"

    if vp_path.exists():
        vp_col_map = {
            "systemicmean": "map",
            "systemicsystolic": "sbp",
            "systemicdiastolic": "dbp",
            "heartrate": "heart_rate",
            "sao2": "spo2",
            "respiration": "resp_rate",
        }
        needed_cols = ["patientunitstayid", "observationoffset"] + list(vp_col_map.keys())
        for chunk in tqdm(
            pd.read_csv(vp_path, chunksize=500_000,
                        usecols=lambda c: c in needed_cols),
            desc="    vitalPeriodic", leave=False
        ):
            chunk = chunk[chunk["patientunitstayid"].isin(cohort_ids)]
            if chunk.empty:
                continue
            # Early time-window filter — drops ~80% of rows
            chunk = chunk[
                (chunk["observationoffset"] >= T0_MIN) &
                (chunk["observationoffset"] < HORIZON_MIN)
            ]
            if chunk.empty:
                continue
            for eicu_col, our_var in vp_col_map.items():
                if eicu_col in chunk.columns:
                    sub = chunk[["patientunitstayid", "observationoffset", eicu_col]].dropna(
                        subset=[eicu_col]
                    )
                    sub = sub.rename(columns={
                        "patientunitstayid": "stay_id",
                        "observationoffset": "offset_min",
                        eicu_col: "value",
                    })
                    sub["variable"] = our_var
                    records.append(sub[["stay_id", "offset_min", "value", "variable"]])
    else:
        print("    WARNING: vitalPeriodic not found!")

    # ── vitalAperiodic (CVP, non-invasive BP) ──
    va_path = EICU_ROOT / "vitalAperiodic.csv.gz"
    if not va_path.exists():
        va_path = EICU_ROOT / "vitalaperiodic.csv.gz"

    if va_path.exists():
        print("  [Vitals] Loading vitalAperiodic...")
        va_col_map = {
            "noninvasivemean": "map",     # Fallback if no invasive MAP
            "noninvasivesystolic": "sbp",
            "noninvasivediastolic": "dbp",
            "cvp": "cvp",                 # Central Venous Pressure
        }
        va_needed = ["patientunitstayid", "observationoffset"] + list(va_col_map.keys())
        for chunk in tqdm(
            pd.read_csv(va_path, chunksize=500_000,
                        usecols=lambda c: c in va_needed),
            desc="    vitalAperiodic", leave=False
        ):
            chunk = chunk[chunk["patientunitstayid"].isin(cohort_ids)]
            if chunk.empty:
                continue
            chunk = chunk[
                (chunk["observationoffset"] >= T0_MIN) &
                (chunk["observationoffset"] < HORIZON_MIN)
            ]
            if chunk.empty:
                continue
            for eicu_col, our_var in va_col_map.items():
                if eicu_col in chunk.columns:
                    sub = chunk[["patientunitstayid", "observationoffset", eicu_col]].dropna(
                        subset=[eicu_col]
                    )
                    sub = sub.rename(columns={
                        "patientunitstayid": "stay_id",
                        "observationoffset": "offset_min",
                        eicu_col: "value",
                    })
                    sub["variable"] = our_var
                    records.append(sub[["stay_id", "offset_min", "value", "variable"]])

    # ── FiO2 from respiratoryCharting ──
    rc_path = EICU_ROOT / "respiratoryCharting.csv.gz"
    if not rc_path.exists():
        rc_path = EICU_ROOT / "respiratorycharting.csv.gz"

    if rc_path.exists():
        print("  [Vitals] Loading respiratoryCharting (FiO2)...")
        fio2_labels = {"fio2", "fio2 (%)", "fio2(%)", "fraction of inspired oxygen"}
        rc_needed = {"patientunitstayid", "respchartoffset",
                     "respchartvaluelabel", "respchartvalue"}
        for chunk in tqdm(
            pd.read_csv(rc_path, chunksize=500_000),
            desc="    respiratoryCharting", leave=False
        ):
            # Select needed columns after read (usecols lambda causes IndexError)
            avail = [c for c in rc_needed if c in chunk.columns]
            chunk = chunk[avail]
            chunk = chunk[chunk["patientunitstayid"].isin(cohort_ids)]
            if chunk.empty:
                continue
            if "respchartoffset" in chunk.columns:
                chunk = chunk[
                    (chunk["respchartoffset"] >= T0_MIN) &
                    (chunk["respchartoffset"] < HORIZON_MIN)
                ]
            if chunk.empty:
                continue
            if "respchartvaluelabel" not in chunk.columns:
                continue

            mask = chunk["respchartvaluelabel"].str.lower().isin(fio2_labels)
            if not mask.any():
                continue
            sub = chunk.loc[mask].copy()
            sub["value"] = pd.to_numeric(sub["respchartvalue"], errors="coerce")
            sub = sub.dropna(subset=["value"])
            # FiO2 sanity: should be 21-100 (percent)
            sub = sub[(sub["value"] >= 21) & (sub["value"] <= 100)]
            if sub.empty:
                continue
            rec = pd.DataFrame({
                "stay_id": sub["patientunitstayid"].values,
                "offset_min": sub["respchartoffset"].values,
                "value": sub["value"].values,
                "variable": "fio2",
            })
            records.append(rec)
    else:
        print("    WARNING: respiratoryCharting not found!")

    # ── nurseCharting fallback for CVP + FiO2 (P1-1) ──
    # nurseCharting is a 151M-row table; only load for cohort stays with
    # CVP or FiO2 labels.  This recovers values from sites that chart these
    # in nursing flowsheets instead of vitalAperiodic / respiratoryCharting.
    nc_path = EICU_ROOT / "nurseCharting.csv.gz"
    if not nc_path.exists():
        nc_path = EICU_ROOT / "nursecharting.csv.gz"

    if nc_path.exists():
        print("  [Vitals] Loading nurseCharting fallback (CVP + FiO2)...")
        # Label sets (lowercased for matching)
        cvp_labels = {"cvp", "central venous pressure"}
        fio2_labels_nc = {"fio2", "fio2 (%)", "fio2(%)", "fraction of inspired oxygen"}
        target_labels = cvp_labels | fio2_labels_nc

        nc_n_cvp = 0
        nc_n_fio2 = 0
        for chunk in tqdm(
            pd.read_csv(nc_path, chunksize=500_000),
            desc="    nurseCharting", leave=False,
        ):
            chunk = chunk[chunk["patientunitstayid"].isin(cohort_ids)]
            if chunk.empty:
                continue
            # Time window filter
            if "nursingchartoffset" in chunk.columns:
                chunk = chunk[
                    (chunk["nursingchartoffset"] >= T0_MIN)
                    & (chunk["nursingchartoffset"] < HORIZON_MIN)
                ]
            if chunk.empty:
                continue
            if "nursingchartcelltypevallabel" not in chunk.columns:
                continue

            label_lower = chunk["nursingchartcelltypevallabel"].str.lower()
            mask = label_lower.isin(target_labels)
            if not mask.any():
                continue

            sub = chunk.loc[mask].copy()
            sub["_label_lower"] = label_lower[mask]
            sub["value"] = pd.to_numeric(sub["nursingchartvalue"], errors="coerce")
            sub = sub.dropna(subset=["value"])
            if sub.empty:
                continue

            # CVP: valid range 0–40 mmHg
            cvp_mask = sub["_label_lower"].isin(cvp_labels)
            cvp_sub = sub.loc[cvp_mask & (sub["value"] >= 0) & (sub["value"] <= 40)]
            if len(cvp_sub):
                nc_n_cvp += len(cvp_sub)
                records.append(pd.DataFrame({
                    "stay_id": cvp_sub["patientunitstayid"].values,
                    "offset_min": cvp_sub["nursingchartoffset"].values,
                    "value": cvp_sub["value"].values,
                    "variable": "cvp",
                }))

            # FiO2: valid range 21–100 %
            fio2_mask = sub["_label_lower"].isin(fio2_labels_nc)
            fio2_sub = sub.loc[fio2_mask & (sub["value"] >= 21) & (sub["value"] <= 100)]
            if len(fio2_sub):
                nc_n_fio2 += len(fio2_sub)
                records.append(pd.DataFrame({
                    "stay_id": fio2_sub["patientunitstayid"].values,
                    "offset_min": fio2_sub["nursingchartoffset"].values,
                    "value": fio2_sub["value"].values,
                    "variable": "fio2",
                }))

        print(f"    nurseCharting fallback: CVP={nc_n_cvp:,}, FiO2={nc_n_fio2:,} records")
    else:
        print("    INFO: nurseCharting not found, skipping CVP/FiO2 fallback")

    if records:
        # Periodic consolidation to reduce memory fragmentation
        vitals = pd.concat(records, ignore_index=True)
        del records
        print(f"  [Vitals] {len(vitals):,} measurements extracted")
        return vitals
    return pd.DataFrame(columns=["stay_id", "offset_min", "value", "variable"])


def extract_labs(cohort_ids: set) -> pd.DataFrame:
    """Extract labs from lab table."""
    print("  [Labs] Loading lab table...")
    lab_path = EICU_ROOT / "lab.csv.gz"
    if not lab_path.exists():
        print("    WARNING: lab table not found!")
        return pd.DataFrame(columns=["stay_id", "offset_min", "value", "variable"])

    records = []
    lab_needed = ["patientunitstayid", "labresultoffset", "labname", "labresult"]
    for chunk in tqdm(
        pd.read_csv(lab_path, chunksize=500_000,
                    usecols=lambda c: c in lab_needed),
        desc="    lab", leave=False
    ):
        chunk = chunk[chunk["patientunitstayid"].isin(cohort_ids)]
        if chunk.empty:
            continue

        # Early time-window filter
        if "labresultoffset" in chunk.columns:
            chunk = chunk[
                (chunk["labresultoffset"] >= T0_MIN) &
                (chunk["labresultoffset"] < HORIZON_MIN)
            ]
        if chunk.empty:
            continue

        # Map lab names
        if "labname" not in chunk.columns:
            continue

        for eicu_name, our_var in LAB_NAME_MAP.items():
            if our_var is None:
                continue
            mask = chunk["labname"].str.lower() == eicu_name.lower()
            if not mask.any():
                continue

            sub = chunk.loc[mask, ["patientunitstayid", "labresultoffset", "labresult"]].copy()
            sub["labresult"] = pd.to_numeric(sub["labresult"], errors="coerce")
            sub = sub.dropna(subset=["labresult"])
            sub = sub.rename(columns={
                "patientunitstayid": "stay_id",
                "labresultoffset": "offset_min",
                "labresult": "value",
            })
            sub["variable"] = our_var
            records.append(sub)

    # ── P0-3: Recover labs (especially lactate) from customLab table ──
    # The standard lab table relies on hospital-specific interfaces.
    # When the interface is unavailable, labs fall into customLab instead.
    # This particularly affects lactate (97% NaN in standard lab).
    customlab_path = EICU_ROOT / "customLab.csv.gz"
    if not customlab_path.exists():
        customlab_path = EICU_ROOT / "customlab.csv.gz"

    n_customlab_before = sum(len(r) for r in records)
    if customlab_path.exists():
        print("  [Labs] Loading customLab (recovering additional labs)...")
        # customLab schema: patientunitstayid, labotheroffset, labothervaluetype,
        #                   labothertypeid, labothershortlabel, labotherresult
        CUSTOMLAB_MAP = {
            "lactate": "lactate",
            "lactic acid": "lactate",
            "creatinine": "creatinine",
            "bun": "bun",
            "potassium": "potassium",
            "sodium": "sodium",
            "bicarbonate": "bicarbonate",
            "hco3": "bicarbonate",
            "glucose": "glucose",
            "hemoglobin": "hemoglobin",
            "hgb": "hemoglobin",
            "wbc": "wbc",
            "platelets": "platelet",
            "albumin": "albumin",
            "total bilirubin": "bilirubin",
            "troponin i": "troponin",
            "troponin t": "troponin",
            "ph": "ph",
        }
        n_custom_records = 0
        for chunk in tqdm(
            pd.read_csv(customlab_path, chunksize=500_000),
            desc="    customLab", leave=False,
        ):
            chunk = chunk[chunk["patientunitstayid"].isin(cohort_ids)]
            if chunk.empty:
                continue

            offset_col = "labotheroffset" if "labotheroffset" in chunk.columns else None
            if offset_col:
                chunk = chunk[
                    (chunk[offset_col] >= T0_MIN) &
                    (chunk[offset_col] < HORIZON_MIN)
                ]
            if chunk.empty:
                continue

            # Try both labothershortlabel and labothervaluetype for matching
            label_col = None
            for candidate in ["labothershortlabel", "labothervaluetype"]:
                if candidate in chunk.columns:
                    label_col = candidate
                    break
            if label_col is None:
                continue

            result_col = "labotherresult" if "labotherresult" in chunk.columns else None
            if result_col is None:
                continue

            labels_lower = chunk[label_col].astype(str).str.lower()
            for custom_name, our_var in CUSTOMLAB_MAP.items():
                mask = labels_lower == custom_name
                if not mask.any():
                    continue
                sub = chunk.loc[mask].copy()
                sub["_value"] = pd.to_numeric(sub[result_col], errors="coerce")
                sub = sub.dropna(subset=["_value"])
                if sub.empty:
                    continue
                rec = sub[["patientunitstayid"]].copy()
                rec = rec.rename(columns={"patientunitstayid": "stay_id"})
                rec["offset_min"] = sub[offset_col].values if offset_col else T0_MIN
                rec["value"] = sub["_value"].values
                rec["variable"] = our_var
                records.append(rec)
                n_custom_records += len(rec)

        print(f"  [Labs] customLab: {n_custom_records:,} additional records recovered")
    else:
        print("  [Labs] customLab not found, skipping P0-3")

    if records:
        labs = pd.concat(records, ignore_index=True)
        # Deduplicate: if multiple lab names map to same variable, keep first per time
        labs = labs.sort_values(["stay_id", "variable", "offset_min"])
        labs = labs.drop_duplicates(subset=["stay_id", "variable", "offset_min"], keep="first")
        n_total = len(labs)
        n_lactate = (labs["variable"] == "lactate").sum()
        print(f"  [Labs] {n_total:,} measurements extracted "
              f"(lactate: {n_lactate:,})")
        return labs
    return pd.DataFrame(columns=["stay_id", "offset_min", "value", "variable"])


def extract_treatments(cohort_ids: set) -> pd.DataFrame:
    """
    Extract treatments: IV fluids, vasopressors (NE-equiv), diuretics.
    Returns per-bin aggregated treatment data.
    """
    print("  [Treatments] Extracting...")
    treat_records = []

    # ── Vasopressors from infusionDrug ──
    inf_path = EICU_ROOT / "infusionDrug.csv.gz"
    if not inf_path.exists():
        inf_path = EICU_ROOT / "infusiondrug.csv.gz"

    if inf_path.exists():
        print("    Loading infusionDrug...")
        for chunk in tqdm(
            pd.read_csv(inf_path, chunksize=500_000),
            desc="    infusionDrug", leave=False
        ):
            chunk = chunk[chunk["patientunitstayid"].isin(cohort_ids)]
            if chunk.empty:
                continue
            # Early time-window filter
            if "infusionoffset" in chunk.columns:
                chunk = chunk[
                    (chunk["infusionoffset"] >= T0_MIN) &
                    (chunk["infusionoffset"] < HORIZON_MIN)
                ]
            if chunk.empty:
                continue

            if "drugname" in chunk.columns:
                drugname_lower = chunk["drugname"].astype(str).str.lower()
                rate_col = "drugrate" if "drugrate" in chunk.columns else "infusionrate"
                offset_col = "infusionoffset" if "infusionoffset" in chunk.columns else None

                # Vasopressors (vectorized, with unit-aware conversion)
                # eICU drugname encodes units, e.g. "Norepinephrine (mcg/min)"
                # MIMIC target: NE-equivalent mcg/kg/min
                # Must parse unit from drugname and convert accordingly:
                #   mcg/min → divide by patientweight (kg) → mcg/kg/min
                #   mcg/kg/min → use directly
                #   units/min (vasopressin) → use NE-equiv factor directly
                #   ml/hr → skip (cannot convert without concentration info)
                has_weight = "patientweight" in chunk.columns
                if has_weight:
                    chunk["_weight_kg"] = pd.to_numeric(
                        chunk["patientweight"], errors="coerce"
                    )
                    # Sanity: weight 30-300 kg; else use 80 kg default
                    chunk["_weight_kg"] = chunk["_weight_kg"].where(
                        chunk["_weight_kg"].between(30, 300), 80.0
                    )
                else:
                    chunk["_weight_kg"] = 80.0  # default fallback

                for vaso_name, ne_factor in VASO_NE_EQUIV.items():
                    mask = drugname_lower.str.contains(vaso_name, na=False)
                    if not mask.any():
                        continue
                    sub = chunk.loc[mask].copy()
                    sub["drugrate_numeric"] = parse_mixed_numeric(
                        sub[rate_col] if rate_col in sub.columns else pd.Series(dtype=object)
                    )
                    sub = sub.dropna(subset=["drugrate_numeric"])
                    if sub.empty:
                        continue

                    # Determine unit from drugname for proper conversion
                    drug_str = sub["drugname"].astype(str).str.lower()
                    is_mcg_per_min = drug_str.str.contains(r"mcg/min\)", na=False) & \
                                     ~drug_str.str.contains(r"mcg/kg/min", na=False)
                    is_mcg_per_kg_min = drug_str.str.contains(r"mcg/kg/min", na=False)
                    is_mcg_per_hr = drug_str.str.contains(r"mcg/hr", na=False)
                    is_units_per_hr = drug_str.str.contains(r"units/hr", na=False)
                    is_units_per_min = drug_str.str.contains(r"units/min", na=False)
                    is_ml_per_hr = drug_str.str.contains(r"ml/hr", na=False)

                    # Start with NaN (unknown/unconvertible)
                    ne_eq = pd.Series(np.nan, index=sub.index)

                    # mcg/min → divide by weight → mcg/kg/min → * ne_factor
                    if is_mcg_per_min.any():
                        ne_eq[is_mcg_per_min] = (
                            sub.loc[is_mcg_per_min, "drugrate_numeric"]
                            / sub.loc[is_mcg_per_min, "_weight_kg"]
                            * ne_factor
                        )
                    # mcg/kg/min → already per-kg → * ne_factor
                    if is_mcg_per_kg_min.any():
                        ne_eq[is_mcg_per_kg_min] = (
                            sub.loc[is_mcg_per_kg_min, "drugrate_numeric"]
                            * ne_factor
                        )
                    # units/min (vasopressin) → * ne_factor directly
                    if is_units_per_min.any():
                        ne_eq[is_units_per_min] = (
                            sub.loc[is_units_per_min, "drugrate_numeric"]
                            * ne_factor
                        )
                    # mcg/hr → divide by 60 and weight → mcg/kg/min
                    if is_mcg_per_hr.any():
                        ne_eq[is_mcg_per_hr] = (
                            sub.loc[is_mcg_per_hr, "drugrate_numeric"]
                            / 60.0
                            / sub.loc[is_mcg_per_hr, "_weight_kg"]
                            * ne_factor
                        )
                    # units/hr (vasopressin) → units/min → * ne_factor
                    if is_units_per_hr.any():
                        ne_eq[is_units_per_hr] = (
                            sub.loc[is_units_per_hr, "drugrate_numeric"]
                            / 60.0
                            * ne_factor
                        )
                    # ml/hr → skip (no concentration info, cannot convert)

                    # For entries with no unit pattern matched and not ml/hr,
                    # assume mcg/min (most common in eICU) and convert
                    unknown_unit = ~(is_mcg_per_min | is_mcg_per_kg_min |
                                     is_mcg_per_hr | is_units_per_hr |
                                     is_units_per_min | is_ml_per_hr)
                    if unknown_unit.any():
                        ne_eq[unknown_unit] = (
                            sub.loc[unknown_unit, "drugrate_numeric"]
                            / sub.loc[unknown_unit, "_weight_kg"]
                            * ne_factor
                        )

                    valid = ne_eq.notna()
                    if not valid.any():
                        continue
                    rec = pd.DataFrame({
                        "stay_id": sub.loc[valid, "patientunitstayid"].values,
                        "offset_min": sub.loc[valid, offset_col].values if offset_col else 0,
                        "variable": "vasopressor_ne_eq",
                        "value": ne_eq[valid].values,
                    })
                    treat_records.append(rec)

                # Diuretics in infusion form (vectorized)
                for diur_name, furo_factor in DIURETIC_FURO_EQUIV.items():
                    mask = drugname_lower.str.contains(diur_name, na=False)
                    if not mask.any():
                        continue
                    sub = chunk.loc[mask].copy()
                    sub["drugrate_numeric"] = parse_mixed_numeric(
                        sub[rate_col] if rate_col in sub.columns else pd.Series(dtype=object)
                    )
                    sub = sub.dropna(subset=["drugrate_numeric"])
                    if sub.empty:
                        continue
                    rec = pd.DataFrame({
                        "stay_id": sub["patientunitstayid"].values,
                        "offset_min": sub[offset_col].values if offset_col else 0,
                        "variable": "diuretic_furo_eq_mg",
                        "value": sub["drugrate_numeric"].values * furo_factor,
                    })
                    treat_records.append(rec)

    # ── IV Fluids from infusionDrug (rate-based continuous infusions) ──
    # eICU records continuous IV infusions in infusionDrug via drugrate (ml/hr).
    # drugname includes unit info, e.g. "NS (ml/hr)", "LR (ml/hr)", "D5W (ml/hr)".
    # volumeoffluid is BAG VOLUME (e.g., 1000ml), NOT infused volume — DO NOT use it.
    # Strategy: identify fluid-type drugs by (ml/hr) unit suffix or known fluid names,
    # exclude known medications, then convert rate → volume per entry.
    # Each infusionDrug row is a point-in-time rate snapshot. Volume is estimated as
    # rate × time_to_next_entry (capped at BIN_HOURS to avoid overestimation).
    n_inf_fluid = 0
    n_inf_fluid_stays = set()
    if inf_path.exists():
        print("    Loading infusionDrug for IV fluid rates (ml/hr)...")

        # Non-fluid drugs to exclude (medications that also have ml/hr variants)
        NON_FLUID_DRUGS = [
            "propofol", "fentanyl", "midazolam", "insulin", "heparin",
            "norepinephrine", "epinephrine", "vasopressin", "phenylephrine",
            "dopamine", "dobutamine", "milrinone", "amiodarone", "nitroglycerin",
            "furosemide", "bumetanide", "torsemide", "lasix",
            "vancomycin", "piperacillin", "ceftriaxone", "meropenem", "zosyn",
            "pantoprazole", "famotidine", "dexmedetomidine", "precedex",
            "ketamine", "rocuronium", "cisatracurium", "nimbex", "vecuronium",
            "nicardipine", "nitroprusside", "esmolol", "diltiazem", "cardizem",
            "lorazepam", "morphine", "hydromorphone", "remifentanil", "dilaudid",
            "tpn", "lipid", "intralipid", "parenteral", "octreotide",
            "amicar", "aminocaproic", "bivalirudin", "argatroban",
            "clevidipine", "labetalol", "hydralazine",
        ]

        # Known IV fluid drugname patterns (positive identification)
        IV_FLUID_NAMES = [
            r"\bns\b", r"\blr\b", r"\bd5", "saline", "sodium chloride",
            "nacl", "0\.9%", "0\.45%", "ringer", "dextrose",
            "plasmalyte", "plasma-lyte", "normosol", "isolyte",
            "albumin 5", "albumin 25",
        ]

        # Collect all fluid-type infusion entries first, then compute volumes
        fluid_inf_records = []

        for chunk in pd.read_csv(inf_path, chunksize=500_000):
            chunk = chunk[chunk["patientunitstayid"].isin(cohort_ids)]
            if chunk.empty:
                continue
            if "infusionoffset" in chunk.columns:
                chunk = chunk[
                    (chunk["infusionoffset"] >= T0_MIN) &
                    (chunk["infusionoffset"] < HORIZON_MIN)
                ]
            if chunk.empty or "drugname" not in chunk.columns:
                continue

            drugname_lower = chunk["drugname"].astype(str).str.lower()
            rate_col = "drugrate" if "drugrate" in chunk.columns else "infusionrate"

            # Strategy 1: drugname with (ml/hr) suffix — likely IV fluid
            is_ml_hr = drugname_lower.str.contains(r"\(ml/hr\)", na=False)

            # Strategy 2: known IV fluid names (any unit)
            is_named_fluid = pd.Series(False, index=chunk.index)
            for pat in IV_FLUID_NAMES:
                is_named_fluid |= drugname_lower.str.contains(pat, na=False)

            # Exclude known medications
            is_non_fluid = pd.Series(False, index=chunk.index)
            for pat in NON_FLUID_DRUGS:
                is_non_fluid |= drugname_lower.str.contains(pat, na=False)

            # Accept: (ml/hr drugs OR named fluids) AND NOT non-fluid medications
            accept = (is_ml_hr | is_named_fluid) & ~is_non_fluid

            if not accept.any():
                continue

            sub = chunk.loc[accept].copy()

            # Parse rate as ml/hr
            sub["_rate_ml_hr"] = parse_mixed_numeric(sub[rate_col])
            sub = sub[sub["_rate_ml_hr"].notna() & (sub["_rate_ml_hr"] > 0) &
                       (sub["_rate_ml_hr"] <= 2000)]  # sanity: max 2L/hr
            if sub.empty:
                continue

            fluid_inf_records.append(sub[["patientunitstayid", "infusionoffset",
                                          "_rate_ml_hr"]].copy())

        # Convert rates to volumes: for each entry, volume = rate × duration
        # Duration = time to next entry for same stay (capped at 2h = 120 min)
        if fluid_inf_records:
            fluid_df = pd.concat(fluid_inf_records, ignore_index=True)
            fluid_df = fluid_df.sort_values(["patientunitstayid", "infusionoffset"])

            # Compute time delta to next entry (within same stay)
            fluid_df["_next_offset"] = fluid_df.groupby("patientunitstayid")["infusionoffset"].shift(-1)
            fluid_df["_dt_min"] = fluid_df["_next_offset"] - fluid_df["infusionoffset"]
            # Cap at 120 min (1 bin); NaN (last entry) defaults to 60 min (1h assumed)
            fluid_df["_dt_min"] = fluid_df["_dt_min"].clip(upper=120).fillna(60)
            # Avoid very short intervals (< 1 min) which may be duplicates
            fluid_df["_dt_min"] = fluid_df["_dt_min"].clip(lower=1)

            # Volume = rate (ml/hr) × duration (hr)
            fluid_df["_vol_ml"] = fluid_df["_rate_ml_hr"] * (fluid_df["_dt_min"] / 60.0)
            # Sanity: cap per-entry volume at 2000ml
            fluid_df["_vol_ml"] = fluid_df["_vol_ml"].clip(upper=2000)

            rec = pd.DataFrame({
                "stay_id": fluid_df["patientunitstayid"].values,
                "offset_min": fluid_df["infusionoffset"].values,
                "variable": "iv_fluid_ml",
                "value": fluid_df["_vol_ml"].values,
            })
            treat_records.append(rec)
            n_inf_fluid = len(rec)
            n_inf_fluid_stays = set(fluid_df["patientunitstayid"].unique())

        print(f"    infusionDrug IV fluid: {n_inf_fluid:,} volume entries "
              f"from {len(n_inf_fluid_stays):,} stays")

    # ── IV Fluids + Urine from intakeOutput ──
    io_path = EICU_ROOT / "intakeOutput.csv.gz"
    if not io_path.exists():
        io_path = EICU_ROOT / "intakeoutput.csv.gz"

    if io_path.exists():
        print("    Loading intakeOutput...")

        # ── IV fluid classification ──
        # Strategy: two-tier matching for MIMIC-aligned IV fluid extraction.
        #
        # Tier 1: cellpath-based (most reliable).
        #   eICU cellpath is a hierarchical label like
        #   "flowsheet|Flowsheet Cell Labels|Intake|Crystalloids|..."
        #   If cellpath contains an intake-fluid keyword, accept.
        #
        # Tier 2: celllabel keyword match (fallback for sites without
        #   structured cellpath).
        #
        # Both tiers exclude medication vehicle volumes (fentanyl, propofol,
        # etc.) and non-fluid entries (body weight, tube feeding, stool).
        #
        # The previous version only used a narrow celllabel regex and missed
        # ~70% of IV fluid entries.  Key misses were:
        #   "I.V.", "Volume Given (mL)", "Volume Infused (mL)",
        #   "dextrose 5%..." labels, generic saline variants.

        # Patterns that indicate IV fluid in cellpath (case-insensitive)
        CELLPATH_FLUID_PATTERNS = [
            "crystalloid", "colloid", "iv fluid", "iv intake",
            "saline", "ringer", "dextrose", "albumin",
        ]
        cellpath_fluid_re = re.compile(
            "|".join(re.escape(p) for p in CELLPATH_FLUID_PATTERNS),
            re.IGNORECASE,
        )

        # Expanded celllabel include patterns (case-insensitive)
        IV_LABEL_INCLUDE = [
            "crystalloid", "normal saline", "saline", "sodium chloride",
            "ns ", "ns-", "nacl", "0.9%", "0.45%",
            "lactated ringer", "ringer", "lr ",
            "d5w", "d5 ", "d5%", "dextrose",
            "plasmalyte", "plasma-lyte", "plasma lyte",
            "iv fluid", "i.v.", "iv bolus", "bolus",
            "volume given", "volume infused",
            "albumin 5", "albumin 25", "albumin ",
        ]
        iv_label_re = re.compile(
            "|".join(re.escape(p) for p in IV_LABEL_INCLUDE),
            re.IGNORECASE,
        )

        # Exclusion patterns — medication vehicle volumes, non-fluids
        IV_LABEL_EXCLUDE = [
            "fentanyl", "insulin", "heparin", "propofol",
            "dexmedetomidine", "amiodarone", "midazolam", "lorazepam",
            "furosemide", "vasopressin", "norepinephrine", "epinephrine",
            "phenylephrine", "dopamine", "dobutamine", "milrinone",
            "hydromorphone", "dilaudid", "morphine", "ketamine",
            "remifentanil", "cisatracurium", "vecuronium", "rocuronium",
            "nitroglycerin", "nitroprusside", "nicardipine", "clevidipine",
            "lipid", "tpn", "parenteral nutrition", "intralipid",
            "potassium chloride", "magnesium", "calcium gluconate",
            "sodium bicarbonate",
            "tube feeding", "residual", "flush",
            "blood", "transfuse", "rbc", "ffp", "platelet",
            "bodyweight", "weight",
            "urine", "foley", "void", "stool", "emesis",
            "drain", "chest tube", "removed",
            "dialys", "cvvh", "crrt",
            "p.o.", "oral",
        ]
        iv_exclude_re = re.compile(
            "|".join(re.escape(p) for p in IV_LABEL_EXCLUDE),
            re.IGNORECASE,
        )

        iv_diag_counts = {"cellpath_hit": 0, "celllabel_hit": 0,
                          "excluded": 0, "total_rows": 0}

        for chunk in tqdm(
            pd.read_csv(io_path, chunksize=500_000),
            desc="    intakeOutput", leave=False
        ):
            chunk = chunk[chunk["patientunitstayid"].isin(cohort_ids)]
            if chunk.empty:
                continue
            # Early time-window filter
            if "intakeoutputoffset" in chunk.columns:
                chunk = chunk[
                    (chunk["intakeoutputoffset"] >= T0_MIN) &
                    (chunk["intakeoutputoffset"] < HORIZON_MIN)
                ]
            if chunk.empty:
                continue

            label_col = "celllabel" if "celllabel" in chunk.columns else None
            if label_col is None:
                continue

            labels_lower = chunk[label_col].astype(str).str.lower()
            offset_col_io = "intakeoutputoffset" if "intakeoutputoffset" in chunk.columns else None
            vol_col = "cellvaluenumeric" if "cellvaluenumeric" in chunk.columns else "cellvalue"

            iv_diag_counts["total_rows"] += len(chunk)

            # ── Tier 1: cellpath-based classification ──
            cellpath_mask = pd.Series(False, index=chunk.index)
            if "cellpath" in chunk.columns:
                cp_lower = chunk["cellpath"].astype(str).str.lower()
                # Must be intake path AND contain a fluid keyword
                is_intake = cp_lower.str.contains("intake", na=False)
                has_fluid_kw = cp_lower.str.contains(cellpath_fluid_re, na=False)
                cellpath_mask = is_intake & has_fluid_kw

            # ── Tier 2: celllabel keyword match ──
            label_include = labels_lower.str.contains(iv_label_re, na=False)

            # ── Combined include (either tier) ──
            include_mask = cellpath_mask | label_include

            # ── Apply exclusions ──
            exclude_mask = labels_lower.str.contains(iv_exclude_re, na=False)
            iv_mask = include_mask & ~exclude_mask

            iv_diag_counts["cellpath_hit"] += int(cellpath_mask.sum())
            iv_diag_counts["celllabel_hit"] += int((label_include & ~cellpath_mask).sum())
            iv_diag_counts["excluded"] += int((include_mask & exclude_mask).sum())

            if iv_mask.any():
                iv_sub = chunk.loc[iv_mask].copy()
                iv_sub["volume"] = parse_mixed_numeric(iv_sub[vol_col])
                iv_sub = iv_sub.dropna(subset=["volume"])
                # Sanity: IV fluid volumes should be positive and < 10L per entry
                iv_sub = iv_sub[(iv_sub["volume"] > 0) & (iv_sub["volume"] <= 10_000)]
                if not iv_sub.empty:
                    rec = pd.DataFrame({
                        "stay_id": iv_sub["patientunitstayid"].values,
                        "offset_min": iv_sub[offset_col_io].values if offset_col_io else 0,
                        "variable": "iv_fluid_ml",
                        "value": iv_sub["volume"].values,
                    })
                    treat_records.append(rec)

            # Urine output — vectorized
            urine_regex = "urine|foley|void"
            urine_mask = labels_lower.str.contains(urine_regex, na=False)
            if urine_mask.any():
                u_sub = chunk.loc[urine_mask].copy()
                u_sub["volume"] = parse_mixed_numeric(u_sub[vol_col])
                u_sub = u_sub.dropna(subset=["volume"])
                if not u_sub.empty:
                    rec = pd.DataFrame({
                        "stay_id": u_sub["patientunitstayid"].values,
                        "offset_min": u_sub[offset_col_io].values if offset_col_io else 0,
                        "variable": "urine_output_ml",
                        "value": u_sub["volume"].abs().values,
                    })
                    treat_records.append(rec)

        # Diagnostic summary for IV fluid extraction
        print(f"    IV fluid extraction diagnostics:")
        print(f"      Total intakeOutput rows in window: {iv_diag_counts['total_rows']:,}")
        print(f"      cellpath-based hits:  {iv_diag_counts['cellpath_hit']:,}")
        print(f"      celllabel-based hits: {iv_diag_counts['celllabel_hit']:,}")
        print(f"      Excluded (med vehicle/non-fluid): {iv_diag_counts['excluded']:,}")
        print(f"      infusionDrug IV fluid records:    {n_inf_fluid:,}")

    # ── Diuretics from medication table ──
    med_path = EICU_ROOT / "medication.csv.gz"
    if med_path.exists():
        print("    Loading medication...")
        for chunk in tqdm(
            pd.read_csv(med_path, chunksize=500_000),
            desc="    medication", leave=False
        ):
            chunk = chunk[chunk["patientunitstayid"].isin(cohort_ids)]
            if chunk.empty:
                continue
            # Early time-window filter
            offset_key = "drugstartoffset" if "drugstartoffset" in chunk.columns else "drugorderoffset"
            if offset_key in chunk.columns:
                chunk = chunk[
                    (chunk[offset_key] >= T0_MIN) &
                    (chunk[offset_key] < HORIZON_MIN)
                ]
            if chunk.empty:
                continue

            if "drugname" not in chunk.columns:
                continue

            drugname_lower = chunk["drugname"].astype(str).str.lower()
            dose_col = "dosage" if "dosage" in chunk.columns else None
            med_offset_col = "drugstartoffset" if "drugstartoffset" in chunk.columns else "drugorderoffset"
            for diur_name, furo_factor in DIURETIC_FURO_EQUIV.items():
                mask = drugname_lower.str.contains(diur_name, na=False)
                if not mask.any():
                    continue
                sub = chunk.loc[mask].copy()
                if dose_col:
                    sub["dose_num"] = parse_mixed_numeric(sub[dose_col])
                    sub = sub.dropna(subset=["dose_num"])
                    if sub.empty:
                        continue
                    rec = pd.DataFrame({
                        "stay_id": sub["patientunitstayid"].values,
                        "offset_min": sub[med_offset_col].values if med_offset_col in sub.columns else 0,
                        "variable": "diuretic_furo_eq_mg",
                        "value": sub["dose_num"].values * furo_factor,
                    })
                    treat_records.append(rec)

    # NOTE: P0-1 treatment table vasopressor extraction REVERTED (2026-03-12).
    # Reason: treatment table provides only binary exposure, and the fixed
    # DEFAULT_VASO_NE_EQ=0.05 mcg/kg/min imputed value introduces noise —
    # model sees "vaso=0.05" for stays where actual rate may be 0.01 or 0.2.
    # The 5,024 affected stays are better classified as "no measured vaso"
    # (which correctly maps to action_idx vaso=none) than noise-injected.

    if treat_records:
        treatments = pd.concat(treat_records, ignore_index=True)
        print(f"  [Treatments] {len(treatments):,} records extracted")
        return treatments
    return pd.DataFrame(columns=["stay_id", "offset_min", "value", "variable"])


def grid_temporal_data(
    vitals: pd.DataFrame,
    labs: pd.DataFrame,
    treatments: pd.DataFrame,
    cohort_ids: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Grid all temporal data into 2-hour bins from T0 (6h post-admission).

    Returns: (grid_df, missingness_df)
    """
    print("\n  Gridding temporal data into 2h bins...")

    # Combine all measurements
    all_data = pd.concat([vitals, labs, treatments], ignore_index=True)
    all_data = all_data[all_data["stay_id"].isin(set(cohort_ids))]

    # Convert offset_min to hours from admission
    all_data["hours_from_admit"] = all_data["offset_min"] / 60.0

    # Filter to T0 → T0+24h window
    all_data = all_data[
        (all_data["hours_from_admit"] >= T0_HOURS) &
        (all_data["hours_from_admit"] < HORIZON_HOURS)
    ]

    # Assign to 2-hour bins
    all_data["bin"] = ((all_data["hours_from_admit"] - T0_HOURS) / BIN_HOURS).astype(int)
    all_data = all_data[(all_data["bin"] >= 0) & (all_data["bin"] < N_BINS)]

    print(f"  Measurements in window: {len(all_data):,}")

    # ── Build grid: stay_id × bin × variable → mean value ──
    # State variables: take mean per bin
    state_data = all_data[all_data["variable"].isin(STATE_VARS)]
    state_pivot = state_data.groupby(["stay_id", "bin", "variable"])["value"].mean().reset_index()
    state_wide = state_pivot.pivot_table(
        index=["stay_id", "bin"], columns="variable", values="value"
    ).reset_index()

    # Treatment variables: aggregate per bin
    #   - Volumes/doses (iv_fluid_ml, diuretic_furo_eq_mg, urine_output_ml): SUM
    #   - Rates (vasopressor_ne_eq, in mcg/kg/min): MEAN (not sum!)
    #     Summing rates inflates values when multiple readings exist per bin.
    treat_vars = ["iv_fluid_ml", "vasopressor_ne_eq", "diuretic_furo_eq_mg", "urine_output_ml"]
    treat_data = all_data[all_data["variable"].isin(treat_vars)]

    # Split rate vs volume variables for correct aggregation
    rate_vars = {"vasopressor_ne_eq"}
    vol_vars = set(treat_vars) - rate_vars

    treat_parts = []
    # Volume variables: sum per bin
    vol_data = treat_data[treat_data["variable"].isin(vol_vars)]
    if not vol_data.empty:
        vol_pivot = vol_data.groupby(["stay_id", "bin", "variable"])["value"].sum().reset_index()
        treat_parts.append(vol_pivot)
    # Rate variables: mean per bin (matches MIMIC 40b time-weighted average approach)
    rate_data = treat_data[treat_data["variable"].isin(rate_vars)]
    if not rate_data.empty:
        rate_pivot = rate_data.groupby(["stay_id", "bin", "variable"])["value"].mean().reset_index()
        treat_parts.append(rate_pivot)

    if treat_parts:
        treat_pivot = pd.concat(treat_parts, ignore_index=True)
    else:
        treat_pivot = pd.DataFrame(columns=["stay_id", "bin", "variable", "value"])
    treat_wide = treat_pivot.pivot_table(
        index=["stay_id", "bin"], columns="variable", values="value"
    ).reset_index()

    # Create full grid skeleton (vectorized)
    print("  Building grid skeleton...")
    idx = pd.MultiIndex.from_product(
        [cohort_ids, range(N_BINS)], names=["stay_id", "bin"]
    )
    grid = pd.DataFrame(index=idx).reset_index()

    # Merge state + treatment data
    grid = grid.merge(state_wide, on=["stay_id", "bin"], how="left")
    grid = grid.merge(treat_wide, on=["stay_id", "bin"], how="left")

    # Fill treatment NaN with 0 (no treatment = zero)
    for tc in treat_vars:
        if tc in grid.columns:
            grid[tc] = grid[tc].fillna(0.0)

    # Add time columns
    grid["bin_start_h"] = T0_HOURS + grid["bin"] * BIN_HOURS
    grid["bin_end_h"] = grid["bin_start_h"] + BIN_HOURS

    # ── Compute fluid balance ──
    if "iv_fluid_ml" in grid.columns and "urine_output_ml" in grid.columns:
        grid["fluid_balance_ml"] = grid["iv_fluid_ml"] - grid["urine_output_ml"]
        grid["fluid_balance_L"] = grid["fluid_balance_ml"] / 1000.0
        grid["cumulative_fluid_balance_L"] = grid.groupby("stay_id")["fluid_balance_L"].cumsum()

    # ── Compute freq_24h: rolling 24h bin-presence count ─────────────────
    # MIMIC 40c definition: rolling sum of BINARY MASK (0/1 per bin) over
    # 12 bins. This counts "how many of the last 12 bins had at least one
    # observation", NOT the raw observation count. Range: [0, 12].
    print("  Computing freq_24h (rolling 24h bin-presence, matching MIMIC 40c)...")
    grid = grid.sort_values(["stay_id", "bin"]).reset_index(drop=True)

    for var in STATE_VARS:
        tmp_col = f"_mask_tmp_{var}"
        if var in grid.columns:
            grid[tmp_col] = grid[var].notna().astype(int)
        else:
            grid[tmp_col] = 0
        grid[f"{var}_freq_24h"] = (
            grid.groupby("stay_id")[tmp_col]
            .transform(lambda x: x.rolling(N_BINS, min_periods=1).sum())
            .astype(int)
        )
        grid.drop(columns=[tmp_col], inplace=True)

    freq_cols = [c for c in grid.columns if c.endswith("_freq_24h")]
    print(f"  freq_24h columns added: {len(freq_cols)}")
    for var in STATE_VARS[:5]:
        if f"{var}_freq_24h" in grid.columns:
            mean_freq = grid[f"{var}_freq_24h"].mean()
            print(f"    {var}_freq_24h: mean={mean_freq:.1f} (MIMIC range: [0, {N_BINS}])")
    print(f"    ... ({len(STATE_VARS) - 5} more)")

    # ── Build missingness features (vectorized) ──
    print("  Computing missingness features...")
    # Ensure grid is sorted for correct forward-fill
    grid = grid.sort_values(["stay_id", "bin"]).reset_index(drop=True)
    miss_cols = []
    for var in tqdm(STATE_VARS, desc="  Missingness features"):
        if var not in grid.columns:
            continue
        # Binary mask: 1 = observed, 0 = missing
        grid[f"{var}_mask"] = grid[var].notna().astype(float)
        miss_cols.append(f"{var}_mask")

        # Time since last measurement: forward-fill the bin number where observed
        grid[f"_obs_bin_{var}"] = np.where(grid[var].notna(), grid["bin"], np.nan)
        grid[f"_obs_bin_{var}"] = grid.groupby("stay_id")[f"_obs_bin_{var}"].ffill()
        grid[f"{var}_tsl_h"] = (grid["bin"] - grid[f"_obs_bin_{var}"]) * BIN_HOURS
        # Leave NaN for never-measured; DTRDataset nan_to_num → 0.0
        # (matches MIMIC 40c behaviour where tsl_h stays NaN until DTRDataset)
        grid.drop(columns=[f"_obs_bin_{var}"], inplace=True)
        miss_cols.append(f"{var}_tsl_h")

        # Never measured flag (per stay)
        never = grid.groupby("stay_id")[f"{var}_mask"].transform("max") == 0
        grid[f"{var}_never_measured"] = never.astype(float)
        miss_cols.append(f"{var}_never_measured")

    # Separate missingness features
    miss_df = grid[["stay_id", "bin"] + miss_cols].copy()

    # Report coverage
    print("\n  Variable coverage (% of stay×bin with at least one observation):")
    for var in STATE_VARS:
        if f"{var}_mask" in grid.columns:
            cov = grid[f"{var}_mask"].mean() * 100
            print(f"    {var:20s}: {cov:5.1f}%")

    print(f"\n  Grid: {len(grid):,} rows ({len(cohort_ids):,} stays × {N_BINS} bins)")

    return grid, miss_df


def main():
    print("=" * 72)
    print("90b_02 — Extract eICU Temporal Data")
    print("=" * 72)
    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 1. Load cohort ───────────────────────────────────────────────────
    print("\n[1] Loading eICU cohort...")
    cohort = pd.read_parquet(STAGING_DIR / "90b_eicu_cohort_v1.parquet")
    cohort_ids = cohort["patientunitstayid"].unique()
    cohort_id_set = set(cohort_ids)
    print(f"  Cohort: {len(cohort_ids):,} stays")

    # ── 2. Extract vitals ────────────────────────────────────────────────
    print("\n[2] Extracting vitals...")
    vitals = extract_vitals(cohort_id_set)

    # ── 3. Extract labs ──────────────────────────────────────────────────
    print("\n[3] Extracting labs...")
    labs = extract_labs(cohort_id_set)

    # ── 4. Extract treatments ────────────────────────────────────────────
    print("\n[4] Extracting treatments...")
    treatments = extract_treatments(cohort_id_set)

    # ── 5. Grid into 2h bins ─────────────────────────────────────────────
    print("\n[5] Gridding into 2h bins...")
    grid, miss_df = grid_temporal_data(vitals, labs, treatments, cohort_ids)

    # ── 6. Save ──────────────────────────────────────────────────────────
    print("\n[6] Saving...")
    grid_path = INTERIM_DIR / "90b_eicu_grid2h_states.parquet"
    grid.to_parquet(grid_path, index=False, engine="pyarrow")
    print(f"  Grid: {grid_path}")

    miss_path = INTERIM_DIR / "90b_eicu_missingness_features.parquet"
    miss_df.to_parquet(miss_path, index=False, engine="pyarrow")
    print(f"  Missingness: {miss_path}")

    # ── 7. Summary report ────────────────────────────────────────────────
    report_lines = [
        "=" * 72,
        "90b_02 — eICU Temporal Data Extraction Summary",
        "=" * 72,
        f"Generated: {ts}",
        "",
        "DATA SOURCES",
        "────────────",
        f"  Vitals:     {len(vitals):>10,} measurements",
        f"  Labs:       {len(labs):>10,} measurements",
        f"  Treatments: {len(treatments):>10,} records",
        "",
        "GRID",
        "────",
        f"  Stays: {len(cohort_ids):,}",
        f"  Bins: {N_BINS} (2h each, T0=6h → T0+24h)",
        f"  Grid rows: {len(grid):,}",
        "",
        "VARIABLE COVERAGE",
        "─────────────────",
    ]
    for var in STATE_VARS:
        if f"{var}_mask" in grid.columns:
            cov = grid[f"{var}_mask"].mean() * 100
            report_lines.append(f"  {var:20s}: {cov:5.1f}%")

    report_lines.extend([
        "",
        "TREATMENT COVERAGE",
        "──────────────────",
    ])
    for tc in ["iv_fluid_ml", "vasopressor_ne_eq", "diuretic_furo_eq_mg", "urine_output_ml"]:
        if tc in grid.columns:
            has = (grid[tc] > 0).mean() * 100
            mean_vol = grid.loc[grid[tc] > 0, tc].mean() if has > 0 else 0
            report_lines.append(f"  {tc:30s}: {has:5.1f}% of bins non-zero, "
                                f"mean(>0)={mean_vol:.1f}")

    # IV fluid distribution diagnostic (critical for action encoding alignment)
    if "iv_fluid_ml" in grid.columns:
        iv = grid["iv_fluid_ml"]
        report_lines.extend([
            "",
            "IV FLUID DISTRIBUTION DIAGNOSTIC",
            "─────────────────────────────────",
            f"  Bins with iv_fluid > 0:     {(iv > 0).mean()*100:.1f}%",
            f"  Bins with iv_fluid > 250ml: {(iv > 250).mean()*100:.1f}%",
            f"  Bins with iv_fluid > 1000ml:{(iv > 1000).mean()*100:.1f}%",
            f"  Mean (all bins):            {iv.mean():.1f} ml",
            f"  Mean (non-zero bins):       {iv[iv > 0].mean():.1f} ml" if (iv > 0).any() else "  Mean (non-zero): N/A",
            f"  Median (non-zero bins):     {iv[iv > 0].median():.1f} ml" if (iv > 0).any() else "  Median (non-zero): N/A",
            f"  Expected action distribution from these thresholds:",
            f"    a0 (fluid=0, <250ml):  {(iv < 250).mean()*100:.1f}%",
            f"    a1 (fluid=1, 250-1000):{((iv >= 250) & (iv <= 1000)).mean()*100:.1f}%",
            f"    a2 (fluid=2, >1000ml): {(iv > 1000).mean()*100:.1f}%",
        ])

    # Vasopressor distribution diagnostic
    if "vasopressor_ne_eq" in grid.columns:
        vaso = grid["vasopressor_ne_eq"]
        report_lines.extend([
            "",
            "VASOPRESSOR NE-EQUIV DISTRIBUTION DIAGNOSTIC",
            "─────────────────────────────────────────────",
            f"  Bins with vaso > 0:          {(vaso > 0).mean()*100:.1f}%",
            f"  Bins with vaso > 0.001:      {(vaso > 0.001).mean()*100:.1f}%",
            f"  Bins with vaso > 0.1:        {(vaso > 0.1).mean()*100:.1f}%",
            f"  Mean (all bins):             {vaso.mean():.4f} mcg/kg/min",
            f"  Mean (non-zero):             {vaso[vaso > 0].mean():.4f}" if (vaso > 0).any() else "  Mean (non-zero): N/A",
            f"  Median (non-zero):           {vaso[vaso > 0].median():.4f}" if (vaso > 0).any() else "  Median (non-zero): N/A",
            f"  P95 (non-zero):              {vaso[vaso > 0].quantile(0.95):.4f}" if (vaso > 0).any() else "  P95: N/A",
            f"  Expected action level distribution:",
            f"    vaso=0 (none, <0.001):     {(vaso < 0.001).mean()*100:.1f}%",
            f"    vaso=1 (low, 0.001-0.1):   {((vaso >= 0.001) & (vaso < 0.1)).mean()*100:.1f}%",
            f"    vaso=2 (high, >=0.1):      {(vaso >= 0.1).mean()*100:.1f}%",
        ])

    # Diuretic distribution diagnostic
    if "diuretic_furo_eq_mg" in grid.columns:
        diur = grid["diuretic_furo_eq_mg"]
        report_lines.extend([
            "",
            "DIURETIC FURO-EQUIV DISTRIBUTION DIAGNOSTIC",
            "────────────────────────────────────────────",
            f"  Bins with diuretic > 0:      {(diur > 0).mean()*100:.1f}%",
            f"  Mean (all bins):             {diur.mean():.2f} mg",
            f"  Mean (non-zero):             {diur[diur > 0].mean():.2f}" if (diur > 0).any() else "  Mean (non-zero): N/A",
        ])

    report_lines.extend([
        "",
        "NEXT STEPS",
        "──────────",
        "  90b_03: Build state tensors with MIMIC normalization",
    ])

    report_path = REPORTS_DIR / "90b_02_eicu_temporal_extraction_summary.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\n  Report: {report_path}")

    total_time = time.time() - t0
    print()
    print("=" * 72)
    print(f"90b_02 COMPLETE. Grid: {len(grid):,} rows. Time: {total_time/60:.1f} min")
    print("=" * 72)


if __name__ == "__main__":
    main()
