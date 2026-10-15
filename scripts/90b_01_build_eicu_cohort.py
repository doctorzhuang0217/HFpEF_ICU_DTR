#!/usr/bin/env python3
"""90b_01 — Build eICU Cohort for External Validation

Protocol: v3.1, Step 90b (sub-step 1)
Compute: C2

Purpose
-------
Build the eICU-CRD cohort for external validation of frozen MIMIC models.
Mirrors MIMIC Steps 10a → 20a → 25b → 25f but adapted for eICU schema.

Key differences from MIMIC:
  1. HFpEF ascertainment is BROADER-COMPATIBLE (29d):
     HF ICD diagnosis + NOT known systolic HF (ICD 428.2x/I50.2x)
  2. No discharge notes → no NLP EF extraction
  3. Patient table structure differs (offsets instead of timestamps)
  4. 28-day mortality derived from hospitaldischargestatus + offset
  5. Demographics: age is string ("> 89" for elderly), handle conversion

Cohort definition (mirrors MIMIC):
  1. Adult ICU stays (age ≥ 18)
  2. ICU LOS ≥ 30h (T0=6h + 24h horizon)
  3. HF phenotyping: HFpEF (broader), HFrEF, no_HF
  4. Pre-T0 hemodynamic gate (at least 1 MAP reading in first 6h)

Inputs
------
- data/raw/eicu/2.0/patient.csv.gz
- data/raw/eicu/2.0/diagnosis.csv.gz
- data/raw/eicu/2.0/apachePatientResult.csv.gz (mortality)

Outputs
-------
- data/staging/90b_eicu_icu_stays.parquet
- data/staging/90b_eicu_hf_phenotype.parquet
- data/staging/90b_eicu_cohort_v1.parquet
- reports/90b_01_eicu_cohort_summary.txt
- tables/90b_01_eicu_cohort_counts.csv

Dependencies
------------
- 90b_00 report reviewed (eICU schema verified)
- 90a model registry frozen
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
EICU_ROOT = ROOT / "data" / "raw" / "eicu" / "2.0"
STAGING_DIR = ROOT / "data" / "staging"
TABLES_DIR = ROOT / "tables"
REPORTS_DIR = ROOT / "reports"

for d in [STAGING_DIR, TABLES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── HF ICD code patterns (29d broader-compatible) ───────────────────────
# eICU diagnosis table uses `diagnosisstring` (text path) and/or `icd9code`
# We search both columns for HF patterns

HF_ICD9 = ["428"]                     # All HF codes
HF_ICD10 = ["I50"]                    # All HF codes
SYSTOLIC_ICD9 = ["428.2"]             # Systolic HF → exclude from HFpEF
SYSTOLIC_ICD10 = ["I50.2"]            # Systolic HF → exclude from HFpEF
DIASTOLIC_ICD9 = ["428.3"]            # Diastolic HF → HFpEF proxy
DIASTOLIC_ICD10 = ["I50.3"]           # Diastolic HF → HFpEF proxy
UNSPECIFIED_ICD9 = ["428.0"]         # Unspecified HF
UNSPECIFIED_ICD10 = ["I50.9"]         # Unspecified HF

# diagnosisstring-based patterns (eICU uses hierarchical paths)
HF_TEXT_PATTERNS = [
    "heart failure",
    "congestive heart failure",
    "CHF",
]
SYSTOLIC_TEXT_PATTERNS = [
    "systolic heart failure",
    "systolic dysfunction",
    "reduced ejection",
    "low ejection",
]
DIASTOLIC_TEXT_PATTERNS = [
    "diastolic heart failure",
    "diastolic dysfunction",
    "preserved ejection",
    "HFpEF",
]


def load_eicu_patients() -> pd.DataFrame:
    """Load and preprocess eICU patient table."""
    print("  Loading patient table...")
    pt = pd.read_csv(EICU_ROOT / "patient.csv.gz")

    # Convert age (string in eICU: "> 89" for elderly)
    pt["age_num"] = pd.to_numeric(
        pt["age"].replace("> 89", 90).replace("", np.nan),
        errors="coerce"
    )

    # Gender encoding
    pt["is_female"] = (pt["gender"] == "Female").astype(int)

    # ICU LOS in hours (unitdischargeoffset is in minutes from unit admit)
    pt["icu_los_hours"] = pt["unitdischargeoffset"] / 60.0

    # Hospital LOS in hours
    pt["hosp_los_hours"] = pt["hospitaldischargeoffset"] / 60.0

    # ICU mortality
    pt["icu_mortality"] = (pt["unitdischargestatus"] == "Expired").astype(int)

    # Hospital mortality
    pt["hospital_mortality"] = (pt["hospitaldischargestatus"] == "Expired").astype(int)

    # 28-day mortality approximation:
    # eICU doesn't have post-discharge follow-up.
    # Best available: hospital mortality within 28 days of ICU admission.
    # hospitaldischargeoffset (minutes from unit admission to hospital discharge)
    # If hospital mortality AND hospitaldischargeoffset <= 28*24*60 → dead within 28d
    # If alive at hospital discharge, assume alive at 28d (underestimate)
    pt["hosp_discharge_days"] = pt["hospitaldischargeoffset"] / (60 * 24)
    pt["mortality_28d"] = (
        (pt["hospital_mortality"] == 1) &
        (pt["hosp_discharge_days"] <= 28)
    ).astype(int)
    # Also count those who died in hospital even if offset > 28 days as hospital death
    # (conservative: some may have died within 28d but discharged offset recorded differently)

    # Admission type (emergency vs elective)
    pt["is_emergency"] = pt["unitadmitsource"].isin([
        "Emergency Department", "Direct Admit", "Other Hospital",
        "Operating Room", "Floor",
    ]).astype(int)

    print(f"  Loaded {len(pt):,} ICU stays, {pt['uniquepid'].nunique():,} patients")
    return pt


def phenotype_hf(pt: pd.DataFrame) -> pd.DataFrame:
    """
    Assign HF phenotype using eICU diagnosis table (29d broader-compatible).

    Strategy:
      1. Search diagnosis table for HF ICD codes + text patterns
      2. Classify: hfpef_broader, hfref, hf_unspecified, no_hf
      3. broader-compatible HFpEF = HF dx + NOT known systolic HF
    """
    print("  Loading diagnosis table...")
    diag = pd.read_csv(EICU_ROOT / "diagnosis.csv.gz")
    print(f"  Diagnosis table: {len(diag):,} rows")

    # Identify columns for ICD codes and diagnosis text
    icd_col = "icd9code" if "icd9code" in diag.columns else None
    text_col = "diagnosisstring" if "diagnosisstring" in diag.columns else None

    all_stay_ids = set(pt["patientunitstayid"].unique())
    hf_stays = set()
    systolic_stays = set()
    diastolic_stays = set()
    unspecified_hf_stays = set()

    # Search ICD codes
    if icd_col and icd_col in diag.columns:
        codes = diag[icd_col].astype(str).fillna("")

        for pattern in HF_ICD9 + HF_ICD10:
            mask = codes.str.startswith(pattern)
            hf_stays |= set(diag.loc[mask, "patientunitstayid"].unique())

        for pattern in SYSTOLIC_ICD9 + SYSTOLIC_ICD10:
            mask = codes.str.startswith(pattern)
            systolic_stays |= set(diag.loc[mask, "patientunitstayid"].unique())

        for pattern in DIASTOLIC_ICD9 + DIASTOLIC_ICD10:
            mask = codes.str.startswith(pattern)
            diastolic_stays |= set(diag.loc[mask, "patientunitstayid"].unique())

        for pattern in UNSPECIFIED_ICD9 + UNSPECIFIED_ICD10:
            mask = codes.str.startswith(pattern)
            unspecified_hf_stays |= set(diag.loc[mask, "patientunitstayid"].unique())

    # Search diagnosis text
    if text_col and text_col in diag.columns:
        text = diag[text_col].astype(str).fillna("").str.lower()

        for pattern in HF_TEXT_PATTERNS:
            mask = text.str.contains(pattern.lower(), na=False)
            hf_stays |= set(diag.loc[mask, "patientunitstayid"].unique())

        for pattern in SYSTOLIC_TEXT_PATTERNS:
            mask = text.str.contains(pattern.lower(), na=False)
            systolic_stays |= set(diag.loc[mask, "patientunitstayid"].unique())

        for pattern in DIASTOLIC_TEXT_PATTERNS:
            mask = text.str.contains(pattern.lower(), na=False)
            diastolic_stays |= set(diag.loc[mask, "patientunitstayid"].unique())

    # Filter to stays in our patient table
    hf_stays &= all_stay_ids
    systolic_stays &= all_stay_ids
    diastolic_stays &= all_stay_ids
    unspecified_hf_stays &= all_stay_ids

    print(f"  HF stays: {len(hf_stays):,}")
    print(f"  Systolic HF stays: {len(systolic_stays):,}")
    print(f"  Diastolic HF stays: {len(diastolic_stays):,}")
    print(f"  Unspecified HF stays: {len(unspecified_hf_stays):,}")

    # Assign phenotype per 29d broader-compatible strategy
    # Priority: diastolic/systolic-specific > unspecified HF > no HF
    pheno = pd.DataFrame({"patientunitstayid": pt["patientunitstayid"].unique()})
    pheno["has_hf"] = pheno["patientunitstayid"].isin(hf_stays).astype(int)
    pheno["has_systolic"] = pheno["patientunitstayid"].isin(systolic_stays).astype(int)
    pheno["has_diastolic"] = pheno["patientunitstayid"].isin(diastolic_stays).astype(int)
    pheno["has_unspecified_hf"] = pheno["patientunitstayid"].isin(unspecified_hf_stays).astype(int)

    # Broader-compatible HFpEF: HF + NOT known systolic
    conditions = [
        pheno["has_diastolic"] == 1,                           # Diastolic HF → HFpEF
        (pheno["has_hf"] == 1) & (pheno["has_systolic"] == 0) & (pheno["has_diastolic"] == 0),  # HF, not systolic → HFpEF (broader)
        pheno["has_systolic"] == 1,                            # Systolic HF → HFrEF
        pheno["has_hf"] == 0,                                  # No HF
    ]
    choices = ["hfpef", "hfpef", "hfref", "no_hf"]
    pheno["hf_subtype"] = np.select(conditions, choices, default="no_hf")

    # Comparator group (matching MIMIC schema)
    pheno["hfpef_broad_flag"] = ((pheno["has_hf"] == 1) & (pheno["has_systolic"] == 0)).astype(int)
    pheno["hfpef_diastolic_strict_flag"] = (pheno["has_diastolic"] == 1).astype(int)

    pheno["comparator_group"] = pheno["hf_subtype"].map({
        "hfpef": "hfpef",
        "hfref": "hfref",
        "no_hf": "no_hf",
    }).fillna("no_hf")

    # Report
    counts = pheno["hf_subtype"].value_counts()
    print("\n  HF phenotype distribution:")
    for label, n in counts.items():
        pct = n / len(pheno) * 100
        print(f"    {label:20s}: {n:>7,} ({pct:5.1f}%)")

    return pheno


def build_exclusion_flags(pt: pd.DataFrame) -> pd.DataFrame:
    """
    Build hard exclusion flags mirroring MIMIC 20e pipeline.

    Uses eICU diagnosis table ICD codes for:
      1. LVAD (left ventricular assist device)
      2. ECMO
      3. Cardiac transplant
      4. Pregnancy
      5. Chronic dialysis / ESRD
      6. Hemorrhagic shock / massive trauma

    Also checks:
      7. T0-ineligible (died or discharged < 6h into ICU)
    """
    print("\n  Building hard exclusion flags (mirroring MIMIC 20e)...")

    # Load diagnosis table for ICD-based exclusions
    diag = pd.read_csv(EICU_ROOT / "diagnosis.csv.gz")
    icd_col = "icd9code" if "icd9code" in diag.columns else None
    text_col = "diagnosisstring" if "diagnosisstring" in diag.columns else None

    stay_ids = set(pt["patientunitstayid"].unique())

    # ICD code patterns for exclusions (ICD-9 + ICD-10 mixed in eICU)
    # LVAD
    LVAD_CODES = ["Z95811", "Z95812", "V4509"]
    LVAD_TEXT = ["ventricular assist", "lvad", "left ventricular assist"]

    # ECMO
    ECMO_TEXT = ["ecmo", "extracorporeal membrane"]

    # Cardiac transplant
    TRANSPLANT_CODES = ["Z941", "V421"]
    TRANSPLANT_TEXT = ["heart transplant", "cardiac transplant"]

    # Pregnancy
    PREGNANCY_ICD9_PREFIXES = ["V22", "V23", "V27", "V28",
                                "63", "64", "65", "66", "67", "68", "69"]
    PREGNANCY_ICD10_PREFIXES = ["O", "Z33", "Z34", "Z3A"]
    PREGNANCY_TEXT = ["pregnan", "obstetric", "gestational"]

    # Chronic dialysis / ESRD
    DIALYSIS_CODES = ["N186", "5856", "Z49", "Z992", "V451", "V560", "V561"]
    DIALYSIS_TEXT = ["end stage renal", "esrd", "chronic dialysis",
                     "hemodialysis", "peritoneal dialysis"]

    # Hemorrhagic shock / massive trauma
    HEMORRHAGE_TEXT = ["hemorrhagic shock", "massive hemorrhage",
                       "massive transfusion"]

    # Initialize flags
    flags = pd.DataFrame({"patientunitstayid": pt["patientunitstayid"].unique()})
    for col in ["excl_lvad", "excl_ecmo", "excl_transplant", "excl_pregnancy",
                "excl_dialysis", "excl_hemorrhage", "excl_t0_ineligible"]:
        flags[col] = 0

    # Helper: find stays matching ICD code prefixes
    def stays_matching_icd(prefixes):
        if not icd_col or icd_col not in diag.columns:
            return set()
        codes = diag[icd_col].astype(str).fillna("")
        matched = set()
        for prefix in prefixes:
            mask = codes.str.startswith(prefix)
            matched |= set(diag.loc[mask, "patientunitstayid"].unique())
        return matched & stay_ids

    # Helper: find stays matching text patterns in diagnosisstring
    def stays_matching_text(patterns):
        if not text_col or text_col not in diag.columns:
            return set()
        text = diag[text_col].astype(str).str.lower()
        matched = set()
        for pattern in patterns:
            mask = text.str.contains(pattern.lower(), na=False)
            matched |= set(diag.loc[mask, "patientunitstayid"].unique())
        return matched & stay_ids

    # 1. LVAD
    lvad = stays_matching_icd(LVAD_CODES) | stays_matching_text(LVAD_TEXT)
    flags.loc[flags["patientunitstayid"].isin(lvad), "excl_lvad"] = 1
    print(f"    LVAD: {len(lvad):,} stays")

    # 2. ECMO
    ecmo = stays_matching_text(ECMO_TEXT)
    flags.loc[flags["patientunitstayid"].isin(ecmo), "excl_ecmo"] = 1
    print(f"    ECMO: {len(ecmo):,} stays")

    # 3. Cardiac transplant
    transplant = stays_matching_icd(TRANSPLANT_CODES) | stays_matching_text(TRANSPLANT_TEXT)
    flags.loc[flags["patientunitstayid"].isin(transplant), "excl_transplant"] = 1
    print(f"    Cardiac transplant: {len(transplant):,} stays")

    # 4. Pregnancy
    pregnancy = (stays_matching_icd(PREGNANCY_ICD9_PREFIXES + PREGNANCY_ICD10_PREFIXES) |
                 stays_matching_text(PREGNANCY_TEXT))
    flags.loc[flags["patientunitstayid"].isin(pregnancy), "excl_pregnancy"] = 1
    print(f"    Pregnancy: {len(pregnancy):,} stays")

    # 5. Chronic dialysis
    dialysis = stays_matching_icd(DIALYSIS_CODES) | stays_matching_text(DIALYSIS_TEXT)
    flags.loc[flags["patientunitstayid"].isin(dialysis), "excl_dialysis"] = 1
    print(f"    Chronic dialysis/ESRD: {len(dialysis):,} stays")

    # 6. Hemorrhagic shock
    hemorrhage = stays_matching_text(HEMORRHAGE_TEXT)
    flags.loc[flags["patientunitstayid"].isin(hemorrhage), "excl_hemorrhage"] = 1
    print(f"    Hemorrhagic shock: {len(hemorrhage):,} stays")

    # 7. T0-ineligible (died/discharged < 6h)
    t0_inelig = pt[pt["icu_los_hours"] < 6]["patientunitstayid"].unique()
    flags.loc[flags["patientunitstayid"].isin(t0_inelig), "excl_t0_ineligible"] = 1
    print(f"    T0-ineligible (LOS < 6h): {len(t0_inelig):,} stays")

    # Any hard exclusion
    excl_cols = [c for c in flags.columns if c.startswith("excl_")]
    flags["any_hard_exclusion"] = (flags[excl_cols].sum(axis=1) > 0).astype(int)
    n_excl = flags["any_hard_exclusion"].sum()
    print(f"    ANY hard exclusion: {n_excl:,} stays "
          f"({n_excl / len(flags) * 100:.1f}%)")

    return flags


def check_hemodynamic_gate(cohort_ids: set) -> set:
    """
    Check hemodynamic gate: at least 1 MAP reading in first 6 hours.
    Mirrors MIMIC Step 25b.

    eICU MAP sources (checked in order):
      1. vitalPeriodic.systemicmean (arterial line MAP)
      2. vitalPeriodic: calculated MAP from systemicsystolic/systemicdiastolic
      3. vitalAperiodic.noninvasivemean (NIBP MAP)
      4. vitalAperiodic: calculated MAP from noninvasivesystolic/noninvasivediastolic
    """
    print("\n  Checking hemodynamic gate (MAP in first 6h)...")

    passed_ids = set()
    T0_MIN = 6 * 60  # 6 hours in minutes

    def _has_valid_map(row_map):
        """Check if MAP value is physiologically plausible."""
        return row_map.notna() & (row_map > 20) & (row_map < 250)

    # ── Source 1+2: vitalPeriodic (arterial line + calculated from art SBP/DBP) ──
    vp_path = EICU_ROOT / "vitalPeriodic.csv.gz"
    if not vp_path.exists():
        vp_path = EICU_ROOT / "vitalperiodic.csv.gz"

    if vp_path.exists():
        for chunk in pd.read_csv(vp_path, chunksize=500_000):
            chunk = chunk[chunk["patientunitstayid"].isin(cohort_ids)]
            if chunk.empty:
                continue
            if "observationoffset" not in chunk.columns:
                continue
            early = chunk[
                (chunk["observationoffset"] >= 0) &
                (chunk["observationoffset"] < T0_MIN)
            ]
            if early.empty:
                continue

            # Source 1: direct systemicmean (arterial line)
            if "systemicmean" in early.columns:
                has_map = early[_has_valid_map(early["systemicmean"])]
                passed_ids |= set(has_map["patientunitstayid"].unique())

            # Source 2: calculate MAP from arterial SBP/DBP
            if "systemicsystolic" in early.columns and "systemicdiastolic" in early.columns:
                sbp = early["systemicsystolic"]
                dbp = early["systemicdiastolic"]
                calc_map = dbp + (sbp - dbp) / 3.0
                has_calc = early[_has_valid_map(calc_map)]
                passed_ids |= set(has_calc["patientunitstayid"].unique())

    n_after_vp = len(passed_ids & cohort_ids)
    print(f"    After vitalPeriodic: {n_after_vp:,}")

    # ── Source 3+4: vitalAperiodic (NIBP, non-invasive) ──
    va_path = EICU_ROOT / "vitalAperiodic.csv.gz"
    if not va_path.exists():
        va_path = EICU_ROOT / "vitalaperiodic.csv.gz"

    if va_path.exists():
        for chunk in pd.read_csv(va_path, chunksize=500_000):
            chunk = chunk[chunk["patientunitstayid"].isin(cohort_ids)]
            if chunk.empty:
                continue
            if "observationoffset" not in chunk.columns:
                continue
            early = chunk[
                (chunk["observationoffset"] >= 0) &
                (chunk["observationoffset"] < T0_MIN)
            ]
            if early.empty:
                continue

            # Source 3: noninvasivemean (NIBP MAP — was incorrectly systemicmean!)
            if "noninvasivemean" in early.columns:
                has_map = early[_has_valid_map(early["noninvasivemean"])]
                passed_ids |= set(has_map["patientunitstayid"].unique())

            # Source 4: calculate MAP from NIBP SBP/DBP
            if "noninvasivesystolic" in early.columns and "noninvasivediastolic" in early.columns:
                sbp = early["noninvasivesystolic"]
                dbp = early["noninvasivediastolic"]
                calc_map = dbp + (sbp - dbp) / 3.0
                has_calc = early[_has_valid_map(calc_map)]
                passed_ids |= set(has_calc["patientunitstayid"].unique())

    n_pass = len(passed_ids & cohort_ids)
    n_total = len(cohort_ids)
    n_nibp_add = n_pass - n_after_vp
    print(f"    After vitalAperiodic (NIBP): {n_pass:,} (+{n_nibp_add:,} from NIBP)")
    print(f"    Hemodynamic gate: {n_pass:,} / {n_total:,} "
          f"({n_pass/n_total*100:.1f}%) have MAP in first 6h")

    return passed_ids


def apply_eligibility_criteria(
    pt: pd.DataFrame,
    pheno: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply eligibility criteria mirroring MIMIC pipeline.

    Criteria:
      1. Age ≥ 18
      2. ICU LOS ≥ 30h (T0=6h + 24h horizon)
      3. First ICU stay per patient (avoid within-patient correlation)
      4. Not missing critical demographics
      5. No hard exclusions (LVAD, ECMO, transplant, pregnancy, dialysis, hemorrhage)
      6. Hemodynamic gate (at least 1 MAP reading in first 6h)
    """
    print("\n  Applying eligibility criteria...")
    cohort = pt.merge(pheno, on="patientunitstayid", how="left")

    n0 = len(cohort)
    counts = {"total_stays": n0}

    # 1. Age ≥ 18
    cohort = cohort[cohort["age_num"] >= 18]
    n1 = len(cohort)
    counts["age_ge18"] = n1
    print(f"    Age ≥ 18: {n1:,} ({n1/n0*100:.1f}%)")

    # 2. ICU LOS ≥ 30h (6h pre-T0 + 24h decision horizon)
    cohort = cohort[cohort["icu_los_hours"] >= 30]
    n2 = len(cohort)
    counts["los_ge30h"] = n2
    print(f"    LOS ≥ 30h: {n2:,} ({n2/n1*100:.1f}% of age-eligible)")

    # 3. First ICU stay per patient
    cohort = cohort.sort_values(["uniquepid", "hospitaladmitoffset"])
    cohort = cohort.drop_duplicates(subset="uniquepid", keep="first")
    n3 = len(cohort)
    counts["first_stay"] = n3
    print(f"    First stay/patient: {n3:,} ({n3/n2*100:.1f}%)")

    # 4. Non-missing demographics
    cohort = cohort.dropna(subset=["age_num"])
    n4 = len(cohort)
    counts["complete_demog"] = n4
    print(f"    Complete demographics: {n4:,}")

    # 5. Hard exclusions (mirroring MIMIC 20e)
    excl_flags = build_exclusion_flags(pt)
    cohort = cohort.merge(
        excl_flags[["patientunitstayid", "any_hard_exclusion"]],
        on="patientunitstayid", how="left"
    )
    cohort["any_hard_exclusion"] = cohort["any_hard_exclusion"].fillna(0)
    n_before_excl = len(cohort)
    cohort = cohort[cohort["any_hard_exclusion"] == 0]
    n5 = len(cohort)
    counts["after_exclusions"] = n5
    print(f"    After hard exclusions: {n5:,} "
          f"(removed {n_before_excl - n5:,})")

    # 6. Hemodynamic gate (at least 1 MAP in first 6h)
    gate_passed = check_hemodynamic_gate(set(cohort["patientunitstayid"].unique()))
    cohort["hemo_gate_passed"] = cohort["patientunitstayid"].isin(gate_passed).astype(int)
    n_before_gate = len(cohort)
    cohort = cohort[cohort["hemo_gate_passed"] == 1]
    n6 = len(cohort)
    counts["hemo_gate"] = n6
    print(f"    Hemodynamic gate: {n6:,} "
          f"(removed {n_before_gate - n6:,})")

    counts["final"] = n6

    # Report by HF subtype
    print("\n  Final cohort by HF subtype:")
    for label, sub in cohort.groupby("hf_subtype"):
        n = len(sub)
        mort = sub["mortality_28d"].mean() * 100
        age = sub["age_num"].mean()
        fem = sub["is_female"].mean() * 100
        print(f"    {label:20s}: n={n:>6,}, mort28d={mort:5.1f}%, "
              f"age={age:.1f}, female={fem:.1f}%")

    return cohort


def main():
    print("=" * 72)
    print("90b_01 — Build eICU Cohort for External Validation")
    print("=" * 72)
    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 1. Load patients ─────────────────────────────────────────────────
    print("\n[1] Loading eICU patients...")
    pt = load_eicu_patients()

    # Save raw stays
    pt.to_parquet(STAGING_DIR / "90b_eicu_icu_stays.parquet", index=False)

    # ── 2. HF phenotyping ────────────────────────────────────────────────
    print("\n[2] HF phenotyping (29d broader-compatible)...")
    pheno = phenotype_hf(pt)

    # Save phenotype
    pheno.to_parquet(STAGING_DIR / "90b_eicu_hf_phenotype.parquet", index=False)

    # ── 3. Eligibility + cohort ──────────────────────────────────────────
    print("\n[3] Applying eligibility criteria...")
    cohort = apply_eligibility_criteria(pt, pheno)

    # ── 4. 29e downgrade assessment ──────────────────────────────────────
    print("\n[4] 29e Downgrade assessment...")
    n_hfpef = (cohort["hf_subtype"] == "hfpef").sum()
    n_hfpef_strict = int(cohort["hfpef_diastolic_strict_flag"].sum())
    print(f"  HFpEF post-eligibility (broad): {n_hfpef:,}")
    print(f"  HFpEF post-eligibility (diastolic strict proxy): {n_hfpef_strict:,}")

    if n_hfpef >= 500:
        role = "FULL_EXTERNAL_VALIDATION"
        print(f"  29e Trigger 1: {n_hfpef} ≥ 500 → FULL VALIDATION ELIGIBLE")
    elif n_hfpef >= 200:
        role = "SENSITIVITY_ONLY"
        print(f"  29e Trigger 1: {n_hfpef} ∈ [200, 500) → SENSITIVITY-ONLY")
    else:
        role = "DESCRIPTIVE_ONLY"
        print(f"  29e Trigger 1: {n_hfpef} < 200 → DESCRIPTIVE ONLY")

    print(f"  29e Trigger 2: BROADER-COMPATIBLE ascertainment → OK with sensitivity")
    print(f"  Assigned role: {role}")

    cohort["eicu_validation_role"] = role

    # ── 5. Save cohort ───────────────────────────────────────────────────
    print("\n[5] Saving cohort...")
    out_path = STAGING_DIR / "90b_eicu_cohort_v1.parquet"
    cohort.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path} ({len(cohort):,} stays)")

    # Counts table
    counts_rows = []
    for label in ["hfpef", "hfref", "no_hf"]:
        sub = cohort[cohort["hf_subtype"] == label]
        counts_rows.append({
            "hf_subtype": label,
            "n_stays": len(sub),
            "mortality_28d": sub["mortality_28d"].mean(),
            "age_mean": sub["age_num"].mean(),
            "female_pct": sub["is_female"].mean(),
        })
    counts_df = pd.DataFrame(counts_rows)
    counts_df.to_csv(TABLES_DIR / "90b_01_eicu_cohort_counts.csv", index=False)

    # ── 6. Summary report ────────────────────────────────────────────────
    report_lines = [
        "=" * 72,
        "90b_01 — eICU Cohort for External Validation",
        "=" * 72,
        f"Generated: {ts}",
        f"Protocol: v3.1, Step 90b",
        "",
        "COHORT CONSTRUCTION",
        "───────────────────",
        f"  Total eICU stays: {len(pt):,}",
        f"  → Age ≥ 18: {(pt['age_num'] >= 18).sum():,}",
        f"  → LOS ≥ 30h: see below",
        f"  → First stay/patient: see below",
        f"  → After hard exclusions (LVAD/ECMO/transplant/pregnancy/dialysis/hemorrhage): see log",
        f"  → Hemodynamic gate (MAP in first 6h): see log",
        f"  → Final cohort: {len(cohort):,}",
        "",
        "HF PHENOTYPE (29d broader-compatible)",
        "──────────────────────────────────────",
    ]
    for _, row in counts_df.iterrows():
        report_lines.append(
            f"  {row['hf_subtype']:20s}: n={row['n_stays']:>6,}, "
            f"mort={row['mortality_28d']*100:5.1f}%, "
            f"age={row['age_mean']:.1f}, "
            f"F={row['female_pct']*100:.1f}%"
        )
    report_lines.extend([
        "",
        "29e DOWNGRADE ASSESSMENT",
        "────────────────────────",
        f"  HFpEF post-eligibility (broad): {n_hfpef:,}",
        f"  HFpEF post-eligibility (diastolic strict proxy): {n_hfpef_strict:,}",
        f"  Trigger 1 (sample size, broad): {'PASS' if n_hfpef >= 500 else 'TRIGGERED'}",
        f"  Strict-proxy size note: {n_hfpef_strict:,} stays (diastolic-only sensitivity)",
        f"  Trigger 2 (ascertainment): BROADER-COMPATIBLE (requires sensitivity)",
        f"  Assigned role: {role}",
        "",
        "MORTALITY DEFINITION",
        "────────────────────",
        "  28-day mortality approximation:",
        "  = hospital death within 28 days of ICU admission",
        "  Limitation: no post-discharge follow-up in eICU",
        "  → Likely UNDERESTIMATES true 28-day mortality",
        "",
        "NEXT STEPS",
        "──────────",
        "  90b_02: Extract eICU vitals + labs temporal data",
        "  90b_03: Build eICU state tensors (MIMIC-normalized)",
    ])

    report_path = REPORTS_DIR / "90b_01_eicu_cohort_summary.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\n  Report: {report_path}")

    total_time = time.time() - t0
    print()
    print("=" * 72)
    print(f"90b_01 COMPLETE. Cohort: {len(cohort):,} stays "
          f"(HFpEF={n_hfpef:,}, role={role})")
    print(f"Time: {total_time:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
