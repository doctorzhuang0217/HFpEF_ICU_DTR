#!/usr/bin/env python3
"""
20e_build_exclusion_flags_mimic — Hard exclusion flags for cohort eligibility.

Purpose
-------
Build binary exclusion flags for each ICU stay. A stay is excluded from the
primary analysis if ANY hard-exclusion flag is TRUE.

Hard exclusions (from protocol Section 6 / target trial table):
  1. LVAD (left ventricular assist device) — ICD procedure codes
  2. ECMO — ICD procedure codes
  3. Cardiac transplant — ICD procedure or diagnosis codes
  4. Pregnancy-related ICU stay — ICD diagnosis codes
  5. Chronic dialysis-dependent ESRD — ICD diagnosis codes
  6. Comfort care / withdrawal of care before T0 — code-status chartevents (10h)
     NOTE: DNR/DNI is NOT a hard exclusion per protocol — retained as covariate
  7. Active massive hemorrhage / trauma-driven hemorrhagic shock — ICD codes
  8. Death before T0 (< 6h) — from 10c timestamps
  9. Discharge/transfer before T0 — from 10c timestamps

Inputs
------
- data/staging/10a_core_icu_stays_mimic.parquet
- data/staging/10b_diagnoses_mimic.parquet
- data/staging/10b_procedures_mimic.parquet
- data/staging/10c_timestamps_mimic.parquet
- data/staging/10h_vent_indicators_mimic.parquet  (code_status rows)

Outputs
-------
- data/staging/20e_exclusion_flags_mimic.parquet
- reports/20e_exclusion_flags_mimic_summary.txt
"""

from __future__ import annotations

import sys
import textwrap
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path

# ── ICD code patterns for exclusions ────────────────────────────────────
# LVAD (ICD-10 procedure codes)
LVAD_PX_ICD10 = {
    "02RA0EZ", "02RA0FZ", "02RA0JZ",  # LVAD insertion
    "02WA0EZ",                           # LVAD revision
    "02PA0EZ",                           # LVAD removal
}
# LVAD diagnosis codes (presence of device)
LVAD_DX_ICD10 = {"Z95811", "Z95812"}  # Presence of left heart assist device

# ECMO (ICD-10 procedure codes)
ECMO_PX_ICD10 = {
    "5A1522F", "5A1523F",  # VA-ECMO
    "5A1522G", "5A1523G",  # AV-ECMO
    "5A1522H", "5A1523H",  # VV-ECMO
}

# Cardiac transplant
TRANSPLANT_PX_ICD10 = {"02YA0Z0", "02YA0Z1", "02YA0Z2"}
TRANSPLANT_DX_ICD10_PREFIX = "Z941"  # heart transplant status

# Pregnancy
PREGNANCY_DX_ICD10_PREFIXES = [
    "O",      # all obstetric codes
    "Z33",    # pregnant state
    "Z34",    # supervision of normal pregnancy
    "Z3A",    # weeks of gestation
]
PREGNANCY_DX_ICD9_PREFIXES = [
    "V22", "V23", "V27", "V28",  # pregnancy supervision/outcome
    "63", "64", "65", "66", "67", "68", "69",  # pregnancy complications
]

# Chronic dialysis / ESRD
ESRD_DX_ICD10 = {"N186"}  # ESRD
CHRONIC_DIALYSIS_DX_ICD10_PREFIXES = ["Z49", "Z992"]  # dialysis encounter/dependence
ESRD_DX_ICD9 = {"5856"}  # ESRD
CHRONIC_DIALYSIS_DX_ICD9_PREFIXES = ["V45", "V56"]

# Active massive hemorrhage / trauma-driven hemorrhagic shock
# Protocol intent: "active massive hemorrhage OR trauma-driven hemorrhagic shock"
# NOT generic hypovolemic shock (dehydration, burns, post-op volume depletion).
#
# Strategy: Trauma-specific shock codes are standalone sufficient.
#           R571 (hypovolemic shock) only qualifies WITH concurrent hemorrhage
#           or trauma diagnosis — otherwise it captures non-hemorrhagic causes.

# Standalone sufficient: trauma-specific shock codes
TRAUMA_SHOCK_DX_ICD10 = {
    "T794",   # traumatic shock (initial encounter)
    "T794A",  # traumatic shock, initial encounter
    "T794D",  # traumatic shock, subsequent encounter
}
# R571 = hypovolemic shock — only counted when co-occurring with hemorrhage/trauma
HYPOVOLEMIC_SHOCK_DX_ICD10 = {"R571"}

# Hemorrhage context codes (must co-occur with R571 to qualify)
HEMORRHAGE_CONTEXT_DX_ICD10_PREFIXES = [
    "I61",    # nontraumatic intracerebral hemorrhage
    "I62",    # other and unspecified nontraumatic intracranial hemorrhage
    "K25",    # gastric ulcer (with hemorrhage subtypes)
    "K26",    # duodenal ulcer (with hemorrhage subtypes)
    "K27",    # peptic ulcer (with hemorrhage subtypes)
    "K920",   # hematemesis
    "K921",   # melena
    "K922",   # GI hemorrhage, unspecified
    "D62",    # acute posthemorrhagic anemia
    "N93",    # other abnormal uterine/vaginal bleeding
    "R04",    # hemorrhage from respiratory passages
    "R31",    # hematuria (gross)
    "R58",    # hemorrhage, not elsewhere classified
    "S06",    # intracranial injury (traumatic)
    "S27",    # injury of thoracic organs (hemothorax)
    "S36",    # injury of intra-abdominal organs
    "T79",    # certain early complications of trauma
]
# Trauma context (any S/T injury code implies trauma-driven)
TRAUMA_DX_ICD10_PREFIXES = ["S", "T"]

# Major trauma-driven hemorrhage (ICD-10 procedure codes for damage control)
HEMORRHAGE_PX_ICD10_PREFIXES = [
    "0W3",    # control bleeding in anatomical regions
]
# ICD-9 equivalents
TRAUMA_SHOCK_DX_ICD9 = {
    "9584",   # traumatic shock
}
HEMORRHAGE_CONTEXT_DX_ICD9_PREFIXES = [
    "430", "431", "432",  # intracranial hemorrhage
    "578",    # GI hemorrhage
    "998",    # postoperative complications (includes hemorrhage)
    "800", "801", "802", "803", "804",  # skull fractures
    "850", "851", "852", "853", "854",  # intracranial injury
    "860", "861", "862", "863", "864", "865", "866", "867", "868", "869",  # internal organ injury
]


def _normalize_icd(code: str) -> str:
    """Remove dots from ICD codes for matching."""
    return str(code).replace(".", "").strip()


def build_icd_exclusion_flags(
    dx: pd.DataFrame, px: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Build per-hadm_id exclusion flags from ICD codes.

    Returns (flags_df, hem_shock_detail) where hem_shock_detail contains
    breakdown counts for hemorrhagic shock code-list audit.
    """
    # Normalize codes
    dx = dx.copy()
    px = px.copy()
    dx["icd_norm"] = dx["icd_code"].apply(_normalize_icd)
    px["icd_norm"] = px["icd_code"].apply(_normalize_icd)

    hadm_ids = set(dx["hadm_id"].unique()) | set(px["hadm_id"].unique())
    flags = pd.DataFrame({"hadm_id": sorted(hadm_ids)})

    # ── LVAD ──
    lvad_px = px[
        (px["icd_version"] == 10) & px["icd_norm"].isin(LVAD_PX_ICD10)
    ]["hadm_id"].unique()
    lvad_dx = dx[
        (dx["icd_version"] == 10) & dx["icd_norm"].isin(LVAD_DX_ICD10)
    ]["hadm_id"].unique()
    flags["excl_lvad"] = flags["hadm_id"].isin(set(lvad_px) | set(lvad_dx))

    # ── ECMO ──
    ecmo_px = px[
        (px["icd_version"] == 10) & px["icd_norm"].isin(ECMO_PX_ICD10)
    ]["hadm_id"].unique()
    flags["excl_ecmo"] = flags["hadm_id"].isin(set(ecmo_px))

    # ── Cardiac transplant ──
    tx_px = px[
        (px["icd_version"] == 10) & px["icd_norm"].isin(TRANSPLANT_PX_ICD10)
    ]["hadm_id"].unique()
    tx_dx = dx[
        (dx["icd_version"] == 10)
        & dx["icd_norm"].str.startswith(TRANSPLANT_DX_ICD10_PREFIX)
    ]["hadm_id"].unique()
    flags["excl_transplant"] = flags["hadm_id"].isin(set(tx_px) | set(tx_dx))

    # ── Pregnancy ──
    preg_10 = dx[
        (dx["icd_version"] == 10)
        & dx["icd_norm"].apply(
            lambda c: any(c.startswith(p) for p in PREGNANCY_DX_ICD10_PREFIXES)
        )
    ]["hadm_id"].unique()
    preg_9 = dx[
        (dx["icd_version"] == 9)
        & dx["icd_norm"].apply(
            lambda c: any(c.startswith(p) for p in PREGNANCY_DX_ICD9_PREFIXES)
        )
    ]["hadm_id"].unique()
    flags["excl_pregnancy"] = flags["hadm_id"].isin(set(preg_10) | set(preg_9))

    # ── Chronic dialysis / ESRD ──
    esrd_10 = dx[
        (dx["icd_version"] == 10) & dx["icd_norm"].isin(ESRD_DX_ICD10)
    ]["hadm_id"].unique()
    dial_10 = dx[
        (dx["icd_version"] == 10)
        & dx["icd_norm"].apply(
            lambda c: any(c.startswith(p) for p in CHRONIC_DIALYSIS_DX_ICD10_PREFIXES)
        )
    ]["hadm_id"].unique()
    esrd_9 = dx[
        (dx["icd_version"] == 9) & dx["icd_norm"].isin(ESRD_DX_ICD9)
    ]["hadm_id"].unique()
    dial_9 = dx[
        (dx["icd_version"] == 9)
        & dx["icd_norm"].apply(
            lambda c: any(c.startswith(p) for p in CHRONIC_DIALYSIS_DX_ICD9_PREFIXES)
        )
    ]["hadm_id"].unique()
    flags["excl_chronic_dialysis"] = flags["hadm_id"].isin(
        set(esrd_10) | set(dial_10) | set(esrd_9) | set(dial_9)
    )

    # ── Hemorrhagic shock / trauma-driven hemorrhage ──
    # Strategy: trauma-specific shock codes are standalone sufficient;
    # R571 (hypovolemic shock) only qualifies WITH concurrent hemorrhage/trauma.

    # 1. Standalone trauma shock codes (sufficient alone)
    trauma_shock_10 = set(dx[
        (dx["icd_version"] == 10) & dx["icd_norm"].isin(TRAUMA_SHOCK_DX_ICD10)
    ]["hadm_id"].unique())
    trauma_shock_9 = set(dx[
        (dx["icd_version"] == 9) & dx["icd_norm"].isin(TRAUMA_SHOCK_DX_ICD9)
    ]["hadm_id"].unique())

    # 2. R571 hypovolemic shock — needs concurrent hemorrhage/trauma context
    r571_hadms = set(dx[
        (dx["icd_version"] == 10) & dx["icd_norm"].isin(HYPOVOLEMIC_SHOCK_DX_ICD10)
    ]["hadm_id"].unique())

    # 3. Hemorrhage/trauma context codes
    hem_context_10 = set(dx[
        (dx["icd_version"] == 10)
        & dx["icd_norm"].apply(
            lambda c: any(c.startswith(p) for p in HEMORRHAGE_CONTEXT_DX_ICD10_PREFIXES)
        )
    ]["hadm_id"].unique())
    trauma_context_10 = set(dx[
        (dx["icd_version"] == 10)
        & dx["icd_norm"].apply(
            lambda c: any(c.startswith(p) for p in TRAUMA_DX_ICD10_PREFIXES)
        )
    ]["hadm_id"].unique())
    hem_context_9 = set(dx[
        (dx["icd_version"] == 9)
        & dx["icd_norm"].apply(
            lambda c: any(c.startswith(p) for p in HEMORRHAGE_CONTEXT_DX_ICD9_PREFIXES)
        )
    ]["hadm_id"].unique())

    # 4. Hemorrhage control procedures (standalone sufficient)
    hem_px_hadms = set(px[
        (px["icd_version"] == 10)
        & px["icd_norm"].apply(
            lambda c: any(c.startswith(p) for p in HEMORRHAGE_PX_ICD10_PREFIXES)
        )
    ]["hadm_id"].unique())

    # Composite:
    #   trauma shock codes (standalone)
    #   OR R571 + hemorrhage/trauma context (requires both)
    #   OR hemorrhage control procedures (standalone)
    all_context = hem_context_10 | trauma_context_10 | hem_context_9
    r571_with_context = r571_hadms & all_context
    r571_without_context = r571_hadms - all_context
    hemorrhagic_excl = (
        trauma_shock_10 | trauma_shock_9
        | r571_with_context
        | hem_px_hadms
    )
    flags["excl_hemorrhagic_shock"] = flags["hadm_id"].isin(hemorrhagic_excl)

    hem_shock_detail = {
        "trauma_shock_standalone": len(trauma_shock_10 | trauma_shock_9),
        "r571_with_context": len(r571_with_context),
        "r571_without_context": len(r571_without_context),
        "hemorrhage_control_px": len(hem_px_hadms),
        "total_excluded": len(hemorrhagic_excl),
    }

    return flags, hem_shock_detail


def main() -> None:
    cfg = load_config()
    staging_dir = resolve_path(cfg, "staging")
    reports_dir = resolve_path(cfg, "reports")
    staging_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("20e_build_exclusion_flags_mimic — Hard exclusion flags")
    print("=" * 70)

    # ── load inputs ──────────────────────────────────────────────────
    print("\n[1/5] Loading inputs ...")
    stays = pd.read_parquet(
        staging_dir / "10a_core_icu_stays_mimic.parquet",
        columns=["stay_id", "hadm_id", "subject_id"],
    )
    n_stays = len(stays)
    print(f"  Stays: {n_stays:,}")

    dx = pd.read_parquet(staging_dir / "10b_diagnoses_mimic.parquet")
    px = pd.read_parquet(staging_dir / "10b_procedures_mimic.parquet")

    # ── ICD-based exclusion flags ────────────────────────────────────
    print("\n[2/5] Building ICD-based exclusion flags ...")
    icd_flags, hem_detail = build_icd_exclusion_flags(dx, px)
    print(f"  LVAD:               {icd_flags['excl_lvad'].sum():,}")
    print(f"  ECMO:               {icd_flags['excl_ecmo'].sum():,}")
    print(f"  Transplant:         {icd_flags['excl_transplant'].sum():,}")
    print(f"  Pregnancy:          {icd_flags['excl_pregnancy'].sum():,}")
    print(f"  Chronic dialysis:   {icd_flags['excl_chronic_dialysis'].sum():,}")
    print(f"  Hemorrhagic shock:  {icd_flags['excl_hemorrhagic_shock'].sum():,}")
    print(f"    ├─ Trauma shock (standalone):        {hem_detail['trauma_shock_standalone']:,}")
    print(f"    ├─ R571 + hemorrhage/trauma context: {hem_detail['r571_with_context']:,}")
    print(f"    ├─ R571 without context (NOT excl):  {hem_detail['r571_without_context']:,}")
    print(f"    └─ Hemorrhage control procedures:    {hem_detail['hemorrhage_control_px']:,}")

    # ── Comfort care / code status before T0 ─────────────────────────
    print("\n[3/5] Building timing and comfort-care exclusions ...")

    # Load 10c timestamps for T0 eligibility
    ts = pd.read_parquet(
        staging_dir / "10c_timestamps_mimic.parquet",
        columns=["stay_id", "time_zero_6h", "t0_6h_eligible"],
    )

    # Load 10h code-status indicators
    vent_ind = pd.read_parquet(staging_dir / "10h_vent_indicators_mimic.parquet")
    code_status = vent_ind[vent_ind["indicator_type"] == "code_status"].copy()

    # Identify comfort-care and DNR/DNI before T0
    # PROTOCOL: comfort/withdrawal → HARD EXCLUSION
    #           DNR/DNI → NOT excluded, retained as covariate (25e, 40x)
    code_status = code_status.merge(
        ts[["stay_id", "time_zero_6h"]], on="stay_id", how="inner"
    )
    # Keep only records before T0
    pre_t0_codes = code_status[code_status["charttime"] < code_status["time_zero_6h"]].copy()

    # Separate patterns: comfort/withdrawal vs DNR/DNI
    comfort_pattern = r"(?i)comfort|CMO|withdraw|palliat"
    dnr_pattern = r"(?i)DNR|do not resuscitate"
    dni_pattern = r"(?i)DNI|do not intubate"

    comfort_only_stays = set()
    dnr_only_stays = set()
    dni_only_stays = set()
    if len(pre_t0_codes) > 0:
        comfort_mask = pre_t0_codes["value"].str.contains(
            comfort_pattern, na=False
        )
        dnr_mask = pre_t0_codes["value"].str.contains(dnr_pattern, na=False)
        dni_mask = pre_t0_codes["value"].str.contains(dni_pattern, na=False)
        comfort_only_stays = set(
            pre_t0_codes.loc[comfort_mask, "stay_id"].unique()
        )
        dnr_only_stays = set(
            pre_t0_codes.loc[dnr_mask, "stay_id"].unique()
        )
        dni_only_stays = set(
            pre_t0_codes.loc[dni_mask, "stay_id"].unique()
        )
    # DNR/DNI-only (no comfort) — these are NOT excluded
    dnr_dni_all = dnr_only_stays | dni_only_stays
    dnr_dni_no_comfort = dnr_dni_all - comfort_only_stays
    comfort_with_dnr_overlap = comfort_only_stays & dnr_dni_all
    print(f"  Comfort/withdrawal before T0: {len(comfort_only_stays):,} (HARD EXCLUSION)")
    print(f"  DNR before T0:                {len(dnr_only_stays):,} (covariate, NOT excluded)")
    print(f"  DNI before T0:                {len(dni_only_stays):,} (covariate, NOT excluded)")
    print(f"  DNR/DNI-only (no comfort):    {len(dnr_dni_no_comfort):,} (RETAINED)")
    print(f"  Comfort + DNR/DNI overlap:    {len(comfort_with_dnr_overlap):,}")

    # T0 ineligible (died or discharged before T0)
    t0_ineligible = set(ts.loc[ts["t0_6h_eligible"] == 0, "stay_id"])
    print(f"  T0-ineligible (died/discharged before 6h): {len(t0_ineligible):,}")

    # ── Merge all flags to stay level ────────────────────────────────
    out = stays.copy()
    out = out.merge(icd_flags, on="hadm_id", how="left")

    # Fill NaN for stays without any ICD data
    excl_cols = [c for c in out.columns if c.startswith("excl_")]
    for c in excl_cols:
        out[c] = out[c].fillna(False)

    # Add timing/comfort flags
    # HARD EXCLUSION: comfort/withdrawal only (NOT DNR/DNI)
    out["excl_comfort_withdrawal_pre_t0"] = out["stay_id"].isin(comfort_only_stays)
    # COVARIATE (not exclusion): DNR/DNI status before T0
    out["flag_dnr_pre_t0"] = out["stay_id"].isin(dnr_only_stays)
    out["flag_dni_pre_t0"] = out["stay_id"].isin(dni_only_stays)
    # excl_hemorrhagic_shock already merged from icd_flags (hadm_id-level)
    out["excl_t0_ineligible"] = out["stay_id"].isin(t0_ineligible)

    # Composite exclusion flag
    all_excl_cols = [c for c in out.columns if c.startswith("excl_")]
    out["any_hard_exclusion"] = out[all_excl_cols].any(axis=1)

    n_excl = out["any_hard_exclusion"].sum()
    n_eligible = n_stays - n_excl
    print(f"\n  Total excluded: {n_excl:,} ({n_excl / n_stays * 100:.1f}%)")
    print(f"  Eligible stays: {n_eligible:,} ({n_eligible / n_stays * 100:.1f}%)")

    # ── write output ─────────────────────────────────────────────────
    out_cols = [
        "stay_id", "hadm_id", "subject_id",
        "excl_lvad", "excl_ecmo", "excl_transplant",
        "excl_pregnancy", "excl_chronic_dialysis",
        "excl_comfort_withdrawal_pre_t0", "excl_hemorrhagic_shock",
        "excl_t0_ineligible",
        "flag_dnr_pre_t0", "flag_dni_pre_t0",
        "any_hard_exclusion",
    ]
    out = out[out_cols].sort_values("stay_id").reset_index(drop=True)

    out_path = staging_dir / "20e_exclusion_flags_mimic.parquet"
    out.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"\n  → {out_path}  ({len(out):,} rows)")

    # ── HF subtype × exclusion cross-tab (A4 audit fix) ──────────────
    print("\n[4/5] Building HF subtype × exclusion cross-tab ...")
    hf_labels = pd.read_parquet(
        staging_dir / "20a_hf_labels_mimic.parquet",
        columns=["stay_id", "hf_subtype", "comparator_group"],
    )
    out_with_hf = out.merge(hf_labels, on="stay_id", how="left")
    out_with_hf["hf_subtype"] = out_with_hf["hf_subtype"].fillna("unknown")

    # Cross-tab: each excl flag × hf_subtype
    xtab_rows = []
    for c in all_excl_cols:
        row = out_with_hf.groupby("hf_subtype")[c].sum().astype(int)
        row.name = c
        xtab_rows.append(row)
    xtab = pd.DataFrame(xtab_rows)
    xtab["total"] = xtab.sum(axis=1)
    xtab_str = xtab.to_string()

    # Also show retained N per subtype
    retained = out_with_hf[~out_with_hf["any_hard_exclusion"]].groupby("hf_subtype").size()
    retained_str = retained.to_string()

    # DNR/DNI covariate counts (not exclusions)
    dnr_counts = out_with_hf.groupby("hf_subtype")[["flag_dnr_pre_t0", "flag_dni_pre_t0"]].sum().astype(int)
    dnr_str = dnr_counts.to_string()

    # ── summary report ───────────────────────────────────────────────
    print("\n[5/5] Generating summary report ...")

    excl_breakdown = []
    for c in all_excl_cols:
        cnt = out[c].sum()
        excl_breakdown.append(f"  {c:38s}: {cnt:>8,}  ({cnt / n_stays * 100:.2f}%)")
    excl_str = "\n".join(excl_breakdown)

    # Comfort/DNR detail
    n_comfort = len(comfort_only_stays)
    n_dnr = len(dnr_only_stays)
    n_dni = len(dni_only_stays)
    n_dnr_dni_no_comfort = len(dnr_dni_no_comfort)

    report = textwrap.dedent(f"""\
    ================================================================
    20e_build_exclusion_flags_mimic — Summary Report
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ================================================================

    EXCLUSION FLAGS (HARD EXCLUSIONS ONLY)
    ──────────────────────────────────────
    Total ICU stays         : {n_stays:>10,}

    Per-flag breakdown:
    {excl_str}

    COMPOSITE
    ─────────
    Any hard exclusion      : {n_excl:>10,}  ({n_excl / n_stays * 100:.1f}%)
    Eligible (no exclusion) : {n_eligible:>10,}  ({n_eligible / n_stays * 100:.1f}%)

    CODE STATUS DETAIL (pre-T0)
    ───────────────────────────
    Comfort/withdrawal (EXCLUDED):  {n_comfort:>8,}
    DNR (NOT excluded, covariate):  {n_dnr:>8,}
    DNI (NOT excluded, covariate):  {n_dni:>8,}
    DNR/DNI-only (no comfort):      {n_dnr_dni_no_comfort:>8,}  (RETAINED in cohort)
    Comfort + DNR/DNI overlap:      {len(comfort_with_dnr_overlap):>8,}

    HEMORRHAGIC SHOCK DETAIL (code-list audit)
    ────────────────────────────────────────────
    Trauma shock codes (standalone):      {hem_detail['trauma_shock_standalone']:>8,}
    R571 + hemorrhage/trauma context:     {hem_detail['r571_with_context']:>8,}
    R571 without context (NOT excluded):  {hem_detail['r571_without_context']:>8,}
    Hemorrhage control procedures:        {hem_detail['hemorrhage_control_px']:>8,}
    Total hemorrhagic shock excluded:     {hem_detail['total_excluded']:>8,}

    HF SUBTYPE × EXCLUSION FLAG CROSS-TAB
    ──────────────────────────────────────
    {textwrap.indent(xtab_str, '  ')}

    RETAINED STAYS BY HF SUBTYPE (after exclusions)
    ─────────────────────────────────────────────────
    {textwrap.indent(retained_str, '  ')}

    DNR/DNI COVARIATES BY HF SUBTYPE (not excluded)
    ─────────────────────────────────────────────────
    {textwrap.indent(dnr_str, '  ')}

    OUTPUT
    ──────
    {out_path}  ({len(out):,} rows, {out.shape[1]} cols)
    """)

    report_path = reports_dir / "20e_exclusion_flags_mimic_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"  → Report: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
