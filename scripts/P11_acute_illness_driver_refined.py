#!/usr/bin/env python3
"""
P11_acute_illness_driver_refined — Refined acute-illness driver classification.

Purpose
-------
Extend 25c reason-for-ICU classification with finer-grained categories that
a granular Table 1 needs: cardiogenic shock (currently lumped
into "other"), septic shock vs sepsis-without-shock, ARDS vs other respiratory
failure, trauma / hemorrhage / neurocritical (currently lumped), and post-op
cardiac vs non-cardiac surgery.

Categories are NOT mutually exclusive; binary flags are emitted for every
admission, plus a single mutually-exclusive priority label
(``acute_driver_refined``) for stratified Table 1 use.

Inputs
------
- data/staging/25a_time_zero_flags_mimic.parquet
- data/staging/10b_diagnoses_mimic.parquet
- data/staging/10b_procedures_mimic.parquet

Outputs
-------
- data/staging/P11_acute_illness_driver_refined_mimic.parquet
- reports/P11_acute_illness_driver_refined_summary.txt
"""

from __future__ import annotations

import sys
import textwrap
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path


def _norm(code: str) -> str:
    return str(code).replace(".", "").strip().upper()


# ── Diagnosis ICD prefixes / exact codes ─────────────────────────────
# Cardiogenic shock — explicit codes only (do not lump with septic shock).
CARDIOGENIC_SHOCK_ICD10 = {"R570"}
CARDIOGENIC_SHOCK_ICD9 = {"78551"}

# Septic shock vs sepsis-without-shock (split 25c sepsis bucket).
SEPTIC_SHOCK_ICD10 = {"R6521", "T8112"}
SEPTIC_SHOCK_ICD9 = {"78552"}
SEPSIS_BROAD_ICD10_PREFIX = ["A40", "A41", "R6520"]
SEPSIS_BROAD_ICD9_PREFIX = ["038", "9959"]

# ARDS — split out from generic respiratory failure.
ARDS_ICD10_PREFIX = ["J80"]
ARDS_ICD9_PREFIX = ["51882"]   # 518.82 = "other pulmonary insufficiency, ARDS"
RESP_FAILURE_ICD10_PREFIX = ["J96", "J18", "J12", "J13", "J14", "J15",
                              "J16", "J17", "J44", "J69"]
RESP_FAILURE_ICD9_PREFIX = ["518", "486", "480", "481", "482", "483", "484", "485"]

# Neurocritical — stroke, ICH, SAH, status epilepticus, cerebral edema, TBI.
NEURO_ICD10_PREFIX = ["I60", "I61", "I62", "I63",   # CVA / hemorrhage
                      "G40",                         # epilepsy / status
                      "G935", "G936",                # cerebral edema / anoxic injury
                      "S06"]                         # TBI
NEURO_ICD9_PREFIX = ["430", "431", "432", "433", "434", "436",
                     "3485", "3481", "853", "854"]

# Trauma / hemorrhage (non-cerebral).
TRAUMA_ICD10_PREFIX = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9",
                       "T07", "T14", "T79"]
TRAUMA_ICD9_PREFIX = [str(c) for c in range(800, 905)] + ["958", "959"]
HEMORRHAGE_ICD10 = {"D62",                            # acute posthemorrhagic anemia
                    "K920", "K921", "K922",           # GI hemorrhage
                    "R58"}                            # hemorrhage NOS
HEMORRHAGE_ICD9_PREFIX = ["578",                      # GI hemorrhage
                          "2851"]                     # acute posthemorrhagic anemia

# AKI primary (carry forward from 25c).
AKI_ICD10 = {"N170", "N171", "N172", "N178", "N179"}
AKI_ICD9 = {"5849", "5845", "5846", "5847"}


# ── Procedure ICD prefixes (cardiac vs non-cardiac surgery) ─────────
# ICD-10-PCS: 7-char alphanumeric; section 02 = "Heart and Great Vessels".
# Hence cardiac surgical procedures begin with "02".
# Coronary stenting (021–02C ranges) and valve repairs all fall here.
CARDIAC_PROC_ICD10_PCS_PREFIX = ["02"]
# ICD-9 procedure codes for cardiac surgery: 35.x (valves/septa), 36.x (vessels),
# 37.x (other heart). Vascular 39.x is NOT counted as cardiac here (mostly access).
CARDIAC_PROC_ICD9_PREFIX = ["35", "36", "37"]


def _has_prefix(codes: pd.Series, prefixes: list[str]) -> pd.Series:
    """Vectorized: True where code starts with any prefix."""
    if not prefixes:
        return pd.Series(False, index=codes.index)
    pat = "|".join(f"^{p}" for p in prefixes)
    return codes.str.contains(pat, regex=True, na=False)


def classify_drivers(
    hadm_ids: set,
    dx: pd.DataFrame,
    px: pd.DataFrame,
) -> pd.DataFrame:
    """Build refined acute-illness driver flags + priority label.

    Memory-efficient: filters once, no DataFrame copies; normalization is
    a derived Series rather than a new column.
    """
    dx = dx.loc[dx["hadm_id"].isin(hadm_ids), ["hadm_id", "icd_code", "icd_version"]]
    dx_norm = (
        dx["icd_code"].astype(str)
        .str.replace(".", "", regex=False).str.upper().str.strip()
    )
    dx_v10 = dx["icd_version"] == 10
    dx_v9 = dx["icd_version"] == 9
    dx_hadm = dx["hadm_id"].to_numpy()

    px = px.loc[px["hadm_id"].isin(hadm_ids), ["hadm_id", "icd_code", "icd_version"]]
    px_norm = (
        px["icd_code"].astype(str)
        .str.replace(".", "", regex=False).str.upper().str.strip()
    )
    px_v10 = px["icd_version"] == 10
    px_v9 = px["icd_version"] == 9
    px_hadm = px["hadm_id"].to_numpy()

    def _hadm_set_dx(prefixes: list[str], v10: bool) -> set:
        ver_mask = (dx_v10 if v10 else dx_v9).to_numpy()
        if not prefixes:
            return set()
        match = _has_prefix(dx_norm, prefixes).to_numpy()
        return set(dx_hadm[ver_mask & match])

    def _hadm_set_dx_exact(code_set: set, v10: bool) -> set:
        ver_mask = (dx_v10 if v10 else dx_v9).to_numpy()
        match = dx_norm.isin(code_set).to_numpy()
        return set(dx_hadm[ver_mask & match])

    def _hadm_set_px(prefixes: list[str], v10: bool) -> set:
        ver_mask = (px_v10 if v10 else px_v9).to_numpy()
        if not prefixes:
            return set()
        match = _has_prefix(px_norm, prefixes).to_numpy()
        return set(px_hadm[ver_mask & match])

    cardiogenic_shock = (
        _hadm_set_dx_exact(CARDIOGENIC_SHOCK_ICD10, True)
        | _hadm_set_dx_exact(CARDIOGENIC_SHOCK_ICD9, False)
    )
    septic_shock = (
        _hadm_set_dx_exact(SEPTIC_SHOCK_ICD10, True)
        | _hadm_set_dx_exact(SEPTIC_SHOCK_ICD9, False)
    )
    sepsis_broad = (
        _hadm_set_dx(SEPSIS_BROAD_ICD10_PREFIX, True)
        | _hadm_set_dx(SEPSIS_BROAD_ICD9_PREFIX, False)
    ) | septic_shock
    sepsis_no_shock = sepsis_broad - septic_shock

    ards = (
        _hadm_set_dx(ARDS_ICD10_PREFIX, True)
        | _hadm_set_dx(ARDS_ICD9_PREFIX, False)
    )
    resp_failure_broad = (
        _hadm_set_dx(RESP_FAILURE_ICD10_PREFIX, True)
        | _hadm_set_dx(RESP_FAILURE_ICD9_PREFIX, False)
    ) | ards
    resp_failure_other = resp_failure_broad - ards

    neuro = (
        _hadm_set_dx(NEURO_ICD10_PREFIX, True)
        | _hadm_set_dx(NEURO_ICD9_PREFIX, False)
    )
    trauma = (
        _hadm_set_dx(TRAUMA_ICD10_PREFIX, True)
        | _hadm_set_dx(TRAUMA_ICD9_PREFIX, False)
    )
    hemorrhage = (
        _hadm_set_dx_exact(HEMORRHAGE_ICD10, True)
        | _hadm_set_dx(HEMORRHAGE_ICD9_PREFIX, False)
    )
    aki = (
        _hadm_set_dx_exact(AKI_ICD10, True)
        | _hadm_set_dx_exact(AKI_ICD9, False)
    )

    # Surgery: split cardiac vs non-cardiac using PROCEDURE codes.
    cardiac_proc = (
        _hadm_set_px(CARDIAC_PROC_ICD10_PCS_PREFIX, True)
        | _hadm_set_px(CARDIAC_PROC_ICD9_PREFIX, False)
    )
    any_proc = set(px_hadm.tolist())
    noncardiac_proc = any_proc - cardiac_proc

    # Build per-hadm result.
    result = pd.DataFrame({"hadm_id": sorted(hadm_ids)})
    flags = {
        "is_cardiogenic_shock": cardiogenic_shock,
        "is_septic_shock": septic_shock,
        "is_sepsis_no_shock": sepsis_no_shock,
        "is_ards": ards,
        "is_resp_failure_other": resp_failure_other,
        "is_neurocritical": neuro,
        "is_trauma": trauma,
        "is_hemorrhage": hemorrhage,
        "is_aki_primary": aki,
        "has_cardiac_procedure": cardiac_proc,
        "has_noncardiac_procedure_only": noncardiac_proc,
    }
    for col, hadm_set in flags.items():
        result[col] = result["hadm_id"].isin(hadm_set)

    # Mutually-exclusive priority label.
    # Order: cardiogenic_shock > septic_shock > sepsis_no_shock > ards >
    #        resp_failure_other > neurocritical > trauma_hemorrhage >
    #        post_op_cardiac > post_op_noncardiac > aki_primary > other.
    result["acute_driver_refined"] = "other"
    priority = [
        ("aki_primary", result["is_aki_primary"]),
        ("post_op_noncardiac", result["has_noncardiac_procedure_only"]),
        ("post_op_cardiac", result["has_cardiac_procedure"]),
        ("trauma_hemorrhage", result["is_trauma"] | result["is_hemorrhage"]),
        ("neurocritical", result["is_neurocritical"]),
        ("resp_failure_other", result["is_resp_failure_other"]),
        ("ards", result["is_ards"]),
        ("sepsis_no_shock", result["is_sepsis_no_shock"]),
        ("septic_shock", result["is_septic_shock"]),
        ("cardiogenic_shock", result["is_cardiogenic_shock"]),
    ]
    # Apply lowest-priority first so highest priority overwrites.
    for label, mask in priority:
        result.loc[mask, "acute_driver_refined"] = label

    return result


def main() -> None:
    cfg = load_config()
    staging_dir = resolve_path(cfg, "staging")
    reports_dir = resolve_path(cfg, "reports")
    staging_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("P11_acute_illness_driver_refined — refined Table-1 acute drivers")
    print("=" * 70)

    print("\n[1/3] Loading inputs ...")
    cohort = pd.read_parquet(
        staging_dir / "25a_time_zero_flags_mimic.parquet",
        columns=["stay_id", "subject_id", "hadm_id",
                 "hf_subtype", "comparator_group"],
    )
    n_index = len(cohort)
    hadm_ids = set(cohort["hadm_id"].unique())
    print(f"  Index stays: {n_index:,}  ({len(hadm_ids):,} admissions)")

    dx = pd.read_parquet(
        staging_dir / "10b_diagnoses_mimic.parquet",
        columns=["hadm_id", "icd_code", "icd_version"],
    )
    px = pd.read_parquet(
        staging_dir / "10b_procedures_mimic.parquet",
        columns=["hadm_id", "icd_code", "icd_version"],
    )
    print(f"  Diagnoses: {len(dx):,} rows  Procedures: {len(px):,} rows")

    print("\n[2/3] Classifying refined acute drivers ...")
    drivers = classify_drivers(hadm_ids, dx, px)
    out = cohort.merge(drivers, on="hadm_id", how="left")
    out["acute_driver_refined"] = out["acute_driver_refined"].fillna("other")
    flag_cols = [c for c in drivers.columns if c.startswith(("is_", "has_"))]
    for c in flag_cols:
        out[c] = out[c].fillna(False)

    dist = out["acute_driver_refined"].value_counts().sort_index()
    for cat, n in dist.items():
        print(f"  {cat:25s}: {n:>8,}  ({n / n_index * 100:.1f}%)")

    out = out.sort_values("stay_id").reset_index(drop=True)
    out_path = staging_dir / "P11_acute_illness_driver_refined_mimic.parquet"
    out.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"\n  → {out_path}  ({len(out):,} rows)")

    print("\n[3/3] Generating summary report ...")

    cross_label = pd.crosstab(
        out["acute_driver_refined"], out["comparator_group"], margins=True,
    )
    flag_summary = pd.DataFrame({
        col: out.groupby("comparator_group")[col].sum()
        for col in flag_cols
    }).T
    flag_summary["total"] = out[flag_cols].sum(axis=0)
    flag_summary = flag_summary.astype(int)

    report = textwrap.dedent(f"""\
    ================================================================
    P11_acute_illness_driver_refined — Summary Report
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ================================================================

    PRIORITY LABEL DISTRIBUTION
    ────────────────────────────
    {textwrap.indent(dist.to_string(), '  ')}

    PRIORITY LABEL × COMPARATOR GROUP
    ───────────────────────────────────
    {textwrap.indent(cross_label.to_string(), '  ')}

    BINARY-FLAG COUNTS BY COMPARATOR GROUP
    ────────────────────────────────────────
    {textwrap.indent(flag_summary.to_string(), '  ')}

    NOTES
    ─────
    - Flags are NOT mutually exclusive; a single admission can carry several
      (e.g. septic shock + AKI + ARDS).
    - acute_driver_refined applies the priority order:
      cardiogenic_shock > septic_shock > sepsis_no_shock > ards >
      resp_failure_other > neurocritical > trauma_hemorrhage >
      post_op_cardiac > post_op_noncardiac > aki_primary > other.
    - Cardiac procedures use ICD-10-PCS section 02 (Heart and Great Vessels)
      and ICD-9 procedure prefixes 35-37; non-cardiac procedure flag is set
      when an admission has any procedure code OUTSIDE that range and no
      cardiac procedure (mutually exclusive with has_cardiac_procedure).
    - This patch supplements 25c without replacing it; both files coexist.

    OUTPUT
    ──────
    {out_path}  ({len(out):,} rows, {out.shape[1]} cols)
    """)

    report_path = reports_dir / "P11_acute_illness_driver_refined_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"  → Report: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
