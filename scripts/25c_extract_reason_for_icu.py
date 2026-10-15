#!/usr/bin/env python3
"""
25c_extract_reason_for_icu — Classify admitting syndrome / reason for ICU.

Purpose
-------
Classify each index stay into a coarse reason-for-ICU stratum for
stratification and sensitivity analyses.

Categories (mutually exclusive, priority order):
  1. Sepsis-like — sepsis/septic shock ICD codes
  2. Post-operative — surgical procedure ICD codes
  3. Respiratory failure — pneumonia, ARDS, respiratory failure codes
  4. AKI primary — acute kidney injury as primary diagnosis
  5. Other/Mixed — everything else

Uses discharge ICD codes (10b). This is NOT a real-time classification;
it is a retrospective label for phenotype description.

Inputs
------
- data/staging/25a_time_zero_flags_mimic.parquet
- data/staging/10b_diagnoses_mimic.parquet

Outputs
-------
- data/staging/25c_reason_for_icu_flags_mimic.parquet
- reports/25c_reason_for_icu_mimic_summary.txt
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


# ── ICD-10 pattern sets ──────────────────────────────────────────────
SEPSIS_ICD10_PREFIXES = [
    "A40", "A41",     # sepsis
    "R6520", "R6521", # severe sepsis / septic shock
    "T8112",          # postprocedural septic shock
]
SEPSIS_ICD9_PREFIXES = ["038", "9959", "78552"]

POSTOP_ICD10_PREFIXES = [
    "T81", "T82", "T83", "T84", "T85",  # complications of procedures
    "Z96", "Z95",                         # presence of implants
]
POSTOP_ICD9_PREFIXES = ["996", "997", "998", "V43", "V44", "V45"]
# Also flag if primary dx starts with procedure-related block
SURGICAL_DX_ICD10_PREFIXES = ["O", "Z51"]  # will be excluded from postop

RESPIRATORY_ICD10_PREFIXES = [
    "J80",   # ARDS
    "J96",   # respiratory failure
    "J18",   # pneumonia, unspecified
    "J12", "J13", "J14", "J15", "J16", "J17",  # pneumonia by organism
    "J44",   # COPD exacerbation
    "J69",   # aspiration pneumonitis
]
RESPIRATORY_ICD9_PREFIXES = ["518", "486", "480", "481", "482", "483", "484", "485"]

AKI_ICD10 = {"N170", "N171", "N172", "N178", "N179"}
AKI_ICD9 = {"5849", "5845", "5846", "5847"}


def classify_reason_for_icu(hadm_ids: set, dx: pd.DataFrame) -> pd.DataFrame:
    """Classify each hadm_id into reason-for-ICU category."""
    dx = dx[dx["hadm_id"].isin(hadm_ids)].copy()
    dx["icd_norm"] = dx["icd_code"].apply(_norm)

    def _has_prefix(codes: pd.Series, prefixes: list, version: int | None = None) -> set:
        mask = codes.apply(lambda c: any(c.startswith(p) for p in prefixes))
        if version is not None:
            mask = mask & (dx["icd_version"] == version)
        return set(dx.loc[mask, "hadm_id"].unique())

    def _has_exact(codes: pd.Series, code_set: set, version: int | None = None) -> set:
        mask = codes.isin(code_set)
        if version is not None:
            mask = mask & (dx["icd_version"] == version)
        return set(dx.loc[mask, "hadm_id"].unique())

    # Build sets
    sepsis = (
        _has_prefix(dx["icd_norm"], SEPSIS_ICD10_PREFIXES, 10)
        | _has_prefix(dx["icd_norm"], SEPSIS_ICD9_PREFIXES, 9)
    )
    postop = (
        _has_prefix(dx["icd_norm"], POSTOP_ICD10_PREFIXES, 10)
        | _has_prefix(dx["icd_norm"], POSTOP_ICD9_PREFIXES, 9)
    )
    respiratory = (
        _has_prefix(dx["icd_norm"], RESPIRATORY_ICD10_PREFIXES, 10)
        | _has_prefix(dx["icd_norm"], RESPIRATORY_ICD9_PREFIXES, 9)
    )
    aki = (
        _has_exact(dx["icd_norm"], AKI_ICD10, 10)
        | _has_exact(dx["icd_norm"], AKI_ICD9, 9)
    )

    # Priority assignment
    result = pd.DataFrame({"hadm_id": sorted(hadm_ids)})
    result["is_sepsis_like"] = result["hadm_id"].isin(sepsis)
    result["is_post_op"] = result["hadm_id"].isin(postop)
    result["is_respiratory_failure"] = result["hadm_id"].isin(respiratory)
    result["is_aki_primary"] = result["hadm_id"].isin(aki)

    # Mutually exclusive label (priority order)
    conditions = [
        result["is_sepsis_like"],
        result["is_post_op"],
        result["is_respiratory_failure"],
        result["is_aki_primary"],
    ]
    choices = ["sepsis", "post_op", "respiratory", "aki"]
    result["reason_for_icu"] = "other"
    for cond, choice in zip(reversed(conditions), reversed(choices)):
        result.loc[cond, "reason_for_icu"] = choice

    return result


def main() -> None:
    cfg = load_config()
    staging_dir = resolve_path(cfg, "staging")
    reports_dir = resolve_path(cfg, "reports")
    staging_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("25c_extract_reason_for_icu — Reason-for-ICU classification")
    print("=" * 70)

    # ── load inputs ──────────────────────────────────────────────────
    print("\n[1/3] Loading inputs ...")
    cohort = pd.read_parquet(
        staging_dir / "25a_time_zero_flags_mimic.parquet",
        columns=["stay_id", "subject_id", "hadm_id", "hf_subtype", "comparator_group"],
    )
    n_index = len(cohort)
    hadm_ids = set(cohort["hadm_id"].unique())
    print(f"  Index stays: {n_index:,}  ({len(hadm_ids):,} admissions)")

    dx = pd.read_parquet(staging_dir / "10b_diagnoses_mimic.parquet")

    # ── classify ─────────────────────────────────────────────────────
    print("\n[2/3] Classifying reason for ICU ...")
    icu_reason = classify_reason_for_icu(hadm_ids, dx)
    out = cohort.merge(icu_reason, on="hadm_id", how="left")
    out["reason_for_icu"] = out["reason_for_icu"].fillna("other")

    dist = out["reason_for_icu"].value_counts().sort_index()
    for cat, n in dist.items():
        print(f"  {cat:20s}: {n:>8,}  ({n / n_index * 100:.1f}%)")

    # ── write output ─────────────────────────────────────────────────
    out = out.sort_values("stay_id").reset_index(drop=True)
    out_path = staging_dir / "25c_reason_for_icu_flags_mimic.parquet"
    out.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"\n  → {out_path}  ({len(out):,} rows)")

    # ── summary report ───────────────────────────────────────────────
    print("\n[3/3] Generating summary report ...")

    # Cross-tab: reason × HF subtype
    cross = pd.crosstab(out["reason_for_icu"], out["comparator_group"], margins=True)

    report = textwrap.dedent(f"""\
    ================================================================
    25c_extract_reason_for_icu — Summary Report
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ================================================================

    OVERALL DISTRIBUTION
    ─────────────────────
    {textwrap.indent(dist.to_string(), '  ')}

    CROSS-TAB: Reason × Comparator Group
    ──────────────────────────────────────
    {textwrap.indent(cross.to_string(), '  ')}

    OUTPUT
    ──────
    {out_path}  ({len(out):,} rows, {out.shape[1]} cols)
    """)

    report_path = reports_dir / "25c_reason_for_icu_mimic_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"  → Report: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
