#!/usr/bin/env python3
"""
20f_select_index_stay_mimic — Select one index ICU stay per patient.

Purpose
-------
Applies the one-stay-per-patient rule from the protocol: for patients with
multiple ICU stays, select the FIRST eligible stay (earliest intime among
non-excluded stays). This prevents within-patient correlation bias in
downstream causal models.

Also merges HF labels (20a) and exclusion flags (20e) into the final
cohort-candidate table that feeds Step 25 and Phase 0.5.

Inputs
------
- data/staging/10a_core_icu_stays_mimic.parquet
- data/staging/20a_hf_labels_mimic.parquet
- data/staging/20e_exclusion_flags_mimic.parquet

Outputs
-------
- data/staging/20f_cohort_candidates_mimic.parquet
- reports/20f_cohort_candidates_mimic_summary.txt
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
    print("20f_select_index_stay_mimic — Index ICU stay selection")
    print("=" * 70)

    # ── load inputs ──────────────────────────────────────────────────
    print("\n[1/4] Loading inputs ...")
    stays = pd.read_parquet(
        staging_dir / "10a_core_icu_stays_mimic.parquet",
        columns=["stay_id", "subject_id", "hadm_id", "intime", "outtime",
                 "icu_los_hours", "hospital_mortality", "age", "gender",
                 "admission_type"],
    )
    hf = pd.read_parquet(staging_dir / "20a_hf_labels_mimic.parquet")
    excl = pd.read_parquet(staging_dir / "20e_exclusion_flags_mimic.parquet")

    n_all = len(stays)
    n_subjects_all = stays["subject_id"].nunique()
    print(f"  All stays: {n_all:,}  ({n_subjects_all:,} subjects)")

    # ── merge labels and exclusions ──────────────────────────────────
    print("\n[2/4] Merging HF labels and exclusion flags ...")
    merged = stays.merge(
        hf.drop(columns=["subject_id", "hadm_id"]),
        on="stay_id", how="left",
    )
    merged = merged.merge(
        excl.drop(columns=["subject_id", "hadm_id"]),
        on="stay_id", how="left",
    )
    # Fill missing exclusion flags
    excl_cols = [c for c in merged.columns if c.startswith("excl_")]
    for c in excl_cols:
        merged[c] = merged[c].fillna(False)
    merged["any_hard_exclusion"] = merged["any_hard_exclusion"].fillna(False)

    n_excluded = merged["any_hard_exclusion"].sum()
    print(f"  Excluded stays: {n_excluded:,}")

    # ── select index stay ────────────────────────────────────────────
    print("\n[3/4] Selecting index ICU stay per patient ...")

    # Among non-excluded stays, pick the first (earliest intime) per subject
    eligible = merged[~merged["any_hard_exclusion"]].copy()
    n_eligible = len(eligible)
    n_subjects_eligible = eligible["subject_id"].nunique()
    print(f"  Eligible stays: {n_eligible:,}  ({n_subjects_eligible:,} subjects)")

    # Sort by subject_id then intime, keep first
    eligible["intime"] = pd.to_datetime(eligible["intime"], errors="coerce")
    eligible = eligible.sort_values(["subject_id", "intime", "stay_id"])
    index_stays = eligible.drop_duplicates(subset=["subject_id"], keep="first").copy()
    index_stays["is_index_stay"] = True
    n_index = len(index_stays)

    print(f"  Index stays selected: {n_index:,}")

    # Mark index stays in the full merged table
    merged["is_index_stay"] = merged["stay_id"].isin(set(index_stays["stay_id"]))

    # ── write output ─────────────────────────────────────────────────
    out_path = staging_dir / "20f_cohort_candidates_mimic.parquet"
    merged = merged.sort_values("stay_id").reset_index(drop=True)
    merged.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"\n  → {out_path}  ({len(merged):,} rows)")

    # ── summary report ───────────────────────────────────────────────
    print("\n[4/4] Generating summary report ...")

    # Subtype breakdown among index stays
    idx = merged[merged["is_index_stay"]]
    idx_subtype = idx["hf_subtype"].value_counts().to_string()
    idx_comp = idx["comparator_group"].value_counts().to_string()

    n_idx_hfpef = idx["is_hfpef"].sum()
    n_idx_hfref = idx["is_hfref"].sum()
    n_idx_hfmref = idx["is_hfmref"].sum()
    n_idx_ef_unknown = idx["is_hf_ef_unknown"].sum()
    n_idx_no_hf = idx["is_no_hf"].sum()

    idx_den = max(n_index, 1)

    report = textwrap.dedent(f"""\
    ================================================================
    20f_select_index_stay_mimic — Summary Report
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ================================================================

    COHORT FLOW
    ───────────
    All ICU stays           : {n_all:>10,}  ({n_subjects_all:,} subjects)
    Hard-excluded           : {n_excluded:>10,}
    Eligible stays          : {n_eligible:>10,}  ({n_subjects_eligible:,} subjects)
    Index stays (1/patient) : {n_index:>10,}

    HF SUBTYPE (among index stays)
    ───────────────────────────────
    {textwrap.indent(idx_subtype, '  ')}

    HFpEF (strict)          : {n_idx_hfpef:>10,}  ({n_idx_hfpef / idx_den * 100:.1f}%)
    HFmrEF                  : {n_idx_hfmref:>10,}  ({n_idx_hfmref / idx_den * 100:.1f}%)
    HFrEF                   : {n_idx_hfref:>10,}  ({n_idx_hfref / idx_den * 100:.1f}%)
    HF, EF unknown          : {n_idx_ef_unknown:>10,}  ({n_idx_ef_unknown / idx_den * 100:.1f}%)
    No HF                   : {n_idx_no_hf:>10,}  ({n_idx_no_hf / idx_den * 100:.1f}%)

    COMPARATOR GROUPS (among index stays)
    ──────────────────────────────────────
    {textwrap.indent(idx_comp, '  ')}

    OUTPUT
    ──────
    {out_path}  ({len(merged):,} rows, {merged.shape[1]} cols)
    Note: is_index_stay=True marks the selected index stay per patient.
    """)

    report_path = reports_dir / "20f_cohort_candidates_mimic_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"  → Report: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
