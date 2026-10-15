#!/usr/bin/env python3
"""
25f_freeze_primary_cohort — Consolidate Step 25 into cohort-candidate manifest.

Purpose
-------
Merge all Step 25 outputs (25a–25e) with 20f cohort candidates into a
single cohort-candidate manifest ready for Phase 0.5 assessment.

Key derived fields:
  - include_in_phase05: index stay + hemodynamic gate passed
  - hfpef_phase05: HFpEF + hemodynamic gate passed
  - Phase 0.5 guardrail evaluation with explicit thresholds

Inputs
------
- data/staging/20f_cohort_candidates_mimic.parquet
- data/staging/25a_time_zero_flags_mimic.parquet
- data/staging/25b_hemodynamic_gate_flags_mimic.parquet
- data/staging/25c_reason_for_icu_flags_mimic.parquet
- data/staging/25d_dominant_syndrome_flags_mimic.parquet
- data/staging/25e_goals_of_care_rrt_flags_mimic.parquet

Outputs
-------
- data/staging/25f_primary_cohort_candidate_v1_mimic.parquet
- reports/25f_cohort_candidate_summary_mimic.txt
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
    print("25f_freeze_primary_cohort — Cohort candidate manifest v1")
    print("=" * 70)

    # ── load all inputs ──────────────────────────────────────────────
    print("\n[1/4] Loading inputs ...")

    cohort_20f = pd.read_parquet(staging_dir / "20f_cohort_candidates_mimic.parquet")
    idx = cohort_20f[cohort_20f["is_index_stay"] == True].copy()  # noqa: E712
    n_index = len(idx)
    print(f"  20f index stays: {n_index:,}")

    ts_25a = pd.read_parquet(staging_dir / "25a_time_zero_flags_mimic.parquet")
    gate_25b = pd.read_parquet(staging_dir / "25b_hemodynamic_gate_flags_mimic.parquet")
    icu_25c = pd.read_parquet(staging_dir / "25c_reason_for_icu_flags_mimic.parquet")
    synd_25d = pd.read_parquet(staging_dir / "25d_dominant_syndrome_flags_mimic.parquet")
    goc_25e = pd.read_parquet(staging_dir / "25e_goals_of_care_rrt_flags_mimic.parquet")

    # ── merge on stay_id ─────────────────────────────────────────────
    print("\n[2/4] Merging 25a–25e ...")
    key = "stay_id"

    # Start from 20f index stays (full columns)
    out = idx.copy()

    # 25a: time-zero columns (drop duplicates from 20f)
    ts_cols = [c for c in ts_25a.columns if c not in out.columns or c == key]
    out = out.merge(ts_25a[ts_cols], on=key, how="inner")
    print(f"  After 25a merge: {len(out):,}")

    # 25b: hemodynamic gate (select unique columns)
    gate_cols = [key] + [c for c in gate_25b.columns
                         if c not in out.columns and c != key]
    out = out.merge(gate_25b[gate_cols], on=key, how="left")
    print(f"  After 25b merge: {len(out):,}")

    # 25c: reason for ICU
    icu_cols = [key] + [c for c in icu_25c.columns
                        if c not in out.columns and c != key]
    out = out.merge(icu_25c[icu_cols], on=key, how="left")
    print(f"  After 25c merge: {len(out):,}")

    # 25d: dominant syndrome
    synd_cols = [key] + [c for c in synd_25d.columns
                         if c not in out.columns and c != key]
    out = out.merge(synd_25d[synd_cols], on=key, how="left")
    print(f"  After 25d merge: {len(out):,}")

    # 25e: goals of care / RRT
    goc_cols = [key] + [c for c in goc_25e.columns
                        if c not in out.columns and c != key]
    out = out.merge(goc_25e[goc_cols], on=key, how="left")
    print(f"  After 25e merge: {len(out):,}")

    # ── Phase 0.5 eligibility flags ──────────────────────────────────
    print("\n[3/4] Computing Phase 0.5 eligibility ...")

    # Fill missing gate flag (shouldn't happen, but safety)
    if "hemo_gate_passed" not in out.columns:
        out["hemo_gate_passed"] = False
    out["hemo_gate_passed"] = out["hemo_gate_passed"].fillna(False)

    out["include_in_phase05"] = out["hemo_gate_passed"]
    out["hfpef_phase05"] = (out["hf_subtype"] == "hfpef") & out["hemo_gate_passed"]

    n_phase05 = int(out["include_in_phase05"].sum())
    n_hfpef_phase05 = int(out["hfpef_phase05"].sum())
    print(f"  Phase 0.5 eligible: {n_phase05:,}")
    print(f"  HFpEF Phase 0.5:    {n_hfpef_phase05:,}")

    # ── write output ─────────────────────────────────────────────────
    out = out.sort_values(["subject_id", "stay_id"]).reset_index(drop=True)
    out_path = staging_dir / "25f_primary_cohort_candidate_v1_mimic.parquet"
    out.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"\n  → {out_path}  ({len(out):,} rows, {out.shape[1]} cols)")

    # ── comprehensive summary report ─────────────────────────────────
    print("\n[4/4] Generating summary report ...")

    # Per-group gate rates
    group_summary = (
        out.groupby("comparator_group")
        .agg(
            total=(key, "count"),
            gate_passed=("hemo_gate_passed", "sum"),
        )
    )
    group_summary["pass_pct"] = (
        group_summary["gate_passed"] / group_summary["total"] * 100
    ).round(1)

    # Reason-for-ICU distribution
    reason_dist = out["reason_for_icu"].value_counts() if "reason_for_icu" in out.columns else pd.Series(dtype=int)

    # Syndrome distribution (among gate-passers)
    phase05_sub = out[out["include_in_phase05"]]
    if "dominant_syndrome" in phase05_sub.columns:
        synd_dist = phase05_sub["dominant_syndrome"].value_counts()
        synd_by_hf = pd.crosstab(
            phase05_sub["dominant_syndrome"],
            phase05_sub["comparator_group"],
            margins=True,
        )
    else:
        synd_dist = pd.Series(dtype=int)
        synd_by_hf = pd.DataFrame()

    # HFpEF syndrome detail
    hfpef_phase05 = phase05_sub[phase05_sub["hf_subtype"] == "hfpef"]
    if "dominant_syndrome" in hfpef_phase05.columns and len(hfpef_phase05) > 0:
        hfpef_synd = hfpef_phase05["dominant_syndrome"].value_counts()
    else:
        hfpef_synd = pd.Series(dtype=int)

    # Guardrail check
    if n_hfpef_phase05 >= 1500:
        guardrail = f"  PASS — HFpEF post-gate = {n_hfpef_phase05:,} (≥1,500): proceed with strict HFpEF"
    elif n_hfpef_phase05 >= 1200:
        guardrail = (
            f"  CAUTION — HFpEF post-gate = {n_hfpef_phase05:,} (1,200–1,499): "
            f"activate diastolic ICD proxy sensitivity"
        )
    else:
        guardrail = (
            f"  ALERT — HFpEF post-gate = {n_hfpef_phase05:,} (<1,200): "
            f"escalate to Phase 0.5 review before Step 30"
        )

    report = textwrap.dedent(f"""\
    ================================================================
    25f_freeze_primary_cohort — Cohort Candidate Summary v1
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ================================================================

    COHORT FLOW
    ────────────
    All ICU stays (20f)       : {len(cohort_20f):>10,}
    Index stays               : {n_index:>10,}
    Phase 0.5 eligible (gate) : {n_phase05:>10,}  ({n_phase05 / n_index * 100:.1f}%)

    HEMODYNAMIC GATE BY COMPARATOR GROUP
    ──────────────────────────────────────
    {textwrap.indent(group_summary.to_string(), '  ')}

    HFpEF PHASE 0.5 DETAIL
    ────────────────────────
    HFpEF index stays         : {int((out['hf_subtype'] == 'hfpef').sum()):>10,}
    HFpEF gate passed         : {n_hfpef_phase05:>10,}

    PHASE 0.5 GUARDRAIL
    ─────────────────────
    {guardrail}

    REASON FOR ICU (all index stays)
    ──────────────────────────────────
    {textwrap.indent(reason_dist.to_string(), '  ') if len(reason_dist) > 0 else '  (not available)'}

    DOMINANT SYNDROME (Phase 0.5 eligible)
    ───────────────────────────────────────
    {textwrap.indent(synd_dist.to_string(), '  ') if len(synd_dist) > 0 else '  (not available)'}

    HFpEF SYNDROME DISTRIBUTION (Phase 0.5 eligible)
    ──────────────────────────────────────────────────
    {textwrap.indent(hfpef_synd.to_string(), '  ') if len(hfpef_synd) > 0 else '  (not available)'}

    SYNDROME × COMPARATOR (Phase 0.5 eligible)
    ────────────────────────────────────────────
    {textwrap.indent(synd_by_hf.to_string(), '  ') if len(synd_by_hf) > 0 else '  (not available)'}

    OUTPUT
    ──────
    {out_path}  ({len(out):,} rows, {out.shape[1]} cols)
    """)

    report_path = reports_dir / "25f_cohort_candidate_summary_mimic.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"  → Report: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
