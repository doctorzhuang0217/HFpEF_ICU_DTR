#!/usr/bin/env python3
"""
30a_phase05_feasibility_table — Main N-flow feasibility table.

Purpose
-------
Produce the master N-flow table tracking cohort attrition from raw ICU stays
through each filtering step, stratified by HF subtype (HFpEF, HFrEF/HFmrEF,
HF-unknown, non-HF) and by database (MIMIC-IV for now; eICU/HiRID columns
reserved for future).

This is the primary evidence table for Phase 0.5 go/no-go adjudication.

Inputs
------
- data/staging/01_stays_mimic.parquet      (all ICU stays)
- data/staging/20a_hf_labels_mimic.parquet (HF phenotyping)
- data/staging/20e_exclusion_flags_mimic.parquet (hard exclusions)
- data/staging/20f_cohort_candidates_mimic.parquet (index stays)
- data/staging/25b_hemodynamic_gate_flags_mimic.parquet (gate flags)
- data/staging/25f_primary_cohort_candidate_v1_mimic.parquet (frozen cohort)

Outputs
-------
- tables/30a_phase05_feasibility_table.csv
- reports/30a_phase05_feasibility_table_summary.txt
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path


def count_by_subtype(df: pd.DataFrame, comparator_col: str = "comparator_group") -> dict:
    """Count stays by HF subtype. Returns dict with subtype keys."""
    if comparator_col not in df.columns:
        return {"total": len(df)}
    vc = df[comparator_col].value_counts()
    result = {"total": len(df)}
    for grp in ["hfpef", "hfref_hfmref", "hf_ef_unknown", "no_hf"]:
        result[grp] = int(vc.get(grp, 0))
    return result


def main() -> None:
    cfg = load_config()
    staging_dir = resolve_path(cfg, "staging")
    tables_dir = resolve_path(cfg, "tables")
    reports_dir = resolve_path(cfg, "reports")
    for d in [tables_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("[30a] Building Phase 0.5 N-flow feasibility table...")

    # ── Load all stage data ───────────────────────────────────────────────
    stays = pd.read_parquet(staging_dir / "01_stays_mimic.parquet")
    hf_labels = pd.read_parquet(staging_dir / "20a_hf_labels_mimic.parquet")
    excl_flags = pd.read_parquet(staging_dir / "20e_exclusion_flags_mimic.parquet")
    index_stays = pd.read_parquet(staging_dir / "20f_cohort_candidates_mimic.parquet")
    gate_flags = pd.read_parquet(staging_dir / "25b_hemodynamic_gate_flags_mimic.parquet")
    cohort = pd.read_parquet(staging_dir / "25f_primary_cohort_candidate_v1_mimic.parquet")

    # Merge comparator_group into each stage
    comp_map = hf_labels[["stay_id", "comparator_group"]].drop_duplicates()

    # ── Build N-flow rows ─────────────────────────────────────────────────
    stages = []

    # Stage 1: All ICU stays
    s1 = stays.merge(comp_map, on="stay_id", how="left")
    s1["comparator_group"] = s1["comparator_group"].fillna("no_hf")
    stages.append(("1_all_icu_stays", count_by_subtype(s1)))

    # Stage 2: After hard exclusions
    eligible = excl_flags[excl_flags["any_hard_exclusion"] == False]["stay_id"]
    s2 = s1[s1["stay_id"].isin(eligible)]
    stages.append(("2_after_hard_exclusions", count_by_subtype(s2)))

    # Stage 3: Index stays (one per patient)
    # 20f contains ALL stays with is_index_stay flag; filter to index stays only
    assert "is_index_stay" in index_stays.columns, (
        "20f_cohort_candidates_mimic.parquet missing 'is_index_stay' column. "
        "Cannot reliably identify index stays."
    )
    s3 = index_stays[index_stays["is_index_stay"] == True].copy()
    if "comparator_group" not in s3.columns:
        s3 = s3.merge(comp_map, on="stay_id", how="left")
        s3["comparator_group"] = s3["comparator_group"].fillna("no_hf")
    # Regression guard: index stays must be ≤ eligible stays (Stage 2)
    s2_total = stages[-1][1]["total"]  # previous stage is Stage 2
    assert len(s3) <= s2_total, (
        f"Stage 3 (index stays={len(s3):,}) > Stage 2 (post-exclusion={s2_total:,}). "
        f"Possible filter regression: check is_index_stay logic."
    )
    stages.append(("3_index_stays", count_by_subtype(s3)))

    # Stage 4: Gate-passed (hemodynamic management)
    if "hemo_gate_passed" in gate_flags.columns:
        gp_ids = gate_flags[gate_flags["hemo_gate_passed"] == True]["stay_id"]
    else:
        gp_ids = gate_flags["stay_id"]
    s4 = s3[s3["stay_id"].isin(gp_ids)]
    stages.append(("4_gate_passed", count_by_subtype(s4)))

    # Stage 5: Phase 0.5 cohort — from frozen cohort (25f) gate_passed
    s5 = cohort[cohort["hemo_gate_passed"] == True].copy()
    if "comparator_group" not in s5.columns:
        s5 = s5.merge(comp_map, on="stay_id", how="left")
        s5["comparator_group"] = s5["comparator_group"].fillna("no_hf")
    # Consistency check: frozen cohort gate-passed should match gate_flags
    assert len(s5) == len(s4), (
        f"Stage 5 (frozen cohort gate_passed={len(s5)}) != "
        f"Stage 4 (gate_flags passed={len(s4)}). "
        f"Investigate 25f freeze vs 25b gate divergence."
    )
    stages.append(("5_phase05_cohort", count_by_subtype(s5)))

    # ── Build table ───────────────────────────────────────────────────────
    rows = []
    prev_total = None
    for stage_name, counts in stages:
        row = {"stage": stage_name, "database": "MIMIC-IV"}
        row.update(counts)
        if prev_total is not None and prev_total > 0:
            row["attrition_pct"] = round((1 - counts["total"] / prev_total) * 100, 1)
        else:
            row["attrition_pct"] = 0.0
        prev_total = counts["total"]
        rows.append(row)

    nflow = pd.DataFrame(rows)

    # ── Compute exclusion breakdown ───────────────────────────────────────
    excl_cols = [c for c in excl_flags.columns if c.startswith("excl_")]
    excl_summary = {}
    for c in excl_cols:
        excl_summary[c] = int(excl_flags[c].sum())

    # ── Gate attrition by subtype ─────────────────────────────────────────
    gate_attrition = {}
    for grp in ["hfpef", "hfref_hfmref", "hf_ef_unknown", "no_hf"]:
        pre = len(s3[s3["comparator_group"] == grp])
        post = len(s4[s4["comparator_group"] == grp])
        gate_attrition[grp] = {
            "pre_gate": pre,
            "post_gate": post,
            "pass_rate_pct": round(post / pre * 100, 1) if pre > 0 else 0,
        }

    # ── Report ────────────────────────────────────────────────────────────
    report_lines = [
        "=" * 72,
        "30a — Phase 0.5 N-Flow Feasibility Table",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Database: MIMIC-IV",
        "",
        "─" * 72,
        "N-FLOW TABLE",
        "─" * 72,
        nflow.to_string(index=False),
        "",
        "─" * 72,
        "EXCLUSION BREAKDOWN (stay-level)",
        "─" * 72,
    ]
    for k, v in excl_summary.items():
        report_lines.append(f"  {k}: {v:,}")

    report_lines.append("")
    report_lines.append("─" * 72)
    report_lines.append("HEMODYNAMIC GATE PASS RATES BY SUBTYPE")
    report_lines.append("─" * 72)
    for grp, vals in gate_attrition.items():
        report_lines.append(
            f"  {grp}: {vals['post_gate']:,}/{vals['pre_gate']:,} "
            f"({vals['pass_rate_pct']:.1f}%)"
        )

    report_lines.append("")
    report_lines.append("─" * 72)
    report_lines.append("PHASE 0.5 GUARDRAIL CHECK")
    report_lines.append("─" * 72)
    hfpef_post = gate_attrition.get("hfpef", {}).get("post_gate", 0)
    if hfpef_post >= 1500:
        report_lines.append(f"  HFpEF post-gate: {hfpef_post:,} >= 1,500 → PASS")
    elif hfpef_post >= 1200:
        report_lines.append(f"  HFpEF post-gate: {hfpef_post:,} >= 1,200 → CONDITIONAL (escalate review)")
    else:
        report_lines.append(f"  HFpEF post-gate: {hfpef_post:,} < 1,200 → FAIL (activate fallback)")

    # ── Save ──────────────────────────────────────────────────────────────
    csv_path = tables_dir / "30a_phase05_feasibility_table.csv"
    nflow.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    report_path = reports_dir / "30a_phase05_feasibility_table_summary.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Saved: {report_path}")

    print("\n[30a] Done.")


if __name__ == "__main__":
    main()
