#!/usr/bin/env python3
"""
30f_phase05_decision_report — Formal Phase 0.5 go/no-go decision memo.

Purpose
-------
Aggregates results from Steps 29a-30e into a formal go/no-go decision
report for Phase 0.5. Evaluates all hard criteria and produces a
structured decision memo.

Hard criteria (all must pass):
1. Minimum eligible N (HFpEF ≥ 1,500)
2. Adequate joint-action cell counts under ≥1 candidate scheme
3. Sufficient 0-6h information density for early phenotyping
4. ≥80% of cohort with LOS covering 24h primary horizon
5. Tolerable expected trimming loss

Inputs
------
- tables/30a_phase05_feasibility_table.csv
- tables/30b_density_0_6h.csv
- tables/30c_action_joint_pre_t0.csv
- tables/30d_los_horizon_coverage.csv
- tables/30e_positivity_precheck.csv
- tables/29a_action_cell_counts.csv
- tables/29b_action_space_feasibility_sweep.csv
- tables/29c_back_of_envelope_support.csv

Outputs
-------
- reports/30f_phase05_decision_report.md
- reports/30f_phase05_decision_report_summary.txt
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path

# Hard thresholds
MIN_HFPEF_N = 1500
MIN_LOS_24H_COVERAGE = 80.0  # percent
MIN_CORE_DENSITY_VARS = 3     # minimum core variables for phenotyping

# Granularity ranking for scheme selection (higher = more granular)
SCHEME_GRANULARITY = {
    "coarse_3": 1,
    "medium_4": 2,
    "fine_5": 3,
}


def check_criterion(name: str, passed: bool, detail: str) -> dict:
    return {
        "criterion": name,
        "status": "PASS" if passed else "FAIL",
        "detail": detail,
    }


def main() -> None:
    cfg = load_config()
    tables_dir = resolve_path(cfg, "tables")
    reports_dir = resolve_path(cfg, "reports")
    for d in [tables_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("[30f] Generating Phase 0.5 decision report...")

    criteria = []

    # ── Criterion 1: Minimum HFpEF N ─────────────────────────────────────
    try:
        nflow = pd.read_csv(tables_dir / "30a_phase05_feasibility_table.csv")
        phase05_row = nflow[nflow["stage"].str.contains("gate_passed|phase05", case=False)]
        if len(phase05_row) > 0:
            hfpef_n = int(phase05_row.iloc[0].get("hfpef", 0))
        else:
            hfpef_n = 0
        criteria.append(check_criterion(
            "C1: Minimum HFpEF N (≥1,500)",
            hfpef_n >= MIN_HFPEF_N,
            f"HFpEF post-gate = {hfpef_n:,}"
        ))
    except FileNotFoundError:
        criteria.append(check_criterion("C1: Minimum HFpEF N", False, "30a output not found"))
        hfpef_n = 0

    # ── Criterion 2: Joint-action cell support ────────────────────────────
    try:
        sweep = pd.read_csv(tables_dir / "29b_action_space_feasibility_sweep.csv")
        hfpef_sweep = sweep[sweep["subgroup"] == "hfpef"]
        # At least one scheme with all cells ≥25 stays
        any_scheme_passes = any(
            row["cells_below_25_stays"] == 0
            for _, row in hfpef_sweep.iterrows()
        )
        best_scheme = None
        if any_scheme_passes:
            passing = hfpef_sweep[hfpef_sweep["cells_below_25_stays"] == 0].copy()
            # Select most granular passing scheme
            passing["granularity"] = passing["scheme"].map(SCHEME_GRANULARITY).fillna(0)
            passing = passing.sort_values("granularity", ascending=False)
            best_scheme = passing.iloc[0]["scheme"]
        detail = f"Most granular scheme with all HFpEF cells ≥25: {best_scheme or 'none'}"
        criteria.append(check_criterion(
            "C2: Joint-action cell support",
            any_scheme_passes,
            detail
        ))
    except FileNotFoundError:
        criteria.append(check_criterion("C2: Joint-action cell support", False, "29b output not found"))
        best_scheme = None

    # ── Criterion 3: Information density ──────────────────────────────────
    try:
        density = pd.read_csv(tables_dir / "30b_density_0_6h.csv")
        core = density[density["density_tier"] == "high_completeness_core"]
        n_core = len(core)
        criteria.append(check_criterion(
            f"C3: 0-6h density (≥{MIN_CORE_DENSITY_VARS} core variables)",
            n_core >= MIN_CORE_DENSITY_VARS,
            f"{n_core} high-completeness core variables identified"
        ))
    except FileNotFoundError:
        criteria.append(check_criterion("C3: Information density", False, "30b output not found"))

    # ── Criterion 4: LOS coverage (absorbing-state-aware) ─────────────────
    try:
        los = pd.read_csv(tables_dir / "30d_los_horizon_coverage.csv")
        all_24 = los[(los["horizon_hours"] == 24) & (los["subgroup"] == "all")]
        # Use absorbing-state-aware coverage if available, else fall back
        if len(all_24) > 0:
            row_24 = all_24.iloc[0]
            if "full_coverage_pct" in row_24.index:
                cov_24 = row_24["full_coverage_pct"]
                raw_24 = row_24.get("raw_coverage_pct", cov_24)
                detail = (
                    f"24h full coverage (absorbing-state-aware) = {cov_24:.1f}% "
                    f"[raw LOS-only = {raw_24:.1f}%]"
                )
            else:
                cov_24 = row_24["coverage_pct"]
                detail = f"24h coverage = {cov_24:.1f}% (pre-absorbing-state fix)"
        else:
            cov_24 = 0
            detail = "no 24h row found"

        # Also report HFpEF-specific coverage
        hfpef_24 = los[(los["horizon_hours"] == 24) & (los["subgroup"] == "hfpef")]
        if len(hfpef_24) > 0:
            hp_row = hfpef_24.iloc[0]
            hp_cov = hp_row.get("full_coverage_pct", hp_row.get("coverage_pct", 0))
            detail += f"; HFpEF = {hp_cov:.1f}%"

        criteria.append(check_criterion(
            f"C4: 24h LOS coverage (≥{MIN_LOS_24H_COVERAGE:.0f}%)",
            cov_24 >= MIN_LOS_24H_COVERAGE,
            detail
        ))
    except FileNotFoundError:
        criteria.append(check_criterion("C4: LOS coverage", False, "30d output not found"))
        cov_24 = 0

    # ── Criterion 5: Trimming loss ────────────────────────────────────────
    try:
        ess = pd.read_csv(tables_dir / "29c_back_of_envelope_support.csv")
        hfpef_ess = ess[(ess["subgroup"] == "hfpef") & (ess["trim_lo"] == 0.05)]
        if len(hfpef_ess) > 0:
            ess_ratio = hfpef_ess.iloc[0]["ess_ratio_approx"]
        else:
            ess_ratio = 0
        criteria.append(check_criterion(
            "C5: ESS retention (≥30% at [0.05,0.95] trim)",
            ess_ratio >= 0.30,
            f"ESS ratio ≈ {ess_ratio:.1%}"
        ))
    except FileNotFoundError:
        criteria.append(check_criterion("C5: Trimming loss", False, "29c output not found"))

    # ── Overall decision ──────────────────────────────────────────────────
    all_pass = all(c["status"] == "PASS" for c in criteria)

    if all_pass:
        decision = "GO — All criteria passed. Proceed to Step C development."
        cohort_strategy = "Broad HFpEF primary cohort (strict definition)"
    else:
        failed = [c["criterion"] for c in criteria if c["status"] == "FAIL"]
        decision = f"CONDITIONAL — {len(failed)} criteria failed: {', '.join(failed)}"
        cohort_strategy = "Review failed criteria; may need action-space coarsening or fallback cohort"

    # ── Generate markdown report ──────────────────────────────────────────
    md_lines = [
        "# 30f — Phase 0.5 Decision Report",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d')}",
        f"**Protocol version:** v3.1",
        f"**Database:** MIMIC-IV",
        "",
        "## Decision",
        "",
        f"**{decision}**",
        "",
        f"**Cohort strategy:** {cohort_strategy}",
        "",
        "## Criteria Evaluation",
        "",
        "| Criterion | Status | Detail |",
        "|-----------|--------|--------|",
    ]
    for c in criteria:
        md_lines.append(f"| {c['criterion']} | {c['status']} | {c['detail']} |")

    md_lines += [
        "",
        "## Key Numbers",
        "",
        f"- HFpEF post-gate: {hfpef_n:,}",
        f"- Recommended action scheme: {best_scheme or 'TBD after execution'}",
        f"- 24h LOS coverage: {cov_24:.1f}%",
        "",
        "## Recommended Next Steps",
        "",
    ]
    if all_pass:
        md_lines += [
            "1. Freeze action-space scheme (recommended: most granular passing scheme)",
            "2. Proceed to Step 40x (covariate panel construction)",
            "3. Implement 45a-45f (data freeze and absorbing-state handling)",
            "4. Begin Step B (early phenotyping) in parallel",
            "",
            "## Contingency Plan",
            "",
            "- If downstream positivity analysis (Step C) reveals issues not caught here,",
            "  coarsen action space to next coarser scheme",
            "- Diastolic HF ICD proxy available as sensitivity analysis (n≈11,087)",
            "- Syndrome-matched fallback pre-registered if broad cohort fails during modeling",
        ]
    else:
        md_lines += [
            "1. Address failing criteria before proceeding",
            "2. Consider action-space coarsening if C2 fails",
            "3. Consider syndrome-matched fallback if C1 fails",
            "4. Escalate to protocol review if multiple criteria fail",
        ]

    md_lines += [
        "",
        "## Protocol Consistency",
        "",
        "- No locked decisions changed",
        "- Broad HFpEF remains primary unless fallback triggered",
        "- This report operationalizes protocol §2.7 Phase 0.5 gate",
    ]

    # ── Save ──────────────────────────────────────────────────────────────
    md_path = reports_dir / "30f_phase05_decision_report.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"  Saved: {md_path}")

    # Text summary
    report_lines = [
        "=" * 72,
        "30f — Phase 0.5 Decision Report Summary",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"DECISION: {decision}",
        f"COHORT: {cohort_strategy}",
        "",
        "CRITERIA:",
    ]
    for c in criteria:
        report_lines.append(f"  [{c['status']}] {c['criterion']}: {c['detail']}")

    report_path = reports_dir / "30f_phase05_decision_report_summary.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Saved: {report_path}")

    print(f"\n  DECISION: {decision}")
    print("[30f] Done.")


if __name__ == "__main__":
    main()
