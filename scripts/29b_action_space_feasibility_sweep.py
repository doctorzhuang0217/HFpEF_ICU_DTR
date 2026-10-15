#!/usr/bin/env python3
"""
29b_action_space_feasibility_sweep — Feasibility sweep across candidate
action-space schemes.

Purpose
-------
For each candidate scheme defined in 29a, evaluate:
  - Per-action cell counts overall and by HF subtype
  - Minimum cell count / proportion below threshold
  - Merged-bin fallback mapping suggestions
  - Trade-off summary: support vs interpretability vs granularity

Also incorporates 29c back-of-envelope support calculation:
  - Joint-action cell counts
  - Expected trimming losses at each positivity clip level
  - Approximate per-phenotype effective sample size

Protocol requirement: Assumption #1 (Consistency) and #2 (Positivity)
from 03g assumption-to-diagnostic map.

Inputs
------
- tables/29a_action_cell_counts.csv     (from 29a)
- configs/29a_candidate_action_schemes.yaml (from 29a)
- data/staging/25f_primary_cohort_candidate_v1_mimic.parquet
- data/staging/10f_meds_infusions_mimic.parquet
- data/staging/10c_timestamps_mimic.parquet

Outputs
-------
- tables/29b_action_space_feasibility_sweep.csv
- tables/29c_back_of_envelope_support.csv
- reports/29b_action_space_feasibility_sweep_summary.txt
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path

# Phase 0.5 thresholds (from protocol + phase05_step29_30_review_recommendations.md)
MIN_CELL_STAYS_HARD = 25       # must merge if below
MIN_CELL_STAYS_PREFERRED = 50  # preferred minimum
MIN_CELL_STAYS_PRIMARY = 100   # preferred for primary analysis bins
ESS_RETENTION_MIN = 0.30       # minimum ESS/N ratio
ESS_RETENTION_DESIRED = 0.40   # desired ESS/N ratio

# Positivity trimming grid (from global config)
TRIMMING_GRID = [
    (0.05, 0.95),
    (0.10, 0.90),
    (0.15, 0.85),
    (0.20, 0.80),
]


def evaluate_scheme_support(cell_counts: pd.DataFrame, scheme_name: str) -> dict:
    """Evaluate support metrics for one scheme."""
    results = {}
    for subgroup in ["all", "hfpef"]:
        sub = cell_counts[
            (cell_counts["scheme"] == scheme_name) &
            (cell_counts["subgroup"] == subgroup)
        ]
        if len(sub) == 0:
            continue

        n_cells_total = len(sub)
        n_nonempty = (sub["n_observations"] > 0).sum()
        n_empty = n_cells_total - n_nonempty

        nonempty = sub[sub["n_observations"] > 0]

        results[subgroup] = {
            "n_cells_total": n_cells_total,
            "n_nonempty": int(n_nonempty),
            "n_empty": int(n_empty),
            "empty_pct": round(n_empty / n_cells_total * 100, 1) if n_cells_total > 0 else 0,
            "min_obs_per_cell": int(nonempty["n_observations"].min()) if len(nonempty) > 0 else 0,
            "median_obs_per_cell": int(nonempty["n_observations"].median()) if len(nonempty) > 0 else 0,
            "min_stays_per_cell": int(nonempty["n_stays"].min()) if len(nonempty) > 0 else 0,
            "median_stays_per_cell": int(nonempty["n_stays"].median()) if len(nonempty) > 0 else 0,
            "cells_below_25_stays": int((nonempty["n_stays"] < 25).sum()),
            "cells_below_50_stays": int((nonempty["n_stays"] < 50).sum()),
            "cells_below_100_stays": int((nonempty["n_stays"] < 100).sum()),
            "total_observations": int(sub["n_observations"].sum()),
            "total_stays": int(nonempty["n_stays"].sum()) if len(nonempty) > 0 else 0,
        }
    return results


def back_of_envelope_ess(cell_counts: pd.DataFrame, scheme_name: str) -> pd.DataFrame:
    """Back-of-envelope ESS estimation.

    Approximates ESS under uniform trimming by computing:
      ESS ≈ N × (1 - fraction_in_extreme_cells)

    where extreme cells are those whose marginal probability would fall
    outside the trimming bounds.
    """
    rows = []
    for subgroup in ["all", "hfpef"]:
        sub = cell_counts[
            (cell_counts["scheme"] == scheme_name) &
            (cell_counts["subgroup"] == subgroup)
        ]
        if len(sub) == 0 or sub["n_observations"].sum() == 0:
            continue

        total_obs = sub["n_observations"].sum()
        nonempty = sub[sub["n_observations"] > 0].copy()
        nonempty["prop"] = nonempty["n_observations"] / total_obs

        for lo, hi in TRIMMING_GRID:
            # Observations in cells with proportion outside [lo, hi]
            # This is a rough approximation: in practice, trimming is on
            # propensity scores, not cell proportions. But cell proportions
            # give a lower bound on how much mass is near the boundary.
            extreme_mask = (nonempty["prop"] < lo / len(nonempty)) | (nonempty["prop"] > hi)
            extreme_obs = nonempty.loc[extreme_mask, "n_observations"].sum()
            trimmed_pct = extreme_obs / total_obs * 100

            retained_obs = total_obs - extreme_obs
            ess_ratio = retained_obs / total_obs if total_obs > 0 else 0

            rows.append({
                "scheme": scheme_name,
                "subgroup": subgroup,
                "trim_lo": lo,
                "trim_hi": hi,
                "total_obs": int(total_obs),
                "extreme_obs": int(extreme_obs),
                "retained_obs": int(retained_obs),
                "trimmed_pct": round(trimmed_pct, 2),
                "ess_ratio_approx": round(ess_ratio, 4),
                "ess_pass_min": ess_ratio >= ESS_RETENTION_MIN,
                "ess_pass_desired": ess_ratio >= ESS_RETENTION_DESIRED,
            })

    return pd.DataFrame(rows)


def suggest_merges(cell_counts: pd.DataFrame, scheme_name: str,
                   subgroup: str = "hfpef") -> list[str]:
    """Suggest bin merges for cells below minimum support threshold."""
    sub = cell_counts[
        (cell_counts["scheme"] == scheme_name) &
        (cell_counts["subgroup"] == subgroup) &
        (cell_counts["n_stays"] < MIN_CELL_STAYS_HARD) &
        (cell_counts["n_observations"] > 0)
    ]
    suggestions = []
    if len(sub) > 0:
        for _, row in sub.iterrows():
            suggestions.append(
                f"  Merge candidate: fluid={row['fluid_cat']}, "
                f"pressor={row['pressor_cat']}, diuretic={row['diuretic_cat']} "
                f"(n_stays={row['n_stays']})"
            )
    return suggestions


def main() -> None:
    cfg = load_config()
    tables_dir = resolve_path(cfg, "tables")
    reports_dir = resolve_path(cfg, "reports")
    for d in [tables_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("[29b] Loading 29a cell counts...")
    cell_counts = pd.read_csv(tables_dir / "29a_action_cell_counts.csv")

    schemes = cell_counts["scheme"].unique()
    print(f"  Schemes found: {list(schemes)}")

    # ── Evaluate each scheme ──────────────────────────────────────────────
    sweep_rows = []
    ess_rows = []
    report_lines = [
        "=" * 72,
        "29b — Action-Space Feasibility Sweep + 29c Back-of-Envelope Support",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Phase 0.5 support thresholds:",
        f"  Hard minimum per cell: {MIN_CELL_STAYS_HARD} stays",
        f"  Preferred minimum: {MIN_CELL_STAYS_PREFERRED} stays",
        f"  Primary analysis preferred: {MIN_CELL_STAYS_PRIMARY} stays",
        f"  ESS retention minimum: {ESS_RETENTION_MIN:.0%}",
        f"  ESS retention desired: {ESS_RETENTION_DESIRED:.0%}",
        "",
    ]

    for scheme_name in schemes:
        print(f"\n[29b] Evaluating scheme: {scheme_name}")
        support = evaluate_scheme_support(cell_counts, scheme_name)

        report_lines.append("─" * 72)
        report_lines.append(f"SCHEME: {scheme_name}")
        report_lines.append("─" * 72)

        for subgroup, metrics in support.items():
            report_lines.append(f"\n  [{subgroup}]")
            for k, v in metrics.items():
                report_lines.append(f"    {k}: {v}")

            sweep_rows.append({
                "scheme": scheme_name,
                "subgroup": subgroup,
                **metrics,
            })

            # Feasibility verdict
            if subgroup == "hfpef":
                cells_bad = metrics["cells_below_25_stays"]
                n_nonempty = metrics["n_nonempty"]
                if cells_bad == 0:
                    verdict = "PASS — all HFpEF cells have ≥25 stays"
                elif cells_bad <= n_nonempty * 0.2:
                    verdict = f"CONDITIONAL — {cells_bad} HFpEF cells <25 stays; consider merging"
                else:
                    verdict = f"FAIL — {cells_bad}/{n_nonempty} HFpEF cells <25 stays; must merge or coarsen"
                report_lines.append(f"    ▶ VERDICT: {verdict}")
                print(f"    [{subgroup}] {verdict}")

        # Merge suggestions
        merges = suggest_merges(cell_counts, scheme_name, "hfpef")
        if merges:
            report_lines.append(f"\n  Suggested merges for HFpEF ({len(merges)} cells):")
            for m in merges[:10]:  # limit output
                report_lines.append(m)
            if len(merges) > 10:
                report_lines.append(f"  ... and {len(merges) - 10} more")

        # Back-of-envelope ESS
        ess_df = back_of_envelope_ess(cell_counts, scheme_name)
        ess_rows.append(ess_df)

        report_lines.append(f"\n  Back-of-envelope ESS (approx):")
        for _, row in ess_df.iterrows():
            flag = "✓" if row["ess_pass_min"] else "✗"
            report_lines.append(
                f"    [{row['subgroup']}] trim=[{row['trim_lo']:.2f},{row['trim_hi']:.2f}]: "
                f"retained={row['retained_obs']:,}/{row['total_obs']:,} "
                f"({row['ess_ratio_approx']:.1%}) {flag}"
            )

    # ── Overall recommendation ────────────────────────────────────────────
    report_lines.append("")
    report_lines.append("═" * 72)
    report_lines.append("OVERALL RECOMMENDATION")
    report_lines.append("═" * 72)
    report_lines.append("")
    report_lines.append(
        "Recommendation will be populated after cluster execution.\n"
        "Decision rule: select the most granular scheme where ALL HFpEF\n"
        "cells have ≥25 stays. If no scheme passes, use coarse_3 with\n"
        "documented merge fallback mapping."
    )

    # ── Save outputs ──────────────────────────────────────────────────────
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_path = tables_dir / "29b_action_space_feasibility_sweep.csv"
    sweep_df.to_csv(sweep_path, index=False)
    print(f"\n  Saved: {sweep_path}")

    ess_all = pd.concat(ess_rows, ignore_index=True) if ess_rows else pd.DataFrame()
    ess_path = tables_dir / "29c_back_of_envelope_support.csv"
    ess_all.to_csv(ess_path, index=False)
    print(f"  Saved: {ess_path}")

    report_path = reports_dir / "29b_action_space_feasibility_sweep_summary.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Saved: {report_path}")

    print("\n[29b+29c] Done.")


if __name__ == "__main__":
    main()
