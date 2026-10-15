#!/usr/bin/env python3
"""
30e_positivity_precheck — Coarse positivity and support audit.

Purpose
-------
Assess positivity (overlap) for the candidate action space before full
modeling. Uses empirical cell frequencies as a propensity proxy (does NOT
fit treatment models at this stage).

This is a COARSE precheck — not a final positivity analysis. The goal is
to identify structural positivity violations (e.g., zero-support cells)
that would make causal inference fundamentally infeasible.

Protocol reference: Assumption #2 (Positivity) from 03g.

Inputs
------
- data/staging/25f_primary_cohort_candidate_v1_mimic.parquet
- tables/29a_action_cell_counts.csv  (from 29a)

Outputs
-------
- tables/30e_positivity_precheck.csv
- reports/30e_positivity_precheck_summary.txt
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path

# Protocol-specified trimming grid
TRIMMING_GRID = [
    (0.05, 0.95),
    (0.10, 0.90),
    (0.15, 0.85),
    (0.20, 0.80),
]


def compute_empirical_propensity(cell_counts: pd.DataFrame, scheme: str,
                                  subgroup: str) -> pd.DataFrame:
    """Compute empirical marginal probabilities as propensity proxies.

    This is a back-of-envelope proxy: the empirical cell proportion serves
    as a lower bound on propensity score extremity. In practice, covariate-
    conditional propensity scores will have MORE variance, so if empirical
    proportions already violate trimming bounds, the problem is structural.

    NOTE: This function does NOT fit treatment models — it uses empirical
    cell frequencies only.  The low-end threshold is scaled by the number
    of non-empty cells (lo / n_nonempty) to avoid systematically flagging
    all cells when the action space has many categories.  This matches the
    convention used in 29b back-of-envelope ESS.
    """
    sub = cell_counts[
        (cell_counts["scheme"] == scheme) &
        (cell_counts["subgroup"] == subgroup)
    ].copy()

    total = sub["n_observations"].sum()
    if total == 0:
        return pd.DataFrame()

    sub["empirical_prop"] = sub["n_observations"] / total
    sub["log_prop"] = np.log(sub["empirical_prop"].clip(lower=1e-10))

    n_nonempty = (sub["n_observations"] > 0).sum()

    # Check against trimming grid — scale low-end by number of cells
    for lo, hi in TRIMMING_GRID:
        col = f"outside_{lo}_{hi}"
        scaled_lo = lo / max(n_nonempty, 1)
        sub[col] = (sub["empirical_prop"] < scaled_lo) | (sub["empirical_prop"] > hi)

    return sub


def main() -> None:
    cfg = load_config()
    tables_dir = resolve_path(cfg, "tables")
    reports_dir = resolve_path(cfg, "reports")
    for d in [tables_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("[30e] Running positivity precheck...")

    # ── Load cell counts from 29a ─────────────────────────────────────────
    cell_path = tables_dir / "29a_action_cell_counts.csv"
    if not cell_path.exists():
        print("  ERROR: 29a_action_cell_counts.csv not found. Run 29a first.")
        return
    cell_counts = pd.read_csv(cell_path)

    cohort = pd.read_parquet(
        resolve_path(cfg, "staging") / "25f_primary_cohort_candidate_v1_mimic.parquet"
    )
    gated = cohort[cohort["hemo_gate_passed"] == True]
    n_total = len(gated)
    n_hfpef = len(gated[gated["comparator_group"] == "hfpef"])

    print(f"  Gate-passed: {n_total:,} (HFpEF: {n_hfpef:,})")

    schemes = cell_counts["scheme"].unique()
    all_results = []

    report_lines = [
        "=" * 72,
        "30e — Positivity Precheck (Empirical Cell Proportions)",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Gate-passed: {n_total:,} (HFpEF: {n_hfpef:,})",
        "",
        "NOTE: This uses empirical cell proportions as a propensity proxy.",
        "Covariate-conditional propensity scores will have MORE variance,",
        "so violations here indicate structural positivity concerns.",
        "",
    ]

    for scheme in schemes:
        report_lines.append("─" * 72)
        report_lines.append(f"SCHEME: {scheme}")
        report_lines.append("─" * 72)

        for subgroup in ["all", "hfpef"]:
            prop_df = compute_empirical_propensity(cell_counts, scheme, subgroup)
            if len(prop_df) == 0:
                continue

            all_results.append(prop_df)

            # Summary
            n_cells = len(prop_df)
            n_empty = (prop_df["n_observations"] == 0).sum()
            n_nonempty = n_cells - n_empty

            report_lines.append(f"\n  [{subgroup}]")
            report_lines.append(f"    Total cells: {n_cells}, non-empty: {n_nonempty}, empty: {n_empty}")

            for lo, hi in TRIMMING_GRID:
                col = f"outside_{lo}_{hi}"
                n_outside = prop_df[col].sum()
                obs_outside = prop_df.loc[prop_df[col], "n_observations"].sum()
                total_obs = prop_df["n_observations"].sum()
                pct_outside = obs_outside / total_obs * 100 if total_obs > 0 else 0
                flag = ""
                if subgroup == "hfpef" and pct_outside > 15:
                    flag = " ← WARNING"
                elif subgroup == "hfpef" and pct_outside > 10:
                    flag = " ← CAUTION"
                report_lines.append(
                    f"    Trim [{lo:.2f},{hi:.2f}]: {n_outside} cells outside, "
                    f"{pct_outside:.1f}% obs affected{flag}"
                )

            # Sparsity: cells with very few observations
            if n_nonempty > 0:
                nonempty = prop_df[prop_df["n_observations"] > 0]
                p5 = nonempty["n_observations"].quantile(0.05)
                p25 = nonempty["n_observations"].quantile(0.25)
                min_obs = nonempty["n_observations"].min()
                report_lines.append(
                    f"    Obs distribution (non-empty): min={min_obs}, p5={p5:.0f}, p25={p25:.0f}"
                )

    # ── Overall verdict ───────────────────────────────────────────────────
    report_lines.append("")
    report_lines.append("═" * 72)
    report_lines.append("OVERALL POSITIVITY ASSESSMENT")
    report_lines.append("═" * 72)
    report_lines.append("")
    report_lines.append("Verdict will be populated after cluster execution.")
    report_lines.append("Decision rules:")
    report_lines.append("  - If ANY scheme has all HFpEF cells non-empty: positivity plausible")
    report_lines.append("  - If coarse_3 has >50% HFpEF cells empty: structural concern")
    report_lines.append("  - Extreme weight share >15% at [0.05,0.95] trim: escalate to review")

    # ── Save ──────────────────────────────────────────────────────────────
    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        csv_path = tables_dir / "30e_positivity_precheck.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

    report_path = reports_dir / "30e_positivity_precheck_summary.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Saved: {report_path}")

    print("\n[30e] Done.")


if __name__ == "__main__":
    main()
