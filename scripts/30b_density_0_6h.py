#!/usr/bin/env python3
"""
30b_density_0_6h — Measure 0-6h information density for early phenotyping.

Purpose
-------
Assess measurement density in the 0-6h pre-T0 window to determine whether
trend-based (≥2 measurements) or snapshot-based phenotyping is feasible.

For each candidate covariate (vitals, labs), reports:
  - Mean measurements per patient in 0-6h
  - Proportion with ≥1, ≥2, ≥3 measurements (trend-capable)
  - Missingness rate
  - Comparison across HF subtypes

Protocol reference: Step 30b — if density too sparse, phenotyping falls back
to rule-based shock vs congestion (Step 60d).

Inputs
------
- data/staging/25f_primary_cohort_candidate_v1_mimic.parquet
- data/staging/10d_vitals_mimic.parquet
- data/staging/10e_labs_mimic.parquet

Outputs
-------
- tables/30b_density_0_6h.csv
- reports/30b_density_0_6h_summary.txt
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path

PRE_T0_HOURS = 6.0


def measure_density(events: pd.DataFrame, cohort_ids: set,
                    variable_col: str, time_col: str,
                    comparator_map: pd.Series) -> pd.DataFrame:
    """Compute measurement density stats per variable.

    events must have: stay_id, <variable_col>, <time_col>
    time_col should contain hours_since_icu_admit or similar,
    already filtered to 0-6h window.
    """
    # Filter to cohort
    events = events[events["stay_id"].isin(cohort_ids)].copy()

    # Filter to 0-6h window
    if time_col in events.columns:
        events = events[(events[time_col] >= 0) & (events[time_col] < PRE_T0_HOURS)]

    n_total = len(cohort_ids)
    results = []

    for var_name, var_df in events.groupby(variable_col):
        counts_per_stay = var_df.groupby("stay_id").size()

        n_with_any = len(counts_per_stay)
        n_with_2plus = (counts_per_stay >= 2).sum()
        n_with_3plus = (counts_per_stay >= 3).sum()
        mean_count = counts_per_stay.mean() if len(counts_per_stay) > 0 else 0

        row = {
            "variable": var_name,
            "n_stays_total": n_total,
            "n_with_any": n_with_any,
            "coverage_pct": round(n_with_any / n_total * 100, 1) if n_total > 0 else 0,
            "n_with_2plus": int(n_with_2plus),
            "trend_capable_pct": round(n_with_2plus / n_total * 100, 1) if n_total > 0 else 0,
            "n_with_3plus": int(n_with_3plus),
            "mean_measurements": round(mean_count, 2),
            "missing_pct": round((1 - n_with_any / n_total) * 100, 1) if n_total > 0 else 100,
        }

        # HFpEF-specific density
        hfpef_stays = set(comparator_map[comparator_map == "hfpef"].index)
        hfpef_counts = counts_per_stay[counts_per_stay.index.isin(hfpef_stays)]
        n_hfpef = len(hfpef_stays)
        n_hfpef_any = len(hfpef_counts)
        n_hfpef_2plus = (hfpef_counts >= 2).sum() if len(hfpef_counts) > 0 else 0

        row["hfpef_n_total"] = n_hfpef
        row["hfpef_coverage_pct"] = round(n_hfpef_any / n_hfpef * 100, 1) if n_hfpef > 0 else 0
        row["hfpef_trend_capable_pct"] = round(n_hfpef_2plus / n_hfpef * 100, 1) if n_hfpef > 0 else 0

        results.append(row)

    return pd.DataFrame(results)


def main() -> None:
    cfg = load_config()
    staging_dir = resolve_path(cfg, "staging")
    tables_dir = resolve_path(cfg, "tables")
    reports_dir = resolve_path(cfg, "reports")
    for d in [tables_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("[30b] Measuring 0-6h information density...")

    # ── Load data ─────────────────────────────────────────────────────────
    cohort = pd.read_parquet(staging_dir / "25f_primary_cohort_candidate_v1_mimic.parquet")
    gate_passed = set(cohort[cohort["hemo_gate_passed"] == True]["stay_id"])
    comparator_map = cohort.set_index("stay_id")["comparator_group"]

    print(f"  Gate-passed stays: {len(gate_passed):,}")

    # Vitals
    vitals = pd.read_parquet(staging_dir / "10d_vitals_mimic.parquet")
    vitals_density = measure_density(
        vitals, gate_passed, variable_col="vital_name",
        time_col="hours_since_icu_admit", comparator_map=comparator_map
    )
    vitals_density["data_source"] = "vitals"
    print(f"  Vitals: {len(vitals_density)} variables measured")

    # Labs
    labs = pd.read_parquet(staging_dir / "10e_labs_mimic.parquet")
    labs_density = measure_density(
        labs, gate_passed, variable_col="lab_name",
        time_col="hours_since_icu_admit", comparator_map=comparator_map
    )
    labs_density["data_source"] = "labs"
    print(f"  Labs: {len(labs_density)} variables measured")

    # Combine
    density = pd.concat([vitals_density, labs_density], ignore_index=True)
    density = density.sort_values("coverage_pct", ascending=False)

    # ── Classify density tiers ────────────────────────────────────────────
    def classify_tier(row):
        if row["coverage_pct"] >= 80 and row["trend_capable_pct"] >= 50:
            return "high_completeness_core"
        elif row["coverage_pct"] >= 50:
            return "moderate_completeness_optional"
        else:
            return "low_completeness_holdout"
    density["density_tier"] = density.apply(classify_tier, axis=1)

    # ── Decision: trend vs snapshot ───────────────────────────────────────
    core_vars = density[density["density_tier"] == "high_completeness_core"]
    trend_feasible = len(core_vars[core_vars["trend_capable_pct"] >= 50])
    total_core = len(core_vars)

    if total_core > 0 and trend_feasible / total_core >= 0.5:
        phenotyping_recommendation = "TREND-BASED (≥50% core variables trend-capable)"
    else:
        phenotyping_recommendation = "SNAPSHOT-BASED (fallback to rule-based shock vs congestion)"

    # ── Report ────────────────────────────────────────────────────────────
    report_lines = [
        "=" * 72,
        "30b — 0-6h Information Density Assessment",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Gate-passed stays: {len(gate_passed):,}",
        f"Phenotyping recommendation: {phenotyping_recommendation}",
        "",
        "─" * 72,
        "DENSITY TABLE (sorted by coverage)",
        "─" * 72,
        density[[
            "variable", "data_source", "coverage_pct", "trend_capable_pct",
            "mean_measurements", "missing_pct", "density_tier",
            "hfpef_coverage_pct", "hfpef_trend_capable_pct"
        ]].to_string(index=False),
        "",
        "─" * 72,
        "TIER SUMMARY",
        "─" * 72,
        density["density_tier"].value_counts().to_string(),
        "",
        f"Core variables (high completeness): {total_core}",
        f"  Of which trend-capable (≥50% with ≥2 measurements): {trend_feasible}",
        f"  Phenotyping recommendation: {phenotyping_recommendation}",
    ]

    # ── Save ──────────────────────────────────────────────────────────────
    csv_path = tables_dir / "30b_density_0_6h.csv"
    density.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    report_path = reports_dir / "30b_density_0_6h_summary.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Saved: {report_path}")

    print(f"\n  Phenotyping recommendation: {phenotyping_recommendation}")
    print("[30b] Done.")


if __name__ == "__main__":
    main()
