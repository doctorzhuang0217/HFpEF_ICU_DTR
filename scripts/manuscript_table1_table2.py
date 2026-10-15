#!/usr/bin/env python3
"""Generate manuscript Table 1 (baseline + 28-day mortality) and Table 2
(trial-emulation overview) CSVs.

Table 1: Extends 35a_comparator_cohort_summary with 28-day mortality.
Table 2: trial-emulation protocol overview (no data dependency).

Outputs:
  tables/manuscript_table1_baseline.csv
  tables/manuscript_table2_trial_emulation.csv
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build_table1():
    """Merge 28-day mortality from 45d outcomes into 35a baseline summary."""

    # ── Load baseline summary ────────────────────────────────────────────
    t1 = pd.read_csv(ROOT / "tables" / "35a_comparator_cohort_summary.csv")
    print(f"[Table 1] Loaded baseline summary: {len(t1)} arms")

    # ── Load outcomes (stay-level) ───────────────────────────────────────
    outcomes = pd.read_parquet(
        ROOT / "data" / "interim" / "45d_outcomes_absorb_v1_mimic.parquet",
        columns=["stay_id", "mortality_28d"],
    )
    print(f"[Table 1] Loaded outcomes: {len(outcomes):,} stays")

    # ── Load cohort with arm assignment ──────────────────────────────────
    cohort_path = ROOT / "data" / "staging" / "35a_comparator_cohorts_mimic.parquet"
    if not cohort_path.exists():
        # Fallback: reconstruct arm from 25f primary cohort
        cohort_path = ROOT / "data" / "staging" / "25f_primary_cohort_candidate_v1_mimic.parquet"
        cohort = pd.read_parquet(cohort_path, columns=["stay_id", "comparator_group"])
        cohort = cohort.rename(columns={"comparator_group": "arm"})
        print(f"[Table 1] Using 25f cohort fallback: {len(cohort):,} stays")
    else:
        cohort = pd.read_parquet(cohort_path)
        # Determine arm column name
        if "analysis_arm" in cohort.columns:
            cohort = cohort.rename(columns={"analysis_arm": "arm"})
        elif "comparator_group" in cohort.columns:
            cohort = cohort.rename(columns={"comparator_group": "arm"})
        cohort = cohort[["stay_id", "arm"]]
        print(f"[Table 1] Using 35a cohort: {len(cohort):,} stays")

    # ── Merge and compute 28-day mortality per arm ───────────────────────
    merged = cohort.merge(outcomes, on="stay_id", how="inner")
    print(f"[Table 1] Merged: {len(merged):,} stays")

    mort28_by_arm = (
        merged.groupby("arm")["mortality_28d"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "mortality_28d_n", "count": "mortality_28d_total"})
        .reset_index()
    )
    mort28_by_arm["mortality_28d_pct"] = (
        mort28_by_arm["mortality_28d_n"] / mort28_by_arm["mortality_28d_total"] * 100
    ).round(1)

    print("\n[Table 1] 28-day mortality by arm:")
    for _, row in mort28_by_arm.iterrows():
        print(f"  {row['arm']}: {int(row['mortality_28d_n'])}/{int(row['mortality_28d_total'])} "
              f"({row['mortality_28d_pct']}%)")

    # ── Merge into baseline table ────────────────────────────────────────
    t1 = t1.merge(
        mort28_by_arm[["arm", "mortality_28d_n", "mortality_28d_pct"]],
        on="arm", how="left",
    )

    # Reorder: put mortality columns together
    # Insert 28-day mortality right after hospital mortality
    cols = list(t1.columns)
    hosp_idx = cols.index("hospital_mortality_pct")
    new_cols = (
        cols[: hosp_idx + 1]
        + ["mortality_28d_n", "mortality_28d_pct"]
        + [c for c in cols[hosp_idx + 1 :] if c not in ("mortality_28d_n", "mortality_28d_pct")]
    )
    t1 = t1[new_cols]

    out_path = ROOT / "tables" / "manuscript_table1_baseline.csv"
    t1.to_csv(out_path, index=False)
    print(f"\n[Table 1] Saved: {out_path}")
    return t1


def build_table2():
    """Build the trial-emulation overview table (no data dependency)."""

    rows = [
        ("Study design", "Target trial emulation using observational ICU data"),
        ("Eligibility", "Adults admitted to ICU with >= 6 h stay; "
         "HF phenotype ascertained from diagnosis codes + echocardiographic EF"),
        ("HF phenotype definition",
         "HFpEF: HF diagnosis + EF >= 50%; "
         "HFrEF: EF < 40%; HFmrEF: EF 40-49%; "
         "No HF: no HF diagnosis codes"),
        ("Time zero (T0)", "ICU admission + 6 hours (sensitivity: +12 h)"),
        ("Decision window", "24 hours post-T0, reassessed every 2 hours (12 decision points)"),
        ("Action space", "3-level fluid x 3-level vasopressor x 2-level diuretic "
         "(18 theoretical cells, 13 after merging sparse cells)"),
        ("Evaluated strategy",
         "Phenotype-aware, guideline-informed dynamic treatment rule: "
         "hypoperfusion-dominant -> early fluid + vasopressor; "
         "congestion-dominant -> avoid unnecessary fluid / decongestion; "
         "mixed/evolving -> rebalance every 2 h"),
        ("Comparator", "Observed clinician-selected haemodynamic management"),
        ("Primary outcome", "28-day all-cause mortality"),
        ("Primary estimator", "G-Transformer with DM rollout evaluation"),
        ("Classical anchor", "MSM/IPTW with WIS evaluation"),
        ("Concordance models",
         "G-Net, ACTIN, CT-Legacy, Mamba-CDSP "
         "(6/6 directionally concordant)"),
        ("External validation", "eICU-CRD (n = 83,765 full; n = 2,680 Tier A)"),
        ("Sensitivity analyses",
         "9 pre-specified: extreme dose quantile, sequential deconfounding, "
         "positivity grid, negative control, E-value/GOC, monitoring intensity, "
         "restricted re-treatment, extended horizon, informative missingness"),
        ("Consistency checks",
         "10 cross-database checks "
         "(5 pre-registered [C1-C5], 5 additional descriptive [C6-C10]); "
         "8/10 supportive"),
    ]

    t2 = pd.DataFrame(rows, columns=["Element", "Description"])
    out_path = ROOT / "tables" / "manuscript_table2_trial_emulation.csv"
    t2.to_csv(out_path, index=False)
    print(f"[Table 2] Saved: {out_path}")
    return t2


if __name__ == "__main__":
    print("=" * 70)
    print("Manuscript Table 1 & Table 2 Generation")
    print("=" * 70)
    build_table1()
    print()
    build_table2()
    print("\nDone.")
