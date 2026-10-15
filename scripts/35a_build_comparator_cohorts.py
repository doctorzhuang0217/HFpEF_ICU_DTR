#!/usr/bin/env python3
"""
35a_build_comparator_cohorts — Construct comparator cohorts for Phase 0.5b.

Purpose
-------
Build HFrEF, HFmrEF, HF-EF-unknown, and non-HF comparator cohorts using
the identical landmark (T0 = 6h), eligibility gate, and event-handling rules
as the HFpEF primary cohort.  All comparator arms are subsets of the same
25f frozen cohort that passed the hemodynamic gate.

Protocol reference: Step 35a (Phase 0.5b descriptive comparator module).

Inputs
------
- data/staging/25f_primary_cohort_candidate_v1_mimic.parquet

Outputs
-------
- data/interim/35a_comparator_cohorts_v1_mimic.parquet
- tables/35a_comparator_cohort_summary.csv
- reports/35a_comparator_cohort_summary.txt
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path


# Comparator group definitions (must match 20a labels)
COMPARATOR_GROUPS = ["hfpef", "hfref_hfmref", "hf_ef_unknown", "no_hf"]

# Columns to carry forward into comparator cohort
ID_COLS = ["stay_id", "subject_id", "hadm_id"]
COHORT_COLS = [
    "comparator_group", "hf_subtype",
    "is_hfpef", "is_hfref", "is_hfmref", "is_hf_ef_unknown", "is_no_hf",
    "ef_value", "ef_source",
]
ADMISSION_COLS = [
    "age", "gender", "intime", "outtime", "icu_los_hours",
    "hospital_mortality", "admission_type",
]
GATE_COLS = [
    "hemo_gate_passed", "is_index_stay", "include_in_phase05",
    "t0_6h_eligible",
]
CLINICAL_COLS = [
    "reason_for_icu", "dominant_syndrome",
    "pre_t0_min_map", "pre_t0_mean_map", "pre_t0_min_spo2",
    "pre_t0_max_rr", "pre_t0_max_lactate", "fluid_balance_L",
]
TIME_COLS = [
    "time_zero_6h", "time_zero_12h",
    "t0_plus_24h", "t0_plus_48h", "t0_plus_72h",
]


def main() -> None:
    cfg = load_config()
    staging_dir = resolve_path(cfg, "staging")
    interim_dir = resolve_path(cfg, "interim")
    tables_dir = resolve_path(cfg, "tables")
    reports_dir = resolve_path(cfg, "reports")
    for d in [interim_dir, tables_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("[35a] Building comparator cohorts...")

    # ── Load frozen cohort ────────────────────────────────────────────────
    cohort = pd.read_parquet(
        staging_dir / "25f_primary_cohort_candidate_v1_mimic.parquet"
    )

    # Apply identical filters as primary HFpEF cohort:
    # 1. Index stays only
    # 2. Hemodynamic gate passed
    assert "is_index_stay" in cohort.columns, "Missing is_index_stay"
    assert "hemo_gate_passed" in cohort.columns, "Missing hemo_gate_passed"

    gated = cohort[
        (cohort["is_index_stay"] == True) & (cohort["hemo_gate_passed"] == True)
    ].copy()
    print(f"  Gate-passed index stays: {len(gated):,}")

    # ── Select columns ───────────────────────────────────────────────────
    all_desired = ID_COLS + COHORT_COLS + ADMISSION_COLS + GATE_COLS + CLINICAL_COLS + TIME_COLS
    available = [c for c in all_desired if c in gated.columns]
    missing = [c for c in all_desired if c not in gated.columns]
    if missing:
        print(f"  Note: {len(missing)} desired columns not in source: {missing}")

    comparator_df = gated[available].copy()

    # ── Assign analysis arm label ─────────────────────────────────────────
    # For 35a, the arm is simply the comparator_group from 20a
    comparator_df["analysis_arm"] = comparator_df["comparator_group"].copy()

    # ── Summary statistics ────────────────────────────────────────────────
    summary_rows = []
    report_lines = [
        "=" * 72,
        "35a — Comparator Cohort Summary",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Database: MIMIC-IV",
        f"Total gate-passed index stays: {len(comparator_df):,}",
        "",
    ]

    for arm in COMPARATOR_GROUPS:
        sub = comparator_df[comparator_df["analysis_arm"] == arm]
        n = len(sub)
        if n == 0:
            continue

        row = {"arm": arm, "n": n}

        # Demographics
        if "age" in sub.columns:
            row["age_median"] = round(sub["age"].median(), 1)
            row["age_iqr_lo"] = round(sub["age"].quantile(0.25), 1)
            row["age_iqr_hi"] = round(sub["age"].quantile(0.75), 1)
        if "gender" in sub.columns:
            row["pct_female"] = round((sub["gender"] == "F").mean() * 100, 1)

        # Outcomes
        if "hospital_mortality" in sub.columns:
            mort = sub["hospital_mortality"].fillna(0).astype(int)
            row["hospital_mortality_n"] = int(mort.sum())
            row["hospital_mortality_pct"] = round(mort.mean() * 100, 1)
        if "icu_los_hours" in sub.columns:
            row["icu_los_median_h"] = round(sub["icu_los_hours"].median(), 1)

        # Reason for ICU
        if "reason_for_icu" in sub.columns:
            for reason in ["sepsis", "post_op", "respiratory", "aki", "other"]:
                cnt = (sub["reason_for_icu"] == reason).sum()
                row[f"icu_reason_{reason}_n"] = int(cnt)
                row[f"icu_reason_{reason}_pct"] = round(cnt / n * 100, 1)

        # Dominant syndrome
        if "dominant_syndrome" in sub.columns:
            for syn in ["shock", "congestion", "mixed", "neither"]:
                cnt = (sub["dominant_syndrome"] == syn).sum()
                row[f"syndrome_{syn}_n"] = int(cnt)
                row[f"syndrome_{syn}_pct"] = round(cnt / n * 100, 1)

        # EF
        if "ef_value" in sub.columns:
            ef_valid = sub["ef_value"].dropna()
            if len(ef_valid) > 0:
                row["ef_median"] = round(ef_valid.median(), 1)
                row["ef_available_n"] = len(ef_valid)
                row["ef_available_pct"] = round(len(ef_valid) / n * 100, 1)

        summary_rows.append(row)

        # Report text
        report_lines.append(f"─── {arm} (N={n:,}) ───")
        if "age_median" in row:
            report_lines.append(
                f"  Age: median {row['age_median']} "
                f"[IQR {row.get('age_iqr_lo', '?')}–{row.get('age_iqr_hi', '?')}]"
            )
        if "pct_female" in row:
            report_lines.append(f"  Female: {row['pct_female']:.1f}%")
        if "hospital_mortality_pct" in row:
            report_lines.append(
                f"  Hospital mortality: {row['hospital_mortality_n']}/{n} "
                f"({row['hospital_mortality_pct']:.1f}%)"
            )
        if "icu_los_median_h" in row:
            report_lines.append(f"  ICU LOS median: {row['icu_los_median_h']:.1f}h")
        if "reason_for_icu" in sub.columns:
            report_lines.append("  Reason for ICU:")
            for reason in ["sepsis", "post_op", "respiratory", "aki", "other"]:
                rn = row.get(f"icu_reason_{reason}_n", 0)
                rp = row.get(f"icu_reason_{reason}_pct", 0)
                report_lines.append(f"    {reason}: {rn:,} ({rp:.1f}%)")
        if "dominant_syndrome" in sub.columns:
            report_lines.append("  Dominant syndrome:")
            for syn in ["shock", "congestion", "mixed", "neither"]:
                sn = row.get(f"syndrome_{syn}_n", 0)
                sp = row.get(f"syndrome_{syn}_pct", 0)
                report_lines.append(f"    {syn}: {sn:,} ({sp:.1f}%)")
        if "ef_median" in row:
            report_lines.append(
                f"  EF: median {row['ef_median']}% "
                f"(available in {row['ef_available_n']:,}/{n:,} = "
                f"{row['ef_available_pct']:.1f}%)"
            )
        report_lines.append("")

    summary_df = pd.DataFrame(summary_rows)

    # ── Save ──────────────────────────────────────────────────────────────
    pq_path = interim_dir / "35a_comparator_cohorts_v1_mimic.parquet"
    comparator_df.to_parquet(pq_path, index=False)
    print(f"  Saved: {pq_path}")

    csv_path = tables_dir / "35a_comparator_cohort_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    report_path = reports_dir / "35a_comparator_cohort_summary.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Saved: {report_path}")

    print("\n[35a] Done.")


if __name__ == "__main__":
    main()
