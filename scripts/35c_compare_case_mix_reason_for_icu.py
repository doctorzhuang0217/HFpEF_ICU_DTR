#!/usr/bin/env python3
"""
35c_compare_case_mix_reason_for_icu — Case-mix comparison across arms.

Purpose
-------
Compare baseline case-mix distributions and reason-for-ICU strata between
HFpEF and each comparator arm (HFrEF/HFmrEF, non-HF, syndrome-matched
non-HF).  Produces standardized mean differences (SMD) for continuous
variables and proportion comparisons for categorical variables.

Protocol reference: Step 35c.

Inputs
------
- data/interim/35a_comparator_cohorts_v1_mimic.parquet
- data/interim/35b_nonhf_matched_v1_mimic.parquet

Outputs
-------
- tables/35c_case_mix_compare.csv
- reports/35c_case_mix_comparison.txt
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path


# Variables to compare
CONTINUOUS_VARS = [
    ("age", "Age (years)"),
    ("ef_value", "Ejection fraction (%)"),
    ("icu_los_hours", "ICU LOS (hours)"),
    ("pre_t0_min_map", "Pre-T0 min MAP (mmHg)"),
    ("pre_t0_mean_map", "Pre-T0 mean MAP (mmHg)"),
    ("pre_t0_min_spo2", "Pre-T0 min SpO2 (%)"),
    ("pre_t0_max_rr", "Pre-T0 max RR (/min)"),
    ("pre_t0_max_lactate", "Pre-T0 max lactate (mmol/L)"),
    ("fluid_balance_L", "Pre-T0 fluid balance (L)"),
]

BINARY_VARS = [
    ("gender", "F", "Female (%)"),
    ("hospital_mortality", 1, "Hospital mortality (%)"),
]

CATEGORICAL_VARS = [
    ("reason_for_icu", "Reason for ICU"),
    ("dominant_syndrome", "Dominant syndrome"),
]


def smd(x1: pd.Series, x2: pd.Series) -> float:
    """Standardized mean difference (continuous)."""
    m1, m2 = x1.mean(), x2.mean()
    s1, s2 = x1.std(), x2.std()
    pooled = np.sqrt((s1**2 + s2**2) / 2)
    if pooled == 0 or np.isnan(pooled):
        return 0.0
    return (m1 - m2) / pooled


def smd_binary(p1: float, p2: float) -> float:
    """SMD for binary/proportion variables."""
    pooled = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    if pooled == 0 or np.isnan(pooled):
        return 0.0
    return (p1 - p2) / pooled


def proportion_equal(series: pd.Series, value: object) -> float:
    """Compute proportion equal to `value` among non-missing observations."""
    valid = series.dropna()
    if len(valid) == 0:
        return 0.0
    return (valid == value).mean()


def compare_pair(
    hfpef: pd.DataFrame, comp: pd.DataFrame, comp_name: str
) -> list[dict]:
    """Compare HFpEF vs one comparator arm."""
    rows = []
    n1, n2 = len(hfpef), len(comp)

    # Continuous variables
    for col, label in CONTINUOUS_VARS:
        if col not in hfpef.columns or col not in comp.columns:
            print(f"    [skip] {label}: column '{col}' not in both cohorts")
            continue
        v1 = hfpef[col].dropna()
        v2 = comp[col].dropna()
        rows.append({
            "comparator": comp_name,
            "variable": label,
            "var_type": "continuous",
            "hfpef_n": len(v1),
            "hfpef_value": f"{v1.median():.1f} [{v1.quantile(0.25):.1f}–{v1.quantile(0.75):.1f}]" if len(v1) > 0 else "—",
            "comp_n": len(v2),
            "comp_value": f"{v2.median():.1f} [{v2.quantile(0.25):.1f}–{v2.quantile(0.75):.1f}]" if len(v2) > 0 else "—",
            "smd": round(smd(v1, v2), 3) if len(v1) > 0 and len(v2) > 0 else None,
        })

    # Binary variables
    for col, pos_val, label in BINARY_VARS:
        if col not in hfpef.columns or col not in comp.columns:
            print(f"    [skip] {label}: column '{col}' not in both cohorts")
            continue
        p1 = proportion_equal(hfpef[col], pos_val)
        p2 = proportion_equal(comp[col], pos_val)
        rows.append({
            "comparator": comp_name,
            "variable": label,
            "var_type": "binary",
            "hfpef_n": int(hfpef[col].notna().sum()),
            "hfpef_value": f"{p1 * 100:.1f}%",
            "comp_n": int(comp[col].notna().sum()),
            "comp_value": f"{p2 * 100:.1f}%",
            "smd": round(smd_binary(p1, p2), 3),
        })

    # Categorical variables
    for col, label in CATEGORICAL_VARS:
        if col not in hfpef.columns or col not in comp.columns:
            print(f"    [skip] {label}: column '{col}' not in both cohorts")
            continue
        cats = sorted(set(hfpef[col].dropna().unique()) | set(comp[col].dropna().unique()))
        for cat in cats:
            p1 = proportion_equal(hfpef[col], cat)
            p2 = proportion_equal(comp[col], cat)
            rows.append({
                "comparator": comp_name,
                "variable": f"{label}: {cat}",
                "var_type": "categorical",
                "hfpef_n": int(hfpef[col].notna().sum()),
                "hfpef_value": f"{p1 * 100:.1f}%",
                "comp_n": int(comp[col].notna().sum()),
                "comp_value": f"{p2 * 100:.1f}%",
                "smd": round(smd_binary(p1, p2), 3),
            })

    return rows


def main() -> None:
    cfg = load_config()
    interim_dir = resolve_path(cfg, "interim")
    tables_dir = resolve_path(cfg, "tables")
    reports_dir = resolve_path(cfg, "reports")
    for d in [tables_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("[35c] Comparing case-mix and reason-for-ICU...")

    # ── Load cohorts ──────────────────────────────────────────────────────
    cohort = pd.read_parquet(interim_dir / "35a_comparator_cohorts_v1_mimic.parquet")
    hfpef = cohort[cohort["analysis_arm"] == "hfpef"]

    comparators = {
        "hfref_hfmref": cohort[cohort["analysis_arm"] == "hfref_hfmref"],
        "no_hf": cohort[cohort["analysis_arm"] == "no_hf"],
    }

    # Add syndrome-matched non-HF if available
    matched_path = interim_dir / "35b_nonhf_matched_v1_mimic.parquet"
    if matched_path.exists():
        matched = pd.read_parquet(matched_path)
        comparators["no_hf_matched"] = matched

    print(f"  HFpEF: {len(hfpef):,}")
    for name, df in comparators.items():
        print(f"  {name}: {len(df):,}")

    # Log missing variables explicitly for auditability
    all_check_vars = (
        [(col, label) for col, label in CONTINUOUS_VARS]
        + [(col, label) for col, _, label in BINARY_VARS]
        + [(col, label) for col, label in CATEGORICAL_VARS]
    )
    for col, label in all_check_vars:
        if col not in cohort.columns:
            print(f"  [skip] {label}: column '{col}' not in cohort")

    # ── Compare each pair ─────────────────────────────────────────────────
    all_rows = []
    for comp_name, comp_df in comparators.items():
        all_rows.extend(compare_pair(hfpef, comp_df, comp_name))

    result_df = pd.DataFrame(all_rows)

    # ── Report ────────────────────────────────────────────────────────────
    report_lines = [
        "=" * 80,
        "35c — Case-Mix & Reason-for-ICU Comparison",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"HFpEF N: {len(hfpef):,}",
        "",
    ]

    for comp_name in comparators:
        comp_rows = result_df[result_df["comparator"] == comp_name]
        comp_n = comparators[comp_name].shape[0]
        report_lines.append(f"─── HFpEF vs {comp_name} (N={comp_n:,}) ───")
        report_lines.append(
            f"  {'Variable':<40} {'HFpEF':<25} {comp_name:<25} {'SMD':<8}"
        )
        report_lines.append("  " + "─" * 96)
        for _, r in comp_rows.iterrows():
            smd_str = f"{r['smd']:.3f}" if r["smd"] is not None else "—"
            smd_flag = " ***" if r["smd"] is not None and abs(r["smd"]) > 0.2 else (
                " **" if r["smd"] is not None and abs(r["smd"]) > 0.1 else ""
            )
            report_lines.append(
                f"  {r['variable']:<40} {r['hfpef_value']:<25} "
                f"{r['comp_value']:<25} {smd_str}{smd_flag}"
            )
        report_lines.append("")

    report_lines.append("SMD flags: *** > 0.2 (large imbalance), ** > 0.1 (moderate)")

    # ── Save ──────────────────────────────────────────────────────────────
    csv_path = tables_dir / "35c_case_mix_compare.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    report_path = reports_dir / "35c_case_mix_comparison.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Saved: {report_path}")

    print("\n[35c] Done.")


if __name__ == "__main__":
    main()
