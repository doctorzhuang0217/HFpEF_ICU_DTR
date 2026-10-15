#!/usr/bin/env python3
"""
35d_compare_treatment_patterns_6h_24h — Early treatment pattern comparison.

Purpose
-------
Compare early (0-6h pre-T0 and T0-24h post-T0) treatment patterns across
HFpEF and comparator arms: fluid volume, vasopressor exposure, diuretic
exposure, and respiratory support.

Protocol reference: Step 35d.

Inputs
------
- data/interim/35a_comparator_cohorts_v1_mimic.parquet
- data/interim/35b_nonhf_matched_v1_mimic.parquet
- data/staging/10f_meds_infusions_mimic.parquet

Outputs
-------
- tables/35d_treatment_pattern_compare.csv
- reports/35d_treatment_pattern_comparison.txt
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path


# Time windows (hours from ICU admission)
WINDOWS = {
    "pre_t0_0_6h": (0, 6),       # Pre-T0: ICU admission to T0
    "post_t0_0_6h": (6, 12),     # First 6h post-T0
    "post_t0_6_24h": (12, 30),   # 6-24h post-T0
    "post_t0_0_24h": (6, 30),    # Full 24h post-T0
}

# Treatment categories to summarize
MED_CLASSES = {
    "vasopressor": ["norepinephrine", "epinephrine", "vasopressin", "phenylephrine", "dopamine"],
    "inotrope": ["dobutamine", "milrinone"],
    "loop_diuretic": ["furosemide", "bumetanide", "torsemide"],
    "iv_fluid": ["crystalloid", "colloid", "other_iv_fluid"],
}


def summarize_treatments(
    meds: pd.DataFrame,
    stay_ids: set,
    window_start: float,
    window_end: float,
) -> dict:
    """Summarize treatment exposures for a set of stays in a time window."""
    n_total = len(stay_ids)
    if n_total == 0:
        return {}

    # Filter medications to window and stays
    sub = meds[
        meds["stay_id"].isin(stay_ids)
        & (meds["hours_since_icu_admit"] < window_end)
    ].copy()

    # For events with duration, check overlap with window
    if "duration_hours" in sub.columns:
        sub["end_hour"] = sub["hours_since_icu_admit"] + sub["duration_hours"].fillna(0)
    else:
        sub["end_hour"] = sub["hours_since_icu_admit"]

    # Keep events that overlap with window
    sub = sub[sub["end_hour"] > window_start]

    result = {}

    for med_class, med_names in MED_CLASSES.items():
        # Filter to this class
        if "med_class" in sub.columns:
            class_sub = sub[sub["med_class"] == med_class]
        elif "med_name" in sub.columns:
            class_sub = sub[sub["med_name"].isin(med_names)]
        else:
            continue

        exposed_stays = class_sub["stay_id"].nunique()
        result[f"{med_class}_exposed_n"] = exposed_stays
        result[f"{med_class}_exposed_pct"] = round(exposed_stays / n_total * 100, 1)

        # Total volume/amount for IV fluids
        if med_class == "iv_fluid" and "amount" in class_sub.columns:
            fluid_per_stay = class_sub.groupby("stay_id")["amount"].sum()
            # Convert to mL (assume mL if not specified)
            result[f"{med_class}_total_mL_median"] = round(fluid_per_stay.median(), 0) if len(fluid_per_stay) > 0 else 0
            result[f"{med_class}_total_mL_iqr_lo"] = round(fluid_per_stay.quantile(0.25), 0) if len(fluid_per_stay) > 0 else 0
            result[f"{med_class}_total_mL_iqr_hi"] = round(fluid_per_stay.quantile(0.75), 0) if len(fluid_per_stay) > 0 else 0

    return result


def main() -> None:
    cfg = load_config()
    staging_dir = resolve_path(cfg, "staging")
    interim_dir = resolve_path(cfg, "interim")
    tables_dir = resolve_path(cfg, "tables")
    reports_dir = resolve_path(cfg, "reports")
    for d in [tables_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("[35d] Comparing treatment patterns...")

    # ── Load cohorts ──────────────────────────────────────────────────────
    cohort = pd.read_parquet(interim_dir / "35a_comparator_cohorts_v1_mimic.parquet")

    arms = {
        "hfpef": cohort[cohort["analysis_arm"] == "hfpef"],
        "hfref_hfmref": cohort[cohort["analysis_arm"] == "hfref_hfmref"],
        "no_hf": cohort[cohort["analysis_arm"] == "no_hf"],
    }

    matched_path = interim_dir / "35b_nonhf_matched_v1_mimic.parquet"
    if matched_path.exists():
        arms["no_hf_matched"] = pd.read_parquet(matched_path)

    for name, df in arms.items():
        print(f"  {name}: {len(df):,}")

    # ── Load medications ──────────────────────────────────────────────────
    meds_path = staging_dir / "10f_meds_infusions_mimic.parquet"
    if not meds_path.exists():
        print("  ERROR: 10f_meds_infusions_mimic.parquet not found.")
        print("  Creating placeholder outputs.")
        pd.DataFrame().to_csv(tables_dir / "35d_treatment_pattern_compare.csv", index=False)
        with open(reports_dir / "35d_treatment_pattern_comparison.txt", "w") as f:
            f.write("35d — Treatment data not yet available (10f not found)\n")
        return

    # Only load stay_ids we need
    all_stay_ids = set()
    for df in arms.values():
        all_stay_ids.update(df["stay_id"].tolist())

    meds = pd.read_parquet(meds_path)
    meds = meds[meds["stay_id"].isin(all_stay_ids)].copy()
    print(f"  Medication records loaded: {len(meds):,}")

    # ── Compare across windows and arms ───────────────────────────────────
    all_rows = []
    report_lines = [
        "=" * 80,
        "35d — Early Treatment Pattern Comparison",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    for window_name, (w_start, w_end) in WINDOWS.items():
        report_lines.append(f"═══ Window: {window_name} ({w_start}–{w_end}h from ICU admit) ═══")
        report_lines.append("")

        for arm_name, arm_df in arms.items():
            stay_ids = set(arm_df["stay_id"].tolist())
            stats = summarize_treatments(meds, stay_ids, w_start, w_end)

            row = {
                "window": window_name,
                "arm": arm_name,
                "n_stays": len(stay_ids),
            }
            row.update(stats)
            all_rows.append(row)

        # Format comparison table for this window
        header_parts = [f"{'Metric':<35}"]
        for arm_name in arms:
            header_parts.append(f"{arm_name:<18}")
        report_lines.append("  " + " ".join(header_parts))
        report_lines.append("  " + "─" * (35 + 18 * len(arms)))

        for med_class in MED_CLASSES:
            val_parts = [f"{med_class} exposed (%)"[:35].ljust(35)]
            for arm_name in arms:
                arm_row = [r for r in all_rows if r["window"] == window_name and r["arm"] == arm_name]
                if arm_row:
                    pct = arm_row[0].get(f"{med_class}_exposed_pct", 0)
                    n = arm_row[0].get(f"{med_class}_exposed_n", 0)
                    val_parts.append(f"{pct:.1f}% ({n:,})".ljust(18))
                else:
                    val_parts.append("—".ljust(18))
            report_lines.append("  " + " ".join(val_parts))

            # Fluid volume for iv_fluid
            if med_class == "iv_fluid":
                vol_parts = [f"  total mL (median [IQR])"[:35].ljust(35)]
                for arm_name in arms:
                    arm_row = [r for r in all_rows if r["window"] == window_name and r["arm"] == arm_name]
                    if arm_row:
                        med = arm_row[0].get(f"{med_class}_total_mL_median", 0)
                        lo = arm_row[0].get(f"{med_class}_total_mL_iqr_lo", 0)
                        hi = arm_row[0].get(f"{med_class}_total_mL_iqr_hi", 0)
                        vol_parts.append(f"{med:.0f} [{lo:.0f}–{hi:.0f}]".ljust(18))
                    else:
                        vol_parts.append("—".ljust(18))
                report_lines.append("  " + " ".join(vol_parts))

        report_lines.append("")

    result_df = pd.DataFrame(all_rows)

    # ── Save ──────────────────────────────────────────────────────────────
    csv_path = tables_dir / "35d_treatment_pattern_compare.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    report_path = reports_dir / "35d_treatment_pattern_comparison.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Saved: {report_path}")

    print("\n[35d] Done.")


if __name__ == "__main__":
    main()
