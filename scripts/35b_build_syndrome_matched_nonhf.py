#!/usr/bin/env python3
"""
35b_build_syndrome_matched_nonhf — Syndrome-matched non-HF comparator.

Purpose
-------
Restrict the non-HF comparator arm to patients whose reason-for-ICU and
dominant-syndrome distributions match HFpEF.  Uses exact stratified
sampling (not propensity matching) to preserve interpretability.

Matching strata: reason_for_icu × dominant_syndrome.
Within each stratum, randomly sample non-HF stays at a target ratio
(default 3:1 non-HF:HFpEF, capped at available stays).

Protocol reference: Step 35b.

Inputs
------
- data/interim/35a_comparator_cohorts_v1_mimic.parquet

Outputs
-------
- data/interim/35b_nonhf_matched_v1_mimic.parquet
- tables/35b_nonhf_matching_summary.csv
- reports/35b_nonhf_matching_summary.txt
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path

MATCH_RATIO = 3  # target non-HF:HFpEF ratio per stratum
RANDOM_SEED = 42


def main() -> None:
    cfg = load_config()
    interim_dir = resolve_path(cfg, "interim")
    tables_dir = resolve_path(cfg, "tables")
    reports_dir = resolve_path(cfg, "reports")
    for d in [tables_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("[35b] Building syndrome-matched non-HF comparator...")

    # ── Load comparator cohorts ───────────────────────────────────────────
    cohort = pd.read_parquet(interim_dir / "35a_comparator_cohorts_v1_mimic.parquet")

    hfpef = cohort[cohort["analysis_arm"] == "hfpef"].copy()
    nonhf = cohort[cohort["analysis_arm"] == "no_hf"].copy()
    print(f"  HFpEF: {len(hfpef):,}")
    print(f"  Non-HF pool: {len(nonhf):,}")

    # ── Define matching strata ────────────────────────────────────────────
    strata_cols = []
    if "reason_for_icu" in hfpef.columns:
        strata_cols.append("reason_for_icu")
    if "dominant_syndrome" in hfpef.columns:
        strata_cols.append("dominant_syndrome")

    summary_rows: list[dict] = []

    if not strata_cols:
        print("  WARNING: No stratification columns available. Using all non-HF.")
        summary_rows = []
        matched = nonhf.copy()
    else:
        # Fill NaN strata with "unknown" for matching
        for col in strata_cols:
            hfpef[col] = hfpef[col].fillna("unknown")
            nonhf[col] = nonhf[col].fillna("unknown")

        # Count HFpEF per stratum
        hfpef_counts = hfpef.groupby(strata_cols).size().reset_index(name="n_hfpef")

        # Sample non-HF within each stratum
        rng = np.random.default_rng(RANDOM_SEED)
        matched_parts = []

        for _, row in hfpef_counts.iterrows():
            # Build stratum filter
            mask = pd.Series(True, index=nonhf.index)
            stratum_label_parts = []
            for col in strata_cols:
                mask = mask & (nonhf[col] == row[col])
                stratum_label_parts.append(f"{col}={row[col]}")
            stratum_label = " × ".join(stratum_label_parts)

            pool = nonhf[mask]
            n_hfpef_stratum = int(row["n_hfpef"])
            n_target = n_hfpef_stratum * MATCH_RATIO
            n_available = len(pool)
            n_sampled = min(n_target, n_available)

            if n_sampled > 0:
                idx = rng.choice(pool.index, size=n_sampled, replace=False)
                matched_parts.append(nonhf.loc[idx])

            summary_rows.append({
                "stratum": stratum_label,
                "n_hfpef": n_hfpef_stratum,
                "n_nonhf_available": n_available,
                "n_target": n_target,
                "n_sampled": n_sampled,
                "ratio_achieved": round(n_sampled / n_hfpef_stratum, 2) if n_hfpef_stratum > 0 else 0,
            })

        if matched_parts:
            matched = pd.concat(matched_parts, ignore_index=True)
        else:
            matched = nonhf.iloc[:0].copy()

    # Mark as syndrome-matched
    matched["match_type"] = "syndrome_matched"
    print(f"  Matched non-HF: {len(matched):,}")

    # ── Summary ───────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()

    report_lines = [
        "=" * 72,
        "35b — Syndrome-Matched Non-HF Comparator",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"HFpEF N: {len(hfpef):,}",
        f"Non-HF pool: {len(nonhf):,}",
        f"Target ratio: {MATCH_RATIO}:1",
        f"Matched non-HF: {len(matched):,}",
        f"Achieved ratio: {len(matched) / len(hfpef):.2f}:1" if len(hfpef) > 0 else "",
        "",
        "─── Per-Stratum Matching ───",
    ]

    for _, sr in summary_df.iterrows():
        report_lines.append(
            f"  {sr['stratum']}: HFpEF={sr['n_hfpef']}, "
            f"pool={sr['n_nonhf_available']}, "
            f"sampled={sr['n_sampled']} (ratio={sr['ratio_achieved']}:1)"
        )

    # Check for strata with poor matching
    if len(summary_df) > 0:
        poor = summary_df[summary_df["ratio_achieved"] < 1.0]
        if len(poor) > 0:
            report_lines.append("")
            report_lines.append("─── WARNING: Under-matched strata (ratio < 1:1) ───")
            for _, sr in poor.iterrows():
                report_lines.append(
                    f"  {sr['stratum']}: ratio={sr['ratio_achieved']}:1 "
                    f"(pool={sr['n_nonhf_available']})"
                )

    # ── Save ──────────────────────────────────────────────────────────────
    pq_path = interim_dir / "35b_nonhf_matched_v1_mimic.parquet"
    matched.to_parquet(pq_path, index=False)
    print(f"  Saved: {pq_path}")

    csv_path = tables_dir / "35b_nonhf_matching_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    report_path = reports_dir / "35b_nonhf_matching_summary.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Saved: {report_path}")

    print("\n[35b] Done.")


if __name__ == "__main__":
    main()
