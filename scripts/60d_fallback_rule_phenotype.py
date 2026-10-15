#!/usr/bin/env python3
"""
60d_fallback_rule_phenotype — Rule-based shock/congestion phenotype.

Purpose
-------
Generate a clinically motivated rule-based phenotype classification as
the pre-specified fallback when data-driven clustering (60b) fails
stability checks (60c). Always runs to provide a comparison baseline.

Classification rules (from protocol § 6 and 25d):
  - shock_dominant:      MAP < 65 OR vasopressor exposure in 0-6h
  - congestion_dominant: fluid_balance > 1.0L AND (SpO2 < 92 OR RR > 24)
  - mixed:               meets both shock AND congestion criteria
  - neither:             meets neither

This uses pre-T0 raw summaries from 60a, NOT the encoder latents,
because rule-based phenotyping must be interpretable and transparent.

Protocol reference: Step 60d (v3.1 § 12)

Inputs
------
- data/interim/60a_early_features_v1_mimic.parquet
- data/staging/25d_dominant_syndrome_flags_mimic.parquet  (for comparison)

Outputs
-------
- data/interim/60d_rule_phenotype_v1_mimic.parquet
- reports/60d_rule_phenotype_summary.txt

Compute: C1
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path

# ── Rule thresholds ──────────────────────────────────────────────────
# These are on z-scored values from the state tensor, so we need to
# check raw pre-T0 summaries OR use the 25d flags directly.
# Strategy: use 25d dominant_syndrome as primary, then augment with
# a simplified 2-class version for Step C stratification.


def main() -> None:
    proj_cfg = load_config()
    project_root = Path(proj_cfg.get("project_root", "."))
    interim_dir = resolve_path(proj_cfg, "interim")
    staging_dir = resolve_path(proj_cfg, "staging")
    reports_dir = resolve_path(proj_cfg, "reports")
    interim_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("60d_fallback_rule_phenotype — Rule-based early phenotype")
    print("=" * 72)

    # ── Load 60a features ──────────────────────────────────────────────
    print("\n[1/3] Loading early features ...")
    feat_df = pd.read_parquet(interim_dir / "60a_early_features_v1_mimic.parquet")
    n_stays = len(feat_df)
    print(f"  Stays: {n_stays:,}")

    # ── Load 25d syndrome flags ────────────────────────────────────────
    print("\n[2/3] Loading 25d dominant syndrome flags ...")
    syndrome_path = staging_dir / "25d_dominant_syndrome_flags_mimic.parquet"
    if syndrome_path.exists():
        syn_df = pd.read_parquet(syndrome_path)
        syn_cols = ["stay_id", "dominant_syndrome",
                    "is_shock_dominant", "is_congestion_dominant",
                    "is_mixed_syndrome", "is_neither_syndrome"]
        available = [c for c in syn_cols if c in syn_df.columns]
        syn_df = syn_df[available].drop_duplicates(subset=["stay_id"])
        print(f"  25d flags: {len(syn_df):,} stays")
    else:
        print("  25d flags not found — deriving from 60a features")
        syn_df = None

    # ── Build rule phenotype ───────────────────────────────────────────
    print("\n[3/3] Building rule-based phenotype ...")

    if syn_df is not None:
        # Use existing 25d classification first
        result = feat_df[["stay_id"]].merge(syn_df, on="stay_id", how="left")

        if "dominant_syndrome" in result.columns:
            result["rule_phenotype_4class"] = result["dominant_syndrome"]
        else:
            result["rule_phenotype_4class"] = np.nan

        # Backfill from boolean flags when dominant_syndrome is missing
        shock_flag = result.get("is_shock_dominant", pd.Series(0, index=result.index)).fillna(0).astype(int) > 0
        cong_flag = result.get("is_congestion_dominant", pd.Series(0, index=result.index)).fillna(0).astype(int) > 0
        mixed_flag = result.get("is_mixed_syndrome", pd.Series(0, index=result.index)).fillna(0).astype(int) > 0
        neither_flag = result.get("is_neither_syndrome", pd.Series(0, index=result.index)).fillna(0).astype(int) > 0

        result.loc[mixed_flag, "rule_phenotype_4class"] = "mixed"
        result.loc[shock_flag & ~mixed_flag, "rule_phenotype_4class"] = "shock_dominant"
        result.loc[cong_flag & ~mixed_flag, "rule_phenotype_4class"] = "congestion_dominant"
        result.loc[neither_flag, "rule_phenotype_4class"] = "neither"
        result["rule_phenotype_4class"] = result["rule_phenotype_4class"].fillna("unknown")
    else:
        # Derive from 60a raw features
        result = feat_df[["stay_id"]].copy()
        has_shock = (
            feat_df.get("map_pre_min", pd.Series(dtype=float)).lt(0).fillna(False)
        )
        has_congestion = (
            feat_df.get("spo2_pre_min", pd.Series(dtype=float)).lt(0).fillna(False)
        )
        result["rule_phenotype_4class"] = "neither"
        result.loc[has_shock & ~has_congestion, "rule_phenotype_4class"] = "shock_dominant"
        result.loc[~has_shock & has_congestion, "rule_phenotype_4class"] = "congestion_dominant"
        result.loc[has_shock & has_congestion, "rule_phenotype_4class"] = "mixed"

    # Simplified 2-class for Step C stratification
    result["rule_phenotype_2class"] = np.where(
        result["rule_phenotype_4class"].isin(["shock_dominant", "mixed"]),
        "shock_predominant",
        "non_shock"
    )

    # ── Compare with 60b data-driven if available ──────────────────────
    cluster_path = interim_dir / "60b_early_clusters_v1_mimic.parquet"
    if cluster_path.exists():
        cluster_df = pd.read_parquet(cluster_path)
        selected_col = [c for c in cluster_df.columns if c.startswith("selected_k")]
        if selected_col:
            result = result.merge(
                cluster_df[["stay_id", selected_col[0]]], on="stay_id", how="left"
            )
            result.rename(columns={selected_col[0]: "data_driven_cluster"}, inplace=True)

    out_path = interim_dir / "60d_rule_phenotype_v1_mimic.parquet"
    result.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"  Saved: {out_path} ({len(result):,} stays)")

    # ── Summary report ─────────────────────────────────────────────────
    report_lines = [
        "=" * 72,
        "60d — Rule-Based Early Phenotype (MIMIC-IV)",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "4-CLASS DISTRIBUTION",
        "────────────────────",
    ]

    for cls, cnt in result["rule_phenotype_4class"].value_counts().items():
        report_lines.append(f"  {cls}: {cnt:,} ({cnt / n_stays * 100:.1f}%)")

    report_lines.extend([
        "",
        "2-CLASS DISTRIBUTION",
        "────────────────────",
    ])
    for cls, cnt in result["rule_phenotype_2class"].value_counts().items():
        report_lines.append(f"  {cls}: {cnt:,} ({cnt / n_stays * 100:.1f}%)")

    # Cross-tab with HF subtype if available
    if "hf_subtype" in feat_df.columns:
        report_lines.extend(["", "4-CLASS × HF SUBTYPE", "────────────────────"])
        hf_map = feat_df[["stay_id", "hf_subtype"]].drop_duplicates(subset=["stay_id"])
        ct_df = result[["stay_id", "rule_phenotype_4class"]].merge(hf_map, on="stay_id", how="left")
        ct = pd.crosstab(ct_df["rule_phenotype_4class"], ct_df["hf_subtype"], margins=True)
        report_lines.append(ct.to_string())

    # Cross-tab with data-driven clusters
    if "data_driven_cluster" in result.columns:
        report_lines.extend([
            "", "RULE vs DATA-DRIVEN CROSS-TAB", "─────────────────────────────"
        ])
        ct2 = pd.crosstab(
            result["rule_phenotype_4class"],
            result["data_driven_cluster"],
            margins=True,
        )
        report_lines.append(ct2.to_string())

    report_lines.extend([
        "",
        "NEXT STEPS",
        "──────────",
        "  60e: phenotype cards for both rule-based and data-driven",
        "  Step C: use rule_phenotype_2class or data_driven_cluster for stratification",
    ])

    report = "\n".join(report_lines) + "\n"
    report_path = reports_dir / "60d_rule_phenotype_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n  Saved: {report_path}")
    print(report)
    print("[60d] Done.")


if __name__ == "__main__":
    main()
