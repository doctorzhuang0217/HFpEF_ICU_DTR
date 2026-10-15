#!/usr/bin/env python3
"""
60a_build_early_phenotype_features — Construct 0-6h early phenotype features.

Purpose
-------
Build a feature matrix for early (pre-T0) phenotyping using the 0-6h window.
Features come from three sources:
  1. Raw pre-T0 summaries: min/max/mean/last of hemodynamics and labs
  2. Missingness profiles: observation density, never-measured flags
  3. Encoder latents: mean-pooled z from bins 0-2 (0-6h) of 50c/55c

The fallback cascade from the protocol (Section 12):
  - Dense data → snapshots + trends
  - Sparse data → snapshots only (no slopes)
  - Extremely sparse → handled downstream by 60d rule-based fallback

Protocol reference: Step 60a (v3.1 § 12 Step B early phenotype)

Inputs
------
- data/interim/45c_state_tensors_v1_mimic.parquet
- data/staging/25f_primary_cohort_candidate_v1_mimic.parquet
- data/interim/50c_latents_baseline_v1_mimic.parquet  (bin-wise)
- data/interim/55c_latents_dann_v1_mimic.parquet  (bin-wise, if available)

Outputs
-------
- data/interim/60a_early_features_v1_mimic.parquet
- reports/60a_early_features_summary.txt

Compute: C2
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path

# ── Pre-T0 window: bins 0, 1, 2 (hours 0-2, 2-4, 4-6) ──────────────
PRE_T0_BINS = [0, 1, 2]

# ── Vital/lab variables to extract summaries from ────────────────────
HEMO_VARS = [
    "map", "sbp", "dbp", "heart_rate", "spo2", "resp_rate", "fio2",
]
LAB_VARS = [
    "lactate", "creatinine", "bun", "potassium", "sodium",
    "bicarbonate", "glucose", "hemoglobin", "wbc", "platelet",
    "albumin", "bilirubin", "troponin",
]
ALL_CLINICAL_VARS = HEMO_VARS + LAB_VARS

# Minimum observed bins (out of 3) to compute a trend
MIN_BINS_FOR_TREND = 2


def _build_raw_summaries(state_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-stay summaries from pre-T0 bins of the state tensor."""
    pre = state_df[state_df["bin"].isin(PRE_T0_BINS)].copy()

    rows = []
    for sid, grp in pre.groupby("stay_id"):
        feat: dict = {"stay_id": sid}

        for var in ALL_CLINICAL_VARS:
            z_col = f"{var}_z"
            mask_col = f"{var}_mask"

            if z_col not in grp.columns:
                continue

            vals = grp[z_col].values
            masks = grp[mask_col].values if mask_col in grp.columns else np.ones(len(vals))
            observed = vals[masks > 0]

            if len(observed) > 0:
                feat[f"{var}_pre_min"] = observed.min()
                feat[f"{var}_pre_max"] = observed.max()
                feat[f"{var}_pre_mean"] = observed.mean()
                feat[f"{var}_pre_last"] = observed[-1]
                feat[f"{var}_pre_range"] = observed.max() - observed.min()

                # Simple trend: last - first (only if >=2 observed bins)
                if len(observed) >= MIN_BINS_FOR_TREND:
                    feat[f"{var}_pre_trend"] = float(observed[-1] - observed[0])
                else:
                    feat[f"{var}_pre_trend"] = np.nan
            else:
                for suffix in ["_pre_min", "_pre_max", "_pre_mean", "_pre_last",
                               "_pre_range", "_pre_trend"]:
                    feat[f"{var}{suffix}"] = np.nan

            # Missingness features
            if mask_col in grp.columns:
                feat[f"{var}_pre_obs_frac"] = masks.mean()

        rows.append(feat)

    return pd.DataFrame(rows)


def _build_missingness_profile(state_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-stay missingness profile from pre-T0 bins."""
    pre = state_df[state_df["bin"].isin(PRE_T0_BINS)].copy()

    rows = []
    for sid, grp in pre.groupby("stay_id"):
        feat: dict = {"stay_id": sid}

        # Count total observed channels
        n_observed = 0
        n_total = 0
        for var in ALL_CLINICAL_VARS:
            nm_col = f"{var}_never_measured"
            if nm_col in grp.columns:
                feat[f"{var}_never_measured"] = int(grp[nm_col].iloc[0] > 0)
                n_total += 1
                if grp[nm_col].iloc[0] == 0:
                    n_observed += 1

        feat["n_vars_ever_measured"] = n_observed
        feat["frac_vars_measured"] = n_observed / max(n_total, 1)

        rows.append(feat)

    return pd.DataFrame(rows)


def _build_latent_pre_t0(
    latents_binwise: pd.DataFrame, label: str
) -> pd.DataFrame:
    """Mean-pool bin-wise latents over pre-T0 bins."""
    pre = latents_binwise[latents_binwise["bin"].isin(PRE_T0_BINS)].copy()
    z_cols = [c for c in pre.columns if c.startswith("z_")]

    pooled = pre.groupby("stay_id")[z_cols].mean().reset_index()
    pooled.columns = ["stay_id"] + [f"{label}_{c}" for c in z_cols]
    return pooled


def main() -> None:
    proj_cfg = load_config()
    project_root = Path(proj_cfg.get("project_root", "."))
    interim_dir = resolve_path(proj_cfg, "interim")
    staging_dir = resolve_path(proj_cfg, "staging")
    reports_dir = resolve_path(proj_cfg, "reports")
    interim_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("60a_build_early_phenotype_features — 0-6h phenotype feature matrix")
    print("=" * 72)

    # ── Load data ──────────────────────────────────────────────────────
    print("\n[1/5] Loading state tensors ...")
    state_df = pd.read_parquet(interim_dir / "45c_state_tensors_v1_mimic.parquet")
    n_stays = state_df["stay_id"].nunique()
    print(f"  State tensor stays: {n_stays:,}")

    # ── Load cohort for static covariates ──────────────────────────────
    print("\n[2/5] Loading cohort for static covariates ...")
    cohort_path = staging_dir / "25f_primary_cohort_candidate_v1_mimic.parquet"
    cohort = pd.read_parquet(cohort_path)
    static_cols_from_cohort = [
        "stay_id", "age", "gender", "hf_subtype", "comparator_group",
        "dominant_syndrome",
    ]
    available_static = [c for c in static_cols_from_cohort if c in cohort.columns]
    cohort_static = cohort[available_static].drop_duplicates(subset=["stay_id"])
    print(f"  Cohort stays: {len(cohort_static):,}")

    # ── Raw pre-T0 summaries ───────────────────────────────────────────
    print("\n[3/5] Computing raw pre-T0 summaries ...")
    raw_feats = _build_raw_summaries(state_df)
    miss_feats = _build_missingness_profile(state_df)
    print(f"  Raw features: {raw_feats.shape[1] - 1} per stay")
    print(f"  Missingness features: {miss_feats.shape[1] - 1} per stay")

    # ── Encoder latents (pre-T0 pooled) ───────────────────────────────
    print("\n[4/5] Loading encoder latents (pre-T0 bins) ...")
    latent_frames = []

    baseline_path = interim_dir / "50c_latents_baseline_v1_mimic.parquet"
    if baseline_path.exists():
        lat_base = pd.read_parquet(baseline_path)
        lat_base_pre = _build_latent_pre_t0(lat_base, "base")
        latent_frames.append(lat_base_pre)
        print(f"  Baseline latents: {len(lat_base_pre):,} stays, "
              f"{lat_base_pre.shape[1] - 1} dims")
    else:
        print("  Baseline latents: not found (skipping)")

    dann_path = interim_dir / "55c_latents_dann_v1_mimic.parquet"
    if dann_path.exists():
        lat_dann = pd.read_parquet(dann_path)
        lat_dann_pre = _build_latent_pre_t0(lat_dann, "dann")
        latent_frames.append(lat_dann_pre)
        print(f"  DANN latents: {len(lat_dann_pre):,} stays, "
              f"{lat_dann_pre.shape[1] - 1} dims")
    else:
        print("  DANN latents: not found (skipping)")

    # ── Merge all features ─────────────────────────────────────────────
    print("\n[5/5] Merging feature matrix ...")
    merged = raw_feats.merge(miss_feats, on="stay_id", how="outer")
    merged = merged.merge(cohort_static, on="stay_id", how="left")
    for lf in latent_frames:
        merged = merged.merge(lf, on="stay_id", how="left")

    # Compute data density score (for fallback cascade decision)
    obs_frac_cols = [c for c in merged.columns if c.endswith("_pre_obs_frac")]
    if obs_frac_cols:
        merged["data_density_score"] = merged[obs_frac_cols].mean(axis=1)
    else:
        merged["data_density_score"] = np.nan

    # Trend coverage: fraction of vars with valid trends
    trend_cols = [c for c in merged.columns if c.endswith("_pre_trend")]
    if trend_cols:
        merged["trend_coverage"] = merged[trend_cols].notna().mean(axis=1)
    else:
        merged["trend_coverage"] = 0.0

    out_path = interim_dir / "60a_early_features_v1_mimic.parquet"
    merged.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"  Saved: {out_path} ({len(merged):,} stays, {merged.shape[1]} columns)")

    # ── Summary report ─────────────────────────────────────────────────
    n_raw = len([c for c in raw_feats.columns if c != "stay_id"])
    n_miss = len([c for c in miss_feats.columns if c != "stay_id"])
    n_latent = sum(lf.shape[1] - 1 for lf in latent_frames)

    density = merged["data_density_score"]
    trend_cov = merged["trend_coverage"]

    report_lines = [
        "=" * 72,
        "60a — Early Phenotype Feature Matrix (MIMIC-IV, 0-6h)",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "DATA",
        "────",
        f"  Total stays: {len(merged):,}",
        f"  Pre-T0 bins used: {PRE_T0_BINS} (0-6h window)",
        "",
        "FEATURE COUNTS",
        "──────────────",
        f"  Raw clinical summaries: {n_raw}",
        f"  Missingness profile: {n_miss}",
        f"  Encoder latents (pre-T0 pooled): {n_latent}",
        f"  Static covariates: {len(available_static) - 1}",
        f"  Derived (density, trend_coverage): 2",
        f"  Total columns: {merged.shape[1]}",
        "",
        "DATA DENSITY (for fallback cascade)",
        "───────────────────────────────────",
        f"  Density score (mean obs_frac across vars):",
        f"    mean={density.mean():.3f}, median={density.median():.3f}",
        f"    P5={density.quantile(0.05):.3f}, P25={density.quantile(0.25):.3f}",
        f"  Trend coverage (frac vars with valid trend):",
        f"    mean={trend_cov.mean():.3f}, median={trend_cov.median():.3f}",
        f"    P5={trend_cov.quantile(0.05):.3f}, P25={trend_cov.quantile(0.25):.3f}",
    ]

    if "hf_subtype" in merged.columns:
        report_lines.extend(["", "DENSITY BY HF SUBTYPE", "─────────────────────"])
        for st, grp in merged.groupby("hf_subtype"):
            d = grp["data_density_score"]
            t = grp["trend_coverage"]
            report_lines.append(
                f"  {st}: n={len(grp):,}, density={d.mean():.3f}, "
                f"trend_cov={t.mean():.3f}"
            )

    report_lines.extend([
        "",
        "FALLBACK CASCADE STATUS",
        "───────────────────────",
        f"  Stays with density >= 0.50: {(density >= 0.50).sum():,} "
        f"({(density >= 0.50).mean() * 100:.1f}%)",
        f"  Stays with density >= 0.33: {(density >= 0.33).sum():,} "
        f"({(density >= 0.33).mean() * 100:.1f}%)",
        f"  Stays with trend_coverage >= 0.50: {(trend_cov >= 0.50).sum():,} "
        f"({(trend_cov >= 0.50).mean() * 100:.1f}%)",
        "  Decision: if density P25 >= 0.33 → proceed with clustering",
        "            if density P25 < 0.33 → snapshot-only mode",
        "",
        "NEXT STEPS",
        "──────────",
        "  60b: cluster early phenotype features",
        "  60d: rule-based fallback (if clustering fails)",
    ])

    report = "\n".join(report_lines) + "\n"
    report_path = reports_dir / "60a_early_features_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n  Saved: {report_path}")
    print(report)
    print("[60a] Done.")


if __name__ == "__main__":
    main()
