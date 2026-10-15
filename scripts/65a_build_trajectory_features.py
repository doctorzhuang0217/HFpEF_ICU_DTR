#!/usr/bin/env python3
"""
65a_build_trajectory_features — Construct 0-24h trajectory phenotype features.

Purpose
-------
Build a feature matrix for trajectory phenotyping using the full post-T0
decision window (bins 0-11, T0 to T0+24h). Unlike early phenotyping (60a)
which uses only the first 6h (bins 0-2), trajectory features capture
temporal dynamics across the entire treatment horizon.

Features come from three sources:
  1. Encoder latent trajectories: per-phase mean-pooled z from 50c/55c
     (early=bins 0-2, mid=bins 3-8, late=bins 9-11)
  2. Temporal dynamics: per-dim slope/variability over the full sequence
  3. Raw clinical trajectory summaries: phase-specific vital/lab profiles

The trajectory phenotype is explanatory only and does NOT feed back into
the primary causal estimand (protocol § 12).

Protocol reference: Step 65a (v3.1 § 12 Step B trajectory phenotype)

Inputs
------
- data/interim/45c_state_tensors_v1_mimic.parquet
- data/staging/25f_primary_cohort_candidate_v1_mimic.parquet
- data/interim/50c_latents_baseline_v1_mimic.parquet  (bin-wise)
- data/interim/55c_latents_dann_v1_mimic.parquet  (bin-wise, if available)

Outputs
-------
- data/interim/65a_trajectory_features_v1_mimic.parquet
- reports/65a_trajectory_features_summary.txt

Compute: C3
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path

# ── Phase definitions (2h bins) ──────────────────────────────────────
EARLY_BINS = [0, 1, 2]      # T0+0h to T0+6h
MID_BINS = [3, 4, 5, 6, 7, 8]  # T0+6h to T0+18h
LATE_BINS = [9, 10, 11]     # T0+18h to T0+24h
ALL_BINS = list(range(12))

# ── Clinical variables for trajectory summaries ──────────────────────
HEMO_VARS = [
    "map", "sbp", "dbp", "heart_rate", "spo2", "resp_rate", "fio2",
]
LAB_VARS = [
    "lactate", "creatinine", "bun", "potassium", "sodium",
    "bicarbonate", "glucose", "hemoglobin", "wbc", "platelet",
    "albumin", "bilirubin",
]
ALL_CLINICAL_VARS = HEMO_VARS + LAB_VARS
PHASES = {"early": EARLY_BINS, "mid": MID_BINS, "late": LATE_BINS}


def _sorted_z_cols(columns: list[str]) -> list[str]:
    """Sort latent columns by numeric suffix: z_0, z_1, ..., z_63."""
    z_cols = [c for c in columns if c.startswith("z_")]
    return sorted(z_cols, key=lambda c: int(c.split("_")[1]))


def _build_phase_summaries(state_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-stay × per-phase clinical summaries from state tensor."""
    rows = []
    for sid, grp in state_df.groupby("stay_id"):
        feat: dict = {"stay_id": sid}

        for phase_name, phase_bins in PHASES.items():
            phase = grp[grp["bin"].isin(phase_bins)]

            for var in ALL_CLINICAL_VARS:
                z_col = f"{var}_z"
                mask_col = f"{var}_mask"

                if z_col not in phase.columns:
                    continue

                vals = phase[z_col].values
                masks = (
                    phase[mask_col].values
                    if mask_col in phase.columns
                    else np.ones(len(vals))
                )
                observed = vals[masks > 0]

                prefix = f"{var}_{phase_name}"
                if len(observed) > 0:
                    feat[f"{prefix}_mean"] = observed.mean()
                    feat[f"{prefix}_last"] = observed[-1]
                else:
                    feat[f"{prefix}_mean"] = np.nan
                    feat[f"{prefix}_last"] = np.nan

        # Full-trajectory dynamics (early-to-late delta)
        for var in ALL_CLINICAL_VARS:
            z_col = f"{var}_z"
            mask_col = f"{var}_mask"

            if z_col not in grp.columns:
                continue

            vals = grp[z_col].values
            masks = (
                grp[mask_col].values
                if mask_col in grp.columns
                else np.ones(len(vals))
            )
            observed = vals[masks > 0]

            if len(observed) >= 3:
                feat[f"{var}_traj_slope"] = float(
                    np.polyfit(np.arange(len(observed)), observed, 1)[0]
                )
                feat[f"{var}_traj_std"] = float(observed.std())
            else:
                feat[f"{var}_traj_slope"] = np.nan
                feat[f"{var}_traj_std"] = np.nan

            # Early-to-late delta
            early_phase = grp[grp["bin"].isin(EARLY_BINS)]
            late_phase = grp[grp["bin"].isin(LATE_BINS)]

            e_vals = early_phase[z_col].values
            e_masks = (
                early_phase[mask_col].values
                if mask_col in early_phase.columns
                else np.ones(len(e_vals))
            )
            l_vals = late_phase[z_col].values
            l_masks = (
                late_phase[mask_col].values
                if mask_col in late_phase.columns
                else np.ones(len(l_vals))
            )
            e_obs = e_vals[e_masks > 0]
            l_obs = l_vals[l_masks > 0]

            if len(e_obs) > 0 and len(l_obs) > 0:
                feat[f"{var}_early_late_delta"] = float(
                    l_obs.mean() - e_obs.mean()
                )
            else:
                feat[f"{var}_early_late_delta"] = np.nan

        rows.append(feat)

    return pd.DataFrame(rows)


def _build_treatment_trajectory(state_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-stay treatment trajectory summaries."""
    treatment_cols = {
        "iv_fluid_ml": "fluid",
        "vasopressor_ne_eq": "vaso",
        "diuretic_furo_eq_mg": "diuretic",
    }

    rows = []
    for sid, grp in state_df.groupby("stay_id"):
        feat: dict = {"stay_id": sid}
        grp = grp.sort_values("bin")

        for raw_col, short_name in treatment_cols.items():
            if raw_col not in grp.columns:
                continue
            vals = grp[raw_col].fillna(0).values

            feat[f"{short_name}_total"] = vals.sum()
            feat[f"{short_name}_max_bin"] = vals.max()
            feat[f"{short_name}_any_bin_count"] = int((vals > 0).sum())

            # Phase-specific totals
            for phase_name, phase_bins in PHASES.items():
                phase_mask = grp["bin"].isin(phase_bins)
                feat[f"{short_name}_{phase_name}_total"] = (
                    grp.loc[phase_mask, raw_col].fillna(0).sum()
                )

            # Trend: late vs early mean
            early_mask = grp["bin"].isin(EARLY_BINS)
            late_mask = grp["bin"].isin(LATE_BINS)
            e_mean = grp.loc[early_mask, raw_col].fillna(0).mean()
            l_mean = grp.loc[late_mask, raw_col].fillna(0).mean()
            feat[f"{short_name}_late_vs_early"] = l_mean - e_mean

        # Cumulative fluid balance if available
        if "cumulative_fluid_balance_L" in grp.columns:
            fb = grp["cumulative_fluid_balance_L"].values
            valid_fb = fb[~np.isnan(fb)]
            if len(valid_fb) > 0:
                feat["cum_fluid_balance_final"] = float(valid_fb[-1])
                feat["cum_fluid_balance_max"] = float(valid_fb.max())
            else:
                feat["cum_fluid_balance_final"] = np.nan
                feat["cum_fluid_balance_max"] = np.nan

        rows.append(feat)

    return pd.DataFrame(rows)


def _build_latent_trajectory(
    latents_binwise: pd.DataFrame, label: str
) -> pd.DataFrame:
    """Build trajectory features from bin-wise encoder latents."""
    z_cols = sorted(
        [c for c in latents_binwise.columns if c.startswith("z_")],
        key=lambda x: int(x.split("_")[1]),
    )
    if not z_cols:
        return pd.DataFrame()

    rows = []
    for sid, grp in latents_binwise.groupby("stay_id"):
        feat: dict = {"stay_id": sid}
        grp = grp.sort_values("bin")

        # Phase-specific mean-pooled latents
        for phase_name, phase_bins in PHASES.items():
            phase = grp[grp["bin"].isin(phase_bins)]
            if len(phase) > 0:
                for i, zc in enumerate(z_cols):
                    feat[f"{label}_{phase_name}_z_{i}"] = phase[zc].mean()
            else:
                for i in range(len(z_cols)):
                    feat[f"{label}_{phase_name}_z_{i}"] = np.nan

        # Per-dim temporal dynamics (slope, std) across bins
        z_matrix = grp[z_cols].values  # (n_bins_present, D)
        bins_present = grp["bin"].values

        if len(bins_present) >= 3:
            for i in range(min(len(z_cols), 16)):  # top 16 dims for efficiency
                dim_vals = z_matrix[:, i]
                slope = np.polyfit(bins_present, dim_vals, 1)[0]
                feat[f"{label}_slope_z_{i}"] = float(slope)
                feat[f"{label}_std_z_{i}"] = float(dim_vals.std())
        else:
            for i in range(min(len(z_cols), 16)):
                feat[f"{label}_slope_z_{i}"] = np.nan
                feat[f"{label}_std_z_{i}"] = np.nan

        # Early-late latent shift (L2 distance between phase means)
        early = grp[grp["bin"].isin(EARLY_BINS)]
        late = grp[grp["bin"].isin(LATE_BINS)]
        if len(early) > 0 and len(late) > 0:
            e_mean = early[z_cols].mean().values
            l_mean = late[z_cols].mean().values
            feat[f"{label}_early_late_l2"] = float(
                np.linalg.norm(l_mean - e_mean)
            )
        else:
            feat[f"{label}_early_late_l2"] = np.nan

        rows.append(feat)

    return pd.DataFrame(rows)


def main() -> None:
    proj_cfg = load_config()
    interim_dir = resolve_path(proj_cfg, "interim")
    staging_dir = resolve_path(proj_cfg, "staging")
    reports_dir = resolve_path(proj_cfg, "reports")
    interim_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("65a_build_trajectory_features — Trajectory phenotype features")
    print("=" * 72)

    # ── Load state tensors ────────────────────────────────────────────
    print("\n[1/5] Loading state tensors ...")
    state_df = pd.read_parquet(interim_dir / "45c_state_tensors_v1_mimic.parquet")
    n_stays = state_df["stay_id"].nunique()
    n_bins_per_stay = state_df.groupby("stay_id")["bin"].nunique()
    print(f"  State tensor stays: {n_stays:,}")
    print(f"  Bins per stay: median={n_bins_per_stay.median():.0f}, "
          f"min={n_bins_per_stay.min()}, max={n_bins_per_stay.max()}")

    # ── Load cohort for static covariates ─────────────────────────────
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

    # ── Clinical trajectory summaries ─────────────────────────────────
    print("\n[3/5] Computing phase-specific clinical summaries ...")
    phase_feats = _build_phase_summaries(state_df)
    print(f"  Phase features: {phase_feats.shape[1] - 1} per stay")

    treatment_feats = _build_treatment_trajectory(state_df)
    print(f"  Treatment features: {treatment_feats.shape[1] - 1} per stay")

    # ── Encoder latent trajectories ───────────────────────────────────
    print("\n[4/5] Loading encoder latent trajectories ...")
    latent_frames = []

    baseline_path = interim_dir / "50c_latents_baseline_v1_mimic.parquet"
    if baseline_path.exists():
        lat_base = pd.read_parquet(baseline_path)
        lat_base_traj = _build_latent_trajectory(lat_base, "base")
        latent_frames.append(lat_base_traj)
        print(f"  Baseline latent trajectory: {lat_base_traj.shape[1] - 1} features")
    else:
        print("  Baseline latents: not found (skipping)")

    dann_path = interim_dir / "55c_latents_dann_v1_mimic.parquet"
    if dann_path.exists():
        lat_dann = pd.read_parquet(dann_path)
        lat_dann_traj = _build_latent_trajectory(lat_dann, "dann")
        latent_frames.append(lat_dann_traj)
        print(f"  DANN latent trajectory: {lat_dann_traj.shape[1] - 1} features")
    else:
        print("  DANN latents: not found (skipping)")

    # ── Merge all features ────────────────────────────────────────────
    print("\n[5/5] Merging trajectory feature matrix ...")
    merged = phase_feats.merge(treatment_feats, on="stay_id", how="outer")
    merged = merged.merge(cohort_static, on="stay_id", how="left")
    for lf in latent_frames:
        if not lf.empty:
            merged = merged.merge(lf, on="stay_id", how="left")

    # Trajectory coverage: fraction of 12 bins with at least one observation
    bins_per_stay = (
        state_df.groupby("stay_id")["bin"]
        .nunique()
        .reset_index()
        .rename(columns={"bin": "traj_bin_coverage"})
    )
    bins_per_stay["traj_bin_coverage"] = bins_per_stay["traj_bin_coverage"] / 12.0
    merged = merged.merge(bins_per_stay, on="stay_id", how="left")

    out_path = interim_dir / "65a_trajectory_features_v1_mimic.parquet"
    merged.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"  Saved: {out_path} ({len(merged):,} stays, {merged.shape[1]} columns)")

    # ── Summary report ────────────────────────────────────────────────
    n_phase = len([c for c in phase_feats.columns if c != "stay_id"])
    n_treatment = len([c for c in treatment_feats.columns if c != "stay_id"])
    n_latent = sum(lf.shape[1] - 1 for lf in latent_frames if not lf.empty)
    traj_cov = merged["traj_bin_coverage"]

    report_lines = [
        "=" * 72,
        "65a — Trajectory Phenotype Feature Matrix (MIMIC-IV, T0+0-24h)",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "DATA",
        "────",
        f"  Total stays: {len(merged):,}",
        f"  Phase windows: early (bins 0-2, T0+0-6h), "
        f"mid (bins 3-8, T0+6-18h), late (bins 9-11, T0+18-24h)",
        "",
        "FEATURE COUNTS",
        "──────────────",
        f"  Phase-specific clinical summaries: {n_phase}",
        f"  Treatment trajectory summaries: {n_treatment}",
        f"  Encoder latent trajectory: {n_latent}",
        f"  Static covariates: {len(available_static) - 1}",
        f"  Trajectory coverage: 1",
        f"  Total columns: {merged.shape[1]}",
        "",
        "TRAJECTORY COVERAGE",
        "───────────────────",
        f"  Bin coverage (12 bins):",
        f"    mean={traj_cov.mean():.3f}, median={traj_cov.median():.3f}",
        f"    P5={traj_cov.quantile(0.05):.3f}, P25={traj_cov.quantile(0.25):.3f}",
        f"  Stays with full 12-bin coverage: "
        f"{(traj_cov >= 1.0).sum():,} ({(traj_cov >= 1.0).mean() * 100:.1f}%)",
        f"  Stays with ≥75% coverage: "
        f"{(traj_cov >= 0.75).sum():,} ({(traj_cov >= 0.75).mean() * 100:.1f}%)",
    ]

    if "hf_subtype" in merged.columns:
        report_lines.extend(["", "TRAJECTORY COVERAGE BY HF SUBTYPE",
                             "──────────────────────────────────"])
        for st, grp in merged.groupby("hf_subtype"):
            tc = grp["traj_bin_coverage"]
            report_lines.append(
                f"  {st}: n={len(grp):,}, coverage={tc.mean():.3f}"
            )

    report_lines.extend([
        "",
        "NEXT STEPS",
        "──────────",
        "  65b: trajectory clustering (GMVAE or K-means/GMM)",
        "  65d: trajectory phenotype cards",
    ])

    report = "\n".join(report_lines) + "\n"
    report_path = reports_dir / "65a_trajectory_features_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n  Saved: {report_path}")
    print(report)
    print("[65a] Done.")


if __name__ == "__main__":
    main()
