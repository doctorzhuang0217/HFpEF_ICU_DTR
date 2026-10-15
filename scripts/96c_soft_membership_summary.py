#!/usr/bin/env python3
"""
96c_soft_membership_summary — GMM soft membership and boundary analysis.

[EXPLORATORY_96X — NOT part of v3.1 core protocol. Results are provisional
and should NOT be cited in the manuscript main body without explicit upgrade.]

Purpose
-------
Extract posterior membership probabilities from GMM fits (K=2, K=4) on the
Step 65a trajectory features. This enables:

  1. Membership entropy: how "certain" is each patient's cluster assignment?
  2. Near-boundary patient identification (entropy > threshold)
  3. Clinical profiling of high-entropy vs low-entropy patients
  4. HF-subtype enrichment in boundary zones
  5. Outcome comparison: boundary vs core patients

This analysis responds to the "fuzzy/soft assignment" recommendation from
external review — acknowledging that many ICU patients do not cleanly belong
to a single trajectory cluster.

Protocol reference: Exploratory extension (post-v3.1, Step 96 series)

Inputs
------
- data/interim/65a_trajectory_features_v1_mimic.parquet
- data/interim/45d_outcomes_absorb_v1_mimic.parquet
- data/processed/phenotype_labels_frozen_v1_mimic.parquet

Outputs
-------
- data/interim/96c_soft_membership_v1_mimic.parquet
- tables/96c_soft_membership_entropy_summary.csv
- tables/96c_boundary_patient_profiles.csv
- tables/96c_boundary_vs_core_outcomes.csv
- figures/96c_entropy_distribution.png
- figures/96c_boundary_clinical_comparison.png
- reports/96c_soft_membership_summary.txt

Compute: C2

[EXPLORATORY_96X]
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path

EXPLORATORY_TAG = "EXPLORATORY_96X"

SEED = 2024
N_INIT = 10
ENTROPY_THRESHOLD_QUANTILE = 0.75  # top 25% entropy → "boundary"

# Clinical variables for profiling
PROFILE_VARS = [
    "map_early_mean", "map_late_mean", "map_traj_slope",
    "heart_rate_early_mean", "heart_rate_late_mean", "heart_rate_traj_slope",
    "lactate_early_mean", "lactate_late_mean", "lactate_traj_slope",
    "creatinine_early_mean", "creatinine_late_mean", "creatinine_traj_slope",
    "spo2_early_mean", "spo2_late_mean", "spo2_traj_slope",
    "fluid_total", "vaso_total", "diuretic_total",
    "cum_fluid_balance_final",
]


def _sorted_z_cols(columns: list[str]) -> list[str]:
    """Sort latent-derived columns by numeric z suffix."""
    def z_idx(col: str) -> int:
        if col.startswith("z_"):
            return int(col.split("_")[1])
        return int(col.rsplit("_", 1)[1])
    return sorted(columns, key=z_idx)


def _select_trajectory_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Select features for trajectory clustering — same hierarchy as 65b."""
    base_phase_cols = _sorted_z_cols(
        [c for c in df.columns if c.startswith("base_") and "_z_" in c]
    )
    if len(base_phase_cols) >= 16:
        print(f"  Using baseline latent trajectory features: {len(base_phase_cols)} dims")
        return df[base_phase_cols].values, base_phase_cols

    dann_phase_cols = _sorted_z_cols(
        [c for c in df.columns if c.startswith("dann_") and "_z_" in c]
    )
    if len(dann_phase_cols) >= 16:
        print(f"  Using DANN latent trajectory features: {len(dann_phase_cols)} dims")
        return df[dann_phase_cols].values, dann_phase_cols

    raw_cols = sorted(
        [c for c in df.columns
         if any(c.endswith(s) for s in ["_mean", "_last", "_slope", "_std",
                                         "_delta", "_total", "_late_vs_early"])
         and c != "stay_id"]
    )
    print(f"  Fallback to raw trajectory features: {len(raw_cols)} dims")
    return df[raw_cols].values, raw_cols


def _compute_entropy(probs: np.ndarray) -> np.ndarray:
    """Compute Shannon entropy of membership probabilities (per row)."""
    # Clip to avoid log(0)
    p = np.clip(probs, 1e-12, 1.0)
    return -np.sum(p * np.log2(p), axis=1)


def _profile_comparison(
    feat_df: pd.DataFrame,
    group_col: str,
    var_list: list[str],
) -> pd.DataFrame:
    """Compare clinical profiles between boundary and core patients."""
    rows = []
    for group_label in sorted(feat_df[group_col].unique()):
        grp = feat_df[feat_df[group_col] == group_label]
        for var in var_list:
            if var not in grp.columns:
                continue
            vals = grp[var].dropna()
            if len(vals) == 0:
                continue
            rows.append({
                "group": group_label,
                "variable": var,
                "n_obs": len(vals),
                "median": round(vals.median(), 4),
                "q25": round(vals.quantile(0.25), 4),
                "q75": round(vals.quantile(0.75), 4),
                "mean": round(vals.mean(), 4),
                "std": round(vals.std(), 4),
            })
    return pd.DataFrame(rows)


def main() -> None:
    proj_cfg = load_config()
    interim_dir = resolve_path(proj_cfg, "interim")
    processed_dir = resolve_path(proj_cfg, "processed")
    reports_dir = resolve_path(proj_cfg, "reports")
    tables_dir = resolve_path(proj_cfg, "tables")
    figures_dir = resolve_path(proj_cfg, "figures")
    for d in [reports_dir, tables_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"96c_soft_membership_summary [{EXPLORATORY_TAG}]")
    print("  GMM soft membership probabilities and boundary analysis")
    print("=" * 72)

    # ── [1/7] Load trajectory features ────────────────────────────────
    print("\n[1/7] Loading trajectory features ...")
    feat_path = interim_dir / "65a_trajectory_features_v1_mimic.parquet"
    if not feat_path.exists():
        print(f"  ERROR: {feat_path} not found")
        sys.exit(1)
    feat_df = pd.read_parquet(feat_path)
    feat_df = feat_df.drop_duplicates(subset=["stay_id"])
    n_stays = len(feat_df)
    print(f"  Stays: {n_stays:,}")

    # ── [2/7] Feature selection and preprocessing ─────────────────────
    print("\n[2/7] Selecting and preprocessing features ...")
    X_raw, feat_cols = _select_trajectory_features(feat_df)

    nan_mask = np.isnan(X_raw)
    n_nan = int(nan_mask.sum())
    if n_nan > 0:
        print(f"  Imputing {n_nan:,} NaN values with column medians")
        col_medians = np.nanmedian(X_raw, axis=0)
        for j in range(X_raw.shape[1]):
            X_raw[nan_mask[:, j], j] = col_medians[j]

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    print(f"  Feature matrix: {X.shape}")

    # ── [3/7] Fit GMM and extract soft memberships ────────────────────
    print("\n[3/7] Fitting GMM models and extracting soft memberships ...")

    soft_df = pd.DataFrame({"stay_id": feat_df["stay_id"].values})

    for k in [2, 4]:
        print(f"\n  Fitting GMM K={k} ...")
        gmm = GaussianMixture(
            n_components=k, n_init=N_INIT, random_state=SEED,
            covariance_type="full", max_iter=200,
        )
        gmm.fit(X)
        probs = gmm.predict_proba(X)
        labels = gmm.predict(X)

        # Store probabilities
        for c in range(k):
            soft_df[f"gmm_k{k}_prob_{c}"] = probs[:, c].round(6)
        soft_df[f"gmm_k{k}_label"] = labels

        # Compute entropy
        entropy = _compute_entropy(probs)
        max_entropy = np.log2(k)
        norm_entropy = entropy / max_entropy  # normalized to [0, 1]
        soft_df[f"gmm_k{k}_entropy"] = entropy.round(6)
        soft_df[f"gmm_k{k}_norm_entropy"] = norm_entropy.round(6)

        # Max probability (confidence)
        max_prob = probs.max(axis=1)
        soft_df[f"gmm_k{k}_max_prob"] = max_prob.round(6)

        # Summary stats
        print(f"    Entropy: mean={entropy.mean():.4f}, median={np.median(entropy):.4f}, "
              f"max={entropy.max():.4f} (max possible={max_entropy:.4f})")
        print(f"    Normalized entropy: mean={norm_entropy.mean():.4f}, "
              f"P75={np.percentile(norm_entropy, 75):.4f}, "
              f"P90={np.percentile(norm_entropy, 90):.4f}")
        print(f"    Max prob: mean={max_prob.mean():.4f}, "
              f"<0.6: {(max_prob < 0.6).sum():,} ({(max_prob < 0.6).mean()*100:.1f}%), "
              f"<0.7: {(max_prob < 0.7).sum():,} ({(max_prob < 0.7).mean()*100:.1f}%)")

    # ── [4/7] Identify boundary patients ──────────────────────────────
    print("\n[4/7] Identifying boundary patients ...")

    entropy_summary_rows = []

    for k in [2, 4]:
        entropy_col = f"gmm_k{k}_norm_entropy"
        threshold = soft_df[entropy_col].quantile(ENTROPY_THRESHOLD_QUANTILE)
        is_boundary = soft_df[entropy_col] >= threshold
        n_boundary = is_boundary.sum()
        soft_df[f"gmm_k{k}_boundary"] = is_boundary.astype(int)

        print(f"\n  K={k}: boundary threshold (P{int(ENTROPY_THRESHOLD_QUANTILE*100)}) = "
              f"{threshold:.4f}")
        print(f"  Boundary patients: {n_boundary:,} ({n_boundary/n_stays*100:.1f}%)")

        # Per-cluster boundary fraction
        for cl in range(k):
            cl_mask = soft_df[f"gmm_k{k}_label"] == cl
            cl_boundary = (is_boundary & cl_mask).sum()
            cl_total = cl_mask.sum()
            entropy_summary_rows.append({
                "K": k,
                "cluster": cl,
                "n_total": int(cl_total),
                "n_boundary": int(cl_boundary),
                "pct_boundary": round(cl_boundary / cl_total * 100, 1) if cl_total > 0 else 0,
                "entropy_mean": round(soft_df.loc[cl_mask, entropy_col].mean(), 4),
                "entropy_std": round(soft_df.loc[cl_mask, entropy_col].std(), 4),
                "max_prob_mean": round(soft_df.loc[cl_mask, f"gmm_k{k}_max_prob"].mean(), 4),
            })
            print(f"    Cluster {cl}: {cl_boundary}/{cl_total} boundary "
                  f"({cl_boundary/cl_total*100:.1f}%)")

    entropy_summary = pd.DataFrame(entropy_summary_rows)

    # ── [5/7] Boundary vs core clinical profiles ─────────────────────
    print("\n[5/7] Comparing boundary vs core patients ...")

    profile_dfs = []
    outcome_comparison_rows = []

    # Load outcomes
    outcomes_path = interim_dir / "45d_outcomes_absorb_v1_mimic.parquet"
    outcomes_df = None
    if outcomes_path.exists():
        outcomes_df = pd.read_parquet(outcomes_path).drop_duplicates(subset=["stay_id"])

    # Load frozen labels for HF subtype
    frozen_path = processed_dir / "phenotype_labels_frozen_v1_mimic.parquet"
    frozen_df = None
    if frozen_path.exists():
        frozen_df = pd.read_parquet(frozen_path).drop_duplicates(subset=["stay_id"])

    for k in [2, 4]:
        boundary_col = f"gmm_k{k}_boundary"
        analysis_df = feat_df.copy()
        analysis_df[boundary_col] = soft_df[boundary_col].values
        analysis_df["zone"] = np.where(
            analysis_df[boundary_col] == 1, "boundary", "core"
        )

        available_vars = [v for v in PROFILE_VARS if v in analysis_df.columns]
        profiles = _profile_comparison(analysis_df, "zone", available_vars)
        profiles["K"] = k
        profile_dfs.append(profiles)

        # Outcome comparison
        if outcomes_df is not None:
            merged = analysis_df[["stay_id", "zone"]].merge(
                outcomes_df[["stay_id", "mortality_28d", "mortality_7d"]],
                on="stay_id", how="left",
            )
            for zone in ["core", "boundary"]:
                grp = merged[merged["zone"] == zone]
                outcome_comparison_rows.append({
                    "K": k,
                    "zone": zone,
                    "n": len(grp),
                    "mortality_28d": round(grp["mortality_28d"].mean() * 100, 1),
                    "mortality_7d": round(grp["mortality_7d"].mean() * 100, 1),
                })

            print(f"\n  K={k} outcome comparison:")
            for row in outcome_comparison_rows[-2:]:
                print(f"    {row['zone']}: n={row['n']:,}, "
                      f"mort_28d={row['mortality_28d']:.1f}%")

        # HF subtype enrichment in boundary zone
        if frozen_df is not None and "hf_subtype" in frozen_df.columns:
            merged_hf = analysis_df[["stay_id", "zone"]].merge(
                frozen_df[["stay_id", "hf_subtype"]], on="stay_id", how="left",
            )
            ct_hf = pd.crosstab(merged_hf["zone"], merged_hf["hf_subtype"],
                                normalize="index")
            print(f"\n  K={k} HF subtype by zone:")
            print(ct_hf.round(3).to_string().replace("\n", "\n    "))

    boundary_profiles = pd.concat(profile_dfs, ignore_index=True) if profile_dfs else pd.DataFrame()
    outcome_comparison = pd.DataFrame(outcome_comparison_rows)

    # ── [6/7] Visualizations ──────────────────────────────────────────
    print("\n[6/7] Generating visualizations ...")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Figure 1: Entropy distribution
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        for i, k in enumerate([2, 4]):
            ax = axes[i]
            entropy_vals = soft_df[f"gmm_k{k}_norm_entropy"].values
            threshold = np.percentile(entropy_vals,
                                      ENTROPY_THRESHOLD_QUANTILE * 100)
            ax.hist(entropy_vals, bins=50, color="steelblue", alpha=0.7,
                    edgecolor="white")
            ax.axvline(threshold, color="red", linestyle="--", linewidth=2,
                       label=f"P{int(ENTROPY_THRESHOLD_QUANTILE*100)} = {threshold:.3f}")
            ax.set_xlabel("Normalized Entropy")
            ax.set_ylabel("Count")
            ax.set_title(f"K={k} Membership Entropy\n[{EXPLORATORY_TAG}]")
            ax.legend(fontsize=8)

        fig.tight_layout()
        entropy_fig_path = figures_dir / "96c_entropy_distribution.png"
        fig.savefig(entropy_fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {entropy_fig_path}")

        # Figure 2: Boundary vs core clinical comparison (K=2)
        if not boundary_profiles.empty:
            k2_profiles = boundary_profiles[boundary_profiles["K"] == 2]
            if not k2_profiles.empty:
                fig, ax = plt.subplots(figsize=(12, 6))

                core_vals = k2_profiles[k2_profiles["group"] == "core"].set_index("variable")
                boundary_vals = k2_profiles[k2_profiles["group"] == "boundary"].set_index("variable")

                common_vars = sorted(set(core_vals.index) & set(boundary_vals.index))
                if len(common_vars) > 0:
                    x_pos = np.arange(len(common_vars))
                    width = 0.35

                    ax.bar(x_pos - width/2, core_vals.loc[common_vars, "median"],
                           width, label="Core", color="steelblue", alpha=0.7)
                    ax.bar(x_pos + width/2, boundary_vals.loc[common_vars, "median"],
                           width, label="Boundary", color="coral", alpha=0.7)

                    ax.set_xticks(x_pos)
                    ax.set_xticklabels([v.replace("_", "\n") for v in common_vars],
                                       fontsize=6, rotation=45, ha="right")
                    ax.set_ylabel("Median Value")
                    ax.set_title(f"Core vs Boundary Patients (K=2)\n[{EXPLORATORY_TAG}]")
                    ax.legend()

                fig.tight_layout()
                clinical_fig_path = figures_dir / "96c_boundary_clinical_comparison.png"
                fig.savefig(clinical_fig_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved: {clinical_fig_path}")
    except ImportError:
        print("  matplotlib not available — skipping visualizations")

    # ── [7/7] Save outputs ────────────────────────────────────────────
    print("\n[7/7] Saving outputs ...")

    soft_path = interim_dir / "96c_soft_membership_v1_mimic.parquet"
    soft_df.to_parquet(soft_path, index=False)
    print(f"  Saved: {soft_path} ({len(soft_df):,} stays, {soft_df.shape[1]} columns)")

    entropy_path = tables_dir / "96c_soft_membership_entropy_summary.csv"
    entropy_summary.to_csv(entropy_path, index=False)
    print(f"  Saved: {entropy_path}")

    if not boundary_profiles.empty:
        profiles_path = tables_dir / "96c_boundary_patient_profiles.csv"
        boundary_profiles.to_csv(profiles_path, index=False)
        print(f"  Saved: {profiles_path}")

    if not outcome_comparison.empty:
        outcome_path = tables_dir / "96c_boundary_vs_core_outcomes.csv"
        outcome_comparison.to_csv(outcome_path, index=False)
        print(f"  Saved: {outcome_path}")

    # ── Report ────────────────────────────────────────────────────────
    report_lines = [
        "=" * 72,
        f"96c — GMM Soft Membership & Boundary Analysis [{EXPLORATORY_TAG}]",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Tag: {EXPLORATORY_TAG}",
        "",
        "STATUS: EXPLORATORY — results are provisional and NOT part of the",
        "v3.1 core protocol. Do not cite in manuscript main body without",
        "explicit upgrade decision.",
        "",
        "PURPOSE",
        "───────",
        "Extract GMM posterior membership probabilities to identify patients",
        "with uncertain cluster assignments (high entropy). This addresses",
        "the external review recommendation for fuzzy/soft phenotype assignment.",
        "",
        "ENTROPY SUMMARY BY CLUSTER",
        "──────────────────────────",
    ]
    for _, row in entropy_summary.iterrows():
        report_lines.append(
            f"  K={int(row['K'])} Cluster {int(row['cluster'])}: "
            f"n={int(row['n_total']):,}, boundary={int(row['n_boundary'])} "
            f"({row['pct_boundary']:.1f}%), "
            f"entropy={row['entropy_mean']:.4f}±{row['entropy_std']:.4f}, "
            f"max_prob={row['max_prob_mean']:.4f}"
        )

    if not outcome_comparison.empty:
        report_lines.extend([
            "",
            "BOUNDARY vs CORE OUTCOMES",
            "─────────────────────────",
        ])
        for _, row in outcome_comparison.iterrows():
            report_lines.append(
                f"  K={int(row['K'])} {row['zone']}: n={int(row['n']):,}, "
                f"mort_28d={row['mortality_28d']:.1f}%, "
                f"mort_7d={row['mortality_7d']:.1f}%"
            )

    report_lines.extend([
        "",
        "INTERPRETATION",
        "──────────────",
        "High-entropy patients sit at the boundary between clusters and may",
        "represent:",
        "  (a) Transitional states between trajectory phenotypes",
        "  (b) Patients whose disease course is genuinely ambiguous at T0+24h",
        "  (c) Subpopulations that require finer-grained phenotyping (K>4)",
        "",
        "If boundary patients show higher mortality, this suggests that",
        "trajectory uncertainty itself may be clinically informative —",
        "analogous to the 85j informative monitoring finding.",
        "",
        "CLINICAL RELEVANCE",
        "───────────────────",
        "Soft membership probabilities can be used downstream for:",
        "  1. Weighted subgroup analysis in StepC (sensitivity)",
        "  2. Identifying patients who may benefit from adaptive strategies",
        "  3. Quantifying the 'sharpness' of the trajectory phenotype boundary",
        "",
        "OUTPUTS",
        "───────",
        "  data/interim/96c_soft_membership_v1_mimic.parquet",
        "  tables/96c_soft_membership_entropy_summary.csv",
        "  tables/96c_boundary_patient_profiles.csv",
        "  tables/96c_boundary_vs_core_outcomes.csv",
        "  figures/96c_entropy_distribution.png",
        "  figures/96c_boundary_clinical_comparison.png",
        "",
        f"[{EXPLORATORY_TAG}] END",
    ])

    report = "\n".join(report_lines) + "\n"
    report_path = reports_dir / "96c_soft_membership_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {report_path}")
    print(report)
    print(f"[96c] Done. [{EXPLORATORY_TAG}]")


if __name__ == "__main__":
    main()
