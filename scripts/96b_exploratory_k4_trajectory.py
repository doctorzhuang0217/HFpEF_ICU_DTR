#!/usr/bin/env python3
"""
96b_exploratory_k4_trajectory — Exploratory K=4 trajectory phenotyping.

[EXPLORATORY_96X — NOT part of v3.1 core protocol. Results are provisional
and should NOT be cited in the manuscript main body without explicit upgrade.]

Purpose
-------
Run K=4 trajectory clustering on the same Step 65a feature matrix, with full
stability bootstrap and phenotype cards. This tests whether the post-T0 disease
course naturally splits into 4 trajectory branches (beyond the primary K=2).

Key questions:
  1. Is K=4 stable (ARI_P5 ≥ 0.40)?
  2. Do the 4 clusters have clinically interpretable differences?
  3. How do K=4 clusters map onto the 2 early phenotypes (crosswalk)?
  4. Does K=4 provide additional outcome stratification beyond K=2?

Protocol reference: Exploratory extension (post-v3.1, Step 96 series)

Inputs
------
- data/interim/65a_trajectory_features_v1_mimic.parquet
- data/interim/45d_outcomes_absorb_v1_mimic.parquet
- data/processed/phenotype_labels_frozen_v1_mimic.parquet

Outputs
-------
- data/interim/96b_trajectory_k4_clusters_v1_mimic.parquet
- tables/96b_trajectory_k4_metrics.csv
- tables/96b_trajectory_k4_stability.csv
- tables/96b_trajectory_k4_phenotype_cards.csv
- tables/96b_trajectory_k4_outcome_summary.csv
- tables/96b_trajectory_k4_vs_k2_crosswalk.csv
- figures/96b_trajectory_k4_heatmap.png
- reports/96b_trajectory_k4_summary.txt

Compute: C2-C3

[EXPLORATORY_96X]
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path

EXPLORATORY_TAG = "EXPLORATORY_96X"

K_EXPLORATORY = 4
K_COMPARE = [2, 3, 4, 5]  # sweep for context
SEED = 2024
N_INIT = 10
N_BOOTSTRAP = 100
MIN_CLUSTER_FRAC = 0.05

# Clinical variables for phenotype cards
CARD_VARS_BY_PHASE = {
    "early": [
        "map_early_mean", "heart_rate_early_mean", "spo2_early_mean",
        "lactate_early_mean", "creatinine_early_mean",
    ],
    "mid": [
        "map_mid_mean", "heart_rate_mid_mean", "spo2_mid_mean",
        "lactate_mid_mean", "creatinine_mid_mean",
    ],
    "late": [
        "map_late_mean", "heart_rate_late_mean", "spo2_late_mean",
        "lactate_late_mean", "creatinine_late_mean",
    ],
}

TRAJ_DYNAMICS_VARS = [
    "map_traj_slope", "heart_rate_traj_slope", "lactate_traj_slope",
    "creatinine_traj_slope", "spo2_traj_slope",
    "map_traj_std", "heart_rate_traj_std", "lactate_traj_std",
    "map_early_late_delta", "heart_rate_early_late_delta",
    "lactate_early_late_delta", "creatinine_early_late_delta",
]

TREATMENT_VARS = [
    "fluid_total", "vaso_total", "diuretic_total",
    "fluid_early_total", "fluid_mid_total", "fluid_late_total",
    "vaso_early_total", "vaso_mid_total", "vaso_late_total",
    "fluid_late_vs_early", "vaso_late_vs_early",
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


def _impute_and_scale(X_raw: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    """Impute NaN with column medians and standardize."""
    nan_mask = np.isnan(X_raw)
    n_nan = int(nan_mask.sum())
    if n_nan > 0:
        print(f"  Imputing {n_nan:,} NaN values with column medians")
        col_medians = np.nanmedian(X_raw, axis=0)
        for j in range(X_raw.shape[1]):
            X_raw[nan_mask[:, j], j] = col_medians[j]
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    return X, scaler


def _compute_cluster_cards(
    feat_df: pd.DataFrame,
    labels: pd.Series,
    label_name: str,
    var_list: list[str],
) -> pd.DataFrame:
    """Compute median + IQR for each variable by cluster."""
    rows = []
    for cluster_id in sorted(labels.unique()):
        mask = labels == cluster_id
        grp = feat_df[mask]
        for var in var_list:
            if var not in grp.columns:
                continue
            vals = grp[var].dropna()
            if len(vals) == 0:
                continue
            rows.append({
                "label_source": label_name,
                "cluster": cluster_id,
                "variable": var,
                "n_obs": len(vals),
                "median": round(vals.median(), 4),
                "q25": round(vals.quantile(0.25), 4),
                "q75": round(vals.quantile(0.75), 4),
                "mean": round(vals.mean(), 4),
                "std": round(vals.std(), 4),
            })
    return pd.DataFrame(rows)


def _compute_outcome_summary(
    labels: pd.Series,
    label_name: str,
    outcomes_df: pd.DataFrame,
    stay_ids: pd.Series,
) -> pd.DataFrame:
    """Compute outcome rates per cluster."""
    merged = pd.DataFrame({"stay_id": stay_ids, "cluster": labels.values})
    merged = merged.merge(
        outcomes_df[["stay_id", "mortality_28d", "mortality_7d"]],
        on="stay_id", how="left",
    )
    rows = []
    for cluster_id in sorted(labels.unique()):
        grp = merged[merged["cluster"] == cluster_id]
        n = len(grp)
        rows.append({
            "label_source": label_name,
            "cluster": cluster_id,
            "n": n,
            "pct": round(n / len(merged) * 100, 1),
            "mortality_28d": round(grp["mortality_28d"].mean() * 100, 1),
            "mortality_7d": round(grp["mortality_7d"].mean() * 100, 1),
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
    print(f"96b_exploratory_k4_trajectory [{EXPLORATORY_TAG}]")
    print("  Exploratory K=4 trajectory phenotyping with stability assessment")
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
    X, scaler = _impute_and_scale(X_raw.copy())
    print(f"  Feature matrix: {X.shape}")

    # ── [3/7] Clustering sweep (K=2,3,4,5 for context) ───────────────
    print(f"\n[3/7] Running clustering sweep (K={K_COMPARE}) ...")
    metrics_rows = []
    labels_dict = {}

    for k in tqdm(K_COMPARE, desc="K sweep", unit="k"):
        # K-means
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED, max_iter=300)
        km_labels = km.fit_predict(X)
        km_sil = silhouette_score(
            X, km_labels, sample_size=min(10000, n_stays), random_state=SEED,
        )
        km_ch = calinski_harabasz_score(X, km_labels)
        _, km_counts = np.unique(km_labels, return_counts=True)
        km_min_frac = km_counts.min() / n_stays

        metrics_rows.append({
            "method": "kmeans", "K": k,
            "silhouette": round(km_sil, 4),
            "calinski_harabasz": round(km_ch, 1),
            "inertia": round(km.inertia_, 1),
            "min_cluster_frac": round(km_min_frac, 4),
        })
        labels_dict[f"kmeans_k{k}"] = km_labels

        # GMM
        gmm = GaussianMixture(
            n_components=k, n_init=N_INIT, random_state=SEED,
            covariance_type="full", max_iter=200,
        )
        gmm_labels = gmm.fit_predict(X)
        gmm_sil = silhouette_score(
            X, gmm_labels, sample_size=min(10000, n_stays), random_state=SEED,
        )
        gmm_ch = calinski_harabasz_score(X, gmm_labels)
        gmm_bic = gmm.bic(X)
        _, gmm_counts = np.unique(gmm_labels, return_counts=True)
        gmm_min_frac = gmm_counts.min() / n_stays

        metrics_rows.append({
            "method": "gmm", "K": k,
            "silhouette": round(gmm_sil, 4),
            "calinski_harabasz": round(gmm_ch, 1),
            "bic": round(gmm_bic, 1),
            "min_cluster_frac": round(gmm_min_frac, 4),
        })
        labels_dict[f"gmm_k{k}"] = gmm_labels

        print(f"  K={k}: kmeans sil={km_sil:.3f} CH={km_ch:.0f} min_frac={km_min_frac:.3f} | "
              f"gmm sil={gmm_sil:.3f} BIC={gmm_bic:.0f}")

    metrics_df = pd.DataFrame(metrics_rows)

    # Focus on K=4
    k4_labels = labels_dict["kmeans_k4"]
    _, k4_counts = np.unique(k4_labels, return_counts=True)
    print(f"\n  K=4 cluster distribution: {dict(zip(range(4), k4_counts))}")
    print(f"  K=4 min cluster fraction: {k4_counts.min() / n_stays:.3f}")

    # ── [4/7] Stability bootstrap for K=4 ─────────────────────────────
    print(f"\n[4/7] Running {N_BOOTSTRAP} bootstrap resamples for stability ...")
    rng = np.random.RandomState(SEED)
    stability_rows = []

    for k in tqdm([2, 3, 4, 5], desc="Stability by K", unit="k"):
        ref_labels = labels_dict[f"kmeans_k{k}"]
        aris, nmis = [], []
        for b in tqdm(range(N_BOOTSTRAP), desc=f"K={k} bootstrap", unit="iter", leave=False):
            idx = rng.choice(n_stays, size=n_stays, replace=True)
            X_boot = X[idx]
            ref_boot = ref_labels[idx]
            km_boot = KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED + b,
                             max_iter=300)
            boot_labels = km_boot.fit_predict(X_boot)
            aris.append(adjusted_rand_score(ref_boot, boot_labels))
            nmis.append(normalized_mutual_info_score(ref_boot, boot_labels))

        ari_arr = np.array(aris)
        nmi_arr = np.array(nmis)
        stable = (ari_arr.mean() >= 0.60) and (np.percentile(ari_arr, 5) >= 0.40)
        stability_rows.append({
            "K": k,
            "ari_mean": round(ari_arr.mean(), 4),
            "ari_std": round(ari_arr.std(), 4),
            "ari_p5": round(np.percentile(ari_arr, 5), 4),
            "nmi_mean": round(nmi_arr.mean(), 4),
            "nmi_std": round(nmi_arr.std(), 4),
            "nmi_p5": round(np.percentile(nmi_arr, 5), 4),
            "n_bootstrap": N_BOOTSTRAP,
            "stable": stable,
        })
        verdict = "PASS" if stable else "FAIL"
        print(f"  K={k}: ARI={ari_arr.mean():.3f}±{ari_arr.std():.3f} "
              f"(P5={np.percentile(ari_arr, 5):.3f}) → {verdict}")

    stability_df = pd.DataFrame(stability_rows)
    k4_stable = stability_df[stability_df["K"] == 4]["stable"].iloc[0]

    # ── [5/7] Phenotype cards for K=4 ─────────────────────────────────
    print("\n[5/7] Computing K=4 phenotype cards ...")
    all_vars = []
    for phase_vars in CARD_VARS_BY_PHASE.values():
        all_vars.extend(phase_vars)
    all_vars.extend(TRAJ_DYNAMICS_VARS)
    all_vars.extend(TREATMENT_VARS)

    k4_series = pd.Series(k4_labels, index=feat_df.index)
    cards = _compute_cluster_cards(feat_df, k4_series, "trajectory_k4", all_vars)
    print(f"  {len(cards)} card rows computed")

    # Outcomes
    outcomes_path = interim_dir / "45d_outcomes_absorb_v1_mimic.parquet"
    if outcomes_path.exists():
        outcomes_df = pd.read_parquet(outcomes_path).drop_duplicates(subset=["stay_id"])
        outcome_summary = _compute_outcome_summary(
            k4_series, "trajectory_k4", outcomes_df, feat_df["stay_id"],
        )
        print("  Outcome summary:")
        for _, row in outcome_summary.iterrows():
            print(f"    Cluster {int(row['cluster'])}: n={int(row['n']):,} ({row['pct']:.1f}%), "
                  f"mort_28d={row['mortality_28d']:.1f}%")
    else:
        outcome_summary = pd.DataFrame()

    # ── [6/7] K=4 vs K=2 crosswalk ───────────────────────────────────
    print("\n[6/7] Computing K=4 vs K=2 crosswalk ...")
    k2_labels = labels_dict["kmeans_k2"]
    ct_k4_k2 = pd.crosstab(
        pd.Series(k4_labels, name="k4_cluster"),
        pd.Series(k2_labels, name="k2_cluster"),
    )
    print("  K=4 × K=2 contingency:")
    print(ct_k4_k2.to_string().replace("\n", "\n    "))

    ct_norm = pd.crosstab(
        pd.Series(k4_labels, name="k4_cluster"),
        pd.Series(k2_labels, name="k2_cluster"),
        normalize="index",
    )
    print("\n  Row-normalized (P(k2 | k4)):")
    print(ct_norm.round(3).to_string().replace("\n", "\n    "))

    # Check: does each K=2 cluster split into ~2 K=4 clusters?
    k4_to_k2_dominant = {}
    for k4_cl in range(4):
        dominant_k2 = ct_norm.loc[k4_cl].idxmax()
        dom_frac = ct_norm.loc[k4_cl].max()
        k4_to_k2_dominant[k4_cl] = (dominant_k2, dom_frac)
        print(f"  K=4 cluster {k4_cl} → primarily K=2 cluster {dominant_k2} ({dom_frac:.1%})")

    # Early phenotype crosswalk
    frozen_path = processed_dir / "phenotype_labels_frozen_v1_mimic.parquet"
    early_crosswalk_lines = []
    if frozen_path.exists():
        frozen = pd.read_parquet(frozen_path).drop_duplicates(subset=["stay_id"])
        k4_df = pd.DataFrame({"stay_id": feat_df["stay_id"].values, "k4_cluster": k4_labels})
        merged_early = k4_df.merge(frozen[["stay_id", "rule_phenotype_2class"]], on="stay_id", how="left")
        ct_early = pd.crosstab(merged_early["k4_cluster"], merged_early["rule_phenotype_2class"])
        ct_early_norm = pd.crosstab(
            merged_early["k4_cluster"], merged_early["rule_phenotype_2class"],
            normalize="index",
        )
        early_crosswalk_lines.append("  K=4 × Early 2-class:")
        early_crosswalk_lines.append(ct_early.to_string().replace("\n", "\n    "))
        early_crosswalk_lines.append("\n  Row-normalized:")
        early_crosswalk_lines.append(ct_early_norm.round(3).to_string().replace("\n", "\n    "))
        for line in early_crosswalk_lines:
            print(line)

    # Visualization
    print("\n  Generating K=4 heatmap ...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not outcome_summary.empty:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Panel A: Cluster sizes + mortality
            ax = axes[0]
            clusters = outcome_summary["cluster"].astype(int)
            bars = ax.bar(clusters, outcome_summary["pct"], color="steelblue", alpha=0.7)
            ax.set_xlabel("K=4 Cluster")
            ax.set_ylabel("Proportion (%)")
            ax.set_title(f"K=4 Cluster Distribution\n[{EXPLORATORY_TAG}]")
            ax2 = ax.twinx()
            ax2.plot(clusters, outcome_summary["mortality_28d"], "ro-", linewidth=2)
            ax2.set_ylabel("28-day Mortality (%)", color="red")

            # Panel B: K=4 vs K=2 heatmap
            ax = axes[1]
            im = ax.imshow(ct_norm.values, cmap="Blues", aspect="auto")
            ax.set_xticks(range(len(ct_norm.columns)))
            ax.set_xticklabels([f"K2={c}" for c in ct_norm.columns])
            ax.set_yticks(range(len(ct_norm.index)))
            ax.set_yticklabels([f"K4={c}" for c in ct_norm.index])
            ax.set_title(f"K=4 → K=2 Mapping\n[{EXPLORATORY_TAG}]")
            for i in range(len(ct_norm.index)):
                for j in range(len(ct_norm.columns)):
                    val = ct_norm.values[i, j]
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=10, color="white" if val > 0.5 else "black")

            fig.tight_layout()
            heatmap_path = figures_dir / "96b_trajectory_k4_heatmap.png"
            fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {heatmap_path}")
    except ImportError:
        print("  matplotlib not available — skipping visualization")

    # ── [7/7] Save outputs ────────────────────────────────────────────
    print("\n[7/7] Saving outputs ...")

    # Cluster labels
    label_df = pd.DataFrame({
        "stay_id": feat_df["stay_id"].values,
        "traj_k4_cluster": k4_labels,
    })
    for k in K_COMPARE:
        label_df[f"traj_k{k}_cluster"] = labels_dict[f"kmeans_k{k}"]
    label_path = interim_dir / "96b_trajectory_k4_clusters_v1_mimic.parquet"
    label_df.to_parquet(label_path, index=False)
    print(f"  Saved: {label_path}")

    metrics_path = tables_dir / "96b_trajectory_k4_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  Saved: {metrics_path}")

    stability_path = tables_dir / "96b_trajectory_k4_stability.csv"
    stability_df.to_csv(stability_path, index=False)
    print(f"  Saved: {stability_path}")

    cards_path = tables_dir / "96b_trajectory_k4_phenotype_cards.csv"
    cards.to_csv(cards_path, index=False)
    print(f"  Saved: {cards_path}")

    if not outcome_summary.empty:
        outcome_path = tables_dir / "96b_trajectory_k4_outcome_summary.csv"
        outcome_summary.to_csv(outcome_path, index=False)
        print(f"  Saved: {outcome_path}")

    crosswalk_path = tables_dir / "96b_trajectory_k4_vs_k2_crosswalk.csv"
    ct_k4_k2.to_csv(crosswalk_path)
    print(f"  Saved: {crosswalk_path}")

    # ── Report ────────────────────────────────────────────────────────
    k4_verdict = "PASS (stable)" if k4_stable else "FAIL (unstable)"
    report_lines = [
        "=" * 72,
        f"96b — Exploratory K=4 Trajectory Phenotyping [{EXPLORATORY_TAG}]",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Tag: {EXPLORATORY_TAG}",
        "",
        "STATUS: EXPLORATORY — results are provisional and NOT part of the",
        "v3.1 core protocol. Do not cite in manuscript main body without",
        "explicit upgrade decision.",
        "",
        "CLUSTERING METRICS (K-means)",
        "────────────────────────────",
    ]
    for _, row in metrics_df[metrics_df["method"] == "kmeans"].iterrows():
        report_lines.append(
            f"  K={int(row['K'])}: sil={row['silhouette']:.4f} "
            f"CH={row['calinski_harabasz']:.0f} min_frac={row['min_cluster_frac']:.3f}"
        )

    report_lines.extend([
        "",
        "STABILITY BOOTSTRAP",
        "───────────────────",
    ])
    for _, row in stability_df.iterrows():
        verdict = "PASS" if row["stable"] else "FAIL"
        report_lines.append(
            f"  K={int(row['K'])}: ARI={row['ari_mean']:.4f}±{row['ari_std']:.4f} "
            f"(P5={row['ari_p5']:.4f}) → {verdict}"
        )

    report_lines.extend([
        "",
        f"K=4 STABILITY VERDICT: {k4_verdict}",
        "",
        "K=4 CLUSTER DISTRIBUTION",
        "────────────────────────",
    ])
    for cl_id in range(4):
        cnt = int(k4_counts[cl_id])
        pct = cnt / n_stays * 100
        report_lines.append(f"  Cluster {cl_id}: {cnt:,} ({pct:.1f}%)")

    if not outcome_summary.empty:
        report_lines.extend([
            "",
            "K=4 OUTCOME SUMMARY",
            "───────────────────",
        ])
        for _, row in outcome_summary.iterrows():
            report_lines.append(
                f"  Cluster {int(row['cluster'])}: n={int(row['n']):,} ({row['pct']:.1f}%), "
                f"mort_28d={row['mortality_28d']:.1f}%, mort_7d={row['mortality_7d']:.1f}%"
            )

    report_lines.extend([
        "",
        "K=4 → K=2 MAPPING",
        "──────────────────",
    ])
    for k4_cl, (k2_cl, frac) in k4_to_k2_dominant.items():
        report_lines.append(f"  K4 cluster {k4_cl} → K2 cluster {k2_cl} ({frac:.1%})")

    if early_crosswalk_lines:
        report_lines.extend([
            "",
            "K=4 × EARLY PHENOTYPE CROSSWALK",
            "───────────────────────────────",
        ] + early_crosswalk_lines)

    report_lines.extend([
        "",
        "INTERPRETATION",
        "──────────────",
        f"K=4 stability: {k4_verdict}",
        "If K=4 is stable and each K=2 cluster splits into ~2 K=4 sub-clusters,",
        "this supports the '2 roots → 4 branches' hierarchical trajectory hypothesis.",
        "K=4 clusters should NOT be used as StepC primary stratifier — they serve",
        "as Aim 3 explanatory enrichment only.",
        "",
        "OUTPUTS",
        "───────",
        "  data/interim/96b_trajectory_k4_clusters_v1_mimic.parquet",
        "  tables/96b_trajectory_k4_metrics.csv",
        "  tables/96b_trajectory_k4_stability.csv",
        "  tables/96b_trajectory_k4_phenotype_cards.csv",
        "  tables/96b_trajectory_k4_outcome_summary.csv",
        "  tables/96b_trajectory_k4_vs_k2_crosswalk.csv",
        "  figures/96b_trajectory_k4_heatmap.png",
        "",
        f"[{EXPLORATORY_TAG}] END",
    ])

    report = "\n".join(report_lines) + "\n"
    report_path = reports_dir / "96b_trajectory_k4_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {report_path}")
    print(report)
    print(f"[96b] Done. [{EXPLORATORY_TAG}]")


if __name__ == "__main__":
    main()
