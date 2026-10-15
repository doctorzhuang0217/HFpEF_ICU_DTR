#!/usr/bin/env python3
"""
60b_cluster_early_baseline — Cluster early phenotype features.

Purpose
-------
Run K-means / Gaussian mixture clustering on early phenotype features
for candidate K values (2-6). Evaluate each K by silhouette score,
Calinski-Harabasz index, and clinical interpretability metrics.
Select best K for downstream phenotype-stratified analysis.

Input features are the baseline encoder pre-T0 latents (base_z_0..63)
because they provide a compact, denoised representation of the 0-6h
window. Raw clinical features are used for phenotype interpretation
only (60e phenotype cards).

Protocol reference: Step 60b (v3.1 § 12)

Inputs
------
- data/interim/60a_early_features_v1_mimic.parquet

Outputs
-------
- data/interim/60b_early_clusters_v1_mimic.parquet  (stay_id + labels per K)
- tables/60b_cluster_metrics.csv
- reports/60b_cluster_early_summary.txt

Compute: C2-C3
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path

K_CANDIDATES = [2, 3, 4, 5, 6]
SEED = 2024
N_INIT = 10


def _select_clustering_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Select features for clustering — prefer encoder latents, fall back to raw."""
    # Prefer baseline encoder latents
    base_cols = sorted([c for c in df.columns if c.startswith("base_z_")])
    if len(base_cols) >= 16:
        print(f"  Using baseline encoder latents: {len(base_cols)} dims")
        X = df[base_cols].values
        return X, base_cols

    # Fall back to DANN latents
    dann_cols = sorted([c for c in df.columns if c.startswith("dann_z_")])
    if len(dann_cols) >= 16:
        print(f"  Using DANN encoder latents: {len(dann_cols)} dims")
        X = df[dann_cols].values
        return X, dann_cols

    # Fall back to raw clinical summaries
    raw_cols = sorted([c for c in df.columns
                       if c.endswith("_pre_mean") or c.endswith("_pre_last")])
    print(f"  Fallback to raw clinical features: {len(raw_cols)} dims")
    X = df[raw_cols].values
    return X, raw_cols


def main() -> None:
    proj_cfg = load_config()
    project_root = Path(proj_cfg.get("project_root", "."))
    interim_dir = resolve_path(proj_cfg, "interim")
    reports_dir = resolve_path(proj_cfg, "reports")
    tables_dir = resolve_path(proj_cfg, "tables")
    interim_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("60b_cluster_early_baseline — Early phenotype clustering")
    print("=" * 72)

    # ── Load features ──────────────────────────────────────────────────
    print("\n[1/4] Loading early features ...")
    df = pd.read_parquet(interim_dir / "60a_early_features_v1_mimic.parquet")
    n_stays = len(df)
    print(f"  Stays: {n_stays:,}")

    # ── Select and prep features ───────────────────────────────────────
    print("\n[2/4] Selecting clustering features ...")
    X_raw, feat_cols = _select_clustering_features(df)

    # Handle NaNs: impute with column median
    nan_mask = np.isnan(X_raw)
    n_nan = nan_mask.sum()
    if n_nan > 0:
        print(f"  Imputing {n_nan:,} NaN values with column medians")
        col_medians = np.nanmedian(X_raw, axis=0)
        for j in range(X_raw.shape[1]):
            X_raw[nan_mask[:, j], j] = col_medians[j]

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    print(f"  Feature matrix: {X.shape}")

    # ── Clustering sweep ───────────────────────────────────────────────
    print("\n[3/4] Running K-means + GMM sweep ...")
    results = []
    labels_dict = {}

    for k in K_CANDIDATES:
        # K-means
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED, max_iter=300)
        km_labels = km.fit_predict(X)
        km_sil = silhouette_score(X, km_labels, sample_size=min(10000, n_stays),
                                  random_state=SEED)
        km_ch = calinski_harabasz_score(X, km_labels)
        km_inertia = km.inertia_

        # Cluster sizes
        _, km_counts = np.unique(km_labels, return_counts=True)
        km_min_frac = km_counts.min() / n_stays

        results.append({
            "method": "kmeans", "K": k,
            "silhouette": round(km_sil, 4),
            "calinski_harabasz": round(km_ch, 1),
            "inertia": round(km_inertia, 1),
            "min_cluster_frac": round(km_min_frac, 4),
        })
        labels_dict[f"kmeans_k{k}"] = km_labels

        # GMM
        gmm = GaussianMixture(n_components=k, n_init=N_INIT, random_state=SEED,
                               covariance_type="full", max_iter=200)
        gmm_labels = gmm.fit_predict(X)
        gmm_sil = silhouette_score(X, gmm_labels, sample_size=min(10000, n_stays),
                                   random_state=SEED)
        gmm_ch = calinski_harabasz_score(X, gmm_labels)
        gmm_bic = gmm.bic(X)
        gmm_aic = gmm.aic(X)

        _, gmm_counts = np.unique(gmm_labels, return_counts=True)
        gmm_min_frac = gmm_counts.min() / n_stays

        results.append({
            "method": "gmm", "K": k,
            "silhouette": round(gmm_sil, 4),
            "calinski_harabasz": round(gmm_ch, 1),
            "bic": round(gmm_bic, 1),
            "aic": round(gmm_aic, 1),
            "min_cluster_frac": round(gmm_min_frac, 4),
        })
        labels_dict[f"gmm_k{k}"] = gmm_labels

        print(f"  K={k}: kmeans sil={km_sil:.3f} CH={km_ch:.0f} | "
              f"gmm sil={gmm_sil:.3f} BIC={gmm_bic:.0f}")

    # ── Select best K ──────────────────────────────────────────────────
    print("\n[4/4] Selecting best K ...")
    res_df = pd.DataFrame(results)

    # Heuristic: best K = highest silhouette among K-means, with min_cluster >= 5%
    km_res = res_df[res_df["method"] == "kmeans"].copy()
    viable = km_res[km_res["min_cluster_frac"] >= 0.05]
    if len(viable) > 0:
        best_row = viable.loc[viable["silhouette"].idxmax()]
    else:
        best_row = km_res.loc[km_res["silhouette"].idxmax()]
    best_k = int(best_row["K"])
    best_method = "kmeans"
    best_sil = best_row["silhouette"]
    print(f"  Selected: K={best_k} ({best_method}, silhouette={best_sil:.4f})")

    # ── Save outputs ───────────────────────────────────────────────────
    label_df = pd.DataFrame({"stay_id": df["stay_id"]})
    for col_name, lab in labels_dict.items():
        label_df[col_name] = lab
    label_df[f"selected_k{best_k}"] = labels_dict[f"kmeans_k{best_k}"]

    out_path = interim_dir / "60b_early_clusters_v1_mimic.parquet"
    label_df.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"  Saved: {out_path}")

    metrics_path = tables_dir / "60b_cluster_metrics.csv"
    res_df.to_csv(metrics_path, index=False)
    print(f"  Saved: {metrics_path}")

    # ── Summary report ─────────────────────────────────────────────────
    report_lines = [
        "=" * 72,
        "60b — Early Phenotype Clustering (MIMIC-IV)",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "DATA",
        "────",
        f"  Stays: {n_stays:,}",
        f"  Feature dims: {X.shape[1]}",
        f"  Feature source: {feat_cols[0].split('_')[0]} "
        f"({'encoder latents' if 'z_' in feat_cols[0] else 'raw clinical'})",
        f"  NaN imputed: {n_nan:,}",
        "",
        "CLUSTERING RESULTS",
        "──────────────────",
    ]

    for _, row in res_df.iterrows():
        extra = ""
        if "bic" in row and pd.notna(row.get("bic")):
            extra = f" BIC={row['bic']:.0f}"
        report_lines.append(
            f"  {row['method']:6s} K={int(row['K'])} | sil={row['silhouette']:.4f} "
            f"CH={row['calinski_harabasz']:.0f} min_frac={row['min_cluster_frac']:.3f}"
            f"{extra}"
        )

    report_lines.extend([
        "",
        "SELECTED MODEL",
        "──────────────",
        f"  Method: {best_method}",
        f"  K: {best_k}",
        f"  Silhouette: {best_sil:.4f}",
        f"  Label column: selected_k{best_k}",
        "",
        "CLUSTER SIZES (selected)",
        "────────────────────────",
    ])
    selected_labels = labels_dict[f"kmeans_k{best_k}"]
    for c in range(best_k):
        n_c = (selected_labels == c).sum()
        report_lines.append(f"  Cluster {c}: {n_c:,} ({n_c / n_stays * 100:.1f}%)")

    report_lines.extend([
        "",
        "NEXT STEPS",
        "──────────",
        "  60c: bootstrap stability assessment (ARI/NMI)",
        "  60e: phenotype cards for clinical interpretation",
    ])

    report = "\n".join(report_lines) + "\n"
    report_path = reports_dir / "60b_cluster_early_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n  Saved: {report_path}")
    print(report)
    print("[60b] Done.")


if __name__ == "__main__":
    main()
