#!/usr/bin/env python3
"""
60c_stability_bootstrap_early — Bootstrap stability for early phenotypes.

Purpose
-------
Assess clustering stability by re-running K-means on bootstrap resamples
and computing Adjusted Rand Index (ARI) and Normalized Mutual Information
(NMI) between bootstrap labels and the reference labels from 60b.
A stable phenotype should show high ARI/NMI across resamples.

Protocol reference: Step 60c (v3.1 § 12)

Inputs
------
- data/interim/60a_early_features_v1_mimic.parquet
- data/interim/60b_early_clusters_v1_mimic.parquet

Outputs
-------
- tables/60c_early_stability.csv
- reports/60c_stability_bootstrap_summary.txt

Compute: C3
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path

N_BOOTSTRAP = 100
SEED = 2024


def _select_clustering_features(df: pd.DataFrame) -> np.ndarray:
    """Same feature selection logic as 60b for consistency."""
    base_cols = sorted([c for c in df.columns if c.startswith("base_z_")])
    if len(base_cols) >= 16:
        return df[base_cols].values, base_cols
    dann_cols = sorted([c for c in df.columns if c.startswith("dann_z_")])
    if len(dann_cols) >= 16:
        return df[dann_cols].values, dann_cols
    raw_cols = sorted([c for c in df.columns
                       if c.endswith("_pre_mean") or c.endswith("_pre_last")])
    return df[raw_cols].values, raw_cols


def main() -> None:
    proj_cfg = load_config()
    project_root = Path(proj_cfg.get("project_root", "."))
    interim_dir = resolve_path(proj_cfg, "interim")
    reports_dir = resolve_path(proj_cfg, "reports")
    tables_dir = resolve_path(proj_cfg, "tables")
    reports_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("60c_stability_bootstrap_early — Phenotype stability assessment")
    print("=" * 72)

    # ── Load data ──────────────────────────────────────────────────────
    print("\n[1/3] Loading data ...")
    feat_df = pd.read_parquet(interim_dir / "60a_early_features_v1_mimic.parquet")
    cluster_df = pd.read_parquet(interim_dir / "60b_early_clusters_v1_mimic.parquet")
    cluster_df = cluster_df.drop_duplicates(subset=["stay_id"])
    n_stays = len(feat_df)
    print(f"  Stays: {n_stays:,}")

    # Identify selected K from 60b
    selected_col = [c for c in cluster_df.columns if c.startswith("selected_k")]
    if not selected_col:
        print("  ERROR: no 'selected_k*' column found in 60b output")
        sys.exit(1)
    selected_col = selected_col[0]
    best_k = int(selected_col.replace("selected_k", ""))

    aligned = feat_df[["stay_id"]].merge(cluster_df, on="stay_id", how="left")
    if aligned[selected_col].isna().any():
        n_missing = int(aligned[selected_col].isna().sum())
        print(f"  ERROR: {n_missing:,} stays missing selected cluster labels")
        sys.exit(1)

    ref_labels = aligned[selected_col].to_numpy()
    print(f"  Selected K from 60b: {best_k}")

    # Also test neighboring K values
    k_values = sorted(set([best_k] + [k for k in [best_k - 1, best_k + 1]
                                       if 2 <= k <= 6]))

    # ── Prepare features ───────────────────────────────────────────────
    print("\n[2/3] Preparing features ...")
    X_raw, feat_cols = _select_clustering_features(feat_df)
    nan_mask = np.isnan(X_raw)
    if nan_mask.sum() > 0:
        col_medians = np.nanmedian(X_raw, axis=0)
        for j in range(X_raw.shape[1]):
            X_raw[nan_mask[:, j], j] = col_medians[j]
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    print(f"  Feature matrix: {X.shape}")

    # ── Bootstrap stability ────────────────────────────────────────────
    print(f"\n[3/3] Running {N_BOOTSTRAP} bootstrap resamples ...")
    rng = np.random.RandomState(SEED)
    results = []

    for k in k_values:
        # Reference labels for this K
        ref_col = f"kmeans_k{k}"
        if ref_col in aligned.columns:
            ref_k = aligned[ref_col].to_numpy()
        else:
            km_ref = KMeans(n_clusters=k, n_init=10, random_state=SEED)
            ref_k = km_ref.fit_predict(X)

        aris = []
        nmis = []
        for b in range(N_BOOTSTRAP):
            idx = rng.choice(n_stays, size=n_stays, replace=True)
            X_boot = X[idx]

            km = KMeans(n_clusters=k, n_init=5, random_state=SEED + b, max_iter=200)
            boot_labels = km.fit_predict(X_boot)

            # Compute metrics on bootstrap samples
            ari = adjusted_rand_score(ref_k[idx], boot_labels)
            nmi = normalized_mutual_info_score(ref_k[idx], boot_labels)
            aris.append(ari)
            nmis.append(nmi)

        results.append({
            "K": k,
            "ari_mean": round(np.mean(aris), 4),
            "ari_std": round(np.std(aris), 4),
            "ari_p5": round(np.percentile(aris, 5), 4),
            "nmi_mean": round(np.mean(nmis), 4),
            "nmi_std": round(np.std(nmis), 4),
            "nmi_p5": round(np.percentile(nmis, 5), 4),
            "n_bootstrap": N_BOOTSTRAP,
        })
        print(f"  K={k}: ARI={np.mean(aris):.3f}±{np.std(aris):.3f}, "
              f"NMI={np.mean(nmis):.3f}±{np.std(nmis):.3f}")

    res_df = pd.DataFrame(results)

    # ── Stability verdict ──────────────────────────────────────────────
    best_row = res_df[res_df["K"] == best_k].iloc[0]
    stable = best_row["ari_mean"] >= 0.60 and best_row["ari_p5"] >= 0.40
    verdict = "PASS" if stable else "FAIL (consider fallback to rule-based 60d)"

    # ── Save outputs ───────────────────────────────────────────────────
    stab_path = tables_dir / "60c_early_stability.csv"
    res_df.to_csv(stab_path, index=False)
    print(f"\n  Saved: {stab_path}")

    report_lines = [
        "=" * 72,
        "60c — Early Phenotype Bootstrap Stability (MIMIC-IV)",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "PARAMETERS",
        "──────────",
        f"  Bootstrap resamples: {N_BOOTSTRAP}",
        f"  K values tested: {k_values}",
        f"  Selected K from 60b: {best_k}",
        f"  Feature dims: {X.shape[1]}",
        "",
        "STABILITY METRICS",
        "─────────────────",
    ]
    for _, row in res_df.iterrows():
        marker = " ← selected" if int(row["K"]) == best_k else ""
        report_lines.append(
            f"  K={int(row['K'])}: ARI={row['ari_mean']:.4f}±{row['ari_std']:.4f} "
            f"(P5={row['ari_p5']:.4f}), "
            f"NMI={row['nmi_mean']:.4f}±{row['nmi_std']:.4f}{marker}"
        )

    report_lines.extend([
        "",
        "STABILITY VERDICT",
        "─────────────────",
        f"  K={best_k}: ARI_mean={best_row['ari_mean']:.4f}, ARI_P5={best_row['ari_p5']:.4f}",
        f"  Threshold: ARI_mean >= 0.60 AND ARI_P5 >= 0.40",
        f"  Result: {verdict}",
        "",
        "NEXT STEPS",
        "──────────",
    ])
    if stable:
        report_lines.append("  60e: generate phenotype cards for clinical interpretation")
    else:
        report_lines.append("  60d: activate rule-based fallback phenotype")
        report_lines.append("  60e: generate phenotype cards for both data-driven and rule-based")

    report = "\n".join(report_lines) + "\n"
    report_path = reports_dir / "60c_stability_bootstrap_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {report_path}")
    print(report)
    print("[60c] Done.")


if __name__ == "__main__":
    main()
