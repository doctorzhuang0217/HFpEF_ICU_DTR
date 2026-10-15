#!/usr/bin/env python3
"""
65c_stability_bootstrap_trajectory — Bootstrap stability for trajectory phenotypes.

Purpose
-------
Assess clustering stability by re-running K-means on bootstrap resamples
and computing ARI and NMI between bootstrap labels and the reference
labels from 65b. Same methodology as 60c but applied to trajectory features.

Protocol reference: Step 65c (v3.1 § 12)

Inputs
------
- data/interim/65a_trajectory_features_v1_mimic.parquet
- data/interim/65b_trajectory_clusters_v1_mimic.parquet

Outputs
-------
- tables/65c_trajectory_stability.csv
- reports/65c_stability_bootstrap_trajectory_summary.txt

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


def _sorted_z_cols(columns: list[str]) -> list[str]:
    """Sort latent-derived columns by numeric z suffix."""
    def z_idx(col: str) -> int:
        return int(col.rsplit("_", 1)[1])

    return sorted(columns, key=z_idx)


def _select_trajectory_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Same feature selection logic as 65b for consistency."""
    def _z_sort_key(col: str) -> tuple:
        """Sort columns by z-dimension numerically (z_0 < z_2 < z_10)."""
        parts = col.split("_z_")
        if len(parts) == 2 and parts[1].isdigit():
            return (parts[0], int(parts[1]))
        return (col, 0)

    base_phase_cols = sorted(
        [c for c in df.columns if c.startswith("base_") and "_z_" in c],
        key=_z_sort_key,
    )
    if len(base_phase_cols) >= 16:
        return df[base_phase_cols].values, base_phase_cols

    dann_phase_cols = sorted(
        [c for c in df.columns if c.startswith("dann_") and "_z_" in c],
        key=_z_sort_key,
    )
    if len(dann_phase_cols) >= 16:
        return df[dann_phase_cols].values, dann_phase_cols

    raw_cols = sorted(
        [c for c in df.columns
         if any(c.endswith(s) for s in ["_mean", "_last", "_slope", "_std",
                                         "_delta", "_total", "_late_vs_early"])
         and c != "stay_id"]
    )
    return df[raw_cols].values, raw_cols


def main() -> None:
    proj_cfg = load_config()
    interim_dir = resolve_path(proj_cfg, "interim")
    reports_dir = resolve_path(proj_cfg, "reports")
    tables_dir = resolve_path(proj_cfg, "tables")
    reports_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("65c_stability_bootstrap_trajectory — Trajectory stability assessment")
    print("=" * 72)

    # ── Load data ─────────────────────────────────────────────────────
    print("\n[1/3] Loading data ...")
    feat_df = pd.read_parquet(
        interim_dir / "65a_trajectory_features_v1_mimic.parquet"
    )
    cluster_df = pd.read_parquet(
        interim_dir / "65b_trajectory_clusters_v1_mimic.parquet"
    )
    cluster_df = cluster_df.drop_duplicates(subset=["stay_id"])
    n_stays = len(feat_df)
    print(f"  Stays: {n_stays:,}")

    # Identify selected K from 65b
    selected_col = [
        c for c in cluster_df.columns if c.startswith("traj_selected_k")
    ]
    if not selected_col:
        print("  ERROR: no 'traj_selected_k*' column found in 65b output")
        sys.exit(1)
    selected_col = selected_col[0]
    best_k = int(selected_col.replace("traj_selected_k", ""))

    # Align via stay_id merge
    aligned = feat_df[["stay_id"]].merge(cluster_df, on="stay_id", how="left")
    if aligned[selected_col].isna().any():
        n_missing = int(aligned[selected_col].isna().sum())
        print(f"  ERROR: {n_missing:,} stays missing trajectory cluster labels")
        sys.exit(1)

    ref_labels = aligned[selected_col].to_numpy()
    print(f"  Selected K from 65b: {best_k}")

    k_values = sorted(
        set([best_k] + [k for k in [best_k - 1, best_k + 1] if 2 <= k <= 6])
    )

    # ── Prepare features ─────────────────────────────────────────────
    print("\n[2/3] Preparing features ...")
    X_raw, feat_cols = _select_trajectory_features(feat_df)
    nan_mask = np.isnan(X_raw)
    if nan_mask.sum() > 0:
        col_medians = np.nanmedian(X_raw, axis=0)
        for j in range(X_raw.shape[1]):
            X_raw[nan_mask[:, j], j] = col_medians[j]
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    print(f"  Feature matrix: {X.shape}")

    # ── Bootstrap stability ──────────────────────────────────────────
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

            km = KMeans(
                n_clusters=k, n_init=5, random_state=SEED + b, max_iter=200
            )
            boot_labels = km.fit_predict(X_boot)

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
        print(
            f"  K={k}: ARI={np.mean(aris):.3f}±{np.std(aris):.3f}, "
            f"NMI={np.mean(nmis):.3f}±{np.std(nmis):.3f}"
        )

    res_df = pd.DataFrame(results)

    # ── Stability verdict ────────────────────────────────────────────
    best_row = res_df[res_df["K"] == best_k].iloc[0]
    stable = best_row["ari_mean"] >= 0.60 and best_row["ari_p5"] >= 0.40
    verdict = "PASS" if stable else "FAIL (trajectory clusters unstable)"

    # ── Save outputs ─────────────────────────────────────────────────
    stab_path = tables_dir / "65c_trajectory_stability.csv"
    res_df.to_csv(stab_path, index=False)
    print(f"\n  Saved: {stab_path}")

    report_lines = [
        "=" * 72,
        "65c — Trajectory Phenotype Bootstrap Stability (MIMIC-IV)",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "PARAMETERS",
        "──────────",
        f"  Bootstrap resamples: {N_BOOTSTRAP}",
        f"  K values tested: {k_values}",
        f"  Selected K from 65b: {best_k}",
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
        f"  K={best_k}: ARI_mean={best_row['ari_mean']:.4f}, "
        f"ARI_P5={best_row['ari_p5']:.4f}",
        f"  Threshold: ARI_mean >= 0.60 AND ARI_P5 >= 0.40",
        f"  Result: {verdict}",
        "",
        "NOTE",
        "────",
        "  Trajectory phenotype is EXPLANATORY ONLY (protocol § 12).",
        "  Stability failure does NOT trigger a fallback cascade —",
        "  trajectory labels are used for enrichment analysis, not",
        "  for primary causal estimation stratification.",
        "",
        "NEXT STEPS",
        "──────────",
        "  65d: trajectory phenotype cards",
        "  65e: freeze all phenotype labels for Step C",
    ])

    report = "\n".join(report_lines) + "\n"
    report_path = reports_dir / "65c_stability_bootstrap_trajectory_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {report_path}")
    print(report)
    print("[65c] Done.")


if __name__ == "__main__":
    main()
