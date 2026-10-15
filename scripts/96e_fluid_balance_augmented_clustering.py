#!/usr/bin/env python3
"""
96e_fluid_balance_augmented_clustering — Re-run trajectory clustering with
cumulative fluid balance features included in the feature matrix.

[EXPLORATORY_96X — NOT part of v3.1 core protocol. Results are provisional
and should NOT be cited in the manuscript main body without explicit upgrade.]

Motivation
----------
Steps 65b / 96b clustered trajectories using phase-pooled encoder latent
features (64 dims), which do not include fluid balance information. The K=4
analysis (96d) showed no HFpEF enrichment differential — all clusters had
~6.7% HFpEF. The hypothesis here is that HFpEF's distinct hemodynamic
substrate manifests primarily in the filling-pressure / volume-response axis,
which the standard latent features do not capture.

Step 65a already extracted two fluid balance features from the state tensors:
  - cum_fluid_balance_final: cumulative fluid balance at end of 24h window
  - cum_fluid_balance_max: maximum cumulative fluid balance during 24h

This script takes the SAME latent trajectory features used by 96b, APPENDS
the fluid balance features, and re-runs K-means/GMM clustering to test
whether the additional fluid-volume dimension reveals HFpEF-specific clusters.

Additionally, we compute a richer set of fluid-balance trajectory features
from the raw state tensors (45c) to augment the feature space:
  - Per-phase (early/mid/late) cumulative fluid balance means
  - Fluid balance slope (early → late)
  - Fluid input vs urine output ratio
  - Diuretic response index (urine output / diuretic dose)

Protocol reference: Exploratory extension (post-v3.1, Step 96 series)

Inputs
------
- data/interim/65a_trajectory_features_v1_mimic.parquet
- data/interim/45c_state_tensors_v1_mimic.parquet
- data/interim/45d_outcomes_absorb_v1_mimic.parquet
- data/processed/phenotype_labels_frozen_v1_mimic.parquet
- configs/45c_normalization_params_v1.yaml

Outputs
-------
- data/interim/96e_fluid_augmented_clusters_v1_mimic.parquet
- tables/96e_fluid_augmented_cluster_summary.csv
- tables/96e_fluid_augmented_hf_enrichment.csv
- tables/96e_fluid_augmented_stability.csv
- tables/96e_fluid_features_summary.csv
- figures/96e_fluid_augmented_cluster_trajectories.png
- figures/96e_fluid_balance_by_cluster.png
- reports/96e_fluid_augmented_summary.txt

Compute: C2-C3

[EXPLORATORY_96X]
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path

EXPLORATORY_TAG = "EXPLORATORY_96X"

# ── Configuration ────────────────────────────────────────────────────────────
SEED = 2024
K_CANDIDATES = [2, 3, 4, 5, 6]
N_INIT = 20
N_BOOTSTRAP = 100
ARI_PASS = 0.60
ARI_P5_PASS = 0.40
MIN_CLUSTER_FRAC = 0.03  # reject if any cluster < 3%

# Phase definitions (matching 65a)
N_BINS = 12
BIN_HOURS = 2
EARLY_BINS = [0, 1, 2, 3]      # 0-8h
MID_BINS = [4, 5, 6, 7]        # 8-16h
LATE_BINS = [8, 9, 10, 11]     # 16-24h


# ── Feature extraction ───────────────────────────────────────────────────────
def build_fluid_trajectory_features(state_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build rich fluid-balance trajectory features from raw 45c state tensors.

    For each stay, compute:
      - cum_fb_early / mid / late: mean cumulative fluid balance per phase
      - cum_fb_slope: linear slope of cumulative FB over 24h
      - cum_fb_final / max: endpoint and peak values
      - fluid_input_total: total fluid input over 24h
      - urine_total: total urine output over 24h
      - net_io_ratio: fluid_input_total / (urine_total + 1) — higher = more retention
      - diuretic_total: total diuretic dose over 24h
      - diuretic_response: urine_total / (diuretic_total + 0.01) — higher = better response
      - fb_late_vs_early: cum_fb_late - cum_fb_early — fluid accumulation trajectory
    """
    print("  Building fluid trajectory features from 45c state tensors ...")

    # Columns we need
    fb_col = "cumulative_fluid_balance_L"
    fi_col = "iv_fluid_ml"
    uo_col = "urine_output_ml"
    di_col = "diuretic_furo_eq_mg"
    fb_ml_col = "fluid_balance_ml"

    # Check availability
    available = {}
    for col_name, col in [("cum_fb", fb_col), ("fluid_input", fi_col),
                           ("urine", uo_col), ("diuretic", di_col),
                           ("fb_ml", fb_ml_col)]:
        if col in state_df.columns:
            available[col_name] = col
            valid_pct = state_df[col].notna().mean() * 100
            print(f"    {col}: {valid_pct:.1f}% non-null")
        else:
            print(f"    {col}: NOT FOUND in state tensors")

    stay_ids = state_df["stay_id"].unique()
    rows = []

    for sid in tqdm(stay_ids, desc="  Computing fluid features", mininterval=5):
        grp = state_df[state_df["stay_id"] == sid].sort_values("bin")
        feat = {"stay_id": sid}

        # ── Cumulative fluid balance features ───────────────────────────
        if "cum_fb" in available:
            fb_vals = grp[available["cum_fb"]].values
            valid = ~np.isnan(fb_vals)

            # Per-phase means
            for phase_name, bins in [("early", EARLY_BINS), ("mid", MID_BINS),
                                      ("late", LATE_BINS)]:
                phase_mask = np.isin(grp["bin"].values, bins) & valid
                if phase_mask.any():
                    feat[f"cum_fb_{phase_name}"] = float(np.nanmean(fb_vals[phase_mask]))
                else:
                    feat[f"cum_fb_{phase_name}"] = np.nan

            # Slope: linear fit over bins with valid data
            valid_bins = grp["bin"].values[valid]
            valid_fb = fb_vals[valid]
            if len(valid_fb) >= 2:
                slope = np.polyfit(valid_bins, valid_fb, 1)[0]
                feat["cum_fb_slope"] = float(slope)
            else:
                feat["cum_fb_slope"] = np.nan

            # Final and max
            if valid.any():
                feat["cum_fb_final"] = float(valid_fb[-1]) if len(valid_fb) > 0 else np.nan
                feat["cum_fb_max"] = float(np.nanmax(fb_vals))
            else:
                feat["cum_fb_final"] = np.nan
                feat["cum_fb_max"] = np.nan

            # Late vs early delta
            if not np.isnan(feat.get("cum_fb_late", np.nan)) and not np.isnan(feat.get("cum_fb_early", np.nan)):
                feat["fb_late_vs_early"] = feat["cum_fb_late"] - feat["cum_fb_early"]
            else:
                feat["fb_late_vs_early"] = np.nan
        else:
            # If no cumulative FB, try to build from per-bin fluid balance
            if "fb_ml" in available:
                fb_ml = grp[available["fb_ml"]].values
                cum_fb = np.nancumsum(np.where(np.isnan(fb_ml), 0, fb_ml)) / 1000  # mL → L
                for phase_name, bins in [("early", EARLY_BINS), ("mid", MID_BINS),
                                          ("late", LATE_BINS)]:
                    phase_mask = np.isin(grp["bin"].values, bins)
                    if phase_mask.any():
                        feat[f"cum_fb_{phase_name}"] = float(np.mean(cum_fb[phase_mask]))
                    else:
                        feat[f"cum_fb_{phase_name}"] = np.nan
                feat["cum_fb_final"] = float(cum_fb[-1]) if len(cum_fb) > 0 else np.nan
                feat["cum_fb_max"] = float(np.nanmax(cum_fb)) if len(cum_fb) > 0 else np.nan
                if len(cum_fb) >= 2:
                    bins_arr = grp["bin"].values
                    slope = np.polyfit(bins_arr, cum_fb, 1)[0]
                    feat["cum_fb_slope"] = float(slope)
                else:
                    feat["cum_fb_slope"] = np.nan
                if not np.isnan(feat.get("cum_fb_late", np.nan)) and not np.isnan(feat.get("cum_fb_early", np.nan)):
                    feat["fb_late_vs_early"] = feat["cum_fb_late"] - feat["cum_fb_early"]
                else:
                    feat["fb_late_vs_early"] = np.nan

        # ── Fluid input total ───────────────────────────────────────────
        if "fluid_input" in available:
            fi = grp[available["fluid_input"]].values
            feat["fluid_input_total"] = float(np.nansum(fi))
        else:
            feat["fluid_input_total"] = np.nan

        # ── Urine output total ──────────────────────────────────────────
        if "urine" in available:
            uo = grp[available["urine"]].values
            feat["urine_total"] = float(np.nansum(uo))
        else:
            feat["urine_total"] = np.nan

        # ── Net I/O ratio ───────────────────────────────────────────────
        if not np.isnan(feat.get("fluid_input_total", np.nan)) and not np.isnan(feat.get("urine_total", np.nan)):
            feat["net_io_ratio"] = feat["fluid_input_total"] / (feat["urine_total"] + 1.0)
        else:
            feat["net_io_ratio"] = np.nan

        # ── Diuretic features ───────────────────────────────────────────
        if "diuretic" in available:
            di = grp[available["diuretic"]].values
            feat["diuretic_total"] = float(np.nansum(di))
            if not np.isnan(feat.get("urine_total", np.nan)):
                feat["diuretic_response"] = feat["urine_total"] / (feat["diuretic_total"] + 0.01)
            else:
                feat["diuretic_response"] = np.nan
        else:
            feat["diuretic_total"] = np.nan
            feat["diuretic_response"] = np.nan

        rows.append(feat)

    fb_df = pd.DataFrame(rows)
    print(f"  Built {len(fb_df)} stays × {len(fb_df.columns) - 1} fluid features")

    # Summary of each feature
    feat_cols = [c for c in fb_df.columns if c != "stay_id"]
    for c in feat_cols:
        valid_pct = fb_df[c].notna().mean() * 100
        print(f"    {c}: valid={valid_pct:.1f}%, "
              f"mean={fb_df[c].mean():.3f}, std={fb_df[c].std():.3f}")

    return fb_df


def select_trajectory_features_with_fluid(
    feat_df: pd.DataFrame, fluid_df: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """
    Merge 65a trajectory features with fluid balance features.
    Use same hierarchy as 65b/96b for base features, then APPEND fluid features.
    """
    # --- Base features (same as 65b _select_trajectory_features) ---
    def _sorted_z_cols(columns):
        def z_idx(col):
            parts = col.replace("base_", "").replace("dann_", "").split("_z_")
            if len(parts) == 2:
                return int(parts[1])
            return int(col.rsplit("_", 1)[1])
        return sorted(columns, key=z_idx)

    base_phase_cols = _sorted_z_cols(
        [c for c in feat_df.columns if c.startswith("base_") and "_z_" in c]
    )

    if len(base_phase_cols) >= 16:
        base_cols = base_phase_cols
        print(f"  Base features: {len(base_cols)} encoder latent dims")
    else:
        dann_phase_cols = _sorted_z_cols(
            [c for c in feat_df.columns if c.startswith("dann_") and "_z_" in c]
        )
        if len(dann_phase_cols) >= 16:
            base_cols = dann_phase_cols
            print(f"  Base features: {len(base_cols)} DANN latent dims")
        else:
            base_cols = sorted(
                [c for c in feat_df.columns
                 if any(c.endswith(s) for s in ["_mean", "_last", "_slope", "_std",
                                                 "_delta", "_total", "_late_vs_early"])
                 and c != "stay_id"]
            )
            print(f"  Base features: {len(base_cols)} raw trajectory dims")

    # --- Fluid features ---
    fluid_cols = [c for c in fluid_df.columns if c != "stay_id"]
    print(f"  Fluid features: {len(fluid_cols)} dims")
    print(f"    Columns: {fluid_cols}")

    # Merge
    merged = feat_df[["stay_id"] + base_cols].merge(
        fluid_df, on="stay_id", how="inner",
    )
    all_feat_cols = base_cols + fluid_cols
    print(f"  Combined: {len(all_feat_cols)} total features, {len(merged)} stays")

    return merged[all_feat_cols].values, all_feat_cols, merged["stay_id"].values


def impute_and_scale(X: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    """Impute NaN with column median, then z-score scale."""
    for j in range(X.shape[1]):
        col = X[:, j]
        mask = np.isnan(col)
        if mask.any():
            median_val = np.nanmedian(col) if not mask.all() else 0.0
            col[mask] = median_val

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


# ── Clustering & stability ───────────────────────────────────────────────────
def run_clustering_sweep(
    X: np.ndarray,
    k_candidates: list[int],
    seed: int,
    n_init: int,
) -> dict:
    """Run K-means for each K, return metrics."""
    results = {}
    for k in k_candidates:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=seed, max_iter=300)
        labels = km.fit_predict(X)

        # Cluster sizes
        _, counts = np.unique(labels, return_counts=True)
        min_frac = counts.min() / len(labels)

        # Silhouette (sample if large)
        n_sil = min(10000, len(X))
        idx = np.random.RandomState(seed).choice(len(X), n_sil, replace=False)
        sil = silhouette_score(X[idx], labels[idx])

        results[k] = {
            "labels": labels,
            "inertia": km.inertia_,
            "silhouette": sil,
            "min_cluster_frac": min_frac,
            "cluster_sizes": dict(zip(*np.unique(labels, return_counts=True))),
        }
        print(f"  K={k}: sil={sil:.3f}, min_frac={min_frac:.3f}, inertia={km.inertia_:.0f}")

    return results


def bootstrap_stability(X: np.ndarray, ref_labels: np.ndarray, K: int,
                         n_resamples: int, seed: int) -> dict:
    """Bootstrap ARI stability test."""
    rng = np.random.RandomState(seed)
    aris, nmis = [], []
    for _ in tqdm(range(n_resamples), desc=f"  Bootstrap K={K}", mininterval=2):
        idx = rng.choice(len(X), size=len(X), replace=True)
        km = KMeans(n_clusters=K, n_init=5, random_state=rng.randint(1e9), max_iter=200)
        km.fit(X[idx])
        boot_labels = km.predict(X)
        aris.append(adjusted_rand_score(ref_labels, boot_labels))
        nmis.append(normalized_mutual_info_score(ref_labels, boot_labels))

    aris = np.array(aris)
    nmis = np.array(nmis)
    return {
        "ari_mean": float(np.mean(aris)),
        "ari_std": float(np.std(aris)),
        "ari_p5": float(np.percentile(aris, 5)),
        "ari_p95": float(np.percentile(aris, 95)),
        "nmi_mean": float(np.mean(nmis)),
        "pass": bool(np.mean(aris) >= ARI_PASS and np.percentile(aris, 5) >= ARI_P5_PASS),
    }


# ── Analysis & plotting ─────────────────────────────────────────────────────
def compute_outcome_summary(
    cluster_labels: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    pheno_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-cluster mortality and HF enrichment."""
    # Determine shock column name (varies across frozen label versions)
    shock_col = None
    for candidate in ["rule_phenotype_2class", "dominant_syndrome"]:
        if candidate in pheno_df.columns:
            shock_col = candidate
            break

    pheno_cols = ["stay_id", "hf_subtype"]
    if shock_col:
        pheno_cols.append(shock_col)

    merged = cluster_labels.merge(
        outcomes_df[["stay_id", "mortality_28d"]].drop_duplicates(),
        on="stay_id", how="inner",
    ).merge(
        pheno_df[pheno_cols].drop_duplicates(),
        on="stay_id", how="inner",
    )

    # Mortality + phenotype distribution
    outcome_rows = []
    for k in sorted(merged["cluster"].unique()):
        ck = merged[merged["cluster"] == k]
        n = len(ck)
        mort = ck["mortality_28d"].mean()
        if shock_col:
            shock_frac = (ck[shock_col] == "shock_predominant").mean()
        else:
            shock_frac = np.nan
        outcome_rows.append({
            "cluster": k, "n": n, "mortality_28d": mort,
            "shock_frac": shock_frac,
        })
    outcome_df = pd.DataFrame(outcome_rows)

    # HF enrichment
    hf_ct = pd.crosstab(merged["cluster"], merged["hf_subtype"], normalize="index")

    return outcome_df, hf_ct


def plot_cluster_trajectories(
    state_df: pd.DataFrame,
    cluster_labels: pd.DataFrame,
    norm_params: dict,
    out_path: Path,
    K: int,
):
    """Plot aggregate trajectories per cluster."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    merged = state_df.merge(cluster_labels[["stay_id", "cluster"]], on="stay_id", how="inner")

    plot_vars = ["map", "heart_rate", "lactate", "spo2", "creatinine"]
    fig, axes = plt.subplots(len(plot_vars), 1, figsize=(12, 3 * len(plot_vars)), sharex=True)
    hours = np.arange(N_BINS) * BIN_HOURS
    colors = plt.cm.Set1(np.linspace(0, 1, max(K, 3)))

    for ax, var in zip(axes, plot_vars):
        z_col = f"{var}_z"
        mask_col = f"{var}_mask"
        if z_col not in merged.columns:
            continue
        for k in range(K):
            cdata = merged[merged["cluster"] == k]
            means, sems = [], []
            for b in range(N_BINS):
                bdata = cdata[cdata["bin"] == b]
                if mask_col in merged.columns:
                    bdata = bdata[bdata[mask_col] > 0]
                vals = bdata[z_col].dropna()
                if len(vals) == 0:
                    means.append(np.nan)
                    sems.append(np.nan)
                    continue
                p = norm_params.get(var, None)
                if p:
                    means.append(vals.mean() * p["std"] + p["mean"])
                    sems.append(vals.std() / np.sqrt(len(vals)) * p["std"])
                else:
                    means.append(vals.mean())
                    sems.append(vals.std() / np.sqrt(len(vals)))
            means = np.array(means)
            sems = np.array(sems)
            n_k = len(cdata["stay_id"].unique())
            ax.plot(hours, means, color=colors[k], marker="o", ms=4, label=f"C{k} (n={n_k})")
            ax.fill_between(hours, means - sems, means + sems, alpha=0.15, color=colors[k])
        ax.set_ylabel(var)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Hours post-T0")
    fig.suptitle(f"96e Fluid-Augmented K={K} Cluster Trajectories [{EXPLORATORY_TAG}]", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_fluid_balance_by_cluster(
    state_df: pd.DataFrame,
    cluster_labels: pd.DataFrame,
    out_path: Path,
    K: int,
):
    """Plot cumulative fluid balance trajectories per cluster."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fb_col = "cumulative_fluid_balance_L"
    if fb_col not in state_df.columns:
        print(f"  WARNING: {fb_col} not in state tensors — skipping fluid balance plot")
        return

    merged = state_df.merge(cluster_labels[["stay_id", "cluster"]], on="stay_id", how="inner")
    hours = np.arange(N_BINS) * BIN_HOURS
    colors = plt.cm.Set1(np.linspace(0, 1, max(K, 3)))

    fig, ax = plt.subplots(figsize=(10, 5))

    for k in range(K):
        cdata = merged[merged["cluster"] == k]
        means, sems = [], []
        for b in range(N_BINS):
            bdata = cdata[cdata["bin"] == b]
            vals = bdata[fb_col].dropna()
            if len(vals) == 0:
                means.append(np.nan)
                sems.append(np.nan)
            else:
                means.append(vals.mean())
                sems.append(vals.std() / np.sqrt(len(vals)))
        means = np.array(means)
        sems = np.array(sems)
        n_k = len(cdata["stay_id"].unique())
        ax.plot(hours, means, color=colors[k], marker="o", ms=5, linewidth=2,
                label=f"C{k} (n={n_k})")
        ax.fill_between(hours, means - sems, means + sems, alpha=0.15, color=colors[k])

    ax.set_xlabel("Hours post-T0", fontsize=12)
    ax.set_ylabel("Cumulative Fluid Balance (L)", fontsize=12)
    ax.set_title(f"96e Cumulative Fluid Balance by Cluster (K={K}) [{EXPLORATORY_TAG}]",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print(f"96e — Fluid Balance Augmented Trajectory Clustering [{EXPLORATORY_TAG}]")
    print("=" * 72)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Started: {ts}\n")

    np.random.seed(SEED)

    proj_cfg = load_config()
    ROOT = Path(proj_cfg["_project_root"])
    FIGURES = resolve_path(proj_cfg, "figures")
    TABLES = resolve_path(proj_cfg, "tables")
    REPORTS = resolve_path(proj_cfg, "reports")
    interim_dir = resolve_path(proj_cfg, "interim")
    processed_dir = resolve_path(proj_cfg, "processed")
    for d in [FIGURES, TABLES, REPORTS]:
        d.mkdir(parents=True, exist_ok=True)

    # ── [1/8] Load data ──────────────────────────────────────────────────
    print("[1/8] Loading data ...")

    feat_path = interim_dir / "65a_trajectory_features_v1_mimic.parquet"
    state_path = interim_dir / "45c_state_tensors_v1_mimic.parquet"
    outcomes_path = interim_dir / "45d_outcomes_absorb_v1_mimic.parquet"
    pheno_path = processed_dir / "phenotype_labels_frozen_v1_mimic.parquet"
    norm_path = ROOT / "configs" / "45c_normalization_params_v1.yaml"

    feat_df = pd.read_parquet(feat_path)
    feat_df = feat_df.drop_duplicates(subset=["stay_id"])
    state_df = pd.read_parquet(state_path)
    outcomes_df = pd.read_parquet(outcomes_path)
    pheno_df = pd.read_parquet(pheno_path)

    with open(norm_path) as f:
        norm_params = yaml.safe_load(f)["params"]

    print(f"  65a features: {len(feat_df)} stays")
    print(f"  State tensors: {len(state_df)} rows ({len(state_df['stay_id'].unique())} stays)")

    # ── [2/8] Build fluid balance trajectory features ────────────────────
    print("\n[2/8] Building fluid balance trajectory features ...")
    fluid_df = build_fluid_trajectory_features(state_df)

    # Save fluid features summary
    fluid_cols = [c for c in fluid_df.columns if c != "stay_id"]
    fluid_summary = []
    for c in fluid_cols:
        fluid_summary.append({
            "feature": c,
            "valid_pct": fluid_df[c].notna().mean() * 100,
            "mean": fluid_df[c].mean(),
            "std": fluid_df[c].std(),
            "median": fluid_df[c].median(),
            "p5": fluid_df[c].quantile(0.05),
            "p95": fluid_df[c].quantile(0.95),
        })
    fluid_summary_df = pd.DataFrame(fluid_summary)
    fluid_summary_df.to_csv(TABLES / "96e_fluid_features_summary.csv", index=False)
    print(f"  Saved fluid feature summary: {len(fluid_summary_df)} features")

    # ── [3/8] Merge and prepare feature matrix ───────────────────────────
    print("\n[3/8] Preparing augmented feature matrix ...")
    X_raw, feat_cols, stay_ids = select_trajectory_features_with_fluid(feat_df, fluid_df)
    X, scaler = impute_and_scale(X_raw.copy())
    print(f"  Final feature matrix: {X.shape} (stays × features)")

    # Also prepare base-only matrix for comparison (same stay_ids as augmented)
    feat_indexed = feat_df.set_index("stay_id")
    base_phase_cols = sorted(
        [c for c in feat_df.columns if c.startswith("base_") and "_z_" in c]
    )
    # Ensure we use only stays present in the augmented set
    common_sids = [s for s in stay_ids if s in feat_indexed.index]
    if len(base_phase_cols) >= 16:
        X_base_raw = feat_indexed.loc[common_sids, base_phase_cols].values
    else:
        fallback_cols = [c for c in feat_indexed.columns
                         if any(c.endswith(s) for s in ["_mean", "_last", "_slope", "_std"])]
        X_base_raw = feat_indexed.loc[common_sids, fallback_cols].values
    X_base, _ = impute_and_scale(X_base_raw.copy())
    print(f"  Base-only matrix: {X_base.shape}")

    # ── [4/8] Clustering sweep (augmented) ───────────────────────────────
    print(f"\n[4/8] Clustering sweep (augmented features, K={K_CANDIDATES}) ...")
    sweep_results = run_clustering_sweep(X, K_CANDIDATES, SEED, N_INIT)

    # Also run base-only for comparison
    print(f"\n  Comparison: base-only clustering ...")
    sweep_base = run_clustering_sweep(X_base, K_CANDIDATES, SEED, N_INIT)

    # ── [5/8] Stability bootstrap for selected K values ──────────────────
    print(f"\n[5/8] Bootstrap stability ...")
    stability_rows = []
    for K in K_CANDIDATES:
        if sweep_results[K]["min_cluster_frac"] < MIN_CLUSTER_FRAC:
            print(f"  K={K}: SKIP — min cluster fraction {sweep_results[K]['min_cluster_frac']:.3f} < {MIN_CLUSTER_FRAC}")
            stability_rows.append({
                "K": K, "type": "augmented", "ari_mean": np.nan,
                "ari_p5": np.nan, "pass": False, "note": "cluster_too_small",
            })
            continue

        stab = bootstrap_stability(X, sweep_results[K]["labels"], K, N_BOOTSTRAP, SEED)
        stability_rows.append({
            "K": K, "type": "augmented",
            "ari_mean": stab["ari_mean"], "ari_p5": stab["ari_p5"],
            "nmi_mean": stab["nmi_mean"],
            "pass": stab["pass"],
        })
        print(f"  K={K} (augmented): ARI={stab['ari_mean']:.3f} [P5={stab['ari_p5']:.3f}] "
              f"{'PASS' if stab['pass'] else 'FAIL'}")

    # Bootstrap base-only for comparison K=4
    stab_base = bootstrap_stability(X_base, sweep_base[4]["labels"], 4, N_BOOTSTRAP, SEED)
    stability_rows.append({
        "K": 4, "type": "base_only",
        "ari_mean": stab_base["ari_mean"], "ari_p5": stab_base["ari_p5"],
        "nmi_mean": stab_base["nmi_mean"],
        "pass": stab_base["pass"],
    })
    print(f"  K=4 (base-only): ARI={stab_base['ari_mean']:.3f} [P5={stab_base['ari_p5']:.3f}]")

    stab_df = pd.DataFrame(stability_rows)
    stab_df.to_csv(TABLES / "96e_fluid_augmented_stability.csv", index=False)

    # ── [6/8] Select best K and compute detailed results ─────────────────
    stable_aug = [r for r in stability_rows
                  if r["type"] == "augmented" and r.get("pass", False)]
    if stable_aug:
        best_k = max(stable_aug, key=lambda r: sweep_results[r["K"]]["silhouette"])["K"]
    else:
        # Fallback: best silhouette among non-degenerate
        valid_ks = [K for K in K_CANDIDATES
                    if sweep_results[K]["min_cluster_frac"] >= MIN_CLUSTER_FRAC]
        best_k = max(valid_ks, key=lambda k: sweep_results[k]["silhouette"]) if valid_ks else 4
        print(f"  WARNING: No K passed stability. Selecting K={best_k} by silhouette")

    print(f"\n[6/8] Selected K={best_k}")

    best_labels = sweep_results[best_k]["labels"]
    cluster_df = pd.DataFrame({"stay_id": stay_ids, "cluster": best_labels})

    # Outcomes & HF enrichment
    outcome_df, hf_enrichment = compute_outcome_summary(cluster_df, outcomes_df, pheno_df)
    print(f"\n  Outcome summary:")
    for _, row in outcome_df.iterrows():
        print(f"    C{int(row['cluster'])}: n={int(row['n'])}, "
              f"mort={100*row['mortality_28d']:.1f}%, shock={100*row['shock_frac']:.1f}%")

    print(f"\n  HF enrichment:")
    hfpef_col = "hfpef" if "hfpef" in hf_enrichment.columns else None
    if hfpef_col:
        for c in hf_enrichment.index:
            print(f"    C{c}: HFpEF={100*hf_enrichment.loc[c, hfpef_col]:.1f}%")
        hfpef_spread = hf_enrichment[hfpef_col].max() - hf_enrichment[hfpef_col].min()
        print(f"  HFpEF spread: {100*hfpef_spread:.1f} percentage points")
    else:
        hfpef_spread = 0.0

    # ── Compare augmented vs base-only ───────────────────────────────────
    base_k = best_k if best_k in sweep_base else 4
    base_labels = sweep_base[base_k]["labels"]
    # Both base and augmented use the same stay_ids (from the merged set)
    if len(base_labels) == len(best_labels):
        ari_vs_base = adjusted_rand_score(best_labels, base_labels)
    else:
        # Length mismatch — compare on overlapping subset
        min_len = min(len(best_labels), len(base_labels))
        ari_vs_base = adjusted_rand_score(best_labels[:min_len], base_labels[:min_len])
    print(f"\n  ARI(augmented vs base-only, K={base_k}): {ari_vs_base:.3f}")

    # ── [7/8] Save outputs ───────────────────────────────────────────────
    print(f"\n[7/8] Saving outputs ...")

    # Cluster labels
    cluster_out = cluster_df.rename(columns={"cluster": "fluid_aug_cluster"})
    cluster_out.to_parquet(
        interim_dir / "96e_fluid_augmented_clusters_v1_mimic.parquet",
        index=False,
    )

    # Summary table
    summary_rows = []
    for K in K_CANDIDATES:
        r = sweep_results[K]
        stab_row = next((s for s in stability_rows if s["K"] == K and s["type"] == "augmented"), {})
        hfpef_range_str = ""
        if K == best_k and hfpef_col:
            vals = hf_enrichment[hfpef_col]
            hfpef_range_str = f"{100*vals.min():.1f}-{100*vals.max():.1f}%"
        summary_rows.append({
            "K": K,
            "silhouette_augmented": r["silhouette"],
            "silhouette_base": sweep_base[K]["silhouette"],
            "ari_mean": stab_row.get("ari_mean", np.nan),
            "ari_p5": stab_row.get("ari_p5", np.nan),
            "stability_pass": stab_row.get("pass", False),
            "min_cluster_frac": r["min_cluster_frac"],
            "hfpef_range": hfpef_range_str if K == best_k else "",
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(TABLES / "96e_fluid_augmented_cluster_summary.csv", index=False)

    # HF enrichment
    hf_enrichment.to_csv(TABLES / "96e_fluid_augmented_hf_enrichment.csv")

    # Plots
    plot_cluster_trajectories(state_df, cluster_df, norm_params,
                              FIGURES / "96e_fluid_augmented_cluster_trajectories.png", best_k)
    plot_fluid_balance_by_cluster(state_df, cluster_df,
                                  FIGURES / "96e_fluid_balance_by_cluster.png", best_k)

    # ── [8/8] Summary report ─────────────────────────────────────────────
    print(f"\n[8/8] Writing summary report ...")

    report_lines = [
        "=" * 72,
        f"96e — Fluid Balance Augmented Trajectory Clustering [{EXPLORATORY_TAG}]",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Tag: {EXPLORATORY_TAG}",
        "",
        "STATUS: EXPLORATORY — results are provisional and NOT part of the",
        "v3.1 core protocol.",
        "",
        "HYPOTHESIS",
        "──────────",
        "HFpEF's distinct hemodynamic substrate manifests primarily in the",
        "filling-pressure / volume-response axis. Standard encoder latent features",
        "(64 dims) optimized for reconstruction may not capture this. Adding",
        "cumulative fluid balance trajectory features should improve HFpEF",
        "enrichment differential across clusters.",
        "",
        "METHOD",
        "──────",
        f"Took 65a latent trajectory features ({len(base_phase_cols)} dims) and appended",
        f"{len(fluid_cols)} fluid balance trajectory features:",
    ]
    for c in fluid_cols:
        report_lines.append(f"  - {c}")
    report_lines.extend([
        "",
        f"Total features: {len(feat_cols)} → K-means sweep K={K_CANDIDATES}",
        f"Bootstrap stability: n={N_BOOTSTRAP}, threshold ARI≥{ARI_PASS} & P5≥{ARI_P5_PASS}",
        "",
        "CLUSTERING RESULTS (augmented vs base-only)",
        "─────────────────────────────────────────────",
    ])
    for _, row in summary_df.iterrows():
        k = int(row["K"])
        ari_str = f"{row['ari_mean']:.3f}" if pd.notna(row['ari_mean']) else "N/A"
        p5_str = f"{row['ari_p5']:.3f}" if pd.notna(row['ari_p5']) else "N/A"
        report_lines.append(
            f"  K={k}: sil_aug={row['silhouette_augmented']:.3f} vs sil_base={row['silhouette_base']:.3f}, "
            f"ARI={ari_str} [P5={p5_str}] "
            f"{'PASS' if row['stability_pass'] else 'FAIL'}"
        )

    report_lines.extend([
        "",
        f"SELECTED K: {best_k}",
        f"ARI(augmented vs base-only): {ari_vs_base:.3f}",
        "",
        "OUTCOME SUMMARY (selected K)",
        "────────────────────────────",
    ])
    for _, row in outcome_df.iterrows():
        report_lines.append(
            f"  C{int(row['cluster'])}: n={int(row['n'])}, "
            f"mort_28d={100*row['mortality_28d']:.1f}%, shock={100*row['shock_frac']:.1f}%"
        )

    report_lines.extend([
        "",
        "HFpEF ENRICHMENT (selected K)",
        "─────────────────────────────",
    ])
    for c in hf_enrichment.index:
        vals = {col: f"{100*hf_enrichment.loc[c, col]:.1f}%" for col in hf_enrichment.columns}
        report_lines.append(f"  C{c}: {vals}")

    if hfpef_col:
        report_lines.extend([
            "",
            f"  HFpEF spread: {100*hfpef_spread:.1f} percentage points",
            f"  Assessment: {'DIFFERENTIAL FOUND — fluid balance features improved separation'if hfpef_spread > 0.03 else 'NO MEANINGFUL DIFFERENTIAL — fluid features did not help'}",
        ])

    report_lines.extend([
        "",
        "COMPARISON WITH 96b (LATENT-ONLY K=4)",
        "──────────────────────────────────────",
        "96b K=4 HFpEF enrichment: 4.0–6.8% (spread: 2.8pp)",
        f"96e K={best_k} HFpEF enrichment: {100*hf_enrichment[hfpef_col].min():.1f}–{100*hf_enrichment[hfpef_col].max():.1f}% (spread: {100*hfpef_spread:.1f}pp)"
        if hfpef_col else "  N/A",
        "",
        "OUTPUTS",
        "───────",
        "  data/interim/96e_fluid_augmented_clusters_v1_mimic.parquet",
        "  figures/96e_fluid_augmented_cluster_trajectories.png",
        "  figures/96e_fluid_balance_by_cluster.png",
        "  tables/96e_fluid_augmented_cluster_summary.csv",
        "  tables/96e_fluid_augmented_hf_enrichment.csv",
        "  tables/96e_fluid_augmented_stability.csv",
        "  tables/96e_fluid_features_summary.csv",
        "  reports/96e_fluid_augmented_summary.txt",
        "",
        f"[{EXPLORATORY_TAG}] END",
    ])

    report_path = REPORTS / "96e_fluid_augmented_summary.txt"
    report_path.write_text("\n".join(report_lines))
    print(f"  Saved: {report_path}")

    print(f"\n{'=' * 72}")
    print(f"96e COMPLETE — [{EXPLORATORY_TAG}]")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
