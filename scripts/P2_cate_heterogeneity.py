#!/usr/bin/env python3
"""P2 — Individual Treatment Effect Heterogeneity (CATE)

Protocol: v3.1-patch, Step P2
Compute tier: C3–C4 (depends on subgroup count)
Depends on: 80c G-Transformer trained model

Objective: Characterize the distribution of individual-level conditional average
treatment effects (CATE), identifying which patient profiles benefit most or
may be harmed by the rule-based policy.

Analysis plan:
  1. Extract per-patient counterfactual outcome predictions from 80c G-Transformer
  2. Compute CATE_i = Y_i(rule) - Y_i(observed) for each patient
  3. Generate CATE distribution, benefit/harm proportions, variable importance,
     benefit subgroup profiles, and calibration checks

Outputs:
  - tables/P2_cate_distribution.csv
  - tables/P2_cate_benefit_predictors.csv
  - figures/P2_cate_histogram.png
  - figures/P2_cate_benefit_profile.png
  - reports/P2_cate_summary.txt
"""

from __future__ import annotations

import datetime
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Project root ──
ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(ROOT / "src"))

from hfpef_dtr.models.dtr_dataset import DTRDataset, build_dtr_datasets, load_dtr_data
from hfpef_dtr.models.dataset import discover_feature_columns, make_train_val_test_split
from hfpef_dtr.models.gtransformer import GTransformer, build_gtransformer_from_config

# ── Paths ──
CONFIG_PATH = ROOT / "configs" / "80_step_c_training_v1.yaml"
MODEL_CKPT = ROOT / "models" / "80c_gtransformer_best.pt"
TABLES_DIR = ROOT / "tables"
FIGURES_DIR = ROOT / "figures"
REPORTS_DIR = ROOT / "reports"

for d in [TABLES_DIR, FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# CATE extraction
# ═══════════════════════════════════════════════════════════════════════


def extract_patient_cate(
    model,
    dataset: DTRDataset,
    device: torch.device,
    batch_size: int = 256,
    use_rollout: bool = True,
) -> pd.DataFrame:
    """
    Extract per-patient CATE: Y_i(rule) - Y_i(observed).

    Returns DataFrame with columns:
      stay_id, y_observed_pred, y_rule_pred, cate, mortality_28d_actual
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_stay_ids = []
    all_obs_probs = []
    all_rule_probs = []
    all_mortality = []

    for batch in tqdm(loader, desc="  CATE extraction", leave=False, mininterval=2):
        temporal = batch["temporal"].to(device)
        static = batch["static"].to(device)
        action = batch["action"].to(device)
        rule_action = batch["rule_action"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        absorbing_mask = batch["absorbing_mask"].to(device)

        with torch.no_grad():
            if use_rollout and hasattr(model, "predict_counterfactual_rollout"):
                out_obs = model.predict_counterfactual_rollout(
                    temporal, static, action, valid_mask, absorbing_mask
                )
                out_rule = model.predict_counterfactual_rollout(
                    temporal, static, rule_action, valid_mask, absorbing_mask
                )
            else:
                out_obs = model.predict_counterfactual(
                    temporal, static, action, valid_mask, absorbing_mask
                )
                out_rule = model.predict_counterfactual(
                    temporal, static, rule_action, valid_mask, absorbing_mask
                )

        all_stay_ids.extend(batch["stay_id"].numpy().tolist())
        all_obs_probs.append(out_obs["outcome_prob"].cpu().numpy().squeeze())
        all_rule_probs.append(out_rule["outcome_prob"].cpu().numpy().squeeze())
        all_mortality.append(batch["mortality_28d"].numpy())

    obs_probs = np.concatenate(all_obs_probs)
    rule_probs = np.concatenate(all_rule_probs)
    mortality = np.concatenate(all_mortality)

    cate_df = pd.DataFrame({
        "stay_id": all_stay_ids,
        "y_observed_pred": obs_probs,
        "y_rule_pred": rule_probs,
        "cate": rule_probs - obs_probs,
        "mortality_28d_actual": mortality,
    })

    return cate_df


# ═══════════════════════════════════════════════════════════════════════
# Benefit predictor analysis (permutation importance)
# ═══════════════════════════════════════════════════════════════════════


def compute_benefit_predictors(
    cate_df: pd.DataFrame,
    covariates_df: pd.DataFrame,
    n_top: int = 10,
) -> pd.DataFrame:
    """
    Identify top predictors of treatment benefit using a gradient-boosted model
    of CATE on baseline covariates.

    Uses sklearn GradientBoostingRegressor with permutation importance.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import cross_val_score

    merged = cate_df[["stay_id", "cate"]].merge(covariates_df, on="stay_id", how="inner")

    # Feature columns: exclude IDs, targets, and CATE-related columns
    exclude = {"stay_id", "y_observed_pred", "y_rule_pred", "cate",
               "mortality_28d_actual", "mortality_28d", "bin"}
    feature_cols = [c for c in merged.columns if c not in exclude and merged[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

    if len(feature_cols) == 0:
        return pd.DataFrame()

    X = merged[feature_cols].fillna(0).values
    y = merged["cate"].values

    # Fit GBR
    gbr = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    gbr.fit(X, y)

    # Permutation importance (more reliable than built-in feature_importances_)
    # n_jobs=1 to avoid fork-based memory explosion on GPU nodes
    perm_imp = permutation_importance(
        gbr, X, y, n_repeats=20, random_state=42, n_jobs=1,
    )

    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": perm_imp.importances_mean,
        "importance_std": perm_imp.importances_std,
    }).sort_values("importance_mean", ascending=False)

    # Cross-val R²
    cv_scores = cross_val_score(gbr, X, y, cv=5, scoring="r2")
    print(f"  CATE predictor R² (5-fold CV): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    return imp_df.head(n_top)


# ═══════════════════════════════════════════════════════════════════════
# Calibration check
# ═══════════════════════════════════════════════════════════════════════


def calibration_by_cate_tertile(cate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare predicted vs observed outcomes in CATE tertiles.
    """
    cate_df = cate_df.copy()
    cate_df["cate_tertile"] = pd.qcut(
        cate_df["cate"], q=3, labels=["T1 (benefit)", "T2 (neutral)", "T3 (harm)"]
    )

    cal = cate_df.groupby("cate_tertile", observed=True).agg(
        n_patients=("stay_id", "count"),
        cate_mean=("cate", "mean"),
        cate_median=("cate", "median"),
        pred_mort_obs=("y_observed_pred", "mean"),
        pred_mort_rule=("y_rule_pred", "mean"),
        actual_mort=("mortality_28d_actual", "mean"),
    ).reset_index()

    return cal


# ═══════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════


def create_cate_histogram(cate_df: pd.DataFrame, pheno_df: pd.DataFrame,
                          output_path: Path) -> None:
    """CATE distribution histogram, overall and by HF subtype."""
    merged = cate_df.merge(
        pheno_df[["stay_id", "hf_subtype"]].drop_duplicates("stay_id"),
        on="stay_id", how="left",
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall distribution
    ax = axes[0]
    cate_vals = merged["cate"].values * 100  # convert to pp
    ax.hist(cate_vals, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.7, linewidth=1)
    pct_benefit = (cate_vals < 0).mean() * 100
    pct_harm = (cate_vals > 0).mean() * 100
    ax.set_xlabel("CATE (pp)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Overall CATE Distribution\n"
                 f"Benefit: {pct_benefit:.1f}%, Harm: {pct_harm:.1f}%", fontsize=11)
    ax.axvspan(ax.get_xlim()[0], 0, alpha=0.05, color="green")
    ax.axvspan(0, ax.get_xlim()[1], alpha=0.05, color="red")

    # By HF subtype
    ax = axes[1]
    _hf_disp = {"hfpef": "HFpEF", "no_hf": "No HF", "hfref": "HFrEF",
                "hfmref": "HFmrEF", "hf_ef_unknown": "HF (EF unknown)"}
    subtypes = merged["hf_subtype"].dropna().unique()
    colors = {"hfpef": "#2196F3", "hfref": "#FF9800", "hfmref": "#4CAF50",
              "no_hf": "#9E9E9E"}
    for st in sorted(subtypes):
        vals = merged.loc[merged["hf_subtype"] == st, "cate"].values * 100
        ax.hist(vals, bins=30, alpha=0.5,
                label=f"{_hf_disp.get(st, st)} (n={len(vals)})",
                color=colors.get(st, "gray"), edgecolor="white")
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.7, linewidth=1)
    ax.set_xlabel("CATE (pp)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("CATE by HF Subtype", fontsize=11)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(str(output_path).replace('.png', '.pdf'), format='pdf',
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"  Saved: {output_path} (+pdf)")


def create_benefit_profile(cate_df: pd.DataFrame, pheno_df: pd.DataFrame,
                           output_path: Path) -> None:
    """
    Benefit profile heatmap: mean CATE by subgroup cross-tabulations.
    """
    merged = cate_df.merge(
        pheno_df.drop_duplicates("stay_id"),
        on="stay_id", how="left",
    )

    # Create profile dimensions
    dims = {}
    if "hf_subtype" in merged.columns:
        dims["HF subtype"] = "hf_subtype"
    if "rule_phenotype_2class" in merged.columns:
        dims["Phenotype"] = "rule_phenotype_2class"
    if "age" in merged.columns:
        merged["age_group"] = pd.cut(
            merged["age"], bins=[0, 65, 75, 200],
            labels=["<65", "65-75", ">75"], right=False,
        )
        dims["Age group"] = "age_group"
    if "sex" in merged.columns:
        dims["Sex"] = "sex"

    if len(dims) < 2:
        print("  Insufficient subgroup columns for benefit profile heatmap")
        return

    # Remap raw HF subtype values for display
    _hf_map = {"hfpef": "HFpEF", "no_hf": "No HF", "hfref": "HFrEF",
               "hfmref": "HFmrEF", "hf_ef_unknown": "HF (EF unknown)",
               "shock_predominant": "Shock", "non_shock": "Non-shock"}
    if "hf_subtype" in merged.columns:
        merged["hf_subtype"] = merged["hf_subtype"].map(
            lambda v: _hf_map.get(v, v))
    if "rule_phenotype_2class" in merged.columns:
        merged["rule_phenotype_2class"] = merged["rule_phenotype_2class"].map(
            lambda v: _hf_map.get(v, v))

    # Build heatmap data: HF subtype × phenotype
    heatmap_pairs = []
    dim_names = list(dims.keys())
    for i in range(len(dim_names)):
        for j in range(i + 1, len(dim_names)):
            heatmap_pairs.append((dim_names[i], dim_names[j]))

    n_pairs = min(len(heatmap_pairs), 4)
    fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 5))
    if n_pairs == 1:
        axes = [axes]

    for ax_idx, (d1_name, d2_name) in enumerate(heatmap_pairs[:n_pairs]):
        d1_col = dims[d1_name]
        d2_col = dims[d2_name]

        pivot = merged.pivot_table(
            values="cate", index=d1_col, columns=d2_col,
            aggfunc="mean",
        ) * 100  # to pp

        if pivot.empty:
            continue

        im = ax = axes[ax_idx]
        ax = axes[ax_idx]
        data = pivot.values
        im = ax.imshow(data, cmap="RdBu_r", aspect="auto",
                       vmin=-max(abs(data.min()), abs(data.max())),
                       vmax=max(abs(data.min()), abs(data.max())))
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=8, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_xlabel(d2_name, fontsize=10)
        ax.set_ylabel(d1_name, fontsize=10)
        ax.set_title(f"Mean CATE (pp)", fontsize=10)

        # Add text annotations
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if not np.isnan(data[i, j]):
                    ax.text(j, i, f"{data[i, j]:+.2f}",
                            ha="center", va="center", fontsize=8,
                            color="white" if abs(data[i, j]) > data.max() * 0.5 else "black")

        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(str(output_path).replace('.png', '.pdf'), format='pdf',
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"  Saved: {output_path} (+pdf)")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main():
    import gc

    print("=" * 72)
    print("P2 — CATE Heterogeneity Analysis")
    print("=" * 72)
    t0 = time.time()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 1. Load config, model, data ──
    print("[1] Loading config and data...")
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load data once, build ONLY test dataset (not train/val) to save memory
    state_df, outcomes_df, propensity_df, temporal_cols, static_cols, merge_map = (
        load_dtr_data(ROOT, config)
    )
    all_stay_ids = state_df["stay_id"].unique()
    split_cfg = config["split"]
    _, _, test_ids = make_train_val_test_split(
        all_stay_ids, outcomes_df,
        train_frac=split_cfg["train_frac"],
        val_frac=split_cfg["val_frac"],
        test_frac=split_cfg["test_frac"],
        stratify_col=split_cfg["stratify_col"],
        seed=split_cfg["seed"],
    )

    test_ds = DTRDataset(
        state_df=state_df,
        outcomes_df=outcomes_df,
        propensity_df=propensity_df,
        stay_ids=test_ids,
        temporal_cols=temporal_cols,
        static_cols=static_cols,
        merge_map=merge_map,
        n_bins=config["grid"]["n_bins"],
    )
    print(f"  Test set: {len(test_ds)} stays")
    print(f"  Temporal features: {test_ds.d_temporal}, Static: {test_ds.d_static}")
    print(f"  Actions: {test_ds.n_actions}")

    # Load phenotype labels
    pheno_path = ROOT / config["data"]["phenotype_labels"]
    pheno_df = pd.read_parquet(pheno_path)

    # ── 2. Load trained G-Transformer ──
    print("[2] Loading 80c G-Transformer checkpoint...")
    ckpt = torch.load(MODEL_CKPT, map_location=device, weights_only=False)
    model = build_gtransformer_from_config(
        d_temporal=test_ds.d_temporal,
        d_static=test_ds.d_static,
        n_actions=test_ds.n_actions,
        model_cfg=config["models"]["80c_gtransformer"],
        grid_cfg=config["grid"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")

    # ── 3. Extract per-patient CATE ──
    print("[3] Extracting per-patient CATE...")
    cate_df = extract_patient_cate(model, test_ds, device, use_rollout=True)

    n_total = len(cate_df)
    n_benefit = (cate_df["cate"] < 0).sum()
    n_harm = (cate_df["cate"] > 0).sum()
    n_neutral = (cate_df["cate"] == 0).sum()
    print(f"  Total patients: {n_total}")
    print(f"  Benefit (CATE < 0): {n_benefit} ({n_benefit/n_total*100:.1f}%)")
    print(f"  Harm (CATE > 0):    {n_harm} ({n_harm/n_total*100:.1f}%)")
    print(f"  Neutral (CATE = 0): {n_neutral} ({n_neutral/n_total*100:.1f}%)")
    print(f"  Mean CATE: {cate_df['cate'].mean()*100:+.2f} pp")
    print(f"  Median CATE: {cate_df['cate'].median()*100:+.2f} pp")
    print(f"  CATE range: [{cate_df['cate'].min()*100:.2f}, {cate_df['cate'].max()*100:.2f}] pp")

    # ── 4. CATE by HF subtype ──
    print("[4] CATE by HF subtype...")
    merged_pheno = cate_df.merge(
        pheno_df[["stay_id", "hf_subtype"]].drop_duplicates("stay_id"),
        on="stay_id", how="left",
    )

    subtype_stats = []
    for st, grp in merged_pheno.groupby("hf_subtype"):
        stats = {
            "hf_subtype": st,
            "n": len(grp),
            "cate_mean_pp": grp["cate"].mean() * 100,
            "cate_median_pp": grp["cate"].median() * 100,
            "cate_sd_pp": grp["cate"].std() * 100,
            "pct_benefit": (grp["cate"] < 0).mean() * 100,
            "pct_harm": (grp["cate"] > 0).mean() * 100,
        }
        subtype_stats.append(stats)
        print(f"  {st}: n={stats['n']}, mean CATE={stats['cate_mean_pp']:+.2f}pp, "
              f"benefit={stats['pct_benefit']:.1f}%")

    # ── 5. Benefit predictor analysis ──
    print("[5] Identifying benefit predictors...")
    # First-bin covariates for each test patient (reuse already-loaded state_df)
    test_state = state_df[state_df["stay_id"].isin(set(test_ids))].copy()

    # Free large DataFrames no longer needed to reduce memory before sklearn
    del state_df, outcomes_df, propensity_df
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    first_bin = test_state.sort_values(["stay_id", "bin"]).drop_duplicates(
        subset="stay_id", keep="first"
    )
    del test_state
    gc.collect()

    # Merge with phenotype info
    covariates = first_bin.merge(
        pheno_df.drop_duplicates("stay_id"),
        on="stay_id", how="left", suffixes=("", "_pheno"),
    )
    del first_bin
    gc.collect()

    benefit_predictors = compute_benefit_predictors(cate_df, covariates, n_top=10)
    if len(benefit_predictors) > 0:
        print("  Top 10 CATE predictors:")
        for _, row in benefit_predictors.iterrows():
            print(f"    {row['feature']:40s}: {row['importance_mean']:.4f} ± {row['importance_std']:.4f}")

    # ── 6. Calibration check ──
    print("[6] Calibration by CATE tertile...")
    cal_df = calibration_by_cate_tertile(cate_df)
    for _, row in cal_df.iterrows():
        print(f"  {row['cate_tertile']}: n={row['n_patients']}, "
              f"mean CATE={row['cate_mean']*100:+.2f}pp, "
              f"pred_mort(obs)={row['pred_mort_obs']:.4f}, "
              f"actual_mort={row['actual_mort']:.4f}")

    # ── 7. Create figures ──
    print("[7] Creating figures...")
    create_cate_histogram(cate_df, pheno_df, FIGURES_DIR / "P2_cate_histogram.png")
    create_benefit_profile(cate_df, pheno_df, FIGURES_DIR / "P2_cate_benefit_profile.png")

    # ── 8. Save outputs ──
    print("[8] Saving outputs...")

    # CATE distribution table
    cate_out = cate_df[["stay_id", "y_observed_pred", "y_rule_pred", "cate",
                         "mortality_28d_actual"]].copy()
    cate_out["cate_pp"] = cate_out["cate"] * 100
    cate_out.to_csv(TABLES_DIR / "P2_cate_distribution.csv", index=False)
    print(f"  Saved: {TABLES_DIR / 'P2_cate_distribution.csv'}")

    # Benefit predictors table
    if len(benefit_predictors) > 0:
        benefit_predictors.to_csv(TABLES_DIR / "P2_cate_benefit_predictors.csv", index=False)
        print(f"  Saved: {TABLES_DIR / 'P2_cate_benefit_predictors.csv'}")

    # Summary report
    report_lines = [
        "=" * 72,
        "P2 — CATE Heterogeneity Analysis",
        "=" * 72,
        f"Generated: {ts}",
        f"Protocol: v3.1-patch, Step P2",
        f"Database: MIMIC-IV",
        f"Model: 80c G-Transformer (rollout DM)",
        f"Test set: {n_total} patients",
        "",
        "OVERALL CATE DISTRIBUTION",
        "─" * 25,
        f"  Mean CATE:   {cate_df['cate'].mean()*100:+.2f} pp",
        f"  Median CATE: {cate_df['cate'].median()*100:+.2f} pp",
        f"  SD CATE:     {cate_df['cate'].std()*100:.2f} pp",
        f"  Range:       [{cate_df['cate'].min()*100:.2f}, {cate_df['cate'].max()*100:.2f}] pp",
        f"  Benefit (CATE < 0): {n_benefit} ({n_benefit/n_total*100:.1f}%)",
        f"  Harm (CATE > 0):    {n_harm} ({n_harm/n_total*100:.1f}%)",
        "",
        "CATE BY HF SUBTYPE",
        "─" * 19,
    ]
    for s in subtype_stats:
        report_lines.append(
            f"  {s['hf_subtype']:10s}: n={s['n']:>5}, "
            f"mean={s['cate_mean_pp']:+.2f}pp, "
            f"benefit={s['pct_benefit']:.1f}%"
        )
    report_lines.extend(["", "TOP BENEFIT PREDICTORS", "─" * 22])
    if len(benefit_predictors) > 0:
        for _, row in benefit_predictors.iterrows():
            report_lines.append(
                f"  {row['feature']:40s}: {row['importance_mean']:.4f} ± {row['importance_std']:.4f}"
            )
    report_lines.extend(["", "CALIBRATION BY CATE TERTILE", "─" * 27])
    for _, row in cal_df.iterrows():
        report_lines.append(
            f"  {row['cate_tertile']}: n={row['n_patients']}, "
            f"CATE={row['cate_mean']*100:+.2f}pp, "
            f"pred(obs)={row['pred_mort_obs']:.4f}, "
            f"actual={row['actual_mort']:.4f}"
        )
    report_lines.extend([
        "",
        "INTERPRETATION",
        "─" * 14,
        "  CATE < 0 = predicted benefit from rule policy (lower mortality).",
        "  CATE > 0 = predicted harm (higher mortality).",
        "  Calibration: predicted observed mortality should track actual mortality.",
        "  If T1 (benefit) patients have lower actual mortality under observed care,",
        "  the model correctly identifies patients whose current treatment is suboptimal.",
        "",
        "MANUSCRIPT PLACEMENT",
        "─" * 20,
        "  Results Section 4; CATE histogram + benefit profile heatmap.",
        "  Supports precision-medicine framing of DTR findings.",
        "",
    ])

    report_path = REPORTS_DIR / "P2_cate_summary.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Saved: {report_path}")

    total_time = time.time() - t0
    print()
    print("=" * 72)
    print(f"P2 COMPLETE. Total time: {total_time / 60:.1f} min")
    print("=" * 72)


if __name__ == "__main__":
    main()
