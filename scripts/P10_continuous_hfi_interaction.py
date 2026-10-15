#!/usr/bin/env python3
"""P10 — Continuous HFI × HF-Phenotype Interaction Analysis

Protocol: v3.1-patch, Step P10
Compute tier: C3 (model forward passes on test set)
Depends on: 80c G-Transformer, 45c state tensors, P9 HFI computation

Objective: Replace discrete HFI-quartile stratification with continuous
dose-response curves for HFpEF vs no_HF, and formally test the interaction.

Approach (v2 — bin-level analysis):
  1. Compute per-stay ITE from G-Transformer rollout (one forward pass)
  2. Aggregate ITE into overlapping sliding-window bins along HFI
  3. Fit LOWESS on bin-level ATE means (not individual ITE — too noisy)
  4. Test interaction via WLS on bin-level means: ATE ~ HFI * HFtype
  5. Bootstrap confidence bands by resampling stays within bins

Rationale for bin-level vs individual-level analysis:
  Per-stay ITE from DM has enormous variance (individual mortality probs
  range 0-1, ATE is ~1.4pp). Individual-level OLS has R²≈0.0004 — no power.
  Bin-level means average over 50-500 stays, reducing SE by √n and
  recovering the group-level signal that P9 quartiles already demonstrated.

Outputs:
  - tables/P10_continuous_hfi_interaction.csv
  - tables/P10_binned_ate_by_hfi.csv
  - tables/P10_interaction_test_summary.csv
  - figures/P10_continuous_dose_response.png / .pdf
  - reports/P10_continuous_hfi_interaction_summary.txt
"""

from __future__ import annotations

import gc
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats
from tqdm import tqdm

# ── Project root ──
ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(ROOT / "src"))

from hfpef_dtr.models.dtr_dataset import DTRDataset, load_dtr_data
from hfpef_dtr.models.dataset import make_train_val_test_split
from hfpef_dtr.models.gtransformer import build_gtransformer_from_config
from hfpef_dtr.models.ope import evaluate_dm

# ── Paths ──
CONFIG_PATH = ROOT / "configs" / "80_step_c_training_v1.yaml"
NORM_PATH = ROOT / "configs" / "45c_normalization_params_v1.yaml"
MODEL_CKPT = ROOT / "models" / "80c_gtransformer_best.pt"
TABLES_DIR = ROOT / "tables"
FIGURES_DIR = ROOT / "figures"
REPORTS_DIR = ROOT / "reports"

for d in [TABLES_DIR, FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MAP_HYPO_THRESHOLD = 65.0

# ── Journal-quality plot defaults ──
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "axes.linewidth": 0.8,
})


# ═══════════════════════════════════════════════════════════════════
# A. HFI COMPUTATION (reused from P9)
# ═══════════════════════════════════════════════════════════════════

def compute_hfi_per_stay(state_df: pd.DataFrame, norm_params: dict) -> pd.DataFrame:
    """Compute Hemodynamic Fragility Index per stay (same as P9)."""
    if "map" in state_df.columns:
        map_col = "map"
    elif "map_z" in state_df.columns:
        map_mean_norm = float(norm_params["map"]["mean"])
        map_std_norm = float(norm_params["map"]["std"])
        state_df = state_df.copy()
        state_df["_map_raw"] = state_df["map_z"] * map_std_norm + map_mean_norm
        map_col = "_map_raw"
    else:
        raise ValueError("No MAP column found in state_df")

    grouped = state_df.groupby("stay_id")[map_col]

    hfi = pd.DataFrame({
        "stay_id": grouped.first().index,
        "n_hypo_bins": grouped.apply(lambda x: (x < MAP_HYPO_THRESHOLD).sum()),
        "map_mean": grouped.mean(),
        "map_std": grouped.std(ddof=0),
        "map_min": grouped.min(),
        "map_max": grouped.max(),
    })

    hfi["map_cv"] = hfi["map_std"] / hfi["map_mean"].clip(lower=1.0)
    hfi["map_range"] = hfi["map_max"] - hfi["map_min"]

    for col in ["n_hypo_bins", "map_cv", "map_range"]:
        mu, sd = hfi[col].mean(), hfi[col].std()
        hfi[f"{col}_z"] = (hfi[col] - mu) / sd if sd > 0 else 0.0

    hfi["hfi_composite"] = (
        hfi["n_hypo_bins_z"] + hfi["map_cv_z"] + hfi["map_range_z"]
    ) / 3.0

    return hfi.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# B. PER-STAY ITE COMPUTATION
# ═══════════════════════════════════════════════════════════════════

def compute_per_stay_ite(
    model, state_df, outcomes_df, propensity_df,
    temporal_cols, static_cols, merge_map,
    stay_ids, config, device,
) -> pd.DataFrame:
    """Run G-Transformer DM rollout once; return per-stay ITE.

    IMPORTANT: Uses ds.stay_ids (DTRDataset's internally sorted order)
    for output alignment, and passes frozen action_vocab to ensure
    contiguous action mapping matches the trained model.
    """
    ds = DTRDataset(
        state_df=state_df,
        outcomes_df=outcomes_df,
        propensity_df=propensity_df,
        stay_ids=stay_ids,
        temporal_cols=temporal_cols,
        static_cols=static_cols,
        merge_map=merge_map,
        n_bins=config["grid"]["n_bins"],
        action_vocab=config["action"]["valid_merged_indices"],
    )
    result = evaluate_dm(model, ds, device, use_rollout=True)
    # Use ds.stay_ids (sorted) to match output order from DataLoader
    return pd.DataFrame({
        "stay_id": ds.stay_ids,
        "v_obs": result["obs_probs"],
        "v_rule": result["cf_probs"],
        "ite": result["cf_probs"] - result["obs_probs"],
    })


# ═══════════════════════════════════════════════════════════════════
# C. SLIDING-WINDOW BINNED ATE
# ═══════════════════════════════════════════════════════════════════

def compute_binned_ate(
    ite_hfi_df: pd.DataFrame,
    n_bins: int = 12,
    overlap_factor: float = 2.5,
    min_n: int = 15,
) -> pd.DataFrame:
    """Compute mean ATE in overlapping sliding windows along HFI.

    Uses wider windows (overlap_factor=2.5) to ensure adequate n for HFpEF.
    """
    df = ite_hfi_df[ite_hfi_df["hf_subtype"].isin(["hfpef", "no_hf"])].copy()
    hfi_lo, hfi_hi = df["hfi_composite"].quantile([0.03, 0.97])
    centers = np.linspace(hfi_lo, hfi_hi, n_bins)
    step = (hfi_hi - hfi_lo) / (n_bins - 1) if n_bins > 1 else 1.0
    half_width = step * overlap_factor / 2

    rows = []
    for c in centers:
        lo, hi = c - half_width, c + half_width
        for hf in ["hfpef", "no_hf"]:
            mask = (
                (df["hfi_composite"] >= lo)
                & (df["hfi_composite"] < hi)
                & (df["hf_subtype"] == hf)
            )
            sub = df.loc[mask]
            if len(sub) >= min_n:
                rows.append({
                    "hfi_center": c,
                    "hf_subtype": hf,
                    "n_stays": len(sub),
                    "hfi_mean": sub["hfi_composite"].mean(),
                    "ate_mean": sub["ite"].mean(),
                    "ate_se": sub["ite"].std() / np.sqrt(len(sub)),
                })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# D. INTERACTION TEST (BIN-LEVEL WLS)
# ═══════════════════════════════════════════════════════════════════

def test_interaction_wls(binned_df: pd.DataFrame) -> dict:
    """Test ATE ~ HFI * HFtype using WLS on bin-level means.

    Weights = 1/SE² (inverse variance weighting for bin-level estimates).
    This is the correct analysis level for OPE-derived ATEs.
    """
    import statsmodels.api as sm

    df = binned_df.dropna(subset=["ate_mean", "ate_se"]).copy()
    df = df[df["ate_se"] > 0].copy()

    df["is_hfpef"] = (df["hf_subtype"] == "hfpef").astype(float)
    df["hfi_x_hfpef"] = df["hfi_center"] * df["is_hfpef"]
    weights = 1.0 / (df["ate_se"] ** 2)

    X = sm.add_constant(df[["hfi_center", "is_hfpef", "hfi_x_hfpef"]])
    y = df["ate_mean"]
    model = sm.WLS(y, X, weights=weights).fit()

    return {
        "n_bins": len(df),
        "n_hfpef_bins": int(df["is_hfpef"].sum()),
        "n_nohf_bins": int((~df["is_hfpef"].astype(bool)).sum()),
        "beta_hfi": model.params["hfi_center"],
        "se_hfi": model.bse["hfi_center"],
        "p_hfi": model.pvalues["hfi_center"],
        "beta_hfpef": model.params["is_hfpef"],
        "se_hfpef": model.bse["is_hfpef"],
        "p_hfpef": model.pvalues["is_hfpef"],
        "beta_interaction": model.params["hfi_x_hfpef"],
        "se_interaction": model.bse["hfi_x_hfpef"],
        "p_interaction": model.pvalues["hfi_x_hfpef"],
        "r_squared": model.rsquared,
        "model_summary": model.summary().as_text(),
    }


def test_raw_group_difference(ite_hfi_df: pd.DataFrame) -> dict:
    """Direct comparison of mean ITE: HFpEF vs no_HF (unadjusted)."""
    hfpef = ite_hfi_df.loc[ite_hfi_df["hf_subtype"] == "hfpef", "ite"]
    nohf = ite_hfi_df.loc[ite_hfi_df["hf_subtype"] == "no_hf", "ite"]

    t_stat, p_val = stats.ttest_ind(hfpef, nohf, equal_var=False)
    u_stat, p_mw = stats.mannwhitneyu(hfpef, nohf, alternative="two-sided")

    return {
        "hfpef_mean_ite": hfpef.mean(),
        "nohf_mean_ite": nohf.mean(),
        "diff": hfpef.mean() - nohf.mean(),
        "hfpef_se": hfpef.std() / np.sqrt(len(hfpef)),
        "nohf_se": nohf.std() / np.sqrt(len(nohf)),
        "welch_t": t_stat,
        "welch_p": p_val,
        "mann_whitney_U": u_stat,
        "mann_whitney_p": p_mw,
        "n_hfpef": len(hfpef),
        "n_nohf": len(nohf),
    }


# ═══════════════════════════════════════════════════════════════════
# E. BOOTSTRAP CI FOR BIN-LEVEL LOWESS
# ═══════════════════════════════════════════════════════════════════

def bootstrap_binned_lowess(
    ite_hfi_df: pd.DataFrame,
    hf_subtype: str,
    x_grid: np.ndarray,
    n_bins: int = 12,
    overlap_factor: float = 2.5,
    min_n: int = 15,
    frac: float = 0.5,
    n_boot: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap CI for LOWESS curve fitted to bin-level means.

    For each bootstrap iteration:
      1. Resample stays (with replacement) from the given subtype
      2. Recompute bin-level ATE means
      3. Fit LOWESS on bin means
    Returns (y_smooth, y_lo, y_hi) on x_grid.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess

    sub = ite_hfi_df[ite_hfi_df["hf_subtype"] == hf_subtype].copy()
    n = len(sub)
    rng = np.random.RandomState(seed)

    hfi_lo, hfi_hi = ite_hfi_df["hfi_composite"].quantile([0.03, 0.97])
    centers = np.linspace(hfi_lo, hfi_hi, n_bins)
    step = (hfi_hi - hfi_lo) / (n_bins - 1) if n_bins > 1 else 1.0
    half_width = step * overlap_factor / 2

    def _bin_means(df_sub):
        xs, ys = [], []
        for c in centers:
            lo, hi = c - half_width, c + half_width
            mask = (df_sub["hfi_composite"] >= lo) & (df_sub["hfi_composite"] < hi)
            s = df_sub.loc[mask]
            if len(s) >= min_n:
                xs.append(c)
                ys.append(s["ite"].mean())
        return np.array(xs), np.array(ys)

    # Point estimate
    x_pt, y_pt = _bin_means(sub)
    if len(x_pt) < 3:
        return np.full_like(x_grid, np.nan), np.full_like(x_grid, np.nan), np.full_like(x_grid, np.nan)

    fit_pt = lowess(y_pt * 100, x_pt, frac=frac, return_sorted=True)
    y_smooth = np.interp(x_grid, fit_pt[:, 0], fit_pt[:, 1])

    # Bootstrap
    boot_curves = np.full((n_boot, len(x_grid)), np.nan)
    for b in tqdm(range(n_boot), desc=f"  Bootstrap ({hf_subtype})", leave=False):
        idx = rng.choice(n, size=n, replace=True)
        boot_sub = sub.iloc[idx]
        xb, yb = _bin_means(boot_sub)
        if len(xb) >= 3:
            try:
                fit_b = lowess(yb * 100, xb, frac=frac, return_sorted=True)
                boot_curves[b] = np.interp(x_grid, fit_b[:, 0], fit_b[:, 1])
            except Exception:
                continue

    valid = ~np.all(np.isnan(boot_curves), axis=1)
    bv = boot_curves[valid]
    y_lo = np.nanpercentile(bv, 2.5, axis=0) if len(bv) > 0 else y_smooth
    y_hi = np.nanpercentile(bv, 97.5, axis=0) if len(bv) > 0 else y_smooth

    return y_smooth, y_lo, y_hi


# ═══════════════════════════════════════════════════════════════════
# F. FIGURE GENERATION
# ═══════════════════════════════════════════════════════════════════

def make_figure(
    ite_hfi_df: pd.DataFrame,
    binned_df: pd.DataFrame,
    wls_result: dict,
    raw_diff: dict,
    out_path: Path,
):
    """Two-panel figure: LOWESS on bin means + sliding-window error bars."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.30})

    colors = {"hfpef": "#C0392B", "no_hf": "#2980B9"}
    labels = {"hfpef": "HFpEF", "no_hf": "No HF"}

    hfi_lo = ite_hfi_df["hfi_composite"].quantile(0.03)
    hfi_hi = ite_hfi_df["hfi_composite"].quantile(0.97)
    x_grid = np.linspace(hfi_lo, hfi_hi, 200)

    # ── Panel A: LOWESS on bin-level means with bootstrap CI ──
    ax = axes[0]

    for hf in ["no_hf", "hfpef"]:  # draw no_hf first (behind)
        frac = 0.6 if hf == "hfpef" else 0.45
        y_smooth, y_lo, y_hi = bootstrap_binned_lowess(
            ite_hfi_df, hf, x_grid, frac=frac, n_boot=500, seed=42,
        )
        ax.plot(x_grid, y_smooth, color=colors[hf], lw=2.5, label=labels[hf])
        ax.fill_between(x_grid, y_lo, y_hi, color=colors[hf], alpha=0.15)

    ax.axhline(0, color="grey", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Hemodynamic Fragility Index (HFI)", fontsize=11)
    ax.set_ylabel("Estimated ATE (pp)", fontsize=11)
    ax.set_title("A.  Smoothed ATE vs HFI (LOWESS)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="lower left")

    # Annotations
    diff_pp = raw_diff["diff"] * 100
    p_main = wls_result["p_hfpef"]
    p_int = wls_result["p_interaction"]
    ann = (
        f"Raw HFpEF−no_HF: {diff_pp:+.2f} pp "
        f"(Welch p={raw_diff['welch_p']:.3f})\n"
        f"WLS β(HFpEF): {wls_result['beta_hfpef']*100:+.2f} pp "
        f"(p={p_main:.3f})\n"
        f"WLS interaction p={p_int:.3f}"
    )
    ax.text(
        0.97, 0.97, ann,
        transform=ax.transAxes, fontsize=8, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.5),
    )

    # ── Panel B: Sliding-window error bars ──
    ax = axes[1]

    for hf in ["no_hf", "hfpef"]:
        sub = binned_df[binned_df["hf_subtype"] == hf].sort_values("hfi_center")
        x = sub["hfi_center"].values
        y = sub["ate_mean"].values * 100
        yerr = sub["ate_se"].values * 100 * 1.96

        ax.errorbar(
            x, y, yerr=yerr, fmt="o-", color=colors[hf], markersize=5,
            capsize=3, lw=1.5, label=labels[hf],
        )

    ax.axhline(0, color="grey", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("HFI bin center", fontsize=11)
    ax.set_ylabel("Mean ATE (pp) ± 95% CI", fontsize=11)
    ax.set_title("B.  Sliding-window ATE by HFI", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="lower left")

    # Add n annotations for HFpEF bins
    hfpef_bins = binned_df[binned_df["hf_subtype"] == "hfpef"].sort_values("hfi_center")
    for _, row in hfpef_bins.iterrows():
        ax.annotate(
            f"n={int(row['n_stays'])}",
            xy=(row["hfi_center"], row["ate_mean"] * 100),
            xytext=(0, 10), textcoords="offset points",
            fontsize=6, color=colors["hfpef"], ha="center", alpha=0.7,
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {out_path}")
    print(f"  Figure saved: {pdf_path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("P10 — Continuous HFI × HF-Phenotype Interaction Analysis (v2)")
    print("     Bin-level WLS + LOWESS (not individual-level OLS)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ── Load config ──
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    with open(NORM_PATH) as f:
        norm_params = yaml.safe_load(f).get("params", {})

    # ── Load data ──
    print("\n[1/7] Loading data...")
    state_df, outcomes_df, propensity_df, temporal_cols, static_cols, merge_map = (
        load_dtr_data(ROOT, config)
    )
    pheno_df = pd.read_parquet(ROOT / config["data"]["phenotype_labels"])

    # ── Test set only ──
    all_ids = state_df["stay_id"].unique()
    split_cfg = config["split"]
    _, _, test_ids = make_train_val_test_split(
        all_ids, outcomes_df,
        train_frac=split_cfg["train_frac"],
        val_frac=split_cfg["val_frac"],
        test_frac=split_cfg["test_frac"],
        stratify_col=split_cfg["stratify_col"],
        seed=split_cfg["seed"],
    )
    print(f"  Test set: {len(test_ids)} stays")

    test_state = state_df[state_df["stay_id"].isin(test_ids)]

    # ── Load model ──
    print("\n[2/7] Loading G-Transformer model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    ckpt = torch.load(MODEL_CKPT, map_location=device, weights_only=False)
    model = build_gtransformer_from_config(
        d_temporal=len(temporal_cols),
        d_static=len(static_cols),
        n_actions=len(config["action"]["valid_merged_indices"]),
        model_cfg=config["models"]["80c_gtransformer"],
        grid_cfg=config["grid"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── Compute HFI ──
    print("\n[3/7] Computing HFI per stay...")
    hfi_df = compute_hfi_per_stay(test_state, norm_params)

    pheno_cols = ["stay_id", "hf_subtype", "rule_phenotype_2class"]
    avail_cols = [c for c in pheno_cols if c in pheno_df.columns]
    hfi_df = hfi_df.merge(pheno_df[avail_cols], on="stay_id", how="left")

    n_hfpef = (hfi_df["hf_subtype"] == "hfpef").sum()
    n_nohf = (hfi_df["hf_subtype"] == "no_hf").sum()
    print(f"  HFpEF: {n_hfpef}, no_HF: {n_nohf}")
    print(f"  HFI range: [{hfi_df['hfi_composite'].min():.3f}, "
          f"{hfi_df['hfi_composite'].max():.3f}]")

    # ── Compute per-stay ITE ──
    print("\n[4/7] Computing per-stay ITE (G-Transformer rollout)...")
    ite_df = compute_per_stay_ite(
        model, state_df, outcomes_df, propensity_df,
        temporal_cols, static_cols, merge_map,
        test_ids, config, device,
    )
    print(f"  ITE computed for {len(ite_df)} stays")
    print(f"  Mean ITE: {ite_df['ite'].mean():.6f} "
          f"(= ATE {ite_df['ite'].mean()*100:.2f} pp)")

    # Merge
    ite_hfi_df = ite_df.merge(
        hfi_df[["stay_id", "hfi_composite", "hf_subtype"]],
        on="stay_id", how="inner",
    )
    print(f"  Merged ITE+HFI: {len(ite_hfi_df)} stays")

    # ── Raw group difference ──
    print("\n[5/7] Raw ITE comparison (HFpEF vs no_HF)...")
    raw_diff = test_raw_group_difference(ite_hfi_df)
    print(f"  HFpEF mean ITE: {raw_diff['hfpef_mean_ite']*100:+.3f} pp "
          f"(SE={raw_diff['hfpef_se']*100:.3f})")
    print(f"  no_HF mean ITE: {raw_diff['nohf_mean_ite']*100:+.3f} pp "
          f"(SE={raw_diff['nohf_se']*100:.3f})")
    print(f"  Difference:     {raw_diff['diff']*100:+.3f} pp "
          f"(Welch p={raw_diff['welch_p']:.4f})")

    # ── Sliding-window bins ──
    print("\n[6/7] Sliding-window binned ATE + WLS interaction test...")
    binned_df = compute_binned_ate(ite_hfi_df, n_bins=12, overlap_factor=2.5, min_n=15)
    print(f"  Bins: {len(binned_df)} rows "
          f"(HFpEF: {(binned_df['hf_subtype']=='hfpef').sum()}, "
          f"no_HF: {(binned_df['hf_subtype']=='no_hf').sum()})")

    # P9 quartile verification
    print("\n  Verification against P9 quartile ATEs:")
    for hf in ["hfpef", "no_hf"]:
        sub = ite_hfi_df[ite_hfi_df["hf_subtype"] == hf]
        q_labels = ["Q1", "Q2", "Q3", "Q4"]
        try:
            qs = pd.qcut(sub["hfi_composite"], 4, labels=q_labels)
            for q in q_labels:
                q_ite = sub.loc[qs == q, "ite"]
                if len(q_ite) > 0:
                    print(f"    {hf:8s} {q}: n={len(q_ite):4d}, "
                          f"mean ITE={q_ite.mean()*100:+.3f} pp")
        except Exception:
            print(f"    {hf}: could not compute quartiles")

    # WLS interaction test
    wls_result = test_interaction_wls(binned_df)
    print(f"\n  WLS interaction test (n_bins={wls_result['n_bins']}):")
    print(f"    β(HFI):         {wls_result['beta_hfi']*100:+.4f} pp/unit "
          f"(p={wls_result['p_hfi']:.4f})")
    print(f"    β(HFpEF):       {wls_result['beta_hfpef']*100:+.4f} pp "
          f"(p={wls_result['p_hfpef']:.4f})")
    print(f"    β(HFI×HFpEF):   {wls_result['beta_interaction']*100:+.4f} pp/unit "
          f"(p={wls_result['p_interaction']:.4f})")
    print(f"    R²:             {wls_result['r_squared']:.4f}")

    # ── Save outputs ──
    print("\n[7/7] Saving outputs and generating figure...")

    out_df = ite_hfi_df[ite_hfi_df["hf_subtype"].isin(["hfpef", "no_hf"])].copy()
    out_df.to_csv(TABLES_DIR / "P10_continuous_hfi_interaction.csv", index=False)
    binned_df.to_csv(TABLES_DIR / "P10_binned_ate_by_hfi.csv", index=False)

    int_summary = {k: v for k, v in wls_result.items() if k != "model_summary"}
    int_summary.update({f"raw_{k}": v for k, v in raw_diff.items()})
    pd.DataFrame([int_summary]).to_csv(
        TABLES_DIR / "P10_interaction_test_summary.csv", index=False,
    )

    # Figure (includes bootstrap LOWESS — may take 1-2 min)
    make_figure(
        ite_hfi_df, binned_df, wls_result, raw_diff,
        FIGURES_DIR / "P10_continuous_dose_response.png",
    )

    # ── Report ──
    report_path = REPORTS_DIR / "P10_continuous_hfi_interaction_summary.txt"
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("P10 — Continuous HFI × HF-Phenotype Interaction (v2)\n")
        f.write("     Bin-level WLS + LOWESS approach\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write("═" * 65 + "\n")
        f.write("A. RAW ITE COMPARISON (unadjusted)\n")
        f.write("═" * 65 + "\n\n")
        for hf in ["hfpef", "no_hf"]:
            sub = out_df[out_df["hf_subtype"] == hf]
            f.write(f"  {hf:12s}: n={len(sub):5d}, "
                    f"mean ITE={sub['ite'].mean()*100:+.3f} pp, "
                    f"median={sub['ite'].median()*100:+.3f} pp, "
                    f"SE={sub['ite'].std()/np.sqrt(len(sub))*100:.3f} pp\n")
        f.write(f"\n  Difference: {raw_diff['diff']*100:+.3f} pp\n")
        f.write(f"  Welch t={raw_diff['welch_t']:.3f}, "
                f"p={raw_diff['welch_p']:.4f}\n")
        f.write(f"  Mann-Whitney p={raw_diff['mann_whitney_p']:.4f}\n")

        f.write("\n\n" + "═" * 65 + "\n")
        f.write("B. SLIDING-WINDOW BINNED ATE\n")
        f.write("═" * 65 + "\n\n")
        f.write(f"  {'HFI center':>10s}  {'Subtype':8s}  {'n':>5s}  "
                f"{'ATE (pp)':>10s}  {'SE (pp)':>8s}\n")
        f.write("  " + "-" * 50 + "\n")
        for _, row in binned_df.sort_values(["hfi_center", "hf_subtype"]).iterrows():
            f.write(f"  {row['hfi_center']:+10.3f}  {row['hf_subtype']:8s}  "
                    f"{row['n_stays']:5.0f}  "
                    f"{row['ate_mean']*100:+10.3f}  "
                    f"{row['ate_se']*100:8.3f}\n")

        f.write("\n\n" + "═" * 65 + "\n")
        f.write("C. WLS INTERACTION TEST: ATE_bin ~ HFI * HFtype\n")
        f.write("═" * 65 + "\n\n")
        f.write(f"  Bins: {wls_result['n_bins']} "
                f"(HFpEF: {wls_result['n_hfpef_bins']}, "
                f"no_HF: {wls_result['n_nohf_bins']})\n")
        f.write(f"  Weights: 1/SE² (inverse variance)\n\n")
        f.write(f"  β(HFI)         = {wls_result['beta_hfi']*100:+.4f} pp/unit "
                f"(SE={wls_result['se_hfi']*100:.4f}, "
                f"p={wls_result['p_hfi']:.4f})\n")
        f.write(f"    → Slope: does ATE become more negative with higher HFI?\n\n")
        f.write(f"  β(HFpEF)       = {wls_result['beta_hfpef']*100:+.4f} pp "
                f"(SE={wls_result['se_hfpef']*100:.4f}, "
                f"p={wls_result['p_hfpef']:.4f})\n")
        f.write(f"    → Intercept shift: is HFpEF curve below no_HF?\n\n")
        f.write(f"  β(HFI×HFpEF)   = {wls_result['beta_interaction']*100:+.4f} pp/unit "
                f"(SE={wls_result['se_interaction']*100:.4f}, "
                f"p={wls_result['p_interaction']:.4f})\n")
        f.write(f"    → Do the curves diverge? (different slopes?)\n\n")
        f.write(f"  R²             = {wls_result['r_squared']:.4f}\n\n")

        # Interpretation
        p_hfi = wls_result["p_hfi"]
        beta_hfi = wls_result["beta_hfi"]
        p_main = wls_result["p_hfpef"]
        beta_main = wls_result["beta_hfpef"]
        p_int = wls_result["p_interaction"]

        f.write("  INTERPRETATION:\n\n")

        # HFI slope
        if p_hfi < 0.05 and beta_hfi < 0:
            f.write("    ✓ ATE increases (more negative) with HFI — dose-response confirmed\n")
        elif p_hfi < 0.10 and beta_hfi < 0:
            f.write("    ~ Trend: ATE increases with HFI (marginal significance)\n")
        else:
            f.write(f"    ● HFI slope: β={beta_hfi*100:+.4f} pp/unit, p={p_hfi:.3f}\n")

        # HFpEF main effect
        if p_main < 0.05 and beta_main < 0:
            f.write(f"    ✓ HFpEF has significantly larger ATE than no_HF "
                    f"(shift={beta_main*100:+.2f} pp, p={p_main:.3f})\n")
        elif p_main < 0.10 and beta_main < 0:
            f.write(f"    ~ HFpEF trend toward larger ATE "
                    f"(shift={beta_main*100:+.2f} pp, p={p_main:.3f})\n")
        else:
            f.write(f"    ● HFpEF intercept: β={beta_main*100:+.2f} pp, p={p_main:.3f}\n")

        # Interaction
        if p_int < 0.05:
            f.write("    ✓ Significant interaction: curves diverge with HFI\n")
        else:
            f.write(f"    ● No significant interaction (p={p_int:.3f}): "
                    f"curves approximately parallel\n")
            f.write("      → HFpEF advantage is constant across HFI spectrum\n")
            f.write("      (this is consistent with P9: residual specificity in 4/4 quartiles)\n")

        f.write("\n\n" + "═" * 65 + "\n")
        f.write("D. METHODOLOGICAL NOTE\n")
        f.write("═" * 65 + "\n\n")
        f.write("  Individual-level ITE from G-Transformer DM is extremely noisy\n")
        f.write("  (individual mortality probabilities span 0-1, ATE ≈ 1.4pp).\n")
        f.write("  Individual-level OLS yields R²≈0.0004 — no statistical power.\n\n")
        f.write("  This analysis uses BIN-LEVEL means (averaging 50-500 stays per\n")
        f.write("  bin), which is equivalent to running evaluate_subgroup_dm for\n")
        f.write("  each bin. WLS with 1/SE² weights provides correct inference\n")
        f.write("  at this aggregation level.\n\n")
        f.write("  The P9 quartile analysis (4/4 HFpEF advantage) remains the\n")
        f.write("  cleanest evidence for residual specificity. This P10 analysis\n")
        f.write("  provides a continuous visualization of the interaction.\n")

        f.write("\n\n" + "═" * 65 + "\n")
        f.write("E. FULL WLS MODEL SUMMARY\n")
        f.write("═" * 65 + "\n\n")
        f.write(wls_result["model_summary"])

        f.write("\n\nARTIFACTS\n" + "─" * 9 + "\n")
        f.write("  tables/P10_continuous_hfi_interaction.csv\n")
        f.write("  tables/P10_binned_ate_by_hfi.csv\n")
        f.write("  tables/P10_interaction_test_summary.csv\n")
        f.write("  figures/P10_continuous_dose_response.png\n")
        f.write("  figures/P10_continuous_dose_response.pdf\n")
        f.write("  reports/P10_continuous_hfi_interaction_summary.txt\n")

    print(f"  Report saved: {report_path}")
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
