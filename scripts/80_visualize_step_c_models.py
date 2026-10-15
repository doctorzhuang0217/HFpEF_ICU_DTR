#!/usr/bin/env python3
"""80_visualize — Step C Model Training & Performance Visualizations

Protocol: v3.1, Step 80 (visualization supplement)
Compute: C1

Purpose
-------
Generate 4 figures that visually summarize Step 80 model training and
off-policy evaluation (OPE) results across all 5 deep DTR models
(80a–80e) plus the HF-subtype × shock interaction analysis (80g).

Figures
-------
  F1. Training loss curves (5 models, train/val)
  F2. Cross-model ATE forest plot with 95% bootstrap CIs
  F3. Shock vs non-shock subgroup ATE comparison (grouped bars)
  F4. HF-subtype × shock interaction heatmap (80g, 2×5 grid)

Inputs
------
- tables/80{a,b,c,d,e}_*_training_history.csv
- tables/80{a,b,c,d,e}_*_policy_values.csv
- tables/80{a,b,c,d,e}_*_bootstrap_ci.csv
- tables/80{a,b,c,d,e}_*_subgroup_results.csv
- tables/80c_gtransformer_rollout_ow1_*.csv
- tables/80g_hf_subtype_x_shock_interaction.csv

Outputs
-------
- figures/80_f1_training_loss_curves.png
- figures/80_f2_cross_model_ate_forest.png
- figures/80_f3_subgroup_shock_comparison.png
- figures/80_f4_interaction_heatmap.png
- reports/80_visualization_manifest.txt
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────
ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
TABLES_DIR = ROOT / "tables"
FIGURES_DIR = ROOT / "figures"
REPORTS_DIR = ROOT / "reports"

for d in [FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("80_viz")

# ── Plot defaults ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "font.family": "sans-serif",
})

# ── Color palette ─────────────────────────────────────────────────────
MODEL_COLORS = {
    "80c G-Transformer": "#d6604d",   # primary — warm red
    "80a GNet":          "#4393c3",   # confirmatory — blue
    "80b ACTIN":         "#5ab4ac",   # confirmatory — teal
    "80d CT-Legacy":     "#998ec3",   # ablation — purple
    "80e Mamba-CDSP":    "#f1a340",   # optional — orange
}

ESTIMATOR_MARKERS = {
    "DM":  "o",
    "DR":  "s",
    "FQE": "D",
    "WIS": "^",
}


# ══════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════

def _safe_read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        log.warning("  Missing: %s", path)
        return None
    return pd.read_csv(path)


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 1: Training Loss Curves
# ══════════════════════════════════════════════════════════════════════

def fig_f1_training_loss_curves():
    """F1: Training loss curves for all 5 models (train + val)."""
    log.info("  F1: Training loss curves...")

    models = [
        ("80c G-Transformer", "80c_gtransformer_rollout_ow1_training_history.csv"),
        ("80a GNet",          "80a_gnet_training_history.csv"),
        ("80b ACTIN",         "80b_actin_training_history.csv"),
        ("80d CT-Legacy",     "80d_ct_legacy_training_history.csv"),
        ("80e Mamba-CDSP",    "80e_mamba_cdsp_training_history.csv"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=False)

    for i, (name, fname) in enumerate(models):
        ax = axes[i]
        df = _safe_read(TABLES_DIR / fname)
        if df is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
            ax.set_title(name, fontsize=11, fontweight="bold")
            continue

        color = MODEL_COLORS[name]
        epochs = df["epoch"]

        # Normalize losses to [0, 1] range for visual comparison
        # (different models have very different loss scales)
        train_loss = df["train_loss"]
        val_loss = df["val_loss"]

        ax.plot(epochs, train_loss, color=color, linewidth=1.5,
                alpha=0.7, label="Train")
        ax.plot(epochs, val_loss, color=color, linewidth=2.0,
                linestyle="--", label="Val")

        # Mark best val epoch
        best_epoch = epochs.iloc[val_loss.values.argmin()]
        best_val = val_loss.min()
        ax.axvline(best_epoch, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
        ax.plot(best_epoch, best_val, "*", color=color, markersize=12, zorder=5)

        ax.set_title(name, fontsize=11, fontweight="bold", color=color)
        ax.set_xlabel("Epoch")
        if i == 0:
            ax.set_ylabel("Loss")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.2)

        # Add annotation for best epoch
        ax.annotate(f"best={best_epoch}",
                    xy=(best_epoch, best_val),
                    xytext=(best_epoch + 5, best_val + (val_loss.max() - val_loss.min()) * 0.1),
                    fontsize=7, color="gray",
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.5))

    fig.suptitle("Step 80: Model Training Convergence", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    path = FIGURES_DIR / "80_f1_training_loss_curves.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: %s", path)
    return path


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 2: Cross-Model ATE Forest Plot
# ══════════════════════════════════════════════════════════════════════

def fig_f2_cross_model_forest():
    """F2: Forest plot — ATE point estimates + 95% CI by model × estimator."""
    log.info("  F2: Cross-model ATE forest plot...")

    # Collect all policy values and bootstrap CIs
    entries = []  # list of dicts: label, model_name, estimator, ate, ci_lo, ci_hi, color

    # --- 75a classical baseline (WIS) ---
    # Hardcoded from documented Step 75a results: ATE=-4.87pp [-5.68, -4.05].
    entries.append({
        "label": "75a MSM-IPTW (WIS)",
        "model_name": "75a",
        "estimator": "WIS",
        "ate": -4.87,
        "ci_lo": -5.68,
        "ci_hi": -4.05,
        "color": "#2166ac",
    })

    # --- 80c G-Transformer (primary, rollout ow=1) ---
    pv_80c = _safe_read(TABLES_DIR / "80c_gtransformer_rollout_ow1_policy_values.csv")
    ci_80c = _safe_read(TABLES_DIR / "80c_gtransformer_rollout_ow1_bootstrap_ci.csv")

    if pv_80c is not None and ci_80c is not None:
        for _, row in pv_80c.iterrows():
            est = row["estimator"]
            ate_pp = row["ate"] * 100
            ci_row = ci_80c[ci_80c["estimator"] == est]
            ci_lo = ci_row["ate_ci_lo"].values[0] * 100 if len(ci_row) > 0 else None
            ci_hi = ci_row["ate_ci_hi"].values[0] * 100 if len(ci_row) > 0 else None
            entries.append({
                "label": f"80c G-Transformer ({est})",
                "model_name": "80c G-Transformer",
                "estimator": est,
                "ate": ate_pp,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "color": MODEL_COLORS["80c G-Transformer"],
            })

    # --- 80a GNet ---
    pv_80a = _safe_read(TABLES_DIR / "80a_gnet_policy_values.csv")
    ci_80a = _safe_read(TABLES_DIR / "80a_gnet_bootstrap_ci.csv")

    if pv_80a is not None and ci_80a is not None:
        for _, row in pv_80a.iterrows():
            est = row["estimator"]
            ate_pp = row["ate"] * 100
            ci_row = ci_80a[ci_80a["estimator"] == est]
            ci_lo = ci_row["ate_ci_lo"].values[0] * 100 if len(ci_row) > 0 else None
            ci_hi = ci_row["ate_ci_hi"].values[0] * 100 if len(ci_row) > 0 else None
            entries.append({
                "label": f"80a GNet ({est})",
                "model_name": "80a GNet",
                "estimator": est,
                "ate": ate_pp,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "color": MODEL_COLORS["80a GNet"],
            })

    # --- 80b ACTIN ---
    pv_80b = _safe_read(TABLES_DIR / "80b_actin_policy_values.csv")
    ci_80b = _safe_read(TABLES_DIR / "80b_actin_bootstrap_ci.csv")

    if pv_80b is not None and ci_80b is not None:
        for _, row in pv_80b.iterrows():
            est = row["estimator"]
            ate_pp = row["ate"] * 100
            ci_row = ci_80b[ci_80b["estimator"] == est]
            ci_lo = ci_row["ate_ci_lo"].values[0] * 100 if len(ci_row) > 0 else None
            ci_hi = ci_row["ate_ci_hi"].values[0] * 100 if len(ci_row) > 0 else None
            entries.append({
                "label": f"80b ACTIN ({est})",
                "model_name": "80b ACTIN",
                "estimator": est,
                "ate": ate_pp,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "color": MODEL_COLORS["80b ACTIN"],
            })

    # --- 80d CT-Legacy (FQE only) ---
    pv_80d = _safe_read(TABLES_DIR / "80d_ct_legacy_policy_values.csv")
    ci_80d = _safe_read(TABLES_DIR / "80d_ct_legacy_bootstrap_ci.csv")

    if pv_80d is not None and ci_80d is not None:
        for _, row in pv_80d.iterrows():
            est = row["estimator"]
            ate_pp = row["ate"] * 100
            ci_row = ci_80d[ci_80d["estimator"] == est]
            ci_lo = ci_row["ate_ci_lo"].values[0] * 100 if len(ci_row) > 0 else None
            ci_hi = ci_row["ate_ci_hi"].values[0] * 100 if len(ci_row) > 0 else None
            entries.append({
                "label": f"80d CT-Legacy ({est})",
                "model_name": "80d CT-Legacy",
                "estimator": est,
                "ate": ate_pp,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "color": MODEL_COLORS["80d CT-Legacy"],
            })

    # --- 80e Mamba-CDSP (FQE only) ---
    pv_80e = _safe_read(TABLES_DIR / "80e_mamba_cdsp_policy_values.csv")
    ci_80e = _safe_read(TABLES_DIR / "80e_mamba_cdsp_bootstrap_ci.csv")

    if pv_80e is not None and ci_80e is not None:
        for _, row in pv_80e.iterrows():
            est = row["estimator"]
            ate_pp = row["ate"] * 100
            ci_row = ci_80e[ci_80e["estimator"] == est]
            ci_lo = ci_row["ate_ci_lo"].values[0] * 100 if len(ci_row) > 0 else None
            ci_hi = ci_row["ate_ci_hi"].values[0] * 100 if len(ci_row) > 0 else None
            entries.append({
                "label": f"80e Mamba-CDSP ({est})",
                "model_name": "80e Mamba-CDSP",
                "estimator": est,
                "ate": ate_pp,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "color": MODEL_COLORS["80e Mamba-CDSP"],
            })

    if not entries:
        log.warning("  No data for F2")
        return None

    # --- Plot ---
    n = len(entries)
    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.55)))

    for i, e in enumerate(entries):
        marker = ESTIMATOR_MARKERS.get(e["estimator"], "o")
        ax.plot(e["ate"], i, marker, color=e["color"], markersize=9, zorder=3,
                markeredgecolor="white", markeredgewidth=0.5)
        if e["ci_lo"] is not None and e["ci_hi"] is not None:
            ax.plot([e["ci_lo"], e["ci_hi"]], [i, i], "-",
                    color=e["color"], linewidth=2.5, alpha=0.7, zorder=2,
                    solid_capstyle="round")
            # ATE annotation
            ax.annotate(f"{e['ate']:.2f}pp",
                        xy=(e["ate"], i), xytext=(8, 0),
                        textcoords="offset points", fontsize=8,
                        color=e["color"], va="center", fontweight="bold")

    ax.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6, zorder=1)

    # Shaded region for primary estimate range
    primary_ci_lo = None
    primary_ci_hi = None
    for e in entries:
        if e["label"] == "80c G-Transformer (DM)":
            primary_ci_lo = e["ci_lo"]
            primary_ci_hi = e["ci_hi"]
    if primary_ci_lo is not None:
        ax.axvspan(primary_ci_lo, primary_ci_hi, alpha=0.08, color=MODEL_COLORS["80c G-Transformer"],
                   zorder=0, label="Primary 95% CI")

    labels = [e["label"] for e in entries]
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("ATE (percentage points)", fontsize=12)
    ax.set_title("Step 80: Cross-Model ATE Comparison\n(negative = rule policy reduces 28-day mortality)",
                 fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)

    # Add separator line between 75a and 80x models
    ax.axhline(0.5, color="gray", linewidth=0.5, alpha=0.4, linestyle="-")

    # Estimator legend
    est_handles = [
        plt.Line2D([0], [0], marker=m, color="gray", linestyle="", markersize=7, label=est)
        for est, m in ESTIMATOR_MARKERS.items()
    ]
    ax.legend(handles=est_handles, title="Estimator", loc="lower left",
              fontsize=9, title_fontsize=10, framealpha=0.9)

    fig.tight_layout()
    path = FIGURES_DIR / "80_f2_cross_model_ate_forest.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: %s", path)
    return path


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 3: Shock vs Non-Shock Subgroup Comparison
# ══════════════════════════════════════════════════════════════════════

def fig_f3_subgroup_shock_comparison():
    """F3: Grouped bar chart — ATE by model in shock vs non-shock strata."""
    log.info("  F3: Subgroup shock comparison...")

    model_files = [
        ("80c G-Transformer", "80c_gtransformer_rollout_ow1_subgroup_results.csv"),
        ("80a GNet",          "80a_gnet_subgroup_results.csv"),
        ("80b ACTIN",         "80b_actin_subgroup_results.csv"),
        ("80d CT-Legacy",     "80d_ct_legacy_subgroup_results.csv"),
        ("80e Mamba-CDSP",    "80e_mamba_cdsp_subgroup_results.csv"),
    ]

    models = []
    non_shock_ate = []
    shock_ate = []

    for name, fname in model_files:
        df = _safe_read(TABLES_DIR / fname)
        if df is None:
            continue
        pheno = df[df["stratifier"] == "rule_phenotype_2class"]
        ns = pheno[pheno["group"] == "non_shock"]
        sh = pheno[pheno["group"] == "shock_predominant"]
        if len(ns) > 0 and len(sh) > 0:
            models.append(name)
            non_shock_ate.append(ns["ate"].values[0] * 100)
            shock_ate.append(sh["ate"].values[0] * 100)

    if not models:
        log.warning("  No subgroup data for F3")
        return None

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_ns = ax.bar(x - width / 2, non_shock_ate, width, label="Non-Shock",
                     color="#4393c3", edgecolor="white", linewidth=0.5, alpha=0.9)
    bars_sh = ax.bar(x + width / 2, shock_ate, width, label="Shock-Predominant",
                     color="#d6604d", edgecolor="white", linewidth=0.5, alpha=0.9)

    # Value labels on bars
    for bars in [bars_ns, bars_sh]:
        for bar in bars:
            val = bar.get_height()
            y_offset = -0.3 if val < 0 else 0.1
            ax.text(bar.get_x() + bar.get_width() / 2, val + y_offset,
                    f"{val:.2f}", ha="center", va="top" if val < 0 else "bottom",
                    fontsize=8, fontweight="bold")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.set_ylabel("ATE (percentage points)", fontsize=12)
    ax.set_title("Step 80: Treatment Effect by Shock Stratum\n"
                 "(shock-predominant patients show 2-4× larger ATE across all models)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="lower left")
    ax.grid(axis="y", alpha=0.25)

    # Add ratio annotations
    for i in range(len(models)):
        if non_shock_ate[i] != 0:
            ratio = shock_ate[i] / non_shock_ate[i]
            y_pos = min(shock_ate[i], non_shock_ate[i]) - 0.5
            ax.annotate(f"{ratio:.1f}×",
                        xy=(x[i], y_pos), fontsize=9, ha="center",
                        color="#666666", fontstyle="italic")

    fig.tight_layout()
    path = FIGURES_DIR / "80_f3_subgroup_shock_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: %s", path)
    return path


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 4: HF-Subtype × Shock Interaction Heatmap
# ══════════════════════════════════════════════════════════════════════

def fig_f4_interaction_heatmap():
    """F4: Heatmap of ATE across HF subtypes × shock strata (80g, primary 80c DM)."""
    log.info("  F4: HF-subtype × shock interaction heatmap...")

    df = _safe_read(TABLES_DIR / "80g_hf_subtype_x_shock_interaction.csv")
    if df is None:
        return None

    # Filter: primary model (80c), DM estimator, non-ALL strata
    primary = df[
        (df["model"] == "80c_gtransformer_rollout_ow1") &
        (df["estimator"] == "DM") &
        (df["shock_stratum"] != "ALL")
    ].copy()

    if len(primary) == 0:
        log.warning("  No primary DM data in 80g")
        return None

    # Pivot: rows=hf_subtype, cols=shock_stratum
    pivot_ate = primary.pivot(index="hf_subtype", columns="shock_stratum", values="ate")
    pivot_n = primary.pivot(index="hf_subtype", columns="shock_stratum", values="n_stays")

    # Reorder: HFpEF first, then others, no_hf last
    subtype_order = ["hfpef", "hfref", "hfmref", "hf_ef_unknown", "no_hf"]
    subtype_order = [s for s in subtype_order if s in pivot_ate.index]
    strata_order = ["non_shock", "shock_predominant"]
    strata_order = [s for s in strata_order if s in pivot_ate.columns]

    pivot_ate = pivot_ate.loc[subtype_order, strata_order]
    pivot_n = pivot_n.loc[subtype_order, strata_order]

    # Convert to percentage points
    ate_pp = pivot_ate * 100

    # Pretty labels
    subtype_labels = {
        "hfpef": "HFpEF",
        "hfref": "HFrEF",
        "hfmref": "HFmrEF",
        "hf_ef_unknown": "HF-EF Unknown",
        "no_hf": "No HF",
    }
    strata_labels = {
        "non_shock": "Non-Shock",
        "shock_predominant": "Shock-Predominant",
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    # Custom colormap: darker = larger (more negative) ATE
    # Use reversed RdBu so blue = more beneficial (more negative ATE)
    vmin = ate_pp.values.min()
    vmax = 0  # zero line

    im = ax.imshow(ate_pp.values, cmap="RdYlBu", aspect="auto",
                   vmin=vmin - 0.5, vmax=0.5)

    # Add text annotations (ATE value + sample size)
    for i in range(len(subtype_order)):
        for j in range(len(strata_order)):
            ate_val = ate_pp.iloc[i, j]
            n_val = int(pivot_n.iloc[i, j])
            text_color = "white" if abs(ate_val) > 2.0 else "black"
            ax.text(j, i, f"{ate_val:.2f}pp\n(n={n_val})",
                    ha="center", va="center", fontsize=11,
                    fontweight="bold", color=text_color)

    # HFpEF highlight border
    hfpef_idx = subtype_order.index("hfpef") if "hfpef" in subtype_order else None
    if hfpef_idx is not None:
        rect = plt.Rectangle((-0.5, hfpef_idx - 0.5), len(strata_order), 1,
                              linewidth=3, edgecolor="#d6604d", facecolor="none",
                              linestyle="-", zorder=5)
        ax.add_patch(rect)

    ax.set_xticks(range(len(strata_order)))
    ax.set_xticklabels([strata_labels.get(s, s) for s in strata_order],
                       fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(subtype_order)))
    ax.set_yticklabels([subtype_labels.get(s, s) for s in subtype_order],
                       fontsize=12, fontweight="bold")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("ATE (percentage points)", fontsize=11)

    ax.set_title("Step 80g: HF-Subtype × Shock Interaction\n"
                 "(80c G-Transformer, DM estimator — HFpEF row highlighted)",
                 fontsize=13, fontweight="bold")

    # Add ratio annotation below
    if hfpef_idx is not None and "no_hf" in subtype_order:
        nohf_idx = subtype_order.index("no_hf")
        ns_col = strata_order.index("non_shock") if "non_shock" in strata_order else 0
        hfpef_ns = ate_pp.iloc[hfpef_idx, ns_col]
        nohf_ns = ate_pp.iloc[nohf_idx, ns_col]
        if nohf_ns != 0:
            ratio = hfpef_ns / nohf_ns
            fig.text(0.5, -0.02,
                     f"Non-shock HFpEF/no-HF ratio: {ratio:.1f}× "
                     f"(HFpEF={hfpef_ns:.2f}pp vs no-HF={nohf_ns:.2f}pp)",
                     ha="center", fontsize=11, fontstyle="italic", color="#d6604d")

    fig.tight_layout()
    path = FIGURES_DIR / "80_f4_interaction_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: %s", path)
    return path


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info("Step 80: Model Visualization")
    log.info("=" * 60)

    outputs = []

    log.info("[1/4] Training loss curves")
    p = fig_f1_training_loss_curves()
    if p:
        outputs.append(str(p))

    log.info("[2/4] Cross-model ATE forest plot")
    p = fig_f2_cross_model_forest()
    if p:
        outputs.append(str(p))

    log.info("[3/4] Subgroup shock comparison")
    p = fig_f3_subgroup_shock_comparison()
    if p:
        outputs.append(str(p))

    log.info("[4/4] HF-subtype × shock interaction heatmap")
    p = fig_f4_interaction_heatmap()
    if p:
        outputs.append(str(p))

    # Write manifest
    manifest_path = REPORTS_DIR / "80_visualization_manifest.txt"
    with open(manifest_path, "w") as f:
        f.write(f"Step 80 Visualization Manifest\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"Figures generated: {len(outputs)}\n\n")
        for o in outputs:
            f.write(f"  {o}\n")
        f.write(f"\nWall time: {time.time() - t0:.1f}s\n")

    log.info("=" * 60)
    log.info("Done — %d figures generated in %.1fs", len(outputs), time.time() - t0)
    log.info("Manifest: %s", manifest_path)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
