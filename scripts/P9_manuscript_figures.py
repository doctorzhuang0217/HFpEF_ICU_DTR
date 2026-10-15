#!/usr/bin/env python3
"""P9 — Manuscript-Quality Figures (PDF vector) for Hemodynamic Fragility Analysis

Generates publication-ready vector PDF figures for P8/P9 results.
Style matches 95b_generate_figures_tables.py conventions.

Outputs (main text):
  - manuscript/figures_pdf/Figure5.pdf — 3-panel composite:
      (A) HFI distribution by HF subtype
      (B) ATE dose-response by HFI quartile
      (C) HFI-matched HFpEF vs no_HF residual specificity

Outputs (supplementary):
  - manuscript/figures_pdf/sFigure8.pdf — P8 action pathway decomposition
  - manuscript/figures_pdf/sFigure9.pdf — P9 hypotensive episode histogram

Compute: C1 (reads CSV only, no model)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType fonts in PDF (editable)
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
TABLES_DIR = ROOT / "tables"
PDF_DIR = ROOT / "manuscript" / "figures_pdf"
PDF_DIR.mkdir(parents=True, exist_ok=True)

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
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.2,
})

# ── Publication-ready label mapping ──
_HF = {"hfpef": "HFpEF", "no_hf": "No HF", "hfref": "HFrEF",
       "hfmref": "HFmrEF", "hf_ef_unknown": "HF (EF unknown)"}

# ── Color palette ──
C_HFPEF = "#C0392B"   # dark red
C_NOHF = "#2980B9"    # dark blue
C_HFREF = "#27AE60"   # green
C_HFMREF = "#E67E22"  # orange
C_Q1 = "#27AE60"
C_Q2 = "#F1C40F"
C_Q3 = "#E67E22"
C_Q4 = "#C0392B"
QUARTILE_COLORS = [C_Q1, C_Q2, C_Q3, C_Q4]


def fig5_composite():
    """Figure 5: Hemodynamic Fragility — 3-panel composite (main text).

    Panel A: HFI distribution (KDE) by HF subtype
    Panel B: ATE dose-response by HFI quartile (overall)
    Panel C: HFI-matched HFpEF vs no_HF (residual specificity)
    """
    print("[Figure 5] Generating 3-panel composite...")

    # Load data
    dist_df = pd.read_csv(TABLES_DIR / "P9_hfi_distribution_by_hf_subtype.csv")
    ate_df = pd.read_csv(TABLES_DIR / "P9_ate_by_hfi_quartile.csv")
    matched_df = pd.read_csv(TABLES_DIR / "P9_hfi_matched_hfpef_vs_nohf.csv")

    overall = ate_df[ate_df["hf_subtype"] == "ALL"].sort_values("quartile")
    quartiles = ["Q1_stable", "Q2", "Q3", "Q4_fragile"]
    q_labels = ["Q1\n(stable)", "Q2", "Q3", "Q4\n(fragile)"]

    # ── Create figure ──
    fig = plt.figure(figsize=(7.2, 7.8))  # ~single column width for journals
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.1], hspace=0.35, wspace=0.35)

    # ── Panel A: HFI distribution ──
    ax_a = fig.add_subplot(gs[0, :])  # full width top
    ax_a.text(-0.08, 1.05, "A", transform=ax_a.transAxes,
              fontsize=14, fontweight="bold", va="top")

    # Build synthetic KDE from summary stats (mean ± std, assume normal)
    x_range = np.linspace(-2.5, 3.5, 300)
    for _, row in dist_df.iterrows():
        grp = row["hf_subtype"]
        mu = row["hfi_composite_mean"]
        sigma = row["hfi_composite_std"]
        if grp == "hfpef":
            color, lw, ls, zorder = C_HFPEF, 2.0, "-", 10
        elif grp == "no_hf":
            color, lw, ls, zorder = C_NOHF, 2.0, "-", 9
        elif grp == "hfref":
            color, lw, ls, zorder = C_HFREF, 1.2, "--", 5
        elif grp == "hfmref":
            color, lw, ls, zorder = C_HFMREF, 1.2, "--", 5
        else:
            continue
        kde_y = stats.norm.pdf(x_range, mu, sigma)
        label = f"{_HF.get(grp, grp)} (n={int(row['n_stays'])})"
        ax_a.plot(x_range, kde_y, color=color, linewidth=lw,
                  linestyle=ls, label=label, zorder=zorder)
        if grp in ("hfpef", "no_hf"):
            ax_a.fill_between(x_range, kde_y, alpha=0.12, color=color, zorder=zorder-1)

    # Annotation: effect size
    hfpef_mean = dist_df.loc[dist_df["hf_subtype"] == "hfpef", "hfi_composite_mean"].values[0]
    nohf_mean = dist_df.loc[dist_df["hf_subtype"] == "no_hf", "hfi_composite_mean"].values[0]
    ax_a.annotate("", xy=(hfpef_mean, 0.42), xytext=(nohf_mean, 0.42),
                  arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.2),
                  zorder=20)
    ax_a.text((hfpef_mean + nohf_mean) / 2, 0.44,
              "Cohen's d = 0.33\np < 10⁻⁷",
              ha="center", va="bottom", fontsize=7.5, color="#333333",
              style="italic", zorder=20,
              bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                        edgecolor="none", alpha=0.85))

    ax_a.set_xlabel("Hemodynamic Fragility Index (composite z-score)")
    ax_a.set_ylabel("Density")
    ax_a.set_title("HFI Distribution by HF Subtype", fontsize=10, fontweight="bold")
    ax_a.legend(loc="upper right", framealpha=0.9, edgecolor="none")
    ax_a.set_xlim(-2.5, 3.5)
    ax_a.set_ylim(0, 0.62)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # ── Panel B: Dose-response ──
    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.text(-0.15, 1.05, "B", transform=ax_b.transAxes,
              fontsize=14, fontweight="bold", va="top")

    x = np.arange(4)
    ates = overall["ate"].values * 100
    bars = ax_b.bar(x, ates, color=QUARTILE_COLORS, edgecolor="white",
                    linewidth=0.8, width=0.7, zorder=3)

    for bar, ate_val in zip(bars, ates):
        ax_b.text(bar.get_x() + bar.get_width() / 2, ate_val - 0.06,
                  f"{ate_val:.1f}", ha="center", va="top",
                  fontsize=8, fontweight="bold", color="white")

    # Trend arrow
    ax_b.annotate("", xy=(3.3, ates[-1] - 0.1), xytext=(3.3, ates[0] + 0.1),
                  arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax_b.text(3.55, (ates[0] + ates[-1]) / 2, "ρ = −1.0\np < 0.001",
              fontsize=7, color="gray", ha="left", va="center", style="italic")

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(q_labels, fontsize=8)
    ax_b.set_ylabel("ATE (percentage points)")
    ax_b.set_title("Dose-Response:\nATE by HFI Quartile", fontsize=10, fontweight="bold")
    ax_b.axhline(y=0, color="black", linewidth=0.5, zorder=1)
    ax_b.set_ylim(min(ates) - 0.4, 0.15)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.set_xlim(-0.5, 4.2)

    # ── Panel C: HFI-matched specificity ──
    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.text(-0.15, 1.05, "C", transform=ax_c.transAxes,
              fontsize=14, fontweight="bold", va="top")

    width = 0.35
    hfpef_ates = []
    nohf_ates = []
    hfpef_ns = []
    nohf_ns = []
    ratios = []

    for q in quartiles:
        hr = matched_df[(matched_df["quartile"] == q) & (matched_df["hf_subtype"] == "hfpef")]
        nr = matched_df[(matched_df["quartile"] == q) & (matched_df["hf_subtype"] == "no_hf")]
        h_ate = hr["ate"].values[0] * 100
        n_ate = nr["ate"].values[0] * 100
        hfpef_ates.append(h_ate)
        nohf_ates.append(n_ate)
        hfpef_ns.append(int(hr["n_stays"].values[0]))
        nohf_ns.append(int(nr["n_stays"].values[0]))
        ratios.append(hr["ate"].values[0] / nr["ate"].values[0])

    bars_h = ax_c.bar(x - width / 2, hfpef_ates, width, label="HFpEF",
                      color=C_HFPEF, alpha=0.85, edgecolor="white", linewidth=0.5, zorder=3)
    bars_n = ax_c.bar(x + width / 2, nohf_ates, width, label="No HF",
                      color=C_NOHF, alpha=0.85, edgecolor="white", linewidth=0.5, zorder=3)

    # Ratio annotations — place below bars with enough clearance
    for i, (q, ratio) in enumerate(zip(quartiles, ratios)):
        y_pos = min(hfpef_ates[i], nohf_ates[i]) - 0.55
        ax_c.text(i, y_pos, f"{ratio:.1f}\u00d7",
                  ha="center", va="top", fontsize=7.5, fontweight="bold",
                  color="#555555", zorder=10,
                  bbox=dict(boxstyle="round,pad=0.18", facecolor="lightyellow",
                            edgecolor="#cccccc", linewidth=0.5, alpha=1.0))

    # Sample size annotations (compact)
    for i in range(4):
        ax_c.text(i, 0.08, f"n={hfpef_ns[i]}/{nohf_ns[i]}",
                  ha="center", va="bottom", fontsize=6.5, color="#666666")

    ax_c.axhline(y=0, color="black", linewidth=0.5, zorder=1)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(q_labels, fontsize=8)
    ax_c.set_ylabel("ATE (percentage points)")
    ax_c.set_title("Residual Specificity:\nHFpEF vs No HF (HFI-matched)",
                    fontsize=10, fontweight="bold")
    ax_c.legend(loc="lower left", framealpha=0.9, edgecolor="none")
    all_ates = hfpef_ates + nohf_ates
    ax_c.set_ylim(min(all_ates) - 1.3, 0.35)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # ── Save ──
    path = PDF_DIR / "Figure5.pdf"
    fig.savefig(path, format="pdf", bbox_inches="tight", dpi=300, facecolor="white")
    fig.savefig(str(path).replace(".pdf", ".png"), format="png",
                bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def sfig8_action_decomposition():
    """sFigure 8: P8 Action pathway decomposition (supplementary)."""
    print("[sFigure 8] P8 action pathway decomposition...")

    p8 = pd.read_csv(TABLES_DIR / "P8_action_mechanism_decomposition.csv")
    ns = p8[p8["stratum"] == "non_shock"]

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))

    pathway_order = ["shock_resuscitation", "decongestion", "maintenance"]
    pw_labels = ["Shock\nresuscitation", "Decongestion", "Maintenance"]

    for ax_idx, (title, rule_col, obs_col) in enumerate([
        ("Rule Policy (recommended)", "rule_pct_", "rule_pct_"),
        ("Observed (clinician actual)", "obs_pct_", "obs_pct_"),
    ]):
        ax = axes[ax_idx]
        x = np.arange(len(pathway_order))
        width = 0.35
        prefix = rule_col if ax_idx == 0 else obs_col

        hfpef_vals = []
        nohf_vals = []
        for pw in ["shock", "decongest", "maintenance"]:
            col = f"{prefix}{pw}"
            hr = ns[ns["hf_subtype"] == "hfpef"][col].values[0] * 100
            nr = ns[ns["hf_subtype"] == "no_hf"][col].values[0] * 100
            hfpef_vals.append(hr)
            nohf_vals.append(nr)

        ax.bar(x - width / 2, hfpef_vals, width, label="HFpEF",
               color=C_HFPEF, alpha=0.85, edgecolor="white")
        ax.bar(x + width / 2, nohf_vals, width, label="No HF",
               color=C_NOHF, alpha=0.85, edgecolor="white")

        for i, (h, n) in enumerate(zip(hfpef_vals, nohf_vals)):
            if h > 1:
                ax.text(i - width / 2, h + 0.5, f"{h:.1f}%", ha="center",
                        fontsize=7, color=C_HFPEF)
            if n > 1:
                ax.text(i + width / 2, n + 0.5, f"{n:.1f}%", ha="center",
                        fontsize=7, color=C_NOHF)

        ax.set_xticks(x)
        ax.set_xticklabels(pw_labels, fontsize=8)
        ax.set_ylabel("Proportion (%)")
        ax.set_title(title, fontsize=9, fontweight="bold")
        if ax_idx == 0:
            ax.legend(loc="upper center", fontsize=7, framealpha=0.9, ncol=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Treatment Pathway Distribution \u2014 Non-Shock Stratum",
                 fontsize=10, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = PDF_DIR / "sFigure8.pdf"
    fig.savefig(path, format="pdf", bbox_inches="tight", dpi=300, facecolor="white")
    fig.savefig(str(path).replace(".pdf", ".png"), format="png",
                bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def sfig9_hypotensive_episodes():
    """sFigure 9: Hypotensive episode histogram (supplementary)."""
    print("[sFigure 9] Hypotensive episode distribution...")

    # We reconstruct from summary stats since we don't have raw HFI per-stay
    # on this machine. Use the matched CSV to get n per quartile.
    dist_df = pd.read_csv(TABLES_DIR / "P9_hfi_distribution_by_hf_subtype.csv")

    fig, ax = plt.subplots(figsize=(5, 3.5))

    # Summary bar chart: key HFI metrics comparison
    metrics = [
        ("Mean hypotensive\nbins (/12)", "n_hypo_bins_mean"),
        ("Any hypotension\n(%)", "n_hypo_bins_pct_any"),
        ("MAP CV\n(×100)", "map_cv_mean"),
    ]

    x = np.arange(len(metrics))
    width = 0.3

    hfpef_row = dist_df[dist_df["hf_subtype"] == "hfpef"].iloc[0]
    nohf_row = dist_df[dist_df["hf_subtype"] == "no_hf"].iloc[0]

    hfpef_vals = [
        hfpef_row["n_hypo_bins_mean"],
        hfpef_row["n_hypo_bins_pct_any"] * 100,
        hfpef_row["map_cv_mean"] * 100,
    ]
    nohf_vals = [
        nohf_row["n_hypo_bins_mean"],
        nohf_row["n_hypo_bins_pct_any"] * 100,
        nohf_row["map_cv_mean"] * 100,
    ]

    bars_h = ax.bar(x - width / 2, hfpef_vals, width, label="HFpEF (n=304)",
                    color=C_HFPEF, alpha=0.85, edgecolor="white")
    bars_n = ax.bar(x + width / 2, nohf_vals, width, label="No HF (n=4,647)",
                    color=C_NOHF, alpha=0.85, edgecolor="white")

    for bars in [bars_h, bars_n]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=7.5)

    # Significance stars
    sigs = ["***", "***", "***"]  # all p < 0.001
    for i, sig in enumerate(sigs):
        max_h = max(hfpef_vals[i], nohf_vals[i])
        ax.text(i, max_h + 2.5, sig, ha="center", va="bottom",
                fontsize=9, color="#555555")

    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics], fontsize=8)
    ax.set_ylabel("Value")
    ax.set_title("Hemodynamic Fragility Metrics:\nHFpEF vs No HF",
                 fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = PDF_DIR / "sFigure9.pdf"
    fig.savefig(path, format="pdf", bbox_inches="tight", dpi=300, facecolor="white")
    fig.savefig(str(path).replace(".pdf", ".png"), format="png",
                bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def main():
    print("=" * 72)
    print("P9 — Manuscript-Quality Figures (PDF vector)")
    print("=" * 72)

    paths = []
    paths.append(fig5_composite())
    paths.append(sfig8_action_decomposition())
    paths.append(sfig9_hypotensive_episodes())

    print(f"\n{'=' * 72}")
    print(f"Generated {len(paths)} figures:")
    for p in paths:
        print(f"  {p}")
    print("=" * 72)


if __name__ == "__main__":
    main()
