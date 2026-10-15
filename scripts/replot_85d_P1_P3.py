#!/usr/bin/env python
"""Lightweight replot script for 85d, P1, P3 figures.

Reads pre-computed CSVs and regenerates ONLY the PNG/PDF figures
with corrected titles (no internal step-ID prefixes).
Runs on login node — no model inference or dataset loading required.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = ROOT / "tables"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

TRIMMING_BRACKETS = [
    (1.0, 99.0), (2.5, 97.5), (5.0, 95.0), (10.0, 90.0), (20.0, 80.0),
]


def replot_85d():
    """Regenerate 85d_positivity_grid_stability.png/pdf."""
    print("[85d] Replotting positivity grid stability ...")
    csv_path = TABLES_DIR / "85d_positivity_grid.csv"
    if not csv_path.exists():
        print(f"  SKIP: {csv_path} not found")
        return
    results_df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 6))

    bracket_order = ["none"] + [f"[{lo},{hi}]" for lo, hi in TRIMMING_BRACKETS]
    x_map = {b: i for i, b in enumerate(bracket_order)}

    for subgroup, marker, color in [
        ("ALL", "o", "#2196F3"),
        ("hfpef", "s", "#E91E63"),
        ("no_hf", "^", "#4CAF50"),
    ]:
        sub = results_df[results_df["subgroup"] == subgroup].copy()
        if sub.empty:
            continue
        sub = sub[sub["bracket"].isin(x_map)].copy()
        sub["x_pos"] = sub["bracket"].map(x_map)
        sub = sub.sort_values("x_pos")

        x_pos = sub["x_pos"].values
        ates = sub["ate"].values
        ci_lo = sub["ate_ci_lo"].values
        ci_hi = sub["ate_ci_hi"].values
        yerr_lo = ates - ci_lo
        yerr_hi = ci_hi - ates
        yerr = np.array([yerr_lo, yerr_hi])
        yerr = np.nan_to_num(yerr, nan=0.0)

        ax.errorbar(
            x_pos, ates, yerr=yerr,
            marker=marker, color=color, label=subgroup,
            capsize=4, linewidth=1.5, markersize=8,
        )

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Trimming Bracket")
    ax.set_ylabel("ATE (mortality risk difference)")
    ax.set_title("Positivity Grid Stability: ATE by Trimming Bracket")
    ax.set_xticks(np.arange(len(bracket_order)))
    ax.set_xticklabels(bracket_order, rotation=45, ha="right")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fig_path = FIGURES_DIR / "85d_positivity_grid_stability.png"
    fig.savefig(fig_path, dpi=150)
    fig.savefig(str(fig_path).replace('.png', '.pdf'), format='pdf',
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig)
    print(f"  Saved: {fig_path} (+pdf)")


def replot_P1():
    """Regenerate P1_guideline_policy_forest_plot.png/pdf."""
    print("[P1] Replotting guideline policy forest plot ...")
    csv_path = TABLES_DIR / "P1_guideline_policy_comparison.csv"
    if not csv_path.exists():
        print(f"  SKIP: {csv_path} not found")
        return
    results_df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, max(6, len(results_df) * 0.4)))

    rows = []
    for _, r in results_df.iterrows():
        label = f"{r['policy']} — {r.get('group', 'Overall')}"
        ate = r["ate"]
        ci_lo = r.get("ate_ci_lo", ate)
        ci_hi = r.get("ate_ci_hi", ate)
        rows.append((label, ate, ci_lo, ci_hi))

    rows = sorted(rows, key=lambda x: x[1])
    labels = [r[0] for r in rows]
    ates = [r[1] for r in rows]
    ci_lo = [r[2] for r in rows]
    ci_hi = [r[3] for r in rows]
    y_pos = np.arange(len(rows))

    ax.errorbar(
        ates, y_pos,
        xerr=[np.array(ates) - np.array(ci_lo),
              np.array(ci_hi) - np.array(ates)],
        fmt="o", color="steelblue", ecolor="gray",
        elinewidth=1.5, capsize=3, markersize=6,
    )
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.6, linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("ATE (mortality risk difference, pp)", fontsize=11)
    ax.set_title("Guideline Policy Comparison \u2014 ATE vs Observed", fontsize=12)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    output_path = FIGURES_DIR / "P1_guideline_policy_forest_plot.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(str(output_path).replace('.png', '.pdf'), format='pdf',
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"  Saved: {output_path} (+pdf)")


def replot_P3():
    """Regenerate P3_cumulative_ate_curve.png/pdf and P3_cumulative_ate_by_hf_subtype.png/pdf."""
    print("[P3] Replotting cumulative ATE curves ...")
    csv_path = TABLES_DIR / "P3_time_varying_ate.csv"
    if not csv_path.exists():
        print(f"  SKIP: {csv_path} not found")
        return
    results_df = pd.read_csv(csv_path)

    # Figure 1: Overall cumulative ATE curve
    overall = results_df[results_df["group"] == "Overall"].sort_values("horizon_hours")
    fig, ax = plt.subplots(figsize=(10, 5))
    hours = overall["horizon_hours"].values
    ates = overall["ate"].values * 100  # to pp

    ax.plot(hours, ates, "o-", color="steelblue", linewidth=2, markersize=6)
    ax.fill_between(hours, 0, ates, alpha=0.1, color="steelblue")
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Rule policy horizon (hours post-T0)", fontsize=11)
    ax.set_ylabel("Cumulative ATE (pp)", fontsize=11)
    ax.set_title("Cumulative Treatment Effect by Rule Policy Horizon", fontsize=12)
    ax.set_xticks(hours)
    ax.grid(alpha=0.3)

    if len(ates) > 1:
        marginal = np.diff(ates)
        max_marginal_idx = np.argmax(np.abs(marginal))
        ax.annotate(
            f"Max marginal\n{hours[max_marginal_idx+1]}h: \u0394={marginal[max_marginal_idx]:+.2f}pp",
            xy=(hours[max_marginal_idx + 1], ates[max_marginal_idx + 1]),
            xytext=(hours[max_marginal_idx + 1] + 2, ates[max_marginal_idx + 1] + 0.3),
            arrowprops=dict(arrowstyle="->", color="gray"),
            fontsize=9, color="gray",
        )

    plt.tight_layout()
    fig_path = FIGURES_DIR / "P3_cumulative_ate_curve.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.savefig(str(fig_path).replace('.png', '.pdf'), format='pdf',
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"  Saved: {fig_path} (+pdf)")

    # Figure 2: By HF subtype
    hf_data = results_df[
        results_df["group"].isin(["Overall", "hfpef", "hfref", "no_hf",
                                   "shock_predominant", "non_shock"])
    ].copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"Overall": "black", "hfpef": "#2196F3", "hfref": "#FF9800",
              "no_hf": "#9E9E9E", "shock_predominant": "#F44336",
              "non_shock": "#4CAF50"}

    for group_name, grp in hf_data.groupby("group"):
        grp = grp.sort_values("horizon_hours")
        ax.plot(grp["horizon_hours"], grp["ate"] * 100, "o-",
                label=f"{group_name} (n={grp['n_stays'].iloc[0]})",
                color=colors.get(group_name, "gray"),
                linewidth=2 if group_name in ["Overall", "hfpef"] else 1.2,
                markersize=5, alpha=0.9 if group_name in ["Overall", "hfpef"] else 0.7)

    ax.axhline(y=0, color="red", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.set_xlabel("Rule policy horizon (hours post-T0)", fontsize=11)
    ax.set_ylabel("Cumulative ATE (pp)", fontsize=11)
    ax.set_title("Cumulative ATE by Subgroup", fontsize=12)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "P3_cumulative_ate_by_hf_subtype.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.savefig(str(fig_path).replace('.png', '.pdf'), format='pdf',
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"  Saved: {fig_path} (+pdf)")


if __name__ == "__main__":
    print("=" * 72)
    print("Lightweight Replot: 85d + P1 + P3 (from saved CSVs)")
    print("=" * 72)

    replot_85d()
    replot_P1()
    replot_P3()

    print("\nDone — all figures regenerated from saved CSVs.")
    print("Next: re-run manuscript_merge_figures.py and manuscript_regenerate_all_vector_figures.py")
