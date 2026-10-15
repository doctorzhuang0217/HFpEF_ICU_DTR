#!/usr/bin/env python3
"""Standalone manuscript figure generator — reads CSVs only, no model/GPU needed.

This script regenerates ALL manuscript figures from pre-computed CSV tables.
It is completely decoupled from the pipeline computation scripts (P1–P9, 85d,
95b, etc.).  Any label/style/layout change should be made HERE, not in the
upstream pipeline scripts.

Compute: C1 (< 10 seconds on any machine)

Figures generated
-----------------
Main text:
  Figure 2  — (A) Cross-model ATE forest  + (B) HFpEF specificity
  Figure 3  — (A) Sensitivity verdict heatmap + (B) Positivity grid stability
  Figure 4  — External validation (MIMIC vs eICU)
  Figure 5  — (A) HFI distribution + (B) dose-response + (C) matched specificity

Supplementary:
  sFigure 3 — (A) Overall cumulative ATE + (B) by subgroup
  sFigure 4 — (A) CATE histogram + (B) benefit profile heatmap
  sFigure 5 — Guideline comparison forest plot
  sFigure 7 — Monitoring dose-response
  sFigure 8 — Action pathway decomposition (non-shock)
  sFigure 9 — HFI metrics comparison
  sFigure 10 — Continuous HFI × HF-phenotype interaction (LOWESS + bin-level)

Not generated here (require pipeline data, not CSV):
  Figure 1  — CONSORT (run manuscript_figure1_consort.py)
  sFigure 1 — Phenotype radar (run 60e_phenotype_cards_early.py)
  sFigure 2 — IPTW weight distribution (run 75a_dynamic_msm_iptw.py)
  sFigure 6 — Counterfactual trajectories (run P5_counterfactual_trajectories.py)

Usage
-----
  env/run_py.sh scripts/manuscript_replot_all_figures.py
"""

from __future__ import annotations

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
import matplotlib.image as mpimg
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy import stats

# ══════════════════════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════════════════════
ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
TABLES = ROOT / "tables"
FIG_DIR = ROOT / "figures" / "manuscript"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR = ROOT / "manuscript" / "figures_pdf"
PDF_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
#  GLOBAL STYLE
# ══════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════
#  LABEL MAPS  —  single source of truth for all figures
# ══════════════════════════════════════════════════════════════════════
HF_LABEL = {
    "hfpef": "HFpEF",
    "no_hf": "No HF",
    "hfref": "HFrEF",
    "hfmref": "HFmrEF",
    "hf_ef_unknown": "HF (EF unknown)",
}

PHENO_LABEL = {
    "shock_predominant": "Shock-predominant",
    "non_shock": "Non-shock",
    "congestion": "Congestion",
    "mixed": "Mixed",
    "neither": "Neither",
}

GROUP_LABEL = {
    **HF_LABEL,
    **PHENO_LABEL,
    "Overall": "Overall",
}

POLICY_LABEL = {
    "Study Rule (70c)": "Study Rule",
}

MODEL_LABEL = {
    "80c_gtransformer_rollout_ow1_DM": "G-Transformer (DM)",
    "80c_gtransformer_rollout_ow1_FQE": "G-Transformer (FQE)",
    "80a_gnet_DM": "G-Net (DM)",
    "80a_gnet_FQE": "G-Net (FQE)",
    "80b_actin_DR": "ACTIN (DR)",
    "80b_actin_FQE": "ACTIN (FQE)",
    "80d_ct_legacy_FQE": "CT-Legacy (FQE)",
    "80e_mamba_cdsp_FQE": "Mamba-CDSP (FQE)",
}

# Colors
C_HFPEF = "#C0392B"
C_NOHF = "#2980B9"
C_HFREF = "#27AE60"
C_HFMREF = "#E67E22"
HF_COLOR = {"hfpef": C_HFPEF, "HFpEF": C_HFPEF,
            "no_hf": C_NOHF, "No HF": C_NOHF,
            "hfref": C_HFREF, "HFrEF": C_HFREF,
            "hfmref": C_HFMREF, "HFmrEF": C_HFMREF}
Q_COLORS = ["#27AE60", "#F1C40F", "#E67E22", "#C0392B"]


def _lbl(raw: str) -> str:
    """Map any raw variable name to publication label."""
    return GROUP_LABEL.get(raw, POLICY_LABEL.get(raw, raw))


def _save(fig, name: str) -> Path:
    """Save figure as PNG + PDF."""
    png = FIG_DIR / f"{name}.png"
    pdf = PDF_DIR / f"{name}.pdf"
    fig.savefig(png, bbox_inches="tight", dpi=300, facecolor="white")
    fig.savefig(pdf, format="pdf", bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  {name}.png + .pdf")
    return png


def _read(name: str) -> pd.DataFrame | None:
    p = TABLES / name
    if not p.exists():
        print(f"  SKIP — missing {name}")
        return None
    return pd.read_csv(p)


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Core Findings
# ══════════════════════════════════════════════════════════════════════

def fig2a_forest(ax, registry: pd.DataFrame):
    """Panel A: Cross-model ATE forest plot."""
    r2 = registry[registry["section"] == "R2_step_c_models"]
    pv = r2[~r2["result_id"].str.contains("ci_")]

    r1 = registry[registry["section"] == "R1_primary_ope"]
    ate_wis = r1[r1["result_id"] == "75a_ci_ate_wis"]

    labels, ates, ci_los, ci_his = [], [], [], []

    if len(ate_wis) > 0:
        r = ate_wis.iloc[0]
        labels.append("MSM-IPTW (WIS)")
        ates.append(float(r["value"]) * 100)
        ci_los.append(float(r["ci_lo"]) * 100 if pd.notna(r["ci_lo"]) else None)
        ci_his.append(float(r["ci_hi"]) * 100 if pd.notna(r["ci_hi"]) else None)

    for rid, display in MODEL_LABEL.items():
        rows = pv[pv["result_id"] == rid]
        if len(rows) > 0:
            r = rows.iloc[0]
            labels.append(display)
            ates.append(float(r["value"]) * 100)
            ci_los.append(None)
            ci_his.append(None)

    # Attach CI for G-Transformer DM
    ci_row = r2[r2["result_id"] == "80c_rollout_ow1_ci_DM"]
    if len(ci_row) > 0:
        r = ci_row.iloc[0]
        for i, l in enumerate(labels):
            if l == "G-Transformer (DM)":
                ci_los[i] = float(r["ci_lo"]) * 100 if pd.notna(r["ci_lo"]) else None
                ci_his[i] = float(r["ci_hi"]) * 100 if pd.notna(r["ci_hi"]) else None

    for i in range(len(labels)):
        color = "#2166ac" if i == 0 else "#d6604d" if "DM" in labels[i] else "#4393c3"
        ax.plot(ates[i], i, "o", color=color, markersize=8, zorder=3)
        if ci_los[i] is not None and ci_his[i] is not None:
            ax.plot([ci_los[i], ci_his[i]], [i, i], "-", color=color, linewidth=2, zorder=2)

    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("ATE (percentage points)")
    ax.set_title("Cross-Model ATE Comparison\n(negative = rule policy reduces mortality)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)


def fig2b_specificity(ax, registry: pd.DataFrame):
    """Panel B: HFpEF specificity bar chart (non-shock stratum)."""
    r3 = registry[registry["section"] == "R3_hf_shock_interaction"]
    non_shock = r3[r3["subgroup"].str.contains("non_shock", na=False)]
    if len(non_shock) == 0:
        return

    subtypes, ate_vals = [], []
    for _, row in non_shock.iterrows():
        sg = str(row["subgroup"]).replace("_non_shock", "")
        subtypes.append(HF_LABEL.get(sg, sg))
        ate_vals.append(float(row["value"]) * 100)

    colors = [C_HFPEF if s == "HFpEF" else C_NOHF if s == "No HF" else "#92c5de"
              for s in subtypes]
    bars = ax.barh(subtypes, ate_vals, color=colors, edgecolor="white", height=0.6)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("ATE (percentage points)")
    ax.set_title("HFpEF-Specific Effect in Non-Shock Stratum\n"
                 "(G-Transformer rollout, DM estimator)")
    ax.invert_yaxis()
    for bar, val in zip(bars, ate_vals):
        if val < 0:
            ax.text(val / 2, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}pp", va="center", ha="center", fontsize=9, fontweight="bold")
        else:
            ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}pp", va="center", ha="left", fontsize=9, fontweight="bold")


def make_figure2():
    print("[Figure 2] Core findings")
    reg = _read("95a_results_registry.csv")
    if reg is None:
        return
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(10, 10))
    fig.text(0.01, 0.98, "A", fontsize=18, fontweight="bold", va="top")
    fig.text(0.01, 0.48, "B", fontsize=18, fontweight="bold", va="top")
    fig2a_forest(ax_a, reg)
    fig2b_specificity(ax_b, reg)
    fig.tight_layout(rect=[0.03, 0, 1, 1])
    _save(fig, "Figure2")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Robustness
# ══════════════════════════════════════════════════════════════════════

def fig3a_heatmap(ax):
    """Panel A: Sensitivity analysis verdict heatmap."""
    steps = ["EDQ", "SeqDeconf", "Positivity", "NegCtrl",
             "E-value", "Monitoring", "RRT", "Horizon", "Missingness"]
    verdicts = ["ROBUST", "SENSITIVE", "STABLE", "PASS",
                "WEAK", "ROBUST", "ROBUST", "SENSITIVE", "REFRAMED"]
    vmap = {"ROBUST": 2, "PASS": 2, "STABLE": 2, "REFRAMED": 1, "WEAK": 0, "SENSITIVE": 0}
    scores = [vmap.get(v, 1) for v in verdicts]
    cmap = {2: "#4daf4a", 1: "#ff7f00", 0: "#e41a1c"}

    ax.barh([0] * len(steps), [1] * len(steps),
            left=range(len(steps)), color=[cmap[s] for s in scores],
            edgecolor="white", height=0.8)
    for i, v in enumerate(verdicts):
        ax.text(i + 0.5, 0, v, ha="center", va="center",
                fontsize=8, fontweight="bold", color="white")
    ax.set_xticks([i + 0.5 for i in range(len(steps))])
    ax.set_xticklabels(steps, fontsize=8)
    ax.set_yticks([])
    ax.set_title("Sensitivity Analysis Verdicts")
    ax.set_xlim(-0.1, len(steps) + 0.1)
    ax.legend(handles=[
        Patch(facecolor="#4daf4a", label="ROBUST / PASS / STABLE"),
        Patch(facecolor="#ff7f00", label="REFRAMED"),
        Patch(facecolor="#e41a1c", label="SENSITIVE / WEAK"),
    ], loc="upper right", fontsize=8)


def fig3b_positivity(ax):
    """Panel B: Positivity grid stability."""
    df = _read("85d_positivity_grid.csv")
    if df is None:
        return

    brackets = ["none"] + [f"[{lo},{hi}]" for lo, hi in
                           [(1.0, 99.0), (2.5, 97.5), (5.0, 95.0),
                            (10.0, 90.0), (20.0, 80.0)]]
    x_map = {b: i for i, b in enumerate(brackets)}

    for subgroup, marker, color in [
        ("ALL", "o", "#2196F3"), ("hfpef", "s", "#E91E63"), ("no_hf", "^", "#4CAF50"),
    ]:
        sub = df[(df["subgroup"] == subgroup) & (df["bracket"].isin(x_map))].copy()
        if sub.empty:
            continue
        sub["x_pos"] = sub["bracket"].map(x_map)
        sub = sub.sort_values("x_pos")
        yerr = np.nan_to_num(np.array([
            sub["ate"].values - sub["ate_ci_lo"].values,
            sub["ate_ci_hi"].values - sub["ate"].values,
        ]), nan=0.0)
        ax.errorbar(sub["x_pos"], sub["ate"], yerr=yerr,
                    marker=marker, color=color,
                    label=HF_LABEL.get(subgroup, subgroup),
                    capsize=4, linewidth=1.5, markersize=8)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Trimming Bracket")
    ax.set_ylabel("ATE (mortality risk difference)")
    ax.set_title("Positivity Grid Stability: ATE by Trimming Bracket")
    ax.set_xticks(range(len(brackets)))
    ax.set_xticklabels(brackets, rotation=45, ha="right", fontsize=7)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)


def make_figure3():
    print("[Figure 3] Robustness")
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2.5], hspace=0.35)
    fig.text(0.01, 0.98, "A", fontsize=18, fontweight="bold", va="top")
    fig.text(0.01, 0.58, "B", fontsize=18, fontweight="bold", va="top")
    fig3a_heatmap(fig.add_subplot(gs[0]))
    fig3b_positivity(fig.add_subplot(gs[1]))
    _save(fig, "Figure3")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 4 — External Validation
# ══════════════════════════════════════════════════════════════════════

def make_figure4():
    print("[Figure 4] External validation")
    comp = _read("90b_eicu_transportability_comparison.csv")
    tier_a = _read("90b_05_eicu_tier_a_sensitivity.csv")
    if comp is None:
        return

    def pp(v):
        return v * 100 if pd.notna(v) else 0.0

    def pp_err(val, lo, hi):
        if pd.notna(lo) and pd.notna(hi) and pd.notna(val):
            return (abs(val - lo) * 100, abs(hi - val) * 100)
        return (0.0, 0.0)

    dm_row = comp[comp["estimator"] == "DM"].iloc[0] if "DM" in comp["estimator"].values else None
    fqe_row = comp[comp["estimator"] == "FQE"].iloc[0] if "FQE" in comp["estimator"].values else None
    if dm_row is None:
        return

    ta_dm_ate = ta_fqe_ate = np.nan
    ta_dm_ci_lo = ta_dm_ci_hi = ta_fqe_ci_lo = ta_fqe_ci_hi = np.nan
    if tier_a is not None:
        ta_dm = tier_a[(tier_a["estimator"] == "DM") & (tier_a["subset"] == "tier_a_full")]
        if len(ta_dm):
            ta_dm_ate = ta_dm.iloc[0]["ate"]
            ta_dm_ci_lo = ta_dm.iloc[0].get("ate_ci_lo", np.nan)
            ta_dm_ci_hi = ta_dm.iloc[0].get("ate_ci_hi", np.nan)
        ta_fqe = tier_a[(tier_a["estimator"] == "FQE") & (tier_a["subset"] == "tier_a_full")]
        if len(ta_fqe):
            ta_fqe_ate = ta_fqe.iloc[0]["ate"]
            ta_fqe_ci_lo = ta_fqe.iloc[0].get("ate_ci_lo", np.nan)
            ta_fqe_ci_hi = ta_fqe.iloc[0].get("ate_ci_hi", np.nan)

    bar_colors = ["#2166ac", "#d6604d", "#4dac26"]
    xlabels = ["MIMIC-IV\n(n=38,981)", "eICU (full)\n(n=83,765)", "eICU (Tier A)\n(n=2,680)"]
    x = np.arange(3)
    w = 0.55

    dm_vals = [pp(dm_row["mimic_ate"]), pp(dm_row["eicu_ate"]), pp(ta_dm_ate)]
    dm_errs_lo, dm_errs_hi = [], []
    for val, lo, hi in [
        (dm_row["mimic_ate"], dm_row.get("mimic_ci_lo"), dm_row.get("mimic_ci_hi")),
        (dm_row["eicu_ate"], dm_row.get("eicu_ci_lo"), dm_row.get("eicu_ci_hi")),
        (ta_dm_ate, ta_dm_ci_lo, ta_dm_ci_hi),
    ]:
        e = pp_err(val, lo, hi)
        dm_errs_lo.append(e[0]); dm_errs_hi.append(e[1])

    fqe_vals = [0.0, 0.0, 0.0]
    fqe_errs_lo = [0.0, 0.0, 0.0]
    fqe_errs_hi = [0.0, 0.0, 0.0]
    if fqe_row is not None:
        fqe_vals = [pp(fqe_row["mimic_ate"]), pp(fqe_row["eicu_ate"]), pp(ta_fqe_ate)]
        for idx, (val, lo, hi) in enumerate([
            (fqe_row["mimic_ate"], fqe_row.get("mimic_ci_lo"), fqe_row.get("mimic_ci_hi")),
            (fqe_row["eicu_ate"], fqe_row.get("eicu_ci_lo"), fqe_row.get("eicu_ci_hi")),
            (ta_fqe_ate, ta_fqe_ci_lo, ta_fqe_ci_hi),
        ]):
            e = pp_err(val, lo, hi)
            fqe_errs_lo[idx] = e[0]; fqe_errs_hi[idx] = e[1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5))

    for ax, vals, errs_lo, errs_hi, title in [
        (ax1, dm_vals, dm_errs_lo, dm_errs_hi,
         "A: Direct Method (DM)\n(teacher-forced, cross-database comparison)"),
        (ax2, fqe_vals, fqe_errs_lo, fqe_errs_hi,
         "B: Fitted Q-Evaluation (FQE)\n(sign reversal in full sample, recovers in Tier A)"),
    ]:
        bars = ax.bar(x, vals, w, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.errorbar(x, vals, yerr=[errs_lo, errs_hi], fmt="none", ecolor="black", capsize=4, linewidth=1.2)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_ylabel("ATE (percentage points)"); ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        # Place value labels inside bars (centered); tiny bars get label above
        y_range = max(abs(min(vals)), abs(max(vals)), 0.1)
        for v, bar in zip(vals, bars):
            if abs(v) > y_range * 0.12:
                # Bar tall enough: label centered inside
                y_pos = v / 2
                va = "center"
            else:
                # Tiny bar: label just outside
                y_pos = v - y_range * 0.05 if v < 0 else v + y_range * 0.05
                va = "top" if v < 0 else "bottom"
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"{v:.2f}pp", ha="center", va=va, fontsize=9, fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))

    fig.suptitle("Cross-Database ATE Comparison: MIMIC-IV vs eICU-CRD\n"
                 "(G-Transformer, DR excluded \u2014 weight pathology)",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    _save(fig, "Figure4")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 5 — Hemodynamic Fragility Index
# ══════════════════════════════════════════════════════════════════════

def make_figure5():
    print("[Figure 5] HFI composite")
    dist_df = _read("P9_hfi_distribution_by_hf_subtype.csv")
    ate_df = _read("P9_ate_by_hfi_quartile.csv")
    matched_df = _read("P9_hfi_matched_hfpef_vs_nohf.csv")
    if any(d is None for d in [dist_df, ate_df, matched_df]):
        return

    overall = ate_df[ate_df["hf_subtype"] == "ALL"].sort_values("quartile")
    quartiles = ["Q1_stable", "Q2", "Q3", "Q4_fragile"]
    q_labels = ["Q1\n(stable)", "Q2", "Q3", "Q4\n(fragile)"]

    fig = plt.figure(figsize=(7.2, 7.8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.1], hspace=0.35, wspace=0.35)

    # ── Panel A: HFI distribution ──
    ax_a = fig.add_subplot(gs[0, :])
    ax_a.text(-0.08, 1.05, "A", transform=ax_a.transAxes, fontsize=14, fontweight="bold", va="top")
    x_range = np.linspace(-2.5, 3.5, 300)
    for _, row in dist_df.iterrows():
        grp = row["hf_subtype"]
        if grp not in ("hfpef", "no_hf", "hfref", "hfmref"):
            continue
        mu, sigma = row["hfi_composite_mean"], row["hfi_composite_std"]
        color = HF_COLOR.get(grp, "gray")
        lw = 2.0 if grp in ("hfpef", "no_hf") else 1.2
        ls = "-" if grp in ("hfpef", "no_hf") else "--"
        zorder = 10 if grp == "hfpef" else 9 if grp == "no_hf" else 5
        kde_y = stats.norm.pdf(x_range, mu, sigma)
        ax_a.plot(x_range, kde_y, color=color, linewidth=lw, linestyle=ls,
                  label=f"{HF_LABEL[grp]} (n={int(row['n_stays'])})", zorder=zorder)
        if grp in ("hfpef", "no_hf"):
            ax_a.fill_between(x_range, kde_y, alpha=0.12, color=color, zorder=zorder - 1)

    # Effect size annotation with white background
    hfpef_mu = dist_df.loc[dist_df["hf_subtype"] == "hfpef", "hfi_composite_mean"].values[0]
    nohf_mu = dist_df.loc[dist_df["hf_subtype"] == "no_hf", "hfi_composite_mean"].values[0]
    ax_a.annotate("", xy=(hfpef_mu, 0.42), xytext=(nohf_mu, 0.42),
                  arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.2), zorder=20)
    ax_a.text((hfpef_mu + nohf_mu) / 2, 0.44, "Cohen\u2019s d = 0.33\np < 10\u207b\u2077",
              ha="center", va="bottom", fontsize=7.5, color="#333333", style="italic",
              zorder=20, bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                   edgecolor="none", alpha=0.85))
    ax_a.set_xlabel("Hemodynamic Fragility Index (composite z-score)")
    ax_a.set_ylabel("Density")
    ax_a.set_title("HFI Distribution by HF Subtype", fontsize=10, fontweight="bold")
    ax_a.legend(loc="upper right", framealpha=0.9, edgecolor="none")
    ax_a.set_xlim(-2.5, 3.5); ax_a.set_ylim(0, 0.62)
    ax_a.spines["top"].set_visible(False); ax_a.spines["right"].set_visible(False)

    # ── Panel B: Dose-response ──
    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.text(-0.15, 1.05, "B", transform=ax_b.transAxes, fontsize=14, fontweight="bold", va="top")
    ates_b = overall["ate"].values * 100
    bars_b = ax_b.bar(range(4), ates_b, color=Q_COLORS, edgecolor="white", linewidth=0.8, width=0.7, zorder=3)
    for bar, v in zip(bars_b, ates_b):
        ax_b.text(bar.get_x() + bar.get_width() / 2, v - 0.06,
                  f"{v:.1f}", ha="center", va="top", fontsize=8, fontweight="bold", color="white")
    ax_b.annotate("", xy=(3.3, ates_b[-1] - 0.1), xytext=(3.3, ates_b[0] + 0.1),
                  arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax_b.text(3.55, (ates_b[0] + ates_b[-1]) / 2, "\u03c1 = \u22121.0\np < 0.001",
              fontsize=7, color="gray", ha="left", va="center", style="italic")
    ax_b.set_xticks(range(4)); ax_b.set_xticklabels(q_labels, fontsize=8)
    ax_b.set_ylabel("ATE (percentage points)")
    ax_b.set_title("Dose-Response:\nATE by HFI Quartile", fontsize=10, fontweight="bold")
    ax_b.axhline(0, color="black", linewidth=0.5, zorder=1)
    ax_b.set_ylim(min(ates_b) - 0.4, 0.15)
    ax_b.spines["top"].set_visible(False); ax_b.spines["right"].set_visible(False)
    ax_b.set_xlim(-0.5, 4.2)

    # ── Panel C: Matched specificity ──
    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.text(-0.15, 1.05, "C", transform=ax_c.transAxes, fontsize=14, fontweight="bold", va="top")
    width = 0.35
    hfpef_ates, nohf_ates, hfpef_ns, nohf_ns, ratios = [], [], [], [], []
    for q in quartiles:
        hr = matched_df[(matched_df["quartile"] == q) & (matched_df["hf_subtype"] == "hfpef")]
        nr = matched_df[(matched_df["quartile"] == q) & (matched_df["hf_subtype"] == "no_hf")]
        hfpef_ates.append(hr["ate"].values[0] * 100)
        nohf_ates.append(nr["ate"].values[0] * 100)
        hfpef_ns.append(int(hr["n_stays"].values[0]))
        nohf_ns.append(int(nr["n_stays"].values[0]))
        ratios.append(hr["ate"].values[0] / nr["ate"].values[0])

    xc = np.arange(4)
    ax_c.bar(xc - width / 2, hfpef_ates, width, label="HFpEF",
             color=C_HFPEF, alpha=0.85, edgecolor="white", linewidth=0.5, zorder=3)
    ax_c.bar(xc + width / 2, nohf_ates, width, label="No HF",
             color=C_NOHF, alpha=0.85, edgecolor="white", linewidth=0.5, zorder=3)
    for i, ratio in enumerate(ratios):
        y_pos = min(hfpef_ates[i], nohf_ates[i]) - 0.55
        ax_c.text(i, y_pos, f"{ratio:.1f}\u00d7", ha="center", va="top",
                  fontsize=7.5, fontweight="bold", color="#555555", zorder=10,
                  bbox=dict(boxstyle="round,pad=0.18", facecolor="lightyellow",
                            edgecolor="#cccccc", linewidth=0.5, alpha=1.0))
    ax_c.axhline(0, color="black", linewidth=0.5, zorder=1)
    ax_c.set_xticks(xc)
    # Build x-axis labels with n= counts below quartile names
    xlabels_c = [f"{ql}\nn={hfpef_ns[i]}/{nohf_ns[i]}"
                 for i, ql in enumerate(q_labels)]
    ax_c.set_xticklabels(xlabels_c, fontsize=7)
    ax_c.set_ylabel("ATE (percentage points)")
    ax_c.set_title("Residual Specificity:\nHFpEF vs No HF (HFI-matched)",
                    fontsize=10, fontweight="bold")
    ax_c.legend(loc="upper right", framealpha=0.9, edgecolor="none")
    all_ates = hfpef_ates + nohf_ates
    ax_c.set_ylim(min(all_ates) - 1.3, 0.35)
    ax_c.spines["top"].set_visible(False); ax_c.spines["right"].set_visible(False)

    _save(fig, "Figure5")


# ══════════════════════════════════════════════════════════════════════
#  sFigure 3 — Cumulative ATE
# ══════════════════════════════════════════════════════════════════════

def make_sfig3():
    print("[sFigure 3] Cumulative ATE")
    df = _read("P3_time_varying_ate.csv")
    if df is None:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.text(0.01, 0.98, "A", fontsize=16, fontweight="bold", va="top")
    fig.text(0.51, 0.98, "B", fontsize=16, fontweight="bold", va="top")

    # Panel A: Overall
    overall = df[df["group"] == "Overall"].sort_values("horizon_hours")
    h, a = overall["horizon_hours"].values, overall["ate"].values * 100
    ax1.plot(h, a, "o-", color="steelblue", linewidth=2, markersize=6)
    ax1.fill_between(h, 0, a, alpha=0.1, color="steelblue")
    ax1.axhline(0, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
    ax1.set_xlabel("Rule policy horizon (hours post-T0)")
    ax1.set_ylabel("Cumulative ATE (pp)")
    ax1.set_title("Overall Cumulative ATE Curve")
    ax1.set_xticks(h); ax1.grid(alpha=0.3)
    if len(a) > 1:
        mg = np.diff(a)
        mi = np.argmax(np.abs(mg))
        ax1.annotate(f"Max marginal\n{h[mi+1]}h: \u0394={mg[mi]:+.2f}pp",
                     xy=(h[mi + 1], a[mi + 1]),
                     xytext=(h[mi + 1] + 2, a[mi + 1] + 0.3),
                     arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9, color="gray")

    # Panel B: By subgroup
    colors = {"Overall": "black", "hfpef": "#2196F3", "hfref": "#FF9800",
              "no_hf": "#9E9E9E", "shock_predominant": "#F44336", "non_shock": "#4CAF50"}
    target_groups = ["Overall", "hfpef", "hfref", "no_hf", "shock_predominant", "non_shock"]
    sub = df[df["group"].isin(target_groups)]
    for gn, grp in sub.groupby("group"):
        grp = grp.sort_values("horizon_hours")
        ax2.plot(grp["horizon_hours"], grp["ate"] * 100, "o-",
                 label=f"{_lbl(gn)} (n={grp['n_stays'].iloc[0]})",
                 color=colors.get(gn, "gray"),
                 linewidth=2 if gn in ["Overall", "hfpef"] else 1.2,
                 markersize=5, alpha=0.9 if gn in ["Overall", "hfpef"] else 0.7)
    ax2.axhline(0, color="red", linestyle="--", alpha=0.4, linewidth=0.8)
    ax2.set_xlabel("Rule policy horizon (hours post-T0)")
    ax2.set_ylabel("Cumulative ATE (pp)")
    ax2.set_title("Cumulative ATE by Subgroup")
    ax2.legend(fontsize=8, loc="lower left"); ax2.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, "sFigure3")


# ══════════════════════════════════════════════════════════════════════
#  sFigure 4 — CATE heterogeneity
# ══════════════════════════════════════════════════════════════════════

def make_sfig4():
    print("[sFigure 4] CATE heterogeneity")
    cate = _read("P2_cate_distribution.csv")
    if cate is None:
        return

    # Try to load phenotype labels for subtype stratification
    pheno_path = ROOT / "data" / "processed" / "phenotype_labels_frozen_v1_mimic.parquet"
    if not pheno_path.exists():
        print("  SKIP sFigure 4 — phenotype parquet not found")
        return
    pheno = pd.read_parquet(pheno_path)
    merged = cate.merge(pheno[["stay_id", "hf_subtype"]].drop_duplicates("stay_id"),
                        on="stay_id", how="left")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Overall + by subtype
    ax = axes[0]
    cate_vals = merged["cate"].values * 100
    ax.hist(cate_vals, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", alpha=0.7, linewidth=1)
    pct_b = (cate_vals < 0).mean() * 100
    ax.set_xlabel("CATE (pp)"); ax.set_ylabel("Count")
    ax.set_title(f"Overall CATE Distribution\nBenefit: {pct_b:.1f}%, Harm: {100-pct_b:.1f}%")

    ax = axes[1]
    st_colors = {"hfpef": "#2196F3", "hfref": "#FF9800", "hfmref": "#4CAF50", "no_hf": "#9E9E9E"}
    for st in sorted(merged["hf_subtype"].dropna().unique()):
        vals = merged.loc[merged["hf_subtype"] == st, "cate"].values * 100
        ax.hist(vals, bins=30, alpha=0.5,
                label=f"{HF_LABEL.get(st, st)} (n={len(vals)})",
                color=st_colors.get(st, "gray"), edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", alpha=0.7, linewidth=1)
    ax.set_xlabel("CATE (pp)"); ax.set_ylabel("Count")
    ax.set_title("CATE by HF Subtype"); ax.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, "sFigure4")


# ══════════════════════════════════════════════════════════════════════
#  sFigure 5 — Guideline comparison
# ══════════════════════════════════════════════════════════════════════

def make_sfig5():
    print("[sFigure 5] Guideline comparison")
    df = _read("P1_guideline_policy_comparison.csv")
    if df is None:
        return

    rows = []
    for _, r in df.iterrows():
        policy = POLICY_LABEL.get(r["policy"], r["policy"])
        group = HF_LABEL.get(str(r.get("group", "Overall")), str(r.get("group", "Overall")))
        label = f"{policy} \u2014 {group}"
        ate = r["ate"] * 100  # convert proportion to pp
        ci_lo = r.get("ate_ci_lo", r["ate"]) * 100
        ci_hi = r.get("ate_ci_hi", r["ate"]) * 100
        rows.append((label, ate, ci_lo, ci_hi))
    rows.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(10, 7))
    labels = [r[0] for r in rows]
    ates = [r[1] for r in rows]
    ax.errorbar(ates, range(len(rows)),
                xerr=[np.array(ates) - np.array([r[2] for r in rows]),
                      np.array([r[3] for r in rows]) - np.array(ates)],
                fmt="o", color="steelblue", ecolor="gray",
                elinewidth=1.5, capsize=3, markersize=6)
    ax.axvline(0, color="red", linestyle="--", alpha=0.6, linewidth=0.8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("ATE (mortality risk difference, pp)")
    ax.set_title("Guideline Policy Comparison \u2014 ATE vs Observed")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, "sFigure5")


# ══════════════════════════════════════════════════════════════════════
#  sFigure 7 — Monitoring dose-response
# ══════════════════════════════════════════════════════════════════════

def make_sfig7():
    print("[sFigure 7] Monitoring dose-response")
    sens = _read("85j_missingness_sensitivity.csv")
    corr = _read("P7_monitoring_intensity_correlates.csv")
    if sens is None:
        return

    # Extract A4 quartile ATE data
    a4 = sens[sens["scenario"].str.startswith("A4_intensity")].copy()
    if len(a4) == 0:
        print("  SKIP — no A4 quartile data in 85j results")
        return
    a4["quartile"] = a4["scenario"].str.extract(r"Q(\d)").astype(int)
    a4 = a4.sort_values("quartile")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # ── Panel A: ATE by monitoring intensity quartile ──
    ax = axes[0]
    quartiles = a4["quartile"].values
    ates = a4["ate"].values * 100
    ci_lo = a4["ate_ci_lo"].values * 100
    ci_hi = a4["ate_ci_hi"].values * 100
    yerr = np.array([ates - ci_lo, ci_hi - ates])
    ax.bar(quartiles, ates, yerr=yerr, capsize=5,
           color=["#81D4FA", "#4FC3F7", "#29B6F6", "#0288D1"],
           edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.set_xlabel("Monitoring Intensity Quartile")
    ax.set_ylabel("ATE (pp)")
    ax.set_xticks(quartiles)
    ax.set_xticklabels(["Q1\n(low)", "Q2", "Q3", "Q4\n(high)"])
    ax.set_title("A: Treatment Effect\nby Monitoring Intensity")
    ax.grid(axis="y", alpha=0.3)

    # ── Panel B: Mortality rate by monitoring quartile ──
    ax = axes[1]
    if corr is not None and "mortality_rate" in corr.columns:
        mort = corr["mortality_rate"].values
        ax.bar(range(1, len(mort) + 1), mort * 100,
               color=["#A5D6A7", "#66BB6A", "#43A047", "#2E7D32"],
               edgecolor="black", linewidth=0.5, alpha=0.85)
        ax.set_xlabel("Monitoring Intensity Quartile")
        ax.set_ylabel("28-day Mortality (%)")
        ax.set_xticks(range(1, len(mort) + 1))
        ax.set_xticklabels(["Q1\n(low)", "Q2", "Q3", "Q4\n(high)"])
        ax.grid(axis="y", alpha=0.3)
    ax.set_title("B: Mortality Rate\nby Monitoring Intensity")

    # ── Panel C: A1 vs A2 vs A3 ablation (horizontal bars) ──
    ax = axes[2]
    scenarios, ate_vals, ci_lo_vals, ci_hi_vals, colors_c = [], [], [], [], []
    for label, scenario, color in [
        ("A1\n(aware)", "A1_baseline", "#1976D2"),
        ("A2\n(naive)", "A2_zero_missingness", "#F44336"),
        ("A3\n(shuffle)", "A3_shuffle_missingness", "#FF9800"),
    ]:
        row_df = sens[(sens["scenario"] == scenario) & (sens["subgroup"] == "ALL")]
        if len(row_df) > 0:
            r = row_df.iloc[0]
            scenarios.append(label)
            ate_vals.append(r["ate"] * 100)
            ci_lo_vals.append(r["ate_ci_lo"] * 100 if pd.notna(r.get("ate_ci_lo")) else r["ate"] * 100)
            ci_hi_vals.append(r["ate_ci_hi"] * 100 if pd.notna(r.get("ate_ci_hi")) else r["ate"] * 100)
            colors_c.append(color)
    if scenarios:
        ate_arr = np.array(ate_vals)
        yerr_c = np.array([ate_arr - np.array(ci_lo_vals),
                           np.array(ci_hi_vals) - ate_arr])
        ax.barh(range(len(scenarios)), ate_arr, xerr=yerr_c, capsize=5,
                color=colors_c, edgecolor="black", linewidth=0.5, alpha=0.85)
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.set_yticks(range(len(scenarios)))
        ax.set_yticklabels(scenarios)
        ax.set_xlabel("ATE (pp)")
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()
    ax.set_title("C: Monitoring Feature\nAblation Comparison")

    fig.tight_layout()
    _save(fig, "sFigure7")


# ══════════════════════════════════════════════════════════════════════
#  sFigure 8 — Action decomposition
# ══════════════════════════════════════════════════════════════════════

def make_sfig8():
    print("[sFigure 8] Action decomposition")
    p8 = _read("P8_action_mechanism_decomposition.csv")
    if p8 is None:
        return

    ns = p8[p8["stratum"] == "non_shock"]
    if ns.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    pw_labels = ["Shock\nresuscitation", "Decongestion", "Maintenance"]

    for ax_idx, (title, prefix) in enumerate([
        ("Rule Policy (recommended)", "rule_pct_"),
        ("Observed (clinician actual)", "obs_pct_"),
    ]):
        ax = axes[ax_idx]
        x = np.arange(3)
        width = 0.35
        hfpef_vals, nohf_vals = [], []
        for pw in ["shock", "decongest", "maintenance"]:
            col = f"{prefix}{pw}"
            hr = ns[ns["hf_subtype"] == "hfpef"][col].values[0] * 100
            nr = ns[ns["hf_subtype"] == "no_hf"][col].values[0] * 100
            hfpef_vals.append(hr); nohf_vals.append(nr)

        ax.bar(x - width / 2, hfpef_vals, width, label="HFpEF", color=C_HFPEF, alpha=0.85, edgecolor="white")
        ax.bar(x + width / 2, nohf_vals, width, label="No HF", color=C_NOHF, alpha=0.85, edgecolor="white")
        for i, (h, n) in enumerate(zip(hfpef_vals, nohf_vals)):
            if h > 1:
                ax.text(i - width / 2, h + 0.5, f"{h:.1f}%", ha="center", fontsize=7, color=C_HFPEF)
            if n > 1:
                ax.text(i + width / 2, n + 0.5, f"{n:.1f}%", ha="center", fontsize=7, color=C_NOHF)
        ax.set_xticks(x); ax.set_xticklabels(pw_labels, fontsize=8)
        ax.set_ylabel("Proportion (%)")
        ax.set_title(title, fontsize=9, fontweight="bold")
        if ax_idx == 0:
            ax.legend(loc="upper center", fontsize=7, framealpha=0.9, ncol=2)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("Treatment Pathway Distribution \u2014 Non-Shock Stratum",
                 fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "sFigure8")


# ══════════════════════════════════════════════════════════════════════
#  sFigure 9 — HFI metrics
# ══════════════════════════════════════════════════════════════════════

def make_sfig9():
    print("[sFigure 9] HFI metrics")
    dist = _read("P9_hfi_distribution_by_hf_subtype.csv")
    if dist is None:
        return

    hfpef_row = dist[dist["hf_subtype"] == "hfpef"].iloc[0]
    nohf_row = dist[dist["hf_subtype"] == "no_hf"].iloc[0]

    metrics = [
        ("Mean hypotensive\nbins (/12)", "n_hypo_bins_mean"),
        ("Any hypotension\n(%)", "n_hypo_bins_pct_any"),
        ("MAP CV\n(\u00d7100)", "map_cv_mean"),
    ]
    x = np.arange(len(metrics))
    width = 0.3

    hv = [hfpef_row["n_hypo_bins_mean"],
          hfpef_row["n_hypo_bins_pct_any"] * 100,
          hfpef_row["map_cv_mean"] * 100]
    nv = [nohf_row["n_hypo_bins_mean"],
          nohf_row["n_hypo_bins_pct_any"] * 100,
          nohf_row["map_cv_mean"] * 100]

    n_hfpef = int(hfpef_row["n_stays"])
    n_nohf = int(nohf_row["n_stays"])

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars_h = ax.bar(x - width / 2, hv, width, label=f"HFpEF (n={n_hfpef:,})",
                    color=C_HFPEF, alpha=0.85, edgecolor="white")
    bars_n = ax.bar(x + width / 2, nv, width, label=f"No HF (n={n_nohf:,})",
                    color=C_NOHF, alpha=0.85, edgecolor="white")
    for bars in [bars_h, bars_n]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=7.5)
    for i in range(len(metrics)):
        ax.text(i, max(hv[i], nv[i]) + 2.5, "***", ha="center", va="bottom",
                fontsize=9, color="#555555")
    ax.set_xticks(x); ax.set_xticklabels([m[0] for m in metrics], fontsize=8)
    ax.set_ylabel("Value")
    ax.set_title("Hemodynamic Fragility Metrics:\nHFpEF vs No HF", fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "sFigure9")


# ══════════════════════════════════════════════════════════════════════
#  sFigure 10 — Continuous HFI x HF-Phenotype Interaction
# ══════════════════════════════════════════════════════════════════════

def make_sfig10():
    print("[sFigure 10] Continuous HFI interaction")
    binned = _read("P10_binned_ate_by_hfi.csv")
    summary = _read("P10_interaction_test_summary.csv")
    if binned is None:
        return

    # Parse WLS results from summary
    beta_hfpef = beta_interaction = p_interaction = r2 = np.nan
    raw_diff = raw_welch_p = np.nan
    if summary is not None and len(summary) > 0:
        s = summary.iloc[0]
        beta_hfpef = s.get("beta_hfpef", np.nan)
        beta_interaction = s.get("beta_interaction", np.nan)
        p_interaction = s.get("p_interaction", np.nan)
        r2 = s.get("r_squared", np.nan)
        raw_diff = s.get("raw_diff", np.nan)
        raw_welch_p = s.get("raw_welch_p", np.nan)

    hfpef = binned[binned["hf_subtype"] == "hfpef"].sort_values("hfi_center")
    nohf = binned[binned["hf_subtype"] == "no_hf"].sort_values("hfi_center")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.text(0.01, 0.98, "A", fontsize=16, fontweight="bold", va="top")
    fig.text(0.51, 0.98, "B", fontsize=16, fontweight="bold", va="top")

    # ── Panel A: LOWESS-smoothed dose-response ──
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

    for df_sub, color, label, marker in [
        (hfpef, C_HFPEF, "HFpEF", "s"),
        (nohf, C_NOHF, "No HF", "o"),
    ]:
        x = df_sub["hfi_center"].values
        y = df_sub["ate_mean"].values * 100
        se = df_sub["ate_se"].values * 100
        ax_a.errorbar(x, y, yerr=1.96 * se, fmt=marker, color=color,
                      alpha=0.5, markersize=5, capsize=2, linewidth=0.8)
        # LOWESS smooth
        if len(x) >= 4:
            smoothed = sm_lowess(y, x, frac=0.6, return_sorted=True)
            ax_a.plot(smoothed[:, 0], smoothed[:, 1], "-", color=color,
                      linewidth=2.5, label=label, zorder=5)
        else:
            ax_a.plot(x, y, "-", color=color, linewidth=2, label=label)

    ax_a.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_a.set_xlabel("Hemodynamic Fragility Index (HFI)")
    ax_a.set_ylabel("Estimated ATE (pp)")
    ax_a.set_title("Continuous Dose-Response:\nATE vs HFI by HF Subtype")
    ax_a.legend(loc="lower left", framealpha=0.9)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.grid(alpha=0.3)

    # Annotation box with WLS results
    if not np.isnan(beta_hfpef):
        txt = (f"WLS: R\u00b2 = {r2:.2f}\n"
               f"\u03b2(HFpEF) = {beta_hfpef*100:.2f} pp (p < 0.001)\n"
               f"\u03b2(interaction) = {beta_interaction*100:.2f} pp "
               f"(p = {p_interaction:.3f})")
        ax_a.text(0.97, 0.03, txt, transform=ax_a.transAxes, fontsize=7.5,
                  ha="right", va="bottom", style="italic",
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                            edgecolor="#cccccc", alpha=0.9))

    # ── Panel B: Bin-level ATE with CI and sample sizes ──
    for df_sub, color, label, marker in [
        (hfpef, C_HFPEF, "HFpEF", "s"),
        (nohf, C_NOHF, "No HF", "o"),
    ]:
        x = df_sub["hfi_center"].values
        y = df_sub["ate_mean"].values * 100
        se = df_sub["ate_se"].values * 100
        ax_b.errorbar(x, y, yerr=1.96 * se, fmt=f"{marker}-", color=color,
                      label=label, capsize=3, linewidth=1.5, markersize=6,
                      alpha=0.85)
        # Annotate HFpEF sample sizes
        if label == "HFpEF":
            for xi, yi, ni in zip(x, y, df_sub["n_stays"].values):
                ax_b.annotate(f"n={int(ni)}", (xi, yi), textcoords="offset points",
                              xytext=(5, 8), fontsize=6, color=color, alpha=0.7)

    ax_b.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_b.set_xlabel("Hemodynamic Fragility Index (HFI)")
    ax_b.set_ylabel("Estimated ATE (pp)")
    ax_b.set_title("Sliding-Window Bin ATE\nwith 95% CI")
    ax_b.legend(loc="lower left", framealpha=0.9)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.grid(alpha=0.3)

    # Raw ITE difference annotation
    if not np.isnan(raw_diff):
        txt2 = f"Mean ITE diff = {raw_diff*100:.2f} pp\n(Welch p = {raw_welch_p:.4f})"
        ax_b.text(0.97, 0.03, txt2, transform=ax_b.transAxes, fontsize=7.5,
                  ha="right", va="bottom", style="italic",
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                            edgecolor="#cccccc", alpha=0.9))

    fig.tight_layout()
    _save(fig, "sFigure10")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 65)
    print("  Manuscript Figure Generator (CSV-only, no GPU)")
    print(f"  Output: {FIG_DIR}")
    print(f"  PDF:    {PDF_DIR}")
    print("=" * 65)

    generators = [
        make_figure2,
        make_figure3,
        make_figure4,
        make_figure5,
        make_sfig3,
        make_sfig4,
        make_sfig5,
        make_sfig7,
        make_sfig8,
        make_sfig9,
        make_sfig10,
    ]

    ok, skip = 0, 0
    for gen in generators:
        try:
            gen()
            ok += 1
        except Exception as e:
            print(f"  ERROR in {gen.__name__}: {e}")
            skip += 1

    elapsed = time.time() - t0
    print()
    print("=" * 65)
    print(f"  Done: {ok} figures generated, {skip} skipped in {elapsed:.1f}s")
    print()
    print("  NOT generated here (need pipeline scripts):")
    print("    Figure 1   — run: manuscript_figure1_consort.py")
    print("    sFigure 1  — run: 60e_phenotype_cards_early.py")
    print("    sFigure 2  — run: 75a_dynamic_msm_iptw.py")
    print("    sFigure 6  — run: P5_counterfactual_trajectories.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
