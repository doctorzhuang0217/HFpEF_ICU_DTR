#!/usr/bin/env python3
"""95b — Generate Manuscript-Ready Figures and Tables

Protocol: v3.1, Step 95b
Compute: C1-C2

Purpose
-------
Generate main and supplementary figures and tables for the manuscript,
drawing from the 95a master results registry and upstream step outputs.

Figures
-------
  F1. Cross-model ATE forest plot (75a + 80a–80e)
  F2. HFpEF specificity bar chart (80g non-shock stratum)
  F3. Sensitivity analysis verdict heatmap (85a–85j)
  F4. External validation comparison (MIMIC vs eICU)
  F5. Trimming sensitivity stability plot (75a / 85d)

Tables
------
  T1. Step C model comparison (manuscript Table 2)
  T2. Sensitivity analysis summary (manuscript Table 3)
  T3. External validation results (manuscript Table 4)
  T4. Consistency checklist (manuscript Supplementary Table)

Inputs
------
- tables/95a_results_registry.csv
- tables/75a_trimming_sensitivity.csv
- tables/85d_positivity_grid.csv
- tables/80g_hf_subtype_x_shock_interaction.csv
- tables/90b_eicu_transportability_comparison.csv
- tables/90e_consistency_checklist.csv

Outputs
-------
- figures/95b_f1_cross_model_ate_forest.png
- figures/95b_f2_hfpef_specificity.png
- figures/95b_f3_sensitivity_heatmap.png
- figures/95b_f4_external_validation.png
- figures/95b_f5_trimming_stability.png
- tables/95b_t1_step_c_comparison.csv
- tables/95b_t2_sensitivity_summary.csv
- tables/95b_t3_external_validation.csv
- tables/95b_t4_consistency_checklist.csv
- reports/95b_figures_tables_manifest.txt
"""

from __future__ import annotations

import logging
import os
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

# ── Paths ─────────────────────────────────────────────────────────────
ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
TABLES_DIR = ROOT / "tables"
FIGURES_DIR = ROOT / "figures"
REPORTS_DIR = ROOT / "reports"

for d in [TABLES_DIR, FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("95b")

# ── Plot defaults ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})


# ══════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════

def _safe_read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        log.warning("  Missing: %s", path.name)
        return None
    return pd.read_csv(path)


# ══════════════════════════════════════════════════════════════════════
#  FIGURE GENERATORS
# ══════════════════════════════════════════════════════════════════════

def fig_f1_cross_model_forest(registry: pd.DataFrame):
    """F1: Cross-model ATE forest plot."""
    log.info("  F1: Cross-model ATE forest plot...")

    # Collect models from R2
    r2 = registry[registry["section"] == "R2_step_c_models"]
    # Only main estimator per model (not bootstrap CI rows)
    pv = r2[~r2["result_id"].str.contains("ci_")]

    # Add 75a WIS
    r1 = registry[registry["section"] == "R1_primary_ope"]
    ate_wis = r1[r1["result_id"] == "75a_ci_ate_wis"]

    labels = []
    ates = []
    ci_los = []
    ci_his = []

    if len(ate_wis) > 0:
        r = ate_wis.iloc[0]
        labels.append("MSM-IPTW (WIS)")
        ates.append(float(r["value"]) * 100)
        ci_los.append(float(r["ci_lo"]) * 100 if pd.notna(r["ci_lo"]) else None)
        ci_his.append(float(r["ci_hi"]) * 100 if pd.notna(r["ci_hi"]) else None)

    model_display = {
        "80c_gtransformer_rollout_ow1_DM": "G-Transformer (DM)",
        "80c_gtransformer_rollout_ow1_FQE": "G-Transformer (FQE)",
        "80a_gnet_DM": "G-Net (DM)",
        "80a_gnet_FQE": "G-Net (FQE)",
        "80b_actin_DR": "ACTIN (DR)",
        "80b_actin_FQE": "ACTIN (FQE)",
        "80d_ct_legacy_FQE": "CT-Legacy (FQE)",
        "80e_mamba_cdsp_FQE": "Mamba-CDSP (FQE)",
    }

    for rid, display in model_display.items():
        rows = pv[pv["result_id"] == rid]
        if len(rows) > 0:
            r = rows.iloc[0]
            labels.append(display)
            ates.append(float(r["value"]) * 100)
            ci_los.append(None)
            ci_his.append(None)

    # Add 80c CI from bootstrap row
    ci_row = r2[r2["result_id"] == "80c_rollout_ow1_ci_DM"]
    if len(ci_row) > 0:
        r = ci_row.iloc[0]
        # Update the 80c DM entry with CI
        for i, l in enumerate(labels):
            if l == "G-Transformer (DM)":
                ci_los[i] = float(r["ci_lo"]) * 100 if pd.notna(r["ci_lo"]) else None
                ci_his[i] = float(r["ci_hi"]) * 100 if pd.notna(r["ci_hi"]) else None

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(labels))

    for i in range(len(labels)):
        color = "#2166ac" if i == 0 else "#d6604d" if "DM" in labels[i] else "#4393c3"
        ax.plot(ates[i], i, "o", color=color, markersize=8, zorder=3)
        if ci_los[i] is not None and ci_his[i] is not None:
            ax.plot([ci_los[i], ci_his[i]], [i, i], "-", color=color, linewidth=2, zorder=2)

    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("ATE (percentage points)")
    ax.set_title("Cross-Model ATE Comparison\n(negative = rule policy reduces mortality)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    path = FIGURES_DIR / "95b_f1_cross_model_ate_forest.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace('.png', '.pdf'), format='pdf',
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig)
    log.info("  Saved: %s (+pdf)", path)
    return path


def fig_f2_hfpef_specificity(registry: pd.DataFrame):
    """F2: HFpEF specificity bar chart (80g non-shock)."""
    log.info("  F2: HFpEF specificity bar chart...")

    r3 = registry[registry["section"] == "R3_hf_shock_interaction"]
    non_shock = r3[r3["subgroup"].str.contains("non_shock", na=False)]

    if len(non_shock) == 0:
        log.warning("  No non-shock data found for F2")
        return None

    # Publication-ready label mapping
    _hf_display = {
        "hfpef": "HFpEF", "no_hf": "No HF", "hfref": "HFrEF",
        "hfmref": "HFmrEF", "hf_ef_unknown": "HF (EF unknown)",
    }

    subtypes = []
    ate_vals = []
    for _, row in non_shock.iterrows():
        sg = str(row["subgroup"]).replace("_non_shock", "")
        ate = float(row["value"]) * 100
        subtypes.append(_hf_display.get(sg, sg))
        ate_vals.append(ate)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d6604d" if s == "HFpEF" else "#4393c3" if s == "No HF" else "#92c5de"
              for s in subtypes]
    bars = ax.barh(subtypes, ate_vals, color=colors, edgecolor="white", height=0.6)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("ATE (percentage points)")
    ax.set_title("HFpEF-Specific Effect in Non-Shock Stratum\n(G-Transformer rollout, DM estimator)")
    ax.invert_yaxis()

    for bar, val in zip(bars, ate_vals):
        # Place label inside bar (right-aligned within bar) to avoid overlapping y-axis labels
        if val < 0:
            ax.text(val / 2, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}pp", va="center", ha="center",
                    fontsize=9, fontweight="bold")
        else:
            ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}pp", va="center", ha="left",
                    fontsize=9, fontweight="bold")

    fig.tight_layout()
    path = FIGURES_DIR / "95b_f2_hfpef_specificity.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace('.png', '.pdf'), format='pdf',
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig)
    log.info("  Saved: %s (+pdf)", path)
    return path


def fig_f3_sensitivity_heatmap():
    """F3: Sensitivity analysis verdict heatmap."""
    log.info("  F3: Sensitivity verdict heatmap...")

    steps = ["EDQ", "SeqDeconf", "Positivity", "NegCtrl",
             "E-value", "Monitoring", "RRT", "Horizon",
             "Missingness"]
    verdicts = ["ROBUST", "SENSITIVE", "STABLE", "PASS",
                "WEAK", "ROBUST", "ROBUST", "SENSITIVE", "REFRAMED"]
    # Map to numeric: ROBUST/PASS/STABLE=2, REFRAMED=1, WEAK/SENSITIVE=0
    verdict_map = {"ROBUST": 2, "PASS": 2, "STABLE": 2, "REFRAMED": 1, "WEAK": 0, "SENSITIVE": 0}
    scores = [verdict_map.get(v, 1) for v in verdicts]

    fig, ax = plt.subplots(figsize=(12, 2.5))
    colors_map = {2: "#4daf4a", 1: "#ff7f00", 0: "#e41a1c"}
    bar_colors = [colors_map[s] for s in scores]

    ax.barh([0] * len(steps), [1] * len(steps),
            left=range(len(steps)), color=bar_colors, edgecolor="white", height=0.8)

    for i, (step, verdict) in enumerate(zip(steps, verdicts)):
        ax.text(i + 0.5, 0, f"{verdict}", ha="center", va="center",
                fontsize=8, fontweight="bold", color="white")

    ax.set_xticks([i + 0.5 for i in range(len(steps))])
    ax.set_xticklabels(steps, fontsize=8)
    ax.set_yticks([])
    ax.set_title("Sensitivity Analysis Verdicts")
    ax.set_xlim(-0.1, len(steps) + 0.1)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4daf4a", label="ROBUST / PASS / STABLE"),
        Patch(facecolor="#ff7f00", label="REFRAMED"),
        Patch(facecolor="#e41a1c", label="SENSITIVE / WEAK"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    fig.tight_layout()
    path = FIGURES_DIR / "95b_f3_sensitivity_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace('.png', '.pdf'), format='pdf',
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig)
    log.info("  Saved: %s (+pdf)", path)
    return path


def fig_f4_external_validation():
    """F4: External validation comparison (MIMIC vs eICU).

    Two-panel design:
      A) DM estimator: MIMIC vs eICU-full vs eICU-Tier-A (with CIs)
      B) FQE estimator: same three groups — highlights Tier A ≈ MIMIC
    DR excluded due to weight pathology (eICU DR ATE ≈ -616pp).
    """
    log.info("  F4: External validation comparison...")

    comp = _safe_read(TABLES_DIR / "90b_eicu_transportability_comparison.csv")
    tier_a = _safe_read(TABLES_DIR / "90b_05_eicu_tier_a_sensitivity.csv")
    if comp is None:
        return None

    # --- Extract values ---
    # MIMIC & eICU-full from transportability comparison
    dm_row = comp[comp["estimator"] == "DM"].iloc[0] if "DM" in comp["estimator"].values else None
    fqe_row = comp[comp["estimator"] == "FQE"].iloc[0] if "FQE" in comp["estimator"].values else None

    if dm_row is None:
        log.warning("  F4: DM row not found in transportability comparison")
        return None

    # Tier A values
    ta_dm_ate, ta_dm_ci_lo, ta_dm_ci_hi = np.nan, np.nan, np.nan
    ta_fqe_ate, ta_fqe_ci_lo, ta_fqe_ci_hi = np.nan, np.nan, np.nan
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

    # Convert to percentage points
    def pp(v):
        return v * 100 if pd.notna(v) else 0.0

    def pp_err(val, lo, hi):
        """Return (lower_err, upper_err) for errorbar, or (0, 0) if missing."""
        if pd.notna(lo) and pd.notna(hi) and pd.notna(val):
            return (abs(val - lo) * 100, abs(hi - val) * 100)
        return (0.0, 0.0)

    # --- Build data for each panel ---
    colors = {"MIMIC-IV": "#2166ac", "eICU (full)": "#d6604d", "eICU (Tier A)": "#4dac26"}
    labels = list(colors.keys())
    bar_colors = list(colors.values())

    # DM values
    dm_vals = [pp(dm_row["mimic_ate"]), pp(dm_row["eicu_ate"]), pp(ta_dm_ate)]
    dm_errs_lo = []
    dm_errs_hi = []
    # MIMIC DM CI
    lo, hi = pp_err(dm_row["mimic_ate"], dm_row.get("mimic_ci_lo"), dm_row.get("mimic_ci_hi"))
    dm_errs_lo.append(lo); dm_errs_hi.append(hi)
    # eICU-full DM CI
    lo, hi = pp_err(dm_row["eicu_ate"], dm_row.get("eicu_ci_lo"), dm_row.get("eicu_ci_hi"))
    dm_errs_lo.append(lo); dm_errs_hi.append(hi)
    # Tier A DM CI
    lo, hi = pp_err(ta_dm_ate, ta_dm_ci_lo, ta_dm_ci_hi)
    dm_errs_lo.append(lo); dm_errs_hi.append(hi)

    # FQE values
    fqe_vals = [0.0, 0.0, 0.0]
    fqe_errs_lo = [0.0, 0.0, 0.0]
    fqe_errs_hi = [0.0, 0.0, 0.0]
    if fqe_row is not None:
        fqe_vals[0] = pp(fqe_row["mimic_ate"])
        fqe_vals[1] = pp(fqe_row["eicu_ate"])
        fqe_vals[2] = pp(ta_fqe_ate)
        # MIMIC FQE CI (may be missing)
        lo, hi = pp_err(fqe_row["mimic_ate"], fqe_row.get("mimic_ci_lo"), fqe_row.get("mimic_ci_hi"))
        fqe_errs_lo[0] = lo; fqe_errs_hi[0] = hi
        # eICU-full FQE CI
        lo, hi = pp_err(fqe_row["eicu_ate"], fqe_row.get("eicu_ci_lo"), fqe_row.get("eicu_ci_hi"))
        fqe_errs_lo[1] = lo; fqe_errs_hi[1] = hi
        # Tier A FQE CI
        lo, hi = pp_err(ta_fqe_ate, ta_fqe_ci_lo, ta_fqe_ci_hi)
        fqe_errs_lo[2] = lo; fqe_errs_hi[2] = hi

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5), sharey=False)
    x = np.arange(3)
    width = 0.55

    # Two-line x-tick labels: name + sample size
    xlabels = ["MIMIC-IV\n(n=38,981)", "eICU (full)\n(n=83,765)", "eICU (Tier A)\n(n=2,680)"]

    # Panel A: DM
    bars1 = ax1.bar(x, dm_vals, width, color=bar_colors, alpha=0.85,
                    edgecolor="black", linewidth=0.5)
    ax1.errorbar(x, dm_vals, yerr=[dm_errs_lo, dm_errs_hi],
                 fmt="none", ecolor="black", capsize=4, linewidth=1.2)
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabels, fontsize=9)
    ax1.set_ylabel("ATE (percentage points)")
    ax1.set_title("A: Direct Method (DM)\n(non-rollout, primary estimator)", fontsize=11)
    ax1.grid(axis="y", alpha=0.3)
    # Add value labels inside bars
    for i, (v, bar) in enumerate(zip(dm_vals, bars1)):
        ax1.text(bar.get_x() + bar.get_width() / 2, v / 2,
                 f"{v:.2f}pp", ha="center", va="center", fontsize=9, fontweight="bold")

    # Panel B: FQE
    bars2 = ax2.bar(x, fqe_vals, width, color=bar_colors, alpha=0.85,
                    edgecolor="black", linewidth=0.5)
    ax2.errorbar(x, fqe_vals, yerr=[fqe_errs_lo, fqe_errs_hi],
                 fmt="none", ecolor="black", capsize=4, linewidth=1.2)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(xlabels, fontsize=9)
    ax2.set_ylabel("ATE (percentage points)")
    ax2.set_title("B: Fitted Q-Evaluation (FQE)\n(sign reversal in full sample, recovers in Tier A)", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)
    # Add value labels inside bars (or above for positive)
    for i, (v, bar) in enumerate(zip(fqe_vals, bars2)):
        if v < -0.5:
            ax2.text(bar.get_x() + bar.get_width() / 2, v / 2,
                     f"{v:.2f}pp", ha="center", va="center", fontsize=9, fontweight="bold")
        else:
            ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.25,
                     f"{v:+.2f}pp", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("Cross-Database ATE Comparison: MIMIC-IV vs eICU-CRD\n"
                 "(G-Transformer, DR excluded \u2014 weight pathology)",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    path = FIGURES_DIR / "95b_f4_external_validation.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace('.png', '.pdf'), format='pdf',
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig)
    log.info("  Saved: %s (+pdf)", path)
    return path


def fig_f5_trimming_stability():
    """F5: Trimming sensitivity stability plot."""
    log.info("  F5: Trimming sensitivity stability...")

    trim = _safe_read(TABLES_DIR / "75a_trimming_sensitivity.csv")
    pos = _safe_read(TABLES_DIR / "85d_positivity_grid.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: 75a WIS trimming
    if trim is not None:
        brackets = [f"[{r['trim_lo']:.0f},{r['trim_hi']:.0f}]" for _, r in trim.iterrows()]
        ates = trim["ate_wis"].values * 100
        axes[0].plot(range(len(ates)), ates, "o-", color="#2166ac", markersize=6)
        axes[0].set_xticks(range(len(brackets)))
        axes[0].set_xticklabels(brackets, rotation=45, ha="right", fontsize=8)
        axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
        axes[0].set_ylabel("ATE (percentage points)")
        axes[0].set_title("A. WIS Trimming Sensitivity")
        axes[0].grid(axis="y", alpha=0.3)

    # Panel B: 85d positivity grid (ALL subgroup)
    if pos is not None:
        all_pos = pos[pos["subgroup"] == "ALL"]
        pct_excluded = all_pos["pct_excluded"].values
        ates_pos = all_pos["ate"].values * 100
        axes[1].plot(pct_excluded, ates_pos, "s-", color="#d6604d", markersize=6)
        axes[1].set_xlabel("% Excluded")
        axes[1].set_ylabel("ATE (percentage points)")
        axes[1].set_title("B. Positivity Grid Stability")
        axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
        axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = FIGURES_DIR / "95b_f5_trimming_stability.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(str(path).replace('.png', '.pdf'), format='pdf',
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig)
    log.info("  Saved: %s (+pdf)", path)
    return path


# ══════════════════════════════════════════════════════════════════════
#  TABLE GENERATORS
# ══════════════════════════════════════════════════════════════════════

def table_t1_step_c(registry: pd.DataFrame) -> Path:
    """T1: Step C model comparison table."""
    log.info("  T1: Step C model comparison...")

    r2 = registry[registry["section"] == "R2_step_c_models"]
    pv = r2[~r2["result_id"].str.contains("ci_")]

    rows = []
    for _, r in pv.iterrows():
        rows.append({
            "model": r["model"],
            "estimator": r["estimator"],
            "ate_pp": round(float(r["value"]) * 100, 2) if pd.notna(r["value"]) else None,
            "note": r["note"],
        })

    # Add 75a
    r1 = registry[registry["section"] == "R1_primary_ope"]
    wis = r1[r1["result_id"] == "75a_ci_ate_wis"]
    if len(wis) > 0:
        r = wis.iloc[0]
        rows.insert(0, {
            "model": "75a_msm_iptw",
            "estimator": "WIS",
            "ate_pp": round(float(r["value"]) * 100, 2),
            "note": f"CI: [{float(r['ci_lo'])*100:.2f}, {float(r['ci_hi'])*100:.2f}]",
        })

    df = pd.DataFrame(rows)
    path = TABLES_DIR / "95b_t1_step_c_comparison.csv"
    df.to_csv(path, index=False)
    log.info("  Saved: %s", path)
    return path


def table_t2_sensitivity() -> Path:
    """T2: Sensitivity analysis summary table."""
    log.info("  T2: Sensitivity summary...")

    rows = [
        {"step": "85a", "analysis": "EDQ (early discharge quality)", "verdict": "ROBUST",
         "detail": "Baseline ATE stable across exclusion/bounding scenarios"},
        {"step": "85b", "analysis": "Sequential deconfounding", "verdict": "SENSITIVE",
         "detail": "ATE attenuates from Q1 (-2.43pp) to Q4 (-0.54pp) by proxy Z-score quartile"},
        {"step": "85d", "analysis": "Positivity grid", "verdict": "STABLE",
         "detail": "ATE stable across trimming brackets [1,99] to [20,80]"},
        {"step": "85e", "analysis": "Negative control outcomes", "verdict": "PASS 3/3",
         "detail": "Shuffled outcome/treatment near zero; pre-T0 lab null"},
        {"step": "85f", "analysis": "E-value + GOC", "verdict": "WEAK E-value, GOC ROBUST",
         "detail": "E-value=1.52 (CI bound=1.49); GOC exclusion ATE unchanged"},
        {"step": "85g", "analysis": "Monitoring intensity", "verdict": "ROBUST",
         "detail": "IPMW adjustment Δ<1%; trend p=0.121 (not significant)"},
        {"step": "85h", "analysis": "RRT sensitivity", "verdict": "ROBUST",
         "detail": "Exclusion/bounding/stratification all consistent"},
        {"step": "85i", "analysis": "Horizon variation", "verdict": "SENSITIVE",
         "detail": "48h sign reversal; 72h HFpEF ATE=-2.67pp (8.9× ratio)"},
        {"step": "85j", "analysis": "Informative missingness", "verdict": "REFRAMED",
         "detail": "Informative monitoring — methodological strength, not weakness"},
    ]

    df = pd.DataFrame(rows)
    path = TABLES_DIR / "95b_t2_sensitivity_summary.csv"
    df.to_csv(path, index=False)
    log.info("  Saved: %s", path)
    return path


def table_t3_external_validation(registry: pd.DataFrame) -> Path:
    """T3: External validation results table."""
    log.info("  T3: External validation results...")

    r5 = registry[registry["section"] == "R5_external_validation"]
    r6 = registry[registry["section"] == "R6_transportability"]

    rows = []
    # Main comparison
    comp = r5[r5["result_id"].str.startswith("90b_eicu_")]
    for _, r in comp.iterrows():
        rows.append({
            "estimator": r["estimator"],
            "eicu_ate_pp": round(float(r["value"]) * 100, 4) if pd.notna(r["value"]) else None,
            "note": r["note"],
            "database": r["database"],
        })

    # Grades
    for _, r in r6.iterrows():
        rows.append({
            "estimator": r["estimator"],
            "eicu_ate_pp": None,
            "note": f"Grade: {r['value']}; {r['note']}",
            "database": "grade",
        })

    df = pd.DataFrame(rows)
    path = TABLES_DIR / "95b_t3_external_validation.csv"
    df.to_csv(path, index=False)
    log.info("  Saved: %s", path)
    return path


def table_t4_consistency() -> Path:
    """T4: Consistency checklist (copy from 90e with formatting)."""
    log.info("  T4: Consistency checklist...")

    src = _safe_read(TABLES_DIR / "90e_consistency_checklist.csv")
    if src is None:
        return None

    path = TABLES_DIR / "95b_t4_consistency_checklist.csv"
    src.to_csv(path, index=False)
    log.info("  Saved: %s", path)
    return path


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("95b — Generate Manuscript-Ready Figures and Tables")
    print("=" * 72)
    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Load registry
    registry = _safe_read(TABLES_DIR / "95a_results_registry.csv")
    if registry is None:
        log.error("95a registry not found — run 95a first")
        return

    log.info("Loaded 95a registry: %d entries", len(registry))

    # ── Figures ──────────────────────────────────────────────────────
    log.info("[1/2] Generating figures...")
    outputs = []

    p = fig_f1_cross_model_forest(registry)
    if p:
        outputs.append(str(p))

    p = fig_f2_hfpef_specificity(registry)
    if p:
        outputs.append(str(p))

    p = fig_f3_sensitivity_heatmap()
    if p:
        outputs.append(str(p))

    p = fig_f4_external_validation()
    if p:
        outputs.append(str(p))

    p = fig_f5_trimming_stability()
    if p:
        outputs.append(str(p))

    # ── Tables ───────────────────────────────────────────────────────
    log.info("[2/2] Generating tables...")

    p = table_t1_step_c(registry)
    if p:
        outputs.append(str(p))

    p = table_t2_sensitivity()
    if p:
        outputs.append(str(p))

    p = table_t3_external_validation(registry)
    if p:
        outputs.append(str(p))

    p = table_t4_consistency()
    if p:
        outputs.append(str(p))

    # ── Manifest ─────────────────────────────────────────────────────
    manifest = [
        f"95b — Figures & Tables Manifest",
        f"Generated: {ts}",
        f"Total outputs: {len(outputs)}",
        "",
    ]
    for o in outputs:
        manifest.append(f"  {o}")

    manifest_path = REPORTS_DIR / "95b_figures_tables_manifest.txt"
    with open(manifest_path, "w") as f:
        f.write("\n".join(manifest) + "\n")
    outputs.append(str(manifest_path))

    elapsed = time.time() - t0
    print()
    print("=" * 72)
    print(f"95b COMPLETE. {len(outputs)} outputs generated in {elapsed:.1f}s.")
    print()
    for o in outputs:
        print(f"  {o}")
    print("=" * 72)


if __name__ == "__main__":
    main()
