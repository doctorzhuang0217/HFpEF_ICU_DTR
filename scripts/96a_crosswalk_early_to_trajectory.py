#!/usr/bin/env python3
"""
96a_crosswalk_early_to_trajectory — Early-to-trajectory phenotype crosswalk.

[EXPLORATORY_96X — NOT part of v3.1 core protocol. Results are provisional
and should NOT be cited in the manuscript main body without explicit upgrade.]

Purpose
-------
Examine how Step 60 early phenotypes (rule-based 2-class and 4-class) map onto
Step 65 trajectory phenotypes (K=2 data-driven). This answers:

  1. Do the 2 early roots each split into distinct trajectory branches?
  2. Is there a "2 roots → 4 branches" hierarchical structure?
  3. Which early×trajectory cells show the most divergent outcomes?

The crosswalk also produces a Sankey/alluvial-style flow diagram and
conditional probability matrices to support Aim 3 interpretation.

Protocol reference: Exploratory extension (post-v3.1, Step 96 series)

Inputs
------
- data/processed/phenotype_labels_frozen_v1_mimic.parquet
- data/interim/65a_trajectory_features_v1_mimic.parquet
- data/interim/45d_outcomes_absorb_v1_mimic.parquet

Outputs
-------
- tables/96a_crosswalk_contingency.csv
- tables/96a_crosswalk_conditional_probs.csv
- tables/96a_crosswalk_outcome_by_cell.csv
- figures/96a_crosswalk_sankey.png
- figures/96a_crosswalk_heatmap.png
- reports/96a_crosswalk_summary.txt

Compute: C1

[EXPLORATORY_96X]
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path

# Tag for manuscript project to filter on
EXPLORATORY_TAG = "EXPLORATORY_96X"

# Clinical variables for cell-level profiling
PROFILE_VARS = [
    "map_early_mean", "map_late_mean", "map_traj_slope",
    "heart_rate_early_mean", "heart_rate_late_mean", "heart_rate_traj_slope",
    "lactate_early_mean", "lactate_late_mean", "lactate_traj_slope",
    "creatinine_early_mean", "creatinine_late_mean", "creatinine_traj_slope",
    "spo2_early_mean", "spo2_late_mean", "spo2_traj_slope",
    "fluid_total", "vaso_total", "diuretic_total",
    "cum_fluid_balance_final",
]


def _compute_cell_profiles(
    merged: pd.DataFrame,
    cell_col: str,
    var_list: list[str],
) -> pd.DataFrame:
    """Compute median + IQR for each variable by crosswalk cell."""
    rows = []
    for cell_id in sorted(merged[cell_col].unique()):
        grp = merged[merged[cell_col] == cell_id]
        for var in var_list:
            if var not in grp.columns:
                continue
            vals = grp[var].dropna()
            if len(vals) == 0:
                continue
            rows.append({
                "cell": cell_id,
                "variable": var,
                "n_obs": len(vals),
                "median": round(vals.median(), 4),
                "q25": round(vals.quantile(0.25), 4),
                "q75": round(vals.quantile(0.75), 4),
                "mean": round(vals.mean(), 4),
                "std": round(vals.std(), 4),
            })
    return pd.DataFrame(rows)


def _plot_sankey(ct: pd.DataFrame, out_path: Path) -> bool:
    """Plot a Sankey/alluvial diagram from contingency table."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  matplotlib not available — skipping Sankey plot")
        return False

    fig, ax = plt.subplots(figsize=(10, 6))

    early_labels = ct.index.tolist()
    traj_labels = ct.columns.tolist()
    n_early = len(early_labels)
    n_traj = len(traj_labels)

    # Positions
    left_y = np.linspace(0.9, 0.1, n_early)
    right_y = np.linspace(0.9, 0.1, n_traj)

    total = ct.values.sum()
    early_totals = ct.sum(axis=1).values
    traj_totals = ct.sum(axis=0).values

    # Color scheme
    colors_early = plt.cm.Set2(np.linspace(0, 1, max(n_early, 3)))
    colors_traj = plt.cm.Set1(np.linspace(0, 1, max(n_traj, 3)))

    # Draw flow bands
    left_offsets = np.zeros(n_early)
    right_offsets = np.zeros(n_traj)

    for i, el in enumerate(early_labels):
        bar_h = (early_totals[i] / total) * 0.7
        ax.barh(left_y[i], 0.05, height=bar_h, left=0.05, color=colors_early[i],
                edgecolor="grey", linewidth=0.5, align="center")
        ax.text(0.02, left_y[i], str(el), ha="right", va="center", fontsize=8,
                fontweight="bold")

        for j, tl in enumerate(traj_labels):
            flow = ct.iloc[i, j]
            if flow == 0:
                continue
            flow_h = (flow / total) * 0.7
            # Simple connecting polygon
            from matplotlib.patches import FancyArrowPatch
            alpha = max(0.15, min(0.7, flow / total * 3))
            ax.annotate(
                "", xy=(0.85, right_y[j] + right_offsets[j]),
                xytext=(0.10, left_y[i] + left_offsets[i]),
                arrowprops=dict(
                    arrowstyle="-",
                    color=(*colors_early[i][:3], alpha),
                    lw=max(1, flow / total * 40),
                    connectionstyle="arc3,rad=0.1",
                ),
            )
            # Annotate flow count
            mid_x = 0.48
            mid_y = (left_y[i] + left_offsets[i] + right_y[j] + right_offsets[j]) / 2
            ax.text(mid_x, mid_y, f"{flow:,}", fontsize=6, ha="center",
                    va="center", color="grey")
            left_offsets[i] += flow_h * 0.3
            right_offsets[j] += flow_h * 0.3

    for j, tl in enumerate(traj_labels):
        bar_h = (traj_totals[j] / total) * 0.7
        ax.barh(right_y[j], 0.05, height=bar_h, left=0.85, color=colors_traj[j],
                edgecolor="grey", linewidth=0.5, align="center")
        ax.text(0.93, right_y[j], f"traj_{tl}", ha="left", va="center",
                fontsize=8, fontweight="bold")

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")
    ax.set_title("Early Phenotype → Trajectory Phenotype Crosswalk\n"
                 f"[{EXPLORATORY_TAG}]", fontsize=11, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_heatmap(outcome_df: pd.DataFrame, out_path: Path) -> bool:
    """Plot outcome heatmap by crosswalk cell."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    if outcome_df.empty:
        return False

    pivot = outcome_df.pivot_table(
        index="early_phenotype", columns="trajectory_cluster",
        values="mortality_28d", aggfunc="first",
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"traj_{c}" for c in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Trajectory Cluster")
    ax.set_ylabel("Early Phenotype")
    ax.set_title(f"28-day Mortality (%) by Early×Trajectory Cell\n[{EXPLORATORY_TAG}]",
                 fontsize=10, fontweight="bold")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        fontsize=9, color="black" if val < 20 else "white")

    fig.colorbar(im, ax=ax, label="Mortality (%)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    proj_cfg = load_config()
    processed_dir = resolve_path(proj_cfg, "processed")
    interim_dir = resolve_path(proj_cfg, "interim")
    reports_dir = resolve_path(proj_cfg, "reports")
    tables_dir = resolve_path(proj_cfg, "tables")
    figures_dir = resolve_path(proj_cfg, "figures")
    for d in [reports_dir, tables_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"96a_crosswalk_early_to_trajectory [{EXPLORATORY_TAG}]")
    print("  Early phenotype → trajectory phenotype crosswalk analysis")
    print("=" * 72)

    # ── [1/6] Load frozen phenotype labels ────────────────────────────
    print("\n[1/6] Loading frozen phenotype labels ...")
    frozen_path = processed_dir / "phenotype_labels_frozen_v1_mimic.parquet"
    if not frozen_path.exists():
        print(f"  ERROR: {frozen_path} not found")
        sys.exit(1)
    frozen = pd.read_parquet(frozen_path)
    frozen = frozen.drop_duplicates(subset=["stay_id"])
    n_stays = len(frozen)
    print(f"  Stays: {n_stays:,}")
    print(f"  Columns: {list(frozen.columns)}")

    # Identify early and trajectory columns
    early_2class_col = "rule_phenotype_2class"
    early_4class_col = "rule_phenotype_4class"
    traj_col = [c for c in frozen.columns if c.startswith("trajectory_data_driven_k")]
    if not traj_col:
        print("  ERROR: no trajectory_data_driven_k* column found")
        sys.exit(1)
    traj_col = traj_col[0]
    print(f"  Early 2-class: {early_2class_col}")
    print(f"  Early 4-class: {early_4class_col}")
    print(f"  Trajectory: {traj_col}")

    # ── [2/6] Contingency tables ──────────────────────────────────────
    print("\n[2/6] Computing contingency tables ...")

    contingency_rows = []

    # 2-class × trajectory
    ct_2 = pd.crosstab(frozen[early_2class_col], frozen[traj_col])
    print(f"\n  Rule 2-class × Trajectory K=2:")
    print(ct_2.to_string().replace("\n", "\n    "))

    ct_2_norm_row = pd.crosstab(
        frozen[early_2class_col], frozen[traj_col], normalize="index"
    )
    print(f"\n  Row-normalized (P(traj | early)):")
    print(ct_2_norm_row.round(3).to_string().replace("\n", "\n    "))

    ct_2_norm_col = pd.crosstab(
        frozen[early_2class_col], frozen[traj_col], normalize="columns"
    )
    print(f"\n  Column-normalized (P(early | traj)):")
    print(ct_2_norm_col.round(3).to_string().replace("\n", "\n    "))

    for early_val in ct_2.index:
        for traj_val in ct_2.columns:
            contingency_rows.append({
                "early_source": early_2class_col,
                "early_value": early_val,
                "traj_source": traj_col,
                "traj_value": traj_val,
                "count": int(ct_2.loc[early_val, traj_val]),
                "p_traj_given_early": round(ct_2_norm_row.loc[early_val, traj_val], 4),
                "p_early_given_traj": round(ct_2_norm_col.loc[early_val, traj_val], 4),
            })

    # 4-class × trajectory
    ct_4 = pd.crosstab(frozen[early_4class_col], frozen[traj_col])
    print(f"\n  Rule 4-class × Trajectory K=2:")
    print(ct_4.to_string().replace("\n", "\n    "))

    ct_4_norm_row = pd.crosstab(
        frozen[early_4class_col], frozen[traj_col], normalize="index"
    )
    for early_val in ct_4.index:
        for traj_val in ct_4.columns:
            contingency_rows.append({
                "early_source": early_4class_col,
                "early_value": early_val,
                "traj_source": traj_col,
                "traj_value": traj_val,
                "count": int(ct_4.loc[early_val, traj_val]),
                "p_traj_given_early": round(ct_4_norm_row.loc[early_val, traj_val], 4),
                "p_early_given_traj": round(
                    pd.crosstab(
                        frozen[early_4class_col], frozen[traj_col],
                        normalize="columns"
                    ).loc[early_val, traj_val], 4
                ),
            })

    # HF subtype × early × trajectory (if available)
    if "hf_subtype" in frozen.columns:
        print(f"\n  HF subtype × Early 2-class × Trajectory:")
        ct_hf = pd.crosstab(
            [frozen["hf_subtype"], frozen[early_2class_col]],
            frozen[traj_col],
        )
        print(ct_hf.to_string().replace("\n", "\n    "))

    contingency_df = pd.DataFrame(contingency_rows)

    # ── [3/6] Load outcomes and compute cell-level mortality ──────────
    print("\n[3/6] Computing cell-level outcome profiles ...")
    outcomes_path = interim_dir / "45d_outcomes_absorb_v1_mimic.parquet"
    outcome_rows = []

    if outcomes_path.exists():
        outcomes_df = pd.read_parquet(outcomes_path)
        outcomes_df = outcomes_df.drop_duplicates(subset=["stay_id"])

        merged = frozen.merge(outcomes_df[["stay_id", "mortality_28d", "mortality_7d"]],
                              on="stay_id", how="left")

        # 2-class crosswalk cells
        for early_val in merged[early_2class_col].unique():
            for traj_val in merged[traj_col].unique():
                mask = (merged[early_2class_col] == early_val) & (merged[traj_col] == traj_val)
                cell = merged[mask]
                if len(cell) == 0:
                    continue
                outcome_rows.append({
                    "early_source": early_2class_col,
                    "early_phenotype": early_val,
                    "trajectory_cluster": traj_val,
                    "n": len(cell),
                    "pct": round(len(cell) / n_stays * 100, 1),
                    "mortality_28d": round(cell["mortality_28d"].mean() * 100, 1),
                    "mortality_7d": round(cell["mortality_7d"].mean() * 100, 1),
                })

        # 4-class crosswalk cells
        for early_val in merged[early_4class_col].unique():
            for traj_val in merged[traj_col].unique():
                mask = (merged[early_4class_col] == early_val) & (merged[traj_col] == traj_val)
                cell = merged[mask]
                if len(cell) == 0:
                    continue
                outcome_rows.append({
                    "early_source": early_4class_col,
                    "early_phenotype": early_val,
                    "trajectory_cluster": traj_val,
                    "n": len(cell),
                    "pct": round(len(cell) / n_stays * 100, 1),
                    "mortality_28d": round(cell["mortality_28d"].mean() * 100, 1),
                    "mortality_7d": round(cell["mortality_7d"].mean() * 100, 1),
                })

        outcome_df = pd.DataFrame(outcome_rows)
        print(f"  {len(outcome_df)} crosswalk cells computed")

        # Key finding: within-early-group trajectory divergence
        print("\n  Outcome divergence within early phenotypes (2-class × trajectory):")
        subset_2 = outcome_df[outcome_df["early_source"] == early_2class_col]
        for ep in subset_2["early_phenotype"].unique():
            ep_cells = subset_2[subset_2["early_phenotype"] == ep]
            if len(ep_cells) >= 2:
                mort_range = ep_cells["mortality_28d"].max() - ep_cells["mortality_28d"].min()
                print(f"    {ep}: mortality range = {mort_range:.1f}pp across trajectory clusters")
    else:
        print("  WARNING: outcomes file not found")
        outcome_df = pd.DataFrame()

    # ── [4/6] Cell-level clinical profiles ────────────────────────────
    print("\n[4/6] Computing cell-level clinical profiles ...")
    feat_path = interim_dir / "65a_trajectory_features_v1_mimic.parquet"
    if feat_path.exists():
        feat_df = pd.read_parquet(feat_path)
        feat_df = feat_df.drop_duplicates(subset=["stay_id"])
        cell_merged = frozen[["stay_id", early_2class_col, traj_col]].merge(
            feat_df, on="stay_id", how="left",
        )
        cell_merged["crosswalk_cell"] = (
            cell_merged[early_2class_col].astype(str) + "_×_traj_"
            + cell_merged[traj_col].astype(str)
        )
        available_vars = [v for v in PROFILE_VARS if v in cell_merged.columns]
        cell_profiles = _compute_cell_profiles(cell_merged, "crosswalk_cell", available_vars)
        print(f"  {len(cell_profiles)} profile rows computed ({len(available_vars)} variables)")
    else:
        cell_profiles = pd.DataFrame()
        print("  WARNING: trajectory features not found — skipping profiles")

    # ── [5/6] Visualizations ──────────────────────────────────────────
    print("\n[5/6] Generating visualizations ...")

    sankey_path = figures_dir / "96a_crosswalk_sankey.png"
    sankey_ok = _plot_sankey(ct_2, sankey_path)
    if sankey_ok:
        print(f"  Saved: {sankey_path}")

    heatmap_path = figures_dir / "96a_crosswalk_heatmap.png"
    heatmap_ok = False
    if not outcome_df.empty:
        subset_2_out = outcome_df[outcome_df["early_source"] == early_2class_col]
        heatmap_ok = _plot_heatmap(subset_2_out, heatmap_path)
        if heatmap_ok:
            print(f"  Saved: {heatmap_path}")

    # ── [6/6] Save outputs ────────────────────────────────────────────
    print("\n[6/6] Saving outputs ...")

    contingency_path = tables_dir / "96a_crosswalk_contingency.csv"
    contingency_df.to_csv(contingency_path, index=False)
    print(f"  Saved: {contingency_path}")

    if not outcome_df.empty:
        outcome_path = tables_dir / "96a_crosswalk_outcome_by_cell.csv"
        outcome_df.to_csv(outcome_path, index=False)
        print(f"  Saved: {outcome_path}")

    if not cell_profiles.empty:
        profiles_path = tables_dir / "96a_crosswalk_cell_profiles.csv"
        cell_profiles.to_csv(profiles_path, index=False)
        print(f"  Saved: {profiles_path}")

    # ── Report ────────────────────────────────────────────────────────
    report_lines = [
        "=" * 72,
        f"96a — Early-to-Trajectory Phenotype Crosswalk [{EXPLORATORY_TAG}]",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Tag: {EXPLORATORY_TAG}",
        "",
        "STATUS: EXPLORATORY — results are provisional and NOT part of the",
        "v3.1 core protocol. Do not cite in manuscript main body without",
        "explicit upgrade decision.",
        "",
        "PURPOSE",
        "───────",
        "Examine whether 2 early phenotypes (Step 60) split into distinct",
        "trajectory branches (Step 65), supporting a '2 roots → 4 branches'",
        "hierarchical disease course interpretation.",
        "",
        "CONTINGENCY: Rule 2-class × Trajectory",
        "────────────────────────────────────────",
        ct_2.to_string(),
        "",
        "Row-normalized P(trajectory | early phenotype):",
        ct_2_norm_row.round(3).to_string(),
        "",
        "CONTINGENCY: Rule 4-class × Trajectory",
        "────────────────────────────────────────",
        ct_4.to_string(),
        "",
    ]

    if not outcome_df.empty:
        report_lines.extend([
            "OUTCOME BY CROSSWALK CELL (2-class × trajectory)",
            "─────────────────────────────────────────────────",
        ])
        subset_2_report = outcome_df[outcome_df["early_source"] == early_2class_col]
        report_lines.append(subset_2_report.to_string(index=False))
        report_lines.append("")

        # Key finding
        report_lines.append("KEY FINDING: Within-early-group trajectory divergence")
        for ep in subset_2_report["early_phenotype"].unique():
            ep_cells = subset_2_report[subset_2_report["early_phenotype"] == ep]
            if len(ep_cells) >= 2:
                mort_range = ep_cells["mortality_28d"].max() - ep_cells["mortality_28d"].min()
                report_lines.append(
                    f"  {ep}: Δmortality_28d = {mort_range:.1f}pp across trajectory clusters"
                )
        report_lines.append("")

    report_lines.extend([
        "INTERPRETATION",
        "──────────────",
        "If within-early-group trajectory divergence is substantial (>5pp mortality",
        "difference), this supports the '2 roots → 4 branches' hypothesis:",
        "  - Each early phenotype funnels patients into trajectory sub-branches",
        "  - Treatment response heterogeneity may explain divergence",
        "  - Step 65 trajectory labels serve as post-hoc explanatory enrichment",
        "",
        "OUTPUTS",
        "───────",
        f"  tables/96a_crosswalk_contingency.csv",
        f"  tables/96a_crosswalk_outcome_by_cell.csv",
        f"  tables/96a_crosswalk_cell_profiles.csv",
        f"  figures/96a_crosswalk_sankey.png {'(generated)' if sankey_ok else '(skipped)'}",
        f"  figures/96a_crosswalk_heatmap.png {'(generated)' if heatmap_ok else '(skipped)'}",
        "",
        f"[{EXPLORATORY_TAG}] END",
    ])

    report = "\n".join(report_lines) + "\n"
    report_path = reports_dir / "96a_crosswalk_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {report_path}")
    print(report)
    print(f"[96a] Done. [{EXPLORATORY_TAG}]")


if __name__ == "__main__":
    main()
