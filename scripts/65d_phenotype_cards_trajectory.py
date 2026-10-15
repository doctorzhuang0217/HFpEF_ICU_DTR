#!/usr/bin/env python3
"""
65d_phenotype_cards_trajectory — Generate trajectory phenotype cards.

Purpose
-------
Produce interpretable phenotype summaries for each trajectory cluster.
Cards show:
  - Phase-specific clinical profiles (early/mid/late)
  - Treatment patterns across phases
  - Trajectory dynamics (slope, variability, early-late delta)
  - Outcome association (mortality rates per cluster)
  - HF-subtype enrichment

These cards support clinical interpretation of trajectory phenotypes
as an explanatory enrichment layer for Aim 3.

Protocol reference: Step 65d (v3.1 § 12)

Inputs
------
- data/interim/65a_trajectory_features_v1_mimic.parquet
- data/interim/65b_trajectory_clusters_v1_mimic.parquet
- data/interim/45d_outcomes_absorb_v1_mimic.parquet

Outputs
-------
- tables/65d_trajectory_phenotype_cards.csv
- tables/65d_trajectory_phenotype_outcomes.csv
- figures/65d_trajectory_heatmap.png  (if matplotlib available)
- reports/65d_trajectory_phenotype_cards_summary.txt

Compute: C1-C2
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path

# Key variables for trajectory characterization
CARD_VARS_BY_PHASE = {
    "early": [
        "map_early_mean", "heart_rate_early_mean", "spo2_early_mean",
        "lactate_early_mean", "creatinine_early_mean",
    ],
    "mid": [
        "map_mid_mean", "heart_rate_mid_mean", "spo2_mid_mean",
        "lactate_mid_mean", "creatinine_mid_mean",
    ],
    "late": [
        "map_late_mean", "heart_rate_late_mean", "spo2_late_mean",
        "lactate_late_mean", "creatinine_late_mean",
    ],
}

TRAJ_DYNAMICS_VARS = [
    "map_traj_slope", "heart_rate_traj_slope", "lactate_traj_slope",
    "creatinine_traj_slope", "spo2_traj_slope",
    "map_traj_std", "heart_rate_traj_std", "lactate_traj_std",
    "map_early_late_delta", "heart_rate_early_late_delta",
    "lactate_early_late_delta", "creatinine_early_late_delta",
]

TREATMENT_VARS = [
    "fluid_total", "vaso_total", "diuretic_total",
    "fluid_early_total", "fluid_mid_total", "fluid_late_total",
    "vaso_early_total", "vaso_mid_total", "vaso_late_total",
    "fluid_late_vs_early", "vaso_late_vs_early",
    "cum_fluid_balance_final",
]


def _compute_cluster_cards(
    feat_df: pd.DataFrame,
    labels: pd.Series,
    label_name: str,
    var_list: list[str],
) -> pd.DataFrame:
    """Compute median + IQR for each variable by cluster."""
    rows = []
    for cluster_id in sorted(labels.unique()):
        mask = labels == cluster_id
        grp = feat_df[mask]
        for var in var_list:
            if var not in grp.columns:
                continue
            vals = grp[var].dropna()
            if len(vals) == 0:
                continue
            rows.append({
                "label_source": label_name,
                "cluster": cluster_id,
                "variable": var,
                "n_obs": len(vals),
                "median": round(vals.median(), 4),
                "q25": round(vals.quantile(0.25), 4),
                "q75": round(vals.quantile(0.75), 4),
                "mean": round(vals.mean(), 4),
                "std": round(vals.std(), 4),
            })
    return pd.DataFrame(rows)


def _compute_outcome_summary(
    labels: pd.Series,
    label_name: str,
    outcomes_df: pd.DataFrame,
    stay_ids: pd.Series,
) -> pd.DataFrame:
    """Compute outcome rates per cluster."""
    merged = pd.DataFrame({"stay_id": stay_ids, "cluster": labels.values})
    merged = merged.merge(
        outcomes_df[["stay_id", "mortality_28d", "mortality_7d"]],
        on="stay_id", how="left",
    )
    rows = []
    for cluster_id in sorted(labels.unique()):
        grp = merged[merged["cluster"] == cluster_id]
        n = len(grp)
        rows.append({
            "label_source": label_name,
            "cluster": cluster_id,
            "n": n,
            "pct": round(n / len(merged) * 100, 1),
            "mortality_28d": round(grp["mortality_28d"].mean() * 100, 1),
            "mortality_7d": round(grp["mortality_7d"].mean() * 100, 1),
        })
    return pd.DataFrame(rows)


def main() -> None:
    proj_cfg = load_config()
    interim_dir = resolve_path(proj_cfg, "interim")
    reports_dir = resolve_path(proj_cfg, "reports")
    tables_dir = resolve_path(proj_cfg, "tables")
    figures_dir = resolve_path(proj_cfg, "figures")
    for d in [reports_dir, tables_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("65d_phenotype_cards_trajectory — Trajectory phenotype cards")
    print("=" * 72)

    # ── Load data ─────────────────────────────────────────────────────
    print("\n[1/4] Loading data ...")
    feat_df = pd.read_parquet(
        interim_dir / "65a_trajectory_features_v1_mimic.parquet"
    )
    cluster_df = pd.read_parquet(
        interim_dir / "65b_trajectory_clusters_v1_mimic.parquet"
    )
    cluster_df = cluster_df.drop_duplicates(subset=["stay_id"])
    n_stays = len(feat_df)
    print(f"  Stays: {n_stays:,}")

    outcomes_path = interim_dir / "45d_outcomes_absorb_v1_mimic.parquet"
    outcomes_df = (
        pd.read_parquet(outcomes_path) if outcomes_path.exists() else None
    )

    # ── Compute cards ─────────────────────────────────────────────────
    print("\n[2/4] Computing trajectory phenotype cards ...")
    all_vars = []
    for phase_vars in CARD_VARS_BY_PHASE.values():
        all_vars.extend(phase_vars)
    all_vars.extend(TRAJ_DYNAMICS_VARS)
    all_vars.extend(TREATMENT_VARS)

    all_cards = []
    all_outcomes = []

    # Data-driven trajectory clusters
    selected_col = [
        c for c in cluster_df.columns if c.startswith("traj_selected_k")
    ]
    if selected_col:
        dd_labels = cluster_df[["stay_id", selected_col[0]]].drop_duplicates(
            subset=["stay_id"]
        )
        dd_labels = feat_df[["stay_id"]].merge(
            dd_labels, on="stay_id", how="left"
        )[selected_col[0]]
        cards = _compute_cluster_cards(
            feat_df, dd_labels, "trajectory_data_driven", all_vars
        )
        all_cards.append(cards)
        print(f"  Data-driven: {dd_labels.nunique()} clusters")

        if outcomes_df is not None:
            oc = _compute_outcome_summary(
                dd_labels, "trajectory_data_driven",
                outcomes_df, feat_df["stay_id"],
            )
            all_outcomes.append(oc)

    # GMVAE clusters if available
    if "gmvae_cluster" in cluster_df.columns:
        gmvae_map = cluster_df[["stay_id", "gmvae_cluster"]].drop_duplicates(
            subset=["stay_id"]
        )
        gmvae_labels = feat_df[["stay_id"]].merge(
            gmvae_map, on="stay_id", how="left"
        )["gmvae_cluster"]
        if gmvae_labels.notna().sum() > 0:
            cards_gmvae = _compute_cluster_cards(
                feat_df, gmvae_labels.fillna(-1).astype(int),
                "trajectory_gmvae", all_vars,
            )
            all_cards.append(cards_gmvae)
            print(f"  GMVAE: {gmvae_labels.nunique()} clusters")

            if outcomes_df is not None:
                oc_gmvae = _compute_outcome_summary(
                    gmvae_labels.fillna(-1).astype(int),
                    "trajectory_gmvae",
                    outcomes_df, feat_df["stay_id"],
                )
                all_outcomes.append(oc_gmvae)

    # ── HF-subtype enrichment ─────────────────────────────────────────
    print("\n[3/4] Computing HF-subtype enrichment ...")
    enrichment_lines = []
    if "hf_subtype" in feat_df.columns and selected_col:
        dd_map = cluster_df[["stay_id", selected_col[0]]].drop_duplicates(
            subset=["stay_id"]
        )
        enrich_df = feat_df[["stay_id", "hf_subtype"]].merge(
            dd_map, on="stay_id", how="left"
        )
        ct = pd.crosstab(
            enrich_df[selected_col[0]],
            enrich_df["hf_subtype"],
            normalize="index",
        )
        enrichment_lines.append(ct.round(3).to_string())
        print("  Cross-tab computed")

    # ── Save outputs ──────────────────────────────────────────────────
    print("\n[4/4] Saving outputs ...")
    cards_all = (
        pd.concat(all_cards, ignore_index=True)
        if all_cards
        else pd.DataFrame()
    )
    cards_path = tables_dir / "65d_trajectory_phenotype_cards.csv"
    cards_all.to_csv(cards_path, index=False)
    print(f"  Saved: {cards_path}")

    if all_outcomes:
        outcomes_all = pd.concat(all_outcomes, ignore_index=True)
        outcomes_path_out = tables_dir / "65d_trajectory_phenotype_outcomes.csv"
        outcomes_all.to_csv(outcomes_path_out, index=False)
        print(f"  Saved: {outcomes_path_out}")

    # ── Phase heatmap ─────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if selected_col and not cards_all.empty:
            dd_cards = cards_all[
                cards_all["label_source"] == "trajectory_data_driven"
            ]
            # Build phase × cluster heatmap for key vitals
            key_vars = ["map", "heart_rate", "lactate", "spo2", "creatinine"]
            phases = ["early", "mid", "late"]
            clusters = sorted(dd_cards["cluster"].unique())

            heatmap_data = []
            y_labels = []
            for var in key_vars:
                for phase in phases:
                    col_name = f"{var}_{phase}_mean"
                    row_vals = []
                    for cid in clusters:
                        match = dd_cards[
                            (dd_cards["cluster"] == cid)
                            & (dd_cards["variable"] == col_name)
                        ]
                        row_vals.append(
                            match["median"].values[0] if len(match) > 0 else 0
                        )
                    heatmap_data.append(row_vals)
                    y_labels.append(f"{var}/{phase}")

            if heatmap_data:
                fig, ax = plt.subplots(figsize=(8, max(6, len(y_labels) * 0.4)))
                im = ax.imshow(
                    heatmap_data, aspect="auto", cmap="RdBu_r",
                    interpolation="nearest",
                )
                ax.set_xticks(range(len(clusters)))
                ax.set_xticklabels([f"C{c}" for c in clusters])
                ax.set_yticks(range(len(y_labels)))
                ax.set_yticklabels(y_labels, fontsize=8)
                ax.set_xlabel("Trajectory Cluster")
                ax.set_title("Trajectory Phenotype Profiles (z-scored medians)")
                fig.colorbar(im, ax=ax, shrink=0.8)
                fig.tight_layout()
                heatmap_path = figures_dir / "65d_trajectory_heatmap.png"
                fig.savefig(heatmap_path, dpi=150)
                plt.close(fig)
                print(f"  Saved: {heatmap_path}")
    except ImportError:
        print("  matplotlib not available — skipping heatmap")

    # ── Summary report ────────────────────────────────────────────────
    report_lines = [
        "=" * 72,
        "65d — Trajectory Phenotype Cards (MIMIC-IV)",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "PHENOTYPE SOURCES",
        "─────────────────",
    ]

    if all_outcomes:
        for oc_df in all_outcomes:
            src = oc_df["label_source"].iloc[0]
            report_lines.append(f"\n  {src}:")
            for _, row in oc_df.iterrows():
                report_lines.append(
                    f"    Cluster {row['cluster']}: n={int(row['n']):,} "
                    f"({row['pct']}%), "
                    f"mort_28d={row['mortality_28d']}%, "
                    f"mort_7d={row['mortality_7d']}%"
                )

    if enrichment_lines:
        report_lines.extend([
            "",
            "HF-SUBTYPE ENRICHMENT (row-normalized)",
            "───────────────────────────────────────",
        ] + enrichment_lines)

    report_lines.extend([
        "",
        "TRAJECTORY DYNAMICS INTERPRETATION",
        "──────────────────────────────────",
        "  - traj_slope > 0: variable increasing over 24h trajectory",
        "  - traj_slope < 0: variable decreasing over trajectory",
        "  - early_late_delta: change from early (0-6h) to late (18-24h) phase",
        "  - Clusters with improving MAP/lactate = treatment-responsive",
        "  - Clusters with worsening trajectory = refractory/deteriorating",
        "",
        "NOTE",
        "────",
        "  Trajectory phenotype is EXPLANATORY ONLY (protocol § 12).",
        "  It provides Aim 3 results but does NOT stratify Step C models.",
        "",
        "NEXT STEPS",
        "──────────",
        "  65e: freeze phenotype labels for Step C",
        "  Step 70: policy framework setup",
    ])

    report = "\n".join(report_lines) + "\n"
    report_path = reports_dir / "65d_trajectory_phenotype_cards_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n  Saved: {report_path}")
    print(report)
    print("[65d] Done.")


if __name__ == "__main__":
    main()
