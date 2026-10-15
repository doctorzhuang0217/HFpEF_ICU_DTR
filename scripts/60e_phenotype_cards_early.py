#!/usr/bin/env python3
"""
60e_phenotype_cards_early — Generate early phenotype cards.

Purpose
-------
Produce interpretable phenotype summaries ("cards") for each early
phenotype cluster. Cards show:
  - Clinical variable distributions (median, IQR) per cluster
  - Key hemodynamic / lab signatures
  - Cluster sizes and HF-subtype enrichment
  - Treatment patterns (fluid, vasopressor, diuretic exposure in 0-6h)

These cards support clinical validation of the discovered phenotypes
before using them for Step C stratification.

Protocol reference: Step 60e (v3.1 § 12)

Inputs
------
- data/interim/60a_early_features_v1_mimic.parquet
- data/interim/60b_early_clusters_v1_mimic.parquet
- data/interim/60d_rule_phenotype_v1_mimic.parquet
- data/interim/45d_outcomes_absorb_v1_mimic.parquet

Outputs
-------
- tables/60e_phenotype_cards.csv
- tables/60e_phenotype_outcomes.csv
- figures/60e_phenotype_radar.png  (if matplotlib available)
- reports/60e_phenotype_cards_summary.txt

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

# Variables for phenotype characterization (raw z-scored pre-T0 means)
CARD_VARS = [
    "map_pre_mean", "heart_rate_pre_mean", "spo2_pre_mean", "resp_rate_pre_mean",
    "lactate_pre_mean", "creatinine_pre_mean", "bun_pre_mean",
    "sodium_pre_mean", "potassium_pre_mean", "bicarbonate_pre_mean",
    "hemoglobin_pre_mean", "wbc_pre_mean", "platelet_pre_mean",
    "albumin_pre_mean", "bilirubin_pre_mean",
    "data_density_score", "trend_coverage",
]


def _compute_cluster_cards(
    feat_df: pd.DataFrame, labels: pd.Series, label_name: str
) -> pd.DataFrame:
    """Compute median + IQR for each variable by cluster."""
    rows = []
    for cluster_id in sorted(labels.unique()):
        mask = labels == cluster_id
        grp = feat_df[mask]
        for var in CARD_VARS:
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
    merged = pd.DataFrame({"stay_id": stay_ids, "cluster": labels})
    merged = merged.merge(outcomes_df[["stay_id", "mortality_28d", "mortality_7d"]],
                          on="stay_id", how="left")
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
    project_root = Path(proj_cfg.get("project_root", "."))
    interim_dir = resolve_path(proj_cfg, "interim")
    reports_dir = resolve_path(proj_cfg, "reports")
    tables_dir = resolve_path(proj_cfg, "tables")
    figures_dir = resolve_path(proj_cfg, "figures")
    for d in [reports_dir, tables_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("60e_phenotype_cards_early — Clinical phenotype characterization")
    print("=" * 72)

    # ── Load data ──────────────────────────────────────────────────────
    print("\n[1/4] Loading data ...")
    feat_df = pd.read_parquet(interim_dir / "60a_early_features_v1_mimic.parquet")
    n_stays = len(feat_df)

    cluster_df = pd.read_parquet(interim_dir / "60b_early_clusters_v1_mimic.parquet")
    rule_df = pd.read_parquet(interim_dir / "60d_rule_phenotype_v1_mimic.parquet")

    outcomes_path = interim_dir / "45d_outcomes_absorb_v1_mimic.parquet"
    outcomes_df = pd.read_parquet(outcomes_path) if outcomes_path.exists() else None

    print(f"  Stays: {n_stays:,}")

    # ── Compute cards ──────────────────────────────────────────────────
    print("\n[2/4] Computing phenotype cards ...")
    all_cards = []
    all_outcomes = []

    # Data-driven clusters
    selected_col = [c for c in cluster_df.columns if c.startswith("selected_k")]
    if selected_col:
        dd_labels = cluster_df[["stay_id", selected_col[0]]].drop_duplicates(subset=["stay_id"])
        dd_labels = feat_df[["stay_id"]].merge(dd_labels, on="stay_id", how="left")[selected_col[0]]
        cards = _compute_cluster_cards(feat_df, dd_labels, "data_driven")
        all_cards.append(cards)
        print(f"  Data-driven: {dd_labels.nunique()} clusters")

        if outcomes_df is not None:
            oc = _compute_outcome_summary(
                dd_labels, "data_driven", outcomes_df, feat_df["stay_id"]
            )
            all_outcomes.append(oc)

    # Rule-based phenotype
    if "rule_phenotype_4class" in rule_df.columns:
        rule_labels = rule_df.set_index("stay_id").loc[
            feat_df["stay_id"], "rule_phenotype_4class"
        ].values
        cards_rule = _compute_cluster_cards(
            feat_df, pd.Series(rule_labels), "rule_4class"
        )
        all_cards.append(cards_rule)
        print(f"  Rule-based: {len(np.unique(rule_labels))} classes")

        if outcomes_df is not None:
            oc_rule = _compute_outcome_summary(
                pd.Series(rule_labels), "rule_4class",
                outcomes_df, feat_df["stay_id"]
            )
            all_outcomes.append(oc_rule)

    # ── HF-subtype enrichment ──────────────────────────────────────────
    print("\n[3/4] Computing HF-subtype enrichment ...")
    enrichment_lines = []
    if "hf_subtype" in feat_df.columns and selected_col:
        dd_map = cluster_df[["stay_id", selected_col[0]]].drop_duplicates(subset=["stay_id"])
        enrich_df = feat_df[["stay_id", "hf_subtype"]].merge(dd_map, on="stay_id", how="left")
        ct = pd.crosstab(
            enrich_df[selected_col[0]],
            enrich_df["hf_subtype"],
            normalize="index",
        )
        enrichment_lines.append(ct.round(3).to_string())
        print("  Cross-tab computed")

    # ── Save outputs ───────────────────────────────────────────────────
    print("\n[4/4] Saving outputs ...")
    cards_all = pd.concat(all_cards, ignore_index=True) if all_cards else pd.DataFrame()
    cards_path = tables_dir / "60e_phenotype_cards.csv"
    cards_all.to_csv(cards_path, index=False)
    print(f"  Saved: {cards_path}")

    if all_outcomes:
        outcomes_all = pd.concat(all_outcomes, ignore_index=True)
        outcomes_path_out = tables_dir / "60e_phenotype_outcomes.csv"
        outcomes_all.to_csv(outcomes_path_out, index=False)
        print(f"  Saved: {outcomes_path_out}")

    # ── Radar plot (rule-based 4-class) ─────────────────────────────────
    PHENOTYPE_ORDER = ["shock_dominant", "congestion_dominant", "mixed", "neither"]
    PHENOTYPE_LABELS = {
        "shock_dominant": "Shock-predominant",
        "congestion_dominant": "Congestion-predominant",
        "mixed": "Mixed",
        "neither": "Neither",
    }
    PHENOTYPE_COLORS = {
        "shock_dominant": "#d62728",
        "congestion_dominant": "#1f77b4",
        "mixed": "#9467bd",
        "neither": "#2ca02c",
    }
    try:
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        import matplotlib.pyplot as plt

        rule_cards = cards_all[cards_all["label_source"] == "rule_4class"] if not cards_all.empty else pd.DataFrame()
        radar_vars = [v for v in CARD_VARS[:8] if not rule_cards.empty and v in rule_cards["variable"].values]

        if len(radar_vars) >= 4:
            fig, ax = plt.subplots(figsize=(8, 8),
                                   subplot_kw=dict(polar=True))
            angles = np.linspace(0, 2 * np.pi, len(radar_vars), endpoint=False)
            angles = np.concatenate([angles, [angles[0]]])

            for pheno in PHENOTYPE_ORDER:
                p_data = rule_cards[rule_cards["cluster"] == pheno]
                if p_data.empty:
                    continue
                vals = []
                for v in radar_vars:
                    row = p_data[p_data["variable"] == v]
                    vals.append(row["median"].values[0] if len(row) > 0 else 0)
                vals.append(vals[0])
                label = PHENOTYPE_LABELS.get(pheno, pheno)
                color = PHENOTYPE_COLORS.get(pheno, None)
                ax.plot(angles, vals, "-o", label=label, markersize=4, color=color)
                ax.fill(angles, vals, alpha=0.08, color=color)

            ax.set_thetagrids(np.degrees(angles[:-1]),
                              [v.replace("_pre_mean", "") for v in radar_vars])
            ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
            ax.set_title("Rule-Based Hemodynamic Phenotype Profiles\n(z-scored medians)",
                         pad=20)
            fig.tight_layout()
            radar_path = figures_dir / "60e_phenotype_radar.png"
            fig.savefig(radar_path, dpi=150, bbox_inches='tight', facecolor='white')
            fig.savefig(str(radar_path).replace('.png', '.pdf'), format='pdf',
                        bbox_inches='tight', dpi=300, facecolor='white')
            plt.close(fig)
            print(f"  Saved: {radar_path} (+pdf)")
    except ImportError:
        print("  matplotlib not available — skipping radar plot")

    # ── Summary report ─────────────────────────────────────────────────
    report_lines = [
        "=" * 72,
        "60e — Early Phenotype Cards (MIMIC-IV)",
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
                    f"    Cluster {row['cluster']}: n={int(row['n']):,} ({row['pct']}%), "
                    f"mort_28d={row['mortality_28d']}%, mort_7d={row['mortality_7d']}%"
                )

    if enrichment_lines:
        report_lines.extend([
            "",
            "HF-SUBTYPE ENRICHMENT (row-normalized)",
            "───────────────────────────────────────",
        ] + enrichment_lines)

    report_lines.extend([
        "",
        "INTERPRETATION GUIDE",
        "────────────────────",
        "  - Positive z-scores = above population mean",
        "  - Negative z-scores = below population mean",
        "  - Clusters with high MAP / low lactate = hemodynamically stable",
        "  - Clusters with low MAP / high lactate = shock-predominant",
        "  - Clusters with low SpO2 / high RR = respiratory-predominant",
        "",
        "NEXT STEPS",
        "──────────",
        "  Step 65: trajectory phenotyping (0-48h, explanatory only)",
        "  Step 70: policy framework setup using phenotype stratification",
    ])

    report = "\n".join(report_lines) + "\n"
    report_path = reports_dir / "60e_phenotype_cards_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n  Saved: {report_path}")
    print(report)
    print("[60e] Done.")


if __name__ == "__main__":
    main()
