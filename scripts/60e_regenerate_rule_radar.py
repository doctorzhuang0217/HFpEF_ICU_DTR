#!/usr/bin/env python3
"""
60e_regenerate_rule_radar — Regenerate sFigure 1 as rule-based 4-class radar.

Reads the existing 60e_phenotype_cards.csv (rule_4class rows) and produces
a radar plot for the 4 rule-based hemodynamic phenotypes:
  shock_dominant, congestion_dominant, mixed, neither.

This replaces the old K=3 data-driven clustering radar.

Outputs
-------
- figures/60e_phenotype_radar.png  (overwrites old K=3 version)
- figures/60e_phenotype_radar.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "tables"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Variables to show on radar (first 8 from CARD_VARS, matching old plot)
RADAR_VARS = [
    "map_pre_mean", "heart_rate_pre_mean", "spo2_pre_mean",
    "resp_rate_pre_mean", "lactate_pre_mean", "creatinine_pre_mean",
    "bun_pre_mean", "sodium_pre_mean",
]

# Display labels (strip _pre_mean suffix)
DISPLAY_LABELS = [v.replace("_pre_mean", "") for v in RADAR_VARS]

# Phenotype display order and labels
PHENOTYPE_ORDER = ["shock_dominant", "congestion_dominant", "mixed", "neither"]
PHENOTYPE_LABELS = {
    "shock_dominant": "Shock-predominant",
    "congestion_dominant": "Congestion-predominant",
    "mixed": "Mixed",
    "neither": "Neither",
}
PHENOTYPE_COLORS = {
    "shock_dominant": "#d62728",      # red
    "congestion_dominant": "#1f77b4", # blue
    "mixed": "#9467bd",              # purple
    "neither": "#2ca02c",            # green
}


def main() -> None:
    cards_path = TABLES_DIR / "60e_phenotype_cards.csv"
    if not cards_path.exists():
        raise FileNotFoundError(f"Missing {cards_path}")

    cards = pd.read_csv(cards_path)
    rule_cards = cards[cards["label_source"] == "rule_4class"].copy()

    if rule_cards.empty:
        raise ValueError("No rule_4class rows in 60e_phenotype_cards.csv")

    # Filter to radar variables
    rule_cards = rule_cards[rule_cards["variable"].isin(RADAR_VARS)]

    # ── Build radar ───────────────────────────────────────────────────
    n_vars = len(RADAR_VARS)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for pheno in PHENOTYPE_ORDER:
        p_data = rule_cards[rule_cards["cluster"] == pheno]
        if p_data.empty:
            continue
        vals = []
        for v in RADAR_VARS:
            row = p_data[p_data["variable"] == v]
            vals.append(row["median"].values[0] if len(row) > 0 else 0)
        vals.append(vals[0])  # close polygon

        label = PHENOTYPE_LABELS.get(pheno, pheno)
        color = PHENOTYPE_COLORS.get(pheno, None)
        ax.plot(angles, vals, "-o", label=label, markersize=4, color=color)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]), DISPLAY_LABELS)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    ax.set_title("Rule-Based Hemodynamic Phenotype Profiles\n(z-scored medians)",
                 pad=20)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURES_DIR / "60e_phenotype_radar.png"
    pdf_path = FIGURES_DIR / "60e_phenotype_radar.pdf"
    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300,
                facecolor="white")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print("Done — sFigure 1 now shows rule-based 4-class phenotypes.")


if __name__ == "__main__":
    main()
