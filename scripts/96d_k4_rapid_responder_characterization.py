#!/usr/bin/env python3
"""
96d_k4_rapid_responder_characterization — K=4 trajectory clinical characterization.

[EXPLORATORY_96X — NOT part of v3.1 core protocol. Results are provisional
and should NOT be cited in the manuscript main body without explicit upgrade.]

Purpose
-------
Characterize the K=4 trajectory clusters, focusing on the K4=2 "shock rapid
responder" subgroup (84.6% shock_predominant, 1.9% mortality) and the K4=0
"convergently deteriorating" subgroup (50/50 shock/non_shock, 27.2% mortality).

Two levels of analysis:
  Level 1 — Group-level aggregate trajectories:
    Mean MAP / lactate / HR / SpO2 / fluid / vasopressor over 12 bins (0-24h),
    stratified by K=4 cluster. One multi-panel figure showing temporal evolution.

  Level 2 — Representative case trajectories:
    P5-style individual patient trajectory plots for selected patients from
    K4=2 (rapid responders) vs K4=0 (convergently bad), with HFpEF focus.

Protocol reference: Exploratory extension (post-v3.1, Step 96 series)

Inputs
------
- data/interim/96b_trajectory_k4_clusters_v1_mimic.parquet
- data/interim/45c_state_tensors_v1_mimic.parquet
- data/interim/45d_outcomes_absorb_v1_mimic.parquet
- data/processed/phenotype_labels_frozen_v1_mimic.parquet
- configs/45c_normalization_params_v1.yaml

Outputs
-------
- figures/96d_k4_aggregate_trajectories.png
- figures/96d_k4_case_rapid_responder_*.png
- figures/96d_k4_case_convergent_bad_*.png
- figures/96d_k4_case_panel.png
- tables/96d_k4_aggregate_trajectory_data.csv
- tables/96d_k4_cluster_hf_enrichment.csv
- tables/96d_k4_selected_cases.csv
- reports/96d_k4_characterization_summary.txt

Compute: C2

[EXPLORATORY_96X]
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path

EXPLORATORY_TAG = "EXPLORATORY_96X"

# Bin / phase definitions (consistent with 65a)
N_BINS = 12
BIN_HOURS = 2
HOURS = np.arange(N_BINS) * BIN_HOURS

# Key variables for trajectory plotting (z-scored columns in state tensor)
PLOT_VARS = {
    "map": {"label": "MAP (mmHg)", "color": "#1976D2", "threshold": 65,
            "threshold_label": "MAP=65"},
    "heart_rate": {"label": "HR (bpm)", "color": "#FF9800", "threshold": None},
    "lactate": {"label": "Lactate (mmol/L)", "color": "#F44336", "threshold": 2.0,
                "threshold_label": "Lactate=2.0"},
    "spo2": {"label": "SpO2 (%)", "color": "#4CAF50", "threshold": 92,
             "threshold_label": "SpO2=92"},
    "creatinine": {"label": "Creatinine (mg/dL)", "color": "#9C27B0", "threshold": None},
}

# Treatment variables in state tensor
TREATMENT_VARS = {
    "fluid_input": {"label": "Fluid (mL/2h)", "color": "#2196F3"},
    "vasopressor_dose": {"label": "Vasopressor (mcg/kg/min)", "color": "#E91E63"},
    "diuretic_dose": {"label": "Diuretic (mg/2h)", "color": "#009688"},
}

# K4 cluster color scheme
K4_COLORS = {
    0: "#D32F2F",  # red — convergently bad (27.2% mort)
    1: "#388E3C",  # green — stable low-risk (7.4%)
    2: "#1565C0",  # blue — rapid responder (1.9%)
    3: "#F57C00",  # orange — intermediate (12.5%)
}

K4_LABELS = {
    0: "K4=0 (converge-bad, 27.2%)",
    1: "K4=1 (stable-low, 7.4%)",
    2: "K4=2 (rapid-resp, 1.9%)",
    3: "K4=3 (intermediate, 12.5%)",
}


def _denorm(z_values: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Denormalize z-scored values back to original scale."""
    return z_values * std + mean


def _compute_group_trajectories(
    state_df: pd.DataFrame,
    k4_labels: pd.DataFrame,
    norm_params: dict,
) -> pd.DataFrame:
    """Compute mean ± SEM trajectories for each K4 cluster and variable."""
    # Merge K4 labels
    merged = state_df.merge(
        k4_labels[["stay_id", "traj_k4_cluster"]],
        on="stay_id", how="inner",
    )

    rows = []
    all_vars = list(PLOT_VARS.keys()) + list(TREATMENT_VARS.keys())

    for var in all_vars:
        z_col = f"{var}_z"
        mask_col = f"{var}_mask"

        # Check which column exists
        if z_col in merged.columns:
            val_col = z_col
            needs_denorm = var in norm_params
        elif var in merged.columns:
            val_col = var
            needs_denorm = False
        else:
            continue

        has_mask = mask_col in merged.columns

        for k4 in sorted(merged["traj_k4_cluster"].unique()):
            cluster_data = merged[merged["traj_k4_cluster"] == k4]

            for b in range(N_BINS):
                bin_data = cluster_data[cluster_data["bin"] == b]
                if has_mask:
                    bin_data = bin_data[bin_data[mask_col] > 0]

                vals = bin_data[val_col].dropna()
                if len(vals) == 0:
                    continue

                mean_z = vals.mean()
                sem_z = vals.std() / np.sqrt(len(vals))

                if needs_denorm:
                    p = norm_params[var]
                    mean_raw = mean_z * p["std"] + p["mean"]
                    sem_raw = sem_z * p["std"]
                else:
                    mean_raw = mean_z
                    sem_raw = sem_z

                rows.append({
                    "variable": var,
                    "k4_cluster": k4,
                    "bin": b,
                    "hour": b * BIN_HOURS,
                    "n_obs": len(vals),
                    "mean_raw": round(mean_raw, 4),
                    "sem_raw": round(sem_raw, 4),
                    "mean_z": round(mean_z, 4),
                })

    return pd.DataFrame(rows)


def _plot_aggregate_trajectories(
    traj_data: pd.DataFrame,
    out_path: Path,
) -> None:
    """Create multi-panel aggregate trajectory figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clinical_vars = [v for v in PLOT_VARS if v in traj_data["variable"].unique()]
    treat_vars = [v for v in TREATMENT_VARS if v in traj_data["variable"].unique()]
    n_clinical = len(clinical_vars)
    n_treat = len(treat_vars)
    n_panels = n_clinical + n_treat

    if n_panels == 0:
        print("  WARNING: no variables found for plotting")
        return

    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0

    # Clinical panels
    for var in clinical_vars:
        ax = axes[panel_idx]
        var_data = traj_data[traj_data["variable"] == var]
        cfg = PLOT_VARS[var]

        for k4 in sorted(var_data["k4_cluster"].unique()):
            k4_data = var_data[var_data["k4_cluster"] == k4].sort_values("hour")
            ax.plot(k4_data["hour"], k4_data["mean_raw"],
                    "o-", color=K4_COLORS.get(k4, "grey"),
                    label=K4_LABELS.get(k4, f"K4={k4}"),
                    linewidth=2, markersize=4)
            ax.fill_between(
                k4_data["hour"],
                k4_data["mean_raw"] - 1.96 * k4_data["sem_raw"],
                k4_data["mean_raw"] + 1.96 * k4_data["sem_raw"],
                color=K4_COLORS.get(k4, "grey"), alpha=0.1,
            )

        if cfg["threshold"] is not None:
            ax.axhline(y=cfg["threshold"], color="grey", linestyle="--",
                       alpha=0.5, label=cfg.get("threshold_label", ""))

        ax.set_ylabel(cfg["label"], fontsize=10)
        ax.legend(fontsize=7, ncol=2, loc="upper right")
        ax.grid(alpha=0.2)
        if panel_idx == 0:
            ax.set_title(f"K=4 Trajectory Cluster Aggregate Profiles (0-24h)\n"
                         f"[{EXPLORATORY_TAG}]", fontsize=12, fontweight="bold")
        panel_idx += 1

    # Treatment panels
    for var in treat_vars:
        ax = axes[panel_idx]
        var_data = traj_data[traj_data["variable"] == var]
        cfg = TREATMENT_VARS[var]

        for k4 in sorted(var_data["k4_cluster"].unique()):
            k4_data = var_data[var_data["k4_cluster"] == k4].sort_values("hour")
            ax.plot(k4_data["hour"], k4_data["mean_raw"],
                    "s-", color=K4_COLORS.get(k4, "grey"),
                    label=K4_LABELS.get(k4, f"K4={k4}"),
                    linewidth=2, markersize=4)
            ax.fill_between(
                k4_data["hour"],
                k4_data["mean_raw"] - 1.96 * k4_data["sem_raw"],
                k4_data["mean_raw"] + 1.96 * k4_data["sem_raw"],
                color=K4_COLORS.get(k4, "grey"), alpha=0.1,
            )

        ax.set_ylabel(cfg["label"], fontsize=10)
        ax.legend(fontsize=7, ncol=2, loc="upper right")
        ax.grid(alpha=0.2)
        panel_idx += 1

    axes[-1].set_xlabel("Hours post-T0", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _select_representative_cases(
    k4_labels: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    pheno_df: pd.DataFrame,
) -> list[dict]:
    """Select representative patients from K4=2 (rapid responders) and K4=0."""
    merged = k4_labels.merge(outcomes_df[["stay_id", "mortality_28d", "mortality_7d"]],
                             on="stay_id", how="left")
    merged = merged.merge(
        pheno_df[["stay_id", "hf_subtype", "rule_phenotype_2class"]].drop_duplicates("stay_id"),
        on="stay_id", how="left",
    )

    selections = []
    target_profiles = [
        # (label, k4_cluster, hf_subtype, shock_status, mortality_preference)
        ("K4=2 rapid-resp, HFpEF shock, survived",
         2, "hfpef", "shock_predominant", 0),
        ("K4=2 rapid-resp, non-HF shock, survived",
         2, "no_hf", "shock_predominant", 0),
        ("K4=2 rapid-resp, HFpEF shock, died",
         2, "hfpef", "shock_predominant", 1),
        ("K4=0 converge-bad, HFpEF, died",
         0, "hfpef", None, 1),
        ("K4=0 converge-bad, non-HF shock, died",
         0, "no_hf", "shock_predominant", 1),
        ("K4=0 converge-bad, non-HF non-shock, died",
         0, "no_hf", "non_shock", 1),
    ]

    for label, k4, hf, pheno, mort_pref in target_profiles:
        mask = merged["traj_k4_cluster"] == k4
        if hf:
            mask &= merged["hf_subtype"] == hf
        if pheno:
            mask &= merged["rule_phenotype_2class"] == pheno
        if mort_pref is not None:
            mask &= merged["mortality_28d"] == mort_pref

        candidates = merged[mask]
        if len(candidates) == 0:
            # Relax mortality constraint
            mask_relaxed = merged["traj_k4_cluster"] == k4
            if hf:
                mask_relaxed &= merged["hf_subtype"] == hf
            if pheno:
                mask_relaxed &= merged["rule_phenotype_2class"] == pheno
            candidates = merged[mask_relaxed]
            if len(candidates) == 0:
                print(f"  SKIP: no candidates for '{label}'")
                continue

        # Pick median-index patient (not extreme)
        selected = candidates.iloc[len(candidates) // 2]
        selections.append({
            "label": label,
            "stay_id": int(selected["stay_id"]),
            "k4_cluster": int(selected["traj_k4_cluster"]),
            "hf_subtype": selected.get("hf_subtype", ""),
            "phenotype": selected.get("rule_phenotype_2class", ""),
            "mortality_28d": int(selected.get("mortality_28d", -1)),
        })

    return selections


def _plot_single_case(
    state_df: pd.DataFrame,
    stay_id: int,
    norm_params: dict,
    patient_info: dict,
    out_path: Path,
) -> bool:
    """Plot individual patient trajectory (MAP, lactate, HR, treatment)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    stay_data = state_df[state_df["stay_id"] == stay_id].sort_values("bin")
    if len(stay_data) == 0:
        print(f"    SKIP: no data for stay_id={stay_id}")
        return False

    fig = plt.figure(figsize=(13, 9))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 0.8], hspace=0.35)

    # ── Panel 1: MAP + HR ──
    ax1 = fig.add_subplot(gs[0])

    for var, color, ylabel, side in [
        ("map", "#1976D2", "MAP (mmHg)", "left"),
        ("heart_rate", "#FF9800", "HR (bpm)", "right"),
    ]:
        z_col = f"{var}_z"
        mask_col = f"{var}_mask"
        if z_col not in stay_data.columns:
            continue

        valid = stay_data
        if mask_col in stay_data.columns:
            valid = stay_data[stay_data[mask_col] > 0]

        vals_z = valid[z_col].values
        bins = valid["bin"].values
        hours = bins * BIN_HOURS

        if var in norm_params:
            vals = _denorm(vals_z, norm_params[var]["mean"], norm_params[var]["std"])
        else:
            vals = vals_z

        if side == "left":
            ax1.plot(hours, vals, "o-", color=color, label=PLOT_VARS[var]["label"],
                     linewidth=2, markersize=5)
            ax1.set_ylabel(PLOT_VARS[var]["label"], fontsize=9, color=color)
            if PLOT_VARS[var]["threshold"]:
                ax1.axhline(y=PLOT_VARS[var]["threshold"], color="red",
                            linestyle="--", alpha=0.4)
        else:
            ax1t = ax1.twinx()
            ax1t.plot(hours, vals, "s-", color=color, label=PLOT_VARS[var]["label"],
                      linewidth=1.5, markersize=4, alpha=0.7)
            ax1t.set_ylabel(PLOT_VARS[var]["label"], fontsize=9, color=color)

    k4_cl = patient_info["k4_cluster"]
    cl_color = K4_COLORS.get(k4_cl, "grey")
    ax1.set_title(
        f"{patient_info['label']}\n"
        f"stay_id={patient_info['stay_id']} | "
        f"HF={patient_info.get('hf_subtype', '')} | "
        f"Pheno={patient_info.get('phenotype', '')} | "
        f"Mort_28d={patient_info.get('mortality_28d', '?')}\n"
        f"[{EXPLORATORY_TAG}]",
        fontsize=10, fontweight="bold", color=cl_color,
    )
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.2)

    # ── Panel 2: Lactate + SpO2 ──
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    for var, color, ylabel, side in [
        ("lactate", "#F44336", "Lactate (mmol/L)", "left"),
        ("spo2", "#4CAF50", "SpO2 (%)", "right"),
    ]:
        z_col = f"{var}_z"
        mask_col = f"{var}_mask"
        if z_col not in stay_data.columns:
            continue

        valid = stay_data
        if mask_col in stay_data.columns:
            valid = stay_data[stay_data[mask_col] > 0]

        vals_z = valid[z_col].values
        bins = valid["bin"].values
        hours = bins * BIN_HOURS

        if var in norm_params:
            vals = _denorm(vals_z, norm_params[var]["mean"], norm_params[var]["std"])
        else:
            vals = vals_z

        marker = "D-" if side == "left" else "^-"
        if side == "left":
            ax2.plot(hours, vals, marker, color=color, label=PLOT_VARS[var]["label"],
                     linewidth=2, markersize=5)
            ax2.set_ylabel(PLOT_VARS[var]["label"], fontsize=9, color=color)
            if PLOT_VARS[var]["threshold"]:
                ax2.axhline(y=PLOT_VARS[var]["threshold"], color=color,
                            linestyle="--", alpha=0.3)
        else:
            ax2t = ax2.twinx()
            ax2t.plot(hours, vals, marker, color=color, label=PLOT_VARS[var]["label"],
                      linewidth=1.5, markersize=4, alpha=0.7)
            ax2t.set_ylabel(PLOT_VARS[var]["label"], fontsize=9, color=color)

    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(alpha=0.2)

    # ── Panel 3: Treatment intensity ──
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    treat_plotted = False
    for var, cfg in TREATMENT_VARS.items():
        z_col = f"{var}_z"
        raw_col = var
        if z_col in stay_data.columns:
            col = z_col
            if var in norm_params:
                vals = _denorm(stay_data[col].values, norm_params[var]["mean"],
                               norm_params[var]["std"])
            else:
                vals = stay_data[col].values
        elif raw_col in stay_data.columns:
            vals = stay_data[raw_col].values
        else:
            continue

        bins = stay_data["bin"].values
        hours = bins * BIN_HOURS
        ax3.bar(hours + list(TREATMENT_VARS.keys()).index(var) * 0.5,
                np.maximum(vals, 0), width=0.45,
                color=cfg["color"], alpha=0.7, label=cfg["label"])
        treat_plotted = True

    if treat_plotted:
        ax3.set_ylabel("Treatment intensity", fontsize=9)
        ax3.legend(fontsize=7, ncol=3, loc="upper right")
    ax3.set_xlabel("Hours post-T0", fontsize=10)
    ax3.grid(alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    proj_cfg = load_config()
    interim_dir = resolve_path(proj_cfg, "interim")
    processed_dir = resolve_path(proj_cfg, "processed")
    reports_dir = resolve_path(proj_cfg, "reports")
    tables_dir = resolve_path(proj_cfg, "tables")
    figures_dir = resolve_path(proj_cfg, "figures")
    root = Path(proj_cfg.get("project_root", Path(__file__).resolve().parent.parent))
    for d in [reports_dir, tables_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"96d_k4_rapid_responder_characterization [{EXPLORATORY_TAG}]")
    print("  K=4 trajectory cluster clinical characterization")
    print("=" * 72)

    # ── [1/7] Load K4 cluster labels ──────────────────────────────────
    print("\n[1/7] Loading K4 cluster labels ...")
    k4_path = interim_dir / "96b_trajectory_k4_clusters_v1_mimic.parquet"
    if not k4_path.exists():
        print(f"  ERROR: {k4_path} not found — run 96b first")
        sys.exit(1)
    k4_labels = pd.read_parquet(k4_path).drop_duplicates(subset=["stay_id"])
    print(f"  Stays: {len(k4_labels):,}")
    print(f"  K4 distribution: {k4_labels['traj_k4_cluster'].value_counts().sort_index().to_dict()}")

    # ── [2/7] Load state tensors ──────────────────────────────────────
    print("\n[2/7] Loading state tensors ...")
    state_path = interim_dir / "45c_state_tensors_v1_mimic.parquet"
    if not state_path.exists():
        print(f"  ERROR: {state_path} not found")
        sys.exit(1)
    state_df = pd.read_parquet(state_path)
    n_stays_state = state_df["stay_id"].nunique()
    print(f"  State tensor: {len(state_df):,} rows, {n_stays_state:,} stays")

    # Load normalization params
    norm_path = root / "configs" / "45c_normalization_params_v1.yaml"
    norm_params = {}
    if norm_path.exists():
        with open(norm_path) as f:
            norm_cfg = yaml.safe_load(f)
        norm_params = norm_cfg.get("params", {})
        print(f"  Normalization params: {len(norm_params)} variables")
    else:
        print("  WARNING: normalization params not found — using z-scores")

    # ── [3/7] Load outcomes and phenotype labels ──────────────────────
    print("\n[3/7] Loading outcomes and phenotype labels ...")
    outcomes_path = interim_dir / "45d_outcomes_absorb_v1_mimic.parquet"
    outcomes_df = pd.read_parquet(outcomes_path).drop_duplicates(subset=["stay_id"]) if outcomes_path.exists() else None

    pheno_path = processed_dir / "phenotype_labels_frozen_v1_mimic.parquet"
    pheno_df = pd.read_parquet(pheno_path).drop_duplicates(subset=["stay_id"]) if pheno_path.exists() else None

    if outcomes_df is not None:
        print(f"  Outcomes: {len(outcomes_df):,} stays")
    if pheno_df is not None:
        print(f"  Phenotype labels: {len(pheno_df):,} stays")

    # HF enrichment by K4 cluster
    if pheno_df is not None and "hf_subtype" in pheno_df.columns:
        merged_hf = k4_labels.merge(pheno_df[["stay_id", "hf_subtype"]],
                                     on="stay_id", how="left")
        hf_enrich = pd.crosstab(merged_hf["traj_k4_cluster"],
                                 merged_hf["hf_subtype"], normalize="index")
        print("\n  HF subtype enrichment by K4 cluster:")
        print(hf_enrich.round(3).to_string().replace("\n", "\n    "))

        hf_enrich_path = tables_dir / "96d_k4_cluster_hf_enrichment.csv"
        hf_enrich.to_csv(hf_enrich_path)
        print(f"  Saved: {hf_enrich_path}")

        # HFpEF counts per cluster
        hf_counts = pd.crosstab(merged_hf["traj_k4_cluster"], merged_hf["hf_subtype"])
        if "hfpef" in hf_counts.columns:
            print("\n  HFpEF counts by K4 cluster:")
            for k4 in sorted(hf_counts.index):
                n_hfpef = hf_counts.loc[k4, "hfpef"]
                n_total = hf_counts.loc[k4].sum()
                print(f"    K4={k4}: {n_hfpef} HFpEF / {n_total} total ({n_hfpef/n_total*100:.1f}%)")

    # ── [4/7] Compute group-level aggregate trajectories ──────────────
    print("\n[4/7] Computing group-level aggregate trajectories ...")
    traj_data = _compute_group_trajectories(state_df, k4_labels, norm_params)
    print(f"  {len(traj_data)} trajectory data points computed")

    traj_data_path = tables_dir / "96d_k4_aggregate_trajectory_data.csv"
    traj_data.to_csv(traj_data_path, index=False)
    print(f"  Saved: {traj_data_path}")

    # ── [4b/7] Compute rapid-responder delta flags ─
    print("\n[4b/7] Computing rapid-responder delta flags ...")
    # For MAP: check if early→mid and early→late show recovery (positive delta)
    # For vasopressor: check if early→mid and early→late show weaning (negative delta)
    delta_rows = []
    feat_path = interim_dir / "65a_trajectory_features_v1_mimic.parquet"
    if feat_path.exists():
        feat_df = pd.read_parquet(feat_path).drop_duplicates(subset=["stay_id"])
        feat_merged = feat_df.merge(
            k4_labels[["stay_id", "traj_k4_cluster"]], on="stay_id", how="inner"
        )
        MAP_PHASE = {"early": "map_early_mean", "mid": "map_mid_mean", "late": "map_late_mean"}
        VASO_PHASE = {"early": "vaso_early_total", "mid": "vaso_mid_total", "late": "vaso_late_total"}

        for k4 in sorted(feat_merged["traj_k4_cluster"].unique()):
            grp = feat_merged[feat_merged["traj_k4_cluster"] == k4]
            map_d_em = (grp[MAP_PHASE["mid"]] - grp[MAP_PHASE["early"]]).median()
            map_d_el = (grp[MAP_PHASE["late"]] - grp[MAP_PHASE["early"]]).median()
            vaso_d_em = (grp[VASO_PHASE["mid"]] - grp[VASO_PHASE["early"]]).median()
            vaso_d_el = (grp[VASO_PHASE["late"]] - grp[VASO_PHASE["early"]]).median()
            map_recovery = bool(map_d_em > 0 and map_d_el > 0)
            vaso_wean = bool(vaso_d_em < 0 and vaso_d_el < 0)
            delta_rows.append({
                "traj_k4_cluster": int(k4),
                "n": len(grp),
                "map_delta_early_mid_median": round(float(map_d_em), 4),
                "map_delta_early_late_median": round(float(map_d_el), 4),
                "vaso_delta_early_mid_median": round(float(vaso_d_em), 4),
                "vaso_delta_early_late_median": round(float(vaso_d_el), 4),
                "map_recovery_flag": map_recovery,
                "vaso_wean_flag": vaso_wean,
            })
            flag_str = f"MAP_recov={'YES' if map_recovery else 'no'}, vaso_wean={'YES' if vaso_wean else 'no'}"
            print(f"  K4={k4}: ΔMAP e→m={map_d_em:+.3f} e→l={map_d_el:+.3f} | "
                  f"ΔVaso e→m={vaso_d_em:+.4f} e→l={vaso_d_el:+.4f} | {flag_str}")

        delta_df = pd.DataFrame(delta_rows)
        delta_path = tables_dir / "96d_k4_cluster_delta_summary.csv"
        delta_df.to_csv(delta_path, index=False)
        print(f"  Saved: {delta_path}")
    else:
        delta_df = pd.DataFrame()
        print("  WARNING: 65a features not found — skipping delta flags")

    # ── [5/7] Plot group-level aggregate trajectories ─────────────────
    print("\n[5/7] Plotting group-level aggregate trajectories ...")
    try:
        agg_fig_path = figures_dir / "96d_k4_aggregate_trajectories.png"
        _plot_aggregate_trajectories(traj_data, agg_fig_path)
        print(f"  Saved: {agg_fig_path}")
    except ImportError:
        print("  matplotlib not available — skipping")

    # ── [6/7] Select and plot representative cases ────────────────────
    print("\n[6/7] Selecting representative cases ...")
    case_summaries = []

    if outcomes_df is not None and pheno_df is not None:
        selections = _select_representative_cases(k4_labels, outcomes_df, pheno_df)
        print(f"  Selected {len(selections)} cases:")
        for s in selections:
            print(f"    {s['label']}: stay_id={s['stay_id']}, "
                  f"K4={s['k4_cluster']}, mort={s['mortality_28d']}")

        # Plot individual cases
        print("\n  Generating individual case trajectory plots ...")
        for i, sel in enumerate(selections):
            safe_label = (sel["label"].replace(" ", "_").replace(",", "")
                         .replace("=", "").lower()[:40])
            filename = f"96d_k4_case_{i+1}_{safe_label}.png"
            out_path = figures_dir / filename
            ok = _plot_single_case(state_df, sel["stay_id"], norm_params, sel, out_path)
            if ok:
                print(f"    Saved: {out_path}")
                sel["figure"] = str(out_path)
            case_summaries.append(sel)

        # Save case selection table
        cases_df = pd.DataFrame(case_summaries)
        cases_path = tables_dir / "96d_k4_selected_cases.csv"
        cases_df.to_csv(cases_path, index=False)
        print(f"  Saved: {cases_path}")

    # ── [7/7] Generate report ─────────────────────────────────────────
    print("\n[7/7] Generating summary report ...")

    report_lines = [
        "=" * 72,
        f"96d — K=4 Trajectory Cluster Characterization [{EXPLORATORY_TAG}]",
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
        "Characterize K=4 trajectory clusters with aggregate temporal profiles",
        "and representative case trajectories, focusing on:",
        "  - K4=2: 'Shock rapid responders' (84.6% shock, 1.9% mort)",
        "  - K4=0: 'Convergently deteriorating' (50/50 shock, 27.2% mort)",
        "",
        "GROUP-LEVEL AGGREGATE TRAJECTORIES",
        "───────────────────────────────────",
        f"Variables plotted: {', '.join(list(PLOT_VARS.keys()) + list(TREATMENT_VARS.keys()))}",
        f"Data points: {len(traj_data)}",
        "See: figures/96d_k4_aggregate_trajectories.png",
        "",
    ]

    if case_summaries:
        report_lines.extend([
            "REPRESENTATIVE CASES",
            "────────────────────",
        ])
        for cs in case_summaries:
            report_lines.append(
                f"  {cs['label']}: stay_id={cs['stay_id']}, "
                f"K4={cs['k4_cluster']}, mort_28d={cs['mortality_28d']}"
            )
        report_lines.append("")

    if not delta_df.empty:
        report_lines.extend([
            "RAPID-RESPONDER DELTA FLAGS",
            "───────────────────────────",
            delta_df.to_string(index=False),
            "",
        ])
        # Key contrast
        d0 = delta_df[delta_df["traj_k4_cluster"] == 0]
        d2 = delta_df[delta_df["traj_k4_cluster"] == 2]
        if not d0.empty and not d2.empty:
            d0r = d0.iloc[0]
            d2r = d2.iloc[0]
            report_lines.extend([
                "KEY CONTRAST: K4=2 vs K4=0",
                "───────────────────────────",
                f"  K4=2 MAP recovery flag:    {bool(d2r['map_recovery_flag'])}",
                f"  K4=0 MAP recovery flag:    {bool(d0r['map_recovery_flag'])}",
                f"  K4=2 Vasopressor wean flag: {bool(d2r['vaso_wean_flag'])}",
                f"  K4=0 Vasopressor wean flag: {bool(d0r['vaso_wean_flag'])}",
                f"  ΔMAP (K4=2 minus K4=0, early→mid): {d2r['map_delta_early_mid_median'] - d0r['map_delta_early_mid_median']:+.4f}",
                f"  ΔVaso (K4=2 minus K4=0, early→mid): {d2r['vaso_delta_early_mid_median'] - d0r['vaso_delta_early_mid_median']:+.4f}",
                "",
            ])

    report_lines.extend([
        "KEY CLINICAL QUESTIONS",
        "─────────────────────",
        "1. Does K4=2 show rapid MAP recovery in first 6h?",
        "2. Does K4=2 show early vasopressor weaning vs K4=0?",
        "3. Does K4=0 show progressive lactate rise / creatinine rise?",
        "4. Is K4=2's low mortality explained by treatment response or",
        "   by lower baseline severity despite shock criteria?",
        "5. What is the HFpEF enrichment in each K4 cluster?",
        "",
        "MANUSCRIPT RELEVANCE",
        "────────────────────",
        "The aggregate trajectory figure (96d) is a strong candidate for",
        "Supplementary Figure. It visually demonstrates that trajectory",
        "phenotypes capture clinically meaningful temporal evolution patterns",
        "beyond the T0 snapshot. The K4=2 'rapid responder' finding enriches",
        "the Discussion of treatment effect heterogeneity.",
        "",
        "OUTPUTS",
        "───────",
        "  figures/96d_k4_aggregate_trajectories.png",
    ])
    for cs in case_summaries:
        if "figure" in cs:
            report_lines.append(f"  {cs['figure']}")
    report_lines.extend([
        "  tables/96d_k4_aggregate_trajectory_data.csv",
        "  tables/96d_k4_cluster_hf_enrichment.csv",
        "  tables/96d_k4_selected_cases.csv",
        "",
        f"[{EXPLORATORY_TAG}] END",
    ])

    report = "\n".join(report_lines) + "\n"
    report_path = reports_dir / "96d_k4_characterization_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {report_path}")
    print(report)
    print(f"[96d] Done. [{EXPLORATORY_TAG}]")


if __name__ == "__main__":
    main()
