#!/usr/bin/env python3
"""P3 — Time-Varying Cumulative Treatment Effect Curve

Protocol: v3.1-patch, Step P3
Compute tier: C3 (direct execution)
Depends on: 80c G-Transformer trained model

Objective: Decompose the 24h aggregate ATE into per-decision-point cumulative
effects, revealing the temporal window of maximum treatment benefit.

For each of the 12 decision points (2h bins, 0-24h post-T0):
  - Run truncated-horizon OPE: apply rule policy up to time t,
    then follow observed policy for remaining steps
  - Plot cumulative ATE(t) and identify maximum marginal benefit window
  - Stratify by HF subtype and phenotype

Outputs:
  - tables/P3_time_varying_ate.csv
  - figures/P3_cumulative_ate_curve.png
  - figures/P3_cumulative_ate_by_hf_subtype.png
  - reports/P3_time_varying_ate_summary.txt
"""

from __future__ import annotations

import datetime
import gc
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Project root ──
ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(ROOT / "src"))

from hfpef_dtr.models.dtr_dataset import DTRDataset, load_dtr_data
from hfpef_dtr.models.dataset import make_train_val_test_split
from hfpef_dtr.models.gtransformer import build_gtransformer_from_config

# ── Paths ──
CONFIG_PATH = ROOT / "configs" / "80_step_c_training_v1.yaml"
MODEL_CKPT = ROOT / "models" / "80c_gtransformer_best.pt"
TABLES_DIR = ROOT / "tables"
FIGURES_DIR = ROOT / "figures"
REPORTS_DIR = ROOT / "reports"

for d in [TABLES_DIR, FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def evaluate_dm_truncated_horizon(
    model,
    dataset: DTRDataset,
    device: torch.device,
    horizon_bins: int,
    batch_size: int = 64,
) -> dict:
    """
    DM evaluation with truncated rule-policy horizon.

    Apply rule_action for bins 0..horizon_bins-1, then observed action
    for bins horizon_bins..T-1.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    obs_probs = []
    cf_probs = []
    n_bins = dataset.n_bins

    for batch in loader:
        temporal = batch["temporal"].to(device)
        static = batch["static"].to(device)
        action = batch["action"].to(device)
        rule_action = batch["rule_action"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        absorbing_mask = batch["absorbing_mask"].to(device)

        # Build hybrid action: rule for t < horizon_bins, observed otherwise
        hybrid_action = action.clone()
        if horizon_bins > 0:
            hybrid_action[:, :min(horizon_bins, n_bins)] = rule_action[:, :min(horizon_bins, n_bins)]

        with torch.no_grad():
            if hasattr(model, "predict_counterfactual_rollout"):
                out_obs = model.predict_counterfactual_rollout(
                    temporal, static, action, valid_mask, absorbing_mask
                )
                out_cf = model.predict_counterfactual_rollout(
                    temporal, static, hybrid_action, valid_mask, absorbing_mask
                )
            else:
                out_obs = model.predict_counterfactual(
                    temporal, static, action, valid_mask, absorbing_mask
                )
                out_cf = model.predict_counterfactual(
                    temporal, static, hybrid_action, valid_mask, absorbing_mask
                )

        obs_probs.append(out_obs["outcome_prob"].cpu().numpy())
        cf_probs.append(out_cf["outcome_prob"].cpu().numpy())

    obs_probs = np.concatenate(obs_probs).squeeze()
    cf_probs = np.concatenate(cf_probs).squeeze()

    return {
        "v_observed": float(obs_probs.mean()),
        "v_rule": float(cf_probs.mean()),
        "ate": float(cf_probs.mean() - obs_probs.mean()),
    }


def main():
    print("=" * 72)
    print("P3 — Time-Varying Cumulative Treatment Effect Curve")
    print("=" * 72)
    t0 = time.time()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 1. Load config, data, model ──
    print("[1] Loading config and data...")
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    state_df, outcomes_df, propensity_df, temporal_cols, static_cols, merge_map = (
        load_dtr_data(ROOT, config)
    )
    split_cfg = config["split"]
    all_stay_ids = state_df["stay_id"].unique()
    _, _, test_ids = make_train_val_test_split(
        all_stay_ids, outcomes_df,
        train_frac=split_cfg["train_frac"],
        val_frac=split_cfg["val_frac"],
        test_frac=split_cfg["test_frac"],
        stratify_col=split_cfg["stratify_col"],
        seed=split_cfg["seed"],
    )

    test_ds = DTRDataset(
        state_df=state_df, outcomes_df=outcomes_df,
        propensity_df=propensity_df, stay_ids=test_ids,
        temporal_cols=temporal_cols, static_cols=static_cols,
        merge_map=merge_map, n_bins=config["grid"]["n_bins"],
    )
    print(f"  Test set: {len(test_ds)} stays, {test_ds.n_actions} actions")

    pheno_path = ROOT / config["data"]["phenotype_labels"]
    pheno_df = pd.read_parquet(pheno_path)

    # Free memory
    del state_df, outcomes_df, propensity_df
    gc.collect()

    print("[2] Loading 80c G-Transformer...")
    ckpt = torch.load(MODEL_CKPT, map_location=device, weights_only=False)
    model = build_gtransformer_from_config(
        d_temporal=test_ds.d_temporal, d_static=test_ds.d_static,
        n_actions=test_ds.n_actions,
        model_cfg=config["models"]["80c_gtransformer"],
        grid_cfg=config["grid"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded from epoch {ckpt['epoch']}")

    # ── 3. Cumulative ATE curve (overall) ──
    print("[3] Computing cumulative ATE curve...")
    n_bins = config["grid"]["n_bins"]
    bin_hours = config["grid"]["bin_hours"]
    eval_batch_size = int(os.environ.get("P3_EVAL_BATCH_SIZE", "64"))
    print(f"  Evaluation batch_size: {eval_batch_size}")

    results = []
    for h in tqdm(range(n_bins + 1), desc="  Horizons"):
        dm = evaluate_dm_truncated_horizon(
            model, test_ds, device, horizon_bins=h, batch_size=eval_batch_size
        )
        hours = h * bin_hours
        result = {
            "horizon_bins": h,
            "horizon_hours": hours,
            "group": "Overall",
            "n_stays": len(test_ds),
            "v_observed": dm["v_observed"],
            "v_rule": dm["v_rule"],
            "ate": dm["ate"],
        }
        results.append(result)
        print(f"    h={hours:2d}h (bins 0..{h-1 if h>0 else 'none'}): ATE={dm['ate']:+.4f}")

    # ── 4. By HF subtype ──
    print("[4] Cumulative ATE by HF subtype...")
    pheno_test = pheno_df[pheno_df["stay_id"].isin(set(test_ids))].copy()

    subgroup_state_df = pd.read_parquet(ROOT / config["data"]["state_tensors"])
    subgroup_outcomes_df = pd.read_parquet(ROOT / config["data"]["outcomes"])
    subgroup_prop_df = pd.read_parquet(ROOT / config["data"]["propensities_coarse3"])

    for strat in ["hf_subtype", "rule_phenotype_2class"]:
        if strat not in pheno_test.columns:
            continue
        for group_name, group_df in pheno_test.groupby(strat):
            group_ids = group_df["stay_id"].values
            if len(group_ids) < 50:
                continue

            sub_ds = DTRDataset(
                state_df=subgroup_state_df,
                outcomes_df=subgroup_outcomes_df,
                propensity_df=subgroup_prop_df,
                stay_ids=group_ids, temporal_cols=temporal_cols,
                static_cols=static_cols, merge_map=merge_map,
                n_bins=n_bins, action_vocab=test_ds.action_vocab,
            )

            for h in tqdm(range(n_bins + 1), desc=f"  {strat}={group_name}", leave=False):
                dm = evaluate_dm_truncated_horizon(
                    model, sub_ds, device, horizon_bins=h, batch_size=eval_batch_size
                )
                results.append({
                    "horizon_bins": h,
                    "horizon_hours": h * bin_hours,
                    "stratifier": strat,
                    "group": str(group_name),
                    "n_stays": len(sub_ds),
                    "v_observed": dm["v_observed"],
                    "v_rule": dm["v_rule"],
                    "ate": dm["ate"],
                })

            del sub_ds
            gc.collect()

    del subgroup_state_df, subgroup_outcomes_df, subgroup_prop_df
    gc.collect()

    # ── 5. Save table ──
    print("[5] Saving outputs...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(TABLES_DIR / "P3_time_varying_ate.csv", index=False)
    print(f"  Saved: {TABLES_DIR / 'P3_time_varying_ate.csv'}")

    # ── 6. Figures ──
    print("[6] Creating figures...")

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

    # Marginal benefit annotation
    if len(ates) > 1:
        marginal = np.diff(ates)
        max_marginal_idx = np.argmax(np.abs(marginal))
        ax.annotate(
            f"Max marginal\n{hours[max_marginal_idx+1]}h: Δ={marginal[max_marginal_idx]:+.2f}pp",
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
        (results_df.get("stratifier", "") == "hf_subtype") |
        (results_df["group"] == "Overall")
    ].copy()
    if "stratifier" not in hf_data.columns:
        hf_data["stratifier"] = ""
    hf_data = results_df[
        results_df["group"].isin(["Overall", "hfpef", "hfref", "no_hf",
                                   "shock_predominant", "non_shock"])
    ].copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    _disp = {"Overall": "Overall", "hfpef": "HFpEF", "hfref": "HFrEF",
             "no_hf": "No HF", "shock_predominant": "Shock-predominant",
             "non_shock": "Non-shock"}
    colors = {"Overall": "black", "hfpef": "#2196F3", "hfref": "#FF9800",
              "no_hf": "#9E9E9E", "shock_predominant": "#F44336",
              "non_shock": "#4CAF50"}

    for group_name, grp in hf_data.groupby("group"):
        grp = grp.sort_values("horizon_hours")
        ax.plot(grp["horizon_hours"], grp["ate"] * 100, "o-",
                label=f"{_disp.get(group_name, group_name)} (n={grp['n_stays'].iloc[0]})",
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

    # ── 7. Summary report ──
    # Find max marginal benefit window
    overall_sorted = overall.sort_values("horizon_hours")
    ates_overall = overall_sorted["ate"].values
    hours_overall = overall_sorted["horizon_hours"].values
    marginal_overall = np.diff(ates_overall)
    max_mg_idx = np.argmax(np.abs(marginal_overall))
    ate_24h = ates_overall[-1] if len(ates_overall) > 0 else 0

    report_lines = [
        "=" * 72,
        "P3 — Time-Varying Cumulative Treatment Effect Curve",
        "=" * 72,
        f"Generated: {ts}",
        f"Protocol: v3.1-patch, Step P3",
        f"Database: MIMIC-IV",
        f"Model: 80c G-Transformer (rollout DM)",
        f"Test set: {len(test_ds)} stays",
        "",
        "METHOD",
        "─" * 6,
        "  For each horizon h (0-24h in 2h bins):",
        "    - Apply rule policy for bins 0..h",
        "    - Follow observed policy for bins h+1..11",
        "    - Evaluate counterfactual outcome via rollout DM",
        "  ATE(h) = E[Y(hybrid_h)] - E[Y(observed)]",
        "",
        "OVERALL CUMULATIVE ATE CURVE",
        "─" * 28,
    ]
    for _, row in overall_sorted.iterrows():
        report_lines.append(
            f"  h={row['horizon_hours']:2.0f}h: ATE={row['ate']*100:+.2f}pp"
        )
    report_lines.extend([
        "",
        f"  Full horizon (24h) ATE: {ate_24h*100:+.2f}pp",
        f"  Max marginal benefit window: {hours_overall[max_mg_idx]}h → {hours_overall[max_mg_idx+1]}h "
        f"(Δ={marginal_overall[max_mg_idx]*100:+.2f}pp)",
        "",
        "SUBGROUP CURVES",
        "─" * 15,
    ])
    subgroups_24h = results_df[results_df["horizon_bins"] == n_bins]
    for _, row in subgroups_24h.iterrows():
        if row["group"] != "Overall":
            report_lines.append(
                f"  {row['group']:25s}: ATE(24h)={row['ate']*100:+.2f}pp (n={row['n_stays']})"
            )
    report_lines.extend([
        "",
        "MANUSCRIPT PLACEMENT",
        "─" * 20,
        "  Results Section or Discussion figure.",
        "  Answers: 'When does the treatment benefit occur?'",
        "",
    ])

    with open(REPORTS_DIR / "P3_time_varying_ate_summary.txt", "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Saved: {REPORTS_DIR / 'P3_time_varying_ate_summary.txt'}")

    total_time = time.time() - t0
    print()
    print("=" * 72)
    print(f"P3 COMPLETE. Total time: {total_time / 60:.1f} min")
    print("=" * 72)


if __name__ == "__main__":
    main()
