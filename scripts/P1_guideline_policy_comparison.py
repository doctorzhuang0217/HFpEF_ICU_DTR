#!/usr/bin/env python3
"""P1 — Guideline-Concordant Policy Comparison

Protocol: v3.1-patch, Step P1
Compute tier: C3 (direct execution)
Depends on: 80c G-Transformer trained model, 75a OPE pipeline, 70c rule policy

Objective: Compare the study's rule-based policy against policies derived from
published clinical guidelines (SSC-2021, ESC-2023) to demonstrate clinical
relevance and contextualize the treatment effect.

Guideline policies:
  1. SSC-2021 (Surviving Sepsis Campaign 2021)
     - MAP target >= 65 mmHg, aggressive initial fluid resuscitation,
       norepinephrine first-line for refractory hypotension
  2. ESC-2023 (ESC Heart Failure Guidelines 2023)
     - Conservative fluid strategy, loop diuretic emphasis for congestion,
       avoidance of fluid overload in HFpEF
  3. Study rule policy (existing 70c baseline)

Analysis:
  - Encode SSC-2021 and ESC-2023 as deterministic policies in the same action space
  - Run OPE (DM via 80c G-Transformer) for each guideline policy
  - Compare V(pi) across all three policies, overall and by HF subtype
  - Report ATE of each guideline policy vs observed, and pairwise comparisons

Outputs:
  - tables/P1_guideline_policy_comparison.csv
  - figures/P1_guideline_policy_forest_plot.png
  - reports/P1_guideline_policy_comparison_summary.txt
"""

from __future__ import annotations

import datetime
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

from hfpef_dtr.models.dtr_dataset import DTRDataset, build_dtr_datasets, load_dtr_data
from hfpef_dtr.models.dataset import discover_feature_columns, make_train_val_test_split
from hfpef_dtr.models.gtransformer import GTransformer, build_gtransformer_from_config
from hfpef_dtr.models.ope import evaluate_dm, bootstrap_ope

# ── Paths ──
CONFIG_PATH = ROOT / "configs" / "80_step_c_training_v1.yaml"
NORM_PATH = ROOT / "configs" / "45c_normalization_params_v1.yaml"
MODEL_CKPT = ROOT / "models" / "80c_gtransformer_best.pt"
TABLES_DIR = ROOT / "tables"
FIGURES_DIR = ROOT / "figures"
REPORTS_DIR = ROOT / "reports"

for d in [TABLES_DIR, FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# Guideline Policy Definitions
# ═══════════════════════════════════════════════════════════════════════
#
# Action encoding: fluid_level * 6 + vaso_level * 2 + diuretic_level
#   fluid:    0=none, 1=low_moderate, 2=high
#   vaso:     0=none, 1=low, 2=high
#   diuretic: 0=none, 1=any

def _encode_action(fluid: int, vaso: int, diuretic: int) -> int:
    return fluid * 6 + vaso * 2 + diuretic


# Pre-compute named action constants
A_AGGRESSIVE_RESUS = _encode_action(2, 2, 0)   # 16: high fluid + high vaso
A_MODERATE_RESUS   = _encode_action(2, 1, 0)   # 14: high fluid + low vaso
A_FLUID_ONLY       = _encode_action(2, 0, 0)   # 12: high fluid, no vaso
A_PERF_SUPPORT     = _encode_action(1, 1, 0)   # 8:  low_mod fluid + low vaso
A_MAINTENANCE      = _encode_action(1, 0, 0)   # 6:  low_mod fluid, no vaso
A_VASO_ONLY        = _encode_action(0, 1, 0)   # 2:  no fluid + low vaso
A_VASO_HIGH        = _encode_action(0, 2, 0)   # 4:  no fluid + high vaso
A_DECONGEST        = _encode_action(0, 0, 1)   # 1:  no fluid + diuretic
A_NOTHING          = _encode_action(0, 0, 0)   # 0:  nothing
A_MAINT_DIURETIC   = _encode_action(1, 0, 1)   # 7:  low_mod fluid + diuretic


def _load_norm_params() -> dict:
    """Load frozen normalization parameters from Step 45c."""
    with open(NORM_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("params", {})


def _denorm(z_vals: np.ndarray, param: dict) -> np.ndarray:
    """Convert z-scored values back to raw units."""
    return z_vals * float(param["std"]) + float(param["mean"])


def apply_ssc2021_policy(state_df: pd.DataFrame, norm_params: dict) -> np.ndarray:
    """
    SSC-2021 (Surviving Sepsis Campaign 2021) policy.

    Core principles:
    - Aggressive initial crystalloid resuscitation (30 mL/kg) for sepsis-induced
      hypoperfusion
    - MAP target >= 65 mmHg; norepinephrine first-line vasopressor
    - Vasopressors if hypotension persists after fluid challenge
    - Continue fluid if lactate remains elevated and fluid-responsive
    - Less attention to fluid overload risk (sepsis-centric)

    Translation to our action space:
    - Shock (MAP<65 + lactate>2): aggressive resuscitation (high fluid + high vaso)
    - Hypotension only (MAP<65): high fluid + low vaso (fluid first, then vasopressor)
    - Hypoperfusion only (lactate>2): moderate fluid resuscitation
    - Stable: maintenance fluids (SSC does not emphasize diuresis)
    """
    n = len(state_df)
    actions = np.full(n, A_MAINTENANCE, dtype=int)

    # Reconstruct raw values from z-scores
    if "map_z" in state_df.columns and "map" in norm_params:
        map_vals = _denorm(state_df["map_z"].values, norm_params["map"])
    elif "map" in state_df.columns:
        map_vals = state_df["map"].values
    else:
        map_vals = np.full(n, 80.0)

    if "lactate_z" in state_df.columns and "lactate" in norm_params:
        lac_vals = _denorm(state_df["lactate_z"].values, norm_params["lactate"])
    elif "lactate" in state_df.columns:
        lac_vals = state_df["lactate"].values
    else:
        lac_vals = np.full(n, 1.5)

    # SSC-2021 rules (priority order)
    # R1: Shock + hypoperfusion → aggressive resuscitation
    mask_shock = (map_vals < 65) & (lac_vals > 2.0)
    actions[mask_shock] = A_AGGRESSIVE_RESUS

    # R2: Hypotension only → high fluid + low vasopressor (fluid-first approach)
    mask_hypo = (map_vals < 65) & (lac_vals <= 2.0)
    actions[mask_hypo] = A_MODERATE_RESUS

    # R3: Hypoperfusion without shock → moderate fluid resuscitation
    mask_perf = (lac_vals > 2.0) & (map_vals >= 65)
    actions[mask_perf] = A_FLUID_ONLY

    # R4: Stable → maintenance (SSC does not trigger diuretics proactively)
    # Already set as default

    return actions


def apply_esc2023_policy(state_df: pd.DataFrame, norm_params: dict) -> np.ndarray:
    """
    ESC-2023 (ESC Heart Failure Guidelines 2023) policy.

    Core principles:
    - Conservative fluid strategy for HF patients
    - Prioritize loop diuretics for volume overload/congestion
    - Avoid fluid overload — particularly relevant in HFpEF
    - Vasopressors only for true cardiogenic shock; prefer inotropes
    - Lower threshold for diuresis; higher threshold for fluid administration

    Translation to our action space:
    - Shock (MAP<65 + lactate>2): vasopressor + conservative fluid (not aggressive)
    - Hypotension only (MAP<65): vasopressor alone (avoid fluid loading)
    - Volume overload (fluid_balance>2L): diuretics aggressively
    - Mild congestion (fluid_balance>0L + SpO2<94): diuretics
    - Stable: minimal fluids or nothing (conservative)
    """
    n = len(state_df)
    actions = np.full(n, A_NOTHING, dtype=int)  # ESC defaults conservative

    # Reconstruct raw values
    if "map_z" in state_df.columns and "map" in norm_params:
        map_vals = _denorm(state_df["map_z"].values, norm_params["map"])
    elif "map" in state_df.columns:
        map_vals = state_df["map"].values
    else:
        map_vals = np.full(n, 80.0)

    if "lactate_z" in state_df.columns and "lactate" in norm_params:
        lac_vals = _denorm(state_df["lactate_z"].values, norm_params["lactate"])
    elif "lactate" in state_df.columns:
        lac_vals = state_df["lactate"].values
    else:
        lac_vals = np.full(n, 1.5)

    # Fluid balance
    if "fluid_balance_L_z" in state_df.columns and "fluid_balance_L" in norm_params:
        fb_vals = _denorm(state_df["fluid_balance_L_z"].values, norm_params["fluid_balance_L"])
    else:
        fb_col = None
        for candidate in ["fluid_balance_cumulative_L", "fluid_balance_L", "cum_fluid_balance_L"]:
            if candidate in state_df.columns:
                fb_col = candidate
                break
        fb_vals = state_df[fb_col].values if fb_col else np.zeros(n)

    if "spo2_z" in state_df.columns and "spo2" in norm_params:
        spo2_vals = _denorm(state_df["spo2_z"].values, norm_params["spo2"])
    elif "spo2" in state_df.columns:
        spo2_vals = state_df["spo2"].values
    else:
        spo2_vals = np.full(n, 97.0)

    # ESC-2023 rules (priority order — HF-aware)
    # R1: Cardiogenic shock → vasopressor + conservative fluid
    mask_shock = (map_vals < 65) & (lac_vals > 2.0)
    actions[mask_shock] = A_PERF_SUPPORT  # low_mod fluid + low vaso (not aggressive)

    # R2: Hypotension without hypoperfusion → vasopressor only (avoid fluid)
    mask_hypo = (map_vals < 65) & (lac_vals <= 2.0)
    actions[mask_hypo] = A_VASO_ONLY

    # R3: Volume overload → diuretics (ESC threshold lower than SSC: 2L vs 3-5L)
    mask_overload = (fb_vals > 2.0) & (map_vals >= 65)
    actions[mask_overload] = A_DECONGEST

    # R4: Mild congestion with respiratory compromise → diuretics
    mask_mild_congest = (fb_vals > 0.0) & (spo2_vals < 94) & (map_vals >= 65) & (fb_vals <= 2.0)
    actions[mask_mild_congest] = A_DECONGEST

    # R5: Stable, no overload → nothing or minimal (conservative)
    # Already default = A_NOTHING

    return actions


# ═══════════════════════════════════════════════════════════════════════
# OPE evaluation with custom policy actions
# ═══════════════════════════════════════════════════════════════════════


def evaluate_dm_custom_policy(
    model,
    dataset: DTRDataset,
    custom_actions: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
    use_rollout: bool = True,
) -> dict:
    """
    DM evaluation for a custom policy (not the pre-stored rule_action).

    custom_actions: (n_stays, n_bins) array of raw action indices, which will
    be remapped to contiguous via dataset._action_to_contiguous.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    obs_probs = []
    cf_probs = []
    batch_start = 0

    for batch in tqdm(loader, desc="    DM custom", leave=False, mininterval=2):
        temporal = batch["temporal"].to(device)
        static = batch["static"].to(device)
        action = batch["action"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        absorbing_mask = batch["absorbing_mask"].to(device)

        bs = temporal.size(0)

        # Remap custom actions for this batch
        raw_slice = custom_actions[batch_start: batch_start + bs]
        custom_contig = np.zeros_like(raw_slice)
        for i in range(bs):
            for t in range(raw_slice.shape[1]):
                custom_contig[i, t] = dataset._action_to_contiguous.get(
                    int(raw_slice[i, t]), 0
                )
        custom_tensor = torch.from_numpy(custom_contig).long().to(device)

        with torch.no_grad():
            # Observed policy outcome
            if use_rollout and hasattr(model, "predict_counterfactual_rollout"):
                out_obs = model.predict_counterfactual_rollout(
                    temporal, static, action, valid_mask, absorbing_mask
                )
            else:
                out_obs = model.predict_counterfactual(
                    temporal, static, action, valid_mask, absorbing_mask
                )
            obs_probs.append(out_obs["outcome_prob"].cpu().numpy())

            # Custom policy outcome
            if use_rollout and hasattr(model, "predict_counterfactual_rollout"):
                out_cf = model.predict_counterfactual_rollout(
                    temporal, static, custom_tensor, valid_mask, absorbing_mask
                )
            else:
                out_cf = model.predict_counterfactual(
                    temporal, static, custom_tensor, valid_mask, absorbing_mask
                )
            cf_probs.append(out_cf["outcome_prob"].cpu().numpy())

        batch_start += bs

    obs_probs = np.concatenate(obs_probs).squeeze()
    cf_probs = np.concatenate(cf_probs).squeeze()

    return {
        "v_observed": float(obs_probs.mean()),
        "v_policy": float(cf_probs.mean()),
        "ate": float(cf_probs.mean() - obs_probs.mean()),
        "obs_probs": obs_probs,
        "cf_probs": cf_probs,
    }


def bootstrap_dm_custom(
    model,
    dataset: DTRDataset,
    custom_actions: np.ndarray,
    device: torch.device,
    n_resamples: int = 500,
    seed: int = 7777,
    use_rollout: bool = True,
) -> dict:
    """Bootstrap CIs for custom policy DM evaluation."""
    rng = np.random.RandomState(seed)
    n = len(dataset)

    boot_ates = []
    boot_v_policies = []

    pbar = tqdm(range(n_resamples), desc="    bootstrap", leave=False, mininterval=2)
    for _ in pbar:
        idx = rng.choice(n, size=n, replace=True)
        from hfpef_dtr.models.ope import _BootstrapSubset
        boot_ds = _BootstrapSubset(dataset, idx)
        boot_actions = custom_actions[idx]

        try:
            result = evaluate_dm_custom_policy(
                model, boot_ds, boot_actions, device,
                use_rollout=use_rollout,
            )
            boot_ates.append(result["ate"])
            boot_v_policies.append(result["v_policy"])
        except Exception:
            continue

        if boot_ates:
            pbar.set_postfix(ate=f"{np.mean(boot_ates):+.4f}", ok=len(boot_ates))

    if not boot_ates:
        return {"n_success": 0}

    boot_ates = np.array(boot_ates)
    return {
        "n_success": len(boot_ates),
        "ate_mean": float(boot_ates.mean()),
        "ate_ci_lo": float(np.percentile(boot_ates, 2.5)),
        "ate_ci_hi": float(np.percentile(boot_ates, 97.5)),
        "v_policy_mean": float(np.mean(boot_v_policies)),
    }


# ═══════════════════════════════════════════════════════════════════════
# Build per-stay action arrays from state data
# ═══════════════════════════════════════════════════════════════════════


def build_policy_actions(
    policy_fn,
    state_df: pd.DataFrame,
    stay_ids: np.ndarray,
    norm_params: dict,
    n_bins: int = 12,
) -> np.ndarray:
    """
    Build (n_stays, n_bins) action array by applying a policy function to state data.
    """
    stay_order = list(stay_ids)
    sid_to_idx = {sid: i for i, sid in enumerate(stay_order)}
    n_stays = len(stay_order)
    actions = np.zeros((n_stays, n_bins), dtype=int)

    df = state_df[state_df["stay_id"].isin(set(stay_ids))].copy()
    policy_actions = policy_fn(df, norm_params)

    for row_i, (_, row) in enumerate(df.iterrows()):
        sid = row["stay_id"]
        if sid not in sid_to_idx:
            continue
        b = int(row["bin"])
        if 0 <= b < n_bins:
            actions[sid_to_idx[sid], b] = int(policy_actions.iloc[row_i]
                                               if hasattr(policy_actions, "iloc")
                                               else policy_actions[row_i])

    return actions


# ═══════════════════════════════════════════════════════════════════════
# Subgroup analysis
# ═══════════════════════════════════════════════════════════════════════


def run_subgroup_analysis(
    model,
    dataset: DTRDataset,
    custom_actions: np.ndarray,
    pheno_df: pd.DataFrame,
    test_ids: np.ndarray,
    device: torch.device,
    policy_name: str,
    state_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    propensity_df: pd.DataFrame,
    temporal_cols: list[str],
    static_cols: list[str],
    merge_map: dict | None,
    config: dict,
    norm_params: dict,
    policy_fn,
    use_rollout: bool = True,
) -> list[dict]:
    """Run DM evaluation for each HF subtype using custom policy actions."""
    pheno_test = pheno_df[pheno_df["stay_id"].isin(set(test_ids))].copy()
    results = []

    for strat in ["hf_subtype"]:
        if strat not in pheno_test.columns:
            continue
        for group_name, group_df in pheno_test.groupby(strat):
            group_ids = group_df["stay_id"].values
            if len(group_ids) < 30:
                continue

            sub_ds = DTRDataset(
                state_df=state_df,
                outcomes_df=outcomes_df,
                propensity_df=propensity_df,
                stay_ids=group_ids,
                temporal_cols=temporal_cols,
                static_cols=static_cols,
                merge_map=merge_map,
                n_bins=config["grid"]["n_bins"],
                action_vocab=dataset.action_vocab,
            )

            # Build custom actions for this subgroup
            sub_actions = build_policy_actions(
                policy_fn, state_df, sub_ds.stay_ids, norm_params,
                n_bins=config["grid"]["n_bins"],
            )

            dm = evaluate_dm_custom_policy(
                model, sub_ds, sub_actions, device, use_rollout=use_rollout,
            )
            results.append({
                "policy": policy_name,
                "stratifier": strat,
                "group": str(group_name),
                "n_stays": len(sub_ds),
                "v_observed": dm["v_observed"],
                "v_policy": dm["v_policy"],
                "ate": dm["ate"],
            })

    return results


# ═══════════════════════════════════════════════════════════════════════
# Forest plot
# ═══════════════════════════════════════════════════════════════════════


def create_forest_plot(results_df: pd.DataFrame, output_path: Path) -> None:
    """Create forest plot comparing policy values across guideline strategies."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Publication-ready label mappings
    _policy_display = {
        "Study Rule (70c)": "Study Rule",
    }
    _group_display = {
        "hfpef": "HFpEF", "no_hf": "No HF", "hfref": "HFrEF",
        "hfmref": "HFmrEF", "hf_ef_unknown": "HF (EF unknown)",
    }

    # Collect unique policy-group combinations
    rows = []
    for _, r in results_df.iterrows():
        policy = _policy_display.get(r["policy"], r["policy"])
        group = _group_display.get(str(r.get("group", "Overall")),
                                    str(r.get("group", "Overall")))
        label = f"{policy} — {group}"
        ate = r["ate"]
        ci_lo = r.get("ate_ci_lo", ate)
        ci_hi = r.get("ate_ci_hi", ate)
        rows.append((label, ate, ci_lo, ci_hi))

    rows = sorted(rows, key=lambda x: x[1])  # sort by ATE
    labels = [r[0] for r in rows]
    ates = [r[1] for r in rows]
    ci_lo = [r[2] for r in rows]
    ci_hi = [r[3] for r in rows]

    y_pos = np.arange(len(rows))

    # Plot
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
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(str(output_path).replace('.png', '.pdf'), format='pdf',
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"  Saved: {output_path} (+pdf)")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 72)
    print("P1 — Guideline-Concordant Policy Comparison")
    print("=" * 72)
    t0 = time.time()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 1. Load config, model, data ──
    print("[1] Loading config and data...")
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load normalization params for policy threshold conversion
    norm_params = _load_norm_params()

    # Load data
    state_df, outcomes_df, propensity_df, temporal_cols, static_cols, merge_map = (
        load_dtr_data(ROOT, config)
    )

    # Build test dataset (same split as training)
    _, _, test_ds = build_dtr_datasets(ROOT, config)
    all_stay_ids = state_df["stay_id"].unique()
    split_cfg = config["split"]
    _, _, test_ids = make_train_val_test_split(
        all_stay_ids, outcomes_df,
        train_frac=split_cfg["train_frac"],
        val_frac=split_cfg["val_frac"],
        test_frac=split_cfg["test_frac"],
        stratify_col=split_cfg["stratify_col"],
        seed=split_cfg["seed"],
    )

    # Load phenotype labels for subgroup analysis
    pheno_path = ROOT / config["data"]["phenotype_labels"]
    pheno_df = pd.read_parquet(pheno_path)

    # ── 2. Load trained G-Transformer ──
    print("[2] Loading 80c G-Transformer checkpoint...")
    ckpt = torch.load(MODEL_CKPT, map_location=device, weights_only=False)
    model = build_gtransformer_from_config(
        d_temporal=test_ds.d_temporal,
        d_static=test_ds.d_static,
        n_actions=test_ds.n_actions,
        model_cfg=config["models"]["80c_gtransformer"],
        grid_cfg=config["grid"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")

    # ── 3. Build policy action arrays ──
    print("[3] Building guideline policy action arrays...")
    n_bins = config["grid"]["n_bins"]
    test_ids_sorted = test_ds.stay_ids

    policies = {
        "Study Rule (70c)": (None, None),  # uses pre-stored rule_action
        "SSC-2021": (apply_ssc2021_policy, None),
        "ESC-2023": (apply_esc2023_policy, None),
    }

    # Build action arrays for guideline policies
    for name, (policy_fn, _) in policies.items():
        if policy_fn is not None:
            actions = build_policy_actions(
                policy_fn, state_df, test_ids_sorted, norm_params, n_bins,
            )
            policies[name] = (policy_fn, actions)
            # Policy action distribution
            flat = actions.flatten()
            unique, counts = np.unique(flat, return_counts=True)
            print(f"  {name}: {len(flat)} bins, {len(unique)} unique actions")
            for a, c in sorted(zip(unique, counts), key=lambda x: -x[1])[:5]:
                print(f"    action {a}: {c} ({c/len(flat)*100:.1f}%)")

    # ── 4. OPE evaluation for each policy ──
    print("[4] Running OPE (DM) for each policy...")
    all_results = []

    for name, (policy_fn, actions) in tqdm(policies.items(), desc="Policies"):
        print(f"\n  === {name} ===")

        if name == "Study Rule (70c)":
            # Use the built-in rule_action from dataset
            dm = evaluate_dm(model, test_ds, device, use_rollout=True)
            result = {
                "policy": name,
                "group": "Overall",
                "n_stays": len(test_ds),
                "v_observed": dm["v_observed"],
                "v_policy": dm["v_rule"],
                "ate": dm["ate"],
            }

            # Bootstrap CI
            def rule_eval_fn(ds, **kwargs):
                return evaluate_dm(model, ds, device, use_rollout=True)
            boot = bootstrap_ope(rule_eval_fn, test_ds, n_resamples=500, seed=7777)
            if boot.get("n_success", 0) > 0:
                result["ate_ci_lo"] = boot["ate_ci_lo"]
                result["ate_ci_hi"] = boot["ate_ci_hi"]
                result["n_bootstrap"] = boot["n_success"]
        else:
            dm = evaluate_dm_custom_policy(
                model, test_ds, actions, device, use_rollout=True,
            )
            result = {
                "policy": name,
                "group": "Overall",
                "n_stays": len(test_ds),
                "v_observed": dm["v_observed"],
                "v_policy": dm["v_policy"],
                "ate": dm["ate"],
            }

            # Bootstrap CI
            boot = bootstrap_dm_custom(
                model, test_ds, actions, device,
                n_resamples=500, seed=7777, use_rollout=True,
            )
            if boot.get("n_success", 0) > 0:
                result["ate_ci_lo"] = boot["ate_ci_lo"]
                result["ate_ci_hi"] = boot["ate_ci_hi"]
                result["n_bootstrap"] = boot["n_success"]

        print(f"    V(obs)={result['v_observed']:.4f}, V(policy)={result['v_policy']:.4f}, ATE={result['ate']:+.4f}")
        if "ate_ci_lo" in result:
            print(f"    CI: [{result['ate_ci_lo']:.4f}, {result['ate_ci_hi']:.4f}]")
        all_results.append(result)

    # ── 5. Subgroup analysis by HF subtype ──
    print("\n[5] Subgroup analysis by HF subtype...")
    for name, (policy_fn, actions) in policies.items():
        if name == "Study Rule (70c)":
            # Rule policy subgroup — use evaluate_dm directly
            pheno_test = pheno_df[pheno_df["stay_id"].isin(set(test_ids))].copy()
            for strat in ["hf_subtype"]:
                if strat not in pheno_test.columns:
                    continue
                for group_name, group_df in pheno_test.groupby(strat):
                    group_ids = group_df["stay_id"].values
                    if len(group_ids) < 30:
                        continue
                    sub_ds = DTRDataset(
                        state_df=state_df, outcomes_df=outcomes_df,
                        propensity_df=propensity_df, stay_ids=group_ids,
                        temporal_cols=temporal_cols, static_cols=static_cols,
                        merge_map=merge_map, n_bins=n_bins,
                        action_vocab=test_ds.action_vocab,
                    )
                    dm = evaluate_dm(model, sub_ds, device, use_rollout=True)
                    all_results.append({
                        "policy": name, "group": str(group_name),
                        "stratifier": strat, "n_stays": len(sub_ds),
                        "v_observed": dm["v_observed"], "v_policy": dm["v_rule"],
                        "ate": dm["ate"],
                    })
        else:
            sg_results = run_subgroup_analysis(
                model, test_ds, actions, pheno_df, test_ids, device, name,
                state_df, outcomes_df, propensity_df, temporal_cols, static_cols,
                merge_map, config, norm_params, policy_fn, use_rollout=True,
            )
            all_results.extend(sg_results)

    for r in all_results:
        if r.get("group", "Overall") != "Overall":
            print(f"  {r['policy']} × {r.get('group','')}: n={r['n_stays']}, ATE={r['ate']:+.4f}")

    # ── 6. Pairwise policy comparisons ──
    print("\n[6] Pairwise comparisons...")
    overall = {r["policy"]: r for r in all_results if r.get("group") == "Overall"}
    pairwise = []
    policy_names = list(overall.keys())
    for i in range(len(policy_names)):
        for j in range(i + 1, len(policy_names)):
            p1, p2 = policy_names[i], policy_names[j]
            diff = overall[p2]["v_policy"] - overall[p1]["v_policy"]
            pairwise.append({
                "policy_a": p1,
                "policy_b": p2,
                "v_a": overall[p1]["v_policy"],
                "v_b": overall[p2]["v_policy"],
                "delta_v": diff,
            })
            print(f"  {p2} vs {p1}: ΔV = {diff:+.4f}")

    # ── 7. Save outputs ──
    print("\n[7] Saving outputs...")

    # Main results table
    results_df = pd.DataFrame(all_results)
    results_path = TABLES_DIR / "P1_guideline_policy_comparison.csv"
    results_df.to_csv(results_path, index=False)
    print(f"  Saved: {results_path}")

    # Pairwise table
    if pairwise:
        pw_df = pd.DataFrame(pairwise)
        pw_path = TABLES_DIR / "P1_pairwise_comparison.csv"
        pw_df.to_csv(pw_path, index=False)
        print(f"  Saved: {pw_path}")

    # Forest plot
    create_forest_plot(results_df, FIGURES_DIR / "P1_guideline_policy_forest_plot.png")

    # Summary report
    report_lines = [
        "=" * 72,
        "P1 — Guideline-Concordant Policy Comparison",
        "=" * 72,
        f"Generated: {ts}",
        f"Protocol: v3.1-patch, Step P1",
        f"Database: MIMIC-IV",
        f"Model: 80c G-Transformer (rollout DM)",
        f"Test set: {len(test_ds)} stays",
        "",
        "GUIDELINE POLICIES",
        "─" * 18,
        "  SSC-2021: Surviving Sepsis Campaign 2021",
        "    - Aggressive fluid resuscitation for sepsis-induced hypoperfusion",
        "    - MAP >= 65, norepinephrine first-line, fluid-first approach",
        "  ESC-2023: ESC Heart Failure Guidelines 2023",
        "    - Conservative fluid, loop diuretics for congestion",
        "    - Vasopressor over fluid for hypotension, lower diuresis threshold",
        "  Study Rule (70c): Hemodynamic-guided rule-based policy",
        "    - Balanced approach: resuscitate shock, decongest overload",
        "",
        "OVERALL RESULTS",
        "─" * 15,
    ]
    for r in all_results:
        if r.get("group") == "Overall":
            ci_str = ""
            if "ate_ci_lo" in r:
                ci_str = f" [{r['ate_ci_lo']:.4f}, {r['ate_ci_hi']:.4f}]"
            report_lines.append(
                f"  {r['policy']:25s}: V(obs)={r['v_observed']:.4f}, "
                f"V(π)={r['v_policy']:.4f}, ATE={r['ate']:+.4f}{ci_str}"
            )
    report_lines.extend(["", "SUBGROUP RESULTS (by HF subtype)", "─" * 32])
    for r in all_results:
        if r.get("group") != "Overall":
            report_lines.append(
                f"  {r['policy']:20s} × {r.get('group',''):10s}: "
                f"n={r['n_stays']:>5}, ATE={r['ate']:+.4f}"
            )
    report_lines.extend(["", "PAIRWISE COMPARISONS", "─" * 20])
    for pw in pairwise:
        report_lines.append(
            f"  {pw['policy_b']} vs {pw['policy_a']}: "
            f"ΔV = {pw['delta_v']:+.4f}"
        )
    report_lines.extend([
        "",
        "INTERPRETATION",
        "─" * 14,
        "  SSC-2021 reflects sepsis-centric aggressive resuscitation.",
        "  ESC-2023 reflects HF-aware conservative fluid management.",
        "  Study rule policy balances both perspectives.",
        "  ATE < 0 indicates lower predicted mortality vs observed care.",
        "",
        "MANUSCRIPT PLACEMENT",
        "─" * 20,
        "  Results Section 3 or Discussion; forest plot figure.",
        "",
    ])

    report_path = REPORTS_DIR / "P1_guideline_policy_comparison_summary.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Saved: {report_path}")

    total_time = time.time() - t0
    print()
    print("=" * 72)
    print(f"P1 COMPLETE. Total time: {total_time / 60:.1f} min")
    print("=" * 72)


if __name__ == "__main__":
    main()
