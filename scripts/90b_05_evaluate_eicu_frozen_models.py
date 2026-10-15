#!/usr/bin/env python3
"""90b_05 — Evaluate Frozen MIMIC Models on eICU Data

Protocol: v3.1, Step 90b (sub-step 5)
Compute: G1 (GPU for model inference) + C3 (bootstrap)

Purpose
-------
Load frozen MIMIC-trained model checkpoints and evaluate them on eICU
external validation data. This is the core transportability test.

Evaluation:
  1. Load frozen checkpoints (80c G-Transformer, 80a G-Net, 80b ACTIN)
  2. Build DTRDataset from eICU state tensors + actions + propensities
  3. Run OPE: DM (non-rollout primary, rollout diagnostic), DR, FQE
  4. Subgroup analysis: HFpEF vs HFrEF vs no_HF
  5. Bootstrap CIs (B=500)
  6. Compare with MIMIC reference results
  7. Apply 29e consistency criteria

All eICU data is used as a single test set (no train/val split).

Inputs
------
- data/interim/90b_eicu_state_tensors_v1.parquet
- data/interim/90b_eicu_outcomes_absorb.parquet
- data/interim/90b_eicu_propensity_coarse3.parquet
- models/80c_gtransformer_best.pt (or rollout_ow1 variant)
- models/80a_gnet_best.pt
- models/80b_actin_best.pt
- configs/80_step_c_training_v1.yaml
- configs/90a_primary_model_registry.json

Outputs
-------
- tables/90b_eicu_*_policy_values.csv
- tables/90b_eicu_*_bootstrap_ci.csv
- tables/90b_eicu_*_subgroup_results.csv
- tables/90b_eicu_transportability_comparison.csv
- reports/90b_05_eicu_external_validation_report.txt
- figures/90b_eicu_forest_plot.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(ROOT / "src"))

from hfpef_dtr.models.dtr_dataset import DTRDataset
from hfpef_dtr.models.gtransformer import GTransformer, build_gtransformer_from_config
from hfpef_dtr.models.gnet import GNet, build_gnet_from_config
from hfpef_dtr.models.actin import ACTIN, build_actin_from_config
from hfpef_dtr.models.ope import (
    evaluate_dm, evaluate_dr, evaluate_fqe,
    train_fqe, bootstrap_ope, bootstrap_fqe_eval_only,
)
from hfpef_dtr.models.step_c_reporting import (
    save_policy_values, save_bootstrap_cis,
    save_subgroup_results, generate_summary_report,
)

MODELS_DIR = ROOT / "models"
TABLES_DIR = ROOT / "tables"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = ROOT / "figures"
INTERIM_DIR = ROOT / "data" / "interim"
STAGING_DIR = ROOT / "data" / "staging"

for d in [TABLES_DIR, REPORTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="90b_05: eICU external validation")
    p.add_argument("--models", nargs="+",
                   default=["80c_gtransformer"],
                   help="Model IDs to evaluate (default: 80c_gtransformer)")
    p.add_argument("--use-rollout", action="store_true",
                   help="Use autoregressive rollout for DM (80c only)")
    p.add_argument("--tag", type=str, default="rollout_ow1",
                   help="Checkpoint tag (e.g., 'rollout_ow1' for best 80c variant)")
    p.add_argument("--bootstrap", type=int, default=500,
                   help="Number of bootstrap resamples")
    p.add_argument("--skip-bootstrap", action="store_true",
                   help="Skip bootstrap CIs (faster for debugging)")
    p.add_argument("--tier-a-threshold", type=float, default=80.0,
                   help="Tier A fluid_nz_pct threshold (default: 80%%)")
    p.add_argument("--infusion-drug-only", action="store_true",
                   help="P1-3: restrict full-cohort eval to hospitals with "
                        "infusionDrug records (has_infusiondrug=True in site CSV)")
    return p.parse_args()


def load_mimic_feature_spec() -> dict:
    """Load MIMIC feature specification from the frozen model registry."""
    registry_path = ROOT / "configs" / "90a_primary_model_registry.json"
    with open(registry_path) as f:
        registry = json.load(f)
    features = registry["features"]
    training = registry["training_config"]
    return {
        "temporal_cols": features["temporal_cols"],  # 118 columns
        "static_cols": features["static_cols"],      # 3 columns
        "n_temporal": features["n_temporal"],         # 118
        "n_static": features["n_static"],             # 3
        "n_actions": training["action"]["n_merged_cells"],  # 13
    }


def align_eicu_to_mimic_features(
    state_df: pd.DataFrame,
    mimic_spec: dict,
) -> pd.DataFrame:
    """
    Ensure eICU state tensor has exactly the MIMIC feature columns.
    Add missing columns as 0, warn about unexpected columns.
    """
    mimic_temporal = mimic_spec["temporal_cols"]
    mimic_static = mimic_spec["static_cols"]
    all_mimic_cols = mimic_temporal + mimic_static

    missing_cols = [c for c in all_mimic_cols if c not in state_df.columns]
    if missing_cols:
        print(f"    WARNING: {len(missing_cols)} MIMIC columns missing in eICU, "
              f"filling with 0: {missing_cols[:5]}...")
        for c in missing_cols:
            state_df[c] = 0.0

    return state_df


def load_eicu_dtr_dataset(config: dict, mimic_spec: dict) -> DTRDataset:
    """
    Build a single DTRDataset from all eICU data (no train/val split).
    Uses MIMIC feature lists from registry (NOT auto-discovery).
    """
    print("  Loading eICU data...")
    state_df = pd.read_parquet(INTERIM_DIR / "90b_eicu_state_tensors_v1.parquet")
    outcomes_df = pd.read_parquet(INTERIM_DIR / "90b_eicu_outcomes_absorb.parquet")
    propensity_df = pd.read_parquet(INTERIM_DIR / "90b_eicu_propensity_coarse3.parquet")

    # Use MIMIC feature lists (NOT discover_feature_columns)
    temporal_cols = mimic_spec["temporal_cols"]
    static_cols = mimic_spec["static_cols"]

    # Align eICU columns to MIMIC feature space
    state_df = align_eicu_to_mimic_features(state_df, mimic_spec)

    # Load MIMIC merge map
    merge_path = ROOT / "configs" / "70b_action_merge_map.yaml"
    merge_map = None
    if merge_path.exists():
        with open(merge_path) as f:
            mm = yaml.safe_load(f)
        if "merge_map" in mm:
            merge_map = {int(k): int(v) for k, v in mm["merge_map"].items()}

    all_stay_ids = state_df["stay_id"].unique()
    print(f"  eICU stays: {len(all_stay_ids):,}")
    print(f"  Temporal: {len(temporal_cols)} (MIMIC registry)")
    print(f"  Static: {len(static_cols)} (MIMIC registry)")

    # Use MIMIC training action vocabulary to ensure contiguous remapping
    # matches frozen model embeddings (eICU may lack some action cells)
    mimic_action_vocab = config["action"]["valid_merged_indices"]

    ds = DTRDataset(
        state_df=state_df,
        outcomes_df=outcomes_df,
        propensity_df=propensity_df,
        stay_ids=all_stay_ids,
        temporal_cols=temporal_cols,
        static_cols=static_cols,
        merge_map=merge_map,
        n_bins=config["grid"]["n_bins"],
        action_vocab=mimic_action_vocab,
    )

    print(f"  DTR dataset: {len(ds)} stays, "
          f"{ds.d_temporal} temporal, {ds.d_static} static, "
          f"{ds.n_actions} actions (MIMIC vocab forced)")

    # ── Feature distribution diagnostic ──
    print("\n  Feature distribution diagnostic (eICU temporal tensor):")
    tsl_cols = [c for c in temporal_cols if c.endswith("_tsl_h")]
    for c in tsl_cols:
        idx = temporal_cols.index(c)
        vals = ds.temporal[:, :, idx].flatten()
        vals_nonzero = vals[vals > 0]
        print(f"    {c:30s}: mean={vals.mean():.2f}, max={vals.max():.1f}, "
              f"n(>100)={int((vals > 100).sum())}")
    # Overall temporal stats
    t_mean = ds.temporal.mean(axis=(0, 1))
    outlier_feats = [(temporal_cols[i], t_mean[i]) for i in range(len(temporal_cols))
                     if abs(t_mean[i]) > 5.0]
    if outlier_feats:
        print(f"  WARNING: {len(outlier_feats)} features with |mean| > 5.0:")
        for name, val in outlier_feats[:10]:
            print(f"    {name:30s}: mean={val:.2f}")

    return ds, state_df


def load_frozen_model(
    model_id: str,
    config: dict,
    d_temporal: int,
    d_static: int,
    n_actions: int,
    device: torch.device,
    tag: str | None = None,
) -> torch.nn.Module:
    """Load a frozen MIMIC-trained model checkpoint."""
    model_cfg = config["models"].get(model_id, {})
    grid_cfg = config["grid"]

    # Determine checkpoint path
    base_name = f"{model_id}_{tag}_best.pt" if tag else f"{model_id}_best.pt"
    ckpt_path = MODELS_DIR / base_name

    # Fallback to base checkpoint if tagged version doesn't exist
    if not ckpt_path.exists() and tag:
        ckpt_path = MODELS_DIR / f"{model_id}_best.pt"
        print(f"  WARNING: {base_name} not found, falling back to {ckpt_path.name}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Build model
    if "gtransformer" in model_id:
        model = build_gtransformer_from_config(
            d_temporal, d_static, n_actions, model_cfg, grid_cfg
        )
    elif "gnet" in model_id:
        model = build_gnet_from_config(
            d_temporal, d_static, n_actions, model_cfg, grid_cfg
        )
    elif "actin" in model_id:
        model = build_actin_from_config(
            d_temporal, d_static, n_actions, model_cfg, grid_cfg
        )
    else:
        raise ValueError(f"Unknown model: {model_id}")

    # Load weights
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", "?")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {model_id}: {n_params:,} params, "
          f"epoch={epoch}, val_loss={val_loss}")

    return model


def diagnose_rollout_state_drift(
    model,
    dataset: DTRDataset,
    device: torch.device,
    batch_size: int = 256,
    max_batches: int = 20,
) -> dict:
    """
    Quantify compounding prediction error in autoregressive rollout.

    Compares predicted states from rollout vs observed states to measure
    how far the MIMIC-trained next_state_head drifts on eICU data.
    This is the key diagnostic for why rollout DM diverges from non-rollout.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    per_step_mse = []  # shape will be (n_batches, T-1)
    per_step_mae = []

    for i, batch in enumerate(tqdm(loader, desc="    Rollout drift", leave=False)):
        if i >= max_batches:
            break

        temporal = batch["temporal"].to(device)
        static = batch["static"].to(device)
        action = batch["action"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        absorbing_mask = batch["absorbing_mask"].to(device)

        # Get rollout predicted states
        out = model.predict_counterfactual_rollout(
            temporal, static, action, valid_mask, absorbing_mask
        )
        pred_states = out["predicted_states"].cpu().numpy()  # (B, T, D)
        obs_states = temporal.cpu().numpy()  # (B, T, D)
        vmask = (valid_mask * absorbing_mask).cpu().numpy()  # (B, T)

        # Per-step MSE/MAE (excluding step 0 which is always observed)
        T = pred_states.shape[1]
        step_mse = np.zeros(T - 1)
        step_mae = np.zeros(T - 1)
        step_count = np.zeros(T - 1)

        for t in range(1, T):
            mask_t = vmask[:, t] > 0
            if mask_t.sum() == 0:
                continue
            diff = pred_states[mask_t, t] - obs_states[mask_t, t]
            step_mse[t - 1] = (diff ** 2).mean()
            step_mae[t - 1] = np.abs(diff).mean()
            step_count[t - 1] = mask_t.sum()

        per_step_mse.append(step_mse)
        per_step_mae.append(step_mae)

    # Average across batches
    avg_mse = np.mean(per_step_mse, axis=0)
    avg_mae = np.mean(per_step_mae, axis=0)

    return {
        "per_step_mse": avg_mse,
        "per_step_mae": avg_mae,
        "drift_ratio": avg_mse[-1] / max(avg_mse[0], 1e-10),  # last/first step ratio
        "total_mae": avg_mae.mean(),
    }


def evaluate_model_on_eicu(
    model,
    model_id: str,
    dataset: DTRDataset,
    config: dict,
    device: torch.device,
    args,
) -> dict:
    """Run full OPE evaluation of a frozen model on eICU data."""
    print(f"\n  Evaluating {model_id} on eICU...")
    use_rollout = args.use_rollout and "gtransformer" in model_id

    ope_results = {}
    methods = config["ope"]["methods_by_model"].get(model_id.split("_rollout")[0], [])

    # ── DM (non-rollout = PRIMARY for external validation) ──
    # Rationale: rollout uses MIMIC-trained next_state_head which suffers
    # compounding prediction error on eICU distribution. Non-rollout uses
    # real eICU states with only action substitution → more reliable.
    if "DM" in methods and hasattr(model, "predict_counterfactual"):
        print("    DM (rollout=False) [PRIMARY for external validation]...")
        dm_noroll = evaluate_dm(model, dataset, device, use_rollout=False)
        ope_results["DM"] = dm_noroll  # Non-rollout is primary for external
        ope_results["DM_norollout"] = dm_noroll
        print(f"      V(obs)={dm_noroll['v_observed']:.4f}, "
              f"V(rule)={dm_noroll['v_rule']:.4f}, "
              f"ATE={dm_noroll['ate']:+.4f}")
        # Outcome logit diagnostics
        obs_logits = np.log(dm_noroll['obs_probs'] / (1 - dm_noroll['obs_probs'] + 1e-15))
        print(f"      outcome_logit: mean={obs_logits.mean():.2f}, "
              f"std={obs_logits.std():.2f}, "
              f"min={obs_logits.min():.2f}, max={obs_logits.max():.2f}")

        if use_rollout:
            print(f"    DM (rollout=True) [DIAGNOSTIC — cross-db state drift]...")
            dm_roll = evaluate_dm(model, dataset, device, use_rollout=True)
            ope_results["DM_rollout"] = dm_roll
            print(f"      V(obs)={dm_roll['v_observed']:.4f}, "
                  f"V(rule)={dm_roll['v_rule']:.4f}, "
                  f"ATE={dm_roll['ate']:+.4f}")
            roll_logits = np.log(dm_roll['obs_probs'] / (1 - dm_roll['obs_probs'] + 1e-15))
            print(f"      outcome_logit: mean={roll_logits.mean():.2f}, "
                  f"std={roll_logits.std():.2f}, "
                  f"min={roll_logits.min():.2f}, max={roll_logits.max():.2f}")

            # ── Rollout state-drift diagnostic ──
            print("    Rollout state-drift diagnostic...")
            drift = diagnose_rollout_state_drift(model, dataset, device)
            ope_results["_rollout_drift"] = drift
            print(f"      per-step MAE: {', '.join(f'{m:.4f}' for m in drift['per_step_mae'][:6])}")
            print(f"      drift ratio (last/first step MSE): {drift['drift_ratio']:.2f}x")
            print(f"      total MAE: {drift['total_mae']:.4f}")
            if drift['drift_ratio'] > 2.0:
                print(f"      ⚠ DRIFT WARNING: {drift['drift_ratio']:.1f}x compounding error "
                      f"→ rollout ATE unreliable for cross-database evaluation")

    # ── DR ──
    if "DR" in methods:
        print("    DR...")
        dr = evaluate_dr(model, dataset, device)
        ope_results["DR"] = dr
        print(f"      V(obs)={dr['v_observed']:.4f}, V(rule)={dr['v_rule']:.4f}, "
              f"ATE={dr['ate']:+.4f}, ESS={dr['ess']:.0f}")
        print(f"      weights: mean={dr['weights_mean']:.4f}, "
              f"median={dr['weights_median']:.4f}")
        # DR model component diagnostics
        print(f"      mu_obs(non-rollout)={dr.get('mu_obs_mean', 'N/A')}, "
              f"mu_rule(non-rollout)={dr.get('mu_rule_mean', 'N/A')}")

    # ── FQE ──
    if "FQE" in methods:
        print("    FQE...")
        fqe_cfg = config["ope"]["fqe"]
        q_net = train_fqe(
            dataset, model, device, fqe_cfg,
            gamma=config["reward"]["gamma"],
        )
        fqe = evaluate_fqe(q_net, dataset, device)
        ope_results["FQE"] = fqe
        print(f"      V(obs)={fqe['v_observed']:.4f}, V(rule)={fqe['v_rule']:.4f}, "
              f"ATE={fqe['ate']:+.4f}")

    # ── Bootstrap CIs ──
    boot_results = {}
    if not args.skip_bootstrap:
        n_boot = args.bootstrap
        seed = config["ope"]["bootstrap"]["seed"]

        if "DM" in ope_results:
            print(f"    Bootstrap DM non-rollout (B={n_boot}) [PRIMARY]...")
            def dm_fn(ds, **kw):
                return evaluate_dm(model, ds, device, use_rollout=False)
            boot_dm = bootstrap_ope(dm_fn, dataset, n_resamples=n_boot, seed=seed)
            boot_results["DM"] = boot_dm
            if boot_dm.get("n_success", 0) > 0:
                print(f"      ATE={boot_dm['ate_mean']:+.4f} "
                      f"[{boot_dm['ate_ci_lo']:.4f}, {boot_dm['ate_ci_hi']:.4f}]")

        if "FQE" in ope_results and "FQE" in methods:
            print(f"    Bootstrap FQE-eval (B={n_boot})...")
            boot_fqe = bootstrap_fqe_eval_only(
                q_net, dataset, device,
                n_resamples=n_boot, seed=seed + 2,
            )
            boot_results["FQE"] = boot_fqe
            if boot_fqe.get("n_success", 0) > 0:
                print(f"      ATE={boot_fqe['ate_mean']:+.4f} "
                      f"[{boot_fqe['ate_ci_lo']:.4f}, {boot_fqe['ate_ci_hi']:.4f}]")

    return {"ope": ope_results, "bootstrap": boot_results}


def run_subgroup_analysis(
    model,
    model_id: str,
    state_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    propensity_df: pd.DataFrame,
    cohort: pd.DataFrame,
    config: dict,
    device: torch.device,
    mimic_spec: dict,
    use_rollout: bool = False,
) -> list[dict]:
    """Run DM evaluation on HF subtype subgroups."""
    print(f"\n  Subgroup analysis for {model_id}...")

    temporal_cols = mimic_spec["temporal_cols"]
    static_cols = mimic_spec["static_cols"]
    merge_path = ROOT / "configs" / "70b_action_merge_map.yaml"
    merge_map = None
    if merge_path.exists():
        with open(merge_path) as f:
            mm = yaml.safe_load(f)
        if "merge_map" in mm:
            merge_map = {int(k): int(v) for k, v in mm["merge_map"].items()}

    results = []

    # HF subtype stratification
    if "hf_subtype" in cohort.columns:
        hf_pheno = cohort[["patientunitstayid", "hf_subtype"]].rename(
            columns={"patientunitstayid": "stay_id"}
        )
    else:
        print("    WARNING: No hf_subtype in cohort, skipping subgroup")
        return results

    for group_name, group_df in hf_pheno.groupby("hf_subtype"):
        group_ids = group_df["stay_id"].values
        if len(group_ids) < 30:
            print(f"    {group_name}: n={len(group_ids)} < 30, skipping")
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
            action_vocab=config["action"]["valid_merged_indices"],
        )

        if hasattr(model, "predict_counterfactual"):
            # Always use non-rollout for external validation subgroups
            # (rollout state transition model doesn't transfer cross-database)
            dm = evaluate_dm(model, sub_ds, device, use_rollout=False)
            results.append({
                "model": f"90b_eicu_{model_id}",
                "stratifier": "hf_subtype",
                "group": str(group_name),
                "n_stays": len(sub_ds),
                "v_observed": dm["v_observed"],
                "v_rule": dm["v_rule"],
                "ate": dm["ate"],
                "database": "eICU",
            })
            print(f"    {group_name}: n={len(sub_ds)}, ATE={dm['ate']:+.4f}")

    return results


def build_transportability_comparison(
    eicu_results: dict,
    mimic_ref: dict,
) -> pd.DataFrame:
    """Compare eICU vs MIMIC results for transportability assessment."""
    rows = []
    for model_id, eicu_data in eicu_results.items():
        for est_name, est_result in eicu_data["ope"].items():
            row = {
                "model": model_id,
                "estimator": est_name,
                "eicu_ate": est_result.get("ate"),
                "eicu_v_observed": est_result.get("v_observed"),
                "eicu_v_rule": est_result.get("v_rule"),
            }
            # Add bootstrap CI
            boot = eicu_data.get("bootstrap", {}).get(est_name, {})
            if boot.get("n_success", 0) > 0:
                row["eicu_ci_lo"] = boot["ate_ci_lo"]
                row["eicu_ci_hi"] = boot["ate_ci_hi"]

            # Add MIMIC reference
            if model_id in mimic_ref:
                mref = mimic_ref[model_id]
                if "policy_values" in mref:
                    for pv in mref["policy_values"]:
                        if pv.get("estimator") == est_name:
                            row["mimic_ate"] = pv.get("ate")
                            row["mimic_v_observed"] = pv.get("v_observed")
                if "bootstrap_ci" in mref:
                    for bc in mref["bootstrap_ci"]:
                        if bc.get("estimator") == est_name:
                            row["mimic_ci_lo"] = bc.get("ate_ci_lo")
                            row["mimic_ci_hi"] = bc.get("ate_ci_hi")

            # Compute consistency metrics
            if "mimic_ate" in row and row.get("eicu_ate") is not None:
                row["ate_ratio"] = row["eicu_ate"] / row["mimic_ate"] if row["mimic_ate"] != 0 else np.nan
                row["same_sign"] = (
                    np.sign(row["eicu_ate"]) == np.sign(row["mimic_ate"])
                )

                # CI overlap check
                if all(k in row for k in ["eicu_ci_lo", "eicu_ci_hi", "mimic_ci_lo", "mimic_ci_hi"]):
                    row["ci_overlap"] = not (
                        row["eicu_ci_hi"] < row["mimic_ci_lo"] or
                        row["mimic_ci_hi"] < row["eicu_ci_lo"]
                    )

            rows.append(row)

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    print("=" * 72)
    print("90b_05 — Evaluate Frozen MIMIC Models on eICU")
    print("=" * 72)
    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 1. Setup ─────────────────────────────────────────────────────────
    print("\n[1] Setup...")
    config_path = ROOT / "configs" / "80_step_c_training_v1.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  WARNING: No GPU, using CPU (will be slow)")

    # ── 2. Load MIMIC feature spec + eICU dataset ──────────────────────
    print("\n[2] Loading MIMIC feature spec and eICU DTR dataset...")
    mimic_spec = load_mimic_feature_spec()
    print(f"  MIMIC spec: {mimic_spec['n_temporal']} temporal, "
          f"{mimic_spec['n_static']} static, {mimic_spec['n_actions']} actions")

    eicu_ds, state_df = load_eicu_dtr_dataset(config, mimic_spec)

    # Also load raw data for subgroup analysis
    outcomes_df = pd.read_parquet(INTERIM_DIR / "90b_eicu_outcomes_absorb.parquet")
    propensity_df = pd.read_parquet(INTERIM_DIR / "90b_eicu_propensity_coarse3.parquet")
    cohort = pd.read_parquet(STAGING_DIR / "90b_eicu_cohort_v1.parquet")

    # Use MIMIC dimensions for model construction (NOT eICU auto-detected)
    mimic_n_actions = mimic_spec["n_actions"]
    print(f"  Model dimensions: d_temporal={eicu_ds.d_temporal}, "
          f"d_static={eicu_ds.d_static}, n_actions={mimic_n_actions} (MIMIC)")

    # ── 2b. P1-3: infusionDrug-only hospital filter ──────────────────────
    infusion_drug_meta = {}
    if args.infusion_drug_only:
        site_csv = TABLES_DIR / "90b_04_eicu_site_heterogeneity.csv"
        if site_csv.exists() and "hospitalid" in cohort.columns:
            site_df_full = pd.read_csv(site_csv)
            if "has_infusiondrug" in site_df_full.columns:
                infdrug_hids = set(
                    site_df_full.loc[site_df_full["has_infusiondrug"], "hospitalid"]
                )
            else:
                # Fallback: vaso_nz_pct > 0 as proxy
                print("  WARNING: has_infusiondrug column missing, using vaso_nz_pct>0 proxy")
                infdrug_hids = set(
                    site_df_full.loc[site_df_full["vaso_nz_pct"] > 0, "hospitalid"]
                )

            n_before = len(cohort)
            keep_stays = set(
                cohort.loc[cohort["hospitalid"].isin(infdrug_hids), "patientunitstayid"]
            )
            cohort = cohort[cohort["patientunitstayid"].isin(keep_stays)].reset_index(drop=True)
            state_df = state_df[state_df["stay_id"].isin(keep_stays)].reset_index(drop=True)
            outcomes_df = outcomes_df[outcomes_df["stay_id"].isin(keep_stays)].reset_index(drop=True)
            propensity_df = propensity_df[propensity_df["stay_id"].isin(keep_stays)].reset_index(drop=True)

            # Rebuild eICU DTRDataset with filtered stays
            merge_path = ROOT / "configs" / "70b_action_merge_map.yaml"
            merge_map = None
            if merge_path.exists():
                with open(merge_path) as f:
                    mm = yaml.safe_load(f)
                if "merge_map" in mm:
                    merge_map = {int(k): int(v) for k, v in mm["merge_map"].items()}

            eicu_ds = DTRDataset(
                state_df=state_df,
                outcomes_df=outcomes_df,
                propensity_df=propensity_df,
                stay_ids=np.array(list(keep_stays)),
                temporal_cols=mimic_spec["temporal_cols"],
                static_cols=mimic_spec["static_cols"],
                merge_map=merge_map,
                n_bins=config["grid"]["n_bins"],
                action_vocab=config["action"]["valid_merged_indices"],
            )

            infusion_drug_meta = {
                "n_hospitals": len(infdrug_hids),
                "n_stays": len(keep_stays),
                "n_excluded": n_before - len(keep_stays),
            }
            print(f"\n  P1-3 infusionDrug filter ACTIVE:")
            print(f"    Hospitals with infusionDrug: {len(infdrug_hids)}")
            print(f"    Stays kept: {len(keep_stays):,} / {n_before:,} "
                  f"(excluded {n_before - len(keep_stays):,})")
            print(f"    Rebuilt DTRDataset: {len(eicu_ds)} stays")
        else:
            print("  WARNING: cannot apply --infusion-drug-only "
                  "(site CSV or hospitalid missing)")

    # ── 3. Load MIMIC reference results ──────────────────────────────────
    print("\n[3] Loading MIMIC reference results...")
    registry_path = ROOT / "configs" / "90a_primary_model_registry.json"
    mimic_ref = {}
    if registry_path.exists():
        with open(registry_path) as f:
            registry = json.load(f)
        mimic_ref = registry.get("mimic_reference_results", {})
        print(f"  MIMIC reference models: {list(mimic_ref.keys())}")

    # ── 4. Tier A site-stratified sensitivity analysis (RUN FIRST) ──────
    # Restrict to hospitals with fluid_nz_pct >= threshold (treatment
    # documentation quality comparable to MIMIC). This tests whether the a0
    # inflation and ATE divergence are driven by site-level documentation
    # gaps vs. genuine transportability issues.
    # Runs BEFORE full cohort so key results are visible early in logs.
    tier_a_thr = args.tier_a_threshold
    tier_a_results = {}
    site_csv = TABLES_DIR / "90b_04_eicu_site_heterogeneity.csv"
    if site_csv.exists():
        print(f"\n{'='*60}")
        print(f"[4] Tier A sensitivity analysis (fluid_nz >= {tier_a_thr:.0f}%)")
        print(f"{'='*60}")

        site_df = pd.read_csv(site_csv)
        tier_a_hids = set(site_df.loc[site_df["fluid_nz_pct"] >= tier_a_thr, "hospitalid"])
        print(f"  Tier A hospitals: {len(tier_a_hids)} / {len(site_df)}")

        # Map stays to hospitals via cohort
        if "hospitalid" in cohort.columns:
            tier_a_stay_ids = set(
                cohort.loc[cohort["hospitalid"].isin(tier_a_hids), "patientunitstayid"]
            )
            print(f"  Tier A stays: {len(tier_a_stay_ids):,} / {len(cohort):,} "
                  f"({100*len(tier_a_stay_ids)/len(cohort):.1f}%)")

            if len(tier_a_stay_ids) >= 100:
                # Build Tier A dataset
                temporal_cols = mimic_spec["temporal_cols"]
                static_cols = mimic_spec["static_cols"]
                merge_path = ROOT / "configs" / "70b_action_merge_map.yaml"
                merge_map = None
                if merge_path.exists():
                    with open(merge_path) as f:
                        mm = yaml.safe_load(f)
                    if "merge_map" in mm:
                        merge_map = {int(k): int(v) for k, v in mm["merge_map"].items()}

                tier_a_ds = DTRDataset(
                    state_df=state_df,
                    outcomes_df=outcomes_df,
                    propensity_df=propensity_df,
                    stay_ids=np.array(list(tier_a_stay_ids)),
                    temporal_cols=temporal_cols,
                    static_cols=static_cols,
                    merge_map=merge_map,
                    n_bins=config["grid"]["n_bins"],
                    action_vocab=config["action"]["valid_merged_indices"],
                )
                print(f"  Tier A dataset: {len(tier_a_ds)} stays, "
                      f"{tier_a_ds.n_actions} actions")

                # Action distribution in Tier A
                a_counts = np.bincount(tier_a_ds.action[:, 0], minlength=tier_a_ds.n_actions)
                a0_pct = 100 * a_counts[0] / a_counts.sum() if a_counts.sum() > 0 else 0
                print(f"  Tier A a0 prevalence: {a0_pct:.1f}% "
                      f"(full cohort for comparison: see 90b_04 report)")

                # Evaluate each model on Tier A (full OPE pipeline)
                n_boot = args.bootstrap
                boot_seed = config["ope"]["bootstrap"]["seed"]
                methods_cfg = config["ope"]["methods_by_model"]

                for model_id in args.models:
                    print(f"\n  {'─'*50}")
                    print(f"  Tier A evaluation: {model_id}")
                    print(f"  {'─'*50}")
                    try:
                        model = load_frozen_model(
                            model_id, config,
                            d_temporal=eicu_ds.d_temporal,
                            d_static=eicu_ds.d_static,
                            n_actions=mimic_n_actions,
                            device=device,
                            tag=args.tag if "gtransformer" in model_id else None,
                        )
                        methods = methods_cfg.get(model_id.split("_rollout")[0], [])

                        ta_ope = {}
                        ta_boot = {}

                        # ── DM non-rollout (PRIMARY) ──
                        use_rollout_ta = args.use_rollout and "gtransformer" in model_id
                        if "DM" in methods and hasattr(model, "predict_counterfactual"):
                            print("    DM (non-rollout) [PRIMARY]...")
                            dm = evaluate_dm(model, tier_a_ds, device, use_rollout=False)
                            ta_ope["DM"] = dm
                            ta_ope["DM_norollout"] = dm
                            print(f"      V(obs)={dm['v_observed']:.4f}, "
                                  f"V(rule)={dm['v_rule']:.4f}, "
                                  f"ATE={dm['ate']:+.4f}")

                            # Bootstrap DM
                            if not args.skip_bootstrap:
                                print(f"    Bootstrap DM (B={n_boot})...")
                                def dm_fn(ds, **kw):
                                    return evaluate_dm(model, ds, device, use_rollout=False)
                                boot_dm = bootstrap_ope(
                                    dm_fn, tier_a_ds,
                                    n_resamples=n_boot, seed=boot_seed,
                                )
                                ta_boot["DM"] = boot_dm
                                if boot_dm.get("n_success", 0) > 0:
                                    print(f"      ATE={boot_dm['ate_mean']:+.4f} "
                                          f"[{boot_dm['ate_ci_lo']:.4f}, "
                                          f"{boot_dm['ate_ci_hi']:.4f}]")

                            # ── DM rollout (DIAGNOSTIC) ──
                            if use_rollout_ta:
                                print("    DM (rollout=True) [DIAGNOSTIC — Tier A]...")
                                dm_roll = evaluate_dm(model, tier_a_ds, device, use_rollout=True)
                                ta_ope["DM_rollout"] = dm_roll
                                print(f"      V(obs)={dm_roll['v_observed']:.4f}, "
                                      f"V(rule)={dm_roll['v_rule']:.4f}, "
                                      f"ATE={dm_roll['ate']:+.4f}")
                                roll_logits = np.log(
                                    dm_roll['obs_probs'] /
                                    (1 - dm_roll['obs_probs'] + 1e-15)
                                )
                                print(f"      outcome_logit: mean={roll_logits.mean():.2f}, "
                                      f"std={roll_logits.std():.2f}")

                                # Rollout state-drift diagnostic on Tier A
                                print("    Rollout state-drift diagnostic (Tier A)...")
                                ta_drift = diagnose_rollout_state_drift(
                                    model, tier_a_ds, device
                                )
                                ta_ope["_rollout_drift"] = ta_drift
                                print(f"      drift ratio: {ta_drift['drift_ratio']:.2f}x, "
                                      f"total MAE: {ta_drift['total_mae']:.4f}")

                                # Bootstrap rollout DM
                                if not args.skip_bootstrap:
                                    print(f"    Bootstrap DM rollout (B={n_boot})...")
                                    def dm_roll_fn(ds, **kw):
                                        return evaluate_dm(model, ds, device, use_rollout=True)
                                    boot_dm_roll = bootstrap_ope(
                                        dm_roll_fn, tier_a_ds,
                                        n_resamples=n_boot, seed=boot_seed + 1,
                                    )
                                    ta_boot["DM_rollout"] = boot_dm_roll
                                    if boot_dm_roll.get("n_success", 0) > 0:
                                        print(f"      ATE={boot_dm_roll['ate_mean']:+.4f} "
                                              f"[{boot_dm_roll['ate_ci_lo']:.4f}, "
                                              f"{boot_dm_roll['ate_ci_hi']:.4f}]")

                        # ── FQE (train on Tier A subset) ──
                        if "FQE" in methods:
                            print("    FQE (training on Tier A subset)...")
                            fqe_cfg = config["ope"]["fqe"]
                            ta_q_net = train_fqe(
                                tier_a_ds, model, device, fqe_cfg,
                                gamma=config["reward"]["gamma"],
                            )
                            fqe = evaluate_fqe(ta_q_net, tier_a_ds, device)
                            ta_ope["FQE"] = fqe
                            print(f"      V(obs)={fqe['v_observed']:.4f}, "
                                  f"V(rule)={fqe['v_rule']:.4f}, "
                                  f"ATE={fqe['ate']:+.4f}")

                            # Bootstrap FQE-eval
                            if not args.skip_bootstrap:
                                print(f"    Bootstrap FQE-eval (B={n_boot})...")
                                boot_fqe = bootstrap_fqe_eval_only(
                                    ta_q_net, tier_a_ds, device,
                                    n_resamples=n_boot, seed=boot_seed + 2,
                                )
                                ta_boot["FQE"] = boot_fqe
                                if boot_fqe.get("n_success", 0) > 0:
                                    print(f"      ATE={boot_fqe['ate_mean']:+.4f} "
                                          f"[{boot_fqe['ate_ci_lo']:.4f}, "
                                          f"{boot_fqe['ate_ci_hi']:.4f}]")

                        tier_a_results[model_id] = {
                            "n_stays": len(tier_a_ds),
                            "n_hospitals": len(tier_a_hids),
                            "a0_pct": a0_pct,
                            "ope": ta_ope,
                            "bootstrap": ta_boot,
                        }

                        # Save Tier A per-model results
                        ta_model_id = f"90b_eicu_tier_a_{model_id}"
                        save_policy_values(ta_ope, ta_model_id, TABLES_DIR)
                        if ta_boot:
                            save_bootstrap_cis(ta_boot, ta_model_id, TABLES_DIR)

                        # ── Subgroup within Tier A ──
                        if "hf_subtype" in cohort.columns:
                            tier_a_cohort = cohort[
                                cohort["patientunitstayid"].isin(tier_a_stay_ids)
                            ]
                            for grp, gdf in tier_a_cohort.groupby("hf_subtype"):
                                gids = gdf["patientunitstayid"].values
                                if len(gids) < 30:
                                    continue
                                g_ds = DTRDataset(
                                    state_df=state_df,
                                    outcomes_df=outcomes_df,
                                    propensity_df=propensity_df,
                                    stay_ids=gids,
                                    temporal_cols=temporal_cols,
                                    static_cols=static_cols,
                                    merge_map=merge_map,
                                    n_bins=config["grid"]["n_bins"],
                                    action_vocab=config["action"]["valid_merged_indices"],
                                )
                                g_dm = evaluate_dm(model, g_ds, device, use_rollout=False)
                                print(f"    Tier A {grp}: n={len(g_ds)}, "
                                      f"ATE={g_dm['ate']:+.4f}")
                                tier_a_results[model_id][f"subgroup_{grp}"] = {
                                    "n": len(g_ds), "ate": g_dm["ate"],
                                    "v_observed": g_dm["v_observed"],
                                    "v_rule": g_dm["v_rule"],
                                }

                    except FileNotFoundError as e:
                        print(f"    SKIPPED: {e}")
                    except Exception as e:
                        print(f"    ERROR in Tier A eval: {e}")
                        import traceback
                        traceback.print_exc()

                # Save Tier A summary CSV
                tier_a_rows = []
                for mid, ta in tier_a_results.items():
                    for est_name, est in ta["ope"].items():
                        if est_name.startswith("_"):
                            continue  # skip internal diagnostics
                        row = {
                            "model": mid, "subset": "tier_a_full",
                            "estimator": est_name,
                            "n_stays": ta["n_stays"],
                            "n_hospitals": ta["n_hospitals"],
                            "a0_pct": ta["a0_pct"],
                            "v_observed": est["v_observed"],
                            "v_rule": est["v_rule"],
                            "ate": est["ate"],
                        }
                        boot = ta.get("bootstrap", {}).get(est_name, {})
                        if boot.get("n_success", 0) > 0:
                            row["ate_ci_lo"] = boot["ate_ci_lo"]
                            row["ate_ci_hi"] = boot["ate_ci_hi"]
                        tier_a_rows.append(row)
                    for key, val in ta.items():
                        if key.startswith("subgroup_") and isinstance(val, dict):
                            tier_a_rows.append({
                                "model": mid,
                                "subset": f"tier_a_{key.replace('subgroup_', '')}",
                                "estimator": "DM",
                                "n_stays": val["n"],
                                "n_hospitals": ta["n_hospitals"],
                                "a0_pct": ta["a0_pct"],
                                "v_observed": val["v_observed"],
                                "v_rule": val["v_rule"],
                                "ate": val["ate"],
                            })
                tier_a_path = TABLES_DIR / "90b_05_eicu_tier_a_sensitivity.csv"
                pd.DataFrame(tier_a_rows).to_csv(tier_a_path, index=False)
                print(f"\n  Tier A results saved: {tier_a_path}")
            else:
                print(f"  WARNING: only {len(tier_a_stay_ids)} Tier A stays, skipping")
        else:
            print("  WARNING: hospitalid not in cohort, cannot filter by site tier")
    else:
        print("\n  [4] Site heterogeneity CSV not found, skipping Tier A analysis")
        print(f"      Expected: {site_csv}")

    # ── 5. Full cohort evaluation ─────────────────────────────────────────
    all_results = {}
    all_subgroups = []

    for model_id in args.models:
        print(f"\n{'='*60}")
        print(f"[5] Full cohort evaluation: {model_id}")
        print(f"{'='*60}")

        try:
            # Load frozen model (use MIMIC n_actions, not eICU auto-detected)
            model = load_frozen_model(
                model_id, config,
                d_temporal=eicu_ds.d_temporal,
                d_static=eicu_ds.d_static,
                n_actions=mimic_n_actions,
                device=device,
                tag=args.tag if "gtransformer" in model_id else None,
            )

            # OPE evaluation
            results = evaluate_model_on_eicu(
                model, model_id, eicu_ds, config, device, args
            )
            all_results[model_id] = results

            # Save per-model results
            eicu_model_id = f"90b_eicu_{model_id}"
            save_policy_values(results["ope"], eicu_model_id, TABLES_DIR)
            if results["bootstrap"]:
                save_bootstrap_cis(results["bootstrap"], eicu_model_id, TABLES_DIR)

            # Subgroup analysis
            use_rollout = args.use_rollout and "gtransformer" in model_id
            sg = run_subgroup_analysis(
                model, model_id,
                state_df, outcomes_df, propensity_df, cohort,
                config, device, mimic_spec=mimic_spec,
                use_rollout=use_rollout,
            )
            all_subgroups.extend(sg)
            if sg:
                save_subgroup_results(sg, eicu_model_id, TABLES_DIR)

        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            continue
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ── 6. Transportability comparison ───────────────────────────────────
    print("\n[6] Building transportability comparison...")
    transport_df = build_transportability_comparison(all_results, mimic_ref)
    transport_path = TABLES_DIR / "90b_eicu_transportability_comparison.csv"
    transport_df.to_csv(transport_path, index=False)
    print(f"  Saved: {transport_path}")

    if not transport_df.empty:
        print("\n  Transportability summary:")
        for _, row in transport_df.iterrows():
            mimic_ate = row.get("mimic_ate", "?")
            eicu_ate = row.get("eicu_ate", "?")
            same_sign = row.get("same_sign", "?")
            ci_overlap = row.get("ci_overlap", "?")
            print(f"    {row['model']} {row['estimator']}: "
                  f"MIMIC={mimic_ate}, eICU={eicu_ate}, "
                  f"same_sign={same_sign}, CI_overlap={ci_overlap}")

    # ── 6. Comprehensive report ──────────────────────────────────────────
    print("\n[7] Generating report...")
    report_lines = [
        "=" * 72,
        "90b_05 — eICU External Validation Report",
        "=" * 72,
        f"Generated: {ts}",
        f"Protocol: v3.1, Step 90b",
        "",
        "DATASET",
        "───────",
        f"  Database: eICU-CRD",
        f"  InfusionDrug filter: {'ACTIVE' if infusion_drug_meta else 'OFF'}",
        f"  Stays: {len(eicu_ds):,}",
        f"  Features: temporal={eicu_ds.d_temporal}, static={eicu_ds.d_static}",
        f"  Actions (model): {mimic_n_actions} (MIMIC), "
        f"{eicu_ds.n_actions} (eICU auto-detected)",
        f"  Ascertainment: BROADER-COMPATIBLE (29d)",
        "",
    ]

    # Per-model results
    for model_id, results in all_results.items():
        report_lines.append(f"MODEL: {model_id}")
        report_lines.append("─" * len(f"MODEL: {model_id}"))
        for est_name, est in results["ope"].items():
            if est_name.startswith("_"):
                continue  # skip internal diagnostics (_rollout_drift etc.)
            line = (f"  {est_name}: V(obs)={est['v_observed']:.4f}, "
                    f"V(rule)={est['v_rule']:.4f}, ATE={est['ate']:+.4f}")
            boot = results.get("bootstrap", {}).get(est_name, {})
            if boot.get("n_success", 0) > 0:
                line += f" [{boot['ate_ci_lo']:.4f}, {boot['ate_ci_hi']:.4f}]"
            report_lines.append(line)
        report_lines.append("")

    # Subgroup results
    if all_subgroups:
        report_lines.append("SUBGROUP ANALYSIS (eICU)")
        report_lines.append("───────────────────────")
        for sg in all_subgroups:
            report_lines.append(
                f"  {sg['model']} | {sg['stratifier']}={sg['group']}: "
                f"n={sg['n_stays']}, ATE={sg['ate']:+.4f}"
            )
        report_lines.append("")

    # Transportability
    report_lines.append("TRANSPORTABILITY COMPARISON (eICU vs MIMIC)")
    report_lines.append("────────────────────────────────────────────")
    if not transport_df.empty:
        for _, row in transport_df.iterrows():
            report_lines.append(
                f"  {row['model']} {row['estimator']}: "
                f"MIMIC ATE={row.get('mimic_ate', '?')}, "
                f"eICU ATE={row.get('eicu_ate', '?')}, "
                f"same_sign={row.get('same_sign', '?')}, "
                f"CI_overlap={row.get('ci_overlap', '?')}"
            )
    else:
        report_lines.append("  No comparison available (missing MIMIC reference)")

    # Tier A sensitivity analysis
    if tier_a_results:
        report_lines.extend([
            "",
            f"TIER A SITE-STRATIFIED SENSITIVITY (fluid_nz >= {tier_a_thr:.0f}%)",
            "─" * 53,
        ])
        for model_id, ta in tier_a_results.items():
            report_lines.append(
                f"  {model_id}: n={ta['n_stays']:,} stays, "
                f"{ta['n_hospitals']} hospitals, a0={ta['a0_pct']:.1f}%"
            )
            for est_name, est in ta.get("ope", {}).items():
                if est_name.startswith("_"):
                    continue  # skip internal diagnostics
                line = (f"    {est_name}: V(obs)={est['v_observed']:.4f}, "
                        f"V(rule)={est['v_rule']:.4f}, ATE={est['ate']:+.4f}")
                boot = ta.get("bootstrap", {}).get(est_name, {})
                if boot.get("n_success", 0) > 0:
                    line += f" [{boot['ate_ci_lo']:.4f}, {boot['ate_ci_hi']:.4f}]"
                report_lines.append(line)
            # Rollout drift for Tier A
            ta_drift = ta.get("ope", {}).get("_rollout_drift")
            if ta_drift:
                report_lines.append(
                    f"    rollout drift: ratio={ta_drift['drift_ratio']:.2f}x, "
                    f"MAE={ta_drift['total_mae']:.4f}"
                )
            # Subgroup results within Tier A
            for key, val in ta.items():
                if key.startswith("subgroup_") and isinstance(val, dict):
                    grp_name = key.replace("subgroup_", "")
                    report_lines.append(
                        f"    {grp_name}: n={val['n']}, ATE={val['ate']:+.4f}"
                    )
        report_lines.append("")

    # Rollout drift diagnostic
    for model_id, results in all_results.items():
        drift = results["ope"].get("_rollout_drift")
        if drift:
            report_lines.extend([
                f"ROLLOUT STATE-DRIFT DIAGNOSTIC ({model_id})",
                "──────────────────────────────────────────",
                f"  Drift ratio (last/first step MSE): {drift['drift_ratio']:.2f}x",
                f"  Total MAE: {drift['total_mae']:.4f}",
                f"  Per-step MAE: {', '.join(f'{m:.4f}' for m in drift['per_step_mae'][:6])}",
                "",
                "  Interpretation: MIMIC-trained next_state_head exhibits compounding",
                "  prediction error on eICU state transitions. The drift ratio quantifies",
                "  how much autoregressive prediction error amplifies over time steps.",
                "  Drift ratio > 2.0 indicates rollout DM is unreliable for cross-database",
                "  evaluation. Non-rollout DM (teacher-forced on real eICU states) is the",
                "  appropriate primary estimator for external validation.",
                "",
            ])

    report_lines.extend([
        "",
        "EJECTFX CALIBRATION NOTE",
        "────────────────────────",
        "  eICU apachePredVar.ejectfx: 98.2% coded -1 (not assessed).",
        "  Only 1.8% of all stays (3,053/171,177) have actual EF values.",
        "  Among ICD-based HFpEF stays with recorded EF (N~155):",
        "    EF >= 50% (true HFpEF):  ~38%",
        "    EF 40-49% (HFmrEF):     ~16%",
        "    EF < 40% (HFrEF):       ~35%",
        "  Implication: broader-compatible ascertainment has ~35% HFrEF",
        "  contamination. This attenuates HFpEF-specific signals and should",
        "  be noted when comparing eICU ATE to MIMIC (strict EF-based).",
        "  No EF data found in nurseCharting or physicalExam tables.",
        "",
        "INTERPRETATION NOTES",
        "────────────────────",
        "  1. eICU HFpEF uses broader-compatible ascertainment (29d)",
        "     → includes some HFmrEF/HFrEF misclassified as HFpEF (see calibration above)",
        "  2. 28-day mortality approximated from hospital discharge",
        "     → likely underestimates true 28d mortality",
        "  3. eICU is multicenter US data → treatment pattern variation expected",
        "  4. Distribution shift in state variables may affect model predictions",
        "  5. Non-rollout DM is the PRIMARY estimator for external validation",
        "     because rollout's autoregressive state predictions suffer compounding",
        "     error from MIMIC-eICU distribution shift (next_state_head trained on",
        "     MIMIC cannot reliably predict eICU state transitions)",
        "  6. Rollout DM serves as a DIAGNOSTIC for state transition model",
        "     transportability — rollout-vs-nonrollout gap quantifies how much",
        "     the G-Transformer's learned dynamics transfer across databases",
        "",
        "ESTIMATOR HIERARCHY (external validation)",
        "──────────────────────────────────────────",
        "  PRIMARY:    DM (non-rollout) — real eICU states + action substitution",
        "  SECONDARY:  FQE — trained Q-network on eICU transitions",
        "  DIAGNOSTIC: DR — collapses under action distribution mismatch",
        "  DIAGNOSTIC: DM (rollout) — measures state-model transportability",
        "",
        "NEXT STEPS",
        "──────────",
        "  90c: HiRID external validation (if eligible)",
        "  90d: Transportability summary memo",
        "  90e: Cross-dataset consistency report",
    ])

    report_path = REPORTS_DIR / "90b_05_eicu_external_validation_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"  Report: {report_path}")

    total_time = time.time() - t0
    print()
    print("=" * 72)
    print(f"90b_05 COMPLETE. Total time: {total_time/60:.1f} min")
    print("=" * 72)


if __name__ == "__main__":
    main()
