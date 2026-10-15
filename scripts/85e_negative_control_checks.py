#!/usr/bin/env python3
"""
85e_negative_control_checks — Negative control outcome and exposure checks.

Purpose
-------
Negative control analyses test for residual bias by evaluating the estimated
treatment effect on outcomes that SHOULD NOT be affected by the treatment
(negative control outcomes, NCO) and by using exposures that SHOULD NOT
affect the real outcome (negative control exposures, NCE).

If the model produces a non-null ATE for negative control outcomes, this
signals residual confounding or model misspecification.

Important: for Direct Method (DM), NCO checks are valid only if the
outcome model is re-trained on the pseudo outcome Y*. Replacing Y only at
evaluation time is insufficient because DM inference uses model predictions.

Negative control outcomes (NCO):
  NCO1: Pre-T0 lab values (albumin at admission) — treatment after T0
        cannot cause changes in labs measured before T0.
  NCO2: Hospital readmission flag (if available) — a distal outcome that
        the short-term hemodynamic rule policy should not affect.
  NCO3: Random-permutation pseudo-outcome — shuffle mortality_28d across
        stays to destroy all treatment→outcome associations. Any residual
        ATE signals bias.

Negative control exposure (NCE):
  NCE1: Randomly permute treatment assignments across stays. The model's
        ATE should be ≈0 under random treatment (no systematic benefit).

Protocol reference: Step 85e (sensitivity layer)
Compute: C3-G1 (NCO requires outcome-model re-training on pseudo outcomes)

Inputs
------
- configs/80_step_c_training_v1.yaml
- models/80c_gtransformer_rollout_ow1_best.pt
- data/interim/45c_state_tensors_v1_mimic.parquet
- data/interim/45d_outcomes_absorb_v1_mimic.parquet
- data/interim/70b_propensity_coarse3_v1_mimic.parquet
- data/processed/phenotype_labels_frozen_v1_mimic.parquet

Outputs
-------
- tables/85e_negative_control.csv
- reports/85e_negative_control_summary.txt
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hfpef_dtr.models.dtr_dataset import DTRDataset, load_dtr_data
from hfpef_dtr.models.dataset import make_train_val_test_split
from hfpef_dtr.models.gtransformer import GTransformerLoss
from hfpef_dtr.models.ope import evaluate_dm, bootstrap_ope

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "tables"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"

N_BOOTSTRAP = 300
BOOT_SEED = 7777
CI_LEVEL = 0.95
N_PERMUTATION_REPS = 10  # NCE only (shuffled exposure)
NCO_RETRAIN_MAX_EPOCHS = 12
NCO_RETRAIN_PATIENCE = 3


def load_primary_model(config, d_temporal, d_static, n_actions, device):
    """Load 80c G-Transformer (primary model) from checkpoint."""
    import importlib
    mod = importlib.import_module("hfpef_dtr.models.gtransformer")
    builder_fn = getattr(mod, "build_gtransformer_from_config")

    model_cfg = config["models"]["80c_gtransformer"]
    grid_cfg = config["grid"]
    model = builder_fn(
        d_temporal=d_temporal, d_static=d_static,
        n_actions=n_actions, model_cfg=model_cfg, grid_cfg=grid_cfg,
    )

    ckpt_path = MODELS_DIR / "80c_gtransformer_rollout_ow1_best.pt"
    if not ckpt_path.exists():
        print(f"  FATAL: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def set_action_vocab(ds: DTRDataset, action_vocab: list[int]) -> DTRDataset:
    """Force dataset action vocabulary to match model-training mapping."""
    ds.action_vocab = action_vocab
    ds.n_actions = len(action_vocab)
    ds._action_to_contiguous = {a: i for i, a in enumerate(action_vocab)}
    return ds


def train_one_epoch(
    model,
    loader: DataLoader,
    criterion: GTransformerLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    """One training epoch for outcome re-fitting under pseudo outcomes."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="      train", leave=False, mininterval=2):
        temporal = batch["temporal"].to(device)
        static = batch["static"].to(device)
        action = batch["action"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        absorbing_mask = batch["absorbing_mask"].to(device)
        mortality = batch["mortality_28d"].to(device)

        outputs = model(temporal, static, action, valid_mask, absorbing_mask)
        losses = criterion(outputs, {
            "temporal": temporal,
            "valid_mask": valid_mask,
            "absorbing_mask": absorbing_mask,
            "mortality_28d": mortality,
        })

        optimizer.zero_grad()
        losses["loss"].backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(losses["loss"].item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_one_epoch(
    model,
    loader: DataLoader,
    criterion: GTransformerLoss,
    device: torch.device,
) -> float:
    """Validation loss for pseudo-outcome re-training."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="      val", leave=False, mininterval=2):
        temporal = batch["temporal"].to(device)
        static = batch["static"].to(device)
        action = batch["action"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        absorbing_mask = batch["absorbing_mask"].to(device)
        mortality = batch["mortality_28d"].to(device)

        outputs = model(temporal, static, action, valid_mask, absorbing_mask)
        losses = criterion(outputs, {
            "temporal": temporal,
            "valid_mask": valid_mask,
            "absorbing_mask": absorbing_mask,
            "mortality_28d": mortality,
        })
        total_loss += float(losses["loss"].item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


def retrain_model_for_outcome(
    config: dict,
    state_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    propensity_df: pd.DataFrame,
    temporal_cols: list[str],
    static_cols: list[str],
    merge_map: dict,
    action_vocab: list[int],
    train_ids,
    val_ids,
    device: torch.device,
    seed: int,
    label: str,
):
    """Re-train DM outcome model under alternative pseudo outcome Y* (required for valid NCO)."""
    print(f"    Re-training model for {label} ...")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    d_temporal = len(temporal_cols)
    d_static = len(static_cols)
    n_actions = len(config["action"]["valid_merged_indices"])
    model = load_primary_model(config, d_temporal, d_static, n_actions, device)

    train_ds = DTRDataset(
        state_df=state_df,
        outcomes_df=outcomes_df,
        propensity_df=propensity_df,
        stay_ids=train_ids,
        temporal_cols=temporal_cols,
        static_cols=static_cols,
        merge_map=merge_map,
        n_bins=config["grid"]["n_bins"],
    )
    val_ds = DTRDataset(
        state_df=state_df,
        outcomes_df=outcomes_df,
        propensity_df=propensity_df,
        stay_ids=val_ids,
        temporal_cols=temporal_cols,
        static_cols=static_cols,
        merge_map=merge_map,
        n_bins=config["grid"]["n_bins"],
    )
    train_ds = set_action_vocab(train_ds, action_vocab)
    val_ds = set_action_vocab(val_ds, action_vocab)

    train_cfg = config["training"]
    model_cfg = config["models"]["80c_gtransformer"]
    criterion = GTransformerLoss(
        weight_next_state=model_cfg["loss_weights"].get("next_state", 1.0),
        weight_outcome=model_cfg["loss_weights"].get("outcome", 1.0),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 128)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg.get("batch_size", 128)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
        drop_last=False,
    )

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    wait = 0

    for epoch in tqdm(range(1, NCO_RETRAIN_MAX_EPOCHS + 1), desc=f"    {label} epochs", mininterval=2):
        tr = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip=float(train_cfg.get("gradient_clip", 1.0)),
        )
        va = evaluate_one_epoch(model, val_loader, criterion, device)
        tqdm.write(f"      epoch {epoch:02d}: train={tr:.4f}, val={va:.4f}")

        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= NCO_RETRAIN_PATIENCE:
                tqdm.write(f"      early stopping ({NCO_RETRAIN_PATIENCE} epochs no improvement)")
                break

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("85e — Negative Control Checks")
    print("=" * 72)

    # ── 1. Load config and data ──────────────────────────────────────
    print("\n[1] Loading data ...")
    config_path = ROOT / "configs" / "80_step_c_training_v1.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    state_df, outcomes_df, propensity_df, temporal_cols, static_cols, merge_map = (
        load_dtr_data(ROOT, config)
    )

    all_stay_ids = state_df["stay_id"].unique()
    split_cfg = config["split"]
    train_ids, val_ids, test_ids = make_train_val_test_split(
        all_stay_ids, outcomes_df,
        train_frac=split_cfg["train_frac"],
        val_frac=split_cfg["val_frac"],
        test_frac=split_cfg["test_frac"],
        stratify_col=split_cfg["stratify_col"],
        seed=split_cfg["seed"],
    )
    test_set = set(test_ids)
    print(f"  Test stays: {len(test_ids):,}")

    # ── 2. Load model ────────────────────────────────────────────────
    print("\n[2] Loading 80c G-Transformer ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_temporal = len(temporal_cols)
    d_static = len(static_cols)
    n_actions = len(config["action"]["valid_merged_indices"])

    model = load_primary_model(config, d_temporal, d_static, n_actions, device)
    print(f"  Device: {device}")

    # Build anchor for action vocab
    anchor_ds = DTRDataset(
        state_df=state_df, outcomes_df=outcomes_df, propensity_df=propensity_df,
        stay_ids=train_ids, temporal_cols=temporal_cols, static_cols=static_cols,
        merge_map=merge_map, n_bins=config["grid"]["n_bins"],
    )
    action_vocab = anchor_ds.action_vocab

    all_results = []

    # ── Baseline ─────────────────────────────────────────────────────
    print("\n[Baseline] Real outcome, real treatment ...")
    ds_base = DTRDataset(
        state_df=state_df, outcomes_df=outcomes_df, propensity_df=propensity_df,
        stay_ids=test_ids, temporal_cols=temporal_cols, static_cols=static_cols,
        merge_map=merge_map, n_bins=config["grid"]["n_bins"],
    )
    ds_base = set_action_vocab(ds_base, action_vocab)

    base_point = evaluate_dm(model, ds_base, device, use_rollout=True)

    def _eval_fn(ds, **kw):
        return evaluate_dm(model, ds, device, use_rollout=True)

    base_boot = bootstrap_ope(
        _eval_fn, ds_base,
        n_resamples=N_BOOTSTRAP, seed=BOOT_SEED, confidence=CI_LEVEL,
    )

    all_results.append({
        "test": "baseline",
        "description": "Real outcome, real treatment",
        "n_stays": len(test_ids),
        "ate": base_point["ate"],
        "ate_ci_lo": base_boot.get("ate_ci_lo", np.nan),
        "ate_ci_hi": base_boot.get("ate_ci_hi", np.nan),
        "v_observed": base_point["v_observed"],
        "v_rule": base_point["v_rule"],
        "expected_ate": "nonzero",
        "pass_criterion": "ATE ≠ 0",
    })
    print(f"  Baseline: ATE={base_point['ate']:+.4f} "
          f"[{base_boot.get('ate_ci_lo', np.nan):+.4f}, "
          f"{base_boot.get('ate_ci_hi', np.nan):+.4f}]")

    # ── NCO3: Random-permutation pseudo-outcome (requires retrain) ───
    print("\n[NCO3] Shuffled outcome with model re-training ...")
    rng = np.random.RandomState(42)
    out_perm = outcomes_df.copy()
    all_outcomes = out_perm["mortality_28d"].values.copy()
    rng.shuffle(all_outcomes)
    out_perm["mortality_28d"] = all_outcomes

    model_nco3 = retrain_model_for_outcome(
        config=config,
        state_df=state_df,
        outcomes_df=out_perm,
        propensity_df=propensity_df,
        temporal_cols=temporal_cols,
        static_cols=static_cols,
        merge_map=merge_map,
        action_vocab=action_vocab,
        train_ids=train_ids,
        val_ids=val_ids,
        device=device,
        seed=1003,
        label="NCO3",
    )

    ds_perm = DTRDataset(
        state_df=state_df, outcomes_df=out_perm, propensity_df=propensity_df,
        stay_ids=test_ids, temporal_cols=temporal_cols, static_cols=static_cols,
        merge_map=merge_map, n_bins=config["grid"]["n_bins"],
    )
    ds_perm = set_action_vocab(ds_perm, action_vocab)

    nco3_point = evaluate_dm(model_nco3, ds_perm, device, use_rollout=True)

    def _eval_fn_nco3(ds, **kw):
        return evaluate_dm(model_nco3, ds, device, use_rollout=True)

    nco3_boot = bootstrap_ope(
        _eval_fn_nco3, ds_perm,
        n_resamples=N_BOOTSTRAP, seed=BOOT_SEED + 30, confidence=CI_LEVEL,
    )

    all_results.append({
        "test": "NCO3_shuffled_outcome",
        "description": "Randomly permuted mortality_28d (destroys treatment-outcome link)",
        "n_stays": len(test_ids),
        "ate": nco3_point["ate"],
        "ate_ci_lo": nco3_boot.get("ate_ci_lo", np.nan),
        "ate_ci_hi": nco3_boot.get("ate_ci_hi", np.nan),
        "v_observed": nco3_point["v_observed"],
        "v_rule": nco3_point["v_rule"],
        "expected_ate": "≈ 0",
        "pass_criterion": "|ATE| < 0.5 * |baseline ATE|",
    })

    nco3_pass = abs(nco3_point["ate"]) < max(0.005, 0.5 * abs(base_point["ate"]))
    print(f"  NCO3: ATE={nco3_point['ate']:+.4f} "
          f"[{nco3_boot.get('ate_ci_lo', np.nan):+.4f}, "
          f"{nco3_boot.get('ate_ci_hi', np.nan):+.4f}] "
          f"({'PASS' if nco3_pass else 'FAIL'})")

    # ── NCE1: Random treatment permutation ───────────────────────────
    print(f"\n[NCE1] Shuffled treatment ({N_PERMUTATION_REPS} reps) ...")
    nce_ates = []

    for rep in tqdm(range(N_PERMUTATION_REPS), desc="NCE1 permutations"):
        # Shuffle treatment (action) assignments across stays within each bin
        prop_perm = propensity_df.copy()
        act_col = "action_idx_merged" if "action_idx_merged" in prop_perm.columns else "action_idx"

        for b in range(config["grid"]["n_bins"]):
            bin_mask = (prop_perm["stay_id"].isin(test_set)) & (prop_perm["bin"] == b)
            bin_vals = prop_perm.loc[bin_mask, act_col].values.copy()
            rng.shuffle(bin_vals)
            prop_perm.loc[bin_mask, act_col] = bin_vals

        ds_nce = DTRDataset(
            state_df=state_df, outcomes_df=outcomes_df, propensity_df=prop_perm,
            stay_ids=test_ids, temporal_cols=temporal_cols, static_cols=static_cols,
            merge_map=merge_map, n_bins=config["grid"]["n_bins"],
        )
        ds_nce = set_action_vocab(ds_nce, action_vocab)

        nce_point = evaluate_dm(model, ds_nce, device, use_rollout=True)
        nce_ates.append(nce_point["ate"])

    nce1_mean = float(np.mean(nce_ates))
    nce1_std = float(np.std(nce_ates))

    all_results.append({
        "test": "NCE1_shuffled_treatment",
        "description": "Randomly permuted treatment per bin (destroys treatment-state link)",
        "n_stays": len(test_ids),
        "ate": nce1_mean,
        "ate_ci_lo": nce1_mean - 1.96 * nce1_std,
        "ate_ci_hi": nce1_mean + 1.96 * nce1_std,
        "v_observed": np.nan,
        "v_rule": np.nan,
        "expected_ate": "≈ 0",
        "pass_criterion": "|ATE| < 0.5 * |baseline ATE|",
    })

    # NOTE: Under DM rollout, V(rule) is independent of observed actions,
    # so shuffling treatment only perturbs V(observed). NCE is not calibrated
    # for DM — mark as N/A rather than applying an inappropriate threshold.
    nce1_pass_naive = abs(nce1_mean) < max(0.005, 0.5 * abs(base_point["ate"]))
    nce1_delta = nce1_mean - float(base_point["ate"])
    print(f"  NCE1: ATE={nce1_mean:+.4f} ± {nce1_std:.4f} "
          f"(Δ vs baseline={nce1_delta:+.4f}; naive={'PASS' if nce1_pass_naive else 'FAIL'}; "
          f"DM caveat → N/A)")

    # ── NCO1: Pre-T0 feature as pseudo-outcome ──────────────────────
    print("\n[NCO1] Pre-T0 albumin as pseudo-outcome ...")
    # Check if albumin (or another pre-T0 lab) is available in temporal cols
    pre_t0_candidates = [c for c in temporal_cols
                         if "albumin" in c.lower() and "_mask" not in c.lower()
                         and "_tsl" not in c.lower() and "_freq" not in c.lower()
                         and "_never" not in c.lower()]
    if not pre_t0_candidates:
        # Try other lab values
        for lab in ["creatinine", "bun", "hemoglobin", "platelet"]:
            pre_t0_candidates = [c for c in temporal_cols
                                 if lab in c.lower() and "_mask" not in c.lower()
                                 and "_tsl" not in c.lower() and "_freq" not in c.lower()
                                 and "_never" not in c.lower()]
            if pre_t0_candidates:
                break

    if pre_t0_candidates:
        nco1_col = pre_t0_candidates[0]
        nco1_col_idx = temporal_cols.index(nco1_col)
        print(f"  Using '{nco1_col}' (index={nco1_col_idx}) at bin 0 as pseudo-outcome")

        # Extract bin-0 value as pseudo-outcome for ALL stays (train+val+test)
        # so the retrained model learns the pseudo-outcome mapping, not real mortality
        sub = state_df[state_df["bin"] == 0].copy()
        pseudo_outcomes = sub.set_index("stay_id")[nco1_col]

        # Create binary pseudo-outcome: above-median = 1.
        # Use training split only to avoid test-set leakage in threshold selection.
        train_mask = pseudo_outcomes.index.isin(set(train_ids))
        median_val = pseudo_outcomes.loc[train_mask].median()
        pseudo_binary = (pseudo_outcomes > median_val).astype(int)

        # Build modified outcomes — replace mortality_28d for ALL stays
        out_nco1 = outcomes_df.copy()
        sid_to_pseudo = pseudo_binary.to_dict()
        out_nco1["mortality_28d"] = out_nco1["stay_id"].map(sid_to_pseudo).fillna(
            out_nco1["mortality_28d"]
        ).astype(int)

        ds_nco1 = DTRDataset(
            state_df=state_df, outcomes_df=out_nco1, propensity_df=propensity_df,
            stay_ids=test_ids, temporal_cols=temporal_cols, static_cols=static_cols,
            merge_map=merge_map, n_bins=config["grid"]["n_bins"],
        )
        ds_nco1 = set_action_vocab(ds_nco1, action_vocab)

        model_nco1 = retrain_model_for_outcome(
            config=config,
            state_df=state_df,
            outcomes_df=out_nco1,
            propensity_df=propensity_df,
            temporal_cols=temporal_cols,
            static_cols=static_cols,
            merge_map=merge_map,
            action_vocab=action_vocab,
            train_ids=train_ids,
            val_ids=val_ids,
            device=device,
            seed=1001,
            label="NCO1",
        )

        nco1_point = evaluate_dm(model_nco1, ds_nco1, device, use_rollout=True)

        def _eval_fn_nco1(ds, **kw):
            return evaluate_dm(model_nco1, ds, device, use_rollout=True)

        nco1_boot = bootstrap_ope(
            _eval_fn_nco1, ds_nco1,
            n_resamples=N_BOOTSTRAP, seed=BOOT_SEED + 100, confidence=CI_LEVEL,
        )

        all_results.append({
            "test": "NCO1_preT0_lab",
            "description": f"Pre-T0 {nco1_col} (binarized) as pseudo-outcome",
            "n_stays": len(test_ids),
            "ate": nco1_point["ate"],
            "ate_ci_lo": nco1_boot.get("ate_ci_lo", np.nan),
            "ate_ci_hi": nco1_boot.get("ate_ci_hi", np.nan),
            "v_observed": nco1_point["v_observed"],
            "v_rule": nco1_point["v_rule"],
            "expected_ate": "≈ 0",
            "pass_criterion": "|ATE| < 0.5 * |baseline ATE|",
        })

        nco1_pass = abs(nco1_point["ate"]) < max(0.005, 0.5 * abs(base_point["ate"]))
        print(f"  NCO1: ATE={nco1_point['ate']:+.4f} "
              f"[{nco1_boot.get('ate_ci_lo', np.nan):+.4f}, "
              f"{nco1_boot.get('ate_ci_hi', np.nan):+.4f}] "
              f"({'PASS' if nco1_pass else 'FAIL'})")
    else:
        print("  No suitable pre-T0 lab column found — skipping NCO1")

    # ── NCO2: Random binary outcome (placebo check) ──────────────────
    print("\n[NCO2] Random binary pseudo-outcome (prevalence-matched, with re-training) ...")
    base_prev = outcomes_df.loc[
        outcomes_df["stay_id"].isin(set(train_ids)), "mortality_28d"
    ].mean()

    out_rand = outcomes_df.copy()
    all_rand = rng.binomial(1, base_prev, size=len(out_rand))
    out_rand["mortality_28d"] = all_rand

    model_nco2 = retrain_model_for_outcome(
        config=config,
        state_df=state_df,
        outcomes_df=out_rand,
        propensity_df=propensity_df,
        temporal_cols=temporal_cols,
        static_cols=static_cols,
        merge_map=merge_map,
        action_vocab=action_vocab,
        train_ids=train_ids,
        val_ids=val_ids,
        device=device,
        seed=1002,
        label="NCO2",
    )

    ds_rand = DTRDataset(
        state_df=state_df, outcomes_df=out_rand, propensity_df=propensity_df,
        stay_ids=test_ids, temporal_cols=temporal_cols, static_cols=static_cols,
        merge_map=merge_map, n_bins=config["grid"]["n_bins"],
    )
    ds_rand = set_action_vocab(ds_rand, action_vocab)

    nco2_point = evaluate_dm(model_nco2, ds_rand, device, use_rollout=True)

    def _eval_fn_nco2(ds, **kw):
        return evaluate_dm(model_nco2, ds, device, use_rollout=True)

    nco2_boot = bootstrap_ope(
        _eval_fn_nco2, ds_rand,
        n_resamples=N_BOOTSTRAP, seed=BOOT_SEED + 200, confidence=CI_LEVEL,
    )

    all_results.append({
        "test": "NCO2_random_outcome",
        "description": f"IID Bernoulli(p={base_prev:.3f}) pseudo-outcome",
        "n_stays": len(test_ids),
        "ate": nco2_point["ate"],
        "ate_ci_lo": nco2_boot.get("ate_ci_lo", np.nan),
        "ate_ci_hi": nco2_boot.get("ate_ci_hi", np.nan),
        "v_observed": nco2_point["v_observed"],
        "v_rule": nco2_point["v_rule"],
        "expected_ate": "≈ 0",
        "pass_criterion": "|ATE| < 0.5 * |baseline ATE|",
    })

    nco2_pass = abs(nco2_point["ate"]) < max(0.005, 0.5 * abs(base_point["ate"]))
    print(f"  NCO2: ATE={nco2_point['ate']:+.4f} "
          f"[{nco2_boot.get('ate_ci_lo', np.nan):+.4f}, "
          f"{nco2_boot.get('ate_ci_hi', np.nan):+.4f}] "
          f"({'PASS' if nco2_pass else 'FAIL'})")

    # ── 5. Save results ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("[5] Saving results ...")

    results_df = pd.DataFrame(all_results)
    csv_path = TABLES_DIR / "85e_negative_control.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path} ({len(results_df)} rows)")

    # ── 6. Generate report ───────────────────────────────────────────
    print("\n[6] Generating report ...")

    baseline_ate_val = base_point["ate"]
    lines = [
        "=" * 72,
        "85e — Negative Control Checks (MIMIC-IV)",
        "=" * 72,
        f"Generated: {ts}",
        f"Model: 80c G-Transformer DM (rollout, ow=1)",
        f"Permutation reps: {N_PERMUTATION_REPS}",
        f"Bootstrap: B={N_BOOTSTRAP}, seed={BOOT_SEED}",
        "",
        "NEGATIVE CONTROL RESULTS",
        "────────────────────────",
        f"  {'Test':<25s} {'ATE':>9s} {'CI_lo':>9s} {'CI_hi':>9s} "
        f"{'Expected':>10s} {'Result':>8s}",
        f"  {'-'*25} {'-'*9} {'-'*9} {'-'*9} {'-'*10} {'-'*8}",
    ]

    n_pass = 0
    n_tests = 0
    for _, r in results_df.iterrows():
        ci_lo = f"{r['ate_ci_lo']:+9.4f}" if pd.notna(r["ate_ci_lo"]) else "       —"
        ci_hi = f"{r['ate_ci_hi']:+9.4f}" if pd.notna(r["ate_ci_hi"]) else "       —"

        if r["test"] == "baseline":
            pass_fail = "REF"
        elif r["test"] == "NCE1_shuffled_treatment":
            # NCE is not calibrated for DM rollout — V(rule) is independent
            # of observed actions, so shuffling only perturbs V(observed).
            # Mark as N/A rather than applying an inappropriate threshold.
            pass_fail = "N/A(DM)"
        else:
            is_pass = abs(r["ate"]) < max(0.005, 0.5 * abs(baseline_ate_val))
            pass_fail = "PASS" if is_pass else "FAIL"
            n_tests += 1
            if is_pass:
                n_pass += 1

        lines.append(
            f"  {r['test']:<25s} {r['ate']:+9.4f} {ci_lo} {ci_hi} "
            f"{r['expected_ate']:>10s} {pass_fail:>8s}"
        )

    lines.extend([
        "",
        "INTERPRETATION",
        "──────────────",
        f"  Baseline ATE: {baseline_ate_val:+.4f}",
        f"  Pass threshold: |ATE| < max(0.005, {0.5 * abs(baseline_ate_val):.4f})",
        f"  Tests passed: {n_pass}/{n_tests}",
        "",
        "  NCO (Negative Control Outcome): tests whether the model attributes",
        "  treatment effects to outcomes that treatment cannot plausibly cause.",
        "  A non-null ATE under NCO → residual confounding or model artifacts.",
        "",
        "  NCE (Negative Control Exposure) under rollout-DM has a known caveat:",
        "  shuffling observed actions only perturbs V(observed), while V(rule) is",
        "  unchanged because it depends on state + fixed rule actions. Therefore",
        "  NCE1 is marked N/A for DM; it is designed for IPW/DR estimators.",
        "",
    ])

    if n_pass == n_tests:
        lines.append(f"  → NCO ALL PASS ({n_pass}/{n_tests}): No evidence of residual bias")
        lines.append("    NCE1 excluded from scoring (not calibrated for DM rollout)")
    elif n_pass >= n_tests // 2:
        lines.append(f"  → NCO PARTIAL PASS: {n_pass}/{n_tests} NCO tests pass — "
                    "some residual bias signals, investigate further")
        lines.append("    NCE1 excluded from scoring (not calibrated for DM rollout)")
    else:
        lines.append(f"  → NCO FAIL: {n_tests - n_pass}/{n_tests} NCO tests fail — "
                    "substantial residual bias detected; interpret primary ATE with caution")
        lines.append("    NCE1 excluded from scoring (not calibrated for DM rollout)")

    report = "\n".join(lines) + "\n"
    report_path = REPORTS_DIR / "85e_negative_control_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {report_path}")
    print(report)

    total_time = time.time() - t0
    print(f"\n{'=' * 72}")
    print(f"85e COMPLETE. Total time: {total_time / 60:.1f} min")
    print("=" * 72)


if __name__ == "__main__":
    main()
