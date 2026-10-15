#!/usr/bin/env python3
"""80d — Train CT-Legacy (Causal Transformer ablation, outcome-only)

Protocol: v3.1, Step 80d
Compute: G1 (single H100 GPU)
Role: Legacy ablation — outcome-only Transformer (no next-state prediction).
      Quantifies the benefit of G-computation in the full G-Transformer (80c).

Key difference from 80c:
  - No next-state prediction head → no G-computation rollout
  - Smaller architecture (d_model=64, n_layers=3, d_ff=128)
  - OPE: FQE only (no DM/DR)
  - Single outcome BCE loss (no multi-task weighting)

Pipeline: train → OPE (FQE) → subgroups → bootstrap → report
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Project root ──
ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(ROOT / "src"))

from hfpef_dtr.models.dtr_dataset import DTRDataset, build_dtr_datasets, load_dtr_data
from hfpef_dtr.models.dataset import discover_feature_columns, make_train_val_test_split
from hfpef_dtr.models.ct_legacy import CTLegacy, CTLegacyLoss, build_ct_legacy_from_config
from hfpef_dtr.models.ope import evaluate_fqe, train_fqe, bootstrap_ope, bootstrap_fqe_eval_only
from hfpef_dtr.models.step_c_reporting import (
    save_policy_values,
    save_bootstrap_cis,
    append_bootstrap_cis,
    save_subgroup_results,
    generate_summary_report,
    save_structured_yaml,
)

CONFIG_PATH = ROOT / "configs" / "80_step_c_training_v1.yaml"
MODEL_ID = "80d_ct_legacy"
TABLES_DIR = ROOT / "tables"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = ROOT / "figures"
MODELS_DIR = ROOT / "models"

for d in [TABLES_DIR, REPORTS_DIR, FIGURES_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("  WARNING: No GPU available, using CPU")
    return device


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="80d CT-Legacy training")
    p.add_argument("--fqe-bootstrap", action="store_true",
                   help="Append an FQE eval-only bootstrap CI row (B=500) to "
                        "{model_id}_bootstrap_ci.csv. Additive, default OFF.")
    p.add_argument("--tag", type=str, default=None,
                   help="Suffix for output files")
    return p.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    n_batches = 0
    pbar = tqdm(loader, desc="  train", leave=False, mininterval=5)
    for batch in pbar:
        temporal = batch["temporal"].to(device)
        static = batch["static"].to(device)
        action = batch["action"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        absorbing_mask = batch["absorbing_mask"].to(device)
        outputs = model(temporal, static, action, valid_mask, absorbing_mask)
        losses = criterion(outputs, {
            "mortality_28d": batch["mortality_28d"].to(device),
        })
        optimizer.zero_grad()
        losses["loss"].backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += losses["loss"].item()
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}")
    return {"loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    pbar = tqdm(loader, desc="  val  ", leave=False, mininterval=5)
    for batch in pbar:
        temporal = batch["temporal"].to(device)
        static = batch["static"].to(device)
        action = batch["action"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        absorbing_mask = batch["absorbing_mask"].to(device)
        outputs = model(temporal, static, action, valid_mask, absorbing_mask)
        losses = criterion(outputs, {
            "mortality_28d": batch["mortality_28d"].to(device),
        })
        total_loss += losses["loss"].item()
        n_batches += 1
    return {"loss": total_loss / max(n_batches, 1)}


def main():
    args = parse_args()
    tag = args.tag
    model_id = f"{MODEL_ID}_{tag}" if tag else MODEL_ID

    print("=" * 72)
    print(f"80d — CT-Legacy Training (MIMIC-IV){f'  [sens: {tag}]' if tag else ''}")
    print("=" * 72)
    t0 = time.time()

    # ── 1. Config ──
    print("[1] Loading config...")
    config = load_config()
    model_cfg = config["models"][MODEL_ID]
    train_cfg = config["training"]
    set_seed(train_cfg.get("seed", 42))
    device = get_device()

    # ── 2. Datasets ──
    print("[2] Building DTR datasets...")
    train_ds, val_ds, test_ds = build_dtr_datasets(ROOT, config)
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=train_cfg.get("pin_memory", True), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=train_cfg.get("pin_memory", True),
    )

    # ── 3. Model ──
    print("[3] Building CT-Legacy...")
    model = build_ct_legacy_from_config(
        d_temporal=train_ds.d_temporal,
        d_static=train_ds.d_static,
        n_actions=train_ds.n_actions,
        model_cfg=model_cfg,
        grid_cfg=config["grid"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Architecture: d_model={model_cfg.get('d_model', 64)}, "
          f"n_layers={model_cfg.get('n_layers', 3)}, "
          f"d_ff={model_cfg.get('d_ff', 128)}")

    criterion = CTLegacyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )
    warmup_epochs = train_cfg["scheduler"].get("warmup_epochs", 10)
    max_epochs = train_cfg["max_epochs"]

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── 4. Training ──
    print("[4] Training CT-Legacy...")
    patience = train_cfg["early_stopping"]["patience"]
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    train_history = []

    for epoch in range(1, max_epochs + 1):
        train_m = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip=train_cfg.get("gradient_clip", 1.0),
        )
        val_m = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        train_history.append({
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "val_loss": val_m["loss"],
            "lr": optimizer.param_groups[0]["lr"],
        })
        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss,
            }, MODELS_DIR / f"{model_id}_best.pt")
        else:
            patience_counter += 1
        if epoch % 5 == 0 or epoch == 1 or patience_counter == 0:
            print(
                f"  Epoch {epoch:3d}/{max_epochs}: "
                f"train={train_m['loss']:.4f}, val={val_m['loss']:.4f}, "
                f"lr={optimizer.param_groups[0]['lr']:.6f}"
                f"{' *best*' if patience_counter == 0 else ''}"
            )
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch} (best={best_epoch})")
            break

    ckpt = torch.load(MODELS_DIR / f"{model_id}_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded best model from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")
    pd.DataFrame(train_history).to_csv(TABLES_DIR / f"{model_id}_training_history.csv", index=False)
    train_time = time.time() - t0
    print(f"  Training time: {train_time / 60:.1f} min")

    # ── 5. OPE (FQE only) ──
    print("[5] OPE evaluation on test set (FQE only)...")
    ope_results = {}

    print("  [5a] FQE...")
    q_net = train_fqe(test_ds, model, device, config["ope"]["fqe"],
                      gamma=config["reward"]["gamma"])
    fqe = evaluate_fqe(q_net, test_ds, device)
    ope_results["FQE"] = fqe
    print(f"    V(obs)={fqe['v_observed']:.4f}, V(rule)={fqe['v_rule']:.4f}, "
          f"ATE={fqe['ate']:+.4f}")

    save_policy_values(ope_results, model_id, TABLES_DIR)

    # ── 6. Subgroups ──
    print("[6] Subgroup analysis...")
    subgroup_results = run_subgroup_analysis(model, config, device, model_id)
    save_subgroup_results(subgroup_results, model_id, TABLES_DIR)
    for sg in subgroup_results:
        print(f"  {sg['stratifier']}={sg['group']}: n={sg['n_stays']}, "
              f"ATE(FQE)={sg['ate']:+.4f}")

    # ── 7. Bootstrap ──
    # NOTE: Use eval-only bootstrap — resamples evaluation with frozen Q-net
    # instead of retraining FQE per sample (>1000x speedup).
    print(f"[7] Bootstrap CIs (FQE eval-only, B={config['ope']['bootstrap']['n_resamples']})...")
    n_boot = config["ope"]["bootstrap"]["n_resamples"]
    boot_seed = config["ope"]["bootstrap"]["seed"]

    boot_fqe = bootstrap_fqe_eval_only(
        q_net, test_ds, device, n_resamples=n_boot, seed=boot_seed + 3)
    boot_results = {"FQE": boot_fqe}
    if boot_fqe.get("n_success", 0) > 0:
        print(f"  FQE: ATE={boot_fqe['ate_mean']:+.4f} "
              f"[{boot_fqe['ate_ci_lo']:.4f}, {boot_fqe['ate_ci_hi']:.4f}] "
              f"(B={boot_fqe['n_success']})")

    save_bootstrap_cis(boot_results, model_id, TABLES_DIR)

    # ── 7b. Optional FQE eval-only bootstrap (additive, default OFF) ──
    # Appends an FQE row (same schema) to the existing {model_id}_bootstrap_ci.csv.
    # Does not alter the default bootstrap output.
    if args.fqe_bootstrap:
        print("[7b] FQE eval-only bootstrap (B=500)...")
        boot_fqe_flag = bootstrap_fqe_eval_only(
            q_net, test_ds, device, n_resamples=500, seed=7777 + 2,
        )
        append_bootstrap_cis({"FQE": boot_fqe_flag}, model_id, TABLES_DIR)
        if boot_fqe_flag.get("n_success", 0) > 0:
            print(f"  FQE: ATE={boot_fqe_flag['ate_mean']:+.4f} "
                  f"[{boot_fqe_flag['ate_ci_lo']:.4f}, {boot_fqe_flag['ate_ci_hi']:.4f}] "
                  f"(B={boot_fqe_flag['n_success']})")

    # ── 8. Report ──
    print("[8] Summary report...")
    training_info = {
        "epochs_total": len(train_history),
        "best_epoch": best_epoch,
        "best_val_loss": f"{best_val_loss:.4f}",
        "n_params": f"{n_params:,}",
        "train_time_min": f"{train_time / 60:.1f}",
        "device": str(device),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "ablation_note": "outcome-only (no next-state), smaller architecture",
    }
    generate_summary_report(model_id, "CT-Legacy", ope_results, boot_results,
                            subgroup_results, training_info, REPORTS_DIR)
    save_structured_yaml(model_id, ope_results, boot_results, training_info, TABLES_DIR)

    total_time = time.time() - t0
    print()
    print("=" * 72)
    print(f"80d COMPLETE ({model_id}). Total time: {total_time / 60:.1f} min")
    print("=" * 72)


def run_subgroup_analysis(model, config, device, model_id=MODEL_ID):
    """Subgroup analysis using FQE estimator."""
    pheno_df = pd.read_parquet(ROOT / config["data"]["phenotype_labels"])
    state_df, outcomes_df, propensity_df, temporal_cols, static_cols, merge_map = (
        load_dtr_data(ROOT, config)
    )
    all_stay_ids = state_df["stay_id"].unique()
    split_cfg = config["split"]
    _, _, test_ids = make_train_val_test_split(
        all_stay_ids, outcomes_df,
        train_frac=split_cfg["train_frac"], val_frac=split_cfg["val_frac"],
        test_frac=split_cfg["test_frac"], stratify_col=split_cfg["stratify_col"],
        seed=split_cfg["seed"],
    )
    pheno_test = pheno_df[pheno_df["stay_id"].isin(set(test_ids))].copy()
    results = []
    for strat in config["ope"]["subgroup_stratifiers"]:
        if strat not in pheno_test.columns:
            continue
        for group_name, group_df in pheno_test.groupby(strat):
            group_ids = group_df["stay_id"].values
            if len(group_ids) < 30:
                continue
            sub_ds = DTRDataset(
                state_df=state_df, outcomes_df=outcomes_df, propensity_df=propensity_df,
                stay_ids=group_ids, temporal_cols=temporal_cols, static_cols=static_cols,
                merge_map=merge_map, n_bins=config["grid"]["n_bins"],
                action_vocab=config["action"]["valid_merged_indices"],
            )
            q = train_fqe(sub_ds, model, device, config["ope"]["fqe"],
                          gamma=config["reward"]["gamma"])
            fqe_result = evaluate_fqe(q, sub_ds, device)
            results.append({
                "model": model_id, "stratifier": strat, "group": str(group_name),
                "n_stays": len(sub_ds), "v_observed": fqe_result["v_observed"],
                "v_rule": fqe_result["v_rule"], "ate": fqe_result["ate"],
            })
    return results


if __name__ == "__main__":
    main()
