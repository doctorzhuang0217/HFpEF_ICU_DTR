#!/usr/bin/env python3
"""80c — Train G-Transformer (primary deep DTR model)

Protocol: v3.1, Step 80c
Compute: G1 (single H100 GPU)
Role: Primary deep pair with 75a (dynamic-MSM / IPTW)

Pipeline:
  1. Load data (45c states, 45d outcomes, 70b propensities, 70c rule policy)
  2. Build DTR train/val/test datasets
  3. Train G-Transformer with early stopping
  4. Evaluate: DM, DR, FQE on test set
  5. Subgroup analysis (phenotype, HF subtype)
  6. Bootstrap CIs
  7. Save checkpoint, OPE tables, summary report
"""

from __future__ import annotations

import argparse
import datetime
import json
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
from hfpef_dtr.models.gtransformer import GTransformer, GTransformerLoss, build_gtransformer_from_config
from hfpef_dtr.models.ope import (
    evaluate_dm,
    evaluate_dr,
    evaluate_fqe,
    train_fqe,
    bootstrap_ope,
    bootstrap_fqe_eval_only,
)
from hfpef_dtr.models.step_c_reporting import (
    save_policy_values,
    save_bootstrap_cis,
    append_bootstrap_cis,
    save_subgroup_results,
    save_trimming_sensitivity,
    generate_summary_report,
    save_structured_yaml,
)

# ── Paths ──
CONFIG_PATH = ROOT / "configs" / "80_step_c_training_v1.yaml"
# Sensitivity (non-default) scheme configs. Selected ONLY via --scheme; the
# default coarse_3 path above is never altered.
SCHEME_CONFIG_PATHS = {
    "coarse_3": CONFIG_PATH,
    "medium_4": ROOT / "configs" / "80_step_c_training_medium4_v1.yaml",
}
MODEL_ID = "80c_gtransformer"
TABLES_DIR = ROOT / "tables"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = ROOT / "figures"
MODELS_DIR = ROOT / "models"

for d in [TABLES_DIR, REPORTS_DIR, FIGURES_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path) as f:
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
    """Set all RNG seeds for reproducible train/val/test behavior."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: GTransformer,
    loader: DataLoader,
    criterion: GTransformerLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> dict:
    model.train()
    total_loss = 0.0
    total_ns = 0.0
    total_out = 0.0
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
            "temporal": temporal,
            "valid_mask": valid_mask,
            "absorbing_mask": absorbing_mask,
            "mortality_28d": batch["mortality_28d"].to(device),
        })

        optimizer.zero_grad()
        losses["loss"].backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += losses["loss"].item()
        total_ns += losses["loss_next_state"].item()
        total_out += losses["loss_outcome"].item()
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss/n_batches:.2f}")

    return {
        "loss": total_loss / max(n_batches, 1),
        "loss_next_state": total_ns / max(n_batches, 1),
        "loss_outcome": total_out / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate_epoch(
    model: GTransformer,
    loader: DataLoader,
    criterion: GTransformerLoss,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_ns = 0.0
    total_out = 0.0
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
            "temporal": temporal,
            "valid_mask": valid_mask,
            "absorbing_mask": absorbing_mask,
            "mortality_28d": batch["mortality_28d"].to(device),
        })

        total_loss += losses["loss"].item()
        total_ns += losses["loss_next_state"].item()
        total_out += losses["loss_outcome"].item()
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "loss_next_state": total_ns / max(n_batches, 1),
        "loss_outcome": total_out / max(n_batches, 1),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="80c G-Transformer training")
    p.add_argument("--scheme", choices=["coarse_3", "medium_4"], default="coarse_3",
                   help="Action-discretization scheme. DEFAULT coarse_3 uses the "
                        "frozen config + output paths (byte-for-byte unchanged); "
                        "medium_4 loads the medium_4 config and writes "
                        "80c_gtransformer_medium4_* artifacts.")
    p.add_argument("--ack-rule-policy-gap", action="store_true", default=False,
                   help="REQUIRED for --scheme medium_4: the medium_4 rule-policy "
                        "action is now RE-ENCODED into the 32-cell vocab by "
                        "70b --scheme medium_4 (wave-2 §F), so the medium_4 OPE is "
                        "VALID. The flag is retained as an explicit opt-in to the "
                        "non-default finer-granularity sensitivity chain; ensure "
                        "70b --scheme medium_4 has been run first.")
    p.add_argument("--outcome-weight", type=float, default=None,
                   help="Override loss_weights.outcome (sensitivity: 5.0, 10.0)")
    p.add_argument("--use-rollout", action="store_true",
                   help="Use autoregressive rollout for DM evaluation only (not training loss)")
    p.add_argument("--dr-bootstrap", action="store_true",
                   help="Add DR bootstrap CI (B=500)")
    p.add_argument("--fqe-bootstrap", action="store_true",
                   help="Add FQE eval-only bootstrap CI (B=500); appends FQE row "
                        "to {model_id}_bootstrap_ci.csv. Additive, default OFF.")
    p.add_argument("--tag", type=str, default=None,
                   help="Suffix for output files (e.g. 'ow5' → 80c_gtransformer_ow5)")
    return p.parse_args()


def infer_outcome_weight_from_tag(tag: str | None) -> float | None:
    """Infer outcome weight from tag pattern (e.g., rollout_ow10 -> 10.0)."""
    if not tag:
        return None
    m = re.search(r"(?:^|_)ow(\d+(?:\.\d+)?)$", tag)
    if m is None:
        return None
    return float(m.group(1))


def main():
    args = parse_args()

    inferred_weight = infer_outcome_weight_from_tag(args.tag)
    if args.outcome_weight is None and inferred_weight is not None:
        args.outcome_weight = inferred_weight
        print(f"  [auto] inferred --outcome-weight={args.outcome_weight} from --tag='{args.tag}'")

    if args.outcome_weight is not None and inferred_weight is not None:
        if not np.isclose(args.outcome_weight, inferred_weight):
            raise ValueError(
                f"Inconsistent arguments: --outcome-weight={args.outcome_weight} "
                f"but --tag='{args.tag}' implies ow{inferred_weight}. "
                "Please align CLI arguments."
            )

    # ── Resolve action-discretization scheme (default coarse_3 = frozen) ──
    scheme = getattr(args, "scheme", "coarse_3")
    if scheme == "medium_4" and not getattr(args, "ack_rule_policy_gap", False):
        raise SystemExit(
            "ABORT (80c): --scheme medium_4 is the non-default finer-granularity "
            "sensitivity chain. The medium_4 rule-policy action re-encoding IS now "
            "implemented (wave-2 §F, in 70b --scheme medium_4), so the DM-rollout "
            "OPE is VALID once 70b --scheme medium_4 has been run. Opt in "
            "explicitly with --ack-rule-policy-gap."
        )
    config_path = SCHEME_CONFIG_PATHS[scheme]
    scheme_tag = None if scheme == "coarse_3" else scheme.replace("_", "")

    # ── Resolve output tag ──
    # For coarse_3 the tag logic is byte-for-byte the original. For a non-default
    # scheme the scheme tag is composed FIRST so outputs land on *_medium4 paths
    # and never collide with the frozen coarse_3 artifacts.
    tag = args.tag
    if tag is None and args.outcome_weight is not None:
        tag = f"ow{args.outcome_weight:g}"
    if scheme_tag is not None:
        tag = scheme_tag if tag is None else f"{scheme_tag}_{tag}"
    model_id = f"{MODEL_ID}_{tag}" if tag else MODEL_ID

    print("=" * 72)
    print(f"80c — G-Transformer Training (MIMIC-IV){f'  [sens: {tag}]' if tag else ''}")
    print("=" * 72)
    t0 = time.time()

    # ── 1. Config & device ──
    print("[1] Loading config...")
    config = load_config(config_path)

    # For a non-default scheme, sync the action vocab from the surviving merged
    # indices recorded by 70b (--scheme medium_4). This only mutates the in-memory
    # medium_4 config; coarse_3 is never touched. Training auto-discovers vocab
    # from data, but subgroup OPE reads config["action"]["valid_merged_indices"].
    if scheme != "coarse_3":
        mm_path = ROOT / config["data"]["merge_map"]
        if mm_path.exists():
            with open(mm_path) as f:
                mm_cfg = yaml.safe_load(f) or {}
            vmi = mm_cfg.get("valid_merged_indices")
            if vmi:
                config["action"]["valid_merged_indices"] = sorted(int(a) for a in vmi)
                print(f"  [scheme={scheme}] valid_merged_indices synced from "
                      f"{mm_path.name}: {config['action']['valid_merged_indices']}")
        else:
            print(f"  [scheme={scheme}] WARNING: merge map {mm_path} not found; "
                  f"using config placeholder valid_merged_indices.")

    model_cfg = config["models"][MODEL_ID]
    train_cfg = config["training"]

    # Apply CLI overrides
    if args.outcome_weight is not None:
        model_cfg["loss_weights"]["outcome"] = args.outcome_weight
        print(f"  [override] outcome_weight = {args.outcome_weight}")

    set_seed(train_cfg.get("seed", 42))
    device = get_device()

    # ── 2. Build datasets ──
    print("[2] Building DTR datasets...")
    train_ds, val_ds, test_ds = build_dtr_datasets(ROOT, config)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=train_cfg.get("pin_memory", True),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=train_cfg.get("pin_memory", True),
    )
    # ── 3. Build model ──
    print("[3] Building G-Transformer...")
    model = build_gtransformer_from_config(
        d_temporal=train_ds.d_temporal,
        d_static=train_ds.d_static,
        n_actions=train_ds.n_actions,
        model_cfg=model_cfg,
        grid_cfg=config["grid"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  d_temporal={train_ds.d_temporal}, d_static={train_ds.d_static}, n_actions={train_ds.n_actions}")

    criterion = GTransformerLoss(
        weight_next_state=model_cfg["loss_weights"]["next_state"],
        weight_outcome=model_cfg["loss_weights"]["outcome"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )

    # Cosine scheduler with warmup
    warmup_epochs = train_cfg["scheduler"].get("warmup_epochs", 10)
    max_epochs = train_cfg["max_epochs"]

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── 4. Training loop ──
    print("[4] Training G-Transformer...")
    patience = train_cfg["early_stopping"]["patience"]
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    train_history = []

    for epoch in range(1, max_epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip=train_cfg.get("gradient_clip", 1.0),
        )
        val_metrics = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        train_history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "lr": optimizer.param_groups[0]["lr"],
        })

        # Early stopping check
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0
            # Save best checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": config,
            }, MODELS_DIR / f"{model_id}_best.pt")
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1 or patience_counter == 0:
            print(
                f"  Epoch {epoch:3d}/{max_epochs}: "
                f"train={train_metrics['loss']:.4f} "
                f"(ns={train_metrics['loss_next_state']:.4f}, out={train_metrics['loss_outcome']:.4f}), "
                f"val={val_metrics['loss']:.4f}, "
                f"lr={optimizer.param_groups[0]['lr']:.6f}"
                f"{' *best*' if patience_counter == 0 else ''}"
            )

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch} (best={best_epoch})")
            break

    # Load best checkpoint
    ckpt = torch.load(MODELS_DIR / f"{model_id}_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded best model from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")

    # Save training history
    pd.DataFrame(train_history).to_csv(TABLES_DIR / f"{model_id}_training_history.csv", index=False)

    train_time = time.time() - t0
    print(f"  Training time: {train_time / 60:.1f} min")

    # ── 5. OPE evaluation on test set ──
    print("[5] OPE evaluation on test set...")

    ope_results = {}
    q_net = None  # populated if FQE is evaluated; reused by optional --fqe-bootstrap

    methods = config["ope"]["methods_by_model"].get(MODEL_ID, [])

    use_rollout = getattr(args, "use_rollout", False)
    if use_rollout:
        print("  ** Autoregressive rollout enabled for DM **")

    if "DM" in methods:
        print("  [5a] Direct Method (DM)...")
        dm_result = evaluate_dm(model, test_ds, device, use_rollout=use_rollout)
        ope_results["DM"] = dm_result
        print(f"    V(obs)={dm_result['v_observed']:.4f}, V(rule)={dm_result['v_rule']:.4f}, ATE={dm_result['ate']:+.4f}")

    if "DR" in methods:
        print("  [5b] Doubly-Robust (DR)...")
        dr_result = evaluate_dr(model, test_ds, device)
        ope_results["DR"] = dr_result
        print(f"    V(obs)={dr_result['v_observed']:.4f}, V(rule)={dr_result['v_rule']:.4f}, ATE={dr_result['ate']:+.4f}, ESS={dr_result['ess']:.0f}")

    if "FQE" in methods:
        print("  [5c] Fitted Q-Evaluation (FQE)...")
        fqe_cfg = config["ope"]["fqe"]
        q_net = train_fqe(
            test_ds, model, device, fqe_cfg,
            gamma=config["reward"]["gamma"],
        )
        fqe_result = evaluate_fqe(q_net, test_ds, device)
        ope_results["FQE"] = fqe_result
        print(f"    V(obs)={fqe_result['v_observed']:.4f}, V(rule)={fqe_result['v_rule']:.4f}, ATE={fqe_result['ate']:+.4f}")

    # Save OPE results
    save_policy_values(ope_results, model_id, TABLES_DIR)

    # ── 6. Subgroup analysis ──
    print("[6] Subgroup analysis...")
    subgroup_results = run_subgroup_analysis(model, config, device, model_id, use_rollout=use_rollout)
    save_subgroup_results(subgroup_results, model_id, TABLES_DIR)
    for sg in subgroup_results:
        print(f"  {sg['stratifier']}={sg['group']}: n={sg['n_stays']}, ATE(DM)={sg['ate']:+.4f}")

    # ── 7. Bootstrap CIs ──
    boot_estimators = ["DM"]
    if args.dr_bootstrap:
        boot_estimators.append("DR")
    print(f"[7] Bootstrap CIs ({'+'.join(boot_estimators)}, B=500)...")
    boot_results = {}

    n_boot = config["ope"]["bootstrap"]["n_resamples"]
    boot_seed = config["ope"]["bootstrap"]["seed"]

    if "DM" in ope_results and "DM" in boot_estimators:
        def dm_eval_fn(ds, **kwargs):
            return evaluate_dm(model, ds, device, use_rollout=use_rollout)
        boot_dm = bootstrap_ope(dm_eval_fn, test_ds, n_resamples=n_boot, seed=boot_seed)
        boot_results["DM"] = boot_dm
        if boot_dm.get("n_success", 0) > 0:
            print(f"  DM: ATE={boot_dm['ate_mean']:+.4f} [{boot_dm['ate_ci_lo']:.4f}, {boot_dm['ate_ci_hi']:.4f}] (B={boot_dm['n_success']})")

    if "DR" in ope_results and "DR" in boot_estimators:
        def dr_eval_fn(ds, **kwargs):
            return evaluate_dr(model, ds, device)
        boot_dr = bootstrap_ope(dr_eval_fn, test_ds, n_resamples=n_boot, seed=boot_seed + 1)
        boot_results["DR"] = boot_dr
        if boot_dr.get("n_success", 0) > 0:
            print(f"  DR: ATE={boot_dr['ate_mean']:+.4f} [{boot_dr['ate_ci_lo']:.4f}, {boot_dr['ate_ci_hi']:.4f}] (B={boot_dr['n_success']})")

    save_bootstrap_cis(boot_results, model_id, TABLES_DIR)

    # ── 7b. Optional FQE eval-only bootstrap (additive, default OFF) ──
    # When --fqe-bootstrap is given, append an FQE row (same schema as DM/DR
    # rows) to the existing {model_id}_bootstrap_ci.csv. Does not alter the
    # default (non-flag) bootstrap output.
    if args.fqe_bootstrap:
        if "FQE" in ope_results and q_net is not None:
            print("[7b] FQE eval-only bootstrap (B=500)...")
            boot_fqe = bootstrap_fqe_eval_only(
                q_net, test_ds, device, n_resamples=500, seed=7777 + 2,
            )
            append_bootstrap_cis({"FQE": boot_fqe}, model_id, TABLES_DIR)
            if boot_fqe.get("n_success", 0) > 0:
                print(f"  FQE: ATE={boot_fqe['ate_mean']:+.4f} "
                      f"[{boot_fqe['ate_ci_lo']:.4f}, {boot_fqe['ate_ci_hi']:.4f}] "
                      f"(B={boot_fqe['n_success']})")
        else:
            print("[7b] --fqe-bootstrap requested but FQE was not evaluated; skipping.")

    # ── 8. Summary report ──
    print("[8] Generating summary report...")
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
    }

    generate_summary_report(
        model_id, "G-Transformer",
        ope_results, boot_results,
        subgroup_results, training_info,
        REPORTS_DIR,
    )
    save_structured_yaml(model_id, ope_results, boot_results, training_info, TABLES_DIR)

    total_time = time.time() - t0
    print()
    print("=" * 72)
    print(f"80c COMPLETE ({model_id}). Total time: {total_time / 60:.1f} min")
    print("=" * 72)


def run_subgroup_analysis(
    model: GTransformer,
    config: dict,
    device: torch.device,
    model_id: str = MODEL_ID,
    use_rollout: bool = False,
) -> list[dict]:
    """Run DM evaluation on each pre-specified subgroup."""
    # Load phenotype labels
    pheno_path = ROOT / config["data"]["phenotype_labels"]
    pheno_df = pd.read_parquet(pheno_path)

    # Load full data for subgroup splits
    state_df, outcomes_df, propensity_df, temporal_cols, static_cols, merge_map = (
        load_dtr_data(ROOT, config)
    )

    # Get test set stay IDs (same split as training)
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

    # Merge phenotype labels with test IDs
    pheno_test = pheno_df[pheno_df["stay_id"].isin(set(test_ids))].copy()

    results = []
    stratifiers = config["ope"]["subgroup_stratifiers"]

    for strat in stratifiers:
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
                action_vocab=config["action"]["valid_merged_indices"],
            )

            dm = evaluate_dm(model, sub_ds, device, use_rollout=use_rollout)
            results.append({
                "model": model_id,
                "stratifier": strat,
                "group": str(group_name),
                "n_stays": len(sub_ds),
                "v_observed": dm["v_observed"],
                "v_rule": dm["v_rule"],
                "ate": dm["ate"],
            })

    return results


if __name__ == "__main__":
    main()
