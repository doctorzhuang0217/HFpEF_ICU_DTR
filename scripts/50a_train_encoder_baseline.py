#!/usr/bin/env python3
"""
50a_train_encoder_baseline — Train missingness-aware baseline encoder.

Purpose
-------
Self-supervised pretraining of a BiGRU encoder on 45c state tensors.
Pretext tasks: masked reconstruction + next-state prediction.
Produces frozen encoder checkpoint for downstream Step B/C use.

Protocol reference: Step 50a (v3.1 § 11 Step A representation learning)

Inputs
------
- data/interim/45c_state_tensors_v1_mimic.parquet
- data/interim/45d_outcomes_absorb_v1_mimic.parquet
- configs/50a_encoder_baseline_config.yaml

Outputs
-------
- models/50a_encoder_baseline/encoder_baseline_best.pt
- models/50a_encoder_baseline/encoder_baseline_last.pt
- reports/50a_encoder_baseline_training_log.csv
- reports/50a_encoder_baseline_summary.txt

Compute: G1 (debug) / G2 (full)

Usage
-----
  # Debug (G1):
  qsub -N step50a_debug env/sge/submit_g1.sh scripts/50a_train_encoder_baseline.py --mode debug
  # Full (G1/G2):
  qsub -N step50a env/sge/submit_g1.sh scripts/50a_train_encoder_baseline.py --mode full
"""

from __future__ import annotations

import argparse
import csv
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path
from hfpef_dtr.models.dataset import (
    ICUStayDataset,
    discover_feature_columns,
    make_train_val_test_split,
)
from hfpef_dtr.models.encoder import build_model_from_config


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_mask(
    temporal: torch.Tensor,
    mask_ratio: float,
    valid_mask: torch.Tensor,
    absorbing_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Create random feature mask for masked reconstruction.

    Returns a (B, T, D) binary tensor where 1 = masked (to predict).
    Only masks valid, non-absorbing bins.
    """
    B, T, D = temporal.shape
    # Random mask per feature per bin
    rand = torch.rand(B, T, D, device=temporal.device)
    feature_mask = (rand < mask_ratio).float()

    # Only mask valid bins that are before absorbing state
    bin_valid = (valid_mask * absorbing_mask).unsqueeze(-1)  # (B, T, 1)
    return feature_mask * bin_valid


def compute_losses(
    model_out: dict,
    batch: dict,
    feature_mask: torch.Tensor,
    pretext_cfg: dict,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Compute all enabled losses."""
    losses = {}
    temporal = batch["temporal"].to(device)
    valid_mask = batch["valid_mask"].to(device)
    absorbing_mask = batch["absorbing_mask"].to(device)
    bin_valid = valid_mask * absorbing_mask  # (B, T)

    # 1. Masked reconstruction loss
    if pretext_cfg.get("masked_reconstruction", {}).get("enabled", True):
        recon = model_out["recon"]  # (B, T, D)
        # Loss only on masked positions
        masked_diff = (recon - temporal) ** 2 * feature_mask
        n_masked = feature_mask.sum().clamp(min=1)
        recon_loss = masked_diff.sum() / n_masked
        w = pretext_cfg["masked_reconstruction"].get("loss_weight", 1.0)
        losses["recon"] = recon_loss * w

    # 2. Next-state prediction loss
    nsp_cfg = pretext_cfg.get("next_state_prediction", {})
    if nsp_cfg.get("enabled", True) and "next_state" in model_out:
        pred_next = model_out["next_state"]   # (B, T-1, D)
        target_next = temporal[:, 1:]         # (B, T-1, D)
        # Mask: both current and next bin must be valid
        nsp_valid = (bin_valid[:, :-1] * bin_valid[:, 1:]).unsqueeze(-1)
        nsp_diff = (pred_next - target_next) ** 2 * nsp_valid
        n_valid = nsp_valid.sum().clamp(min=1)
        nsp_loss = nsp_diff.sum() / n_valid
        w = nsp_cfg.get("loss_weight", 0.5)
        losses["next_state"] = nsp_loss * w

    # 3. Mortality head
    mort_cfg = pretext_cfg.get("mortality_head", {})
    if mort_cfg.get("enabled", False) and "mortality_logit" in model_out:
        logit = model_out["mortality_logit"].squeeze(-1)
        target = batch["mortality_28d"].to(device)
        mort_loss = nn.functional.binary_cross_entropy_with_logits(logit, target)
        w = mort_cfg.get("loss_weight", 0.1)
        losses["mortality"] = mort_loss * w

    # Total
    losses["total"] = sum(losses.values())
    return losses


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    pretext_cfg: dict,
    mask_ratio: float,
    gradient_clip: float,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    """Train for one epoch, return average losses."""
    model.train()
    accum = {}
    n_batches = 0

    for batch in loader:
        temporal = batch["temporal"].to(device)
        static = batch["static"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        absorbing_mask = batch["absorbing_mask"].to(device)

        # Create random feature mask
        feature_mask = create_mask(temporal, mask_ratio, valid_mask, absorbing_mask)

        # Mask input (replace masked features with 0)
        masked_input = temporal * (1 - feature_mask)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(masked_input, static, valid_mask)
            losses = compute_losses(out, batch, feature_mask, pretext_cfg, device)

        scaler.scale(losses["total"]).backward()
        if gradient_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()

        for k, v in losses.items():
            accum[k] = accum.get(k, 0.0) + v.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in accum.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    pretext_cfg: dict,
    mask_ratio: float,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    """Evaluate on validation set."""
    model.eval()
    accum = {}
    n_batches = 0

    for batch in loader:
        temporal = batch["temporal"].to(device)
        static = batch["static"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        absorbing_mask = batch["absorbing_mask"].to(device)

        feature_mask = create_mask(temporal, mask_ratio, valid_mask, absorbing_mask)
        masked_input = temporal * (1 - feature_mask)

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(masked_input, static, valid_mask)
            losses = compute_losses(out, batch, feature_mask, pretext_cfg, device)

        for k, v in losses.items():
            accum[k] = accum.get(k, 0.0) + v.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in accum.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["debug", "full"], default="debug")
    args = parser.parse_args()

    proj_cfg = load_config()
    configs_dir = resolve_path(proj_cfg, "configs")
    reports_dir = resolve_path(proj_cfg, "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ── Load encoder config ────────────────────────────────────────────
    enc_cfg_path = configs_dir / "50a_encoder_baseline_config.yaml"
    with open(enc_cfg_path) as f:
        enc_cfg = yaml.safe_load(f)

    model_cfg = enc_cfg["model"]
    pretext_cfg = enc_cfg["pretext"]
    data_cfg = enc_cfg["data"]
    train_cfg = enc_cfg["training"][args.mode]
    output_cfg = enc_cfg["output"]

    seed = enc_cfg.get("seed", 42)
    deterministic = enc_cfg.get("deterministic", True)
    set_seed(seed, deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = enc_cfg["training"].get("mixed_precision", True) and device.type == "cuda"

    print("=" * 72)
    print(f"50a_train_encoder_baseline — mode={args.mode}")
    print("=" * 72)
    print(f"  Device: {device}")
    print(f"  AMP: {use_amp}")
    print(f"  Seed: {seed}")

    # ── Load data ──────────────────────────────────────────────────────
    print("\n[1/5] Loading data ...")
    project_root = Path(proj_cfg.get("project_root", "."))
    state_path = project_root / data_cfg["state_tensor_path"]
    outcomes_path = project_root / data_cfg["outcomes_path"]

    state_df = pd.read_parquet(state_path)
    outcomes_df = pd.read_parquet(outcomes_path)
    print(f"  State tensors: {state_df.shape}")
    print(f"  Outcomes: {outcomes_df.shape}")

    # ── Discover features ──────────────────────────────────────────────
    print("\n[2/5] Discovering features ...")
    temporal_cols, static_cols = discover_feature_columns(state_df)
    print(f"  Temporal features: {len(temporal_cols)}")
    print(f"  Static features: {len(static_cols)}")

    if len(temporal_cols) == 0:
        print("  ERROR: No temporal features found. Check 45c column naming.")
        sys.exit(1)

    # ── Split ──────────────────────────────────────────────────────────
    print("\n[3/5] Creating train/val/test split ...")
    all_stays = state_df["stay_id"].unique()
    train_ids, val_ids, test_ids = make_train_val_test_split(
        all_stays, outcomes_df,
        train_frac=data_cfg.get("train_frac", 0.7),
        val_frac=data_cfg.get("val_frac", 0.15),
        test_frac=data_cfg.get("test_frac", 0.15),
        stratify_col=data_cfg.get("stratify_by", "mortality_28d"),
        seed=data_cfg.get("split_seed", 2024),
    )
    print(f"  Train: {len(train_ids):,}, Val: {len(val_ids):,}, Test: {len(test_ids):,}")

    # Save split for reproducibility
    ckpt_dir = project_root / output_cfg["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        ckpt_dir / "data_split.npz",
        train=train_ids, val=val_ids, test=test_ids,
    )

    # ── Datasets ───────────────────────────────────────────────────────
    mask_post = data_cfg.get("mask_post_absorbing", True)
    ds_kwargs = dict(
        state_df=state_df, outcomes_df=outcomes_df,
        temporal_cols=temporal_cols, static_cols=static_cols,
    )
    train_ds = ICUStayDataset(stay_ids=train_ids, **ds_kwargs)
    val_ds = ICUStayDataset(stay_ids=val_ids, **ds_kwargs)

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=train_cfg.get("num_workers", 4), pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=train_cfg.get("num_workers", 4), pin_memory=True,
    )

    # ── Model ──────────────────────────────────────────────────────────
    print("\n[4/5] Building model ...")
    model = build_model_from_config(
        d_temporal=len(temporal_cols),
        d_static=len(static_cols),
        model_cfg=model_cfg,
        pretext_cfg=pretext_cfg,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # ── Optimizer & scheduler ──────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )

    scheduler = None
    if train_cfg.get("lr_scheduler") == "cosine_warmup":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        warmup_epochs = train_cfg.get("warmup_epochs", 5)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=train_cfg["max_epochs"] - warmup_epochs,
        )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    mask_ratio = pretext_cfg.get("masked_reconstruction", {}).get("mask_ratio", 0.15)
    gradient_clip = train_cfg.get("gradient_clip", 1.0)
    patience = train_cfg.get("early_stopping_patience", 15)

    # ── Training loop ──────────────────────────────────────────────────
    print(f"\n[5/5] Training (max_epochs={train_cfg['max_epochs']}) ...")
    log_path = project_root / output_cfg["training_log"]
    log_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = []

    with open(log_path, "w", newline="") as logf:
        writer = csv.writer(logf)
        writer.writerow(["epoch", "train_total", "train_recon", "train_next_state",
                         "val_total", "val_recon", "val_next_state", "lr", "time_s"])

        for epoch in range(1, train_cfg["max_epochs"] + 1):
            t0 = time.time()

            # Warmup: linear LR ramp
            warmup_epochs = train_cfg.get("warmup_epochs", 0)
            if epoch <= warmup_epochs:
                warmup_lr = train_cfg["learning_rate"] * epoch / max(warmup_epochs, 1)
                for pg in optimizer.param_groups:
                    pg["lr"] = warmup_lr

            train_losses = train_one_epoch(
                model, train_loader, optimizer, scaler,
                pretext_cfg, mask_ratio, gradient_clip, device, use_amp,
            )
            val_losses = evaluate(
                model, val_loader, pretext_cfg, mask_ratio, device, use_amp,
            )

            if scheduler and epoch > warmup_epochs:
                scheduler.step()

            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]

            writer.writerow([
                epoch,
                f"{train_losses.get('total', 0):.6f}",
                f"{train_losses.get('recon', 0):.6f}",
                f"{train_losses.get('next_state', 0):.6f}",
                f"{val_losses.get('total', 0):.6f}",
                f"{val_losses.get('recon', 0):.6f}",
                f"{val_losses.get('next_state', 0):.6f}",
                f"{lr:.2e}",
                f"{elapsed:.1f}",
            ])
            logf.flush()

            history.append({
                "epoch": epoch, "train_total": train_losses.get("total", 0),
                "val_total": val_losses.get("total", 0), "time_s": elapsed,
            })

            # Logging
            if epoch % max(1, train_cfg["max_epochs"] // 20) == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:>3d}/{train_cfg['max_epochs']} | "
                    f"train={train_losses.get('total', 0):.4f} "
                    f"val={val_losses.get('total', 0):.4f} "
                    f"lr={lr:.2e} | {elapsed:.1f}s"
                )

            # Checkpointing
            val_total = val_losses.get("total", float("inf"))
            if val_total < best_val_loss:
                best_val_loss = val_total
                epochs_no_improve = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_total,
                    "config": enc_cfg,
                    "temporal_cols": temporal_cols,
                    "static_cols": static_cols,
                    "d_temporal": len(temporal_cols),
                    "d_static": len(static_cols),
                }, ckpt_dir / output_cfg["best_model_name"])
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    # Save last checkpoint
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_loss": val_total,
        "config": enc_cfg,
        "temporal_cols": temporal_cols,
        "static_cols": static_cols,
    }, ckpt_dir / output_cfg["last_model_name"])

    # ── Summary report ─────────────────────────────────────────────────
    report_lines = [
        "=" * 72,
        "50a — Baseline Encoder Training Summary (MIMIC-IV)",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Mode: {args.mode}",
        f"Device: {device}",
        "",
        "DATA",
        "────",
        f"  Stays: {len(all_stays):,}",
        f"  Train/Val/Test: {len(train_ids):,} / {len(val_ids):,} / {len(test_ids):,}",
        f"  Temporal features: {len(temporal_cols)}",
        f"  Static features: {len(static_cols)}",
        "",
        "MODEL",
        "─────",
        f"  Architecture: BiGRU (hidden={model_cfg['hidden_dim']}, "
        f"latent={model_cfg['latent_dim']}, layers={model_cfg['n_layers']})",
        f"  Parameters: {n_params:,}",
        f"  Pretext tasks: recon={pretext_cfg['masked_reconstruction']['enabled']}, "
        f"next_state={pretext_cfg['next_state_prediction']['enabled']}",
        "",
        "TRAINING",
        "────────",
        f"  Epochs run: {epoch}",
        f"  Best val loss: {best_val_loss:.6f} (epoch {history[np.argmin([h['val_total'] for h in history])]['epoch']})",
        f"  Final train loss: {history[-1]['train_total']:.6f}",
        f"  Avg time per epoch: {np.mean([h['time_s'] for h in history]):.1f}s",
        "",
        "OUTPUTS",
        "───────",
        f"  Best checkpoint: {ckpt_dir / output_cfg['best_model_name']}",
        f"  Training log: {log_path}",
    ]

    report = "\n".join(report_lines) + "\n"
    report_path = project_root / output_cfg["summary_report"]
    report_path.write_text(report, encoding="utf-8")
    print(f"\n  Saved: {report_path}")
    print(report)
    print("[50a] Done.")


if __name__ == "__main__":
    main()
