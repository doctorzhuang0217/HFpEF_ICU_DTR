#!/usr/bin/env python3
"""
55a_train_encoder_dann — Train domain-adversarial encoder (DANN).

Purpose
-------
Train a BiGRU encoder with gradient-reversal domain adversary to learn
domain-invariant representations. Uses the same pretext tasks as 50a
(masked reconstruction + next-state prediction) with an additional
adversarial loss that penalizes domain-predictable latent features.

When only MIMIC is available, uses train/val/test pseudo-domain labels
as a sanity check (encoder should learn to ignore split identity).
When eICU/HiRID state tensors become available, switch domain_label_strategy
to "database" for true multi-database domain adaptation.

Protocol reference: Step 55a (v3.1 § 11, Assumption 9 transportability)

Inputs
------
- configs/55a_encoder_dann_config.yaml
- data/interim/45c_state_tensors_v1_mimic.parquet
- data/interim/45d_outcomes_absorb_v1_mimic.parquet
- models/50a_encoder_baseline/encoder_baseline_best.pt  (warm-start)

Outputs
-------
- models/55a_encoder_dann/encoder_dann_best.pt
- models/55a_encoder_dann/encoder_dann_last.pt
- models/55a_encoder_dann/data_split.npz
- reports/55a_encoder_dann_training_log.csv
- reports/55a_encoder_dann_summary.txt

Compute: G1 (debug) / G2 (full)

Usage
-----
  # Debug:
  qsub -N step55a_debug env/sge/submit_g1.sh scripts/55a_train_encoder_dann.py --mode debug
  # Full:
  qsub -N step55a -q gpuh.q -l h100=1 env/sge/submit_g1.sh scripts/55a_train_encoder_dann.py --mode full
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
from hfpef_dtr.models.encoder import build_dann_model_from_config


def set_seed(seed: int, deterministic: bool = True) -> None:
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
    B, T, D = temporal.shape
    rand = torch.rand(B, T, D, device=temporal.device)
    feature_mask = (rand < mask_ratio).float()
    bin_valid = (valid_mask * absorbing_mask).unsqueeze(-1)
    return feature_mask * bin_valid


def dann_alpha_schedule(epoch: int, max_alpha: float, warmup_epochs: int) -> float:
    """Sigmoid-ramp schedule for gradient reversal strength."""
    if warmup_epochs <= 0:
        return max_alpha
    p = min(epoch / max(warmup_epochs, 1), 1.0)
    return float(max_alpha * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0))


def assign_domain_labels(
    stay_ids: np.ndarray,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: np.ndarray,
    strategy: str,
) -> np.ndarray:
    """
    Assign domain labels per stay_id.

    strategy="split": deterministic pseudo-domains via stay_id hashing (n_domains=3).
    strategy="database": requires external stay-to-database mapping.
    """
    if strategy == "split":
        # NOTE:
        # If we directly map train/val/test membership to domains, each split-specific
        # dataset has only a single domain label (e.g., train set == domain 0), which
        # makes adversarial learning degenerate. Instead, for MIMIC-only sanity checks,
        # construct deterministic pseudo-domains from stay_id hashing so every split
        # contains mixed domains and the discriminator receives a meaningful signal.
        return np.array([hash(int(sid)) % 3 for sid in stay_ids], dtype=np.int64)
    else:
        raise ValueError(f"Domain label strategy '{strategy}' not implemented yet. "
                         "Use 'split' for MIMIC-only or implement 'database' mapping.")


def compute_losses(
    model_out: dict,
    batch: dict,
    feature_mask: torch.Tensor,
    pretext_cfg: dict,
    dann_cfg: dict,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Compute reconstruction + next-state + domain adversarial losses."""
    losses = {}
    temporal = batch["temporal"].to(device)
    valid_mask = batch["valid_mask"].to(device)
    absorbing_mask = batch["absorbing_mask"].to(device)
    bin_valid = valid_mask * absorbing_mask

    # 1. Masked reconstruction
    if pretext_cfg.get("masked_reconstruction", {}).get("enabled", True):
        recon = model_out["recon"]
        masked_diff = (recon - temporal) ** 2 * feature_mask
        n_masked = feature_mask.sum().clamp(min=1)
        recon_loss = masked_diff.sum() / n_masked
        w = pretext_cfg["masked_reconstruction"].get("loss_weight", 1.0)
        losses["recon"] = recon_loss * w

    # 2. Next-state prediction
    nsp_cfg = pretext_cfg.get("next_state_prediction", {})
    if nsp_cfg.get("enabled", True) and "next_state" in model_out:
        pred_next = model_out["next_state"]
        target_next = temporal[:, 1:]
        nsp_valid = (bin_valid[:, :-1] * bin_valid[:, 1:]).unsqueeze(-1)
        nsp_diff = (pred_next - target_next) ** 2 * nsp_valid
        n_valid = nsp_valid.sum().clamp(min=1)
        nsp_loss = nsp_diff.sum() / n_valid
        w = nsp_cfg.get("loss_weight", 0.5)
        losses["next_state"] = nsp_loss * w

    # 3. Domain adversarial loss
    if "domain_logits" in model_out and "domain_label" in batch:
        domain_logits = model_out["domain_logits"]
        domain_labels = batch["domain_label"].to(device).long()
        domain_loss = nn.functional.cross_entropy(domain_logits, domain_labels)
        gamma = dann_cfg.get("domain_loss_weight", 0.05)
        losses["domain"] = domain_loss * gamma

    # Total
    losses["total"] = sum(losses.values())
    return losses


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    pretext_cfg: dict,
    dann_cfg: dict,
    mask_ratio: float,
    gradient_clip: float,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    model.train()
    accum = {}
    n_batches = 0

    for batch in loader:
        temporal = batch["temporal"].to(device)
        static = batch["static"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        absorbing_mask = batch["absorbing_mask"].to(device)

        feature_mask = create_mask(temporal, mask_ratio, valid_mask, absorbing_mask)
        masked_input = temporal * (1 - feature_mask)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(masked_input, static, valid_mask)
            losses = compute_losses(out, batch, feature_mask, pretext_cfg, dann_cfg, device)

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
    dann_cfg: dict,
    mask_ratio: float,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
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
            losses = compute_losses(out, batch, feature_mask, pretext_cfg, dann_cfg, device)

        for k, v in losses.items():
            accum[k] = accum.get(k, 0.0) + v.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in accum.items()}


class DANNDataset(ICUStayDataset):
    """ICUStayDataset extended with domain labels."""

    def __init__(self, domain_labels: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        # Map stay_ids to domain labels
        sid_to_domain = {sid: dl for sid, dl in zip(kwargs["stay_ids"], domain_labels)}
        self._domain_labels = np.array([
            sid_to_domain.get(sid, 0) for sid in self.stay_ids
        ], dtype=np.int64)

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        item["domain_label"] = torch.tensor(self._domain_labels[idx], dtype=torch.long)
        return item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["debug", "full"], default="debug")
    args = parser.parse_args()

    proj_cfg = load_config()
    configs_dir = resolve_path(proj_cfg, "configs")
    reports_dir = resolve_path(proj_cfg, "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ── Load config ─────────────────────────────────────────────────────
    cfg_path = configs_dir / "55a_encoder_dann_config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    pretext_cfg = cfg["pretext"]
    dann_cfg = cfg["dann"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"][args.mode]
    output_cfg = cfg["output"]

    seed = cfg.get("seed", 42)
    set_seed(seed, cfg.get("deterministic", True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg["training"].get("mixed_precision", True) and device.type == "cuda"

    print("=" * 72)
    print(f"55a_train_encoder_dann — mode={args.mode}")
    print("=" * 72)
    print(f"  Device: {device}")
    print(f"  Domain strategy: {dann_cfg['domain_label_strategy']}")
    print(f"  Warm-start from 50a: {dann_cfg.get('warm_start', False)}")

    # ── Load data ───────────────────────────────────────────────────────
    print("\n[1/6] Loading data ...")
    project_root = Path(proj_cfg.get("project_root", "."))
    state_df = pd.read_parquet(project_root / data_cfg["state_tensor_path"])
    outcomes_df = pd.read_parquet(project_root / data_cfg["outcomes_path"])
    print(f"  State tensors: {state_df.shape}")

    # ── Discover features ───────────────────────────────────────────────
    print("\n[2/6] Discovering features ...")
    temporal_cols, static_cols = discover_feature_columns(state_df)
    print(f"  Temporal: {len(temporal_cols)}, Static: {len(static_cols)}")

    # ── Split ───────────────────────────────────────────────────────────
    print("\n[3/6] Creating splits + domain labels ...")
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

    ckpt_dir = project_root / output_cfg["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(ckpt_dir / "data_split.npz", train=train_ids, val=val_ids, test=test_ids)

    # Domain labels
    strategy = dann_cfg.get("domain_label_strategy", "split")
    train_domain = assign_domain_labels(train_ids, train_ids, val_ids, test_ids, strategy)
    val_domain = assign_domain_labels(val_ids, train_ids, val_ids, test_ids, strategy)

    n_domains = len(np.unique(np.concatenate([train_domain, val_domain])))
    print(f"  Domain strategy: {strategy}, n_domains={n_domains}")

    # ── Datasets ────────────────────────────────────────────────────────
    ds_kwargs = dict(
        state_df=state_df, outcomes_df=outcomes_df,
        temporal_cols=temporal_cols, static_cols=static_cols,
    )
    train_ds = DANNDataset(domain_labels=train_domain, stay_ids=train_ids, **ds_kwargs)
    val_ds = DANNDataset(domain_labels=val_domain, stay_ids=val_ids, **ds_kwargs)

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=train_cfg.get("num_workers", 4), pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=train_cfg.get("num_workers", 4), pin_memory=True,
    )

    # ── Model ───────────────────────────────────────────────────────────
    print("\n[4/6] Building DANN model ...")
    dann_cfg_model = {**dann_cfg, "n_domains": n_domains}
    model = build_dann_model_from_config(
        d_temporal=len(temporal_cols),
        d_static=len(static_cols),
        model_cfg=model_cfg,
        pretext_cfg=pretext_cfg,
        dann_cfg=dann_cfg_model,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_encoder = sum(p.numel() for p in model.encoder.parameters())
    n_disc = sum(p.numel() for p in model.domain_disc.parameters())
    print(f"  Total params: {n_params:,} (encoder: {n_encoder:,}, discriminator: {n_disc:,})")

    # Warm-start from baseline
    n_loaded = 0
    if dann_cfg.get("warm_start", False):
        baseline_path = project_root / dann_cfg["baseline_checkpoint"]
        if baseline_path.exists():
            print(f"  Warm-starting from {baseline_path} ...")
            baseline_ckpt = torch.load(baseline_path, map_location=device, weights_only=False)
            n_loaded = model.load_baseline_weights(baseline_ckpt)
            print(f"  Loaded {n_loaded} parameter tensors from baseline")
        else:
            print(f"  WARNING: baseline checkpoint not found at {baseline_path}")

    # ── Optimizer ───────────────────────────────────────────────────────
    print("\n[5/6] Setting up optimizer ...")
    # Separate parameter groups: encoder+decoder vs domain discriminator
    encoder_params = list(model.encoder.parameters()) + \
                     list(model.recon_decoder.parameters())
    if model.enable_next_state:
        encoder_params += list(model.next_state_head.parameters())

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": train_cfg["learning_rate"],
         "weight_decay": train_cfg.get("weight_decay", 0.0)},
        {"params": model.domain_disc.parameters(),
         "lr": cfg["training"].get("domain_lr", 1e-3),
         "weight_decay": cfg["training"].get("domain_weight_decay", 1e-4)},
    ])

    scheduler = None
    if train_cfg.get("lr_scheduler") == "cosine_warmup":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        warmup_epochs = train_cfg.get("warmup_epochs", 5)
        scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg["max_epochs"] - warmup_epochs)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    mask_ratio = pretext_cfg.get("masked_reconstruction", {}).get("mask_ratio", 0.15)
    gradient_clip = train_cfg.get("gradient_clip", 1.0)
    patience = train_cfg.get("early_stopping_patience", 20)

    # ── Training loop ───────────────────────────────────────────────────
    print(f"\n[6/6] Training (max_epochs={train_cfg['max_epochs']}) ...")
    log_path = project_root / output_cfg["training_log"]
    log_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = []

    with open(log_path, "w", newline="") as logf:
        writer = csv.writer(logf)
        writer.writerow([
            "epoch", "train_total", "train_recon", "train_next_state", "train_domain",
            "val_total", "val_recon", "val_next_state", "val_domain",
            "lr", "dann_alpha", "time_s",
        ])

        for epoch in range(1, train_cfg["max_epochs"] + 1):
            t0 = time.time()

            # Alpha schedule for gradient reversal
            alpha = dann_alpha_schedule(
                epoch, dann_cfg.get("max_alpha", 1.0),
                dann_cfg.get("alpha_warmup_epochs", 20),
            )
            model.set_dann_alpha(alpha)

            # Warmup LR
            warmup_epochs = train_cfg.get("warmup_epochs", 0)
            if epoch <= warmup_epochs:
                warmup_lr = train_cfg["learning_rate"] * epoch / max(warmup_epochs, 1)
                for pg in optimizer.param_groups:
                    pg["lr"] = warmup_lr

            train_losses = train_one_epoch(
                model, train_loader, optimizer, scaler,
                pretext_cfg, dann_cfg, mask_ratio, gradient_clip, device, use_amp,
            )
            val_losses = evaluate(
                model, val_loader, pretext_cfg, dann_cfg, mask_ratio, device, use_amp,
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
                f"{train_losses.get('domain', 0):.6f}",
                f"{val_losses.get('total', 0):.6f}",
                f"{val_losses.get('recon', 0):.6f}",
                f"{val_losses.get('next_state', 0):.6f}",
                f"{val_losses.get('domain', 0):.6f}",
                f"{lr:.2e}",
                f"{alpha:.4f}",
                f"{elapsed:.1f}",
            ])
            logf.flush()

            history.append({
                "epoch": epoch, "train_total": train_losses.get("total", 0),
                "val_total": val_losses.get("total", 0),
                "train_domain": train_losses.get("domain", 0),
                "val_domain": val_losses.get("domain", 0),
                "alpha": alpha, "time_s": elapsed,
            })

            if epoch % max(1, train_cfg["max_epochs"] // 20) == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:>3d}/{train_cfg['max_epochs']} | "
                    f"train={train_losses.get('total', 0):.4f} "
                    f"val={val_losses.get('total', 0):.4f} "
                    f"domain={train_losses.get('domain', 0):.4f} "
                    f"α={alpha:.3f} lr={lr:.2e} | {elapsed:.1f}s"
                )

            # Early stopping on val total (recon + nsp; domain is regularization)
            val_recon_nsp = val_losses.get("recon", 0) + val_losses.get("next_state", 0)
            if val_recon_nsp < best_val_loss:
                best_val_loss = val_recon_nsp
                epochs_no_improve = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_recon_nsp,
                    "val_domain_loss": val_losses.get("domain", 0),
                    "dann_alpha": alpha,
                    "config": cfg,
                    "temporal_cols": temporal_cols,
                    "static_cols": static_cols,
                    "d_temporal": len(temporal_cols),
                    "d_static": len(static_cols),
                    "n_domains": n_domains,
                    "domain_label_strategy": strategy,
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
        "val_loss": val_recon_nsp,
        "config": cfg,
        "temporal_cols": temporal_cols,
        "static_cols": static_cols,
    }, ckpt_dir / output_cfg["last_model_name"])

    # ── Summary report ──────────────────────────────────────────────────
    best_ep = history[np.argmin([h["val_total"] for h in history])]["epoch"]
    report_lines = [
        "=" * 72,
        "55a — DANN Encoder Training Summary (MIMIC-IV)",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Mode: {args.mode}",
        f"Device: {device}",
        "",
        "DATA",
        "────",
        f"  Stays: {len(all_stays):,}",
        f"  Train/Val/Test: {len(train_ids):,} / {len(val_ids):,} / {len(test_ids):,}",
        f"  Domain strategy: {strategy} (n_domains={n_domains})",
        "",
        "MODEL",
        "─────",
        f"  Architecture: BiGRU + DANN (encoder={n_encoder:,}, discriminator={n_disc:,})",
        f"  Total params: {n_params:,}",
        f"  Warm-start from 50a: {'yes' if n_loaded > 0 else 'no'} ({n_loaded} tensors)",
        "",
        "TRAINING",
        "────────",
        f"  Epochs run: {epoch}",
        f"  Best val loss (recon+nsp): {best_val_loss:.6f} (epoch {best_ep})",
        f"  Final train loss: {history[-1]['train_total']:.6f}",
        f"  Final domain loss: {history[-1]['train_domain']:.6f}",
        f"  Final α: {history[-1]['alpha']:.4f}",
        f"  γ (domain weight): {dann_cfg.get('domain_loss_weight', 0.05)}",
        f"  Avg epoch time: {np.mean([h['time_s'] for h in history]):.1f}s",
        "",
        "OUTPUTS",
        "───────",
        f"  Best checkpoint: {ckpt_dir / output_cfg['best_model_name']}",
        f"  Training log: {log_path}",
        "",
        "NEXT STEPS",
        "──────────",
        "  55b: probe/audit representation quality + domain leakage test",
        "  55c: export latents for downstream use",
        "  55d: freeze encoder versions registry",
    ]

    report = "\n".join(report_lines) + "\n"
    report_path = project_root / output_cfg["summary_report"]
    report_path.write_text(report, encoding="utf-8")
    print(f"\n  Saved: {report_path}")
    print(report)
    print("[55a] Done.")


if __name__ == "__main__":
    main()
