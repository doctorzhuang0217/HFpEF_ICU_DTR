#!/usr/bin/env python3
"""90b_07 — eICU Recalibrated Model Sensitivity Analysis (P1-2)

Protocol: v3.1, Step 90b (sub-step 7, manuscript enhancement P1-2)
Compute: G1 (GPU)

Purpose
-------
Sensitivity analysis comparing FROZEN vs RECALIBRATED MIMIC models on eICU.
Standard practice for external validation: (1) evaluate frozen model (90b_05),
(2) fine-tune outcome head on target domain, (3) compare ATEs.

Recalibration strategy (conservative):
  - Freeze transformer encoder + next_state_head
  - Fine-tune outcome_head only on eICU Tier A data
  - Train/val split within Tier A (80/20)
  - Compare: ATE_frozen vs ATE_recalibrated

This does NOT replace the frozen evaluation (which is the primary analysis).
Recalibration is a sensitivity check that quantifies how much of the
ATE difference is due to outcome-model miscalibration vs genuine
treatment effect difference.

Inputs
------
- data/interim/90b_eicu_state_tensors_v1.parquet
- data/interim/90b_eicu_outcomes_absorb.parquet
- data/interim/90b_eicu_propensity_coarse3.parquet
- models/80c_gtransformer_rollout_ow1_best.pt (frozen checkpoint)
- tables/90b_04_eicu_site_heterogeneity.csv
- configs/80_step_c_training_v1.yaml

Outputs
-------
- tables/90b_07_recalibrated_sensitivity.csv
- models/90b_07_recalibrated_outcome_head.pt
- reports/90b_07_recalibration_sensitivity_report.txt
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import yaml
from tqdm import tqdm

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(ROOT / "src"))

from hfpef_dtr.models.dtr_dataset import DTRDataset
from hfpef_dtr.models.gtransformer import GTransformer, build_gtransformer_from_config
from hfpef_dtr.models.ope import (
    evaluate_dm, bootstrap_ope,
)
from hfpef_dtr.models.step_c_reporting import save_policy_values, save_bootstrap_cis

MODELS_DIR = ROOT / "models"
TABLES_DIR = ROOT / "tables"
REPORTS_DIR = ROOT / "reports"
INTERIM_DIR = ROOT / "data" / "interim"
STAGING_DIR = ROOT / "data" / "staging"

for d in [TABLES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(
        description="90b_07: eICU recalibrated model sensitivity (P1-2)"
    )
    p.add_argument("--tag", type=str, default="rollout_ow1",
                   help="Checkpoint tag for frozen model")
    p.add_argument("--tier-a-threshold", type=float, default=80.0,
                   help="Tier A fluid_nz_pct threshold (default: 80%%)")
    p.add_argument("--epochs", type=int, default=30,
                   help="Fine-tuning epochs for outcome head")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Learning rate for outcome head fine-tuning")
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="Validation fraction within Tier A")
    p.add_argument("--bootstrap", type=int, default=500,
                   help="Number of bootstrap resamples")
    p.add_argument("--skip-bootstrap", action="store_true",
                   help="Skip bootstrap CIs")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    return p.parse_args()


def load_frozen_gtransformer(
    config: dict,
    d_temporal: int,
    d_static: int,
    n_actions: int,
    device: torch.device,
    tag: str | None = None,
) -> GTransformer:
    """Load the frozen MIMIC-trained G-Transformer checkpoint."""
    model_id = "80c_gtransformer"
    model_cfg = config["models"].get(model_id, {})
    grid_cfg = config["grid"]

    base_name = f"{model_id}_{tag}_best.pt" if tag else f"{model_id}_best.pt"
    ckpt_path = MODELS_DIR / base_name
    if not ckpt_path.exists() and tag:
        ckpt_path = MODELS_DIR / f"{model_id}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = build_gtransformer_from_config(
        d_temporal, d_static, n_actions, model_cfg, grid_cfg
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded frozen G-Transformer: {n_params:,} params, "
          f"epoch={ckpt.get('epoch', '?')}")
    return model


def freeze_encoder_and_state_head(model: GTransformer):
    """Freeze everything except outcome_head."""
    for name, param in model.named_parameters():
        if "outcome_head" not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} "
          f"({100*trainable/total:.1f}% — outcome_head only)")


def finetune_outcome_head(
    model: GTransformer,
    train_ds: DTRDataset,
    val_ds: DTRDataset,
    device: torch.device,
    epochs: int,
    lr: float,
) -> dict:
    """Fine-tune outcome_head on eICU data.  Returns training history."""
    model.train()
    # Only optimize outcome_head parameters
    optimizer = torch.optim.Adam(
        [p for p in model.outcome_head.parameters()],
        lr=lr, weight_decay=1e-5,
    )
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    best_val_loss = float("inf")
    best_state = None

    for epoch in tqdm(range(epochs), desc="  Fine-tuning outcome_head"):
        # ── Train ──
        model.train()
        train_losses = []
        for batch in train_loader:
            temporal = batch["temporal"].to(device)
            static = batch["static"].to(device) if "static" in batch else None
            action = batch["action"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            absorbing_mask = batch["absorbing_mask"].to(device)
            outcome = batch["mortality_28d"].float().to(device)

            out = model(temporal, static, action, valid_mask, absorbing_mask)
            logit = out["outcome_logit"].squeeze(-1)
            loss = criterion(logit, outcome)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # ── Validate ──
        model.eval()
        val_losses = []
        val_logits_all = []
        val_labels_all = []
        with torch.no_grad():
            for batch in val_loader:
                temporal = batch["temporal"].to(device)
                static = batch["static"].to(device) if "static" in batch else None
                action = batch["action"].to(device)
                valid_mask = batch["valid_mask"].to(device)
                absorbing_mask = batch["absorbing_mask"].to(device)
                outcome = batch["mortality_28d"].float().to(device)

                out = model(temporal, static, action, valid_mask, absorbing_mask)
                logit = out["outcome_logit"].squeeze(-1)
                loss = criterion(logit, outcome)
                val_losses.append(loss.item())
                val_logits_all.append(logit.cpu().numpy())
                val_labels_all.append(outcome.cpu().numpy())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        # AUC
        try:
            from sklearn.metrics import roc_auc_score
            all_logits = np.concatenate(val_logits_all)
            all_labels = np.concatenate(val_labels_all)
            auc = roc_auc_score(all_labels, all_logits)
        except Exception:
            auc = float("nan")
        history["val_auc"].append(auc)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = copy.deepcopy(model.outcome_head.state_dict())

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}: train={avg_train:.4f}, "
                  f"val={avg_val:.4f}, AUC={auc:.4f}")

    # Restore best
    if best_state is not None:
        model.outcome_head.load_state_dict(best_state)
        print(f"  Restored best val_loss={best_val_loss:.4f}")

    model.eval()
    return history


def main():
    args = parse_args()

    print("=" * 72)
    print("90b_07 — eICU Recalibrated Model Sensitivity (P1-2)")
    print("=" * 72)
    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rng = np.random.RandomState(args.seed)

    # ── 1. Setup ──────────────────────────────────────────────────────────
    print("\n[1] Setup...")
    config_path = ROOT / "configs" / "80_step_c_training_v1.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # ── 2. Load MIMIC feature spec ────────────────────────────────────────
    print("\n[2] Loading feature spec + data...")
    registry_path = ROOT / "configs" / "90a_primary_model_registry.json"
    with open(registry_path) as f:
        registry = json.load(f)
    features = registry["features"]
    training = registry["training_config"]
    mimic_spec = {
        "temporal_cols": features["temporal_cols"],
        "static_cols": features["static_cols"],
        "n_temporal": features["n_temporal"],
        "n_static": features["n_static"],
        "n_actions": training["action"]["n_merged_cells"],
    }

    # Load eICU data
    state_df = pd.read_parquet(INTERIM_DIR / "90b_eicu_state_tensors_v1.parquet")
    outcomes_df = pd.read_parquet(INTERIM_DIR / "90b_eicu_outcomes_absorb.parquet")
    propensity_df = pd.read_parquet(INTERIM_DIR / "90b_eicu_propensity_coarse3.parquet")
    cohort = pd.read_parquet(STAGING_DIR / "90b_eicu_cohort_v1.parquet")

    # Align features
    temporal_cols = mimic_spec["temporal_cols"]
    static_cols = mimic_spec["static_cols"]
    missing = [c for c in temporal_cols + static_cols if c not in state_df.columns]
    for c in missing:
        state_df[c] = 0.0

    # Load merge map
    merge_path = ROOT / "configs" / "70b_action_merge_map.yaml"
    merge_map = None
    if merge_path.exists():
        with open(merge_path) as f:
            mm = yaml.safe_load(f)
        if "merge_map" in mm:
            merge_map = {int(k): int(v) for k, v in mm["merge_map"].items()}

    mimic_n_actions = mimic_spec["n_actions"]

    # ── 3. Tier A hospital filter ─────────────────────────────────────────
    print("\n[3] Applying Tier A hospital filter...")
    site_csv = TABLES_DIR / "90b_04_eicu_site_heterogeneity.csv"
    if not site_csv.exists():
        print("  ERROR: site heterogeneity CSV not found, cannot proceed")
        sys.exit(1)

    site_df = pd.read_csv(site_csv)
    tier_a_hids = set(
        site_df.loc[site_df["fluid_nz_pct"] >= args.tier_a_threshold, "hospitalid"]
    )
    tier_a_stays = set(
        cohort.loc[cohort["hospitalid"].isin(tier_a_hids), "patientunitstayid"]
    )
    print(f"  Tier A (fluid_nz >= {args.tier_a_threshold:.0f}%): "
          f"{len(tier_a_hids)} hospitals, {len(tier_a_stays):,} stays")

    if len(tier_a_stays) < 200:
        print(f"  ERROR: only {len(tier_a_stays)} Tier A stays, need >= 200")
        sys.exit(1)

    # ── 4. Train/val split within Tier A ──────────────────────────────────
    print("\n[4] Creating train/val split...")
    tier_a_stay_arr = np.array(sorted(tier_a_stays))
    rng.shuffle(tier_a_stay_arr)
    n_val = max(50, int(len(tier_a_stay_arr) * args.val_frac))
    val_stays = set(tier_a_stay_arr[:n_val])
    train_stays = set(tier_a_stay_arr[n_val:])
    print(f"  Train: {len(train_stays):,}, Val: {len(val_stays):,}")

    # Build datasets
    ds_kwargs = dict(
        state_df=state_df,
        outcomes_df=outcomes_df,
        propensity_df=propensity_df,
        temporal_cols=temporal_cols,
        static_cols=static_cols,
        merge_map=merge_map,
        n_bins=config["grid"]["n_bins"],
        action_vocab=config["action"]["valid_merged_indices"],
    )
    train_ds = DTRDataset(stay_ids=np.array(list(train_stays)), **ds_kwargs)
    val_ds = DTRDataset(stay_ids=np.array(list(val_stays)), **ds_kwargs)
    full_tier_a_ds = DTRDataset(stay_ids=tier_a_stay_arr, **ds_kwargs)

    # Also build full-cohort dataset for comparison
    all_stays = state_df["stay_id"].unique()
    full_ds = DTRDataset(stay_ids=all_stays, **ds_kwargs)

    print(f"  Train DS: {len(train_ds)}, Val DS: {len(val_ds)}, "
          f"Full Tier A: {len(full_tier_a_ds)}, Full cohort: {len(full_ds)}")

    # ── 5. Load frozen model + evaluate BEFORE recalibration ──────────────
    print("\n[5] Evaluating FROZEN model...")
    frozen_model = load_frozen_gtransformer(
        config, full_ds.d_temporal, full_ds.d_static,
        mimic_n_actions, device, tag=args.tag,
    )

    # Frozen on Tier A
    frozen_tier_a = evaluate_dm(frozen_model, full_tier_a_ds, device, use_rollout=False)
    print(f"  Frozen (Tier A): V(obs)={frozen_tier_a['v_observed']:.4f}, "
          f"V(rule)={frozen_tier_a['v_rule']:.4f}, ATE={frozen_tier_a['ate']:+.4f}")

    # Frozen on full cohort
    frozen_full = evaluate_dm(frozen_model, full_ds, device, use_rollout=False)
    print(f"  Frozen (full):   V(obs)={frozen_full['v_observed']:.4f}, "
          f"V(rule)={frozen_full['v_rule']:.4f}, ATE={frozen_full['ate']:+.4f}")

    # Bootstrap frozen
    frozen_boot_ta = {}
    frozen_boot_full = {}
    if not args.skip_bootstrap:
        print(f"  Bootstrap frozen Tier A (B={args.bootstrap})...")
        def dm_fn(ds, **kw):
            return evaluate_dm(frozen_model, ds, device, use_rollout=False)
        frozen_boot_ta = bootstrap_ope(
            dm_fn, full_tier_a_ds,
            n_resamples=args.bootstrap, seed=args.seed,
        )
        if frozen_boot_ta.get("n_success", 0) > 0:
            print(f"    ATE={frozen_boot_ta['ate_mean']:+.4f} "
                  f"[{frozen_boot_ta['ate_ci_lo']:.4f}, {frozen_boot_ta['ate_ci_hi']:.4f}]")

        print(f"  Bootstrap frozen full (B={args.bootstrap})...")
        frozen_boot_full = bootstrap_ope(
            dm_fn, full_ds,
            n_resamples=args.bootstrap, seed=args.seed + 1,
        )
        if frozen_boot_full.get("n_success", 0) > 0:
            print(f"    ATE={frozen_boot_full['ate_mean']:+.4f} "
                  f"[{frozen_boot_full['ate_ci_lo']:.4f}, {frozen_boot_full['ate_ci_hi']:.4f}]")

    # ── 6. Recalibrate: fine-tune outcome_head on Tier A ──────────────────
    print("\n[6] Recalibrating outcome_head on Tier A...")
    recal_model = load_frozen_gtransformer(
        config, full_ds.d_temporal, full_ds.d_static,
        mimic_n_actions, device, tag=args.tag,
    )
    freeze_encoder_and_state_head(recal_model)

    history = finetune_outcome_head(
        recal_model, train_ds, val_ds, device,
        epochs=args.epochs, lr=args.lr,
    )

    # Save recalibrated outcome_head
    recal_path = MODELS_DIR / "90b_07_recalibrated_outcome_head.pt"
    torch.save({
        "outcome_head_state_dict": recal_model.outcome_head.state_dict(),
        "history": history,
        "tier_a_threshold": args.tier_a_threshold,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "epochs": args.epochs,
        "lr": args.lr,
    }, recal_path)
    print(f"  Saved: {recal_path}")

    # ── 7. Evaluate RECALIBRATED model ────────────────────────────────────
    print("\n[7] Evaluating RECALIBRATED model...")

    # Recalibrated on Tier A
    recal_tier_a = evaluate_dm(recal_model, full_tier_a_ds, device, use_rollout=False)
    print(f"  Recal (Tier A): V(obs)={recal_tier_a['v_observed']:.4f}, "
          f"V(rule)={recal_tier_a['v_rule']:.4f}, ATE={recal_tier_a['ate']:+.4f}")

    # Recalibrated on full cohort
    recal_full = evaluate_dm(recal_model, full_ds, device, use_rollout=False)
    print(f"  Recal (full):   V(obs)={recal_full['v_observed']:.4f}, "
          f"V(rule)={recal_full['v_rule']:.4f}, ATE={recal_full['ate']:+.4f}")

    # Bootstrap recalibrated
    recal_boot_ta = {}
    recal_boot_full = {}
    if not args.skip_bootstrap:
        print(f"  Bootstrap recalibrated Tier A (B={args.bootstrap})...")
        def dm_recal_fn(ds, **kw):
            return evaluate_dm(recal_model, ds, device, use_rollout=False)
        recal_boot_ta = bootstrap_ope(
            dm_recal_fn, full_tier_a_ds,
            n_resamples=args.bootstrap, seed=args.seed + 2,
        )
        if recal_boot_ta.get("n_success", 0) > 0:
            print(f"    ATE={recal_boot_ta['ate_mean']:+.4f} "
                  f"[{recal_boot_ta['ate_ci_lo']:.4f}, {recal_boot_ta['ate_ci_hi']:.4f}]")

        print(f"  Bootstrap recalibrated full (B={args.bootstrap})...")
        recal_boot_full = bootstrap_ope(
            dm_recal_fn, full_ds,
            n_resamples=args.bootstrap, seed=args.seed + 3,
        )
        if recal_boot_full.get("n_success", 0) > 0:
            print(f"    ATE={recal_boot_full['ate_mean']:+.4f} "
                  f"[{recal_boot_full['ate_ci_lo']:.4f}, {recal_boot_full['ate_ci_hi']:.4f}]")

    # ── 8. Comparison table ───────────────────────────────────────────────
    print("\n[8] Building comparison table...")
    rows = []
    for label, subset, frz, rec, fb, rb in [
        ("tier_a", "Tier A", frozen_tier_a, recal_tier_a,
         frozen_boot_ta, recal_boot_ta),
        ("full", "Full cohort", frozen_full, recal_full,
         frozen_boot_full, recal_boot_full),
    ]:
        row = {
            "subset": label,
            "n_stays": len(full_tier_a_ds) if label == "tier_a" else len(full_ds),
            "frozen_ate": frz["ate"],
            "frozen_v_obs": frz["v_observed"],
            "frozen_v_rule": frz["v_rule"],
            "recal_ate": rec["ate"],
            "recal_v_obs": rec["v_observed"],
            "recal_v_rule": rec["v_rule"],
            "ate_delta": rec["ate"] - frz["ate"],
            "same_sign": np.sign(frz["ate"]) == np.sign(rec["ate"]),
        }
        if fb.get("n_success", 0) > 0:
            row["frozen_ci_lo"] = fb["ate_ci_lo"]
            row["frozen_ci_hi"] = fb["ate_ci_hi"]
        if rb.get("n_success", 0) > 0:
            row["recal_ci_lo"] = rb["ate_ci_lo"]
            row["recal_ci_hi"] = rb["ate_ci_hi"]
        rows.append(row)

    comp_df = pd.DataFrame(rows)
    comp_path = TABLES_DIR / "90b_07_recalibrated_sensitivity.csv"
    comp_df.to_csv(comp_path, index=False)
    print(f"  Saved: {comp_path}")

    # ── 9. Report ─────────────────────────────────────────────────────────
    print("\n[9] Generating report...")
    report_lines = [
        "=" * 72,
        "90b_07 — eICU Recalibrated Model Sensitivity (P1-2)",
        "=" * 72,
        f"Generated: {ts}",
        f"Protocol: v3.1, Step 90b, P1-2 manuscript enhancement",
        "",
        "DESIGN",
        "──────",
        f"  Frozen checkpoint: 80c_gtransformer (tag={args.tag})",
        f"  Recalibration: outcome_head fine-tuning only",
        f"  Encoder + next_state_head: FROZEN (no gradient)",
        f"  Tier A threshold: fluid_nz >= {args.tier_a_threshold:.0f}%",
        f"  Fine-tuning: {args.epochs} epochs, lr={args.lr}, val_frac={args.val_frac}",
        f"  Train: {len(train_ds):,} stays, Val: {len(val_ds):,} stays",
        "",
        "TRAINING HISTORY",
        "────────────────",
        f"  Final train_loss: {history['train_loss'][-1]:.4f}",
        f"  Best val_loss:    {min(history['val_loss']):.4f}",
        f"  Final val_AUC:    {history['val_auc'][-1]:.4f}",
        "",
        "COMPARISON: FROZEN vs RECALIBRATED",
        "──────────────────────────────────",
    ]

    for _, row in comp_df.iterrows():
        report_lines.append(f"  {row['subset'].upper()}: n={row['n_stays']:,}")
        frz_line = f"    Frozen:       ATE={row['frozen_ate']:+.4f}"
        if "frozen_ci_lo" in row and pd.notna(row.get("frozen_ci_lo")):
            frz_line += f" [{row['frozen_ci_lo']:.4f}, {row['frozen_ci_hi']:.4f}]"
        report_lines.append(frz_line)

        rec_line = f"    Recalibrated: ATE={row['recal_ate']:+.4f}"
        if "recal_ci_lo" in row and pd.notna(row.get("recal_ci_lo")):
            rec_line += f" [{row['recal_ci_lo']:.4f}, {row['recal_ci_hi']:.4f}]"
        report_lines.append(rec_line)

        report_lines.append(
            f"    Delta: {row['ate_delta']:+.4f}, same_sign={row['same_sign']}"
        )
        report_lines.append("")

    report_lines.extend([
        "INTERPRETATION",
        "──────────────",
        "  If frozen and recalibrated ATEs have the same sign and similar",
        "  magnitude (|delta| < 0.02), the frozen model is well-calibrated",
        "  for eICU and the treatment effect estimate is robust to outcome",
        "  model specification.",
        "",
        "  If ATEs differ substantially, the difference quantifies how much",
        "  of the transportability gap is attributable to outcome-model",
        "  miscalibration (vs genuine treatment effect heterogeneity).",
        "",
        "ARTIFACTS",
        "─────────",
        f"  Comparison table: {comp_path}",
        f"  Recalibrated weights: {recal_path}",
    ])

    report_path = REPORTS_DIR / "90b_07_recalibration_sensitivity_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"  Report: {report_path}")

    total_time = time.time() - t0
    print()
    print("=" * 72)
    print(f"90b_07 COMPLETE. Total time: {total_time/60:.1f} min")
    print("=" * 72)


if __name__ == "__main__":
    main()
