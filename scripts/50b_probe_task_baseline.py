#!/usr/bin/env python3
"""
50b_probe_task_baseline — Evaluate baseline encoder with probe tasks.

Purpose
-------
Freeze the pretrained encoder from 50a and evaluate representation quality
using linear/shallow probe classifiers on:
  1. 28-day mortality prediction (primary)
  2. 7-day mortality prediction
  3. Next-state reconstruction error (held-out test set)
  4. Absorbing state type classification (death/discharge/observed)

All probes use frozen encoder features (no encoder fine-tuning).

Protocol reference: Step 50b (v3.1 § 11)

Inputs
------
- models/50a_encoder_baseline/encoder_baseline_best.pt
- models/50a_encoder_baseline/data_split.npz
- data/interim/45c_state_tensors_v1_mimic.parquet
- data/interim/45d_outcomes_absorb_v1_mimic.parquet

Outputs
-------
- tables/50b_probe_baseline_results.csv
- reports/50b_probe_baseline_summary.txt

Compute: G1
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path
from hfpef_dtr.models.dataset import ICUStayDataset, discover_feature_columns
from hfpef_dtr.models.encoder import build_model_from_config


def extract_latents(
    model: torch.nn.Module,
    dataset: ICUStayDataset,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract latent z_t from frozen encoder.

    Returns
    -------
    z_pooled : (N, latent_dim) — masked mean-pooled over time
    z_full : (N, T, latent_dim) — per-bin latents
    """
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2,
    )

    z_pooled_list = []
    z_full_list = []

    with torch.no_grad():
        for batch in loader:
            temporal = batch["temporal"].to(device)
            static = batch["static"].to(device)
            valid_mask = batch["valid_mask"].to(device)

            out = model(temporal, static, valid_mask)
            z = out["z"]  # (B, T, latent_dim)

            # Masked mean pooling
            mask = valid_mask.unsqueeze(-1)
            z_masked = z * mask
            z_pool = z_masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            z_pooled_list.append(z_pool.cpu().numpy())
            z_full_list.append(z.cpu().numpy())

    return np.concatenate(z_pooled_list), np.concatenate(z_full_list)


def run_classification_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_name: str,
) -> dict:
    """Run logistic regression probe and return metrics."""
    # Skip if degenerate
    if len(np.unique(y_train)) < 2:
        return {"task": task_name, "train_auroc": np.nan, "val_auroc": np.nan,
                "test_auroc": np.nan, "test_auprc": np.nan, "note": "single_class"}

    clf = LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs", class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    results = {"task": task_name}
    for split_name, X, y in [("train", X_train, y_train),
                              ("val", X_val, y_val),
                              ("test", X_test, y_test)]:
        if len(np.unique(y)) < 2:
            results[f"{split_name}_auroc"] = np.nan
            continue
        proba = clf.predict_proba(X)[:, 1]
        results[f"{split_name}_auroc"] = roc_auc_score(y, proba)
        if split_name == "test":
            results["test_auprc"] = average_precision_score(y, proba)
            results["test_acc"] = accuracy_score(y, clf.predict(X))
            results["test_prevalence"] = y.mean()

    return results


def main() -> None:
    proj_cfg = load_config()
    project_root = Path(proj_cfg.get("project_root", "."))
    reports_dir = resolve_path(proj_cfg, "reports")
    tables_dir = resolve_path(proj_cfg, "tables")
    reports_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("50b_probe_task_baseline — Encoder probe evaluation")
    print("=" * 72)

    # ── Load checkpoint ────────────────────────────────────────────────
    print("\n[1/4] Loading model ...")
    ckpt_dir = project_root / "models" / "50a_encoder_baseline"
    ckpt = torch.load(ckpt_dir / "encoder_baseline_best.pt", map_location=device)

    enc_cfg = ckpt["config"]
    temporal_cols = ckpt["temporal_cols"]
    static_cols = ckpt["static_cols"]

    model = build_model_from_config(
        d_temporal=ckpt["d_temporal"],
        d_static=ckpt["d_static"],
        model_cfg=enc_cfg["model"],
        pretext_cfg=enc_cfg["pretext"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded best checkpoint (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f})")

    # ── Load data & splits ─────────────────────────────────────────────
    print("\n[2/4] Loading data ...")
    data_cfg = enc_cfg["data"]
    state_df = pd.read_parquet(project_root / data_cfg["state_tensor_path"])
    outcomes_df = pd.read_parquet(project_root / data_cfg["outcomes_path"])

    splits = np.load(ckpt_dir / "data_split.npz")
    train_ids, val_ids, test_ids = splits["train"], splits["val"], splits["test"]
    print(f"  Train: {len(train_ids):,}, Val: {len(val_ids):,}, Test: {len(test_ids):,}")

    # ── Extract latents ────────────────────────────────────────────────
    print("\n[3/4] Extracting latents ...")
    ds_kwargs = dict(
        state_df=state_df, outcomes_df=outcomes_df,
        temporal_cols=temporal_cols, static_cols=static_cols,
    )

    splits_data = {}
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        ds = ICUStayDataset(stay_ids=ids, **ds_kwargs)
        z_pool, z_full = extract_latents(model, ds, device)
        splits_data[name] = {
            "z_pool": z_pool,
            "z_full": z_full,
            "stay_ids": ds.stay_ids,
            "mortality_28d": ds.mortality_28d,
            "mortality_7d": ds.mortality_7d,
        }
        print(f"  {name}: z_pool shape={z_pool.shape}")

    # Get absorbing_type labels
    absorbing_map = (
        outcomes_df.drop_duplicates("stay_id")
        .set_index("stay_id")["absorbing_type"]
    )
    for name in splits_data:
        sids = splits_data[name]["stay_ids"]
        splits_data[name]["absorbing_death"] = np.array([
            1 if absorbing_map.get(sid) == "death" else 0
            for sid in sids
        ])

    # ── Probe tasks ────────────────────────────────────────────────────
    print("\n[4/4] Running probe tasks ...")
    results = []

    # Probe 1: 28-day mortality
    r = run_classification_probe(
        splits_data["train"]["z_pool"], splits_data["train"]["mortality_28d"],
        splits_data["val"]["z_pool"], splits_data["val"]["mortality_28d"],
        splits_data["test"]["z_pool"], splits_data["test"]["mortality_28d"],
        "mortality_28d",
    )
    results.append(r)
    print(f"  mortality_28d: test AUROC={r.get('test_auroc', float('nan')):.3f}")

    # Probe 2: 7-day mortality
    r = run_classification_probe(
        splits_data["train"]["z_pool"], splits_data["train"]["mortality_7d"],
        splits_data["val"]["z_pool"], splits_data["val"]["mortality_7d"],
        splits_data["test"]["z_pool"], splits_data["test"]["mortality_7d"],
        "mortality_7d",
    )
    results.append(r)
    print(f"  mortality_7d: test AUROC={r.get('test_auroc', float('nan')):.3f}")

    # Probe 3: Death in 24h window (absorbing_type == death)
    r = run_classification_probe(
        splits_data["train"]["z_pool"], splits_data["train"]["absorbing_death"],
        splits_data["val"]["z_pool"], splits_data["val"]["absorbing_death"],
        splits_data["test"]["z_pool"], splits_data["test"]["absorbing_death"],
        "death_in_window",
    )
    results.append(r)
    print(f"  death_in_window: test AUROC={r.get('test_auroc', float('nan')):.3f}")

    # Probe 4: Reconstruction error on test set (summary stat, not classification)
    test_ds = ICUStayDataset(stay_ids=test_ids, **ds_kwargs)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=512, shuffle=False)
    recon_errors = []
    with torch.no_grad():
        for batch in test_loader:
            temporal = batch["temporal"].to(device)
            static = batch["static"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            out = model(temporal, static, valid_mask)
            recon = out["recon"]
            diff = ((recon - temporal) ** 2 * valid_mask.unsqueeze(-1)).sum() / valid_mask.sum()
            recon_errors.append(diff.item())
    mean_recon = np.mean(recon_errors)
    results.append({"task": "test_recon_mse", "test_auroc": mean_recon, "note": "MSE, not AUROC"})
    print(f"  test_recon_mse: {mean_recon:.6f}")

    # ── Save results ───────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    csv_path = tables_dir / "50b_probe_baseline_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # ── Summary report ─────────────────────────────────────────────────
    report_lines = [
        "=" * 72,
        "50b — Baseline Encoder Probe Results (MIMIC-IV)",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Encoder: epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f}",
        f"Latent dim: {enc_cfg['model']['latent_dim']}",
        "",
        "PROBE RESULTS",
        "─────────────",
    ]
    for r in results:
        task = r["task"]
        if "note" in r and r["note"] == "MSE, not AUROC":
            report_lines.append(f"  {task}: MSE = {r['test_auroc']:.6f}")
        elif isinstance(r.get("test_auroc"), float):
            auroc = r.get("test_auroc", float("nan"))
            auprc = r.get("test_auprc", float("nan"))
            prev = r.get("test_prevalence", float("nan"))
            report_lines.append(
                f"  {task}: test AUROC = {auroc:.3f}, "
                f"AUPRC = {auprc:.3f}, prev = {prev:.3f}"
            )
        else:
            report_lines.append(f"  {task}: {r.get('note', 'N/A')}")

    report_lines.extend([
        "",
        "INTERPRETATION GUIDE",
        "────────────────────",
        "  AUROC > 0.70: encoder captures clinically relevant signal",
        "  AUROC > 0.80: strong representation for downstream use",
        "  Reconstruction MSE should decrease vs random-init baseline",
        "",
        "  NOTE: probes use mean-pooled z_t; per-bin probes may differ.",
        "  Albumin/CVP interpretations should focus on missingness channels,",
        "  not raw values (high median-fill artifact, per 45c review).",
    ])

    report = "\n".join(report_lines) + "\n"
    report_path = reports_dir / "50b_probe_baseline_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {report_path}")
    print(report)
    print("[50b] Done.")


if __name__ == "__main__":
    main()
