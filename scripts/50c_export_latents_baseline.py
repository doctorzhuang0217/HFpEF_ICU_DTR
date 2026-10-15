#!/usr/bin/env python3
"""
50c_export_latents_baseline — Export per-bin latent states z_t from baseline encoder.

Purpose
-------
Load the best checkpoint from 50a and produce per-stay × per-bin latent
representations for downstream use in Step B (phenotyping) and Step C
(causal DTR models). Also exports mean-pooled per-stay representations.

Protocol reference: Step 50c (v3.1 § 11)

Inputs
------
- models/50a_encoder_baseline/encoder_baseline_best.pt
- data/interim/45c_state_tensors_v1_mimic.parquet
- data/interim/45d_outcomes_absorb_v1_mimic.parquet

Outputs
-------
- data/interim/50c_latents_baseline_v1_mimic.parquet  (stay_id × bin × z_0..z_63)
- data/interim/50c_latents_baseline_pooled_v1_mimic.parquet  (stay_id × z_0..z_63)
- reports/50c_export_latents_baseline_summary.txt

Compute: G1
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path
from hfpef_dtr.models.dataset import ICUStayDataset
from hfpef_dtr.models.encoder import build_model_from_config


def main() -> None:
    proj_cfg = load_config()
    project_root = Path(proj_cfg.get("project_root", "."))
    interim_dir = resolve_path(proj_cfg, "interim")
    reports_dir = resolve_path(proj_cfg, "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("50c_export_latents_baseline — Export latent states z_t")
    print("=" * 72)

    # ── Load checkpoint ────────────────────────────────────────────────
    print("\n[1/3] Loading model ...")
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

    latent_dim = enc_cfg["model"]["latent_dim"]
    print(f"  Loaded encoder (latent_dim={latent_dim})")

    # ── Load data ──────────────────────────────────────────────────────
    print("\n[2/3] Loading data ...")
    data_cfg = enc_cfg["data"]
    state_df = pd.read_parquet(project_root / data_cfg["state_tensor_path"])
    outcomes_df = pd.read_parquet(project_root / data_cfg["outcomes_path"])

    all_stays = state_df["stay_id"].unique()
    n_stays = len(all_stays)
    print(f"  Total stays: {n_stays:,}")

    # Build dataset for ALL stays (not just train)
    dataset = ICUStayDataset(
        state_df=state_df,
        outcomes_df=outcomes_df,
        stay_ids=all_stays,
        temporal_cols=temporal_cols,
        static_cols=static_cols,
    )

    # ── Export latents ─────────────────────────────────────────────────
    print("\n[3/3] Exporting latents ...")
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=512, shuffle=False, num_workers=2, pin_memory=True,
    )

    z_col_names = [f"z_{i}" for i in range(latent_dim)]
    all_z_binwise = []
    all_meta_binwise = []
    all_z_pooled = []
    all_sid_pooled = []

    with torch.no_grad():
        for batch in loader:
            temporal = batch["temporal"].to(device)
            static = batch["static"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            stay_ids = batch["stay_id"].numpy()

            out = model(temporal, static, valid_mask)
            z = out["z"].cpu().numpy()  # (B, T, latent_dim)
            vm = valid_mask.cpu().numpy()  # (B, T)

            B, T, D = z.shape
            for i in range(B):
                sid = int(stay_ids[i])

                # Per-bin latents (vectorized over valid bins)
                valid_bins = np.where(vm[i] > 0)[0]
                if len(valid_bins) > 0:
                    all_z_binwise.append(z[i, valid_bins])  # (n_valid, D)
                    meta = np.column_stack([
                        np.full(len(valid_bins), sid, dtype=np.int64),
                        valid_bins.astype(np.int64),
                    ])
                    all_meta_binwise.append(meta)

                # Mean-pooled latent
                mask_sum = vm[i].sum()
                if mask_sum > 0:
                    z_pool = (z[i] * vm[i, :, np.newaxis]).sum(axis=0) / mask_sum
                else:
                    z_pool = np.zeros(D)
                all_z_pooled.append(z_pool)
                all_sid_pooled.append(sid)

    # ── Build DataFrames at once ─────────────────────────────────────
    meta_arr = np.concatenate(all_meta_binwise)
    z_bin_arr = np.concatenate(all_z_binwise)
    df_bin = pd.DataFrame(
        np.column_stack([meta_arr, z_bin_arr]),
        columns=["stay_id", "bin"] + z_col_names,
    )
    df_bin["stay_id"] = df_bin["stay_id"].astype(np.int64)
    df_bin["bin"] = df_bin["bin"].astype(np.int64)

    z_pool_arr_raw = np.stack(all_z_pooled)
    df_pool = pd.DataFrame(z_pool_arr_raw, columns=z_col_names)
    df_pool.insert(0, "stay_id", np.array(all_sid_pooled, dtype=np.int64))

    bin_path = interim_dir / "50c_latents_baseline_v1_mimic.parquet"
    pool_path = interim_dir / "50c_latents_baseline_pooled_v1_mimic.parquet"

    df_bin.to_parquet(bin_path, index=False, engine="pyarrow")
    df_pool.to_parquet(pool_path, index=False, engine="pyarrow")

    print(f"  Saved bin-wise: {bin_path} ({len(df_bin):,} rows)")
    print(f"  Saved pooled:   {pool_path} ({len(df_pool):,} rows)")

    # ── Summary report ─────────────────────────────────────────────────
    z_pool_arr = df_pool[z_col_names].values
    z_norms = np.linalg.norm(z_pool_arr, axis=1)

    report_lines = [
        "=" * 72,
        "50c — Baseline Latent Export Summary (MIMIC-IV)",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Encoder: epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f}",
        "",
        "EXPORT DIMENSIONS",
        "──────────────────",
        f"  Stays: {n_stays:,}",
        f"  Latent dim: {latent_dim}",
        f"  Bin-wise rows: {len(df_bin):,}",
        f"  Pooled rows: {len(df_pool):,}",
        "",
        "LATENT STATISTICS (pooled)",
        "──────────────────────────",
        f"  z-norm: mean={z_norms.mean():.3f}, std={z_norms.std():.3f}, "
        f"min={z_norms.min():.3f}, max={z_norms.max():.3f}",
        f"  Per-dim mean range: [{z_pool_arr.mean(axis=0).min():.3f}, "
        f"{z_pool_arr.mean(axis=0).max():.3f}]",
        f"  Per-dim std range:  [{z_pool_arr.std(axis=0).min():.3f}, "
        f"{z_pool_arr.std(axis=0).max():.3f}]",
        "",
        "  Collapsed dims (std < 0.01): "
        f"{(z_pool_arr.std(axis=0) < 0.01).sum()} / {latent_dim}",
        "",
        "DOWNSTREAM USE",
        "──────────────",
        "  Step B phenotyping: use pooled latents (50c_latents_baseline_pooled)",
        "  Step C models: use bin-wise latents (50c_latents_baseline) as state input",
        "  85c ablation: compare raw features vs these latents vs 55c DANN latents",
    ]

    # Add split membership info
    splits_path = ckpt_dir / "data_split.npz"
    if splits_path.exists():
        splits = np.load(splits_path)
        report_lines.extend([
            "",
            "DATA SPLIT (from 50a)",
            "─────────────────────",
            f"  Train: {len(splits['train']):,}",
            f"  Val:   {len(splits['val']):,}",
            f"  Test:  {len(splits['test']):,}",
        ])

    report = "\n".join(report_lines) + "\n"
    report_path = reports_dir / "50c_export_latents_baseline_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n  Saved: {report_path}")
    print(report)
    print("[50c] Done.")


if __name__ == "__main__":
    main()
