#!/usr/bin/env python3
"""
55c_export_latents_dann — Export per-bin latent states from DANN encoder.

Purpose
-------
Load the best DANN checkpoint from 55a and produce per-stay × per-bin
and mean-pooled latent representations. Same structure as 50c but using
the domain-invariant encoder.

Protocol reference: Step 55c (v3.1 § 11)

Inputs
------
- models/55a_encoder_dann/encoder_dann_best.pt
- data/interim/45c_state_tensors_v1_mimic.parquet
- data/interim/45d_outcomes_absorb_v1_mimic.parquet

Outputs
-------
- data/interim/55c_latents_dann_v1_mimic.parquet  (stay_id × bin × z_0..z_63)
- data/interim/55c_latents_dann_pooled_v1_mimic.parquet  (stay_id × z_0..z_63)
- reports/55c_export_latents_dann_summary.txt

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
from hfpef_dtr.models.encoder import build_dann_model_from_config


def main() -> None:
    proj_cfg = load_config()
    project_root = Path(proj_cfg.get("project_root", "."))
    interim_dir = resolve_path(proj_cfg, "interim")
    reports_dir = resolve_path(proj_cfg, "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("55c_export_latents_dann — Export DANN latent states z_t")
    print("=" * 72)

    # ── Load checkpoint ─────────────────────────────────────────────────
    print("\n[1/3] Loading DANN model ...")
    ckpt_dir = project_root / "models" / "55a_encoder_dann"
    ckpt = torch.load(ckpt_dir / "encoder_dann_best.pt", map_location=device, weights_only=False)

    cfg = ckpt["config"]
    temporal_cols = ckpt["temporal_cols"]
    static_cols = ckpt["static_cols"]

    dann_cfg_model = {**cfg["dann"], "n_domains": ckpt.get("n_domains", 2)}
    model = build_dann_model_from_config(
        d_temporal=ckpt["d_temporal"],
        d_static=ckpt["d_static"],
        model_cfg=cfg["model"],
        pretext_cfg=cfg["pretext"],
        dann_cfg=dann_cfg_model,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    latent_dim = cfg["model"]["latent_dim"]
    print(f"  Loaded DANN encoder (latent_dim={latent_dim}, "
          f"epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f})")

    # ── Load data ───────────────────────────────────────────────────────
    print("\n[2/3] Loading data ...")
    data_cfg = cfg["data"]
    state_df = pd.read_parquet(project_root / data_cfg["state_tensor_path"])
    outcomes_df = pd.read_parquet(project_root / data_cfg["outcomes_path"])

    all_stays = state_df["stay_id"].unique()
    n_stays = len(all_stays)
    print(f"  Total stays: {n_stays:,}")

    dataset = ICUStayDataset(
        state_df=state_df, outcomes_df=outcomes_df,
        stay_ids=all_stays,
        temporal_cols=temporal_cols, static_cols=static_cols,
    )

    # ── Export latents ──────────────────────────────────────────────────
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
            z = out["z"].cpu().numpy()
            vm = valid_mask.cpu().numpy()

            B, T, D = z.shape
            for i in range(B):
                sid = int(stay_ids[i])

                valid_bins = np.where(vm[i] > 0)[0]
                if len(valid_bins) > 0:
                    all_z_binwise.append(z[i, valid_bins])
                    meta = np.column_stack([
                        np.full(len(valid_bins), sid, dtype=np.int64),
                        valid_bins.astype(np.int64),
                    ])
                    all_meta_binwise.append(meta)

                mask_sum = vm[i].sum()
                if mask_sum > 0:
                    z_pool = (z[i] * vm[i, :, np.newaxis]).sum(axis=0) / mask_sum
                else:
                    z_pool = np.zeros(D)
                all_z_pooled.append(z_pool)
                all_sid_pooled.append(sid)

    # Build DataFrames
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

    bin_path = interim_dir / "55c_latents_dann_v1_mimic.parquet"
    pool_path = interim_dir / "55c_latents_dann_pooled_v1_mimic.parquet"

    df_bin.to_parquet(bin_path, index=False, engine="pyarrow")
    df_pool.to_parquet(pool_path, index=False, engine="pyarrow")

    print(f"  Saved bin-wise: {bin_path} ({len(df_bin):,} rows)")
    print(f"  Saved pooled:   {pool_path} ({len(df_pool):,} rows)")

    # ── Summary report ──────────────────────────────────────────────────
    z_pool_vals = df_pool[z_col_names].values
    z_norms = np.linalg.norm(z_pool_vals, axis=1)

    report_lines = [
        "=" * 72,
        "55c — DANN Latent Export Summary (MIMIC-IV)",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Encoder: DANN, epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f}",
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
        f"  Per-dim mean range: [{z_pool_vals.mean(0).min():.3f}, {z_pool_vals.mean(0).max():.3f}]",
        f"  Per-dim std range:  [{z_pool_vals.std(0).min():.3f}, {z_pool_vals.std(0).max():.3f}]",
        f"  Collapsed dims (std < 0.01): {(z_pool_vals.std(0) < 0.01).sum()} / {latent_dim}",
        "",
        "DOWNSTREAM USE",
        "──────────────",
        "  Step B phenotyping: use pooled latents (55c_latents_dann_pooled)",
        "  Step C models: use bin-wise latents (55c_latents_dann) as domain-inv state",
        "  85c ablation: compare 50c baseline vs 55c DANN vs raw features",
    ]

    splits_path = ckpt_dir / "data_split.npz"
    if splits_path.exists():
        splits = np.load(splits_path)
        report_lines.extend([
            "",
            "DATA SPLIT (from 55a)",
            "─────────────────────",
            f"  Train: {len(splits['train']):,}",
            f"  Val:   {len(splits['val']):,}",
            f"  Test:  {len(splits['test']):,}",
        ])

    report = "\n".join(report_lines) + "\n"
    report_path = reports_dir / "55c_export_latents_dann_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n  Saved: {report_path}")
    print(report)
    print("[55c] Done.")


if __name__ == "__main__":
    main()
