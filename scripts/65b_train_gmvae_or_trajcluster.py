#!/usr/bin/env python3
"""
65b_train_gmvae_or_trajcluster — Trajectory phenotype clustering.

Purpose
-------
Fit trajectory phenotype models on the full post-T0 feature matrix from 65a.
Two tracks are implemented:

  Track A (CPU, primary):
    K-means + GMM sweep on phase-pooled encoder latent trajectory features,
    analogous to 60b for early phenotypes.

  Track B (GPU, optional):
    Gaussian Mixture VAE (GMVAE) on bin-wise encoder latent sequences.
    Learns a latent trajectory space with a K-component Gaussian mixture
    prior, producing soft cluster assignments per stay.

Track A runs first and is always produced. Track B runs only when --mode=gmvae
is specified and GPU is available.

Protocol reference: Step 65b (v3.1 § 12 Step B trajectory phenotype)

Inputs
------
- data/interim/65a_trajectory_features_v1_mimic.parquet
- data/interim/50c_latents_baseline_v1_mimic.parquet  (for GMVAE)
- configs/65b_trajectory_cluster_config.yaml

Outputs
-------
- data/interim/65b_trajectory_clusters_v1_mimic.parquet  (stay_id + labels)
- tables/65b_trajectory_cluster_metrics.csv
- models/65b_gmvae/gmvae_best.pt  (if Track B)
- reports/65b_trajectory_cluster_summary.txt

Compute: C3 (Track A) / G1 (Track B)

Usage
-----
  # Track A only (CPU):
  env/run_py.sh scripts/65b_train_gmvae_or_trajcluster.py --mode cluster
  # Track B (GPU):
  qsub -N step65b env/sge/submit_g1.sh scripts/65b_train_gmvae_or_trajcluster.py --mode gmvae
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path

DEFAULT_K_CANDIDATES = [2, 3, 4, 5, 6]
DEFAULT_SEED = 2024
DEFAULT_N_INIT = 10


def _sorted_z_cols(columns: list[str]) -> list[str]:
    """Sort latent-derived columns by numeric z suffix."""
    def z_idx(col: str) -> int:
        if col.startswith("z_"):
            return int(col.split("_")[1])
        return int(col.rsplit("_", 1)[1])

    return sorted(columns, key=z_idx)


def _select_trajectory_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Select features for trajectory clustering — prefer latent trajectory features."""
    # Prefer phase-pooled baseline latent trajectory features
    base_phase_cols = _sorted_z_cols(
        [c for c in df.columns if c.startswith("base_") and "_z_" in c]
    )
    if len(base_phase_cols) >= 16:
        print(f"  Using baseline latent trajectory features: {len(base_phase_cols)} dims")
        return df[base_phase_cols].values, base_phase_cols

    # Fall back to DANN latent trajectory features
    dann_phase_cols = _sorted_z_cols(
        [c for c in df.columns if c.startswith("dann_") and "_z_" in c]
    )
    if len(dann_phase_cols) >= 16:
        print(f"  Using DANN latent trajectory features: {len(dann_phase_cols)} dims")
        return df[dann_phase_cols].values, dann_phase_cols

    # Fall back to raw phase clinical features + treatment trajectory
    raw_cols = sorted(
        [c for c in df.columns
         if any(c.endswith(s) for s in ["_mean", "_last", "_slope", "_std",
                                         "_delta", "_total", "_late_vs_early"])
         and c != "stay_id"]
    )
    print(f"  Fallback to raw trajectory features: {len(raw_cols)} dims")
    return df[raw_cols].values, raw_cols


def run_track_a(
    df: pd.DataFrame,
    k_candidates: list[int],
    seed: int,
    n_init: int,
    min_cluster_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame, int, str]:
    """Track A: K-means + GMM sweep on trajectory features."""
    print("\n[Track A] K-means + GMM clustering sweep ...")

    X_raw, feat_cols = _select_trajectory_features(df)
    n_stays = len(df)

    # Handle NaNs: impute with column median
    nan_mask = np.isnan(X_raw)
    n_nan = int(nan_mask.sum())
    if n_nan > 0:
        print(f"  Imputing {n_nan:,} NaN values with column medians")
        col_medians = np.nanmedian(X_raw, axis=0)
        for j in range(X_raw.shape[1]):
            X_raw[nan_mask[:, j], j] = col_medians[j]

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    print(f"  Feature matrix: {X.shape}")

    # ── Clustering sweep ─────────────────────────────────────────────
    results = []
    labels_dict = {}

    for k in k_candidates:
        # K-means
        km = KMeans(n_clusters=k, n_init=n_init, random_state=seed, max_iter=300)
        km_labels = km.fit_predict(X)
        km_sil = silhouette_score(
            X, km_labels, sample_size=min(10000, n_stays), random_state=seed
        )
        km_ch = calinski_harabasz_score(X, km_labels)

        _, km_counts = np.unique(km_labels, return_counts=True)
        km_min_frac = km_counts.min() / n_stays

        results.append({
            "method": "kmeans", "K": k,
            "silhouette": round(km_sil, 4),
            "calinski_harabasz": round(km_ch, 1),
            "inertia": round(km.inertia_, 1),
            "min_cluster_frac": round(km_min_frac, 4),
        })
        labels_dict[f"kmeans_k{k}"] = km_labels

        # GMM
        gmm = GaussianMixture(
            n_components=k, n_init=n_init, random_state=seed,
            covariance_type="full", max_iter=200,
        )
        gmm_labels = gmm.fit_predict(X)
        gmm_sil = silhouette_score(
            X, gmm_labels, sample_size=min(10000, n_stays), random_state=seed
        )
        gmm_ch = calinski_harabasz_score(X, gmm_labels)
        gmm_bic = gmm.bic(X)

        _, gmm_counts = np.unique(gmm_labels, return_counts=True)
        gmm_min_frac = gmm_counts.min() / n_stays

        results.append({
            "method": "gmm", "K": k,
            "silhouette": round(gmm_sil, 4),
            "calinski_harabasz": round(gmm_ch, 1),
            "bic": round(gmm_bic, 1),
            "aic": round(gmm.aic(X), 1),
            "min_cluster_frac": round(gmm_min_frac, 4),
        })
        labels_dict[f"gmm_k{k}"] = gmm_labels

        print(f"  K={k}: kmeans sil={km_sil:.3f} CH={km_ch:.0f} | "
              f"gmm sil={gmm_sil:.3f} BIC={gmm_bic:.0f}")

    # ── Select best K ────────────────────────────────────────────────
    res_df = pd.DataFrame(results)
    km_res = res_df[res_df["method"] == "kmeans"].copy()
    viable = km_res[km_res["min_cluster_frac"] >= min_cluster_frac]
    if len(viable) > 0:
        best_row = viable.loc[viable["silhouette"].idxmax()]
    else:
        best_row = km_res.loc[km_res["silhouette"].idxmax()]
    best_k = int(best_row["K"])
    best_sil = best_row["silhouette"]
    print(f"  Selected: K={best_k} (kmeans, silhouette={best_sil:.4f})")

    # ── Build label DataFrame ────────────────────────────────────────
    label_df = pd.DataFrame({"stay_id": df["stay_id"]})
    for col_name, lab in labels_dict.items():
        label_df[col_name] = lab
    label_df[f"traj_selected_k{best_k}"] = labels_dict[f"kmeans_k{best_k}"]

    return label_df, res_df, best_k, "kmeans"


def run_track_b_gmvae(
    df: pd.DataFrame,
    interim_dir: Path,
    model_dir: Path,
    cfg: dict,
) -> pd.DataFrame | None:
    """Track B: GMVAE on bin-wise encoder latent sequences."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("  [Track B] PyTorch not available — skipping GMVAE")
        return None

    if not torch.cuda.is_available():
        print("  [Track B] No GPU available — skipping GMVAE")
        return None

    print("\n[Track B] Training GMVAE ...")
    device = torch.device("cuda")

    # Load bin-wise latents
    baseline_path = interim_dir / "50c_latents_baseline_v1_mimic.parquet"
    if not baseline_path.exists():
        print("  Bin-wise latents not found — skipping GMVAE")
        return None

    lat_df = pd.read_parquet(baseline_path)
    z_cols = sorted(
        [c for c in lat_df.columns if c.startswith("z_")],
        key=lambda x: int(x.split("_")[1]),
    )
    n_z = len(z_cols)
    n_bins = 12

    # Build (N, T, D) tensor
    stay_ids = sorted(lat_df["stay_id"].unique())
    n_stays = len(stay_ids)
    sid_to_idx = {s: i for i, s in enumerate(stay_ids)}

    X = np.zeros((n_stays, n_bins, n_z), dtype=np.float32)
    mask = np.zeros((n_stays, n_bins), dtype=np.float32)

    for _, row in lat_df.iterrows():
        sid = int(row["stay_id"])
        b = int(row["bin"])
        if sid in sid_to_idx and 0 <= b < n_bins:
            idx = sid_to_idx[sid]
            X[idx, b, :] = row[z_cols].values.astype(np.float32)
            mask[idx, b] = 1.0

    print(f"  Latent sequences: {X.shape} ({n_stays:,} stays)")

    # ── GMVAE architecture ───────────────────────────────────────────
    gmvae_cfg = cfg.get("gmvae", {})
    K = gmvae_cfg.get("n_components", 4)
    h_dim = gmvae_cfg.get("hidden_dim", 128)
    z_dim = gmvae_cfg.get("gmvae_latent_dim", 32)
    epochs = gmvae_cfg.get("max_epochs", 50)
    batch_size = gmvae_cfg.get("batch_size", 256)
    lr = gmvae_cfg.get("learning_rate", 1e-3)

    class TrajectoryGMVAE(nn.Module):
        """GMVAE: GRU encoder → GMM-prior VAE latent → GRU decoder."""

        def __init__(self, input_dim, hidden_dim, latent_dim, n_components):
            super().__init__()
            self.n_components = n_components
            self.latent_dim = latent_dim

            # Encoder: GRU on input sequences
            self.encoder_gru = nn.GRU(
                input_dim, hidden_dim, num_layers=1,
                batch_first=True, bidirectional=True,
            )
            # q(z|x): mean and logvar from encoded representation
            self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

            # q(c|x): cluster assignment from encoded representation
            self.fc_cluster = nn.Linear(hidden_dim * 2, n_components)

            # GMM prior parameters (learnable)
            self.prior_mu = nn.Parameter(
                torch.randn(n_components, latent_dim) * 0.1
            )
            self.prior_logvar = nn.Parameter(
                torch.zeros(n_components, latent_dim)
            )
            self.prior_logpi = nn.Parameter(
                torch.zeros(n_components)
            )

            # Decoder: project z → GRU → reconstruct
            self.decoder_proj = nn.Linear(latent_dim, hidden_dim)
            self.decoder_gru = nn.GRU(
                hidden_dim, hidden_dim, num_layers=1, batch_first=True,
            )
            self.decoder_out = nn.Linear(hidden_dim, input_dim)

        def encode(self, x, mask_seq):
            """Encode sequence to pooled representation."""
            out, _ = self.encoder_gru(x)  # (B, T, 2H)
            # Masked mean pooling
            m = mask_seq.unsqueeze(-1)  # (B, T, 1)
            pooled = (out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
            return pooled

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z, seq_len):
            """Decode z to reconstructed sequence."""
            h = self.decoder_proj(z).unsqueeze(1).expand(-1, seq_len, -1)
            out, _ = self.decoder_gru(h)
            return self.decoder_out(out)

        def forward(self, x, mask_seq):
            h = self.encode(x, mask_seq)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            z = self.reparameterize(mu, logvar)
            logits_c = self.fc_cluster(h)
            recon = self.decode(z, x.shape[1])
            return recon, mu, logvar, logits_c, z

        def compute_loss(self, x, mask_seq):
            recon, mu, logvar, logits_c, z = self.forward(x, mask_seq)

            # Reconstruction loss (masked)
            m = mask_seq.unsqueeze(-1)
            recon_loss = ((recon - x) ** 2 * m).sum() / m.sum().clamp(min=1)

            # q(c|x) — use log_softmax for numerical stability
            log_q_c = torch.log_softmax(logits_c, dim=-1)  # (B, K)
            q_c = log_q_c.exp()

            # KL divergence with GMM prior
            log_prior_pi = torch.log_softmax(self.prior_logpi, dim=0)  # (K,)
            prior_logvar_clamped = self.prior_logvar.clamp(-10, 10)
            kl = torch.tensor(0.0, device=x.device)
            for k_idx in range(self.n_components):
                # KL(q(z|x) || p(z|c=k))
                kl_k = 0.5 * (
                    prior_logvar_clamped[k_idx] - logvar
                    + (logvar.exp() + (mu - self.prior_mu[k_idx]) ** 2)
                    / prior_logvar_clamped[k_idx].exp()
                    - 1
                ).sum(dim=-1)  # (B,)
                # Weighted by q(c=k|x) — log-space for stability
                kl = kl + (q_c[:, k_idx] * (
                    kl_k - log_prior_pi[k_idx] + log_q_c[:, k_idx]
                )).mean()

            return recon_loss + kl, recon_loss.item(), kl.item()

    model = TrajectoryGMVAE(n_z, h_dim, z_dim, K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Training
    X_t = torch.from_numpy(X).to(device)
    mask_t = torch.from_numpy(mask).to(device)

    best_loss = float("inf")
    model_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_stays, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_stays, batch_size):
            idx = perm[i : i + batch_size]
            x_batch = X_t[idx]
            m_batch = mask_t[idx]

            loss, _, _ = model.compute_loss(x_batch, m_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "loss": best_loss,
                "config": gmvae_cfg,
                "n_components": K,
                "latent_dim": z_dim,
            }, model_dir / "gmvae_best.pt")

    # Extract cluster assignments
    model.eval()
    with torch.no_grad():
        _, _, _, logits_c, z_all = model(X_t, mask_t)
        cluster_probs = torch.softmax(logits_c, dim=-1).cpu().numpy()
        gmvae_labels = cluster_probs.argmax(axis=1)

    gmvae_df = pd.DataFrame({
        "stay_id": stay_ids,
        "gmvae_cluster": gmvae_labels,
    })
    for k_idx in range(K):
        gmvae_df[f"gmvae_prob_k{k_idx}"] = cluster_probs[:, k_idx]

    print(f"  GMVAE: K={K}, best_loss={best_loss:.4f}")
    cluster_counts = np.bincount(gmvae_labels, minlength=K)
    for k_idx in range(K):
        print(f"    Cluster {k_idx}: {cluster_counts[k_idx]:,} "
              f"({cluster_counts[k_idx] / n_stays * 100:.1f}%)")

    return gmvae_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["cluster", "gmvae", "both"], default="cluster",
        help="cluster=Track A only (CPU), gmvae=Track B (GPU), both=A+B",
    )
    args = parser.parse_args()

    proj_cfg = load_config()
    project_root = Path(proj_cfg.get("project_root", "."))
    interim_dir = resolve_path(proj_cfg, "interim")
    reports_dir = resolve_path(proj_cfg, "reports")
    tables_dir = resolve_path(proj_cfg, "tables")
    models_dir = project_root / "models" / "65b_gmvae"
    for d in [interim_dir, reports_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load config
    configs_dir = resolve_path(proj_cfg, "configs")
    cfg_path = configs_dir / "65b_trajectory_cluster_config.yaml"
    cfg = {}
    if cfg_path.exists():
        import yaml
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}

    print("=" * 72)
    print(f"65b_train_gmvae_or_trajcluster — mode={args.mode}")
    print("=" * 72)

    # ── Load trajectory features ──────────────────────────────────────
    print("\n[1/3] Loading trajectory features ...")
    df = pd.read_parquet(interim_dir / "65a_trajectory_features_v1_mimic.parquet")
    n_stays = len(df)
    print(f"  Stays: {n_stays:,}")

    # ── Track A: clustering sweep ─────────────────────────────────────
    print("\n[2/3] Running Track A clustering ...")
    cl_cfg = cfg.get("clustering", {})
    k_candidates = cl_cfg.get("k_candidates", DEFAULT_K_CANDIDATES)
    seed = int(cfg.get("seed", cl_cfg.get("seed", DEFAULT_SEED)))
    n_init = int(cl_cfg.get("n_init", DEFAULT_N_INIT))
    min_cluster_frac = float(cl_cfg.get("min_cluster_frac", 0.05))

    label_df, res_df, best_k, best_method = run_track_a(
        df, k_candidates, seed, n_init, min_cluster_frac
    )

    # ── Track B: GMVAE (optional) ─────────────────────────────────────
    gmvae_df = None
    if args.mode in ("gmvae", "both"):
        print("\n[3/3] Running Track B GMVAE ...")
        gmvae_df = run_track_b_gmvae(df, interim_dir, models_dir, cfg)
        if gmvae_df is not None:
            label_df = label_df.merge(gmvae_df, on="stay_id", how="left")
    else:
        print("\n[3/3] Skipping Track B (use --mode=gmvae to enable)")

    # ── Save outputs ──────────────────────────────────────────────────
    out_path = interim_dir / "65b_trajectory_clusters_v1_mimic.parquet"
    label_df.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"\n  Saved: {out_path} ({len(label_df):,} stays)")

    metrics_path = tables_dir / "65b_trajectory_cluster_metrics.csv"
    res_df.to_csv(metrics_path, index=False)
    print(f"  Saved: {metrics_path}")

    # ── Summary report ────────────────────────────────────────────────
    best_row = res_df[
        (res_df["method"] == "kmeans") & (res_df["K"] == best_k)
    ].iloc[0]

    report_lines = [
        "=" * 72,
        "65b — Trajectory Phenotype Clustering (MIMIC-IV)",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Mode: {args.mode}",
        "",
        "DATA",
        "────",
        f"  Stays: {n_stays:,}",
        "",
        "TRACK A: K-MEANS + GMM SWEEP",
        "─────────────────────────────",
    ]

    for _, row in res_df.iterrows():
        extra = ""
        if "bic" in row and pd.notna(row.get("bic")):
            extra = f" BIC={row['bic']:.0f}"
        report_lines.append(
            f"  {row['method']:6s} K={int(row['K'])} | sil={row['silhouette']:.4f} "
            f"CH={row['calinski_harabasz']:.0f} min_frac={row['min_cluster_frac']:.3f}"
            f"{extra}"
        )

    report_lines.extend([
        "",
        "SELECTED MODEL (Track A)",
        "────────────────────────",
        f"  Method: {best_method}",
        f"  K: {best_k}",
        f"  Silhouette: {best_row['silhouette']:.4f}",
        f"  Label column: traj_selected_k{best_k}",
        "",
        "CLUSTER SIZES (selected)",
        "────────────────────────",
    ])
    selected_labels = label_df[f"traj_selected_k{best_k}"].values
    for c in range(best_k):
        n_c = int((selected_labels == c).sum())
        report_lines.append(
            f"  Cluster {c}: {n_c:,} ({n_c / n_stays * 100:.1f}%)"
        )

    if gmvae_df is not None:
        report_lines.extend([
            "",
            "TRACK B: GMVAE",
            "──────────────",
            f"  See models/65b_gmvae/gmvae_best.pt for checkpoint",
        ])
        gc = label_df["gmvae_cluster"].value_counts().sort_index()
        for k_idx, cnt in gc.items():
            report_lines.append(
                f"  GMVAE cluster {k_idx}: {cnt:,} ({cnt / n_stays * 100:.1f}%)"
            )

    report_lines.extend([
        "",
        "NOTE",
        "────",
        "  Trajectory phenotype is EXPLANATORY ONLY (protocol § 12).",
        "  It does NOT feed back into the primary causal estimand.",
        "",
        "NEXT STEPS",
        "──────────",
        "  65c: bootstrap stability assessment",
        "  65d: trajectory phenotype cards",
    ])

    report = "\n".join(report_lines) + "\n"
    report_path = reports_dir / "65b_trajectory_cluster_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n  Saved: {report_path}")
    print(report)
    print("[65b] Done.")


if __name__ == "__main__":
    main()
