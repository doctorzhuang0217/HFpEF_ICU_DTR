#!/usr/bin/env python3
"""
50d_transfer_eval_baseline — Evaluate baseline encoder transfer readiness.

Purpose
-------
Assess whether the baseline encoder latent space is suitable for
cross-database generalization by running internal diagnostics on MIMIC-IV
and (when available) applying the frozen encoder to eICU/HiRID data.

Diagnostics performed:
  1. Split leakage test: can a classifier distinguish train/val/test
     from latents alone? (should fail → ~33% accuracy)
  2. Latent distribution analysis: per-split z statistics, KL divergence
  3. HF-subtype separability: do latents cluster by comparator_group?
  4. [External] Apply frozen encoder to eICU/HiRID and compare latent
     distributions (when external state tensors become available)

Protocol reference: Step 50d (v3.1 § 11, Assumption 9 transportability)

Inputs
------
- models/50a_encoder_baseline/encoder_baseline_best.pt
- models/50a_encoder_baseline/data_split.npz
- data/interim/50c_latents_baseline_pooled_v1_mimic.parquet
- data/staging/25f_primary_cohort_candidate_v1_mimic.parquet  (for comparator_group)
- [optional] data/interim/45c_state_tensors_v1_eicu.parquet
- [optional] data/interim/45c_state_tensors_v1_hirid.parquet

Outputs
-------
- tables/50d_transfer_eval_baseline.csv
- reports/50d_transfer_eval_baseline_summary.txt

Compute: C2
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path


def split_leakage_test(
    z_pool: np.ndarray,
    split_labels: np.ndarray,
    n_bootstrap: int = 5,
    seed: int = 42,
) -> dict:
    """
    Test if train/val/test split is predictable from latent features.

    If the encoder memorizes split-specific patterns, a classifier can
    distinguish which split a latent came from. We expect ~33% accuracy
    (3-class chance level).
    """
    rng = np.random.RandomState(seed)
    accs = []
    for i in range(n_bootstrap):
        clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", random_state=seed + i)
        # Use 80% for training, 20% for testing within the combined data
        n = len(z_pool)
        idx = rng.permutation(n)
        n_train = int(0.8 * n)
        train_idx, test_idx = idx[:n_train], idx[n_train:]

        clf.fit(z_pool[train_idx], split_labels[train_idx])
        pred = clf.predict(z_pool[test_idx])
        accs.append(accuracy_score(split_labels[test_idx], pred))

    return {
        "leakage_accuracy_mean": np.mean(accs),
        "leakage_accuracy_std": np.std(accs),
        "chance_level": 1.0 / len(np.unique(split_labels)),
        "pass": np.mean(accs) < 0.40,  # should be close to 33%
    }


def latent_distribution_analysis(
    z_pool: np.ndarray,
    split_labels: np.ndarray,
    split_names: list[str],
) -> dict:
    """Compare latent distributions across splits."""
    results = {}
    unique_splits = sorted(np.unique(split_labels))

    for s in unique_splits:
        mask = split_labels == s
        z_s = z_pool[mask]
        name = split_names[s] if s < len(split_names) else f"split_{s}"
        results[f"{name}_z_norm_mean"] = float(np.linalg.norm(z_s, axis=1).mean())
        results[f"{name}_z_norm_std"] = float(np.linalg.norm(z_s, axis=1).std())
        results[f"{name}_z_mean_range"] = f"[{z_s.mean(0).min():.3f}, {z_s.mean(0).max():.3f}]"
        results[f"{name}_n"] = int(mask.sum())

    # Pairwise KL divergence on per-dim marginals (discretized)
    kl_pairs = []
    for i in range(len(unique_splits)):
        for j in range(i + 1, len(unique_splits)):
            z_i = z_pool[split_labels == unique_splits[i]]
            z_j = z_pool[split_labels == unique_splits[j]]
            kl_sum = 0.0
            for d in range(z_pool.shape[1]):
                bins = np.linspace(
                    min(z_i[:, d].min(), z_j[:, d].min()) - 0.1,
                    max(z_i[:, d].max(), z_j[:, d].max()) + 0.1,
                    50,
                )
                p = np.histogram(z_i[:, d], bins=bins, density=True)[0] + 1e-8
                q = np.histogram(z_j[:, d], bins=bins, density=True)[0] + 1e-8
                p /= p.sum()
                q /= q.sum()
                kl_sum += entropy(p, q)
            n_i = split_names[unique_splits[i]]
            n_j = split_names[unique_splits[j]]
            kl_pairs.append((n_i, n_j, kl_sum / z_pool.shape[1]))

    results["kl_pairs"] = kl_pairs
    return results


def hf_subtype_separability(
    z_pool: np.ndarray,
    stay_ids: np.ndarray,
    cohort_df: pd.DataFrame,
) -> dict:
    """Test if latents encode HF-subtype information (expected: yes)."""
    cohort_indexed = cohort_df.set_index("stay_id")
    labels = []
    valid_idx = []
    for i, sid in enumerate(stay_ids):
        if sid in cohort_indexed.index:
            cg = cohort_indexed.loc[sid, "comparator_group"]
            if isinstance(cg, str):
                labels.append(cg)
                valid_idx.append(i)

    if len(valid_idx) < 100:
        return {"note": "insufficient cohort overlap"}

    labels = np.array(labels)
    z_valid = z_pool[valid_idx]

    # Binary: HFpEF vs rest
    y_hfpef = (labels == "hfpef").astype(int)
    if len(np.unique(y_hfpef)) < 2:
        return {"note": "single class for HFpEF binary"}

    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", class_weight="balanced")
    # Simple 5-fold-like holdout
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(clf, z_valid, y_hfpef, cv=5, scoring="roc_auc")

    return {
        "hfpef_vs_rest_auroc_mean": float(scores.mean()),
        "hfpef_vs_rest_auroc_std": float(scores.std()),
        "n_hfpef": int(y_hfpef.sum()),
        "n_total": len(y_hfpef),
        "group_distribution": {g: int((labels == g).sum()) for g in np.unique(labels)},
    }


def try_external_eval(
    project_root: Path,
    enc_cfg: dict,
    device_str: str = "cpu",
) -> dict | None:
    """
    Attempt to load external database state tensors and apply frozen encoder.
    Returns None if external data not available.
    """
    import torch
    from hfpef_dtr.models.dataset import ICUStayDataset, discover_feature_columns
    from hfpef_dtr.models.encoder import build_model_from_config

    external_results = {}

    for db_name, tensor_path in [
        ("eicu", "data/interim/45c_state_tensors_v1_eicu.parquet"),
        ("hirid", "data/interim/45c_state_tensors_v1_hirid.parquet"),
    ]:
        full_path = project_root / tensor_path
        if not full_path.exists():
            external_results[db_name] = {"status": "not_available", "path": tensor_path}
            continue

        print(f"\n  [{db_name}] Loading external state tensors ...")
        ext_df = pd.read_parquet(full_path)
        ext_stays = ext_df["stay_id"].unique()
        print(f"  [{db_name}] {len(ext_stays):,} stays")

        # Load encoder
        ckpt_dir = project_root / "models" / "50a_encoder_baseline"
        device = torch.device(device_str)
        ckpt = torch.load(
            ckpt_dir / "encoder_baseline_best.pt",
            map_location=device,
            weights_only=True,
        )
        temporal_cols = ckpt["temporal_cols"]
        static_cols = ckpt["static_cols"]

        # Check column overlap
        available_cols = set(ext_df.columns)
        missing_temporal = [c for c in temporal_cols if c not in available_cols]
        missing_static = [c for c in static_cols if c not in available_cols]

        if missing_temporal:
            external_results[db_name] = {
                "status": "column_mismatch",
                "missing_temporal": missing_temporal[:10],
                "missing_static": missing_static[:10],
                "total_missing": len(missing_temporal) + len(missing_static),
            }
            continue

        # Build dataset and extract latents
        model = build_model_from_config(
            d_temporal=ckpt["d_temporal"],
            d_static=ckpt["d_static"],
            model_cfg=enc_cfg["model"],
            pretext_cfg=enc_cfg["pretext"],
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Load outcomes if available
        outcomes_path = project_root / f"data/interim/45d_outcomes_absorb_v1_{db_name}.parquet"
        outcomes_df = pd.read_parquet(outcomes_path) if outcomes_path.exists() else None

        dataset = ICUStayDataset(
            state_df=ext_df, outcomes_df=outcomes_df,
            stay_ids=ext_stays,
            temporal_cols=temporal_cols, static_cols=static_cols,
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=512, shuffle=False, num_workers=2,
        )

        z_pooled = []
        with torch.no_grad():
            for batch in loader:
                temporal = batch["temporal"].to(device)
                static = batch["static"].to(device)
                valid_mask = batch["valid_mask"].to(device)
                out = model(temporal, static, valid_mask)
                z = out["z"]
                mask = valid_mask.unsqueeze(-1)
                z_pool = (z * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                z_pooled.append(z_pool.cpu().numpy())

        z_ext = np.concatenate(z_pooled)
        z_norms = np.linalg.norm(z_ext, axis=1)

        external_results[db_name] = {
            "status": "success",
            "n_stays": len(ext_stays),
            "z_norm_mean": float(z_norms.mean()),
            "z_norm_std": float(z_norms.std()),
            "z_mean_range": f"[{z_ext.mean(0).min():.3f}, {z_ext.mean(0).max():.3f}]",
            "collapsed_dims": int((z_ext.std(0) < 0.01).sum()),
        }
        print(f"  [{db_name}] z-norm mean={z_norms.mean():.3f}, collapsed={int((z_ext.std(0) < 0.01).sum())}")

    return external_results if any(
        v.get("status") != "not_available" for v in external_results.values()
    ) else None


def main() -> None:
    proj_cfg = load_config()
    project_root = Path(proj_cfg.get("project_root", "."))
    reports_dir = resolve_path(proj_cfg, "reports")
    tables_dir = resolve_path(proj_cfg, "tables")
    reports_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("50d_transfer_eval_baseline — Transfer readiness evaluation")
    print("=" * 72)

    # ── Load latents and splits ─────────────────────────────────────────
    print("\n[1/4] Loading latents and splits ...")
    pool_path = project_root / "data/interim/50c_latents_baseline_pooled_v1_mimic.parquet"
    df_pool = pd.read_parquet(pool_path)
    z_cols = [c for c in df_pool.columns if c.startswith("z_")]
    z_pool = df_pool[z_cols].values
    stay_ids = df_pool["stay_id"].values
    print(f"  Pooled latents: {z_pool.shape}")

    ckpt_dir = project_root / "models" / "50a_encoder_baseline"
    splits = np.load(ckpt_dir / "data_split.npz")
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    split_labels = np.array([
        0 if sid in train_set else 1 if sid in val_set else 2
        for sid in stay_ids
    ])
    split_names = ["train", "val", "test"]
    print(f"  Splits: train={len(train_set):,}, val={len(val_set):,}, test={len(test_set):,}")

    # ── Split leakage test ──────────────────────────────────────────────
    print("\n[2/4] Split leakage test ...")
    leakage = split_leakage_test(z_pool, split_labels)
    status = "PASS" if leakage["pass"] else "FAIL"
    print(f"  Accuracy: {leakage['leakage_accuracy_mean']:.3f} "
          f"± {leakage['leakage_accuracy_std']:.3f} "
          f"(chance={leakage['chance_level']:.3f}) → {status}")

    # ── Distribution analysis ───────────────────────────────────────────
    print("\n[3/4] Latent distribution analysis ...")
    dist = latent_distribution_analysis(z_pool, split_labels, split_names)
    for pair_name_i, pair_name_j, kl in dist["kl_pairs"]:
        print(f"  KL({pair_name_i}||{pair_name_j}) mean per-dim = {kl:.4f}")

    # ── HF subtype separability ─────────────────────────────────────────
    print("\n[4/4] HF-subtype separability ...")
    cohort_path = project_root / "data/staging/25f_primary_cohort_candidate_v1_mimic.parquet"
    hf_sep = {}
    if cohort_path.exists():
        cohort_df = pd.read_parquet(cohort_path, columns=["stay_id", "comparator_group"])
        hf_sep = hf_subtype_separability(z_pool, stay_ids, cohort_df)
        if "hfpef_vs_rest_auroc_mean" in hf_sep:
            print(f"  HFpEF vs rest AUROC: {hf_sep['hfpef_vs_rest_auroc_mean']:.3f} "
                  f"± {hf_sep['hfpef_vs_rest_auroc_std']:.3f}")
            if hf_sep.get("group_distribution"):
                for g, n in hf_sep["group_distribution"].items():
                    print(f"    {g}: {n}")
        else:
            print(f"  {hf_sep.get('note', 'N/A')}")
    else:
        print("  Cohort file not found — skipping HF separability test")
        hf_sep = {"note": "cohort file not found"}

    # ── External database evaluation ────────────────────────────────────
    import yaml
    enc_cfg_path = project_root / "configs/50a_encoder_baseline_config.yaml"
    with open(enc_cfg_path) as f:
        enc_cfg = yaml.safe_load(f)

    ext_results = try_external_eval(project_root, enc_cfg)
    if ext_results is None:
        print("\n  External databases: not available (eICU/HiRID state tensors not built yet)")

    # ── Save results ────────────────────────────────────────────────────
    rows = [
        {"metric": "split_leakage_accuracy", "value": leakage["leakage_accuracy_mean"]},
        {"metric": "split_leakage_std", "value": leakage["leakage_accuracy_std"]},
        {"metric": "split_leakage_pass", "value": int(leakage["pass"])},
    ]
    for pair_i, pair_j, kl in dist["kl_pairs"]:
        rows.append({"metric": f"kl_{pair_i}_vs_{pair_j}", "value": kl})
    if "hfpef_vs_rest_auroc_mean" in hf_sep:
        rows.append({"metric": "hfpef_vs_rest_auroc", "value": hf_sep["hfpef_vs_rest_auroc_mean"]})
    if ext_results:
        for db, info in ext_results.items():
            rows.append({"metric": f"ext_{db}_status", "value": info["status"]})
            if info["status"] == "success":
                rows.append({"metric": f"ext_{db}_z_norm_mean", "value": info["z_norm_mean"]})

    csv_path = tables_dir / "50d_transfer_eval_baseline.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # ── Summary report ──────────────────────────────────────────────────
    report_lines = [
        "=" * 72,
        "50d — Baseline Encoder Transfer Readiness (MIMIC-IV)",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "SPLIT LEAKAGE TEST",
        "──────────────────",
        f"  Accuracy: {leakage['leakage_accuracy_mean']:.3f} "
        f"± {leakage['leakage_accuracy_std']:.3f}",
        f"  Chance level: {leakage['chance_level']:.3f}",
        f"  Result: {status}",
        "  Interpretation: encoder does NOT memorize split-specific patterns"
        if leakage["pass"] else
        "  WARNING: encoder may have split-dependent artifacts",
        "",
        "LATENT DISTRIBUTION (per split)",
        "────────────────────────────────",
    ]
    for name in split_names:
        n_key = f"{name}_n"
        norm_key = f"{name}_z_norm_mean"
        if n_key in dist:
            report_lines.append(
                f"  {name}: n={dist[n_key]:,}, z-norm={dist[norm_key]:.3f} "
                f"± {dist[f'{name}_z_norm_std']:.3f}, "
                f"mean-range={dist[f'{name}_z_mean_range']}"
            )

    report_lines.append("")
    report_lines.append("KL DIVERGENCE (mean per-dim)")
    report_lines.append("────────────────────────────")
    for pair_i, pair_j, kl in dist["kl_pairs"]:
        report_lines.append(f"  KL({pair_i}||{pair_j}) = {kl:.4f}")
    report_lines.append("  Interpretation: KL < 0.05 = well-matched; > 0.20 = distribution shift")

    report_lines.extend(["", "HF-SUBTYPE SEPARABILITY", "───────────────────────"])
    if "hfpef_vs_rest_auroc_mean" in hf_sep:
        report_lines.append(
            f"  HFpEF vs rest AUROC: {hf_sep['hfpef_vs_rest_auroc_mean']:.3f} "
            f"± {hf_sep['hfpef_vs_rest_auroc_std']:.3f}"
        )
        report_lines.append(
            f"  n_HFpEF={hf_sep['n_hfpef']}, n_total={hf_sep['n_total']}"
        )
        if hf_sep.get("group_distribution"):
            for g, n in sorted(hf_sep["group_distribution"].items()):
                report_lines.append(f"    {g}: {n}")
        report_lines.append(
            "  Interpretation: AUROC > 0.70 = encoder captures HF-subtype signal"
        )
    else:
        report_lines.append(f"  {hf_sep.get('note', 'N/A')}")

    report_lines.extend(["", "EXTERNAL DATABASE EVALUATION", "────────────────────────────"])
    if ext_results:
        for db, info in ext_results.items():
            if info["status"] == "success":
                report_lines.append(
                    f"  {db}: z-norm={info['z_norm_mean']:.3f} ± {info['z_norm_std']:.3f}, "
                    f"collapsed={info['collapsed_dims']}/{z_pool.shape[1]}"
                )
            else:
                report_lines.append(f"  {db}: {info['status']}")
    else:
        report_lines.append("  eICU: not available (45c state tensors not built)")
        report_lines.append("  HiRID: not available (45c state tensors not built)")
        report_lines.append("  Action: complete eICU/HiRID 10x-45c pipeline, then rerun 50d")

    report_lines.extend([
        "",
        "TRANSFER READINESS SUMMARY",
        "──────────────────────────",
        f"  Split leakage: {status}",
    ])
    if "hfpef_vs_rest_auroc_mean" in hf_sep:
        hf_status = "PASS" if hf_sep["hfpef_vs_rest_auroc_mean"] > 0.60 else "MARGINAL"
        report_lines.append(f"  HF-subtype signal: {hf_status}")
    report_lines.append(
        "  External transfer: PENDING (awaiting eICU/HiRID state tensors)"
        if not ext_results else
        "  External transfer: EVALUATED"
    )

    report = "\n".join(report_lines) + "\n"
    report_path = reports_dir / "50d_transfer_eval_baseline_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {report_path}")
    print(report)
    print("[50d] Done.")


if __name__ == "__main__":
    main()
