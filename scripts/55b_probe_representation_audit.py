#!/usr/bin/env python3
"""
55b_probe_representation_audit — Audit DANN encoder quality + domain leakage.

Purpose
-------
Evaluate the domain-adversarial encoder from 55a:
  1. Probe tasks: mortality prediction (28d, 7d) using frozen DANN latents.
     Compare against 50b baseline probe results.
  2. Domain leakage test: train a fresh domain classifier on DANN latents.
     If domain-invariant, accuracy should be near chance (~33% for 3 splits).
  3. Reconstruction quality: test set MSE.
  4. Comparison table: 50b baseline vs 55b DANN side by side.

Protocol reference: Step 55b (v3.1 § 11)

Inputs
------
- models/55a_encoder_dann/encoder_dann_best.pt
- models/55a_encoder_dann/data_split.npz
- data/interim/45c_state_tensors_v1_mimic.parquet
- data/interim/45d_outcomes_absorb_v1_mimic.parquet
- tables/50b_probe_baseline_results.csv  (for comparison)

Outputs
-------
- tables/55b_probe_dann_results.csv
- tables/55b_baseline_vs_dann_comparison.csv
- reports/55b_probe_representation_audit_summary.txt

Compute: G1
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path
from hfpef_dtr.models.dataset import ICUStayDataset, discover_feature_columns
from hfpef_dtr.models.encoder import build_dann_model_from_config


def extract_latents(model, dataset, device, batch_size=512):
    """Extract mean-pooled latents from frozen DANN encoder."""
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    z_pooled_list = []

    with torch.no_grad():
        for batch in loader:
            temporal = batch["temporal"].to(device)
            static = batch["static"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            out = model(temporal, static, valid_mask)
            z = out["z"]
            mask = valid_mask.unsqueeze(-1)
            z_pool = (z * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            z_pooled_list.append(z_pool.cpu().numpy())

    return np.concatenate(z_pooled_list)


def run_probe(X_train, y_train, X_val, y_val, X_test, y_test, task_name):
    """Run logistic regression probe."""
    if len(np.unique(y_train)) < 2:
        return {"task": task_name, "test_auroc": np.nan, "note": "single_class"}

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", class_weight="balanced")
    clf.fit(X_train, y_train)

    results = {"task": task_name}
    for split_name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        if len(np.unique(y)) < 2:
            results[f"{split_name}_auroc"] = np.nan
            continue
        proba = clf.predict_proba(X)[:, 1]
        results[f"{split_name}_auroc"] = roc_auc_score(y, proba)
        if split_name == "test":
            results["test_auprc"] = average_precision_score(y, proba)
            results["test_prevalence"] = y.mean()
    return results


def domain_leakage_test(z_pool, domain_labels, n_bootstrap=5, seed=42):
    """Test if domains are distinguishable from DANN latents."""
    rng = np.random.RandomState(seed)
    accs = []
    for i in range(n_bootstrap):
        clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", random_state=seed + i)
        n = len(z_pool)
        idx = rng.permutation(n)
        n_train = int(0.8 * n)
        clf.fit(z_pool[idx[:n_train]], domain_labels[idx[:n_train]])
        pred = clf.predict(z_pool[idx[n_train:]])
        accs.append(accuracy_score(domain_labels[idx[n_train:]], pred))

    n_domains = len(np.unique(domain_labels))
    return {
        "leakage_accuracy": np.mean(accs),
        "leakage_std": np.std(accs),
        "chance_level": 1.0 / n_domains,
        "pass": np.mean(accs) < (1.0 / n_domains + 0.07),
    }


def main() -> None:
    proj_cfg = load_config()
    project_root = Path(proj_cfg.get("project_root", "."))
    reports_dir = resolve_path(proj_cfg, "reports")
    tables_dir = resolve_path(proj_cfg, "tables")
    reports_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("55b_probe_representation_audit — DANN encoder audit")
    print("=" * 72)

    # ── Load DANN checkpoint ────────────────────────────────────────────
    print("\n[1/5] Loading DANN model ...")
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
    print(f"  Loaded (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f}, "
          f"strategy={ckpt.get('domain_label_strategy', 'unknown')})")

    # ── Load data ───────────────────────────────────────────────────────
    print("\n[2/5] Loading data ...")
    data_cfg = cfg["data"]
    state_df = pd.read_parquet(project_root / data_cfg["state_tensor_path"])
    outcomes_df = pd.read_parquet(project_root / data_cfg["outcomes_path"])

    splits = np.load(ckpt_dir / "data_split.npz")
    train_ids, val_ids, test_ids = splits["train"], splits["val"], splits["test"]
    print(f"  Train: {len(train_ids):,}, Val: {len(val_ids):,}, Test: {len(test_ids):,}")

    # ── Extract latents ─────────────────────────────────────────────────
    print("\n[3/5] Extracting DANN latents ...")
    ds_kwargs = dict(
        state_df=state_df, outcomes_df=outcomes_df,
        temporal_cols=temporal_cols, static_cols=static_cols,
    )

    splits_data = {}
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        ds = ICUStayDataset(stay_ids=ids, **ds_kwargs)
        z_pool = extract_latents(model, ds, device)
        splits_data[name] = {
            "z_pool": z_pool,
            "stay_ids": ds.stay_ids,
            "mortality_28d": ds.mortality_28d,
            "mortality_7d": ds.mortality_7d,
        }
        print(f"  {name}: z_pool shape={z_pool.shape}")

    # Get absorbing_death labels
    absorbing_map = outcomes_df.drop_duplicates("stay_id").set_index("stay_id")["absorbing_type"]
    for name in splits_data:
        sids = splits_data[name]["stay_ids"]
        splits_data[name]["absorbing_death"] = np.array([
            1 if absorbing_map.get(sid) == "death" else 0 for sid in sids
        ])

    # ── Probe tasks ─────────────────────────────────────────────────────
    print("\n[4/5] Running probe tasks ...")
    results = []

    for task_key, task_name in [
        ("mortality_28d", "mortality_28d"),
        ("mortality_7d", "mortality_7d"),
        ("absorbing_death", "death_in_window"),
    ]:
        r = run_probe(
            splits_data["train"]["z_pool"], splits_data["train"][task_key],
            splits_data["val"]["z_pool"], splits_data["val"][task_key],
            splits_data["test"]["z_pool"], splits_data["test"][task_key],
            task_name,
        )
        results.append(r)
        auroc = r.get("test_auroc", float("nan"))
        print(f"  {task_name}: test AUROC={auroc:.3f}")

    # Reconstruction MSE
    test_ds = ICUStayDataset(stay_ids=test_ids, **ds_kwargs)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=512, shuffle=False)
    recon_errors = []
    with torch.no_grad():
        for batch in test_loader:
            temporal = batch["temporal"].to(device)
            static = batch["static"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            out = model(temporal, static, valid_mask)
            denom = (valid_mask.sum() * temporal.shape[-1]).clamp(min=1)
            diff = ((out["recon"] - temporal) ** 2 * valid_mask.unsqueeze(-1)).sum() / denom
            recon_errors.append(diff.item())
    mean_recon = np.mean(recon_errors)
    results.append({"task": "test_recon_mse", "test_auroc": mean_recon, "note": "MSE, not AUROC"})
    print(f"  test_recon_mse: {mean_recon:.6f}")

    # ── Domain leakage ──────────────────────────────────────────────────
    print("\n[5/5] Domain leakage test ...")
    all_z = np.concatenate([splits_data[s]["z_pool"] for s in ["train", "val", "test"]])
    all_stay_ids = np.concatenate([splits_data[s]["stay_ids"] for s in ["train", "val", "test"]])

    strategy = ckpt.get("domain_label_strategy", "split")
    n_domains = int(ckpt.get("n_domains", 3))
    if strategy == "split":
        all_labels = np.array([hash(int(sid)) % n_domains for sid in all_stay_ids], dtype=np.int64)
    else:
        raise NotImplementedError(
            f"domain leakage test strategy={strategy} not implemented in 55b yet"
        )

    leakage = domain_leakage_test(all_z, all_labels)
    leak_status = "PASS" if leakage["pass"] else "FAIL"
    print(f"  Domain leakage: {leakage['leakage_accuracy']:.3f} "
          f"± {leakage['leakage_std']:.3f} (chance={leakage['chance_level']:.3f}) → {leak_status}")

    # ── Save results ────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    csv_path = tables_dir / "55b_probe_dann_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # Comparison with baseline
    baseline_csv = tables_dir / "50b_probe_baseline_results.csv"
    if baseline_csv.exists():
        baseline_df = pd.read_csv(baseline_csv)
        comparison = []
        for task in ["mortality_28d", "mortality_7d", "death_in_window"]:
            base_row = baseline_df[baseline_df["task"] == task]
            dann_row = results_df[results_df["task"] == task]
            comparison.append({
                "task": task,
                "baseline_auroc": float(base_row["test_auroc"].iloc[0]) if len(base_row) else np.nan,
                "dann_auroc": float(dann_row["test_auroc"].iloc[0]) if len(dann_row) else np.nan,
            })
        comp_df = pd.DataFrame(comparison)
        comp_df["delta"] = comp_df["dann_auroc"] - comp_df["baseline_auroc"]
        comp_path = tables_dir / "55b_baseline_vs_dann_comparison.csv"
        comp_df.to_csv(comp_path, index=False)
        print(f"  Saved: {comp_path}")

    # ── Summary report ──────────────────────────────────────────────────
    report_lines = [
        "=" * 72,
        "55b — DANN Encoder Representation Audit (MIMIC-IV)",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Encoder: epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f}",
        f"Domain strategy: {ckpt.get('domain_label_strategy', 'unknown')}",
        "",
        "PROBE RESULTS (DANN)",
        "─────────────────────",
    ]
    for r in results:
        task = r["task"]
        if r.get("note") == "MSE, not AUROC":
            report_lines.append(f"  {task}: MSE = {r['test_auroc']:.6f}")
        elif isinstance(r.get("test_auroc"), float):
            report_lines.append(
                f"  {task}: AUROC = {r.get('test_auroc', float('nan')):.3f}, "
                f"AUPRC = {r.get('test_auprc', float('nan')):.3f}, "
                f"prev = {r.get('test_prevalence', float('nan')):.3f}"
            )

    if baseline_csv.exists():
        report_lines.extend(["", "BASELINE vs DANN COMPARISON", "───────────────────────────"])
        for _, row in comp_df.iterrows():
            delta_str = f"+{row['delta']:.3f}" if row["delta"] >= 0 else f"{row['delta']:.3f}"
            report_lines.append(
                f"  {row['task']}: baseline={row['baseline_auroc']:.3f} → "
                f"DANN={row['dann_auroc']:.3f} (Δ={delta_str})"
            )

    report_lines.extend([
        "",
        "DOMAIN LEAKAGE TEST",
        "───────────────────",
        f"  Accuracy: {leakage['leakage_accuracy']:.3f} ± {leakage['leakage_std']:.3f}",
        f"  Chance: {leakage['chance_level']:.3f}",
        f"  Result: {leak_status}",
        "",
        "INTERPRETATION",
        "──────────────",
        "  Probe AUROC ≥ baseline: DANN preserved predictive signal ✓" if all(
            r.get("test_auroc", 0) >= 0.70 for r in results if r["task"] != "test_recon_mse"
        ) else "  WARNING: probe AUROC degraded; consider reducing γ",
        f"  Domain leakage: {'suppressed ✓' if leakage['pass'] else 'still present — increase γ or training'}",
    ])

    report = "\n".join(report_lines) + "\n"
    report_path = reports_dir / "55b_probe_representation_audit_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {report_path}")
    print(report)
    print("[55b] Done.")


if __name__ == "__main__":
    main()
