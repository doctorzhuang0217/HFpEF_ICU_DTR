#!/usr/bin/env python3
"""
80g_bootstrap_interaction_ci — Bootstrap CIs for HFpEF × shock 2×2 interaction.

Purpose
-------
Compute bootstrap confidence intervals and interaction p-value for the
non-shock stratum HFpEF-specific signal (2.8× ratio from 80g point estimates).

Addresses peer review feedback: "non-shock HFpEF n=163 needs uncertainty
quantification and formal interaction test."

Design:
  - Primary model only: 80c G-Transformer DM rollout (ow=1)
  - B=500 patient-level bootstrap resamples within each 2×2 cell
  - Compute: per-cell ATE CI, ratio CI, interaction (difference-in-differences) CI + p

Protocol reference: Step 80g supplement (peer review response)

Inputs
------
- models/80c_gtransformer_rollout_ow1_best.pt
- data/processed/phenotype_labels_frozen_v1_mimic.parquet
- configs/80_step_c_training_v1.yaml

Outputs
-------
- tables/80g_bootstrap_interaction_ci.csv
- reports/80g_bootstrap_interaction_ci_summary.txt

Compute: C3-C4 (CPU forward passes × 500 bootstrap, ~2-4h)
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hfpef_dtr.config import load_config, resolve_path
from hfpef_dtr.models.dtr_dataset import DTRDataset, load_dtr_data
from hfpef_dtr.models.dataset import discover_feature_columns, make_train_val_test_split
from hfpef_dtr.models.ope import evaluate_dm

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "tables"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"

# Bootstrap config
B = 500          # number of bootstrap resamples
SEED = 42        # base seed
CI_LEVEL = 0.95  # 95% CI


def load_gtransformer(config, d_temporal, d_static, n_actions, device):
    """Load the primary 80c G-Transformer model."""
    import importlib
    mod = importlib.import_module("hfpef_dtr.models.gtransformer")
    builder_fn = getattr(mod, "build_gtransformer_from_config")
    model_cfg = config["models"]["80c_gtransformer"]
    grid_cfg = config["grid"]
    model = builder_fn(
        d_temporal=d_temporal, d_static=d_static,
        n_actions=n_actions, model_cfg=model_cfg, grid_cfg=grid_cfg,
    )
    ckpt_path = MODELS_DIR / "80c_gtransformer_rollout_ow1_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate_subgroup(model, state_df, outcomes_df, propensity_df,
                      temporal_cols, static_cols, merge_map,
                      stay_ids, config, device):
    """Evaluate DM rollout ATE for a subgroup."""
    if len(stay_ids) < 20:
        return None
    sub_ds = DTRDataset(
        state_df=state_df,
        outcomes_df=outcomes_df,
        propensity_df=propensity_df,
        stay_ids=stay_ids,
        temporal_cols=temporal_cols,
        static_cols=static_cols,
        merge_map=merge_map,
        n_bins=config["grid"]["n_bins"],
        action_vocab=config["action"]["valid_merged_indices"],
    )
    try:
        result = evaluate_dm(model, sub_ds, device, use_rollout=True)
        return result["ate"]
    except Exception:
        return None


def main() -> None:
    t0_wall = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("80g — Bootstrap CIs for HFpEF × Shock 2×2 Interaction")
    print(f"B={B}, CI={CI_LEVEL*100:.0f}%, seed={SEED}")
    print("=" * 72)

    # ── 1. Load config and data ──────────────────────────────────────
    print("\n[1] Loading data ...")
    config_path = ROOT / "configs" / "80_step_c_training_v1.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    state_df, outcomes_df, propensity_df, temporal_cols, static_cols, merge_map = (
        load_dtr_data(ROOT, config)
    )

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
    print(f"  Test stays: {len(test_ids):,}")

    # ── 2. Load phenotype labels ─────────────────────────────────────
    print("\n[2] Loading phenotype labels ...")
    pheno_path = ROOT / config["data"]["phenotype_labels"]
    pheno_df = pd.read_parquet(pheno_path)
    pheno_test = pheno_df[pheno_df["stay_id"].isin(set(test_ids))].copy()

    # Define 4 key cells
    cells = {
        ("hfpef", "non_shock"): None,
        ("hfpef", "shock_predominant"): None,
        ("no_hf", "non_shock"): None,
        ("no_hf", "shock_predominant"): None,
    }
    cell_ids = {}
    for (hf, shock) in cells:
        mask = (pheno_test["hf_subtype"] == hf) & (pheno_test["rule_phenotype_2class"] == shock)
        ids = pheno_test.loc[mask, "stay_id"].values
        cell_ids[(hf, shock)] = ids
        print(f"  {hf:12s} × {shock:20s}: n={len(ids)}")

    # ── 3. Load model ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_temporal = len(temporal_cols)
    d_static = len(static_cols)
    n_actions = len(config["action"]["valid_merged_indices"])
    print(f"\n[3] Loading G-Transformer (device={device}) ...")
    model = load_gtransformer(config, d_temporal, d_static, n_actions, device)

    # ── 4. Point estimates (original) ────────────────────────────────
    print("\n[4] Point estimates ...")
    point_ates = {}
    for (hf, shock), ids in cell_ids.items():
        ate = evaluate_subgroup(
            model, state_df, outcomes_df, propensity_df,
            temporal_cols, static_cols, merge_map,
            ids, config, device,
        )
        point_ates[(hf, shock)] = ate
        print(f"  {hf:12s} × {shock:20s}: ATE={ate:+.4f}" if ate else
              f"  {hf:12s} × {shock:20s}: FAILED")

    # ── 5. Bootstrap ─────────────────────────────────────────────────
    print(f"\n[5] Bootstrap ({B} resamples) ...")
    rng = np.random.RandomState(SEED)
    boot_ates = {k: [] for k in cells}

    for b in tqdm(range(B), desc="Bootstrap", unit="rep"):
        for (hf, shock), ids in cell_ids.items():
            # Patient-level resample with replacement
            boot_ids = rng.choice(ids, size=len(ids), replace=True)
            ate = evaluate_subgroup(
                model, state_df, outcomes_df, propensity_df,
                temporal_cols, static_cols, merge_map,
                boot_ids, config, device,
            )
            if ate is not None:
                boot_ates[(hf, shock)].append(ate)
            else:
                boot_ates[(hf, shock)].append(np.nan)

    # ── 6. Compute CIs and interaction ───────────────────────────────
    print("\n[6] Computing CIs and interaction test ...")
    alpha = 1 - CI_LEVEL
    results = []

    for (hf, shock) in cells:
        arr = np.array(boot_ates[(hf, shock)])
        arr_valid = arr[~np.isnan(arr)]
        ci_lo = np.percentile(arr_valid, 100 * alpha / 2)
        ci_hi = np.percentile(arr_valid, 100 * (1 - alpha / 2))
        se = np.std(arr_valid, ddof=1)
        results.append({
            "hf_subtype": hf, "shock_stratum": shock,
            "n_stays": len(cell_ids[(hf, shock)]),
            "ate_point": point_ates.get((hf, shock), np.nan),
            "ate_boot_mean": np.mean(arr_valid),
            "ate_ci_lo": ci_lo, "ate_ci_hi": ci_hi,
            "ate_se": se, "n_valid_boot": len(arr_valid),
        })

    # Interaction: difference-in-differences
    # DiD = (ATE_hfpef_nonshock - ATE_nohf_nonshock) - (ATE_hfpef_shock - ATE_nohf_shock)
    # Simpler: just test non-shock difference = ATE_hfpef_nonshock - ATE_nohf_nonshock
    arr_hfpef_ns = np.array(boot_ates[("hfpef", "non_shock")])
    arr_nohf_ns = np.array(boot_ates[("no_hf", "non_shock")])
    arr_hfpef_sh = np.array(boot_ates[("hfpef", "shock_predominant")])
    arr_nohf_sh = np.array(boot_ates[("no_hf", "shock_predominant")])

    # Non-shock difference
    diff_nonshock = arr_hfpef_ns - arr_nohf_ns
    diff_nonshock_valid = diff_nonshock[~np.isnan(diff_nonshock)]
    diff_ns_ci_lo = np.percentile(diff_nonshock_valid, 100 * alpha / 2)
    diff_ns_ci_hi = np.percentile(diff_nonshock_valid, 100 * (1 - alpha / 2))
    # p-value: proportion of bootstrap where difference >= 0
    p_nonshock_diff = np.mean(diff_nonshock_valid >= 0)

    # Ratio: HFpEF/no_hf in non-shock stratum
    ratio_ns = arr_hfpef_ns / arr_nohf_ns
    ratio_ns_valid = ratio_ns[~np.isnan(ratio_ns) & ~np.isinf(ratio_ns)]
    ratio_ns_ci_lo = np.percentile(ratio_ns_valid, 100 * alpha / 2) if len(ratio_ns_valid) > 10 else np.nan
    ratio_ns_ci_hi = np.percentile(ratio_ns_valid, 100 * (1 - alpha / 2)) if len(ratio_ns_valid) > 10 else np.nan

    # Full DiD
    did = (arr_hfpef_ns - arr_nohf_ns) - (arr_hfpef_sh - arr_nohf_sh)
    did_valid = did[~np.isnan(did)]
    did_ci_lo = np.percentile(did_valid, 100 * alpha / 2) if len(did_valid) > 10 else np.nan
    did_ci_hi = np.percentile(did_valid, 100 * (1 - alpha / 2)) if len(did_valid) > 10 else np.nan
    p_did = np.mean(did_valid >= 0) if len(did_valid) > 10 else np.nan

    # ── 7. Save results ──────────────────────────────────────────────
    print("\n[7] Saving results ...")
    results_df = pd.DataFrame(results)
    csv_path = TABLES_DIR / "80g_bootstrap_interaction_ci.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # ── 8. Report ────────────────────────────────────────────────────
    elapsed = time.time() - t0_wall
    report_lines = [
        "=" * 72,
        "80g — Bootstrap CIs for HFpEF × Shock 2×2 Interaction",
        "=" * 72,
        f"Generated: {ts}",
        f"Model: G-Transformer DM rollout (ow=1)",
        f"Bootstrap: B={B}, seed={SEED}, CI={CI_LEVEL*100:.0f}%",
        f"Wall time: {elapsed/60:.1f} min",
        "",
        "PER-CELL RESULTS",
        "─────────────────",
    ]

    for r in results:
        report_lines.append(
            f"  {r['hf_subtype']:12s} × {r['shock_stratum']:20s}: "
            f"n={r['n_stays']:4d}  ATE={r['ate_point']:+.4f}  "
            f"CI=[{r['ate_ci_lo']:+.4f}, {r['ate_ci_hi']:+.4f}]  "
            f"SE={r['ate_se']:.4f}  (valid={r['n_valid_boot']})"
        )

    report_lines.extend([
        "",
        "INTERACTION TESTS",
        "─────────────────",
        f"  Non-shock HFpEF–no_hf difference:",
        f"    Point: {point_ates.get(('hfpef','non_shock'),0) - point_ates.get(('no_hf','non_shock'),0):+.4f}",
        f"    CI: [{diff_ns_ci_lo:+.4f}, {diff_ns_ci_hi:+.4f}]",
        f"    p(diff≥0): {p_nonshock_diff:.4f}",
        "",
        f"  Non-shock ratio (HFpEF/no_hf):",
        f"    Point: {point_ates.get(('hfpef','non_shock'),1) / point_ates.get(('no_hf','non_shock'),1):.2f}×",
        f"    CI: [{ratio_ns_ci_lo:.2f}×, {ratio_ns_ci_hi:.2f}×]",
        "",
        f"  Difference-in-differences (DiD):",
        f"    Formula: (HFpEF_ns - noHF_ns) - (HFpEF_sh - noHF_sh)",
        f"    CI: [{did_ci_lo:+.4f}, {did_ci_hi:+.4f}]",
        f"    p(DiD≥0): {p_did:.4f}",
        "",
        "INTERPRETATION",
        "──────────────",
        "  If non-shock difference CI excludes 0 and p < 0.05:",
        "    → HFpEF-specific signal statistically significant",
        "  If DiD CI excludes 0: interaction effect confirmed",
        "    (HFpEF-specific benefit differs between shock strata)",
    ])

    report_path = REPORTS_DIR / "80g_bootstrap_interaction_ci_summary.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"  Report: {report_path}")
    print(f"\n80g bootstrap CI COMPLETE. ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
