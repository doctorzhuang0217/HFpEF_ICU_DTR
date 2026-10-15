#!/usr/bin/env python3
"""
85a_edq_early_discharge_sensitivity — Early Discharge Quantitative sensitivity.

Purpose
-------
Test robustness of Step C ATE estimates to early discharge handling.

Early discharge (27.3% of stays) creates informative censoring: patients
discharged alive before the 24h observation window ends have truncated
treatment trajectories. If ATE estimates change substantially under
different early-discharge treatments, the finding is fragile.

Scenarios tested:
  S0 (baseline): Include all stays as-is (original Step C result)
  S1 (exclude):  Exclude all early-discharge stays entirely
  S2 (best-case): Recode early-discharge as mortality_28d=0 (alive assumption)
  S3 (worst-case): Recode early-discharge as mortality_28d=1 (death assumption)
  S4 (Δ sweep): Apply increasing exclusion of early bins
      (exclude stays discharged before bin 3, then bin 6, then bin 9)

For each scenario, re-evaluate ATE using 80c G-Transformer DM (primary)
with bootstrap CIs. Also compute subgroup ATEs for HFpEF vs no_hf.

Protocol reference: Step 85a (sensitivity layer)
Compute: C3 (bootstrap-heavy CPU, ~16 cores recommended)

Inputs
------
- configs/80_step_c_training_v1.yaml
- models/80c_gtransformer_rollout_ow1_best.pt
- data/interim/45c_state_tensors_v1_mimic.parquet
- data/interim/45d_outcomes_absorb_v1_mimic.parquet
- data/interim/70b_propensity_coarse3_v1_mimic.parquet
- data/processed/phenotype_labels_frozen_v1_mimic.parquet

Outputs
-------
- tables/85a_edq_sensitivity.csv
- reports/85a_edq_sensitivity_summary.txt
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

from hfpef_dtr.models.dtr_dataset import DTRDataset, load_dtr_data
from hfpef_dtr.models.dataset import make_train_val_test_split
from hfpef_dtr.models.ope import evaluate_dm, bootstrap_ope

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "tables"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"

N_BOOTSTRAP = 500
BOOT_SEED = 7777
CI_LEVEL = 0.95


def load_primary_model(config, d_temporal, d_static, n_actions, device):
    """Load 80c G-Transformer (primary model) from checkpoint."""
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
    if not ckpt_path.exists():
        print(f"  FATAL: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def build_scenario_dataset(
    state_df, outcomes_df, propensity_df,
    temporal_cols, static_cols, merge_map,
    stay_ids, config, scenario_outcomes=None,
):
    """Build DTRDataset, optionally with modified outcomes."""
    out_df = scenario_outcomes if scenario_outcomes is not None else outcomes_df
    ds = DTRDataset(
        state_df=state_df,
        outcomes_df=out_df,
        propensity_df=propensity_df,
        stay_ids=stay_ids,
        temporal_cols=temporal_cols,
        static_cols=static_cols,
        merge_map=merge_map,
        n_bins=config["grid"]["n_bins"],
    )
    if "action_vocab" in config:
        action_vocab = config["action_vocab"]
        ds.action_vocab = action_vocab
        ds.n_actions = len(action_vocab)
        ds._action_to_contiguous = {a: i for i, a in enumerate(action_vocab)}
    return ds


def evaluate_scenario(model, dataset, device, n_boot=N_BOOTSTRAP, boot_seed=BOOT_SEED):
    """Evaluate DM ATE with bootstrap CIs."""
    # Point estimate
    point = evaluate_dm(model, dataset, device, use_rollout=True)

    # Bootstrap CIs
    def _eval_fn(ds, **kw):
        return evaluate_dm(model, ds, device, use_rollout=True)

    boot = bootstrap_ope(
        _eval_fn, dataset,
        n_resamples=n_boot, seed=boot_seed, confidence=CI_LEVEL,
    )

    return {
        "ate": point["ate"],
        "v_observed": point["v_observed"],
        "v_rule": point["v_rule"],
        "ate_ci_lo": boot.get("ate_ci_lo", np.nan),
        "ate_ci_hi": boot.get("ate_ci_hi", np.nan),
        "n_boot_success": boot.get("n_success", 0),
    }


def main() -> None:
    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("85a — Early Discharge Quantitative (EDQ) Sensitivity")
    print("=" * 72)

    # ── 1. Load config and data ──────────────────────────────────────
    print("\n[1] Loading data ...")
    config_path = ROOT / "configs" / "80_step_c_training_v1.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    state_df, outcomes_df, propensity_df, temporal_cols, static_cols, merge_map = (
        load_dtr_data(ROOT, config)
    )

    # Test set IDs
    all_stay_ids = state_df["stay_id"].unique()
    split_cfg = config["split"]
    train_ids, _, test_ids = make_train_val_test_split(
        all_stay_ids, outcomes_df,
        train_frac=split_cfg["train_frac"],
        val_frac=split_cfg["val_frac"],
        test_frac=split_cfg["test_frac"],
        stratify_col=split_cfg["stratify_col"],
        seed=split_cfg["seed"],
    )
    test_set = set(test_ids)
    print(f"  Test stays: {len(test_ids):,}")

    # Early-discharge info from outcomes
    out_test = outcomes_df[outcomes_df["stay_id"].isin(test_set)].copy()
    early_disc_ids = set(out_test.loc[out_test["early_discharge"] == 1, "stay_id"])
    non_early_ids = set(out_test.loc[out_test["early_discharge"] == 0, "stay_id"])
    print(f"  Early discharge: {len(early_disc_ids):,} ({len(early_disc_ids)/len(test_ids)*100:.1f}%)")
    print(f"  Non-early:       {len(non_early_ids):,}")

    # Phenotype labels for subgroup analysis
    pheno_path = ROOT / config["data"]["phenotype_labels"]
    pheno_df = pd.read_parquet(pheno_path)
    pheno_test = pheno_df[pheno_df["stay_id"].isin(test_set)].set_index("stay_id")

    # ── 2. Load primary model ────────────────────────────────────────
    print("\n[2] Loading 80c G-Transformer ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_temporal = len(temporal_cols)
    d_static = len(static_cols)
    n_actions = len(config["action"]["valid_merged_indices"])

    model = load_primary_model(config, d_temporal, d_static, n_actions, device)
    print(f"  Device: {device}")

    # ── 3. Define scenarios ──────────────────────────────────────────
    print("\n[3] Defining scenarios ...")

    scenarios = []

    # S0: baseline (all stays)
    scenarios.append({
        "name": "S0_baseline",
        "description": "All stays, original outcomes",
        "stay_ids": test_ids,
        "outcomes_mod": None,
        "boot_seed": BOOT_SEED + 0,
    })

    # S1: exclude early-discharge stays
    s1_ids = np.array(sorted(non_early_ids))
    scenarios.append({
        "name": "S1_exclude_early",
        "description": "Exclude early-discharge stays",
        "stay_ids": s1_ids,
        "outcomes_mod": None,
        "boot_seed": BOOT_SEED + 101,
    })

    # S2: recode early-discharge as alive (mortality_28d=0)
    out_s2 = outcomes_df.copy()
    ed_mask = out_s2["stay_id"].isin(early_disc_ids)
    out_s2.loc[ed_mask, "mortality_28d"] = 0
    scenarios.append({
        "name": "S2_bestcase_alive",
        "description": "Early-discharge recoded as mortality_28d=0",
        "stay_ids": test_ids,
        "outcomes_mod": out_s2,
        "boot_seed": BOOT_SEED + 202,
    })

    # S3: recode early-discharge as dead (mortality_28d=1)
    out_s3 = outcomes_df.copy()
    out_s3.loc[ed_mask, "mortality_28d"] = 1
    scenarios.append({
        "name": "S3_worstcase_dead",
        "description": "Early-discharge recoded as mortality_28d=1",
        "stay_ids": test_ids,
        "outcomes_mod": out_s3,
        "boot_seed": BOOT_SEED + 303,
    })

    # S4a-c: exclude stays discharged before bin threshold
    for threshold_bin in [3, 6, 9]:
        threshold_hours = threshold_bin * 2  # bins are 2h each
        # Keep stays either not early-discharged OR discharged after threshold_bin
        keep_mask = (
            (out_test["early_discharge"] == 0)
            | (out_test["absorbing_state_bin"] >= threshold_bin)
        )
        s4_ids = np.array(sorted(out_test.loc[keep_mask, "stay_id"]))
        scenarios.append({
            "name": f"S4_exclude_before_bin{threshold_bin}",
            "description": f"Exclude stays discharged before bin {threshold_bin} ({threshold_hours}h post-T0)",
            "stay_ids": s4_ids,
            "outcomes_mod": None,
            "boot_seed": BOOT_SEED + 400 + threshold_bin,
        })

    for s in scenarios:
        print(f"  {s['name']}: n={len(s['stay_ids']):,} — {s['description']}")

    # ── 4. Evaluate each scenario ────────────────────────────────────
    print("\n[4] Evaluating scenarios ...")
    all_results = []

    # Ensure consistent action remapping with model training.
    anchor_ds = DTRDataset(
        state_df=state_df, outcomes_df=outcomes_df, propensity_df=propensity_df,
        stay_ids=train_ids, temporal_cols=temporal_cols, static_cols=static_cols,
        merge_map=merge_map, n_bins=config["grid"]["n_bins"],
    )
    scenario_cfg = {**config, "action_vocab": anchor_ds.action_vocab}

    for s in tqdm(scenarios, desc="Scenarios", unit="scenario"):
        print(f"\n  ── {s['name']} (n={len(s['stay_ids']):,}) ──")

        out_mod = s["outcomes_mod"]
        ds = build_scenario_dataset(
            state_df, outcomes_df if out_mod is None else out_mod,
            propensity_df, temporal_cols, static_cols, merge_map,
            s["stay_ids"], scenario_cfg,
        )

        # Full-sample ATE
        result = evaluate_scenario(model, ds, device, boot_seed=s["boot_seed"])
        row = {
            "scenario": s["name"],
            "subgroup": "ALL",
            "n_stays": len(s["stay_ids"]),
            **result,
        }
        all_results.append(row)
        print(f"    ALL:   ATE={result['ate']:+.4f} [{result['ate_ci_lo']:+.4f}, {result['ate_ci_hi']:+.4f}]")

        # Subgroup: HFpEF vs no_hf
        for hf_grp in ["hfpef", "no_hf"]:
            grp_sids = pheno_test.loc[
                (pheno_test.index.isin(set(s["stay_ids"])))
                & (pheno_test["hf_subtype"] == hf_grp)
            ].index.values
            if len(grp_sids) < 20:
                print(f"    {hf_grp}: n={len(grp_sids)} — TOO SMALL")
                continue

            grp_ds = build_scenario_dataset(
                state_df, outcomes_df if out_mod is None else out_mod,
                propensity_df, temporal_cols, static_cols, merge_map,
                grp_sids, scenario_cfg,
            )
            grp_result = evaluate_scenario(
                model, grp_ds, device, n_boot=200, boot_seed=s["boot_seed"] + 17
            )
            grp_row = {
                "scenario": s["name"],
                "subgroup": hf_grp,
                "n_stays": len(grp_sids),
                **grp_result,
            }
            all_results.append(grp_row)
            print(f"    {hf_grp}: ATE={grp_result['ate']:+.4f} "
                  f"[{grp_result['ate_ci_lo']:+.4f}, {grp_result['ate_ci_hi']:+.4f}]")

    # ── 5. Save results ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("[5] Saving results ...")

    results_df = pd.DataFrame(all_results)
    csv_path = TABLES_DIR / "85a_edq_sensitivity.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path} ({len(results_df)} rows)")

    # ── 6. Generate report ───────────────────────────────────────────
    print("\n[6] Generating report ...")

    baseline_ate = results_df.loc[
        (results_df["scenario"] == "S0_baseline") & (results_df["subgroup"] == "ALL"),
        "ate",
    ].values[0]

    lines = [
        "=" * 72,
        "85a — Early Discharge Quantitative (EDQ) Sensitivity (MIMIC-IV)",
        "=" * 72,
        f"Generated: {ts}",
        f"Primary model: 80c G-Transformer DM (rollout, ow=1)",
        f"Bootstrap: B={N_BOOTSTRAP}, seed={BOOT_SEED}, CI={CI_LEVEL*100:.0f}%",
        "",
        "PURPOSE",
        "───────",
        "  Test whether ATE estimates are robust to different handling of",
        "  early-discharge stays (informative censoring sensitivity).",
        "",
        "RESULTS",
        "───────",
        f"  {'Scenario':<30s} {'Subgroup':<10s} {'n':>6s} {'ATE':>9s} "
        f"{'CI_lo':>9s} {'CI_hi':>9s} {'Δ vs S0':>9s}",
        f"  {'-'*30} {'-'*10} {'-'*6} {'-'*9} {'-'*9} {'-'*9} {'-'*9}",
    ]

    for _, r in results_df.iterrows():
        delta = r["ate"] - baseline_ate if r["subgroup"] == "ALL" else np.nan
        delta_str = f"{delta:+9.4f}" if pd.notna(delta) and r["scenario"] != "S0_baseline" else "       —"
        lines.append(
            f"  {r['scenario']:<30s} {r['subgroup']:<10s} {r['n_stays']:6d} "
            f"{r['ate']:+9.4f} {r['ate_ci_lo']:+9.4f} {r['ate_ci_hi']:+9.4f} {delta_str}"
        )

    # Interpretation
    lines.extend([
        "",
        "INTERPRETATION",
        "──────────────",
        f"  Baseline ATE (S0): {baseline_ate:+.4f}",
        "",
        "  Robustness criteria:",
        "    - ROBUST if all scenario ATEs are same sign and within 50% of S0",
        "    - SENSITIVE if S1 (exclude) or S3 (worst-case) change sign",
        "    - HFpEF-SPECIFIC if HFpEF ATE remains larger than no_hf across scenarios",
    ])

    # Check robustness
    all_ates = results_df.loc[results_df["subgroup"] == "ALL"]
    same_sign = all(all_ates["ate"] < 0) if baseline_ate < 0 else all(all_ates["ate"] > 0)
    max_deviation = (all_ates["ate"] - baseline_ate).abs().max()

    if same_sign and max_deviation < abs(baseline_ate) * 0.5:
        lines.append("  → ROBUST: All scenarios maintain sign and magnitude")
    elif same_sign:
        lines.append(f"  → PARTIALLY ROBUST: Same sign but magnitude varies (max Δ={max_deviation:.4f})")
    else:
        lines.append("  → SENSITIVE: Sign changes across scenarios — interpret with caution")

    report = "\n".join(lines) + "\n"
    report_path = REPORTS_DIR / "85a_edq_sensitivity_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {report_path}")
    print(report)

    total_time = time.time() - t0
    print(f"\n{'=' * 72}")
    print(f"85a COMPLETE. Total time: {total_time / 60:.1f} min")
    print("=" * 72)


if __name__ == "__main__":
    main()
