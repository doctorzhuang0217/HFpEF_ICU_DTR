#!/usr/bin/env python3
"""
85h_rrt_crrt_handling_sensitivity — RRT/CRRT handling sensitivity analysis.

Purpose
-------
Test whether alternate handling of RRT/CRRT materially changes Step C ATE
estimates. RRT/CRRT is an intercurrent event that fundamentally alters fluid
management. The primary analysis treats RRT stays as regular participants;
this sensitivity tests three alternatives.

Scenarios tested:
  S0 (baseline):   Include all stays as-is (original Step C result)
  S1 (exclude RRT): Exclude stays with any RRT/CRRT during ICU stay
  S2 (exclude pre-T0 RRT): Exclude stays with RRT initiated before T0
  S3 (censor at RRT): For post-T0 RRT stays, recode mortality_28d
      under best-case (alive) and worst-case (dead) bounding
  S4 (stratify): Evaluate ATE separately in RRT vs non-RRT subgroups

For each scenario, re-evaluate ATE using 80c G-Transformer DM (primary)
with bootstrap CIs. Also compute HFpEF vs no_hf subgroup ATEs.

Protocol reference: Step 85h (sensitivity layer), Assumption #6 (no
  informative censoring), Target Trial Table (intercurrent events)
Compute: C2-C3 (bootstrap-heavy CPU, ~16 cores recommended)

Inputs
------
- configs/80_step_c_training_v1.yaml
- models/80c_gtransformer_rollout_ow1_best.pt
- data/interim/45c_state_tensors_v1_mimic.parquet
- data/interim/45d_outcomes_absorb_v1_mimic.parquet
- data/interim/70b_propensity_coarse3_v1_mimic.parquet
- data/processed/phenotype_labels_frozen_v1_mimic.parquet
- data/staging/25e_goals_of_care_rrt_flags_mimic.parquet

Outputs
-------
- tables/85h_rrt_sensitivity.csv
- reports/85h_rrt_sensitivity_summary.txt
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
    point = evaluate_dm(model, dataset, device, use_rollout=True)

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


def evaluate_with_subgroups(
    model, device, state_df, outcomes_df, propensity_df,
    temporal_cols, static_cols, merge_map, scenario_cfg,
    stay_ids, pheno_test, scenario_name, boot_seed,
    outcomes_mod=None, extra_subgroups=None,
):
    """Evaluate a scenario for ALL + HFpEF/no_hf subgroups. Returns list of result dicts."""
    results = []
    out_df = outcomes_mod if outcomes_mod is not None else outcomes_df

    # Full-sample ATE
    ds = build_scenario_dataset(
        state_df, out_df, propensity_df,
        temporal_cols, static_cols, merge_map,
        stay_ids, scenario_cfg,
    )
    result = evaluate_scenario(model, ds, device, boot_seed=boot_seed)
    results.append({
        "scenario": scenario_name,
        "subgroup": "ALL",
        "n_stays": len(stay_ids),
        **result,
    })
    print(f"    ALL:   ATE={result['ate']:+.4f} "
          f"[{result['ate_ci_lo']:+.4f}, {result['ate_ci_hi']:+.4f}]")

    # HFpEF / no_hf subgroups
    subgroup_list = ["hfpef", "no_hf"]
    if extra_subgroups:
        subgroup_list.extend(extra_subgroups)

    for hf_grp in subgroup_list:
        grp_sids = pheno_test.loc[
            (pheno_test.index.isin(set(stay_ids)))
            & (pheno_test["hf_subtype"] == hf_grp)
        ].index.values
        if len(grp_sids) < 20:
            print(f"    {hf_grp}: n={len(grp_sids)} — TOO SMALL, skipped")
            continue

        grp_ds = build_scenario_dataset(
            state_df, out_df, propensity_df,
            temporal_cols, static_cols, merge_map,
            grp_sids, scenario_cfg,
        )
        grp_result = evaluate_scenario(
            model, grp_ds, device, n_boot=200, boot_seed=boot_seed + 17
        )
        results.append({
            "scenario": scenario_name,
            "subgroup": hf_grp,
            "n_stays": len(grp_sids),
            **grp_result,
        })
        print(f"    {hf_grp}: ATE={grp_result['ate']:+.4f} "
              f"[{grp_result['ate_ci_lo']:+.4f}, {grp_result['ate_ci_hi']:+.4f}]")

    return results


def main() -> None:
    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("85h — RRT/CRRT Handling Sensitivity")
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

    # ── 2. Load RRT flags ────────────────────────────────────────────
    print("\n[2] Loading RRT/CRRT flags ...")
    rrt_path = ROOT / "data" / "staging" / "25e_goals_of_care_rrt_flags_mimic.parquet"
    if not rrt_path.exists():
        print(f"  FATAL: RRT flags not found: {rrt_path}")
        sys.exit(1)
    rrt_flags = pd.read_parquet(rrt_path)

    # Also check the outcomes table for the rrt column (consistency audit)
    rrt_col = None
    for candidate in ["rrt_crrt_any_during_stay", "has_rrt"]:
        if candidate in outcomes_df.columns:
            rrt_col = candidate
            break

    required_rrt_cols = {
        "stay_id",
        "rrt_crrt_any_during_stay",
        "rrt_crrt_initiated_pre_t0",
    }
    missing_rrt_cols = required_rrt_cols - set(rrt_flags.columns)
    if missing_rrt_cols:
        print(f"  FATAL: Missing required RRT columns: {sorted(missing_rrt_cols)}")
        sys.exit(1)

    # Build RRT stay sets from 25e flags
    rrt_test = rrt_flags[rrt_flags["stay_id"].isin(test_set)].copy()
    covered_ids = set(rrt_test["stay_id"])
    missing_ids = test_set - covered_ids
    if missing_ids:
        print(
            f"  WARNING: {len(missing_ids):,} test stays missing in 25e RRT flags; "
            "treating as non-RRT"
        )

    rrt_any_ids = set(rrt_test.loc[rrt_test["rrt_crrt_any_during_stay"] == True, "stay_id"])
    rrt_pre_t0_ids = set(rrt_test.loc[rrt_test["rrt_crrt_initiated_pre_t0"] == True, "stay_id"])
    rrt_post_t0_only_ids = rrt_any_ids - rrt_pre_t0_ids
    no_rrt_ids = test_set - rrt_any_ids

    print(f"  RRT any during stay (test): {len(rrt_any_ids):,} "
          f"({len(rrt_any_ids)/len(test_ids)*100:.1f}%)")
    print(f"  RRT pre-T0 only (test):     {len(rrt_pre_t0_ids):,}")
    print(f"  RRT post-T0 only (test):    {len(rrt_post_t0_only_ids):,}")
    print(f"  No RRT (test):              {len(no_rrt_ids):,}")

    if rrt_col is not None:
        out_test_rrt = outcomes_df[outcomes_df["stay_id"].isin(test_set)][["stay_id", rrt_col]].drop_duplicates("stay_id")
        out_rrt_any_ids = set(out_test_rrt.loc[out_test_rrt[rrt_col] == 1, "stay_id"])
        disagreement = len(rrt_any_ids.symmetric_difference(out_rrt_any_ids))
        print(
            f"  Cross-check vs outcomes.{rrt_col}: "
            f"{disagreement:,} stay_id disagreements"
        )
    else:
        print("  WARNING: No RRT indicator in outcomes_df (rrt_crrt_any_during_stay/has_rrt)")

    # Phenotype labels for subgroup analysis
    pheno_path = ROOT / config["data"]["phenotype_labels"]
    pheno_df = pd.read_parquet(pheno_path)
    pheno_test = pheno_df[pheno_df["stay_id"].isin(test_set)].set_index("stay_id")

    # ── 3. Load primary model ────────────────────────────────────────
    print("\n[3] Loading 80c G-Transformer ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_temporal = len(temporal_cols)
    d_static = len(static_cols)
    n_actions = len(config["action"]["valid_merged_indices"])

    model = load_primary_model(config, d_temporal, d_static, n_actions, device)
    print(f"  Device: {device}")

    # Anchor action vocab from training set
    anchor_ds = DTRDataset(
        state_df=state_df, outcomes_df=outcomes_df, propensity_df=propensity_df,
        stay_ids=train_ids, temporal_cols=temporal_cols, static_cols=static_cols,
        merge_map=merge_map, n_bins=config["grid"]["n_bins"],
    )
    scenario_cfg = {**config, "action_vocab": anchor_ds.action_vocab}

    # ── 4. Define and evaluate scenarios ─────────────────────────────
    print("\n[4] Evaluating scenarios ...")
    all_results = []

    # --- S3a/S3b outcome modifications prepared once ---
    out_s3a = outcomes_df.copy()
    s3a_mask = out_s3a["stay_id"].isin(rrt_post_t0_only_ids)
    out_s3a.loc[s3a_mask, "mortality_28d"] = 0

    out_s3b = outcomes_df.copy()
    out_s3b.loc[s3a_mask, "mortality_28d"] = 1

    scenarios = [
        {
            "name": "S0_baseline",
            "stay_ids": test_ids,
            "outcomes_mod": None,
            "boot_seed": BOOT_SEED + 0,
            "extra_note": None,
        },
        {
            "name": "S1_exclude_rrt_any",
            "stay_ids": np.array(sorted(no_rrt_ids)),
            "outcomes_mod": None,
            "boot_seed": BOOT_SEED + 100,
            "extra_note": None,
        },
        {
            "name": "S2_exclude_rrt_pre_t0",
            "stay_ids": np.array(sorted(test_set - rrt_pre_t0_ids)),
            "outcomes_mod": None,
            "boot_seed": BOOT_SEED + 200,
            "extra_note": None,
        },
        {
            "name": "S3a_rrt_bestcase_alive",
            "stay_ids": test_ids,
            "outcomes_mod": out_s3a,
            "boot_seed": BOOT_SEED + 300,
            "extra_note": f"recoded={s3a_mask.sum():,}",
        },
        {
            "name": "S3b_rrt_worstcase_dead",
            "stay_ids": test_ids,
            "outcomes_mod": out_s3b,
            "boot_seed": BOOT_SEED + 350,
            "extra_note": f"recoded={s3a_mask.sum():,}",
        },
    ]

    for scenario in tqdm(scenarios, desc="Scenarios S0-S3", unit="scenario"):
        extra = f", {scenario['extra_note']}" if scenario["extra_note"] else ""
        print(f"\n  ── {scenario['name']} (n={len(scenario['stay_ids']):,}{extra}) ──")
        scenario_results = evaluate_with_subgroups(
            model, device, state_df, outcomes_df, propensity_df,
            temporal_cols, static_cols, merge_map, scenario_cfg,
            scenario["stay_ids"], pheno_test, scenario["name"], scenario["boot_seed"],
            outcomes_mod=scenario["outcomes_mod"],
        )
        all_results.extend(scenario_results)

    # --- S4: Stratify — RRT subgroup only ---
    s4_rrt_ids = np.array(sorted(rrt_any_ids))
    if len(s4_rrt_ids) >= 30:
        print(f"\n  ── S4_rrt_only (n={len(s4_rrt_ids):,}) ──")
        s4_ds = build_scenario_dataset(
            state_df, outcomes_df, propensity_df,
            temporal_cols, static_cols, merge_map,
            s4_rrt_ids, scenario_cfg,
        )
        s4_result = evaluate_scenario(
            model, s4_ds, device, n_boot=200, boot_seed=BOOT_SEED + 400
        )
        all_results.append({
            "scenario": "S4_rrt_only",
            "subgroup": "ALL",
            "n_stays": len(s4_rrt_ids),
            **s4_result,
        })
        print(f"    ALL:   ATE={s4_result['ate']:+.4f} "
              f"[{s4_result['ate_ci_lo']:+.4f}, {s4_result['ate_ci_hi']:+.4f}]")

        # HFpEF within RRT subgroup
        for hf_grp in ["hfpef", "no_hf"]:
            grp_sids = pheno_test.loc[
                (pheno_test.index.isin(rrt_any_ids))
                & (pheno_test["hf_subtype"] == hf_grp)
            ].index.values
            if len(grp_sids) >= 20:
                grp_ds = build_scenario_dataset(
                    state_df, outcomes_df, propensity_df,
                    temporal_cols, static_cols, merge_map,
                    grp_sids, scenario_cfg,
                )
                grp_result = evaluate_scenario(
                    model, grp_ds, device, n_boot=200, boot_seed=BOOT_SEED + 417
                )
                all_results.append({
                    "scenario": "S4_rrt_only",
                    "subgroup": hf_grp,
                    "n_stays": len(grp_sids),
                    **grp_result,
                })
                print(f"    {hf_grp}: ATE={grp_result['ate']:+.4f} "
                      f"[{grp_result['ate_ci_lo']:+.4f}, {grp_result['ate_ci_hi']:+.4f}]")
            else:
                print(f"    {hf_grp}: n={len(grp_sids)} — TOO SMALL, skipped")
    else:
        print(f"\n  ── S4_rrt_only: SKIPPED (n={len(s4_rrt_ids)} < 30) ──")

    # --- S4b: Stratify — non-RRT subgroup only (for comparison) ---
    s4b_ids = np.array(sorted(no_rrt_ids))
    print(f"\n  ── S4b_no_rrt_only (n={len(s4b_ids):,}) ──")
    s4b_ds = build_scenario_dataset(
        state_df, outcomes_df, propensity_df,
        temporal_cols, static_cols, merge_map,
        s4b_ids, scenario_cfg,
    )
    s4b_result = evaluate_scenario(
        model, s4b_ds, device, n_boot=200, boot_seed=BOOT_SEED + 450
    )
    all_results.append({
        "scenario": "S4b_no_rrt_only",
        "subgroup": "ALL",
        "n_stays": len(s4b_ids),
        **s4b_result,
    })
    print(f"    ALL:   ATE={s4b_result['ate']:+.4f} "
          f"[{s4b_result['ate_ci_lo']:+.4f}, {s4b_result['ate_ci_hi']:+.4f}]")

    # ── 5. Save results ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("[5] Saving results ...")

    results_df = pd.DataFrame(all_results)
    csv_path = TABLES_DIR / "85h_rrt_sensitivity.csv"
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
        "85h — RRT/CRRT Handling Sensitivity (MIMIC-IV)",
        "=" * 72,
        f"Generated: {ts}",
        f"Primary model: 80c G-Transformer DM (rollout, ow=1)",
        f"Bootstrap: B={N_BOOTSTRAP}, seed={BOOT_SEED}, CI={CI_LEVEL*100:.0f}%",
        "",
        "RRT/CRRT PREVALENCE (test set)",
        "──────────────────────────────",
        f"  Any RRT during stay:  {len(rrt_any_ids):,} "
        f"({len(rrt_any_ids)/len(test_ids)*100:.1f}%)",
        f"  Pre-T0 RRT:           {len(rrt_pre_t0_ids):,}",
        f"  Post-T0 RRT only:     {len(rrt_post_t0_only_ids):,}",
        f"  No RRT:               {len(no_rrt_ids):,}",
        "",
        "PURPOSE",
        "───────",
        "  RRT/CRRT is an intercurrent event that fundamentally alters fluid",
        "  management. Test whether ATE estimates are robust to alternate RRT",
        "  handling: (S1) exclude all RRT, (S2) exclude pre-T0 RRT,",
        "  (S3a/b) bound post-T0 RRT outcomes, (S4) stratify by RRT status.",
        "",
        "RESULTS",
        "───────",
        f"  {'Scenario':<28s} {'Subgroup':<10s} {'n':>6s} {'ATE':>9s} "
        f"{'CI_lo':>9s} {'CI_hi':>9s} {'Δ vs S0':>9s}",
        f"  {'-'*28} {'-'*10} {'-'*6} {'-'*9} {'-'*9} {'-'*9} {'-'*9}",
    ]

    for _, r in results_df.iterrows():
        delta = r["ate"] - baseline_ate if r["subgroup"] == "ALL" else np.nan
        delta_str = (
            f"{delta:+9.4f}"
            if pd.notna(delta) and r["scenario"] != "S0_baseline"
            else "       —"
        )
        lines.append(
            f"  {r['scenario']:<28s} {r['subgroup']:<10s} {r['n_stays']:6d} "
            f"{r['ate']:+9.4f} {r['ate_ci_lo']:+9.4f} {r['ate_ci_hi']:+9.4f} {delta_str}"
        )

    # Interpretation
    lines.extend([
        "",
        "INTERPRETATION",
        "──────────────",
        f"  Baseline ATE (S0): {baseline_ate:+.4f}",
        "",
        "  S1 (exclude all RRT): If ATE persists → treatment effect is not",
        "    driven by RRT/CRRT patients whose fluid management is atypical.",
        "",
        "  S2 (exclude pre-T0 RRT): Tests whether pre-existing RRT",
        "    (prevalent users) confounds the estimated effect.",
        "",
        "  S3a/b (best/worst-case bounding): Tippett-style bounds —",
        "    if sign is preserved under both extreme outcome recoding for",
        "    post-T0 RRT initiators, the conclusion is robust to RRT as",
        "    informative censoring.",
        "",
        "  S4 (stratify): If ATE differs between RRT vs non-RRT →",
        "    effect modification by RRT status (expected: RRT patients",
        "    have altered fluid dynamics; ATE magnitude may differ).",
        "",
        "  Robustness criteria:",
        "    - ROBUST if S1/S2 ATEs same sign and within 50% of S0",
        "    - ROBUST if S3a/S3b both same sign as S0 (bounding test)",
        "    - HFpEF-SPECIFIC if HFpEF ATE > no_hf across scenarios",
    ])

    # Check robustness
    exclusion_scenarios = results_df.loc[
        (results_df["scenario"].isin(["S0_baseline", "S1_exclude_rrt_any", "S2_exclude_rrt_pre_t0"]))
        & (results_df["subgroup"] == "ALL")
    ]
    bounding_scenarios = results_df.loc[
        (results_df["scenario"].isin(["S3a_rrt_bestcase_alive", "S3b_rrt_worstcase_dead"]))
        & (results_df["subgroup"] == "ALL")
    ]

    excl_same_sign = all(
        (exclusion_scenarios["ate"] < 0) if baseline_ate < 0
        else (exclusion_scenarios["ate"] > 0)
    )
    excl_max_dev = (exclusion_scenarios["ate"] - baseline_ate).abs().max()

    bound_same_sign = True
    if len(bounding_scenarios) > 0:
        bound_same_sign = all(
            (bounding_scenarios["ate"] < 0) if baseline_ate < 0
            else (bounding_scenarios["ate"] > 0)
        )

    if excl_same_sign and excl_max_dev < abs(baseline_ate) * 0.5 and bound_same_sign:
        verdict = "ROBUST"
        lines.append(
            "  → ROBUST: Exclusion scenarios maintain sign/magnitude; "
            "bounding test passed"
        )
    elif excl_same_sign and bound_same_sign:
        verdict = "PARTIALLY ROBUST"
        lines.append(
            f"  → PARTIALLY ROBUST: Same sign but magnitude varies "
            f"(max Δ={excl_max_dev:.4f})"
        )
    else:
        verdict = "SENSITIVE"
        lines.append(
            "  → SENSITIVE: Sign changes under RRT handling alternatives — "
            "interpret with caution"
        )

    report = "\n".join(lines) + "\n"
    report_path = REPORTS_DIR / "85h_rrt_sensitivity_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {report_path}")
    print(report)

    total_time = time.time() - t0
    print(f"\n{'=' * 72}")
    print(f"85h COMPLETE. Total time: {total_time / 60:.1f} min")
    print(f"Verdict: {verdict}")
    print("=" * 72)


if __name__ == "__main__":
    main()
