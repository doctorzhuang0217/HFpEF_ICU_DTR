#!/usr/bin/env python3
"""P10 Diagnostic — Full-cohort vs Subgroup Per-Patient Prediction Comparison

Purpose: Determine whether G-Transformer DM rollout gives DIFFERENT per-patient
predictions when a patient is evaluated as part of the full test cohort vs. as
part of a subgroup-only DTRDataset.

This diagnostic resolves the P9 vs P10 discrepancy:
  - P9 (subgroup evaluation): HFpEF ATE = -2.31pp, no_HF = -1.22pp
  - P10 (full-cohort, per-stay ITE): HFpEF = -1.11pp, no_HF = -1.39pp
  - Overall ATE matches (~-1.37pp), but subgroup allocation is reversed.

If per-patient predictions match → P10 code/data bug
If per-patient predictions differ → DTRDataset or model bug (should not happen
    for a deterministic model with per-patient inference)

Compute tier: C3 (two forward passes on test set)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# ── Project root ──
ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(ROOT / "src"))

from hfpef_dtr.models.dtr_dataset import DTRDataset, load_dtr_data
from hfpef_dtr.models.dataset import make_train_val_test_split
from hfpef_dtr.models.gtransformer import build_gtransformer_from_config
from hfpef_dtr.models.ope import evaluate_dm


CONFIG_PATH = ROOT / "configs" / "80_step_c_training_v1.yaml"
MODEL_CKPT = ROOT / "models" / "80c_gtransformer_best.pt"


def main():
    print("=" * 70)
    print("P10 DIAGNOSTIC: Full-cohort vs Subgroup Prediction Comparison")
    print("=" * 70)

    # ── Load config & data ──
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    state_df, outcomes_df, propensity_df, temporal_cols, static_cols, merge_map = (
        load_dtr_data(ROOT, config)
    )
    pheno_df = pd.read_parquet(ROOT / config["data"]["phenotype_labels"])

    # ── Test split ──
    all_ids = state_df["stay_id"].unique()
    split_cfg = config["split"]
    _, _, test_ids = make_train_val_test_split(
        all_ids, outcomes_df,
        train_frac=split_cfg["train_frac"],
        val_frac=split_cfg["val_frac"],
        test_frac=split_cfg["test_frac"],
        stratify_col=split_cfg["stratify_col"],
        seed=split_cfg["seed"],
    )
    print(f"Test set: {len(test_ids)} stays")

    # ── Identify HFpEF and no_HF test stays ──
    pheno_test = pheno_df[pheno_df["stay_id"].isin(test_ids)]
    hfpef_ids = pheno_test.loc[pheno_test["hf_subtype"] == "hfpef", "stay_id"].values
    nohf_ids = pheno_test.loc[pheno_test["hf_subtype"] == "no_hf", "stay_id"].values
    hfpef_ids = np.sort(hfpef_ids)
    nohf_ids = np.sort(nohf_ids)
    print(f"HFpEF test stays: {len(hfpef_ids)}")
    print(f"no_HF test stays: {len(nohf_ids)}")

    # ── Load model ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    ckpt = torch.load(MODEL_CKPT, map_location=device, weights_only=False)
    model = build_gtransformer_from_config(
        d_temporal=len(temporal_cols),
        d_static=len(static_cols),
        n_actions=len(config["action"]["valid_merged_indices"]),
        model_cfg=config["models"]["80c_gtransformer"],
        grid_cfg=config["grid"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    action_vocab = config["action"]["valid_merged_indices"]

    # ══════════════════════════════════════════════════════════════
    # TEST 1: Full cohort evaluation → extract HFpEF per-patient ITE
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST 1: Full-cohort evaluation (all test patients)")
    print("=" * 60)

    ds_full = DTRDataset(
        state_df=state_df, outcomes_df=outcomes_df, propensity_df=propensity_df,
        stay_ids=test_ids, temporal_cols=temporal_cols, static_cols=static_cols,
        merge_map=merge_map, n_bins=config["grid"]["n_bins"],
        action_vocab=action_vocab,
    )
    result_full = evaluate_dm(model, ds_full, device, use_rollout=True)

    # Map predictions back to stay_ids (DTRDataset sorts internally)
    full_df = pd.DataFrame({
        "stay_id": ds_full.stay_ids,  # Use DTRDataset's sorted stay_ids
        "obs_prob": result_full["obs_probs"],
        "cf_prob": result_full["cf_probs"],
        "ite": result_full["cf_probs"] - result_full["obs_probs"],
    })
    full_df = full_df.merge(
        pheno_df[["stay_id", "hf_subtype"]], on="stay_id", how="left"
    )

    full_hfpef = full_df[full_df["hf_subtype"] == "hfpef"]
    full_nohf = full_df[full_df["hf_subtype"] == "no_hf"]

    print(f"\n  Full-cohort overall ATE: {full_df['ite'].mean()*100:.4f} pp")
    print(f"  Full-cohort HFpEF (n={len(full_hfpef)}): "
          f"mean ITE = {full_hfpef['ite'].mean()*100:.4f} pp, "
          f"v_obs = {full_hfpef['obs_prob'].mean():.6f}, "
          f"v_cf = {full_hfpef['cf_prob'].mean():.6f}")
    print(f"  Full-cohort no_HF (n={len(full_nohf)}): "
          f"mean ITE = {full_nohf['ite'].mean()*100:.4f} pp, "
          f"v_obs = {full_nohf['obs_prob'].mean():.6f}, "
          f"v_cf = {full_nohf['cf_prob'].mean():.6f}")

    # ══════════════════════════════════════════════════════════════
    # TEST 2: Subgroup-only evaluation (P9 style)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST 2: Subgroup-only evaluation (P9 / 80g style)")
    print("=" * 60)

    # HFpEF subgroup
    ds_hfpef = DTRDataset(
        state_df=state_df, outcomes_df=outcomes_df, propensity_df=propensity_df,
        stay_ids=hfpef_ids, temporal_cols=temporal_cols, static_cols=static_cols,
        merge_map=merge_map, n_bins=config["grid"]["n_bins"],
        action_vocab=action_vocab,
    )
    result_hfpef = evaluate_dm(model, ds_hfpef, device, use_rollout=True)

    sub_hfpef_df = pd.DataFrame({
        "stay_id": ds_hfpef.stay_ids,
        "obs_prob": result_hfpef["obs_probs"],
        "cf_prob": result_hfpef["cf_probs"],
        "ite": result_hfpef["cf_probs"] - result_hfpef["obs_probs"],
    })

    print(f"\n  Subgroup HFpEF (n={len(sub_hfpef_df)}): "
          f"mean ITE = {sub_hfpef_df['ite'].mean()*100:.4f} pp, "
          f"v_obs = {sub_hfpef_df['obs_prob'].mean():.6f}, "
          f"v_cf = {sub_hfpef_df['cf_prob'].mean():.6f}")

    # no_HF subgroup
    ds_nohf = DTRDataset(
        state_df=state_df, outcomes_df=outcomes_df, propensity_df=propensity_df,
        stay_ids=nohf_ids, temporal_cols=temporal_cols, static_cols=static_cols,
        merge_map=merge_map, n_bins=config["grid"]["n_bins"],
        action_vocab=action_vocab,
    )
    result_nohf = evaluate_dm(model, ds_nohf, device, use_rollout=True)

    sub_nohf_df = pd.DataFrame({
        "stay_id": ds_nohf.stay_ids,
        "obs_prob": result_nohf["obs_probs"],
        "cf_prob": result_nohf["cf_probs"],
        "ite": result_nohf["cf_probs"] - result_nohf["obs_probs"],
    })

    print(f"  Subgroup no_HF (n={len(sub_nohf_df)}): "
          f"mean ITE = {sub_nohf_df['ite'].mean()*100:.4f} pp, "
          f"v_obs = {sub_nohf_df['obs_prob'].mean():.6f}, "
          f"v_cf = {sub_nohf_df['cf_prob'].mean():.6f}")

    # ══════════════════════════════════════════════════════════════
    # TEST 3: Per-patient comparison
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST 3: Per-patient prediction comparison")
    print("=" * 60)

    # Compare HFpEF patients
    merged_hfpef = full_hfpef[["stay_id", "obs_prob", "cf_prob", "ite"]].merge(
        sub_hfpef_df[["stay_id", "obs_prob", "cf_prob", "ite"]],
        on="stay_id", suffixes=("_full", "_sub"),
    )

    obs_diff = np.abs(merged_hfpef["obs_prob_full"] - merged_hfpef["obs_prob_sub"])
    cf_diff = np.abs(merged_hfpef["cf_prob_full"] - merged_hfpef["cf_prob_sub"])
    ite_diff = np.abs(merged_hfpef["ite_full"] - merged_hfpef["ite_sub"])

    print(f"\n  HFpEF per-patient comparison (n={len(merged_hfpef)}):")
    print(f"    obs_prob diff: max={obs_diff.max():.8f}, mean={obs_diff.mean():.8f}")
    print(f"    cf_prob diff:  max={cf_diff.max():.8f}, mean={cf_diff.mean():.8f}")
    print(f"    ITE diff:      max={ite_diff.max():.8f}, mean={ite_diff.mean():.8f}")

    if obs_diff.max() < 1e-5:
        print("    → MATCH: Per-patient predictions are identical (within float32 precision)")
        print("    → The P10 discrepancy was a BUG in P10 code, not a model issue")
    else:
        print("    → MISMATCH: Per-patient predictions DIFFER between full-cohort and subgroup!")
        print("    → This indicates a DTRDataset or model issue")

        # Show worst mismatches
        worst = merged_hfpef.nlargest(5, "ite_full")
        print("\n    Top 5 largest ITE mismatches:")
        for _, row in worst.iterrows():
            print(f"      stay_id={row['stay_id']}: "
                  f"ITE_full={row['ite_full']*100:.4f}pp, "
                  f"ITE_sub={row['ite_sub']*100:.4f}pp, "
                  f"diff={abs(row['ite_full']-row['ite_sub'])*100:.4f}pp")

    # Compare no_HF patients (sample)
    merged_nohf = full_nohf[["stay_id", "obs_prob", "cf_prob", "ite"]].merge(
        sub_nohf_df[["stay_id", "obs_prob", "cf_prob", "ite"]],
        on="stay_id", suffixes=("_full", "_sub"),
    )

    obs_diff_nohf = np.abs(merged_nohf["obs_prob_full"] - merged_nohf["obs_prob_sub"])
    cf_diff_nohf = np.abs(merged_nohf["cf_prob_full"] - merged_nohf["cf_prob_sub"])

    print(f"\n  no_HF per-patient comparison (n={len(merged_nohf)}):")
    print(f"    obs_prob diff: max={obs_diff_nohf.max():.8f}, mean={obs_diff_nohf.mean():.8f}")
    print(f"    cf_prob diff:  max={cf_diff_nohf.max():.8f}, mean={cf_diff_nohf.mean():.8f}")

    # ══════════════════════════════════════════════════════════════
    # TEST 4: Action vocab check
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST 4: Action vocabulary comparison")
    print("=" * 60)

    print(f"  Full-cohort action_vocab: {ds_full.action_vocab}")
    print(f"  HFpEF subgroup action_vocab: {ds_hfpef.action_vocab}")
    print(f"  no_HF subgroup action_vocab: {ds_nohf.action_vocab}")
    print(f"  Config action_vocab: {action_vocab}")

    # Check contiguous mappings
    full_map = ds_full._action_to_contiguous
    sub_map = ds_hfpef._action_to_contiguous
    if full_map == sub_map:
        print("  → Contiguous mappings IDENTICAL")
    else:
        print("  → Contiguous mappings DIFFER!")
        for k in sorted(set(full_map) | set(sub_map)):
            f_v = full_map.get(k, "N/A")
            s_v = sub_map.get(k, "N/A")
            if f_v != s_v:
                print(f"    Action {k}: full→{f_v}, sub→{s_v}")

    # ══════════════════════════════════════════════════════════════
    # TEST 5: DTRDataset data integrity check
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TEST 5: DTRDataset data integrity (spot check)")
    print("=" * 60)

    # Pick first 5 HFpEF stay_ids and compare tensor data
    common_ids = np.intersect1d(ds_full.stay_ids, ds_hfpef.stay_ids)[:5]
    print(f"  Checking {len(common_ids)} common stays...")

    for sid in common_ids:
        # Find index in each dataset
        full_idx = np.where(ds_full.stay_ids == sid)[0][0]
        sub_idx = np.where(ds_hfpef.stay_ids == sid)[0][0]

        # Compare tensors
        temp_match = np.allclose(ds_full.temporal[full_idx], ds_hfpef.temporal[sub_idx], atol=1e-6)
        stat_match = np.allclose(ds_full.static[full_idx], ds_hfpef.static[sub_idx], atol=1e-6)
        act_match = np.array_equal(ds_full.action[full_idx], ds_hfpef.action[sub_idx])
        rule_match = np.array_equal(ds_full.rule_action[full_idx], ds_hfpef.rule_action[sub_idx])
        mask_match = np.array_equal(ds_full.valid_mask[full_idx], ds_hfpef.valid_mask[sub_idx])
        abs_match = np.array_equal(ds_full.absorbing_mask[full_idx], ds_hfpef.absorbing_mask[sub_idx])
        mort_match = ds_full.mortality_28d[full_idx] == ds_hfpef.mortality_28d[sub_idx]
        prop_match = np.allclose(ds_full.propensity[full_idx], ds_hfpef.propensity[sub_idx], atol=1e-6)
        rew_match = np.allclose(ds_full.reward[full_idx], ds_hfpef.reward[sub_idx], atol=1e-6)

        all_match = all([temp_match, stat_match, act_match, rule_match,
                        mask_match, abs_match, mort_match, prop_match, rew_match])

        status = "OK" if all_match else "MISMATCH"
        print(f"  stay_id={sid}: {status}", end="")
        if not all_match:
            mismatches = []
            if not temp_match: mismatches.append("temporal")
            if not stat_match: mismatches.append("static")
            if not act_match: mismatches.append("action")
            if not rule_match: mismatches.append("rule_action")
            if not mask_match: mismatches.append("valid_mask")
            if not abs_match: mismatches.append("absorbing_mask")
            if not mort_match: mismatches.append("mortality_28d")
            if not prop_match: mismatches.append("propensity")
            if not rew_match: mismatches.append("reward")
            print(f" [{', '.join(mismatches)}]", end="")
        print()

    # ══════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n  Full-cohort HFpEF ATE:     {full_hfpef['ite'].mean()*100:+.4f} pp")
    print(f"  Subgroup-only HFpEF ATE:   {sub_hfpef_df['ite'].mean()*100:+.4f} pp")
    print(f"  Difference:                {(full_hfpef['ite'].mean() - sub_hfpef_df['ite'].mean())*100:+.4f} pp")
    print()
    print(f"  Full-cohort no_HF ATE:     {full_nohf['ite'].mean()*100:+.4f} pp")
    print(f"  Subgroup-only no_HF ATE:   {sub_nohf_df['ite'].mean()*100:+.4f} pp")
    print(f"  Difference:                {(full_nohf['ite'].mean() - sub_nohf_df['ite'].mean())*100:+.4f} pp")
    print()
    print(f"  Full-cohort overall ATE:   {full_df['ite'].mean()*100:+.4f} pp")
    print(f"  P9 reference HFpEF ATE:    -2.31 pp (weighted from quartiles)")
    print(f"  P9 reference no_HF ATE:    -1.22 pp (weighted from quartiles)")

    if obs_diff.max() < 1e-5:
        print("\n  CONCLUSION: Predictions match. If P10 showed different numbers,")
        print("  the bug was in P10's stay_id mapping or merge, not in DTRDataset.")
        print("  P9's subgroup results are correct. P10 should use ds.stay_ids")
        print("  (not the input stay_ids) when building the output DataFrame.")
    else:
        print("\n  CONCLUSION: Predictions DIFFER between modes!")
        print("  This requires further investigation into DTRDataset behavior.")

    # Save diagnostic output
    diag_path = ROOT / "reports" / "P10_diagnostic_output.csv"
    merged_hfpef.to_csv(diag_path, index=False)
    print(f"\n  Saved: {diag_path}")


if __name__ == "__main__":
    main()
