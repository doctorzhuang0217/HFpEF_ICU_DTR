#!/usr/bin/env python3
"""
85g_monitoring_process_sensitivity — Monitoring process sensitivity (light).

Purpose
-------
Test whether monitoring intensity *as a confounding process* drives Step C
conclusions. This is distinct from 85j (which ablates missingness *features*
inside the model). Here we treat monitoring intensity as an external
confounder and apply three diagnostics:

  G1 (IPMW stratified):
      Compute ATE within monitoring intensity strata, then produce an
      IPMW-reweighted aggregate ATE. If the monitoring-adjusted ATE differs
      materially from baseline → monitoring intensity confounds the estimate.

  G2 (propensity overlap):
      Check whether treatment propensity distributions differ across
      monitoring intensity quartiles. Divergent propensity by monitoring
      level signals violation of treatment-monitoring exchangeability.

  G3 (dose-response trend):
      Fit a linear trend of ATE vs monitoring intensity quartile midpoint.
      A significant slope suggests the treatment effect is an artifact of
      monitoring frequency (sicker patients get monitored more and also
      have worse outcomes regardless of treatment).

  G4 (HFpEF-specific monitoring check):
      Compare monitoring intensity distribution between HFpEF and non-HF.
      If HFpEF patients are systematically more/less monitored, the
      HFpEF-specific ATE may be confounded by differential monitoring.

All analyses use the frozen 80c model — no retraining.

Protocol reference: Step 85g (sensitivity layer), Assumption #8 (no
  informative missingness) — parallel to 85j but distinct scope.
Compute: C2-C3 (CPU-based, no GPU needed)

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
- tables/85g_monitoring_sensitivity.csv
- reports/85g_monitoring_sensitivity_summary.txt
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
N_QUARTILES = 4


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


def compute_measurement_intensity(
    state_df: pd.DataFrame,
    temporal_cols: list[str],
    stay_ids: np.ndarray,
) -> pd.DataFrame:
    """
    Compute per-stay measurement intensity as mean of _mask features.

    Higher intensity = more measurements = potentially sicker patient.
    Reuses the same logic as 85j for consistency.
    """
    mask_cols = [c for c in temporal_cols if c.endswith("_mask")]
    if not mask_cols:
        print("  WARNING: No _mask columns found; using raw obs count")
        sub = state_df[state_df["stay_id"].isin(set(stay_ids))]
        counts = sub.groupby("stay_id").size().reset_index(name="measurement_intensity")
        return counts

    sub = state_df[state_df["stay_id"].isin(set(stay_ids))].copy()
    intensity = sub.groupby("stay_id")[mask_cols].mean().mean(axis=1).reset_index()
    intensity.columns = ["stay_id", "measurement_intensity"]
    return intensity


def compute_ipmw(intensity_df: pd.DataFrame, n_quartiles: int = 4) -> pd.DataFrame:
    """
    Compute Inverse Probability of Monitoring Weights (IPMW).

    For each quartile q, P(Q=q | covariates). In this light version we use
    marginal probabilities (no covariate adjustment) — weight = 1/P(Q=q)
    normalized so weights within each quartile sum to the quartile size.

    This stabilized marginal IPMW tests: "what if all patients had the same
    probability of being in each monitoring intensity group?"
    """
    q_edges = intensity_df["measurement_intensity"].quantile(
        [i / n_quartiles for i in range(1, n_quartiles)]
    ).values
    intensity_df = intensity_df.copy()
    intensity_df["intensity_q"] = np.digitize(
        intensity_df["measurement_intensity"].values, q_edges
    ) + 1

    # Marginal probability of each quartile
    q_counts = intensity_df["intensity_q"].value_counts()
    n_total = len(intensity_df)
    intensity_df["p_q"] = intensity_df["intensity_q"].map(q_counts) / n_total

    # Stabilized weight: target distribution is uniform
    target_p = 1.0 / n_quartiles
    intensity_df["ipmw"] = target_p / intensity_df["p_q"]

    return intensity_df


def main() -> None:
    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("85g — Monitoring Process Sensitivity (light)")
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

    # Phenotype labels
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

    # Anchor action vocab
    anchor_ds = DTRDataset(
        state_df=state_df, outcomes_df=outcomes_df, propensity_df=propensity_df,
        stay_ids=train_ids, temporal_cols=temporal_cols, static_cols=static_cols,
        merge_map=merge_map, n_bins=config["grid"]["n_bins"],
    )
    scenario_cfg = {**config, "action_vocab": anchor_ds.action_vocab}

    # ── 3. Compute measurement intensity ─────────────────────────────
    print("\n[3] Computing measurement intensity ...")
    intensity_df = compute_measurement_intensity(state_df, temporal_cols, test_ids)
    intensity_df = intensity_df[intensity_df["stay_id"].isin(test_set)]
    intensity_df = compute_ipmw(intensity_df, N_QUARTILES)

    q_dist = intensity_df["intensity_q"].value_counts().sort_index()
    print(f"  Quartile distribution: {q_dist.to_dict()}")
    print(f"  Intensity range: [{intensity_df['measurement_intensity'].min():.4f}, "
          f"{intensity_df['measurement_intensity'].max():.4f}]")
    print(f"  Intensity mean: {intensity_df['measurement_intensity'].mean():.4f}")

    int_indexed = intensity_df.set_index("stay_id")

    all_results = []

    # ── G0: Baseline ATE ─────────────────────────────────────────────
    print("\n[G0] Baseline ATE ...")
    ds_base = build_scenario_dataset(
        state_df, outcomes_df, propensity_df,
        temporal_cols, static_cols, merge_map,
        test_ids, scenario_cfg,
    )
    base_result = evaluate_scenario(model, ds_base, device, boot_seed=BOOT_SEED)
    all_results.append({
        "scenario": "G0_baseline",
        "subgroup": "ALL",
        "n_stays": len(test_ids),
        **base_result,
    })
    baseline_ate = base_result["ate"]
    print(f"  ATE={baseline_ate:+.4f} "
          f"[{base_result['ate_ci_lo']:+.4f}, {base_result['ate_ci_hi']:+.4f}]")

    # ── G1: IPMW-stratified ATE ──────────────────────────────────────
    print("\n[G1] IPMW-stratified ATE ...")

    q_ates = []
    q_ns = []
    q_labels = []

    for q in tqdm(range(1, N_QUARTILES + 1), desc="G1 IPMW quartiles"):
        q_sids = int_indexed.loc[int_indexed["intensity_q"] == q].index.values
        q_sids = np.array(sorted(set(q_sids) & test_set))

        if len(q_sids) < 30:
            print(f"  Q{q}: n={len(q_sids)} — TOO SMALL, skipped")
            continue

        q_ds = build_scenario_dataset(
            state_df, outcomes_df, propensity_df,
            temporal_cols, static_cols, merge_map,
            q_sids, scenario_cfg,
        )
        q_result = evaluate_scenario(
            model, q_ds, device, n_boot=200, boot_seed=BOOT_SEED + q * 10
        )

        all_results.append({
            "scenario": f"G1_intensity_Q{q}",
            "subgroup": "ALL",
            "n_stays": len(q_sids),
            **q_result,
        })

        q_ates.append(q_result["ate"])
        q_ns.append(len(q_sids))
        q_labels.append(q)

        print(f"  Q{q}: n={len(q_sids):,}, ATE={q_result['ate']:+.4f} "
              f"[{q_result['ate_ci_lo']:+.4f}, {q_result['ate_ci_hi']:+.4f}]")

    # IPMW-weighted aggregate ATE
    if q_ates:
        q_ates_arr = np.array(q_ates)
        q_ns_arr = np.array(q_ns)

        # Standardized aggregate: Σ_q target_p * ATE_q.
        # If some quartiles were skipped (small n), renormalize over available target mass.
        target_weights = np.array([1.0 / N_QUARTILES] * len(q_ates_arr))
        if len(q_ates_arr) < N_QUARTILES:
            target_weights = target_weights / target_weights.sum()

        ipmw_ate = np.average(q_ates_arr, weights=target_weights)
        # Simple n-weighted (for comparison)
        naive_ate = np.average(q_ates_arr, weights=q_ns_arr)

        all_results.append({
            "scenario": "G1_ipmw_aggregate",
            "subgroup": "ALL",
            "n_stays": sum(q_ns),
            "ate": ipmw_ate,
            "v_observed": np.nan,
            "v_rule": np.nan,
            "ate_ci_lo": np.nan,
            "ate_ci_hi": np.nan,
            "n_boot_success": 0,
        })
        all_results.append({
            "scenario": "G1_naive_aggregate",
            "subgroup": "ALL",
            "n_stays": sum(q_ns),
            "ate": naive_ate,
            "v_observed": np.nan,
            "v_rule": np.nan,
            "ate_ci_lo": np.nan,
            "ate_ci_hi": np.nan,
            "n_boot_success": 0,
        })
        print(f"\n  IPMW aggregate ATE: {ipmw_ate:+.4f}")
        print(f"  Naive aggregate ATE: {naive_ate:+.4f}")
        print(f"  Baseline ATE:        {baseline_ate:+.4f}")
        if len(q_ates_arr) < N_QUARTILES:
            print(
                f"  WARNING: only {len(q_ates_arr)}/{N_QUARTILES} quartiles "
                "contributed to IPMW aggregate (small-cell skip)"
            )

    # ── G2: Propensity overlap by monitoring quartile ────────────────
    print("\n[G2] Propensity overlap by monitoring quartile ...")

    # Use propensity scores from 70b
    prop_test = propensity_df[propensity_df["stay_id"].isin(test_set)].copy()

    # Find propensity columns (behavior policy probabilities)
    prop_cols = [c for c in prop_test.columns if c.startswith("pi_behav_")]
    if not prop_cols:
        # Fallback: look for any probability-like columns
        prop_cols = [c for c in prop_test.columns
                     if c not in ("stay_id", "time_bin") and prop_test[c].dtype == float]

    if prop_cols:
        # Compute per-stay mean propensity entropy as a summary statistic
        prop_stay = prop_test.groupby("stay_id")[prop_cols].mean()

        # Entropy of action probabilities per stay
        def _entropy(row):
            p = row.values
            p = p[p > 0]
            return -np.sum(p * np.log(p + 1e-12))

        prop_stay["prop_entropy"] = prop_stay.apply(_entropy, axis=1)

        # Merge with intensity quartiles
        prop_int = prop_stay[["prop_entropy"]].join(
            int_indexed[["intensity_q"]], how="inner"
        )

        g2_stats = []
        for q in range(1, N_QUARTILES + 1):
            q_entropy = prop_int.loc[prop_int["intensity_q"] == q, "prop_entropy"]
            g2_stats.append({
                "quartile": q,
                "n": len(q_entropy),
                "mean_entropy": q_entropy.mean(),
                "std_entropy": q_entropy.std(),
            })
            print(f"  Q{q}: n={len(q_entropy):,}, "
                  f"mean propensity entropy={q_entropy.mean():.4f} "
                  f"(sd={q_entropy.std():.4f})")

        # Add to results as metadata rows
        for s in g2_stats:
            all_results.append({
                "scenario": f"G2_prop_overlap_Q{s['quartile']}",
                "subgroup": "ALL",
                "n_stays": s["n"],
                "ate": s["mean_entropy"],  # repurpose ATE column for entropy
                "v_observed": s["std_entropy"],
                "v_rule": np.nan,
                "ate_ci_lo": np.nan,
                "ate_ci_hi": np.nan,
                "n_boot_success": 0,
            })
    else:
        print("  WARNING: No propensity columns found — G2 skipped")

    # ── G3: Dose-response trend ──────────────────────────────────────
    print("\n[G3] Dose-response trend (ATE vs monitoring intensity) ...")

    if len(q_ates) >= 3:
        # Simple linear regression: ATE ~ quartile midpoint
        q_midpoints = np.array([
            intensity_df.loc[
                intensity_df["intensity_q"] == q, "measurement_intensity"
            ].median()
            for q in q_labels
        ])

        # Ensure arrays match
        n_valid = min(len(q_midpoints), len(q_ates))
        x = q_midpoints[:n_valid]
        y = np.array(q_ates[:n_valid])

        if len(x) >= 3:
            # Fit linear trend
            slope, intercept = np.polyfit(x, y, 1)

            # Permutation test for slope significance (lightweight)
            rng = np.random.default_rng(BOOT_SEED)
            n_perm = 1000
            null_slopes = np.empty(n_perm)
            for i in tqdm(range(n_perm), desc="G3 permutation test", leave=False):
                y_perm = rng.permutation(y)
                null_slopes[i] = np.polyfit(x, y_perm, 1)[0]
            p_trend = np.mean(np.abs(null_slopes) >= np.abs(slope))

            print(f"  Slope: {slope:+.6f}")
            print(f"  Intercept: {intercept:+.6f}")
            print(f"  p-value (permutation, n={n_perm}): {p_trend:.4f}")

            all_results.append({
                "scenario": "G3_trend_slope",
                "subgroup": "ALL",
                "n_stays": sum(q_ns[:n_valid]),
                "ate": slope,
                "v_observed": intercept,
                "v_rule": p_trend,
                "ate_ci_lo": np.nan,
                "ate_ci_hi": np.nan,
                "n_boot_success": n_perm,
            })
    else:
        print("  WARNING: Too few quartiles for trend analysis — G3 skipped")

    # ── G4: HFpEF-specific monitoring check ──────────────────────────
    print("\n[G4] HFpEF-specific monitoring check ...")

    for hf_grp in ["hfpef", "no_hf"]:
        grp_sids = pheno_test.loc[
            pheno_test["hf_subtype"] == hf_grp
        ].index.values
        grp_intensity = int_indexed.loc[
            int_indexed.index.isin(set(grp_sids)),
            "measurement_intensity"
        ]

        if len(grp_intensity) < 20:
            print(f"  {hf_grp}: n={len(grp_intensity)} — TOO SMALL")
            continue

        grp_mean = grp_intensity.mean()
        grp_std = grp_intensity.std()
        grp_q_dist = int_indexed.loc[
            int_indexed.index.isin(set(grp_sids)), "intensity_q"
        ].value_counts().sort_index()

        all_results.append({
            "scenario": f"G4_monitoring_profile",
            "subgroup": hf_grp,
            "n_stays": len(grp_intensity),
            "ate": grp_mean,
            "v_observed": grp_std,
            "v_rule": np.nan,
            "ate_ci_lo": np.nan,
            "ate_ci_hi": np.nan,
            "n_boot_success": 0,
        })

        print(f"  {hf_grp}: n={len(grp_intensity):,}, "
              f"mean intensity={grp_mean:.4f} (sd={grp_std:.4f}), "
              f"Q dist={grp_q_dist.to_dict()}")

    # Also compute per-quartile ATE for HFpEF subgroup
    print("\n  HFpEF ATE by monitoring quartile:")
    for q in tqdm(range(1, N_QUARTILES + 1), desc="G4 HFpEF quartiles"):
        hfpef_q_sids = pheno_test.loc[
            (pheno_test["hf_subtype"] == "hfpef")
            & (pheno_test.index.isin(
                int_indexed.loc[int_indexed["intensity_q"] == q].index
            ))
        ].index.values

        if len(hfpef_q_sids) < 20:
            print(f"  Q{q}: n={len(hfpef_q_sids)} — TOO SMALL")
            continue

        hfpef_q_ds = build_scenario_dataset(
            state_df, outcomes_df, propensity_df,
            temporal_cols, static_cols, merge_map,
            hfpef_q_sids, scenario_cfg,
        )
        hfpef_q_result = evaluate_scenario(
            model, hfpef_q_ds, device, n_boot=100, boot_seed=BOOT_SEED + 500 + q
        )
        all_results.append({
            "scenario": f"G4_hfpef_Q{q}",
            "subgroup": "hfpef",
            "n_stays": len(hfpef_q_sids),
            **hfpef_q_result,
        })
        print(f"    Q{q}: n={len(hfpef_q_sids):,}, ATE={hfpef_q_result['ate']:+.4f}")

    # ── 5. Save results ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("[5] Saving results ...")

    results_df = pd.DataFrame(all_results)
    csv_path = TABLES_DIR / "85g_monitoring_sensitivity.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path} ({len(results_df)} rows)")

    # ── 6. Generate report ───────────────────────────────────────────
    print("\n[6] Generating report ...")

    # Extract key numbers for report
    g1_rows = results_df[results_df["scenario"].str.startswith("G1_intensity_Q")]
    ipmw_row = results_df[results_df["scenario"] == "G1_ipmw_aggregate"]
    g3_row = results_df[results_df["scenario"] == "G3_trend_slope"]

    ipmw_ate = ipmw_row["ate"].values[0] if len(ipmw_row) > 0 else np.nan
    trend_slope = g3_row["ate"].values[0] if len(g3_row) > 0 else np.nan
    trend_p = g3_row["v_rule"].values[0] if len(g3_row) > 0 else np.nan

    lines = [
        "=" * 72,
        "85g — Monitoring Process Sensitivity (MIMIC-IV)",
        "=" * 72,
        f"Generated: {ts}",
        f"Primary model: 80c G-Transformer DM (rollout, ow=1)",
        f"Bootstrap: B={N_BOOTSTRAP}, seed={BOOT_SEED}, CI={CI_LEVEL*100:.0f}%",
        f"Monitoring intensity: mean of _mask features across all bins/variables",
        "",
        "DISTINCTION FROM 85j",
        "────────────────────",
        "  85j tests whether missingness FEATURES inside the model drive ATE",
        "    (model-internal ablation of _mask/_tsl_h/_freq_24h channels).",
        "  85g tests whether the monitoring PROCESS (external confounder)",
        "    drives conclusions (IPMW reweighting + propensity overlap +",
        "    dose-response trend across monitoring intensity quartiles).",
        "",
        "G1 — IPMW-STRATIFIED ATE",
        "────────────────────────",
        f"  {'Quartile':<12s} {'n':>6s} {'ATE':>9s} {'CI_lo':>9s} {'CI_hi':>9s}",
        f"  {'-'*12} {'-'*6} {'-'*9} {'-'*9} {'-'*9}",
    ]

    for _, r in g1_rows.iterrows():
        lines.append(
            f"  {r['scenario']:<12s} {r['n_stays']:6d} {r['ate']:+9.4f} "
            f"{r['ate_ci_lo']:+9.4f} {r['ate_ci_hi']:+9.4f}"
        )

    lines.extend([
        "",
        f"  IPMW-weighted aggregate ATE: {ipmw_ate:+.4f}",
        f"  Baseline ATE:                {baseline_ate:+.4f}",
        f"  Δ (IPMW − baseline):         {ipmw_ate - baseline_ate:+.4f}" if pd.notna(ipmw_ate) else "",
    ])

    # G2 summary
    g2_rows = results_df[results_df["scenario"].str.startswith("G2_")]
    if len(g2_rows) > 0:
        lines.extend([
            "",
            "G2 — PROPENSITY OVERLAP BY MONITORING QUARTILE",
            "───────────────────────────────────────────────",
            f"  {'Quartile':<20s} {'n':>6s} {'Mean Entropy':>14s} {'SD':>9s}",
            f"  {'-'*20} {'-'*6} {'-'*14} {'-'*9}",
        ])
        for _, r in g2_rows.iterrows():
            lines.append(
                f"  {r['scenario']:<20s} {r['n_stays']:6d} "
                f"{r['ate']:14.4f} {r['v_observed']:9.4f}"
            )
        lines.extend([
            "",
            "  If entropy varies across quartiles → propensity model",
            "  assigns different action distributions by monitoring level",
            "  (potential treatment-monitoring confounding).",
        ])

    # G3 summary
    if pd.notna(trend_slope):
        lines.extend([
            "",
            "G3 — DOSE-RESPONSE TREND",
            "────────────────────────",
            f"  Linear trend slope:  {trend_slope:+.6f}",
            f"  Permutation p-value: {trend_p:.4f}",
            "",
            "  Significant slope (p < 0.05) → ATE varies systematically with",
            "  monitoring intensity, suggesting possible confounding by",
            "  measurement frequency.",
        ])

    # G4 summary
    g4_profile = results_df[results_df["scenario"] == "G4_monitoring_profile"]
    g4_hfpef_q = results_df[results_df["scenario"].str.startswith("G4_hfpef_Q")]
    if len(g4_profile) > 0:
        lines.extend([
            "",
            "G4 — HFpEF-SPECIFIC MONITORING CHECK",
            "─────────────────────────────────────",
        ])
        for _, r in g4_profile.iterrows():
            lines.append(
                f"  {r['subgroup']:<10s}: n={r['n_stays']:,}, "
                f"mean intensity={r['ate']:.4f} (sd={r['v_observed']:.4f})"
            )

        if len(g4_hfpef_q) > 0:
            lines.append("")
            lines.append("  HFpEF ATE by monitoring quartile:")
            for _, r in g4_hfpef_q.iterrows():
                lines.append(
                    f"    {r['scenario']}: n={r['n_stays']:,}, ATE={r['ate']:+.4f}"
                )

    # Overall interpretation
    lines.extend([
        "",
        "INTERPRETATION",
        "──────────────",
        f"  Baseline ATE: {baseline_ate:+.4f}",
        "",
        "  Robustness criteria:",
        "    - ROBUST if IPMW ATE within 50% of baseline AND trend p > 0.05",
        "    - SENSITIVE if IPMW ATE differs > 50% OR trend p < 0.05",
        "    - INCONCLUSIVE if propensity overlap varies dramatically by quartile",
    ])

    # Verdict
    ipmw_delta_pct = (
        abs(ipmw_ate - baseline_ate) / abs(baseline_ate) * 100
        if pd.notna(ipmw_ate) and abs(baseline_ate) > 1e-10 else np.nan
    )
    trend_sig = trend_p < 0.05 if pd.notna(trend_p) else False

    if pd.notna(ipmw_delta_pct) and ipmw_delta_pct < 50 and not trend_sig:
        verdict = "ROBUST"
        lines.append(
            f"  → ROBUST: IPMW Δ={ipmw_delta_pct:.1f}% (< 50%), "
            f"trend p={trend_p:.4f} (> 0.05)"
        )
    elif pd.notna(ipmw_delta_pct) and ipmw_delta_pct < 50:
        verdict = "PARTIALLY ROBUST"
        lines.append(
            f"  → PARTIALLY ROBUST: IPMW Δ={ipmw_delta_pct:.1f}% (< 50%) "
            f"but significant trend (p={trend_p:.4f})"
        )
    elif pd.notna(ipmw_delta_pct):
        verdict = "SENSITIVE"
        lines.append(
            f"  → SENSITIVE: IPMW Δ={ipmw_delta_pct:.1f}% (≥ 50%) — "
            f"monitoring intensity may confound the treatment effect"
        )
    else:
        verdict = "INCONCLUSIVE"
        lines.append("  → INCONCLUSIVE: insufficient data for IPMW assessment")

    report = "\n".join(lines) + "\n"
    report_path = REPORTS_DIR / "85g_monitoring_sensitivity_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {report_path}")
    print(report)

    total_time = time.time() - t0
    print(f"\n{'=' * 72}")
    print(f"85g COMPLETE. Total time: {total_time / 60:.1f} min")
    print(f"Verdict: {verdict}")
    print("=" * 72)


if __name__ == "__main__":
    main()
