#!/usr/bin/env python3
"""
85b_sequential_deconfounder_sensitivity — Sequential Deconfounder exploratory
hidden-confounding sensitivity analysis.

Purpose
-------
Fit a latent factor model (PPCA / VAE) to treatment sequences to detect
potential hidden confounders. If adding the inferred latent factor as a
covariate materially changes the ATE, this signals possible unmeasured
confounding. This is EXPLORATORY — the sequential deconfounder has known
limitations and is not a primary causal estimator.

Method (Wang & Blei 2019, adapted for DTR):
  1. Fit a probabilistic PCA (PPCA) on the treatment sequence A_{1:T}
     conditional on observed covariates X_{1:T} to recover latent factor Z.
  2. Check "substitute confounder" validity: Z must predict both treatment
     and outcome after conditioning on X.
  3. Re-evaluate 80c G-Transformer DM ATE on the ORIGINAL model
     (no retraining — model is frozen) but stratified by Z quantiles.
     If ATE varies systematically with Z → possible hidden confounding.
  4. Compute sensitivity parameter: max |ATE(Z=q1) - ATE(Z=q4)|
     as a measure of confounding vulnerability.

Note: We intentionally do NOT retrain the outcome model with Z. The protocol
specifies this as a sensitivity diagnostic, not a debiasing step.

Protocol reference: Step 85b (sensitivity layer)
Compute: C3 (PPCA is fast; ATE re-evaluation with bootstrap is the bottleneck)

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
- tables/85b_seqdeconf_sensitivity.csv
- reports/85b_seqdeconf_sensitivity_summary.txt
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hfpef_dtr.models.dtr_dataset import DTRDataset, load_dtr_data
from hfpef_dtr.models.dataset import make_train_val_test_split
from hfpef_dtr.models.ope import evaluate_dm, bootstrap_ope

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "tables"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"

N_LATENT_DIMS = 5  # Number of latent dimensions for PPCA
N_QUANTILES = 4    # Stratify by Z quantiles
N_BOOTSTRAP = 200  # Fewer bootstraps for subgroup (speed)
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


def fit_sequential_deconfounder(
    state_df: pd.DataFrame,
    propensity_df: pd.DataFrame,
    stay_ids: np.ndarray,
    temporal_cols: list[str],
    n_bins: int = 12,
    n_latent: int = N_LATENT_DIMS,
) -> tuple[pd.DataFrame, PCA, np.ndarray, int]:
    """
    Fit PPCA on treatment sequences residualized on state covariates.

    Returns (DataFrame, PCA, beta, n_actions_fit).
    """
    print("  [2a] Building treatment-residual matrix ...")
    stay_set = set(stay_ids)

    # Build per-stay action sequence matrix (n_stays, n_bins)
    prop_sub = propensity_df[propensity_df["stay_id"].isin(stay_set)].copy()
    act_col = "action_idx_merged" if "action_idx_merged" in prop_sub.columns else "action_idx"

    # Pivot to (stay_id, bin) → action
    action_wide = prop_sub.pivot_table(
        index="stay_id", columns="bin", values=act_col, aggfunc="first",
    ).reindex(columns=range(n_bins), fill_value=0).fillna(0).astype(int)

    # Pivot state features: mean per bin (already z-scored)
    state_sub = state_df[state_df["stay_id"].isin(stay_set)].copy()
    # Flatten state to per-stay summary (mean across bins)
    state_summary = state_sub.groupby("stay_id")[temporal_cols].mean().reindex(
        action_wide.index
    ).fillna(0)

    # One-hot encode actions per bin
    n_actions = prop_sub[act_col].max() + 1
    action_oh_list = []
    for b in range(n_bins):
        for a in range(n_actions):
            action_oh_list.append(
                (action_wide[b] == a).astype(float).rename(f"a{b}_c{a}")
            )
    action_oh = pd.concat(action_oh_list, axis=1)

    # Residualize: regress each action indicator on observed state summary.
    # This approximates PPCA on A_{1:T} | X and avoids simply re-encoding policy.
    ridge = 1e-4
    X_state = np.hstack([state_summary.values, np.ones((len(state_summary), 1))])
    Y_action = action_oh.values
    xtx = X_state.T @ X_state
    beta = np.linalg.solve(
        xtx + ridge * np.eye(xtx.shape[0]),
        X_state.T @ Y_action,
    )
    residual = Y_action - X_state @ beta

    print(f"  [2b] Fitting PPCA (n_components={n_latent}) on action sequences ...")
    X = residual  # (n_stays, n_bins * n_actions)

    pca = PCA(n_components=min(n_latent, X.shape[1], X.shape[0] - 1))
    Z = pca.fit_transform(X)  # (n_stays, n_latent_actual)
    n_actual = Z.shape[1]

    explained_var = pca.explained_variance_ratio_.sum()
    print(f"  [2c] PPCA fit: {n_actual} components, "
          f"explained variance = {explained_var:.3f}")

    # Build result DataFrame
    result = pd.DataFrame({"stay_id": action_wide.index})
    for d in range(n_actual):
        result[f"z_{d}"] = Z[:, d]

    # Primary latent score: first component
    z_primary = Z[:, 0]
    quantile_edges = np.percentile(z_primary, [25, 50, 75])
    z_q4 = np.digitize(z_primary, quantile_edges) + 1  # 1,2,3,4
    result["z_q4"] = z_q4

    return result, pca, beta, n_actions


def project_sequential_deconfounder(
    state_df: pd.DataFrame,
    propensity_df: pd.DataFrame,
    stay_ids: np.ndarray,
    temporal_cols: list[str],
    pca: PCA,
    beta: np.ndarray,
    n_bins: int,
    n_actions: int,
) -> pd.DataFrame:
    """Project new stays into trained PPCA latent space with train-fitted residualizer."""
    stay_set = set(stay_ids)
    prop_sub = propensity_df[propensity_df["stay_id"].isin(stay_set)].copy()
    act_col = "action_idx_merged" if "action_idx_merged" in prop_sub.columns else "action_idx"

    action_wide = prop_sub.pivot_table(
        index="stay_id", columns="bin", values=act_col, aggfunc="first",
    ).reindex(columns=range(n_bins), fill_value=0).fillna(0).astype(int)

    state_sub = state_df[state_df["stay_id"].isin(stay_set)].copy()
    state_summary = state_sub.groupby("stay_id")[temporal_cols].mean().reindex(
        action_wide.index
    ).fillna(0)
    action_oh_list = []
    for b in range(n_bins):
        for a in range(n_actions):
            action_oh_list.append(
                (action_wide[b] == a).astype(float).rename(f"a{b}_c{a}")
            )
    action_oh = pd.concat(action_oh_list, axis=1)

    X_state = np.hstack([state_summary.values, np.ones((len(state_summary), 1))])
    residual = action_oh.values - X_state @ beta
    Z = pca.transform(residual)

    result = pd.DataFrame({"stay_id": action_wide.index})
    for d in range(Z.shape[1]):
        result[f"z_{d}"] = Z[:, d]
    return result


def check_substitute_confounder(
    z_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    propensity_df: pd.DataFrame,
    stay_ids: np.ndarray,
    n_bins: int = 12,
) -> dict:
    """
    Validate Z as substitute confounder: must predict both treatment and outcome
    after conditioning on observed covariates (approximately).
    """
    stay_set = set(stay_ids)
    z_indexed = z_df.set_index("stay_id")
    z_cols = [c for c in z_indexed.columns if c.startswith("z_") and c != "z_q4"]

    # Outcome prediction: Z → mortality_28d
    out_sub = outcomes_df[outcomes_df["stay_id"].isin(stay_set)].set_index("stay_id")
    common = z_indexed.index.intersection(out_sub.index)
    Z_out = z_indexed.loc[common, z_cols].values
    y_out = out_sub.loc[common, "mortality_28d"].values

    if len(np.unique(y_out)) < 2:
        return {"z_predicts_outcome": False, "z_predicts_treatment": False,
                "outcome_auc": 0.5, "treatment_acc": 0.0}

    lr_out = LogisticRegression(max_iter=500, solver="lbfgs")
    lr_out.fit(Z_out, y_out)
    y_pred = lr_out.predict_proba(Z_out)[:, 1]
    try:
        auc_out = roc_auc_score(y_out, y_pred)
    except ValueError:
        auc_out = 0.5

    # Treatment prediction: Z → action at bin 0
    prop_sub = propensity_df[propensity_df["stay_id"].isin(stay_set)]
    act_col = "action_idx_merged" if "action_idx_merged" in prop_sub.columns else "action_idx"
    bin0 = prop_sub[prop_sub["bin"] == 0].set_index("stay_id")
    common_t = z_indexed.index.intersection(bin0.index)
    Z_treat = z_indexed.loc[common_t, z_cols].values
    a_treat = bin0.loc[common_t, act_col].values

    lr_treat = LogisticRegression(max_iter=500, solver="lbfgs")
    lr_treat.fit(Z_treat, a_treat)
    acc = lr_treat.score(Z_treat, a_treat)

    return {
        "z_predicts_outcome": auc_out > 0.55,
        "z_predicts_treatment": acc > 0.15,  # Better than uniform random
        "outcome_auc": float(auc_out),
        "treatment_acc": float(acc),
    }


def main() -> None:
    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("85b — Sequential Deconfounder Sensitivity Analysis")
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
    train_ids, _, test_ids = make_train_val_test_split(
        all_stay_ids, outcomes_df,
        train_frac=split_cfg["train_frac"],
        val_frac=split_cfg["val_frac"],
        test_frac=split_cfg["test_frac"],
        stratify_col=split_cfg["stratify_col"],
        seed=split_cfg["seed"],
    )
    print(f"  Train stays: {len(train_ids):,}")
    print(f"  Test stays:  {len(test_ids):,}")

    # ── 2. Fit PPCA on TRAIN set, project onto TEST set ──────────────
    print("\n[2] Fitting Sequential Deconfounder (PPCA) ...")
    z_train, pca_model, residual_beta, n_actions_fit = fit_sequential_deconfounder(
        state_df, propensity_df,
        train_ids, temporal_cols,
        n_bins=config["grid"]["n_bins"],
        n_latent=N_LATENT_DIMS,
    )

    # Project test set through same PCA
    print("  [2d] Projecting test set ...")
    z_test = project_sequential_deconfounder(
        state_df, propensity_df,
        test_ids, temporal_cols,
        pca=pca_model,
        beta=residual_beta,
        n_bins=config["grid"]["n_bins"],
        n_actions=n_actions_fit,
    )
    # Re-quantile using train set edges for consistency
    z_cols = [c for c in z_train.columns if c.startswith("z_") and c != "z_q4"]
    train_z0 = z_train["z_0"].values
    q_edges = np.percentile(train_z0, [25, 50, 75])
    z_test["z_q4"] = np.digitize(z_test["z_0"].values, q_edges) + 1

    print(f"  Test set Z quantile distribution: {pd.Series(z_test['z_q4']).value_counts().sort_index().to_dict()}")

    # ── 3. Substitute confounder validity check ──────────────────────
    print("\n[3] Checking substitute confounder validity ...")
    validity = check_substitute_confounder(
        z_test, outcomes_df, propensity_df, test_ids,
        n_bins=config["grid"]["n_bins"],
    )
    print(f"  Z → Outcome AUC:    {validity['outcome_auc']:.3f} "
          f"({'PASS' if validity['z_predicts_outcome'] else 'FAIL: Z does not predict outcome'})")
    print(f"  Z → Treatment Acc:  {validity['treatment_acc']:.3f} "
          f"({'PASS' if validity['z_predicts_treatment'] else 'FAIL: Z does not predict treatment'})")

    # ── 4. Load model and evaluate ATE by Z quantile ─────────────────
    print("\n[4] Loading 80c G-Transformer ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_temporal = len(temporal_cols)
    d_static = len(static_cols)
    n_actions = len(config["action"]["valid_merged_indices"])

    model = load_primary_model(config, d_temporal, d_static, n_actions, device)

    print("\n[5] Evaluating ATE by Z-quantile strata ...")
    z_test_indexed = z_test.set_index("stay_id")

    # Keep action vocabulary aligned with model training split.
    anchor_ds = DTRDataset(
        state_df=state_df, outcomes_df=outcomes_df, propensity_df=propensity_df,
        stay_ids=train_ids, temporal_cols=temporal_cols, static_cols=static_cols,
        merge_map=merge_map, n_bins=config["grid"]["n_bins"],
    )
    action_vocab = anchor_ds.action_vocab

    all_results = []

    # Full-sample baseline
    ds_full = DTRDataset(
        state_df=state_df, outcomes_df=outcomes_df, propensity_df=propensity_df,
        stay_ids=test_ids, temporal_cols=temporal_cols, static_cols=static_cols,
        merge_map=merge_map, n_bins=config["grid"]["n_bins"],
    )
    ds_full.action_vocab = action_vocab
    ds_full.n_actions = len(action_vocab)
    ds_full._action_to_contiguous = {a: i for i, a in enumerate(action_vocab)}
    full_result = evaluate_dm(model, ds_full, device, use_rollout=True)
    all_results.append({
        "stratum": "ALL",
        "n_stays": len(test_ids),
        "ate": full_result["ate"],
        "v_observed": full_result["v_observed"],
        "v_rule": full_result["v_rule"],
        "ate_ci_lo": np.nan,
        "ate_ci_hi": np.nan,
    })
    print(f"  ALL: ATE={full_result['ate']:+.4f}")

    # Per-quantile
    for q in tqdm(range(1, N_QUANTILES + 1), desc="Z-quantiles", unit="q"):
        q_sids = z_test_indexed.loc[z_test_indexed["z_q4"] == q].index.values
        q_sids = np.array(sorted(set(q_sids) & set(test_ids)))

        if len(q_sids) < 30:
            print(f"  Q{q}: n={len(q_sids)} — TOO SMALL")
            continue

        q_ds = DTRDataset(
            state_df=state_df, outcomes_df=outcomes_df, propensity_df=propensity_df,
            stay_ids=q_sids, temporal_cols=temporal_cols, static_cols=static_cols,
            merge_map=merge_map, n_bins=config["grid"]["n_bins"],
        )
        q_ds.action_vocab = action_vocab
        q_ds.n_actions = len(action_vocab)
        q_ds._action_to_contiguous = {a: i for i, a in enumerate(action_vocab)}

        q_point = evaluate_dm(model, q_ds, device, use_rollout=True)

        # Bootstrap CI
        def _eval_fn(ds, **kw):
            return evaluate_dm(model, ds, device, use_rollout=True)

        q_boot = bootstrap_ope(
            _eval_fn, q_ds,
            n_resamples=N_BOOTSTRAP, seed=BOOT_SEED + q, confidence=CI_LEVEL,
        )

        row = {
            "stratum": f"Z_Q{q}",
            "n_stays": len(q_sids),
            "ate": q_point["ate"],
            "v_observed": q_point["v_observed"],
            "v_rule": q_point["v_rule"],
            "ate_ci_lo": q_boot.get("ate_ci_lo", np.nan),
            "ate_ci_hi": q_boot.get("ate_ci_hi", np.nan),
        }
        all_results.append(row)
        print(f"  Q{q}: n={len(q_sids):,}, ATE={q_point['ate']:+.4f} "
              f"[{q_boot.get('ate_ci_lo', np.nan):+.4f}, {q_boot.get('ate_ci_hi', np.nan):+.4f}]")

    # ── 6. Compute sensitivity parameter ─────────────────────────────
    print("\n[6] Computing sensitivity parameter ...")
    q_ates = [r["ate"] for r in all_results if r["stratum"].startswith("Z_Q")]
    if len(q_ates) >= 2:
        sensitivity_param = max(q_ates) - min(q_ates)
        print(f"  Sensitivity parameter: |max(ATE) - min(ATE)| = {sensitivity_param:.4f}")
    else:
        sensitivity_param = np.nan

    # ── 7. Save results ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("[7] Saving results ...")

    results_df = pd.DataFrame(all_results)
    csv_path = TABLES_DIR / "85b_seqdeconf_sensitivity.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path} ({len(results_df)} rows)")

    # ── 8. Generate report ───────────────────────────────────────────
    lines = [
        "=" * 72,
        "85b — Sequential Deconfounder Sensitivity Analysis (MIMIC-IV)",
        "=" * 72,
        f"Generated: {ts}",
        f"Model: 80c G-Transformer DM (rollout, ow=1)",
        f"Method: PPCA on treatment sequences, {N_LATENT_DIMS} latent dims",
        f"Quantiles: {N_QUANTILES}, Bootstrap: B={N_BOOTSTRAP}",
        "",
        "SUBSTITUTE CONFOUNDER VALIDITY",
        "──────────────────────────────",
        f"  Z → Outcome (mortality_28d) AUC: {validity['outcome_auc']:.3f} "
        f"{'✓' if validity['z_predicts_outcome'] else '✗'}",
        f"  Z → Treatment (action_bin0) Acc: {validity['treatment_acc']:.3f} "
        f"{'✓' if validity['z_predicts_treatment'] else '✗'}",
        "",
        "  Interpretation: Z must predict both outcome and treatment to serve",
        "  as a substitute confounder. If either FAILS, the PPCA latent does",
        "  not capture meaningful hidden confounding and the test is moot.",
        "",
        "ATE BY LATENT QUANTILE",
        "──────────────────────",
        f"  {'Stratum':<10s} {'n':>6s} {'ATE':>9s} {'CI_lo':>9s} {'CI_hi':>9s}",
        f"  {'-'*10} {'-'*6} {'-'*9} {'-'*9} {'-'*9}",
    ]

    for _, r in results_df.iterrows():
        ci_lo = f"{r['ate_ci_lo']:+9.4f}" if pd.notna(r["ate_ci_lo"]) else "       —"
        ci_hi = f"{r['ate_ci_hi']:+9.4f}" if pd.notna(r["ate_ci_hi"]) else "       —"
        lines.append(
            f"  {r['stratum']:<10s} {r['n_stays']:6d} {r['ate']:+9.4f} {ci_lo} {ci_hi}"
        )

    lines.extend([
        "",
        f"  Sensitivity parameter (max - min ATE across quantiles): "
        f"{sensitivity_param:.4f}" if pd.notna(sensitivity_param) else "  N/A",
        "",
        "INTERPRETATION",
        "──────────────",
        "  If ATE is similar across Z quantiles (sensitivity param < 0.5 * |ATE_baseline|):",
        "    → Hidden confounding is unlikely to explain the treatment effect",
        "  If ATE varies systematically with Z:",
        "    → Cannot rule out hidden confounding; report as limitation",
        "",
        "  NOTE: This is an EXPLORATORY diagnostic, not a causal correction.",
        "  The sequential deconfounder has known theoretical limitations",
        "  (see D'Amour 2019). Results should supplement, not replace,",
        "  the primary causal analysis.",
    ])

    report = "\n".join(lines) + "\n"
    report_path = REPORTS_DIR / "85b_seqdeconf_sensitivity_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {report_path}")
    print(report)

    total_time = time.time() - t0
    print(f"\n{'=' * 72}")
    print(f"85b COMPLETE. Total time: {total_time / 60:.1f} min")
    print("=" * 72)


if __name__ == "__main__":
    main()
