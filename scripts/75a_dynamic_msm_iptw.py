#!/usr/bin/env python3
"""
75a_dynamic_msm_iptw — Primary classical causal DTR evaluation.

Purpose
-------
Evaluate dynamic treatment regimes using stabilised IPTW and marginal
structural models (MSMs).  This is the *primary classical baseline* in the
Step C model hierarchy (paired with G-Transformer 80c).

Method summary
--------------
1. Stabilised IPTW weights:
     SW_i = prod_t  P(A_t | A_{<t})  /  P(A_t | L_t, A_{<t})
   Numerator: marginal treatment model (logistic, action | prior actions).
   Denominator: 70b behavior-policy propensities (already fitted).

2. Dynamic MSM:
     logit P(Y=1 | cum_treatment; V) = beta' * f(cum_treatment, V)
   Fit by weighted logistic regression with SW weights.

3. Policy evaluation (WIS + DR):
   - Weighted Importance Sampling (self-normalised) for rule-based policy.
   - Doubly-robust estimator using a simple outcome model.

4. Subgroup / sensitivity:
   - Phenotype strata: rule_phenotype_2class, early_data_driven_k2
   - Trimming grid from 70f
   - HF-subtype stratification
   - coarse_2 sensitivity for HFpEF arm
   - Bootstrap CIs (B=500)

Protocol reference: Step 75a (v3.1), SAP 03f §8
Compute: C3

Inputs
------
- data/interim/45c_state_tensors_v1_mimic.parquet   (state features)
- data/interim/45b_actions_2h_v1_mimic.parquet       (discretised actions)
- data/interim/45d_outcomes_absorb_v1_mimic.parquet   (outcomes)
- data/interim/70b_propensity_coarse3_v1_mimic.parquet (propensities)
- data/interim/70c_rule_policy_actions_v1_mimic.parquet (rule-policy actions)
- data/processed/phenotype_labels_frozen_v1_mimic.parquet (phenotype strata)
- configs/70b_action_merge_map.yaml
- configs/70f_trimming_grid_v1.yaml
- configs/70d_rewards_metrics_v1.yaml
- configs/70e_ope_config_v1.yaml

Outputs
-------
- data/interim/75a_iptw_weights_v1_mimic.parquet
- tables/75a_msm_coefficients.csv
- tables/75a_policy_values.csv
- tables/75a_policy_values_by_stratum.csv
- tables/75a_trimming_sensitivity.csv
- tables/75a_bootstrap_ci.csv
- reports/75a_dynamic_msm_iptw_summary.txt
- figures/75a_weight_distribution.png
- figures/75a_policy_value_forest.png
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path

# ── Constants ────────────────────────────────────────────────────────────────
N_BINS = 12
EPSILON_SOFTEN = 0.05          # ε-greedy softening for deterministic rule policy
PROP_FLOOR = 0.005             # from 70f config
N_BOOTSTRAP = 500
RNG_SEED = 42


# ═════════════════════════════════════════════════════════════════════════════
# 1. Data loading
# ═════════════════════════════════════════════════════════════════════════════

def load_all_data(cfg: dict) -> dict:
    """Load and merge all upstream datasets needed for 75a."""
    interim = resolve_path(cfg, "interim")
    processed = resolve_path(cfg, "processed")
    configs_dir = resolve_path(cfg, "configs")

    print("[1] Loading upstream data...")

    # 1a. Actions
    actions = pd.read_parquet(
        interim / "45b_actions_2h_v1_mimic.parquet",
        columns=["stay_id", "bin", "action_idx"],
    )
    print(f"  Actions: {len(actions):,} rows")

    # 1b. Outcomes (per-stay)
    outcomes = pd.read_parquet(
        interim / "45d_outcomes_absorb_v1_mimic.parquet",
    )
    print(f"  Outcomes: {len(outcomes):,} stays")

    # 1c. Propensities (per obs)
    prop_cols = ["stay_id", "bin", "action_idx_merged", "propensity",
                 "propensity_clipped"]
    # Also load per-action probabilities for WIS
    prop_df = pd.read_parquet(interim / "70b_propensity_coarse3_v1_mimic.parquet")
    prob_cols = [c for c in prop_df.columns if c.startswith("prob_a")]
    print(f"  Propensities: {len(prop_df):,} rows, {len(prob_cols)} action-prob cols")

    # 1d. Rule-policy actions
    rule_df = pd.read_parquet(
        interim / "70c_rule_policy_actions_v1_mimic.parquet",
        columns=["stay_id", "bin", "rule_policy_action"],
    )
    print(f"  Rule-policy actions: {len(rule_df):,} rows")

    # 1e. Phenotype labels
    pheno = pd.read_parquet(processed / "phenotype_labels_frozen_v1_mimic.parquet")
    pheno_cols = [c for c in pheno.columns
                  if c in ("stay_id", "rule_phenotype_2class",
                           "rule_phenotype_4class", "early_data_driven_k2")]
    pheno = pheno[pheno_cols]
    print(f"  Phenotype labels: {len(pheno):,} stays")

    # 1f. State tensor (for MSM covariates: cumulative treatment + static)
    import pyarrow.parquet as pq
    st_path = interim / "45c_state_tensors_v1_mimic.parquet"
    all_st_cols = pq.read_schema(st_path).names
    want_cols = ["stay_id", "bin", "cumulative_fluid_balance_L",
                 "hf_subtype", "age", "gender"]
    use_cols = [c for c in want_cols if c in all_st_cols]
    st = pd.read_parquet(st_path, columns=use_cols)
    print(f"  State tensor subset: {len(st):,} rows, cols={use_cols}")

    # 1g. Action merge map
    merge_map_path = configs_dir / "70b_action_merge_map.yaml"
    with open(merge_map_path) as f:
        merge_cfg = yaml.safe_load(f)
    merge_map = {int(k): int(v) for k, v in merge_cfg.get("merge_map", {}).items()}
    print(f"  Merge map: {len(merge_map)} entries")

    # 1h. Trimming grid
    with open(configs_dir / "70f_trimming_grid_v1.yaml") as f:
        trim_cfg = yaml.safe_load(f)

    return {
        "actions": actions,
        "outcomes": outcomes,
        "propensities": prop_df,
        "prob_cols": prob_cols,
        "rule_actions": rule_df,
        "phenotypes": pheno,
        "states": st,
        "merge_map": merge_map,
        "trim_cfg": trim_cfg,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 2. Build analysis panel
# ═════════════════════════════════════════════════════════════════════════════

def build_panel(data: dict) -> pd.DataFrame:
    """Merge all datasets into a single analysis panel."""
    print("\n[2] Building analysis panel...")

    panel = data["actions"].copy()

    # Merge propensities
    prop_merge_cols = ["stay_id", "bin", "action_idx_merged",
                       "propensity", "propensity_clipped"] + data["prob_cols"]
    panel = panel.merge(
        data["propensities"][prop_merge_cols],
        on=["stay_id", "bin"], how="left",
    )

    # Apply merge map to actions
    merge_map = data["merge_map"]
    panel["action_merged"] = panel["action_idx"].map(
        lambda x: merge_map.get(x, x)
    )

    # Merge rule-policy actions
    panel = panel.merge(data["rule_actions"], on=["stay_id", "bin"], how="left")

    # Apply merge map to rule-policy actions too
    panel["rule_action_merged"] = panel["rule_policy_action"].map(
        lambda x: merge_map.get(x, x)
    )

    # Merge outcomes (per-stay)
    outcome_cols = ["stay_id", "mortality_28d", "mortality_7d",
                    "absorbing_state_bin", "absorbing_type"]
    avail_cols = [c for c in outcome_cols if c in data["outcomes"].columns]
    panel = panel.merge(data["outcomes"][avail_cols], on="stay_id", how="left")

    # Merge phenotype labels
    panel = panel.merge(data["phenotypes"], on="stay_id", how="left")

    # Merge state features (for MSM + HF subtype)
    state_merge = data["states"][
        [c for c in data["states"].columns
         if c in ("stay_id", "bin", "cumulative_fluid_balance_L",
                  "hf_subtype", "age", "gender")]
    ]
    panel = panel.merge(state_merge, on=["stay_id", "bin"], how="left")

    # Mark post-absorbing rows (no treatment effect defined after death/discharge)
    if "absorbing_state_bin" in panel.columns:
        panel["post_absorbing"] = (
            panel["absorbing_state_bin"].notna()
            & (panel["bin"] > panel["absorbing_state_bin"])
        )
    else:
        panel["post_absorbing"] = False

    n_stays = panel["stay_id"].nunique()
    n_obs = len(panel)
    n_post_abs = panel["post_absorbing"].sum()
    print(f"  Panel: {n_obs:,} rows, {n_stays:,} stays")
    print(f"  Post-absorbing rows (excluded from weighting): {n_post_abs:,}")

    return panel


# ═════════════════════════════════════════════════════════════════════════════
# 3. Stabilised IPTW weights
# ═════════════════════════════════════════════════════════════════════════════

def fit_numerator_model(panel: pd.DataFrame) -> np.ndarray:
    """
    Fit marginal treatment model P(A_t | A_{<t}) for IPTW numerator.

    Uses a simple logistic regression of current action on lagged actions.
    Returns numerator probabilities for each observation.
    """
    print("\n[3a] Fitting numerator model P(A_t | A_bar_{<t})...")

    # Build lag features: action at t-1, t-2
    panel_sorted = panel.sort_values(["stay_id", "bin"])
    lag_features = []
    for lag in [1, 2]:
        col = f"action_lag{lag}"
        panel_sorted[col] = panel_sorted.groupby("stay_id")["action_merged"].shift(lag)
        lag_features.append(col)

    # Add bin indicator
    lag_features.append("bin")

    # Exclude post-absorbing rows from fitting
    fit_mask = ~panel_sorted["post_absorbing"]
    X_fit = panel_sorted.loc[fit_mask, lag_features].fillna(-1).values
    y_fit = panel_sorted.loc[fit_mask, "action_merged"].values

    n_classes = len(np.unique(y_fit))
    print(f"  Features: {len(lag_features)}, classes: {n_classes}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = LogisticRegression(
            max_iter=500, solver="lbfgs", C=1.0,
        )
        lr.fit(X_fit, y_fit)

    train_acc = lr.score(X_fit, y_fit)
    print(f"  Numerator model accuracy: {train_acc:.4f}")

    # Predict numerator probabilities for ALL rows
    X_all = panel_sorted[lag_features].fillna(-1).values
    probs_all = lr.predict_proba(X_all)
    classes = lr.classes_

    # Get P(observed action) from numerator model
    actions_all = panel_sorted["action_merged"].values
    numer_probs = np.zeros(len(panel_sorted))
    for i, a in enumerate(actions_all):
        idx = np.where(classes == a)[0]
        if len(idx) > 0:
            numer_probs[i] = probs_all[i, idx[0]]
        else:
            numer_probs[i] = 1.0 / n_classes  # fallback for unseen

    numer_probs = np.clip(numer_probs, PROP_FLOOR, 1.0)

    # Restore original order
    numer_series = pd.Series(numer_probs, index=panel_sorted.index)
    return numer_series.loc[panel.index].values


def compute_stabilised_weights(
    panel: pd.DataFrame, numer_probs: np.ndarray
) -> pd.DataFrame:
    """
    Compute stabilised cumulative IPTW weights.

    SW_i(t) = prod_{s=0}^{t} numer_s / denom_s
    where denom = P(A_t | L_t, A_{<t}) from 70b behavior policy.
    """
    print("\n[3b] Computing stabilised IPTW weights...")

    denom = panel["propensity_clipped"].values
    numer = numer_probs

    # Per-step importance ratio
    step_ratio = numer / denom
    step_ratio = np.clip(step_ratio, 0.01, 200.0)  # safety cap per-step

    panel = panel.copy()
    panel["step_ratio"] = step_ratio
    panel["log_step_ratio"] = np.log(step_ratio)

    # Cumulative product within each stay (on log scale for stability)
    panel["cum_log_ratio"] = panel.groupby("stay_id")["log_step_ratio"].cumsum()
    panel["sw_raw"] = np.exp(panel["cum_log_ratio"])

    # Zero out post-absorbing weights
    panel.loc[panel["post_absorbing"], "sw_raw"] = 0.0

    # Get terminal (last active bin) weight per stay
    active = panel[~panel["post_absorbing"]]
    terminal_idx = active.groupby("stay_id")["bin"].idxmax()
    panel["is_terminal"] = False
    panel.loc[terminal_idx, "is_terminal"] = True
    panel["sw_terminal"] = panel["sw_raw"]
    panel.loc[~panel["is_terminal"], "sw_terminal"] = np.nan

    sw = panel.loc[panel["is_terminal"], "sw_raw"]
    print(f"  Stabilised weights (terminal): mean={sw.mean():.3f}, "
          f"median={sw.median():.3f}, P95={np.percentile(sw, 95):.3f}, "
          f"P99={np.percentile(sw, 99):.3f}, max={sw.max():.3f}")

    return panel


# ═════════════════════════════════════════════════════════════════════════════
# 4. Apply trimming
# ═════════════════════════════════════════════════════════════════════════════

def apply_trimming(
    panel: pd.DataFrame, lo_pct: float, hi_pct: float
) -> pd.DataFrame:
    """Trim stabilised weights at the given percentiles."""
    active = panel[panel["is_terminal"]].copy()
    w = active["sw_raw"].values
    lo_val = np.percentile(w, lo_pct)
    hi_val = np.percentile(w, hi_pct)
    trimmed = np.clip(w, lo_val, hi_val)
    # Self-normalise
    trimmed = trimmed / trimmed.sum() * len(trimmed)
    active["sw_trimmed"] = trimmed

    # Merge back
    panel = panel.copy()
    panel["sw_trimmed"] = np.nan
    panel.loc[active.index, "sw_trimmed"] = active["sw_trimmed"].values
    return panel


# ═════════════════════════════════════════════════════════════════════════════
# 5. Dynamic MSM
# ═════════════════════════════════════════════════════════════════════════════

def fit_dynamic_msm(panel: pd.DataFrame) -> dict:
    """
    Fit a weighted marginal structural model.

    MSM: logit P(Y=1) = β₀ + β₁·mean_fluid_cat + β₂·mean_vaso_cat
                       + β₃·any_diuretic_fraction

    Weighted by trimmed stabilised IPTW weights.
    Returns coefficients and per-stay predicted risks.
    """
    print("\n[5] Fitting dynamic MSM (weighted logistic regression)...")

    # Summarise treatment trajectory per stay (mean category levels)
    actions = panel[~panel["post_absorbing"]].copy()
    # Decode action components from action_merged
    actions["fluid_cat"] = actions["action_merged"] // 6
    actions["vaso_cat"] = (actions["action_merged"] % 6) // 2
    actions["diuretic_cat"] = actions["action_merged"] % 2

    trt_summary = actions.groupby("stay_id").agg(
        mean_fluid=("fluid_cat", "mean"),
        mean_vaso=("vaso_cat", "mean"),
        diuretic_frac=("diuretic_cat", "mean"),
        n_active_bins=("bin", "count"),
    ).reset_index()

    # Merge outcome + weights
    terminal = panel[panel["is_terminal"]][
        ["stay_id", "mortality_28d", "sw_trimmed", "hf_subtype"]
    ].copy()
    msm_df = trt_summary.merge(terminal, on="stay_id", how="inner")
    msm_df = msm_df.dropna(subset=["sw_trimmed", "mortality_28d"])

    X_cols = ["mean_fluid", "mean_vaso", "diuretic_frac"]
    X = msm_df[X_cols].values
    y = msm_df["mortality_28d"].values.astype(int)
    w = msm_df["sw_trimmed"].values

    print(f"  MSM data: {len(msm_df):,} stays, {y.sum()} deaths")
    print(f"  Treatment summary ranges: fluid=[{X[:,0].min():.2f},{X[:,0].max():.2f}], "
          f"vaso=[{X[:,1].min():.2f},{X[:,1].max():.2f}], "
          f"diuretic=[{X[:,2].min():.2f},{X[:,2].max():.2f}]")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = LogisticRegression(
            max_iter=1000, solver="lbfgs", penalty=None,
        )
        lr.fit(X, y, sample_weight=w)

    coefs = dict(zip(X_cols, lr.coef_[0]))
    coefs["intercept"] = lr.intercept_[0]

    print("  MSM coefficients:")
    for name, val in coefs.items():
        or_val = np.exp(val) if name != "intercept" else np.exp(val)
        print(f"    {name}: β={val:.4f}, OR={or_val:.4f}")

    # Predicted risk under observed treatment
    msm_df["pred_risk"] = lr.predict_proba(X)[:, 1]

    return {
        "model": lr,
        "coefs": coefs,
        "X_cols": X_cols,
        "msm_df": msm_df,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 6. Policy evaluation (WIS + DR)
# ═════════════════════════════════════════════════════════════════════════════

def compute_policy_weights(
    panel: pd.DataFrame, prob_cols: list[str], merge_map: dict,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute per-trajectory importance weights for the softened rule policy.

    π_ε(a|s) = (1-ε)·I(a = rule(s)) + ε/|A|
    w_i = prod_t π_ε(A_t|S_t) / μ(A_t|S_t)
    """
    if verbose:
        print("\n[6a] Computing rule-policy importance weights (WIS)...")

    active = panel[~panel["post_absorbing"]].copy()
    n_actions = len(prob_cols)

    # Rule policy probability for observed action under ε-greedy softening
    rule_match = (active["action_merged"] == active["rule_action_merged"]).astype(float)
    pi_rule = (1 - EPSILON_SOFTEN) * rule_match + EPSILON_SOFTEN / n_actions

    # Behavior policy probability for observed action
    mu = active["propensity_clipped"].values

    # Per-step ratio
    step_ratio = pi_rule.values / mu
    step_ratio = np.clip(step_ratio, 0.001, 200.0)

    active = active.copy()
    active["wis_step_ratio"] = step_ratio
    active["wis_log_ratio"] = np.log(step_ratio)
    active["wis_cum_log"] = active.groupby("stay_id")["wis_log_ratio"].cumsum()
    active["wis_raw"] = np.exp(active["wis_cum_log"])

    # Terminal weight per stay
    terminal_idx = active.groupby("stay_id")["bin"].idxmax()
    wis_terminal = active.loc[terminal_idx, ["stay_id", "wis_raw"]].set_index("stay_id")

    return wis_terminal


def wis_estimator(
    weights: pd.Series, outcomes: pd.Series
) -> dict:
    """Self-normalised weighted importance sampling estimator."""
    w = weights.values
    y = outcomes.values
    # Self-normalise
    w_norm = w / w.sum()
    v_hat = np.sum(w_norm * y)
    # Effective sample size
    ess = (np.sum(w) ** 2) / np.sum(w ** 2)
    return {"value": float(v_hat), "ess": float(ess), "n": len(w)}


def fit_outcome_model(panel: pd.DataFrame) -> LogisticRegression:
    """Simple outcome model for doubly-robust estimation."""
    print("  Fitting outcome model for DR...")

    terminal = panel[panel["is_terminal"]].copy()
    terminal = terminal.dropna(subset=["mortality_28d"])

    # Decode actions
    terminal["fluid_cat"] = terminal["action_merged"] // 6
    terminal["vaso_cat"] = (terminal["action_merged"] % 6) // 2
    terminal["diuretic_cat"] = terminal["action_merged"] % 2

    # Build features: treatment summary + available covariates
    trt_cols = ["fluid_cat", "vaso_cat", "diuretic_cat"]
    X = terminal[trt_cols].fillna(0).values
    y = terminal["mortality_28d"].values.astype(int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = LogisticRegression(max_iter=500, solver="lbfgs", penalty=None)
        lr.fit(X, y)

    print(f"  Outcome model accuracy: {lr.score(X, y):.4f}")
    return lr


def dr_estimator(
    wis_weights: pd.Series,
    outcomes: pd.Series,
    predicted_outcomes: pd.Series,
) -> dict:
    """Doubly-robust estimator."""
    w = wis_weights.values
    y = outcomes.values
    y_hat = predicted_outcomes.values
    w_norm = w / w.sum()

    dr_value = float(np.mean(y_hat) + np.sum(w_norm * (y - y_hat)))
    return {"value": dr_value, "n": len(w)}


def evaluate_policies(
    panel: pd.DataFrame, prob_cols: list[str], merge_map: dict,
    trim_lo: float, trim_hi: float,
) -> dict:
    """Evaluate observed and rule-based policies."""
    print("\n[6] Evaluating policies...")

    # ── Observed policy value (simple mean) ──
    terminal = panel[panel["is_terminal"]].copy()
    terminal = terminal.dropna(subset=["mortality_28d"])
    obs_value = terminal["mortality_28d"].mean()
    print(f"  Observed policy value (mortality rate): {obs_value:.4f}")

    # ── Rule-policy WIS ──
    wis_terminal = compute_policy_weights(panel, prob_cols, merge_map, verbose=True)
    wis_terminal = wis_terminal.join(
        terminal.set_index("stay_id")[["mortality_28d"]], how="inner"
    )

    # Trim WIS weights
    w = wis_terminal["wis_raw"].values
    lo_val = np.percentile(w, trim_lo)
    hi_val = np.percentile(w, trim_hi)
    wis_terminal["wis_trimmed"] = np.clip(w, lo_val, hi_val)

    wis_result = wis_estimator(
        wis_terminal["wis_trimmed"], wis_terminal["mortality_28d"]
    )
    print(f"  Rule-policy WIS value: {wis_result['value']:.4f} "
          f"(ESS={wis_result['ess']:.0f})")

    # ── DR estimator ──
    print("\n[6b] Doubly-robust estimation...")
    outcome_model = fit_outcome_model(panel)

    # Predicted outcomes under observed treatment
    terminal_for_dr = terminal.copy()
    terminal_for_dr["fluid_cat"] = terminal_for_dr["action_merged"] // 6
    terminal_for_dr["vaso_cat"] = (terminal_for_dr["action_merged"] % 6) // 2
    terminal_for_dr["diuretic_cat"] = terminal_for_dr["action_merged"] % 2
    trt_cols = ["fluid_cat", "vaso_cat", "diuretic_cat"]
    y_hat = outcome_model.predict_proba(
        terminal_for_dr[trt_cols].fillna(0).values
    )[:, 1]
    terminal_for_dr["y_hat"] = y_hat

    # Align with WIS
    dr_df = wis_terminal.join(
        terminal_for_dr.set_index("stay_id")[["y_hat"]], how="inner"
    )
    dr_result = dr_estimator(
        dr_df["wis_trimmed"], dr_df["mortality_28d"], dr_df["y_hat"]
    )
    print(f"  DR value: {dr_result['value']:.4f}")

    return {
        "observed_value": obs_value,
        "wis_result": wis_result,
        "dr_result": dr_result,
        "wis_terminal": wis_terminal,
        "outcome_model": outcome_model,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 7. Subgroup analysis
# ═════════════════════════════════════════════════════════════════════════════

def subgroup_analysis(
    panel: pd.DataFrame, prob_cols: list[str], merge_map: dict,
    trim_lo: float, trim_hi: float,
) -> pd.DataFrame:
    """Run policy evaluation within phenotype and HF-subtype strata."""
    print("\n[7] Subgroup analyses...")

    results = []
    stratifiers = []

    if "rule_phenotype_2class" in panel.columns:
        stratifiers.append("rule_phenotype_2class")
    if "early_data_driven_k2" in panel.columns:
        stratifiers.append("early_data_driven_k2")
    if "hf_subtype" in panel.columns:
        stratifiers.append("hf_subtype")

    for strat_col in stratifiers:
        groups = panel[strat_col].dropna().unique()
        for grp in sorted(groups, key=str):
            mask = panel[strat_col] == grp
            sub = panel[mask].copy()
            n_stays = sub["stay_id"].nunique()
            if n_stays < 30:
                print(f"  Skipping {strat_col}={grp} (n={n_stays} < 30)")
                continue

            terminal = sub[sub["is_terminal"]].dropna(subset=["mortality_28d"])
            obs_value = terminal["mortality_28d"].mean()

            # WIS for rule policy
            wis_t = compute_policy_weights(sub, prob_cols, merge_map, verbose=False)
            wis_t = wis_t.join(
                terminal.set_index("stay_id")[["mortality_28d"]], how="inner"
            )
            w = wis_t["wis_raw"].values
            if len(w) > 10:
                lo_v = np.percentile(w, trim_lo)
                hi_v = np.percentile(w, trim_hi)
                wis_t["wis_trimmed"] = np.clip(w, lo_v, hi_v)
                wis_res = wis_estimator(wis_t["wis_trimmed"], wis_t["mortality_28d"])
            else:
                wis_res = {"value": np.nan, "ess": 0.0, "n": len(w)}

            results.append({
                "stratifier": strat_col,
                "group": str(grp),
                "n_stays": n_stays,
                "observed_mortality": obs_value,
                "wis_rule_value": wis_res["value"],
                "wis_ess": wis_res["ess"],
                "ate_wis": wis_res["value"] - obs_value,
            })
            print(f"  {strat_col}={grp}: n={n_stays}, "
                  f"obs={obs_value:.4f}, WIS={wis_res['value']:.4f}")

    return pd.DataFrame(results)


# ═════════════════════════════════════════════════════════════════════════════
# 8. Trimming sensitivity
# ═════════════════════════════════════════════════════════════════════════════

def trimming_sensitivity(
    panel: pd.DataFrame, prob_cols: list[str], merge_map: dict,
    trim_cfg: dict,
) -> pd.DataFrame:
    """Evaluate rule policy across all trimming brackets."""
    print("\n[8] Trimming sensitivity...")

    results_list = trim_cfg["weight_trimming_grid"]["results"]
    rows = []

    for r in results_list:
        lo = r["percentile_lo"]
        hi = r["percentile_hi"]

        terminal = panel[panel["is_terminal"]].dropna(subset=["mortality_28d"])
        obs_value = terminal["mortality_28d"].mean()

        wis_t = compute_policy_weights(panel, prob_cols, merge_map, verbose=False)
        wis_t = wis_t.join(
            terminal.set_index("stay_id")[["mortality_28d"]], how="inner"
        )
        w = wis_t["wis_raw"].values
        lo_v = np.percentile(w, lo)
        hi_v = np.percentile(w, hi)
        wis_t["wis_trimmed"] = np.clip(w, lo_v, hi_v)
        wis_res = wis_estimator(wis_t["wis_trimmed"], wis_t["mortality_28d"])

        rows.append({
            "trim_lo": lo,
            "trim_hi": hi,
            "observed_mortality": obs_value,
            "wis_rule_value": wis_res["value"],
            "wis_ess": wis_res["ess"],
            "ate_wis": wis_res["value"] - obs_value,
        })
        print(f"  [{lo:5.1f}, {hi:5.1f}]: WIS={wis_res['value']:.4f}, "
              f"ESS={wis_res['ess']:.0f}")

    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# 9. Bootstrap CIs
# ═════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(
    panel: pd.DataFrame, prob_cols: list[str], merge_map: dict,
    trim_lo: float, trim_hi: float, n_boot: int = N_BOOTSTRAP,
) -> dict:
    """Bootstrap confidence intervals for policy values."""
    print(f"\n[9] Bootstrap ({n_boot} resamples)...")

    stay_ids = panel["stay_id"].unique()
    rng = np.random.RandomState(RNG_SEED)

    obs_boots = []
    wis_boots = []
    dr_boots = []
    wis_failures = 0

    for b in range(n_boot):
        if (b + 1) % 100 == 0:
            print(f"  Bootstrap {b+1}/{n_boot}...")

        # Resample stays (weight-based bootstrap avoids expensive row duplication)
        boot_ids = rng.choice(stay_ids, size=len(stay_ids), replace=True)
        boot_counts = pd.Series(boot_ids).value_counts()
        boot_panel = panel[panel["stay_id"].isin(boot_counts.index)].copy()
        boot_panel["_bw"] = boot_panel["stay_id"].map(boot_counts).values

        terminal = boot_panel[boot_panel["is_terminal"]].copy()
        terminal = terminal.dropna(subset=["mortality_28d"])

        if len(terminal) < 10:
            continue

        # Observed (weighted by bootstrap counts)
        bw_t = terminal["_bw"].values
        obs_boots.append(
            np.average(terminal["mortality_28d"].values, weights=bw_t)
        )

        # WIS (multiply WIS weights by bootstrap counts, then self-normalise)
        try:
            wis_t = compute_policy_weights(boot_panel, prob_cols, merge_map, verbose=False)
            wis_t = wis_t.join(
                terminal.set_index("stay_id")[["mortality_28d", "_bw"]], how="inner"
            )
            w = wis_t["wis_raw"].values * wis_t["_bw"].values
            lo_v = np.percentile(w, trim_lo)
            hi_v = np.percentile(w, trim_hi)
            w_trimmed = np.clip(w, lo_v, hi_v)
            w_norm = w_trimmed / w_trimmed.sum()
            wis_boots.append(float(np.sum(w_norm * wis_t["mortality_28d"].values)))
        except Exception:
            wis_failures += 1

    # Compute CIs
    ci = {}
    if obs_boots:
        ci["observed"] = {
            "mean": float(np.mean(obs_boots)),
            "ci_lo": float(np.percentile(obs_boots, 2.5)),
            "ci_hi": float(np.percentile(obs_boots, 97.5)),
        }
    if wis_boots:
        ci["wis_rule"] = {
            "mean": float(np.mean(wis_boots)),
            "ci_lo": float(np.percentile(wis_boots, 2.5)),
            "ci_hi": float(np.percentile(wis_boots, 97.5)),
        }
    if wis_boots and obs_boots:
        n_common = min(len(wis_boots), len(obs_boots))
        ate_boots = [wis_boots[i] - obs_boots[i] for i in range(n_common)]
        ci["ate_wis"] = {
            "mean": float(np.mean(ate_boots)),
            "ci_lo": float(np.percentile(ate_boots, 2.5)),
            "ci_hi": float(np.percentile(ate_boots, 97.5)),
        }

    print(f"  Bootstrap complete: {len(obs_boots)} obs, {len(wis_boots)} WIS")
    if wis_failures > 0:
        print(f"  Warning: {wis_failures} bootstrap replicates failed during WIS computation")
    for key, vals in ci.items():
        print(f"  {key}: {vals['mean']:.4f} "
              f"[{vals['ci_lo']:.4f}, {vals['ci_hi']:.4f}]")

    return ci


# ═════════════════════════════════════════════════════════════════════════════
# 10. Figures
# ═════════════════════════════════════════════════════════════════════════════

def plot_weight_distribution(panel: pd.DataFrame, fig_dir: Path) -> None:
    """Plot stabilised weight distribution after [5,95] trimming.

    Shows the trimmed (primary) weight distribution used in the MSM analysis,
    annotated with the effective sample size (ESS) and trimming fraction.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    terminal = panel[panel["is_terminal"]].copy()
    w_raw = terminal["sw_raw"].values
    w_trim = terminal["sw_trimmed"].dropna().values

    # Compute stats
    lo_cut = np.percentile(w_raw, 5)
    hi_cut = np.percentile(w_raw, 95)
    n_total = len(w_raw)
    n_trimmed_obs = int(np.sum((w_raw < lo_cut) | (w_raw > hi_cut)))
    trim_pct = n_trimmed_obs / n_total * 100
    ess = (np.sum(w_trim) ** 2) / np.sum(w_trim ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: trimmed weight histogram ──
    axes[0].hist(w_trim, bins=80, edgecolor="black", alpha=0.7, color="#4878cf")
    axes[0].set_xlabel("Stabilised IPTW weight (trimmed)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Stabilised Weight Distribution\n([P5, P95] trimmed)")
    axes[0].axvline(1.0, color="red", linestyle="--", linewidth=1.2, label="w = 1")
    axes[0].legend(loc="upper right")
    # Annotation box
    stats_text = (
        f"Trimming: [P5, P95]\n"
        f"Trimmed: {n_trimmed_obs:,} / {n_total:,} ({trim_pct:.0f}%)\n"
        f"ESS = {ess:,.0f}"
    )
    axes[0].text(0.97, 0.70, stats_text, transform=axes[0].transAxes,
                 fontsize=9, verticalalignment="top", horizontalalignment="right",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8))

    # ── Right: log-scale trimmed weights ──
    log_w = np.log(w_trim + 1e-10)
    axes[1].hist(log_w, bins=80, edgecolor="black", alpha=0.7, color="#4878cf")
    axes[1].set_xlabel("log(Stabilised IPTW weight)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Log Weight Distribution (after trimming)")
    axes[1].axvline(0.0, color="red", linestyle="--", linewidth=1.2, label="log(w) = 0")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    path = fig_dir / "75a_weight_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.savefig(str(path).replace('.png', '.pdf'), format='pdf',
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"  Saved: {path} (+pdf)")


def plot_policy_value_forest(
    policy_results: dict, subgroup_df: pd.DataFrame,
    bootstrap_ci: dict, fig_dir: Path,
) -> None:
    """Forest plot of policy values across strata."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    # Overall
    obs_ci = bootstrap_ci.get("observed", {})
    wis_ci = bootstrap_ci.get("wis_rule", {})
    rows.append({
        "label": "Overall (observed)",
        "value": policy_results["observed_value"],
        "ci_lo": obs_ci.get("ci_lo", np.nan),
        "ci_hi": obs_ci.get("ci_hi", np.nan),
        "type": "observed",
    })
    rows.append({
        "label": "Overall (rule WIS)",
        "value": policy_results["wis_result"]["value"],
        "ci_lo": wis_ci.get("ci_lo", np.nan),
        "ci_hi": wis_ci.get("ci_hi", np.nan),
        "type": "rule",
    })

    # Subgroups
    for _, row in subgroup_df.iterrows():
        rows.append({
            "label": f"{row['stratifier']}={row['group']} (obs)",
            "value": row["observed_mortality"],
            "ci_lo": np.nan, "ci_hi": np.nan,
            "type": "observed",
        })
        rows.append({
            "label": f"{row['stratifier']}={row['group']} (WIS)",
            "value": row["wis_rule_value"],
            "ci_lo": np.nan, "ci_hi": np.nan,
            "type": "rule",
        })

    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.4)))
    y_pos = np.arange(len(plot_df))

    colors = {"observed": "steelblue", "rule": "darkorange"}
    for _, row in plot_df.iterrows():
        idx = plot_df.index.get_loc(row.name)
        color = colors.get(row["type"], "gray")
        ax.errorbar(
            row["value"], idx,
            xerr=[[row["value"] - row["ci_lo"]] if not np.isnan(row["ci_lo"]) else [0],
                  [row["ci_hi"] - row["value"]] if not np.isnan(row["ci_hi"]) else [0]],
            fmt="o", color=color, capsize=3, markersize=6,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["label"], fontsize=8)
    ax.set_xlabel("28-day Mortality Rate")
    ax.set_title("Policy Value Estimates (Observed vs Rule-Based)")
    ax.axvline(plot_df.iloc[0]["value"], color="gray", linestyle=":", alpha=0.5)
    ax.invert_yaxis()
    plt.tight_layout()

    path = fig_dir / "75a_policy_value_forest.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.savefig(str(path).replace('.png', '.pdf'), format='pdf',
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"  Saved: {path} (+pdf)")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    proj_cfg = load_config()
    interim_dir = resolve_path(proj_cfg, "interim")
    tables_dir = resolve_path(proj_cfg, "tables")
    reports_dir = resolve_path(proj_cfg, "reports")
    figures_dir = resolve_path(proj_cfg, "figures")

    print("=" * 72)
    print("75a — Dynamic MSM / IPTW (Primary Classical DTR Evaluation)")
    print("=" * 72)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Load ──────────────────────────────────────────────────────────────
    data = load_all_data(proj_cfg)

    # ── Build panel ───────────────────────────────────────────────────────
    panel = build_panel(data)

    # ── Numerator model ───────────────────────────────────────────────────
    numer_probs = fit_numerator_model(panel)

    # ── Stabilised weights ────────────────────────────────────────────────
    panel = compute_stabilised_weights(panel, numer_probs)

    # ── Primary trimming (from 70f config) ────────────────────────────────
    primary_trim = data["trim_cfg"]["weight_trimming_grid"]["primary"]
    trim_lo = primary_trim["lo"]
    trim_hi = primary_trim["hi"]
    print(f"\n[4] Applying primary trimming: [P{trim_lo}, P{trim_hi}]")
    panel = apply_trimming(panel, trim_lo, trim_hi)

    # Save weights
    weight_path = interim_dir / "75a_iptw_weights_v1_mimic.parquet"
    weight_cols = ["stay_id", "bin", "step_ratio", "sw_raw",
                   "is_terminal", "sw_trimmed", "post_absorbing"]
    panel[weight_cols].to_parquet(weight_path, index=False)
    print(f"  Saved: {weight_path}")

    # ── Dynamic MSM ───────────────────────────────────────────────────────
    msm_results = fit_dynamic_msm(panel)

    # Save MSM coefficients
    tables_dir.mkdir(parents=True, exist_ok=True)
    coef_df = pd.DataFrame([msm_results["coefs"]])
    coef_path = tables_dir / "75a_msm_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)
    print(f"  Saved: {coef_path}")

    # ── Policy evaluation ─────────────────────────────────────────────────
    policy_results = evaluate_policies(
        panel, data["prob_cols"], data["merge_map"], trim_lo, trim_hi,
    )

    # Save policy values
    pv_df = pd.DataFrame([{
        "policy": "observed",
        "estimator": "mean",
        "value": policy_results["observed_value"],
    }, {
        "policy": "rule_v1",
        "estimator": "WIS",
        "value": policy_results["wis_result"]["value"],
        "ess": policy_results["wis_result"]["ess"],
    }, {
        "policy": "rule_v1",
        "estimator": "DR",
        "value": policy_results["dr_result"]["value"],
    }])
    pv_path = tables_dir / "75a_policy_values.csv"
    pv_df.to_csv(pv_path, index=False)
    print(f"  Saved: {pv_path}")

    # ── Subgroup analysis ─────────────────────────────────────────────────
    subgroup_df = subgroup_analysis(
        panel, data["prob_cols"], data["merge_map"], trim_lo, trim_hi,
    )
    sub_path = tables_dir / "75a_policy_values_by_stratum.csv"
    subgroup_df.to_csv(sub_path, index=False)
    print(f"  Saved: {sub_path}")

    # ── Trimming sensitivity ──────────────────────────────────────────────
    trim_sens_df = trimming_sensitivity(
        panel, data["prob_cols"], data["merge_map"], data["trim_cfg"],
    )
    trim_path = tables_dir / "75a_trimming_sensitivity.csv"
    trim_sens_df.to_csv(trim_path, index=False)
    print(f"  Saved: {trim_path}")

    # ── Bootstrap CIs ─────────────────────────────────────────────────────
    ci = bootstrap_ci(
        panel, data["prob_cols"], data["merge_map"], trim_lo, trim_hi,
    )
    ci_df = pd.DataFrame([
        {"estimand": k, **v} for k, v in ci.items()
    ])
    ci_path = tables_dir / "75a_bootstrap_ci.csv"
    ci_df.to_csv(ci_path, index=False)
    print(f"  Saved: {ci_path}")

    # ── Figures ───────────────────────────────────────────────────────────
    print("\n[10] Generating figures...")
    plot_weight_distribution(panel, figures_dir)
    plot_policy_value_forest(policy_results, subgroup_df, ci, figures_dir)

    # ── Summary report ────────────────────────────────────────────────────
    print("\n[11] Writing summary report...")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "75a_dynamic_msm_iptw_summary.txt"

    lines = [
        "=" * 72,
        "75a — Dynamic MSM / IPTW (MIMIC-IV)",
        "=" * 72,
        f"Generated: {ts}",
        f"Protocol: v3.1, Step 75a (primary classical DTR baseline)",
        "",
        "DATA",
        "────",
        f"  Stays: {panel['stay_id'].nunique():,}",
        f"  Observations: {len(panel):,}",
        f"  Post-absorbing excluded: {panel['post_absorbing'].sum():,}",
        f"  Mortality (28d): {policy_results['observed_value']:.4f}",
        "",
        "STABILISED IPTW",
        "────────────────",
    ]
    terminal = panel[panel["is_terminal"]]
    sw = terminal["sw_raw"]
    lines.extend([
        f"  Numerator: P(A_t | A_bar_{{<t}}) — marginal logistic model",
        f"  Denominator: P(A_t | L_t, A_{{<t}}) — 70b behavior policy",
        f"  Raw weights: mean={sw.mean():.3f}, median={sw.median():.3f}, "
        f"P95={np.percentile(sw, 95):.3f}, P99={np.percentile(sw, 99):.3f}",
        f"  Primary trimming: [P{trim_lo}, P{trim_hi}]",
        "",
        "DYNAMIC MSM",
        "───────────",
    ])
    for name, val in msm_results["coefs"].items():
        or_val = np.exp(val)
        lines.append(f"  {name}: β={val:.4f}, OR={or_val:.4f}")

    lines.extend([
        "",
        "POLICY EVALUATION",
        "─────────────────",
        f"  Observed mortality: {policy_results['observed_value']:.4f}",
        f"  Rule-policy WIS:   {policy_results['wis_result']['value']:.4f} "
        f"(ESS={policy_results['wis_result']['ess']:.0f})",
        f"  Rule-policy DR:    {policy_results['dr_result']['value']:.4f}",
        f"  ε-softening: {EPSILON_SOFTEN}",
    ])

    # ATE
    ate_wis = policy_results["wis_result"]["value"] - policy_results["observed_value"]
    ate_dr = policy_results["dr_result"]["value"] - policy_results["observed_value"]
    lines.extend([
        "",
        "AVERAGE TREATMENT EFFECT (rule vs observed)",
        "────────────────────────────────────────────",
        f"  ATE (WIS):  {ate_wis:+.4f}",
        f"  ATE (DR):   {ate_dr:+.4f}",
    ])

    # Bootstrap CIs
    if ci:
        lines.extend(["", "BOOTSTRAP CIs (B=500, 95%)", "──────────────────────────"])
        for k, v in ci.items():
            lines.append(f"  {k}: {v['mean']:.4f} [{v['ci_lo']:.4f}, {v['ci_hi']:.4f}]")

    # Subgroup
    if not subgroup_df.empty:
        lines.extend(["", "SUBGROUP ANALYSIS", "─────────────────"])
        for _, row in subgroup_df.iterrows():
            lines.append(
                f"  {row['stratifier']}={row['group']}: "
                f"n={row['n_stays']}, obs={row['observed_mortality']:.4f}, "
                f"WIS={row['wis_rule_value']:.4f}, "
                f"ATE={row['ate_wis']:+.4f}"
            )

    # Trimming sensitivity
    if not trim_sens_df.empty:
        lines.extend(["", "TRIMMING SENSITIVITY", "────────────────────"])
        for _, row in trim_sens_df.iterrows():
            lines.append(
                f"  [{row['trim_lo']:5.1f}, {row['trim_hi']:5.1f}]: "
                f"WIS={row['wis_rule_value']:.4f}, ESS={row['wis_ess']:.0f}, "
                f"ATE={row['ate_wis']:+.4f}"
            )

    lines.extend([
        "",
        "INTERPRETATION NOTES",
        "────────────────────",
        "  - Negative ATE → rule policy associated with LOWER mortality",
        "  - Positive ATE → rule policy associated with HIGHER mortality",
        "  - ESS << N indicates high variance / limited policy support",
        "  - Results are sensitive to positivity assumption; check trimming stability",
        "  - This is a classical IPTW baseline; compare with G-Transformer (80c)",
        "",
        "NEXT STEPS",
        "──────────",
        "  75c: classical baseline summary",
        "  80c: train G-Transformer (primary deep pair)",
    ])

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {report_path}")

    print("\n" + "=" * 72)
    print("75a COMPLETE.")
    print("=" * 72)


if __name__ == "__main__":
    main()
