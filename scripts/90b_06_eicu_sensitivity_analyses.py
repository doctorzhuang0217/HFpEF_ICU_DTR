#!/usr/bin/env python3
"""90b_06 — eICU External Validation Sensitivity Analyses

Protocol: v3.1, Step 90b (sub-step 6)
Compute: G1 (GPU for model inference) + C3 (bootstrap)

Purpose
-------
Four sensitivity analyses to improve eICU–MIMIC alignment interpretation:

  S1. Support-restricted OPE (P1)
      Mask stay×bin pairs where rule-action propensity < threshold,
      re-evaluate DM/DR on supported subset only.
  S2. Endpoint dual-definition (P2)
      Re-evaluate with hospital_mortality and icu_mortality alongside
      28-day mortality.
  S3. Strict HFpEF sub-cohort (P2)
      Restrict to diastolic-ICD-coded HFpEF (hfpef_diastolic_strict_flag=1)
      and re-evaluate DM.
  S4. Action support diagnostics (P2)
      Quantify per-action coverage in eICU vs MIMIC, flag low-support cells,
      report rule-policy overlap statistics.

All analyses reuse frozen MIMIC checkpoints and eICU interim data from
90b_01–05. No locked decisions are changed.

Inputs
------
- data/interim/90b_eicu_state_tensors_v1.parquet
- data/interim/90b_eicu_outcomes_absorb.parquet
- data/interim/90b_eicu_propensity_coarse3.parquet
- data/interim/90b_eicu_actions_2h.parquet
- data/staging/90b_eicu_cohort_v1.parquet
- models/80c_gtransformer_rollout_ow1_best.pt (or fallback)
- configs/80_step_c_training_v1.yaml
- configs/90a_primary_model_registry.json
- configs/70b_action_merge_map.yaml

Outputs
-------
- tables/90b_06_support_restricted_results.csv
- tables/90b_06_endpoint_sensitivity_results.csv
- tables/90b_06_strict_hfpef_results.csv
- tables/90b_06_action_support_diagnostics.csv
- reports/90b_06_eicu_sensitivity_report.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(ROOT / "src"))

from hfpef_dtr.models.dtr_dataset import DTRDataset
from hfpef_dtr.models.gtransformer import build_gtransformer_from_config
from hfpef_dtr.models.ope import evaluate_dm, evaluate_dr

MODELS_DIR = ROOT / "models"
TABLES_DIR = ROOT / "tables"
REPORTS_DIR = ROOT / "reports"
INTERIM_DIR = ROOT / "data" / "interim"
STAGING_DIR = ROOT / "data" / "staging"

for d in [TABLES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="90b_06: eICU sensitivity analyses")
    p.add_argument("--tag", type=str, default="rollout_ow1",
                   help="Checkpoint tag for 80c model")
    p.add_argument("--support-thresholds", nargs="+", type=float,
                   default=[0.01, 0.02, 0.05],
                   help="Rule-action propensity thresholds for S1")
    p.add_argument("--skip-s1", action="store_true", help="Skip support-restricted OPE")
    p.add_argument("--skip-s2", action="store_true", help="Skip endpoint sensitivity")
    p.add_argument("--skip-s3", action="store_true", help="Skip strict HFpEF")
    p.add_argument("--skip-s4", action="store_true", help="Skip action diagnostics")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════

def load_mimic_spec() -> dict:
    """Load frozen MIMIC feature specification."""
    path = ROOT / "configs" / "90a_primary_model_registry.json"
    with open(path) as f:
        reg = json.load(f)
    return {
        "temporal_cols": reg["features"]["temporal_cols"],
        "static_cols": reg["features"]["static_cols"],
        "n_temporal": reg["features"]["n_temporal"],
        "n_static": reg["features"]["n_static"],
        "n_actions": reg["training_config"]["action"]["n_merged_cells"],
    }


def align_features(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Ensure MIMIC feature columns exist, fill missing with 0."""
    all_cols = spec["temporal_cols"] + spec["static_cols"]
    for c in all_cols:
        if c not in df.columns:
            df[c] = 0.0
    return df


def load_merge_map() -> dict | None:
    path = ROOT / "configs" / "70b_action_merge_map.yaml"
    if path.exists():
        with open(path) as f:
            mm = yaml.safe_load(f)
        if "merge_map" in mm:
            return {int(k): int(v) for k, v in mm["merge_map"].items()}
    return None


def make_dataset(
    state_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    propensity_df: pd.DataFrame,
    stay_ids: np.ndarray,
    spec: dict,
    config: dict,
) -> DTRDataset:
    """Build a DTRDataset for a subset of stays."""
    return DTRDataset(
        state_df=state_df,
        outcomes_df=outcomes_df,
        propensity_df=propensity_df,
        stay_ids=stay_ids,
        temporal_cols=spec["temporal_cols"],
        static_cols=spec["static_cols"],
        merge_map=load_merge_map(),
        n_bins=config["grid"]["n_bins"],
        action_vocab=config["action"]["valid_merged_indices"],
    )


def load_model(config: dict, spec: dict, ds: DTRDataset, device, tag: str):
    """Load frozen 80c G-Transformer."""
    model_id = "80c_gtransformer"
    model_cfg = config["models"].get(model_id, {})
    grid_cfg = config["grid"]

    base = f"{model_id}_{tag}_best.pt" if tag else f"{model_id}_best.pt"
    ckpt_path = MODELS_DIR / base
    if not ckpt_path.exists() and tag:
        ckpt_path = MODELS_DIR / f"{model_id}_best.pt"

    model = build_gtransformer_from_config(
        ds.d_temporal, ds.d_static, spec["n_actions"], model_cfg, grid_cfg
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"  Loaded {model_id}: {sum(p.numel() for p in model.parameters()):,} params")
    return model


# ══════════════════════════════════════════════════════════════════════
# S1: Support-restricted OPE
# ══════════════════════════════════════════════════════════════════════

def run_support_restricted_ope(
    model, full_ds, propensity_df, state_df, outcomes_df,
    spec, config, device, thresholds: list[float],
) -> pd.DataFrame:
    """
    For each threshold τ, identify stays where min per-bin rule-action
    propensity ≥ τ across all valid bins. Re-evaluate DM on that subset.
    """
    print("\n" + "=" * 60)
    print("S1: Support-Restricted OPE")
    print("=" * 60)

    merge_map = load_merge_map()
    n_bins = config["grid"]["n_bins"]
    rows = []

    # First, compute per-stay minimum rule-action propensity
    # propensity_df has: stay_id, bin, rule_action_merged, prob_a0..prob_a17
    print("  Computing per-stay minimum rule-action propensity...")
    prop = propensity_df.copy()

    # Vectorized: look up prob_a{rule_action} for each row
    prob_cols = [f"prob_a{i}" for i in range(18)]
    avail_prob_cols = [c for c in prob_cols if c in prop.columns]
    prob_matrix = prop[avail_prob_cols].values  # (N, n_avail)
    rule_actions = prop["rule_action_merged"].values.astype(int)
    # Map action index to column position
    col_map = {int(c.split("_a")[1]): i for i, c in enumerate(avail_prob_cols)}
    rule_col_idx = np.array([col_map.get(ra, -1) for ra in rule_actions])
    valid_mask_prop = rule_col_idx >= 0
    rule_probs_arr = np.zeros(len(prop), dtype=np.float64)
    rule_probs_arr[valid_mask_prop] = prob_matrix[
        np.where(valid_mask_prop)[0], rule_col_idx[valid_mask_prop]
    ]
    prop["rule_propensity"] = rule_probs_arr

    # Vectorized: per-stay min rule propensity
    stay_min_prop = prop.groupby("stay_id")["rule_propensity"].min().reset_index()
    stay_min_prop.columns = ["stay_id", "min_rule_propensity"]

    # Also compute mean for diagnostics
    stay_mean_prop = prop.groupby("stay_id")["rule_propensity"].mean().reset_index()
    stay_mean_prop.columns = ["stay_id", "mean_rule_propensity"]
    stay_support = stay_min_prop.merge(stay_mean_prop, on="stay_id")

    all_stays = full_ds.stay_ids
    print(f"  Total stays: {len(all_stays):,}")
    print(f"  Min rule-propensity distribution: "
          f"p50={stay_support['min_rule_propensity'].median():.4f}, "
          f"p10={stay_support['min_rule_propensity'].quantile(0.1):.4f}, "
          f"p90={stay_support['min_rule_propensity'].quantile(0.9):.4f}")

    # Full-sample baseline
    print("\n  Baseline (full sample)...")
    dm_full = evaluate_dm(model, full_ds, device, use_rollout=True)
    rows.append({
        "threshold": 0.0, "label": "full_sample",
        "n_stays": len(full_ds), "pct_retained": 100.0,
        "v_observed": dm_full["v_observed"],
        "v_rule": dm_full["v_rule"],
        "ate": dm_full["ate"],
    })
    print(f"    Full: n={len(full_ds)}, ATE={dm_full['ate']:+.4f}")

    # Threshold sweeps
    for tau in thresholds:
        supported = stay_support[stay_support["min_rule_propensity"] >= tau]
        supported_ids = supported["stay_id"].values
        n_supported = len(supported_ids)
        pct = n_supported / len(all_stays) * 100

        print(f"\n  τ={tau:.3f}: {n_supported:,} stays ({pct:.1f}% retained)")

        if n_supported < 30:
            print(f"    SKIP: too few stays (<30)")
            rows.append({
                "threshold": tau, "label": f"tau_{tau}",
                "n_stays": n_supported, "pct_retained": pct,
                "v_observed": np.nan, "v_rule": np.nan, "ate": np.nan,
            })
            continue

        sub_ds = make_dataset(
            state_df, outcomes_df, propensity_df,
            supported_ids, spec, config,
        )

        dm = evaluate_dm(model, sub_ds, device, use_rollout=True)
        rows.append({
            "threshold": tau, "label": f"tau_{tau}",
            "n_stays": n_supported, "pct_retained": pct,
            "v_observed": dm["v_observed"],
            "v_rule": dm["v_rule"],
            "ate": dm["ate"],
        })
        print(f"    ATE={dm['ate']:+.4f} (V_obs={dm['v_observed']:.4f}, "
              f"V_rule={dm['v_rule']:.4f})")

    df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "90b_06_support_restricted_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    return df


# ══════════════════════════════════════════════════════════════════════
# S2: Endpoint dual-definition
# ══════════════════════════════════════════════════════════════════════

def run_endpoint_sensitivity(
    model, state_df, propensity_df, cohort,
    spec, config, device,
) -> pd.DataFrame:
    """
    Re-evaluate DM with hospital_mortality and icu_mortality as outcomes,
    alongside the default mortality_28d.
    """
    print("\n" + "=" * 60)
    print("S2: Endpoint Dual-Definition Sensitivity")
    print("=" * 60)

    merge_map = load_merge_map()
    n_bins = config["grid"]["n_bins"]
    rows = []

    # Base outcomes
    base_outcomes = pd.read_parquet(INTERIM_DIR / "90b_eicu_outcomes_absorb.parquet")
    all_stay_ids = base_outcomes["stay_id"].unique()

    # Map from cohort: we need hospital_mortality, icu_mortality
    cohort_map = cohort.set_index("patientunitstayid")

    endpoint_defs = {
        "mortality_28d": "mortality_28d",
        "hospital_mortality": "hospital_mortality",
        "icu_mortality": "icu_mortality",
    }

    for ep_label, ep_col in endpoint_defs.items():
        print(f"\n  Endpoint: {ep_label}")

        if ep_col not in cohort.columns:
            print(f"    WARNING: {ep_col} not in cohort, skipping")
            continue

        # Build modified outcomes dataframe
        alt_outcomes = base_outcomes.copy()
        # Map new outcome from cohort
        ep_map = cohort.set_index("patientunitstayid")[ep_col].to_dict()
        alt_outcomes["mortality_28d"] = alt_outcomes["stay_id"].map(ep_map).fillna(0).astype(int)

        prevalence = alt_outcomes["mortality_28d"].mean()
        print(f"    Prevalence: {prevalence*100:.1f}%")

        ds = make_dataset(state_df, alt_outcomes, propensity_df,
                          all_stay_ids, spec, config)

        dm = evaluate_dm(model, ds, device, use_rollout=True)
        rows.append({
            "endpoint": ep_label,
            "prevalence": prevalence,
            "n_stays": len(ds),
            "v_observed": dm["v_observed"],
            "v_rule": dm["v_rule"],
            "ate": dm["ate"],
            "ate_sign": "negative" if dm["ate"] < 0 else "positive",
        })
        print(f"    ATE={dm['ate']:+.4f} (V_obs={dm['v_observed']:.4f}, "
              f"V_rule={dm['v_rule']:.4f})")

    df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "90b_06_endpoint_sensitivity_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    return df


# ══════════════════════════════════════════════════════════════════════
# S3: Strict HFpEF sub-cohort
# ══════════════════════════════════════════════════════════════════════

def run_strict_hfpef_sensitivity(
    model, state_df, outcomes_df, propensity_df, cohort,
    spec, config, device,
) -> pd.DataFrame:
    """
    Restrict to diastolic-ICD-coded HFpEF (hfpef_diastolic_strict_flag=1)
    and re-evaluate DM. Compare with broad HFpEF and no_HF.
    """
    print("\n" + "=" * 60)
    print("S3: Strict HFpEF Sub-Cohort Sensitivity")
    print("=" * 60)

    rows = []

    subcohorts = {
        "all_eicu": cohort["patientunitstayid"].values,
        "hfpef_broad": cohort.loc[
            cohort["hf_subtype"] == "hfpef", "patientunitstayid"
        ].values,
        "hfpef_strict": cohort.loc[
            cohort["hfpef_diastolic_strict_flag"] == 1, "patientunitstayid"
        ].values,
        "hfpef_broad_only": cohort.loc[
            (cohort["hf_subtype"] == "hfpef") &
            (cohort["hfpef_diastolic_strict_flag"] == 0),
            "patientunitstayid"
        ].values,
        "hfref": cohort.loc[
            cohort["hf_subtype"] == "hfref", "patientunitstayid"
        ].values,
        "no_hf": cohort.loc[
            cohort["hf_subtype"] == "no_hf", "patientunitstayid"
        ].values,
    }

    for label, stay_ids in subcohorts.items():
        n = len(stay_ids)
        print(f"\n  {label}: n={n:,}")

        if n < 30:
            print(f"    SKIP: too few stays (<30)")
            rows.append({
                "subcohort": label, "n_stays": n,
                "mortality_rate": np.nan,
                "v_observed": np.nan, "v_rule": np.nan, "ate": np.nan,
            })
            continue

        # Mortality rate for this subcohort
        mort_rate = outcomes_df[
            outcomes_df["stay_id"].isin(stay_ids)
        ]["mortality_28d"].mean()

        ds = make_dataset(state_df, outcomes_df, propensity_df,
                          stay_ids, spec, config)

        dm = evaluate_dm(model, ds, device, use_rollout=True)
        rows.append({
            "subcohort": label, "n_stays": n,
            "mortality_rate": mort_rate,
            "v_observed": dm["v_observed"],
            "v_rule": dm["v_rule"],
            "ate": dm["ate"],
        })
        print(f"    mort={mort_rate*100:.1f}%, ATE={dm['ate']:+.4f}")

    df = pd.DataFrame(rows)

    # Compute strict-vs-broad comparison
    strict_row = df[df["subcohort"] == "hfpef_strict"]
    broad_row = df[df["subcohort"] == "hfpef_broad"]
    if len(strict_row) > 0 and len(broad_row) > 0:
        s_ate = strict_row.iloc[0]["ate"]
        b_ate = broad_row.iloc[0]["ate"]
        if not (np.isnan(s_ate) or np.isnan(b_ate)):
            same_sign = np.sign(s_ate) == np.sign(b_ate)
            print(f"\n  Strict vs Broad HFpEF: same_sign={same_sign}, "
                  f"strict ATE={s_ate:+.4f}, broad ATE={b_ate:+.4f}")

    out_path = TABLES_DIR / "90b_06_strict_hfpef_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    return df


# ══════════════════════════════════════════════════════════════════════
# S4: Action support diagnostics
# ══════════════════════════════════════════════════════════════════════

def run_action_support_diagnostics(
    propensity_df: pd.DataFrame,
    actions_df: pd.DataFrame,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """
    Quantify per-action coverage: observed frequency, rule frequency,
    behavior-policy mean propensity, and low-support flags.
    """
    print("\n" + "=" * 60)
    print("S4: Action Support Diagnostics")
    print("=" * 60)

    merge_map = load_merge_map() or {}

    # All possible merged actions
    all_actions = sorted(actions_df["action_idx_merged"].unique())
    n_total = len(actions_df)

    rows = []
    for a_idx in all_actions:
        # Observed frequency
        obs_count = (actions_df["action_idx_merged"] == a_idx).sum()
        obs_pct = obs_count / n_total * 100

        # Rule frequency
        rule_count = (actions_df["rule_action_merged"] == a_idx).sum()
        rule_pct = rule_count / n_total * 100

        # Mean behavior-policy probability for this action
        prob_col = f"prob_a{a_idx}"
        if prob_col in propensity_df.columns:
            mean_prop = propensity_df[prob_col].mean()
            median_prop = propensity_df[prob_col].median()
        else:
            mean_prop = np.nan
            median_prop = np.nan

        # Decode action
        f_level = a_idx // 6
        v_level = (a_idx % 6) // 2
        d_level = a_idx % 2

        # Support flags
        low_obs = obs_pct < 1.0
        low_rule = rule_pct < 1.0
        low_prop = (mean_prop < 0.01) if not np.isnan(mean_prop) else True

        rows.append({
            "action_idx": a_idx,
            "fluid": f_level, "vaso": v_level, "diuretic": d_level,
            "obs_count": obs_count, "obs_pct": obs_pct,
            "rule_count": rule_count, "rule_pct": rule_pct,
            "mean_propensity": mean_prop,
            "median_propensity": median_prop,
            "low_obs_flag": low_obs,
            "low_rule_flag": low_rule,
            "low_propensity_flag": low_prop,
        })

    df = pd.DataFrame(rows)

    print("\n  Action support table:")
    print(f"  {'Action':>8s} {'f,v,d':>6s} {'Obs%':>7s} {'Rule%':>7s} "
          f"{'MeanProp':>9s} {'Flags':>12s}")
    print("  " + "-" * 55)
    for _, row in df.iterrows():
        flags = []
        if row["low_obs_flag"]:
            flags.append("LOW_OBS")
        if row["low_propensity_flag"]:
            flags.append("LOW_PROP")
        flag_str = ",".join(flags) if flags else "OK"
        print(f"  a{row['action_idx']:>6d} ({row['fluid']},{row['vaso']},{row['diuretic']}) "
              f"{row['obs_pct']:>6.1f}% {row['rule_pct']:>6.1f}% "
              f"{row['mean_propensity']:>8.4f}  {flag_str}")

    # Rule-observed overlap analysis
    n_agree = (actions_df["action_idx_merged"] == actions_df["rule_action_merged"]).sum()
    agreement = n_agree / n_total * 100
    print(f"\n  Rule-observed agreement: {agreement:.1f}%")

    # Rule actions that fall on zero-support observed cells
    rule_actions = set(actions_df["rule_action_merged"].unique())
    obs_actions = set(actions_df["action_idx_merged"].unique())
    unsupported_rule = rule_actions - obs_actions
    if unsupported_rule:
        print(f"  WARNING: Rule actions with ZERO observed support: {unsupported_rule}")
    else:
        print(f"  All rule actions have observed support: PASS")

    # Per-HF-subtype action support
    print("\n  Action support by HF subtype:")
    cohort_ids = cohort.set_index("patientunitstayid")
    for subtype in ["hfpef", "hfref", "no_hf"]:
        sub_ids = cohort.loc[cohort["hf_subtype"] == subtype, "patientunitstayid"].values
        sub_actions = actions_df[actions_df["stay_id"].isin(sub_ids)]
        if len(sub_actions) == 0:
            continue
        n_cells = sub_actions["action_idx_merged"].nunique()
        rule_agree = (sub_actions["action_idx_merged"] == sub_actions["rule_action_merged"]).mean() * 100
        print(f"    {subtype}: n_bins={len(sub_actions):,}, "
              f"active_cells={n_cells}, rule_agree={rule_agree:.1f}%")

    out_path = TABLES_DIR / "90b_06_action_support_diagnostics.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    return df


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    print("=" * 72)
    print("90b_06 — eICU External Validation Sensitivity Analyses")
    print("=" * 72)
    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Setup ──
    print("\n[0] Setup...")
    config_path = ROOT / "configs" / "80_step_c_training_v1.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}" +
          (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

    spec = load_mimic_spec()
    print(f"  MIMIC spec: {spec['n_temporal']}T + {spec['n_static']}S, "
          f"{spec['n_actions']} actions")

    # ── Load data ──
    print("\n[1] Loading data...")
    state_df = pd.read_parquet(INTERIM_DIR / "90b_eicu_state_tensors_v1.parquet")
    state_df = align_features(state_df, spec)
    outcomes_df = pd.read_parquet(INTERIM_DIR / "90b_eicu_outcomes_absorb.parquet")
    propensity_df = pd.read_parquet(INTERIM_DIR / "90b_eicu_propensity_coarse3.parquet")
    actions_df = pd.read_parquet(INTERIM_DIR / "90b_eicu_actions_2h.parquet")
    cohort = pd.read_parquet(STAGING_DIR / "90b_eicu_cohort_v1.parquet")

    all_stay_ids = outcomes_df["stay_id"].unique()
    print(f"  Stays: {len(all_stay_ids):,}")

    # ── Build full dataset & load model (shared across S1–S3) ──
    model = None
    full_ds = None

    need_model = not (args.skip_s1 and args.skip_s2 and args.skip_s3)
    if need_model:
        print("\n[2] Loading model & building dataset...")
        full_ds = make_dataset(state_df, outcomes_df, propensity_df,
                               all_stay_ids, spec, config)
        model = load_model(config, spec, full_ds, device, args.tag)

    # ── S1: Support-restricted OPE ──
    s1_df = None
    if not args.skip_s1:
        s1_df = run_support_restricted_ope(
            model, full_ds, propensity_df, state_df, outcomes_df,
            spec, config, device, args.support_thresholds,
        )

    # ── S2: Endpoint dual-definition ──
    s2_df = None
    if not args.skip_s2:
        s2_df = run_endpoint_sensitivity(
            model, state_df, propensity_df, cohort,
            spec, config, device,
        )

    # ── S3: Strict HFpEF sub-cohort ──
    s3_df = None
    if not args.skip_s3:
        s3_df = run_strict_hfpef_sensitivity(
            model, state_df, outcomes_df, propensity_df, cohort,
            spec, config, device,
        )

    # ── S4: Action support diagnostics ──
    s4_df = None
    if not args.skip_s4:
        s4_df = run_action_support_diagnostics(
            propensity_df, actions_df, cohort,
        )

    # ── Report ──
    print("\n[R] Generating report...")
    report_lines = [
        "=" * 72,
        "90b_06 — eICU External Validation Sensitivity Analyses",
        "=" * 72,
        f"Generated: {ts}",
        f"Protocol: v3.1, Step 90b (sensitivity)",
        "",
    ]

    # S1 summary
    if s1_df is not None:
        report_lines.append("S1: SUPPORT-RESTRICTED OPE")
        report_lines.append("─" * 40)
        full_ate = s1_df.loc[s1_df["label"] == "full_sample", "ate"].values
        full_ate_str = f"{full_ate[0]:+.4f}" if len(full_ate) > 0 else "N/A"
        report_lines.append(f"  Full-sample ATE: {full_ate_str}")
        for _, row in s1_df.iterrows():
            if row["label"] == "full_sample":
                continue
            ate_str = f"{row['ate']:+.4f}" if not np.isnan(row["ate"]) else "N/A"
            report_lines.append(
                f"  τ={row['threshold']:.3f}: n={row['n_stays']:,} "
                f"({row['pct_retained']:.1f}%), ATE={ate_str}"
            )
        # Sign stability assessment
        valid_s1 = s1_df.dropna(subset=["ate"])
        if len(valid_s1) > 1:
            signs = np.sign(valid_s1["ate"].values)
            sign_stable = (signs == signs[0]).all()
            report_lines.append(
                f"  Sign stability across thresholds: "
                f"{'STABLE' if sign_stable else 'UNSTABLE'}"
            )
        report_lines.append("")

    # S2 summary
    if s2_df is not None:
        report_lines.append("S2: ENDPOINT DUAL-DEFINITION")
        report_lines.append("─" * 40)
        for _, row in s2_df.iterrows():
            report_lines.append(
                f"  {row['endpoint']}: prev={row['prevalence']*100:.1f}%, "
                f"ATE={row['ate']:+.4f} ({row['ate_sign']})"
            )
        # Direction consistency
        if len(s2_df) > 1:
            same_dir = s2_df["ate_sign"].nunique() == 1
            report_lines.append(
                f"  Direction consistency: {'CONSISTENT' if same_dir else 'MIXED'}"
            )
        report_lines.append("")

    # S3 summary
    if s3_df is not None:
        report_lines.append("S3: STRICT HFpEF SUB-COHORT")
        report_lines.append("─" * 40)
        for _, row in s3_df.iterrows():
            ate_str = f"{row['ate']:+.4f}" if not np.isnan(row["ate"]) else "N/A"
            mort_str = f"{row['mortality_rate']*100:.1f}%" if not np.isnan(row["mortality_rate"]) else "N/A"
            report_lines.append(
                f"  {row['subcohort']}: n={row['n_stays']:,}, "
                f"mort={mort_str}, ATE={ate_str}"
            )
        report_lines.append("")

    # S4 summary
    if s4_df is not None:
        report_lines.append("S4: ACTION SUPPORT DIAGNOSTICS")
        report_lines.append("─" * 40)
        n_low_obs = s4_df["low_obs_flag"].sum()
        n_low_prop = s4_df["low_propensity_flag"].sum()
        n_total_actions = len(s4_df)
        report_lines.append(f"  Total merged actions: {n_total_actions}")
        report_lines.append(f"  Low-observation (<1%): {n_low_obs}")
        report_lines.append(f"  Low-propensity (<0.01): {n_low_prop}")
        report_lines.append("")

    # Overall interpretation
    report_lines.append("INTERPRETATION")
    report_lines.append("─" * 40)
    report_lines.append("  These sensitivity analyses test whether the eICU ATE")
    report_lines.append("  sign change vs MIMIC is due to:")
    report_lines.append("    a) Support violation → check S1 sign stability")
    report_lines.append("    b) Endpoint mismatch → check S2 direction consistency")
    report_lines.append("    c) Phenotype heterogeneity → check S3 strict vs broad")
    report_lines.append("    d) Action space mismatch → check S4 coverage")
    report_lines.append("")
    report_lines.append("  If all sensitivities show same sign as full eICU, the")
    report_lines.append("  finding is a true transportability limitation, not an")
    report_lines.append("  alignment artifact.")

    report_path = REPORTS_DIR / "90b_06_eicu_sensitivity_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"  Report: {report_path}")

    total = time.time() - t0
    print()
    print("=" * 72)
    print(f"90b_06 COMPLETE. Time: {total/60:.1f} min")
    print("=" * 72)


if __name__ == "__main__":
    main()
