#!/usr/bin/env python3
"""90b_04 — Build eICU Actions, Outcomes, Rule Policy & Behavior Policy

Protocol: v3.1, Step 90b (sub-step 4)
Compute: C2

Purpose
-------
Build eICU action encoding, outcomes, rule policy assignments, and
behavior policy (propensity model). Mirrors MIMIC 45a/45b/45d/70b/70c.

Key design decisions for external validation:
  1. Action encoding: SAME as MIMIC (fluid*6 + vaso*2 + diuretic)
  2. Action merge map: SAME as MIMIC (from 70b_action_merge_map.yaml)
  3. Rule policy: SAME thresholds as MIMIC (MAP<65, lactate>2, etc.)
     Applied to RAW (un-normalized) eICU values
  4. Behavior policy: NEW model fit on eICU data
     (captures eICU clinician treatment patterns)
  5. Outcomes: 28-day mortality from eICU discharge data
  6. Absorbing states: early discharge/death before end of horizon

Inputs
------
- data/interim/90b_eicu_state_tensors_v1.parquet
- data/interim/90b_eicu_grid2h_states.parquet
- data/staging/90b_eicu_cohort_v1.parquet
- configs/70b_action_merge_map.yaml (MIMIC — FROZEN)
- configs/70c_rule_policy_v1.yaml (MIMIC — FROZEN)

Outputs
-------
- data/interim/90b_eicu_actions_2h.parquet
- data/interim/90b_eicu_outcomes_absorb.parquet
- data/interim/90b_eicu_propensity_coarse3.parquet
- data/interim/90b_eicu_rule_actions.parquet
- models/90b_eicu_behavior_policy_coarse3.pkl
- reports/90b_04_eicu_actions_propensity_summary.txt
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(ROOT / "src"))

from hfpef_dtr.models.dataset import discover_feature_columns

INTERIM_DIR = ROOT / "data" / "interim"
STAGING_DIR = ROOT / "data" / "staging"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = ROOT / "tables"

for d in [INTERIM_DIR, MODELS_DIR, REPORTS_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Load FROZEN action thresholds from MIMIC config ──
# CRITICAL: Must match MIMIC exactly for valid external validation.
# Previous bug: hardcoded [250, 1000] for fluid (WRONG — MIMIC uses [1, 250]).
_ACTION_CFG_PATH = ROOT / "configs" / "action_space_v1.yaml"
with open(_ACTION_CFG_PATH) as _f:
    _action_cfg = yaml.safe_load(_f)

_fluid_bins = _action_cfg["action_dimensions"]["iv_fluid_ml_per_2h"]["bins"]
_vaso_bins = _action_cfg["action_dimensions"]["vasopressor_ne_eq_mcg_kg_min"]["bins"]
_diur_bins = _action_cfg["action_dimensions"]["diuretic_furo_eq_mg_per_2h"]["bins"]

# Convert "inf" strings to float
def _parse_bins(bins):
    return [float(b) if str(b).lower() != "inf" else np.inf for b in bins]

FLUID_BINS = _parse_bins(_fluid_bins)    # [0, 1, 250, inf]
VASO_BINS = _parse_bins(_vaso_bins)       # [0, 0.001, 0.1, inf]
DIUR_BINS = _parse_bins(_diur_bins)      # [0, 0.001, inf]
N_RAW_ACTIONS = 18

print(f"  Action thresholds (from {_ACTION_CFG_PATH.name}):")
print(f"    Fluid bins:    {FLUID_BINS}")
print(f"    Vaso bins:     {VASO_BINS}")
print(f"    Diuretic bins: {DIUR_BINS}")


def encode_actions(grid: pd.DataFrame) -> pd.DataFrame:
    """Encode treatment actions into joint action index (matching MIMIC)."""
    print("  Encoding actions...")

    # Fluid level — MIMIC frozen: [0, 1, 250, inf], right=False (left-closed)
    fluid_col = "iv_fluid_ml"
    if fluid_col not in grid.columns:
        grid[fluid_col] = 0.0
    grid["fluid_level"] = pd.cut(
        grid[fluid_col].fillna(0),
        bins=FLUID_BINS,
        labels=[0, 1, 2],
        right=False,        # CRITICAL: match MIMIC 45b (left-closed intervals)
        include_lowest=True,
    ).astype(int)

    # Vasopressor level — MIMIC frozen: [0, 0.001, 0.1, inf], right=False
    vaso_col = "vasopressor_ne_eq"
    if vaso_col not in grid.columns:
        grid[vaso_col] = 0.0
    grid["vaso_level"] = pd.cut(
        grid[vaso_col].fillna(0),
        bins=VASO_BINS,
        labels=[0, 1, 2],
        right=False,        # CRITICAL: match MIMIC 45b (left-closed intervals)
        include_lowest=True,
    ).astype(int)

    # Diuretic level — MIMIC frozen: [0, 0.001, inf], right=False
    diur_col = "diuretic_furo_eq_mg"
    if diur_col not in grid.columns:
        grid[diur_col] = 0.0
    grid["diuretic_level"] = pd.cut(
        grid[diur_col].fillna(0),
        bins=DIUR_BINS,
        labels=[0, 1],
        right=False,        # CRITICAL: match MIMIC 45b
        include_lowest=True,
    ).astype(int)

    # Joint action index
    grid["action_idx"] = (
        grid["fluid_level"] * 6 + grid["vaso_level"] * 2 + grid["diuretic_level"]
    )

    # Report distribution
    counts = grid["action_idx"].value_counts().sort_index()
    print("  Action distribution:")
    for idx, cnt in counts.items():
        pct = cnt / len(grid) * 100
        f = idx // 6
        v = (idx % 6) // 2
        d = idx % 2
        print(f"    a{idx:2d} (f={f},v={v},d={d}): {cnt:>8,} ({pct:5.1f}%)")

    return grid


def apply_merge_map(grid: pd.DataFrame) -> pd.DataFrame:
    """Apply MIMIC merge map to actions."""
    merge_path = ROOT / "configs" / "70b_action_merge_map.yaml"
    with open(merge_path) as f:
        merge_cfg = yaml.safe_load(f)

    merge_map = {int(k): int(v) for k, v in merge_cfg.get("merge_map", {}).items()}
    print(f"  Merge map (from MIMIC): {merge_map}")

    grid["action_idx_merged"] = grid["action_idx"].replace(merge_map)
    n_merged = grid["action_idx_merged"].nunique()
    print(f"  Actions after merge: {n_merged}")

    return grid


def assign_rule_policy(grid: pd.DataFrame) -> pd.DataFrame:
    """
    Assign rule policy actions using clinical values with LOCF + safe defaults.

    Same thresholds as MIMIC (70c_rule_policy_v1.yaml).

    CRITICAL FIX (2026-03-12):
    The MIMIC pipeline (Step 45c → 70c) applies LOCF + population median
    imputation BEFORE rule assignment, so rule conditions never encounter NaN.
    The eICU pipeline was applying rules on RAW unimputed values, causing
    NaN comparisons to return False (e.g., NaN > 2.0 → False, NaN ≤ 2.0 → False).

    With lactate 97.1% NaN in eICU, BOTH hypotension (requires lactate ≤ 2)
    and shock (requires lactate > 2) conditions failed → 97.9% defaulted to a6.

    Fix: apply LOCF + clinically safe defaults before rule evaluation,
    matching MIMIC 45c → 70c pipeline behavior. Safe defaults are chosen
    to be clinically unremarkable (i.e., they do NOT trigger any rule),
    matching the MIMIC 70c fallback logic.
    """
    print("  Assigning rule policy...")

    # Load rule policy thresholds
    rule_path = ROOT / "configs" / "70c_rule_policy_v1.yaml"
    with open(rule_path) as f:
        rule_cfg = yaml.safe_load(f)

    thresholds = rule_cfg["thresholds"]
    map_threshold = thresholds["map_hypotension"]          # 65
    lactate_threshold = thresholds["lactate_hypoperfusion"] # 2.0
    fb_threshold = thresholds["fluid_balance_overload_L"]   # 3.0
    fb_severe = thresholds["fluid_balance_severe_overload_L"]  # 5.0
    spo2_threshold = thresholds["spo2_hypoxemia"]           # 94

    # ── Impute raw values (matching MIMIC 45c → 70c pipeline) ──
    # Step 1: LOCF within each stay
    # Step 2: Fill remaining NaN with clinically safe defaults
    #         (same as MIMIC 70c fallback: values that do NOT trigger rules)
    SAFE_DEFAULTS = {
        "map": 80.0,       # Above hypotension threshold (65)
        "lactate": 1.5,    # Below hypoperfusion threshold (2.0)
        "spo2": 97.0,      # Above hypoxemia threshold (94)
    }

    rule_vars = ["map", "lactate", "spo2"]
    grid_sorted = grid.sort_values(["stay_id", "bin"])

    nan_report = {}
    for var in rule_vars:
        if var not in grid_sorted.columns:
            grid_sorted[var] = np.nan

        n_nan_before = grid_sorted[var].isna().sum()

        # LOCF within stay
        grid_sorted[f"_{var}_rule"] = grid_sorted.groupby("stay_id")[var].ffill()
        n_nan_after_locf = grid_sorted[f"_{var}_rule"].isna().sum()

        # Fill remaining NaN with safe default
        grid_sorted[f"_{var}_rule"] = grid_sorted[f"_{var}_rule"].fillna(SAFE_DEFAULTS[var])
        n_nan_after_fill = grid_sorted[f"_{var}_rule"].isna().sum()

        nan_report[var] = {
            "raw_nan_pct": n_nan_before / len(grid_sorted) * 100,
            "after_locf_nan_pct": n_nan_after_locf / len(grid_sorted) * 100,
        }
        print(f"    {var}: raw NaN={n_nan_before/len(grid_sorted)*100:.1f}% "
              f"→ after LOCF={n_nan_after_locf/len(grid_sorted)*100:.1f}% "
              f"→ filled with {SAFE_DEFAULTS[var]}")

    # Use imputed values for rule evaluation
    raw_map = grid_sorted[f"_map_rule"]
    raw_lactate = grid_sorted[f"_lactate_rule"]
    raw_spo2 = grid_sorted[f"_spo2_rule"]
    raw_fb = grid_sorted["cumulative_fluid_balance_L"] if "cumulative_fluid_balance_L" in grid_sorted.columns else pd.Series(0.0, index=grid_sorted.index)
    raw_fb = raw_fb.fillna(0.0)

    # Default: maintenance (fluid=low, vaso=none, diuretic=none) → action 6
    rule_action = pd.Series(6, index=grid_sorted.index, dtype=int)

    # Priority 5: severe volume overload → no fluid, diuretic
    mask_svo = raw_fb > fb_severe
    rule_action[mask_svo] = 1

    # Priority 4: volume overload + hypoxemia
    mask_vo_hyp = (raw_fb > fb_threshold) & (raw_spo2 < spo2_threshold)
    rule_action[mask_vo_hyp] = 1

    # Priority 3: hypoperfusion only (lactate > 2, MAP ≥ 65)
    mask_hp = (raw_lactate > lactate_threshold) & (raw_map >= map_threshold)
    rule_action[mask_hp] = 8

    # Priority 2: hypotension only (MAP < 65, lactate ≤ 2)
    mask_hypo = (raw_map < map_threshold) & (raw_lactate <= lactate_threshold)
    rule_action[mask_hypo] = 14

    # Priority 1: shock + hypoperfusion
    mask_shock = (raw_map < map_threshold) & (raw_lactate > lactate_threshold)
    rule_action[mask_shock] = 16

    # Report trigger rates for diagnostics
    n_total = len(grid_sorted)
    print(f"\n  Rule trigger diagnostics:")
    print(f"    MAP < {map_threshold}: {(raw_map < map_threshold).sum()/n_total*100:.1f}%")
    print(f"    Lactate > {lactate_threshold}: {(raw_lactate > lactate_threshold).sum()/n_total*100:.1f}%")
    print(f"    SpO2 < {spo2_threshold}: {(raw_spo2 < spo2_threshold).sum()/n_total*100:.1f}%")
    print(f"    FB > {fb_threshold}L: {(raw_fb > fb_threshold).sum()/n_total*100:.1f}%")
    print(f"    FB > {fb_severe}L: {(raw_fb > fb_severe).sum()/n_total*100:.1f}%")

    # Store rule actions in grid_sorted (aligned by index)
    grid_sorted["rule_action"] = rule_action

    # Clean up temp columns
    for var in rule_vars:
        grid_sorted.drop(columns=[f"_{var}_rule"], inplace=True)

    # Restore original index order
    grid = grid_sorted.sort_index()

    # Apply merge map to rule actions too
    merge_path = ROOT / "configs" / "70b_action_merge_map.yaml"
    with open(merge_path) as f:
        merge_cfg = yaml.safe_load(f)
    merge_map = {int(k): int(v) for k, v in merge_cfg.get("merge_map", {}).items()}

    grid["rule_action_merged"] = grid["rule_action"].replace(merge_map)

    # Report
    rule_counts = grid["rule_action_merged"].value_counts().sort_index()
    print("  Rule policy distribution:")
    for idx, cnt in rule_counts.items():
        pct = cnt / len(grid) * 100
        print(f"    rule_a{idx:2d}: {cnt:>8,} ({pct:5.1f}%)")

    # Rule-observed agreement
    agree = (grid["action_idx_merged"] == grid["rule_action_merged"]).mean()
    print(f"  Rule-observed agreement: {agree*100:.1f}%")

    return grid


def build_outcomes(cohort: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Build outcomes + absorbing state indicators."""
    print("  Building outcomes...")

    outcomes = cohort[["patientunitstayid"]].rename(
        columns={"patientunitstayid": "stay_id"}
    ).copy()

    # 28-day mortality
    outcomes = outcomes.merge(
        cohort[["patientunitstayid", "mortality_28d"]].rename(
            columns={"patientunitstayid": "stay_id"}
        ),
        on="stay_id", how="left"
    )
    outcomes["mortality_28d"] = outcomes["mortality_28d"].fillna(0).astype(int)

    # Absorbing state: early discharge or death
    # If patient is discharged from ICU before end of our 24h horizon
    # unitdischargeoffset (minutes from ICU admission)
    T0_MIN = 6 * 60  # T0 in minutes
    BIN_MIN = 2 * 60  # 2h bins in minutes
    HORIZON_END_MIN = T0_MIN + 12 * BIN_MIN

    absorb_df = cohort[["patientunitstayid", "unitdischargeoffset"]].copy()
    absorb_df = absorb_df.rename(columns={"patientunitstayid": "stay_id"})
    early_mask = absorb_df["unitdischargeoffset"] < HORIZON_END_MIN
    absorb_df["absorbing_state_bin"] = np.where(
        early_mask,
        ((absorb_df["unitdischargeoffset"] - T0_MIN) / BIN_MIN).astype(int).clip(0, 11),
        np.nan,
    )
    absorb_df = absorb_df[["stay_id", "absorbing_state_bin"]]
    outcomes = outcomes.merge(absorb_df, on="stay_id", how="left")

    # Absorbing type
    outcomes["absorbing_type"] = "none"
    has_absorb = outcomes["absorbing_state_bin"].notna()
    mort_mask = outcomes["mortality_28d"] == 1
    outcomes.loc[has_absorb & mort_mask, "absorbing_type"] = "death"
    outcomes.loc[has_absorb & ~mort_mask, "absorbing_type"] = "discharge"

    n_absorb = has_absorb.sum()
    n_total = len(outcomes)
    print(f"  Early absorbing: {n_absorb:,}/{n_total:,} ({n_absorb/n_total*100:.1f}%)")
    print(f"  28d mortality: {outcomes['mortality_28d'].mean()*100:.1f}%")

    return outcomes


def fit_eicu_behavior_policy(grid: pd.DataFrame, state_tensors: pd.DataFrame) -> tuple:
    """Fit behavior policy on eICU data (NEW model, same architecture as MIMIC)."""
    print("  Fitting eICU behavior policy...")

    # Discover feature columns from state tensors
    temporal_cols, static_cols = discover_feature_columns(state_tensors)

    # Build feature matrix (matching MIMIC 70b)
    feature_cols = temporal_cols.copy()

    # Add bin as feature
    grid["bin_normalized"] = grid["bin"] / 11.0
    feature_cols.append("bin_normalized")

    # Add numeric static cols
    string_cols = {"gender", "admission_type", "hf_subtype",
                   "comparator_group", "dominant_syndrome", "reason_for_icu"}
    for sc in static_cols:
        if sc in state_tensors.columns and sc not in string_cols:
            feature_cols.append(sc)

    # Filter to available columns
    merged = grid[["stay_id", "bin", "action_idx_merged"]].merge(
        state_tensors[["stay_id", "bin"] + [c for c in feature_cols if c in state_tensors.columns]],
        on=["stay_id", "bin"], how="left"
    )

    avail_features = [c for c in feature_cols if c in merged.columns]
    X = merged[avail_features].values.astype(np.float32)
    np.nan_to_num(X, copy=False, nan=0.0)
    y = merged["action_idx_merged"].values.astype(int)

    print(f"    Features: {len(avail_features)}, Observations: {len(X):,}")

    # Fit logistic regression (same as MIMIC 70b)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    clf = LogisticRegression(
        solver="lbfgs", max_iter=1000, C=1.0,
        random_state=2024, n_jobs=-1,
    )
    clf.fit(X_s, y)

    train_acc = (clf.predict(X_s) == y).mean()
    print(f"    Training accuracy: {train_acc:.4f}")

    # Predict probabilities
    probs = clf.predict_proba(X_s)
    known_classes = clf.classes_
    full_probs = np.zeros((len(y), N_RAW_ACTIONS), dtype=np.float32)
    for i, c in enumerate(known_classes):
        if c < N_RAW_ACTIONS:
            full_probs[:, c] = probs[:, i]

    # Build propensity table
    prop_df = grid[["stay_id", "bin", "action_idx", "action_idx_merged",
                     "rule_action_merged"]].copy()
    prop_df["propensity"] = full_probs[np.arange(len(y)), y]
    prop_df["propensity_clipped"] = prop_df["propensity"].clip(lower=0.005)

    for aidx in range(N_RAW_ACTIONS):
        prop_df[f"prob_a{aidx}"] = full_probs[:, aidx]

    # Save model
    model_path = MODELS_DIR / "90b_eicu_behavior_policy_coarse3.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": clf, "scaler": scaler,
            "feature_cols": avail_features,
            "database": "eICU",
        }, f)
    print(f"    Saved: {model_path}")

    return prop_df, train_acc, avail_features


def main():
    print("=" * 72)
    print("90b_04 — Build eICU Actions, Outcomes & Behavior Policy")
    print("=" * 72)
    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 1. Load data ─────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    grid = pd.read_parquet(INTERIM_DIR / "90b_eicu_grid2h_states.parquet")
    state_tensors = pd.read_parquet(INTERIM_DIR / "90b_eicu_state_tensors_v1.parquet")
    cohort = pd.read_parquet(STAGING_DIR / "90b_eicu_cohort_v1.parquet")
    print(f"  Grid: {len(grid):,} rows, {grid['stay_id'].nunique():,} stays")

    # ── 2. Encode actions ────────────────────────────────────────────────
    print("\n[2] Encoding actions...")
    grid = encode_actions(grid)
    grid = apply_merge_map(grid)

    # ── 3. Rule policy ───────────────────────────────────────────────────
    print("\n[3] Assigning rule policy...")
    grid = assign_rule_policy(grid)

    # ── 4. Outcomes + absorbing states ───────────────────────────────────
    print("\n[4] Building outcomes...")
    outcomes = build_outcomes(cohort, grid)

    # ── 5. Fit behavior policy ───────────────────────────────────────────
    print("\n[5] Fitting eICU behavior policy...")
    prop_df, bp_acc, bp_features = fit_eicu_behavior_policy(grid, state_tensors)

    # ── 5b. Site-level heterogeneity analysis (P0-2) ─────────────────────
    print("\n[5b] Site-level heterogeneity analysis...")
    site_report_lines = []
    if "hospitalid" in cohort.columns:
        # Map hospitalid to grid via stay_id ↔ patientunitstayid
        stay_to_hosp = cohort.set_index("patientunitstayid")["hospitalid"].to_dict()
        grid["hospitalid"] = grid["stay_id"].map(stay_to_hosp)

        n_hospitals = grid["hospitalid"].nunique()
        print(f"  Hospitals in cohort: {n_hospitals}")

        # Per-hospital action distribution
        hosp_stats = []
        for hid, hdf in grid.groupby("hospitalid"):
            n_stays_h = hdf["stay_id"].nunique()
            n_bins_h = len(hdf)
            a0_pct = (hdf["action_idx_merged"] == 0).mean() * 100
            fluid_nz = (hdf["iv_fluid_ml"].fillna(0) > 0).mean() * 100 if "iv_fluid_ml" in hdf.columns else 0
            vaso_nz = (hdf["vasopressor_ne_eq"].fillna(0) > 0).mean() * 100 if "vasopressor_ne_eq" in hdf.columns else 0
            diur_nz = (hdf["diuretic_furo_eq_mg"].fillna(0) > 0).mean() * 100 if "diuretic_furo_eq_mg" in hdf.columns else 0
            # has_infusiondrug: vasopressors come exclusively from infusionDrug,
            # so vaso_nz > 0 indicates the hospital has infusionDrug records.
            # Fluid infusions from infusionDrug also contribute to iv_fluid_ml
            # but cannot be separated from intakeOutput at the grid level.
            has_infdrug = vaso_nz > 0 or (fluid_nz > 0 and vaso_nz > 0)
            hosp_stats.append({
                "hospitalid": hid,
                "n_stays": n_stays_h,
                "n_bins": n_bins_h,
                "a0_pct": a0_pct,
                "fluid_nz_pct": fluid_nz,
                "vaso_nz_pct": vaso_nz,
                "diur_nz_pct": diur_nz,
                "has_infusiondrug": has_infdrug,
            })
        hosp_df = pd.DataFrame(hosp_stats).sort_values("a0_pct")

        # Save full per-hospital table
        hosp_path = TABLES_DIR / "90b_04_eicu_site_heterogeneity.csv"
        hosp_df.to_csv(hosp_path, index=False)
        print(f"  Saved: {hosp_path}")

        # Tier classification based on fluid documentation quality
        # Tier A: fluid_nz >= 15% (MIMIC-like documentation)
        # Tier B: fluid_nz 5-15% (moderate documentation)
        # Tier C: fluid_nz < 5% (poor documentation / no intake data)
        hosp_df["doc_tier"] = pd.cut(
            hosp_df["fluid_nz_pct"],
            bins=[-1, 5, 15, 100],
            labels=["C_poor", "B_moderate", "A_good"],
        )
        tier_summary = hosp_df.groupby("doc_tier", observed=True).agg(
            n_hospitals=("hospitalid", "nunique"),
            n_stays=("n_stays", "sum"),
            a0_pct_mean=("a0_pct", "mean"),
            fluid_nz_mean=("fluid_nz_pct", "mean"),
            vaso_nz_mean=("vaso_nz_pct", "mean"),
        ).reset_index()

        print("\n  Documentation quality tiers:")
        print(f"  {'Tier':12s} {'Hosp':>5s} {'Stays':>7s} {'a0%':>6s} {'Fluid%':>7s} {'Vaso%':>6s}")
        for _, r in tier_summary.iterrows():
            print(f"  {r['doc_tier']:12s} {r['n_hospitals']:5d} {r['n_stays']:7,.0f} "
                  f"{r['a0_pct_mean']:6.1f} {r['fluid_nz_mean']:7.1f} {r['vaso_nz_mean']:6.1f}")

        # Summary statistics
        a0_p10 = hosp_df["a0_pct"].quantile(0.10)
        a0_p50 = hosp_df["a0_pct"].quantile(0.50)
        a0_p90 = hosp_df["a0_pct"].quantile(0.90)
        n_zero_fluid = (hosp_df["fluid_nz_pct"] == 0).sum()

        print(f"\n  a0% distribution across hospitals: "
              f"P10={a0_p10:.1f}%, P50={a0_p50:.1f}%, P90={a0_p90:.1f}%")
        print(f"  Hospitals with ZERO fluid documentation: {n_zero_fluid}/{n_hospitals}")

        n_has_infdrug = hosp_df["has_infusiondrug"].sum()
        infdrug_stays = grid.loc[
            grid["hospitalid"].isin(hosp_df.loc[hosp_df["has_infusiondrug"], "hospitalid"]),
            "stay_id",
        ].nunique()
        print(f"  Hospitals with infusionDrug records: {n_has_infdrug}/{n_hospitals} "
              f"({infdrug_stays:,} stays)")

        # Identify "MIMIC-like" hospitals (Tier A) for sensitivity analysis
        tier_a_hospitals = set(hosp_df.loc[hosp_df["doc_tier"] == "A_good", "hospitalid"])
        tier_a_stays = grid.loc[grid["hospitalid"].isin(tier_a_hospitals), "stay_id"].nunique()
        tier_a_a0 = (grid.loc[grid["hospitalid"].isin(tier_a_hospitals), "action_idx_merged"] == 0).mean() * 100
        print(f"\n  Tier A ('MIMIC-like') subset: {len(tier_a_hospitals)} hospitals, "
              f"{tier_a_stays:,} stays, a0={tier_a_a0:.1f}%")

        site_report_lines = [
            "",
            "SITE-LEVEL HETEROGENEITY (P0-2)",
            "────────────────────────────────",
            f"  Total hospitals: {n_hospitals}",
            f"  a0% across hospitals: P10={a0_p10:.1f}%, P50={a0_p50:.1f}%, P90={a0_p90:.1f}%",
            f"  Hospitals with zero fluid: {n_zero_fluid}/{n_hospitals}",
            f"  Hospitals with infusionDrug: {n_has_infdrug}/{n_hospitals} ({infdrug_stays:,} stays)",
            f"  Tier A (fluid_nz>=15%): {len(tier_a_hospitals)} hosp, {tier_a_stays:,} stays, a0={tier_a_a0:.1f}%",
            f"  See: tables/90b_04_eicu_site_heterogeneity.csv",
        ]

        # Clean up temp column
        grid.drop(columns=["hospitalid"], inplace=True)
    else:
        print("  WARNING: hospitalid not in cohort, skipping site analysis")

    # ── 6. Save outputs ──────────────────────────────────────────────────
    print("\n[6] Saving outputs...")

    actions_path = INTERIM_DIR / "90b_eicu_actions_2h.parquet"
    grid[["stay_id", "bin", "action_idx", "action_idx_merged",
          "fluid_level", "vaso_level", "diuretic_level",
          "rule_action", "rule_action_merged"]].to_parquet(actions_path, index=False)
    print(f"  Actions: {actions_path}")

    outcomes_path = INTERIM_DIR / "90b_eicu_outcomes_absorb.parquet"
    outcomes.to_parquet(outcomes_path, index=False)
    print(f"  Outcomes: {outcomes_path}")

    prop_path = INTERIM_DIR / "90b_eicu_propensity_coarse3.parquet"
    prop_df.to_parquet(prop_path, index=False)
    print(f"  Propensity: {prop_path}")

    # ── 7. Summary report ────────────────────────────────────────────────
    n_stays = grid["stay_id"].nunique()
    report_lines = [
        "=" * 72,
        "90b_04 — eICU Actions, Outcomes & Behavior Policy",
        "=" * 72,
        f"Generated: {ts}",
        "",
        "ACTION ENCODING",
        "───────────────",
        f"  Scheme: coarse_3 (fluid×3 × vaso×3 × diuretic×2 = 18 raw)",
        f"  After merge: {grid['action_idx_merged'].nunique()} cells",
        f"  Merge map: MIMIC (frozen)",
        "",
        "RULE POLICY",
        "───────────",
        f"  Thresholds: MIMIC (frozen)",
        f"  Rule-observed agreement: {(grid['action_idx_merged'] == grid['rule_action_merged']).mean()*100:.1f}%",
        "",
        "OUTCOMES",
        "────────",
        f"  28-day mortality: {outcomes['mortality_28d'].mean()*100:.1f}%",
        f"  Early absorbing: {outcomes['absorbing_state_bin'].notna().sum():,}/{n_stays:,}",
        "",
        "BEHAVIOR POLICY (eICU-specific)",
        "────────────────────────────────",
        f"  Model: LogisticRegression (C=1.0, lbfgs)",
        f"  Features: {len(bp_features)}",
        f"  Training accuracy: {bp_acc:.4f}",
        "",
    ]
    report_lines.extend(site_report_lines)
    report_lines.extend([
        "",
        "NEXT STEPS",
        "──────────",
        "  90b_05: Evaluate frozen MIMIC models on eICU data",
    ])

    report_path = REPORTS_DIR / "90b_04_eicu_actions_propensity_summary.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\n  Report: {report_path}")

    total_time = time.time() - t0
    print()
    print("=" * 72)
    print(f"90b_04 COMPLETE. Time: {total_time/60:.1f} min")
    print("=" * 72)


if __name__ == "__main__":
    main()
