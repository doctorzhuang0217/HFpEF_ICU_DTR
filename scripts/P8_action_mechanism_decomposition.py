#!/usr/bin/env python3
"""P8 — Action Mechanism Decomposition: HFpEF vs no_HF in Non-Shock Stratum

Protocol: v3.1-patch, Step P8
Compute tier: C1 (lightweight, no model inference)
Depends on: 70c rule policy, 45b observed actions, 65e phenotype labels

Objective: Demonstrate that the HFpEF-specific treatment effect (2.8x ratio in
non-shock stratum, Step 80g) operates through the fluid/diuretic pathway rather
than the shock/vasopressor pathway.

Analysis:
  1. Compare RULE policy action distributions: HFpEF vs no_HF in non-shock stratum
     → Do the rules recommend different treatments based on hemodynamic state?
  2. Compare OBSERVED (clinician) action distributions: HFpEF vs no_HF
     → Do clinicians actually treat them differently?
  3. Rule-observed agreement gap: which group has more "room to improve"?
  4. Action-level decomposition by clinical pathway:
     - Shock pathway: actions 8 (perfusion support), 14 (resuscitate fluid), 16 (resuscitate aggressive)
     - Decongestion pathway: action 1 (decongest)
     - Maintenance: action 6 (default)
  5. Per-bin rule trigger analysis: which clinical rules fire more often for HFpEF?

Key insight: if HFpEF's larger ATE comes from congestion-related rule triggers
(P4: volume_overload_hypoxemia, P5: severe_volume_overload) and clinician
under-treatment of congestion, it confirms a HFpEF-specific mechanism
independent of shock management.

Outputs:
  - tables/P8_action_mechanism_decomposition.csv
  - tables/P8_rule_trigger_by_hf_subtype.csv
  - figures/P8_action_distribution_comparison.png
  - figures/P8_rule_observed_gap.png
  - reports/P8_action_mechanism_decomposition_summary.txt
"""

from __future__ import annotations

import os
import sys
import time
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
from scipy import stats

# ── Project root ──
ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(ROOT / "src"))

from hfpef_dtr.config import load_config, resolve_path

# ── Paths ──
CONFIG_PATH = ROOT / "configs" / "80_step_c_training_v1.yaml"
NORM_PATH = ROOT / "configs" / "45c_normalization_params_v1.yaml"
TABLES_DIR = ROOT / "tables"
FIGURES_DIR = ROOT / "figures"
REPORTS_DIR = ROOT / "reports"

for d in [TABLES_DIR, FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Action definitions ──
ACTION_NAMES = {
    0: "nothing",
    1: "decongest",
    2: "decongest+fluid",
    4: "vaso_low_only",
    6: "maintenance",
    7: "maintenance+diuretic",
    8: "perfusion_support",
    9: "perfusion_support+diuretic",
    10: "vaso_high_only",
    12: "fluid_high_only",
    13: "fluid_high+diuretic",
    14: "resuscitate_fluid",
    16: "resuscitate_aggressive",
}

# Clinical pathway grouping
PATHWAY_SHOCK = {8, 14, 16}           # vasopressor / aggressive fluid
PATHWAY_DECONGEST = {1, 7, 9, 13}     # any action involving diuretics (except pure shock)
PATHWAY_MAINTENANCE = {0, 6}           # monitoring / maintenance
PATHWAY_OTHER = {2, 4, 10, 12}        # mixed / uncommon

PATHWAY_LABELS = {
    "shock_resuscitation": PATHWAY_SHOCK,
    "decongestion": PATHWAY_DECONGEST,
    "maintenance": PATHWAY_MAINTENANCE,
    "other": PATHWAY_OTHER,
}

# Rule trigger mapping (which hemodynamic rule produces each action)
# From 70c: P1→16, P2→14, P3→8, P4/P5→1, P6→6
RULE_TRIGGER_MAP = {
    16: "P1_shock_hypoperfusion",   # MAP<65 + lactate>2
    14: "P2_hypotension_only",      # MAP<65
    8:  "P3_hypoperfusion_only",    # lactate>2, MAP>=65
    1:  "P4P5_volume_overload",     # fluid_balance > threshold
    6:  "P6_default_maintenance",   # otherwise
}


def load_norm_params() -> dict:
    """Load normalization parameters for z-score inversion."""
    with open(NORM_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("params", {})


def _cumulative_fluid_balance_L(
    state: pd.DataFrame | pd.Series, norm_params: dict,
) -> np.ndarray:
    """Resolve cumulative balance in litres; never substitute interval balance.

    Raw cumulative columns take precedence over normalized cumulative columns.
    This resolves the operand only; it does not change the upstream time grid.
    """
    columns = state.columns if isinstance(state, pd.DataFrame) else state.index
    aliases = ("cumulative_fluid_balance_L", "fluid_balance_cumulative_L",
               "cum_fluid_balance_L")
    for column in aliases:
        if column in columns:
            return np.asarray(state[column], dtype=float)
    for column in aliases:
        z_column = column + "_z"
        if z_column in columns:
            params = norm_params.get(column)
            if not isinstance(params, dict) or not {"mean", "std"} <= params.keys():
                raise ValueError(f"{z_column} requires matching {column} normalization parameters")
            mean, std = float(params["mean"]), float(params["std"])
            if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
                raise ValueError(f"Invalid normalization parameters for {column}")
            return np.asarray(state[z_column], dtype=float) * std + mean
    raise ValueError(
        "Cumulative fluid balance is required: expected cumulative_fluid_balance_L "
        "(or a cumulative alias / normalized counterpart with matching parameters). "
        "Interval fluid_balance_L / fluid_balance_L_z cannot be used instead."
    )


def classify_rule_trigger(state_row: pd.Series, norm_params: dict) -> str:
    """Determine which rule trigger applies to a state observation.

    Reproduces 70c rule logic to label each observation with the
    clinical trigger that would fire.
    """
    # Invert z-scores to raw values
    if "map_z" in state_row.index:
        map_val = state_row["map_z"] * norm_params["map"]["std"] + norm_params["map"]["mean"]
    else:
        map_val = state_row.get("map", 80.0)

    if "lactate_z" in state_row.index:
        lac_val = state_row["lactate_z"] * norm_params["lactate"]["std"] + norm_params["lactate"]["mean"]
    else:
        lac_val = state_row.get("lactate", 1.5)

    fb_val = float(_cumulative_fluid_balance_L(state_row, norm_params))

    if "spo2_z" in state_row.index:
        spo2_val = state_row["spo2_z"] * norm_params["spo2"]["std"] + norm_params["spo2"]["mean"]
    else:
        spo2_val = state_row.get("spo2", 97.0)

    # Priority-ordered rules (same as 70c)
    if map_val < 65 and lac_val > 2.0:
        return "P1_shock_hypoperfusion"
    elif map_val < 65:
        return "P2_hypotension_only"
    elif lac_val > 2.0:
        return "P3_hypoperfusion_only"
    elif fb_val > 3.0 and spo2_val < 94:
        return "P4_volume_overload_hypoxemia"
    elif fb_val > 5.0:
        return "P5_severe_volume_overload"
    else:
        return "P6_default_maintenance"


def classify_rule_trigger_vectorized(state_df: pd.DataFrame, norm_params: dict) -> pd.Series:
    """Vectorized version of rule trigger classification."""
    n = len(state_df)

    # MAP
    if "map_z" in state_df.columns:
        map_vals = state_df["map_z"].values * float(norm_params["map"]["std"]) + float(norm_params["map"]["mean"])
    elif "map" in state_df.columns:
        map_vals = state_df["map"].values
    else:
        map_vals = np.full(n, 80.0)

    # Lactate
    if "lactate_z" in state_df.columns:
        lac_vals = state_df["lactate_z"].values * float(norm_params["lactate"]["std"]) + float(norm_params["lactate"]["mean"])
    elif "lactate" in state_df.columns:
        lac_vals = state_df["lactate"].values
    else:
        lac_vals = np.full(n, 1.5)

    # Cumulative fluid balance (L), not the current two-hour interval balance.
    fb_vals = _cumulative_fluid_balance_L(state_df, norm_params)

    # SpO2
    if "spo2_z" in state_df.columns:
        spo2_vals = state_df["spo2_z"].values * float(norm_params["spo2"]["std"]) + float(norm_params["spo2"]["mean"])
    elif "spo2" in state_df.columns:
        spo2_vals = state_df["spo2"].values
    else:
        spo2_vals = np.full(n, 97.0)

    # Apply rules in priority order
    triggers = np.full(n, "P6_default_maintenance", dtype=object)

    # P4 precedes P5 in the scalar classifier and the documented 70c rules.
    mask_p4 = (fb_vals > 3.0) & (spo2_vals < 94) & (map_vals >= 65) & (lac_vals <= 2.0)
    triggers[mask_p4] = "P4_volume_overload_hypoxemia"

    mask_p5 = (fb_vals > 5.0) & (map_vals >= 65) & (lac_vals <= 2.0) & ~mask_p4
    triggers[mask_p5] = "P5_severe_volume_overload"

    mask_p3 = (lac_vals > 2.0) & (map_vals >= 65)
    triggers[mask_p3] = "P3_hypoperfusion_only"

    mask_p2 = (map_vals < 65) & (lac_vals <= 2.0)
    triggers[mask_p2] = "P2_hypotension_only"

    mask_p1 = (map_vals < 65) & (lac_vals > 2.0)
    triggers[mask_p1] = "P1_shock_hypoperfusion"

    return pd.Series(triggers, index=state_df.index, name="rule_trigger")


def action_to_pathway(action_idx: int) -> str:
    """Map action index to clinical pathway category."""
    for pathway_name, action_set in PATHWAY_LABELS.items():
        if action_idx in action_set:
            return pathway_name
    return "other"


def main() -> None:
    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 72)
    print("P8 — Action Mechanism Decomposition")
    print("     HFpEF vs no_HF in Non-Shock Stratum")
    print("=" * 72)

    # ── 1. Load config ──
    print("\n[1] Loading configuration...")
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    proj_cfg = load_config()
    interim_dir = resolve_path(proj_cfg, "interim")

    # ── 2. Load data ──
    print("\n[2] Loading data...")

    # State tensors (for rule trigger analysis)
    state_path = ROOT / config["data"]["state_tensors"]
    state_df = pd.read_parquet(state_path)
    print(f"  State tensors: {len(state_df):,} obs, {state_df['stay_id'].nunique():,} stays")

    # Observed actions
    actions_path = interim_dir / "45b_actions_2h_v1_mimic.parquet"
    actions_df = pd.read_parquet(actions_path, columns=["stay_id", "bin", "action_idx"])
    actions_df.rename(columns={"action_idx": "observed_action"}, inplace=True)

    # Rule policy actions
    rule_path = interim_dir / "70c_rule_policy_actions_v1_mimic.parquet"
    rule_df = pd.read_parquet(rule_path, columns=["stay_id", "bin", "rule_policy_action"])

    # Phenotype labels
    pheno_path = ROOT / config["data"]["phenotype_labels"]
    pheno_df = pd.read_parquet(pheno_path)

    print(f"  Phenotype labels: {len(pheno_df):,} stays")
    print(f"  Columns: {list(pheno_df.columns)}")

    # ── 3. Test set split (same as 80g) ──
    print("\n[3] Applying test set split (same as 80g)...")
    from hfpef_dtr.models.dataset import make_train_val_test_split
    outcomes_path = ROOT / config["data"]["outcomes"]
    outcomes_df = pd.read_parquet(outcomes_path)
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
    test_set = set(test_ids)
    print(f"  Test stays: {len(test_ids):,}")

    # ── 4. Merge all data ──
    print("\n[4] Merging datasets...")
    merged = state_df[state_df["stay_id"].isin(test_set)].copy()
    merged = merged.merge(actions_df, on=["stay_id", "bin"], how="left")
    merged = merged.merge(rule_df, on=["stay_id", "bin"], how="left")

    # Avoid column collisions: drop cols from merged that also exist in pheno_df
    pheno_cols = ["hf_subtype", "rule_phenotype_2class"]
    for col in pheno_cols:
        if col in merged.columns:
            merged.drop(columns=[col], inplace=True)
    merged = merged.merge(
        pheno_df[["stay_id"] + pheno_cols],
        on="stay_id", how="left",
    )
    print(f"  Merged test observations: {len(merged):,}")
    print(f"  Columns after merge: {list(merged.columns)}")
    print(f"  HF subtypes: {merged['hf_subtype'].value_counts().to_dict()}")
    print(f"  Shock strata: {merged['rule_phenotype_2class'].value_counts().to_dict()}")

    # ── 5. Classify rule triggers ──
    print("\n[5] Classifying rule triggers (vectorized)...")
    norm_params = load_norm_params()
    merged["rule_trigger"] = classify_rule_trigger_vectorized(merged, norm_params)
    print(f"  Rule trigger distribution:\n{merged['rule_trigger'].value_counts().to_string()}")

    # ── 6. Add pathway labels ──
    merged["rule_pathway"] = merged["rule_policy_action"].map(action_to_pathway)
    merged["observed_pathway"] = merged["observed_action"].map(action_to_pathway)

    # ── 7. Non-shock stratum analysis ──
    print("\n" + "=" * 72)
    print("[7] NON-SHOCK STRATUM ANALYSIS")
    print("=" * 72)

    non_shock = merged[merged["rule_phenotype_2class"] == "non_shock"].copy()
    print(f"\n  Non-shock observations: {len(non_shock):,}")
    print(f"  Non-shock stays: {non_shock['stay_id'].nunique():,}")

    # Focus groups
    groups = ["hfpef", "no_hf"]
    results_rows = []

    for grp in groups:
        grp_data = non_shock[non_shock["hf_subtype"] == grp]
        n_obs = len(grp_data)
        n_stays = grp_data["stay_id"].nunique()
        print(f"\n  ── {grp} (n_stays={n_stays}, n_obs={n_obs}) ──")

        # 7a. Rule action distribution
        rule_dist = grp_data["rule_policy_action"].value_counts(normalize=True).sort_index()
        print(f"    Rule policy action distribution:")
        for aidx, pct in rule_dist.items():
            name = ACTION_NAMES.get(aidx, f"action_{aidx}")
            print(f"      {aidx:2d} ({name}): {pct*100:.1f}%")

        # 7b. Observed action distribution
        obs_dist = grp_data["observed_action"].value_counts(normalize=True).sort_index()
        print(f"    Observed action distribution:")
        for aidx, pct in obs_dist.items():
            name = ACTION_NAMES.get(aidx, f"action_{aidx}")
            if pct >= 0.01:
                print(f"      {aidx:2d} ({name}): {pct*100:.1f}%")

        # 7c. Pathway distribution (rule)
        rule_pw = grp_data["rule_pathway"].value_counts(normalize=True)
        print(f"    Rule pathway distribution:")
        for pw, pct in rule_pw.items():
            print(f"      {pw}: {pct*100:.1f}%")

        # 7d. Pathway distribution (observed)
        obs_pw = grp_data["observed_pathway"].value_counts(normalize=True)
        print(f"    Observed pathway distribution:")
        for pw, pct in obs_pw.items():
            print(f"      {pw}: {pct*100:.1f}%")

        # 7e. Rule trigger distribution
        trigger_dist = grp_data["rule_trigger"].value_counts(normalize=True)
        print(f"    Rule trigger distribution:")
        for trig, pct in trigger_dist.items():
            print(f"      {trig}: {pct*100:.1f}%")

        # 7f. Agreement rate
        agreement = (grp_data["rule_policy_action"] == grp_data["observed_action"]).mean()
        print(f"    Rule-observed agreement: {agreement*100:.1f}%")

        # 7g. Decongest-specific analysis
        rule_decongest_pct = (grp_data["rule_pathway"] == "decongestion").mean()
        obs_decongest_pct = (grp_data["observed_pathway"] == "decongestion").mean()
        decongest_gap = rule_decongest_pct - obs_decongest_pct
        print(f"    Decongestion: rule={rule_decongest_pct*100:.2f}%, "
              f"observed={obs_decongest_pct*100:.2f}%, gap={decongest_gap*100:.2f}pp")

        # 7h. Shock pathway analysis
        rule_shock_pct = (grp_data["rule_pathway"] == "shock_resuscitation").mean()
        obs_shock_pct = (grp_data["observed_pathway"] == "shock_resuscitation").mean()
        shock_gap = rule_shock_pct - obs_shock_pct
        print(f"    Shock pathway: rule={rule_shock_pct*100:.2f}%, "
              f"observed={obs_shock_pct*100:.2f}%, gap={shock_gap*100:.2f}pp")

        # Volume overload triggers (P4+P5 combined)
        vol_overload_triggers = (
            grp_data["rule_trigger"].isin(["P4_volume_overload_hypoxemia", "P5_severe_volume_overload"])
        ).mean()
        print(f"    Volume overload triggers (P4+P5): {vol_overload_triggers*100:.2f}%")

        # Store for CSV
        results_rows.append({
            "stratum": "non_shock",
            "hf_subtype": grp,
            "n_stays": n_stays,
            "n_obs": n_obs,
            "rule_pct_shock": rule_shock_pct,
            "rule_pct_decongest": rule_decongest_pct,
            "rule_pct_maintenance": (grp_data["rule_pathway"] == "maintenance").mean(),
            "obs_pct_shock": obs_shock_pct,
            "obs_pct_decongest": obs_decongest_pct,
            "obs_pct_maintenance": (grp_data["observed_pathway"] == "maintenance").mean(),
            "agreement_rate": agreement,
            "decongest_gap_pp": decongest_gap * 100,
            "shock_gap_pp": shock_gap * 100,
            "vol_overload_trigger_pct": vol_overload_triggers,
            "P1_trigger_pct": (grp_data["rule_trigger"] == "P1_shock_hypoperfusion").mean(),
            "P2_trigger_pct": (grp_data["rule_trigger"] == "P2_hypotension_only").mean(),
            "P3_trigger_pct": (grp_data["rule_trigger"] == "P3_hypoperfusion_only").mean(),
            "P4_trigger_pct": (grp_data["rule_trigger"] == "P4_volume_overload_hypoxemia").mean(),
            "P5_trigger_pct": (grp_data["rule_trigger"] == "P5_severe_volume_overload").mean(),
            "P6_trigger_pct": (grp_data["rule_trigger"] == "P6_default_maintenance").mean(),
        })

    # ── 8. Statistical tests ──
    print("\n" + "=" * 72)
    print("[8] STATISTICAL TESTS")
    print("=" * 72)

    hfpef_data = non_shock[non_shock["hf_subtype"] == "hfpef"]
    nohf_data = non_shock[non_shock["hf_subtype"] == "no_hf"]

    # 8a. Chi-square on rule action distribution
    all_actions = sorted(set(hfpef_data["rule_policy_action"].unique()) |
                         set(nohf_data["rule_policy_action"].unique()))
    contingency_rule = pd.DataFrame({
        "hfpef": hfpef_data["rule_policy_action"].value_counts().reindex(all_actions, fill_value=0),
        "no_hf": nohf_data["rule_policy_action"].value_counts().reindex(all_actions, fill_value=0),
    })
    chi2_rule, p_rule, dof_rule, _ = stats.chi2_contingency(contingency_rule.T)
    print(f"\n  Chi-square (rule action distribution): chi2={chi2_rule:.2f}, "
          f"p={p_rule:.2e}, dof={dof_rule}")

    # 8b. Chi-square on observed action distribution
    all_obs_actions = sorted(set(hfpef_data["observed_action"].dropna().unique().astype(int)) |
                             set(nohf_data["observed_action"].dropna().unique().astype(int)))
    contingency_obs = pd.DataFrame({
        "hfpef": hfpef_data["observed_action"].value_counts().reindex(all_obs_actions, fill_value=0),
        "no_hf": nohf_data["observed_action"].value_counts().reindex(all_obs_actions, fill_value=0),
    })
    chi2_obs, p_obs, dof_obs, _ = stats.chi2_contingency(contingency_obs.T)
    print(f"  Chi-square (observed action distribution): chi2={chi2_obs:.2f}, "
          f"p={p_obs:.2e}, dof={dof_obs}")

    # 8c. Chi-square on rule trigger distribution
    all_triggers = sorted(set(hfpef_data["rule_trigger"].unique()) |
                          set(nohf_data["rule_trigger"].unique()))
    contingency_trig = pd.DataFrame({
        "hfpef": hfpef_data["rule_trigger"].value_counts().reindex(all_triggers, fill_value=0),
        "no_hf": nohf_data["rule_trigger"].value_counts().reindex(all_triggers, fill_value=0),
    })
    chi2_trig, p_trig, dof_trig, _ = stats.chi2_contingency(contingency_trig.T)
    print(f"  Chi-square (rule trigger distribution): chi2={chi2_trig:.2f}, "
          f"p={p_trig:.2e}, dof={dof_trig}")

    # 8d. Two-proportion z-test: decongestion trigger (P4+P5) HFpEF vs no_HF
    hfpef_vol = hfpef_data["rule_trigger"].isin(
        ["P4_volume_overload_hypoxemia", "P5_severe_volume_overload"]
    )
    nohf_vol = nohf_data["rule_trigger"].isin(
        ["P4_volume_overload_hypoxemia", "P5_severe_volume_overload"]
    )
    n1, n2 = len(hfpef_vol), len(nohf_vol)
    p1, p2 = hfpef_vol.mean(), nohf_vol.mean()
    if p1 + p2 > 0:
        p_pool = (hfpef_vol.sum() + nohf_vol.sum()) / (n1 + n2)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        z_vol = (p1 - p2) / se if se > 0 else 0.0
        p_vol = 2 * (1 - stats.norm.cdf(abs(z_vol)))
    else:
        z_vol, p_vol = 0.0, 1.0
    print(f"\n  Volume overload trigger (P4+P5):")
    print(f"    HFpEF: {p1*100:.2f}%  vs  no_HF: {p2*100:.2f}%")
    print(f"    Difference: {(p1-p2)*100:.2f}pp, z={z_vol:.2f}, p={p_vol:.4f}")

    # 8e. Agreement rate difference (z-test)
    agree_hfpef = (hfpef_data["rule_policy_action"] == hfpef_data["observed_action"]).mean()
    agree_nohf = (nohf_data["rule_policy_action"] == nohf_data["observed_action"]).mean()
    n_h, n_n = len(hfpef_data), len(nohf_data)
    p_pool_agree = ((hfpef_data["rule_policy_action"] == hfpef_data["observed_action"]).sum() +
                    (nohf_data["rule_policy_action"] == nohf_data["observed_action"]).sum()) / (n_h + n_n)
    se_agree = np.sqrt(p_pool_agree * (1 - p_pool_agree) * (1/n_h + 1/n_n))
    z_agree = (agree_hfpef - agree_nohf) / se_agree if se_agree > 0 else 0.0
    p_agree = 2 * (1 - stats.norm.cdf(abs(z_agree)))
    print(f"\n  Rule-observed agreement rate:")
    print(f"    HFpEF: {agree_hfpef*100:.1f}%  vs  no_HF: {agree_nohf*100:.1f}%")
    print(f"    Difference: {(agree_hfpef - agree_nohf)*100:.1f}pp, z={z_agree:.2f}, p={p_agree:.4f}")

    # ── 9. Also analyze SHOCK stratum for comparison ──
    print("\n" + "=" * 72)
    print("[9] SHOCK STRATUM COMPARISON (for contrast)")
    print("=" * 72)

    shock = merged[merged["rule_phenotype_2class"] == "shock_predominant"]
    for grp in groups:
        grp_data = shock[shock["hf_subtype"] == grp]
        n_stays = grp_data["stay_id"].nunique()
        n_obs = len(grp_data)
        rule_pw = grp_data["rule_pathway"].value_counts(normalize=True)
        trigger_dist = grp_data["rule_trigger"].value_counts(normalize=True)
        agreement = (grp_data["rule_policy_action"] == grp_data["observed_action"]).mean()

        print(f"\n  {grp} (n_stays={n_stays}):")
        print(f"    Rule pathway: {dict(rule_pw.round(3))}")
        print(f"    Top triggers: {dict(trigger_dist.head(3).round(3))}")
        print(f"    Agreement: {agreement*100:.1f}%")

        results_rows.append({
            "stratum": "shock_predominant",
            "hf_subtype": grp,
            "n_stays": n_stays,
            "n_obs": n_obs,
            "rule_pct_shock": (grp_data["rule_pathway"] == "shock_resuscitation").mean(),
            "rule_pct_decongest": (grp_data["rule_pathway"] == "decongestion").mean(),
            "rule_pct_maintenance": (grp_data["rule_pathway"] == "maintenance").mean(),
            "obs_pct_shock": (grp_data["observed_pathway"] == "shock_resuscitation").mean(),
            "obs_pct_decongest": (grp_data["observed_pathway"] == "decongestion").mean(),
            "obs_pct_maintenance": (grp_data["observed_pathway"] == "maintenance").mean(),
            "agreement_rate": agreement,
            "decongest_gap_pp": ((grp_data["rule_pathway"] == "decongestion").mean() -
                                  (grp_data["observed_pathway"] == "decongestion").mean()) * 100,
            "shock_gap_pp": ((grp_data["rule_pathway"] == "shock_resuscitation").mean() -
                              (grp_data["observed_pathway"] == "shock_resuscitation").mean()) * 100,
            "vol_overload_trigger_pct": grp_data["rule_trigger"].isin(
                ["P4_volume_overload_hypoxemia", "P5_severe_volume_overload"]).mean(),
            "P1_trigger_pct": (grp_data["rule_trigger"] == "P1_shock_hypoperfusion").mean(),
            "P2_trigger_pct": (grp_data["rule_trigger"] == "P2_hypotension_only").mean(),
            "P3_trigger_pct": (grp_data["rule_trigger"] == "P3_hypoperfusion_only").mean(),
            "P4_trigger_pct": (grp_data["rule_trigger"] == "P4_volume_overload_hypoxemia").mean(),
            "P5_trigger_pct": (grp_data["rule_trigger"] == "P5_severe_volume_overload").mean(),
            "P6_trigger_pct": (grp_data["rule_trigger"] == "P6_default_maintenance").mean(),
        })

    # ── 10. Per-bin rule trigger analysis (non-shock, HFpEF vs no_HF) ──
    print("\n" + "=" * 72)
    print("[10] PER-BIN RULE TRIGGER EVOLUTION (non-shock)")
    print("=" * 72)

    trigger_by_bin = []
    for grp in groups:
        grp_data = non_shock[non_shock["hf_subtype"] == grp]
        for b in range(12):
            bin_data = grp_data[grp_data["bin"] == b]
            if len(bin_data) == 0:
                continue
            vol_pct = bin_data["rule_trigger"].isin(
                ["P4_volume_overload_hypoxemia", "P5_severe_volume_overload"]
            ).mean()
            shock_pct = bin_data["rule_trigger"].isin(
                ["P1_shock_hypoperfusion", "P2_hypotension_only"]
            ).mean()
            trigger_by_bin.append({
                "hf_subtype": grp,
                "bin": b,
                "hours": b * 2,
                "vol_overload_pct": vol_pct,
                "shock_trigger_pct": shock_pct,
                "n_obs": len(bin_data),
            })

    trigger_bin_df = pd.DataFrame(trigger_by_bin)
    print(trigger_bin_df.pivot_table(
        index="bin", columns="hf_subtype",
        values=["vol_overload_pct", "shock_trigger_pct"],
        aggfunc="first",
    ).to_string())

    # ── 11. Save results ──
    print("\n" + "=" * 72)
    print("[11] Saving outputs...")

    results_df = pd.DataFrame(results_rows)
    csv_path = TABLES_DIR / "P8_action_mechanism_decomposition.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    trigger_csv = TABLES_DIR / "P8_rule_trigger_by_hf_subtype.csv"
    trigger_bin_df.to_csv(trigger_csv, index=False)
    print(f"  Saved: {trigger_csv}")

    # ── 12. Figures ──
    print("\n[12] Generating figures...")

    # Figure 1: Action pathway comparison (non-shock)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pathway_order = ["shock_resuscitation", "decongestion", "maintenance", "other"]
    colors = {"shock_resuscitation": "#e74c3c", "decongestion": "#3498db",
              "maintenance": "#2ecc71", "other": "#95a5a6"}

    for ax_idx, (label, data_col) in enumerate([
        ("Rule Policy", "rule_pathway"),
        ("Observed (Clinician)", "observed_pathway"),
    ]):
        ax = axes[ax_idx]
        x = np.arange(len(pathway_order))
        width = 0.35

        hfpef_ns = non_shock[non_shock["hf_subtype"] == "hfpef"]
        nohf_ns = non_shock[non_shock["hf_subtype"] == "no_hf"]

        hfpef_pcts = [hfpef_ns[data_col].eq(pw).mean() * 100 for pw in pathway_order]
        nohf_pcts = [nohf_ns[data_col].eq(pw).mean() * 100 for pw in pathway_order]

        bars1 = ax.bar(x - width/2, hfpef_pcts, width, label="HFpEF", color="#e74c3c", alpha=0.8)
        bars2 = ax.bar(x + width/2, nohf_pcts, width, label="No HF", color="#3498db", alpha=0.8)

        ax.set_xlabel("Clinical Pathway")
        ax.set_ylabel("Proportion (%)")
        ax.set_title(f"{label}\n(Non-Shock Stratum)")
        ax.set_xticks(x)
        ax.set_xticklabels(pathway_order, rotation=30, ha="right", fontsize=9)
        ax.legend()

        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                if h > 0.5:
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                            f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Treatment Pathway Distribution \u2014 HFpEF vs No HF (Non-Shock)", fontsize=13)
    plt.tight_layout()
    fig1_path = FIGURES_DIR / "P8_action_distribution_comparison.png"
    fig.savefig(fig1_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig1_path}")

    # Figure 2: Rule-observed gap by pathway
    fig, ax = plt.subplots(figsize=(8, 5))
    pathways_plot = ["shock_resuscitation", "decongestion", "maintenance"]
    x = np.arange(len(pathways_plot))
    width = 0.35

    hfpef_gaps = []
    nohf_gaps = []
    for pw in pathways_plot:
        hfpef_ns = non_shock[non_shock["hf_subtype"] == "hfpef"]
        nohf_ns = non_shock[non_shock["hf_subtype"] == "no_hf"]
        hfpef_gap = (hfpef_ns["rule_pathway"].eq(pw).mean() - hfpef_ns["observed_pathway"].eq(pw).mean()) * 100
        nohf_gap = (nohf_ns["rule_pathway"].eq(pw).mean() - nohf_ns["observed_pathway"].eq(pw).mean()) * 100
        hfpef_gaps.append(hfpef_gap)
        nohf_gaps.append(nohf_gap)

    ax.bar(x - width/2, hfpef_gaps, width, label="HFpEF", color="#e74c3c", alpha=0.8)
    ax.bar(x + width/2, nohf_gaps, width, label="No HF", color="#3498db", alpha=0.8)
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Clinical Pathway")
    ax.set_ylabel("Rule − Observed Gap (percentage points)")
    ax.set_title("Treatment Gap (Rule vs Clinician)\nNon-Shock Stratum")
    ax.set_xticks(x)
    _pw_d = {"shock_resuscitation": "Shock resus.", "decongestion": "Decongestion",
             "maintenance": "Maintenance"}
    ax.set_xticklabels([_pw_d.get(p, p) for p in pathways_plot], fontsize=10)
    ax.legend()
    plt.tight_layout()
    fig2_path = FIGURES_DIR / "P8_rule_observed_gap.png"
    fig.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig2_path}")

    # Figure 3: Per-bin volume overload trigger evolution
    fig, ax = plt.subplots(figsize=(8, 5))
    _hf_d = {"hfpef": "HFpEF", "no_hf": "No HF"}
    for grp, color, ls in [("hfpef", "#e74c3c", "-"), ("no_hf", "#3498db", "--")]:
        grp_bins = trigger_bin_df[trigger_bin_df["hf_subtype"] == grp]
        ax.plot(grp_bins["hours"], grp_bins["vol_overload_pct"] * 100,
                color=color, linestyle=ls, marker="o", markersize=4,
                label=_hf_d.get(grp, grp))
    ax.set_xlabel("Hours After T0")
    ax.set_ylabel("Volume Overload Trigger Rate (%)")
    ax.set_title("Volume Overload Rule Trigger\nNon-Shock Stratum, by HF Subtype")
    ax.legend()
    ax.set_xticks(range(0, 24, 2))
    plt.tight_layout()
    fig3_path = FIGURES_DIR / "P8_vol_overload_trigger_by_bin.png"
    fig.savefig(fig3_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig3_path}")

    # ── 13. Generate report ──
    print("\n[13] Generating report...")

    # Extract key numbers
    hfpef_row = [r for r in results_rows if r["hf_subtype"] == "hfpef" and r["stratum"] == "non_shock"][0]
    nohf_row = [r for r in results_rows if r["hf_subtype"] == "no_hf" and r["stratum"] == "non_shock"][0]

    lines = [
        "=" * 72,
        "P8 — Action Mechanism Decomposition (MIMIC-IV)",
        "     HFpEF vs no_HF in Non-Shock Stratum",
        "=" * 72,
        f"Generated: {ts}",
        "",
        "PURPOSE",
        "───────",
        "  Demonstrate that the HFpEF-specific treatment effect (80g: 2.8× ratio",
        "  in non-shock stratum) operates through the fluid/diuretic pathway",
        "  rather than the shock/vasopressor pathway.",
        "",
        "═══════════════════════════════════════════════════════════════════════",
        "KEY FINDINGS",
        "═══════════════════════════════════════════════════════════════════════",
        "",
        "1. RULE TRIGGER DISTRIBUTION (non-shock stratum)",
        "─────────────────────────────────────────────────",
        f"   Volume overload triggers (P4+P5):",
        f"     HFpEF: {hfpef_row['vol_overload_trigger_pct']*100:.2f}%",
        f"     no_HF: {nohf_row['vol_overload_trigger_pct']*100:.2f}%",
        f"     z-test: z={z_vol:.2f}, p={p_vol:.4f}",
        "",
        f"   Shock triggers (P1+P2):",
        f"     HFpEF: {(hfpef_row['P1_trigger_pct']+hfpef_row['P2_trigger_pct'])*100:.2f}%",
        f"     no_HF: {(nohf_row['P1_trigger_pct']+nohf_row['P2_trigger_pct'])*100:.2f}%",
        "",
        "2. TREATMENT PATHWAY DISTRIBUTION (non-shock stratum)",
        "──────────────────────────────────────────────────────",
        f"   Rule policy — shock pathway:",
        f"     HFpEF: {hfpef_row['rule_pct_shock']*100:.1f}%  vs  no_HF: {nohf_row['rule_pct_shock']*100:.1f}%",
        f"   Rule policy — decongestion pathway:",
        f"     HFpEF: {hfpef_row['rule_pct_decongest']*100:.1f}%  vs  no_HF: {nohf_row['rule_pct_decongest']*100:.1f}%",
        f"   Rule policy — maintenance:",
        f"     HFpEF: {hfpef_row['rule_pct_maintenance']*100:.1f}%  vs  no_HF: {nohf_row['rule_pct_maintenance']*100:.1f}%",
        "",
        f"   Observed — shock pathway:",
        f"     HFpEF: {hfpef_row['obs_pct_shock']*100:.1f}%  vs  no_HF: {nohf_row['obs_pct_shock']*100:.1f}%",
        f"   Observed — decongestion pathway:",
        f"     HFpEF: {hfpef_row['obs_pct_decongest']*100:.1f}%  vs  no_HF: {nohf_row['obs_pct_decongest']*100:.1f}%",
        "",
        f"   Chi-square (rule actions): chi2={chi2_rule:.2f}, p={p_rule:.2e}",
        f"   Chi-square (observed actions): chi2={chi2_obs:.2f}, p={p_obs:.2e}",
        "",
        "3. RULE-OBSERVED AGREEMENT GAP (non-shock stratum)",
        "───────────────────────────────────────────────────",
        f"   HFpEF agreement: {hfpef_row['agreement_rate']*100:.1f}%",
        f"   no_HF agreement: {nohf_row['agreement_rate']*100:.1f}%",
        f"   Difference: {(hfpef_row['agreement_rate'] - nohf_row['agreement_rate'])*100:.1f}pp "
        f"(z={z_agree:.2f}, p={p_agree:.4f})",
        "",
        f"   Decongestion gap (rule − observed):",
        f"     HFpEF: {hfpef_row['decongest_gap_pp']:+.2f}pp",
        f"     no_HF: {nohf_row['decongest_gap_pp']:+.2f}pp",
        "",
        "═══════════════════════════════════════════════════════════════════════",
        "INTERPRETATION",
        "═══════════════════════════════════════════════════════════════════════",
        "",
        "  Scenario A: HFpEF has MORE volume-overload triggers than no_HF",
        "    → HFpEF has more congestion substrate; rule policy decongests more;",
        "      treatment effect comes from decongestion pathway (HFpEF-specific)",
        "",
        "  Scenario B: HFpEF has SIMILAR triggers but LOWER clinician agreement",
        "    → Clinicians under-recognize HFpEF congestion in ICU;",
        "      rule policy corrects this gap (HFpEF-specific practice gap)",
        "",
        "  Scenario C: HFpEF has MORE shock triggers (would undermine thesis)",
        "    → Non-shock classification doesn't fully separate hemodynamics;",
        "      effect may still be partly shock-driven",
        "",
        "  Combined with P1 (SSC-2021 ATE≈0 for HFpEF, ESC-2023 ATE=-1.66pp)",
        "  and 80g (2.8× non-shock ratio), the mechanism decomposition completes",
        "  the evidence chain for HFpEF-specific treatment effect attribution.",
        "",
        "ARTIFACTS",
        "─────────",
        f"  tables/P8_action_mechanism_decomposition.csv",
        f"  tables/P8_rule_trigger_by_hf_subtype.csv",
        f"  figures/P8_action_distribution_comparison.png",
        f"  figures/P8_rule_observed_gap.png",
        f"  figures/P8_vol_overload_trigger_by_bin.png",
    ]

    report_path = REPORTS_DIR / "P8_action_mechanism_decomposition_summary.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {report_path}")

    total_time = time.time() - t0
    print(f"\n{'=' * 72}")
    print(f"P8 COMPLETE. Total time: {total_time:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
