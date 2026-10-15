#!/usr/bin/env python3
"""P1-secondary — Secondary Outcomes IPTW / Concordance Analysis

Protocol: v3.1-patch, Step P1 (supplementary)
Compute tier: C2-C3 (CPU only)
Depends on: 45c state tensors, 45d outcomes, 70b propensity, 70c rule policy,
            10h vent indicators, 10e labs, phenotype labels

Objective: Evaluate 6 secondary outcomes under the rule-based treatment policy
using a simplified IPTW/WIS approach and rule-concordance stratification.

Secondary outcomes:
  1. 7-day mortality (from 45d)
  2. ICU-free days (28 - ICU LOS, capped at 0)
  3. 72h net fluid balance (cumulative fluid balance at bin 36)
  4. 72h intubation/IMV escalation (new IMV after T0)
  5. 28-day ventilator-free days (VFD-28)
  6. AKI progression (simplified KDIGO stage 1+)

Analysis:
  - Rule-concordance stratification: top vs bottom tertile of per-patient
    concordance rate (fraction of 12 bins where rule action == observed action)
  - WIS: importance-weighted outcome using behavior policy propensities
  - Stratified by: overall, HFpEF, no_hf

Outputs:
  - tables/P1_secondary_outcomes_summary.csv
  - reports/P1_secondary_outcomes_summary.txt
"""

from __future__ import annotations

import datetime
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Project root ──
ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(ROOT / "src"))

from hfpef_dtr.config import load_config, resolve_path

# ── Constants ──
T0_HOURS = 6
BIN_HOURS = 2
N_BINS = 12  # 12 bins × 2h = 24h post-T0
BOOTSTRAP_B = 500
BOOTSTRAP_SEED = 2024


def _bootstrap_ci(
    values: np.ndarray,
    B: int = BOOTSTRAP_B,
    alpha: float = 0.05,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Bootstrap mean and (1-alpha) CI."""
    rng = np.random.RandomState(seed)
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan
    boot_means = np.empty(B)
    for b in range(B):
        idx = rng.randint(0, n, size=n)
        boot_means[b] = np.nanmean(values[idx])
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(np.nanmean(values)), float(lo), float(hi)


def _bootstrap_diff_ci(
    vals_a: np.ndarray,
    vals_b: np.ndarray,
    B: int = BOOTSTRAP_B,
    alpha: float = 0.05,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Bootstrap difference of means (a - b) and (1-alpha) CI."""
    rng = np.random.RandomState(seed)
    na, nb = len(vals_a), len(vals_b)
    if na == 0 or nb == 0:
        return np.nan, np.nan, np.nan
    boot_diffs = np.empty(B)
    for b in range(B):
        ma = np.nanmean(vals_a[rng.randint(0, na, size=na)])
        mb = np.nanmean(vals_b[rng.randint(0, nb, size=nb)])
        boot_diffs[b] = ma - mb
    lo = np.percentile(boot_diffs, 100 * alpha / 2)
    hi = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
    return float(np.nanmean(vals_a) - np.nanmean(vals_b)), float(lo), float(hi)


# ══════════════════════════════════════════════════════════════════════════════
# Outcome construction functions
# ══════════════════════════════════════════════════════════════════════════════

def build_mortality_7d(outcomes_df: pd.DataFrame) -> pd.Series:
    """Outcome 1: 7-day mortality (directly from 45d)."""
    print("  [1] 7-day mortality from 45d ...")
    if "mortality_7d" not in outcomes_df.columns:
        raise KeyError("mortality_7d not found in 45d outcomes")
    out = outcomes_df.set_index("stay_id")["mortality_7d"].astype(float)
    print(f"      N={len(out):,}, mean={out.mean():.4f}")
    return out


def build_icu_free_days(outcomes_df: pd.DataFrame) -> pd.Series:
    """Outcome 2: ICU-free days = max(0, 28 - icu_los_days)."""
    print("  [2] ICU-free days ...")
    if "icu_los_days" not in outcomes_df.columns:
        raise KeyError("icu_los_days not found in 45d outcomes")
    ifd = np.maximum(0.0, 28.0 - outcomes_df["icu_los_days"].values)
    # If patient died within 28 days, set ICU-free days to 0
    if "mortality_28d" in outcomes_df.columns:
        died = outcomes_df["mortality_28d"].values == 1
        ifd[died] = 0.0
    out = pd.Series(ifd, index=outcomes_df["stay_id"].values, name="icu_free_days")
    out.index.name = "stay_id"
    print(f"      N={len(out):,}, mean={out.mean():.1f}, median={np.nanmedian(out):.1f}")
    return out


def build_72h_fluid_balance(
    state_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
) -> pd.Series:
    """Outcome 3: 72h net fluid balance.

    72h post-T0 = bin index 36 (72h / 2h per bin = 36 bins post-T0).
    But state tensors only cover 12 bins (24h). Use the last available
    cumulative fluid balance as proxy.
    """
    print("  [3] 72h fluid balance ...")

    fb_col = None
    for candidate in [
        "cumulative_fluid_balance_L",
        "fluid_balance_cumulative_L",
        "cum_fluid_balance_L",
    ]:
        if candidate in state_df.columns:
            fb_col = candidate
            break

    if fb_col is None:
        raise KeyError(
            "No cumulative fluid balance column found in state tensors. "
            f"Available: {[c for c in state_df.columns if 'fluid' in c.lower()]}"
        )

    # State tensors have bins 0-11 (24h post-T0). 72h data is not directly
    # available. Use end-of-window (bin 11) cumulative fluid balance as proxy.
    last_bin = state_df.groupby("stay_id")[fb_col].last()
    # Merge with outcomes to ensure alignment
    stay_ids = outcomes_df["stay_id"].values
    out = last_bin.reindex(stay_ids)
    out.index.name = "stay_id"
    out.name = "fluid_balance_24h_L"  # honest label: only 24h available
    print(f"      N={out.notna().sum():,} (of {len(out):,}), "
          f"mean={out.mean():.2f} L, median={out.median():.2f} L")
    print("      NOTE: using 24h (end-of-window) as proxy for 72h "
          "(72h data not in state tensors)")
    return out


def build_imv_escalation(
    vent_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
) -> pd.Series:
    """Outcome 4: 72h intubation/IMV escalation.

    IMV escalation = any IMV present after T0 but NOT at T0.
    Uses indicator_type == 'mech_vent' or 'vent_mode' from 10h.
    """
    print("  [4] IMV escalation ...")

    stay_ids = outcomes_df["stay_id"].values

    # Filter to mechanical ventilation indicators
    imv_types = {"mech_vent", "vent_mode"}
    avail_types = set(vent_df["indicator_type"].unique()) if "indicator_type" in vent_df.columns else set()
    target_types = imv_types & avail_types
    if not target_types:
        # Fallback: use any vent indicator
        print("      WARNING: no mech_vent/vent_mode found, using all vent indicators")
        vent_sub = vent_df.copy()
    else:
        vent_sub = vent_df[vent_df["indicator_type"].isin(target_types)].copy()

    if "hours_since_icu_admit" not in vent_sub.columns:
        raise KeyError("hours_since_icu_admit not in vent indicators")

    # T0 = 6h post-admission; 72h window = up to 78h post-admission
    t0_h = T0_HOURS
    window_end_h = T0_HOURS + 72.0

    escalation = pd.Series(0, index=stay_ids, name="imv_escalation")
    escalation.index.name = "stay_id"

    for sid in tqdm(stay_ids, desc="      IMV escalation", mininterval=2.0):
        pat = vent_sub[vent_sub["stay_id"] == sid]
        if len(pat) == 0:
            continue
        # Check if IMV was present at T0 (within +-1h of T0 for tolerance)
        at_t0 = pat[
            (pat["hours_since_icu_admit"] >= t0_h - 1.0)
            & (pat["hours_since_icu_admit"] <= t0_h + 1.0)
        ]
        # Check if IMV was present after T0 (up to 72h post-T0)
        after_t0 = pat[
            (pat["hours_since_icu_admit"] > t0_h + 1.0)
            & (pat["hours_since_icu_admit"] <= window_end_h)
        ]
        if len(at_t0) == 0 and len(after_t0) > 0:
            escalation.loc[sid] = 1

    n_esc = int(escalation.sum())
    print(f"      N={len(escalation):,}, escalation={n_esc} "
          f"({n_esc / len(escalation) * 100:.1f}%)")
    return escalation


def build_vfd_28(
    vent_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
) -> pd.Series:
    """Outcome 5: 28-day ventilator-free days (VFD-28).

    Count days (out of 28) free from invasive mechanical ventilation.
    If patient died within 28 days, VFD=0.
    """
    print("  [5] 28-day ventilator-free days ...")

    stay_ids = outcomes_df["stay_id"].values

    # Filter to mechanical ventilation indicators
    imv_types = {"mech_vent", "vent_mode"}
    avail_types = set(vent_df["indicator_type"].unique()) if "indicator_type" in vent_df.columns else set()
    target_types = imv_types & avail_types
    if not target_types:
        print("      WARNING: no mech_vent/vent_mode found, using all vent indicators")
        vent_sub = vent_df.copy()
    else:
        vent_sub = vent_df[vent_df["indicator_type"].isin(target_types)].copy()

    if "hours_since_icu_admit" not in vent_sub.columns:
        raise KeyError("hours_since_icu_admit not in vent indicators")

    # For each stay, find which days (out of 28 from ICU admission) had IMV
    vfd = pd.Series(28.0, index=stay_ids, name="vfd_28")
    vfd.index.name = "stay_id"

    # Group vent data by stay_id for faster lookup
    vent_grouped = vent_sub.groupby("stay_id")["hours_since_icu_admit"].apply(list).to_dict()

    for sid in tqdm(stay_ids, desc="      VFD-28", mininterval=2.0):
        hours_list = vent_grouped.get(sid, [])
        if not hours_list:
            continue
        # Convert hours to days, find unique days with IMV
        vent_days = set()
        for h in hours_list:
            if 0 <= h <= 28 * 24:
                vent_days.add(int(h / 24))
        vfd.loc[sid] = max(0.0, 28.0 - len(vent_days))

    # If patient died within 28 days, VFD=0
    if "mortality_28d" in outcomes_df.columns:
        died_mask = outcomes_df.set_index("stay_id")["mortality_28d"].reindex(stay_ids) == 1
        vfd[died_mask.values] = 0.0

    print(f"      N={len(vfd):,}, mean={vfd.mean():.1f}, median={np.nanmedian(vfd):.1f}")
    return vfd


def build_aki_progression(
    labs_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
) -> pd.Series:
    """Outcome 6: AKI progression (simplified KDIGO stage 1+).

    Baseline = last pre-T0 creatinine.
    Post-T0 max creatinine.
    AKI progression = post-T0 max >= 1.5x baseline OR >= baseline + 0.3 mg/dL.
    """
    print("  [6] AKI progression ...")

    stay_ids = outcomes_df["stay_id"].values

    # Filter labs to creatinine
    lab_col = "lab_name" if "lab_name" in labs_df.columns else "label"
    creat = labs_df[labs_df[lab_col].str.lower().str.contains("creatinine", na=False)].copy()
    print(f"      Creatinine records: {len(creat):,}")

    if len(creat) == 0:
        raise ValueError("No creatinine records found in labs")

    if "hours_since_icu_admit" not in creat.columns:
        raise KeyError("hours_since_icu_admit not in labs")

    val_col = "valuenum" if "valuenum" in creat.columns else "value"

    aki = pd.Series(np.nan, index=stay_ids, name="aki_progression")
    aki.index.name = "stay_id"

    # Group by stay_id for faster lookup
    creat_grouped = creat.groupby("stay_id")

    n_with_data = 0
    for sid in tqdm(stay_ids, desc="      AKI progression", mininterval=2.0):
        try:
            pat = creat_grouped.get_group(sid)
        except KeyError:
            continue

        pre_t0 = pat[pat["hours_since_icu_admit"] <= T0_HOURS]
        post_t0 = pat[pat["hours_since_icu_admit"] > T0_HOURS]

        if len(pre_t0) == 0 or len(post_t0) == 0:
            continue

        baseline = pre_t0[val_col].iloc[-1]  # last pre-T0 value
        post_max = post_t0[val_col].max()

        if pd.isna(baseline) or pd.isna(post_max) or baseline <= 0:
            continue

        n_with_data += 1
        # Simplified KDIGO stage 1+
        if post_max >= 1.5 * baseline or post_max >= baseline + 0.3:
            aki.loc[sid] = 1.0
        else:
            aki.loc[sid] = 0.0

    n_aki = int(aki.sum()) if aki.notna().any() else 0
    n_valid = int(aki.notna().sum())
    print(f"      N with paired creatinine={n_valid:,}, "
          f"AKI={n_aki} ({n_aki / max(n_valid, 1) * 100:.1f}%)")
    return aki


# ══════════════════════════════════════════════════════════════════════════════
# IPTW / Concordance analysis
# ══════════════════════════════════════════════════════════════════════════════

def compute_rule_concordance(rule_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-patient rule concordance rate.

    rule_df has columns: stay_id, bin, action_idx, rule_policy_action,
    rule_agrees_with_observed.

    Returns DataFrame with stay_id, concordance_rate, concordance_tertile.
    """
    conc = rule_df.groupby("stay_id")["rule_agrees_with_observed"].mean()
    conc = conc.reset_index()
    conc.columns = ["stay_id", "concordance_rate"]

    # Tertile split
    q33 = conc["concordance_rate"].quantile(1 / 3)
    q67 = conc["concordance_rate"].quantile(2 / 3)
    conc["concordance_tertile"] = pd.cut(
        conc["concordance_rate"],
        bins=[-np.inf, q33, q67, np.inf],
        labels=["bottom", "middle", "top"],
    )
    return conc


def compute_wis_outcome(
    rule_df: pd.DataFrame,
    prop_df: pd.DataFrame,
    outcome: pd.Series,
) -> dict:
    """Compute WIS-weighted outcome under rule policy.

    Simplified trajectory-level WIS:
    For each patient, w_i = product over bins of pi_rule(a_t|s_t) / pi_b(a_t|s_t)
    where pi_rule = 1 if action == rule action, 0 otherwise (deterministic).
    So w_i = product of 1/pi_b(a_t|s_t) for bins where observed == rule,
    and w_i = 0 otherwise (rule doesn't match observed for at least one bin).

    For practical WIS, we use the concordance-weighted approach:
    weight = mean propensity ratio across bins (softer than strict product).
    """
    # Merge rule actions with propensities
    merged = rule_df[["stay_id", "bin", "action_idx", "rule_policy_action"]].merge(
        prop_df[["stay_id", "bin", "propensity_clipped"]],
        on=["stay_id", "bin"],
        how="inner",
    )

    # For concordant bins: weight contribution = 1/propensity
    # For discordant bins: weight contribution = 0 (strict) or small (soft)
    merged["agrees"] = (merged["action_idx"] == merged["rule_policy_action"]).astype(float)
    merged["bin_weight"] = merged["agrees"] / merged["propensity_clipped"]

    # Per-patient: mean weight (soft WIS approach)
    patient_weights = merged.groupby("stay_id")["bin_weight"].mean()

    # Align with outcome
    common_ids = outcome.index.intersection(patient_weights.index)
    if len(common_ids) == 0:
        return {"wis_mean": np.nan, "wis_ci_lo": np.nan, "wis_ci_hi": np.nan,
                "n_wis": 0}

    w = patient_weights.loc[common_ids].values
    y = outcome.loc[common_ids].values

    # Normalize weights
    valid = np.isfinite(w) & np.isfinite(y)
    w, y = w[valid], y[valid]
    if len(w) == 0 or w.sum() == 0:
        return {"wis_mean": np.nan, "wis_ci_lo": np.nan, "wis_ci_hi": np.nan,
                "n_wis": 0}

    w_norm = w / w.sum()
    wis_mean = float(np.sum(w_norm * y))

    # Bootstrap CI for WIS
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    n = len(w)
    boot_wis = np.empty(BOOTSTRAP_B)
    for b in range(BOOTSTRAP_B):
        idx = rng.randint(0, n, size=n)
        wb = w[idx]
        yb = y[idx]
        if wb.sum() > 0:
            boot_wis[b] = np.sum((wb / wb.sum()) * yb)
        else:
            boot_wis[b] = np.nan
    lo = float(np.nanpercentile(boot_wis, 2.5))
    hi = float(np.nanpercentile(boot_wis, 97.5))

    return {"wis_mean": wis_mean, "wis_ci_lo": lo, "wis_ci_hi": hi,
            "n_wis": int(valid.sum())}


def analyze_outcome(
    outcome_name: str,
    outcome: pd.Series,
    concordance_df: pd.DataFrame,
    rule_df: pd.DataFrame,
    prop_df: pd.DataFrame | None,
    labels_df: pd.DataFrame,
) -> list[dict]:
    """Run concordance + WIS analysis for one outcome, stratified."""
    results = []

    # Merge concordance info with outcome
    conc = concordance_df.set_index("stay_id")
    labels = labels_df.set_index("stay_id") if "stay_id" in labels_df.columns else labels_df

    # Define subgroups
    subgroups = {"overall": None}
    hf_col = None
    for candidate in ["hf_subtype", "comparator_group"]:
        if candidate in labels.columns:
            hf_col = candidate
            break

    if hf_col:
        if "HFpEF" in labels[hf_col].values:
            subgroups["HFpEF"] = labels[labels[hf_col] == "HFpEF"].index
        if "no_hf" in labels[hf_col].values:
            subgroups["no_hf"] = labels[labels[hf_col] == "no_hf"].index
        # Also try no_HF / No_HF variations
        for variant in ["no_HF", "No_HF", "noHF", "non_hf"]:
            if variant in labels[hf_col].values and "no_hf" not in subgroups:
                subgroups["no_hf"] = labels[labels[hf_col] == variant].index

    for subgroup_name, subgroup_ids in tqdm(
        subgroups.items(), desc=f"    {outcome_name} subgroups", leave=False
    ):
        # Filter outcome to subgroup
        if subgroup_ids is not None:
            valid_ids = outcome.index.intersection(subgroup_ids)
            sub_outcome = outcome.loc[valid_ids]
        else:
            sub_outcome = outcome

        # Drop NaN outcomes
        sub_outcome = sub_outcome.dropna()
        if len(sub_outcome) == 0:
            results.append({
                "outcome": outcome_name,
                "subgroup": subgroup_name,
                "n": 0,
                "note": "no valid data",
            })
            continue

        # Overall mean
        overall_mean, overall_lo, overall_hi = _bootstrap_ci(sub_outcome.values)

        # Concordance stratification
        sub_conc = conc.reindex(sub_outcome.index).dropna(subset=["concordance_tertile"])
        top_ids = sub_conc[sub_conc["concordance_tertile"] == "top"].index
        bot_ids = sub_conc[sub_conc["concordance_tertile"] == "bottom"].index

        top_vals = sub_outcome.reindex(top_ids).dropna().values
        bot_vals = sub_outcome.reindex(bot_ids).dropna().values

        top_mean, top_lo, top_hi = _bootstrap_ci(top_vals)
        bot_mean, bot_lo, bot_hi = _bootstrap_ci(bot_vals)
        diff, diff_lo, diff_hi = _bootstrap_diff_ci(top_vals, bot_vals)

        row = {
            "outcome": outcome_name,
            "subgroup": subgroup_name,
            "n": len(sub_outcome),
            "overall_mean": round(overall_mean, 4),
            "overall_ci_lo": round(overall_lo, 4),
            "overall_ci_hi": round(overall_hi, 4),
            "n_top_tertile": len(top_vals),
            "n_bot_tertile": len(bot_vals),
            "concordant_mean": round(top_mean, 4) if not np.isnan(top_mean) else np.nan,
            "concordant_ci_lo": round(top_lo, 4) if not np.isnan(top_lo) else np.nan,
            "concordant_ci_hi": round(top_hi, 4) if not np.isnan(top_hi) else np.nan,
            "discordant_mean": round(bot_mean, 4) if not np.isnan(bot_mean) else np.nan,
            "discordant_ci_lo": round(bot_lo, 4) if not np.isnan(bot_lo) else np.nan,
            "discordant_ci_hi": round(bot_hi, 4) if not np.isnan(bot_hi) else np.nan,
            "diff_conc_disc": round(diff, 4) if not np.isnan(diff) else np.nan,
            "diff_ci_lo": round(diff_lo, 4) if not np.isnan(diff_lo) else np.nan,
            "diff_ci_hi": round(diff_hi, 4) if not np.isnan(diff_hi) else np.nan,
        }

        # WIS if propensity available
        if prop_df is not None:
            if subgroup_ids is not None:
                sub_rule = rule_df[rule_df["stay_id"].isin(valid_ids)]
                sub_prop = prop_df[prop_df["stay_id"].isin(valid_ids)]
            else:
                sub_rule = rule_df
                sub_prop = prop_df
            wis_result = compute_wis_outcome(sub_rule, sub_prop, sub_outcome)
            row.update({
                "wis_mean": round(wis_result["wis_mean"], 4)
                    if not np.isnan(wis_result["wis_mean"]) else np.nan,
                "wis_ci_lo": round(wis_result["wis_ci_lo"], 4)
                    if not np.isnan(wis_result["wis_ci_lo"]) else np.nan,
                "wis_ci_hi": round(wis_result["wis_ci_hi"], 4)
                    if not np.isnan(wis_result["wis_ci_hi"]) else np.nan,
                "n_wis": wis_result["n_wis"],
            })
        else:
            row.update({
                "wis_mean": np.nan, "wis_ci_lo": np.nan,
                "wis_ci_hi": np.nan, "n_wis": 0,
            })

        results.append(row)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ts_start = datetime.datetime.now()
    cfg = load_config()
    interim_dir = resolve_path(cfg, "interim")
    staging_dir = resolve_path(cfg, "staging")
    processed_dir = resolve_path(cfg, "processed")
    tables_dir = resolve_path(cfg, "tables")
    reports_dir = resolve_path(cfg, "reports")
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("P1-secondary — Secondary Outcomes IPTW / Concordance Analysis")
    print("=" * 72)
    print(f"Started: {ts_start.strftime('%Y-%m-%d %H:%M:%S')}")

    # ══════════════════════════════════════════════════════════════════════
    # 1. Load shared data
    # ══════════════════════════════════════════════════════════════════════
    print("\n[1/5] Loading shared data ...")

    # 1a. Outcomes (45d)
    outcomes_path = interim_dir / "45d_outcomes_absorb_v1_mimic.parquet"
    print(f"  Loading outcomes: {outcomes_path}")
    outcomes_df = pd.read_parquet(outcomes_path)
    n_stays = len(outcomes_df)
    print(f"  Stays: {n_stays:,}")
    print(f"  Outcome columns: {list(outcomes_df.columns)}")

    # 1b. Rule policy actions (70c)
    rule_path = interim_dir / "70c_rule_policy_actions_v1_mimic.parquet"
    print(f"  Loading rule policy: {rule_path}")
    rule_df = pd.read_parquet(rule_path)
    print(f"  Rule policy: {len(rule_df):,} obs")
    # Columns: stay_id, bin, action_idx, rule_policy_action, rule_agrees_with_observed

    # 1c. Propensity (70b) — try multiple paths
    prop_df = None
    prop_candidates = [
        interim_dir / "70b_propensity_coarse3_v1_mimic.parquet",
        interim_dir / "70b_propensity_coarse2_v1_mimic.parquet",
    ]
    for pp in prop_candidates:
        if pp.exists():
            print(f"  Loading propensity: {pp}")
            prop_df = pd.read_parquet(pp)
            # Ensure propensity_clipped exists
            if "propensity_clipped" not in prop_df.columns and "propensity" in prop_df.columns:
                prop_df["propensity_clipped"] = prop_df["propensity"].clip(lower=0.005)
            print(f"  Propensity: {len(prop_df):,} obs")
            break
    if prop_df is None:
        print("  WARNING: No propensity file found — WIS analysis will be skipped")

    # 1d. Phenotype labels
    labels_path = processed_dir / "phenotype_labels_frozen_v1_mimic.parquet"
    print(f"  Loading phenotype labels: {labels_path}")
    labels_df = pd.read_parquet(labels_path)
    print(f"  Labels: {len(labels_df):,} stays")
    hf_col = None
    for candidate in ["hf_subtype", "comparator_group"]:
        if candidate in labels_df.columns:
            hf_col = candidate
            print(f"  HF subtype column: {hf_col}")
            print(f"  Distribution: {labels_df[hf_col].value_counts().to_dict()}")
            break

    # ══════════════════════════════════════════════════════════════════════
    # 2. Compute rule concordance
    # ══════════════════════════════════════════════════════════════════════
    print("\n[2/5] Computing rule concordance ...")
    concordance_df = compute_rule_concordance(rule_df)
    print(f"  Concordance rates: mean={concordance_df['concordance_rate'].mean():.4f}, "
          f"median={concordance_df['concordance_rate'].median():.4f}")
    tertile_counts = concordance_df["concordance_tertile"].value_counts()
    for t, c in tertile_counts.items():
        print(f"    {t}: {c:,}")

    # ══════════════════════════════════════════════════════════════════════
    # 3. Construct secondary outcomes
    # ══════════════════════════════════════════════════════════════════════
    print("\n[3/5] Constructing secondary outcomes ...")

    outcomes = {}

    # Outcome 1: 7-day mortality
    try:
        outcomes["mortality_7d"] = build_mortality_7d(outcomes_df)
    except Exception as e:
        print(f"  SKIPPED mortality_7d: {e}")
        traceback.print_exc()

    # Outcome 2: ICU-free days
    try:
        outcomes["icu_free_days"] = build_icu_free_days(outcomes_df)
    except Exception as e:
        print(f"  SKIPPED icu_free_days: {e}")
        traceback.print_exc()

    # Outcome 3: 72h fluid balance (using 24h proxy)
    try:
        state_path = interim_dir / "45c_state_tensors_v1_mimic.parquet"
        print(f"  Loading state tensors: {state_path}")
        state_df = pd.read_parquet(state_path)
        outcomes["fluid_balance_24h_L"] = build_72h_fluid_balance(state_df, outcomes_df)
        del state_df  # free memory
    except Exception as e:
        print(f"  SKIPPED fluid_balance: {e}")
        traceback.print_exc()

    # Load vent data once for outcomes 4 and 5
    vent_df = None
    vent_path = staging_dir / "10h_vent_indicators_mimic.parquet"
    if vent_path.exists():
        try:
            print(f"  Loading vent indicators: {vent_path}")
            vent_df = pd.read_parquet(vent_path)
        except Exception as e:
            print(f"  WARNING: Could not load vent indicators: {e}")

    # Outcome 4: IMV escalation
    try:
        if vent_df is None:
            raise FileNotFoundError(f"Vent indicators not available: {vent_path}")
        outcomes["imv_escalation"] = build_imv_escalation(vent_df, outcomes_df)
    except Exception as e:
        print(f"  SKIPPED imv_escalation: {e}")
        traceback.print_exc()

    # Outcome 5: VFD-28
    try:
        if vent_df is None:
            raise FileNotFoundError(f"Vent indicators not available: {vent_path}")
        outcomes["vfd_28"] = build_vfd_28(vent_df, outcomes_df)
    except Exception as e:
        print(f"  SKIPPED vfd_28: {e}")
        traceback.print_exc()

    # Outcome 6: AKI progression
    try:
        labs_path = staging_dir / "10e_labs_long_mimic.parquet"
        if not labs_path.exists():
            # Try alternative name
            labs_path = staging_dir / "10e_labs_mimic.parquet"
        print(f"  Loading labs: {labs_path}")
        labs_df = pd.read_parquet(labs_path)
        outcomes["aki_progression"] = build_aki_progression(labs_df, outcomes_df)
        del labs_df
    except Exception as e:
        print(f"  SKIPPED aki_progression: {e}")
        traceback.print_exc()

    print(f"\n  Successfully constructed {len(outcomes)} / 6 outcomes: "
          f"{list(outcomes.keys())}")

    if len(outcomes) == 0:
        print("\nERROR: No outcomes could be constructed. Exiting.")
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════════════
    # 4. Analyze each outcome
    # ══════════════════════════════════════════════════════════════════════
    print("\n[4/5] Analyzing outcomes (concordance + WIS) ...")

    all_results = []
    for outcome_name, outcome_series in tqdm(
        outcomes.items(), desc="  Outcomes", leave=True
    ):
        print(f"\n  --- {outcome_name} ---")
        try:
            result_rows = analyze_outcome(
                outcome_name=outcome_name,
                outcome=outcome_series,
                concordance_df=concordance_df,
                rule_df=rule_df,
                prop_df=prop_df,
                labels_df=labels_df,
            )
            all_results.extend(result_rows)
        except Exception as e:
            print(f"  ERROR analyzing {outcome_name}: {e}")
            traceback.print_exc()
            all_results.append({
                "outcome": outcome_name,
                "subgroup": "overall",
                "note": f"analysis failed: {e}",
            })

    # ══════════════════════════════════════════════════════════════════════
    # 5. Save results
    # ══════════════════════════════════════════════════════════════════════
    print("\n[5/5] Saving results ...")

    results_df = pd.DataFrame(all_results)
    csv_path = tables_dir / "P1_secondary_outcomes_summary.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # ── Text report ──
    ts_end = datetime.datetime.now()
    report_lines = [
        "=" * 72,
        "P1-secondary — Secondary Outcomes IPTW / Concordance Analysis",
        "=" * 72,
        f"Generated: {ts_end.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Runtime: {(ts_end - ts_start).total_seconds():.0f}s",
        f"Bootstrap iterations: {BOOTSTRAP_B}",
        f"Stays: {n_stays:,}",
        "",
        "RULE CONCORDANCE",
        "─────────────────",
        f"  Mean concordance rate: {concordance_df['concordance_rate'].mean():.4f}",
        f"  Median:                {concordance_df['concordance_rate'].median():.4f}",
    ]
    for t, c in tertile_counts.items():
        report_lines.append(f"  {t}: {c:,}")

    report_lines.extend(["", "OUTCOMES CONSTRUCTED", "────────────────────"])
    for oname in outcomes:
        s = outcomes[oname]
        n_valid = int(s.notna().sum())
        report_lines.append(f"  {oname}: N={n_valid:,}, mean={s.mean():.4f}")

    skipped = set(
        ["mortality_7d", "icu_free_days", "fluid_balance_24h_L",
         "imv_escalation", "vfd_28", "aki_progression"]
    ) - set(outcomes.keys())
    if skipped:
        report_lines.extend(["", "OUTCOMES SKIPPED (data not available)",
                             "─────────────────────────────────────"])
        for s in sorted(skipped):
            report_lines.append(f"  {s}: N/A")

    report_lines.extend(["", "RESULTS BY OUTCOME AND SUBGROUP",
                         "────────────────────────────────"])

    for _, row in results_df.iterrows():
        oname = row.get("outcome", "?")
        sg = row.get("subgroup", "?")
        n = row.get("n", 0)

        if "note" in row and pd.notna(row.get("note")):
            report_lines.append(f"\n  {oname} [{sg}]: {row['note']}")
            continue

        report_lines.append(f"\n  {oname} [{sg}] (N={n})")

        if pd.notna(row.get("overall_mean")):
            report_lines.append(
                f"    Overall:    {row['overall_mean']:.4f} "
                f"[{row.get('overall_ci_lo', np.nan):.4f}, "
                f"{row.get('overall_ci_hi', np.nan):.4f}]"
            )

        if pd.notna(row.get("concordant_mean")):
            report_lines.append(
                f"    Concordant (top tertile, n={row.get('n_top_tertile', '?')}): "
                f"{row['concordant_mean']:.4f} "
                f"[{row.get('concordant_ci_lo', np.nan):.4f}, "
                f"{row.get('concordant_ci_hi', np.nan):.4f}]"
            )

        if pd.notna(row.get("discordant_mean")):
            report_lines.append(
                f"    Discordant (bot tertile, n={row.get('n_bot_tertile', '?')}): "
                f"{row['discordant_mean']:.4f} "
                f"[{row.get('discordant_ci_lo', np.nan):.4f}, "
                f"{row.get('discordant_ci_hi', np.nan):.4f}]"
            )

        if pd.notna(row.get("diff_conc_disc")):
            report_lines.append(
                f"    Diff (conc-disc): {row['diff_conc_disc']:.4f} "
                f"[{row.get('diff_ci_lo', np.nan):.4f}, "
                f"{row.get('diff_ci_hi', np.nan):.4f}]"
            )

        if pd.notna(row.get("wis_mean")):
            report_lines.append(
                f"    WIS (rule policy): {row['wis_mean']:.4f} "
                f"[{row.get('wis_ci_lo', np.nan):.4f}, "
                f"{row.get('wis_ci_hi', np.nan):.4f}] "
                f"(n_wis={row.get('n_wis', '?')})"
            )

    report_lines.extend([
        "",
        "INTERPRETATION NOTES",
        "────────────────────",
        "  - Concordant = patients whose observed treatment most closely matched",
        "    the rule policy (top tertile of concordance rate).",
        "  - Discordant = bottom tertile of concordance rate.",
        "  - Diff (conc-disc): positive = concordant group has higher outcome value.",
        "    For mortality/AKI: negative diff = concordant group did better.",
        "    For ICU-free days/VFD: positive diff = concordant group did better.",
        "  - WIS: weighted-importance-sampling estimate of outcome under rule policy.",
        "  - fluid_balance_24h_L: uses end-of-24h-window value as proxy for 72h",
        "    (72h data not available in 12-bin state tensors).",
        "  - Causal interpretation requires caution: concordance is observational.",
    ])

    report = "\n".join(report_lines) + "\n"
    report_path = reports_dir / "P1_secondary_outcomes_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {report_path}")
    print(report)

    print(f"\n[P1-secondary] COMPLETE. "
          f"Runtime: {(ts_end - ts_start).total_seconds():.0f}s")


if __name__ == "__main__":
    main()
