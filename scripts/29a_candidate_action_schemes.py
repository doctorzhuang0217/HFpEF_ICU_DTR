#!/usr/bin/env python3
"""
29a_candidate_action_schemes — Define and evaluate candidate action-space
discretization schemes for the DTR analysis.

Purpose
-------
Define 2-3 candidate action-space schemes with explicit bin cutpoints for
three treatment dimensions (IV fluids, vasopressors, diuretics) on the 2h
decision grid. Produce empirical distribution summaries from the frozen
cohort to validate that bins are clinically meaningful and have adequate
support.

Protocol requirement
--------------------
"Before Phase 0.5, run at least 2-3 candidate action-space schemes.
 Do not interpret positivity failure under one arbitrarily fine action space
 as a global no-go." (locked decision §2.6)

Treatment dimensions
--------------------
1. IV fluid volume (mL per 2h bin)
2. Vasopressor dose (NE-equivalent mcg/kg/min)
3. IV loop diuretic dose (furosemide-equivalent mg per 2h bin)

Inputs
------
- data/staging/25f_primary_cohort_candidate_v1_mimic.parquet
- data/staging/10f_meds_infusions_mimic.parquet
- data/staging/10c_timestamps_mimic.parquet

Outputs
-------
- configs/29a_candidate_action_schemes.yaml
- reports/29a_candidate_action_schemes_summary.txt
- tables/29a_action_marginal_distributions.csv
"""

from __future__ import annotations

import sys
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path

# ══════════════════════════════════════════════════════════════════════════
# NE-equivalent conversion factors (mcg/kg/min → NE-equivalent mcg/kg/min)
# Standard references: Overgaard & Džavík 2008; Jentzer et al. 2018
# These will be versioned in 40e; used here for feasibility assessment only.
# ══════════════════════════════════════════════════════════════════════════
NE_EQUIVALENT_FACTORS = {
    "norepinephrine": 1.0,
    "epinephrine":    1.0,      # roughly equivalent at low-moderate doses
    "phenylephrine":  0.1,      # 10:1 potency ratio
    "dopamine":       0.01,     # approximate at vasopressor-range doses (>5 mcg/kg/min)
    "vasopressin":    2.5,      # 0.04 U/min ≈ 0.1 mcg/kg/min NE (for ~80kg); use 2.5 as heuristic
    "dobutamine":     0.0,      # inotrope — excluded from vasopressor dose
    "milrinone":      0.0,      # inotrope — excluded from vasopressor dose
}

# Furosemide-equivalent conversion factors
FURO_EQUIVALENT_FACTORS = {
    "furosemide":  1.0,
    "bumetanide":  40.0,     # 1 mg bumetanide ≈ 40 mg furosemide
    "torsemide":   2.0,      # 20 mg torsemide ≈ 40 mg furosemide (so factor = 2.0)
}

# ── Decision grid parameters ─────────────────────────────────────────────
GRID_BIN_HOURS = 2
PRIMARY_HORIZON_HOURS = 24
T0_HOURS = 6   # time-zero = ICU admission + 6h
N_GRID_BINS = PRIMARY_HORIZON_HOURS // GRID_BIN_HOURS  # = 12

# ══════════════════════════════════════════════════════════════════════════
# Candidate action-space schemes
# ══════════════════════════════════════════════════════════════════════════
# Scheme A: "Coarse-3" — 3 bins per dimension → 3×3×2=18 joint cells
# Scheme B: "Medium-4" — 4 bins for fluid/pressor, 2 for diuretic → 4×4×2=32 cells
# Scheme C: "Fine-5"   — 5 bins for fluid, 4 for pressor, 3 for diuretic → 5×4×3=60 cells

ACTION_SCHEMES = {
    "coarse_3": {
        "description": "Minimal granularity for maximum support; 3×3×2 = 18 joint cells",
        "iv_fluid_ml_per_2h": {
            "bins": [0, 1, 250, float("inf")],
            "labels": ["none", "low_moderate", "high"],
            "note": "0=none, >0 to <250=low/moderate, >=250=high (bolus-range)"
        },
        "vasopressor_ne_eq_mcg_kg_min": {
            "bins": [0, 0.001, 0.1, float("inf")],
            "labels": ["none", "low", "high"],
            "note": "0=none, >0 to <0.1=low, >=0.1=high"
        },
        "diuretic_furo_eq_mg_per_2h": {
            "bins": [0, 0.001, float("inf")],
            "labels": ["none", "any"],
            "note": "Binary: none vs any exposure"
        },
        "joint_cells": 18
    },
    "medium_4": {
        "description": "Moderate granularity balancing interpretability and support; 4×4×2 = 32 joint cells",
        "iv_fluid_ml_per_2h": {
            "bins": [0, 1, 100, 500, float("inf")],
            "labels": ["none", "minimal", "moderate", "aggressive"],
            "note": "0=none, 1-99=minimal/maintenance, 100-499=moderate, >=500=aggressive"
        },
        "vasopressor_ne_eq_mcg_kg_min": {
            "bins": [0, 0.001, 0.05, 0.2, float("inf")],
            "labels": ["none", "low", "moderate", "high"],
            "note": "0=none, >0 to <0.05=low, 0.05-0.2=moderate, >0.2=high"
        },
        "diuretic_furo_eq_mg_per_2h": {
            "bins": [0, 0.001, float("inf")],
            "labels": ["none", "any"],
            "note": "Binary: none vs any. 2-bin for Phase 0.5 stability."
        },
        "joint_cells": 32
    },
    "fine_5": {
        "description": "Maximum clinical interpretability; 5×4×3 = 60 joint cells — may have support issues",
        "iv_fluid_ml_per_2h": {
            "bins": [0, 1, 50, 250, 1000, float("inf")],
            "labels": ["none", "kvo", "low", "moderate_bolus", "high_resuscitation"],
            "note": "0=none, 1-49=KVO/maintenance, 50-249=low, 250-999=moderate bolus, >=1000=resuscitation"
        },
        "vasopressor_ne_eq_mcg_kg_min": {
            "bins": [0, 0.001, 0.05, 0.2, float("inf")],
            "labels": ["none", "low", "moderate", "high"],
            "note": "Same as medium — 4 pressor bins is clinically standard"
        },
        "diuretic_furo_eq_mg_per_2h": {
            "bins": [0, 0.001, 40, float("inf")],
            "labels": ["none", "low", "high"],
            "note": "0=none, >0 to <40mg=low (<=1 standard dose), >=40mg=high"
        },
        "joint_cells": 60
    },
}


def compute_ne_equivalent(meds: pd.DataFrame, weight: pd.Series) -> pd.DataFrame:
    """Compute NE-equivalent dose for vasopressor/inotrope rows.

    Parameters
    ----------
    meds : DataFrame with columns [stay_id, med_name, rate, rateuom, amount, amountuom]
    weight : Series indexed by stay_id with patient weight in kg

    Returns
    -------
    DataFrame with [stay_id, starttime, endtime, ne_eq_mcg_kg_min]
    """
    vp = meds[meds["med_name"].isin(NE_EQUIVALENT_FACTORS)].copy()
    vp["ne_factor"] = vp["med_name"].map(NE_EQUIVALENT_FACTORS)

    # Rate-based vasopressors: rate is typically in mcg/kg/min or mcg/min
    # For mcg/min, we need to divide by weight
    vp = vp.merge(weight.rename("body_weight_kg"), left_on="stay_id",
                  right_index=True, how="left")
    vp["body_weight_kg"] = vp["body_weight_kg"].fillna(80.0)  # default 80kg

    # Determine if rate is weight-based already
    rate_is_per_kg = vp["rateuom"].str.contains("kg", case=False, na=False)
    vp["rate_mcg_kg_min"] = np.where(
        rate_is_per_kg,
        vp["rate"],
        vp["rate"] / vp["body_weight_kg"]
    )
    # Handle units: some may be mg/hr → convert
    rate_is_mg = vp["rateuom"].str.contains("mg", case=False, na=False)
    vp.loc[rate_is_mg, "rate_mcg_kg_min"] = (
        vp.loc[rate_is_mg, "rate"] * 1000 / 60 / vp.loc[rate_is_mg, "body_weight_kg"]
    )

    # Vasopressin special handling: typically units/min or units/hr
    vaso_mask = vp["med_name"] == "vasopressin"
    rate_is_units_hr = vp["rateuom"].str.contains("units/hour|units/hr|U/hr",
                                                   case=False, na=False)
    rate_is_units_min = vp["rateuom"].str.contains("units/min|U/min",
                                                    case=False, na=False)
    # Convert vasopressin units/hr to units/min
    vp.loc[vaso_mask & rate_is_units_hr, "rate_mcg_kg_min"] = (
        vp.loc[vaso_mask & rate_is_units_hr, "rate"] / 60
    )
    vp.loc[vaso_mask & rate_is_units_min, "rate_mcg_kg_min"] = (
        vp.loc[vaso_mask & rate_is_units_min, "rate"]
    )
    # For vasopressin, factor converts units/min to NE-equivalent
    # 0.04 U/min ≈ 0.1 mcg/kg/min NE → factor = 0.1/0.04 = 2.5
    vp.loc[vaso_mask, "rate_mcg_kg_min"] = (
        vp.loc[vaso_mask, "rate_mcg_kg_min"] * NE_EQUIVALENT_FACTORS["vasopressin"]
    )
    vp.loc[~vaso_mask, "rate_mcg_kg_min"] = (
        vp.loc[~vaso_mask, "rate_mcg_kg_min"] * vp.loc[~vaso_mask, "ne_factor"]
    )

    return vp[["stay_id", "starttime", "endtime", "rate_mcg_kg_min",
               "med_name", "ne_factor"]].rename(
        columns={"rate_mcg_kg_min": "ne_eq_mcg_kg_min"})


def compute_furo_equivalent(meds: pd.DataFrame) -> pd.DataFrame:
    """Compute furosemide-equivalent dose for diuretic rows.

    Returns DataFrame with [stay_id, starttime, amount_furo_eq_mg]
    """
    diur = meds[meds["med_name"].isin(FURO_EQUIVALENT_FACTORS)].copy()
    diur["furo_factor"] = diur["med_name"].map(FURO_EQUIVALENT_FACTORS)
    diur["furo_eq_mg"] = diur["amount"] * diur["furo_factor"]
    return diur[["stay_id", "starttime", "furo_eq_mg", "med_name"]]


def assign_to_grid_bins(events: pd.DataFrame, timestamps: pd.DataFrame,
                        time_col: str = "starttime") -> pd.DataFrame:
    """Assign events to 2h grid bins relative to T0.

    Returns events with added column 'grid_bin' (0-based, 0 = T0 to T0+2h).
    Events before T0 or after horizon are dropped.
    """
    events = events.merge(
        timestamps[["stay_id", "time_zero_6h"]],
        on="stay_id", how="left"
    )
    events["hours_since_t0"] = (
        (pd.to_datetime(events[time_col]) - pd.to_datetime(events["time_zero_6h"]))
        .dt.total_seconds() / 3600
    )
    # Keep only post-T0 within primary horizon
    mask = (events["hours_since_t0"] >= 0) & (events["hours_since_t0"] < PRIMARY_HORIZON_HOURS)
    events = events[mask].copy()
    events["grid_bin"] = (events["hours_since_t0"] // GRID_BIN_HOURS).astype(int)
    return events


def discretize_actions(fluid_per_bin: pd.DataFrame,
                       pressor_per_bin: pd.DataFrame,
                       diuretic_per_bin: pd.DataFrame,
                       scheme: dict) -> pd.DataFrame:
    """Discretize continuous action values into categorical bins per scheme.

    Each input should have columns: [stay_id, grid_bin, <value_col>]
    Returns a merged DataFrame with categorical action labels.
    """
    # Create full grid scaffold
    stay_ids = sorted(set(
        fluid_per_bin["stay_id"].unique().tolist()
        + pressor_per_bin["stay_id"].unique().tolist()
        + diuretic_per_bin["stay_id"].unique().tolist()
    ))
    grid_bins = list(range(N_GRID_BINS))
    scaffold = pd.DataFrame(
        [(s, g) for s in stay_ids for g in grid_bins],
        columns=["stay_id", "grid_bin"]
    )

    # Merge continuous values
    scaffold = scaffold.merge(fluid_per_bin, on=["stay_id", "grid_bin"], how="left")
    scaffold = scaffold.merge(pressor_per_bin, on=["stay_id", "grid_bin"], how="left")
    scaffold = scaffold.merge(diuretic_per_bin, on=["stay_id", "grid_bin"], how="left")

    # Fill NaN with 0 (no treatment in that bin)
    scaffold["iv_fluid_ml"] = scaffold["iv_fluid_ml"].fillna(0)
    scaffold["ne_eq_mcg_kg_min"] = scaffold["ne_eq_mcg_kg_min"].fillna(0)
    scaffold["furo_eq_mg"] = scaffold["furo_eq_mg"].fillna(0)

    # Discretize
    s_fluid = scheme["iv_fluid_ml_per_2h"]
    s_pressor = scheme["vasopressor_ne_eq_mcg_kg_min"]
    s_diuretic = scheme["diuretic_furo_eq_mg_per_2h"]

    scaffold["fluid_cat"] = pd.cut(
        scaffold["iv_fluid_ml"],
        bins=s_fluid["bins"], labels=s_fluid["labels"],
        right=False, include_lowest=True
    )
    scaffold["pressor_cat"] = pd.cut(
        scaffold["ne_eq_mcg_kg_min"],
        bins=s_pressor["bins"], labels=s_pressor["labels"],
        right=False, include_lowest=True
    )
    scaffold["diuretic_cat"] = pd.cut(
        scaffold["furo_eq_mg"],
        bins=s_diuretic["bins"], labels=s_diuretic["labels"],
        right=False, include_lowest=True
    )

    return scaffold


def compute_joint_cell_counts(discretized: pd.DataFrame, cohort: pd.DataFrame,
                              label: str) -> pd.DataFrame:
    """Compute joint action cell counts overall and by HFpEF subgroup."""
    # Merge comparator_group
    discretized = discretized.merge(
        cohort[["stay_id", "comparator_group"]],
        on="stay_id", how="left"
    )

    results = []
    for group_label, group_df in [("all", discretized),
                                  ("hfpef", discretized[discretized["comparator_group"] == "hfpef"])]:
        if len(group_df) == 0:
            continue
        cell_counts = (
            group_df
            .groupby(["fluid_cat", "pressor_cat", "diuretic_cat"])
            .size()
            .reset_index(name="n_observations")
        )
        cell_counts["n_stays"] = (
            group_df
            .groupby(["fluid_cat", "pressor_cat", "diuretic_cat"])["stay_id"]
            .nunique()
            .values
        )
        cell_counts["scheme"] = label
        cell_counts["subgroup"] = group_label
        results.append(cell_counts)

    return pd.concat(results, ignore_index=True)


def main() -> None:
    cfg = load_config()
    staging_dir = resolve_path(cfg, "staging")
    reports_dir = resolve_path(cfg, "reports")
    tables_dir = resolve_path(cfg, "tables")
    configs_dir = resolve_path(cfg, "configs")
    for d in [reports_dir, tables_dir, configs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("[29a] Loading frozen cohort and treatment data...")

    # ── Load data ─────────────────────────────────────────────────────────
    cohort = pd.read_parquet(staging_dir / "25f_primary_cohort_candidate_v1_mimic.parquet")
    meds = pd.read_parquet(staging_dir / "10f_meds_infusions_mimic.parquet")
    timestamps = pd.read_parquet(staging_dir / "10c_timestamps_mimic.parquet")

    # Filter to gate-passed cohort only
    gate_passed = set(cohort[cohort["hemo_gate_passed"] == True]["stay_id"])
    print(f"  Gate-passed stays: {len(gate_passed):,}")

    meds = meds[meds["stay_id"].isin(gate_passed)].copy()
    cohort_gated = cohort[cohort["stay_id"].isin(gate_passed)].copy()

    # ── Get patient weights ───────────────────────────────────────────────
    weight_path = staging_dir / "10g_weight_mimic.parquet"
    if weight_path.exists():
        weight_df = pd.read_parquet(weight_path)
        # Use median weight per stay
        weight = weight_df.groupby("stay_id")["valuenum"].median()
    else:
        weight = pd.Series(dtype=float)
    print(f"  Stays with weight data: {len(weight):,}")

    # ── Compute NE-equivalent vasopressor doses ───────────────────────────
    print("[29a] Computing NE-equivalent doses...")
    vp_meds = meds[meds["med_class"].isin(["vasopressor", "inotrope"])].copy()
    if len(vp_meds) > 0:
        ne_eq = compute_ne_equivalent(vp_meds, weight)
        ne_eq_grid = assign_to_grid_bins(ne_eq, timestamps, time_col="starttime")
        # Aggregate: max NE-eq per stay per grid bin (concurrent infusions sum)
        pressor_per_bin = (
            ne_eq_grid
            .groupby(["stay_id", "grid_bin"])["ne_eq_mcg_kg_min"]
            .sum()  # sum concurrent vasopressors
            .reset_index()
        )
    else:
        pressor_per_bin = pd.DataFrame(columns=["stay_id", "grid_bin", "ne_eq_mcg_kg_min"])
    print(f"  Vasopressor grid-bin observations: {len(pressor_per_bin):,}")

    # ── Compute furosemide-equivalent diuretic doses ──────────────────────
    print("[29a] Computing furosemide-equivalent doses...")
    diur_meds = meds[meds["med_class"] == "loop_diuretic"].copy()
    if len(diur_meds) > 0:
        furo_eq = compute_furo_equivalent(diur_meds)
        furo_eq_grid = assign_to_grid_bins(furo_eq, timestamps, time_col="starttime")
        diuretic_per_bin = (
            furo_eq_grid
            .groupby(["stay_id", "grid_bin"])["furo_eq_mg"]
            .sum()
            .reset_index()
        )
    else:
        diuretic_per_bin = pd.DataFrame(columns=["stay_id", "grid_bin", "furo_eq_mg"])
    print(f"  Diuretic grid-bin observations: {len(diuretic_per_bin):,}")

    # ── Compute IV fluid volumes per grid bin ─────────────────────────────
    print("[29a] Computing IV fluid volumes per grid bin...")
    fluid_meds = meds[meds["med_class"] == "iv_fluid"].copy()
    if len(fluid_meds) > 0:
        fluid_grid = assign_to_grid_bins(fluid_meds, timestamps, time_col="starttime")
        fluid_per_bin = (
            fluid_grid
            .groupby(["stay_id", "grid_bin"])["amount"]
            .sum()
            .reset_index()
            .rename(columns={"amount": "iv_fluid_ml"})
        )
    else:
        fluid_per_bin = pd.DataFrame(columns=["stay_id", "grid_bin", "iv_fluid_ml"])
    print(f"  Fluid grid-bin observations: {len(fluid_per_bin):,}")

    # ── Marginal distribution summaries ───────────────────────────────────
    print("\n[29a] ══ Marginal Distribution Summaries ══")
    marginals = []

    # Fluid marginals
    if len(fluid_per_bin) > 0:
        nonzero_fluid = fluid_per_bin[fluid_per_bin["iv_fluid_ml"] > 0]["iv_fluid_ml"]
        print(f"\n  IV Fluid (mL/2h, non-zero only):")
        print(f"    N observations: {len(nonzero_fluid):,}")
        print(f"    Percentiles: p5={nonzero_fluid.quantile(0.05):.0f}, "
              f"p25={nonzero_fluid.quantile(0.25):.0f}, "
              f"p50={nonzero_fluid.quantile(0.50):.0f}, "
              f"p75={nonzero_fluid.quantile(0.75):.0f}, "
              f"p95={nonzero_fluid.quantile(0.95):.0f}")
        for p in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
            marginals.append({
                "dimension": "iv_fluid_ml_per_2h", "stat": f"p{int(p*100)}",
                "value": nonzero_fluid.quantile(p), "scope": "nonzero"
            })

    # Pressor marginals
    if len(pressor_per_bin) > 0:
        nonzero_pressor = pressor_per_bin[pressor_per_bin["ne_eq_mcg_kg_min"] > 0]["ne_eq_mcg_kg_min"]
        print(f"\n  Vasopressor NE-eq (mcg/kg/min, non-zero only):")
        print(f"    N observations: {len(nonzero_pressor):,}")
        print(f"    Percentiles: p5={nonzero_pressor.quantile(0.05):.4f}, "
              f"p25={nonzero_pressor.quantile(0.25):.4f}, "
              f"p50={nonzero_pressor.quantile(0.50):.4f}, "
              f"p75={nonzero_pressor.quantile(0.75):.4f}, "
              f"p95={nonzero_pressor.quantile(0.95):.4f}")
        for p in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
            marginals.append({
                "dimension": "vasopressor_ne_eq_mcg_kg_min", "stat": f"p{int(p*100)}",
                "value": nonzero_pressor.quantile(p), "scope": "nonzero"
            })

    # Diuretic marginals
    if len(diuretic_per_bin) > 0:
        nonzero_diur = diuretic_per_bin[diuretic_per_bin["furo_eq_mg"] > 0]["furo_eq_mg"]
        print(f"\n  Diuretic furo-eq (mg/2h, non-zero only):")
        print(f"    N observations: {len(nonzero_diur):,}")
        if len(nonzero_diur) > 0:
            print(f"    Percentiles: p5={nonzero_diur.quantile(0.05):.1f}, "
                  f"p25={nonzero_diur.quantile(0.25):.1f}, "
                  f"p50={nonzero_diur.quantile(0.50):.1f}, "
                  f"p75={nonzero_diur.quantile(0.75):.1f}, "
                  f"p95={nonzero_diur.quantile(0.95):.1f}")
            for p in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
                marginals.append({
                    "dimension": "diuretic_furo_eq_mg_per_2h", "stat": f"p{int(p*100)}",
                    "value": nonzero_diur.quantile(p), "scope": "nonzero"
                })

    marginals_df = pd.DataFrame(marginals)

    # ── Evaluate each scheme ──────────────────────────────────────────────
    print("\n[29a] ══ Evaluating Candidate Schemes ══")
    all_cell_counts = []

    for scheme_name, scheme in ACTION_SCHEMES.items():
        print(f"\n  ── Scheme: {scheme_name} ({scheme['description']}) ──")

        discretized = discretize_actions(
            fluid_per_bin, pressor_per_bin, diuretic_per_bin, scheme
        )

        # Cell counts
        cell_counts = compute_joint_cell_counts(discretized, cohort_gated, scheme_name)
        all_cell_counts.append(cell_counts)

        # Summary stats
        for subgroup in ["all", "hfpef"]:
            sub_cells = cell_counts[cell_counts["subgroup"] == subgroup]
            if len(sub_cells) == 0:
                continue
            n_nonempty = (sub_cells["n_observations"] > 0).sum()
            n_total_cells = scheme["joint_cells"]
            min_cell = sub_cells["n_observations"].min()
            min_stay_cell = sub_cells["n_stays"].min()
            below_25 = (sub_cells["n_stays"] < 25).sum()
            below_50 = (sub_cells["n_stays"] < 50).sum()
            below_100 = (sub_cells["n_stays"] < 100).sum()

            print(f"    [{subgroup}] Non-empty cells: {n_nonempty}/{n_total_cells}")
            print(f"    [{subgroup}] Min obs/cell: {min_cell:,}, Min stays/cell: {min_stay_cell:,}")
            print(f"    [{subgroup}] Cells with <25 stays: {below_25}, <50: {below_50}, <100: {below_100}")

    # Combine all cell counts
    cell_counts_all = pd.concat(all_cell_counts, ignore_index=True)

    # ── Save outputs ──────────────────────────────────────────────────────
    # 1. YAML config
    yaml_output = {
        "step": "29a_candidate_action_schemes",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "protocol_version": "v3.1",
        "ne_equivalent_factors": NE_EQUIVALENT_FACTORS,
        "furo_equivalent_factors": FURO_EQUIVALENT_FACTORS,
        "grid": {
            "bin_hours": GRID_BIN_HOURS,
            "primary_horizon_hours": PRIMARY_HORIZON_HOURS,
            "n_bins": N_GRID_BINS,
            "t0_hours": T0_HOURS
        },
        "schemes": {},
    }
    for scheme_name, scheme in ACTION_SCHEMES.items():
        # Convert inf for YAML serialization
        scheme_out = {}
        for key, val in scheme.items():
            if isinstance(val, dict) and "bins" in val:
                val_copy = dict(val)
                val_copy["bins"] = [str(b) if b == float("inf") else b for b in val["bins"]]
                scheme_out[key] = val_copy
            else:
                scheme_out[key] = val
        yaml_output["schemes"][scheme_name] = scheme_out

    yaml_path = configs_dir / "29a_candidate_action_schemes.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_output, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)
    print(f"\n  Saved: {yaml_path}")

    # 2. Cell counts CSV
    csv_path = tables_dir / "29a_action_cell_counts.csv"
    cell_counts_all.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # 3. Marginal distributions CSV
    marg_path = tables_dir / "29a_action_marginal_distributions.csv"
    marginals_df.to_csv(marg_path, index=False)
    print(f"  Saved: {marg_path}")

    # 4. Summary report
    report_lines = [
        "=" * 72,
        "29a — Candidate Action-Space Schemes: Summary Report",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Cohort: gate-passed stays = {len(gate_passed):,}",
        f"HFpEF gate-passed: {len(cohort_gated[cohort_gated['comparator_group'] == 'hfpef']):,}",
        f"Decision grid: {GRID_BIN_HOURS}h bins × {N_GRID_BINS} = {PRIMARY_HORIZON_HOURS}h horizon",
        "",
        "NE-equivalent factors:",
    ]
    for med, factor in NE_EQUIVALENT_FACTORS.items():
        report_lines.append(f"  {med}: {factor}")
    report_lines.append("")
    report_lines.append("Furosemide-equivalent factors:")
    for med, factor in FURO_EQUIVALENT_FACTORS.items():
        report_lines.append(f"  {med}: {factor}")

    report_lines.append("")
    report_lines.append("─" * 72)
    report_lines.append("MARGINAL DISTRIBUTIONS (non-zero observations)")
    report_lines.append("─" * 72)
    report_lines.append(marginals_df.to_string(index=False))

    report_lines.append("")
    report_lines.append("─" * 72)
    report_lines.append("SCHEME EVALUATION SUMMARY")
    report_lines.append("─" * 72)

    for scheme_name in ACTION_SCHEMES:
        scheme = ACTION_SCHEMES[scheme_name]
        report_lines.append(f"\n  Scheme: {scheme_name}")
        report_lines.append(f"  Description: {scheme['description']}")
        for subgroup in ["all", "hfpef"]:
            sub_cells = cell_counts_all[
                (cell_counts_all["scheme"] == scheme_name) &
                (cell_counts_all["subgroup"] == subgroup)
            ]
            if len(sub_cells) == 0:
                report_lines.append(f"    [{subgroup}] No data")
                continue
            n_nonempty = (sub_cells["n_observations"] > 0).sum()
            min_stays = sub_cells["n_stays"].min()
            below_25 = (sub_cells["n_stays"] < 25).sum()
            below_50 = (sub_cells["n_stays"] < 50).sum()
            report_lines.append(
                f"    [{subgroup}] Non-empty: {n_nonempty}/{scheme['joint_cells']}, "
                f"min stays/cell: {min_stays:,}, "
                f"cells <25 stays: {below_25}, <50 stays: {below_50}"
            )

    report_lines.append("")
    report_lines.append("─" * 72)
    report_lines.append("FULL CELL COUNTS BY SCHEME")
    report_lines.append("─" * 72)
    report_lines.append(cell_counts_all.to_string(index=False))

    report_path = reports_dir / "29a_candidate_action_schemes_summary.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Saved: {report_path}")

    print("\n[29a] Done.")


if __name__ == "__main__":
    main()
