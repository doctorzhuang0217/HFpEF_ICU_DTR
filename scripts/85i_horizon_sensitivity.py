#!/usr/bin/env python3
"""
85i_horizon_sensitivity — 48h/72h decision-horizon sensitivity analysis.

Purpose
-------
Compare ATE estimates across 24h (primary), 48h, and 72h decision horizons.
The primary analysis uses a 24h window (12 × 2h bins from T0). This script
rebuilds the entire pipeline for extended horizons (48h = 24 bins, 72h = 36 bins)
and retrains the G-Transformer on each, then compares OPE results.

Pipeline per horizon:
  Phase A — Data preparation:
    A1: Grid builder        (adapted 40b)
    A2: Missingness features (adapted 40c)
    A3: State tensors        (adapted 45c)
    A4: Outcomes & absorbing (adapted 45d)
    A5: Behavior policy      (adapted 70b)
  Phase B — Model training & OPE:
    B1: Train G-Transformer  (adapted 80c)
    B2: OPE evaluation + bootstrap
    B3: Subgroup analysis
  Phase C — Cross-horizon comparison:
    C1: Load 24h primary results
    C2: Compare horizons, compute verdict
    C3: Generate report

Protocol reference: Step 85i (v3.1 § sensitivity battery)
Compute: G2 (GPU for training) + C3 (CPU for data prep + bootstrap)
Pre-specified in protocol v3.1: "48h and 72h are prespecified sensitivities."
Feasibility: 30d LOS — 48h coverage 98.6% (HFpEF 98.1%), 72h 98.3% (97.6%).

Inputs
------
Primary 24h results (loaded, not re-run):
- tables/80c_gtransformer_rollout_ow1_policy_values.csv
- tables/80c_gtransformer_rollout_ow1_bootstrap_cis.csv
- tables/80c_gtransformer_rollout_ow1_subgroup_results.csv

Raw data (rebuilt for each horizon):
- data/staging/25f_primary_cohort_candidate_v1_mimic.parquet
- data/staging/10d_vitals_mimic.parquet
- data/staging/10e_labs_mimic.parquet
- data/staging/10f_meds_infusions_mimic.parquet
- data/staging/10g_output_events_mimic.parquet
- data/staging/10a_core_icu_stays_mimic.parquet
- configs/29a_candidate_action_schemes.yaml
- configs/80_step_c_training_v1.yaml

Outputs
-------
Per horizon (48h, 72h):
- data/interim/85i_grid2h_states_{H}h_mimic.parquet
- data/interim/85i_missingness_features_{H}h_mimic.parquet
- data/interim/85i_state_tensors_{H}h_v1_mimic.parquet
- data/interim/85i_outcomes_absorb_{H}h_v1_mimic.parquet
- data/interim/85i_propensity_{H}h_v1_mimic.parquet
- models/85i_gtransformer_{H}h_best.pt

Comparison:
- tables/85i_horizon_sensitivity.csv
- tables/85i_horizon_subgroup.csv
- reports/85i_horizon_sensitivity_summary.txt

Multi-seed validation (when --seeds is specified):
- tables/85i_horizon_multiseed_per_seed.csv       (per-seed detailed results)
- tables/85i_horizon_multiseed_agg.csv             (mean/std/range across seeds)
- tables/85i_horizon_multiseed_subgroup_per_seed.csv
- tables/85i_horizon_multiseed_subgroup_agg.csv
- reports/85i_horizon_multiseed_summary.txt
Note: original single-seed outputs are NOT overwritten.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hfpef_dtr.config import load_config, resolve_path
from hfpef_dtr.models.dataset import discover_feature_columns, make_train_val_test_split
from hfpef_dtr.models.dtr_dataset import DTRDataset
from hfpef_dtr.models.gtransformer import (
    GTransformer,
    GTransformerLoss,
    build_gtransformer_from_config,
)
from hfpef_dtr.models.ope import bootstrap_ope, evaluate_dm

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "tables"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
INTERIM_DIR = ROOT / "data" / "interim"

for d in [TABLES_DIR, REPORTS_DIR, MODELS_DIR, INTERIM_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Constants ───────────────────────────────────────────────────────────
T0_HOURS = 6
BIN_HOURS = 2
PRIMARY_N_BINS = 12  # 24h reference
SENSITIVITY_HORIZONS = {48: 24, 72: 36}  # horizon_hours → n_bins

N_BOOTSTRAP = 300
BOOT_SEED = 7777
CI_LEVEL = 0.95

# Rare-cell merge threshold (from 45g / 70b)
VERY_RARE_THRESHOLD = 0.001

# ── Vital / lab / med aggregation rules (from 40b) ─────────────────────
VITAL_AGG = {
    "heart_rate": ("heart_rate", "mean"),
    "spo2": ("spo2", "mean"),
    "resp_rate": ("resp_rate", "mean"),
    "cvp": ("cvp", "mean"),
    "temperature_c": ("temperature", "mean"),
    "fio2": ("fio2", "max"),
}
BP_INVASIVE = {"map_invasive": "map", "sbp_invasive": "sbp", "dbp_invasive": "dbp"}
BP_NONINVASIVE = {"map_noninvasive": "map", "sbp_noninvasive": "sbp", "dbp_noninvasive": "dbp"}

LAB_AGG = {
    "lactate": ("lactate", "max"),
    "creatinine": ("creatinine", "max"),
    "bun": ("bun", "max"),
    "potassium": ("potassium", "last"),
    "sodium": ("sodium", "last"),
    "bicarbonate": ("bicarbonate", "min"),
    "glucose": ("glucose", "mean"),
    "ph": ("ph", "min"),
    "hemoglobin": ("hemoglobin", "min"),
    "wbc": ("wbc", "max"),
    "platelet": ("platelet", "min"),
    "albumin": ("albumin", "min"),
    "bilirubin_total": ("bilirubin", "max"),
    "troponin_t": ("troponin", "max"),
    "bnp": ("bnp", "max"),
}

STATE_VARS = [
    "map", "sbp", "dbp", "heart_rate", "spo2", "cvp", "resp_rate",
    "temperature", "fio2",
    "lactate", "creatinine", "bun", "potassium", "sodium",
    "bicarbonate", "glucose", "ph",
    "hemoglobin", "wbc", "platelet", "albumin", "bilirubin",
    "troponin", "bnp",
]
DROP_VARS = ["bnp", "temperature"]

STATIC_COVARIATES = [
    "age", "gender", "admission_type",
    "hf_subtype", "comparator_group",
    "dominant_syndrome", "reason_for_icu",
    "hemo_gate_passed", "include_in_phase05",
]

FLUID_BALANCE_CLIP_L = (-10.0, 15.0)


def _to_edges(raw_bins: list) -> np.ndarray:
    """Convert YAML bin edges (including 'inf') to numeric edges."""
    return np.array([
        np.inf if (isinstance(b, str) and b.lower() == "inf") else float(b)
        for b in raw_bins
    ], dtype=float)


def _discretize_right_open(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Replicate pd.cut(..., right=False, include_lowest=True) category codes."""
    n_levels = len(edges) - 1
    codes = np.searchsorted(edges, values, side="right") - 1
    return np.clip(codes, 0, n_levels - 1).astype(np.int16)


# ════════════════════════════════════════════════════════════════════════
# Phase A: Data preparation helpers
# ════════════════════════════════════════════════════════════════════════

def _agg_func(method: str):
    return "last" if method == "last" else method


def build_grid_for_horizon(
    n_bins: int,
    horizon_hours: int,
    stay_ids: set,
    vitals: pd.DataFrame,
    labs: pd.DataFrame,
    meds: pd.DataFrame,
    outputs_raw: pd.DataFrame,
    ne_factors: dict,
    furo_factors: dict,
) -> pd.DataFrame:
    """Adapted from 40b: build 2h-binned state/action grid for given horizon."""

    def assign_bin(hours_col: pd.Series) -> pd.Series:
        return ((hours_col - T0_HOURS) / BIN_HOURS).astype(int)

    # ── Skeleton ──
    skeleton = pd.DataFrame([
        {"stay_id": sid, "bin": b}
        for sid in stay_ids for b in range(n_bins)
    ])
    skeleton["bin_start_h"] = T0_HOURS + skeleton["bin"] * BIN_HOURS
    skeleton["bin_end_h"] = skeleton["bin_start_h"] + BIN_HOURS

    # ── Vitals ──
    v = vitals[
        vitals["stay_id"].isin(stay_ids)
        & vitals["hours_since_icu_admit"].between(T0_HOURS, T0_HOURS + horizon_hours, inclusive="left")
    ].copy()
    v["bin"] = assign_bin(v["hours_since_icu_admit"])
    v = v[(v["bin"] >= 0) & (v["bin"] < n_bins)]
    v = v.sort_values(["stay_id", "hours_since_icu_admit"])

    vital_parts = []
    for vname, (canonical, agg) in VITAL_AGG.items():
        sub = v[v["vital_name"] == vname]
        if len(sub) == 0:
            continue
        grouped = sub.groupby(["stay_id", "bin"])["valuenum"].agg(_agg_func(agg))
        grouped.name = canonical
        vital_parts.append(grouped)

    for inv_name, canonical in BP_INVASIVE.items():
        noninv_name = inv_name.replace("invasive", "noninvasive")
        inv = v[v["vital_name"] == inv_name]
        noninv = v[v["vital_name"] == noninv_name]
        inv_agg = inv.groupby(["stay_id", "bin"])["valuenum"].mean() if len(inv) > 0 else pd.Series(dtype=float)
        noninv_agg = noninv.groupby(["stay_id", "bin"])["valuenum"].mean() if len(noninv) > 0 else pd.Series(dtype=float)
        combined = inv_agg.combine_first(noninv_agg)
        combined.name = canonical
        vital_parts.append(combined)

    vitals_grid = pd.concat(vital_parts, axis=1).reset_index() if vital_parts else pd.DataFrame(columns=["stay_id", "bin"])

    # ── Labs ──
    la = labs[
        labs["stay_id"].isin(stay_ids)
        & labs["hours_since_icu_admit"].between(T0_HOURS, T0_HOURS + horizon_hours, inclusive="left")
    ].copy()
    la["bin"] = assign_bin(la["hours_since_icu_admit"])
    la = la[(la["bin"] >= 0) & (la["bin"] < n_bins)]
    la = la.sort_values(["stay_id", "hours_since_icu_admit"])

    lab_parts = []
    for lname, (canonical, agg) in LAB_AGG.items():
        sub = la[la["lab_name"] == lname]
        if len(sub) == 0:
            continue
        grouped = sub.groupby(["stay_id", "bin"])["valuenum"].agg(_agg_func(agg))
        grouped.name = canonical
        lab_parts.append(grouped)

    labs_grid = pd.concat(lab_parts, axis=1).reset_index() if lab_parts else pd.DataFrame(columns=["stay_id", "bin"])

    # ── Meds: IV Fluids ──
    m = meds[meds["stay_id"].isin(stay_ids)].copy()
    fluids = m[m["med_class"] == "iv_fluid"].copy()
    fluids = fluids[fluids["hours_since_icu_admit"].between(T0_HOURS, T0_HOURS + horizon_hours, inclusive="left")]
    fluids["bin"] = assign_bin(fluids["hours_since_icu_admit"])
    fluids = fluids[(fluids["bin"] >= 0) & (fluids["bin"] < n_bins)]

    amt_col = "amount" if "amount" in fluids.columns else "totalamount"
    if amt_col in fluids.columns and len(fluids) > 0:
        fluid_grid = fluids.groupby(["stay_id", "bin"])[amt_col].sum().rename("iv_fluid_ml").reset_index()
    else:
        fluid_grid = pd.DataFrame(columns=["stay_id", "bin", "iv_fluid_ml"])

    # ── Meds: Vasopressors (interval overlap) ──
    pressors = m[m["med_class"] == "vasopressor"].copy()
    pressors = pressors.dropna(subset=["rate"])
    pressor_grid = _interval_to_bins_horizon(pressors, stay_ids, ne_factors, "vasopressor_ne_eq", n_bins, horizon_hours)

    # ── Meds: Diuretics ──
    diuretics = m[m["med_class"] == "loop_diuretic"].copy()
    diuretics = diuretics[diuretics["hours_since_icu_admit"].between(T0_HOURS, T0_HOURS + horizon_hours, inclusive="left")]
    diuretics["bin"] = assign_bin(diuretics["hours_since_icu_admit"])
    diuretics = diuretics[(diuretics["bin"] >= 0) & (diuretics["bin"] < n_bins)]

    if len(diuretics) > 0:
        diuretics["furo_eq_mg"] = diuretics.apply(
            lambda r: (r.get("amount", 0) or 0) * furo_factors.get(r["med_name"], 0), axis=1
        )
        diuretic_grid = diuretics.groupby(["stay_id", "bin"])["furo_eq_mg"].sum().rename("diuretic_furo_eq_mg").reset_index()
    else:
        diuretic_grid = pd.DataFrame(columns=["stay_id", "bin", "diuretic_furo_eq_mg"])

    meds_grid = fluid_grid
    for g in [pressor_grid, diuretic_grid]:
        if len(g) > 0:
            meds_grid = meds_grid.merge(g, on=["stay_id", "bin"], how="outer")

    # ── Outputs (urine) ──
    o = outputs_raw[
        outputs_raw["stay_id"].isin(stay_ids)
        & outputs_raw["hours_since_icu_admit"].between(T0_HOURS, T0_HOURS + horizon_hours, inclusive="left")
    ].copy()
    o["bin"] = assign_bin(o["hours_since_icu_admit"])
    o = o[(o["bin"] >= 0) & (o["bin"] < n_bins)]
    urine = o[o["output_type"] == "urine"]
    output_grid = (
        urine.groupby(["stay_id", "bin"])["value"].sum().rename("urine_output_ml").reset_index()
        if len(urine) > 0 else pd.DataFrame(columns=["stay_id", "bin", "urine_output_ml"])
    )

    # ── D3: output documentation absent flag ──
    pre_t0_output = outputs_raw[
        outputs_raw["stay_id"].isin(stay_ids)
        & outputs_raw["hours_since_icu_admit"].between(0, T0_HOURS, inclusive="left")
    ]
    stays_with_output = set(pre_t0_output["stay_id"].unique())
    doc_flag = pd.DataFrame({"stay_id": list(stay_ids)})
    doc_flag["output_documentation_absent"] = ~doc_flag["stay_id"].isin(stays_with_output)

    # ── Merge ──
    grid = skeleton.copy()
    for g in [vitals_grid, labs_grid, meds_grid, output_grid]:
        if len(g) > 0 and "stay_id" in g.columns and "bin" in g.columns:
            grid = grid.merge(g, on=["stay_id", "bin"], how="left")

    # ── Derived fields ──
    grid["iv_fluid_ml"] = grid["iv_fluid_ml"].fillna(0)
    grid["urine_output_ml"] = grid["urine_output_ml"].fillna(0)
    grid["fluid_balance_ml"] = grid["iv_fluid_ml"] - grid["urine_output_ml"]
    grid["fluid_balance_L"] = grid["fluid_balance_ml"] / 1000.0
    grid["fluid_balance_L_clipped"] = grid["fluid_balance_L"].clip(*FLUID_BALANCE_CLIP_L)

    grid = grid.sort_values(["stay_id", "bin"])
    grid["cumulative_fluid_balance_L"] = grid.groupby("stay_id")["fluid_balance_L_clipped"].cumsum()

    for col in ["iv_fluid_ml", "vasopressor_ne_eq", "diuretic_furo_eq_mg"]:
        if col in grid.columns:
            grid[col] = grid[col].fillna(0)

    grid = grid.merge(doc_flag, on="stay_id", how="left")
    grid = grid.sort_values(["stay_id", "bin"]).reset_index(drop=True)

    return grid


def _interval_to_bins_horizon(
    df: pd.DataFrame, stay_ids: set, equiv_factors: dict,
    out_col: str, n_bins: int, horizon_hours: int,
) -> pd.DataFrame:
    """Vasopressor interval-to-bin conversion (adapted for variable horizon)."""
    if len(df) == 0:
        return pd.DataFrame(columns=["stay_id", "bin", out_col])

    rows = []
    pbar = tqdm(df.iterrows(), total=len(df), desc="      pressor interval→bin", leave=False, mininterval=5)
    for _, r in pbar:
        sid = r["stay_id"]
        if sid not in stay_ids:
            continue
        start_h = r["hours_since_icu_admit"]
        dur = r.get("duration_hours", 0) or 0
        end_h = start_h + max(dur, 0.01)
        rate_raw = r.get("rate", 0) or 0
        weight = r.get("patientweight", 0) or 0
        med = r.get("med_name", "")
        factor = equiv_factors.get(med, 0)
        rateuom = str(r.get("rateuom", "")).lower()
        if "kg" in rateuom:
            ne_rate = rate_raw * factor
        elif weight > 0:
            ne_rate = (rate_raw / weight) * factor
        else:
            ne_rate = rate_raw * factor

        for b in range(n_bins):
            bin_start = T0_HOURS + b * BIN_HOURS
            bin_end = bin_start + BIN_HOURS
            overlap_start = max(start_h, bin_start)
            overlap_end = min(end_h, bin_end)
            if overlap_start < overlap_end:
                overlap_frac = (overlap_end - overlap_start) / BIN_HOURS
                rows.append({"stay_id": sid, "bin": b, out_col: ne_rate, "_weight": overlap_frac})

    if not rows:
        return pd.DataFrame(columns=["stay_id", "bin", out_col])

    rdf = pd.DataFrame(rows)
    rdf["_weighted_val"] = rdf[out_col] * rdf["_weight"]
    grouped = rdf.groupby(["stay_id", "bin"]).agg(
        _sum_weighted=("_weighted_val", "sum"), _sum_weight=("_weight", "sum"),
    )
    grouped[out_col] = grouped["_sum_weighted"] / grouped["_sum_weight"].clip(lower=1e-9)
    return grouped[[out_col]].reset_index()


def build_missingness_features(grid: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """Adapted from 40c: generate missingness-informative features."""
    grid = grid.sort_values(["stay_id", "bin"]).reset_index(drop=True)
    available_vars = [v for v in STATE_VARS if v in grid.columns]

    new_cols: dict[str, pd.Series] = {}
    trailing_window = min(12, n_bins)

    for var in available_vars:
        mask_col = f"{var}_mask"
        mask = grid[var].notna().astype(int)
        new_cols[mask_col] = mask

        # Time since last
        last_obs_bin = grid["bin"].where(grid[var].notna())
        last_obs_bin = last_obs_bin.groupby(grid["stay_id"]).ffill()
        new_cols[f"{var}_tsl_h"] = (grid["bin"] - last_obs_bin) * BIN_HOURS

        # Frequency in trailing 24h (always 12-bin window regardless of horizon)
        freq = mask.groupby(grid["stay_id"]).transform(
            lambda x: x.rolling(trailing_window, min_periods=1).sum()
        )
        new_cols[f"{var}_freq_24h"] = freq.astype(int)

        # Never measured
        ever_measured = mask.groupby(grid["stay_id"]).transform("max")
        new_cols[f"{var}_never_measured"] = (ever_measured == 0).astype(int)

    grid = pd.concat([grid, pd.DataFrame(new_cols, index=grid.index)], axis=1)
    return grid


def build_state_tensors(
    grid_with_miss: pd.DataFrame, cohort: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Adapted from 45c: LOCF + z-score + tensor assembly."""
    grid = grid_with_miss.copy()
    active_vars = [v for v in STATE_VARS if v not in DROP_VARS and v in grid.columns]

    # LOCF
    grid = grid.sort_values(["stay_id", "bin"]).reset_index(drop=True)
    for var in active_vars:
        grid[var] = grid.groupby("stay_id")[var].ffill()

    # Population median fill
    pop_medians = {}
    for var in active_vars:
        med = grid[var].median()
        pop_medians[var] = float(med) if pd.notna(med) else 0.0
        grid[var] = grid[var].fillna(pop_medians[var])

    # Z-score
    norm_params = {}
    for var in active_vars:
        mu = float(grid[var].mean())
        sigma = float(grid[var].std())
        if sigma < 1e-10:
            sigma = 1.0
        norm_params[var] = {"mean": mu, "std": sigma, "pop_median": pop_medians[var]}
        grid[f"{var}_z"] = (grid[var] - mu) / sigma

    # Merge static covariates
    static_avail = [c for c in STATIC_COVARIATES if c in cohort.columns]
    static_df = cohort[["stay_id"] + static_avail].drop_duplicates(subset="stay_id")
    tensor = grid.merge(static_df, on="stay_id", how="left")

    return tensor, norm_params


def build_outcomes_for_horizon(
    cohort: pd.DataFrame, core: pd.DataFrame,
    n_bins: int, horizon_hours: int,
) -> pd.DataFrame:
    """Adapted from 45d: outcomes with horizon-specific absorbing states."""
    cohort_gated = cohort[cohort["hemo_gate_passed"] == True].copy()  # noqa: E712

    # Core columns
    core_cols = ["stay_id"]
    if "days_to_death_from_icu_admit" in core.columns:
        core_cols.append("days_to_death_from_icu_admit")
    if "icu_los_hours" in core.columns:
        core_cols.append("icu_los_hours")
    elif "los" in core.columns:
        core_cols.append("los")

    out = cohort_gated[["stay_id"]].drop_duplicates().copy()
    out = out.merge(core[core_cols].drop_duplicates(subset="stay_id"), on="stay_id", how="left")

    # Mortality
    if "mortality_28d" in cohort_gated.columns:
        mort28 = cohort_gated[["stay_id", "mortality_28d"]].drop_duplicates(subset="stay_id")
        out = out.merge(mort28, on="stay_id", how="left")
    elif "days_to_death_from_icu_admit" in out.columns:
        dtd = out["days_to_death_from_icu_admit"]
        out["mortality_28d"] = ((dtd >= 0) & (dtd <= 28)).astype(int)
    else:
        if "hospital_mortality" in cohort_gated.columns:
            hm = cohort_gated[["stay_id", "hospital_mortality"]].drop_duplicates(subset="stay_id")
            out = out.merge(hm, on="stay_id", how="left")
            out["mortality_28d"] = out["hospital_mortality"].fillna(0).astype(int)

    out["mortality_28d"] = out["mortality_28d"].fillna(0).astype(int)

    if "days_to_death_from_icu_admit" in out.columns:
        dtd = out["days_to_death_from_icu_admit"]
        out["mortality_7d"] = ((dtd >= 0) & (dtd <= 7)).astype(int)
    else:
        out["mortality_7d"] = 0
    out["mortality_7d"] = out["mortality_7d"].fillna(0).astype(int)

    # ICU LOS
    if "icu_los_hours" in out.columns:
        out["icu_los_days"] = out["icu_los_hours"].astype(float) / 24.0
    elif "los" in out.columns:
        out["icu_los_days"] = out["los"].astype(float)
    else:
        out["icu_los_days"] = np.nan

    # Absorbing states (horizon-specific)
    t0_days = T0_HOURS / 24.0
    end_window_days = (T0_HOURS + horizon_hours) / 24.0

    absorb_type = []
    absorb_bin = []

    for _, row in out.iterrows():
        dtd = row.get("days_to_death_from_icu_admit", np.nan)
        los = row.get("icu_los_days", np.nan)

        if pd.notna(dtd) and t0_days <= dtd <= end_window_days:
            hours_from_t0 = (dtd - t0_days) * 24
            bin_idx = min(int(hours_from_t0 / BIN_HOURS), n_bins - 1)
            absorb_type.append("death")
            absorb_bin.append(bin_idx)
        elif pd.notna(los) and los < end_window_days and (pd.isna(dtd) or dtd > end_window_days):
            if los >= t0_days:
                hours_from_t0 = (los - t0_days) * 24
                bin_idx = min(int(hours_from_t0 / BIN_HOURS), n_bins - 1)
                absorb_type.append("discharge")
                absorb_bin.append(bin_idx)
            else:
                absorb_type.append("censored")
                absorb_bin.append(np.nan)
        else:
            absorb_type.append("observed")
            absorb_bin.append(np.nan)

    out["absorbing_type"] = absorb_type
    out["absorbing_state_bin"] = pd.to_numeric(absorb_bin, errors="coerce")
    out["early_discharge"] = (out["absorbing_type"] == "discharge").astype(int)

    # RRT flag
    for rrt_col in ["rrt_crrt_any_during_stay", "has_rrt"]:
        if rrt_col in cohort_gated.columns:
            rrt = cohort_gated[["stay_id", rrt_col]].drop_duplicates(subset="stay_id")
            out = out.merge(rrt, on="stay_id", how="left")
            out[rrt_col] = out[rrt_col].fillna(0).astype(int)

    # Clean up
    for c in ["icu_los_hours", "los", "days_to_death_from_icu_admit", "hospital_mortality"]:
        if c in out.columns and c not in ("mortality_28d", "mortality_7d"):
            out = out.drop(columns=[c], errors="ignore")

    return out


def build_action_index(state_df: pd.DataFrame, action_space_cfg: dict) -> pd.DataFrame:
    """Compute action_idx from continuous action variables using coarse_3 encoding.

    coarse_3 = fluid_level(3) × vaso_level(3) × diuretic_level(2) = 18 cells
    Encoding: fluid_level * 6 + vaso_level * 2 + diuretic_level
    """
    df = state_df.copy()

    dims = action_space_cfg["action_dimensions"]

    # Fluid: use frozen 45a/45b bins from action_space_v1
    fluid = df["iv_fluid_ml"].fillna(0).to_numpy(dtype=float)
    fluid_edges = _to_edges(dims["iv_fluid_ml_per_2h"]["bins"])
    fluid_level = _discretize_right_open(fluid, fluid_edges)

    # Vaso: 0=none, 1=low, 2=high
    vaso = df["vasopressor_ne_eq"].values if "vasopressor_ne_eq" in df.columns else np.zeros(len(df))
    vaso_edges = _to_edges(dims["vasopressor_ne_eq_mcg_kg_min"]["bins"])
    vaso_level = _discretize_right_open(np.nan_to_num(vaso, nan=0.0), vaso_edges)

    # Diuretic: 0=none, 1=any
    diuretic = df["diuretic_furo_eq_mg"].values if "diuretic_furo_eq_mg" in df.columns else np.zeros(len(df))
    diuretic_edges = _to_edges(dims["diuretic_furo_eq_mg_per_2h"]["bins"])
    diuretic_level = _discretize_right_open(np.nan_to_num(diuretic, nan=0.0), diuretic_edges)

    df["action_idx"] = fluid_level * 6 + vaso_level * 2 + diuretic_level
    return df


def fit_behavior_policy_for_horizon(
    state_df: pd.DataFrame,
    temporal_cols: list[str],
    static_cols: list[str],
    n_bins: int,
) -> pd.DataFrame:
    """Adapted from 70b: fit behavior policy and compute propensities."""
    # Build action indices
    action_space_cfg_path = ROOT / "configs" / "action_space_v1.yaml"
    with open(action_space_cfg_path) as f:
        action_space_cfg = yaml.safe_load(f)
    df = build_action_index(state_df, action_space_cfg)

    # Auto-merge rare cells
    action_counts = df["action_idx"].value_counts()
    total = action_counts.sum()
    merge_map = {}
    for aidx in range(18):
        freq = action_counts.get(aidx, 0) / total
        if freq < VERY_RARE_THRESHOLD and aidx % 2 == 1:
            merge_map[aidx] = aidx - 1

    if merge_map:
        df["action_idx_merged"] = df["action_idx"].replace(merge_map)
    else:
        df["action_idx_merged"] = df["action_idx"]

    # Build feature matrix: z-scored temporal + normalized bin + numeric static covariates
    feature_cols = [c for c in temporal_cols if c in df.columns]
    numeric_static_cols = [
        c for c in static_cols
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    df["bin_normalized"] = df["bin"] / max((n_bins - 1), 1)
    feature_cols = feature_cols + ["bin_normalized"] + numeric_static_cols
    X = df[feature_cols].values.astype(np.float32)
    np.nan_to_num(X, copy=False, nan=0.0)

    y = df["action_idx_merged"].values

    # Fit logistic regression
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    unique_actions = np.unique(y)
    n_classes = len(unique_actions)

    clf = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0, random_state=2024, n_jobs=-1)
    clf.fit(X_s, y)

    # Predict propensities
    proba = clf.predict_proba(X_s)
    class_order = clf.classes_

    prop_vals = np.zeros(len(df), dtype=np.float32)
    for i in range(len(df)):
        a = y[i]
        cls_idx = np.searchsorted(class_order, a) if a in class_order else -1
        if cls_idx >= 0 and cls_idx < len(class_order) and class_order[cls_idx] == a:
            prop_vals[i] = proba[i, cls_idx]
        else:
            prop_vals[i] = 1e-6

    df["propensity"] = prop_vals
    df["propensity_clipped"] = np.clip(prop_vals, 0.01, 0.99)

    # Rule policy: hemodynamic-guided (matching 70c_define_rule_policy_v1)
    # Action encoding: fluid_level * 6 + vaso_level * 2 + diuretic_level
    RULE_RESUSCITATE_AGGRESSIVE = 2 * 6 + 2 * 2 + 0  # 16
    RULE_RESUSCITATE_FLUID      = 2 * 6 + 1 * 2 + 0  # 14  # noqa: E221
    RULE_PERFUSION_SUPPORT      = 1 * 6 + 1 * 2 + 0  # 8   # noqa: E221
    RULE_DECONGEST              = 0 * 6 + 0 * 2 + 1  # 1   # noqa: E221
    RULE_MAINTENANCE            = 1 * 6 + 0 * 2 + 0  # 6   # noqa: E221

    rule = np.full(len(df), RULE_MAINTENANCE, dtype=int)

    # Reconstruct raw vitals from z-scored columns (build_state_tensors z-scores in-place)
    # Try raw first, fall back to z-scored
    if "map" in df.columns:
        map_vals = df["map"].values
    elif "map_z" in df.columns:
        map_vals = df["map_z"].values  # already z-scored; convert via approx thresholds
        # z-threshold for MAP<65: if pop mean~75, std~15 → z=(65-75)/15 ≈ -0.67
        # Use conservative z < -0.5 as proxy
        map_vals = np.where(np.isnan(map_vals), 80.0, map_vals * 15.0 + 75.0)
    else:
        map_vals = np.full(len(df), 80.0)

    if "lactate" in df.columns:
        lac_vals = df["lactate"].values
    elif "lactate_z" in df.columns:
        lac_vals = np.where(np.isnan(df["lactate_z"].values), 1.5,
                            df["lactate_z"].values * 1.5 + 2.0)
    else:
        lac_vals = np.full(len(df), 1.5)

    # Fluid balance
    fb_vals = np.zeros(len(df))
    for fb_candidate in ["cumulative_fluid_balance_L", "fluid_balance_cumulative_L",
                         "fluid_balance_L", "cum_fluid_balance_L"]:
        if fb_candidate in df.columns:
            fb_vals = df[fb_candidate].fillna(0).values
            break

    if "spo2" in df.columns:
        spo2_vals = df["spo2"].values
    elif "spo2_z" in df.columns:
        spo2_vals = np.where(np.isnan(df["spo2_z"].values), 97.0,
                             df["spo2_z"].values * 3.0 + 96.0)
    else:
        spo2_vals = np.full(len(df), 97.0)

    # Apply rules in priority order (matching 70c)
    mask_shock_perf = (map_vals < 65) & (lac_vals > 2.0)
    rule[mask_shock_perf] = RULE_RESUSCITATE_AGGRESSIVE

    mask_hypo = (map_vals < 65) & ~mask_shock_perf
    rule[mask_hypo] = RULE_RESUSCITATE_FLUID

    mask_perf = (lac_vals > 2.0) & (map_vals >= 65)
    rule[mask_perf] = RULE_PERFUSION_SUPPORT

    mask_overload_hypox = (fb_vals > 3.0) & (spo2_vals < 94) & \
                          (map_vals >= 65) & (lac_vals <= 2.0)
    rule[mask_overload_hypox] = RULE_DECONGEST

    mask_severe_overload = (fb_vals > 5.0) & (map_vals >= 65) & \
                           (lac_vals <= 2.0) & ~mask_overload_hypox
    rule[mask_severe_overload] = RULE_DECONGEST

    df["rule_action"] = rule
    rule_merged = rule.copy()
    if merge_map:
        for old_a, new_a in merge_map.items():
            rule_merged[rule == old_a] = new_a
        df["rule_action_merged"] = rule_merged
    else:
        df["rule_action_merged"] = rule

    # Diagnostic: rule policy distribution
    rule_counts = pd.Series(rule).value_counts().sort_index()
    rule_names = {RULE_RESUSCITATE_AGGRESSIVE: "resuscitate_aggressive(16)",
                  RULE_RESUSCITATE_FLUID: "resuscitate_fluid(14)",
                  RULE_PERFUSION_SUPPORT: "perfusion_support(8)",
                  RULE_DECONGEST: "decongest(1)", RULE_MAINTENANCE: "maintenance(6)"}
    print("    Rule policy distribution:")
    for aidx, cnt in rule_counts.items():
        name = rule_names.get(aidx, f"action_{aidx}")
        print(f"      {name}: {cnt:,} ({cnt/len(df)*100:.1f}%)")

    # Return propensity-augmented dataframe
    prop_cols = ["stay_id", "bin", "action_idx", "action_idx_merged",
                 "propensity", "propensity_clipped", "rule_action", "rule_action_merged"]
    return df[prop_cols]


# ════════════════════════════════════════════════════════════════════════
# Phase B: Model training + OPE
# ════════════════════════════════════════════════════════════════════════

def train_gtransformer_for_horizon(
    state_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    propensity_df: pd.DataFrame,
    config: dict,
    n_bins: int,
    horizon_hours: int,
    device: torch.device,
    tag: str,
    training_seed: int | None = None,
) -> tuple[GTransformer, DTRDataset, dict]:
    """Train G-Transformer for a specific horizon and return model + test dataset + OPE results.

    Parameters
    ----------
    training_seed : int or None
        If provided, overrides the config seed for model initialization and
        training. Checkpoint is saved with ``_s{seed}`` suffix to avoid
        overwriting the default run.
    """
    temporal_cols, static_cols = discover_feature_columns(state_df)

    # Merge map from propensity
    merge_map_path = ROOT / "configs" / "70b_action_merge_map.yaml"
    merge_map = None
    if merge_map_path.exists():
        with open(merge_map_path) as f:
            mm = yaml.safe_load(f)
        if "merge_map" in mm:
            merge_map = {int(k): int(v) for k, v in mm["merge_map"].items()}

    # Build train/val/test splits
    all_stay_ids = state_df["stay_id"].unique()
    split_cfg = config["split"]
    train_ids, val_ids, test_ids = make_train_val_test_split(
        all_stay_ids, outcomes_df,
        train_frac=split_cfg["train_frac"],
        val_frac=split_cfg["val_frac"],
        test_frac=split_cfg["test_frac"],
        stratify_col=split_cfg["stratify_col"],
        seed=split_cfg["seed"],
    )

    common_kwargs = dict(
        state_df=state_df, outcomes_df=outcomes_df, propensity_df=propensity_df,
        temporal_cols=temporal_cols, static_cols=static_cols,
        merge_map=merge_map, n_bins=n_bins,
    )

    train_ds = DTRDataset(stay_ids=train_ids, **common_kwargs)
    action_vocab = train_ds.action_vocab

    def _build_split(stay_ids):
        ds = DTRDataset(stay_ids=stay_ids, **common_kwargs)
        ds.action_vocab = action_vocab
        ds.n_actions = len(action_vocab)
        ds._action_to_contiguous = {a: i for i, a in enumerate(action_vocab)}
        return ds

    val_ds = _build_split(val_ids)
    test_ds = _build_split(test_ids)

    print(f"    DTR splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"    Features: temporal={train_ds.d_temporal}, static={train_ds.d_static}, actions={train_ds.n_actions}")

    # Build model
    model_cfg = config["models"]["80c_gtransformer"]
    grid_cfg = {"n_bins": n_bins, "bin_hours": BIN_HOURS, "horizon_hours": horizon_hours, "t0_hours": T0_HOURS}

    model = build_gtransformer_from_config(
        d_temporal=train_ds.d_temporal,
        d_static=train_ds.d_static,
        n_actions=train_ds.n_actions,
        model_cfg=model_cfg,
        grid_cfg=grid_cfg,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Model params: {n_params:,}")

    # Training setup
    train_cfg = config["training"]
    outcome_weight = 1.0  # Default (primary analysis uses ow=1)
    criterion = GTransformerLoss(
        weight_next_state=model_cfg["loss_weights"]["next_state"],
        weight_outcome=outcome_weight,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )

    max_epochs = train_cfg["max_epochs"]
    warmup_epochs = train_cfg["scheduler"].get("warmup_epochs", 10)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=min(train_cfg.get("num_workers", 4), 4),
        pin_memory=train_cfg.get("pin_memory", True), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=min(train_cfg.get("num_workers", 4), 4),
        pin_memory=train_cfg.get("pin_memory", True),
    )

    # Training loop
    patience = train_cfg["early_stopping"]["patience"]
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    actual_seed = training_seed if training_seed is not None else train_cfg.get("seed", 42)
    seed_suffix = f"_s{actual_seed}" if training_seed is not None else ""
    ckpt_path = MODELS_DIR / f"85i_gtransformer_{tag}{seed_suffix}_best.pt"
    grad_clip = train_cfg.get("gradient_clip", 1.0)

    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)

    for epoch in tqdm(range(1, max_epochs + 1), desc=f"    Train {tag}", unit="ep"):
        # Train
        model.train()
        total_loss = 0.0
        n_batches = 0
        train_pbar = tqdm(train_loader, desc=f"      train ep{epoch:03d}", leave=False, mininterval=5)
        for batch in train_pbar:
            temporal = batch["temporal"].to(device)
            static = batch["static"].to(device)
            action = batch["action"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            absorbing_mask = batch["absorbing_mask"].to(device)

            outputs = model(temporal, static, action, valid_mask, absorbing_mask)
            losses = criterion(outputs, {
                "temporal": temporal, "valid_mask": valid_mask,
                "absorbing_mask": absorbing_mask,
                "mortality_28d": batch["mortality_28d"].to(device),
            })
            optimizer.zero_grad()
            losses["loss"].backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += losses["loss"].item()
            n_batches += 1
            train_pbar.set_postfix(loss=f"{total_loss / max(n_batches, 1):.4f}")

        # Validate
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"      val   ep{epoch:03d}", leave=False, mininterval=5)
            for batch in val_pbar:
                temporal = batch["temporal"].to(device)
                static = batch["static"].to(device)
                action = batch["action"].to(device)
                valid_mask = batch["valid_mask"].to(device)
                absorbing_mask = batch["absorbing_mask"].to(device)
                outputs = model(temporal, static, action, valid_mask, absorbing_mask)
                losses = criterion(outputs, {
                    "temporal": temporal, "valid_mask": valid_mask,
                    "absorbing_mask": absorbing_mask,
                    "mortality_28d": batch["mortality_28d"].to(device),
                })
                val_loss += losses["loss"].item()
                val_batches += 1
                val_pbar.set_postfix(loss=f"{val_loss / max(val_batches, 1):.4f}")

        avg_val = val_loss / max(val_batches, 1)
        scheduler.step()

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss, "grid_cfg": grid_cfg,
            }, ckpt_path)
        else:
            patience_counter += 1

        if epoch % 10 == 0 or patience_counter == 0:
            avg_train = total_loss / max(n_batches, 1)
            print(f"      Ep {epoch:3d}: train={avg_train:.4f}, val={avg_val:.4f}"
                  f"{' *best*' if patience_counter == 0 else ''}")

        if patience_counter >= patience:
            print(f"      Early stop at ep {epoch} (best={best_epoch})")
            break

    # Load best
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"    Best model: epoch {best_epoch}, val_loss={best_val_loss:.4f}")

    # OPE: DM with rollout
    print(f"    OPE (DM rollout) ...")
    dm_result = evaluate_dm(model, test_ds, device, use_rollout=True)
    print(f"      V(obs)={dm_result['v_observed']:.4f}, V(rule)={dm_result['v_rule']:.4f}, ATE={dm_result['ate']:+.4f}")

    # Bootstrap
    print(f"    Bootstrap (B={N_BOOTSTRAP}) ...")
    def dm_eval_fn(ds, **kw):
        return evaluate_dm(model, ds, device, use_rollout=True)
    boot = bootstrap_ope(dm_eval_fn, test_ds, n_resamples=N_BOOTSTRAP, seed=BOOT_SEED)

    ope_results = {
        "horizon_hours": horizon_hours,
        "n_bins": n_bins,
        "ate": dm_result["ate"],
        "v_observed": dm_result["v_observed"],
        "v_rule": dm_result["v_rule"],
        "ate_ci_lo": boot.get("ate_ci_lo", np.nan),
        "ate_ci_hi": boot.get("ate_ci_hi", np.nan),
        "n_boot_success": boot.get("n_success", 0),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "training_seed": actual_seed,
    }

    return model, test_ds, ope_results


def run_subgroup_analysis(
    model: GTransformer,
    test_ds: DTRDataset,
    state_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    propensity_df: pd.DataFrame,
    config: dict,
    n_bins: int,
    device: torch.device,
    tag: str,
) -> list[dict]:
    """Subgroup DM evaluation for HFpEF vs no_hf."""
    pheno_path = ROOT / config["data"]["phenotype_labels"]
    pheno_df = pd.read_parquet(pheno_path)

    temporal_cols, static_cols = discover_feature_columns(state_df)
    merge_map_path = ROOT / "configs" / "70b_action_merge_map.yaml"
    merge_map = None
    if merge_map_path.exists():
        with open(merge_map_path) as f:
            mm = yaml.safe_load(f)
        if "merge_map" in mm:
            merge_map = {int(k): int(v) for k, v in mm["merge_map"].items()}

    test_sids = set(test_ds.stay_ids)
    pheno_test = pheno_df[pheno_df["stay_id"].isin(test_sids)].copy()
    action_vocab = test_ds.action_vocab

    results = []
    for hf_grp in ["hfpef", "no_hf"]:
        if "hf_subtype" not in pheno_test.columns:
            continue
        grp_sids = pheno_test.loc[pheno_test["hf_subtype"] == hf_grp, "stay_id"].values
        if len(grp_sids) < 30:
            print(f"      {hf_grp}: n={len(grp_sids)} — too small, skipped")
            continue

        sub_ds = DTRDataset(
            state_df=state_df, outcomes_df=outcomes_df, propensity_df=propensity_df,
            stay_ids=grp_sids, temporal_cols=temporal_cols, static_cols=static_cols,
            merge_map=merge_map, n_bins=n_bins,
        )
        sub_ds.action_vocab = action_vocab
        sub_ds.n_actions = len(action_vocab)
        sub_ds._action_to_contiguous = {a: i for i, a in enumerate(action_vocab)}

        dm = evaluate_dm(model, sub_ds, device, use_rollout=True)

        # Subgroup bootstrap
        def _eval_fn(ds, **kw):
            return evaluate_dm(model, ds, device, use_rollout=True)
        boot = bootstrap_ope(_eval_fn, sub_ds, n_resamples=200, seed=BOOT_SEED + 17)

        results.append({
            "horizon": tag,
            "subgroup": hf_grp,
            "n_stays": len(grp_sids),
            "ate": dm["ate"],
            "v_observed": dm["v_observed"],
            "v_rule": dm["v_rule"],
            "ate_ci_lo": boot.get("ate_ci_lo", np.nan),
            "ate_ci_hi": boot.get("ate_ci_hi", np.nan),
        })
        print(f"      {hf_grp}: n={len(grp_sids)}, ATE={dm['ate']:+.4f} "
              f"[{boot.get('ate_ci_lo', np.nan):+.4f}, {boot.get('ate_ci_hi', np.nan):+.4f}]")

    return results


# ════════════════════════════════════════════════════════════════════════
# Phase C: Cross-horizon comparison + report
# ════════════════════════════════════════════════════════════════════════

def load_primary_24h_results() -> tuple[dict, list[dict]]:
    """Load existing 24h primary results from Step 80c outputs."""
    pv_path = TABLES_DIR / "80c_gtransformer_rollout_ow1_policy_values.csv"
    boot_path = TABLES_DIR / "80c_gtransformer_rollout_ow1_bootstrap_ci.csv"
    sub_path = TABLES_DIR / "80c_gtransformer_rollout_ow1_subgroup_results.csv"

    primary = {"horizon_hours": 24, "n_bins": 12}

    if pv_path.exists():
        pv = pd.read_csv(pv_path)
        # Column is "estimator" (not "method") in 80c outputs
        est_col = "estimator" if "estimator" in pv.columns else "method"
        dm_mask = pv[est_col] == "DM"
        dm_row = pv[dm_mask].iloc[0] if dm_mask.any() else pv.iloc[0]
        primary["ate"] = float(dm_row.get("ate", np.nan))
        primary["v_observed"] = float(dm_row.get("v_observed", np.nan))
        primary["v_rule"] = float(dm_row.get("v_rule", np.nan))
    else:
        print(f"  WARNING: Primary 24h results not found at {pv_path}")
        primary["ate"] = np.nan
        primary["v_observed"] = np.nan
        primary["v_rule"] = np.nan

    if boot_path.exists():
        bc = pd.read_csv(boot_path)
        est_col = "estimator" if "estimator" in bc.columns else "method"
        dm_mask = bc[est_col] == "DM"
        dm_row = bc[dm_mask].iloc[0] if dm_mask.any() else bc.iloc[0]
        primary["ate_ci_lo"] = float(dm_row.get("ate_ci_lo", np.nan))
        primary["ate_ci_hi"] = float(dm_row.get("ate_ci_hi", np.nan))
    else:
        primary["ate_ci_lo"] = np.nan
        primary["ate_ci_hi"] = np.nan

    # Subgroup
    subgroup_results = []
    if sub_path.exists():
        sub = pd.read_csv(sub_path)
        for _, row in sub.iterrows():
            if row.get("stratifier") == "hf_subtype":
                subgroup_results.append({
                    "horizon": "24h",
                    "subgroup": row["group"],
                    "n_stays": int(row.get("n_stays", 0)),
                    "ate": float(row.get("ate", np.nan)),
                    "v_observed": float(row.get("v_observed", np.nan)),
                    "v_rule": float(row.get("v_rule", np.nan)),
                    "ate_ci_lo": np.nan,
                    "ate_ci_hi": np.nan,
                })

    return primary, subgroup_results


def generate_report(
    all_results: list[dict],
    subgroup_results: list[dict],
    horizon_stats: dict,
    ts: str,
) -> str:
    """Generate human-readable comparison report."""
    lines = [
        "=" * 72,
        "85i — Horizon Sensitivity Analysis (24h / 48h / 72h) (MIMIC-IV)",
        "=" * 72,
        f"Generated: {ts}",
        f"Model: G-Transformer DM (rollout, ow=1)",
        f"Bootstrap: B={N_BOOTSTRAP}, seed={BOOT_SEED}, CI={CI_LEVEL*100:.0f}%",
        "",
        "PURPOSE",
        "───────",
        "  Compare ATE estimates across 24h (primary), 48h, and 72h decision",
        "  horizons. Protocol v3.1 specifies 48h and 72h as pre-specified",
        "  horizon sensitivities. A ROBUST finding is one where ATE sign and",
        "  approximate magnitude are preserved across horizons.",
        "",
    ]

    # Horizon coverage stats
    lines.append("HORIZON COVERAGE (from 30d LOS feasibility)")
    lines.append("────────────────────────────────────────────")
    for h, stats in horizon_stats.items():
        lines.append(f"  {h}h: {stats.get('coverage_all', '—')} (all), "
                     f"{stats.get('coverage_hfpef', '—')} (HFpEF), "
                     f"early_disc={stats.get('early_disc_pct', '—')}")
    lines.append("")

    # Main results table
    lines.append("MAIN RESULTS (Full Test Set)")
    lines.append("────────────────────────────")
    lines.append(f"  {'Horizon':<10} {'n_bins':>6} {'ATE':>9} {'CI_lo':>9} {'CI_hi':>9} "
                 f"{'V(obs)':>9} {'V(rule)':>9} {'Δ vs 24h':>9}")
    lines.append(f"  {'─'*10} {'─'*6} {'─'*9} {'─'*9} {'─'*9} {'─'*9} {'─'*9} {'─'*9}")

    baseline_ate = None
    for r in all_results:
        if r["horizon_hours"] == 24:
            baseline_ate = r["ate"]

    for r in all_results:
        h = r["horizon_hours"]
        delta = r["ate"] - baseline_ate if baseline_ate is not None and h != 24 else np.nan
        delta_str = f"{delta:+9.4f}" if pd.notna(delta) else "       —"
        lines.append(
            f"  {h:>4}h     {r.get('n_bins', '?'):>6} {r['ate']:+9.4f} "
            f"{r.get('ate_ci_lo', np.nan):+9.4f} {r.get('ate_ci_hi', np.nan):+9.4f} "
            f"{r.get('v_observed', np.nan):9.4f} {r.get('v_rule', np.nan):9.4f} {delta_str}"
        )
    lines.append("")

    # Subgroup results
    if subgroup_results:
        lines.append("SUBGROUP RESULTS (HFpEF vs no_hf)")
        lines.append("──────────────────────────────────")
        lines.append(f"  {'Horizon':<10} {'Subgroup':<10} {'n':>6} {'ATE':>9} {'CI_lo':>9} {'CI_hi':>9}")
        lines.append(f"  {'─'*10} {'─'*10} {'─'*6} {'─'*9} {'─'*9} {'─'*9}")
        for r in subgroup_results:
            lines.append(
                f"  {r['horizon']:<10} {r['subgroup']:<10} {r['n_stays']:>6} "
                f"{r['ate']:+9.4f} {r.get('ate_ci_lo', np.nan):+9.4f} {r.get('ate_ci_hi', np.nan):+9.4f}"
            )
        lines.append("")

    # Interpretation
    lines.append("INTERPRETATION")
    lines.append("──────────────")
    if baseline_ate is not None:
        lines.append(f"  Baseline ATE (24h): {baseline_ate:+.4f}")
    lines.append("")
    lines.append("  Robustness criteria:")
    lines.append("    - ROBUST if all horizons maintain same ATE sign and |Δ| < 50% of 24h ATE")
    lines.append("    - PARTIALLY ROBUST if same sign but magnitude varies >50%")
    lines.append("    - SENSITIVE if ATE sign reverses in extended horizons")
    lines.append("    - ATTENUATED if |ATE| systematically decreases with longer horizons")
    lines.append("      (expected: longer horizons dilute treatment effect → mild attenuation is normal)")
    lines.append("")

    # Verdict
    extended = [r for r in all_results if r["horizon_hours"] != 24]
    if baseline_ate is not None and extended:
        all_same_sign = all(
            (r["ate"] < 0) == (baseline_ate < 0) for r in extended if pd.notna(r["ate"])
        )
        max_deviation = max(abs(r["ate"] - baseline_ate) for r in extended if pd.notna(r["ate"]))
        ate_decreasing = all(
            abs(r["ate"]) <= abs(baseline_ate) * 1.2 for r in extended if pd.notna(r["ate"])
        )

        if all_same_sign and max_deviation < abs(baseline_ate) * 0.5:
            verdict = "ROBUST"
            lines.append(f"  → VERDICT: {verdict}")
            lines.append("    All horizons preserve sign and magnitude (Δ < 50%)")
        elif all_same_sign and ate_decreasing:
            verdict = "ROBUST (attenuated)"
            lines.append(f"  → VERDICT: {verdict}")
            lines.append("    Sign preserved; magnitude decreases with longer horizon (expected)")
        elif all_same_sign:
            verdict = "PARTIALLY ROBUST"
            lines.append(f"  → VERDICT: {verdict}")
            lines.append(f"    Same sign but magnitude varies (max Δ={max_deviation:.4f})")
        else:
            verdict = "SENSITIVE"
            lines.append(f"  → VERDICT: {verdict}")
            lines.append("    ATE sign reverses across horizons — interpret with caution")
    else:
        verdict = "INCONCLUSIVE"
        lines.append(f"  → VERDICT: {verdict}")
        lines.append("    Insufficient data to compare across horizons")

    lines.append("")
    lines.append("NOTE: Mild ATE attenuation with longer horizons is expected and does not")
    lines.append("undermine the primary finding. Treatment effects measured over 24h may dilute")
    lines.append("when extended to 48h/72h because later bins contribute less treatment contrast.")

    return "\n".join(lines) + "\n"


def generate_multiseed_report(
    agg_results: list[dict],
    per_seed_results: list[dict],
    per_seed_subgroup: list[dict],
    horizon_stats: dict,
    ts: str,
) -> str:
    """Generate multi-seed validation report."""
    lines = [
        "=" * 72,
        "85i — Multi-Seed Horizon Sensitivity Validation (MIMIC-IV)",
        "=" * 72,
        f"Generated: {ts}",
        f"Model: G-Transformer DM (rollout, ow=1)",
        f"Bootstrap per seed: B={N_BOOTSTRAP}, CI={CI_LEVEL*100:.0f}%",
        "",
    ]

    # Collect seeds used
    seeds_used = sorted(set(r["training_seed"] for r in per_seed_results))
    lines.append(f"Seeds: {seeds_used} (n={len(seeds_used)})")
    lines.append("")

    # Per-seed detail table
    lines.append("PER-SEED RESULTS")
    lines.append("─────────────────")
    lines.append(f"  {'Horizon':<8} {'Seed':>6} {'ATE':>10} {'V(obs)':>9} {'V(rule)':>9} "
                 f"{'best_ep':>8} {'val_loss':>9}")
    lines.append(f"  {'─'*8} {'─'*6} {'─'*10} {'─'*9} {'─'*9} {'─'*8} {'─'*9}")
    for r in sorted(per_seed_results, key=lambda x: (x["horizon_hours"], x["training_seed"])):
        lines.append(
            f"  {r['horizon_hours']:>4}h   {r['training_seed']:>6} {r['ate']:+10.4f} "
            f"{r.get('v_observed', np.nan):9.4f} {r.get('v_rule', np.nan):9.4f} "
            f"{r.get('best_epoch', '?'):>8} {r.get('best_val_loss', np.nan):9.4f}"
        )
    lines.append("")

    # Aggregated summary
    lines.append("AGGREGATED ACROSS SEEDS (mean ± std [min, max])")
    lines.append("────────────────────────────────────────────────")
    # Skip 24h primary ONLY when it is the single-model row (no multi-seed stats).
    # When 24h is itself multi-seeded (--horizons 24 --seeds ...), the aggregate
    # row carries ate_std/min/max and must be shown like any other horizon.
    for r in agg_results:
        h = r["horizon_hours"]
        if h == 24 and "ate_std" not in r:
            lines.append(f"  {h:>4}h: ATE={r['ate']:+.4f} (primary, single model)")
            continue
        if "ate_std" not in r:
            lines.append(f"  {h:>4}h: ATE={r['ate']:+.4f} (single seed)")
            continue
        lines.append(
            f"  {h:>4}h: ATE={r['ate']:+.4f} ± {r['ate_std']:.4f} "
            f"[{r['ate_min']:+.4f}, {r['ate_max']:+.4f}]  (n_seeds={r['n_seeds']})"
        )
    lines.append("")

    # Subgroup aggregation
    if per_seed_subgroup:
        lines.append("SUBGROUP AGGREGATED (HFpEF vs no_hf)")
        lines.append("─────────────────────────────────────")
        sub_df = pd.DataFrame(per_seed_subgroup)
        lines.append(f"  {'Horizon':<8} {'Subgroup':<10} {'n':>5} {'ATE_mean':>10} {'ATE_std':>9} "
                     f"{'ATE_min':>10} {'ATE_max':>10} {'sign_consistent':>16}")
        lines.append(f"  {'─'*8} {'─'*10} {'─'*5} {'─'*10} {'─'*9} {'─'*10} {'─'*10} {'─'*16}")
        for (hz, sg), grp in sub_df.groupby(["horizon", "subgroup"]):
            ate_vals = grp["ate"].values
            all_same = "YES" if (all(a < 0 for a in ate_vals) or all(a >= 0 for a in ate_vals)) else "NO"
            lines.append(
                f"  {hz:<8} {sg:<10} {grp['n_stays'].iloc[0]:>5} {ate_vals.mean():+10.4f} "
                f"{ate_vals.std():9.4f} {ate_vals.min():+10.4f} {ate_vals.max():+10.4f} "
                f"{all_same:>16}"
            )
        lines.append("")

    # Stability verdict
    lines.append("MULTI-SEED STABILITY VERDICT")
    lines.append("────────────────────────────")
    for r in agg_results:
        h = r["horizon_hours"]
        # Skip rows without multi-seed stats (incl. the single-model 24h primary);
        # a multi-seeded 24h aggregate (has ate_std) is included like any horizon.
        if "ate_std" not in r:
            continue
        # Check if all seeds agree on sign
        h_seeds = [s for s in per_seed_results if s["horizon_hours"] == h]
        signs = [s["ate"] > 0 for s in h_seeds]
        sign_consistent = all(signs) or not any(signs)
        cv = abs(r["ate_std"] / r["ate"]) if abs(r["ate"]) > 1e-6 else float("inf")
        if sign_consistent and cv < 0.5:
            verdict = "STABLE"
        elif sign_consistent:
            verdict = "SIGN-STABLE (high variance)"
        else:
            verdict = "UNSTABLE (sign flips across seeds)"
        lines.append(f"  {h:>4}h: {verdict}  (CV={cv:.2f}, sign_consistent={sign_consistent})")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  STABLE: all seeds agree on sign, CV < 0.5 → result is seed-robust")
    lines.append("  SIGN-STABLE: all seeds agree on sign, CV >= 0.5 → direction robust, magnitude uncertain")
    lines.append("  UNSTABLE: seeds disagree on ATE sign → result is seed-dependent (not reliable)")
    lines.append("")

    return "\n".join(lines) + "\n"


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="85i Horizon Sensitivity Analysis (48h/72h)")
    p.add_argument("--horizons", nargs="+", type=int, default=[48, 72],
                   help="Horizons to test (default: 48 72)")
    p.add_argument("--skip-data-prep", action="store_true",
                   help="Skip data prep if intermediate files already exist")
    p.add_argument("--skip-training", action="store_true",
                   help="Skip training if model checkpoints exist")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Training seeds for multi-seed validation (e.g. --seeds 42 123 456 789 1024). "
                        "When specified, trains a separate model per seed, then aggregates ATE "
                        "mean/std/range across seeds. Results saved with _multiseed suffix. "
                        "Original single-seed outputs are NOT overwritten.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 72)
    print("85i — Horizon Sensitivity Analysis (24h / 48h / 72h)")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Load base config ──
    config_path = ROOT / "configs" / "80_step_c_training_v1.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # ── Load raw data (shared across horizons) ──
    print("\n[Phase 0] Loading raw data (shared) ...")
    hfpef_cfg = load_config()
    staging_dir = resolve_path(hfpef_cfg, "staging")

    cohort = pd.read_parquet(staging_dir / "25f_primary_cohort_candidate_v1_mimic.parquet")
    gated = cohort[(cohort["is_index_stay"] == True) & (cohort["hemo_gate_passed"] == True)]  # noqa: E712
    stay_ids = set(gated["stay_id"])
    n_stays = len(stay_ids)
    print(f"  Gate-passed stays: {n_stays:,}")

    print("  Loading vitals ...")
    vitals = pd.read_parquet(staging_dir / "10d_vitals_mimic.parquet")
    print("  Loading labs ...")
    labs = pd.read_parquet(staging_dir / "10e_labs_mimic.parquet")
    print("  Loading meds ...")
    meds_raw = pd.read_parquet(staging_dir / "10f_meds_infusions_mimic.parquet")
    print("  Loading outputs ...")
    outputs_raw = pd.read_parquet(staging_dir / "10g_output_events_mimic.parquet")
    print("  Loading core stays ...")
    core = pd.read_parquet(staging_dir / "10a_core_icu_stays_mimic.parquet")

    action_cfg_path = ROOT / "configs" / "29a_candidate_action_schemes.yaml"
    with open(action_cfg_path) as f:
        action_cfg = yaml.safe_load(f)
    ne_factors = action_cfg["ne_equivalent_factors"]
    furo_factors = action_cfg["furo_equivalent_factors"]

    # ── Phase C1: Load primary 24h results ──
    print("\n[Phase C1] Loading 24h primary results ...")
    primary_result, primary_subgroups = load_primary_24h_results()
    if pd.notna(primary_result.get("ate")):
        print(f"  24h ATE={primary_result['ate']:+.4f} "
              f"[{primary_result.get('ate_ci_lo', np.nan):+.4f}, {primary_result.get('ate_ci_hi', np.nan):+.4f}]")
    else:
        print("  WARNING: 24h primary results not available — will still run extended horizons")

    all_main_results = [primary_result]
    all_subgroup_results = list(primary_subgroups)
    # Multi-seed collections (only used when --seeds is specified)
    is_multiseed = args.seeds is not None
    all_main_results_multiseed = []
    all_subgroup_results_multiseed = []
    horizon_stats = {
        24: {"coverage_all": "99.7%", "coverage_hfpef": "99.5%", "early_disc_pct": "27.3%"},
        48: {"coverage_all": "98.6%", "coverage_hfpef": "98.1%", "early_disc_pct": "TBD"},
        72: {"coverage_all": "98.3%", "coverage_hfpef": "97.6%", "early_disc_pct": "TBD"},
    }

    # ── Process each extended horizon ──
    for horizon_hours in args.horizons:
        n_bins = horizon_hours // BIN_HOURS
        tag = f"{horizon_hours}h"
        print(f"\n{'━' * 72}")
        print(f"  HORIZON: {horizon_hours}h ({n_bins} bins)")
        print(f"{'━' * 72}")

        # ── Phase A: Data preparation ──
        grid_path = INTERIM_DIR / f"85i_grid2h_states_{tag}_mimic.parquet"
        miss_path = INTERIM_DIR / f"85i_missingness_features_{tag}_mimic.parquet"
        tensor_path = INTERIM_DIR / f"85i_state_tensors_{tag}_v1_mimic.parquet"
        outcomes_path = INTERIM_DIR / f"85i_outcomes_absorb_{tag}_v1_mimic.parquet"
        prop_path = INTERIM_DIR / f"85i_propensity_{tag}_v1_mimic.parquet"

        if args.skip_data_prep and all(p.exists() for p in [tensor_path, outcomes_path, prop_path]):
            print(f"\n  [A] Skipping data prep (files exist)")
            state_df = pd.read_parquet(tensor_path)
            outcomes_df = pd.read_parquet(outcomes_path)
            propensity_df = pd.read_parquet(prop_path)
        else:
            # A1: Grid building
            print(f"\n  [A1] Building {tag} grid ({n_bins} bins) ...")
            t_a1 = time.time()
            grid = build_grid_for_horizon(
                n_bins, horizon_hours, stay_ids,
                vitals, labs, meds_raw, outputs_raw,
                ne_factors, furo_factors,
            )
            grid.to_parquet(grid_path, index=False, engine="pyarrow")
            print(f"    → {grid_path}  ({len(grid):,} rows)")
            print(f"    Time: {time.time() - t_a1:.1f}s")

            # A2: Missingness features
            print(f"\n  [A2] Building missingness features ...")
            t_a2 = time.time()
            grid_with_miss = build_missingness_features(grid, n_bins)
            grid_with_miss.to_parquet(miss_path, index=False, engine="pyarrow")
            print(f"    → {miss_path}  ({grid_with_miss.shape[1]} cols)")
            print(f"    Time: {time.time() - t_a2:.1f}s")

            # A3: State tensors
            print(f"\n  [A3] Building state tensors ...")
            t_a3 = time.time()
            state_df, norm_params = build_state_tensors(grid_with_miss, cohort)
            state_df.to_parquet(tensor_path, index=False, engine="pyarrow")
            print(f"    → {tensor_path}  ({len(state_df):,} rows, {state_df.shape[1]} cols)")
            print(f"    Time: {time.time() - t_a3:.1f}s")

            # Free grid memory
            del grid, grid_with_miss

            # A4: Outcomes
            print(f"\n  [A4] Building outcomes (horizon={horizon_hours}h) ...")
            t_a4 = time.time()
            outcomes_df = build_outcomes_for_horizon(cohort, core, n_bins, horizon_hours)
            outcomes_df.to_parquet(outcomes_path, index=False, engine="pyarrow")
            early_disc_pct = outcomes_df["early_discharge"].mean() * 100
            horizon_stats[horizon_hours]["early_disc_pct"] = f"{early_disc_pct:.1f}%"
            print(f"    → {outcomes_path}  ({len(outcomes_df):,} stays)")
            print(f"    Early discharge ({tag}): {early_disc_pct:.1f}%")
            print(f"    Mortality 28d: {outcomes_df['mortality_28d'].mean()*100:.1f}%")
            print(f"    Time: {time.time() - t_a4:.1f}s")

            # A5: Behavior policy
            print(f"\n  [A5] Fitting behavior policy ...")
            t_a5 = time.time()
            temporal_cols, static_cols = discover_feature_columns(state_df)
            propensity_df = fit_behavior_policy_for_horizon(
                state_df, temporal_cols, static_cols, n_bins,
            )
            propensity_df.to_parquet(prop_path, index=False, engine="pyarrow")
            print(f"    → {prop_path}  ({len(propensity_df):,} rows)")
            print(f"    Time: {time.time() - t_a5:.1f}s")

        # ── Determine seed list for this horizon ──
        seed_list = args.seeds if args.seeds is not None else [None]

        per_seed_main = []     # collect per-seed main results
        per_seed_subgroup = [] # collect per-seed subgroup results

        for seed_idx, cur_seed in enumerate(seed_list):
            seed_label = f"s{cur_seed}" if cur_seed is not None else "default"

            if is_multiseed:
                print(f"\n  ── Seed {seed_idx+1}/{len(seed_list)}: {seed_label} ──")

            # ── Phase B: Model training + OPE ──
            seed_suffix = f"_s{cur_seed}" if cur_seed is not None else ""
            ckpt_path = MODELS_DIR / f"85i_gtransformer_{tag}{seed_suffix}_best.pt"

            if args.skip_training and ckpt_path.exists():
                print(f"\n  [B] Skipping training (checkpoint exists: {ckpt_path.name})")
                # Load model and evaluate
                temporal_cols, static_cols = discover_feature_columns(state_df)
                merge_map_path = ROOT / "configs" / "70b_action_merge_map.yaml"
                merge_map = None
                if merge_map_path.exists():
                    with open(merge_map_path) as f:
                        mm = yaml.safe_load(f)
                    if "merge_map" in mm:
                        merge_map = {int(k): int(v) for k, v in mm["merge_map"].items()}

                all_stay_ids = state_df["stay_id"].unique()
                split_cfg = config["split"]
                _, _, test_ids = make_train_val_test_split(
                    all_stay_ids, outcomes_df,
                    train_frac=split_cfg["train_frac"], val_frac=split_cfg["val_frac"],
                    test_frac=split_cfg["test_frac"], stratify_col=split_cfg["stratify_col"],
                    seed=split_cfg["seed"],
                )
                train_ids_for_vocab, _, _ = make_train_val_test_split(
                    all_stay_ids, outcomes_df,
                    train_frac=split_cfg["train_frac"], val_frac=split_cfg["val_frac"],
                    test_frac=split_cfg["test_frac"], stratify_col=split_cfg["stratify_col"],
                    seed=split_cfg["seed"],
                )

                common_kwargs = dict(
                    state_df=state_df, outcomes_df=outcomes_df, propensity_df=propensity_df,
                    temporal_cols=temporal_cols, static_cols=static_cols,
                    merge_map=merge_map, n_bins=n_bins,
                )
                train_ds_tmp = DTRDataset(stay_ids=train_ids_for_vocab, **common_kwargs)
                action_vocab = train_ds_tmp.action_vocab

                test_ds = DTRDataset(stay_ids=test_ids, **common_kwargs)
                test_ds.action_vocab = action_vocab
                test_ds.n_actions = len(action_vocab)
                test_ds._action_to_contiguous = {a: i for i, a in enumerate(action_vocab)}

                grid_cfg = {"n_bins": n_bins, "bin_hours": BIN_HOURS,
                            "horizon_hours": horizon_hours, "t0_hours": T0_HOURS}
                model_cfg = config["models"]["80c_gtransformer"]
                model = build_gtransformer_from_config(
                    d_temporal=test_ds.d_temporal, d_static=test_ds.d_static,
                    n_actions=test_ds.n_actions, model_cfg=model_cfg, grid_cfg=grid_cfg,
                ).to(device)
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()

                dm_result = evaluate_dm(model, test_ds, device, use_rollout=True)
                def dm_eval_fn(ds, **kw):
                    return evaluate_dm(model, ds, device, use_rollout=True)
                boot = bootstrap_ope(dm_eval_fn, test_ds, n_resamples=N_BOOTSTRAP, seed=BOOT_SEED)

                ope_results = {
                    "horizon_hours": horizon_hours, "n_bins": n_bins,
                    "ate": dm_result["ate"], "v_observed": dm_result["v_observed"],
                    "v_rule": dm_result["v_rule"],
                    "ate_ci_lo": boot.get("ate_ci_lo", np.nan),
                    "ate_ci_hi": boot.get("ate_ci_hi", np.nan),
                    "n_boot_success": boot.get("n_success", 0),
                    "best_epoch": ckpt.get("epoch", 0),
                    "best_val_loss": ckpt.get("val_loss", np.nan),
                    "n_train": len(train_ds_tmp), "n_test": len(test_ds),
                    "training_seed": cur_seed if cur_seed is not None else int(config["training"].get("seed", 42)),
                }
                del train_ds_tmp
            else:
                print(f"\n  [B] Training G-Transformer for {tag} (seed={seed_label}) ...")
                t_b = time.time()
                model, test_ds, ope_results = train_gtransformer_for_horizon(
                    state_df, outcomes_df, propensity_df,
                    config, n_bins, horizon_hours, device, tag,
                    training_seed=cur_seed,
                )
                print(f"    Training time: {(time.time() - t_b) / 60:.1f} min")

            per_seed_main.append(ope_results)

            # Subgroup analysis
            print(f"\n  [B3] Subgroup analysis ({tag}, {seed_label}) ...")
            sub_results = run_subgroup_analysis(
                model, test_ds, state_df, outcomes_df, propensity_df,
                config, n_bins, device, tag,
            )
            # Tag each subgroup result with seed
            for sr in sub_results:
                sr["training_seed"] = ope_results["training_seed"]
            per_seed_subgroup.extend(sub_results)

            # Clean up GPU memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ── Collect results for this horizon ──
        if is_multiseed:
            # For multi-seed: store all per-seed results, pick median for main
            all_main_results_multiseed.extend(per_seed_main)
            all_subgroup_results_multiseed.extend(per_seed_subgroup)
            # Also add the mean-aggregated result to all_main_results for report
            mean_ate = np.mean([r["ate"] for r in per_seed_main])
            std_ate = np.std([r["ate"] for r in per_seed_main])
            min_ate = np.min([r["ate"] for r in per_seed_main])
            max_ate = np.max([r["ate"] for r in per_seed_main])
            agg_result = {
                "horizon_hours": horizon_hours, "n_bins": per_seed_main[0]["n_bins"],
                "ate": mean_ate, "v_observed": np.mean([r["v_observed"] for r in per_seed_main]),
                "v_rule": np.mean([r["v_rule"] for r in per_seed_main]),
                "ate_ci_lo": np.nan, "ate_ci_hi": np.nan,
                "ate_std": std_ate, "ate_min": min_ate, "ate_max": max_ate,
                "n_seeds": len(per_seed_main),
                "n_train": per_seed_main[0].get("n_train"), "n_test": per_seed_main[0].get("n_test"),
            }
            all_main_results.append(agg_result)
            print(f"\n  ── Multi-seed aggregate ({tag}): ATE mean={mean_ate:+.4f} "
                  f"std={std_ate:.4f} range=[{min_ate:+.4f}, {max_ate:+.4f}] ──")
        else:
            all_main_results.append(per_seed_main[0])
            all_subgroup_results.extend(per_seed_subgroup)

    # ── Phase C: Comparison + report ──
    print(f"\n{'━' * 72}")
    print("  PHASE C: Cross-horizon Comparison")
    print(f"{'━' * 72}")

    if is_multiseed:
        # Save per-seed detailed results (new files, never overwrite originals)
        perseed_main_csv = TABLES_DIR / "85i_horizon_multiseed_per_seed.csv"
        pd.DataFrame(all_main_results_multiseed).to_csv(perseed_main_csv, index=False)
        print(f"  → {perseed_main_csv}")

        if all_subgroup_results_multiseed:
            perseed_sub_csv = TABLES_DIR / "85i_horizon_multiseed_subgroup_per_seed.csv"
            pd.DataFrame(all_subgroup_results_multiseed).to_csv(perseed_sub_csv, index=False)
            print(f"  → {perseed_sub_csv}")

            # Aggregate subgroup results across seeds
            sub_agg_rows = []
            sub_ms_df = pd.DataFrame(all_subgroup_results_multiseed)
            for (hz, sg), grp in sub_ms_df.groupby(["horizon", "subgroup"]):
                sub_agg_rows.append({
                    "horizon": hz, "subgroup": sg,
                    "n_stays": grp["n_stays"].iloc[0],
                    "ate_mean": grp["ate"].mean(), "ate_std": grp["ate"].std(),
                    "ate_min": grp["ate"].min(), "ate_max": grp["ate"].max(),
                    "n_seeds": len(grp),
                })
            sub_agg_csv = TABLES_DIR / "85i_horizon_multiseed_subgroup_agg.csv"
            pd.DataFrame(sub_agg_rows).to_csv(sub_agg_csv, index=False)
            print(f"  → {sub_agg_csv}")

        # Save aggregated main results
        agg_csv = TABLES_DIR / "85i_horizon_multiseed_agg.csv"
        pd.DataFrame(all_main_results).to_csv(agg_csv, index=False)
        print(f"  → {agg_csv}")

        # Generate multi-seed report
        report = generate_multiseed_report(
            all_main_results, all_main_results_multiseed,
            all_subgroup_results_multiseed, horizon_stats, ts,
        )
        report_path = REPORTS_DIR / "85i_horizon_multiseed_summary.txt"
        report_path.write_text(report, encoding="utf-8")
        print(f"  → {report_path}")
        print(report)
    else:
        # Save main results (original filenames)
        main_df = pd.DataFrame(all_main_results)
        main_csv = TABLES_DIR / "85i_horizon_sensitivity.csv"
        main_df.to_csv(main_csv, index=False)
        print(f"  → {main_csv}")

        # Save subgroup results
        if all_subgroup_results:
            sub_df = pd.DataFrame(all_subgroup_results)
            sub_csv = TABLES_DIR / "85i_horizon_subgroup.csv"
            sub_df.to_csv(sub_csv, index=False)
            print(f"  → {sub_csv}")

        # Generate report
        report = generate_report(all_main_results, all_subgroup_results, horizon_stats, ts)
        report_path = REPORTS_DIR / "85i_horizon_sensitivity_summary.txt"
        report_path.write_text(report, encoding="utf-8")
        print(f"  → {report_path}")
        print(report)

    total_time = time.time() - t_start
    print(f"\n{'=' * 72}")
    print(f"85i COMPLETE. Total time: {total_time / 60:.1f} min")
    print("=" * 72)


if __name__ == "__main__":
    main()
