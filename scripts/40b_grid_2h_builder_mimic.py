#!/usr/bin/env python3
"""
40b_grid_2h_builder_mimic — Aggregate longitudinal data to a common 2h grid.

Purpose
-------
Build the analysis-ready 2h-binned state/action tensor from longitudinal
clinical data (vitals, labs, meds, outputs).  Each row = one stay × one
2h bin.  Time grid runs from T0 (ICU admission + 6h) through T0 + 24h
(primary horizon), yielding 12 bins per stay.

Design decisions applied:
  D1 — fluid_balance_L clipped to [-10, +15] (physiologic bounds)
  D3 — output_documentation_absent flag for stays with zero output in 0-T0

Protocol reference: Step 40b (C3 for MIMIC).

Inputs
------
- data/staging/25f_primary_cohort_candidate_v1_mimic.parquet
- data/staging/10d_vitals_mimic.parquet
- data/staging/10e_labs_mimic.parquet
- data/staging/10f_meds_infusions_mimic.parquet
- data/staging/10g_output_events_mimic.parquet
- configs/29a_candidate_action_schemes.yaml

Outputs
-------
- data/interim/40b_grid2h_states_mimic.parquet
- reports/40b_grid2h_builder_mimic_summary.txt
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

ROOT = Path(__file__).resolve().parents[1]

# ── Grid parameters ────────────────────────────────────────────────────
T0_HOURS = 6       # hours post-admission
BIN_HOURS = 2
N_BINS = 12        # 12 bins × 2h = 24h horizon
HORIZON_HOURS = N_BINS * BIN_HOURS  # 24h

# ── D1: physiologic clipping bounds ───────────────────────────────────
FLUID_BALANCE_CLIP_L = (-10.0, 15.0)

# ── Vital sign aggregation rules ──────────────────────────────────────
# (vital_name_in_10d → canonical_name, aggregation)
# For invasive/non-invasive BP, prefer invasive then fall back
VITAL_AGG = {
    "heart_rate":       ("heart_rate",  "mean"),
    "spo2":             ("spo2",        "mean"),
    "resp_rate":        ("resp_rate",   "mean"),
    "cvp":              ("cvp",         "mean"),
    "temperature_c":    ("temperature", "mean"),
    "fio2":             ("fio2",        "max"),
}

# MAP/SBP/DBP: prefer invasive, fallback to non-invasive
BP_INVASIVE = {
    "map_invasive":  "map",
    "sbp_invasive":  "sbp",
    "dbp_invasive":  "dbp",
}
BP_NONINVASIVE = {
    "map_noninvasive":  "map",
    "sbp_noninvasive":  "sbp",
    "dbp_noninvasive":  "dbp",
}

# ── Lab aggregation rules ─────────────────────────────────────────────
LAB_AGG = {
    "lactate":      ("lactate",      "max"),
    "creatinine":   ("creatinine",   "max"),
    "bun":          ("bun",          "max"),
    "potassium":    ("potassium",    "last"),
    "sodium":       ("sodium",       "last"),
    "bicarbonate":  ("bicarbonate",  "min"),
    "glucose":      ("glucose",      "mean"),
    "ph":           ("ph",           "min"),
    "hemoglobin":   ("hemoglobin",   "min"),
    "wbc":          ("wbc",          "max"),
    "platelet":     ("platelet",     "min"),
    "albumin":      ("albumin",      "min"),
    "bilirubin_total": ("bilirubin", "max"),
    "troponin_t":   ("troponin",     "max"),
    "bnp":          ("bnp",          "max"),
}


def _assign_bin(hours_col: pd.Series) -> pd.Series:
    """Map hours_since_icu_admit to 0-indexed bin within the decision grid."""
    return ((hours_col - T0_HOURS) / BIN_HOURS).astype(int)


def _agg_func(method: str):
    """Return a pandas-compatible aggregation function."""
    if method == "last":
        return "last"
    return method


def build_vitals_grid(
    vitals: pd.DataFrame, stay_ids: set
) -> pd.DataFrame:
    """Aggregate vitals into 2h bins."""
    v = vitals[
        vitals["stay_id"].isin(stay_ids)
        & vitals["hours_since_icu_admit"].between(
            T0_HOURS, T0_HOURS + HORIZON_HOURS, inclusive="left"
        )
    ].copy()
    v["bin"] = _assign_bin(v["hours_since_icu_admit"])
    v = v[(v["bin"] >= 0) & (v["bin"] < N_BINS)]

    # Sort for 'last' aggregation
    v = v.sort_values(["stay_id", "hours_since_icu_admit"])

    result_parts = []

    # Standard vitals
    for vname, (canonical, agg) in VITAL_AGG.items():
        sub = v[v["vital_name"] == vname]
        if len(sub) == 0:
            continue
        grouped = sub.groupby(["stay_id", "bin"])["valuenum"].agg(_agg_func(agg))
        grouped.name = canonical
        result_parts.append(grouped)

    # Blood pressure: prefer invasive, fill with non-invasive
    for inv_name, canonical in BP_INVASIVE.items():
        noninv_name = inv_name.replace("invasive", "noninvasive")
        inv = v[v["vital_name"] == inv_name]
        noninv = v[v["vital_name"] == noninv_name]

        inv_agg = inv.groupby(["stay_id", "bin"])["valuenum"].mean() if len(inv) > 0 else pd.Series(dtype=float)
        noninv_agg = noninv.groupby(["stay_id", "bin"])["valuenum"].mean() if len(noninv) > 0 else pd.Series(dtype=float)

        # Combine: invasive preferred
        combined = inv_agg.combine_first(noninv_agg)
        combined.name = canonical
        result_parts.append(combined)

    if not result_parts:
        return pd.DataFrame(columns=["stay_id", "bin"])

    grid = pd.concat(result_parts, axis=1).reset_index()
    return grid


def build_labs_grid(
    labs: pd.DataFrame, stay_ids: set
) -> pd.DataFrame:
    """Aggregate labs into 2h bins."""
    la = labs[
        labs["stay_id"].isin(stay_ids)
        & labs["hours_since_icu_admit"].between(
            T0_HOURS, T0_HOURS + HORIZON_HOURS, inclusive="left"
        )
    ].copy()
    la["bin"] = _assign_bin(la["hours_since_icu_admit"])
    la = la[(la["bin"] >= 0) & (la["bin"] < N_BINS)]
    la = la.sort_values(["stay_id", "hours_since_icu_admit"])

    result_parts = []
    for lname, (canonical, agg) in LAB_AGG.items():
        sub = la[la["lab_name"] == lname]
        if len(sub) == 0:
            continue
        grouped = sub.groupby(["stay_id", "bin"])["valuenum"].agg(_agg_func(agg))
        grouped.name = canonical
        result_parts.append(grouped)

    if not result_parts:
        return pd.DataFrame(columns=["stay_id", "bin"])

    grid = pd.concat(result_parts, axis=1).reset_index()
    return grid


def build_meds_grid(
    meds: pd.DataFrame, stay_ids: set, ne_factors: dict, furo_factors: dict
) -> pd.DataFrame:
    """Aggregate medications into 2h bins.

    - IV fluids: sum volume (mL) per bin
    - Vasopressors: time-weighted mean NE-equivalent rate (mcg/kg/min) per bin
    - Diuretics: sum furosemide-equivalent dose (mg) per bin
    """
    m = meds[meds["stay_id"].isin(stay_ids)].copy()

    # ── IV Fluids (point amounts) ──────────────────────────────────────
    fluids = m[m["med_class"] == "iv_fluid"].copy()
    fluids = fluids[
        fluids["hours_since_icu_admit"].between(
            T0_HOURS, T0_HOURS + HORIZON_HOURS, inclusive="left"
        )
    ]
    fluids["bin"] = _assign_bin(fluids["hours_since_icu_admit"])
    fluids = fluids[(fluids["bin"] >= 0) & (fluids["bin"] < N_BINS)]

    amt_col = "amount" if "amount" in fluids.columns else "totalamount"
    if amt_col in fluids.columns and len(fluids) > 0:
        fluid_grid = (
            fluids.groupby(["stay_id", "bin"])[amt_col]
            .sum()
            .rename("iv_fluid_ml")
            .reset_index()
        )
    else:
        fluid_grid = pd.DataFrame(columns=["stay_id", "bin", "iv_fluid_ml"])

    # ── Vasopressors (interval-based, need overlap calculation) ────────
    pressors = m[m["med_class"] == "vasopressor"].copy()
    pressors = pressors.dropna(subset=["rate"])
    pressor_grid = _interval_to_bins(
        pressors, stay_ids, ne_factors, "vasopressor_ne_eq", agg="mean"
    )

    # ── Diuretics (bolus doses) ────────────────────────────────────────
    diuretics = m[m["med_class"] == "loop_diuretic"].copy()
    diuretics = diuretics[
        diuretics["hours_since_icu_admit"].between(
            T0_HOURS, T0_HOURS + HORIZON_HOURS, inclusive="left"
        )
    ]
    diuretics["bin"] = _assign_bin(diuretics["hours_since_icu_admit"])
    diuretics = diuretics[(diuretics["bin"] >= 0) & (diuretics["bin"] < N_BINS)]

    if len(diuretics) > 0:
        diuretics["furo_eq_mg"] = diuretics.apply(
            lambda r: (r.get("amount", 0) or 0) * furo_factors.get(r["med_name"], 0),
            axis=1,
        )
        diuretic_grid = (
            diuretics.groupby(["stay_id", "bin"])["furo_eq_mg"]
            .sum()
            .rename("diuretic_furo_eq_mg")
            .reset_index()
        )
    else:
        diuretic_grid = pd.DataFrame(
            columns=["stay_id", "bin", "diuretic_furo_eq_mg"]
        )

    # ── Merge all med grids ────────────────────────────────────────────
    grid = fluid_grid
    for g in [pressor_grid, diuretic_grid]:
        if len(g) > 0:
            grid = grid.merge(g, on=["stay_id", "bin"], how="outer")

    return grid


def _interval_to_bins(
    df: pd.DataFrame,
    stay_ids: set,
    equiv_factors: dict,
    out_col: str,
    agg: str = "mean",
) -> pd.DataFrame:
    """Convert interval-based infusions to 2h bin values.

    For each infusion row with starttime/endtime, compute the overlap
    with each 2h bin and assign a time-weighted contribution.
    """
    if len(df) == 0:
        return pd.DataFrame(columns=["stay_id", "bin", out_col])

    rows = []
    for _, r in df.iterrows():
        sid = r["stay_id"]
        if sid not in stay_ids:
            continue

        start_h = r["hours_since_icu_admit"]
        dur = r.get("duration_hours", 0) or 0
        end_h = start_h + max(dur, 0.01)  # avoid zero-duration

        rate_raw = r.get("rate", 0) or 0
        weight = r.get("patientweight", 0) or 0
        med = r.get("med_name", "")
        factor = equiv_factors.get(med, 0)

        # NE-equivalent rate: rate is typically mcg/min or mcg/kg/min
        # If patientweight available and rate seems total (not per-kg),
        # normalize. For simplicity, assume rate is already mcg/kg/min
        # if rateuom contains "kg". Otherwise divide by weight.
        rateuom = str(r.get("rateuom", "")).lower()
        if "kg" in rateuom:
            ne_rate = rate_raw * factor
        elif weight > 0:
            ne_rate = (rate_raw / weight) * factor
        else:
            ne_rate = rate_raw * factor  # fallback

        # Distribute across overlapping bins
        for b in range(N_BINS):
            bin_start = T0_HOURS + b * BIN_HOURS
            bin_end = bin_start + BIN_HOURS
            overlap_start = max(start_h, bin_start)
            overlap_end = min(end_h, bin_end)
            if overlap_start < overlap_end:
                overlap_frac = (overlap_end - overlap_start) / BIN_HOURS
                rows.append({
                    "stay_id": sid,
                    "bin": b,
                    out_col: ne_rate,
                    "_weight": overlap_frac,
                })

    if not rows:
        return pd.DataFrame(columns=["stay_id", "bin", out_col])

    rdf = pd.DataFrame(rows)

    if agg == "mean":
        # Time-weighted mean
        rdf["_weighted_val"] = rdf[out_col] * rdf["_weight"]
        grouped = rdf.groupby(["stay_id", "bin"]).agg(
            _sum_weighted=("_weighted_val", "sum"),
            _sum_weight=("_weight", "sum"),
        )
        grouped[out_col] = grouped["_sum_weighted"] / grouped["_sum_weight"].clip(lower=1e-9)
        return grouped[[out_col]].reset_index()
    else:
        return rdf.groupby(["stay_id", "bin"])[out_col].sum().reset_index()


def build_output_grid(
    outputs: pd.DataFrame, stay_ids: set
) -> pd.DataFrame:
    """Aggregate urine output into 2h bins."""
    o = outputs[
        outputs["stay_id"].isin(stay_ids)
        & outputs["hours_since_icu_admit"].between(
            T0_HOURS, T0_HOURS + HORIZON_HOURS, inclusive="left"
        )
    ].copy()
    o["bin"] = _assign_bin(o["hours_since_icu_admit"])
    o = o[(o["bin"] >= 0) & (o["bin"] < N_BINS)]

    # Separate urine from other output
    urine = o[o["output_type"] == "urine"]
    urine_grid = (
        urine.groupby(["stay_id", "bin"])["value"]
        .sum()
        .rename("urine_output_ml")
        .reset_index()
    ) if len(urine) > 0 else pd.DataFrame(columns=["stay_id", "bin", "urine_output_ml"])

    return urine_grid


def main() -> None:
    cfg = load_config()
    staging_dir = resolve_path(cfg, "staging")
    interim_dir = resolve_path(cfg, "interim")
    reports_dir = resolve_path(cfg, "reports")
    for d in [interim_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("40b_grid_2h_builder_mimic — 2h state/action grid")
    print("=" * 70)

    # ── Load equivalence factors ───────────────────────────────────────
    action_cfg_path = ROOT / "configs" / "29a_candidate_action_schemes.yaml"
    with open(action_cfg_path) as f:
        action_cfg = yaml.safe_load(f)
    ne_factors = action_cfg["ne_equivalent_factors"]
    furo_factors = action_cfg["furo_equivalent_factors"]

    # ── Load cohort ────────────────────────────────────────────────────
    print("\n[1/6] Loading cohort ...")
    cohort = pd.read_parquet(
        staging_dir / "25f_primary_cohort_candidate_v1_mimic.parquet"
    )
    gated = cohort[
        (cohort["is_index_stay"] == True) & (cohort["hemo_gate_passed"] == True)  # noqa: E712
    ]
    stay_ids = set(gated["stay_id"])
    n_stays = len(stay_ids)
    print(f"  Gate-passed stays: {n_stays:,}")

    # ── Build skeleton grid ────────────────────────────────────────────
    print("\n[2/6] Building skeleton grid ...")
    skeleton = pd.DataFrame([
        {"stay_id": sid, "bin": b}
        for sid in stay_ids
        for b in range(N_BINS)
    ])
    skeleton["bin_start_h"] = T0_HOURS + skeleton["bin"] * BIN_HOURS
    skeleton["bin_end_h"] = skeleton["bin_start_h"] + BIN_HOURS
    print(f"  Skeleton: {len(skeleton):,} rows ({n_stays:,} stays × {N_BINS} bins)")

    # ── Aggregate vitals ──────────────────────────────────────────────
    print("\n[3/6] Aggregating vitals ...")
    vitals = pd.read_parquet(staging_dir / "10d_vitals_mimic.parquet")
    vitals_grid = build_vitals_grid(vitals, stay_ids)
    n_vital_cols = len([c for c in vitals_grid.columns if c not in ("stay_id", "bin")])
    print(f"  Vital columns: {n_vital_cols}")
    del vitals

    # ── Aggregate labs ─────────────────────────────────────────────────
    print("\n[4/6] Aggregating labs ...")
    labs = pd.read_parquet(staging_dir / "10e_labs_mimic.parquet")
    labs_grid = build_labs_grid(labs, stay_ids)
    n_lab_cols = len([c for c in labs_grid.columns if c not in ("stay_id", "bin")])
    print(f"  Lab columns: {n_lab_cols}")
    del labs

    # ── Aggregate meds ─────────────────────────────────────────────────
    print("\n[5/6] Aggregating meds ...")
    meds = pd.read_parquet(staging_dir / "10f_meds_infusions_mimic.parquet")
    meds_grid = build_meds_grid(meds, stay_ids, ne_factors, furo_factors)
    n_med_cols = len([c for c in meds_grid.columns if c not in ("stay_id", "bin")])
    print(f"  Med/action columns: {n_med_cols}")

    # D3: pre-T0 output documentation flag
    print("  Computing output_documentation_absent flag ...")
    outputs_raw = pd.read_parquet(staging_dir / "10g_output_events_mimic.parquet")
    pre_t0_output = outputs_raw[
        outputs_raw["stay_id"].isin(stay_ids)
        & outputs_raw["hours_since_icu_admit"].between(0, T0_HOURS, inclusive="left")
    ]
    stays_with_output = set(pre_t0_output["stay_id"].unique())
    doc_flag = pd.DataFrame({
        "stay_id": list(stay_ids),
    })
    doc_flag["output_documentation_absent"] = ~doc_flag["stay_id"].isin(stays_with_output)
    del meds

    # ── Aggregate outputs ──────────────────────────────────────────────
    print("\n[6/6] Aggregating outputs ...")
    output_grid = build_output_grid(outputs_raw, stay_ids)
    del outputs_raw

    # ── Merge everything onto skeleton ─────────────────────────────────
    print("\nMerging all grids ...")
    grid = skeleton.copy()

    for g, label in [
        (vitals_grid, "vitals"),
        (labs_grid, "labs"),
        (meds_grid, "meds"),
        (output_grid, "outputs"),
    ]:
        if len(g) > 0 and "stay_id" in g.columns and "bin" in g.columns:
            before = grid.shape[1]
            grid = grid.merge(g, on=["stay_id", "bin"], how="left")
            added = grid.shape[1] - before
            print(f"  + {label}: {added} columns")

    # ── Derived fields ─────────────────────────────────────────────────
    # Fluid balance per bin
    grid["iv_fluid_ml"] = grid["iv_fluid_ml"].fillna(0)
    grid["urine_output_ml"] = grid["urine_output_ml"].fillna(0)
    grid["fluid_balance_ml"] = grid["iv_fluid_ml"] - grid["urine_output_ml"]
    grid["fluid_balance_L"] = grid["fluid_balance_ml"] / 1000.0

    # D1: clip fluid balance
    grid["fluid_balance_L_clipped"] = grid["fluid_balance_L"].clip(
        lower=FLUID_BALANCE_CLIP_L[0], upper=FLUID_BALANCE_CLIP_L[1]
    )

    # Cumulative fluid balance from T0
    grid = grid.sort_values(["stay_id", "bin"])
    grid["cumulative_fluid_balance_L"] = grid.groupby("stay_id")["fluid_balance_L_clipped"].cumsum()

    # Fill action NaN with 0 (no treatment = zero dose)
    for col in ["iv_fluid_ml", "vasopressor_ne_eq", "diuretic_furo_eq_mg"]:
        if col in grid.columns:
            grid[col] = grid[col].fillna(0)

    # D3: merge output documentation flag
    grid = grid.merge(doc_flag, on="stay_id", how="left")

    # ── Save ───────────────────────────────────────────────────────────
    grid = grid.sort_values(["stay_id", "bin"]).reset_index(drop=True)
    out_path = interim_dir / "40b_grid2h_states_mimic.parquet"
    grid.to_parquet(out_path, index=False, engine="pyarrow")

    n_rows = len(grid)
    n_cols = grid.shape[1]
    state_cols = [c for c in grid.columns if c not in (
        "stay_id", "bin", "bin_start_h", "bin_end_h",
        "iv_fluid_ml", "vasopressor_ne_eq", "diuretic_furo_eq_mg",
        "fluid_balance_ml", "fluid_balance_L", "fluid_balance_L_clipped",
        "cumulative_fluid_balance_L", "urine_output_ml",
        "output_documentation_absent",
    )]
    action_cols = ["iv_fluid_ml", "vasopressor_ne_eq", "diuretic_furo_eq_mg"]

    # Missingness summary
    miss_pct = grid.isnull().mean()

    print(f"\n  → {out_path}")
    print(f"  Rows: {n_rows:,}  ({n_stays:,} stays × {N_BINS} bins)")
    print(f"  Columns: {n_cols}")

    # ── Report ─────────────────────────────────────────────────────────
    report_lines = [
        "=" * 72,
        "40b — 2h Grid Builder Summary (MIMIC-IV)",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Grid: T0={T0_HOURS}h, bin={BIN_HOURS}h, bins={N_BINS}, horizon={HORIZON_HOURS}h",
        f"Gate-passed stays: {n_stays:,}",
        f"Total rows: {n_rows:,}",
        f"Total columns: {n_cols}",
        "",
        "COLUMN INVENTORY",
        "─────────────────",
    ]
    for c in sorted(grid.columns):
        mp = miss_pct[c] * 100
        report_lines.append(f"  {c:<40} miss={mp:.1f}%")

    report_lines.append("")
    report_lines.append("DESIGN DECISIONS APPLIED")
    report_lines.append("────────────────────────")
    report_lines.append(f"  D1: fluid_balance_L clipped to [{FLUID_BALANCE_CLIP_L[0]}, {FLUID_BALANCE_CLIP_L[1]}]L")
    n_clipped = int(
        ((grid["fluid_balance_L"] < FLUID_BALANCE_CLIP_L[0])
         | (grid["fluid_balance_L"] > FLUID_BALANCE_CLIP_L[1])).sum()
    )
    report_lines.append(f"      Rows clipped: {n_clipped:,} / {n_rows:,} ({n_clipped/n_rows*100:.3f}%)")
    n_doc_absent = int(grid["output_documentation_absent"].sum()) // N_BINS
    report_lines.append(f"  D3: output_documentation_absent stays: {n_doc_absent:,} / {n_stays:,}")

    report_lines.append("")
    report_lines.append("ACTION VARIABLE SUMMARY (across all bins)")
    report_lines.append("──────────────────────────────────────────")
    for ac in action_cols:
        if ac in grid.columns:
            desc = grid[ac].describe()
            nonzero = (grid[ac] > 0).sum()
            report_lines.append(
                f"  {ac}: mean={desc['mean']:.2f}, "
                f"median={desc['50%']:.2f}, "
                f"max={desc['max']:.2f}, "
                f"nonzero={nonzero:,}/{n_rows:,} ({nonzero/n_rows*100:.1f}%)"
            )

    report_path = reports_dir / "40b_grid2h_builder_mimic_summary.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Report: {report_path}")

    print("\n[40b] Done.")


if __name__ == "__main__":
    main()
