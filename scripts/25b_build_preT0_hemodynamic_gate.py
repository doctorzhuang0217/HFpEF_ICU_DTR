#!/usr/bin/env python3
"""
25b_build_preT0_hemodynamic_gate — Pre-T0 hemodynamic-management eligibility.

Purpose
-------
Determine hemodynamic-management eligibility using ONLY pre-T0 information
(0 to 6 hours post-ICU admission). A patient passes the gate if ANY of:
  1. IV fluid bolus/resuscitation ≥ 250 mL (clinically meaningful threshold)
  2. Vasopressor or inotrope (start or continuation)
  3. IV loop diuretic exposure
  4. Low-perfusion / shock proxy: MAP < 65 AND lactate > 2.0

This gate defines the clinical substrate justifying the DTR analysis.

Outputs include per-subgroup pass rates for Phase 0.5 monitoring, with
special focus on HFpEF retention per checklist guardrails.

Inputs
------
- data/staging/25a_time_zero_flags_mimic.parquet
- data/staging/10d_vitals_mimic.parquet
- data/staging/10e_labs_mimic.parquet
- data/staging/10f_meds_infusions_mimic.parquet

Outputs
-------
- data/staging/25b_hemodynamic_gate_flags_mimic.parquet
- reports/25b_hemodynamic_gate_mimic_summary.txt
"""

from __future__ import annotations

import sys
import textwrap
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path

# ── Hemodynamic gate thresholds ───────────────────────────────────────
MAP_HYPOTENSION_THRESHOLD = 65.0     # mmHg
LACTATE_ELEVATION_THRESHOLD = 2.0    # mmol/L
PRE_T0_HOURS = 6.0                   # window: 0–6h post-admission
# IV fluid: protocol requires "clinically meaningful bolus/resuscitation"
# Primary threshold: ≥ 250 mL total IV fluid in 0–6h window
# Sensitivity thresholds reported alongside for transparency
IV_FLUID_VOLUME_THRESHOLD_ML = 250.0           # primary
IV_FLUID_SENSITIVITY_THRESHOLDS_ML = [0, 250, 500]  # for sweep report


def main() -> None:
    cfg = load_config()
    staging_dir = resolve_path(cfg, "staging")
    reports_dir = resolve_path(cfg, "reports")
    staging_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("25b_build_preT0_hemodynamic_gate — Hemodynamic eligibility gate")
    print("=" * 70)

    # ── load index cohort ────────────────────────────────────────────
    print("\n[1/6] Loading index cohort from 25a ...")
    cohort = pd.read_parquet(
        staging_dir / "25a_time_zero_flags_mimic.parquet",
        columns=["stay_id", "subject_id", "hadm_id", "hf_subtype",
                 "comparator_group", "intime", "time_zero_6h",
                 "pre_t0_start", "pre_t0_end"],
    )
    n_index = len(cohort)
    print(f"  Index stays: {n_index:,}")

    # ── pre-T0 vitals (MAP) ─────────────────────────────────────────
    print("\n[2/6] Extracting pre-T0 vitals (MAP) ...")
    vitals = pd.read_parquet(staging_dir / "10d_vitals_mimic.parquet")
    # Filter to index stays and pre-T0 window
    vitals = vitals[vitals["stay_id"].isin(cohort["stay_id"])].copy()
    vitals = vitals[vitals["hours_since_icu_admit"].between(0, PRE_T0_HOURS, inclusive="left")]

    # MAP: 10d uses vital_name "map_invasive" and "map_noninvasive"
    map_vitals = vitals[vitals["vital_name"].isin(["map_invasive", "map_noninvasive"])].copy()
    map_summary = (
        map_vitals.groupby("stay_id")["valuenum"]
        .agg(pre_t0_min_map="min", pre_t0_mean_map="mean")
    )
    map_below_65 = set(map_summary[map_summary["pre_t0_min_map"] < MAP_HYPOTENSION_THRESHOLD].index)
    print(f"  Stays with any MAP reading pre-T0: {len(map_summary):,}")
    print(f"  Stays with MAP < {MAP_HYPOTENSION_THRESHOLD}: {len(map_below_65):,}")

    # ── pre-T0 lactate ──────────────────────────────────────────────
    print("\n[3/6] Extracting pre-T0 lactate ...")
    labs = pd.read_parquet(staging_dir / "10e_labs_mimic.parquet")
    labs = labs[labs["stay_id"].isin(cohort["stay_id"])].copy()
    labs = labs[labs["hours_since_icu_admit"].between(0, PRE_T0_HOURS, inclusive="left")]
    lactate = labs[labs["lab_name"] == "lactate"].copy()
    lactate_summary = (
        lactate.groupby("stay_id")["valuenum"]
        .agg(pre_t0_max_lactate="max")
    )
    lactate_high = set(
        lactate_summary[lactate_summary["pre_t0_max_lactate"] > LACTATE_ELEVATION_THRESHOLD].index
    )
    print(f"  Stays with lactate pre-T0: {len(lactate_summary):,}")
    print(f"  Stays with lactate > {LACTATE_ELEVATION_THRESHOLD}: {len(lactate_high):,}")

    # ── pre-T0 medications ──────────────────────────────────────────
    print("\n[4/6] Extracting pre-T0 medications ...")
    meds = pd.read_parquet(staging_dir / "10f_meds_infusions_mimic.parquet")
    meds = meds[meds["stay_id"].isin(cohort["stay_id"])].copy()
    meds = meds[meds["hours_since_icu_admit"].between(0, PRE_T0_HOURS, inclusive="left")]

    # Vasopressors & inotropes
    # Prefer med_class flags from 10f; keep med_name fallback for robustness.
    pressor_names = {"norepinephrine", "epinephrine", "phenylephrine",
                     "vasopressin", "dopamine"}
    inotrope_names = {"dobutamine", "milrinone"}
    pressor_stays = set(
        meds[
            (meds["med_class"] == "vasopressor")
            | meds["med_name"].isin(pressor_names)
        ]["stay_id"].unique()
    )
    inotrope_stays = set(
        meds[
            (meds["med_class"] == "inotrope")
            | meds["med_name"].isin(inotrope_names)
        ]["stay_id"].unique()
    )
    print(f"  Stays with vasopressor pre-T0: {len(pressor_stays):,}")
    print(f"  Stays with inotrope pre-T0:    {len(inotrope_stays):,}")

    # IV loop diuretics
    diuretic_names = {"furosemide", "bumetanide", "torsemide"}
    diuretic_stays = set(
        meds[
            (meds["med_class"] == "loop_diuretic")
            | meds["med_name"].isin(diuretic_names)
        ]["stay_id"].unique()
    )
    print(f"  Stays with IV diuretic pre-T0: {len(diuretic_stays):,}")

    # IV fluids — protocol: "clinically meaningful IV fluid bolus/resuscitation"
    # 10f uses med_class="iv_fluid" with amount column (mL)
    fluid_meds = meds[meds["med_class"] == "iv_fluid"].copy()

    # Aggregate total IV fluid volume per stay in pre-T0 window
    fluid_vol_per_stay = (
        fluid_meds.groupby("stay_id")["amount"]
        .sum()
        .rename("pre_t0_iv_fluid_ml")
    )

    # Primary: stays with total IV fluid ≥ threshold
    fluid_stays = set(
        fluid_vol_per_stay[
            fluid_vol_per_stay >= IV_FLUID_VOLUME_THRESHOLD_ML
        ].index
    )
    fluid_any_stays = set(fluid_vol_per_stay.index)  # for sensitivity sweep
    print(f"  Stays with any IV fluid pre-T0:      {len(fluid_any_stays):,}")
    print(f"  Stays with IV fluid ≥ {IV_FLUID_VOLUME_THRESHOLD_ML:.0f} mL: {len(fluid_stays):,}")

    # Sensitivity sweep: gate pass rates under different fluid thresholds
    fluid_sweep = {}
    for thresh in IV_FLUID_SENSITIVITY_THRESHOLDS_ML:
        if thresh == 0:
            sweep_stays = fluid_any_stays
        else:
            sweep_stays = set(
                fluid_vol_per_stay[fluid_vol_per_stay >= thresh].index
            )
        fluid_sweep[thresh] = sweep_stays
    print(f"  Fluid threshold sensitivity sweep:")
    for thresh, stays_set in fluid_sweep.items():
        print(f"    ≥ {thresh:>4.0f} mL: {len(stays_set):,} stays")

    # ── build gate flags ─────────────────────────────────────────────
    print("\n[5/6] Building gate flags ...")
    out = cohort[["stay_id", "subject_id", "hadm_id",
                  "hf_subtype", "comparator_group"]].copy()

    # Individual criteria
    out["hemo_fluid"] = out["stay_id"].isin(fluid_stays)
    out["hemo_pressor"] = out["stay_id"].isin(pressor_stays | inotrope_stays)
    out["hemo_diuretic"] = out["stay_id"].isin(diuretic_stays)
    out["hemo_shock_proxy"] = out["stay_id"].isin(map_below_65 & lactate_high)

    # Data availability flags
    out["vitals_available_pre_t0"] = out["stay_id"].isin(set(map_summary.index))
    out["labs_available_pre_t0"] = out["stay_id"].isin(set(lactate_summary.index))
    out["meds_available_pre_t0"] = out["stay_id"].isin(set(meds["stay_id"].unique()))

    # Add summary vitals and fluid volume
    out = out.merge(map_summary, on="stay_id", how="left")
    out = out.merge(lactate_summary, on="stay_id", how="left")
    out = out.merge(fluid_vol_per_stay.reset_index(), on="stay_id", how="left")
    out["pre_t0_iv_fluid_ml"] = out["pre_t0_iv_fluid_ml"].fillna(0.0)

    # Composite gate: ANY criterion met
    out["hemo_gate_passed"] = (
        out["hemo_fluid"]
        | out["hemo_pressor"]
        | out["hemo_diuretic"]
        | out["hemo_shock_proxy"]
    )

    n_passed = out["hemo_gate_passed"].sum()
    n_failed = n_index - n_passed
    print(f"  Gate PASSED: {n_passed:,} ({n_passed / n_index * 100:.1f}%)")
    print(f"  Gate FAILED: {n_failed:,} ({n_failed / n_index * 100:.1f}%)")

    # ── write output ─────────────────────────────────────────────────
    out = out.sort_values("stay_id").reset_index(drop=True)
    out_path = staging_dir / "25b_hemodynamic_gate_flags_mimic.parquet"
    out.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"\n  → {out_path}  ({len(out):,} rows, {out.shape[1]} cols)")

    # ── summary report with per-subgroup breakdown ───────────────────
    print("\n[6/6] Generating summary report ...")

    # Per-subgroup gate pass rates
    subgroup = (
        out.groupby("comparator_group")
        .agg(
            total=("stay_id", "count"),
            gate_passed=("hemo_gate_passed", "sum"),
            fluid=("hemo_fluid", "sum"),
            pressor=("hemo_pressor", "sum"),
            diuretic=("hemo_diuretic", "sum"),
            shock_proxy=("hemo_shock_proxy", "sum"),
        )
    )
    subgroup["pass_pct"] = (subgroup["gate_passed"] / subgroup["total"] * 100).round(1)

    # HFpEF-specific breakdown
    hfpef = out[out["hf_subtype"] == "hfpef"]
    n_hfpef = len(hfpef)
    n_hfpef_pass = int(hfpef["hemo_gate_passed"].sum())
    n_hfpef_fluid = int(hfpef["hemo_fluid"].sum())
    n_hfpef_pressor = int(hfpef["hemo_pressor"].sum())
    n_hfpef_diuretic = int(hfpef["hemo_diuretic"].sum())
    n_hfpef_shock = int(hfpef["hemo_shock_proxy"].sum())

    # Phase 0.5 guardrail check
    guardrail_msg = ""
    if n_hfpef_pass >= 1500:
        guardrail_msg = f"  ✓ HFpEF post-gate = {n_hfpef_pass:,} (≥1,500): PROCEED with strict HFpEF"
    elif n_hfpef_pass >= 1200:
        guardrail_msg = (
            f"  ⚠ HFpEF post-gate = {n_hfpef_pass:,} (1,200–1,499): "
            f"ACTIVATE diastolic ICD proxy sensitivity"
        )
    else:
        guardrail_msg = (
            f"  ✗ HFpEF post-gate = {n_hfpef_pass:,} (<1,200): "
            f"ESCALATE to Phase 0.5 review"
        )

    # Build fluid threshold sensitivity sweep table
    sweep_lines = []
    for thresh in IV_FLUID_SENSITIVITY_THRESHOLDS_ML:
        s = fluid_sweep[thresh]
        # Compute gate pass with this threshold
        sweep_fluid = out["stay_id"].isin(s)
        sweep_gate = (
            sweep_fluid
            | out["hemo_pressor"]
            | out["hemo_diuretic"]
            | out["hemo_shock_proxy"]
        )
        n_pass = int(sweep_gate.sum())
        # HFpEF specific
        hfpef_mask = out["hf_subtype"] == "hfpef"
        hfpef_fluid = hfpef_mask & sweep_fluid
        hfpef_gate = hfpef_mask & sweep_gate
        n_hfpef_sweep = int(hfpef_gate.sum())
        sweep_lines.append(
            f"    ≥ {thresh:>4.0f} mL:  gate={n_pass:>6,} ({n_pass/n_index*100:.1f}%)  "
            f"HFpEF={n_hfpef_sweep:>5,} ({n_hfpef_sweep/max(n_hfpef,1)*100:.1f}%)  "
            f"fluid-only={int(sweep_fluid.sum()):>6,}"
        )
    sweep_str = "\n".join(sweep_lines)

    report = textwrap.dedent(f"""\
    ================================================================
    25b_build_preT0_hemodynamic_gate — Summary Report
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ================================================================

    COHORT
    ──────
    Index stays             : {n_index:>10,}

    HEMODYNAMIC GATE (any of 4 criteria in 0–{PRE_T0_HOURS:.0f}h window)
    ─────────────────────────────────────────────────────────────
    Gate PASSED             : {n_passed:>10,}  ({n_passed / n_index * 100:.1f}%)
    Gate FAILED             : {n_failed:>10,}  ({n_failed / n_index * 100:.1f}%)

    IV FLUID THRESHOLD      : ≥ {IV_FLUID_VOLUME_THRESHOLD_ML:.0f} mL (clinically meaningful bolus)

    PER-CRITERION BREAKDOWN (not mutually exclusive)
    ─────────────────────────────────────────────────
    IV fluid (≥{IV_FLUID_VOLUME_THRESHOLD_ML:.0f}mL)  : {out['hemo_fluid'].sum():>10,}  ({out['hemo_fluid'].mean() * 100:.1f}%)
    Vasopressor/inotrope    : {out['hemo_pressor'].sum():>10,}  ({out['hemo_pressor'].mean() * 100:.1f}%)
    IV loop diuretic        : {out['hemo_diuretic'].sum():>10,}  ({out['hemo_diuretic'].mean() * 100:.1f}%)
    Shock proxy (MAP<65+Lac>2): {out['hemo_shock_proxy'].sum():>10,}  ({out['hemo_shock_proxy'].mean() * 100:.1f}%)

    IV FLUID THRESHOLD SENSITIVITY SWEEP
    ──────────────────────────────────────
    {sweep_str}

    PASS RATES BY COMPARATOR GROUP
    ────────────────────────────────
    {textwrap.indent(subgroup.to_string(), '  ')}

    HFpEF DETAIL
    ─────────────
    Total HFpEF index       : {n_hfpef:>10,}
    Gate passed             : {n_hfpef_pass:>10,}  ({n_hfpef_pass / max(n_hfpef, 1) * 100:.1f}%)
      IV fluid              : {n_hfpef_fluid:>10,}  ({n_hfpef_fluid / max(n_hfpef, 1) * 100:.1f}%)
      Vasopressor/inotrope  : {n_hfpef_pressor:>10,}  ({n_hfpef_pressor / max(n_hfpef, 1) * 100:.1f}%)
      IV loop diuretic      : {n_hfpef_diuretic:>10,}  ({n_hfpef_diuretic / max(n_hfpef, 1) * 100:.1f}%)
      Shock proxy           : {n_hfpef_shock:>10,}  ({n_hfpef_shock / max(n_hfpef, 1) * 100:.1f}%)

    PHASE 0.5 GUARDRAIL CHECK
    ──────────────────────────
    {guardrail_msg}

    DATA AVAILABILITY
    ──────────────────
    Vitals (MAP) pre-T0     : {out['vitals_available_pre_t0'].sum():>10,}  ({out['vitals_available_pre_t0'].mean() * 100:.1f}%)
    Labs (lactate) pre-T0   : {out['labs_available_pre_t0'].sum():>10,}  ({out['labs_available_pre_t0'].mean() * 100:.1f}%)
    Meds pre-T0             : {out['meds_available_pre_t0'].sum():>10,}  ({out['meds_available_pre_t0'].mean() * 100:.1f}%)

    OUTPUT
    ──────
    {out_path}  ({len(out):,} rows, {out.shape[1]} cols)
    """)

    report_path = reports_dir / "25b_hemodynamic_gate_mimic_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"  → Report: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
