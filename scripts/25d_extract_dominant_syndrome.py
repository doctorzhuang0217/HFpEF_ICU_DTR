#!/usr/bin/env python3
"""
25d_extract_dominant_syndrome — Classify acute hemodynamic substrate at T0.

Purpose
-------
Classify each index stay into a dominant hemodynamic syndrome using pre-T0
data. This captures the physiologic context at T0 and is the major
stratifier for treatment-effect heterogeneity analysis.

Categories:
  - shock_dominant:      MAP < 65 + lactate > 2, or vasopressor-dependent
  - congestion_dominant: cumulative fluid balance > 2 L + respiratory distress proxy
  - mixed:               features of both shock and congestion
  - neither:             does not meet shock OR congestion thresholds

This is a phenotyping marker, NOT a causal diagnosis.

Inputs
------
- data/staging/25a_time_zero_flags_mimic.parquet
- data/staging/10d_vitals_mimic.parquet
- data/staging/10e_labs_mimic.parquet
- data/staging/10f_meds_infusions_mimic.parquet
- data/staging/10g_output_events_mimic.parquet

Outputs
-------
- data/staging/25d_dominant_syndrome_flags_mimic.parquet
- reports/25d_dominant_syndrome_mimic_summary.txt
"""

from __future__ import annotations

import sys
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path

# ── Thresholds ────────────────────────────────────────────────────────
MAP_SHOCK_THRESH = 65.0       # mmHg
LACTATE_SHOCK_THRESH = 2.0    # mmol/L
FLUID_BALANCE_CONGEST_L = 2.0 # litres positive balance → congestion proxy
SPO2_CONGEST_THRESH = 92.0    # % — respiratory distress proxy
RR_CONGEST_THRESH = 24.0      # breaths/min — respiratory distress proxy
PRE_T0_HOURS = 6.0


def main() -> None:
    cfg = load_config()
    staging_dir = resolve_path(cfg, "staging")
    reports_dir = resolve_path(cfg, "reports")
    staging_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("25d_extract_dominant_syndrome — Hemodynamic substrate classification")
    print("=" * 70)

    # ── load index cohort ────────────────────────────────────────────
    print("\n[1/5] Loading index cohort ...")
    cohort = pd.read_parquet(
        staging_dir / "25a_time_zero_flags_mimic.parquet",
        columns=["stay_id", "subject_id", "hadm_id", "hf_subtype",
                 "comparator_group", "intime", "time_zero_6h"],
    )
    stay_ids = set(cohort["stay_id"])
    n_index = len(cohort)
    print(f"  Index stays: {n_index:,}")

    # ── pre-T0 vitals ───────────────────────────────────────────────
    print("\n[2/5] Loading pre-T0 vitals ...")
    vitals = pd.read_parquet(staging_dir / "10d_vitals_mimic.parquet")
    vitals = vitals[vitals["stay_id"].isin(stay_ids)].copy()
    vitals = vitals[vitals["hours_since_icu_admit"].between(0, PRE_T0_HOURS, inclusive="left")]

    # MAP
    map_v = vitals[vitals["vital_name"].isin(["map_invasive", "map_noninvasive"])]
    map_agg = map_v.groupby("stay_id")["valuenum"].agg(
        pre_t0_min_map="min", pre_t0_mean_map="mean"
    )

    # SpO2
    spo2_v = vitals[vitals["vital_name"] == "spo2"]
    spo2_agg = spo2_v.groupby("stay_id")["valuenum"].agg(pre_t0_min_spo2="min")

    # RR
    rr_v = vitals[vitals["vital_name"] == "resp_rate"]
    rr_agg = rr_v.groupby("stay_id")["valuenum"].agg(pre_t0_max_rr="max")

    # ── pre-T0 lactate ──────────────────────────────────────────────
    print("\n[3/5] Loading pre-T0 lactate ...")
    labs = pd.read_parquet(staging_dir / "10e_labs_mimic.parquet")
    labs = labs[labs["stay_id"].isin(stay_ids)].copy()
    labs = labs[labs["hours_since_icu_admit"].between(0, PRE_T0_HOURS, inclusive="left")]
    lac = labs[labs["lab_name"] == "lactate"]
    lac_agg = lac.groupby("stay_id")["valuenum"].agg(pre_t0_max_lactate="max")

    # ── pre-T0 medications (vasopressors as shock indicator) ────────
    print("\n[4/5] Loading pre-T0 meds & fluid balance ...")
    meds = pd.read_parquet(staging_dir / "10f_meds_infusions_mimic.parquet")
    meds = meds[meds["stay_id"].isin(stay_ids)].copy()
    meds = meds[meds["hours_since_icu_admit"].between(0, PRE_T0_HOURS, inclusive="left")]

    pressor_names = {"norepinephrine", "epinephrine", "phenylephrine",
                     "vasopressin", "dopamine"}
    pressor_stays = set(meds[meds["med_name"].isin(pressor_names)]["stay_id"].unique())

    # Fluid input pre-T0 — 10f uses med_class="iv_fluid", amount in mL
    fluid_input = meds[meds["med_class"] == "iv_fluid"].copy()

    # Sum fluid amounts per stay
    amount_col = "amount" if "amount" in fluid_input.columns else "totalamount"
    if amount_col in fluid_input.columns and len(fluid_input) > 0:
        fluid_sum = fluid_input.groupby("stay_id")[amount_col].sum().rename("fluid_input_ml")
    else:
        fluid_sum = pd.Series(dtype=float, name="fluid_input_ml")

    # Fluid output pre-T0
    output_events = pd.read_parquet(staging_dir / "10g_output_events_mimic.parquet")
    output_events = output_events[output_events["stay_id"].isin(stay_ids)].copy()
    output_events = output_events[
        output_events["hours_since_icu_admit"].between(0, PRE_T0_HOURS, inclusive="left")
    ]
    output_sum = output_events.groupby("stay_id")["value"].sum().rename("fluid_output_ml")

    # ── build syndrome flags ─────────────────────────────────────────
    print("\n[5/5] Classifying dominant syndrome ...")
    out = cohort[["stay_id", "subject_id", "hadm_id",
                  "hf_subtype", "comparator_group"]].copy()

    out = out.merge(map_agg, on="stay_id", how="left")
    out = out.merge(spo2_agg, on="stay_id", how="left")
    out = out.merge(rr_agg, on="stay_id", how="left")
    out = out.merge(lac_agg, on="stay_id", how="left")
    out = out.merge(fluid_sum.reset_index() if len(fluid_sum) > 0
                    else pd.DataFrame(columns=["stay_id", "fluid_input_ml"]),
                    on="stay_id", how="left")
    out = out.merge(output_sum.reset_index() if len(output_sum) > 0
                    else pd.DataFrame(columns=["stay_id", "fluid_output_ml"]),
                    on="stay_id", how="left")

    out["fluid_input_ml"] = out["fluid_input_ml"].fillna(0)
    out["fluid_output_ml"] = out["fluid_output_ml"].fillna(0)
    out["fluid_balance_ml"] = out["fluid_input_ml"] - out["fluid_output_ml"]
    out["fluid_balance_L"] = out["fluid_balance_ml"] / 1000.0

    # Shock criteria
    has_map = out["pre_t0_min_map"].notna()
    has_lac = out["pre_t0_max_lactate"].notna()
    shock_hemodynamic = (
        has_map & has_lac
        & (out["pre_t0_min_map"] < MAP_SHOCK_THRESH)
        & (out["pre_t0_max_lactate"] > LACTATE_SHOCK_THRESH)
    )
    shock_pressor = out["stay_id"].isin(pressor_stays)
    out["is_shock_dominant"] = shock_hemodynamic | shock_pressor

    # Congestion criteria
    positive_balance = out["fluid_balance_L"] > FLUID_BALANCE_CONGEST_L
    resp_distress = (
        (out["pre_t0_min_spo2"].notna() & (out["pre_t0_min_spo2"] < SPO2_CONGEST_THRESH))
        | (out["pre_t0_max_rr"].notna() & (out["pre_t0_max_rr"] > RR_CONGEST_THRESH))
    )
    out["is_congestion_dominant"] = positive_balance & resp_distress

    # Mixed vs neither
    out["is_mixed_syndrome"] = out["is_shock_dominant"] & out["is_congestion_dominant"]
    out["is_neither_syndrome"] = ~out["is_shock_dominant"] & ~out["is_congestion_dominant"]

    # Final label
    out["dominant_syndrome"] = "neither"
    out.loc[out["is_shock_dominant"] & ~out["is_congestion_dominant"], "dominant_syndrome"] = "shock"
    out.loc[out["is_congestion_dominant"] & ~out["is_shock_dominant"], "dominant_syndrome"] = "congestion"
    out.loc[out["is_mixed_syndrome"], "dominant_syndrome"] = "mixed"

    dist = out["dominant_syndrome"].value_counts()
    for cat, n in dist.items():
        print(f"  {cat:20s}: {n:>8,}  ({n / n_index * 100:.1f}%)")

    # ── write output ─────────────────────────────────────────────────
    out = out.sort_values("stay_id").reset_index(drop=True)
    out_path = staging_dir / "25d_dominant_syndrome_flags_mimic.parquet"
    out.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"\n  → {out_path}  ({len(out):,} rows, {out.shape[1]} cols)")

    # ── summary report ───────────────────────────────────────────────
    cross = pd.crosstab(out["dominant_syndrome"], out["comparator_group"], margins=True)

    report = textwrap.dedent(f"""\
    ================================================================
    25d_extract_dominant_syndrome — Summary Report
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ================================================================

    OVERALL DISTRIBUTION
    ─────────────────────
    {textwrap.indent(dist.to_string(), '  ')}

    CROSS-TAB: Syndrome × Comparator Group
    ────────────────────────────────────────
    {textwrap.indent(cross.to_string(), '  ')}

    FLUID BALANCE (pre-T0, L)
    ──────────────────────────
    {textwrap.indent(out['fluid_balance_L'].describe().round(2).to_string(), '  ')}

    OUTPUT
    ──────
    {out_path}  ({len(out):,} rows, {out.shape[1]} cols)
    """)

    report_path = reports_dir / "25d_dominant_syndrome_mimic_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"  → Report: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
