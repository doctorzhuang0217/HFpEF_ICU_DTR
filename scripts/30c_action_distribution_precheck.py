#!/usr/bin/env python3
"""
30c_action_distribution_precheck — Marginal and joint action distributions.

Purpose
-------
Summarize early marginal and joint distributions of fluid, vasopressor,
and diuretic actions in the 0-6h pre-T0 window. Identifies rare action
combinations before full modeling and provides support tables for
Phase 0.5 adjudication.

Inputs
------
- data/staging/25f_primary_cohort_candidate_v1_mimic.parquet
- data/staging/10f_meds_infusions_mimic.parquet
- data/staging/10c_timestamps_mimic.parquet

Outputs
-------
- tables/30c_action_marginals.csv
- tables/30c_action_joint_pre_t0.csv
- reports/30c_action_distribution_precheck_summary.txt
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path

PRE_T0_HOURS = 6.0


def main() -> None:
    cfg = load_config()
    staging_dir = resolve_path(cfg, "staging")
    tables_dir = resolve_path(cfg, "tables")
    reports_dir = resolve_path(cfg, "reports")
    for d in [tables_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("[30c] Computing action distributions...")

    # ── Load data ─────────────────────────────────────────────────────────
    cohort = pd.read_parquet(staging_dir / "25f_primary_cohort_candidate_v1_mimic.parquet")
    meds = pd.read_parquet(staging_dir / "10f_meds_infusions_mimic.parquet")
    timestamps = pd.read_parquet(staging_dir / "10c_timestamps_mimic.parquet")

    gate_passed = set(cohort[cohort["hemo_gate_passed"] == True]["stay_id"])
    meds = meds[meds["stay_id"].isin(gate_passed)].copy()
    comp_map = cohort.set_index("stay_id")["comparator_group"]

    print(f"  Gate-passed stays: {len(gate_passed):,}")

    # ── Filter to pre-T0 window (0-6h) ───────────────────────────────────
    ts = timestamps[["stay_id", "time_zero_6h"]].drop_duplicates()
    meds = meds.merge(ts, on="stay_id", how="left")
    meds["hours_since_admit"] = (
        (pd.to_datetime(meds["starttime"]) - pd.to_datetime(meds["time_zero_6h"]).dt.tz_localize(None) + pd.Timedelta(hours=6))
        .dt.total_seconds() / 3600
    )
    # Keep only 0 to 6h (pre-T0)
    meds_pre = meds[(meds["hours_since_admit"] >= 0) & (meds["hours_since_admit"] < PRE_T0_HOURS)].copy()

    # ── Marginal distributions ────────────────────────────────────────────
    marginals = []

    # Fluid: total mL in 0-6h per stay
    fluid = meds_pre[meds_pre["med_class"] == "iv_fluid"]
    fluid_per_stay = fluid.groupby("stay_id")["amount"].sum().rename("iv_fluid_ml_pre_t0")
    # Add zeros for stays with no fluid
    all_stays = pd.Series(0.0, index=list(gate_passed), name="iv_fluid_ml_pre_t0")
    all_stays.update(fluid_per_stay)
    fluid_per_stay = all_stays

    for label, values in [("iv_fluid_ml_pre_t0", fluid_per_stay)]:
        desc = values.describe(percentiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
        for stat, val in desc.items():
            marginals.append({"dimension": label, "stat": stat, "value": round(val, 2),
                              "scope": "all"})
        # Non-zero only
        nz = values[values > 0]
        if len(nz) > 0:
            desc_nz = nz.describe(percentiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
            for stat, val in desc_nz.items():
                marginals.append({"dimension": label, "stat": stat, "value": round(val, 2),
                                  "scope": "nonzero"})

    # Vasopressor: any vs none in 0-6h
    vp = meds_pre[meds_pre["med_class"] == "vasopressor"]
    vp_stays = set(vp["stay_id"].unique())
    marginals.append({"dimension": "vasopressor_pre_t0", "stat": "n_exposed",
                      "value": len(vp_stays), "scope": "all"})
    marginals.append({"dimension": "vasopressor_pre_t0", "stat": "pct_exposed",
                      "value": round(len(vp_stays) / len(gate_passed) * 100, 1), "scope": "all"})

    # Diuretic: any vs none in 0-6h
    diur = meds_pre[meds_pre["med_class"] == "loop_diuretic"]
    diur_stays = set(diur["stay_id"].unique())
    marginals.append({"dimension": "diuretic_pre_t0", "stat": "n_exposed",
                      "value": len(diur_stays), "scope": "all"})
    marginals.append({"dimension": "diuretic_pre_t0", "stat": "pct_exposed",
                      "value": round(len(diur_stays) / len(gate_passed) * 100, 1), "scope": "all"})

    # Inotrope: any vs none
    ino = meds_pre[meds_pre["med_class"] == "inotrope"]
    ino_stays = set(ino["stay_id"].unique())
    marginals.append({"dimension": "inotrope_pre_t0", "stat": "n_exposed",
                      "value": len(ino_stays), "scope": "all"})
    marginals.append({"dimension": "inotrope_pre_t0", "stat": "pct_exposed",
                      "value": round(len(ino_stays) / len(gate_passed) * 100, 1), "scope": "all"})

    marginals_df = pd.DataFrame(marginals)

    # ── Joint action distribution (pre-T0) ────────────────────────────────
    # Classify each stay into: fluid (none/any), pressor (none/any), diuretic (none/any)
    joint = pd.DataFrame({"stay_id": list(gate_passed)})
    joint["has_fluid"] = joint["stay_id"].isin(fluid_per_stay[fluid_per_stay > 0].index)
    joint["has_fluid_250"] = joint["stay_id"].isin(fluid_per_stay[fluid_per_stay >= 250].index)
    joint["has_pressor"] = joint["stay_id"].isin(vp_stays)
    joint["has_diuretic"] = joint["stay_id"].isin(diur_stays)
    joint = joint.merge(comp_map.rename("comparator_group"), left_on="stay_id",
                        right_index=True, how="left")

    # Joint counts
    joint_counts = []
    for subgroup_label, sub_df in [("all", joint),
                                    ("hfpef", joint[joint["comparator_group"] == "hfpef"])]:
        cross = (
            sub_df
            .groupby(["has_fluid_250", "has_pressor", "has_diuretic"])
            .size()
            .reset_index(name="n_stays")
        )
        cross["subgroup"] = subgroup_label
        cross["pct"] = round(cross["n_stays"] / len(sub_df) * 100, 1)
        joint_counts.append(cross)

    joint_df = pd.concat(joint_counts, ignore_index=True)

    # ── By-subtype marginals ──────────────────────────────────────────────
    subtype_lines = []
    for grp in ["hfpef", "hfref_hfmref", "hf_ef_unknown", "no_hf"]:
        grp_stays = set(comp_map[comp_map == grp].index) & gate_passed
        n = len(grp_stays)
        if n == 0:
            continue
        n_fluid = len(grp_stays & set(fluid_per_stay[fluid_per_stay >= 250].index))
        n_vp = len(grp_stays & vp_stays)
        n_diur = len(grp_stays & diur_stays)
        subtype_lines.append(
            f"  {grp}: n={n:,}, fluid≥250mL={n_fluid}({round(n_fluid/n*100,1)}%), "
            f"pressor={n_vp}({round(n_vp/n*100,1)}%), "
            f"diuretic={n_diur}({round(n_diur/n*100,1)}%)"
        )

    # ── Report ────────────────────────────────────────────────────────────
    report_lines = [
        "=" * 72,
        "30c — Action Distribution Precheck (Pre-T0 0-6h)",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Gate-passed stays: {len(gate_passed):,}",
        "",
        "─" * 72,
        "MARGINAL DISTRIBUTIONS",
        "─" * 72,
        marginals_df.to_string(index=False),
        "",
        "─" * 72,
        "BY-SUBTYPE MARGINALS (Pre-T0)",
        "─" * 72,
    ] + subtype_lines + [
        "",
        "─" * 72,
        "JOINT ACTION DISTRIBUTION (fluid≥250mL × pressor × diuretic)",
        "─" * 72,
        joint_df.to_string(index=False),
    ]

    # Flag rare combinations
    rare_all = joint_df[(joint_df["subgroup"] == "all") & (joint_df["n_stays"] < 50)]
    rare_hfpef = joint_df[(joint_df["subgroup"] == "hfpef") & (joint_df["n_stays"] < 25)]
    report_lines.append("")
    report_lines.append("RARE COMBINATIONS (potential positivity concern):")
    report_lines.append(f"  All: {len(rare_all)} cells with <50 stays")
    report_lines.append(f"  HFpEF: {len(rare_hfpef)} cells with <25 stays")

    # ── Save ──────────────────────────────────────────────────────────────
    marginals_df.to_csv(tables_dir / "30c_action_marginals.csv", index=False)
    joint_df.to_csv(tables_dir / "30c_action_joint_pre_t0.csv", index=False)
    print(f"  Saved: tables/30c_action_marginals.csv")
    print(f"  Saved: tables/30c_action_joint_pre_t0.csv")

    report_path = reports_dir / "30c_action_distribution_precheck_summary.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Saved: {report_path}")

    print("\n[30c] Done.")


if __name__ == "__main__":
    main()
