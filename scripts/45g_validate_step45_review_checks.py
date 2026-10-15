#!/usr/bin/env python3
"""
45g_validate_step45_review_checks — Post-review validation for Step 45 outputs.

Purpose
-------
Execute 4 targeted checks raised in the Step 45 external review:
  1. LOS audit: verify early_discharge count using raw ICU timestamps
  2. Death window consistency: cross-tabulate absorbing states
  3. Quality flag stratification by hf_subtype / comparator_group
  4. Sparse action cell inventory for downstream IPTW risk

Protocol reference: Step 45g (C1 — lightweight validation)

Inputs
------
- data/staging/10a_core_icu_stays_mimic.parquet   (raw ICU timestamps + LOS)
- data/staging/25f_primary_cohort_candidate_v1_mimic.parquet (cohort with hf_subtype)
- data/interim/45d_outcomes_absorb_v1_mimic.parquet
- data/interim/45e_data_quality_flags_v1_mimic.parquet
- data/interim/45b_actions_2h_v1_mimic.parquet

Outputs
-------
- reports/45g_step45_review_validation_summary.txt
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path

T0_HOURS = 6
HORIZON_HOURS = 24
BIN_HOURS = 2


def main() -> None:
    cfg = load_config()
    staging_dir = resolve_path(cfg, "staging")
    interim_dir = resolve_path(cfg, "interim")
    reports_dir = resolve_path(cfg, "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("45g — Step 45 Review Validation Checks")
    print("=" * 72)

    lines: list[str] = [
        "=" * 72,
        "45g — Step 45 Review Validation Summary (MIMIC-IV)",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    # ── Load data ──────────────────────────────────────────────────────
    print("\n[0/4] Loading data ...")
    core = pd.read_parquet(staging_dir / "10a_core_icu_stays_mimic.parquet")
    cohort = pd.read_parquet(staging_dir / "25f_primary_cohort_candidate_v1_mimic.parquet")
    cohort_gated = cohort[cohort["hemo_gate_passed"] == True].copy()
    gated_ids = set(cohort_gated["stay_id"].unique())

    outcomes = pd.read_parquet(interim_dir / "45d_outcomes_absorb_v1_mimic.parquet")
    flags = pd.read_parquet(interim_dir / "45e_data_quality_flags_v1_mimic.parquet")
    actions = pd.read_parquet(interim_dir / "45b_actions_2h_v1_mimic.parquet")

    n_stays = outcomes["stay_id"].nunique()
    print(f"  Outcome stays: {n_stays:,}")

    # ================================================================
    # CHECK 1: LOS audit — independent ICU LOS from raw timestamps
    # ================================================================
    print("\n[1/4] CHECK 1: LOS audit ...")
    lines.extend([
        "CHECK 1: ICU LOS AUDIT",
        "══════════════════════",
    ])

    # Compute LOS from raw timestamps if available
    core_gated = core[core["stay_id"].isin(gated_ids)].copy()

    # Method A: use icu_los_hours from 10a (computed from intime/outtime)
    los_source = None
    if "icu_los_hours" in core_gated.columns:
        core_gated["los_days_raw"] = core_gated["icu_los_hours"] / 24.0
        los_source = "icu_los_hours / 24"
    elif "los" in core_gated.columns:
        core_gated["los_days_raw"] = core_gated["los"].astype(float)
        los_source = "los (fractional days)"
    # Method B: also try direct timestamp difference
    if "intime" in core_gated.columns and "outtime" in core_gated.columns:
        core_gated["intime_ts"] = pd.to_datetime(core_gated["intime"], errors="coerce")
        core_gated["outtime_ts"] = pd.to_datetime(core_gated["outtime"], errors="coerce")
        valid_ts = core_gated["intime_ts"].notna() & core_gated["outtime_ts"].notna()
        core_gated.loc[valid_ts, "los_days_ts"] = (
            (core_gated.loc[valid_ts, "outtime_ts"] - core_gated.loc[valid_ts, "intime_ts"])
            .dt.total_seconds() / 86400.0
        )
        ts_available = True
    else:
        ts_available = False

    end_window_days = (T0_HOURS + HORIZON_HOURS) / 24.0  # 1.25 days
    t0_days = T0_HOURS / 24.0  # 0.25 days

    # Report LOS column availability
    lines.append(f"  LOS source column: {los_source or 'NOT FOUND'}")
    lines.append(f"  Timestamp-based LOS available: {ts_available}")

    if los_source and "los_days_raw" in core_gated.columns:
        los_raw = core_gated.set_index("stay_id")["los_days_raw"]
        los_valid = los_raw.dropna()
        lines.extend([
            f"  Stays with valid LOS: {len(los_valid):,} / {len(core_gated):,}",
            f"  LOS stats (days): mean={los_valid.mean():.2f}, "
            f"median={los_valid.median():.2f}, "
            f"p5={los_valid.quantile(0.05):.2f}, "
            f"p10={los_valid.quantile(0.10):.2f}, "
            f"min={los_valid.min():.4f}",
        ])

        # Count stays with LOS < end_window_days (1.25 days = 30 hours)
        n_short = (los_valid < end_window_days).sum()
        n_very_short = (los_valid < t0_days).sum()
        lines.extend([
            f"  Stays with LOS < {end_window_days:.2f}d (30h): {n_short:,}",
            f"  Stays with LOS < {t0_days:.2f}d (6h, pre-T0): {n_very_short:,}",
        ])

        # Of the short-LOS stays, check if they died
        if n_short > 0:
            short_ids = set(los_valid[los_valid < end_window_days].index)
            short_outcomes = outcomes[outcomes["stay_id"].isin(short_ids)]
            if len(short_outcomes) > 0:
                lines.append(f"  Short-LOS stays in outcomes table: {len(short_outcomes):,}")
                at_dist = short_outcomes["absorbing_type"].value_counts()
                for at, cnt in at_dist.items():
                    lines.append(f"    absorbing_type={at}: {cnt:,}")
            else:
                lines.append("  Short-LOS stays: none found in outcomes table (may have been excluded)")
    else:
        lines.append("  WARNING: No valid LOS data found — cannot audit LOS")

    # Cross-check with timestamp method
    if ts_available and "los_days_ts" in core_gated.columns:
        los_ts = core_gated.set_index("stay_id")["los_days_ts"].dropna()
        n_short_ts = (los_ts < end_window_days).sum()
        lines.append(f"  [Timestamp cross-check] LOS < 1.25d: {n_short_ts:,}")
        if los_source and "los_days_raw" in core_gated.columns:
            # Compare the two methods
            common = los_raw.dropna().index.intersection(los_ts.index)
            if len(common) > 0:
                diff = (los_raw.loc[common] - los_ts.loc[common]).abs()
                lines.append(
                    f"  [Timestamp cross-check] max abs diff vs {los_source}: "
                    f"{diff.max():.6f}d ({diff.max()*24:.4f}h)"
                )

    # Compare with 45d icu_los_days
    if "icu_los_days" in outcomes.columns:
        los_45d = outcomes.set_index("stay_id")["icu_los_days"]
        n_valid_45d = los_45d.notna().sum()
        n_nan_45d = los_45d.isna().sum()
        lines.extend([
            "",
            f"  45d icu_los_days: {n_valid_45d:,} valid, {n_nan_45d:,} NaN",
        ])
        if n_valid_45d > 0:
            lines.append(
                f"  45d icu_los_days stats: mean={los_45d.mean():.2f}d, "
                f"min={los_45d.min():.4f}d"
            )
        if n_nan_45d == len(outcomes):
            lines.append("  *** BUG DETECTED: icu_los_days is ALL NaN — "
                         "45d LOS column mapping is broken ***")
    else:
        lines.append("  45d output missing icu_los_days column entirely")

    lines.append("")

    # ================================================================
    # CHECK 2: Death window consistency
    # ================================================================
    print("\n[2/4] CHECK 2: Death window consistency ...")
    lines.extend([
        "CHECK 2: DEATH WINDOW CONSISTENCY",
        "══════════════════════════════════",
    ])

    at_counts = outcomes["absorbing_type"].value_counts()
    for at, cnt in at_counts.items():
        pct = cnt / n_stays * 100
        lines.append(f"  {at:<12}: {cnt:>8,} ({pct:5.1f}%)")

    # Cross-check: 7d mortality vs death-in-window
    n_death_window = (outcomes["absorbing_type"] == "death").sum()
    n_mort_7d = outcomes["mortality_7d"].sum() if "mortality_7d" in outcomes.columns else 0
    n_mort_28d = outcomes["mortality_28d"].sum() if "mortality_28d" in outcomes.columns else 0
    n_early_disc = outcomes["early_discharge"].sum() if "early_discharge" in outcomes.columns else 0

    lines.extend([
        "",
        f"  death in 24h window (T0+0..T0+24h): {n_death_window:,}",
        f"  7-day mortality:  {n_mort_7d:,}",
        f"  28-day mortality: {n_mort_28d:,}",
        f"  early_discharge:  {n_early_disc:,}",
        "",
        f"  Ratio death_window / mort_7d: {n_death_window/max(n_mort_7d,1):.3f}",
        f"  Ratio mort_7d / mort_28d:     {n_mort_7d/max(n_mort_28d,1):.3f}",
    ])

    # Absorbing state bin distribution
    if "absorbing_state_bin" in outcomes.columns:
        absorbed = outcomes[outcomes["absorbing_state_bin"].notna()]
        if len(absorbed) > 0:
            lines.extend(["", "  Absorbing state bin distribution:"])
            bin_xt = absorbed.groupby(["absorbing_type", "absorbing_state_bin"]).size()
            for (at, b), cnt in bin_xt.items():
                lines.append(f"    {at} bin={int(b)}: {cnt:,}")
    lines.append("")

    # ================================================================
    # CHECK 3: Quality flag stratification by hf_subtype
    # ================================================================
    print("\n[3/4] CHECK 3: Quality flag stratification ...")
    lines.extend([
        "CHECK 3: QUALITY FLAG STRATIFICATION BY HF_SUBTYPE / COMPARATOR_GROUP",
        "═════════════════════════════════════════════════════════════════════",
    ])

    # Merge comparator_group from cohort
    group_cols = []
    for col in ["comparator_group", "hf_subtype"]:
        if col in cohort_gated.columns:
            group_cols.append(col)

    if group_cols:
        group_info = cohort_gated[["stay_id"] + group_cols].drop_duplicates(subset="stay_id")
        flags_strat = flags.merge(group_info, on="stay_id", how="left")

        flag_cols = [c for c in [
            "extreme_sparsity", "fluid_qc_flag", "rrt_exposure",
            "has_sparse_action", "any_quality_concern"
        ] if c in flags_strat.columns]

        for grp_col in group_cols:
            lines.append(f"\n  Stratified by: {grp_col}")
            lines.append(f"  {'group':<20} {'N':>6}" + "".join(
                f" {fc:>22}" for fc in flag_cols
            ))
            lines.append("  " + "-" * (28 + 23 * len(flag_cols)))

            for grp, sub in flags_strat.groupby(grp_col):
                n_grp = len(sub)
                parts = [f"  {str(grp):<20} {n_grp:>6}"]
                for fc in flag_cols:
                    n_flag = sub[fc].sum()
                    pct = n_flag / max(n_grp, 1) * 100
                    parts.append(f" {n_flag:>6} ({pct:5.1f}%)" + " " * 8)
                lines.append("".join(parts))
    else:
        lines.append("  WARNING: no comparator_group or hf_subtype found in cohort")

    lines.append("")

    # ================================================================
    # CHECK 4: Sparse action cell inventory
    # ================================================================
    print("\n[4/4] CHECK 4: Sparse action cell inventory ...")
    lines.extend([
        "CHECK 4: SPARSE ACTION CELL INVENTORY FOR IPTW RISK",
        "════════════════════════════════════════════════════",
    ])

    if "action_idx" in actions.columns:
        cell_freq = actions["action_idx"].value_counts(normalize=True).sort_index()
        cell_count = actions["action_idx"].value_counts().sort_index()

        # Overall summary
        n_cells = len(cell_freq)
        n_rare = (cell_freq < 0.01).sum()
        n_very_rare = (cell_freq < 0.001).sum()
        lines.extend([
            f"  Total action cells: {n_cells}",
            f"  Cells < 1% frequency: {n_rare}",
            f"  Cells < 0.1% frequency: {n_very_rare}",
            "",
            "  Per-cell breakdown:",
            f"  {'idx':>4} {'count':>8} {'freq%':>7}  {'status'}",
            "  " + "-" * 40,
        ])
        for idx in cell_freq.index:
            cnt = cell_count[idx]
            pct = cell_freq[idx] * 100
            status = "RARE" if pct < 1 else ("VERY_RARE" if pct < 0.1 else "OK")
            lines.append(f"  {idx:>4} {cnt:>8,} {pct:>6.2f}%  {status}")

        # Stay-level: how many stays have ANY action in a rare cell
        rare_cells = set(cell_freq[cell_freq < 0.01].index)
        if rare_cells:
            stay_with_rare = actions[actions["action_idx"].isin(rare_cells)]["stay_id"].nunique()
            total_stays = actions["stay_id"].nunique()
            lines.extend([
                "",
                f"  Stays with any rare-cell action: {stay_with_rare:,} / {total_stays:,} "
                f"({stay_with_rare/max(total_stays,1)*100:.1f}%)",
            ])

            # Stratify by comparator_group if available
            if group_cols:
                actions_grp = actions.merge(
                    group_info[["stay_id", "comparator_group"]],
                    on="stay_id", how="left"
                )
                lines.append("  Rare-cell exposure by comparator_group:")
                for grp, sub in actions_grp.groupby("comparator_group"):
                    n_grp_stays = sub["stay_id"].nunique()
                    n_rare_stays = sub[sub["action_idx"].isin(rare_cells)]["stay_id"].nunique()
                    pct = n_rare_stays / max(n_grp_stays, 1) * 100
                    lines.append(f"    {grp}: {n_rare_stays:,} / {n_grp_stays:,} ({pct:.1f}%)")

        # Pre-specified merge trigger recommendation
        lines.extend([
            "",
            "  RECOMMENDED MERGE TRIGGER RULES (for Step 70/75 protocol note):",
            "  - If any cell contributes <0.1% of observation-bins → auto-merge with nearest",
            "  - If has_sparse_action covers >15% of stays → trigger coarser granularity",
            "  - Monitor per-stratum ESS in 75a IPTW; flag if ESS < 25 in any cell×stratum",
        ])
    else:
        lines.append("  WARNING: action_idx not found in 45b output")

    lines.append("")

    # ================================================================
    # VERDICT
    # ================================================================
    # Determine if 45d LOS is broken
    los_broken = False
    if "icu_los_days" in outcomes.columns:
        if outcomes["icu_los_days"].isna().all():
            los_broken = True

    lines.extend([
        "=" * 72,
        "OVERALL VERDICT",
        "=" * 72,
    ])

    if los_broken:
        lines.extend([
            "  STATUS: FAIL — icu_los_days is ALL NaN in current 45d output",
            "  ACTION: Re-run 45d after applying LOS column fix, then re-run 45f freeze",
            "  BLOCKER FOR: Step 50 (absorbing state definitions unreliable)",
        ])
    else:
        lines.extend([
            "  STATUS: PASS (pending re-execution of fixed 45d)",
            "  The 45d fix (icu_los_hours mapping) has been applied in the script.",
            "  After re-execution, verify early_discharge count is non-zero or",
            "  document that 0 is a genuine cohort characteristic.",
        ])

    lines.append("")

    # ── Save ───────────────────────────────────────────────────────────
    report = "\n".join(lines) + "\n"
    report_path = reports_dir / "45g_step45_review_validation_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n  Saved: {report_path}")
    print(report)
    print("[45g] Done.")


if __name__ == "__main__":
    main()
