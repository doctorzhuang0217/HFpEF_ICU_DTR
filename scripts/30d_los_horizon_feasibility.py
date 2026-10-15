#!/usr/bin/env python3
"""
30d_los_horizon_feasibility — LOS coverage for decision horizons.

Purpose
-------
Estimate what proportion of the cohort has complete observation over
the 24h (primary), 48h, and 72h decision horizons post-T0.

Per protocol v3.1 §7, death and alive ICU discharge are absorbing states.
A stay has a "complete observation" for a given horizon if ANY of:
  (a) ICU LOS post-T0 >= horizon  (stayed through the full window), OR
  (b) died in ICU before horizon   (absorbing state: death), OR
  (c) discharged alive from ICU before horizon WITHOUT subsequent ICU
      readmission within 24h (absorbing state: alive discharge).

ICU-to-ICU transfers (same hadm_id, another stay_id starting within
TRANSFER_GAP_HOURS of the current stay's outtime) are true right-censoring
events that reduce coverage.

Protocol requirement: ≥80% of cohort must have complete observation
for the primary 24h horizon (SAP §13).

Inputs
------
- data/staging/25f_primary_cohort_candidate_v1_mimic.parquet
- data/staging/10a_core_icu_stays_mimic.parquet  (for transfer detection)

Outputs
-------
- tables/30d_los_horizon_coverage.csv
- reports/30d_los_horizon_feasibility_summary.txt
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path

T0_HOURS = 6  # T0 = ICU admit + 6h
HORIZONS = [24, 48, 72]
MIN_COVERAGE_24H = 0.80  # protocol requirement
TRANSFER_GAP_HOURS = 24  # max gap to consider as ICU-to-ICU transfer


def _compute_los_post_t0(gated: pd.DataFrame) -> pd.Series:
    """Compute post-T0 LOS in hours, trying available columns."""
    if "los_hours" in gated.columns:
        return gated["los_hours"] - T0_HOURS
    if "los_icu_hours" in gated.columns:
        return gated["los_icu_hours"] - T0_HOURS
    if "icu_los_hours" in gated.columns:
        return gated["icu_los_hours"] - T0_HOURS
    if "los" in gated.columns:
        return gated["los"] * 24 - T0_HOURS
    if "intime" in gated.columns and "outtime" in gated.columns:
        return (
            (pd.to_datetime(gated["outtime"]) - pd.to_datetime(gated["intime"]))
            .dt.total_seconds() / 3600 - T0_HOURS
        )
    raise ValueError("Cannot determine LOS. No suitable column found.")


def _detect_death_flag(df: pd.DataFrame) -> pd.Series:
    """Return boolean series: True if patient died in hospital."""
    for col in ("hospital_mortality", "hospital_expire_flag"):
        if col in df.columns:
            return df[col].fillna(0).astype(int) == 1
    return pd.Series(False, index=df.index)


def _detect_icu_transfers(
    gated: pd.DataFrame, staging_dir: Path
) -> pd.Series:
    """Detect ICU-to-ICU transfers by checking for subsequent ICU stays.

    For each stay in the gated cohort, check whether the same hadm_id
    has another ICU stay whose intime is within TRANSFER_GAP_HOURS of
    the current stay's outtime.  If so, the current stay ended via
    transfer (censoring), not via absorbing-state discharge.

    Returns boolean Series indexed like gated: True = transfer.
    """
    # Load all ICU stays (not just index / not just gated)
    all_stays_path = staging_dir / "10a_core_icu_stays_mimic.parquet"
    if not all_stays_path.exists():
        print("  WARNING: 10a_core_icu_stays_mimic.parquet not found; "
              "cannot detect ICU transfers. Falling back to conservative mode.")
        return pd.Series(False, index=gated.index)

    all_stays = pd.read_parquet(
        all_stays_path,
        columns=["stay_id", "hadm_id", "intime", "outtime"],
    )
    all_stays["intime"] = pd.to_datetime(all_stays["intime"])
    all_stays["outtime"] = pd.to_datetime(all_stays["outtime"])

    # For each gated stay, find if there is a subsequent stay in same hadm_id
    gated_sub = gated[["stay_id", "hadm_id"]].copy()
    gated_sub["outtime_dt"] = pd.to_datetime(gated["outtime"])

    # Get all subsequent stays: same hadm_id, different stay_id
    merged = gated_sub.merge(
        all_stays.rename(columns={
            "stay_id": "next_stay_id",
            "intime": "next_intime",
        })[["next_stay_id", "hadm_id", "next_intime"]],
        on="hadm_id",
        how="left",
    )
    # Filter: next stay must be different and start within gap after current outtime
    merged = merged[merged["stay_id"] != merged["next_stay_id"]]
    merged["gap_hours"] = (
        (merged["next_intime"] - merged["outtime_dt"]).dt.total_seconds() / 3600
    )
    # Transfer = next stay starts within [0, TRANSFER_GAP_HOURS] of current outtime
    transfers = merged[
        (merged["gap_hours"] >= 0) & (merged["gap_hours"] <= TRANSFER_GAP_HOURS)
    ]
    transfer_stay_ids = set(transfers["stay_id"].unique())

    return gated["stay_id"].isin(transfer_stay_ids)


def main() -> None:
    cfg = load_config()
    staging_dir = resolve_path(cfg, "staging")
    tables_dir = resolve_path(cfg, "tables")
    reports_dir = resolve_path(cfg, "reports")
    for d in [tables_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("[30d] Assessing LOS-horizon feasibility...")

    # ── Load cohort ───────────────────────────────────────────────────────
    cohort = pd.read_parquet(staging_dir / "25f_primary_cohort_candidate_v1_mimic.parquet")
    gated = cohort[cohort["hemo_gate_passed"] == True].copy()
    print(f"  Gate-passed stays: {len(gated):,}")

    # ── Compute LOS post-T0 ──────────────────────────────────────────────
    gated["los_post_t0_hours"] = _compute_los_post_t0(gated)

    # ── Death flag ────────────────────────────────────────────────────────
    gated["_died"] = _detect_death_flag(gated)

    # ── ICU transfer detection ────────────────────────────────────────────
    gated["_is_transfer"] = _detect_icu_transfers(gated, staging_dir)
    n_transfers = int(gated["_is_transfer"].sum())
    transfer_pct_all = (n_transfers / len(gated) * 100) if len(gated) else 0.0
    print(f"  ICU-to-ICU transfers detected: {n_transfers:,} "
          f"({transfer_pct_all:.1f}% of gate-passed stays)")

    comp_map = gated.set_index("stay_id")["comparator_group"]

    # ── Horizon coverage ──────────────────────────────────────────────────
    rows = []
    report_lines = [
        "=" * 72,
        "30d — LOS-Horizon Feasibility Assessment (absorbing-state-aware)",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Gate-passed stays: {len(gated):,}",
        f"ICU-to-ICU transfers detected: {n_transfers:,} "
        f"({transfer_pct_all:.1f}% of gate-passed stays)",
        f"T0 = ICU admission + {T0_HOURS}h",
        f"Transfer gap threshold: {TRANSFER_GAP_HOURS}h",
        "",
        "Coverage definition (protocol v3.1 §7):",
        "  A stay is 'complete' for a horizon if:",
        "    (a) ICU LOS post-T0 >= horizon (full window), OR",
        "    (b) died before horizon (absorbing state), OR",
        "    (c) discharged alive before horizon AND not a transfer.",
        "  ICU-to-ICU transfers before horizon are RIGHT-CENSORED.",
        "",
        "Three metrics reported:",
        "  raw_coverage  = stays with LOS >= horizon (old definition)",
        "  full_coverage = stays with complete observation (transfer-adjusted)",
        "  Note: full_coverage = raw + death_absorb + alive_absorb (excl. transfers)",
        "",
    ]

    for horizon in HORIZONS:
        report_lines.append(f"─── Horizon: {horizon}h post-T0 ───")
        for subgroup in ["all", "hfpef", "hfref_hfmref", "hf_ef_unknown", "no_hf"]:
            if subgroup == "all":
                sub = gated
            else:
                sub_ids = comp_map[comp_map == subgroup].index
                sub = gated[gated["stay_id"].isin(sub_ids)]
            if len(sub) == 0:
                continue

            n_total = len(sub)
            short = sub["los_post_t0_hours"] < horizon

            # (a) Stayed through the full window
            n_los_covered = int((~short).sum())

            # Among early exits:
            early = short
            early_died = early & sub["_died"]
            early_alive = early & ~sub["_died"]
            early_transfer = early & sub["_is_transfer"]
            early_absorb_alive = early_alive & ~sub["_is_transfer"]

            # (b) Died before horizon (absorbing)
            n_died_before = int(early_died.sum())

            # (c) Alive discharge, NOT transfer (absorbing)
            n_alive_absorb = int(early_absorb_alive.sum())

            # (d) Transfer before horizon (censored — NOT covered)
            n_transfer_censored = int(early_transfer.sum())

            # Full coverage = (a) + (b) + (c); transfers are excluded
            n_full_covered = n_los_covered + n_died_before + n_alive_absorb

            raw_pct = n_los_covered / n_total * 100
            full_pct = n_full_covered / n_total * 100

            row = {
                "horizon_hours": horizon,
                "subgroup": subgroup,
                "n_total": n_total,
                "n_los_covered": n_los_covered,
                "raw_coverage_pct": round(raw_pct, 1),
                "n_died_before_horizon": n_died_before,
                "n_alive_discharged_absorbing": n_alive_absorb,
                "n_transfer_censored": n_transfer_censored,
                "n_full_covered": n_full_covered,
                "full_coverage_pct": round(full_pct, 1),
            }
            rows.append(row)

            # Pass/fail on full_coverage (transfer-adjusted)
            pass_flag = ""
            if horizon == 24 and subgroup in ("all", "hfpef"):
                pass_flag = " ← PASS" if full_pct >= MIN_COVERAGE_24H * 100 else " ← FAIL"
            report_lines.append(
                f"  {subgroup}: raw={n_los_covered:,}/{n_total:,} ({raw_pct:.1f}%)  "
                f"died={n_died_before:,}  alive_absorb={n_alive_absorb:,}  "
                f"transfer={n_transfer_censored:,}  "
                f"full={n_full_covered:,}/{n_total:,} ({full_pct:.1f}%){pass_flag}"
            )
        report_lines.append("")

    coverage_df = pd.DataFrame(rows)

    # ── Protocol check ────────────────────────────────────────────────────
    report_lines.append("─── Protocol Check ───")

    all_24h = coverage_df[(coverage_df["horizon_hours"] == 24) & (coverage_df["subgroup"] == "all")]
    if len(all_24h) > 0:
        full_24 = all_24h.iloc[0]["full_coverage_pct"]
        raw_24 = all_24h.iloc[0]["raw_coverage_pct"]
        verdict_full = "PASS" if full_24 >= MIN_COVERAGE_24H * 100 else "FAIL"
        verdict_raw = "PASS" if raw_24 >= MIN_COVERAGE_24H * 100 else "FAIL"
        report_lines.append(
            f"  24h full_coverage (transfer-adjusted, ≥{MIN_COVERAGE_24H:.0%}): "
            f"{verdict_full} ({full_24:.1f}%)"
        )
        report_lines.append(
            f"  24h raw_coverage  (LOS-only, old definition): "
            f"{verdict_raw} ({raw_24:.1f}%)"
        )
    else:
        full_24 = 0

    # Per-subgroup check for HFpEF
    hfpef_24h = coverage_df[
        (coverage_df["horizon_hours"] == 24) & (coverage_df["subgroup"] == "hfpef")
    ]
    if len(hfpef_24h) > 0:
        hfpef_full = hfpef_24h.iloc[0]["full_coverage_pct"]
        hfpef_raw = hfpef_24h.iloc[0]["raw_coverage_pct"]
        report_lines.append(
            f"  24h HFpEF full_coverage: {hfpef_full:.1f}% | raw: {hfpef_raw:.1f}%"
        )

    # LOS descriptive stats
    report_lines.append("")
    report_lines.append("─── LOS Post-T0 Descriptive Stats ───")
    desc = gated["los_post_t0_hours"].describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95])
    for stat, val in desc.items():
        report_lines.append(f"  {stat}: {val:.1f}h")

    # Early-discharge breakdown
    early_24 = gated[gated["los_post_t0_hours"] < 24]
    n_early = len(early_24)
    n_early_died = int(early_24["_died"].sum())
    n_early_transfer = int(early_24["_is_transfer"].sum())
    n_early_absorb = n_early - n_early_died - n_early_transfer
    report_lines.append("")
    report_lines.append("─── Early Exit Breakdown (LOS post-T0 < 24h) ───")
    early_pct = (n_early / len(gated) * 100) if len(gated) else 0.0
    early_transfer_pct = (n_early_transfer / n_early * 100) if n_early else 0.0
    report_lines.append(f"  Total early exits: {n_early:,} ({early_pct:.1f}%)")
    report_lines.append(f"    Died (absorbing):              {n_early_died:,}")
    report_lines.append(f"    Alive discharge (absorbing):   {n_early_absorb:,}")
    report_lines.append(
        f"    ICU transfer (censored):       {n_early_transfer:,} "
        f"({early_transfer_pct:.1f}% of early exits)"
    )

    # ── Save ──────────────────────────────────────────────────────────────
    csv_path = tables_dir / "30d_los_horizon_coverage.csv"
    coverage_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    report_path = reports_dir / "30d_los_horizon_feasibility_summary.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Saved: {report_path}")

    print("\n[30d] Done.")


if __name__ == "__main__":
    main()
