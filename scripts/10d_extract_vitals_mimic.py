#!/usr/bin/env python3
"""
10d_extract_vitals_mimic — Extract longitudinal vital signs from chartevents.

Purpose
-------
Extract hemodynamic and respiratory vital signs from MIMIC-IV chartevents
(433M rows) for the ICU cohort.  Outputs a long-format table used by:
  - 25b: pre-T0 hemodynamic gate (MAP, SBP in 0-6h window)
  - 30x: feasibility and descriptive tables
  - 40x-45x: covariate construction for causal models

Strategy for handling the large chartevents table:
  - Read in chunks to limit memory
  - Filter to target itemids (vitals only) during read
  - Scope to cohort stay_ids from 10a

Inputs
------
- data/staging/10a_core_icu_stays_mimic.parquet  (stay_id list + intime)
- data/raw/mimiciv/icu/chartevents.csv.gz         (433M rows)
- data/raw/mimiciv/icu/d_items.csv.gz             (itemid lookup)

Outputs
-------
- data/staging/10d_vitals_mimic.parquet
- reports/10d_vitals_mimic_summary.txt
"""

from __future__ import annotations

import sys
import textwrap
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── project imports ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, dataset_entrypoint, resolve_path

# ── target vital-sign itemids (MIMIC-IV standard) ─────────────────────────
# These will be verified against d_items at runtime.
VITAL_ITEMIDS: dict[str, list[int]] = {
    # Blood pressure — arterial line (invasive)
    "sbp_invasive":  [220050],
    "dbp_invasive":  [220051],
    "map_invasive":  [220052],
    # Blood pressure — non-invasive (cuff)
    "sbp_noninvasive":  [220179],
    "dbp_noninvasive":  [220180],
    "map_noninvasive":  [220181],
    # Heart rate
    "heart_rate":    [220045],
    # SpO2
    "spo2":          [220277],
    # CVP
    "cvp":           [220074],
    # Respiratory rate
    "resp_rate":     [220210],
    # Temperature
    "temperature_f": [223761],
    "temperature_c": [223762],
    # FiO2
    "fio2":          [223835],
}

# Flatten for filtering
ALL_VITAL_ITEMIDS = set()
ITEMID_TO_VITAL: dict[int, str] = {}
for vital_name, ids in VITAL_ITEMIDS.items():
    for iid in ids:
        ALL_VITAL_ITEMIDS.add(iid)
        ITEMID_TO_VITAL[iid] = vital_name

# Physiological range filters (obvious data errors)
RANGE_FILTERS: dict[str, tuple[float, float]] = {
    "sbp_invasive":     (20, 300),
    "dbp_invasive":     (5, 225),
    "map_invasive":     (10, 250),
    "sbp_noninvasive":  (20, 300),
    "dbp_noninvasive":  (5, 225),
    "map_noninvasive":  (10, 250),
    "heart_rate":       (15, 300),
    "spo2":             (50, 100),
    "cvp":              (-10, 40),
    "resp_rate":        (2, 70),
    "temperature_f":    (85, 115),
    "temperature_c":    (28, 45),
    "fio2":             (21, 100),
}

CHUNK_SIZE = 5_000_000  # rows per chunk for chartevents


def main() -> None:
    cfg = load_config()
    mimic_root = dataset_entrypoint(cfg, "mimic")
    staging_dir = resolve_path(cfg, "staging")
    reports_dir = resolve_path(cfg, "reports")
    staging_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("10d_extract_vitals_mimic — Longitudinal vital signs")
    print("=" * 70)

    # ── load cohort stay_ids and intimes ───────────────────────────────
    print("\n[1/5] Loading cohort from 10a ...")
    stays = pd.read_parquet(
        staging_dir / "10a_core_icu_stays_mimic.parquet",
        columns=["stay_id", "intime"],
    )
    cohort_stay_ids = set(stays["stay_id"].unique())
    intime_map = stays.set_index("stay_id")["intime"]
    print(f"  Cohort: {len(cohort_stay_ids):,} stay_ids")

    # ── verify itemids against d_items ─────────────────────────────────
    print("\n[2/5] Verifying itemids against d_items ...")
    d_items = pd.read_csv(mimic_root / "icu" / "d_items.csv.gz")
    found = d_items[d_items["itemid"].isin(ALL_VITAL_ITEMIDS)]
    print(f"  Target itemids: {len(ALL_VITAL_ITEMIDS)}")
    print(f"  Found in d_items: {len(found)}")
    missing_ids = ALL_VITAL_ITEMIDS - set(found["itemid"])
    if missing_ids:
        print(f"  WARNING: itemids not in d_items: {missing_ids}")
    for _, row in found.iterrows():
        print(f"    {row['itemid']:>6d} → {row['label']}")

    # ── chunked read of chartevents ────────────────────────────────────
    print(f"\n[3/5] Reading chartevents in chunks of {CHUNK_SIZE:,} ...")
    ce_path = mimic_root / "icu" / "chartevents.csv.gz"

    chunks_out = []
    n_read = 0
    n_kept = 0
    chunk_i = 0

    reader = pd.read_csv(
        ce_path,
        chunksize=CHUNK_SIZE,
        usecols=["stay_id", "charttime", "itemid", "valuenum", "valueuom"],
        dtype={"stay_id": "Int64", "itemid": "Int64"},
    )

    for chunk in reader:
        chunk_i += 1
        n_read += len(chunk)

        # Filter to target itemids
        mask = chunk["itemid"].isin(ALL_VITAL_ITEMIDS)
        filtered = chunk.loc[mask].copy()

        # Filter to cohort stay_ids
        filtered = filtered[filtered["stay_id"].isin(cohort_stay_ids)]

        # Drop rows with null valuenum
        filtered = filtered.dropna(subset=["valuenum"])

        if len(filtered) > 0:
            chunks_out.append(filtered)
            n_kept += len(filtered)

        if chunk_i % 10 == 0:
            print(f"    chunk {chunk_i}: read {n_read:,}, kept {n_kept:,}")

    print(f"  Total read: {n_read:,} rows")
    print(f"  Total kept: {n_kept:,} rows (vitals for cohort)")

    # ── assemble and enrich ────────────────────────────────────────────
    print("\n[4/5] Assembling and enriching ...")
    vitals = pd.concat(chunks_out, ignore_index=True)

    # Parse charttime
    vitals["charttime"] = pd.to_datetime(vitals["charttime"], errors="coerce")

    # Map itemid to vital name
    vitals["vital_name"] = vitals["itemid"].map(ITEMID_TO_VITAL)

    # Time since ICU admission (hours)
    vitals = vitals.merge(
        intime_map.reset_index(), on="stay_id", how="left"
    )
    vitals["hours_since_icu_admit"] = (
        (vitals["charttime"] - vitals["intime"]).dt.total_seconds() / 3600.0
    )

    # Apply physiological range filters
    n_before_range = len(vitals)
    range_mask = pd.Series(True, index=vitals.index)
    for vname, (lo, hi) in RANGE_FILTERS.items():
        v_mask = vitals["vital_name"] == vname
        val_mask = vitals["valuenum"].between(lo, hi)
        range_mask &= ~v_mask | (v_mask & val_mask)

    vitals = vitals[range_mask].copy()
    n_range_removed = n_before_range - len(vitals)
    print(f"  Range filter removed: {n_range_removed:,} rows "
          f"({n_range_removed / n_before_range * 100:.2f}%)")

    # Select and order output columns
    out_cols = [
        "stay_id", "charttime", "itemid", "vital_name",
        "valuenum", "valueuom", "hours_since_icu_admit",
    ]
    vitals_out = vitals[out_cols].copy()
    vitals_out = vitals_out.sort_values(
        ["stay_id", "charttime", "vital_name"]
    ).reset_index(drop=True)

    # ── write output ───────────────────────────────────────────────────
    out_path = staging_dir / "10d_vitals_mimic.parquet"
    vitals_out.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"  → {out_path}  ({len(vitals_out):,} rows)")

    # ── summary report ─────────────────────────────────────────────────
    print("\n[5/5] Generating summary report ...")

    vital_counts = vitals_out["vital_name"].value_counts().to_string()
    n_stays_with_vitals = vitals_out["stay_id"].nunique()
    stays_no_vitals = cohort_stay_ids - set(vitals_out["stay_id"].unique())

    # Per-vital descriptive stats
    vital_stats_parts = []
    for vname in sorted(vitals_out["vital_name"].unique()):
        v = vitals_out.loc[vitals_out["vital_name"] == vname, "valuenum"]
        vital_stats_parts.append(
            f"  {vname:25s}: n={len(v):>10,}  "
            f"mean={v.mean():7.1f}  sd={v.std():7.1f}  "
            f"[{v.min():.1f}, {v.quantile(0.25):.1f}, "
            f"{v.median():.1f}, {v.quantile(0.75):.1f}, {v.max():.1f}]"
        )
    vital_stats_str = "\n".join(vital_stats_parts)

    report = textwrap.dedent(f"""\
    ================================================================
    10d_extract_vitals_mimic — Summary Report
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ================================================================

    EXTRACTION
    ──────────
    Chartevents rows read   : {n_read:>14,}
    Rows kept (vitals)      : {n_kept:>14,}
    After range filter      : {len(vitals_out):>14,}
    Range-filtered out      : {n_range_removed:>14,}

    COVERAGE
    ────────
    Stays with any vital    : {n_stays_with_vitals:>10,}
    Stays without vitals    : {len(stays_no_vitals):>10,}
    Coverage                : {n_stays_with_vitals / len(cohort_stay_ids) * 100:.1f}%

    VITAL SIGN COUNTS
    ──────────────────
    {textwrap.indent(vital_counts, '  ')}

    VITAL SIGN DISTRIBUTIONS
    ─────────────────────────
    {vital_stats_str}

    OUTPUT
    ──────
    {out_path}  ({len(vitals_out):,} rows, {vitals_out.shape[1]} cols)
    """)

    report_path = reports_dir / "10d_vitals_mimic_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"  → Report: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
