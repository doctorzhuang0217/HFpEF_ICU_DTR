#!/usr/bin/env python3
"""
10g_extract_io_urine_weight_mimic — Extract I/O, urine output, and body weight.

Purpose
-------
Extract fluid output records (outputevents, 5.4M rows) and body-weight
measurements (from chartevents subset) for the ICU cohort.

Used by:
  - 25b: pre-T0 hemodynamic gate (urine output as perfusion indicator)
  - 40b-40d: fluid balance calculation (input from 10f minus output here)
  - 40x: covariate construction (weight-based dosing normalization)

Inputs
------
- data/staging/10a_core_icu_stays_mimic.parquet
- data/raw/mimiciv/icu/outputevents.csv.gz  (5.4M rows)
- data/raw/mimiciv/icu/chartevents.csv.gz   (weight itemids only — small subset)
- data/raw/mimiciv/icu/d_items.csv.gz

Outputs
-------
- data/staging/10g_output_events_mimic.parquet  (urine + other outputs)
- data/staging/10g_weight_mimic.parquet          (body weight records)
- reports/10g_io_urine_weight_mimic_summary.txt
"""

from __future__ import annotations

import sys
import textwrap
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, dataset_entrypoint, resolve_path

# ── urine output itemids (d_items namespace) ──────────────────────────────
# Resolved at runtime; these are common MIMIC-IV itemids for urine.
URINE_LABEL_PATTERNS = [
    "Foley",
    "Void",
    "Urine",
    "Urinary",
    "Condom Cath",
    "Nephrostomy",
    "Suprapubic",
    "Ileoconduit",
]

# ── weight itemids (chartevents, d_items namespace) ───────────────────────
WEIGHT_LABEL_PATTERNS = [
    "Admit Wt",
    "Daily Weight",
    "Admission Weight",
]
WEIGHT_RANGE = (20, 300)  # kg — physiological range filter (applied AFTER lbs→kg)
LBS_TO_KG = 0.453592
# Known lbs itemids — values are converted to kg before range filtering
LBS_WEIGHT_LABEL_PATTERNS = ["lbs", "pounds"]

CHUNK_SIZE = 5_000_000  # for chartevents weight extraction

def extract_weight_from_10d(staging_dir: Path, cohort_stay_ids: set[int], wt_itemids: set[int]) -> pd.DataFrame:
    """Reuse 10d vitals output when present to avoid a full chartevents scan."""
    vitals_path = staging_dir / "10d_vitals_mimic.parquet"
    if not vitals_path.exists():
        return pd.DataFrame(columns=["stay_id", "charttime", "itemid", "valuenum", "valueuom"])

    cols = ["stay_id", "charttime", "itemid", "valuenum", "valueuom"]
    vitals = pd.read_parquet(vitals_path, columns=cols)
    wt = vitals[
        vitals["stay_id"].isin(cohort_stay_ids)
        & vitals["itemid"].isin(wt_itemids)
        & vitals["valuenum"].notna()
    ].copy()
    return wt


def extract_weight_from_chartevents(
    ce_path: Path, cohort_stay_ids: set[int], wt_itemids: set[int]
) -> tuple[pd.DataFrame, int]:
    """Fallback: scan chartevents in chunks and keep only weight rows."""
    wt_chunks = []
    n_ce_read = 0

    reader = pd.read_csv(
        ce_path,
        chunksize=CHUNK_SIZE,
        usecols=["stay_id", "charttime", "itemid", "valuenum", "valueuom"],
        dtype={"stay_id": "Int64", "itemid": "Int64"},
    )
    for chunk in reader:
        n_ce_read += len(chunk)
        filtered = chunk[chunk["itemid"].isin(wt_itemids)].copy()
        filtered = filtered[filtered["stay_id"].isin(cohort_stay_ids)]
        filtered = filtered.dropna(subset=["valuenum"])
        if len(filtered) > 0:
            wt_chunks.append(filtered)

    wt = pd.concat(wt_chunks, ignore_index=True) if wt_chunks else pd.DataFrame(
        columns=["stay_id", "charttime", "itemid", "valuenum", "valueuom"]
    )
    return wt, n_ce_read


def resolve_output_itemids(
    d_items: pd.DataFrame,
) -> tuple[dict[int, str], set[int]]:
    """Resolve urine/output itemids from d_items."""
    oe_items = d_items[d_items["linksto"] == "outputevents"]
    itemid_map: dict[int, str] = {}

    # Urine-specific items
    for pattern in URINE_LABEL_PATTERNS:
        mask = oe_items["label"].str.contains(pattern, case=False, na=False)
        for _, row in oe_items.loc[mask].iterrows():
            itemid_map[int(row["itemid"])] = "urine"

    # All remaining output items (drains, chest tube, etc.)
    for _, row in oe_items.iterrows():
        iid = int(row["itemid"])
        if iid not in itemid_map:
            itemid_map[iid] = "other_output"

    return itemid_map, set(itemid_map.keys())


def resolve_weight_itemids(
    d_items: pd.DataFrame,
) -> tuple[dict[int, str], set[int], set[int]]:
    """Resolve weight itemids from d_items (chartevents namespace).

    Returns:
        itemid_map: itemid → label type
        all_itemids: set of all weight itemids
        lbs_itemids: subset of itemids whose values are in lbs (need conversion)
    """
    ce_items = d_items[d_items["linksto"] == "chartevents"]
    itemid_map: dict[int, str] = {}
    lbs_itemids: set[int] = set()
    for pattern in WEIGHT_LABEL_PATTERNS:
        mask = ce_items["label"].str.contains(pattern, case=False, na=False)
        for _, row in ce_items.loc[mask].iterrows():
            iid = int(row["itemid"])
            label = str(row["label"])
            itemid_map[iid] = "weight_kg"
            # Detect lbs items by label
            if any(p in label.lower() for p in LBS_WEIGHT_LABEL_PATTERNS):
                lbs_itemids.add(iid)
    return itemid_map, set(itemid_map.keys()), lbs_itemids


def main() -> None:
    cfg = load_config()
    mimic_root = dataset_entrypoint(cfg, "mimic")
    staging_dir = resolve_path(cfg, "staging")
    reports_dir = resolve_path(cfg, "reports")
    staging_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("10g_extract_io_urine_weight_mimic — I/O, urine, weight")
    print("=" * 70)

    # ── load cohort ────────────────────────────────────────────────────
    print("\n[1/6] Loading cohort from 10a ...")
    stays = pd.read_parquet(
        staging_dir / "10a_core_icu_stays_mimic.parquet",
        columns=["stay_id", "intime"],
    )
    cohort_stay_ids = set(stays["stay_id"].unique())
    intime_map = stays.set_index("stay_id")["intime"]
    print(f"  Cohort: {len(cohort_stay_ids):,} stay_ids")

    # ── resolve itemids ────────────────────────────────────────────────
    print("\n[2/6] Resolving itemids ...")
    d_items = pd.read_csv(mimic_root / "icu" / "d_items.csv.gz")

    oe_itemid_map, oe_itemids = resolve_output_itemids(d_items)
    wt_itemid_map, wt_itemids, wt_lbs_itemids = resolve_weight_itemids(d_items)

    n_urine_ids = sum(1 for v in oe_itemid_map.values() if v == "urine")
    n_other_ids = sum(1 for v in oe_itemid_map.values() if v == "other_output")
    print(f"  Output itemids: {n_urine_ids} urine, {n_other_ids} other")
    print(f"  Weight itemids: {len(wt_itemid_map)}")
    for iid, wname in sorted(wt_itemid_map.items()):
        label = d_items.loc[d_items["itemid"] == iid, "label"].iloc[0]
        unit_note = " [lbs→kg]" if iid in wt_lbs_itemids else ""
        print(f"    {iid:>6d} → {label}{unit_note}")

    # ── extract outputevents (5.4M rows — fits in memory) ─────────────
    print("\n[3/6] Loading outputevents ...")
    oe = pd.read_csv(
        mimic_root / "icu" / "outputevents.csv.gz",
        usecols=["stay_id", "charttime", "itemid", "value", "valueuom"],
        dtype={"stay_id": "Int64", "itemid": "Int64"},
    )
    n_oe_total = len(oe)
    print(f"  Raw: {n_oe_total:,} rows")

    oe = oe[oe["stay_id"].isin(cohort_stay_ids)].copy()
    oe = oe.dropna(subset=["value"])
    print(f"  After cohort filter + dropna: {len(oe):,} rows")

    # Map itemid to output type
    oe["output_type"] = oe["itemid"].map(oe_itemid_map)
    oe = oe.dropna(subset=["output_type"])

    # Parse charttime
    oe["charttime"] = pd.to_datetime(oe["charttime"], errors="coerce")

    # Add hours_since_icu_admit
    oe = oe.merge(intime_map.reset_index(), on="stay_id", how="left")
    oe["hours_since_icu_admit"] = (
        (oe["charttime"] - oe["intime"]).dt.total_seconds() / 3600.0
    )

    # Remove obviously negative values (output should be ≥ 0)
    n_neg = (oe["value"] < 0).sum()
    oe = oe[oe["value"] >= 0].copy()
    if n_neg > 0:
        print(f"  Removed {n_neg:,} negative output values")

    oe_out_cols = [
        "stay_id", "charttime", "itemid", "output_type",
        "value", "valueuom", "hours_since_icu_admit",
    ]
    oe_out = oe[oe_out_cols].sort_values(
        ["stay_id", "charttime"]
    ).reset_index(drop=True)

    oe_path = staging_dir / "10g_output_events_mimic.parquet"
    oe_out.to_parquet(oe_path, index=False, engine="pyarrow")
    print(f"  → {oe_path}  ({len(oe_out):,} rows)")

    # ── extract weight (prefer 10d reuse, fallback to chartevents scan) ──
    print(f"\n[4/6] Extracting weight records ...")
    wt = extract_weight_from_10d(staging_dir, cohort_stay_ids, wt_itemids)
    n_ce_read = 0
    if len(wt) > 0:
        print(f"  Reused 10d_vitals_mimic.parquet: {len(wt):,} candidate weight rows")
    else:
        ce_path = mimic_root / "icu" / "chartevents.csv.gz"
        print(
            f"  10d output unavailable for weight; scanning chartevents in chunks of {CHUNK_SIZE:,} ..."
        )
        wt, n_ce_read = extract_weight_from_chartevents(ce_path, cohort_stay_ids, wt_itemids)
        print(f"  Chartevents scanned: {n_ce_read:,} rows")

    wt["charttime"] = pd.to_datetime(wt["charttime"], errors="coerce")
    wt["weight_type"] = wt["itemid"].map(wt_itemid_map)

    # Convert lbs → kg for items recorded in pounds
    if wt_lbs_itemids:
        lbs_mask = wt["itemid"].isin(wt_lbs_itemids)
        n_lbs = lbs_mask.sum()
        wt.loc[lbs_mask, "valuenum"] = wt.loc[lbs_mask, "valuenum"] * LBS_TO_KG
        print(f"  Converted {n_lbs:,} lbs records to kg")

    # Range filter (applied after unit conversion)
    n_before = len(wt)
    wt = wt[wt["valuenum"].between(*WEIGHT_RANGE)].copy()
    n_wt_removed = n_before - len(wt)
    if n_wt_removed > 0:
        print(f"  Weight range filter removed: {n_wt_removed:,} rows")

    # Add hours_since_icu_admit
    wt = wt.merge(intime_map.reset_index(), on="stay_id", how="left")
    wt["hours_since_icu_admit"] = (
        (wt["charttime"] - wt["intime"]).dt.total_seconds() / 3600.0
    )

    wt_out_cols = [
        "stay_id", "charttime", "itemid", "weight_type",
        "valuenum", "valueuom", "hours_since_icu_admit",
    ]
    wt_out = wt[wt_out_cols].sort_values(
        ["stay_id", "charttime"]
    ).reset_index(drop=True)

    wt_path = staging_dir / "10g_weight_mimic.parquet"
    wt_out.to_parquet(wt_path, index=False, engine="pyarrow")
    print(f"  → {wt_path}  ({len(wt_out):,} rows)")

    # ── summary report ─────────────────────────────────────────────────
    print("\n[5/6] Generating summary report ...")

    n_oe_stays = oe_out["stay_id"].nunique()
    n_wt_stays = wt_out["stay_id"].nunique() if len(wt_out) > 0 else 0
    urine_only = oe_out[oe_out["output_type"] == "urine"]
    n_urine_stays = urine_only["stay_id"].nunique()

    oe_type_counts = oe_out["output_type"].value_counts().to_string()

    wt_stats = ""
    if len(wt_out) > 0:
        ws = wt_out["valuenum"]
        wt_stats = (
            f"  n={len(ws):,}  mean={ws.mean():.1f} kg  "
            f"sd={ws.std():.1f}  [{ws.min():.1f}, {ws.median():.1f}, {ws.max():.1f}]"
        )

    report = textwrap.dedent(f"""\
    ================================================================
    10g_extract_io_urine_weight_mimic — Summary Report
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ================================================================

    OUTPUT EVENTS
    ─────────────
    Raw outputevents        : {n_oe_total:>12,}
    Cohort-filtered         : {len(oe_out):>12,}
    Stays with output data  : {n_oe_stays:>12,}  ({n_oe_stays / len(cohort_stay_ids) * 100:.1f}%)
    Stays with urine data   : {n_urine_stays:>12,}  ({n_urine_stays / len(cohort_stay_ids) * 100:.1f}%)

    Output type counts:
    {textwrap.indent(oe_type_counts, '  ')}

    BODY WEIGHT
    ───────────
    Weight records          : {len(wt_out):>12,}
    Stays with weight       : {n_wt_stays:>12,}  ({n_wt_stays / len(cohort_stay_ids) * 100:.1f}%)
    Range filter removed    : {n_wt_removed:>12,}
    Weight stats            : {wt_stats}

    OUTPUTS
    ───────
    {oe_path}  ({len(oe_out):,} rows)
    {wt_path}   ({len(wt_out):,} rows)
    """)

    report_path = reports_dir / "10g_io_urine_weight_mimic_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"  → Report: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
