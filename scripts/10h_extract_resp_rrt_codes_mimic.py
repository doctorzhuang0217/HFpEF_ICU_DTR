#!/usr/bin/env python3
"""
10h_extract_resp_rrt_codes_mimic — Respiratory support, RRT/CRRT, and code status.

Purpose
-------
Extract:
  1. Mechanical ventilation / respiratory support markers (procedureevents + chartevents)
  2. RRT/CRRT records (procedureevents)
  3. Code status / goals-of-care signals (procedures_icd + chartevents)

Used by:
  - 20e: hard exclusion (active ECMO → exclude; comfort care → sensitivity)
  - 25b: pre-T0 gate (mechanical ventilation as severity marker)
  - 30x: descriptive tables
  - 40x: time-varying covariates

Inputs
------
- data/staging/10a_core_icu_stays_mimic.parquet
- data/raw/mimiciv/icu/procedureevents.csv.gz   (809K rows)
- data/raw/mimiciv/icu/chartevents.csv.gz        (subset: vent/code-status itemids)
- data/raw/mimiciv/icu/d_items.csv.gz

Outputs
-------
- data/staging/10h_procedures_mimic.parquet       (RRT + vent from procedureevents)
- data/staging/10h_vent_indicators_mimic.parquet   (ventilation from chartevents)
- reports/10h_resp_rrt_codes_mimic_summary.txt
"""

from __future__ import annotations

import sys
import textwrap
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, dataset_entrypoint, resolve_path

# ── ventilation / respiratory support itemids (chartevents) ───────────────
# Resolved at runtime from d_items labels
VENT_LABEL_PATTERNS = {
    "mech_vent":   ["Mechanical Ventilation", "Ventilator Mode"],
    "vent_mode":   ["Ventilator Mode"],
    "fio2":        ["Inspired O2 Fraction"],
    "peep":        ["PEEP set", "PEEP"],
    "tidal_vol":   ["Tidal Volume"],
    "o2_device":   ["O2 Delivery Device"],
    "hfnc":        ["High Flow"],
}

# ── RRT / CRRT patterns (procedureevents ordercategoryname) ───────────────
RRT_PATTERNS = [
    "Dialysis",
    "CRRT",
    "CVVH",
    "CVVHD",
    "CVVHDF",
    "Hemodialysis",
    "IHD",
    "SLED",
    "SCUF",
    "Continuous Renal",
    "Peritoneal Dialysis",
]

# ── code status patterns (chartevents) ────────────────────────────────────
CODE_STATUS_LABEL_PATTERNS = [
    "Code Status",
    "Goals of Care",
    "DNR",
    "Comfort",
]

CHUNK_SIZE = 5_000_000


def resolve_vent_itemids(
    d_items: pd.DataFrame,
) -> tuple[dict[int, str], set[int]]:
    """Resolve ventilation/respiratory chartevents itemids."""
    ce_items = d_items[d_items["linksto"] == "chartevents"]
    itemid_map: dict[int, str] = {}
    for vent_type, patterns in VENT_LABEL_PATTERNS.items():
        for pattern in patterns:
            mask = ce_items["label"].str.contains(pattern, case=False, na=False)
            for _, row in ce_items.loc[mask].iterrows():
                iid = int(row["itemid"])
                if iid not in itemid_map:  # first match wins
                    itemid_map[iid] = vent_type
    return itemid_map, set(itemid_map.keys())


def resolve_code_status_itemids(
    d_items: pd.DataFrame,
) -> tuple[dict[int, str], set[int]]:
    """Resolve code-status chartevents itemids."""
    ce_items = d_items[d_items["linksto"] == "chartevents"]
    itemid_map: dict[int, str] = {}
    for pattern in CODE_STATUS_LABEL_PATTERNS:
        mask = ce_items["label"].str.contains(pattern, case=False, na=False)
        for _, row in ce_items.loc[mask].iterrows():
            itemid_map[int(row["itemid"])] = "code_status"
    return itemid_map, set(itemid_map.keys())


def main() -> None:
    cfg = load_config()
    mimic_root = dataset_entrypoint(cfg, "mimic")
    staging_dir = resolve_path(cfg, "staging")
    reports_dir = resolve_path(cfg, "reports")
    staging_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("10h_extract_resp_rrt_codes_mimic — Resp, RRT, code status")
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

    vent_map, vent_itemids = resolve_vent_itemids(d_items)
    code_map, code_itemids = resolve_code_status_itemids(d_items)
    all_ce_itemids = vent_itemids | code_itemids
    combined_ce_map = {**vent_map, **code_map}

    print(f"  Vent/resp itemids: {len(vent_map)}")
    print(f"  Code status itemids: {len(code_map)}")
    for iid in sorted(vent_map):
        label = d_items.loc[d_items["itemid"] == iid, "label"].iloc[0]
        print(f"    {iid:>6d} → {vent_map[iid]:15s}  ({label})")
    for iid in sorted(code_map):
        label = d_items.loc[d_items["itemid"] == iid, "label"].iloc[0]
        print(f"    {iid:>6d} → code_status       ({label})")

    # ── procedureevents (809K rows — in memory) ───────────────────────
    print("\n[3/6] Loading procedureevents ...")
    pe_cols = [
        "stay_id", "starttime", "endtime", "itemid",
        "value", "valueuom", "ordercategoryname",
        "statusdescription",
    ]
    pe = pd.read_csv(
        mimic_root / "icu" / "procedureevents.csv.gz",
        usecols=pe_cols,
        dtype={"stay_id": "Int64", "itemid": "Int64"},
    )
    n_pe_total = len(pe)
    print(f"  Raw: {n_pe_total:,} rows")

    pe = pe[pe["stay_id"].isin(cohort_stay_ids)].copy()
    print(f"  After cohort filter: {len(pe):,} rows")

    # Classify RRT — use pre-built label lookup for speed
    itemid_label = d_items.set_index("itemid")["label"].str.lower().to_dict()
    rrt_mask = pd.Series(False, index=pe.index)
    for pattern in RRT_PATTERNS:
        rrt_mask |= pe["ordercategoryname"].str.contains(
            pattern, case=False, na=False
        )
        pat_lower = pattern.lower()
        rrt_mask |= pe["itemid"].map(
            lambda x, p=pat_lower: p in itemid_label.get(x, "")
        )

    pe["procedure_type"] = "other"
    pe.loc[rrt_mask, "procedure_type"] = "rrt_crrt"

    # Also flag ventilation-related procedures
    vent_proc_mask = pe["ordercategoryname"].str.contains(
        "Ventilation|Intubation|Extubation", case=False, na=False
    )
    pe.loc[vent_proc_mask & ~rrt_mask, "procedure_type"] = "ventilation"

    # Parse timestamps
    pe["starttime"] = pd.to_datetime(pe["starttime"], errors="coerce")
    pe["endtime"] = pd.to_datetime(pe["endtime"], errors="coerce")

    pe = pe.merge(intime_map.reset_index(), on="stay_id", how="left")
    pe["hours_since_icu_admit"] = (
        (pe["starttime"] - pe["intime"]).dt.total_seconds() / 3600.0
    )

    pe_out_cols = [
        "stay_id", "starttime", "endtime", "itemid",
        "procedure_type", "ordercategoryname",
        "value", "valueuom", "statusdescription",
        "hours_since_icu_admit",
    ]
    pe_out = pe[pe_out_cols].sort_values(
        ["stay_id", "starttime"]
    ).reset_index(drop=True)

    pe_path = staging_dir / "10h_procedures_mimic.parquet"
    pe_out.to_parquet(pe_path, index=False, engine="pyarrow")
    print(f"  → {pe_path}  ({len(pe_out):,} rows)")

    # ── ventilation indicators from chartevents (chunked) ──────────────
    print(f"\n[4/6] Extracting vent/code-status from chartevents ...")
    ce_path = mimic_root / "icu" / "chartevents.csv.gz"

    ce_chunks = []
    n_ce_read = 0

    reader = pd.read_csv(
        ce_path,
        chunksize=CHUNK_SIZE,
        usecols=["stay_id", "charttime", "itemid", "value", "valuenum", "valueuom"],
        dtype={"stay_id": "Int64", "itemid": "Int64"},
    )
    for chunk in reader:
        n_ce_read += len(chunk)
        filtered = chunk[chunk["itemid"].isin(all_ce_itemids)].copy()
        filtered = filtered[filtered["stay_id"].isin(cohort_stay_ids)]
        # Keep rows even without valuenum (text values are important here)
        if len(filtered) > 0:
            ce_chunks.append(filtered)

    print(f"  Chartevents scanned: {n_ce_read:,} rows")

    if ce_chunks:
        vent_ce = pd.concat(ce_chunks, ignore_index=True)
    else:
        vent_ce = pd.DataFrame(
            columns=["stay_id", "charttime", "itemid", "value", "valuenum", "valueuom"]
        )

    print(f"  Vent/code rows extracted: {len(vent_ce):,}")

    vent_ce["charttime"] = pd.to_datetime(vent_ce["charttime"], errors="coerce")
    vent_ce["indicator_type"] = vent_ce["itemid"].map(combined_ce_map)

    vent_ce = vent_ce.merge(intime_map.reset_index(), on="stay_id", how="left")
    vent_ce["hours_since_icu_admit"] = (
        (vent_ce["charttime"] - vent_ce["intime"]).dt.total_seconds() / 3600.0
    )

    vent_out_cols = [
        "stay_id", "charttime", "itemid", "indicator_type",
        "value", "valuenum", "valueuom", "hours_since_icu_admit",
    ]
    vent_out = vent_ce[vent_out_cols].sort_values(
        ["stay_id", "charttime"]
    ).reset_index(drop=True)

    vent_path = staging_dir / "10h_vent_indicators_mimic.parquet"
    vent_out.to_parquet(vent_path, index=False, engine="pyarrow")
    print(f"  → {vent_path}  ({len(vent_out):,} rows)")

    # ── summary report ─────────────────────────────────────────────────
    print("\n[5/6] Generating summary report ...")

    n_rrt_stays = pe_out.loc[
        pe_out["procedure_type"] == "rrt_crrt", "stay_id"
    ].nunique()
    n_vent_proc_stays = pe_out.loc[
        pe_out["procedure_type"] == "ventilation", "stay_id"
    ].nunique()
    n_vent_ce_stays = vent_out["stay_id"].nunique()

    proc_type_counts = pe_out["procedure_type"].value_counts().to_string()
    vent_type_counts = vent_out["indicator_type"].value_counts().to_string()

    report = textwrap.dedent(f"""\
    ================================================================
    10h_extract_resp_rrt_codes_mimic — Summary Report
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ================================================================

    PROCEDURE EVENTS
    ─────────────────
    Raw procedureevents     : {n_pe_total:>12,}
    Cohort-filtered         : {len(pe_out):>12,}

    Procedure type counts:
    {textwrap.indent(proc_type_counts, '  ')}

    Stays with RRT/CRRT     : {n_rrt_stays:>10,}  ({n_rrt_stays / len(cohort_stay_ids) * 100:.1f}%)
    Stays with vent proc    : {n_vent_proc_stays:>10,}  ({n_vent_proc_stays / len(cohort_stay_ids) * 100:.1f}%)

    VENTILATION / CODE STATUS INDICATORS (from chartevents)
    ────────────────────────────────────────────────────────
    Total rows              : {len(vent_out):>12,}
    Stays with indicators   : {n_vent_ce_stays:>10,}  ({n_vent_ce_stays / len(cohort_stay_ids) * 100:.1f}%)

    Indicator type counts:
    {textwrap.indent(vent_type_counts, '  ')}

    OUTPUTS
    ───────
    {pe_path}       ({len(pe_out):,} rows)
    {vent_path}  ({len(vent_out):,} rows)
    """)

    report_path = reports_dir / "10h_resp_rrt_codes_mimic_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"  → Report: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
