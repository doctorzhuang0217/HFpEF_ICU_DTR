#!/usr/bin/env python3
"""
10f_extract_meds_infusions_mimic — Extract vasopressors, inotropes, diuretics, IV fluids.

Purpose
-------
Extract ICU medication infusion records (inputevents, 11M rows) for:
  - Vasopressors: norepinephrine, vasopressin, epinephrine, phenylephrine, dopamine
  - Inotropes: dobutamine, milrinone
  - Loop diuretics: furosemide, bumetanide, torsemide
  - IV fluids: crystalloids (NS, LR), colloids (albumin)

Used by:
  - 25b: pre-T0 hemodynamic gate (vasopressor/diuretic/fluid in 0-6h)
  - 40d-40f: action variable construction (fluid volumes, NE-equivalence, diuretic dose)

Note: inputevents uses d_items (ICU) namespace and links by stay_id.
      Table is ~11M rows — fits in memory without chunking.

Inputs
------
- data/staging/10a_core_icu_stays_mimic.parquet
- data/raw/mimiciv/icu/inputevents.csv.gz
- data/raw/mimiciv/icu/d_items.csv.gz

Outputs
-------
- data/staging/10f_meds_infusions_mimic.parquet
- reports/10f_meds_infusions_mimic_summary.txt
"""

from __future__ import annotations

import sys
import textwrap
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, dataset_entrypoint, resolve_path

# ── target medication categories ──────────────────────────────────────────
# Resolved at runtime by matching d_items.label against these patterns.
# ordercategoryname in inputevents is also used to cross-validate.
MED_CATEGORIES: dict[str, dict] = {
    # Vasopressors
    "norepinephrine":  {"label_patterns": ["Norepinephrine", "Levophed"],
                        "med_class": "vasopressor"},
    "epinephrine":     {"label_patterns": ["Epinephrine"],
                        "med_class": "vasopressor"},
    "vasopressin":     {"label_patterns": ["Vasopressin"],
                        "med_class": "vasopressor"},
    "phenylephrine":   {"label_patterns": ["Phenylephrine"],
                        "med_class": "vasopressor"},
    "dopamine":        {"label_patterns": ["Dopamine"],
                        "med_class": "vasopressor"},
    # Inotropes
    "dobutamine":      {"label_patterns": ["Dobutamine"],
                        "med_class": "inotrope"},
    "milrinone":       {"label_patterns": ["Milrinone"],
                        "med_class": "inotrope"},
    # Loop diuretics
    "furosemide":      {"label_patterns": ["Furosemide", "Lasix"],
                        "med_class": "loop_diuretic"},
    "bumetanide":      {"label_patterns": ["Bumetanide", "Bumex"],
                        "med_class": "loop_diuretic"},
    "torsemide":       {"label_patterns": ["Torsemide"],
                        "med_class": "loop_diuretic"},
}

# IV fluid categories — matched by ordercategoryname in inputevents
FLUID_CATEGORIES: dict[str, list[str]] = {
    "crystalloid":  ["Normal Saline", "Lactated Ringers",
                     "Dextrose 5%", "D5W", "0.9% Sodium Chloride",
                     "Sodium Chloride 0.9%", "Ringer"],
    "colloid":      ["Albumin 5%", "Albumin 25%", "Albumin"],
}

# Dialysis solutions are not exposure fluids for treatment action construction.
FLUID_EXCLUDE_PATTERNS = ["dialys", "cvvh", "cvvhd", "rrt", "crrt"]

# Generic fluid categories to catch if free-text category is broad.
GENERIC_FLUID_CATEGORY_PATTERNS = ["iv", "fluid", "hydration"]




def resolve_med_itemids(
    d_items: pd.DataFrame,
) -> tuple[dict[int, str], dict[int, str]]:
    """Match target medications against d_items by label pattern.

    Returns:
        itemid_to_med: mapping of itemid → canonical med name
        itemid_to_class: mapping of itemid → med_class
    """
    itemid_to_med: dict[int, str] = {}
    itemid_to_class: dict[int, str] = {}
    # Restrict to inputevents-linked items
    ie_items = d_items[d_items["linksto"] == "inputevents"]
    for med_name, spec in MED_CATEGORIES.items():
        for pattern in spec["label_patterns"]:
            mask = ie_items["label"].str.contains(pattern, case=False, na=False)
            for _, row in ie_items.loc[mask].iterrows():
                iid = int(row["itemid"])
                if iid not in itemid_to_med:  # first match wins; prevents NE→epi
                    itemid_to_med[iid] = med_name
                    itemid_to_class[iid] = spec["med_class"]
    return itemid_to_med, itemid_to_class


def main() -> None:
    cfg = load_config()
    mimic_root = dataset_entrypoint(cfg, "mimic")
    staging_dir = resolve_path(cfg, "staging")
    reports_dir = resolve_path(cfg, "reports")
    staging_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("10f_extract_meds_infusions_mimic — Meds & infusions")
    print("=" * 70)

    # ── load cohort ────────────────────────────────────────────────────
    print("\n[1/5] Loading cohort from 10a ...")
    stays = pd.read_parquet(
        staging_dir / "10a_core_icu_stays_mimic.parquet",
        columns=["stay_id", "intime"],
    )
    cohort_stay_ids = set(stays["stay_id"].unique())
    intime_map = stays.set_index("stay_id")["intime"]
    print(f"  Cohort: {len(cohort_stay_ids):,} stay_ids")

    # ── resolve medication itemids ─────────────────────────────────────
    print("\n[2/5] Resolving medication itemids ...")
    d_items = pd.read_csv(mimic_root / "icu" / "d_items.csv.gz")
    itemid_to_med, itemid_to_class = resolve_med_itemids(d_items)
    print(f"  Resolved {len(itemid_to_med)} itemids:")
    for iid, mname in sorted(itemid_to_med.items(), key=lambda x: (x[1], x[0])):
        label = d_items.loc[d_items["itemid"] == iid, "label"].iloc[0]
        cls = itemid_to_class[iid]
        print(f"    {iid:>6d} → {mname:20s} [{cls}]  ({label})")

    # ── load inputevents (fits in memory: ~11M rows) ──────────────────
    print("\n[3/5] Loading inputevents ...")
    ie_cols = [
        "stay_id", "starttime", "endtime", "itemid",
        "amount", "amountuom", "rate", "rateuom",
        "ordercategoryname", "patientweight", "statusdescription",
    ]
    ie = pd.read_csv(
        mimic_root / "icu" / "inputevents.csv.gz",
        usecols=ie_cols,
        dtype={"stay_id": "Int64", "itemid": "Int64"},
    )
    n_ie_total = len(ie)
    print(f"  Raw inputevents: {n_ie_total:,} rows")

    # Scope to cohort
    ie = ie[ie["stay_id"].isin(cohort_stay_ids)].copy()
    print(f"  After cohort filter: {len(ie):,} rows")

    # Exclude 'Rewritten' orders (superseded by a newer order — keeping them
    # double-counts the actual administered volume/dose)
    n_rewritten = (ie["statusdescription"] == "Rewritten").sum()
    ie = ie[ie["statusdescription"] != "Rewritten"].copy()
    print(f"  Removed {n_rewritten:,} Rewritten rows → {len(ie):,} remaining")

    # ── classify records ───────────────────────────────────────────────
    print("\n[4/5] Classifying medications and fluids ...")

    # Medications (vasopressors, inotropes, diuretics) — by itemid
    med_mask = ie["itemid"].isin(itemid_to_med)
    meds = ie[med_mask].copy()
    meds["med_name"] = meds["itemid"].map(itemid_to_med)
    meds["med_class"] = meds["itemid"].map(itemid_to_class)
    meds["record_type"] = "medication"
    print(f"  Medications matched: {len(meds):,} rows")

    # IV Fluids — by ordercategoryname pattern matching
    non_med = ie[~med_mask].copy()

    # Build a label lookup for d_items (for additional matching)
    itemid_label = d_items.set_index("itemid")["label"].fillna("").str.lower().to_dict()
    item_label_series = non_med["itemid"].map(itemid_label).fillna("")
    ordercat_series = non_med["ordercategoryname"].fillna("").str.lower()
    exclusion_regex = "|".join(FLUID_EXCLUDE_PATTERNS)
    excluded_mask = (
        ordercat_series.str.contains(exclusion_regex, case=False, na=False)
        | item_label_series.str.contains(exclusion_regex, case=False, na=False)
    )

    fluid_records = []
    matched_indices: set = set()
    for fluid_class, patterns in FLUID_CATEGORIES.items():
        for pattern in patterns:
            pat_lower = pattern.lower()
            # Match ordercategoryname
            fmask = non_med["ordercategoryname"].str.contains(
                pattern, case=False, na=False
            )
            # Also match d_items label
            fmask |= non_med["itemid"].map(
                lambda x, p=pat_lower: p in itemid_label.get(x, "")
            )
            # Exclude dialysis solutions and already matched rows
            fmask &= ~excluded_mask
            fmask &= ~non_med.index.isin(matched_indices)
            matched = non_med[fmask].copy()
            if len(matched) > 0:
                matched["med_name"] = fluid_class
                matched["med_class"] = "iv_fluid"
                matched["record_type"] = "fluid"
                fluid_records.append(matched)
                matched_indices.update(matched.index)

    # Also catch remaining "IV Fluids" / "Fluids" category generically
    generic_fluid_regex = "|".join(GENERIC_FLUID_CATEGORY_PATTERNS)
    generic_fluid_mask = non_med["ordercategoryname"].str.contains(
        generic_fluid_regex, case=False, na=False
    )
    generic_fluids = non_med[
        generic_fluid_mask & ~excluded_mask & ~non_med.index.isin(matched_indices)
    ].copy()
    if len(generic_fluids) > 0:
        generic_fluids["med_name"] = "other_iv_fluid"
        generic_fluids["med_class"] = "iv_fluid"
        generic_fluids["record_type"] = "fluid"
        fluid_records.append(generic_fluids)

    if fluid_records:
        fluids = pd.concat(fluid_records, ignore_index=False)
        print(f"  IV fluids matched: {len(fluids):,} rows")
    else:
        fluids = pd.DataFrame(columns=meds.columns)
        print("  IV fluids matched: 0 rows")

    # Combine
    combined = pd.concat([meds, fluids], ignore_index=True)

    # Parse timestamps
    combined["starttime"] = pd.to_datetime(combined["starttime"], errors="coerce")
    combined["endtime"] = pd.to_datetime(combined["endtime"], errors="coerce")

    # Add hours_since_icu_admit
    combined = combined.merge(
        intime_map.reset_index(), on="stay_id", how="left"
    )
    combined["hours_since_icu_admit"] = (
        (combined["starttime"] - combined["intime"]).dt.total_seconds() / 3600.0
    )
    combined["duration_hours"] = (
        (combined["endtime"] - combined["starttime"]).dt.total_seconds() / 3600.0
    )

    # Select output columns
    out_cols = [
        "stay_id", "starttime", "endtime", "itemid",
        "med_name", "med_class", "record_type",
        "amount", "amountuom", "rate", "rateuom",
        "ordercategoryname", "patientweight",
        "hours_since_icu_admit", "duration_hours",
    ]
    out = combined[out_cols].copy()
    out = out.sort_values(["stay_id", "starttime", "med_name"]).reset_index(drop=True)

    # ── write output ───────────────────────────────────────────────────
    out_path = staging_dir / "10f_meds_infusions_mimic.parquet"
    out.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"  → {out_path}  ({len(out):,} rows)")

    # ── summary report ─────────────────────────────────────────────────
    print("\n[5/5] Generating summary report ...")

    class_counts = out.groupby(["med_class", "med_name"]).size().to_string()
    n_stays = out["stay_id"].nunique()
    stays_by_class = out.groupby("med_class")["stay_id"].nunique().to_string()

    report = textwrap.dedent(f"""\
    ================================================================
    10f_extract_meds_infusions_mimic — Summary Report
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ================================================================

    EXTRACTION
    ──────────
    Total inputevents rows  : {n_ie_total:>12,}
    Rewritten rows removed  : {n_rewritten:>12,}
    Cohort-filtered rows    : {len(ie):>12,}
    Medication records      : {len(meds):>12,}
    Fluid records           : {len(fluids):>12,}
    Total output records    : {len(out):>12,}

    COVERAGE
    ────────
    Stays with any record   : {n_stays:>10,}  ({n_stays / len(cohort_stay_ids) * 100:.1f}%)

    Stays by med_class:
    {textwrap.indent(stays_by_class, '  ')}

    RECORD COUNTS (by class and name)
    ──────────────────────────────────
    {textwrap.indent(class_counts, '  ')}

    OUTPUT
    ──────
    {out_path}  ({len(out):,} rows, {out.shape[1]} cols)
    """)

    report_path = reports_dir / "10f_meds_infusions_mimic_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"  → Report: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
