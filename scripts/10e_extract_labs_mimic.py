#!/usr/bin/env python3
"""
10e_extract_labs_mimic — Extract longitudinal laboratory values from labevents.

Purpose
-------
Extract target lab tests from MIMIC-IV labevents (158M rows) for the ICU cohort.
Outputs a long-format table used by:
  - 25b: pre-T0 hemodynamic gate (lactate ≥ 4 as shock proxy)
  - 30x: feasibility and descriptive tables
  - 40x-45x: covariate construction for causal models

Note: labevents uses d_labitems namespace (NOT d_items used by chartevents).
      hadm_id is used for linkage (labevents lacks stay_id).

Inputs
------
- data/staging/10a_core_icu_stays_mimic.parquet  (hadm_id list + intime)
- data/raw/mimiciv/hosp/labevents.csv.gz          (158M rows)
- data/raw/mimiciv/hosp/d_labitems.csv.gz          (itemid lookup)

Outputs
-------
- data/staging/10e_labs_mimic.parquet
- reports/10e_labs_mimic_summary.txt
"""

from __future__ import annotations

import sys
import textwrap
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, dataset_entrypoint, resolve_path

# ── target lab tests ──────────────────────────────────────────────────────
# Labels matched against d_labitems.label (case-insensitive substring).
# MIMIC-IV labevents itemids are NOT the same as d_items (chartevents).
# We resolve itemids at runtime from d_labitems.
TARGET_LABS: dict[str, dict] = {
    # key: canonical name; label_patterns: substrings matched in d_labitems.label
    # fluid: restrict to "Blood" etc. where ambiguity exists
    # Patterns are regex (pandas str.contains default); use ^...$ for exact match
    # to avoid false positives on substring matches.
    "lactate":       {"label_patterns": [r"^Lactate$"],  "fluid": "Blood",
                      "range": (0.1, 30.0), "unit": "mmol/L"},
    "creatinine":    {"label_patterns": ["Creatinine"],    "fluid": "Blood",
                      "range": (0.1, 30.0), "unit": "mg/dL"},
    "bun":           {"label_patterns": ["Urea Nitrogen"], "fluid": "Blood",
                      "range": (1, 200),    "unit": "mg/dL"},
    "potassium":     {"label_patterns": ["Potassium"],     "fluid": "Blood",
                      "range": (1.5, 10.0), "unit": "mEq/L"},
    "sodium":        {"label_patterns": ["Sodium"],        "fluid": "Blood",
                      "range": (100, 180),  "unit": "mEq/L"},
    "bicarbonate":   {"label_patterns": ["Bicarbonate"],   "fluid": "Blood",
                      "range": (5, 50),     "unit": "mEq/L"},
    "glucose":       {"label_patterns": ["Glucose"],       "fluid": "Blood",
                      "range": (10, 1000),  "unit": "mg/dL"},
    "hemoglobin":    {"label_patterns": [r"^Hemoglobin$"], "fluid": "Blood",
                      "range": (2, 25),     "unit": "g/dL"},
    "wbc":           {"label_patterns": ["White Blood Cells"], "fluid": None,
                      "range": (0.1, 200),  "unit": "K/uL"},
    "platelet":      {"label_patterns": ["Platelet Count"], "fluid": None,
                      "range": (5, 1500),   "unit": "K/uL"},
    "albumin":       {"label_patterns": ["Albumin"],       "fluid": "Blood",
                      "range": (0.5, 6.0),  "unit": "g/dL"},
    "bilirubin_total": {"label_patterns": ["Bilirubin, Total"], "fluid": "Blood",
                      "range": (0.1, 60),   "unit": "mg/dL"},
    "alt":           {"label_patterns": ["Alanine Aminotransferase"], "fluid": None,
                      "range": (1, 10000),  "unit": "IU/L"},
    "ast":           {"label_patterns": ["Aspartate Aminotransferase"], "fluid": None,
                      "range": (1, 10000),  "unit": "IU/L"},
    "troponin_t":    {"label_patterns": ["Troponin T"],    "fluid": None,
                      "range": (0.0, 50),   "unit": "ng/mL"},
    "bnp":           {"label_patterns": ["NTproBNP", "NT-proBNP"],
                      "fluid": "Blood",
                      "range": (1, 70000),  "unit": "pg/mL"},
    "ph":            {"label_patterns": [r"^pH$"],         "fluid": "Blood",
                      "range": (6.5, 8.0),  "unit": "units"},
    "pco2":          {"label_patterns": ["pCO2"],          "fluid": "Blood",
                      "range": (10, 150),   "unit": "mmHg"},
    "po2":           {"label_patterns": ["pO2"],           "fluid": "Blood",
                      "range": (10, 600),   "unit": "mmHg"},
}

CHUNK_SIZE = 5_000_000


def resolve_lab_itemids(d_labitems: pd.DataFrame) -> tuple[dict[int, str], set[int]]:
    """Match target lab labels against d_labitems and return itemid→lab_name mapping."""
    itemid_map: dict[int, str] = {}
    lab_labels = d_labitems["label"].fillna("")
    for lab_name, spec in TARGET_LABS.items():
        for pattern in spec["label_patterns"]:
            mask = lab_labels.str.contains(pattern, case=False, na=False)
            if spec.get("fluid"):
                mask &= d_labitems["fluid"].str.contains(
                    spec["fluid"], case=False, na=False
                )

            # Keep Troponin T specific; avoid pulling Troponin I into T bucket.
            if lab_name == "troponin_t":
                mask &= ~lab_labels.str.contains("Troponin I", case=False, na=False)

            # Keep BNP specific; avoid the common "proBNP ratio" type text matches.
            if lab_name == "bnp":
                mask &= ~lab_labels.str.contains("ratio", case=False, na=False)

            matched = d_labitems.loc[mask]
            for _, row in matched.iterrows():
                itemid_map[int(row["itemid"])] = lab_name
    return itemid_map, set(itemid_map.keys())


def main() -> None:
    cfg = load_config()
    mimic_root = dataset_entrypoint(cfg, "mimic")
    staging_dir = resolve_path(cfg, "staging")
    reports_dir = resolve_path(cfg, "reports")
    staging_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("10e_extract_labs_mimic — Longitudinal laboratory values")
    print("=" * 70)

    # ── load cohort ────────────────────────────────────────────────────
    print("\n[1/5] Loading cohort from 10a ...")
    stays = pd.read_parquet(
        staging_dir / "10a_core_icu_stays_mimic.parquet",
        columns=["subject_id", "hadm_id", "stay_id", "intime"],
    )
    # labevents links by hadm_id (no stay_id column)
    # For multi-ICU-stay admissions we need hadm→stay mapping with intime
    cohort_hadm_ids = set(stays["hadm_id"].unique())
    print(f"  Cohort: {len(cohort_hadm_ids):,} unique hadm_ids")

    # ── resolve itemids from d_labitems ────────────────────────────────
    print("\n[2/5] Resolving lab itemids from d_labitems ...")
    d_labitems = pd.read_csv(mimic_root / "hosp" / "d_labitems.csv.gz")
    itemid_map, target_itemids = resolve_lab_itemids(d_labitems)
    print(f"  Resolved {len(itemid_map)} itemids across {len(TARGET_LABS)} lab types:")
    for iid, lname in sorted(itemid_map.items(), key=lambda x: (x[1], x[0])):
        label = d_labitems.loc[d_labitems["itemid"] == iid, "label"].iloc[0]
        print(f"    {iid:>6d} → {lname:20s} ({label})")

    # ── chunked read of labevents ──────────────────────────────────────
    print(f"\n[3/5] Reading labevents in chunks of {CHUNK_SIZE:,} ...")
    le_path = mimic_root / "hosp" / "labevents.csv.gz"

    chunks_out = []
    n_read = 0
    n_kept = 0
    chunk_i = 0

    reader = pd.read_csv(
        le_path,
        chunksize=CHUNK_SIZE,
        usecols=["hadm_id", "itemid", "charttime", "valuenum", "valueuom"],
        dtype={"hadm_id": "Int64", "itemid": "Int64"},
    )

    for chunk in reader:
        chunk_i += 1
        n_read += len(chunk)

        # Filter to target itemids
        mask = chunk["itemid"].isin(target_itemids)
        filtered = chunk.loc[mask].copy()

        # Filter to cohort hadm_ids
        filtered = filtered[filtered["hadm_id"].isin(cohort_hadm_ids)]

        # Drop rows with null valuenum
        filtered = filtered.dropna(subset=["valuenum"])

        if len(filtered) > 0:
            chunks_out.append(filtered)
            n_kept += len(filtered)

        if chunk_i % 5 == 0:
            print(f"    chunk {chunk_i}: read {n_read:,}, kept {n_kept:,}")

    print(f"  Total read: {n_read:,} rows")
    print(f"  Total kept: {n_kept:,} rows (labs for cohort)")

    # ── assemble and enrich ────────────────────────────────────────────
    print("\n[4/5] Assembling and enriching ...")
    if not chunks_out:
        raise RuntimeError(
            "No labevents matched the requested itemids/cohort; check itemid resolution and inputs."
        )

    labs = pd.concat(chunks_out, ignore_index=True)

    # Parse charttime
    labs["charttime"] = pd.to_datetime(labs["charttime"], errors="coerce")

    # Map itemid to lab name
    labs["lab_name"] = labs["itemid"].map(itemid_map)

    # Link to stay_id and intime via hadm_id
    # For multi-ICU-stay admissions, assign lab to nearest ICU stay by time
    hadm_stays = stays[["hadm_id", "stay_id", "intime"]].copy()
    labs = labs.merge(hadm_stays, on="hadm_id", how="inner")

    # For multi-stay hadm_ids: keep the assignment where charttime is closest
    # to intime (and preferably after intime)
    labs["hours_since_icu_admit"] = (
        (labs["charttime"] - labs["intime"]).dt.total_seconds() / 3600.0
    )

    # For admissions with multiple ICU stays:
    #   1) prefer non-negative assignment (event at/after ICU admit)
    #   2) if all negative, allow up to -24h lookback
    #   3) then choose smallest absolute distance to admit
    labs["is_non_negative"] = labs["hours_since_icu_admit"] >= 0
    labs["within_lookback"] = labs["hours_since_icu_admit"] >= -24
    labs["abs_hours"] = labs["hours_since_icu_admit"].abs()
    labs = labs.sort_values(
        [
            "hadm_id",
            "charttime",
            "itemid",
            "valuenum",
            "is_non_negative",
            "within_lookback",
            "abs_hours",
        ],
        ascending=[True, True, True, True, False, False, True],
    ).drop_duplicates(
        subset=["hadm_id", "charttime", "itemid", "valuenum"],
        keep="first",
    )
    labs.drop(columns=["abs_hours", "is_non_negative", "within_lookback"], inplace=True)

    # Apply physiological range filters
    n_before_range = len(labs)
    range_mask = pd.Series(True, index=labs.index)
    for lab_name, spec in TARGET_LABS.items():
        lo, hi = spec["range"]
        v_mask = labs["lab_name"] == lab_name
        val_mask = labs["valuenum"].between(lo, hi)
        range_mask &= ~v_mask | (v_mask & val_mask)

    labs = labs[range_mask].copy()
    n_range_removed = n_before_range - len(labs)
    print(f"  Range filter removed: {n_range_removed:,} rows "
          f"({n_range_removed / n_before_range * 100:.2f}%)")

    # Select output columns
    out_cols = [
        "stay_id", "hadm_id", "charttime", "itemid", "lab_name",
        "valuenum", "valueuom", "hours_since_icu_admit",
    ]
    labs_out = labs[out_cols].copy()
    labs_out = labs_out.sort_values(
        ["stay_id", "charttime", "lab_name"]
    ).reset_index(drop=True)

    # ── write output ───────────────────────────────────────────────────
    out_path = staging_dir / "10e_labs_mimic.parquet"
    labs_out.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"  → {out_path}  ({len(labs_out):,} rows)")

    # ── summary report ─────────────────────────────────────────────────
    print("\n[5/5] Generating summary report ...")

    lab_counts = labs_out["lab_name"].value_counts().to_string()
    n_stays_with_labs = labs_out["stay_id"].nunique()

    lab_stats_parts = []
    for lname in sorted(labs_out["lab_name"].unique()):
        v = labs_out.loc[labs_out["lab_name"] == lname, "valuenum"]
        lab_stats_parts.append(
            f"  {lname:20s}: n={len(v):>10,}  "
            f"mean={v.mean():8.1f}  sd={v.std():8.1f}  "
            f"[{v.min():.1f}, {v.quantile(0.25):.1f}, "
            f"{v.median():.1f}, {v.quantile(0.75):.1f}, {v.max():.1f}]"
        )
    lab_stats_str = "\n".join(lab_stats_parts)

    report = textwrap.dedent(f"""\
    ================================================================
    10e_extract_labs_mimic — Summary Report
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ================================================================

    EXTRACTION
    ──────────
    Labevents rows read     : {n_read:>14,}
    Rows kept (target labs) : {n_kept:>14,}
    After stay-assignment   : {n_before_range:>14,}
    After range filter      : {len(labs_out):>14,}
    Range-filtered out      : {n_range_removed:>14,}

    ITEMID RESOLUTION
    ─────────────────
    Target lab types        : {len(TARGET_LABS)}
    Resolved itemids        : {len(itemid_map)}

    COVERAGE
    ────────
    Stays with any lab      : {n_stays_with_labs:>10,}
    Coverage                : {n_stays_with_labs / len(stays):>10.1%}

    LAB COUNTS
    ──────────
    {textwrap.indent(lab_counts, '  ')}

    LAB DISTRIBUTIONS
    ──────────────────
    {lab_stats_str}

    OUTPUT
    ──────
    {out_path}  ({len(labs_out):,} rows, {labs_out.shape[1]} cols)
    """)

    report_path = reports_dir / "10e_labs_mimic_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"  → Report: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
