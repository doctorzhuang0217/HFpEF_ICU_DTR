#!/usr/bin/env python3
"""
20a_build_hf_labels_mimic — HF candidate flags, EF typing, HFpEF flag, comparators.

Purpose
-------
Consolidated implementation of Steps 20a–20d:
  20a: Build HF candidate flags from ICD diagnosis codes
  20b: Extract ejection fraction from charted ICU observations and classify HF subtypes
  20c: Build primary HFpEF flag (strict: HF dx + EF >= 50%)
  20d: Assign comparator labels (HFpEF / HFmrEF / HFrEF / non-HF)

Output is a single stay-level table with all HF phenotyping columns.

Scientific framing (from protocol):
  - MIMIC-IV uses "strict" HFpEF ascertainment: HF diagnosis + EF >= 50%
  - Primary comparators: HFpEF vs HFrEF/HFmrEF; HFpEF vs non-HF
  - Non-HF = absent HF diagnosis codes (I50.x / 428.x)
  - EF source hierarchy (highest priority first):
    1. chartevents structured EF (itemids 227008, 226272) — sparse but exact
    2. mimic-iv-note discharge summaries — NLP regex extraction of EF values
    3. diastolic-HF ICD proxy — fallback when no numeric EF available

Inputs
------
- data/staging/10a_core_icu_stays_mimic.parquet  (stay backbone)
- data/staging/10b_diagnoses_mimic.parquet        (ICD diagnosis codes)
- data/raw/mimiciv/icu/chartevents.csv.gz         (charted EF values)
- data/raw/mimiciv/icu/d_items.csv.gz             (item dictionary for EF lookup)
- data/raw/mimiciv/note/2.2/note/discharge.csv.gz (discharge summaries for NLP EF)

Outputs
-------
- data/staging/20a_hf_labels_mimic.parquet
- reports/20a_hf_labels_mimic_summary.txt
"""

from __future__ import annotations

import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, dataset_entrypoint, resolve_path

# ── HF ICD code definitions ─────────────────────────────────────────────
# ICD-10: I50.x  (heart failure)
# ICD-9:  428.x  (heart failure)
#
# Diastolic-specific subcodes (for sensitivity: ICD-only HFpEF proxy):
#   ICD-10: I50.3x (diastolic HF), I50.31/32/33
#   ICD-9:  428.3x (diastolic HF), 428.30/31/32/33
DIASTOLIC_ICD10 = {"I5030", "I5031", "I5032", "I5033"}
DIASTOLIC_ICD9 = {"42830", "42831", "42832", "42833"}

# ── EF classification thresholds (ESC / AHA) ────────────────────────────
EF_HFPEF_THRESHOLD = 50    # EF >= 50% → HFpEF
EF_HFMREF_LOWER = 40       # 40 <= EF < 50 → HFmrEF
# EF < 40 → HFrEF


def extract_hf_flags(dx: pd.DataFrame) -> pd.DataFrame:
    """Build HF candidate flags from ICD diagnosis codes.

    Returns per-hadm_id DataFrame with:
        has_hf: any HF ICD code present
        has_hf_icd10: any I50.x code present
        has_hf_icd9: any 428.x code present
        has_diastolic_hf_icd: diastolic-specific code present
        hf_icd_codes: semicolon-joined list of HF ICD codes
    """
    dx = dx.copy()
    dx["icd_code"] = dx["icd_code"].fillna("").astype(str)

    # ICD-10 HF
    hf10 = dx[(dx["icd_version"] == 10) & dx["icd_code"].str.startswith("I50")].copy()
    # ICD-9 HF
    hf9 = dx[(dx["icd_version"] == 9) & dx["icd_code"].str.startswith("428")].copy()
    hf_all = pd.concat([hf10, hf9], ignore_index=True)

    if len(hf_all) == 0:
        return pd.DataFrame(columns=[
            "hadm_id", "has_hf", "has_hf_icd10", "has_hf_icd9",
            "has_diastolic_hf_icd", "hf_icd_codes",
        ])

    # Per-admission aggregation
    hadm_hf = hf_all.groupby("hadm_id").agg(
        has_hf_icd10=("icd_version", lambda s: (s == 10).any()),
        has_hf_icd9=("icd_version", lambda s: (s == 9).any()),
        has_diastolic_hf_icd=("icd_code", lambda s: bool(
            set(s.str.replace(".", "", regex=False)) & (DIASTOLIC_ICD10 | DIASTOLIC_ICD9)
        )),
        hf_icd_codes=("icd_code", lambda s: ";".join(sorted(s.unique()))),
    ).reset_index()
    hadm_hf["has_hf"] = True

    return hadm_hf


def _resolve_ef_itemids(d_items: pd.DataFrame) -> dict[int, str]:
    """Find EF-related itemids in d_items by label pattern matching.

    MIMIC-IV d_items labels for EF vary; known items include:
      - 227008: "Ejection Fraction"  (APACHE IV score component)
      - 226272: "EF (CCO)"           (Continuous Cardiac Output, Hemodynamics)
    Returns dict mapping itemid → label for matched items.
    """
    # Restrict to chartevents dictionary entries when d_items includes linksto.
    if "linksto" in d_items.columns:
        d_items = d_items[d_items["linksto"] == "chartevents"].copy()

    labels = d_items["label"].fillna("").astype(str)
    abbreviations = (
        d_items["abbreviation"].fillna("").astype(str)
        if "abbreviation" in d_items.columns
        else pd.Series("", index=d_items.index)
    )
    categories = (
        d_items["category"].fillna("").astype(str)
        if "category" in d_items.columns
        else pd.Series("", index=d_items.index)
    )

    # Pass 1: direct cardiac EF naming.
    direct_mask = labels.str.contains(r"(?i)ejection\s+fraction|\bLVEF\b", na=False)

    # Pass 2: standalone "EF" abbreviation only when accompanied by cardiac context.
    # Check label, abbreviation, AND category for cardiac context (e.g. 226272
    # "EF (CCO)" has label without cardiac keywords but category "Hemodynamics").
    ef_token_mask = (labels + " " + abbreviations).str.contains(
        r"(?i)(?:^|[^a-z0-9])EF(?:[^a-z0-9]|$)", na=False
    )
    cardiac_context_mask = (labels + " " + categories).str.contains(
        r"(?i)hemodyn|cardi|echo|ventric|CCO|cardiac.output", na=False
    )

    hits = d_items.loc[direct_mask | (ef_token_mask & cardiac_context_mask), ["itemid", "label"]].copy()
    mapping = dict(zip(hits["itemid"], hits["label"]))
    return mapping


def extract_ef_from_chartevents(
    mimic_root: Path, subject_ids: set[int], stay_ids: set[int], hadm_ids: set[int]
) -> pd.DataFrame:
    """Extract ejection fraction from MIMIC-IV chartevents via d_items lookup.

    Returns per-hadm_id DataFrame with:
        ef_value: most recent numeric EF (%)
        ef_date: charttime of the EF measurement
        ef_source: d_items label that matched
    """
    # ── resolve EF itemids from d_items ──
    d_items = pd.read_csv(mimic_root / "icu" / "d_items.csv.gz")
    ef_itemids = _resolve_ef_itemids(d_items)
    print(f"    d_items EF itemids found: {len(ef_itemids)}")
    for iid, label in sorted(ef_itemids.items()):
        print(f"      {iid}: {label}")

    if len(ef_itemids) == 0:
        print("    WARNING: No EF-related itemids found in d_items")
        return pd.DataFrame(columns=[
            "hadm_id", "ef_value", "ef_date", "ef_source",
        ])

    # ── scan chartevents for matching rows ──
    ef_rows = []
    ce_path = mimic_root / "icu" / "chartevents.csv.gz"
    target_ids = set(ef_itemids.keys())
    for chunk in pd.read_csv(
        ce_path,
        usecols=["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum"],
        chunksize=2_000_000,
    ):
        mask = (
            chunk["itemid"].isin(target_ids)
            & chunk["stay_id"].isin(stay_ids)
            & chunk["hadm_id"].isin(hadm_ids)
            & chunk["subject_id"].isin(subject_ids)
        )
        matched = chunk.loc[mask].copy()
        if len(matched) > 0:
            ef_rows.append(matched)

    if len(ef_rows) == 0:
        print("    No EF rows found in chartevents")
        return pd.DataFrame(columns=[
            "hadm_id", "ef_value", "ef_date", "ef_source",
        ])

    ef = pd.concat(ef_rows, ignore_index=True)
    print(f"    Raw EF chartevents rows: {len(ef):,}")

    # ── parse and filter ──
    ef["valuenum"] = pd.to_numeric(ef["valuenum"], errors="coerce")
    ef = ef[ef["valuenum"].between(5, 90)].copy()
    print(f"    After range filter (5-90%): {len(ef):,}")

    ef["charttime"] = pd.to_datetime(ef["charttime"], errors="coerce")
    ef["ef_source"] = ef["itemid"].map(ef_itemids)

    # Keep most recent EF per admission to avoid cross-admission leakage.
    ef = ef.sort_values(
        ["hadm_id", "charttime"], ascending=[True, False]
    ).drop_duplicates(subset=["hadm_id"], keep="first")

    result = ef[["hadm_id", "valuenum", "charttime", "ef_source"]].copy()
    result.rename(columns={
        "valuenum": "ef_value",
        "charttime": "ef_date",
    }, inplace=True)

    return result.reset_index(drop=True)


# ── NLP patterns for EF extraction from discharge notes ──────────────
# Numeric EF: "EF 55%", "LVEF of 60-65%", "ejection fraction = 35%", etc.
_EF_NUMERIC_RE = re.compile(
    r"(?:(?:LV\s*)?(?:EF|ejection\s+fraction))"   # anchor: EF / LVEF / ejection fraction
    r"\s*(?:of|is|was|:|\=|estimated\s+at)?\s*"    # optional connectors
    r"(?:~|>|<|>=|<=|approximately|approx\.?)?\s*"  # optional qualifiers
    r"(\d{1,2})\s*(?:-|to)\s*(\d{1,2})\s*%"        # range: "55-60%"
    r"|"
    r"(?:(?:LV\s*)?(?:EF|ejection\s+fraction))"
    r"\s*(?:of|is|was|:|\=|estimated\s+at)?\s*"
    r"(?:~|>|<|>=|<=|approximately|approx\.?)?\s*"
    r"(\d{1,2})\s*%",                               # point: "55%"
    re.IGNORECASE,
)

# Qualitative EF descriptions → midpoint numeric estimate
_EF_QUALITATIVE = {
    r"normal\s+(?:systolic\s+)?(?:LV\s+)?function": 60,
    r"(?:preserved|hyperdynamic)\s+(?:systolic\s+)?(?:LV\s+)?function": 60,
    r"low[\s-]?normal\s+(?:systolic\s+)?function": 52,
    r"mild(?:ly)?\s+(?:reduced|decreased|depressed)\s+(?:systolic\s+)?(?:LV\s+)?function": 45,
    r"moderate(?:ly)?\s+(?:reduced|decreased|depressed)\s+(?:systolic\s+)?(?:LV\s+)?function": 35,
    r"moderate(?:ly)?[\s-]+to[\s-]+severe(?:ly)?\s+(?:reduced|decreased|depressed)": 27,
    r"severe(?:ly)?\s+(?:reduced|decreased|depressed)\s+(?:systolic\s+)?(?:LV\s+)?function": 20,
}
_EF_QUALITATIVE_RES = [
    (re.compile(pat, re.IGNORECASE), val)
    for pat, val in _EF_QUALITATIVE.items()
]

# Echo section headers to focus NLP search
_ECHO_HEADER_RE = re.compile(
    r"(?:ECHO|TTE|TEE|[Ee]chocardiogra[mp]|TRANSTHORACIC|TRANSESOPHAGEAL)",
)


def _extract_ef_from_text(text: str) -> tuple[float | None, str]:
    """Extract EF from a single note text.

    Returns (ef_value, method) where method is 'numeric' or 'qualitative'.
    For numeric ranges, takes the lower bound (conservative for HFpEF).
    """
    if not isinstance(text, str):
        return None, ""

    # Try to narrow to echo-related paragraphs (±500 chars around header).
    # If no echo header found, search full text.
    search_regions: list[str] = []
    for m in _ECHO_HEADER_RE.finditer(text):
        start = max(0, m.start() - 200)
        end = min(len(text), m.end() + 1500)
        search_regions.append(text[start:end])
    if not search_regions:
        search_regions = [text]

    # Pass 1: numeric EF. Collect all candidates and keep the last mention,
    # which is often the most up-to-date summary value in discharge notes.
    numeric_candidates: list[tuple[float, str]] = []
    for region in search_regions:
        for m in _EF_NUMERIC_RE.finditer(region):
            range_lo, range_hi, point = m.group(1), m.group(2), m.group(3)
            if range_lo and range_hi:
                lo, hi = float(range_lo), float(range_hi)
                if 5 <= lo <= 90 and 5 <= hi <= 90:
                    numeric_candidates.append((lo, "numeric_range"))  # conservative for ranges
            elif point:
                val = float(point)
                if 5 <= val <= 90:
                    numeric_candidates.append((val, "numeric_point"))
    if numeric_candidates:
        return numeric_candidates[-1]

    # Pass 2: qualitative description
    qualitative_candidates: list[float] = []
    for region in search_regions:
        for pat_re, val in _EF_QUALITATIVE_RES:
            if pat_re.search(region):
                qualitative_candidates.append(float(val))
    if qualitative_candidates:
        return qualitative_candidates[-1], "qualitative"

    return None, ""


def extract_ef_from_discharge_notes(
    note_dir: Path, subject_ids: set[int], hadm_ids: set[int],
) -> pd.DataFrame:
    """Extract ejection fraction from MIMIC-IV-Note discharge summaries.

    Returns per-hadm_id DataFrame with:
        ef_value: extracted numeric EF (%)
        ef_date: note charttime
        ef_source: extraction method (numeric_point / numeric_range / qualitative)
    """
    discharge_path = note_dir / "discharge.csv.gz"
    if not discharge_path.exists():
        print("    WARNING: discharge.csv.gz not found; skipping note EF extraction")
        return pd.DataFrame(columns=["hadm_id", "ef_value", "ef_date", "ef_source"])

    print(f"    Reading {discharge_path} ...")
    # Read in chunks to limit memory for 1.1GB file
    ef_rows = []
    n_notes_scanned = 0
    for chunk in pd.read_csv(
        discharge_path,
        usecols=["subject_id", "hadm_id", "charttime", "text"],
        chunksize=10_000,
    ):
        # Filter to cohort
        mask = chunk["subject_id"].isin(subject_ids) & chunk["hadm_id"].isin(hadm_ids)
        cohort_notes = chunk.loc[mask].copy()
        n_notes_scanned += len(cohort_notes)

        for row in cohort_notes.itertuples(index=False):
            ef_val, ef_method = _extract_ef_from_text(row.text)
            if ef_val is not None:
                ef_rows.append({
                    "subject_id": row.subject_id,
                    "hadm_id": row.hadm_id,
                    "ef_value": ef_val,
                    "ef_date": row.charttime,
                    "ef_source": f"discharge_note_{ef_method}",
                })

    print(f"    Discharge notes scanned (cohort): {n_notes_scanned:,}")
    print(f"    Notes with EF extracted: {len(ef_rows):,}")

    if len(ef_rows) == 0:
        return pd.DataFrame(columns=["hadm_id", "ef_value", "ef_date", "ef_source"])

    ef = pd.DataFrame(ef_rows)
    ef["ef_date"] = pd.to_datetime(ef["ef_date"], errors="coerce")

    # Keep most recent EF per admission (matching chartevents logic)
    ef = ef.sort_values(
        ["hadm_id", "ef_date"], ascending=[True, False]
    ).drop_duplicates(subset=["hadm_id"], keep="first")

    # Method breakdown
    method_counts = ef["ef_source"].value_counts()
    for method, cnt in method_counts.items():
        print(f"      {method}: {cnt:,}")

    return ef[["hadm_id", "ef_value", "ef_date", "ef_source"]].reset_index(drop=True)


def classify_hf_subtype(row: pd.Series) -> str:
    """Classify HF subtype based on EF value and HF diagnosis."""
    if not row["has_hf"]:
        return "no_hf"
    if pd.isna(row["ef_value"]):
        return "hf_ef_unknown"
    if row["ef_value"] >= EF_HFPEF_THRESHOLD:
        return "hfpef"
    if row["ef_value"] >= EF_HFMREF_LOWER:
        return "hfmref"
    return "hfref"


def main() -> None:
    cfg = load_config()
    mimic_root = dataset_entrypoint(cfg, "mimic")
    staging_dir = resolve_path(cfg, "staging")
    reports_dir = resolve_path(cfg, "reports")
    staging_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("20a_build_hf_labels_mimic — HF phenotyping (20a–20d)")
    print("=" * 70)

    # ── load cohort backbone ──────────────────────────────────────────
    print("\n[1/6] Loading cohort backbone ...")
    stays = pd.read_parquet(
        staging_dir / "10a_core_icu_stays_mimic.parquet",
        columns=["subject_id", "hadm_id", "stay_id"],
    )
    n_stays = len(stays)
    n_subjects = stays["subject_id"].nunique()
    print(f"  Stays: {n_stays:,}  Subjects: {n_subjects:,}")

    # ── Step 20a: HF candidate flags from ICD codes ──────────────────
    print("\n[2/6] Building HF candidate flags (20a) ...")
    dx = pd.read_parquet(staging_dir / "10b_diagnoses_mimic.parquet")
    hf_flags = extract_hf_flags(dx)
    n_hadm_hf = len(hf_flags)
    print(f"  Admissions with any HF ICD: {n_hadm_hf:,}")

    # Merge HF flags to stays (via hadm_id)
    out = stays.merge(hf_flags, on="hadm_id", how="left")
    out["has_hf"] = out["has_hf"].fillna(False)
    out["has_hf_icd10"] = out["has_hf_icd10"].fillna(False)
    out["has_hf_icd9"] = out["has_hf_icd9"].fillna(False)
    out["has_diastolic_hf_icd"] = out["has_diastolic_hf_icd"].fillna(False)

    n_stays_hf = out["has_hf"].sum()
    print(f"  Stays with HF flag: {n_stays_hf:,} ({n_stays_hf / n_stays * 100:.1f}%)")

    # ── Step 20b: Extract EF (chartevents + discharge notes) ─────────
    cohort_subjects = set(stays["subject_id"].unique())
    cohort_stays = set(stays["stay_id"].unique())
    cohort_hadms = set(stays["hadm_id"].unique())

    # Source 1: chartevents (structured, highest priority)
    print("\n[3/6] Extracting EF from chartevents (20b-chartevents) ...")
    ef_chart = extract_ef_from_chartevents(mimic_root, cohort_subjects, cohort_stays, cohort_hadms)
    n_ef_chart = len(ef_chart)
    print(f"  Admissions with chartevents EF: {n_ef_chart:,} ({n_ef_chart / len(cohort_hadms) * 100:.1f}%)")

    # Source 2: discharge notes (NLP)
    print("\n[4/6] Extracting EF from discharge notes (20b-notes) ...")
    note_subdir = cfg["datasets"]["mimic"].get("subdirs", {}).get("note")
    if note_subdir:
        note_dir = mimic_root / note_subdir
    else:
        note_dir = mimic_root / "note" / "2.2" / "note"
    ef_note = extract_ef_from_discharge_notes(note_dir, cohort_subjects, cohort_hadms)
    n_ef_note = len(ef_note)
    print(f"  Admissions with note EF: {n_ef_note:,} ({n_ef_note / len(cohort_hadms) * 100:.1f}%)")

    # Merge: chartevents takes priority, notes fill gaps
    if n_ef_chart > 0 and n_ef_note > 0:
        chart_hadms = set(ef_chart["hadm_id"])
        ef_note_new = ef_note[~ef_note["hadm_id"].isin(chart_hadms)].copy()
        ef_data = pd.concat([ef_chart, ef_note_new], ignore_index=True)
        n_overlap = n_ef_note - len(ef_note_new)
        print(f"\n  EF merge: {n_ef_chart:,} chartevents + {len(ef_note_new):,} note-only "
              f"({n_overlap:,} overlapping admissions → chartevents kept)")
    elif n_ef_chart > 0:
        ef_data = ef_chart
    elif n_ef_note > 0:
        ef_data = ef_note
    else:
        ef_data = pd.DataFrame(columns=["hadm_id", "ef_value", "ef_date", "ef_source"])

    n_ef = len(ef_data)
    ef_coverage = n_ef / len(cohort_hadms) if cohort_hadms else 0.0
    print(f"  Combined EF coverage: {n_ef:,} admissions ({ef_coverage * 100:.1f}%)")
    if ef_coverage < 0.05:
        print("  WARNING: Low EF coverage (<5%); HFpEF classification will rely more on ICD diastolic proxy.")

    if n_ef > 0:
        print(f"  EF distribution: mean={ef_data['ef_value'].mean():.1f}%, "
              f"median={ef_data['ef_value'].median():.0f}%, "
              f"range=[{ef_data['ef_value'].min():.0f}, {ef_data['ef_value'].max():.0f}]")

    # Merge EF to stays (via hadm_id)
    out = out.merge(ef_data, on="hadm_id", how="left")

    # ── Step 20c: Classify HF subtypes ───────────────────────────────
    print("\n[5/6] Classifying HF subtypes (20c+20d) ...")
    out["hf_subtype"] = out.apply(classify_hf_subtype, axis=1)

    # Convenience flags
    out["is_hfpef"] = out["hf_subtype"] == "hfpef"
    out["is_hfmref"] = out["hf_subtype"] == "hfmref"
    out["is_hfref"] = out["hf_subtype"] == "hfref"
    out["is_hf_ef_unknown"] = out["hf_subtype"] == "hf_ef_unknown"
    out["is_no_hf"] = out["hf_subtype"] == "no_hf"

    # Sensitivity: ICD-only diastolic HF proxy (for subjects without EF)
    out["hfpef_icd_proxy"] = out["has_diastolic_hf_icd"]

    # Comparator group assignment
    # Primary pair 1: HFpEF vs HFrEF/HFmrEF
    # Primary pair 2: HFpEF vs non-HF
    out["comparator_group"] = out["hf_subtype"].map({
        "hfpef": "hfpef",
        "hfmref": "hfref_hfmref",
        "hfref": "hfref_hfmref",
        "hf_ef_unknown": "hf_ef_unknown",
        "no_hf": "no_hf",
    })

    subtype_counts = out["hf_subtype"].value_counts()
    print(f"  Subtype distribution:")
    for st, cnt in subtype_counts.items():
        print(f"    {st:20s}: {cnt:>8,}  ({cnt / n_stays * 100:.1f}%)")

    # ── write output ─────────────────────────────────────────────────
    out_cols = [
        "stay_id", "subject_id", "hadm_id",
        # 20a: HF flags
        "has_hf", "has_hf_icd10", "has_hf_icd9",
        "has_diastolic_hf_icd", "hf_icd_codes",
        # 20b: EF
        "ef_value", "ef_date", "ef_source",
        # 20c: subtype
        "hf_subtype", "is_hfpef", "is_hfmref", "is_hfref",
        "is_hf_ef_unknown", "is_no_hf",
        # 20d: comparator + sensitivity proxy
        "comparator_group", "hfpef_icd_proxy",
    ]
    out = out[out_cols].sort_values("stay_id").reset_index(drop=True)

    out_path = staging_dir / "20a_hf_labels_mimic.parquet"
    out.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"\n  → {out_path}  ({len(out):,} rows)")

    # ── summary report ───────────────────────────────────────────────
    print("\n[6/6] Generating summary report ...")

    comp_counts = out["comparator_group"].value_counts().to_string()
    n_hfpef = out["is_hfpef"].sum()
    n_hfref = out["is_hfref"].sum()
    n_hfmref = out["is_hfmref"].sum()
    n_ef_unknown = out["is_hf_ef_unknown"].sum()
    n_no_hf = out["is_no_hf"].sum()
    n_icd_proxy = out["hfpef_icd_proxy"].sum()

    # EF stats among HF patients
    hf_ef = out.loc[out["has_hf"] & out["ef_value"].notna(), "ef_value"]

    ef_stats = ""
    if len(hf_ef) > 0:
        ef_stats = (
            f"  n={len(hf_ef):,}  mean={hf_ef.mean():.1f}%  "
            f"median={hf_ef.median():.0f}%  "
            f"[{hf_ef.min():.0f}, {hf_ef.quantile(0.25):.0f}, "
            f"{hf_ef.quantile(0.75):.0f}, {hf_ef.max():.0f}]"
        )

    # EF source breakdown
    ef_source_counts = out.loc[out["ef_value"].notna(), "ef_source"].value_counts()
    ef_source_lines = "\n".join(
        f"    {src:40s}: {cnt:>8,}" for src, cnt in ef_source_counts.items()
    ) if len(ef_source_counts) > 0 else "    (none)"

    report = textwrap.dedent(f"""\
    ================================================================
    20a_build_hf_labels_mimic — Summary Report (Steps 20a–20d)
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ================================================================

    STEP 20a: HF CANDIDATE FLAGS
    ─────────────────────────────
    Total ICU stays         : {n_stays:>10,}
    Stays with HF ICD code  : {n_stays_hf:>10,}  ({n_stays_hf / n_stays * 100:.1f}%)
    Stays without HF        : {n_stays - n_stays_hf:>10,}  ({(n_stays - n_stays_hf) / n_stays * 100:.1f}%)
    Diastolic HF ICD proxy  : {n_icd_proxy:>10,}

    STEP 20b: EJECTION FRACTION
    ────────────────────────────
    Admissions with EF (chartevents): {n_ef_chart:>10,}  ({n_ef_chart / len(cohort_hadms) * 100:.1f}% of admissions)
    Admissions with EF (notes NLP) : {n_ef_note:>10,}  ({n_ef_note / len(cohort_hadms) * 100:.1f}% of admissions)
    Combined unique EF coverage    : {n_ef:>10,}  ({n_ef / len(cohort_hadms) * 100:.1f}% of admissions)
    EF among HF patients    : {ef_stats}

    EF source breakdown:
{ef_source_lines}

    STEP 20c: HF SUBTYPE CLASSIFICATION
    ─────────────────────────────────────
    HFpEF (EF >= 50%)      : {n_hfpef:>10,}  ({n_hfpef / n_stays * 100:.1f}%)
    HFmrEF (EF 40-49%)     : {n_hfmref:>10,}  ({n_hfmref / n_stays * 100:.1f}%)
    HFrEF (EF < 40%)       : {n_hfref:>10,}  ({n_hfref / n_stays * 100:.1f}%)
    HF, EF unknown          : {n_ef_unknown:>10,}  ({n_ef_unknown / n_stays * 100:.1f}%)
    No HF                   : {n_no_hf:>10,}  ({n_no_hf / n_stays * 100:.1f}%)

    STEP 20d: COMPARATOR GROUPS
    ────────────────────────────
    {textwrap.indent(comp_counts, '  ')}

    OUTPUT
    ──────
    {out_path}  ({len(out):,} rows, {out.shape[1]} cols)
    """)

    report_path = reports_dir / "20a_hf_labels_mimic_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"  → Report: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
