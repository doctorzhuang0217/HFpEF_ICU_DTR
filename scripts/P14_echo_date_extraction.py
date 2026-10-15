#!/usr/bin/env python3
"""
P14_echo_date_extraction — Conservative NLP for the *referenced* echo date.

Purpose
-------
P12 reported that for every HFpEF stay, the classifying EF was extracted by
NLP from the discharge summary, where `ef_date` records the discharge-note
charttime — NOT when the underlying echocardiogram was performed. P12 was
therefore unable to bucket HFpEF EF measurements pre-/post-T0 and the row
was reported as `note_undated`.

P14 takes a second NLP pass over discharge.csv.gz that targets the echo
*reference date* with a CONSERVATIVE regex set (high precision, low recall).
We extract dates only when they are explicitly anchored to an echo verb:

    "TTE on YYYY-MM-DD"               "echo on MM/DD/YYYY"
    "TEE dated MMM DD, YYYY"          "echocardiogram performed YYYY-MM-DD"

Phrases like "recent echo", "prior echo", or "TTE X days ago" are NOT
parsed, because the relative-time inference is brittle and reviewers can
push back on it.

A separate keyword bucket flags echoes anchored to the index admission:

    "TTE on admission"     "echo done this admission"     "TTE today"
    "echo earlier today"   "echo this hospitalization"

These get a `this_admission_keyword` flag without an explicit date.

Buckets (relative to admittime / intime / T0):
  - pre_admission           (echo_date < admittime)
  - hospital_pre_icu        (admittime ≤ echo_date < intime)
  - icu_pre_t0              (intime ≤ echo_date < time_zero_6h)
  - icu_post_t0             (echo_date ≥ time_zero_6h)
  - this_admission_keyword  (no date, but admission-anchored phrase)
  - no_referenced_date      (no parseable date and no admission keyword)

Inputs
------
- data/raw/mimiciv/note/2.2/note/discharge.csv.gz
- data/staging/20a_hf_labels_mimic.parquet
- data/staging/25a_time_zero_flags_mimic.parquet

Outputs
-------
- data/staging/P14_echo_date_extraction_mimic.parquet
- reports/P14_echo_date_extraction_summary.txt
"""

from __future__ import annotations

import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, dataset_entrypoint, resolve_path


# Anchor: an echo verb / noun must be within ECHO_LOOKBEHIND of the date.
ECHO_VERB = r"(?:TTE|TEE|echo(?:cardiogram)?(?:s)?)"
ECHO_VERB_RE = re.compile(ECHO_VERB, re.IGNORECASE)

# Conservative date forms.
#   YYYY-MM-DD or YYYY/MM/DD       (ISO-like)
#   M/D/YY or MM/DD/YYYY           (US)
#   D Month YYYY                   (e.g. "5 May 2026")
#   Month D, YYYY                  (e.g. "May 5, 2026")
DATE_PATTERNS = [
    (
        re.compile(
            ECHO_VERB + r"\s+(?:was\s+)?(?:on|dated|performed|completed|done|obtained)\s+"
            r"(?:on\s+)?(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b",
            re.IGNORECASE,
        ),
        "iso",
    ),
    (
        re.compile(
            ECHO_VERB + r"\s+(?:was\s+)?(?:on|dated|performed|completed|done|obtained)\s+"
            r"(?:on\s+)?(\d{1,2}/\d{1,2}/(?:\d{2}|\d{4}))\b",
            re.IGNORECASE,
        ),
        "us",
    ),
    (
        re.compile(
            ECHO_VERB + r"\s+(?:was\s+)?(?:on|dated|performed|completed|done|obtained)\s+"
            r"(?:on\s+)?"
            r"((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
            r"Dec(?:ember)?)\s+\d{1,2},?\s+\d{4})",
            re.IGNORECASE,
        ),
        "month_name",
    ),
    (
        re.compile(
            ECHO_VERB + r"\s+(?:was\s+)?(?:on|dated|performed|completed|done|obtained)\s+"
            r"(?:on\s+)?"
            r"(\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
            r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
            r"Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})",
            re.IGNORECASE,
        ),
        "day_month",
    ),
]

# Admission-anchored keyword phrases (no explicit date).
ADMISSION_KEYWORD_RE = re.compile(
    ECHO_VERB
    + r"\s+(?:was\s+)?(?:done\s+|performed\s+|obtained\s+)?"
    r"(?:on\s+)?"
    r"(?:this\s+admission|on\s+admission|today|this\s+hospitalization|"
    r"during\s+this\s+admission|earlier\s+today|this\s+morning|"
    r"on\s+arrival)",
    re.IGNORECASE,
)


def _parse_date(s: str, kind: str) -> pd.Timestamp | None:
    """Conservative date parsing — return NaT on any ambiguity."""
    s = s.strip()
    try:
        if kind == "iso":
            ts = pd.to_datetime(s.replace("/", "-"), errors="coerce")
        elif kind == "us":
            # US M/D/Y. Accept 2-digit and 4-digit year.
            ts = pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")
            if pd.isna(ts):
                ts = pd.to_datetime(s, format="%m/%d/%y", errors="coerce")
        elif kind in ("month_name", "day_month"):
            ts = pd.to_datetime(s, errors="coerce")
        else:
            return None
    except (ValueError, TypeError):
        return None
    if pd.isna(ts):
        return None
    # Sanity: drop dates outside MIMIC-IV plausible range.
    if not (pd.Timestamp("1990-01-01") <= ts <= pd.Timestamp("2030-12-31")):
        return None
    return ts


def extract_echo_dates(text: str) -> tuple[pd.Timestamp | None, bool, str]:
    """Return (referenced_echo_date, this_admission_keyword, match_kind).

    On multiple matches, returns the FIRST. match_kind is one of
    'iso' / 'us' / 'month_name' / 'day_month' / 'admission_keyword' / ''.
    """
    if not isinstance(text, str) or len(text) < 20:
        return None, False, ""

    # First, try explicit dates.
    for rx, kind in DATE_PATTERNS:
        m = rx.search(text)
        if m:
            ts = _parse_date(m.group(1), kind)
            if ts is not None:
                return ts, False, kind

    # Fallback: admission-anchored keyword without a date.
    if ADMISSION_KEYWORD_RE.search(text):
        return None, True, "admission_keyword"

    return None, False, ""


def main() -> None:
    cfg = load_config()
    mimic_root = dataset_entrypoint(cfg, "mimic")
    staging_dir = resolve_path(cfg, "staging")
    reports_dir = resolve_path(cfg, "reports")
    staging_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("P14_echo_date_extraction — referenced echo date NLP (conservative)")
    print("=" * 70)

    print("\n[1/4] Loading cohort + 20a EF source ...")
    cohort = pd.read_parquet(
        staging_dir / "25a_time_zero_flags_mimic.parquet",
        columns=["stay_id", "subject_id", "hadm_id",
                 "hf_subtype", "comparator_group",
                 "admittime", "intime", "time_zero_6h"],
    )
    for col in ("admittime", "intime", "time_zero_6h"):
        cohort[col] = pd.to_datetime(cohort[col], errors="coerce")

    hf = pd.read_parquet(
        staging_dir / "20a_hf_labels_mimic.parquet",
        columns=["stay_id", "ef_value", "ef_source"],
    )
    cohort = cohort.merge(hf, on="stay_id", how="left")

    # Stays whose classifying EF came from a discharge note (i.e. the ones
    # P12 could not timestamp). Restrict NLP to those hadm_ids to save time.
    note_mask = cohort["ef_source"].fillna("").astype(str).str.startswith("discharge_note_")
    target_hadms = set(cohort.loc[note_mask, "hadm_id"].unique())
    print(f"  Target admissions (note-derived EF): {len(target_hadms):,}")

    print("\n[2/4] Scanning discharge.csv.gz for echo-anchored dates ...")
    note_subdir = cfg["datasets"]["mimic"].get("subdirs", {}).get("note")
    note_dir = mimic_root / note_subdir if note_subdir else mimic_root / "note" / "2.2" / "note"
    discharge_path = note_dir / "discharge.csv.gz"
    if not discharge_path.exists():
        print(f"  ERROR: {discharge_path} not found.")
        sys.exit(1)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda it, **kw: it  # noqa: E731

    rows = []
    n_scanned = 0
    n_matched = 0
    pbar = tqdm(
        pd.read_csv(
            discharge_path,
            usecols=["subject_id", "hadm_id", "charttime", "text"],
            chunksize=10_000,
        ),
        desc="    discharge scan",
        unit="chunk",
    )
    for chunk in pbar:
        sub = chunk[chunk["hadm_id"].isin(target_hadms)]
        n_scanned += len(sub)
        for row in sub.itertuples(index=False):
            ts, kw, kind = extract_echo_dates(row.text)
            if ts is not None or kw:
                rows.append({
                    "subject_id": row.subject_id,
                    "hadm_id": row.hadm_id,
                    "note_charttime": row.charttime,
                    "referenced_echo_date": ts,
                    "this_admission_keyword": kw,
                    "match_kind": kind,
                })
                n_matched += 1
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix({"scanned": n_scanned, "matched": n_matched})

    print(f"\n  Notes scanned (target cohort): {n_scanned:,}")
    print(f"  Notes with referenced echo date or admission keyword: {n_matched:,}")

    if rows:
        ref = pd.DataFrame(rows)
        ref["referenced_echo_date"] = pd.to_datetime(
            ref["referenced_echo_date"], errors="coerce",
        )
        # Per hadm_id keep the earliest referenced echo date if multiple notes.
        ref = ref.sort_values(["hadm_id", "referenced_echo_date"], na_position="last")
        ref = ref.drop_duplicates(subset=["hadm_id"], keep="first")
    else:
        ref = pd.DataFrame(columns=[
            "subject_id", "hadm_id", "note_charttime",
            "referenced_echo_date", "this_admission_keyword", "match_kind",
        ])

    print("\n[3/4] Bucketing referenced dates relative to admittime / intime / T0 ...")
    out = cohort.merge(
        ref[["hadm_id", "referenced_echo_date",
             "this_admission_keyword", "match_kind"]],
        on="hadm_id", how="left",
    )

    bucket = pd.Series("no_referenced_date", index=out.index, dtype=object)
    has_date = out["referenced_echo_date"].notna()
    bucket[has_date & (out["referenced_echo_date"] < out["admittime"])] = "pre_admission"
    bucket[has_date
           & (out["referenced_echo_date"] >= out["admittime"])
           & (out["referenced_echo_date"] < out["intime"])] = "hospital_pre_icu"
    bucket[has_date
           & (out["referenced_echo_date"] >= out["intime"])
           & (out["referenced_echo_date"] < out["time_zero_6h"])] = "icu_pre_t0"
    bucket[has_date & (out["referenced_echo_date"] >= out["time_zero_6h"])] = "icu_post_t0"
    bucket[(~has_date) & out["this_admission_keyword"].fillna(False)] = "this_admission_keyword"
    out["referenced_echo_bucket"] = bucket

    # Restrict reporting to note-derived classifying EF (that's what we set
    # out to address). All other stays are kept in the parquet for
    # completeness but get bucket = "no_referenced_date".
    is_note = out["ef_source"].fillna("").astype(str).str.startswith("discharge_note_")
    out["is_note_derived_ef"] = is_note

    print("\n[4/4] Writing output + report ...")
    out_cols = [
        "stay_id", "subject_id", "hadm_id",
        "hf_subtype", "comparator_group",
        "ef_value", "ef_source", "is_note_derived_ef",
        "admittime", "intime", "time_zero_6h",
        "referenced_echo_date", "match_kind",
        "this_admission_keyword", "referenced_echo_bucket",
    ]
    out["this_admission_keyword"] = out["this_admission_keyword"].fillna(False).astype(bool)
    out = out[out_cols].sort_values("stay_id").reset_index(drop=True)
    out_path = staging_dir / "P14_echo_date_extraction_mimic.parquet"
    out.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"  → {out_path}  ({len(out):,} rows)")

    bucket_order = [
        "pre_admission", "hospital_pre_icu", "icu_pre_t0", "icu_post_t0",
        "this_admission_keyword", "no_referenced_date",
    ]
    out["referenced_echo_bucket"] = pd.Categorical(
        out["referenced_echo_bucket"], categories=bucket_order, ordered=True,
    )

    # Restrict the headline tables to note-derived EF stays — that's the
    # population whose timing was ambiguous in P12.
    note_only = out[out["is_note_derived_ef"]]
    cross = pd.crosstab(
        note_only["referenced_echo_bucket"],
        note_only["comparator_group"],
        margins=True, dropna=False,
    )
    hfpef = note_only[note_only["hf_subtype"] == "hfpef"]
    hfpef_dist = (
        hfpef["referenced_echo_bucket"].value_counts(dropna=False)
        .reindex(bucket_order, fill_value=0)
    )
    n_hfpef = max(len(hfpef), 1)
    hfpef_lines = "\n".join(
        f"    {b:25s}: {n:>8,}  ({n / n_hfpef * 100:.1f}%)"
        for b, n in hfpef_dist.items()
    )

    match_kind_dist = (
        out.loc[out["referenced_echo_date"].notna(), "match_kind"]
        .value_counts().sort_index()
    )

    report = textwrap.dedent(f"""\
    ================================================================
    P14_echo_date_extraction — Summary Report
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ================================================================

    Note-derived classifying EF stays      : {int(out['is_note_derived_ef'].sum()):>8,}
    Of which: explicit referenced echo date: {int(note_only['referenced_echo_date'].notna().sum()):>8,}
              admission-keyword (no date)  : {int((note_only['referenced_echo_bucket'] == 'this_admission_keyword').sum()):>8,}
              still undated                : {int((note_only['referenced_echo_bucket'] == 'no_referenced_date').sum()):>8,}

    REFERENCED-ECHO BUCKET × COMPARATOR (note-derived EF stays only)
    ─────────────────────────────────────────────────────────────────
    {textwrap.indent(cross.to_string(), '  ')}

    HFpEF SUBSET — referenced echo timing ({len(hfpef):,} stays)
    ───────────────────────────────────────────────────────────
{hfpef_lines}

    EXPLICIT-DATE MATCH-KIND BREAKDOWN
    ────────────────────────────────────
    {textwrap.indent(match_kind_dist.to_string(), '  ')}

    NOTES
    ─────
    - Conservative regex set: only matches when the date follows an echo
      verb (TTE / TEE / echo / echocardiogram) plus an action word (on /
      dated / performed / completed / done / obtained). Phrases like
      'recent echo' or 'prior echo' are NOT parsed (low recall by design).
    - 'this_admission_keyword' captures explicit anchors to the index
      admission (e.g. 'TTE on admission', 'echo done today') when a
      date isn't given. These are NOT bucketed by date but DO confirm
      the echo was during the index admission.
    - Bucketing uses admittime / intime / time_zero_6h from 25a.

    OUTPUT
    ──────
    {out_path}  ({len(out):,} rows, {out.shape[1]} cols)
    """)

    report_path = reports_dir / "P14_echo_date_extraction_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"  → Report: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
