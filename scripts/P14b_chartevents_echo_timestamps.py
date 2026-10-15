#!/usr/bin/env python3
"""
P14b_chartevents_echo_timestamps — Find echo procedure timestamps in MIMIC-IV.

Purpose
-------
P14 (NLP-based) returned 0 explicitly-dated referenced echoes from
discharge summaries (96.9 % of the HFpEF cohort remains undated). This
script bypasses NLP and asks: does MIMIC-IV ICU expose echocardiogram
timestamps via *structured* tables (procedureevents, datetimeevents)?

Strategy
--------
1. Scan d_items.csv.gz for any item whose label/abbreviation/category
   matches an echo cue (echo / TTE / TEE / echocardiogr).
2. Group the matched itemids by their `linksto` table:
      procedureevents   — has starttime/endtime per procedure
      datetimeevents    — stores a date value (e.g. "Last Echo Date") with
                          its own charttime
      chartevents       — already covered by 20a's EF extraction
3. Pull the relevant rows from procedureevents.csv.gz and
   datetimeevents.csv.gz for the cohort, take the EARLIEST timestamp per
   admission, and bucket vs intime / T0 / admittime.
4. Cross-reference with P14's note-NLP output to compute combined coverage:
      structured-only / NLP-only / both / neither.

Outputs
-------
- data/staging/P14b_chartevents_echo_timestamps_mimic.parquet
- reports/P14b_chartevents_echo_timestamps_summary.txt
- tables/P14b_echo_itemid_inventory.csv          (which itemids matched)
"""

from __future__ import annotations

import sys
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, dataset_entrypoint, resolve_path


def resolve_echo_itemids(d_items: pd.DataFrame) -> pd.DataFrame:
    """Find echo-related items via case-insensitive cue matching.

    Returns a DataFrame with columns: itemid, label, linksto, abbreviation,
    category, match_field (which column triggered the match).
    """
    cues = r"(?i)\b(?:echo(?:cardiogra[mp])?|TTE|TEE|trans(?:thoracic|esophageal)\s*echo)\b"
    fields = ["label", "abbreviation", "category"]
    keep = pd.Series(False, index=d_items.index)
    match_field = pd.Series("", index=d_items.index, dtype=object)
    for f in fields:
        if f in d_items.columns:
            hit = d_items[f].fillna("").astype(str).str.contains(cues, regex=True, na=False)
            new = hit & (~keep)
            match_field[new] = f
            keep = keep | hit
    cols = ["itemid", "label", "linksto"]
    if "abbreviation" in d_items.columns:
        cols.append("abbreviation")
    if "category" in d_items.columns:
        cols.append("category")
    out = d_items.loc[keep, cols].copy()
    out["match_field"] = match_field[keep].values
    return out.reset_index(drop=True)


def scan_procedureevents(
    mimic_root: Path, target_itemids: set,
    subject_ids: set, stay_ids: set, hadm_ids: set,
) -> pd.DataFrame:
    """Pull echo procedureevents rows for the cohort."""
    path = mimic_root / "icu" / "procedureevents.csv.gz"
    if not path.exists() or not target_itemids:
        return pd.DataFrame(columns=["hadm_id", "stay_id", "subject_id",
                                      "itemid", "starttime", "endtime", "label"])

    cols = ["subject_id", "hadm_id", "stay_id", "itemid", "starttime", "endtime"]
    rows = []
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda it, **kw: it  # noqa: E731

    pbar = tqdm(
        pd.read_csv(path, usecols=cols, chunksize=500_000),
        desc="    procedureevents", unit="chunk",
    )
    for chunk in pbar:
        m = (
            chunk["itemid"].isin(target_itemids)
            & chunk["stay_id"].isin(stay_ids)
            & chunk["hadm_id"].isin(hadm_ids)
            & chunk["subject_id"].isin(subject_ids)
        )
        sub = chunk.loc[m]
        if len(sub) > 0:
            rows.append(sub.copy())
    if not rows:
        return pd.DataFrame(columns=cols + ["label"])
    df = pd.concat(rows, ignore_index=True)
    df["starttime"] = pd.to_datetime(df["starttime"], errors="coerce")
    df["endtime"] = pd.to_datetime(df["endtime"], errors="coerce")
    return df


def scan_datetimeevents(
    mimic_root: Path, target_itemids: set,
    subject_ids: set, stay_ids: set, hadm_ids: set,
) -> pd.DataFrame:
    """Pull echo datetimeevents rows for the cohort.

    datetimeevents stores a *date value* (column 'value' as a timestamp
    string) along with its charttime (when the value was recorded). For an
    echo "Last Echo Date" item, `value` is the date the echo was performed
    and `charttime` is when the nurse charted it.
    """
    path = mimic_root / "icu" / "datetimeevents.csv.gz"
    if not path.exists() or not target_itemids:
        return pd.DataFrame(columns=["hadm_id", "stay_id", "subject_id",
                                      "itemid", "charttime", "value"])

    cols = ["subject_id", "hadm_id", "stay_id", "itemid", "charttime", "value"]
    rows = []
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda it, **kw: it  # noqa: E731

    pbar = tqdm(
        pd.read_csv(path, usecols=cols, chunksize=1_000_000),
        desc="    datetimeevents", unit="chunk",
    )
    for chunk in pbar:
        m = (
            chunk["itemid"].isin(target_itemids)
            & chunk["stay_id"].isin(stay_ids)
            & chunk["hadm_id"].isin(hadm_ids)
            & chunk["subject_id"].isin(subject_ids)
        )
        sub = chunk.loc[m]
        if len(sub) > 0:
            rows.append(sub.copy())
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.concat(rows, ignore_index=True)
    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    df["echo_value_time"] = pd.to_datetime(df["value"], errors="coerce")
    return df


def _bucket(ts, admittime, intime, t0):
    if pd.isna(ts) or pd.isna(intime) or pd.isna(t0):
        return None
    if pd.notna(admittime) and ts < admittime:
        return "pre_admission"
    if ts < intime:
        return "hospital_pre_icu"
    if ts < t0:
        return "icu_pre_t0"
    return "icu_post_t0"


def main() -> None:
    cfg = load_config()
    mimic_root = dataset_entrypoint(cfg, "mimic")
    staging_dir = resolve_path(cfg, "staging")
    reports_dir = resolve_path(cfg, "reports")
    tables_dir = resolve_path(cfg, "tables")
    for d in (staging_dir, reports_dir, tables_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("P14b_chartevents_echo_timestamps — structured echo timestamps")
    print("=" * 70)

    print("\n[1/5] Resolving echo-related itemids ...")
    d_items = pd.read_csv(mimic_root / "icu" / "d_items.csv.gz")
    inv = resolve_echo_itemids(d_items)
    print(f"  Echo-cue itemids found: {len(inv)}")
    inv_path = tables_dir / "P14b_echo_itemid_inventory.csv"
    inv.to_csv(inv_path, index=False)
    print(f"  → {inv_path}")
    if inv.empty:
        print("  No echo itemids found in d_items — nothing to do.")
        sys.exit(0)
    print("\n  Inventory by linksto:")
    print(inv.groupby("linksto").size().to_string())

    print("\n[2/5] Loading cohort ...")
    cohort = pd.read_parquet(
        staging_dir / "25a_time_zero_flags_mimic.parquet",
        columns=["stay_id", "subject_id", "hadm_id",
                 "hf_subtype", "comparator_group",
                 "admittime", "intime", "time_zero_6h"],
    )
    for col in ("admittime", "intime", "time_zero_6h"):
        cohort[col] = pd.to_datetime(cohort[col], errors="coerce")
    subj_ids = set(cohort["subject_id"].unique())
    stay_ids = set(cohort["stay_id"].unique())
    hadm_ids = set(cohort["hadm_id"].unique())
    print(f"  Cohort stays: {len(cohort):,}")

    proc_ids = set(inv.loc[inv["linksto"] == "procedureevents", "itemid"])
    date_ids = set(inv.loc[inv["linksto"] == "datetimeevents", "itemid"])

    print(f"\n[3/5] Scanning procedureevents ({len(proc_ids)} itemids) ...")
    proc = scan_procedureevents(mimic_root, proc_ids, subj_ids, stay_ids, hadm_ids)
    print(f"  Echo procedurevents rows for cohort: {len(proc):,}")

    print(f"\n[4/5] Scanning datetimeevents ({len(date_ids)} itemids) ...")
    dte = scan_datetimeevents(mimic_root, date_ids, subj_ids, stay_ids, hadm_ids)
    print(f"  Echo datetimeevents rows for cohort: {len(dte):,}")

    # Per-admission earliest echo timestamp.
    rows = []
    if len(proc) > 0:
        for hadm, sg in proc.groupby("hadm_id"):
            ts = sg["starttime"].dropna().min()
            if pd.notna(ts):
                rows.append({"hadm_id": hadm, "echo_ts": ts,
                             "ts_source": "procedureevents.starttime"})
    if len(dte) > 0:
        for hadm, sg in dte.groupby("hadm_id"):
            # Prefer the value (echo perform date) over charttime when
            # present — value is the actual echo date.
            ts_value = sg["echo_value_time"].dropna().min()
            ts_chart = sg["charttime"].dropna().min()
            ts = ts_value if pd.notna(ts_value) else ts_chart
            src = "datetimeevents.value" if pd.notna(ts_value) else "datetimeevents.charttime"
            if pd.notna(ts):
                rows.append({"hadm_id": hadm, "echo_ts": ts, "ts_source": src})
    ts_df = pd.DataFrame(rows)
    if not ts_df.empty:
        ts_df = ts_df.sort_values(["hadm_id", "echo_ts"]).drop_duplicates(
            subset=["hadm_id"], keep="first",
        )

    out = cohort.merge(ts_df, on="hadm_id", how="left")
    out["echo_bucket"] = [
        _bucket(t, a, i, z)
        for t, a, i, z in zip(
            out["echo_ts"], out["admittime"], out["intime"], out["time_zero_6h"],
        )
    ]
    out["echo_bucket"] = out["echo_bucket"].fillna("no_structured_echo_ts")

    print("\n[5/5] Writing outputs + report ...")
    out_path = staging_dir / "P14b_chartevents_echo_timestamps_mimic.parquet"
    keep_cols = [
        "stay_id", "subject_id", "hadm_id",
        "hf_subtype", "comparator_group",
        "admittime", "intime", "time_zero_6h",
        "echo_ts", "ts_source", "echo_bucket",
    ]
    out[keep_cols].to_parquet(out_path, index=False, engine="pyarrow")
    print(f"  → {out_path}  ({len(out):,} rows)")

    bucket_order = [
        "pre_admission", "hospital_pre_icu", "icu_pre_t0", "icu_post_t0",
        "no_structured_echo_ts",
    ]
    out["echo_bucket"] = pd.Categorical(
        out["echo_bucket"], categories=bucket_order, ordered=True,
    )
    cross = pd.crosstab(out["echo_bucket"], out["comparator_group"],
                        margins=True, dropna=False)
    hfpef = out[out["hf_subtype"] == "hfpef"]
    n_hfpef = max(len(hfpef), 1)
    hfpef_dist = (hfpef["echo_bucket"].value_counts(dropna=False)
                  .reindex(bucket_order, fill_value=0))
    hfpef_lines = "\n".join(
        f"    {b:25s}: {n:>8,}  ({n / n_hfpef * 100:.1f}%)"
        for b, n in hfpef_dist.items()
    )
    src_breakdown = (out.loc[out["echo_ts"].notna(), "ts_source"]
                     .value_counts().to_string())

    report = textwrap.dedent(f"""\
    ================================================================
    P14b_chartevents_echo_timestamps — Summary Report
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ================================================================

    Echo-cue itemids in d_items                : {len(inv):>8,}
      procedureevents itemids                  : {len(proc_ids):>8,}
      datetimeevents itemids                   : {len(date_ids):>8,}

    Cohort stays                               : {len(cohort):>8,}
    Stays with structured echo timestamp       : {int(out['echo_ts'].notna().sum()):>8,}

    SOURCE TABLE BREAKDOWN
    ───────────────────────
    {textwrap.indent(src_breakdown, '  ')}

    BUCKET × COMPARATOR (full cohort)
    ──────────────────────────────────
    {textwrap.indent(cross.to_string(), '  ')}

    HFpEF SUBSET — structured echo bucket ({len(hfpef):,} stays)
    ───────────────────────────────────────────────────────────
{hfpef_lines}

    NOTES
    ─────
    - This script bypasses NLP entirely and reads structured ICU tables.
    - procedureevents.starttime is the time the echo procedure was
      performed (best-quality timestamp).
    - datetimeevents stores a date *value* and its own charttime; we
      prefer the value (= performed date) over charttime when present.
    - A single admission's earliest matched timestamp is taken across
      both source tables.
    - Coverage depends on which echo itemids MIMIC-IV exposes; many ICU
      echoes are charted only as free-text in the discharge summary and
      will not appear here. P14 (NLP) covers that complementary fraction.

    OUTPUTS
    ───────
    Per-stay parquet : {out_path}
    Itemid inventory : {inv_path}
    """)
    report_path = reports_dir / "P14b_chartevents_echo_timestamps_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"  → Report: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
