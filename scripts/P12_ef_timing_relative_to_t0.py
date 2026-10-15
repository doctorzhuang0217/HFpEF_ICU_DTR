#!/usr/bin/env python3
"""
P12_ef_timing_relative_to_t0 — When was the classifying EF measured?

Purpose
-------
This script asks whether HFpEF classification is based on EF measured BEFORE
the critical illness episode. 20a keeps only the most-recent EF per admission
and does not bucket it relative to ICU admission or T0. This script:

  1. Re-scans MIMIC chartevents for ALL EF measurements (not just most recent)
     using the same d_items resolution as 20a.
  2. Buckets each measurement's charttime against intime (ICU admission) and
     time_zero_6h (T0 = intime + 6h):
        - pre_icu      : charttime < intime              (ED / floor / external)
        - icu_pre_t0   : intime ≤ charttime < time_zero_6h
        - icu_post_t0  : charttime ≥ time_zero_6h
  3. Pulls the classifying EF (the one 20a used) and locates it in those
     buckets. For note-NLP-derived EF, the timestamp is the discharge-note
     charttime (NOT the underlying echo time); these are flagged separately
     as `note_undated`.

Outputs a per-stay table with timing breakdown + a Table-1-style summary.

Inputs
------
- data/staging/25a_time_zero_flags_mimic.parquet
- data/staging/20a_hf_labels_mimic.parquet
- data/raw/mimiciv/icu/chartevents.csv.gz
- data/raw/mimiciv/icu/d_items.csv.gz

Outputs
-------
- data/staging/P12_ef_timing_mimic.parquet
- reports/P12_ef_timing_summary.txt
"""

from __future__ import annotations

import sys
import textwrap
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, dataset_entrypoint, resolve_path

# Re-use 20a's EF itemid resolver to stay consistent.
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
import importlib.util


def _load_20a_module():
    spec = importlib.util.spec_from_file_location(
        "_step20a",
        Path(__file__).resolve().parent / "20a_build_hf_labels_mimic.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def scan_all_ef_chartevents(
    mimic_root: Path,
    subject_ids: set[int],
    stay_ids: set[int],
    hadm_ids: set[int],
) -> pd.DataFrame:
    """Scan chartevents for ALL EF measurements (no per-hadm dedup).

    Returns DataFrame with columns: hadm_id, stay_id, subject_id,
    charttime, ef_value, itemid, ef_source.
    """
    mod20a = _load_20a_module()
    d_items = pd.read_csv(mimic_root / "icu" / "d_items.csv.gz")
    ef_itemids = mod20a._resolve_ef_itemids(d_items)
    print(f"    EF itemids resolved: {len(ef_itemids)} -> {sorted(ef_itemids)}")
    if not ef_itemids:
        return pd.DataFrame(columns=[
            "hadm_id", "stay_id", "subject_id",
            "charttime", "ef_value", "itemid", "ef_source",
        ])

    target_ids = set(ef_itemids.keys())
    rows = []
    ce_path = mimic_root / "icu" / "chartevents.csv.gz"
    # MIMIC-IV chartevents has ~330M rows; chunksize=2M -> ~165 chunks.
    # Wall time depends on disk + gzip throughput, typically 30-50 minutes.
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda it, **kw: it  # noqa: E731

    n_matched = 0
    pbar = tqdm(
        pd.read_csv(
            ce_path,
            usecols=["subject_id", "hadm_id", "stay_id",
                     "charttime", "itemid", "valuenum"],
            chunksize=2_000_000,
        ),
        desc="    chartevents EF scan",
        unit="chunk",
        total=None,
    )
    for chunk in pbar:
        mask = (
            chunk["itemid"].isin(target_ids)
            & chunk["stay_id"].isin(stay_ids)
            & chunk["hadm_id"].isin(hadm_ids)
            & chunk["subject_id"].isin(subject_ids)
        )
        matched = chunk.loc[mask].copy()
        if len(matched) > 0:
            rows.append(matched)
            n_matched += len(matched)
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix({"matched_rows": n_matched})

    if not rows:
        return pd.DataFrame(columns=[
            "hadm_id", "stay_id", "subject_id",
            "charttime", "ef_value", "itemid", "ef_source",
        ])

    ef = pd.concat(rows, ignore_index=True)
    ef["valuenum"] = pd.to_numeric(ef["valuenum"], errors="coerce")
    ef = ef[ef["valuenum"].between(5, 90)].copy()
    ef["charttime"] = pd.to_datetime(ef["charttime"], errors="coerce")
    ef = ef.dropna(subset=["charttime"])
    ef["ef_source"] = ef["itemid"].map(ef_itemids)
    ef = ef.rename(columns={"valuenum": "ef_value"})
    return ef[[
        "hadm_id", "stay_id", "subject_id",
        "charttime", "ef_value", "itemid", "ef_source",
    ]].reset_index(drop=True)


def _bucket(ef_date, intime, t0):
    """Bucket a single timestamp (used only for scalar lookups)."""
    if pd.isna(ef_date) or pd.isna(intime) or pd.isna(t0):
        return None
    if ef_date < intime:
        return "pre_icu"
    if ef_date < t0:
        return "icu_pre_t0"
    return "icu_post_t0"


def bucket_vec(ef_date: pd.Series, intime: pd.Series, t0: pd.Series) -> pd.Series:
    """Vectorized bucketing: returns Series of pre_icu / icu_pre_t0 /
    icu_post_t0 / NaN (when any timestamp is missing)."""
    out = pd.Series(pd.NA, index=ef_date.index, dtype=object)
    valid = ef_date.notna() & intime.notna() & t0.notna()
    pre = valid & (ef_date < intime)
    mid = valid & (ef_date >= intime) & (ef_date < t0)
    post = valid & (ef_date >= t0)
    out[pre] = "pre_icu"
    out[mid] = "icu_pre_t0"
    out[post] = "icu_post_t0"
    return out


def main() -> None:
    cfg = load_config()
    mimic_root = dataset_entrypoint(cfg, "mimic")
    staging_dir = resolve_path(cfg, "staging")
    reports_dir = resolve_path(cfg, "reports")
    staging_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("P12_ef_timing_relative_to_t0 — EF timing buckets")
    print("=" * 70)

    print("\n[1/4] Loading cohort + classification EF ...")
    cohort = pd.read_parquet(
        staging_dir / "25a_time_zero_flags_mimic.parquet",
        columns=["stay_id", "subject_id", "hadm_id",
                 "hf_subtype", "comparator_group",
                 "intime", "time_zero_6h"],
    )
    cohort["intime"] = pd.to_datetime(cohort["intime"], errors="coerce")
    cohort["time_zero_6h"] = pd.to_datetime(cohort["time_zero_6h"], errors="coerce")
    print(f"  Cohort stays: {len(cohort):,}")

    hf = pd.read_parquet(
        staging_dir / "20a_hf_labels_mimic.parquet",
        columns=["stay_id", "ef_value", "ef_date", "ef_source"],
    )
    hf["ef_date"] = pd.to_datetime(hf["ef_date"], errors="coerce")
    out = cohort.merge(hf, on="stay_id", how="left")

    # Classification-EF source family (vectorized).
    src = out["ef_source"].astype("string")
    fam = pd.Series("chartevents", index=out.index, dtype=object)
    fam[src.isna()] = "no_ef"
    fam[src.fillna("").str.startswith("discharge_note_")] = "note_NLP"
    out["classification_ef_source_family"] = fam

    # Classification-EF bucket (only meaningful for chartevents source).
    chart_bucket = bucket_vec(out["ef_date"], out["intime"], out["time_zero_6h"])
    cbucket = pd.Series("unknown", index=out.index, dtype=object)
    cbucket[fam == "no_ef"] = "no_ef"
    cbucket[fam == "note_NLP"] = "note_undated"
    is_chart = (fam == "chartevents")
    cbucket[is_chart & chart_bucket.notna()] = chart_bucket[is_chart & chart_bucket.notna()]
    out["classification_ef_bucket"] = cbucket

    print("\n[2/4] Scanning all chartevents EF measurements (no dedup) ...")
    cohort_subjects = set(cohort["subject_id"].unique())
    cohort_stays = set(cohort["stay_id"].unique())
    cohort_hadms = set(cohort["hadm_id"].unique())
    ef_all = scan_all_ef_chartevents(
        mimic_root, cohort_subjects, cohort_stays, cohort_hadms,
    )
    print(f"  Raw chartevents EF rows (post range filter): {len(ef_all):,}")

    if len(ef_all) > 0:
        # Bucket each measurement (vectorized).
        timeline = ef_all.merge(
            cohort[["stay_id", "intime", "time_zero_6h"]],
            on="stay_id", how="left",
        )
        timeline["bucket"] = bucket_vec(
            timeline["charttime"], timeline["intime"], timeline["time_zero_6h"],
        )
        # Per-stay counts per bucket.
        counts = (
            timeline.assign(_one=1)
            .pivot_table(
                index="stay_id", columns="bucket", values="_one",
                aggfunc="sum", fill_value=0,
            )
        )
        for col in ("pre_icu", "icu_pre_t0", "icu_post_t0"):
            if col not in counts.columns:
                counts[col] = 0
        counts = counts[["pre_icu", "icu_pre_t0", "icu_post_t0"]].astype(int)
        counts.columns = [f"n_chart_ef_{c}" for c in counts.columns]
        counts["n_chart_ef_total"] = counts.sum(axis=1)

        earliest = (
            timeline.sort_values(["stay_id", "charttime"])
            .groupby("stay_id")
            .agg(earliest_chart_ef_charttime=("charttime", "first"),
                 earliest_chart_ef_value=("ef_value", "first"))
            .reset_index()
        )
        cohort_idx = cohort.set_index("stay_id")
        earliest_intime = cohort_idx["intime"].reindex(earliest["stay_id"]).to_numpy()
        earliest_t0 = cohort_idx["time_zero_6h"].reindex(earliest["stay_id"]).to_numpy()
        earliest["earliest_chart_ef_bucket"] = bucket_vec(
            pd.Series(earliest["earliest_chart_ef_charttime"].to_numpy()),
            pd.Series(earliest_intime),
            pd.Series(earliest_t0),
        ).to_numpy()
        per_stay = counts.reset_index().merge(earliest, on="stay_id", how="outer")
    else:
        per_stay = pd.DataFrame(columns=[
            "stay_id",
            "n_chart_ef_pre_icu", "n_chart_ef_icu_pre_t0", "n_chart_ef_icu_post_t0",
            "n_chart_ef_total",
            "earliest_chart_ef_charttime", "earliest_chart_ef_value",
            "earliest_chart_ef_bucket",
        ])

    out = out.merge(per_stay, on="stay_id", how="left")
    for col in ("n_chart_ef_pre_icu", "n_chart_ef_icu_pre_t0",
                "n_chart_ef_icu_post_t0", "n_chart_ef_total"):
        out[col] = out[col].fillna(0).astype(int)

    print("\n[3/4] Writing per-stay timing table ...")
    out_cols = [
        "stay_id", "subject_id", "hadm_id",
        "hf_subtype", "comparator_group",
        "ef_value", "ef_date", "ef_source",
        "classification_ef_source_family", "classification_ef_bucket",
        "n_chart_ef_pre_icu", "n_chart_ef_icu_pre_t0",
        "n_chart_ef_icu_post_t0", "n_chart_ef_total",
        "earliest_chart_ef_charttime", "earliest_chart_ef_value",
        "earliest_chart_ef_bucket",
    ]
    out = out[out_cols].sort_values("stay_id").reset_index(drop=True)
    out_path = staging_dir / "P12_ef_timing_mimic.parquet"
    out.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"  → {out_path}  ({len(out):,} rows)")

    print("\n[4/4] Generating summary report ...")
    bucket_order = [
        "pre_icu", "icu_pre_t0", "icu_post_t0",
        "note_undated", "no_ef", "unknown",
    ]
    out["classification_ef_bucket"] = pd.Categorical(
        out["classification_ef_bucket"], categories=bucket_order, ordered=True,
    )
    cross = pd.crosstab(
        out["classification_ef_bucket"], out["comparator_group"], margins=True,
        dropna=False,
    )

    # HFpEF-only timing breakdown.
    hfpef = out[out["hf_subtype"] == "hfpef"].copy()
    hfpef_dist = (
        hfpef["classification_ef_bucket"].value_counts(dropna=False)
        .reindex(bucket_order, fill_value=0)
    )
    n_hfpef = max(len(hfpef), 1)
    hfpef_lines = "\n".join(
        f"    {b:18s}: {n:>8,}  ({n / n_hfpef * 100:.1f}%)"
        for b, n in hfpef_dist.items()
    )

    earliest_dist = (
        out["earliest_chart_ef_bucket"].fillna("(none)")
        .value_counts().sort_index()
    )

    report = textwrap.dedent(f"""\
    ================================================================
    P12_ef_timing_relative_to_t0 — Summary Report
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ================================================================

    CLASSIFICATION-EF BUCKET × COMPARATOR GROUP
    ────────────────────────────────────────────
    {textwrap.indent(cross.to_string(), '  ')}

    HFpEF SUBSET — CLASSIFICATION-EF TIMING ({len(hfpef):,} stays)
    ───────────────────────────────────────────────────────────────
{hfpef_lines}

    EARLIEST CHARTEVENTS EF BUCKET (across all stays with chartevents EF)
    ──────────────────────────────────────────────────────────────────────
    {textwrap.indent(earliest_dist.to_string(), '  ')}

    NOTES
    ─────
    - `chartevents` EF rows have charttime that reflects when the value was
      charted in MIMIC; bucketing against intime / T0 is meaningful.
    - `note_NLP` EF rows derive from discharge-summary NLP; ef_date is the
      note charttime (typically near discharge), NOT the underlying echo
      timestamp. These are reported as `note_undated` and should NOT be
      interpreted as post-T0 measurements.
    - `pre_icu` count > 0 means the patient had at least one EF charted
      BEFORE ICU admission (e.g. in the ED or during a prior floor stay
      within the same hospitalization). This addresses the
      question about whether HFpEF classification reflects pre-illness EF.

    OUTPUT
    ──────
    {out_path}  ({len(out):,} rows, {out.shape[1]} cols)
    """)

    report_path = reports_dir / "P12_ef_timing_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"  → Report: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
