#!/usr/bin/env python3
"""
05a_inventory_raw_tables_mimic — Enumerate and audit all MIMIC-IV source tables.

Purpose
-------
Scan every .csv.gz file under data/raw/mimiciv/{hosp,icu}, record:
  - table name, subdirectory, file size
  - row count
  - column names and dtypes
  - identifier columns present (subject_id, hadm_id, stay_id)
  - timestamp columns detected
  - missingness per column
  - temporal coverage (min/max of first detected timestamp column)

This step is read-only — it does NOT extract or transform data.
Its output supports planning for Steps 10a–10h (extraction).

Inputs
------
- data/raw/mimiciv/hosp/*.csv.gz
- data/raw/mimiciv/icu/*.csv.gz

Outputs
-------
- reports/05a_mimic_table_manifest.csv   (machine-readable manifest)
- reports/05a_mimic_schema_audit.txt     (human-readable summary)
"""

from __future__ import annotations

import sys
import csv
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── project imports ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, dataset_entrypoint, resolve_path

# Columns we always look for as identifiers
ID_COLS = {"subject_id", "hadm_id", "stay_id", "patientunitstayid"}
# Common timestamp column name patterns
TS_PATTERNS = {"time", "date"}

# Tables that are very large — sample only for schema + missingness
LARGE_TABLE_THRESHOLD = 50_000_000  # 50M rows: sample instead of full read


def _detect_timestamp_cols(columns: list[str]) -> list[str]:
    """Heuristic: columns containing 'time' or 'date' in their name."""
    return [c for c in columns if any(p in c.lower() for p in TS_PATTERNS)]


def _audit_table(filepath: Path, nrows_sample: int = 100_000) -> dict:
    """Audit a single .csv.gz table and return metadata dict."""
    info: dict = {
        "file": str(filepath.name),
        "subdir": filepath.parent.name,
        "rel_path": f"{filepath.parent.name}/{filepath.name}",
        "size_mb": round(filepath.stat().st_size / (1024 * 1024), 1),
    }

    # Read header only first to get columns
    header_df = pd.read_csv(filepath, nrows=0)
    cols = list(header_df.columns)
    info["n_columns"] = len(cols)
    info["columns"] = ", ".join(cols)

    # Detect identifier and timestamp columns
    id_present = [c for c in cols if c in ID_COLS]
    ts_cols = _detect_timestamp_cols(cols)
    info["id_columns"] = ", ".join(id_present) if id_present else "(none)"
    info["timestamp_columns"] = ", ".join(ts_cols) if ts_cols else "(none)"

    # Count rows — for small files read fully, for large ones use wc-like approach
    # We'll read in chunks to count rows without loading everything into memory
    row_count = 0
    chunk_iter = pd.read_csv(filepath, chunksize=500_000, usecols=[cols[0]])
    for chunk in chunk_iter:
        row_count += len(chunk)
    info["row_count"] = row_count

    # Decide whether to sample or read fully for missingness
    if row_count > LARGE_TABLE_THRESHOLD:
        sample_df = pd.read_csv(filepath, nrows=nrows_sample)
        info["sampled"] = True
        info["sample_size"] = nrows_sample
    else:
        sample_df = pd.read_csv(filepath)
        info["sampled"] = False
        info["sample_size"] = row_count

    # Missingness per column
    miss_pct = (sample_df.isna().sum() / len(sample_df) * 100).round(1)
    miss_summary = {col: f"{pct}%" for col, pct in miss_pct.items() if pct > 0}
    info["columns_with_missing"] = len(miss_summary)
    info["missingness_detail"] = "; ".join(
        f"{k}={v}" for k, v in sorted(miss_summary.items())
    ) if miss_summary else "(none)"

    # Dtypes
    dtype_summary = sample_df.dtypes.astype(str).to_dict()
    info["dtypes"] = "; ".join(f"{k}={v}" for k, v in dtype_summary.items())

    # Temporal coverage (first timestamp column)
    info["temporal_min"] = ""
    info["temporal_max"] = ""
    if ts_cols:
        first_ts = ts_cols[0]
        try:
            ts_series = pd.to_datetime(sample_df[first_ts], errors="coerce")
            ts_valid = ts_series.dropna()
            if len(ts_valid) > 0:
                info["temporal_min"] = str(ts_valid.min())
                info["temporal_max"] = str(ts_valid.max())
                info["temporal_column_used"] = first_ts
        except Exception:
            pass

    # Unique identifier counts
    for id_col in id_present:
        info[f"nunique_{id_col}"] = int(sample_df[id_col].nunique())

    return info


def main() -> None:
    cfg = load_config()
    mimic_root = dataset_entrypoint(cfg, "mimic")
    reports_dir = resolve_path(cfg, "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("05a_inventory_raw_tables_mimic — MIMIC-IV Table Inventory & Audit")
    print("=" * 70)

    # ── discover all csv.gz files ──────────────────────────────────────
    print("\n[1/3] Discovering tables ...")

    table_files = sorted(mimic_root.rglob("*.csv.gz"))
    if not table_files:
        raise FileNotFoundError(
            f"No .csv.gz files found under {mimic_root}. "
            "Check that the data symlink is configured correctly."
        )
    print(f"  Found {len(table_files)} tables")
    for f in table_files:
        print(f"    {f.parent.name}/{f.name}")

    # ── audit each table ───────────────────────────────────────────────
    print("\n[2/3] Auditing tables (this may take a while for large tables) ...")

    results = []
    for i, fpath in enumerate(table_files, 1):
        rel = f"{fpath.parent.name}/{fpath.name}"
        print(f"  [{i}/{len(table_files)}] {rel} ...", end=" ", flush=True)
        try:
            info = _audit_table(fpath)
            print(f"{info['row_count']:,} rows, {info['size_mb']} MB")
            results.append(info)
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "file": fpath.name,
                "subdir": fpath.parent.name,
                "rel_path": rel,
                "error": str(e),
            })

    # ── write outputs ──────────────────────────────────────────────────
    print("\n[3/3] Writing outputs ...")

    # CSV manifest
    manifest_path = reports_dir / "05a_mimic_table_manifest.csv"
    if results:
        # Collect all keys across all result dicts
        all_keys = []
        for r in results:
            for k in r.keys():
                if k not in all_keys:
                    all_keys.append(k)

        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"  → Manifest: {manifest_path}")

    # Human-readable summary
    audit_path = reports_dir / "05a_mimic_schema_audit.txt"
    lines = [
        "=" * 70,
        "05a_inventory_raw_tables_mimic — Schema Audit Report",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"MIMIC root: {mimic_root}",
        "=" * 70,
        "",
        f"Total tables found: {len(table_files)}",
        "",
    ]

    # Summary table
    lines.append(f"{'Table':<45s} {'Rows':>12s} {'Size MB':>8s} {'Cols':>5s} {'IDs':<30s}")
    lines.append("-" * 105)
    total_rows = 0
    for r in results:
        if "error" in r:
            lines.append(f"{r.get('rel_path', '?'):<45s} ERROR: {r['error']}")
            continue
        total_rows += r["row_count"]
        lines.append(
            f"{r['rel_path']:<45s} {r['row_count']:>12,} {r['size_mb']:>8.1f} "
            f"{r['n_columns']:>5d} {r.get('id_columns', ''):<30s}"
        )
    lines.append("-" * 105)
    lines.append(f"{'TOTAL':<45s} {total_rows:>12,}")
    lines.append("")

    # Per-table detail blocks
    lines.append("=" * 70)
    lines.append("DETAILED PER-TABLE AUDIT")
    lines.append("=" * 70)
    for r in results:
        if "error" in r:
            continue
        lines.append("")
        lines.append(f"--- {r['rel_path']} ---")
        lines.append(f"  Rows         : {r['row_count']:,}")
        lines.append(f"  File size    : {r['size_mb']} MB")
        lines.append(f"  Columns ({r['n_columns']}): {r['columns']}")
        lines.append(f"  IDs present  : {r['id_columns']}")
        lines.append(f"  Timestamps   : {r['timestamp_columns']}")
        if r.get("temporal_min"):
            lines.append(f"  Temporal range: {r['temporal_min']} → {r['temporal_max']} (col: {r.get('temporal_column_used', '?')})")
        lines.append(f"  Sampled      : {'yes (' + str(r['sample_size']) + ' rows)' if r['sampled'] else 'no (full read)'}")
        lines.append(f"  Missing cols : {r['columns_with_missing']}")
        if r["missingness_detail"] != "(none)":
            lines.append(f"  Missingness  : {r['missingness_detail']}")
        # Unique ID counts
        for id_col in ["subject_id", "hadm_id", "stay_id"]:
            key = f"nunique_{id_col}"
            if key in r:
                lines.append(f"  Unique {id_col}: {r[key]:,}")

    lines.append("")
    lines.append(f"Manifest CSV: {manifest_path}")
    lines.append("")

    report_text = "\n".join(lines)
    audit_path.write_text(report_text, encoding="utf-8")
    print(f"  → Audit report: {audit_path}")
    print(f"\n  Total rows across all tables: {total_rows:,}")
    print(report_text)
    print("\nDone.")


if __name__ == "__main__":
    main()
