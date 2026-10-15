#!/usr/bin/env python3
"""
P13b_restrictive_cm_code_audit — Audit which ICD codes drive is_restrictive_cm.

Question motivating this audit
------------------------------
P13 reports is_restrictive_cm = 8.5 % in the HFrEF/HFmrEF arm versus 2.0 %
in HFpEF. This is *opposite* to the canonical clinical signal (restrictive
physiology should preferentially co-segregate with preserved EF). The
suspicion is that the underlying flag is dominated by ICD-9 **425.4**
("Other primary cardiomyopathies"), which in practice has been used as a
catch-all for dilated / unspecified CM rather than for true restrictive CM,
and is therefore polluting the HFrEF/HFmrEF cell.

This script tests that hypothesis by breaking down is_restrictive_cm by:
  - which underlying code triggered the flag (I42.5 vs I42.3 vs 425.4)
  - HF subtype (HFpEF / HFrEF/HFmrEF / HF-EF-unknown / no-HF)
  - per-cell EF distribution (median / IQR / % EF<40 / % EF>=50)

If HFrEF/HFmrEF restrictive_cm patients are overwhelmingly 425.4-driven
with low EF, the flag is mis-specified for that arm and the P13 row should
be re-interpreted accordingly.

Inputs
------
- data/staging/25a_time_zero_flags_mimic.parquet
- data/staging/20a_hf_labels_mimic.parquet     (for ef_value)
- data/staging/10b_diagnoses_mimic.parquet

Outputs
-------
- tables/P13b_restrictive_cm_code_audit.csv
- reports/P13b_restrictive_cm_code_audit.txt
"""

from __future__ import annotations

import sys
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path


# Same code set P13 uses for is_restrictive_cm.
RESTRICTIVE_CM_ICD10 = {"I425", "I423"}
RESTRICTIVE_CM_ICD9 = {"4254"}


def main() -> None:
    cfg = load_config()
    staging_dir = resolve_path(cfg, "staging")
    reports_dir = resolve_path(cfg, "reports")
    tables_dir = resolve_path(cfg, "tables")
    for d in (reports_dir, tables_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("P13b_restrictive_cm_code_audit — break down is_restrictive_cm")
    print("=" * 70)

    print("\n[1/4] Loading inputs ...")
    cohort = pd.read_parquet(
        staging_dir / "25a_time_zero_flags_mimic.parquet",
        columns=["stay_id", "subject_id", "hadm_id",
                 "hf_subtype", "comparator_group"],
    )
    n_index = len(cohort)
    print(f"  Cohort stays: {n_index:,}")

    ef = pd.read_parquet(
        staging_dir / "20a_hf_labels_mimic.parquet",
        columns=["stay_id", "ef_value", "ef_source"],
    )
    cohort = cohort.merge(ef, on="stay_id", how="left")

    dx = pd.read_parquet(
        staging_dir / "10b_diagnoses_mimic.parquet",
        columns=["hadm_id", "icd_code", "icd_version"],
    )
    print(f"  Diagnoses: {len(dx):,} rows")

    print("\n[2/4] Locating restrictive-CM ICD hits ...")
    icd_norm = (
        dx["icd_code"].astype(str)
        .str.replace(".", "", regex=False).str.upper().str.strip()
    )
    v10 = (dx["icd_version"] == 10).to_numpy()
    v9 = (dx["icd_version"] == 9).to_numpy()

    hit_i425 = (icd_norm == "I425").to_numpy() & v10
    hit_i423 = (icd_norm == "I423").to_numpy() & v10
    hit_4254 = (icd_norm == "4254").to_numpy() & v9

    hadm = dx["hadm_id"].to_numpy()
    s_i425 = set(hadm[hit_i425])
    s_i423 = set(hadm[hit_i423])
    s_4254 = set(hadm[hit_4254])

    print(f"  Unique admissions with I42.5 (ICD-10 restrictive CM): {len(s_i425):,}")
    print(f"  Unique admissions with I42.3 (ICD-10 endomyocardial): {len(s_i423):,}")
    print(f"  Unique admissions with 425.4 (ICD-9 other primary CM): {len(s_4254):,}")

    cohort["hit_i425"] = cohort["hadm_id"].isin(s_i425)
    cohort["hit_i423"] = cohort["hadm_id"].isin(s_i423)
    cohort["hit_4254"] = cohort["hadm_id"].isin(s_4254)
    cohort["any_restrictive_cm_flag"] = (
        cohort["hit_i425"] | cohort["hit_i423"] | cohort["hit_4254"]
    )

    # Mutually-exclusive code source priority (most specific first):
    #   I42.5 > I42.3 > 425.4. Patients with multiple codes are listed under
    #   the most specific they carry.
    def _src(row):
        if row["hit_i425"]:
            return "I42.5_ICD10_restrictive"
        if row["hit_i423"]:
            return "I42.3_ICD10_endomyocardial"
        if row["hit_4254"]:
            return "425.4_ICD9_other_primary_CM"
        return "(none)"

    cohort["restrictive_cm_code_source"] = cohort.apply(_src, axis=1)

    print("\n[3/4] Cross-tab + EF distributions ...")
    flagged = cohort[cohort["any_restrictive_cm_flag"]].copy()

    # 1) Code source × HF subtype counts.
    cross_count = pd.crosstab(
        flagged["restrictive_cm_code_source"],
        flagged["comparator_group"], margins=True,
    )

    # 2) Code source × HF subtype EF stats.
    def _ef_stats(s: pd.Series) -> dict:
        s = s.dropna()
        if len(s) == 0:
            return {"n": 0, "median": np.nan, "q25": np.nan, "q75": np.nan,
                    "pct_lt40": np.nan, "pct_ge50": np.nan}
        return {
            "n": int(len(s)),
            "median": float(s.median()),
            "q25": float(s.quantile(0.25)),
            "q75": float(s.quantile(0.75)),
            "pct_lt40": float((s < 40).mean() * 100),
            "pct_ge50": float((s >= 50).mean() * 100),
        }

    rows = []
    for src, sg in flagged.groupby("restrictive_cm_code_source", sort=True):
        for grp, gg in sg.groupby("comparator_group", sort=True):
            stats = _ef_stats(gg["ef_value"])
            rows.append({
                "code_source": src,
                "comparator_group": grp,
                **stats,
            })
    ef_table = pd.DataFrame(rows)

    # 3) Within HFrEF/HFmrEF specifically: what fraction of restrictive_cm
    #    flag carriers come from each code source?
    hfref = flagged[flagged["comparator_group"] == "hfref_hfmref"]
    hfref_src_share = (
        hfref["restrictive_cm_code_source"].value_counts(normalize=True) * 100
    ).round(1)

    hfpef = flagged[flagged["comparator_group"] == "hfpef"]
    hfpef_src_share = (
        hfpef["restrictive_cm_code_source"].value_counts(normalize=True) * 100
    ).round(1)

    # 4) Save audit table.
    audit_path = tables_dir / "P13b_restrictive_cm_code_audit.csv"
    ef_table.to_csv(audit_path, index=False)
    print(f"  → {audit_path}")

    print("\n[4/4] Generating report ...")
    report = textwrap.dedent(f"""\
    ================================================================
    P13b_restrictive_cm_code_audit — Audit Report
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ================================================================

    Cohort: {n_index:,} index ICU stays
    Total admissions flagged is_restrictive_cm: {int(flagged.shape[0]):,}
      I42.5 (ICD-10 restrictive CM)        : {len(s_i425):>6,}
      I42.3 (ICD-10 endomyocardial)        : {len(s_i423):>6,}
      425.4 (ICD-9  other primary CM)      : {len(s_4254):>6,}
      (Note: a single admission may carry multiple codes; the cross-tab
       below assigns each to its most-specific code source.)

    CODE-SOURCE × HF SUBTYPE  (restrictive_cm-flagged admissions only)
    ───────────────────────────────────────────────────────────────────
    {textwrap.indent(cross_count.to_string(), '  ')}

    HFrEF/HFmrEF — composition by code source (% of arm's flag carriers)
    ─────────────────────────────────────────────────────────────────────
    {textwrap.indent(hfref_src_share.to_string(), '  ')}

    HFpEF — composition by code source (% of arm's flag carriers)
    ──────────────────────────────────────────────────────────────
    {textwrap.indent(hfpef_src_share.to_string(), '  ')}

    EF DISTRIBUTION  (per code source × HF subtype)
    ─────────────────────────────────────────────────
    {textwrap.indent(ef_table.to_string(index=False), '  ')}

    INTERPRETATION GUIDE
    ─────────────────────
    - If HFrEF/HFmrEF flag carriers are dominated by 425.4_ICD9 (>80 %)
      with median EF < 40 %, the is_restrictive_cm flag is being polluted
      by "other primary cardiomyopathy" coding (which in practice is used
      as a catch-all for dilated / unspecified CM, not true restrictive CM).
      → Action: re-define is_restrictive_cm to exclude 425.4 alone, and
        re-issue the P13 row for restrictive_cm.
    - If HFpEF flag carriers split similarly between I42.5 and 425.4 with
      EF >= 50 %, the original P13 number for HFpEF is structurally clean.

    OUTPUT
    ──────
    Audit CSV: {audit_path}
    """)

    report_path = reports_dir / "P13b_restrictive_cm_code_audit.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"  → Report: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
