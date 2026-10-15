#!/usr/bin/env python3
"""P9b_pretT0_hfi — Pre-T0 (0-6h) Hemodynamic Fragility Index.

Protocol: supplementary analysis (checks whether the post-T0 HFI of P9
may be a treatment artifact).
Compute tier: C1 (read pre-T0 vitals + groupby stats; ~1 min).

Purpose
-------
Re-compute the Hemodynamic Fragility Index using only **pre-T0** MAP
records (ICU admission to T0 = +6 h). If pre-T0 HFI continues to
separate HFpEF from no-HF, this strengthens the interpretation of HFI
as a baseline vulnerability marker rather than a downstream
consequence of differential treatment.

Index components (per stay, on the pre-T0 0-6 h window):

  1. frac_below_65 = (# MAP records below 65 mmHg) / (# MAP records).
     Analog of P9's `n_hypo_bins / 12`, on a record-count rather than
     fixed-bin basis (the pre-T0 window has variable record density).
  2. map_cv      = SD(MAP) / mean(MAP).
  3. map_range   = max(MAP) - min(MAP).

Composite pre_t0_hfi = unweighted average of the three z-standardised
components (matches `scripts/P9_hemodynamic_fragility_index.py`).

Stays with fewer than 2 valid MAP records in the pre-T0 window are
flagged `n_records_lt2 = 1` and excluded from the composite (their
SD/range/CV is undefined). They are reported separately.

Inputs
------
- data/staging/10d_vitals_mimic.parquet
- data/staging/25f_primary_cohort_candidate_v1_mimic.parquet

Outputs
-------
- tables/P9b_pretT0_hfi_per_stay.parquet
- tables/P9b_pretT0_hfi_by_hf_subtype.csv
- tables/P9b_pretT0_hfi_quartile_by_hf_subtype.csv
- reports/P9b_pretT0_hfi_summary.txt

CLI
---
    env/run_py.sh scripts/P9b_pretT0_hfi.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
STAGING_DIR = ROOT / "data" / "staging"
TABLES_DIR = ROOT / "tables"
REPORTS_DIR = ROOT / "reports"
for d in [TABLES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PRE_T0_HOURS = 6.0
MAP_HYPO_THRESHOLD = 65.0
MAP_VITAL_NAMES = ("map_invasive", "map_noninvasive")
HF_SUBTYPE_ORDER = ["hfpef", "hfref_hfmref", "hf_ef_unknown", "no_hf"]


def load_pre_t0_map() -> pd.DataFrame:
    print("[1/4] Loading pre-T0 MAP records ...")
    vitals_path = STAGING_DIR / "10d_vitals_mimic.parquet"
    cols = ["stay_id", "vital_name", "valuenum", "hours_since_icu_admit"]
    vitals = pd.read_parquet(vitals_path, columns=cols)
    vitals = vitals[vitals["vital_name"].isin(MAP_VITAL_NAMES)]
    vitals = vitals[
        vitals["hours_since_icu_admit"].between(0, PRE_T0_HOURS, inclusive="left")
    ].copy()
    vitals = vitals.dropna(subset=["valuenum"])
    # Light QC: drop physiologic outliers
    vitals = vitals[(vitals["valuenum"] >= 20) & (vitals["valuenum"] <= 250)]
    print(
        f"  Pre-T0 MAP records: {len(vitals):,} rows from "
        f"{vitals['stay_id'].nunique():,} stays"
    )
    return vitals


def compute_pre_t0_components(vitals: pd.DataFrame) -> pd.DataFrame:
    print("[2/4] Computing per-stay components ...")
    grouped = vitals.groupby("stay_id")["valuenum"]
    out = pd.DataFrame(
        {
            "stay_id": grouped.first().index,
            "n_records": grouped.size().values,
            "map_mean": grouped.mean().values,
            "map_std": grouped.std(ddof=0).values,
            "map_min": grouped.min().values,
            "map_max": grouped.max().values,
            "n_below_65": grouped.apply(lambda x: int((x < MAP_HYPO_THRESHOLD).sum())).values,
        }
    )
    out["frac_below_65"] = out["n_below_65"] / out["n_records"].clip(lower=1)
    out["map_cv"] = out["map_std"] / out["map_mean"].clip(lower=1.0)
    out["map_range"] = out["map_max"] - out["map_min"]
    out["n_records_lt2"] = (out["n_records"] < 2).astype(int)
    return out


def compute_composite(per_stay: pd.DataFrame) -> pd.DataFrame:
    """Z-score the three components on the analytic subset (n_records >= 2)
    and average. Flagged stays receive NaN composite."""
    print("[3/4] Computing composite pre_t0_hfi ...")
    valid = per_stay["n_records_lt2"] == 0
    sub = per_stay.loc[valid].copy()
    for col in ["frac_below_65", "map_cv", "map_range"]:
        mu = sub[col].mean()
        sigma = sub[col].std(ddof=1)
        sub[f"{col}_z"] = (sub[col] - mu) / sigma if sigma > 0 else 0.0
    sub["pre_t0_hfi"] = (
        sub["frac_below_65_z"] + sub["map_cv_z"] + sub["map_range_z"]
    ) / 3.0
    sub["pre_t0_hfi_quartile"] = pd.qcut(
        sub["pre_t0_hfi"],
        q=4,
        labels=["Q1_stable", "Q2", "Q3", "Q4_fragile"],
        duplicates="drop",
    )
    keep = ["stay_id", "frac_below_65_z", "map_cv_z", "map_range_z",
            "pre_t0_hfi", "pre_t0_hfi_quartile"]
    return per_stay.merge(sub[keep], on="stay_id", how="left")


def join_hf_subtype(per_stay: pd.DataFrame) -> pd.DataFrame:
    cohort_path = STAGING_DIR / "25f_primary_cohort_candidate_v1_mimic.parquet"
    cohort = pd.read_parquet(
        cohort_path,
        columns=["stay_id", "hf_subtype", "comparator_group",
                 "is_index_stay", "hemo_gate_passed"],
    )
    cohort = cohort[(cohort["is_index_stay"]) & (cohort["hemo_gate_passed"])]
    arm = cohort["comparator_group"].fillna(cohort["hf_subtype"])
    cohort["arm"] = arm
    return per_stay.merge(
        cohort[["stay_id", "hf_subtype", "arm"]], on="stay_id", how="inner"
    )


def summarize_by_subtype(per_stay: pd.DataFrame) -> pd.DataFrame:
    print("[4/4] Summarizing by HF subtype ...")
    rows = []
    for arm in HF_SUBTYPE_ORDER:
        sub = per_stay[per_stay["arm"] == arm]
        sub_valid = sub[sub["pre_t0_hfi"].notna()]
        if len(sub_valid) == 0:
            continue
        rows.append(
            {
                "arm": arm,
                "n_total": int(len(sub)),
                "n_with_hfi": int(len(sub_valid)),
                "pre_t0_hfi_mean": float(sub_valid["pre_t0_hfi"].mean()),
                "pre_t0_hfi_sd": float(sub_valid["pre_t0_hfi"].std(ddof=1)),
                "frac_below_65_mean": float(sub_valid["frac_below_65"].mean()),
                "map_cv_mean": float(sub_valid["map_cv"].mean()),
                "map_range_mean": float(sub_valid["map_range"].mean()),
            }
        )
    summary = pd.DataFrame(rows)

    # Welch's t-test and Cohen's d for HFpEF vs each other arm
    hfpef_x = per_stay.loc[
        (per_stay["arm"] == "hfpef") & per_stay["pre_t0_hfi"].notna(), "pre_t0_hfi"
    ].values
    welch_rows = []
    for arm in HF_SUBTYPE_ORDER:
        if arm == "hfpef":
            continue
        other_x = per_stay.loc[
            (per_stay["arm"] == arm) & per_stay["pre_t0_hfi"].notna(), "pre_t0_hfi"
        ].values
        if len(other_x) < 2 or len(hfpef_x) < 2:
            continue
        t, p = stats.ttest_ind(hfpef_x, other_x, equal_var=False)
        pooled_sd = np.sqrt(
            (hfpef_x.var(ddof=1) + other_x.var(ddof=1)) / 2.0
        )
        d = (hfpef_x.mean() - other_x.mean()) / pooled_sd if pooled_sd > 0 else np.nan
        welch_rows.append(
            {
                "comparison": f"hfpef_vs_{arm}",
                "n_hfpef": int(len(hfpef_x)),
                "n_other": int(len(other_x)),
                "mean_diff": float(hfpef_x.mean() - other_x.mean()),
                "welch_t": float(t),
                "welch_p": float(p),
                "cohens_d": float(d),
            }
        )
    welch = pd.DataFrame(welch_rows)
    return summary, welch


def quartile_distribution(per_stay: pd.DataFrame) -> pd.DataFrame:
    valid = per_stay[per_stay["pre_t0_hfi_quartile"].notna()]
    table = (
        pd.crosstab(valid["arm"], valid["pre_t0_hfi_quartile"], normalize="index") * 100.0
    ).round(1)
    table.columns = [f"{c}_pct" for c in table.columns]
    counts = pd.crosstab(valid["arm"], valid["pre_t0_hfi_quartile"])
    counts.columns = [f"{c}_n" for c in counts.columns]
    return pd.concat([counts, table], axis=1).reindex(HF_SUBTYPE_ORDER).reset_index()


def write_report(summary: pd.DataFrame, welch: pd.DataFrame, qdist: pd.DataFrame, n_total: int) -> None:
    lines = [
        "=" * 72,
        "P9b — Pre-T0 (0-6 h) Hemodynamic Fragility Index Summary",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 72,
        f"Total per-stay records: {n_total:,}",
        "",
        "PRE-T0 HFI BY HF SUBTYPE",
        "-" * 72,
        summary.to_string(index=False),
        "",
        "WELCH HFPEF vs OTHER ARMS",
        "-" * 72,
        welch.to_string(index=False) if len(welch) else "(insufficient data)",
        "",
        "QUARTILE DISTRIBUTION BY ARM (% within arm)",
        "-" * 72,
        qdist.to_string(index=False),
        "",
        "INTERPRETATION",
        "-" * 72,
        "If pre-T0 HFI separates hfpef from no_hf in the same direction as the",
        "post-T0 HFI reported in P9 (manuscript Section X), this supports HFI",
        "as a baseline vulnerability marker that exists prior to and is not",
        "purely a downstream consequence of differential treatment.",
        "",
    ]
    text = "\n".join(lines)
    out_path = REPORTS_DIR / "P9b_pretT0_hfi_summary.txt"
    out_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved: {out_path}")


def main() -> None:
    print("=" * 72)
    print("P9b — Pre-T0 (0-6 h) Hemodynamic Fragility Index")
    print("=" * 72)

    vitals = load_pre_t0_map()
    per_stay = compute_pre_t0_components(vitals)
    per_stay = compute_composite(per_stay)
    per_stay = join_hf_subtype(per_stay)

    out_parquet = TABLES_DIR / "P9b_pretT0_hfi_per_stay.parquet"
    per_stay.to_parquet(out_parquet, index=False)
    print(f"\nSaved per-stay table: {out_parquet} ({len(per_stay):,} rows)")

    summary, welch = summarize_by_subtype(per_stay)
    summary_path = TABLES_DIR / "P9b_pretT0_hfi_by_hf_subtype.csv"
    welch_path = TABLES_DIR / "P9b_pretT0_hfi_welch_vs_hfpef.csv"
    summary.to_csv(summary_path, index=False)
    welch.to_csv(welch_path, index=False)
    print(f"Saved subtype summary: {summary_path}")
    print(f"Saved Welch comparisons: {welch_path}")

    qdist = quartile_distribution(per_stay)
    qdist_path = TABLES_DIR / "P9b_pretT0_hfi_quartile_by_hf_subtype.csv"
    qdist.to_csv(qdist_path, index=False)
    print(f"Saved quartile distribution: {qdist_path}")

    write_report(summary, welch, qdist, n_total=len(per_stay))


if __name__ == "__main__":
    sys.exit(main())
