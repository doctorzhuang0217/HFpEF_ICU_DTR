#!/usr/bin/env python3
"""P_table1_statistical_tests — Add p-values and SMD to manuscript Table 1.

Protocol: supplementary analysis (between-group statistical comparisons
for Table 1).
Compute tier: C1 (read + groupby + scipy tests; ~10 s).

Purpose
-------
The frozen `tables/manuscript_table1_baseline.csv` is descriptive only
(counts, medians, percentages). This adds explicit
between-group inference. This script computes, for every Table 1 row
characteristic:

  - omnibus p-value across the four arms (HFpEF / HFmrEF+HFrEF /
    HF-EF-unknown / no-HF):
      * continuous           -> Kruskal-Wallis H test
      * binary               -> chi-square (Fisher's exact when any
                                expected cell < 5)
      * multi-level cat      -> chi-square (Cramer's V also reported)
  - pairwise SMD vs HFpEF for every other arm and the maximum
    absolute SMD vs HFpEF (an effect size that does not depend on n).

The table is **not** intended to drive multiple-testing corrections;
it is a descriptive cross-check. Tests are reported as descriptive
comparisons, not as confirmatory analyses.

Inputs
------
- data/interim/35a_comparator_cohorts_v1_mimic.parquet
- data/interim/45d_outcomes_absorb_v1_mimic.parquet  (mortality_28d)

Output
------
- tables/manuscript_table1_baseline_stats.csv

CLI
---
    env/run_py.sh scripts/P_table1_statistical_tests.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
INTERIM_DIR = ROOT / "data" / "interim"
TABLES_DIR = ROOT / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

ARMS = ["hfpef", "hfref_hfmref", "hf_ef_unknown", "no_hf"]
REFERENCE_ARM = "hfpef"
OTHER_ARMS = [a for a in ARMS if a != REFERENCE_ARM]

# Continuous variables: omnibus = Kruskal-Wallis; SMD = standardized mean difference.
CONTINUOUS_VARS = [
    ("age", "Age (years)"),
    ("icu_los_hours", "ICU length of stay (hours)"),
    ("ef_value", "Ejection fraction (%)"),
]

# Binary variables: omnibus = chi-square (or Fisher when expected counts low).
BINARY_VARS = [
    ("female", "% Female", lambda df: (df["gender"] == "F").astype(int)),
    ("hospital_mortality", "Hospital mortality", lambda df: df["hospital_mortality"].fillna(0).astype(int)),
    ("mortality_28d", "28-day mortality", lambda df: df["mortality_28d"].fillna(0).astype(int)),
]

# Multi-level categorical variables: omnibus = chi-square. SMD reported per level.
CATEGORICAL_VARS = [
    ("dominant_syndrome", "Dominant hemodynamic syndrome", ["shock", "congestion", "mixed", "neither"]),
    ("reason_for_icu", "Reason for ICU admission", None),  # levels auto-detected
]


# ── effect size helpers ─────────────────────────────────────────────


def smd_continuous(x1: pd.Series, x2: pd.Series) -> float:
    x1, x2 = x1.dropna(), x2.dropna()
    if len(x1) < 2 or len(x2) < 2:
        return np.nan
    m1, m2 = x1.mean(), x2.mean()
    v1, v2 = x1.var(ddof=1), x2.var(ddof=1)
    pooled = np.sqrt((v1 + v2) / 2.0)
    if pooled == 0 or np.isnan(pooled):
        return np.nan
    return float((m1 - m2) / pooled)


def smd_binary(p1_n: tuple[int, int], p2_n: tuple[int, int]) -> float:
    """SMD for proportions. p1_n = (events, total)."""
    e1, n1 = p1_n
    e2, n2 = p2_n
    if n1 == 0 or n2 == 0:
        return np.nan
    p1, p2 = e1 / n1, e2 / n2
    pooled = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2.0)
    if pooled == 0:
        return 0.0 if p1 == p2 else np.nan
    return float((p1 - p2) / pooled)


def cramers_v(table: np.ndarray) -> float:
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    r, c = table.shape
    if n == 0 or min(r - 1, c - 1) == 0:
        return np.nan
    return float(np.sqrt(chi2 / (n * min(r - 1, c - 1))))


# ── statistical tests ──────────────────────────────────────────────


def test_continuous(df: pd.DataFrame, var: str) -> dict:
    samples = [df.loc[df["arm"] == arm, var].dropna().values for arm in ARMS]
    samples_nonempty = [s for s in samples if len(s) >= 2]
    if len(samples_nonempty) < 2:
        return {"p_overall": np.nan, "test_used": "insufficient_data"}
    h, p = stats.kruskal(*samples_nonempty)
    return {"p_overall": float(p), "test_used": "kruskal_wallis", "test_stat": float(h)}


def test_binary(df: pd.DataFrame, indicator: pd.Series) -> dict:
    table = (
        pd.crosstab(df["arm"], indicator)
        .reindex(index=ARMS, fill_value=0)
    )
    if table.shape[1] < 2:
        # all 0 or all 1 — no variation
        return {"p_overall": 1.0, "test_used": "no_variation"}
    table_arr = table.values
    chi2, p_chi, _, expected = stats.chi2_contingency(table_arr, correction=False)
    test_used = "chi2"
    if (expected < 5).any():
        # 2 × 4 → not directly Fisher; fall back to chi2 with continuity warning
        # but for binary outcomes use Freeman-Halton via Fisher when possible.
        try:
            p_chi = stats.fisher_exact(table_arr) if table_arr.shape == (2, 2) else (chi2, p_chi)
            p_chi = p_chi[1] if isinstance(p_chi, tuple) else p_chi
            test_used = "fisher" if table_arr.shape == (2, 2) else "chi2_low_expected"
        except ValueError:
            test_used = "chi2_low_expected"
    return {"p_overall": float(p_chi), "test_used": test_used, "test_stat": float(chi2)}


def test_categorical(df: pd.DataFrame, var: str, levels: list[str] | None) -> dict:
    if levels is None:
        levels = sorted(df[var].dropna().unique().tolist())
    table = (
        pd.crosstab(df["arm"], df[var])
        .reindex(index=ARMS, columns=levels, fill_value=0)
    )
    if table.shape[1] < 2:
        return {"p_overall": np.nan, "test_used": "single_level", "cramers_v": np.nan, "levels": levels}
    chi2, p, _, expected = stats.chi2_contingency(table.values, correction=False)
    v = cramers_v(table.values)
    test_used = "chi2_low_expected" if (expected < 5).any() else "chi2"
    return {
        "p_overall": float(p),
        "test_used": test_used,
        "cramers_v": float(v),
        "test_stat": float(chi2),
        "levels": levels,
    }


# ── pairwise SMD vs HFpEF ──────────────────────────────────────────


def smd_continuous_vs_ref(df: pd.DataFrame, var: str) -> dict:
    ref = df.loc[df["arm"] == REFERENCE_ARM, var]
    out: dict = {}
    smds: list[float] = []
    for arm in OTHER_ARMS:
        cur = df.loc[df["arm"] == arm, var]
        s = smd_continuous(ref, cur)
        out[f"smd_{REFERENCE_ARM}_vs_{arm}"] = s
        if not np.isnan(s):
            smds.append(abs(s))
    out["smd_max_abs_vs_hfpef"] = max(smds) if smds else np.nan
    return out


def smd_binary_vs_ref(df: pd.DataFrame, indicator: pd.Series) -> dict:
    df_local = df.assign(_ind=indicator.values)
    ref_sub = df_local[df_local["arm"] == REFERENCE_ARM]
    p_ref = (int(ref_sub["_ind"].sum()), len(ref_sub))
    out: dict = {}
    smds: list[float] = []
    for arm in OTHER_ARMS:
        sub = df_local[df_local["arm"] == arm]
        p_arm = (int(sub["_ind"].sum()), len(sub))
        s = smd_binary(p_ref, p_arm)
        out[f"smd_{REFERENCE_ARM}_vs_{arm}"] = s
        if not np.isnan(s):
            smds.append(abs(s))
    out["smd_max_abs_vs_hfpef"] = max(smds) if smds else np.nan
    return out


def smd_categorical_vs_ref(df: pd.DataFrame, var: str, levels: list[str]) -> dict:
    """For each level, treat as binary (level vs not-level) and compute SMD vs HFpEF.
    Report the maximum absolute level-wise SMD per arm."""
    out: dict = {}
    smds_overall: list[float] = []
    for arm in OTHER_ARMS:
        per_level: list[float] = []
        for lvl in levels:
            ind_ref = (df.loc[df["arm"] == REFERENCE_ARM, var] == lvl).astype(int)
            ind_arm = (df.loc[df["arm"] == arm, var] == lvl).astype(int)
            p_ref = (int(ind_ref.sum()), len(ind_ref))
            p_arm = (int(ind_arm.sum()), len(ind_arm))
            s = smd_binary(p_ref, p_arm)
            if not np.isnan(s):
                per_level.append(abs(s))
        max_pl = max(per_level) if per_level else np.nan
        out[f"smd_{REFERENCE_ARM}_vs_{arm}_max_level"] = max_pl
        if not np.isnan(max_pl):
            smds_overall.append(max_pl)
    out["smd_max_abs_vs_hfpef"] = max(smds_overall) if smds_overall else np.nan
    return out


# ── descriptive summaries (per arm) ────────────────────────────────


def summarize_continuous(df: pd.DataFrame, var: str) -> dict:
    out: dict = {}
    for arm in ARMS:
        x = df.loc[df["arm"] == arm, var].dropna()
        if len(x) == 0:
            out[f"{arm}_summary"] = "NA"
        else:
            out[f"{arm}_summary"] = (
                f"{x.median():.1f} [{x.quantile(0.25):.1f}, {x.quantile(0.75):.1f}] (n={len(x)})"
            )
    return out


def summarize_binary(df: pd.DataFrame, indicator: pd.Series) -> dict:
    df_local = df.assign(_ind=indicator.values)
    out: dict = {}
    for arm in ARMS:
        sub = df_local[df_local["arm"] == arm]
        n = len(sub)
        e = int(sub["_ind"].sum())
        out[f"{arm}_summary"] = f"{e}/{n} ({e / n * 100:.1f}%)" if n else "NA"
    return out


def summarize_categorical(df: pd.DataFrame, var: str, levels: list[str]) -> dict:
    out: dict = {}
    for arm in ARMS:
        sub = df[df["arm"] == arm]
        n = len(sub)
        if n == 0:
            out[f"{arm}_summary"] = "NA"
            continue
        parts = []
        for lvl in levels:
            cnt = int((sub[var] == lvl).sum())
            parts.append(f"{lvl}={cnt} ({cnt / n * 100:.1f}%)")
        out[f"{arm}_summary"] = "; ".join(parts)
    return out


# ── pipeline ───────────────────────────────────────────────────────


def load_per_stay() -> pd.DataFrame:
    cohort = pd.read_parquet(INTERIM_DIR / "35a_comparator_cohorts_v1_mimic.parquet")
    if "comparator_group" in cohort.columns:
        cohort = cohort.rename(columns={"comparator_group": "arm"})
    elif "analysis_arm" in cohort.columns:
        cohort = cohort.rename(columns={"analysis_arm": "arm"})

    outcomes_path = INTERIM_DIR / "45d_outcomes_absorb_v1_mimic.parquet"
    if outcomes_path.exists():
        outcomes = pd.read_parquet(outcomes_path, columns=["stay_id", "mortality_28d"])
        cohort = cohort.merge(outcomes, on="stay_id", how="left")
    else:
        print(f"  warning: {outcomes_path} missing — skipping mortality_28d test")
        cohort["mortality_28d"] = np.nan

    n_per_arm = cohort["arm"].value_counts().to_dict()
    print(f"  per-stay cohort: {len(cohort):,} stays, arm distribution = {n_per_arm}")
    return cohort


def main() -> None:
    print("=" * 72)
    print("Table 1 statistical tests (omnibus p + SMD vs HFpEF)")
    print("=" * 72)
    df = load_per_stay()
    df = df[df["arm"].isin(ARMS)].copy()

    rows: list[dict] = []

    # Continuous variables
    for var, label in CONTINUOUS_VARS:
        if var not in df.columns:
            print(f"  skip {var}: not in cohort columns")
            continue
        row: dict = {"variable": var, "label": label, "type": "continuous"}
        row.update(summarize_continuous(df, var))
        row.update(test_continuous(df, var))
        row.update(smd_continuous_vs_ref(df, var))
        rows.append(row)

    # Binary variables
    for var, label, fn in BINARY_VARS:
        try:
            indicator = fn(df)
        except KeyError:
            print(f"  skip {var}: source columns missing")
            continue
        if indicator.isna().all():
            continue
        row = {"variable": var, "label": label, "type": "binary"}
        row.update(summarize_binary(df, indicator))
        row.update(test_binary(df, indicator))
        row.update(smd_binary_vs_ref(df, indicator))
        rows.append(row)

    # Multi-level categorical variables
    for var, label, levels in CATEGORICAL_VARS:
        if var not in df.columns:
            print(f"  skip {var}: not in cohort columns")
            continue
        if levels is None:
            levels = sorted(df[var].dropna().unique().tolist())
        row = {"variable": var, "label": label, "type": "categorical", "levels": ", ".join(map(str, levels))}
        row.update(summarize_categorical(df, var, levels))
        cat_test = test_categorical(df, var, levels)
        row["p_overall"] = cat_test["p_overall"]
        row["test_used"] = cat_test["test_used"]
        row["cramers_v"] = cat_test.get("cramers_v")
        row.update(smd_categorical_vs_ref(df, var, levels))
        rows.append(row)

    out = pd.DataFrame(rows)

    # Round numeric columns for readability
    for col in out.columns:
        if out[col].dtype.kind == "f":
            out[col] = out[col].round(4)

    out_path = TABLES_DIR / "manuscript_table1_baseline_stats.csv"
    out.to_csv(out_path, index=False)

    print("\n" + out.to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    sys.exit(main())
