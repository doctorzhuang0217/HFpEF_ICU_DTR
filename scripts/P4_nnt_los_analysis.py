#!/usr/bin/env python3
"""P4 — NNT and ICU LOS Secondary Endpoint

Protocol: v3.1-patch, Step P4
Compute tier: C2 (lightweight)
Depends on: 80c primary results, cohort data

Objective: Compute Number Needed to Treat (NNT) and evaluate whether the
rule policy affects ICU length of stay as a secondary endpoint.

Analysis:
  1. NNT = 1/|ATE| with CI from inverted ATE CI
  2. NNT by HF subtype and phenotype stratum
  3. ICU LOS stratified by predicted CATE tertiles (proxy analysis)

Outputs:
  - tables/P4_nnt_by_subgroup.csv
  - tables/P4_los_comparison.csv
  - reports/P4_nnt_los_summary.txt
"""

from __future__ import annotations

import datetime
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ── Project root ──
ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(ROOT / "src"))

TABLES_DIR = ROOT / "tables"
REPORTS_DIR = ROOT / "reports"

for d in [TABLES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def compute_nnt(ate: float, ate_ci_lo: float = None, ate_ci_hi: float = None) -> dict:
    """Compute NNT with CI (inverted ATE CI)."""
    nnt = 1.0 / abs(ate) if ate != 0 else float("inf")
    result = {"ate": ate, "nnt": nnt}

    if ate_ci_lo is not None and ate_ci_hi is not None:
        # For negative ATE (benefit): NNT CI = [1/|ci_lo|, 1/|ci_hi|]
        # Note: more negative ATE → smaller NNT (better)
        if ate_ci_lo != 0 and ate_ci_hi != 0:
            nnt_hi = 1.0 / abs(ate_ci_hi)  # smaller |ATE| → larger NNT
            nnt_lo = 1.0 / abs(ate_ci_lo)  # larger |ATE| → smaller NNT
            result["nnt_ci_lo"] = min(nnt_lo, nnt_hi)
            result["nnt_ci_hi"] = max(nnt_lo, nnt_hi)
        result["ate_ci_lo"] = ate_ci_lo
        result["ate_ci_hi"] = ate_ci_hi

    return result


def main():
    print("=" * 72)
    print("P4 — NNT and ICU LOS Secondary Endpoint")
    print("=" * 72)
    t0 = time.time()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 1. Load existing results ──
    print("[1] Loading 80c results and cohort data...")

    # Primary 80c DM results — use rollout_ow1 (primary estimate per 80c CRITICAL note)
    policy_values = pd.read_csv(TABLES_DIR / "80c_gtransformer_rollout_ow1_policy_values.csv")
    bootstrap_ci = pd.read_csv(TABLES_DIR / "80c_gtransformer_rollout_ow1_bootstrap_ci.csv")
    subgroup_results = pd.read_csv(TABLES_DIR / "80c_gtransformer_rollout_ow1_subgroup_results.csv")

    # CATE distribution (from P2)
    cate_path = TABLES_DIR / "P2_cate_distribution.csv"
    has_cate = cate_path.exists()
    if has_cate:
        cate_df = pd.read_csv(cate_path)
        print(f"  Loaded P2 CATE distribution ({len(cate_df)} patients)")
    else:
        print("  WARNING: P2 CATE distribution not found, skipping LOS-by-CATE analysis")

    # Load outcomes for LOS data
    with open(ROOT / "configs" / "80_step_c_training_v1.yaml") as f:
        config = yaml.safe_load(f)

    outcomes_df = pd.read_parquet(ROOT / config["data"]["outcomes"])
    pheno_df = pd.read_parquet(ROOT / config["data"]["phenotype_labels"])

    # ── 2. NNT from primary estimates ──
    print("\n[2] Computing NNT from primary estimates...")
    nnt_rows = []

    # Overall estimates by estimator.
    # Overall OPE estimates are evaluated on the held-out test set, not the full
    # analytic cohort; derive the test-set N from a complete partition of it
    # (rule_phenotype_2class = non-shock + shock-predominant covers all test stays).
    _test_part = subgroup_results.loc[
        subgroup_results["stratifier"] == "rule_phenotype_2class", "n_stays"
    ]
    overall_n = int(_test_part.sum()) if len(_test_part) else len(outcomes_df)
    for _, pv in policy_values.iterrows():
        est = pv["estimator"]
        ate = pv["ate"]

        # Get bootstrap CI if available
        boot_row = bootstrap_ci[bootstrap_ci["estimator"] == est]
        ci_lo = ci_hi = None
        if len(boot_row) > 0 and boot_row.iloc[0].get("n_success", 0) > 0:
            ci_lo = boot_row.iloc[0]["ate_ci_lo"]
            ci_hi = boot_row.iloc[0]["ate_ci_hi"]

        nnt = compute_nnt(ate, ci_lo, ci_hi)
        row = {
            "group": "Overall",
            "stratifier": "",
            "estimator": est,
            "n_stays": overall_n,
            **nnt,
        }
        nnt_rows.append(row)
        ci_str = f" [{nnt.get('nnt_ci_lo', 0):.0f}, {nnt.get('nnt_ci_hi', 0):.0f}]" if "nnt_ci_lo" in nnt else ""
        print(f"  {est}: ATE={ate:+.4f} → NNT={nnt['nnt']:.0f}{ci_str}")

    # ── 3. NNT by subgroup ──
    print("\n[3] NNT by subgroup (DM estimator)...")
    for _, sg in subgroup_results.iterrows():
        if sg.get("ate", 0) == 0:
            continue
        nnt = compute_nnt(sg["ate"])
        nnt_rows.append({
            "group": sg["group"],
            "stratifier": sg["stratifier"],
            "estimator": "DM",
            "n_stays": sg["n_stays"],
            **nnt,
        })
        print(f"  {sg['stratifier']}={sg['group']}: ATE={sg['ate']:+.4f} → NNT={nnt['nnt']:.0f} (n={sg['n_stays']})")

    nnt_df = pd.DataFrame(nnt_rows)
    nnt_df.to_csv(TABLES_DIR / "P4_nnt_by_subgroup.csv", index=False)
    print(f"  Saved: {TABLES_DIR / 'P4_nnt_by_subgroup.csv'}")

    # ── 4. ICU LOS analysis by CATE tertile ──
    print("\n[4] ICU LOS by CATE tertile...")

    # Identify LOS column
    los_candidates = ["icu_los_hours", "los_icu_hours", "icu_los_days",
                       "los_icu_days", "icu_length_of_stay_hours"]
    los_col = None
    for c in los_candidates:
        if c in outcomes_df.columns:
            los_col = c
            break

    los_rows = []
    if los_col and has_cate:
        print(f"  Using LOS column: {los_col}")

        # Merge CATE with outcomes
        merged = cate_df.merge(outcomes_df[["stay_id", los_col]], on="stay_id", how="inner")

        # Convert to days if in hours
        if "hour" in los_col:
            merged["los_days"] = merged[los_col] / 24.0
        else:
            merged["los_days"] = merged[los_col]

        # Create CATE tertiles
        merged["cate_tertile"] = pd.qcut(
            merged["cate"], q=3,
            labels=["T1 (benefit)", "T2 (neutral)", "T3 (harm)"]
        )

        # Also merge phenotype
        merged = merged.merge(
            pheno_df[["stay_id", "hf_subtype"]].drop_duplicates("stay_id"),
            on="stay_id", how="left",
        )

        # LOS by CATE tertile
        for tertile, grp in merged.groupby("cate_tertile", observed=True):
            row = {
                "group": str(tertile),
                "stratifier": "cate_tertile",
                "n_stays": len(grp),
                "los_mean_days": grp["los_days"].mean(),
                "los_median_days": grp["los_days"].median(),
                "los_sd_days": grp["los_days"].std(),
                "los_q25_days": grp["los_days"].quantile(0.25),
                "los_q75_days": grp["los_days"].quantile(0.75),
                "mortality_rate": grp["mortality_28d_actual"].mean(),
                "cate_mean_pp": grp["cate"].mean() * 100,
            }
            los_rows.append(row)
            print(f"  {tertile}: n={row['n_stays']}, LOS={row['los_median_days']:.1f}d (median), "
                  f"mort={row['mortality_rate']:.3f}")

        # LOS by HF subtype
        for st, grp in merged.groupby("hf_subtype"):
            los_rows.append({
                "group": str(st),
                "stratifier": "hf_subtype",
                "n_stays": len(grp),
                "los_mean_days": grp["los_days"].mean(),
                "los_median_days": grp["los_days"].median(),
                "los_sd_days": grp["los_days"].std(),
                "los_q25_days": grp["los_days"].quantile(0.25),
                "los_q75_days": grp["los_days"].quantile(0.75),
                "mortality_rate": grp["mortality_28d_actual"].mean(),
                "cate_mean_pp": grp["cate"].mean() * 100,
            })

        # LOS by CATE tertile × HF subtype
        for (tertile, st), grp in merged.groupby(["cate_tertile", "hf_subtype"], observed=True):
            if len(grp) >= 20:
                los_rows.append({
                    "group": f"{tertile} × {st}",
                    "stratifier": "cate_tertile × hf_subtype",
                    "n_stays": len(grp),
                    "los_mean_days": grp["los_days"].mean(),
                    "los_median_days": grp["los_days"].median(),
                    "los_sd_days": grp["los_days"].std(),
                    "los_q25_days": grp["los_days"].quantile(0.25),
                    "los_q75_days": grp["los_days"].quantile(0.75),
                    "mortality_rate": grp["mortality_28d_actual"].mean(),
                    "cate_mean_pp": grp["cate"].mean() * 100,
                })
    elif not los_col:
        print("  WARNING: No LOS column found in outcomes data")
    else:
        print("  WARNING: No CATE data, skipping LOS-by-CATE analysis")

    los_df = pd.DataFrame(los_rows)
    if len(los_df) > 0:
        los_df.to_csv(TABLES_DIR / "P4_los_comparison.csv", index=False)
        print(f"  Saved: {TABLES_DIR / 'P4_los_comparison.csv'}")

    # ── 5. Context: published ICU RCT NNTs ──
    print("\n[5] NNT context...")
    context = [
        ("ARDS-Net (low Vt)", 11, "NEJM 2000"),
        ("ANDROMEDA-SHOCK", 14, "JAMA 2019"),
        ("NICE-SUGAR", 38, "NEJM 2009"),
        ("This study (DM)", round(1.0 / abs(policy_values[policy_values["estimator"] == "DM"].iloc[0]["ate"])), "80c DM"),
    ]

    # ── 6. Summary report ──
    print("\n[6] Generating summary report...")
    report_lines = [
        "=" * 72,
        "P4 — NNT and ICU LOS Secondary Endpoint",
        "=" * 72,
        f"Generated: {ts}",
        f"Protocol: v3.1-patch, Step P4",
        f"Database: MIMIC-IV",
        "",
        "NNT BY ESTIMATOR (Overall)",
        "─" * 27,
    ]
    for _, row in nnt_df[nnt_df["group"] == "Overall"].iterrows():
        ci_str = ""
        if "nnt_ci_lo" in row and pd.notna(row.get("nnt_ci_lo")):
            ci_str = f" [{row['nnt_ci_lo']:.0f}, {row['nnt_ci_hi']:.0f}]"
        report_lines.append(
            f"  {row['estimator']:6s}: ATE={row['ate']:+.4f} → NNT={row['nnt']:.0f}{ci_str}"
        )

    report_lines.extend(["", "NNT BY SUBGROUP (DM)", "─" * 20])
    for _, row in nnt_df[nnt_df["group"] != "Overall"].iterrows():
        report_lines.append(
            f"  {row['stratifier']}={row['group']:25s}: NNT={row['nnt']:.0f} (n={row['n_stays']})"
        )

    report_lines.extend(["", "CONTEXT: PUBLISHED ICU RCT NNTs", "─" * 31])
    for trial, nnt, ref in context:
        report_lines.append(f"  {trial:40s}: NNT={nnt:>4d} ({ref})")

    if len(los_df) > 0:
        report_lines.extend(["", "ICU LOS BY CATE TERTILE", "─" * 23])
        for _, row in los_df[los_df["stratifier"] == "cate_tertile"].iterrows():
            report_lines.append(
                f"  {row['group']:15s}: n={row['n_stays']:>5}, "
                f"LOS={row['los_median_days']:.1f}d (median), "
                f"mort={row['mortality_rate']:.3f}, "
                f"CATE={row['cate_mean_pp']:+.2f}pp"
            )
        report_lines.extend(["", "ICU LOS BY HF SUBTYPE", "─" * 21])
        for _, row in los_df[los_df["stratifier"] == "hf_subtype"].iterrows():
            report_lines.append(
                f"  {row['group']:15s}: n={row['n_stays']:>5}, "
                f"LOS={row['los_median_days']:.1f}d (median), "
                f"CATE={row['cate_mean_pp']:+.2f}pp"
            )

    report_lines.extend([
        "",
        "INTERPRETATION",
        "─" * 14,
        "  NNT provides clinical interpretability: 'how many patients need to be",
        "  treated with the rule policy to prevent one additional death?'",
        "  Lower NNT = more efficient treatment effect.",
        "  LOS-by-CATE analysis is a proxy; the G-Transformer does not directly",
        "  model ICU discharge timing.",
        "",
        "MANUSCRIPT PLACEMENT",
        "─" * 20,
        "  Discussion paragraph + Table (NNT context for clinical interpretation).",
        "",
    ])

    with open(REPORTS_DIR / "P4_nnt_los_summary.txt", "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Saved: {REPORTS_DIR / 'P4_nnt_los_summary.txt'}")

    total_time = time.time() - t0
    print()
    print("=" * 72)
    print(f"P4 COMPLETE. Total time: {total_time:.1f} sec")
    print("=" * 72)


if __name__ == "__main__":
    main()
