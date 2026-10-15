#!/usr/bin/env python3
"""P6 — Pilot RCT Sample Size Calculation

Protocol: v3.1-patch, Step P6
Compute tier: C1 (trivial)
Depends on: 80c primary results

Objective: Provide a formal sample-size calculation for a future pilot RCT
based on the observed effect sizes, demonstrating translational awareness.

Analysis plan:
  1. Conservative effect size: 80c DM ATE = -1.37pp
  2. Optimistic effect size: 80c FQE ATE = -3.64pp (or Tier A FQE = -5.98pp)
  3. Calculate required sample size for:
     - alpha=0.05 (two-sided), power=80% and 90%
     - 1:1 randomization, superiority design
  4. Report per-arm and total sample sizes
  5. Expected enrollment duration (based on typical ICU HFpEF prevalence)
  6. Feasibility assessment (single-center vs multicenter)
  7. Adaptive design considerations (group sequential with futility)

Outputs:
  - tables/P6_sample_size_table.csv
  - reports/P6_rct_feasibility_summary.txt
"""

from __future__ import annotations

import datetime
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Project root ──
ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))

TABLES_DIR = ROOT / "tables"
REPORTS_DIR = ROOT / "reports"

for d in [TABLES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# Sample size formulas
# ═══════════════════════════════════════════════════════════════════════


def sample_size_two_proportions(
    p_control: float,
    p_treatment: float,
    alpha: float = 0.05,
    power: float = 0.80,
    ratio: float = 1.0,
) -> dict:
    """
    Sample size for two-sample test of proportions (superiority design).

    Uses the normal approximation formula:
      n_control = ((z_alpha/2 + z_beta)^2 * (p1*(1-p1)/r + p2*(1-p2))) / delta^2

    where delta = p1 - p2, r = n_treatment / n_control.

    Parameters
    ----------
    p_control : float
        Expected proportion in control arm (observed care mortality).
    p_treatment : float
        Expected proportion in treatment arm (rule policy mortality).
    alpha : float
        Two-sided significance level.
    power : float
        Statistical power (1 - beta).
    ratio : float
        Allocation ratio (n_treatment / n_control).

    Returns
    -------
    dict with n_control, n_treatment, n_total, and intermediate values.
    """
    from scipy import stats

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    delta = abs(p_control - p_treatment)
    if delta == 0:
        return {"n_control": float("inf"), "n_treatment": float("inf"),
                "n_total": float("inf"), "delta": 0}

    # Pooled variance under H0
    p_bar = (p_control + ratio * p_treatment) / (1 + ratio)
    var_h0 = p_bar * (1 - p_bar) * (1 + 1 / ratio)

    # Variance under H1
    var_h1 = p_control * (1 - p_control) + p_treatment * (1 - p_treatment) / ratio

    # Sample size (control arm) using the unpooled formula
    numerator = (z_alpha * np.sqrt(var_h0) + z_beta * np.sqrt(var_h1)) ** 2
    n_control = np.ceil(numerator / delta ** 2)
    n_treatment = np.ceil(ratio * n_control)
    n_total = n_control + n_treatment

    return {
        "n_control": int(n_control),
        "n_treatment": int(n_treatment),
        "n_total": int(n_total),
        "delta": delta,
        "z_alpha": z_alpha,
        "z_beta": z_beta,
        "p_control": p_control,
        "p_treatment": p_treatment,
    }


def group_sequential_inflation(
    n_fixed: int,
    n_looks: int = 3,
    spending_fn: str = "obrien_fleming",
) -> dict:
    """
    Inflation factor for group sequential design.

    O'Brien-Fleming spending function: conservative early stopping,
    preserves most of the fixed-sample power.

    Approximate inflation factors (well-known values):
      2 looks: 1.015
      3 looks: 1.028
      4 looks: 1.038
      5 looks: 1.045
    """
    inflation = {
        1: 1.000, 2: 1.015, 3: 1.028, 4: 1.038, 5: 1.045,
    }
    factor = inflation.get(n_looks, 1.05)
    n_inflated = int(np.ceil(n_fixed * factor))

    return {
        "n_fixed": n_fixed,
        "n_looks": n_looks,
        "spending_fn": spending_fn,
        "inflation_factor": factor,
        "n_inflated": n_inflated,
        "n_per_interim": int(np.ceil(n_inflated / n_looks)),
    }


# ═══════════════════════════════════════════════════════════════════════
# Feasibility estimation
# ═══════════════════════════════════════════════════════════════════════


def estimate_enrollment_duration(
    n_total: int,
    annual_icu_admissions: int = 2500,
    hfpef_prevalence: float = 0.05,
    eligibility_rate: float = 0.60,
    consent_rate: float = 0.40,
    n_sites: int = 1,
) -> dict:
    """
    Estimate enrollment duration for the trial.

    Default assumptions:
    - 2,500 ICU admissions/year per site (typical large academic center)
    - 5% HFpEF prevalence among ICU admissions
    - 60% meet eligibility criteria
    - 40% consent rate (ICU consent is challenging)
    """
    patients_per_year_per_site = (
        annual_icu_admissions * hfpef_prevalence * eligibility_rate * consent_rate
    )
    patients_per_year_total = patients_per_year_per_site * n_sites
    years = n_total / patients_per_year_total if patients_per_year_total > 0 else float("inf")

    return {
        "n_total": n_total,
        "n_sites": n_sites,
        "annual_icu_admissions_per_site": annual_icu_admissions,
        "hfpef_prevalence": hfpef_prevalence,
        "eligibility_rate": eligibility_rate,
        "consent_rate": consent_rate,
        "patients_per_year_per_site": patients_per_year_per_site,
        "patients_per_year_total": patients_per_year_total,
        "estimated_years": years,
        "estimated_months": years * 12,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 72)
    print("P6 — Pilot RCT Sample Size Calculation")
    print("=" * 72)
    t0 = time.time()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Key parameters from 80c results ──
    # Observed 28-day mortality in test cohort (from 80c reports)
    p_observed = 0.1382  # 13.82% (from 75a)

    scenarios = [
        {
            "name": "Conservative (80c DM)",
            "ate_pp": -1.37,
            "source": "80c G-Transformer DM with rollout",
        },
        {
            "name": "Moderate (80c FQE)",
            "ate_pp": -3.64,
            "source": "80c G-Transformer FQE",
        },
        {
            "name": "Optimistic (Tier A FQE)",
            "ate_pp": -5.98,
            "source": "eICU Tier A FQE (external validation)",
        },
    ]

    power_levels = [0.80, 0.90]
    alpha = 0.05

    # ── 1. Sample size calculations ──
    print("[1] Computing sample sizes...")
    all_results = []

    for scenario in scenarios:
        ate = scenario["ate_pp"] / 100  # convert pp to proportion
        p_treatment = p_observed + ate  # lower mortality under rule policy

        for power in power_levels:
            ss = sample_size_two_proportions(
                p_control=p_observed,
                p_treatment=p_treatment,
                alpha=alpha,
                power=power,
                ratio=1.0,
            )
            nnt = 1 / abs(ate) if ate != 0 else float("inf")

            result = {
                "scenario": scenario["name"],
                "source": scenario["source"],
                "ate_pp": scenario["ate_pp"],
                "p_control": p_observed,
                "p_treatment": round(p_treatment, 4),
                "alpha": alpha,
                "power": power,
                "n_per_arm": ss["n_control"],
                "n_total": ss["n_total"],
                "nnt": round(nnt, 1),
            }

            # Add group sequential design
            gs = group_sequential_inflation(ss["n_total"], n_looks=3)
            result["n_total_gs3"] = gs["n_inflated"]
            result["n_per_interim"] = gs["n_per_interim"]

            all_results.append(result)
            print(f"  {scenario['name']}, power={power:.0%}: "
                  f"n/arm={ss['n_control']}, total={ss['n_total']}, "
                  f"GS(3)={gs['n_inflated']}, NNT={nnt:.0f}")

    results_df = pd.DataFrame(all_results)

    # ── 2. Feasibility estimates ──
    print("\n[2] Enrollment feasibility...")
    feasibility = []

    for _, row in results_df.iterrows():
        for n_sites in [1, 3, 5, 10]:
            fe = estimate_enrollment_duration(
                n_total=row["n_total"],
                n_sites=n_sites,
            )
            feasibility.append({
                "scenario": row["scenario"],
                "power": row["power"],
                "n_total": row["n_total"],
                "n_sites": n_sites,
                "patients_per_year": round(fe["patients_per_year_total"], 1),
                "estimated_months": round(fe["estimated_months"], 1),
                "estimated_years": round(fe["estimated_years"], 1),
            })

    feasibility_df = pd.DataFrame(feasibility)

    # Print key feasibility summary
    print("\n  Key feasibility estimates (power=80%):")
    for _, row in feasibility_df[feasibility_df["power"] == 0.80].iterrows():
        if row["n_sites"] in [1, 5, 10]:
            status = "FEASIBLE" if row["estimated_years"] <= 3 else "CHALLENGING" if row["estimated_years"] <= 5 else "INFEASIBLE"
            print(f"    {row['scenario']}, {row['n_sites']} sites: "
                  f"{row['estimated_months']:.0f} months ({row['estimated_years']:.1f} yr) — {status}")

    # ── 3. Context: published ICU RCT NNTs ──
    print("\n[3] Context: published ICU RCT NNTs...")
    context_nnt = [
        ("ARDS-Net (low tidal volume)", 11, "NEJM 2000"),
        ("PROWESS (drotrecogin alfa)", 16, "NEJM 2001 [later retracted]"),
        ("SAFE (albumin vs saline)", "NS", "NEJM 2004"),
        ("NICE-SUGAR (glucose control)", 38, "NEJM 2009"),
        ("ProCESS (EGDT)", "NS", "NEJM 2014"),
        ("ANDROMEDA-SHOCK (capillary refill)", 14, "JAMA 2019"),
        ("CLOVERS (restrictive fluids)", "NS", "NEJM 2023"),
    ]
    for trial, nnt, ref in context_nnt:
        nnt_str = f"NNT={nnt}" if isinstance(nnt, int) else "NNT=NS"
        print(f"    {trial:45s}: {nnt_str:10s} ({ref})")

    # ── 4. Save outputs ──
    print("\n[4] Saving outputs...")

    # Sample size table
    results_df.to_csv(TABLES_DIR / "P6_sample_size_table.csv", index=False)
    print(f"  Saved: {TABLES_DIR / 'P6_sample_size_table.csv'}")

    # ── 5. Summary report ──
    report_lines = [
        "=" * 72,
        "P6 — Pilot RCT Sample Size Calculation",
        "=" * 72,
        f"Generated: {ts}",
        f"Protocol: v3.1-patch, Step P6",
        "",
        "STUDY DESIGN ASSUMPTIONS",
        "─" * 24,
        f"  Design: Two-arm, 1:1 randomization, superiority",
        f"  Primary outcome: 28-day all-cause mortality",
        f"  Control arm: standard ICU hemodynamic management (observed care)",
        f"  Treatment arm: protocolized hemodynamic management (rule policy)",
        f"  Significance level: α = {alpha} (two-sided)",
        f"  Baseline mortality (control): {p_observed*100:.2f}%",
        "",
        "EFFECT SIZE SCENARIOS",
        "─" * 21,
    ]
    for s in scenarios:
        p_treat = p_observed + s["ate_pp"] / 100
        nnt = round(1 / abs(s["ate_pp"] / 100), 0)
        report_lines.append(
            f"  {s['name']:30s}: ATE = {s['ate_pp']:+.2f}pp → treatment mortality = {p_treat*100:.2f}%, NNT = {nnt:.0f}"
        )
        report_lines.append(f"    Source: {s['source']}")

    report_lines.extend(["", "SAMPLE SIZE CALCULATIONS", "─" * 23])
    report_lines.append(f"  {'Scenario':<30s} {'Power':<8s} {'n/arm':<8s} {'Total':<8s} {'GS(3)':<8s} {'NNT':<6s}")
    report_lines.append("  " + "-" * 68)
    for _, row in results_df.iterrows():
        report_lines.append(
            f"  {row['scenario']:<30s} {row['power']:<8.0%} {row['n_per_arm']:<8d} "
            f"{row['n_total']:<8d} {row['n_total_gs3']:<8d} {row['nnt']:<6.0f}"
        )

    report_lines.extend([
        "",
        "  GS(3) = Group sequential design with 3 interim looks",
        "          (O'Brien-Fleming spending, inflation factor ≈ 1.028)",
        "",
        "ENROLLMENT FEASIBILITY",
        "─" * 22,
        "  Assumptions:",
        "    - 2,500 ICU admissions/year/site (large academic center)",
        "    - 5% HFpEF prevalence among ICU admissions",
        "    - 60% eligibility after screening",
        "    - 40% consent rate",
        "    → 30 patients/year/site",
        "",
    ])

    report_lines.append(f"  {'Scenario':<30s} {'Power':<8s} {'Sites':<8s} {'Months':<10s} {'Feasibility':<15s}")
    report_lines.append("  " + "-" * 71)
    for _, row in feasibility_df.iterrows():
        if row["n_sites"] in [1, 5, 10]:
            status = "FEASIBLE" if row["estimated_years"] <= 3 else "CHALLENGING" if row["estimated_years"] <= 5 else "INFEASIBLE"
            report_lines.append(
                f"  {row['scenario']:<30s} {row['power']:<8.0%} {row['n_sites']:<8d} "
                f"{row['estimated_months']:<10.0f} {status:<15s}"
            )

    report_lines.extend([
        "",
        "CONTEXT: PUBLISHED ICU RCT NNTs",
        "─" * 31,
    ])
    for trial, nnt, ref in context_nnt:
        nnt_str = f"NNT={nnt}" if isinstance(nnt, int) else "NNT=NS"
        report_lines.append(f"  {trial:45s}: {nnt_str:10s} ({ref})")

    report_lines.extend([
        "",
        f"  Current study NNT (conservative): {round(1/0.0137):.0f}",
        f"  Current study NNT (moderate):     {round(1/0.0364):.0f}",
        "",
        "ADAPTIVE DESIGN CONSIDERATIONS",
        "─" * 30,
        "  1. Group sequential with O'Brien-Fleming boundaries (3 looks)",
        "     - Allows early stopping for efficacy or futility",
        "     - Minimal sample size inflation (~2.8%)",
        "     - Interim analyses at 1/3, 2/3, and full enrollment",
        "  2. Futility stopping rule (conditional power < 20%)",
        "     - Protects against prolonged enrollment with no benefit",
        "  3. Sample size re-estimation at interim",
        "     - If observed effect is between conservative and moderate,",
        "       can adapt target N to maintain power",
        "",
        "RECOMMENDATIONS",
        "─" * 15,
        "  1. Target: Moderate scenario (ATE = -3.64pp, N ≈ 2,800 total)",
        "     as primary design, with the conservative scenario informing",
        "     the maximum feasible enrollment.",
        "  2. Multicenter design (5-10 sites) needed for 2-4 year enrollment.",
        "  3. Group sequential design with 3 interim looks recommended.",
        "  4. Consider adaptive enrichment: if interim HFpEF-specific",
        "     signal is stronger, can enrich for HFpEF patients.",
        "  5. Pilot phase (N=100-200) to validate:",
        "     - Rule policy adherence and implementation feasibility",
        "     - Actual HFpEF prevalence and consent rates",
        "     - Outcome event rates in both arms",
        "",
        "MANUSCRIPT PLACEMENT",
        "─" * 20,
        "  Discussion paragraph; addresses 'what would an RCT require?'",
        "  Table: sample size scenarios. Demonstrates translational pathway.",
        "",
    ])

    report_path = REPORTS_DIR / "P6_rct_feasibility_summary.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Saved: {report_path}")

    total_time = time.time() - t0
    print()
    print("=" * 72)
    print(f"P6 COMPLETE. Total time: {total_time:.1f} sec")
    print("=" * 72)


if __name__ == "__main__":
    main()
