#!/usr/bin/env python3
"""
40d_fluid_harmonization_qa — Fluid balance QA and congestion audit.

Purpose
-------
Audit the fluid balance calculation chain from 25d, identify physiologic
outliers, run congestion threshold sensitivity analysis, and produce a
QA report documenting findings and recommendations.

Key questions from Step 35e review:
  1. Why is congestion-dominant only 0.7% of HFpEF (14/2,080)?
  2. Is the extreme negative minimum of fluid_balance_L physiologically plausible?
  3. Are congestion thresholds (>2L AND resp distress) too strict?

Protocol reference: Step 40d (C3).

Inputs
------
- data/staging/25d_dominant_syndrome_flags_mimic.parquet

Outputs
-------
- tables/40d_fluid_balance_qa.csv
- tables/40d_congestion_sensitivity.csv
- reports/40d_fluid_harmonization_qa.txt
"""

from __future__ import annotations

import sys
import textwrap
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path

# Physiologic bounds for QC
FLUID_INPUT_MAX_ML_6H = 15_000   # 15L in 6h is extreme but possible (massive resus)
FLUID_OUTPUT_MAX_ML_6H = 10_000  # 10L output in 6h is extremely unlikely
FLUID_BALANCE_PHYSIO_RANGE_L = (-10, 15)  # anything outside is suspect

# 25d original thresholds
ORIG_FB_THRESH_L = 2.0
ORIG_SPO2_THRESH = 92.0
ORIG_RR_THRESH = 24.0

# Sensitivity thresholds to test
FB_THRESHOLDS_L = [0.5, 1.0, 1.5, 2.0, 3.0]
SPO2_THRESHOLDS = [90.0, 92.0, 94.0, 96.0]
RR_THRESHOLDS = [20.0, 22.0, 24.0, 28.0]


def main() -> None:
    cfg = load_config()
    staging_dir = resolve_path(cfg, "staging")
    tables_dir = resolve_path(cfg, "tables")
    reports_dir = resolve_path(cfg, "reports")
    for d in [tables_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("[40d] Fluid harmonization QA and congestion audit...")

    # ── Load 25d syndrome flags ────────────────────────────────────────
    synd = pd.read_parquet(staging_dir / "25d_dominant_syndrome_flags_mimic.parquet")
    n_total = len(synd)
    print(f"  Total stays: {n_total:,}")

    # Focus on HFpEF for congestion analysis
    hfpef = synd[synd["comparator_group"] == "hfpef"].copy()
    n_hfpef = len(hfpef)
    print(f"  HFpEF stays: {n_hfpef:,}")
    if n_hfpef == 0:
        raise ValueError("No HFpEF rows found in 25d output; cannot run 40d sensitivity analysis.")

    report_lines = [
        "=" * 80,
        "40d — Fluid Harmonization QA & Congestion Audit",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total stays: {n_total:,}",
        f"HFpEF stays: {n_hfpef:,}",
        "",
    ]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 1: Fluid balance distribution audit
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    report_lines.append("§1 FLUID BALANCE DISTRIBUTION (all stays, pre-T0 0–6h)")
    report_lines.append("─" * 60)

    fb = synd["fluid_balance_L"]
    fi = synd["fluid_input_ml"]
    fo = synd["fluid_output_ml"]

    for label, series, unit in [
        ("Fluid input", fi, "mL"),
        ("Fluid output", fo, "mL"),
        ("Fluid balance", fb, "L"),
    ]:
        desc = series.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        report_lines.append(f"\n  {label} ({unit}):")
        for stat, val in desc.items():
            report_lines.append(f"    {stat:<8} {val:>12.2f}")

    # Outlier detection
    report_lines.append("\n  OUTLIER FLAGS:")
    n_fi_extreme = int((fi > FLUID_INPUT_MAX_ML_6H).sum())
    n_fo_extreme = int((fo > FLUID_OUTPUT_MAX_ML_6H).sum())
    n_fb_lo = int((fb < FLUID_BALANCE_PHYSIO_RANGE_L[0]).sum())
    n_fb_hi = int((fb > FLUID_BALANCE_PHYSIO_RANGE_L[1]).sum())
    n_fi_zero = int((fi == 0).sum())
    n_fo_zero = int((fo == 0).sum())

    report_lines.append(f"    Fluid input > {FLUID_INPUT_MAX_ML_6H:,}mL:     {n_fi_extreme:,} ({n_fi_extreme/n_total*100:.2f}%)")
    report_lines.append(f"    Fluid output > {FLUID_OUTPUT_MAX_ML_6H:,}mL:    {n_fo_extreme:,} ({n_fo_extreme/n_total*100:.2f}%)")
    report_lines.append(f"    Fluid balance < {FLUID_BALANCE_PHYSIO_RANGE_L[0]}L:        {n_fb_lo:,} ({n_fb_lo/n_total*100:.2f}%)")
    report_lines.append(f"    Fluid balance > {FLUID_BALANCE_PHYSIO_RANGE_L[1]}L:         {n_fb_hi:,} ({n_fb_hi/n_total*100:.2f}%)")
    report_lines.append(f"    Fluid input = 0:                  {n_fi_zero:,} ({n_fi_zero/n_total*100:.1f}%)")
    report_lines.append(f"    Fluid output = 0:                 {n_fo_zero:,} ({n_fo_zero/n_total*100:.1f}%)")

    # Worst outliers
    if n_fb_lo > 0:
        worst_neg = synd.nsmallest(5, "fluid_balance_L")[["stay_id", "fluid_input_ml", "fluid_output_ml", "fluid_balance_L", "comparator_group"]]
        report_lines.append("\n  WORST NEGATIVE OUTLIERS:")
        for _, r in worst_neg.iterrows():
            report_lines.append(
                f"    stay {r['stay_id']}: in={r['fluid_input_ml']:.0f}mL, "
                f"out={r['fluid_output_ml']:.0f}mL, bal={r['fluid_balance_L']:.2f}L "
                f"({r['comparator_group']})"
            )

    report_lines.append("")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 2: Congestion criteria decomposition (HFpEF only)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    report_lines.append("§2 CONGESTION CRITERIA DECOMPOSITION (HFpEF only)")
    report_lines.append("─" * 60)

    # Individual criteria
    fb_pos_2l = (hfpef["fluid_balance_L"] > ORIG_FB_THRESH_L)
    spo2_low = hfpef["pre_t0_min_spo2"].notna() & (hfpef["pre_t0_min_spo2"] < ORIG_SPO2_THRESH)
    rr_high = hfpef["pre_t0_max_rr"].notna() & (hfpef["pre_t0_max_rr"] > ORIG_RR_THRESH)
    resp_distress = spo2_low | rr_high
    congestion_and = fb_pos_2l & resp_distress
    congestion_or = fb_pos_2l | resp_distress

    report_lines.append(f"\n  Original 25d thresholds:")
    report_lines.append(f"    fluid_balance > {ORIG_FB_THRESH_L}L:             {int(fb_pos_2l.sum()):>6,} / {n_hfpef:,}  ({fb_pos_2l.mean()*100:.1f}%)")
    report_lines.append(f"    SpO2 < {ORIG_SPO2_THRESH}%:                       {int(spo2_low.sum()):>6,} / {n_hfpef:,}  ({spo2_low.mean()*100:.1f}%)")
    report_lines.append(f"    RR > {ORIG_RR_THRESH}:                           {int(rr_high.sum()):>6,} / {n_hfpef:,}  ({rr_high.mean()*100:.1f}%)")
    report_lines.append(f"    Respiratory distress (SpO2 OR RR): {int(resp_distress.sum()):>6,} / {n_hfpef:,}  ({resp_distress.mean()*100:.1f}%)")
    report_lines.append(f"    Congestion (FB AND resp):          {int(congestion_and.sum()):>6,} / {n_hfpef:,}  ({congestion_and.mean()*100:.1f}%)")
    report_lines.append(f"    Congestion (FB OR resp):           {int(congestion_or.sum()):>6,} / {n_hfpef:,}  ({congestion_or.mean()*100:.1f}%)")

    # Why so few meet FB > 2L?
    fb_hfpef = hfpef["fluid_balance_L"]
    report_lines.append(f"\n  HFpEF fluid_balance_L distribution:")
    desc_hfpef = fb_hfpef.describe(percentiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    for stat, val in desc_hfpef.items():
        report_lines.append(f"    {stat:<8} {val:>10.3f}")

    # Check how many HFpEF have negative balance (more output than input)
    n_neg = int((fb_hfpef < 0).sum())
    n_pos = int((fb_hfpef > 0).sum())
    n_zero = int((fb_hfpef == 0).sum())
    report_lines.append(f"\n  HFpEF balance direction:")
    report_lines.append(f"    Negative (output > input): {n_neg:,} ({n_neg/n_hfpef*100:.1f}%)")
    report_lines.append(f"    Zero (exact):              {n_zero:,} ({n_zero/n_hfpef*100:.1f}%)")
    report_lines.append(f"    Positive (input > output): {n_pos:,} ({n_pos/n_hfpef*100:.1f}%)")

    report_lines.append("")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 3: Congestion threshold sensitivity (HFpEF)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    report_lines.append("§3 CONGESTION THRESHOLD SENSITIVITY (HFpEF)")
    report_lines.append("─" * 60)

    sensitivity_rows = []
    for fb_t in FB_THRESHOLDS_L:
        fb_crit = hfpef["fluid_balance_L"] > fb_t
        n_fb = int(fb_crit.sum())

        for logic in ["AND", "OR"]:
            # AND: fluid balance AND resp distress
            # OR: fluid balance OR resp distress
            if logic == "AND":
                combined = fb_crit & resp_distress
            else:
                combined = fb_crit | resp_distress
            n_comb = int(combined.sum())

            sensitivity_rows.append({
                "fb_threshold_L": fb_t,
                "logic": logic,
                "n_fb_only": n_fb,
                "pct_fb_only": round(n_fb / n_hfpef * 100, 1),
                "n_combined": n_comb,
                "pct_combined": round(n_comb / n_hfpef * 100, 1),
            })

    sens_df = pd.DataFrame(sensitivity_rows)

    report_lines.append(f"\n  Resp distress = SpO2<{ORIG_SPO2_THRESH}% OR RR>{ORIG_RR_THRESH}")
    report_lines.append(f"  {'FB thresh':<12} {'Logic':<6} {'N(FB only)':<12} {'%(FB)':<8} {'N(combined)':<12} {'%(comb)':<8}")
    report_lines.append("  " + "─" * 56)
    for _, r in sens_df.iterrows():
        marker = " ← current" if r["fb_threshold_L"] == ORIG_FB_THRESH_L and r["logic"] == "AND" else ""
        report_lines.append(
            f"  {r['fb_threshold_L']:<12.1f} {r['logic']:<6} "
            f"{r['n_fb_only']:<12} {r['pct_fb_only']:<8.1f} "
            f"{r['n_combined']:<12} {r['pct_combined']:<8.1f}{marker}"
        )
    report_lines.append("")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 4: Fluid input/output component audit
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    report_lines.append("§4 FLUID INPUT/OUTPUT COMPONENT AUDIT (HFpEF)")
    report_lines.append("─" * 60)

    fi_hfpef = hfpef["fluid_input_ml"]
    fo_hfpef = hfpef["fluid_output_ml"]

    report_lines.append(f"\n  Fluid input (mL, 0–6h):")
    for stat, val in fi_hfpef.describe().items():
        report_lines.append(f"    {stat:<8} {val:>12.1f}")

    report_lines.append(f"\n  Fluid output (mL, 0–6h):")
    for stat, val in fo_hfpef.describe().items():
        report_lines.append(f"    {stat:<8} {val:>12.1f}")

    n_fi_zero_h = int((fi_hfpef == 0).sum())
    n_fo_zero_h = int((fo_hfpef == 0).sum())
    report_lines.append(f"\n  Zero-value flags (HFpEF):")
    report_lines.append(f"    Input = 0:  {n_fi_zero_h:,} ({n_fi_zero_h/n_hfpef*100:.1f}%)")
    report_lines.append(f"    Output = 0: {n_fo_zero_h:,} ({n_fo_zero_h/n_hfpef*100:.1f}%)")

    report_lines.append("")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 5: Cross-group comparison
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    report_lines.append("§5 FLUID BALANCE BY COMPARATOR GROUP")
    report_lines.append("─" * 60)
    report_lines.append(f"\n  {'Group':<20} {'N':>7} {'median':>8} {'IQR':>18} {'%>2L':>8} {'%cong':>8}")
    report_lines.append("  " + "─" * 68)

    for grp in ["hfpef", "hfref_hfmref", "hf_ef_unknown", "no_hf"]:
        sub = synd[synd["comparator_group"] == grp]
        if len(sub) == 0:
            continue
        fb_sub = sub["fluid_balance_L"]
        med = fb_sub.median()
        q25 = fb_sub.quantile(0.25)
        q75 = fb_sub.quantile(0.75)
        pct_above = (fb_sub > 2.0).mean() * 100
        n_cong = int((sub["dominant_syndrome"] == "congestion").sum())
        pct_cong = n_cong / len(sub) * 100
        report_lines.append(
            f"  {grp:<20} {len(sub):>7,} {med:>8.2f} [{q25:.2f}–{q75:.2f}] "
            f"{pct_above:>7.1f}% {pct_cong:>7.1f}%"
        )
    report_lines.append("")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 6: Recommendations
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    report_lines.append("§6 RECOMMENDATIONS")
    report_lines.append("─" * 60)
    curr_and = sens_df[
        (sens_df["fb_threshold_L"] == ORIG_FB_THRESH_L) & (sens_df["logic"] == "AND")
    ].iloc[0]
    alt_and = sens_df[
        (sens_df["fb_threshold_L"] == 1.0) & (sens_df["logic"] == "AND")
    ].iloc[0]
    gain_pts = float(alt_and["pct_combined"] - curr_and["pct_combined"])

    report_lines.append(f"""
  R1 — Physiologic bounds: Apply Winsorization or clipping to fluid_balance_L
        at [-10, +15]L to handle outliers before modeling.

  R2 — Congestion threshold: The current AND logic (FB>2L AND resp distress)
        is too strict for 0–6h pre-T0 window. Consider:
        (a) Lower FB threshold to 1.0L (captures ~{gain_pts:.1f} percentage points more);
        (b) Use OR logic instead of AND;
        (c) Add BNP/NT-proBNP or weight gain as additional congestion markers
            when available (Step 10e extraction);
        (d) Consider diuretic exposure as a proxy for clinician-recognized
            congestion (observational signal).

  R3 — Output completeness: If fluid_output_ml=0 is common, some patients
        may lack charted output events (documentation artifact, not anuria).
        Flagging these stays is needed before modeling.

  R4 — Sensitivity in DTR: Run congestion definition as a sensitivity
        parameter in Step 85j (informative missingness sensitivity).
""")

    # ── Save outputs ───────────────────────────────────────────────────
    # QA summary table
    qa_rows = []
    for grp in ["hfpef", "hfref_hfmref", "hf_ef_unknown", "no_hf"]:
        sub = synd[synd["comparator_group"] == grp]
        if len(sub) == 0:
            continue
        fb_sub = sub["fluid_balance_L"]
        qa_rows.append({
            "group": grp,
            "n": len(sub),
            "fb_median_L": round(fb_sub.median(), 3),
            "fb_q25_L": round(fb_sub.quantile(0.25), 3),
            "fb_q75_L": round(fb_sub.quantile(0.75), 3),
            "fb_min_L": round(fb_sub.min(), 3),
            "fb_max_L": round(fb_sub.max(), 3),
            "pct_above_2L": round((fb_sub > 2.0).mean() * 100, 2),
            "pct_below_neg10L": round((fb_sub < -10).mean() * 100, 2),
            "n_congestion": int((sub["dominant_syndrome"] == "congestion").sum()),
            "pct_congestion": round((sub["dominant_syndrome"] == "congestion").mean() * 100, 2),
            "n_output_zero": int((sub["fluid_output_ml"] == 0).sum()),
            "pct_output_zero": round((sub["fluid_output_ml"] == 0).mean() * 100, 2),
        })

    qa_df = pd.DataFrame(qa_rows)
    qa_path = tables_dir / "40d_fluid_balance_qa.csv"
    qa_df.to_csv(qa_path, index=False)
    print(f"  Saved: {qa_path}")

    sens_path = tables_dir / "40d_congestion_sensitivity.csv"
    sens_df.to_csv(sens_path, index=False)
    print(f"  Saved: {sens_path}")

    report_path = reports_dir / "40d_fluid_harmonization_qa.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Saved: {report_path}")

    print("\n[40d] Done.")


if __name__ == "__main__":
    main()
