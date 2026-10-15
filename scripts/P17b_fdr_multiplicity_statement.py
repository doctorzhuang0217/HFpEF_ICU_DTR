#!/usr/bin/env python3
"""
P17b_fdr_multiplicity_statement — p-value family registry + FDR multiplicity statement.

Protocol: v3.1 patch (enhancement analysis G; see
the supplementary-analysis specification, section G).
Compute: NONE (local / login-node; stdlib-only — uses only `csv` and `math`).

Purpose
-------
Enumerate every explicit, reported p-value across the project's committed result
tables, organise them into pre-specified statistical *families*, and apply a
Benjamini-Hochberg (BH) false-discovery-rate (FDR) correction WITHIN the single
exploratory family (subgroups / secondary outcomes / sensitivity hypothesis
tests). This produces a transparent multiplicity registry plus a ready-to-paste
FDR methods paragraph for the manuscript / supplement.

Multiplicity handling rules (pre-specified)
------------------------------------------
1. Primary ATE exemption. The pre-specified primary estimand (the dynamic
   conservative-fluid DTR average treatment effect, reported as a bootstrap CI
   in 75a/80c) is a single confirmatory estimand reported as an interval, not a
   hypothesis test. It is recorded for completeness with included_in_fdr=False
   (exempt) and is NOT entered into any BH family.

2. Exploratory family BH correction. All explicit p-values arising from
   exploratory hypothesis tests — subtype/HFI interaction tests, the
   pre-T0 HFI subtype contrasts, and any secondary-outcome hypothesis tests —
   are pooled into one exploratory family and BH-adjusted together. The
   BH q-value (adjusted p-value, monotone-enforced) is reported per test.

3. Table 1 / descriptive baseline tests excluded. Omnibus baseline-balance
   p-values (Kruskal-Wallis / chi-square in Table 1) are descriptive
   cohort-characterisation statistics, not study hypotheses. They are recorded
   with included_in_fdr=False and excluded from BH (standard practice: Table 1
   p-values are not corrected and ideally not emphasised).

4. CI-based inferences are NOT BH tests. Bootstrap-CI-only results
   (80g interaction strata, 85e negative controls, P1 concordant-vs-discordant
   secondary-outcome differences, P16a/P16b subgroup ATEs) carry NO explicit
   p-value. They are counted as CI-based inferences: the script reports how many
   such CIs exclude the null (0 for ATE/difference estimands; for negative
   controls a CI excluding null is a *flag*, not a positive finding). These are
   recorded with p_value empty and included_in_fdr=False.

Robustness
----------
Every source table is read defensively. A missing file is skipped with a logged
WARNING (the registry simply omits that family); a present file with an
unexpected schema contributes only the p-value columns it actually has.

Inputs (all under tables/; any may be absent — skipped with a warning)
---------------------------------------------------------------------
- tables/P10_interaction_test_summary.csv      (explicit p: interaction tests)
- tables/P9b_pretT0_hfi_welch_vs_hfpef.csv      (explicit p: HFI subtype contrasts)
- tables/P1_secondary_outcomes_summary.csv      (CI-only: secondary-outcome diffs)
- tables/manuscript_table1_baseline_stats.csv   (explicit p: descriptive, exempt)
- tables/80g_bootstrap_interaction_ci.csv       (CI-only: interaction strata)
- tables/85e_negative_control.csv               (CI-only: negative controls)
- tables/90e_consistency_checklist.csv          (pass/fail: consistency, not tests)
- tables/P16a_subgroup_sensitivity_iptw.csv     (CI-only: subgroup ATEs) [optional]
- tables/P16b_subgroup_dm_rollout_ow1.csv       (CI-only: subgroup ATEs) [optional]

Outputs
-------
- tables/P17b_pvalue_family_registry.csv
    columns: family, test_label, source_table, p_value,
             is_confirmatory, included_in_fdr, bh_q_value, note
- (stdout) FDR methods paragraph + registry summary.

Usage
-----
    env/run_py.sh scripts/P17b_fdr_multiplicity_statement.py
    # optional:
    env/run_py.sh scripts/P17b_fdr_multiplicity_statement.py --fdr-alpha 0.05
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
from pathlib import Path

# ── Paths (PROJECT_ROOT env override, mirrors project convention) ─────
ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
TABLES_DIR = ROOT / "tables"

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("P17b")

# ── Family labels (pre-specified) ─────────────────────────────────────
FAM_PRIMARY = "primary_ate"            # single pre-specified estimand (exempt)
FAM_EXPLORATORY = "exploratory"        # subgroups / secondary / sensitivity (BH)
FAM_DESCRIPTIVE = "descriptive_table1"  # Table 1 baseline tests (exempt)
FAM_CI = "ci_based_inference"          # CI-only estimands (no p; counted, not BH)

OUTPUT_COLS = [
    "family",
    "test_label",
    "source_table",
    "p_value",
    "is_confirmatory",
    "included_in_fdr",
    "bh_q_value",
    "note",
]


# ── Helpers ───────────────────────────────────────────────────────────
def read_csv_rows(path: Path) -> list[dict] | None:
    """Read a CSV into a list of dict rows. Return None if absent/unreadable."""
    if not path.exists():
        log.warning("SKIP (absent): %s", path.name)
        return None
    try:
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
    except Exception as exc:  # noqa: BLE001 — be robust to any read error
        log.warning("SKIP (read error %s): %s", exc, path.name)
        return None
    log.info("Loaded %s (%d rows)", path.name, len(rows))
    return rows


def parse_float(value) -> float | None:
    """Parse a float p-value cell; return None for empty / NA / unparseable."""
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.upper() in {"NA", "NAN", "NONE", "—", "-"}:
        return None
    try:
        f = float(s)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


def fmt_p(p: float | None) -> str:
    """Format a p-value for CSV output (preserve small exponents)."""
    if p is None:
        return ""
    if p == 0.0:
        return "0"
    if p < 1e-4 or p >= 1e4:
        return f"{p:.3e}"
    return f"{p:.6g}"


def ci_excludes_null(lo, hi, null_value: float = 0.0) -> bool | None:
    """Return True if [lo, hi] excludes `null_value`; None if unparseable."""
    flo = parse_float(lo)
    fhi = parse_float(hi)
    if flo is None or fhi is None:
        return None
    lo_v, hi_v = (flo, fhi) if flo <= fhi else (fhi, flo)
    return not (lo_v <= null_value <= hi_v)


def benjamini_hochberg(pvals: list[float], alpha: float) -> tuple[list[float], int]:
    """Compute BH-adjusted q-values (monotone-enforced) for a list of p-values.

    Returns (q_values_in_original_order, n_discoveries_at_alpha).
    Implements the standard step-up BH adjusted-p (a.k.a. q-value) transform:
        q_(i) = min over k>=i of [ p_(k) * m / k ], clamped to <= 1.
    """
    m = len(pvals)
    if m == 0:
        return [], 0
    # rank ascending, keep original indices
    order = sorted(range(m), key=lambda i: pvals[i])
    raw = [0.0] * m
    for rank, idx in enumerate(order, start=1):
        raw[idx] = min(pvals[idx] * m / rank, 1.0)
    # enforce monotonicity from largest p downwards
    q = [0.0] * m
    running_min = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        running_min = min(running_min, raw[idx])
        q[idx] = running_min
    # number of discoveries: largest rank k with p_(k) <= k/m * alpha
    n_disc = 0
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= (rank / m) * alpha:
            n_disc = rank
    return q, n_disc


# ── Per-source extractors ─────────────────────────────────────────────
def collect_explicit_pvalues() -> list[dict]:
    """Build registry rows that carry an EXPLICIT p-value.

    Each row is a dict with the OUTPUT_COLS keys except bh_q_value (filled
    later for the exploratory family). is_confirmatory / included_in_fdr /
    family set per the pre-specified rules above.
    """
    rows: list[dict] = []

    # ── Exploratory: P10 interaction test summary ─────────────────────
    p10 = read_csv_rows(TABLES_DIR / "P10_interaction_test_summary.csv")
    if p10:
        # one summary row; pull each explicit p-value column it carries.
        p10_pmap = [
            ("p_interaction", "HFI x HFpEF binned-ITE interaction (OLS)"),
            ("p_hfi", "HFI main effect on binned ITE (OLS)"),
            ("p_hfpef", "HFpEF main effect on binned ITE (OLS)"),
            ("raw_welch_p", "Raw ITE HFpEF vs no_HF (Welch t)"),
            ("raw_mann_whitney_p", "Raw ITE HFpEF vs no_HF (Mann-Whitney U)"),
        ]
        for r in p10:
            for col, label in p10_pmap:
                p = parse_float(r.get(col))
                if p is None:
                    continue
                rows.append({
                    "family": FAM_EXPLORATORY,
                    "test_label": label,
                    "source_table": "P10_interaction_test_summary.csv",
                    "p_value": p,
                    "is_confirmatory": False,
                    "included_in_fdr": True,
                    "note": "exploratory CATE/interaction hypothesis test",
                })

    # ── Exploratory: P9b pre-T0 HFI Welch contrasts ───────────────────
    p9b = read_csv_rows(TABLES_DIR / "P9b_pretT0_hfi_welch_vs_hfpef.csv")
    if p9b:
        for r in p9b:
            p = parse_float(r.get("welch_p"))
            if p is None:
                continue
            comp = (r.get("comparison") or "").strip() or "unspecified_contrast"
            rows.append({
                "family": FAM_EXPLORATORY,
                "test_label": f"pre-T0 HFI {comp} (Welch t)",
                "source_table": "P9b_pretT0_hfi_welch_vs_hfpef.csv",
                "p_value": p,
                "is_confirmatory": False,
                "included_in_fdr": True,
                "note": "exploratory pre-T0 fragility-index subtype contrast",
            })

    # NOTE: 80g2/80g3 subtype-difference contrasts are NOT added here. Their
    # `p_diff_ge0` is a DIRECTION-AWARE bootstrap tail probability P(diff>=0)
    # (≈1 for a strong positive difference), not a p-value under a unified
    # alternative — feeding it to Benjamini-Hochberg would be invalid. They are
    # instead recorded as CI-based inferences in collect_ci_based().

    # ── Descriptive (exempt): Table 1 baseline omnibus tests ──────────
    t1 = read_csv_rows(TABLES_DIR / "manuscript_table1_baseline_stats.csv")
    if t1:
        for r in t1:
            p = parse_float(r.get("p_overall"))
            if p is None:
                continue
            var = (r.get("variable") or r.get("label") or "variable").strip()
            test = (r.get("test_used") or "omnibus").strip()
            rows.append({
                "family": FAM_DESCRIPTIVE,
                "test_label": f"Table 1 balance: {var} ({test})",
                "source_table": "manuscript_table1_baseline_stats.csv",
                "p_value": p,
                "is_confirmatory": False,
                "included_in_fdr": False,
                "note": "descriptive baseline-characterisation omnibus; not corrected",
            })

    return rows


def collect_ci_based() -> tuple[list[dict], dict]:
    """Build registry rows for CI-only estimands (no explicit p-value).

    Returns (rows, counts) where counts summarises how many CIs exclude the
    null per source. These rows are NOT BH tests: included_in_fdr=False, p empty.
    """
    rows: list[dict] = []
    counts: dict[str, dict[str, int]] = {}

    def _add_ci_source(fname, lo_col, hi_col, label_cols, note,
                       confirmatory=False, skip_fn=None):
        data = read_csv_rows(TABLES_DIR / fname)
        if data is None:
            return
        n_excl = 0
        n_eval = 0
        for r in data:
            if skip_fn is not None and skip_fn(r):
                continue
            verdict = ci_excludes_null(r.get(lo_col), r.get(hi_col))
            label = " / ".join(
                str(r.get(c, "")).strip() for c in label_cols if r.get(c)
            ).strip() or "estimand"
            if verdict is not None:
                n_eval += 1
                if verdict:
                    n_excl += 1
            rows.append({
                "family": FAM_CI,
                "test_label": f"{label}",
                "source_table": fname,
                "p_value": None,  # CI-only: no explicit p
                "is_confirmatory": confirmatory,
                "included_in_fdr": False,
                "note": (
                    f"CI-based ({'excludes' if verdict else 'includes' if verdict is not None else 'unparseable'} "
                    f"null); {note}"
                ),
            })
        counts[fname] = {"n_eval": n_eval, "n_excl_null": n_excl}

    # 80g interaction strata (HFpEF/no_HF x shock/non-shock ATE CIs)
    _add_ci_source(
        "80g_bootstrap_interaction_ci.csv",
        "ate_ci_lo", "ate_ci_hi",
        ["hf_subtype", "shock_stratum"],
        "subtype x shock interaction stratum ATE",
    )
    # 80g2/80g3 subtype-difference contrasts as CI-based inferences (NOT BH):
    # p_diff_ge0 is a directional bootstrap tail probability, so these are counted
    # by whether the difference CI excludes null, and excluded from the FDR family.
    _add_ci_source(
        "80g2_subtype_difference_ci.csv",
        "diff_ci_lo", "diff_ci_hi",
        ["contrast", "shock_stratum"],
        "subtype-difference ATE (CI-based; p_diff_ge0 is a directional bootstrap tail prob)",
    )
    _add_ci_source(
        "80g3_ef_lability_contamination_sensitivity.csv",
        "diff_ci_lo", "diff_ci_hi",
        ["analysis", "shock_stratum"],
        "EF-lability contamination difference (CI-based; p_diff_ge0 directional)",
    )
    # 85e negative controls (CI excluding null = a FLAG, not a positive finding).
    # IMPORTANT: skip the `baseline` row (test=baseline / expected_ate=nonzero) — it
    # is the REAL primary effect (the -1.37pp estimate, "real outcome, real treatment"),
    # NOT a negative control; its CI excluding null is the desired result, not a
    # residual-bias flag. Only the genuine NCO*/NCE* controls belong in this family.
    _add_ci_source(
        "85e_negative_control.csv",
        "ate_ci_lo", "ate_ci_hi",
        ["test", "description"],
        "negative-control ATE (CI excluding null = residual-bias FLAG)",
        skip_fn=lambda r: (
            str(r.get("test", "")).strip().lower() == "baseline"
            or str(r.get("expected_ate", "")).strip().lower() == "nonzero"
        ),
    )
    # P1 secondary outcomes: concordant-vs-discordant difference CIs
    _add_ci_source(
        "P1_secondary_outcomes_summary.csv",
        "diff_ci_lo", "diff_ci_hi",
        ["outcome", "subgroup"],
        "secondary-outcome concordant-vs-discordant difference",
    )
    # P16a IPTW subgroup ATEs (optional)
    _add_ci_source(
        "P16a_subgroup_sensitivity_iptw.csv",
        "bootstrap_ci_lo", "bootstrap_ci_hi",
        ["label"],
        "acute-driver subgroup ATE (IPTW)",
    )
    # P16b G-Transformer DM subgroup ATEs (optional)
    _add_ci_source(
        "P16b_subgroup_dm_rollout_ow1.csv",
        "ci_lo", "ci_hi",
        ["label"],
        "acute-driver subgroup ATE (DM rollout)",
    )

    return rows, counts


def collect_primary_exempt() -> list[dict]:
    """Record the single pre-specified primary ATE estimand as FDR-exempt.

    The primary estimand is reported as a bootstrap CI (75a/80c), not as a
    hypothesis test, so it has no explicit p-value and is exempt from BH.
    """
    return [{
        "family": FAM_PRIMARY,
        "test_label": "Primary dynamic conservative-fluid DTR ATE (bootstrap CI)",
        "source_table": "75a_bootstrap_ci.csv / 80c_gtransformer_rollout_ow1_bootstrap_ci.csv",
        "p_value": None,
        "is_confirmatory": True,
        "included_in_fdr": False,
        "note": "single pre-specified confirmatory estimand; interval-based, EXEMPT from BH",
    }]


def collect_consistency_note() -> dict:
    """Summarise 90e pre-registered consistency checklist (pass/fail, not tests)."""
    data = read_csv_rows(TABLES_DIR / "90e_consistency_checklist.csv")
    if data is None:
        return {"present": False}
    n_pass = sum(1 for r in data if str(r.get("pass", "")).strip().lower() == "true")
    n_total = len(data)
    return {"present": True, "n_pass": n_pass, "n_total": n_total}


# ── FDR methods paragraph ─────────────────────────────────────────────
def build_methods_paragraph(
    n_expl: int, n_disc: int, alpha: float,
    ci_counts: dict, consistency: dict,
) -> str:
    """Compose the ready-to-paste FDR methods paragraph."""
    ci_total = sum(c["n_eval"] for c in ci_counts.values())
    ci_excl = sum(c["n_excl_null"] for c in ci_counts.values())
    cons_str = ""
    if consistency.get("present"):
        cons_str = (
            f" The {consistency['n_total']} pre-registered cross-dataset "
            f"consistency criteria ({consistency['n_pass']} met) are evaluated as "
            f"binary pass/fail rules and are likewise not hypothesis tests subject "
            f"to FDR adjustment."
        )
    return (
        "Multiplicity and false-discovery-rate control. The primary endpoint was a "
        "single pre-specified estimand — the average treatment effect of the dynamic "
        "conservative-fluid treatment regime — reported as a bootstrap confidence "
        "interval rather than a hypothesis test; it is therefore exempt from "
        "multiplicity correction and stands as the sole confirmatory analysis. All "
        f"explicit p-values from exploratory analyses (subgroup, interaction/CATE, "
        f"and pre-T0 fragility-index contrasts; n = {n_expl} tests) were pooled into "
        "a single exploratory family and adjusted for multiple comparisons with the "
        "Benjamini-Hochberg procedure controlling the false discovery rate at "
        f"{alpha:.2f}. Under this correction, {n_disc} of {n_expl} exploratory tests "
        "remained discoveries (BH q < "
        f"{alpha:.2f}). Baseline-characteristic comparisons (Table 1) are reported as "
        "descriptive omnibus statistics and were not corrected, consistent with "
        "recommendations against emphasising Table 1 p-values. Estimands reported "
        "only as bootstrap confidence intervals — interaction-stratum and "
        "subgroup average treatment effects, negative-control effects, and "
        "secondary-outcome concordant-versus-discordant differences — are interval-"
        f"based inferences rather than Benjamini-Hochberg tests; of {ci_total} such "
        f"confidence intervals evaluated, {ci_excl} excluded the null (for the "
        "negative-control family, a confidence interval excluding the null is "
        "interpreted as a residual-bias flag rather than a positive finding)."
        f"{cons_str}"
    )


# ── Main ──────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="P17b — p-value family registry + BH FDR multiplicity statement"
    )
    p.add_argument(
        "--fdr-alpha", type=float, default=0.05,
        help="Benjamini-Hochberg target FDR for the exploratory family (default 0.05)",
    )
    p.add_argument(
        "--out", type=str,
        default=str(TABLES_DIR / "P17b_pvalue_family_registry.csv"),
        help="Output registry CSV path",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 72)
    print("P17b — p-value family registry + FDR multiplicity statement")
    print("=" * 72)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Collect all registry rows ──────────────────────────────────
    log.info("[1/4] Collecting p-value sources...")
    rows: list[dict] = []
    rows.extend(collect_primary_exempt())
    rows.extend(collect_explicit_pvalues())
    ci_rows, ci_counts = collect_ci_based()
    rows.extend(ci_rows)
    consistency = collect_consistency_note()

    # ── 2. Apply BH within the exploratory family ─────────────────────
    log.info("[2/4] Applying Benjamini-Hochberg within exploratory family...")
    expl_rows = [r for r in rows if r["family"] == FAM_EXPLORATORY and r["included_in_fdr"]]
    expl_p = [r["p_value"] for r in expl_rows]
    q_vals, n_disc = benjamini_hochberg(expl_p, args.fdr_alpha)
    for r, q in zip(expl_rows, q_vals):
        r["bh_q_value"] = q
    # All non-exploratory / non-included rows carry no BH q-value.
    for r in rows:
        if "bh_q_value" not in r:
            r["bh_q_value"] = None
    log.info(
        "  exploratory family: %d tests, %d BH discoveries at alpha=%.2f",
        len(expl_rows), n_disc, args.fdr_alpha,
    )

    # ── 3. Write registry CSV ─────────────────────────────────────────
    log.info("[3/4] Writing registry CSV...")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLS)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "family": r["family"],
                "test_label": r["test_label"],
                "source_table": r["source_table"],
                "p_value": fmt_p(r.get("p_value")),
                "is_confirmatory": bool(r["is_confirmatory"]),
                "included_in_fdr": bool(r["included_in_fdr"]),
                "bh_q_value": fmt_p(r.get("bh_q_value")),
                "note": r.get("note", ""),
            })
    log.info("  Saved: %s (%d rows)", out_path, len(rows))

    # ── 4. Print methods paragraph + summary ──────────────────────────
    log.info("[4/4] Composing FDR methods paragraph...")
    n_expl = len(expl_rows)
    n_desc = sum(1 for r in rows if r["family"] == FAM_DESCRIPTIVE)
    n_ci = sum(1 for r in rows if r["family"] == FAM_CI)
    n_primary = sum(1 for r in rows if r["family"] == FAM_PRIMARY)

    paragraph = build_methods_paragraph(
        n_expl, n_disc, args.fdr_alpha, ci_counts, consistency
    )

    print()
    print("─" * 72)
    print("REGISTRY SUMMARY")
    print("─" * 72)
    print(f"  primary (exempt)        : {n_primary}")
    print(f"  exploratory (BH-tested) : {n_expl}  -> {n_disc} discoveries (q<{args.fdr_alpha})")
    print(f"  descriptive Table 1     : {n_desc}  (exempt, uncorrected)")
    print(f"  CI-based inferences     : {n_ci}")
    for fname, c in ci_counts.items():
        print(f"      {fname}: {c['n_excl_null']}/{c['n_eval']} CIs exclude null")
    if consistency.get("present"):
        print(f"  90e consistency checks  : {consistency['n_pass']}/{consistency['n_total']} pass (pass/fail, not tests)")
    print()
    print("─" * 72)
    print("FDR METHODS PARAGRAPH (copy-paste)")
    print("─" * 72)
    print(paragraph)
    print()
    print("=" * 72)
    print(f"P17b COMPLETE. Registry: {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
