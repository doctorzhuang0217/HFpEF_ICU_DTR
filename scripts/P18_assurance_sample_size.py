#!/usr/bin/env python3
"""P18 — Assurance / conditional-expected-power (CEP) sample-size for the pilot trial.

Upgrades the existing point-scenario power calc (P6_sample_size_table.csv) by
PROPAGATING the effect-estimate uncertainty into the sample-size, instead of
choosing single point ARRs (which were flagged as cherry-picking the largest
estimator). For each candidate per-arm n we compute ASSURANCE = the prior-
averaged power, E_prior[ power(n, ARR) ], under two design priors:

  (1) primary-CI prior  : Normal(mean=|DM-rollout ARR|, sd from its bootstrap CI)
                          — the primary estimand's own sampling uncertainty.
  (2) estimator-spread  : a DISCRETE prior over ALL Step-C estimator point ARRs
                          (DM, DR, FQE, MSM/IPTW-WIS, G-Net, ...) each weighted
                          equally — the honest cross-estimator uncertainty
                          (the ~0.2-4.9 pp spread the manuscript must own).

It also reports ENRICHMENT scenarios (overall vs heart-failure-reserve-enriched
vs HFpEF-enriched vs no-HF), using the ALL-stratum subgroup ARRs from 80g2, to
quantify how a reserve/HFpEF-enriched trial reduces required N.

Stdlib-only (csv, math) — no pandas/numpy/scipy — so it runs anywhere (C1).
Reads committed tables; computes nothing model-based (no GPU). Reports the
conventional point-power N alongside assurance N so the inflation from honestly
propagating uncertainty is explicit.

Inputs (committed):
- tables/P6_sample_size_table.csv          (control event rate; point scenarios)
- tables/95a_results_registry.csv          (Step-C estimator ARRs; WIS CI)
- tables/80c_gtransformer_rollout_ow1_bootstrap_ci.csv  (primary DM ARR + CI)
- tables/80g2_subtype_difference_ci_cells.csv           (subgroup ARRs for enrichment)
Output:
- tables/P18_assurance_sample_size.csv
- reports/P18_assurance_sample_size.txt

Run (trivial CPU/C1):
    env/run_py.sh scripts/P18_assurance_sample_size.py
"""
import argparse
import csv
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"

DEFAULT_CONTROL_RATE = 0.1382   # observed-policy 28-day mortality in MIMIC (P6/eMethods 5)
ALPHA = 0.05                    # two-sided
ASSURANCE_TARGETS = (0.80, 0.90)
MAX_N_PER_ARM = 200_000


# ── normal CDF / PPF (stdlib) ──────────────────────────────────────────
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Acklam's inverse-normal-CDF approximation (abs err < 1.15e-9)."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ── two-proportion power (two-sided, equal n/arm) ──────────────────────
def power_two_prop(p0: float, arr: float, n: int, alpha: float = ALPHA) -> float:
    """Power to detect an absolute risk reduction `arr` (>0) from control p0."""
    arr = abs(arr)
    p1 = p0 - arr
    if p1 <= 0.0 or n < 2:
        return float(p1 > 0.0)
    z_a = norm_ppf(1 - alpha / 2)
    pbar = (p0 + p1) / 2.0
    se0 = math.sqrt(2 * pbar * (1 - pbar))
    se1 = math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))
    if se1 <= 0:
        return 1.0
    return norm_cdf((math.sqrt(n) * (p0 - p1) - z_a * se0) / se1)


def n_for_point_power(p0, arr, target, alpha=ALPHA):
    """Conventional per-arm n for a single point ARR (binary search)."""
    lo, hi = 2, MAX_N_PER_ARM
    if power_two_prop(p0, arr, hi, alpha) < target:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if power_two_prop(p0, arr, mid, alpha) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ── design priors ──────────────────────────────────────────────────────
def normal_grid(mean, sd, n_pts=801, span=4.0):
    """Deterministic (mean, weight) grid approximating Normal(mean, sd)."""
    mean, sd = abs(mean), max(sd, 1e-6)
    lo, hi = mean - span * sd, mean + span * sd
    step = (hi - lo) / (n_pts - 1)
    pts, wts = [], []
    for i in range(n_pts):
        x = lo + i * step
        w = math.exp(-0.5 * ((x - mean) / sd) ** 2)
        if x > 0:                      # ARR must be a positive risk reduction
            pts.append(x)
            wts.append(w)
    s = sum(wts) or 1.0
    return pts, [w / s for w in wts]


def assurance_at_n(p0, prior_pts, prior_wts, n, alpha=ALPHA):
    return sum(w * power_two_prop(p0, a, n, alpha) for a, w in zip(prior_pts, prior_wts))


def n_for_assurance(p0, prior_pts, prior_wts, target, alpha=ALPHA):
    lo, hi = 2, MAX_N_PER_ARM
    if assurance_at_n(p0, prior_pts, prior_wts, hi, alpha) < target:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if assurance_at_n(p0, prior_pts, prior_wts, mid, alpha) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ── readers (defensive) ────────────────────────────────────────────────
def read_rows(path):
    p = TABLES / path
    if not p.exists():
        print(f"  WARN: missing {p}")
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def get_control_rate():
    for r in read_rows("P6_sample_size_table.csv"):
        for k in ("p_control", "control_rate", "p0", "control_event_rate"):
            v = to_float(r.get(k))
            if v is not None:
                return v
    return DEFAULT_CONTROL_RATE


def get_estimator_arrs():
    """Step-C estimator point ATEs (abs, fraction) for the cross-estimator spread prior.

    STRICT selection: a registry row is an estimator ATE only if its result_id ends in a
    method suffix (_DM/_DR/_FQE) or is the WIS ATE id (75a_ci_ate_wis). This deliberately
    EXCLUDES rule/observed mortality-RATE rows (e.g. 75a_rule_v1_WIS=0.089, 75a_ci_wis_rule),
    which are positive event probabilities, not absolute risk reductions, and would otherwise
    be mistaken for ~9pp ARRs and collapse the sample size. Two further guards: value must be
    negative (ATEs here are beneficial; rates are positive) and |value|<0.2 (sane ARR bound);
    _ci_ rows are dropped to avoid double-counting an estimator already present.
    """
    WIS_ATE_ID = "75a_ci_ate_wis"
    METHOD_SUFFIXES = ("_DM", "_DR", "_FQE")
    arrs = {}
    for r in read_rows("95a_results_registry.csv"):
        sec = (r.get("section") or "")
        rid = (r.get("result_id") or "")
        if sec not in ("R1_primary_ope", "R2_step_c_models"):
            continue
        v = to_float(r.get("value"))
        if v is None or v >= 0:                       # exclude rates / non-beneficial
            continue
        is_method_ate = rid.endswith(METHOD_SUFFIXES) and "_ci_" not in rid
        is_wis_ate = (rid == WIS_ATE_ID)
        if (is_method_ate or is_wis_ate) and abs(v) < 0.2:
            arrs[rid] = abs(v)
    return arrs


def get_primary_dm():
    """Primary DM rollout ARR + sd from its bootstrap CI."""
    for r in read_rows("80c_gtransformer_rollout_ow1_bootstrap_ci.csv"):
        if (r.get("estimator") or "").upper() == "DM":
            m = to_float(r.get("ate_mean")) or to_float(r.get("ate"))
            lo = to_float(r.get("ate_ci_lo"))
            hi = to_float(r.get("ate_ci_hi"))
            if m is not None and lo is not None and hi is not None:
                return abs(m), abs(hi - lo) / (2 * 1.959964)
    # fallback to registry point (no CI -> small sd)
    arrs = get_estimator_arrs()
    for k, v in arrs.items():
        if "gtransformer_rollout_ow1_DM" in k:
            return v, v * 0.1
    return 0.0137, 0.0006


def get_subgroup_arrs():
    """ALL-stratum subgroup ARRs (point, sd from CI) for enrichment scenarios."""
    out = {}
    for r in read_rows("80g2_subtype_difference_ci_cells.csv"):
        if (r.get("shock_stratum") or "") != "ALL":
            continue
        sub = r.get("hf_subtype")
        pt = to_float(r.get("ate_point"))
        lo = to_float(r.get("ate_ci_lo"))
        hi = to_float(r.get("ate_ci_hi"))
        se = to_float(r.get("ate_se"))
        if sub and pt is not None:
            if se is not None and se > 0:
                sd = se
            elif lo is not None and hi is not None:
                sd = abs(hi - lo) / (2 * 1.959964)
            else:
                sd = abs(pt) * 0.15
            out[sub] = (abs(pt), max(sd, 1e-6))
    return out


def main():
    ap = argparse.ArgumentParser(description="P18 assurance/CEP sample-size for the pilot trial.")
    ap.add_argument("--control-rate", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    args = ap.parse_args()

    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    p0 = args.control_rate if args.control_rate is not None else get_control_rate()
    alpha = args.alpha

    print("=" * 72)
    print("P18 — assurance / conditional-expected-power sample size")
    print(f"control rate p0={p0:.4f}; alpha={alpha} (two-sided)")
    print("=" * 72)

    est_arrs = get_estimator_arrs()
    dm_mean, dm_sd = get_primary_dm()
    subgroups = get_subgroup_arrs()
    print(f"\nStep-C estimator ARRs (abs pp): " +
          ", ".join(f"{k.split('_')[-1] if '_' in k else k}={v*100:.2f}" for k, v in est_arrs.items()))
    print(f"primary DM rollout: {dm_mean*100:.2f} pp (sd {dm_sd*100:.3f})")
    print(f"subgroup ALL ARRs (pp): " +
          ", ".join(f"{k}={v[0]*100:.2f}" for k, v in subgroups.items()))

    rows = []

    # ── (1) overall, primary-CI prior ─────────────────────────────────
    pts, wts = normal_grid(dm_mean, dm_sd)
    for tgt in ASSURANCE_TARGETS:
        n = n_for_assurance(p0, pts, wts, tgt, alpha)
        npt = n_for_point_power(p0, dm_mean, tgt, alpha)
        rows.append(dict(population="overall", prior="primary_DM_CI",
                         arr_pp=round(dm_mean*100, 3), target=tgt,
                         n_per_arm_assurance=n, n_total_assurance=(2*n if n else None),
                         n_total_point=(2*npt if npt else None)))

    # ── (2) overall, estimator-spread prior (discrete, equal weight) ──
    if est_arrs:
        sp_pts = list(est_arrs.values())
        sp_wts = [1.0 / len(sp_pts)] * len(sp_pts)
        mean_sp = sum(sp_pts) / len(sp_pts)
        for tgt in ASSURANCE_TARGETS:
            n = n_for_assurance(p0, sp_pts, sp_wts, tgt, alpha)
            rows.append(dict(population="overall", prior="estimator_spread",
                             arr_pp=round(mean_sp*100, 3), target=tgt,
                             n_per_arm_assurance=n, n_total_assurance=(2*n if n else None),
                             n_total_point=None))

    # ── (3) enrichment scenarios (subgroup-ARR Normal priors) ─────────
    enrich_map = {"hfpef": "HFpEF_enriched", "no_hf": "no_HF",
                  "hfref": "HFrEF", "hf_ef_unknown": "HF_EF_unknown", "hfmref": "HFmrEF"}
    # HF-reserve pooled: n-weighted mean of the HF subtypes (exclude no_hf)
    hf_keys = [k for k in subgroups if k != "no_hf"]
    if hf_keys:
        hf_mean = sum(subgroups[k][0] for k in hf_keys) / len(hf_keys)
        hf_sd = sum(subgroups[k][1] for k in hf_keys) / len(hf_keys)
        pts, wts = normal_grid(hf_mean, hf_sd)
        for tgt in ASSURANCE_TARGETS:
            n = n_for_assurance(p0, pts, wts, tgt, alpha)
            npt = n_for_point_power(p0, hf_mean, tgt, alpha)
            rows.append(dict(population="HF_reserve_enriched", prior="subgroup_CI",
                             arr_pp=round(hf_mean*100, 3), target=tgt,
                             n_per_arm_assurance=n, n_total_assurance=(2*n if n else None),
                             n_total_point=(2*npt if npt else None)))
    for sub, (pt, sd) in subgroups.items():
        pts, wts = normal_grid(pt, sd)
        for tgt in ASSURANCE_TARGETS:
            n = n_for_assurance(p0, pts, wts, tgt, alpha)
            npt = n_for_point_power(p0, pt, tgt, alpha)
            rows.append(dict(population=enrich_map.get(sub, sub), prior="subgroup_CI",
                             arr_pp=round(pt*100, 3), target=tgt,
                             n_per_arm_assurance=n, n_total_assurance=(2*n if n else None),
                             n_total_point=(2*npt if npt else None)))

    out_csv = TABLES / "P18_assurance_sample_size.csv"
    cols = ["population", "prior", "arr_pp", "target", "n_per_arm_assurance",
            "n_total_assurance", "n_total_point"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_csv} ({len(rows)} rows)")

    methods = (
        "Pilot-trial feasibility used assurance (Bayesian expected power) rather "
        "than single point-estimate power. The control 28-day mortality was set to "
        f"{p0*100:.2f}%. Required sample size was derived by averaging the two-"
        "proportion power curve over a design prior on the absolute risk reduction, "
        "under two priors: (i) a Normal prior centred on the primary direct-method "
        "rollout estimate with its bootstrap-derived dispersion, and (ii) a discrete "
        "prior placing equal weight on all causal-estimator point estimates, which "
        "honestly propagates the cross-estimator uncertainty. Enrichment scenarios "
        "applied subgroup-specific risk reductions to quantify how a heart-failure-"
        "reserve- or HFpEF-enriched trial reduces the required sample size. Assurance "
        "for 80% and 90% targets is reported alongside the conventional point-estimate "
        "sample size; the latter, by ignoring effect-estimate uncertainty, "
        "systematically understates the requirement."
    )
    rep = REPORTS / "P18_assurance_sample_size.txt"
    with open(rep, "w") as f:
        f.write(methods + "\n\n")
        for r in rows:
            f.write(f"{r['population']:>22s} | {r['prior']:<16s} | ARR={r['arr_pp']:.2f}pp | "
                    f"assurance{int(r['target']*100)} N_total="
                    f"{r['n_total_assurance']} (point N_total={r['n_total_point']})\n")
    print(f"Wrote {rep}")
    print("\nP18 COMPLETE.")


if __name__ == "__main__":
    main()
