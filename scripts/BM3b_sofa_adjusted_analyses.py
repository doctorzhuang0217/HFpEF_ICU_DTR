#!/usr/bin/env python3
"""
BM3b_sofa_adjusted_analyses — Baseline-SOFA additions to Table 2, the
fragility-index Cox model and the G-Transformer contrast (BMC Medicine
revision, BM3b).

Purpose
-------
Uses the stay-level pre-T0 SOFA from BM3a to answer "is the HFpEF signal just
baseline severity?":

  (i)   Table 2 addition: SOFA median (IQR), mean (SD) and share >= 6 by
        heart-failure stratum (HFpEF; HF with EF < 50 % = hfref + hfmref;
        hf_ef_unknown; no_hf; all), the five non-cardiovascular components,
        component means, HFpEF-vs-comparator SMD and Mann-Whitney p, and the
        Kruskal-Wallis p across the four groups.
  (ii)  Fragility-index Cox model in HFpEF: E04 model B3 rebuilt exactly as in
        scripts/ejhf_figures_v1/E04_km_cox.py (HFI_preT0 per unit + ADJ_FULL:
        age/10, male, worst pre-T0 lactate and creatinine, SpO2 nadir,
        respiratory-rate peak, pre-T0 IV fluid, retained missing indicators;
        development-split median imputation; lifelines CoxPHFitter, Efron),
        then + SOFA total and, as a sensitivity, + the five non-cardiovascular
        components only (cardiovascular SOFA shares MAP with the HFI); also +
        SOFA under the alternative ETT-GCS convention when BM3a stored it.
        HR (95 % CI) of HFI per unit and per development-split SD, SOFA HR per
        point, attenuation of log-HR on the same rows, n, events, concordance,
        Schoenfeld p. REPRO_CHECK against the published B3 HR 1.20 (1.05-1.38).
  (iii) G-Transformer contrast by baseline SOFA tertile (test set; BM1
        per-stay predictions): RD of the non-adaptive and adaptive rule vs usual
        care by SOFA tertile (cut-points from all test stays) overall and in
        HFpEF, the T3 - T1 difference, with a paired patient-level bootstrap
        (percentile 95 % CI, models fixed), and the Spearman correlation between
        the per-patient contrast and SOFA (bootstrap CI). Skipped with a clear
        message if the BM1 per-stay file is absent.
  (iv)  The exact BM2 command that reruns the clone-censor-weighting analysis
        with SOFA in the censoring models.

Protocol reference: BMC Medicine revision BM3b (spec bmc_medicine_revision
SPEC.md); E04 model B3 (EJHF figure pipeline).
Compute: C2 (direct execution, env/run_py.sh): about 1-3 min on the real data
(1,000 paired bootstrap reps over ~5,800 test stays; a handful of Cox fits on
~2,000 HFpEF stays).

Inputs
------
- data/interim/bmc_medicine/BM3a_sofa_preT0.parquet (+ .meta.json)  [--sofa]
- data/interim/ejhf_figures_v1/F2_hfi_pret0.parquet                 [--hfi]
- data/interim/ejhf_figures_v1/E01_split_manifest.csv               [--split]
  (relative paths are tried under PROJECT_ROOT, then under
  /home/drzhuang/HFpEF_ICU_DTR_ejhf)
- data/interim/bmc_medicine/BM1_perstay_predictions.parquet (+ .meta.json) [--bm1]
- data/processed/phenotype_labels_frozen_v1_mimic.parquet (via configs/80_step_c_training_v1.yaml)

Outputs
-------
- tables/bmc_medicine/BM3b_table2_sofa_by_hf.csv
- tables/bmc_medicine/BM3b_table2_comparisons.csv
- tables/bmc_medicine/BM3b_cox_hfi_sofa.csv             (all terms of every model)
- tables/bmc_medicine/BM3b_cox_hfi_summary.csv          (HFI / SOFA rows, per unit and per SD)
- tables/bmc_medicine/BM3b_sofa_tertile_bootstrap_long.csv
- tables/bmc_medicine/BM3b_sofa_tertile_summary.csv
- tables/bmc_medicine/BM3b_sofa_contrast_spearman.csv
- reports/bmc_medicine/BM3b_summary.txt (ends with the PASTE-BACK BLOCK)

Usage
-----
    env/run_py.sh scripts/bmc_medicine/BM3b_sofa_adjusted_analyses.py
    env/run_py.sh scripts/bmc_medicine/BM3b_sofa_adjusted_analyses.py --quick   # 20 bootstrap reps
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import log, phase  # noqa: E402

SCRIPT_VERSION = "1.0"
TAG = "BM3b"
N_PHASES = 6
SOFA_DEFAULT = "data/interim/bmc_medicine/BM3a_sofa_preT0.parquet"
HFI_DEFAULT = "data/interim/ejhf_figures_v1/F2_hfi_pret0.parquet"
SPLIT_DEFAULT = "data/interim/ejhf_figures_v1/E01_split_manifest.csv"
BM1_DEFAULT = "data/interim/bmc_medicine/BM1_perstay_predictions.parquet"
COMPONENTS = ["resp", "coag", "liver", "cardio", "cns", "renal"]
HF_GROUPS = OrderedDict([("all", None), ("hfpef", ["hfpef"]), ("hf_ef_lt50", ["hfref", "hfmref"]),
                         ("hf_ef_unknown", ["hf_ef_unknown"]), ("no_hf", ["no_hf"])])
MIN_CELL = 10                                  # E04 constant (missing-indicator entry rule)
# published E04 model B3 (tables/ejhf_figures_v1/S3_cox_full_specification.csv, F2D_cox_forest.csv)
B3_TARGET = {"hr": 1.2022, "lo": 1.0492, "hi": 1.3776, "n": 1988, "events": 274, "dev_sd": 0.7833,
             "hr_sd": 1.155, "lo_sd": 1.038, "hi_sd": 1.285}
B3_TOL = 0.01
ARM_COLS = ["y_uc", "y_rule_nonadaptive", "y_rule_adaptive", "y_rule_adaptive_fbapprox", "y_legacy_zero"]
CONTRASTS_ALL = [("rd_rule_na", "y_rule_nonadaptive", "y_uc"), ("rd_rule_ad", "y_rule_adaptive", "y_uc"),
                 ("rd_rule_ad_fbapprox", "y_rule_adaptive_fbapprox", "y_uc"),
                 ("rd_legacy_zero", "y_legacy_zero", "y_uc"), ("obs_minus_uc", "mortality_28d", "y_uc")]
TERTILE_LABELS = ["T1", "T2", "T3"]


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BM3b: SOFA-adjusted analyses (Table 2, HFI Cox, SOFA-tertile contrast)")
    common.add_common_args(p, n_boot_default=1000, model_based=False)
    p.add_argument("--sofa", default=SOFA_DEFAULT, help="BM3a stay-level SOFA parquet")
    p.add_argument("--hfi", default=HFI_DEFAULT, help="E03 F2_hfi_pret0.parquet (EJHF interim)")
    p.add_argument("--split", default=SPLIT_DEFAULT, help="E01 split manifest CSV (EJHF interim)")
    p.add_argument("--bm1", default=BM1_DEFAULT, help="BM1 per-stay predictions parquet")
    p.add_argument("--sofa-col", default="sofa_total", help="SOFA column used throughout")
    return p.parse_args(argv)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════
def read_meta(path: Path) -> dict:
    mp = path.with_suffix(".meta.json")
    if mp.is_file():
        try:
            return json.loads(mp.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover
            return {}
    return {}


def med_iqr(x: np.ndarray, d: int = 0) -> str:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not x.size:
        return "NA"
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return f"{q2:.{d}f} ({q1:.{d}f}-{q3:.{d}f})"


def smd(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def mw_p(a, b) -> float:
    try:
        return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue) if len(a) and len(b) else float("nan")
    except ValueError:
        return float("nan")


def fmt_p(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "NA"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def hr_str(hr, lo, hi, d: int = 2) -> str:
    return common.fmt_ci(hr, lo, hi, d)


# ═══════════════════════════════════════════════════════════════════════
# (ii) E04 model B3 frame — port of E04_km_cox.py lines 120-215 (no n == 2,080 assertion)
# ═══════════════════════════════════════════════════════════════════════
def build_e04_frames(hfi: pd.DataFrame, split: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame, list, dict]:
    diag = {}
    if "split" not in hfi.columns:
        if split is None:
            raise ValueError("F2 frame has no 'split' column and no split manifest was given")
        hfi = hfi.merge(split[["stay_id", "split"]], on="stay_id", how="left")

    d = hfi[hfi["hf_subtype"] == "hfpef"].copy()
    d["surv_event"] = pd.to_numeric(d["surv_event"], errors="coerce").fillna(0).astype(int)
    d["surv_time_days"] = pd.to_numeric(d["surv_time_days"], errors="coerce")
    diag["hfpef_n"], diag["hfpef_events"] = int(len(d)), int(d["surv_event"].sum())
    log(f"  HFpEF n={diag['hfpef_n']:,} (E04 expects 2,080; not asserted here), events={diag['hfpef_events']:,}")
    zero = (d["surv_time_days"] <= 0) & (d["surv_event"] == 1)
    diag["n_zero_time_events"] = int(zero.sum())
    if diag["n_zero_time_events"]:
        d.loc[zero, "surv_time_days"] = 1e-6

    dev_mask = d["split"] == "train"
    d["age10"] = pd.to_numeric(d["age"], errors="coerce") / 10.0
    d["male"] = d["gender"].astype(str).str.upper().str.startswith("M").astype(int)
    d["fluid_l"] = pd.to_numeric(d["pre_t0_iv_fluid_ml"], errors="coerce") / 1000.0
    cont = ["age10", "lactate_max", "creatinine_max", "spo2_min", "rr_max", "fluid_l"]
    miss_frac, dev_median = {}, {}
    for v in cont:
        x = pd.to_numeric(d[v], errors="coerce")
        miss_frac[v] = float(x.isna().mean())
        dev_median[v] = float(np.nanmedian(pd.to_numeric(d.loc[dev_mask, v], errors="coerce")))
        d[v] = x.fillna(dev_median[v])
    diag["missing_fraction"] = miss_frac
    diag["dev_medians"] = dev_median

    ind_cols, lab_ind = [], {}
    for v in ["lactate_max", "creatinine_max", "spo2_min", "rr_max", "fluid_l"]:
        if miss_frac[v] > 0.02:
            src = v if v in hfi.columns else None
            if src is None:   # fluid_l is derived; E04 indexes hfi by the same name (never > 2 % in E04)
                lab_ind[v] = pd.to_numeric(hfi.loc[d.index, "pre_t0_iv_fluid_ml"], errors="coerce").isna().astype(int).to_numpy()
            else:
                lab_ind[v] = pd.to_numeric(hfi.loc[d.index, v], errors="coerce").isna().astype(int).to_numpy()
    if {"lactate_max", "creatinine_max"} <= set(lab_ind):
        agree = float((lab_ind["lactate_max"] == lab_ind["creatinine_max"]).mean())
        diag["lab_missing_indicator_agreement"] = agree
        if agree > 0.95:
            d["labs_missing"] = ((lab_ind["lactate_max"] + lab_ind["creatinine_max"]) > 0).astype(int)
            ind_cols.append("labs_missing")
            for v in ["lactate_max", "creatinine_max"]:
                lab_ind.pop(v)
    for v, arr in lab_ind.items():
        nm = f"{v}_missing"
        d[nm] = arr
        if MIN_CELL <= int(arr.sum()) <= len(d) - MIN_CELL:
            ind_cols.append(nm)
    diag["missing_indicators_entered"] = ind_cols
    adj_full = ["age10", "male", "lactate_max", "creatinine_max", "spo2_min", "rr_max", "fluid_l"] + ind_cols

    dh = d[d["hfi_coverage_ok"] & np.isfinite(pd.to_numeric(d["hfi_pret0"], errors="coerce"))].copy()
    dh["hfi_pret0"] = pd.to_numeric(dh["hfi_pret0"], errors="coerce")
    dh["hfi_z"] = dh["hfi_pret0"].astype(float)
    diag["hfi_subset_n"], diag["hfi_subset_events"] = int(len(dh)), int(dh["surv_event"].sum())
    diag["dev_sd"] = float(hfi.loc[(hfi["split"] == "train") & hfi["hfi_coverage_ok"], "hfi_pret0"].std(ddof=1))
    return d, dh, adj_full, diag


def dmat(frame: pd.DataFrame, terms: list) -> pd.DataFrame:
    X = frame[["surv_time_days", "surv_event"]].copy()
    for t in terms:
        X[t] = pd.to_numeric(frame[t], errors="coerce").astype(float)
    return X.dropna()


def fit_cox(frame: pd.DataFrame, terms: list, label: str, rows: pd.Index | None = None) -> dict:
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test
    X = dmat(frame, terms)
    if rows is not None:
        X = X.loc[X.index.intersection(rows)]
    res = {"label": label, "terms": terms, "n": int(len(X)), "events": int(X["surv_event"].sum()),
           "status": "complete", "note": ""}
    if res["events"] < 2 or len(X) < len(terms) + 2:
        res.update(status="blocked", note="too few rows/events")
        return res
    cph = CoxPHFitter()
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cph.fit(X, "surv_time_days", "surv_event")
            conv = [str(x.message) for x in w if "converg" in str(x.message).lower()]
    except Exception as exc:
        res.update(status="blocked", note=f"fit failed: {type(exc).__name__}: {exc}")
        log(f"  {label}: FIT FAILED {exc}")
        return res
    if conv:
        res["note"] = "convergence warning: " + conv[0][:120]
    sm = cph.summary
    res["summary"] = sm
    res["concordance"] = float(cph.concordance_index_)
    res["loglik"] = float(cph.log_likelihood_)
    res["index"] = X.index
    try:
        ph = proportional_hazard_test(cph, X, time_transform="rank").summary
        res["ph_p"] = {str(k[0] if isinstance(k, tuple) else k): float(v) for k, v in ph["p"].items()}
    except Exception:
        res["ph_p"] = {}
    log(f"  {label}: n={res['n']:,} events={res['events']} params={len(terms)} "
        f"concordance={res['concordance']:.3f}" + (f" [{res['note']}]" if res["note"] else ""))
    return res


def term(res: dict, t: str, scale: float = 1.0) -> dict:
    if res.get("status") != "complete" or t not in res["summary"].index:
        return {"hr": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan, "coef": np.nan, "se": np.nan}
    r = res["summary"].loc[t]
    return {"coef": float(r["coef"]), "se": float(r["se(coef)"]), "hr": float(np.exp(r["coef"] * scale)),
            "lo": float(np.exp(r["coef lower 95%"] * scale)), "hi": float(np.exp(r["coef upper 95%"] * scale)),
            "p": float(r["p"])}


# ═══════════════════════════════════════════════════════════════════════
# (iii) SOFA tertiles
# ═══════════════════════════════════════════════════════════════════════
def tertile_cut(x: np.ndarray) -> tuple[np.ndarray, tuple, str]:
    """Tie-aware tertiles of a discrete score: <=c1, (c1,c2], >c2 with c1/c2 the 1/3 and 2/3 quantiles;
    if c1 == c2 the groups become < c, == c, > c."""
    c1, c2 = np.quantile(x, [1 / 3, 2 / 3])
    if c1 < c2:
        g = np.where(x <= c1, 0, np.where(x <= c2, 1, 2))
        desc = f"T1<= {c1:g}; T2 {c1:g}<x<= {c2:g}; T3> {c2:g}"
    else:
        g = np.where(x < c1, 0, np.where(x == c1, 1, 2))
        desc = f"T1< {c1:g}; T2= {c1:g}; T3> {c1:g} (tied quantiles)"
    return g, (float(c1), float(c2)), desc


def spearman_boot(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int, desc: str) -> dict:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = int(x.size)
    if n < 5 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"n": n, "rho": np.nan, "p": np.nan, "lo": np.nan, "hi": np.nan}
    r = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    draws = np.full(n_boot, np.nan)
    for b in tqdm(range(n_boot), desc=f"  {desc}", mininterval=5, leave=False):
        i = rng.integers(0, n, size=n)
        if np.ptp(x[i]) > 0 and np.ptp(y[i]) > 0:
            draws[b] = stats.spearmanr(x[i], y[i]).statistic
    dr = draws[np.isfinite(draws)]
    lo, hi = (np.percentile(dr, [2.5, 97.5]) if dr.size else (np.nan, np.nan))
    return {"n": n, "rho": float(r.statistic), "p": float(r.pvalue), "lo": float(lo), "hi": float(hi)}


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main(argv=None) -> int:
    t_start = time.time()
    args = common.apply_quick(parse_args(argv))
    print("=" * 72, flush=True)
    print(f"BM3b — SOFA-adjusted analyses (v{SCRIPT_VERSION})", flush=True)
    print("=" * 72, flush=True)

    # ── 1. Inputs ────────────────────────────────────────────────────
    t0 = phase(1, N_PHASES, "Load BM3a SOFA, phenotype strata, F2 HFI frame, BM1 per-stay predictions")
    root = common.find_root(__file__)
    paths = common.setup_paths(root)
    log(f"  PROJECT_ROOT={root}")
    sofa_path = common.resolve_input_path(root, args.sofa, fallback_roots=(), required=False)
    if sofa_path is None:
        raise FileNotFoundError(f"BM3a SOFA file not found ({args.sofa}); run BM3a_extract_sofa_preT0.py first")
    sofa = pd.read_parquet(sofa_path)
    sofa_meta = read_meta(sofa_path)
    sofa_quick = bool(sofa_meta.get("quick", False)) or any((sofa_meta.get("raw_truncated") or {}).values())
    if sofa_quick:
        log("  WARNING: the SOFA file comes from a QUICK / truncated BM3a run (not for interpretation)")
    sc = args.sofa_col
    if sc not in sofa.columns:
        raise KeyError(f"--sofa-col {sc} not in {sofa_path.name}; columns: {list(sofa.columns)}")
    sofa["stay_id"] = pd.to_numeric(sofa["stay_id"]).astype("int64")
    if "sofa_noncv" not in sofa.columns:
        sofa["sofa_noncv"] = sofa[sc] - sofa["sofa_cardio"]
    alt_cols = [c for c in sofa.columns if c.startswith("sofa_total_ett_")]
    alt_col = alt_cols[0] if alt_cols else None
    log(f"  SOFA: {len(sofa):,} stays ({sofa_path.name}; quick={sofa_quick}; ETT primary "
        f"{sofa_meta.get('gcs_ett_primary', 'NA')}; alternative column {alt_col})")
    cfg = common.load_yaml(root / common.CONFIG_REL)
    pheno = pd.read_parquet(root / cfg["data"]["phenotype_labels"],
                            columns=["stay_id", "hf_subtype", "rule_phenotype_2class"])
    pheno["stay_id"] = pd.to_numeric(pheno["stay_id"]).astype("int64")
    hfi_path = common.resolve_input_path(root, args.hfi)
    split_path = common.resolve_input_path(root, args.split, required=False)
    hfi = pd.read_parquet(hfi_path)
    split = pd.read_csv(split_path) if split_path is not None else None
    log(f"  F2 HFI frame: {len(hfi):,} rows ({hfi_path}); split manifest: {split_path}")
    split_agree = np.nan
    if split is not None and "split" in hfi.columns:
        mm = hfi[["stay_id", "split"]].merge(split[["stay_id", "split"]], on="stay_id", how="inner",
                                             suffixes=("", "_e01"))
        split_agree = float((mm["split"] == mm["split_e01"]).mean()) if len(mm) else np.nan
        log(f"  F2 'split' vs E01 manifest agreement: {100 * split_agree:.2f}% on {len(mm):,} stays")
    bm1_path = common.resolve_input_path(root, args.bm1, fallback_roots=(), required=False)
    bm1 = pd.read_parquet(bm1_path) if bm1_path is not None else None
    bm1_meta = read_meta(bm1_path) if bm1_path is not None else {}
    if bm1 is None:
        log(f"  BM1 per-stay file not found ({args.bm1}): part (iii) will be SKIPPED "
            f"(run BM1_adaptive_rollout.py first)")
    else:
        log(f"  BM1 per-stay: {len(bm1):,} stays (quick={bm1_meta.get('quick')}, checkpoint "
            f"{bm1_meta.get('checkpoint')})")
    log(f"  [{time.time() - t0:.1f}s]")

    # ── 2. (i) Table 2 addition ──────────────────────────────────────
    t0 = phase(2, N_PHASES, "(i) Table 2: SOFA by heart-failure stratum")
    fr = sofa.merge(pheno, on="stay_id", how="left")
    n_no_ph = int(fr["hf_subtype"].isna().sum())
    if n_no_ph:
        log(f"  WARNING: {n_no_ph} SOFA stays without phenotype labels (in 'all' only)")
    hf = fr["hf_subtype"].astype(str)
    gmask = OrderedDict((g, np.ones(len(fr), bool) if v is None else hf.isin(v).to_numpy()) for g, v in HF_GROUPS.items())
    t2_rows = []
    for g, m in gmask.items():
        sub = fr.loc[m]
        x = sub[sc].to_numpy(float)
        nc = sub["sofa_noncv"].to_numpy(float)
        row = {"group": g, "n": int(m.sum()),
               "sofa_median": float(np.median(x)) if x.size else np.nan,
               "sofa_q1": float(np.percentile(x, 25)) if x.size else np.nan,
               "sofa_q3": float(np.percentile(x, 75)) if x.size else np.nan,
               "sofa_mean": float(x.mean()) if x.size else np.nan,
               "sofa_sd": float(x.std(ddof=1)) if x.size > 1 else np.nan,
               "sofa_ge6_n": int((x >= 6).sum()), "sofa_ge6_pct": 100 * float((x >= 6).mean()) if x.size else np.nan,
               "sofa_ge2_pct": 100 * float((x >= 2).mean()) if x.size else np.nan,
               "sofa_noncv_median": float(np.median(nc)) if nc.size else np.nan,
               "sofa_noncv_q1": float(np.percentile(nc, 25)) if nc.size else np.nan,
               "sofa_noncv_q3": float(np.percentile(nc, 75)) if nc.size else np.nan,
               "any_component_missing_pct": (100 * float((sub["n_components_missing"] > 0).mean())
                                             if "n_components_missing" in sub and len(sub) else np.nan)}
        for c in COMPONENTS:
            if f"sofa_{c}" in sub:
                row[f"{c}_mean"] = float(sub[f"sofa_{c}"].mean()) if len(sub) else np.nan
        if alt_col:
            row["sofa_alt_ett_median"] = float(sub[alt_col].median()) if len(sub) else np.nan
            row["sofa_alt_ett_mean"] = float(sub[alt_col].mean()) if len(sub) else np.nan
        t2_rows.append(row)
    t2 = pd.DataFrame(t2_rows)
    common.save_table(t2, paths["tables"] / "BM3b_table2_sofa_by_hf.csv")
    xs = {g: fr.loc[m, sc].to_numpy(float) for g, m in gmask.items()}
    cmp_rows = []
    for other in ("hf_ef_lt50", "no_hf", "hf_ef_unknown"):
        for col, lab in ((sc, "sofa_total"), ("sofa_noncv", "sofa_noncv")):
            a, b = fr.loc[gmask["hfpef"], col].to_numpy(float), fr.loc[gmask[other], col].to_numpy(float)
            cmp_rows.append({"comparison": f"hfpef_vs_{other}", "variable": lab, "n_a": a.size, "n_b": b.size,
                             "mean_diff": (a.mean() - b.mean()) if a.size and b.size else np.nan,
                             "smd": smd(a, b), "p_mannwhitney": mw_p(a, b)})
        a, b = fr.loc[gmask["hfpef"], sc].to_numpy(float), fr.loc[gmask[other], sc].to_numpy(float)
        cmp_rows.append({"comparison": f"hfpef_vs_{other}", "variable": "sofa_ge6", "n_a": a.size, "n_b": b.size,
                         "mean_diff": 100 * ((a >= 6).mean() - (b >= 6).mean()) if a.size and b.size else np.nan,
                         "smd": np.nan,
                         "p_mannwhitney": (float(stats.chi2_contingency(
                             [[int((a >= 6).sum()), int((a < 6).sum())], [int((b >= 6).sum()), int((b < 6).sum())]])[1])
                             if a.size and b.size and ((a >= 6).sum() + (b >= 6).sum()) > 0
                             and ((a < 6).sum() + (b < 6).sum()) > 0 else np.nan)})
    groups4 = [xs[g] for g in ("hfpef", "hf_ef_lt50", "hf_ef_unknown", "no_hf") if xs[g].size]
    try:
        p_kw = float(stats.kruskal(*groups4).pvalue) if len(groups4) >= 2 else np.nan
    except ValueError:
        p_kw = np.nan
    cmp_rows.append({"comparison": "kruskal_wallis_4_groups", "variable": "sofa_total", "n_a": int(sum(map(len, groups4))),
                     "n_b": np.nan, "mean_diff": np.nan, "smd": np.nan, "p_mannwhitney": p_kw})
    cmp = pd.DataFrame(cmp_rows)
    common.save_table(cmp, paths["tables"] / "BM3b_table2_comparisons.csv")
    for r in t2_rows:
        log(f"  {r['group']:14s} n={r['n']:>6,}  SOFA {common.fmt_num(r['sofa_median'], 0)} "
            f"({common.fmt_num(r['sofa_q1'], 0)}-{common.fmt_num(r['sofa_q3'], 0)}); >=6 "
            f"{common.fmt_num(r['sofa_ge6_pct'], 1)}%")
    log(f"  [{time.time() - t0:.1f}s]")

    # ── 3. (ii) Fragility-index Cox model ────────────────────────────
    t0 = phase(3, N_PHASES, "(ii) HFI_preT0 Cox model B3 in HFpEF, without and with SOFA")
    d, dh, adj_full, cdiag = build_e04_frames(hfi, split)
    dev_sd = cdiag["dev_sd"]
    log(f"  ADJ_FULL: {adj_full}; development-split HFI SD {dev_sd:.4f} (published {B3_TARGET['dev_sd']})")
    smap = sofa.set_index("stay_id")
    for col in [sc, "sofa_noncv"] + ([alt_col] if alt_col else []):
        dh[col] = dh["stay_id"].map(smap[col]).astype(float)
    n_sofa_missing = int(dh[sc].isna().sum())
    if n_sofa_missing:
        log(f"  WARNING: {n_sofa_missing} HFpEF HFI stays without SOFA (excluded from SOFA models)")
    base_terms = ["hfi_z"] + adj_full
    models = OrderedDict()
    models["B3"] = fit_cox(dh, base_terms, "B3 (replication)")
    rows_sofa = dmat(dh, base_terms + [sc]).index
    models["B3_sofa_rows"] = fit_cox(dh, base_terms, "B3 on SOFA-complete rows", rows=rows_sofa)
    models["B3_sofa"] = fit_cox(dh, base_terms + [sc], "B3 + SOFA total")
    models["B3_sofa_noncv"] = fit_cox(dh, base_terms + ["sofa_noncv"], "B3 + non-CV SOFA (5 components)")
    if alt_col:
        models["B3_sofa_alt"] = fit_cox(dh, base_terms + [alt_col], f"B3 + SOFA ({alt_col})")
    cox_rows, summ_rows = [], []
    for mid, res in models.items():
        if res.get("status") != "complete":
            cox_rows.append({"model": mid, "model_label": res["label"], "term": "(model)", "status": res["status"],
                             "note": res["note"], "n": res["n"], "n_events": res["events"]})
            continue
        for t in res["terms"]:
            tt = term(res, t)
            cox_rows.append({"model": mid, "model_label": res["label"], "term": t, "coef": tt["coef"], "se": tt["se"],
                             "hr": tt["hr"], "hr_ci_lo": tt["lo"], "hr_ci_hi": tt["hi"], "p_value": tt["p"],
                             "schoenfeld_p": res["ph_p"].get(t, np.nan), "n": res["n"], "n_events": res["events"],
                             "n_parameters": len(res["terms"]), "concordance": res["concordance"],
                             "log_likelihood": res["loglik"], "status": "complete", "note": res["note"]})
    cox = pd.DataFrame(cox_rows)
    common.save_table(cox, paths["tables"] / "BM3b_cox_hfi_sofa.csv")
    H = {mid: {"unit": term(r, "hfi_z"), "sd": term(r, "hfi_z", dev_sd)} for mid, r in models.items()}
    sofa_terms = {"B3_sofa": sc, "B3_sofa_noncv": "sofa_noncv", "B3_sofa_alt": alt_col}
    S = {mid: term(models[mid], t) for mid, t in sofa_terms.items() if mid in models and t}
    ref_coef = H["B3_sofa_rows"]["unit"]["coef"]
    atten = {}
    for mid in ("B3_sofa", "B3_sofa_noncv", "B3_sofa_alt"):
        if mid in models:
            c = H[mid]["unit"]["coef"]
            atten[mid] = 100.0 * (ref_coef - c) / ref_coef if np.isfinite(ref_coef) and ref_coef != 0 else np.nan
    for mid, res in models.items():
        u, s_ = H[mid]["unit"], H[mid]["sd"]
        summ_rows.append({"model": mid, "model_label": res["label"], "status": res.get("status"),
                          "n": res["n"], "n_events": res["events"],
                          "hfi_hr_per_unit": u["hr"], "hfi_ci_lo": u["lo"], "hfi_ci_hi": u["hi"], "hfi_p": u["p"],
                          "hfi_hr_per_dev_sd": s_["hr"], "hfi_sd_ci_lo": s_["lo"], "hfi_sd_ci_hi": s_["hi"],
                          "dev_sd": dev_sd,
                          "sofa_hr_per_point": S.get(mid, {}).get("hr", np.nan),
                          "sofa_ci_lo": S.get(mid, {}).get("lo", np.nan), "sofa_ci_hi": S.get(mid, {}).get("hi", np.nan),
                          "sofa_p": S.get(mid, {}).get("p", np.nan),
                          "hfi_attenuation_pct_vs_same_rows": atten.get(mid, np.nan),
                          "concordance": res.get("concordance", np.nan),
                          "hfi_schoenfeld_p": res.get("ph_p", {}).get("hfi_z", np.nan) if res.get("ph_p") else np.nan})
    summ = pd.DataFrame(summ_rows)
    common.save_table(summ, paths["tables"] / "BM3b_cox_hfi_summary.csv")
    b3 = H["B3"]["unit"]
    rc_ok = bool(np.isfinite(b3["hr"]) and abs(b3["hr"] - B3_TARGET["hr"]) <= B3_TOL
                 and abs(b3["lo"] - B3_TARGET["lo"]) <= B3_TOL and abs(b3["hi"] - B3_TARGET["hi"]) <= B3_TOL)
    rc_line = (f"REPRO_CHECK: {'PASS' if rc_ok else 'FAIL'} (E04 model B3 HFI HR per unit "
               f"{hr_str(b3['hr'], b3['lo'], b3['hi'], 3)} vs {B3_TARGET['hr']} ({B3_TARGET['lo']}-{B3_TARGET['hi']}), "
               f"tolerance {B3_TOL} on HR and both limits)")
    n_ok = models["B3"]["n"] == B3_TARGET["n"] and models["B3"]["events"] == B3_TARGET["events"]
    rc_n_line = (f"REPRO_CHECK_N: {'PASS' if n_ok else 'FAIL'} (B3 n={models['B3']['n']:,} events="
                 f"{models['B3']['events']} vs {B3_TARGET['n']:,} / {B3_TARGET['events']})")
    sd_ok = abs(dev_sd - B3_TARGET["dev_sd"]) <= 0.001
    rc_sd_line = (f"REPRO_CHECK_DEVSD: {'PASS' if sd_ok else 'FAIL'} (development-split HFI SD {dev_sd:.4f} vs "
                  f"{B3_TARGET['dev_sd']})")
    for line in (rc_line, rc_n_line, rc_sd_line):
        print(line, flush=True)
    log(f"  [{time.time() - t0:.1f}s]")

    # ── 4. (iii) G-Transformer contrast by SOFA tertile ──────────────
    t0 = phase(4, N_PHASES, f"(iii) G-Transformer contrast by baseline SOFA tertile ({args.n_boot} bootstrap reps)")
    iii = {"status": "skipped"}
    if bm1 is not None:
        b = bm1.copy()
        b["stay_id"] = pd.to_numeric(b["stay_id"]).astype("int64")
        b = b.merge(sofa[["stay_id", sc]], on="stay_id", how="left")
        n_nosofa = int(b[sc].isna().sum())
        if n_nosofa:
            log(f"  WARNING: {n_nosofa} BM1 test stays without SOFA (dropped from part iii)")
        b = b.dropna(subset=[sc]).reset_index(drop=True)
        if "hf_subtype" not in b or b["hf_subtype"].isna().all():
            b = b.drop(columns=[c for c in ("hf_subtype", "rule_phenotype_2class") if c in b]).merge(
                pheno, on="stay_id", how="left")
        arms = [c for c in ARM_COLS if c in b.columns]
        contrasts = [c for c in CONTRASTS_ALL if c[1] in b.columns and c[2] in b.columns]
        x = b[sc].to_numpy(float)
        tert, cuts, cut_desc = tertile_cut(x)
        b["tertile"] = np.array(TERTILE_LABELS)[tert]
        is_hfpef = (b["hf_subtype"].astype(str) == "hfpef").to_numpy()
        strata = OrderedDict()
        strata["all"] = np.ones(len(b), bool)
        for k, lab in enumerate(TERTILE_LABELS):
            strata[f"all_{lab}"] = tert == k
        strata["hfpef"] = is_hfpef
        for k, lab in enumerate(TERTILE_LABELS):
            strata[f"hfpef_{lab}"] = is_hfpef & (tert == k)
        sc_pairs = [("all_T3_minus_T1", "all_T3", "all_T1"), ("hfpef_T3_minus_T1", "hfpef_T3", "hfpef_T1")]
        res = common.paired_bootstrap(b, ["mortality_28d"] + arms, strata, n_boot=args.n_boot, seed=args.seed,
                                      contrasts=contrasts, stratum_contrasts=sc_pairs, scale=100.0,
                                      desc="paired bootstrap by SOFA tertile")
        res.insert(0, "sofa_col", sc)
        common.save_table(res, paths["tables"] / "BM3b_sofa_tertile_bootstrap_long.csv")
        wide = []
        for s, m in strata.items():
            row = {"stratum": s, "n": int(m.sum()), "deaths": int(b.loc[m, "mortality_28d"].sum()),
                   "sofa_mean": float(np.mean(x[m])) if m.any() else np.nan,
                   "sofa_median": float(np.median(x[m])) if m.any() else np.nan}
            for c in ["mortality_28d"] + arms:
                rr = common.get_est(res, s, c, "mean")
                row[f"{c}_pct"] = rr["estimate"]
            for cname, _, _ in contrasts:
                rr = common.get_est(res, s, cname, "diff")
                row[f"{cname}_pp"], row[f"{cname}_lo"], row[f"{cname}_hi"] = rr["estimate"], rr["ci_lo"], rr["ci_hi"]
            wide.append(row)
        wide = pd.DataFrame(wide)
        common.save_table(wide, paths["tables"] / "BM3b_sofa_tertile_summary.csv")
        sp_rows = []
        for cname, a_, b_ in contrasts:
            if cname == "obs_minus_uc":
                continue
            dlt = (b[a_] - b[b_]).to_numpy(float) * 100.0
            for s, m in (("all", strata["all"]), ("hfpef", strata["hfpef"])):
                r = spearman_boot(x[m], dlt[m], args.n_boot, args.seed, f"spearman {cname} {s}")
                sp_rows.append({"contrast": cname, "stratum": s, **r})
        spear = pd.DataFrame(sp_rows)
        common.save_table(spear, paths["tables"] / "BM3b_sofa_contrast_spearman.csv")
        iii = {"status": "complete", "res": res, "wide": wide, "spear": spear, "cuts": cuts, "cut_desc": cut_desc,
               "contrasts": contrasts, "strata": strata, "n": int(len(b)), "n_nosofa": n_nosofa,
               "sc_pairs": sc_pairs}
        log(f"  tertile cut-points ({sc}): {cut_desc}; n per tertile "
            + ", ".join(f"{s}={int(m.sum())}" for s, m in strata.items()))
    log(f"  [{time.time() - t0:.1f}s]")

    # ── 5. (iv) BM2 command ──────────────────────────────────────────
    t0 = phase(5, N_PHASES, "(iv) BM2 clone-censor-weighting rerun with SOFA")
    sofa_rel = (str(sofa_path.relative_to(root)) if sofa_path.is_relative_to(root) else str(sofa_path))
    bm2_direct = (f"export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; env/run_py.sh "
                  f"scripts/bmc_medicine/BM2_ccw_dynamic_rule.py --n-jobs 8 --sofa {sofa_rel}")
    # submit_c4.sh sets no BLAS thread limits; qsub -V forwards these to the job
    bm2_qsub = (f"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                f"qsub -N sBM2_ccw_sofa env/sge/submit_c4.sh scripts/bmc_medicine/BM2_ccw_dynamic_rule.py "
                f"--sofa {sofa_rel}")
    print("BM2_SOFA_COMMAND (C4 qsub, recommended): " + bm2_qsub, flush=True)
    print("BM2_SOFA_COMMAND (direct): " + bm2_direct, flush=True)
    log(f"  [{time.time() - t0:.1f}s]")

    # ── 6. Report ────────────────────────────────────────────────────
    t0 = phase(6, N_PHASES, "Report and paste-back block")
    runtime_min = (time.time() - t_start) / 60.0
    T2 = {r["group"]: r for r in t2_rows}
    C = {(r["comparison"], r["variable"]): r for r in cmp_rows}
    L = common.report_header("BM3b — SOFA-adjusted analyses (BMC Medicine revision)", root, args)
    L += [f"Script version {SCRIPT_VERSION}; runtime {runtime_min:.1f} min"
          + ("  [QUICK: 20 bootstrap reps]" if args.quick else ""),
          f"SOFA file: {sofa_path.name} ({len(sofa):,} stays; quick/truncated={sofa_quick}; column {sc}; "
          f"ETT-GCS primary {sofa_meta.get('gcs_ett_primary', 'NA')}; lookback "
          f"{sofa_meta.get('lookback_hours', 'NA')} h)",
          "** WARNING: SOFA from a QUICK / truncated BM3a run — rerun BM3a in full **" if sofa_quick else "",
          "",
          "(i) Table 2 addition — baseline (ICU admission to T0) SOFA by heart-failure stratum:",
          f"  {'group':14s} {'n':>7s}  {'median (IQR)':>12s}  {'mean (SD)':>12s}  {'>=6 %':>6s}  "
          f"{'non-CV median (IQR)':>19s}"]
    for g, r in T2.items():
        L.append(f"  {g:14s} {r['n']:>7,}  {med_iqr(fr.loc[gmask[g], sc].to_numpy(float)):>12s}  "
                 f"{common.fmt_num(r['sofa_mean']) + ' (' + common.fmt_num(r['sofa_sd']) + ')':>12s}  "
                 f"{common.fmt_num(r['sofa_ge6_pct'], 1):>6s}  "
                 f"{med_iqr(fr.loc[gmask[g], 'sofa_noncv'].to_numpy(float)):>19s}")
    L.append("  component means (resp/coag/liver/cardio/cns/renal): ")
    for g, r in T2.items():
        L.append(f"    {g:14s} " + "/".join(common.fmt_num(r.get(f"{c}_mean", np.nan)) for c in COMPONENTS))
    for other in ("hf_ef_lt50", "no_hf", "hf_ef_unknown"):
        c1, c2, c3 = C[(f"hfpef_vs_{other}", "sofa_total")], C[(f"hfpef_vs_{other}", "sofa_noncv")], \
            C[(f"hfpef_vs_{other}", "sofa_ge6")]
        L.append(f"  HFpEF vs {other:13s}: SOFA mean diff {common.fmt_num(c1['mean_diff'])}, SMD "
                 f"{common.fmt_num(c1['smd'])}, Mann-Whitney p {fmt_p(c1['p_mannwhitney'])}; non-CV SMD "
                 f"{common.fmt_num(c2['smd'])}; >=6 diff {common.fmt_num(c3['mean_diff'], 1)} pp (chi2 p "
                 f"{fmt_p(c3['p_mannwhitney'])})")
    L.append(f"  Kruskal-Wallis across HFpEF / EF<50 / EF unknown / no-HF: p {fmt_p(p_kw)}")
    L += ["", "(ii) HFI_preT0 Cox model in HFpEF (E04 model B3 rebuilt; lifelines CoxPHFitter):",
          f"  HFpEF n={cdiag['hfpef_n']:,}, events={cdiag['hfpef_events']:,}; HFI coverage-OK subset "
          f"n={cdiag['hfi_subset_n']:,}, events={cdiag['hfi_subset_events']:,}; development-split HFI SD {dev_sd:.4f}",
          f"  ADJ_FULL = {', '.join(adj_full)}",
          f"  F2 split vs E01 manifest agreement: {common.fmt_num(100 * split_agree, 2) if np.isfinite(split_agree) else 'NA'}%",
          f"  {'model':16s} {'n':>6s} {'ev':>5s}  {'HFI HR/unit (95% CI)':>24s}  {'HFI HR/SD (95% CI)':>24s}  "
          f"{'SOFA HR/point':>22s}  {'atten%':>6s}  {'C':>5s}"]
    for _, r in summ.iterrows():
        L.append(f"  {r['model']:16s} {int(r['n']):>6,} {int(r['n_events']):>5d}  "
                 f"{hr_str(r['hfi_hr_per_unit'], r['hfi_ci_lo'], r['hfi_ci_hi'], 3):>24s}  "
                 f"{hr_str(r['hfi_hr_per_dev_sd'], r['hfi_sd_ci_lo'], r['hfi_sd_ci_hi'], 3):>24s}  "
                 f"{(hr_str(r['sofa_hr_per_point'], r['sofa_ci_lo'], r['sofa_ci_hi'], 3) if np.isfinite(r['sofa_hr_per_point']) else '-'):>22s}  "
                 f"{common.fmt_num(r['hfi_attenuation_pct_vs_same_rows'], 1):>6s}  {common.fmt_num(r['concordance'], 3):>5s}")
    L += ["  attenuation = 100 x (log HR_B3 - log HR_B3+SOFA) / log HR_B3 on the same SOFA-complete rows",
          "  " + rc_line, "  " + rc_n_line, "  " + rc_sd_line,
          "  The non-CV model drops cardiovascular SOFA, which shares the MAP information of the HFI."]
    L += ["", "(iii) G-Transformer contrast by baseline SOFA tertile (BM1 test-set per-stay predictions):"]
    if iii["status"] == "complete":
        res = iii["res"]
        L.append(f"  BM1 file: {bm1_path.name} (quick={bm1_meta.get('quick')}, checkpoint {bm1_meta.get('checkpoint')}); "
                 f"n={iii['n']:,} with SOFA ({iii['n_nosofa']} without)")
        if bm1_meta.get("quick"):
            L.append("  ** WARNING: BM1 per-stay file is from a QUICK run (subsample) — rerun BM1 in full **")
        L.append(f"  tertile cut-points ({sc}, all test stays): {iii['cut_desc']}")
        L.append(f"  {'stratum':10s} {'n':>6s} {'SOFA':>5s} {'obs%':>6s} {'UC%':>6s}  "
                 + "  ".join(f"{c[0]:>24s}" for c in iii["contrasts"] if c[0] != "obs_minus_uc"))
        for _, w in iii["wide"].iterrows():
            L.append(f"  {w['stratum']:10s} {int(w['n']):>6,} {common.fmt_num(w['sofa_mean'], 1):>5s} "
                     f"{common.fmt_num(w['mortality_28d_pct']):>6s} {common.fmt_num(w.get('y_uc_pct', np.nan)):>6s}  "
                     + "  ".join(f"{common.fmt_ci(w[f'{c[0]}_pp'], w[f'{c[0]}_lo'], w[f'{c[0]}_hi'], 2):>24s}"
                                 for c in iii["contrasts"] if c[0] != "obs_minus_uc"))
        for scn, _, _ in iii["sc_pairs"]:
            L.append(f"  {scn}: " + "; ".join(
                f"{c[0]} {common.fmt_row_ci(common.get_est(res, scn, c[0], 'diff_of_diff'))}"
                for c in iii["contrasts"] if c[0] != "obs_minus_uc"))
        L.append("  Spearman rho between SOFA and the per-patient contrast (pp), bootstrap 95% CI:")
        for _, r in iii["spear"].iterrows():
            L.append(f"    {r['contrast']:22s} {r['stratum']:6s} n={int(r['n']):>6,} rho "
                     f"{common.fmt_ci(r['rho'], r['lo'], r['hi'], 3)} p {fmt_p(r['p'])}")
        L.append("  negative RD favours the rule; a more negative RD in T3 (diff_of_diff < 0) means the model")
        L.append("  attributes more benefit to sicker patients; rho < 0 means the same on a per-patient scale.")
    else:
        L.append(f"  SKIPPED: BM1 per-stay file not found ({args.bm1}); run BM1_adaptive_rollout.py first.")
    L += ["", "(iv) BM2 clone-censor-weighting rerun with baseline SOFA in the censoring models:",
          "  " + bm2_qsub, "  or directly: " + bm2_direct,
          "  BM2 adds sofa_total (median-imputed, + missing indicator if any) to every censoring model and",
          "  writes reports/bmc_medicine/BM2_sofa_summary.txt and tables BM2_*_sofa.csv (paste tag BM2_SOFA);",
          "  compare its RDs with BM2_summary.txt from the run without --sofa."]

    P = OrderedDict()
    P["script"] = "BM3b_sofa_adjusted_analyses"
    P["version"] = SCRIPT_VERSION
    P["quick"] = bool(args.quick)
    P["n_boot"] = int(args.n_boot)
    P["seed"] = int(args.seed)
    P["sofa_file_quick"] = sofa_quick
    P["sofa_col"] = sc
    P["sofa_gcs_ett_primary"] = sofa_meta.get("gcs_ett_primary", "NA")
    P["sofa_lookback_hours"] = sofa_meta.get("lookback_hours", "NA")
    P["n_sofa"] = int(len(sofa))
    for g, r in T2.items():
        P[f"t2_{g}_n"] = int(r["n"])
        P[f"t2_{g}_sofa_median_iqr"] = med_iqr(fr.loc[gmask[g], sc].to_numpy(float))
        P[f"t2_{g}_sofa_mean_sd"] = f"{common.fmt_num(r['sofa_mean'])} ({common.fmt_num(r['sofa_sd'])})"
        P[f"t2_{g}_sofa_ge6_n_pct"] = f"{r['sofa_ge6_n']} ({common.fmt_num(r['sofa_ge6_pct'], 1)}%)"
        P[f"t2_{g}_sofa_noncv_median_iqr"] = med_iqr(fr.loc[gmask[g], "sofa_noncv"].to_numpy(float))
        P[f"t2_{g}_component_means"] = "/".join(common.fmt_num(r.get(f"{c}_mean", np.nan)) for c in COMPONENTS)
        if alt_col:
            P[f"t2_{g}_sofa_alt_ett_mean"] = float(r.get("sofa_alt_ett_mean", np.nan))
    for other in ("hf_ef_lt50", "no_hf", "hf_ef_unknown"):
        c1, c2, c3 = C[(f"hfpef_vs_{other}", "sofa_total")], C[(f"hfpef_vs_{other}", "sofa_noncv")], \
            C[(f"hfpef_vs_{other}", "sofa_ge6")]
        P[f"t2_hfpef_vs_{other}_mean_diff"] = float(c1["mean_diff"])
        P[f"t2_hfpef_vs_{other}_smd"] = float(c1["smd"])
        P[f"t2_hfpef_vs_{other}_p_mw"] = fmt_p(c1["p_mannwhitney"])
        P[f"t2_hfpef_vs_{other}_noncv_smd"] = float(c2["smd"])
        P[f"t2_hfpef_vs_{other}_ge6_diff_pp"] = float(c3["mean_diff"])
    P["t2_p_kruskal_4groups"] = fmt_p(p_kw)
    P["cox_hfpef_n"] = cdiag["hfpef_n"]
    P["cox_hfpef_events"] = cdiag["hfpef_events"]
    P["cox_adj_full"] = ",".join(adj_full)
    P["cox_dev_sd"] = dev_sd
    P["cox_split_agreement_pct"] = 100 * split_agree if np.isfinite(split_agree) else float("nan")
    P["REPRO_CHECK"] = "PASS" if rc_ok else "FAIL"
    P["REPRO_CHECK_N"] = "PASS" if n_ok else "FAIL"
    P["REPRO_CHECK_DEVSD"] = "PASS" if sd_ok else "FAIL"
    for mid, res in models.items():
        u, s_ = H[mid]["unit"], H[mid]["sd"]
        P[f"{mid}_status"] = res.get("status")
        P[f"{mid}_n_events"] = f"{res['n']}/{res['events']}"
        P[f"{mid}_hfi_hr_unit"] = hr_str(u["hr"], u["lo"], u["hi"], 3)
        P[f"{mid}_hfi_p"] = fmt_p(u["p"])
        P[f"{mid}_hfi_hr_sd"] = hr_str(s_["hr"], s_["lo"], s_["hi"], 3)
        if mid in S:
            P[f"{mid}_sofa_hr_point"] = hr_str(S[mid]["hr"], S[mid]["lo"], S[mid]["hi"], 3)
            P[f"{mid}_sofa_p"] = fmt_p(S[mid]["p"])
        if mid in atten:
            P[f"{mid}_hfi_attenuation_pct"] = float(atten[mid])
        if res.get("status") == "complete":
            P[f"{mid}_concordance"] = float(res["concordance"])
            P[f"{mid}_hfi_schoenfeld_p"] = fmt_p(res["ph_p"].get("hfi_z", np.nan))
        if res.get("note"):
            P[f"{mid}_note"] = res["note"]
    P["iii_status"] = iii["status"] if iii["status"] == "complete" else f"skipped (BM1 per-stay file not found: {args.bm1})"
    if iii["status"] == "complete":
        res = iii["res"]
        P["iii_bm1_quick"] = bool(bm1_meta.get("quick", False))
        P["iii_bm1_checkpoint"] = str(bm1_meta.get("checkpoint", "NA"))
        P["iii_n"] = iii["n"]
        P["iii_n_without_sofa"] = iii["n_nosofa"]
        P["iii_tertile_cuts"] = iii["cut_desc"]
        for s, m in iii["strata"].items():
            w = iii["wide"][iii["wide"]["stratum"] == s].iloc[0]
            P[f"iii_{s}_n_deaths"] = f"{int(w['n'])}/{int(w['deaths'])}"
            P[f"iii_{s}_sofa_mean"] = float(w["sofa_mean"])
            P[f"iii_{s}_obs_mort_pct"] = float(w["mortality_28d_pct"])
            if "y_uc_pct" in w:
                P[f"iii_{s}_risk_uc_pct"] = float(w["y_uc_pct"])
            for c in iii["contrasts"]:
                if c[0] in ("rd_rule_na", "rd_rule_ad", "rd_rule_ad_fbapprox", "rd_legacy_zero"):
                    P[f"iii_{s}_{c[0]}_pp"] = common.fmt_ci(w[f"{c[0]}_pp"], w[f"{c[0]}_lo"], w[f"{c[0]}_hi"], 3)
        for scn, _, _ in iii["sc_pairs"]:
            for c in iii["contrasts"]:
                if c[0] in ("rd_rule_na", "rd_rule_ad"):
                    P[f"iii_{scn}_{c[0]}_pp"] = common.fmt_row_ci(common.get_est(res, scn, c[0], "diff_of_diff"), 3)
        for _, r in iii["spear"].iterrows():
            P[f"iii_spearman_{r['contrast']}_{r['stratum']}"] = (f"{common.fmt_ci(r['rho'], r['lo'], r['hi'], 3)} "
                                                                 f"p={fmt_p(r['p'])} n={int(r['n'])}")
    P["BM2_SOFA_COMMAND"] = bm2_qsub
    P["runtime_min"] = runtime_min
    common.write_report(paths["reports"] / "BM3b_summary.txt", L, P, TAG)
    log(f"  [{time.time() - t0:.1f}s]")
    log(f"Done in {runtime_min:.1f} min.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
