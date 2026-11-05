#!/usr/bin/env python3
"""
BM2_ccw_dynamic_rule — Clone-censor-weight (IPCW) per-protocol estimate of the
70c dynamic rule versus the natural course, with no outcome model
(BMC Medicine revision, BM2).

Purpose
-------
The primary Step C contrast is a G-Transformer g-computation (direct-method)
estimate: every number depends on the outcome / transition model. Reviewers
asked for an estimate of the dynamic rule that does NOT rely on an outcome model.
BM2 emulates the per-protocol trial "follow the 70c rule during the first H
2-h intervals after T0 (then usual care)" versus the natural course (usual
care) in the full analytic cohort, by artificial censoring at the first
deviation from the rule and inverse-probability-of-censoring weighting (IPCW):

  * Active intervals: bins 0..11 present in the 45c state tensors and
    bin <= absorbing_state_bin when not null (DTRDataset convention).
  * Adherence at an active interval (observed raw coarse_3 action vs the 70c
    rule action; raw = fluid*6 + vaso*2 + diuretic):
      strict  : merged observed action == merged rule action;
      relaxed : (obs vaso>0) == (rule vaso>0) AND (obs diuretic>0) ==
                (rule diuretic>0) AND |obs fluid level - rule fluid level| <= 1
                (PRIMARY).
  * Horizons H in {3, 6, 12} intervals (6, 12, 24 h; H = 12 primary):
    adherence is required only in active intervals with bin < H; a patient
    absorbed earlier (death / discharge) is complete once absorbed.
  * Censoring: at the first required interval with non-adherence.
  * IPCW: pooled logistic models of P(adherent at t | uncensored, history),
    fitted on the active person-intervals of uncensored patients, separately
    for each required rule category (raw 16, 14, 8, 6, 1); 16/14/8 are pooled
    (with category indicators) if any of them has < --min-events (200)
    intervals in either outcome class, and any category still below that
    count is pooled with the remaining small ones / the largest model.
    Covariates: bin (categorical), current state z-values (all *_z), current
    measurement masks (*_mask), previous interval's observed fluid / vaso /
    diuretic levels (dummies; 0 at bin 0), age, sex, admission-type dummies,
    hf_subtype dummies, rule_phenotype_2class (shock), hf_subtype x shock
    (stratum indicators), and baseline SOFA when --sofa is given (+ missing
    indicator). Continuous covariates are standardised; sklearn
    LogisticRegression(lbfgs, C=1e4, max_iter=5000). Fitted probabilities are
    clipped to [0.01, 1]. Weight of a fully adherent patient = prod 1/p_hat
    over its required intervals; truncated at the 99th percentile of the
    weights of adherent patients (also untruncated and 99.9th percentile).
  * Estimands per stratum: risk under the rule = weighted (Hajek) mean of
    mortality_28d among fully adherent patients; natural-course risk =
    observed mortality of all patients in the stratum; RD = rule - natural
    course (pp; negative favours the rule); naive = unweighted mortality among
    adherent patients (direction of confounding by adherence).
  * Strata: all; HFpEF; HFpEF non-shock; HFpEF shock; no-HF non-shock; HFrEF;
    plus the HFpEF-minus-no-HF non-shock difference of RDs. Stratum estimates
    use the pooled censoring models (all patients) with stratum indicators.
    Feasibility: < --min-adherent (50) fully adherent patients in a stratum ->
    "not estimable (n_adherent<50)".
  * Inference: patient-level nonparametric bootstrap (200 reps, percentile
    95% CI) with the censoring models REFITTED in every resample (frequency
    weights; warm start from the full-sample fit; model grouping fixed from
    the full sample; truncation caps recomputed), parallelised with joblib.
  * Sensitivity: strict adherence; H = 3 and 6; weight truncation none /
    p99.9; relaxed H = 12 with cumulative fluid balance (L, recorded) added to
    the censoring covariates (the rule's P4/P5 operand; not in the spec list,
    reported separately; --no-fb-sensitivity to skip).
  * Diagnostics: per-category adherence table (share of active intervals
    adherent by required rule category and stratum, with the criterion that
    fails), adherence "survival" by bin, censoring-model sizes / events / AUC /
    convergence, share of p_hat < 0.05, Kish ESS and weight distribution,
    baseline covariate balance (SMD) of the weighted adherent sample against
    the whole cohort, and RULE_SIDECAR_CHECK (the 70c rule recomputed from the
    raw 45c columns with the cumulative fluid balance must reproduce the 70c
    sidecar; the pre-v1.0.1 70c used the interval balance).

Protocol reference: BMC Medicine revision, analysis BM2 (CCW / IPCW, no outcome model).
Compute: C3 (direct, --n-jobs 8) or C4 (qsub, 32 slots); CPU only, no torch.
Benchmark (fixture replicated to 39,000 stays / 0.41 M person-intervals, 85
covariate columns, single thread): data + design ~10 s; cold point fits of all
configurations ~30 s; one warm-started bootstrap replicate of all
configurations ~5 s (primary only ~1.5 s). Expected on the real cohort
(38,981 stays, <= 0.47 M person-intervals, harder optimisation): ~5-15 min wall
with 8 workers (C3 direct), ~3-8 min on C4 with 32 workers; allow up to 3x if
the censoring models converge slowly. Memory: design matrix ~0.3 GB (float64),
memory-mapped and shared by the workers; peak ~1.5-2 GB in the main process,
~0.5 GB per worker.

Inputs
------
- configs/80_step_c_training_v1.yaml (data paths, n_bins), configs/70b_action_merge_map.yaml,
  configs/70c_rule_policy_v1.yaml (thresholds)
- data/interim/45c_state_tensors_v1_mimic.parquet (stay_id, bin, *_z, *_mask, age, gender,
  admission_type, hf_subtype, raw map/lactate/spo2, cumulative_fluid_balance_L, fluid_balance_L)
- data/interim/70b_propensity_coarse3_v1_mimic.parquet (action_idx, action_idx_merged)
  [fallback: data/interim/45b_actions_2h_v1_mimic.parquet action_idx + merge map]
- data/interim/70c_rule_policy_actions_v1_mimic.parquet (rule_policy_action)
- data/interim/45d_outcomes_absorb_v1_mimic.parquet (mortality_28d, absorbing_state_bin)
- phenotype labels (stay_id, rule_phenotype_2class; hf_subtype optional): config data.phenotype_labels
  (data/processed/phenotype_labels_frozen_v1_mimic.parquet) unless --phenotype-labels is given; the
  reported estimates used --phenotype-labels data/interim/60d_rule_phenotype_v1_mimic.parquet (the
  rule-based classification of Table 2; hf_subtype then comes from 45c)
- optional --sofa data/interim/bmc_medicine/BM3a_sofa_preT0.parquet (stay_id, sofa_total)

Outputs (suffix "_sofa" when --sofa is given)
-------
- tables/bmc_medicine/BM2_ccw_estimates_long.csv     (config x truncation x stratum x quantity, CI)
- tables/bmc_medicine/BM2_ccw_primary_summary.csv     (wide; relaxed H=12, p99 truncation)
- tables/bmc_medicine/BM2_weight_diagnostics.csv      (config x truncation x stratum)
- tables/bmc_medicine/BM2_censoring_models.csv        (config x model group)
- tables/bmc_medicine/BM2_censoring_coefficients.csv  (primary configuration)
- tables/bmc_medicine/BM2_adherence_by_category.csv   (definition x stratum x rule category)
- tables/bmc_medicine/BM2_adherence_survival.csv      (definition x stratum x bin)
- tables/bmc_medicine/BM2_balance.csv                 (baseline SMDs, primary configuration)
- reports/bmc_medicine/BM2_summary.txt                (ends with the PASTE-BACK block, tag BM2 / BM2_SOFA)
- data/interim/bmc_medicine/BM2_perstay_primary_weights.parquet (stay-level; never sync)
- a temporary bootstrap context (data/interim/bmc_medicine/_BM2_ctx_<pid>.joblib), deleted at exit

Usage
-----
    # Shirokane (project root), C4 qsub (32 slots; --n-jobs defaults to $NSLOTS).
    # submit_c4.sh sets no BLAS thread limits; qsub -V forwards the prefix below:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        qsub -N sBM2_ccw env/sge/submit_c4.sh scripts/bmc_medicine/BM2_ccw_dynamic_rule.py
    # with baseline SOFA from BM3a:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        qsub -N sBM2_ccw_sofa env/sge/submit_c4.sh scripts/bmc_medicine/BM2_ccw_dynamic_rule.py \\
        --sofa data/interim/bmc_medicine/BM3a_sofa_preT0.parquet
    # C3 direct (login / interactive node, 8 threads):
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    env/run_py.sh scripts/bmc_medicine/BM2_ccw_dynamic_rule.py --n-jobs 8
    # smoke test:
    env/run_py.sh scripts/bmc_medicine/BM2_ccw_dynamic_rule.py --quick --n-jobs 2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

log = common.log
phase = common.phase

SCRIPT_VERSION = "1.0.1"
TAG = "BM2"
N_PHASES = 8
EXPECTED_N_STAYS = 38981

RULE_CODES = (16, 14, 8, 6, 1)                 # raw coarse_3 rule actions (70c)
PERFUSION_CODES = (16, 14, 8)
RULE_CODE_LABEL = OrderedDict([
    (16, "P1_shock_hypoperf_f2v2d0"),
    (14, "P2_hypotension_f2v1d0"),
    (8, "P3_hypoperf_f1v1d0"),
    (6, "P6_default_f1v0d0"),
    (1, "P4P5_overload_f0v0d1"),
])
DEFINITIONS = ("relaxed", "strict")
TRUNCATIONS = OrderedDict([("p99", 99.0), ("none", None), ("p999", 99.9)])
PRIMARY_TRUNC = "p99"
SPEC_STRATA = ["all", "hfpef", "hfpef_nonshock", "hfpef_shock", "nohf_nonshock", "hfref"]
DOD = ("hfpef_nonshock_minus_nohf_nonshock", "hfpef_nonshock", "nohf_nonshock")
HF_DUMMY_LEVELS = ("hfpef", "hfref", "hfmref", "hf_ef_unknown")   # reference: no_hf
RULE_COL_CANDIDATES = ("rule_policy_action", "rule_action", "rule_action_merged")
Q_IDX = {"risk_rule": 0, "risk_naive": 1, "risk_nc": 2}


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BM2 clone-censor-weight / IPCW per-protocol estimate of the 70c rule")
    common.add_common_args(p, n_boot_default=200, model_based=False)
    p.add_argument("--n-jobs", type=int, default=int(os.environ.get("NSLOTS") or 8),
                   help="joblib workers for the bootstrap (default $NSLOTS or 8)")
    p.add_argument("--horizons", default="3,6,12", help="adherence horizons in 2-h intervals (comma list)")
    p.add_argument("--primary-horizon", type=int, default=12)
    p.add_argument("--sofa", default=None,
                   help="optional BM3a parquet (stay_id, sofa_total), e.g. "
                        "data/interim/bmc_medicine/BM3a_sofa_preT0.parquet; adds baseline SOFA to the covariates")
    p.add_argument("--min-events", type=int, default=200,
                   help="pool rule categories whose at-risk intervals have fewer events in either class")
    p.add_argument("--min-adherent", type=int, default=50,
                   help="feasibility: fewer fully adherent patients in a stratum -> not estimable")
    p.add_argument("--clip-low", type=float, default=0.01, help="lower clip of fitted adherence probabilities")
    p.add_argument("--lr-C", type=float, default=1e4, help="inverse L2 strength of LogisticRegression")
    p.add_argument("--max-iter", type=int, default=5000)
    p.add_argument("--lr-tol", type=float, default=1e-6,
                   help="lbfgs tolerance (sklearn default 1e-4 leaves fitted p differing by ~1e-2 between starts; "
                        "1e-6 gives ~3e-4, which matters because weights are products over up to 12 intervals)")
    p.add_argument("--rare-level-share", type=float, default=0.01,
                   help="admission types below this patient share are pooled into OTHER")
    p.add_argument("--rule-source", choices=["sidecar", "recompute"], default="sidecar",
                   help="'sidecar' = 70c parquet (spec); 'recompute' = 70c rule re-applied to the raw 45c "
                        "columns with the cumulative fluid balance (use if RULE_SIDECAR_CHECK fails)")
    p.add_argument("--boot-scope", choices=["all", "primary"], default="all",
                   help="bootstrap every configuration (default) or only relaxed H=primary")
    p.add_argument("--no-fb-sensitivity", action="store_true",
                   help="skip the relaxed-H12 sensitivity with cumulative fluid balance as a covariate")
    p.add_argument("--no-warm-start", action="store_true",
                   help="bootstrap fits start from zero instead of the full-sample coefficients")
    p.add_argument("--phenotype-labels", default=None,
                   help="phenotype label parquet for the strata and the shock covariates (stay_id, "
                        "rule_phenotype_2class; hf_subtype optional, else the 45c value); default: config "
                        "data.phenotype_labels. The reported estimates used the rule-based 60d labels: "
                        "--phenotype-labels data/interim/60d_rule_phenotype_v1_mimic.parquet")
    return p.parse_args(argv)


# ═══════════════════════════════════════════════════════════════════════
# Small helpers
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    name: str
    definition: str
    horizon: int
    fb_cov: bool = False
    groups: list = field(default_factory=list)      # [(label, codes tuple, feature idx array)]
    boot: bool = True


def decode_levels(raw: np.ndarray):
    raw = np.asarray(raw, dtype=np.int64)
    return raw // 6, (raw % 6) // 2, raw % 2


def rule_raw_np(map_v, lac_v, spo2_v, fb_v, thr: dict) -> np.ndarray:
    """70c apply_rule_policy (numpy; NaN comparisons are False exactly as in 70c)."""
    map_v, lac_v, spo2_v, fb_v = (np.asarray(x, dtype=np.float64) for x in (map_v, lac_v, spo2_v, fb_v))
    with np.errstate(invalid="ignore"):
        p1 = (map_v < thr["map_hypotension"]) & (lac_v > thr["lactate_hypoperfusion"])
        p2 = (map_v < thr["map_hypotension"]) & ~p1
        p3 = (lac_v > thr["lactate_hypoperfusion"]) & (map_v >= thr["map_hypotension"])
        p4 = ((fb_v > thr["fluid_balance_overload_L"]) & (spo2_v < thr["spo2_hypoxemia"])
              & (map_v >= thr["map_hypotension"]) & (lac_v <= thr["lactate_hypoperfusion"]))
        p5 = ((fb_v > thr["fluid_balance_severe_overload_L"]) & (map_v >= thr["map_hypotension"])
              & (lac_v <= thr["lactate_hypoperfusion"]) & ~p4)
    out = np.full(map_v.shape, 6, dtype=np.int64)
    out[p1] = 16
    out[p2] = 14
    out[p3] = 8
    out[p4] = 1
    out[p5] = 1
    return out


def parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq
        return list(pq.ParquetFile(path).schema_arrow.names)
    except Exception:  # pragma: no cover - pyarrow missing
        return list(pd.read_parquet(path).columns)


def weighted_percentile(values: np.ndarray, counts: np.ndarray, q: float) -> float:
    """Percentile (numpy linear) of ``values`` with integer frequency ``counts``."""
    reps = np.repeat(values, counts.astype(np.int64))
    return float(np.percentile(reps, q)) if reps.size else float("nan")


def pct(x) -> float:
    return 100.0 * float(x) if x is not None and np.isfinite(x) else float("nan")


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def ci_text(est, lo, hi, d: int = 2) -> str:
    return common.fmt_ci(est, lo, hi, d)


def code_str(d: dict, codes=RULE_CODES, digits: int = 1) -> str:
    return ";".join(f"{c}={common.fmt_num(d.get(c, float('nan')), digits)}" for c in codes)


# ═══════════════════════════════════════════════════════════════════════
# Data loading and person-interval construction
# ═══════════════════════════════════════════════════════════════════════
def load_inputs(root: Path, config: dict, args: argparse.Namespace) -> dict:
    data = config["data"]
    n_bins = int(config["grid"]["n_bins"])
    st_path = root / data["state_tensors"]
    cols = parquet_columns(st_path)
    z_cols = [c for c in cols if c.endswith("_z")]
    mask_cols = [c for c in cols if c.endswith("_mask")]
    extra = [c for c in ("age", "gender", "admission_type", "hf_subtype", "map", "lactate", "spo2",
                         "cumulative_fluid_balance_L", "fluid_balance_L") if c in cols]
    t0 = time.time()
    st = pd.read_parquet(st_path, columns=["stay_id", "bin"] + z_cols + mask_cols + extra)
    st = st[(st["bin"] >= 0) & (st["bin"] < n_bins)]
    z_cols = [c for c in z_cols if pd.api.types.is_numeric_dtype(st[c])]
    mask_cols = [c for c in mask_cols if pd.api.types.is_numeric_dtype(st[c])]
    log(f"  45c: {len(st):,} rows, {st['stay_id'].nunique():,} stays; {len(z_cols)} *_z, "
        f"{len(mask_cols)} *_mask  [{time.time() - t0:.1f}s]")
    if st.duplicated(["stay_id", "bin"]).any():
        raise ValueError("45c has duplicate (stay_id, bin) rows")

    out = pd.read_parquet(root / data["outcomes"], columns=["stay_id", "mortality_28d", "absorbing_state_bin"])
    if out["stay_id"].duplicated().any():
        raise ValueError("45d has duplicate stay_id rows")

    prop_path = root / data["propensities_coarse3"]
    if prop_path.is_file():
        pcols = parquet_columns(prop_path)
        acts = pd.read_parquet(prop_path, columns=["stay_id", "bin", "action_idx"]
                               + (["action_idx_merged"] if "action_idx_merged" in pcols else []))
        act_src = prop_path.name
    else:
        alt = root / "data/interim/45b_actions_2h_v1_mimic.parquet"
        log(f"WARNING: {prop_path} not found; observed actions from {alt.name}")
        acts = pd.read_parquet(alt, columns=["stay_id", "bin", "action_idx"])
        act_src = alt.name
    if acts.duplicated(["stay_id", "bin"]).any():
        raise ValueError(f"{act_src} has duplicate (stay_id, bin) rows")

    rule_path = root / data.get("rule_policy_actions", "data/interim/70c_rule_policy_actions_v1_mimic.parquet")
    rcols = parquet_columns(rule_path)
    rule_col = next((c for c in RULE_COL_CANDIDATES if c in rcols), None)
    if rule_col is None:
        raise ValueError(f"{rule_path.name} has none of {RULE_COL_CANDIDATES}")
    rules = pd.read_parquet(rule_path, columns=["stay_id", "bin", rule_col]).rename(columns={rule_col: "rule_raw"})
    if rules.duplicated(["stay_id", "bin"]).any():
        raise ValueError(f"{rule_path.name} has duplicate (stay_id, bin) rows")

    ph_path = root / (args.phenotype_labels or data["phenotype_labels"])
    if not ph_path.is_file():
        raise FileNotFoundError(f"phenotype labels not found: {ph_path}")
    phc = parquet_columns(ph_path)
    pheno_src = ph_path.name + ("" if "hf_subtype" in phc else " (hf_subtype from 45c)")
    log(f"  phenotype labels: {pheno_src}")
    pheno = pd.read_parquet(ph_path, columns=["stay_id"] + [c for c in ("hf_subtype", "rule_phenotype_2class")
                                                             if c in phc]).drop_duplicates("stay_id")

    merge_map = {}
    mm_path = root / data["merge_map"]
    if mm_path.is_file():
        merge_map = {int(k): int(v) for k, v in (common.load_yaml(mm_path).get("merge_map") or {}).items()}
    thr = dict(common.DEFAULT_THRESHOLDS)
    rc_path = root / data.get("rule_policy", common.RULE_CFG_REL)
    if rc_path.is_file():
        for k, v in (common.load_yaml(rc_path).get("thresholds") or {}).items():
            if k in thr:
                if not math.isclose(float(v), thr[k]):
                    log(f"WARNING: 70c threshold {k}={v} differs from the spec value {thr[k]}; using config value")
                thr[k] = float(v)

    sofa = None
    sofa_meta: dict = {}
    if args.sofa:
        sp = common.resolve_input_path(root, args.sofa, fallback_roots=(), required=True)
        sofa = pd.read_parquet(sp, columns=["stay_id", "sofa_total"]).drop_duplicates("stay_id")
        log(f"  SOFA: {sp} ({len(sofa):,} stays)")
        mp = sp.with_suffix(".meta.json")
        if mp.is_file():
            try:
                sofa_meta = json.loads(mp.read_text(encoding="utf-8"))
            except Exception:  # pragma: no cover
                sofa_meta = {}
        sofa_meta["_partial"] = bool(sofa_meta.get("quick", False)) or any(
            bool(v) for v in (sofa_meta.get("raw_truncated") or {}).values())
        sofa_meta["_has_meta"] = mp.is_file()
        if sofa_meta["_partial"]:
            log("WARNING: the BM3a SOFA file comes from a --quick / --max-chunks run (subsample or truncated raw "
                "passes); rerun BM3a in full before interpreting BM2 --sofa")
        elif not sofa_meta["_has_meta"]:
            log(f"WARNING: no {mp.name} next to the SOFA file; cannot verify that BM3a ran in full")
    log(f"  observed actions: {act_src}; rule actions: {rule_path.name}:{rule_col}; merge_map={merge_map}")
    log(f"  thresholds: {thr}")
    return {"state": st, "outcomes": out, "actions": acts, "rules": rules, "pheno": pheno, "sofa": sofa,
            "sofa_meta": sofa_meta,
            "z_cols": z_cols, "mask_cols": mask_cols, "merge_map": merge_map, "thresholds": thr,
            "n_bins": n_bins, "rule_source_file": f"{rule_path.name}:{rule_col}", "action_source_file": act_src,
            "pheno_source_file": pheno_src}


def build_cohort(inp: dict, args: argparse.Namespace) -> dict:
    """Person-interval arrays (active intervals with bin < max H) and patient-level frame."""
    st, out = inp["state"], inp["outcomes"]
    n_bins = inp["n_bins"]
    max_h = max(args.horizon_list)
    ids = np.sort(st["stay_id"].unique())
    if args.quick:
        ids = common.subsample_ids(ids, common.QUICK_N_STAYS, args.seed)
        log(f"  QUICK: cohort subsample n={len(ids):,}")
    no_out = np.setdiff1d(ids, out["stay_id"].to_numpy())
    if no_out.size:
        log(f"WARNING: {no_out.size} stays without a 45d outcome row are dropped")
        ids = np.setdiff1d(ids, no_out)
    n = len(ids)
    id_set = set(ids.tolist())

    # ---- patient level ---------------------------------------------------
    first = st[st["stay_id"].isin(id_set)].sort_values(["stay_id", "bin"]).drop_duplicates("stay_id")
    pat = pd.DataFrame({"stay_id": ids})
    pat = pat.merge(out, on="stay_id", how="left", validate="one_to_one")
    stat_cols = [c for c in ("age", "gender", "admission_type", "hf_subtype") if c in first.columns]
    pat = pat.merge(first[["stay_id"] + stat_cols].rename(columns={"hf_subtype": "hf_subtype_45c"}),
                    on="stay_id", how="left", validate="one_to_one")
    pat = pat.merge(inp["pheno"], on="stay_id", how="left", validate="one_to_one")
    if "hf_subtype" not in pat.columns:
        pat["hf_subtype"] = pat.get("hf_subtype_45c")
    elif "hf_subtype_45c" in pat.columns:
        miss = pat["hf_subtype"].isna()
        if miss.any():
            log(f"WARNING: {int(miss.sum())} stays without phenotype hf_subtype; using the 45c value")
            pat.loc[miss, "hf_subtype"] = pat.loc[miss, "hf_subtype_45c"]
    if "rule_phenotype_2class" not in pat.columns:
        raise ValueError("phenotype labels lack rule_phenotype_2class")
    n_miss_sh = int(pat["rule_phenotype_2class"].isna().sum())
    if n_miss_sh:
        log(f"WARNING: {n_miss_sh} stays without rule_phenotype_2class (treated as non_shock in the covariates)")
    if inp["sofa"] is not None:
        pat = pat.merge(inp["sofa"], on="stay_id", how="left", validate="one_to_one")
    pat["mortality_28d"] = pd.to_numeric(pat["mortality_28d"], errors="coerce")
    if pat["mortality_28d"].isna().any():
        raise ValueError("mortality_28d missing for cohort stays")

    # ---- intervals ---------------------------------------------------------
    rows = st[st["stay_id"].isin(id_set)].sort_values(["stay_id", "bin"]).reset_index(drop=True)
    ab = rows["stay_id"].map(pat.set_index("stay_id")["absorbing_state_bin"])
    active = ab.isna().to_numpy() | (rows["bin"].to_numpy() <= ab.to_numpy())
    n_active_all = int(active.sum())
    rows = rows[active & (rows["bin"].to_numpy() < max_h)].reset_index(drop=True)
    acts = inp["actions"]
    rows = rows.merge(acts, on=["stay_id", "bin"], how="left", validate="one_to_one")
    rows = rows.merge(inp["rules"], on=["stay_id", "bin"], how="left", validate="one_to_one")
    miss_obs = int(rows["action_idx"].isna().sum())
    miss_rule = int(rows["rule_raw"].isna().sum())
    if miss_obs or miss_rule:
        raise ValueError(f"Observed/rule actions do not cover every active interval: missing observed={miss_obs}, "
                         f"missing rule={miss_rule}. No zero-action fallback is allowed.")
    mm = inp["merge_map"]
    obs_raw = rows["action_idx"].to_numpy().astype(np.int64)
    lut = np.arange(18, dtype=np.int64)
    for k, v in mm.items():
        lut[k] = v
    if "action_idx_merged" in rows.columns:
        obs_merged = rows["action_idx_merged"].to_numpy().astype(np.int64)
        n_inc = int((lut[obs_raw] != obs_merged).sum())
        if n_inc:
            log(f"WARNING: {n_inc} intervals where action_idx_merged != merge_map(action_idx)")
    else:
        obs_merged = lut[obs_raw]
    rule_sidecar = rows["rule_raw"].to_numpy().astype(np.int64)

    # ---- rule sidecar check (70c recomputed from the raw 45c columns) ------
    thr = inp["thresholds"]
    chk = {"status": "NA", "agree_cumfb": float("nan"), "agree_intervalfb": float("nan"), "pairs": {}}
    rule_recomputed = None
    if all(c in rows.columns for c in ("map", "lactate", "spo2", "cumulative_fluid_balance_L")):
        rule_recomputed = rule_raw_np(rows["map"], rows["lactate"], rows["spo2"],
                                      rows["cumulative_fluid_balance_L"], thr)
        chk["agree_cumfb"] = float((rule_recomputed == rule_sidecar).mean())
        if "fluid_balance_L" in rows.columns:
            r_int = rule_raw_np(rows["map"], rows["lactate"], rows["spo2"], rows["fluid_balance_L"], thr)
            chk["agree_intervalfb"] = float((r_int == rule_sidecar).mean())
        dis = rule_recomputed != rule_sidecar
        if dis.any():
            vc = pd.Series(list(zip(rule_sidecar[dis], rule_recomputed[dis]))).value_counts().head(8)
            chk["pairs"] = {f"{a}->{b}": int(c) for (a, b), c in vc.items()}
        chk["status"] = "PASS" if chk["agree_cumfb"] == 1.0 else "FAIL"
    log(f"RULE_SIDECAR_CHECK: {chk['status']} (70c sidecar vs rule recomputed from raw 45c with cumulative FB: "
        f"{common.fmt_num(pct(chk['agree_cumfb']), 3)}%; with interval FB: "
        f"{common.fmt_num(pct(chk['agree_intervalfb']), 3)}%; disagreements sidecar->recomputed {chk['pairs']})")
    if chk["status"] == "FAIL":
        log("WARNING: the 70c sidecar is not the corrected-70c output (if the interval-FB agreement is 100% it is "
            "the pre-v1.0.1 file). Regenerate it with the corrected 70c or rerun with --rule-source recompute.")
    if args.rule_source == "recompute":
        if rule_recomputed is None:
            raise ValueError("--rule-source recompute needs raw map/lactate/spo2/cumulative_fluid_balance_L in 45c")
        rule_raw = rule_recomputed
        log("  rule actions: RECOMPUTED from the raw 45c columns (--rule-source recompute)")
    else:
        rule_raw = rule_sidecar
    bad = sorted(set(np.unique(rule_raw).tolist()) - set(RULE_CODES))
    if bad:
        raise ValueError(f"unexpected rule action codes {bad}; expected {RULE_CODES}")
    rule_merged = lut[rule_raw]

    # ---- adherence -------------------------------------------------------------
    of, ov, od = decode_levels(obs_raw)
    rf, rv, rd = decode_levels(rule_raw)
    mf, mv, md = decode_levels(obs_merged)
    rmf, rmv, rmd = decode_levels(rule_merged)
    relaxed_fail = {"vaso": (ov > 0) != (rv > 0), "diuretic": (od > 0) != (rd > 0), "fluid": np.abs(of - rf) > 1}
    strict_fail = {"vaso": mv != rmv, "diuretic": md != rmd, "fluid": mf != rmf}
    adh = {"relaxed": ~(relaxed_fail["vaso"] | relaxed_fail["diuretic"] | relaxed_fail["fluid"]),
           "strict": obs_merged == rule_merged}
    n_strict_not_relaxed = int((adh["strict"] & ~adh["relaxed"]).sum())

    # previous interval's observed raw action (0 at bin 0 / no row)
    prev = acts[["stay_id", "bin", "action_idx"]].rename(columns={"action_idx": "prev_raw"}).copy()
    prev["bin"] = prev["bin"] + 1
    prev_raw = rows[["stay_id", "bin"]].merge(prev, on=["stay_id", "bin"], how="left")["prev_raw"]
    prev_raw = prev_raw.fillna(0).to_numpy().astype(np.int64)

    row_pat = pd.Index(ids).get_indexer(rows["stay_id"].to_numpy()).astype(np.int64)
    if (row_pat < 0).any():
        raise RuntimeError("interval rows reference stays outside the cohort")
    log(f"  cohort: {n:,} stays; active intervals {n_active_all:,} (bin<{n_bins}); "
        f"used (bin<{max_h}) {len(rows):,}; deaths {int(pat['mortality_28d'].sum()):,} "
        f"({100 * pat['mortality_28d'].mean():.2f}%)")
    return {"ids": ids, "pat": pat, "rows": rows, "row_pat": row_pat, "row_bin": rows["bin"].to_numpy().astype(np.int64),
            "obs_raw": obs_raw, "obs_merged": obs_merged, "rule_raw": rule_raw, "adh": adh,
            "fail": {"relaxed": relaxed_fail, "strict": strict_fail}, "prev_raw": prev_raw,
            "n_active_all": n_active_all, "rule_check": chk, "n_strict_not_relaxed": n_strict_not_relaxed}


def build_design(coh: dict, inp: dict, args: argparse.Namespace) -> dict:
    """Row-level covariate matrix (float64), feature names and index sets."""
    rows, pat, row_pat = coh["rows"], coh["pat"], coh["row_pat"]
    n_rows = len(rows)
    max_h = max(args.horizon_list)
    blocks, names, cont = [], [], []

    def add(mat, nm, is_cont):
        mat = np.asarray(mat, dtype=np.float64)
        if mat.ndim == 1:
            mat = mat[:, None]
            nm = [nm]
        blocks.append(mat)
        names.extend(nm)
        cont.extend([is_cont] * len(nm))

    rb = coh["row_bin"]
    add(np.stack([(rb == b) for b in range(1, max_h)], axis=1) if max_h > 1 else np.zeros((n_rows, 0)),
        [f"bin_{b}" for b in range(1, max_h)], False)
    zc = inp["z_cols"]
    add(np.nan_to_num(rows[zc].to_numpy(dtype=np.float64), nan=0.0), list(zc), True)
    mc = inp["mask_cols"]
    add(np.nan_to_num(rows[mc].to_numpy(dtype=np.float64), nan=0.0), list(mc), False)
    pf, pv, pd_ = decode_levels(coh["prev_raw"])
    add(np.stack([pf == 1, pf == 2, pv == 1, pv == 2, pd_ == 1], axis=1),
        ["prev_fluid_1", "prev_fluid_2", "prev_vaso_1", "prev_vaso_2", "prev_diuretic"], False)

    # patient-level covariates
    P = OrderedDict()
    age = pd.to_numeric(pat.get("age"), errors="coerce") if "age" in pat else pd.Series(np.nan, index=pat.index)
    P["age"] = (age.fillna(age.median()).to_numpy(), True)
    g = pat.get("gender", pd.Series("", index=pat.index)).astype(str).str.upper().str[:1]
    P["female"] = ((g == "F").to_numpy(), False)
    adm = pat.get("admission_type", pd.Series("UNKNOWN", index=pat.index)).fillna("UNKNOWN").astype(str)
    share = adm.value_counts(normalize=True)
    keep = [lv for lv in share.index if share[lv] >= args.rare_level_share]
    adm = adm.where(adm.isin(keep), "OTHER")
    levels = adm.value_counts().index.tolist()
    ref = levels[0]
    for lv in levels[1:]:
        P[f"adm_{lv}"] = ((adm == lv).to_numpy(), False)
    hf = pat["hf_subtype"].astype(str)
    sh = (pat["rule_phenotype_2class"].astype(str) == "shock_predominant").to_numpy()
    for lv in HF_DUMMY_LEVELS:
        P[f"hf_{lv}"] = ((hf == lv).to_numpy(), False)
    P["shock_predominant"] = (sh, False)
    for lv in HF_DUMMY_LEVELS:
        P[f"hf_{lv}_x_shock"] = ((hf == lv).to_numpy() & sh, False)
    sofa_info = None
    if "sofa_total" in pat.columns:
        s = pd.to_numeric(pat["sofa_total"], errors="coerce")
        nm = int(s.isna().sum())
        P["sofa_total"] = (s.fillna(s.median()).to_numpy(), True)
        if nm:
            P["sofa_missing"] = (s.isna().to_numpy(), False)
        sofa_info = {"n_missing": nm, "median": float(s.median())}
    for k, (v, is_c) in P.items():
        add(np.asarray(v, dtype=np.float64)[row_pat], k, is_c)

    rc = coh["rule_raw"]
    rulecat_idx = {}
    for c in RULE_CODES:
        rulecat_idx[c] = len(names)
        add(rc == c, f"rulecat_{c}", False)
    fb_idx = None
    if "cumulative_fluid_balance_L" in rows.columns:
        fb_idx = len(names)
        add(np.nan_to_num(rows["cumulative_fluid_balance_L"].to_numpy(dtype=np.float64), nan=0.0),
            "cumulative_fluid_balance_L", True)

    X = np.concatenate(blocks, axis=1)
    cont = np.asarray(cont, dtype=bool)
    mu = X[:, cont].mean(axis=0)
    sd = X[:, cont].std(axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-12)] = 1.0
    X[:, cont] = (X[:, cont] - mu) / sd
    X = np.ascontiguousarray(X)
    special = set(rulecat_idx.values()) | ({fb_idx} if fb_idx is not None else set())
    base_idx = np.array([j for j in range(len(names)) if j not in special], dtype=np.int64)
    log(f"  design: {X.shape[0]:,} person-intervals x {X.shape[1]} columns "
        f"({X.nbytes / 1e6:.0f} MB); base covariates {len(base_idx)}; admission reference '{ref}'; "
        f"levels {levels}")
    return {"X": X, "names": names, "base_idx": base_idx, "rulecat_idx": rulecat_idx, "fb_idx": fb_idx,
            "adm_levels": levels, "adm_ref": ref, "sofa_info": sofa_info}


# ═══════════════════════════════════════════════════════════════════════
# Censoring structure and models
# ═══════════════════════════════════════════════════════════════════════
def censoring_structure(row_pat, row_bin, adh_row, horizon: int, n: int):
    """at-risk rows, fully adherent patients and first non-adherent bin for one (definition, H)."""
    required = row_bin < horizon
    nonadh = required & ~adh_row
    cs = np.cumsum(nonadh.astype(np.int64))
    # exclusive cumulative count within patient (rows sorted by patient then bin)
    excl = cs - nonadh
    if len(row_pat):
        start = np.r_[0, np.flatnonzero(np.diff(row_pat)) + 1]
        seg_id = np.repeat(np.arange(len(start)), np.diff(np.r_[start, len(row_pat)]))
        prior = excl - excl[start][seg_id]
    else:
        prior = excl
    at_risk = required & (prior == 0)
    n_nonadh = np.bincount(row_pat, weights=nonadh.astype(np.float64), minlength=n)
    full = n_nonadh == 0
    return at_risk, full


def first_nonadherent_bin(row_pat, row_bin, adh_row, n: int, n_bins: int) -> np.ndarray:
    fb = np.full(n, n_bins + 100, dtype=np.int64)
    sel = ~adh_row
    np.minimum.at(fb, row_pat[sel], row_bin[sel])
    return fb


def group_categories(counts_by_code: dict, min_events: int) -> list[tuple[str, tuple]]:
    """Model groups (label, codes): separate per rule code; pool 16/14/8 if any has < min_events
    in either class; then pool remaining small groups together and, if still small, into the largest."""
    present = [c for c in RULE_CODES if counts_by_code.get(c, (0, 0))[0] + counts_by_code.get(c, (0, 0))[1] > 0]

    def mincls(codes):
        return min(sum(counts_by_code[c][0] for c in codes), sum(counts_by_code[c][1] for c in codes))

    def total(codes):
        return sum(counts_by_code[c][0] + counts_by_code[c][1] for c in codes)

    groups = [(c,) for c in present]
    perf = tuple(c for c in PERFUSION_CODES if c in present)
    if perf and any(mincls((c,)) < min_events for c in perf):
        groups = [perf] + [g for g in groups if g[0] not in perf]
    small = [g for g in groups if mincls(g) < min_events]
    if small and len(groups) > 1:
        big = [g for g in groups if g not in small]
        pooled = tuple(c for c in RULE_CODES if any(c in g for g in small))
        if mincls(pooled) < min_events and big:
            largest = max(big, key=total)
            big.remove(largest)
            pooled = tuple(c for c in RULE_CODES if c in pooled or c in largest)
        groups = big + [pooled]
        groups = sorted(groups, key=lambda g: RULE_CODES.index(g[0]))
    return [("+".join(map(str, g)), g) for g in groups]


def fit_logistic(Xg, yg, swg, C: float, max_iter: int, warm=None, tol: float = 1e-6) -> dict:
    """Weighted logistic regression for P(adherent); constant fit when one class is absent."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    n1 = float(swg[yg].sum())
    n0 = float(swg[~yg].sum())
    if n1 <= 0 or n0 <= 0:
        return {"kind": "constant", "p": 1.0 if n0 <= 0 else 0.0, "converged": True, "n_iter": 0}
    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter, tol=tol, warm_start=warm is not None)
    if warm is not None and warm.get("kind") == "logit":
        clf.coef_ = np.asarray(warm["coef"], dtype=np.float64).reshape(1, -1).copy()
        clf.intercept_ = np.array([float(warm["intercept"])])
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(Xg, yg.astype(np.int64), sample_weight=swg)
    conv = not any(issubclass(w.category, ConvergenceWarning) for w in wl)
    return {"kind": "logit", "coef": clf.coef_.ravel().copy(), "intercept": float(clf.intercept_[0]),
            "converged": conv, "n_iter": int(np.max(clf.n_iter_))}


def predict_fit(fit: dict, Xg) -> np.ndarray:
    if fit["kind"] == "constant":
        return np.full(Xg.shape[0], fit["p"], dtype=np.float64)
    return sigmoid(Xg @ fit["coef"] + fit["intercept"])


def estimate(ctx: dict, counts: np.ndarray, cfg_ids, diag: bool = False, warm: bool = True) -> dict:
    """Point (counts = 1) or bootstrap (counts = resample frequencies) estimates.

    Returns draw (n_cfg, n_trunc, S, 3) [risk_rule, risk_naive, risk_nc] (fractions); with diag=True
    also fits, p_hat diagnostics, weights per patient and per-stratum weight summaries.
    """
    X, row_pat, row_code = ctx["X"], ctx["row_pat"], ctx["row_code"]
    Y, M = ctx["Y"], ctx["strata_M"]
    n = len(Y)
    S = M.shape[0]
    counts = np.asarray(counts, dtype=np.float64)
    cY = counts * Y
    n_s = M @ counts
    with np.errstate(invalid="ignore", divide="ignore"):
        risk_nc = (M @ cY) / n_s
    truncs = list(ctx["truncations"].items())
    draw = np.full((len(cfg_ids), len(truncs), S, 3), np.nan)
    info = {"n_nonconverged": 0, "n_constant": 0, "fits": {}, "per_cfg": {}}
    rows_pos = counts[row_pat] > 0
    for k, ci in enumerate(cfg_ids):
        cfg = ctx["configs"][ci]
        adh = ctx["adh"][cfg["definition"]]
        at_risk = ctx["at_risk"][ci]
        full = ctx["full"][ci]
        ar_rows = np.flatnonzero(at_risk & rows_pos)
        p_full = np.full(len(row_pat), np.nan)
        fits = []
        for label, codes, feat in cfg["groups"]:
            sel = ar_rows[np.isin(row_code[ar_rows], codes)]
            if sel.size == 0:
                fits.append({"label": label, "kind": "empty", "n_rows": 0})
                continue
            Xg = X[np.ix_(sel, feat)]
            yg = adh[sel]
            swg = counts[row_pat[sel]]
            w0 = ctx["warm"].get((ci, label)) if warm else None
            fit = fit_logistic(Xg, yg, swg, ctx["C"], ctx["max_iter"], warm=w0, tol=ctx.get("tol", 1e-6))
            info["n_nonconverged"] += int(not fit["converged"])
            info["n_constant"] += int(fit["kind"] == "constant")
            p = predict_fit(fit, Xg)
            p_full[sel] = p
            if diag:
                fit.update(label=label, codes=codes, n_rows=int(sel.size), n_adherent=int(yg.sum()),
                           n_nonadherent=int((~yg).sum()), mean_p=float(p.mean()),
                           share_p_lt005=float((p < 0.05).mean()),
                           share_clipped=float((p < ctx["clip_low"]).mean()))
                try:
                    from sklearn.metrics import roc_auc_score
                    fit["auc"] = float(roc_auc_score(yg, p)) if 0 < yg.sum() < yg.size else float("nan")
                except Exception:
                    fit["auc"] = float("nan")
            fits.append(fit)
        pc = np.clip(p_full, ctx["clip_low"], 1.0)
        full_pos = full & (counts > 0)
        R = ar_rows[full_pos[row_pat[ar_rows]]]
        if np.isnan(pc[R]).any():
            raise RuntimeError(f"{cfg['name']}: missing fitted probabilities on weighted rows")
        logw = np.bincount(row_pat[R], weights=-np.log(pc[R]), minlength=n)
        w = np.where(full, np.exp(logw), 0.0)
        cA = counts * full
        with np.errstate(invalid="ignore", divide="ignore"):
            risk_naive = (M @ (cY * full)) / (M @ cA)
        caps = {}
        for ti, (tname, q) in enumerate(truncs):
            if q is None:
                wt = w
                caps[tname] = float("nan")
            else:
                cap = weighted_percentile(w[full_pos], counts[full_pos], q)
                caps[tname] = cap
                wt = np.minimum(w, cap)
            with np.errstate(invalid="ignore", divide="ignore"):
                draw[k, ti, :, 0] = (M @ (cA * wt * Y)) / (M @ (cA * wt))
            draw[k, ti, :, 1] = risk_naive
            draw[k, ti, :, 2] = risk_nc
        if diag:
            ar_all = np.flatnonzero(at_risk)
            ph = p_full[ar_all]
            wrows = np.flatnonzero(at_risk & full[row_pat])
            wsum = {}
            for ti, (tname, q) in enumerate(truncs):
                wt = w if q is None else np.minimum(w, caps[tname])
                for si in range(S):
                    m = M[si].astype(bool) & full
                    ws = wt[m]
                    wu = w[m]
                    wsum[(tname, si)] = {
                        "n_adherent": int(m.sum()),
                        "ess": float(ws.sum() ** 2 / (ws ** 2).sum()) if ws.size and (ws ** 2).sum() > 0 else float("nan"),
                        "w_mean": float(ws.mean()) if ws.size else float("nan"),
                        "w_p50": float(np.percentile(ws, 50)) if ws.size else float("nan"),
                        "w_p99": float(np.percentile(ws, 99)) if ws.size else float("nan"),
                        "w_max": float(ws.max()) if ws.size else float("nan"),
                        "w_untrunc_mean": float(wu.mean()) if wu.size else float("nan"),
                        "w_untrunc_p50": float(np.percentile(wu, 50)) if wu.size else float("nan"),
                        "w_untrunc_p99": float(np.percentile(wu, 99)) if wu.size else float("nan"),
                        "w_untrunc_max": float(wu.max()) if wu.size else float("nan"),
                        "share_truncated": float((wu > caps[tname]).mean()) if (wu.size and q is not None) else 0.0,
                    }
            info["per_cfg"][ci] = {
                "fits": fits, "caps": caps, "w": w, "wsum": wsum,
                "n_at_risk_rows": int(ar_all.size),
                "share_p_lt005_atrisk": float((ph < 0.05).mean()) if ph.size else float("nan"),
                "share_p_clipped_atrisk": float((ph < ctx["clip_low"]).mean()) if ph.size else float("nan"),
                "share_p_lt005_weighted_rows": float((p_full[wrows] < 0.05).mean()) if wrows.size else float("nan"),
                "min_p_atrisk": float(np.nanmin(ph)) if ph.size else float("nan"),
            }
    return {"draw": draw, **info}


# ═══════════════════════════════════════════════════════════════════════
# Bootstrap (joblib)
# ═══════════════════════════════════════════════════════════════════════
def _boot_task(ctx_src, b: int, seed: int, cfg_ids, warm: bool) -> dict:
    """One bootstrap replicate (runs in a joblib worker; context memory-mapped from disk)."""
    import joblib
    from threadpoolctl import threadpool_limits
    ctx = joblib.load(ctx_src, mmap_mode="r") if isinstance(ctx_src, (str, Path)) else ctx_src
    n = len(ctx["Y"])
    rng = np.random.default_rng([int(seed), int(b)])
    counts = np.bincount(rng.integers(0, n, size=n), minlength=n).astype(np.float64)
    with threadpool_limits(limits=1):
        res = estimate(ctx, counts, cfg_ids, diag=False, warm=warm)
    return {"b": b, "draw": res["draw"], "n_nonconverged": res["n_nonconverged"], "n_constant": res["n_constant"]}


def run_bootstrap(ctx: dict, cfg_ids, args, ctx_path: Path) -> tuple[np.ndarray, dict]:
    n_boot = int(args.n_boot)
    S = ctx["strata_M"].shape[0]
    draws = np.full((n_boot, len(cfg_ids), len(ctx["truncations"]), S, 3), np.nan)
    stats = {"n_nonconverged": 0, "n_constant": 0, "n_done": 0}
    if n_boot <= 0:
        return draws, stats
    warm = not args.no_warm_start
    n_jobs = max(1, int(args.n_jobs))
    tick = max(1, n_boot // 10)

    def consume(gen):
        for r in tqdm(gen, total=n_boot, desc="  CCW bootstrap (refit)", mininterval=10):
            draws[r["b"]] = r["draw"]
            stats["n_nonconverged"] += r["n_nonconverged"]
            stats["n_constant"] += r["n_constant"]
            stats["n_done"] += 1
            if stats["n_done"] % tick == 0:
                log(f"    bootstrap {stats['n_done']}/{n_boot} done")

    if n_jobs == 1:
        consume(_boot_task(ctx, b, args.seed, cfg_ids, warm) for b in range(n_boot))
        return draws, stats
    import joblib
    from joblib import Parallel, delayed
    joblib.dump(ctx, ctx_path)
    log(f"  bootstrap context -> {ctx_path} ({ctx_path.stat().st_size / 1e6:.0f} MB, memory-mapped by workers)")
    tasks = [delayed(_boot_task)(str(ctx_path), b, args.seed, cfg_ids, warm) for b in range(n_boot)]
    par = None
    for mode in ("generator_unordered", "generator"):
        try:
            par = Parallel(n_jobs=n_jobs, return_as=mode, batch_size=1)
            break
        except (TypeError, ValueError):
            par = None
    if par is not None:
        consume(par(tasks))
    else:  # old joblib: chunked list mode
        par = Parallel(n_jobs=n_jobs, batch_size=1)
        chunk = n_jobs * 4

        def chunked():
            for i in range(0, n_boot, chunk):
                for r in par(tasks[i:i + chunk]):
                    yield r
        consume(chunked())
    return draws, stats


# ═══════════════════════════════════════════════════════════════════════
# Descriptive adherence tables
# ═══════════════════════════════════════════════════════════════════════
def adherence_by_category(coh, strata, ctx, primary_by_def) -> pd.DataFrame:
    rows_out = []
    row_pat, rule = coh["row_pat"], coh["rule_raw"]
    req = coh["row_bin"] < ctx["primary_horizon"]
    for d in DEFINITIONS:
        adh = coh["adh"][d]
        fail = coh["fail"][d]
        ar = ctx["at_risk"][primary_by_def[d]]
        for sname, smask in strata.items():
            in_s = smask[row_pat] & req
            for code in ("all",) + RULE_CODES:
                m = in_s if code == "all" else in_s & (rule == code)
                mr = m & ar
                nint = int(m.sum())
                rows_out.append({
                    "definition": d, "stratum": sname, "rule_code": code,
                    "rule_label": "all" if code == "all" else RULE_CODE_LABEL[code],
                    "n_intervals": nint,
                    "share_of_stratum_intervals_pct": pct(nint / max(int(in_s.sum()), 1)),
                    "adherent_pct": pct(adh[m].mean()) if nint else float("nan"),
                    "fail_vaso_pct": pct(fail["vaso"][m].mean()) if nint else float("nan"),
                    "fail_diuretic_pct": pct(fail["diuretic"][m].mean()) if nint else float("nan"),
                    "fail_fluid_pct": pct(fail["fluid"][m].mean()) if nint else float("nan"),
                    "n_at_risk_primaryH": int(mr.sum()),
                    "adherent_at_risk_primaryH_pct": pct(adh[mr].mean()) if mr.any() else float("nan"),
                })
    return pd.DataFrame(rows_out)


def adherence_survival(coh, strata, n: int, n_bins: int) -> pd.DataFrame:
    out = []
    for d in DEFINITIONS:
        fnb = first_nonadherent_bin(coh["row_pat"], coh["row_bin"], coh["adh"][d], n, n_bins)
        for sname, smask in strata.items():
            ns = int(smask.sum())
            for t in range(n_bins):
                out.append({"definition": d, "stratum": sname, "bin": t, "n": ns,
                            "still_adherent_pct": pct((fnb[smask] > t).mean()) if ns else float("nan")})
    return pd.DataFrame(out)


def balance_table(coh, ctx, w, full) -> pd.DataFrame:
    """Baseline SMDs: whole cohort vs adherent (unweighted, IPCW-weighted)."""
    pat, rows = coh["pat"], coh["rows"]
    b0 = rows[rows["bin"] == 0].set_index("stay_id")
    covs = OrderedDict()
    covs["age"] = pd.to_numeric(pat.get("age"), errors="coerce").to_numpy(dtype=float)
    covs["female"] = (pat.get("gender", pd.Series("", index=pat.index)).astype(str).str.upper().str[:1] == "F"
                      ).to_numpy(dtype=float)
    hf = pat["hf_subtype"].astype(str)
    covs["hfpef"] = (hf == "hfpef").to_numpy(dtype=float)
    covs["no_hf"] = (hf == "no_hf").to_numpy(dtype=float)
    covs["shock_predominant"] = (pat["rule_phenotype_2class"].astype(str) == "shock_predominant").to_numpy(dtype=float)
    for c in ("map_z", "lactate_z", "spo2_z", "cumulative_fluid_balance_L"):
        if c in b0.columns:
            covs[f"{c}_bin0"] = pat["stay_id"].map(b0[c]).to_numpy(dtype=float)
    if "sofa_total" in pat.columns:
        covs["sofa_total"] = pd.to_numeric(pat["sofa_total"], errors="coerce").to_numpy(dtype=float)
    out = []
    for k, v in covs.items():
        ok = np.isfinite(v)
        m_all = float(v[ok].mean()) if ok.any() else float("nan")
        sd_all = float(v[ok].std()) if ok.any() else float("nan")
        a = ok & full
        m_un = float(v[a].mean()) if a.any() else float("nan")
        m_w = float(np.average(v[a], weights=w[a])) if a.any() and w[a].sum() > 0 else float("nan")
        sd = sd_all if sd_all and np.isfinite(sd_all) and sd_all > 0 else float("nan")
        out.append({"covariate": k, "mean_all": m_all, "mean_adherent_unweighted": m_un,
                    "mean_adherent_weighted": m_w, "smd_unweighted": (m_un - m_all) / sd,
                    "smd_weighted": (m_w - m_all) / sd})
    return pd.DataFrame(out)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main(argv=None) -> int:
    t_start = time.time()
    args = common.apply_quick(parse_args(argv))
    args.horizon_list = sorted({int(h) for h in str(args.horizons).split(",") if h.strip()} | {args.primary_horizon},
                               reverse=True)
    sofa_run = bool(args.sofa)
    tag = "BM2_SOFA" if sofa_run else TAG
    sfx = "_sofa" if sofa_run else ""
    print("=" * 72, flush=True)
    print(f"BM2 — Clone-censor-weight / IPCW per-protocol estimate of the 70c rule (v{SCRIPT_VERSION})", flush=True)
    print("=" * 72, flush=True)

    # ── 1. Inputs ───────────────────────────────────────────────────────
    t0 = phase(1, N_PHASES, "Load 45c / 45d / 70b / 70c / phenotype labels (no torch, no outcome model)")
    root = common.find_root(__file__)
    paths = common.setup_paths(root)
    log(f"  PROJECT_ROOT={root}")
    config = common.load_yaml(root / common.CONFIG_REL)
    inp = load_inputs(root, config, args)
    n_bins = inp["n_bins"]
    if max(args.horizon_list) > n_bins:
        raise ValueError(f"horizon {max(args.horizon_list)} > n_bins {n_bins}")
    log(f"  [{time.time() - t0:.1f}s]")

    # ── 2. Cohort, active intervals, adherence ──────────────────────────
    t0 = phase(2, N_PHASES, "Cohort, active intervals, rule sidecar check, adherence (strict / relaxed)")
    coh = build_cohort(inp, args)
    pat = coh["pat"]
    n = len(pat)
    strata = common.standard_strata(pat)
    strata_names = list(strata)
    M = np.stack([strata[s] for s in strata_names]).astype(np.float64)
    for d in DEFINITIONS:
        log(f"  share of used intervals adherent ({d}): {pct(coh['adh'][d].mean()):.2f}%")
    log(f"  intervals strict-adherent but relaxed-non-adherent (merged-cell artefact): {coh['n_strict_not_relaxed']}")
    log(f"  [{time.time() - t0:.1f}s]")

    # ── 3. Design matrix ────────────────────────────────────────────────
    t0 = phase(3, N_PHASES, "Censoring-model covariates (standardised design matrix)")
    des = build_design(coh, inp, args)
    log(f"  [{time.time() - t0:.1f}s]")

    # ── 4. Configurations, censoring structure, model groups ──────────────
    t0 = phase(4, N_PHASES, "Configurations, artificial censoring and censoring-model groups")
    prim_h = args.primary_horizon
    configs: list[Config] = []
    for d in DEFINITIONS:
        for h in args.horizon_list:
            configs.append(Config(name=f"{d}_H{h}", definition=d, horizon=h))
    if not args.no_fb_sensitivity and des["fb_idx"] is not None:
        configs.append(Config(name=f"relaxed_H{prim_h}_fbcov", definition="relaxed", horizon=prim_h, fb_cov=True))
    primary_name = f"relaxed_H{prim_h}"
    cfg_index = {c.name: i for i, c in enumerate(configs)}
    primary_ci = cfg_index[primary_name]
    primary_by_def = {d: cfg_index[f"{d}_H{prim_h}"] for d in DEFINITIONS}
    at_risk_list, full_list = [], []
    model_rows = []
    for ci, cfg in enumerate(configs):
        adh = coh["adh"][cfg.definition]
        ar, full = censoring_structure(coh["row_pat"], coh["row_bin"], adh, cfg.horizon, n)
        at_risk_list.append(ar)
        full_list.append(full)
        cnt = {}
        for c in RULE_CODES:
            m = ar & (coh["rule_raw"] == c)
            cnt[c] = (int((adh[m]).sum()), int((~adh[m]).sum()))
        groups = group_categories(cnt, args.min_events)
        feat_base = des["base_idx"]
        if cfg.fb_cov:
            feat_base = np.r_[feat_base, des["fb_idx"]]
        cfg.groups = []
        for label, codes in groups:
            feat = np.r_[feat_base, [des["rulecat_idx"][c] for c in codes[1:]]].astype(np.int64)
            cfg.groups.append((label, tuple(codes), feat))
        cfg.boot = args.boot_scope == "all" or ci == primary_ci
        log(f"  {cfg.name:<22} at-risk intervals={int(ar.sum()):>9,}  fully adherent={int(full.sum()):>7,} "
            f"({pct(full.mean()):5.1f}%)  groups={[g[0] for g in groups]}  "
            f"events(adh,non) by code={ {c: cnt[c] for c in RULE_CODES} }")
        model_rows.append(cnt)
    ctx = {
        "X": des["X"], "row_pat": coh["row_pat"], "row_code": coh["rule_raw"].astype(np.int64),
        "Y": pat["mortality_28d"].to_numpy(dtype=np.float64), "strata_M": M,
        "adh": {d: coh["adh"][d] for d in DEFINITIONS},
        "at_risk": at_risk_list, "full": full_list,
        "configs": [{"name": c.name, "definition": c.definition, "horizon": c.horizon, "groups": c.groups}
                    for c in configs],
        "truncations": TRUNCATIONS, "C": float(args.lr_C), "max_iter": int(args.max_iter), "tol": float(args.lr_tol),
        "clip_low": float(args.clip_low), "horizons": args.horizon_list, "primary_horizon": prim_h, "warm": {},
    }
    log(f"  [{time.time() - t0:.1f}s]")

    # ── 5. Point estimates ──────────────────────────────────────────────
    t0 = phase(5, N_PHASES, f"Fit censoring models and IPCW point estimates ({len(configs)} configurations)")
    point = {}
    for ci in tqdm(range(len(configs)), desc="  censoring models", mininterval=5):
        tc = time.time()
        r = estimate(ctx, np.ones(n), [ci], diag=True, warm=False)
        point[ci] = {"draw": r["draw"][0], **r["per_cfg"][ci]}
        for f in r["per_cfg"][ci]["fits"]:
            if f.get("kind") == "logit":
                ctx["warm"][(ci, f["label"])] = {"kind": "logit", "coef": f["coef"], "intercept": f["intercept"]}
        all_i = strata_names.index("all")
        rd = 100 * (point[ci]["draw"][0, all_i, 0] - point[ci]["draw"][0, all_i, 2])
        log(f"    {configs[ci].name:<22} RD_all(p99)={common.fmt_num(rd, 2)} pp; fits="
            + ", ".join(f"{f['label']}:{f.get('kind')}/{'ok' if f.get('converged', True) else 'NONCONV'}"
                        f"/it={f.get('n_iter', 0)}" for f in r["per_cfg"][ci]["fits"])
            + f"  [{time.time() - tc:.1f}s]")
    n_nonconv_point = sum(int(not f.get("converged", True)) for ci in point for f in point[ci]["fits"])
    log(f"  point fits non-converged: {n_nonconv_point}  [{time.time() - t0:.1f}s]")

    # ── 6. Bootstrap ────────────────────────────────────────────────────
    boot_ids = [ci for ci, c in enumerate(configs) if c.boot]
    t0 = phase(6, N_PHASES, f"Patient-level bootstrap with refitted censoring models: {args.n_boot} reps x "
                            f"{len(boot_ids)} configurations, n_jobs={args.n_jobs}")
    ctx_path = paths["interim"] / f"_BM2_ctx_{os.getpid()}.joblib"
    try:
        draws, bstats = run_bootstrap(ctx, boot_ids, args, ctx_path)
    finally:
        if ctx_path.exists():
            ctx_path.unlink()
            log(f"  removed temporary bootstrap context {ctx_path.name}")
    log(f"  bootstrap done: {bstats['n_done']} reps; non-converged fits {bstats['n_nonconverged']}; "
        f"constant (one-class) fits {bstats['n_constant']}  [{(time.time() - t0) / 60:.1f} min]")

    # ── 7. Assemble estimates and tables ─────────────────────────────────
    t0 = phase(7, N_PHASES, "Assemble estimates, CIs, diagnostics and tables")
    alpha = 2.5
    tr_names = list(TRUNCATIONS)
    n_adh_point = {ci: point[ci]["wsum"] for ci in point}
    long_rows = []
    est = {}   # (cfg_name, trunc, stratum, qty) -> dict

    def summarise(vals_point, vals_draw):
        dr = vals_draw[np.isfinite(vals_draw)] if vals_draw is not None else np.array([])
        lo, hi = (np.percentile(dr, [alpha, 100 - alpha]) if dr.size >= 2 else (np.nan, np.nan))
        se = float(np.std(dr, ddof=1)) if dr.size > 1 else np.nan
        return float(vals_point), float(lo), float(hi), se, int(dr.size)

    for ci, cfg in enumerate(configs):
        bpos = boot_ids.index(ci) if ci in boot_ids else None
        for ti, tname in enumerate(tr_names):
            pd_ = point[ci]["draw"][ti]                       # (S, 3)
            bd = draws[:, bpos, ti] if bpos is not None else None   # (B, S, 3)
            rd_by_s = {}
            for si, s in enumerate(strata_names):
                n_adh = n_adh_point[ci][(tname, si)]["n_adherent"]
                ok = n_adh >= args.min_adherent
                q = {
                    "risk_rule_pct": (100 * pd_[si, 0], None if bd is None else 100 * bd[:, si, 0]),
                    "risk_naive_pct": (100 * pd_[si, 1], None if bd is None else 100 * bd[:, si, 1]),
                    "risk_nc_pct": (100 * pd_[si, 2], None if bd is None else 100 * bd[:, si, 2]),
                    "rd_pp": (100 * (pd_[si, 0] - pd_[si, 2]), None if bd is None else 100 * (bd[:, si, 0] - bd[:, si, 2])),
                    "rd_naive_pp": (100 * (pd_[si, 1] - pd_[si, 2]),
                                    None if bd is None else 100 * (bd[:, si, 1] - bd[:, si, 2])),
                }
                rd_by_s[s] = (q["rd_pp"], q["rd_naive_pp"], ok)
                for qty, (pv, dv) in q.items():
                    e, lo, hi, se, nv = summarise(pv, dv)
                    estimable = ok or qty == "risk_nc_pct"
                    rec = {"config": cfg.name, "definition": cfg.definition, "horizon": cfg.horizon,
                           "fb_covariate": cfg.fb_cov, "truncation": tname, "stratum": s, "quantity": qty,
                           "n": int(M[si].sum()), "n_adherent": n_adh, "estimable": bool(estimable),
                           "estimate": e if estimable else np.nan, "ci_lo": lo if estimable else np.nan,
                           "ci_hi": hi if estimable else np.nan, "se_boot": se if estimable else np.nan,
                           "n_boot_valid": nv, "bootstrapped": bd is not None,
                           "note": "" if estimable else f"not estimable (n_adherent<{args.min_adherent})"}
                    long_rows.append(rec)
                    est[(cfg.name, tname, s, qty)] = rec
            name, sa, sb = DOD
            if sa in rd_by_s and sb in rd_by_s:
                for qty, j in (("rd_pp", 0), ("rd_naive_pp", 1)):
                    (pa, da), (pb, db) = rd_by_s[sa][j], rd_by_s[sb][j]
                    ok = rd_by_s[sa][2] and rd_by_s[sb][2]
                    e, lo, hi, se, nv = summarise(pa - pb, None if da is None else da - db)
                    rec = {"config": cfg.name, "definition": cfg.definition, "horizon": cfg.horizon,
                           "fb_covariate": cfg.fb_cov, "truncation": tname, "stratum": name, "quantity": qty,
                           "n": int(M[strata_names.index(sa)].sum() + M[strata_names.index(sb)].sum()),
                           "n_adherent": int(n_adh_point[ci][(tname, strata_names.index(sa))]["n_adherent"]
                                             + n_adh_point[ci][(tname, strata_names.index(sb))]["n_adherent"]),
                           "estimable": bool(ok), "estimate": e if ok else np.nan, "ci_lo": lo if ok else np.nan,
                           "ci_hi": hi if ok else np.nan, "se_boot": se if ok else np.nan, "n_boot_valid": nv,
                           "bootstrapped": da is not None,
                           "note": "" if ok else f"not estimable (n_adherent<{args.min_adherent} in a stratum)"}
                    long_rows.append(rec)
                    est[(cfg.name, tname, name, qty)] = rec
    long_df = pd.DataFrame(long_rows)
    common.save_table(long_df, paths["tables"] / f"BM2_ccw_estimates_long{sfx}.csv")

    def E(cfg_name, s, qty, tname=PRIMARY_TRUNC):
        return est[(cfg_name, tname, s, qty)]

    def show(rec, d=2):
        if not rec["estimable"]:
            return rec["note"]
        if not rec["bootstrapped"]:
            return f"{common.fmt_num(rec['estimate'], d)} (no bootstrap)"
        return ci_text(rec["estimate"], rec["ci_lo"], rec["ci_hi"], d)

    # weight diagnostics table
    wd_rows = []
    for ci, cfg in enumerate(configs):
        for tname in tr_names:
            for si, s in enumerate(strata_names):
                d = point[ci]["wsum"][(tname, si)]
                wd_rows.append({"config": cfg.name, "truncation": tname, "stratum": s, "n": int(M[si].sum()),
                                "cap": point[ci]["caps"][tname], **d})
    wd_df = pd.DataFrame(wd_rows)
    common.save_table(wd_df, paths["tables"] / f"BM2_weight_diagnostics{sfx}.csv")

    # censoring model table + primary coefficients
    cm_rows, coef_rows = [], []
    for ci, cfg in enumerate(configs):
        for f in point[ci]["fits"]:
            cm_rows.append({"config": cfg.name, "group": f["label"], "kind": f.get("kind"),
                            "n_rows": f.get("n_rows", 0), "n_adherent": f.get("n_adherent", 0),
                            "n_nonadherent": f.get("n_nonadherent", 0), "auc": f.get("auc", np.nan),
                            "mean_p": f.get("mean_p", np.nan), "share_p_lt005": f.get("share_p_lt005", np.nan),
                            "share_clipped": f.get("share_clipped", np.nan), "converged": f.get("converged", True),
                            "n_iter": f.get("n_iter", 0), "n_features": len(cfg.groups[[g[0] for g in cfg.groups]
                                                                                     .index(f["label"])][2])})
            if ci == primary_ci and f.get("kind") == "logit":
                feat = cfg.groups[[g[0] for g in cfg.groups].index(f["label"])][2]
                coef_rows.append({"config": cfg.name, "group": f["label"], "feature": "(intercept)",
                                  "coef_std_scale": f["intercept"]})
                for j, c in zip(feat, f["coef"]):
                    coef_rows.append({"config": cfg.name, "group": f["label"], "feature": des["names"][j],
                                      "coef_std_scale": float(c)})
    cm_df = pd.DataFrame(cm_rows)
    common.save_table(cm_df, paths["tables"] / f"BM2_censoring_models{sfx}.csv")
    common.save_table(pd.DataFrame(coef_rows), paths["tables"] / f"BM2_censoring_coefficients{sfx}.csv")

    # descriptive adherence
    cat_df = adherence_by_category(coh, strata, ctx, primary_by_def)
    common.save_table(cat_df, paths["tables"] / f"BM2_adherence_by_category{sfx}.csv")
    surv_df = adherence_survival(coh, strata, n, n_bins)
    common.save_table(surv_df, paths["tables"] / f"BM2_adherence_survival{sfx}.csv")

    # balance (primary, p99 truncation)
    w_prim = point[primary_ci]["w"]
    cap_prim = point[primary_ci]["caps"][PRIMARY_TRUNC]
    w_prim_t = np.minimum(w_prim, cap_prim) if np.isfinite(cap_prim) else w_prim
    full_prim = ctx["full"][primary_ci]
    bal_df = balance_table(coh, ctx, w_prim_t, full_prim)
    common.save_table(bal_df, paths["tables"] / f"BM2_balance{sfx}.csv")

    # primary wide summary
    wide = []
    full_strict = ctx["full"][primary_by_def["strict"]]
    for si, s in enumerate(strata_names):
        ws = point[primary_ci]["wsum"][(PRIMARY_TRUNC, si)]
        sm = strata[s]
        wide.append({
            "stratum": s, "n": int(sm.sum()), "deaths": int(pat["mortality_28d"].to_numpy()[sm].sum()),
            "nc_risk_pct": E(primary_name, s, "risk_nc_pct")["estimate"],
            "n_adherent_relaxed": int((full_prim & sm).sum()),
            "pct_adherent_relaxed": pct((full_prim & sm).sum() / max(int(sm.sum()), 1)),
            "n_adherent_strict": int((full_strict & sm).sum()),
            "pct_adherent_strict": pct((full_strict & sm).sum() / max(int(sm.sum()), 1)),
            "naive_risk_pct": E(primary_name, s, "risk_naive_pct")["estimate"],
            "ipcw_risk_pct": E(primary_name, s, "risk_rule_pct")["estimate"],
            "ipcw_risk_ci_lo": E(primary_name, s, "risk_rule_pct")["ci_lo"],
            "ipcw_risk_ci_hi": E(primary_name, s, "risk_rule_pct")["ci_hi"],
            "rd_pp": E(primary_name, s, "rd_pp")["estimate"], "rd_ci_lo": E(primary_name, s, "rd_pp")["ci_lo"],
            "rd_ci_hi": E(primary_name, s, "rd_pp")["ci_hi"],
            "rd_naive_pp": E(primary_name, s, "rd_naive_pp")["estimate"],
            "rd_naive_ci_lo": E(primary_name, s, "rd_naive_pp")["ci_lo"],
            "rd_naive_ci_hi": E(primary_name, s, "rd_naive_pp")["ci_hi"],
            "estimable": E(primary_name, s, "rd_pp")["estimable"],
            "ess": ws["ess"], "w_mean": ws["w_untrunc_mean"], "w_p50": ws["w_untrunc_p50"],
            "w_p99": ws["w_untrunc_p99"], "w_max": ws["w_untrunc_max"],
        })
    wide_df = pd.DataFrame(wide)
    common.save_table(wide_df, paths["tables"] / f"BM2_ccw_primary_summary{sfx}.csv")

    # stay-level primary weights (interim only)
    common.save_interim(pd.DataFrame({
        "stay_id": coh["ids"], "fully_adherent_relaxed_H": full_prim, "fully_adherent_strict_H": full_strict,
        "ipcw_weight_untruncated": w_prim, "ipcw_weight_p99": np.where(full_prim, w_prim_t, 0.0)}),
        root, f"BM2_perstay_primary_weights{sfx}.parquet")
    log(f"  [{time.time() - t0:.1f}s]")

    # ── 8. Report ───────────────────────────────────────────────────────
    t0 = phase(8, N_PHASES, "Report and PASTE-BACK block")
    runtime_min = (time.time() - t_start) / 60.0
    chk = coh["rule_check"]
    n_ok = n == EXPECTED_N_STAYS
    n_line = (f"REPRO_CHECK_N: {'PASS' if n_ok else 'FAIL'} (n_stays={n:,} vs expected {EXPECTED_N_STAYS:,}"
              f"{'; quick subsample' if args.quick else ''})")
    print(n_line, flush=True)
    rule = coh["rule_raw"]
    req = coh["row_bin"] < prim_h
    hf_rows = strata["hfpef"][coh["row_pat"]] & req
    mix_all = {c: pct((rule[req] == c).mean()) for c in RULE_CODES}
    mix_hf = {c: pct((rule[hf_rows] == c).mean()) if hf_rows.any() else np.nan for c in RULE_CODES}

    def cat_get(d, s, code, col):
        r = cat_df[(cat_df["definition"] == d) & (cat_df["stratum"] == s) & (cat_df["rule_code"] == code)]
        return float(r[col].iloc[0]) if len(r) else float("nan")

    L = common.report_header("BM2 — Clone-censor-weight / IPCW per-protocol estimate of the 70c dynamic rule "
                             "vs the natural course (no outcome model)", root, args)
    L += [
        "Estimand: 28-day mortality if every patient had followed the 70c rule during the first H 2-h intervals",
        "after T0 (then usual care), versus the natural course (observed usual care), in the full analytic cohort.",
        "Artificial censoring at the first deviation; IPCW from pooled logistic adherence models per required rule",
        f"category; weights clipped at p_hat>={args.clip_low}, truncated at the {PRIMARY_TRUNC} of adherent weights.",
        f"Primary: relaxed adherence, H={prim_h} ({2 * prim_h} h). RD = rule - natural course (pp); negative favours "
        "the rule.",
        "",
        f"Inputs: actions={inp['action_source_file']}; rules={inp['rule_source_file']} (source: {args.rule_source}); "
        f"phenotype labels={inp['pheno_source_file']};",
        f"  censoring covariates ({len(des['base_idx'])} base columns): bin dummies, {len(inp['z_cols'])} *_z, "
        f"{len(inp['mask_cols'])} *_mask, previous fluid/vaso/diuretic level dummies, age, female, "
        f"admission type ({len(des['adm_levels']) - 1} dummies; ref '{des['adm_ref']}'), hf_subtype (4 dummies), "
        f"shock, hf_subtype x shock" + (", baseline SOFA" if sofa_run else "") + ".",
        f"  LogisticRegression(lbfgs, C={args.lr_C:g}, max_iter={args.max_iter}, tol={args.lr_tol:g}); bootstrap reps={args.n_boot} "
        f"(scope={args.boot_scope}, refit, warm start={'no' if args.no_warm_start else 'yes'}), n_jobs={args.n_jobs}.",
        "",
        f"Cohort: {n:,} stays; deaths {int(pat['mortality_28d'].sum()):,} ({100 * pat['mortality_28d'].mean():.2f}%); "
        f"active intervals {coh['n_active_all']:,}.",
        "  " + n_line,
        f"  RULE_SIDECAR_CHECK: {chk['status']} (agreement with cumulative FB {common.fmt_num(pct(chk['agree_cumfb']), 3)}%;"
        f" interval FB {common.fmt_num(pct(chk['agree_intervalfb']), 3)}%; pairs {chk['pairs']})",
        f"  Rule mix on active intervals (bin<{prim_h}), %: all {code_str(mix_all)}; HFpEF {code_str(mix_hf)}",
        f"  Intervals strict-adherent but relaxed-non-adherent (merged rare cells): {coh['n_strict_not_relaxed']}",
    ]
    if sofa_run and des["sofa_info"]:
        sm = inp.get("sofa_meta") or {}
        L.append(f"  Baseline SOFA: median {des['sofa_info']['median']:.1f}; missing {des['sofa_info']['n_missing']}"
                 + (" (median-imputed + indicator)" if des["sofa_info"]["n_missing"] else "")
                 + f"; BM3a file quick/truncated={sm.get('_partial', 'NA') if sm.get('_has_meta') else 'NA (no meta)'}"
                 + f", lookback {sm.get('lookback_hours', 'NA')} h, ETT-GCS {sm.get('gcs_ett_primary', 'NA')}")
        if sm.get("_partial"):
            L.append("  ** WARNING: the BM3a SOFA file is from a QUICK / truncated run -- rerun BM3a in full **")
    L += ["", f"Per-category adherence (active intervals, bin<{prim_h}; at-risk = uncensored under H={prim_h}):"]
    hdr = (f"  {'def':<8}{'stratum':<16}{'rule':<26}{'n_int':>8}{'mix%':>7}{'adh%':>7}{'fail_v%':>8}"
           f"{'fail_d%':>8}{'fail_f%':>8}{'n_atrisk':>9}{'adh_ar%':>8}")
    L.append(hdr)
    for d in DEFINITIONS:
        for s in ("all", "hfpef", "hfpef_nonshock", "nohf_nonshock"):
            for code in ("all",) + RULE_CODES:
                r = cat_df[(cat_df["definition"] == d) & (cat_df["stratum"] == s) & (cat_df["rule_code"] == code)].iloc[0]
                L.append(f"  {d:<8}{s:<16}{r['rule_label']:<26}{int(r['n_intervals']):>8}"
                         f"{common.fmt_num(r['share_of_stratum_intervals_pct'], 1):>7}"
                         f"{common.fmt_num(r['adherent_pct'], 1):>7}{common.fmt_num(r['fail_vaso_pct'], 1):>8}"
                         f"{common.fmt_num(r['fail_diuretic_pct'], 1):>8}{common.fmt_num(r['fail_fluid_pct'], 1):>8}"
                         f"{int(r['n_at_risk_primaryH']):>9}{common.fmt_num(r['adherent_at_risk_primaryH_pct'], 1):>8}")
    L += ["  Note: relaxed fluid criterion |obs - rule level| <= 1 can fail only for rule 16/14 (high fluid; observed none)",
          "  or rule 1 (no fluid; observed high); for the default (low/moderate) only vasopressor / diuretic use counts.",
          "  ESS is computed on the p99-truncated weights; weight mean/p50/p99/max are untruncated.",
          "", "Full adherence by configuration (patients):"]
    for ci, cfg in enumerate(configs):
        if cfg.fb_cov:
            continue
        fl = ctx["full"][ci]
        L.append(f"  {cfg.name:<14} all {int(fl.sum()):>7,} ({pct(fl.mean()):5.1f}%)  "
                 + "  ".join(f"{s} {int((fl & strata[s]).sum()):,} ({pct((fl & strata[s]).sum() / max(int(strata[s].sum()), 1)):.1f}%)"
                             for s in SPEC_STRATA[1:]))
    sv = surv_df[(surv_df["stratum"] == "all")]
    for d in DEFINITIONS:
        L.append(f"  still adherent by bin ({d}, all), %: "
                 + ",".join(common.fmt_num(v, 1) for v in sv[sv["definition"] == d]["still_adherent_pct"]))
    L += ["", "Censoring models (full sample):"]
    for ci, cfg in enumerate(configs):
        pc = point[ci]
        L.append(f"  {cfg.name:<22} at-risk rows {pc['n_at_risk_rows']:,}; p_hat<0.05 {pct(pc['share_p_lt005_atrisk']):.2f}% "
                 f"(weighted rows {pct(pc['share_p_lt005_weighted_rows']):.2f}%); clipped {pct(pc['share_p_clipped_atrisk']):.2f}%; "
                 f"cap p99={common.fmt_num(pc['caps']['p99'], 2)} p99.9={common.fmt_num(pc['caps']['p999'], 2)}")
        for f in pc["fits"]:
            L.append(f"      group {f['label']:<10} {f.get('kind'):<8} rows={f.get('n_rows', 0):>8,} adh={f.get('n_adherent', 0):>8,} "
                     f"non={f.get('n_nonadherent', 0):>8,} AUC={common.fmt_num(f.get('auc', np.nan), 3)} "
                     f"mean_p={common.fmt_num(f.get('mean_p', np.nan), 3)} converged={f.get('converged', True)} "
                     f"iter={f.get('n_iter', 0)}")
    L += ["", f"PRIMARY RESULT ({primary_name}, truncation {PRIMARY_TRUNC}; risks %, RD pp, 95% bootstrap CI):"]
    L.append(f"  {'stratum':<16}{'n':>7}{'deaths':>7}{'NC risk':>9}{'n_adh':>7}{'adh%':>6}{'ESS':>8}{'w_max':>9}"
             f" | IPCW risk (CI) | RD (CI) | naive RD (CI)")
    for r in wide:
        s = r["stratum"]
        L.append(f"  {s:<16}{r['n']:>7}{r['deaths']:>7}{common.fmt_num(r['nc_risk_pct'], 2):>9}"
                 f"{r['n_adherent_relaxed']:>7}{common.fmt_num(r['pct_adherent_relaxed'], 1):>6}"
                 f"{common.fmt_num(r['ess'], 1):>8}{common.fmt_num(r['w_max'], 1):>9}"
                 f" | {show(E(primary_name, s, 'risk_rule_pct'))} | {show(E(primary_name, s, 'rd_pp'))}"
                 f" | {show(E(primary_name, s, 'rd_naive_pp'))}")
    L.append(f"  {DOD[0]}: RD {show(E(primary_name, DOD[0], 'rd_pp'))}; naive {show(E(primary_name, DOD[0], 'rd_naive_pp'))}")
    L += ["", "Weight truncation sensitivity (primary configuration), RD pp:"]
    for s in SPEC_STRATA + [DOD[0]]:
        L.append(f"  {s:<36}" + "  ".join(f"{t}: {show(E(primary_name, s, 'rd_pp', t))}" for t in tr_names))
    L += ["", "Secondary configurations (truncation p99), RD pp:"]
    for cfg in configs:
        if cfg.name == primary_name:
            continue
        L.append(f"  {cfg.name}:")
        for s in SPEC_STRATA + [DOD[0]]:
            rec = E(cfg.name, s, "rd_pp")
            L.append(f"      {s:<36} n_adh={rec['n_adherent']:>6}  RD {show(rec)}  naive {show(E(cfg.name, s, 'rd_naive_pp'))}")
    L += ["", "Baseline balance, primary (SMD vs whole cohort; |SMD|<0.1 = balanced):"]
    for _, r in bal_df.iterrows():
        L.append(f"  {r['covariate']:<34} all={common.fmt_num(r['mean_all'], 3):>8} adh={common.fmt_num(r['mean_adherent_unweighted'], 3):>8} "
                 f"adh_w={common.fmt_num(r['mean_adherent_weighted'], 3):>8}  SMD {common.fmt_num(r['smd_unweighted'], 3):>7} -> "
                 f"{common.fmt_num(r['smd_weighted'], 3):>7}")
    L += ["", "How to read:",
          "  * RD < 0 with a CI excluding 0 supports benefit of following the rule without any outcome model. The",
          "    estimand (adhere to the rule for H intervals, then usual care, vs the natural course) differs from the",
          "    G-Transformer contrast (rule in every interval vs usual care, both under model rollout), so agreement",
          "    in SIGN and rough size is what matters, not equality. The published G-Transformer reference",
          f"    ({common.REPRO_TARGETS['rd_all']} / {common.REPRO_TARGETS['rd_hfpef']} pp, keys gtransformer_published_*) "
          "came from the pre-v1.0.1 code whose 'rule' arm",
          "    was the zero-action arm (common.LEGACY_RULE_NOTE); compare BM2 with BM1's corrected rule-arm RDs.",
          "  * naive RD vs IPCW RD shows confounding by adherence: if naive is more negative, healthier patients were",
          "    the ones whose care matched the rule; IPCW removes the measured part of that.",
          "  * ESS << n_adherent, p_hat<0.05 shares of several %, w_max >> cap, or truncation-dependent signs indicate",
          "    positivity problems; then the estimate is fragile. Strata flagged 'not estimable' had < "
          f"{args.min_adherent} adherent patients.",
          f"  * Runtime {runtime_min:.1f} min."]

    # ── paste-back block ─────────────────────────────────────────────
    P = OrderedDict()
    P["script_version"] = SCRIPT_VERSION
    P["sofa_adjusted"] = sofa_run
    if sofa_run:
        sm = inp.get("sofa_meta") or {}
        P["sofa_file_quick"] = sm.get("_partial", "NA") if sm.get("_has_meta") else "NA"
    P["quick"] = bool(args.quick)
    P["n_stays"] = n
    P["n_deaths"] = int(pat["mortality_28d"].sum())
    P["mortality_pct"] = 100 * float(pat["mortality_28d"].mean())
    P["n_active_intervals"] = coh["n_active_all"]
    P["n_boot"] = int(args.n_boot)
    P["n_boot_done"] = bstats["n_done"]
    P["boot_scope"] = args.boot_scope
    P["boot_nonconverged_fits"] = bstats["n_nonconverged"]
    P["boot_constant_fits"] = bstats["n_constant"]
    P["point_nonconverged_fits"] = n_nonconv_point
    P["n_jobs"] = int(args.n_jobs)
    P["seed"] = int(args.seed)
    P["n_censoring_covariates"] = int(len(des["base_idx"]))
    P["REPRO_CHECK_N"] = "PASS" if n_ok else "FAIL"
    P["rule_source"] = args.rule_source
    P["RULE_SIDECAR_CHECK"] = chk["status"]
    P["rule_sidecar_agree_cumfb_pct"] = pct(chk["agree_cumfb"])
    P["rule_sidecar_agree_intervalfb_pct"] = pct(chk["agree_intervalfb"])
    P["rule_mix_all_pct"] = code_str(mix_all)
    P["rule_mix_hfpef_pct"] = code_str(mix_hf)
    for d in DEFINITIONS:
        P[f"adh_interval_{d}_all_pct"] = cat_get(d, "all", "all", "adherent_pct")
        P[f"adh_interval_{d}_hfpef_pct"] = cat_get(d, "hfpef", "all", "adherent_pct")
        P[f"adh_interval_{d}_by_code_pct"] = code_str({c: cat_get(d, "all", c, "adherent_pct") for c in RULE_CODES})
    P["relaxed_fail_default_pct"] = (f"vaso={common.fmt_num(cat_get('relaxed', 'all', 6, 'fail_vaso_pct'), 1)};"
                                     f"diuretic={common.fmt_num(cat_get('relaxed', 'all', 6, 'fail_diuretic_pct'), 1)};"
                                     f"fluid={common.fmt_num(cat_get('relaxed', 'all', 6, 'fail_fluid_pct'), 1)}")
    P["relaxed_fail_diuretic_rule_pct"] = (f"vaso={common.fmt_num(cat_get('relaxed', 'all', 1, 'fail_vaso_pct'), 1)};"
                                           f"diuretic={common.fmt_num(cat_get('relaxed', 'all', 1, 'fail_diuretic_pct'), 1)};"
                                           f"fluid={common.fmt_num(cat_get('relaxed', 'all', 1, 'fail_fluid_pct'), 1)}")
    P["n_strict_not_relaxed_intervals"] = coh["n_strict_not_relaxed"]
    for ci, cfg in enumerate(configs):
        if not cfg.fb_cov:
            P[f"full_adh_{cfg.name}_pct"] = pct(ctx["full"][ci].mean())
    for d in DEFINITIONS:
        P[f"adh_survival_{d}_all_by_bin_pct"] = ",".join(
            common.fmt_num(v, 1) for v in sv[sv["definition"] == d]["still_adherent_pct"])
    pp = point[primary_ci]
    P["primary_config"] = f"{primary_name}/{PRIMARY_TRUNC}"
    P["primary_model_groups"] = "|".join(f["label"] for f in pp["fits"])
    P["primary_model_rows"] = ";".join(f"{f['label']}={f.get('n_rows', 0)}" for f in pp["fits"])
    P["primary_model_events_adh_non"] = ";".join(f"{f['label']}={f.get('n_adherent', 0)}/{f.get('n_nonadherent', 0)}"
                                                 for f in pp["fits"])
    P["primary_model_auc"] = ";".join(f"{f['label']}={common.fmt_num(f.get('auc', np.nan), 3)}" for f in pp["fits"])
    P["primary_share_phat_lt005_atrisk_pct"] = pct(pp["share_p_lt005_atrisk"])
    P["primary_share_phat_lt005_weighted_rows_pct"] = pct(pp["share_p_lt005_weighted_rows"])
    P["primary_share_phat_clipped_pct"] = pct(pp["share_p_clipped_atrisk"])
    P["primary_trunc_cap_p99"] = float(pp["caps"]["p99"])
    P["primary_trunc_cap_p999"] = float(pp["caps"]["p999"])
    for r in wide:
        s = r["stratum"]
        P[f"{s}_n"] = r["n"]
        P[f"{s}_deaths"] = r["deaths"]
        P[f"{s}_nc_risk_pct"] = show(E(primary_name, s, "risk_nc_pct"))
        P[f"{s}_n_adh_relaxed"] = r["n_adherent_relaxed"]
        P[f"{s}_pct_adh_relaxed"] = r["pct_adherent_relaxed"]
        P[f"{s}_n_adh_strict"] = r["n_adherent_strict"]
        P[f"{s}_pct_adh_strict"] = r["pct_adherent_strict"]
        P[f"{s}_naive_risk_pct"] = show(E(primary_name, s, "risk_naive_pct"))
        P[f"{s}_ipcw_risk_pct"] = show(E(primary_name, s, "risk_rule_pct"))
        P[f"{s}_rd_pp"] = show(E(primary_name, s, "rd_pp"))
        P[f"{s}_rd_naive_pp"] = show(E(primary_name, s, "rd_naive_pp"))
        P[f"{s}_ess_trunc_p99"] = r["ess"]
        P[f"{s}_w_untrunc_mean"] = r["w_mean"]
        P[f"{s}_w_untrunc_p50"] = r["w_p50"]
        P[f"{s}_w_untrunc_p99"] = r["w_p99"]
        P[f"{s}_w_untrunc_max"] = r["w_max"]
        P[f"{s}_rd_pp_trunc_none"] = show(E(primary_name, s, "rd_pp", "none"))
        P[f"{s}_rd_pp_trunc_p999"] = show(E(primary_name, s, "rd_pp", "p999"))
    P["dod_nonshock_hfpef_minus_nohf_rd_pp"] = show(E(primary_name, DOD[0], "rd_pp"))
    P["dod_nonshock_hfpef_minus_nohf_rd_naive_pp"] = show(E(primary_name, DOD[0], "rd_naive_pp"))
    P["dod_nonshock_rd_pp_trunc_none"] = show(E(primary_name, DOD[0], "rd_pp", "none"))
    for cfg in configs:
        if cfg.name == primary_name:
            continue
        ci = cfg_index[cfg.name]
        P[f"{cfg.name}_all_n_adh"] = int(point[ci]["wsum"][(PRIMARY_TRUNC, strata_names.index("all"))]["n_adherent"])
        P[f"{cfg.name}_all_ess"] = float(point[ci]["wsum"][(PRIMARY_TRUNC, strata_names.index("all"))]["ess"])
        P[f"{cfg.name}_hfpef_n_adh"] = int(point[ci]["wsum"][(PRIMARY_TRUNC, strata_names.index("hfpef"))]["n_adherent"])
        P[f"{cfg.name}_share_phat_lt005_atrisk_pct"] = pct(point[ci]["share_p_lt005_atrisk"])
        for s in ("all", "hfpef", "hfpef_nonshock", "hfpef_shock", "nohf_nonshock"):
            P[f"{cfg.name}_{s}_rd_pp"] = show(E(cfg.name, s, "rd_pp"))
        P[f"{cfg.name}_dod_rd_pp"] = show(E(cfg.name, DOD[0], "rd_pp"))
    P["balance_max_abs_smd_unweighted"] = float(np.nanmax(np.abs(bal_df["smd_unweighted"]))) if len(bal_df) else np.nan
    P["balance_max_abs_smd_weighted"] = float(np.nanmax(np.abs(bal_df["smd_weighted"]))) if len(bal_df) else np.nan
    P["balance_smd_weighted"] = ";".join(f"{r['covariate']}={common.fmt_num(r['smd_weighted'], 3)}"
                                         for _, r in bal_df.iterrows())

    def rec_of(s, qty="rd_pp", t=PRIMARY_TRUNC, cfg_name=primary_name):
        return E(cfg_name, s, qty, t)

    def excl0(rec):
        return bool(rec["estimable"] and np.isfinite(rec["ci_lo"]) and np.isfinite(rec["ci_hi"])
                    and (rec["ci_hi"] < 0 or rec["ci_lo"] > 0))

    def sign(rec):
        return np.sign(rec["estimate"]) if rec["estimable"] and np.isfinite(rec["estimate"]) else np.nan

    # Flags are "NA" (not False) when an input is not estimable / not bootstrapped, so that e.g. a
    # not-estimable HFpEF stratum is never pasted back as "hfpef_rd_negative: False".
    def has_ci(rec):
        return bool(rec["estimable"] and np.isfinite(rec["ci_lo"]) and np.isfinite(rec["ci_hi"]))

    def flag_neg(rec):
        sg = sign(rec)
        return bool(sg < 0) if np.isfinite(sg) else "NA"

    def flag_ci(rec):
        return excl0(rec) if has_ci(rec) else "NA"

    def flag_same(signs):
        return bool(len(set(signs)) == 1) if np.all(np.isfinite(signs)) else "NA"

    P["flag_primary_all_rd_negative"] = flag_neg(rec_of("all"))
    P["flag_primary_all_ci_excludes_0"] = flag_ci(rec_of("all"))
    P["flag_primary_hfpef_rd_negative"] = flag_neg(rec_of("hfpef"))
    P["flag_primary_hfpef_ci_excludes_0"] = flag_ci(rec_of("hfpef"))
    P["flag_primary_hfpef_nonshock_ci_excludes_0"] = flag_ci(rec_of("hfpef_nonshock"))
    P["flag_primary_dod_ci_excludes_0"] = flag_ci(rec_of(DOD[0]))
    ra, rn = rec_of("all"), rec_of("all", "rd_naive_pp")
    P["flag_naive_more_negative_than_ipcw_all"] = (bool(rn["estimate"] < ra["estimate"])
                                                   if np.isfinite(sign(ra)) and np.isfinite(sign(rn)) else "NA")
    P["flag_relaxed_horizon_signs_consistent"] = flag_same(
        [sign(rec_of("all", cfg_name=f"relaxed_H{h}")) for h in args.horizon_list])
    P["flag_truncation_signs_consistent"] = flag_same([sign(rec_of("all", t=t)) for t in tr_names])
    P["flag_strict_sign_same_as_relaxed"] = flag_same(
        [sign(rec_of("all")), sign(rec_of("all", cfg_name=f"strict_H{prim_h}"))])
    ess_all = point[primary_ci]["wsum"][(PRIMARY_TRUNC, strata_names.index("all"))]
    P["primary_ess_over_n_adherent_all"] = (ess_all["ess"] / ess_all["n_adherent"]
                                            if ess_all["n_adherent"] else float("nan"))
    P["gtransformer_published_rd_all_pp_reference"] = common.REPRO_TARGETS["rd_all"]
    P["gtransformer_published_rd_hfpef_pp_reference"] = common.REPRO_TARGETS["rd_hfpef"]
    P["runtime_min"] = runtime_min

    common.write_report(paths["reports"] / f"BM2{sfx}_summary.txt", L, P, tag)
    log(f"  [{time.time() - t0:.1f}s]")
    log(f"BM2 done in {(time.time() - t_start) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
