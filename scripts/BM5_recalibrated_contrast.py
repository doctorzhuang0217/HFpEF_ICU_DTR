#!/usr/bin/env python3
"""
BM5_recalibrated_contrast — Recalibration of the G-Transformer outcome model on
the validation set and the recalibrated rule-versus-usual-care contrast (BMC
Medicine revision, BM5).

Purpose
-------
The Step C contrast is a difference of two MODEL risks (rule arm minus usual
care) produced by the same outcome head. If that head is miscalibrated (e.g. it
under-predicts 28-day mortality, or its probabilities are too extreme), the
absolute risk difference in percentage points is distorted. BM5 asks whether the
contrast survives recalibration:

  1. Validation set (frozen 80c split; in 80c it served only for early stopping on
     the total training loss, never for the test contrast): G-Transformer rollout
     with the OBSERVED actions
     (= usual-care arm, model.predict_counterfactual_rollout) -> p_val.
  2. Recalibration maps fitted ONCE on (p_val, observed mortality_28d):
       logistic        logit P(Y=1) = a + b*logit(p)   [PRIMARY; statsmodels GLM
                       Binomial, unpenalised]
       intercept       logit P(Y=1) = a + logit(p)     (recalibration-in-the-large,
                       slope fixed at 1; offset GLM)    [extra sensitivity]
       isotonic        sklearn IsotonicRegression(out_of_bounds='clip')
                                                        [SENSITIVITY]
       logistic_hfpef  logit P(Y=1) = a + b*logit(p) + c*HFpEF   [OPTIONAL]
  3. Test set: every map is applied to the per-stay risks of each arm
       uc    usual care (observed actions)
       na    rule, non-adaptive (stored 70c actions; published design, corrected code)
       ad    rule, adaptive (BM1 primary dynamic arm; recorded cumulative fluid balance)
       adfb  rule, adaptive with approximate fluid balance (BM1 sensitivity)
       zero  legacy zero-action arm (the pre-v1.0.1 'rule' arm; = BM4 S0)
     taken from the BM1 per-stay file when it matches this test set, checkpoint,
     usual-care AND non-adaptive rule predictions (both re-rolled here; a stale
     BM1 file from an older 70c rule sidecar is rejected), otherwise recomputed
     here through common.
  4. Per stratum (all; HFpEF; HFpEF non-shock; HFpEF shock; no-HF non-shock;
     HFrEF; extended strata in the CSV): observed mortality, raw and
     recalibrated usual-care risk and the observed-minus-predicted gap,
     recalibrated risks and RDs (rule - usual care, adaptive - non-adaptive),
     the change of each RD caused by recalibration, and the HFpEF-minus-no-HF
     non-shock difference of RDs, all with paired patient-level bootstrap
     percentile 95% CIs (recalibration fixed; resampling test patients only).
  5. Calibration of the usual-care risk in the TEST set before and after each
     map (overall, HFpEF and the other spec strata): mean predicted vs observed,
     O/E, calibration intercept (calibration-in-the-large, offset GLM) and slope
     (GLM on logit p) with Wald 95% CIs, Brier score (and scaled Brier), AUC;
     decile (overall) / quintile (HFpEF) calibration tables; the paired-bootstrap
     CI of the Brier change.

Interpretation: recalibration with an increasing map cannot change the SIGN of a
per-patient contrast (stratum means can still change sign when per-patient
contrasts have mixed signs), but it rescales absolute RDs (logistic: by roughly
b * p'(1-p') / (p(1-p))) and redistributes them across strata. The contrast is
robust if the recalibrated RDs keep their sign, their CIs still exclude 0, and
the change of RD is small relative to the RD. The approach assumes that the
miscalibration map estimated on usual-care predictions also applies to the
model's predictions under the counterfactual action sequences.

Protocol reference: BMC Medicine revision, analysis BM5 (extends the Step 80c DM
rollout; cf. 90b_07 outcome-head recalibration on eICU). Spec: SPEC.md BM5.
Compute: G1 per spec (qsub env/sge/submit_g1.sh; ~3-6 min wall, dominated by
data loading and three DTRDataset builds (train vocabulary, validation, test)).
The model work is 3 fixed rollouts (validation UC, test UC, test rule) when the
BM1 per-stay file is reused, or 4 fixed + 2 adaptive rollouts when recomputed;
on CPU (C2 direct, --device cpu, 8 threads) the whole script takes ~4-8 min
(~6-10 min with --arms-source recompute).

Inputs (through common.load_all)
------
- configs/80_step_c_training_v1.yaml (+ 45c normalisation, 70b merge map, 70c rule)
- data/interim/45c_state_tensors_v1_mimic.parquet
- data/interim/45d_outcomes_absorb_v1_mimic.parquet
- data/interim/70b_propensity_coarse3_v1_mimic.parquet
- data/interim/70c_rule_policy_actions_v1_mimic.parquet
- data/processed/phenotype_labels_frozen_v1_mimic.parquet
- models/80c_gtransformer_rollout_ow1_best.pt (default; the checkpoint behind the
  published numbers) or --checkpoint models/80c_gtransformer_best.pt (P2 export)
- data/interim/bmc_medicine/BM1_perstay_predictions.parquet (+ .meta.json), optional

Outputs
-------
- tables/bmc_medicine/BM5_recalibration_models.csv     (fitted maps: terms, CIs)
- tables/bmc_medicine/BM5_calibration_metrics.csv      (val/test x stratum x method)
- tables/bmc_medicine/BM5_calibration_groups.csv       (deciles / quintiles)
- tables/bmc_medicine/BM5_strata_summary.csv           (wide: stratum x method)
- tables/bmc_medicine/BM5_bootstrap_long.csv           (all bootstrap estimates)
- reports/bmc_medicine/BM5_summary.txt                 (ends with the PASTE-BACK BLOCK)
- data/interim/bmc_medicine/BM5_perstay_recalibrated.parquet (STAY-LEVEL; never sync)

Usage
-----
    # Shirokane (project root), GPU, after BM1 (reuses its per-stay predictions):
    qsub -N sBM5_recal -q gpua.q -l a100=1 env/sge/submit_g1.sh \\
        scripts/bmc_medicine/BM5_recalibrated_contrast.py \\
        --checkpoint models/80c_gtransformer_rollout_ow1_best.pt
    # CPU direct (C2):
    export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
    env/run_py.sh scripts/bmc_medicine/BM5_recalibrated_contrast.py --device cpu \\
        --checkpoint models/80c_gtransformer_rollout_ow1_best.pt
    # smoke test:
    env/run_py.sh scripts/bmc_medicine/BM5_recalibrated_contrast.py --quick --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import log, phase  # noqa: E402

import torch  # noqa: E402

SCRIPT_VERSION = "1.0"
TAG = "BM5"
N_PHASES = 8
BM1_PERSTAY_REL = "data/interim/bmc_medicine/BM1_perstay_predictions.parquet"
PERSTAY_FILE = "BM5_perstay_recalibrated.parquet"

METHODS = OrderedDict([
    ("raw", "no recalibration (model rollout probability)"),
    ("logistic", "logistic a + b*logit(p), GLM Binomial on validation [PRIMARY]"),
    ("intercept", "recalibration-in-the-large a + logit(p) (offset GLM; slope fixed at 1)"),
    ("isotonic", "isotonic regression, sklearn out_of_bounds='clip' [SENSITIVITY]"),
    ("logistic_hfpef", "a + b*logit(p) + c*HFpEF, GLM Binomial on validation [OPTIONAL]"),
])
SECONDARY_METHODS = ["intercept", "isotonic", "logistic_hfpef"]
ARMS = OrderedDict([
    ("uc", ("y_uc", "usual care (observed actions)")),
    ("na", ("y_rule_nonadaptive", "rule, non-adaptive (stored 70c actions)")),
    ("ad", ("y_rule_adaptive", "rule, adaptive, recorded fluid balance (BM1 primary)")),
    ("adfb", ("y_rule_adaptive_fbapprox", "rule, adaptive, approximate fluid balance")),
    ("zero", ("y_legacy_zero", "legacy zero-action arm (pre-v1.0.1 'rule'; = BM4 S0)")),
])
# contrast key -> (arm a, arm b): RD = risk(a) - risk(b)
RD_SPECS = OrderedDict([
    ("rd_na", ("na", "uc")),
    ("rd_ad", ("ad", "uc")),
    ("rd_adfb", ("adfb", "uc")),
    ("rd_zero", ("zero", "uc")),
    ("ad_minus_na", ("ad", "na")),
])
SPEC_STRATA = ["all", "hfpef", "hfpef_nonshock", "hfpef_shock", "nohf_nonshock", "hfref"]
SECONDARY_STRATA = ["all", "hfpef", "hfpef_nonshock", "nohf_nonshock"]
CALIB_PASTE_STRATA = ["all", "hfpef"]
DOD_NAME = common.STANDARD_STRATUM_CONTRASTS[0][0]          # hfpef_nonshock_minus_nohf_nonshock


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BM5 recalibration of the outcome model and recalibrated contrast",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(p, n_boot_default=1000, model_based=True)
    p.add_argument("--arms-source", choices=("auto", "bm1", "recompute"), default="auto",
                   help="test-set arm predictions: 'auto' = BM1 per-stay file if it matches this test set, "
                        "checkpoint and usual-care predictions, else recompute; 'bm1' = require it; "
                        "'recompute' = always recompute through common")
    p.add_argument("--bm1-perstay", default=BM1_PERSTAY_REL,
                   help="BM1 per-stay predictions (relative to PROJECT_ROOT or absolute)")
    p.add_argument("--uc-tol", type=float, default=1e-4,
                   help="max |y_uc(BM1) - y_uc(recomputed)| (probability scale) to accept the BM1 file")
    p.add_argument("--eps", type=float, default=1e-6,
                   help="probability clip before logit inside the recalibration maps")
    p.add_argument("--calib-eps", type=float, default=1e-3,
                   help="probability clip before logit for the calibration intercept / slope of the ISOTONIC "
                        "predictions (keeps fitted values of exactly 0 or 1 from dominating the slope); the "
                        "continuous maps (raw, logistic, intercept, logistic_hfpef) use --eps")
    p.add_argument("--threads", type=int,
                   default=int(os.environ.get("TORCH_NUM_THREADS") or os.environ.get("NSLOTS") or 8),
                   help="torch CPU threads (default TORCH_NUM_THREADS / NSLOTS / 8)")
    p.add_argument("--no-save-perstay", action="store_true",
                   help=f"do not write data/interim/bmc_medicine/{PERSTAY_FILE}")
    return p.parse_args(argv)


# ═══════════════════════════════════════════════════════════════════════
# Recalibration and calibration helpers
# ═══════════════════════════════════════════════════════════════════════
def logit(p, eps: float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def expit(x) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def glm_binomial(y, X, offset=None) -> dict:
    """Unpenalised logistic GLM (statsmodels); never raises. Separation -> ok=False."""
    import statsmodels.api as sm
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    k = X.shape[1]
    out = {"ok": False, "params": np.full(k, np.nan), "bse": np.full(k, np.nan),
           "ci": np.full((k, 2), np.nan), "converged": False, "note": "", "n": int(len(y)),
           "events": int(np.nansum(y))}
    if len(y) < 2 * k + 2 or out["events"] < 1 or out["events"] > len(y) - 1:
        out["note"] = f"degenerate (n={len(y)}, events={out['events']})"
        return out
    if np.linalg.matrix_rank(X) < k:
        out["note"] = "rank-deficient design"
        return out
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            res = sm.GLM(y, X, family=sm.families.Binomial(), offset=offset).fit(maxiter=200)
        except Exception as exc:  # PerfectSeparationError (older statsmodels), LinAlgError ...
            out["note"] = f"{type(exc).__name__}: {str(exc)[:80]}"
            return out
    notes = sorted({w.category.__name__ for w in caught})
    params = np.asarray(res.params, dtype=np.float64)
    separated = any("Separation" in n for n in notes)
    out.update(params=params, bse=np.asarray(res.bse, dtype=np.float64),
               ci=np.asarray(res.conf_int(alpha=0.05), dtype=np.float64),
               converged=bool(getattr(res, "converged", True)), note=";".join(notes))
    out["ok"] = bool(np.all(np.isfinite(params)) and not separated)
    return out


@dataclass
class Recalibrator:
    method: str
    ok: bool
    fn: Callable | None = None
    terms: list = field(default_factory=list)       # [(term, est, se, lo, hi)]
    n_fit: int = 0
    events_fit: int = 0
    converged: bool = True
    note: str = ""
    extra: dict = field(default_factory=dict)

    def apply(self, p: np.ndarray, hfpef: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        if not self.ok or self.fn is None:
            return np.full_like(p, np.nan)
        return np.asarray(self.fn(p, np.asarray(hfpef, dtype=np.float64)), dtype=np.float64)

    def term(self, name: str) -> tuple:
        for t in self.terms:
            if t[0] == name:
                return t
        return (name, np.nan, np.nan, np.nan, np.nan)


def _glm_terms(fit: dict, names: list[str]) -> list[tuple]:
    return [(nm, float(fit["params"][i]), float(fit["bse"][i]), float(fit["ci"][i, 0]), float(fit["ci"][i, 1]))
            for i, nm in enumerate(names)]


def fit_recalibrators(p_val: np.ndarray, y_val: np.ndarray, h_val: np.ndarray, eps: float) -> "OrderedDict":
    from sklearn.isotonic import IsotonicRegression
    lp = logit(p_val, eps)
    n, ev = int(len(y_val)), int(y_val.sum())
    rc: OrderedDict = OrderedDict()
    rc["raw"] = Recalibrator("raw", True, lambda p, h: p, n_fit=n, events_fit=ev)

    f = glm_binomial(y_val, np.column_stack([np.ones(n), lp]))
    a, b = f["params"]
    rc["logistic"] = Recalibrator("logistic", f["ok"], (lambda p, h, a=a, b=b: expit(a + b * logit(p, eps))),
                                  _glm_terms(f, ["a_intercept", "b_slope"]), n, ev, f["converged"], f["note"])

    f = glm_binomial(y_val, np.ones((n, 1)), offset=lp)
    a0 = f["params"][0]
    rc["intercept"] = Recalibrator("intercept", f["ok"], (lambda p, h, a=a0: expit(a + logit(p, eps))),
                                   _glm_terms(f, ["a_intercept_offset"]), n, ev, f["converged"], f["note"])

    iso_ok, iso_note, iso = False, "", None
    if 0 < ev < n:
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        iso.fit(np.asarray(p_val, dtype=np.float64), np.asarray(y_val, dtype=np.float64))
        iso_ok = True
    else:
        iso_note = f"degenerate (n={n}, events={ev})"
    extra = {}
    if iso_ok:
        yt = np.asarray(iso.y_thresholds_, dtype=np.float64)
        extra = {"n_levels": int(len(np.unique(yt))), "min_fitted": float(yt.min()),
                 "max_fitted": float(yt.max()), "x_min": float(np.min(iso.X_thresholds_)),
                 "x_max": float(np.max(iso.X_thresholds_))}
    rc["isotonic"] = Recalibrator("isotonic", iso_ok, (lambda p, h, m=iso: m.predict(p)) if iso_ok else None,
                                  [("n_levels", extra.get("n_levels", np.nan), np.nan, np.nan, np.nan),
                                   ("min_fitted", extra.get("min_fitted", np.nan), np.nan, np.nan, np.nan),
                                   ("max_fitted", extra.get("max_fitted", np.nan), np.nan, np.nan, np.nan)],
                                  n, ev, True, iso_note, extra)

    f = glm_binomial(y_val, np.column_stack([np.ones(n), lp, h_val]))
    a2, b2, c2 = f["params"]
    note = f["note"]
    if int(h_val.sum()) == 0:
        note = (note + ";" if note else "") + "no HFpEF stays in the validation set"
    rc["logistic_hfpef"] = Recalibrator(
        "logistic_hfpef", f["ok"], (lambda p, h, a=a2, b=b2, c=c2: expit(a + b * logit(p, eps) + c * h)),
        _glm_terms(f, ["a_intercept", "b_slope", "c_hfpef"]), n, ev, f["converged"], note,
        {"n_hfpef_val": int(h_val.sum()), "events_hfpef_val": int(y_val[h_val > 0].sum())})
    return rc


def calib_clip(method: str, args: argparse.Namespace) -> float:
    """Clip for the calibration GLMs: --calib-eps only for isotonic (exact 0/1 fitted values); a
    coarse 1e-3 clip on the continuous maps would distort their slope (and the by-construction
    0 / 1 of the logistic map on the validation set)."""
    return float(args.calib_eps) if method == "isotonic" else float(args.eps)


def calib_metrics(y: np.ndarray, p: np.ndarray, eps: float) -> dict:
    """Calibration of predictions p for outcomes y (usual-care arm only)."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(y)
    y, p = y[ok], p[ok]
    n = int(len(y))
    nan = float("nan")
    out = {"n": n, "events": int(y.sum()) if n else 0, "obs_pct": nan, "pred_pct": nan, "o_minus_e_pp": nan,
           "oe_ratio": nan, "citl": nan, "citl_lo": nan, "citl_hi": nan, "slope": nan, "slope_lo": nan,
           "slope_hi": nan, "brier": nan, "brier_scaled": nan, "auc": nan, "note": ""}
    if n == 0:
        out["note"] = "empty"
        return out
    obs, pred = float(y.mean()), float(p.mean())
    out.update(obs_pct=100 * obs, pred_pct=100 * pred, o_minus_e_pp=100 * (obs - pred),
               oe_ratio=obs / pred if pred > 0 else nan, brier=float(np.mean((y - p) ** 2)))
    if 0 < obs < 1:
        out["brier_scaled"] = 1.0 - out["brier"] / (obs * (1 - obs))
        out["auc"] = float(roc_auc_score(y, p))
    lp = logit(p, eps)
    f_c = glm_binomial(y, np.ones((n, 1)), offset=lp)
    if f_c["ok"]:
        out.update(citl=float(f_c["params"][0]), citl_lo=float(f_c["ci"][0, 0]), citl_hi=float(f_c["ci"][0, 1]))
    f_s = glm_binomial(y, np.column_stack([np.ones(n), lp]))
    if f_s["ok"]:
        out.update(slope=float(f_s["params"][1]), slope_lo=float(f_s["ci"][1, 0]), slope_hi=float(f_s["ci"][1, 1]))
    out["note"] = ";".join(x for x in (f_c["note"], f_s["note"]) if x)
    return out


def calib_groups(y: np.ndarray, base_p: np.ndarray, preds: dict, n_groups: int) -> list[dict]:
    """Observed vs mean predicted by groups of the RAW usual-care risk (same groups for every method)."""
    n = len(y)
    if n == 0:
        return []
    g = min(int(n_groups), n)
    grp = pd.qcut(pd.Series(base_p).rank(method="first"), g, labels=False).to_numpy()
    rows = []
    for k in range(g):
        m = grp == k
        r = {"group": k + 1, "n_groups": g, "n": int(m.sum()), "events": int(y[m].sum()),
             "obs_pct": 100 * float(y[m].mean())}
        for meth, p in preds.items():
            r[f"pred_{meth}_pct"] = 100 * float(np.nanmean(p[m])) if np.isfinite(p[m]).any() else np.nan
        rows.append(r)
    return rows


# ═══════════════════════════════════════════════════════════════════════
# BM1 per-stay reuse
# ═══════════════════════════════════════════════════════════════════════
def load_bm1_perstay(path: Path, stay_ids: np.ndarray, ckpt_name: str, y_uc_new: np.ndarray,
                     mort_new: np.ndarray, tol: float,
                     y_na_new: np.ndarray | None = None) -> tuple[pd.DataFrame | None, dict]:
    """Aligned BM1 arm predictions (dataset order) or None with the rejection reason.

    Besides y_uc, the stored non-adaptive rule arm is compared with a fresh rollout
    (y_na_new): a BM1 file produced before the 70c rule sidecar / rule config was
    regenerated matches on usual care but not on the rule arm and must be rejected.
    """
    info = {"path": str(path), "accepted": False, "reason": "", "uc_max_abs_diff": float("nan"),
            "na_max_abs_diff": float("nan"), "meta_quick": None, "meta_checkpoint": None}
    if not path.is_file():
        info["reason"] = "BM1 per-stay file not found"
        return None, info
    meta_path = path.with_suffix(".meta.json")
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            info["meta_quick"] = meta.get("quick")
            info["meta_checkpoint"] = meta.get("checkpoint")
        except (OSError, ValueError):
            meta = {}
        if info["meta_checkpoint"] and Path(str(info["meta_checkpoint"])).name != ckpt_name:
            info["reason"] = f"checkpoint differs (BM1 {Path(str(info['meta_checkpoint'])).name} vs {ckpt_name})"
            return None, info
    df = pd.read_parquet(path)
    need = ["stay_id", "mortality_28d"] + [c for c, _ in ARMS.values()]
    miss = [c for c in need if c not in df.columns]
    if miss:
        info["reason"] = f"columns missing: {miss}"
        return None, info
    if df["stay_id"].duplicated().any():
        info["reason"] = "duplicated stay_id"
        return None, info
    have = set(df["stay_id"].tolist())
    n_missing = int(sum(1 for s in stay_ids.tolist() if s not in have))
    if n_missing:
        info["reason"] = (f"BM1 file lacks {n_missing} of {len(stay_ids):,} test stays "
                          f"(BM1 n={len(df):,}; BM1 quick={info['meta_quick']})")
        return None, info
    al = df.set_index("stay_id").reindex(stay_ids)
    diff = float(np.max(np.abs(al["y_uc"].to_numpy(dtype=np.float64) - y_uc_new))) if len(al) else 0.0
    info["uc_max_abs_diff"] = diff
    if not np.isfinite(diff) or diff > tol:
        info["reason"] = f"usual-care predictions differ (max |diff|={diff:.3g} > {tol:g})"
        return None, info
    if y_na_new is not None:
        dna = float(np.max(np.abs(al["y_rule_nonadaptive"].to_numpy(dtype=np.float64) - y_na_new))) \
            if len(al) else 0.0
        info["na_max_abs_diff"] = dna
        if not np.isfinite(dna) or dna > tol:
            info["reason"] = (f"non-adaptive rule predictions differ (max |diff|={dna:.3g} > {tol:g}; "
                              f"70c rule actions changed since BM1?)")
            return None, info
    if not np.array_equal(al["mortality_28d"].to_numpy(dtype=np.float64), mort_new):
        info["reason"] = "mortality_28d differs"
        return None, info
    if al[[c for c, _ in ARMS.values()]].isna().any().any():
        info["reason"] = "NaN arm predictions"
        return None, info
    info["accepted"] = True
    if len(df) > len(stay_ids):
        info["reason"] = f"subset of BM1 file used ({len(stay_ids):,} of {len(df):,} stays)"
    return al.reset_index(), info


# ═══════════════════════════════════════════════════════════════════════
# Formatting helpers
# ═══════════════════════════════════════════════════════════════════════
def ci_str(row: pd.Series, d: int = 3) -> str:
    return common.fmt_ci(row["estimate"], row["ci_lo"], row["ci_hi"], d)


def ci_excludes_zero(row: pd.Series) -> bool:
    lo, hi = float(row["ci_lo"]), float(row["ci_hi"])
    return bool(np.isfinite(lo) and np.isfinite(hi) and (hi < 0 or lo > 0))


def same_sign(a: float, b: float) -> bool:
    return bool(np.isfinite(a) and np.isfinite(b) and np.sign(a) == np.sign(b) and a != 0)


def ratio(a: float, b: float) -> float:
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and b != 0 else float("nan")


def pcol(arm: str, meth: str) -> str:
    return f"p_{arm}_{meth}"


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main(argv=None) -> int:
    t_start = time.time()
    args = common.apply_quick(parse_args(argv))
    torch.set_num_threads(max(1, int(args.threads)))
    torch.manual_seed(args.seed)
    print("=" * 72, flush=True)
    print(f"BM5 — Recalibrated outcome model and recalibrated contrast (v{SCRIPT_VERSION})", flush=True)
    print("=" * 72, flush=True)

    # ── 1. Data ──────────────────────────────────────────────────────
    t0 = phase(1, N_PHASES, "Load data, split and training action vocabulary")
    root = common.find_root(__file__)
    paths = common.setup_paths(root)
    log(f"  PROJECT_ROOT={root}")
    bundle = common.load_all(root)
    vocab = common.training_vocab(bundle, method="dataset")
    val_ids = np.sort(np.asarray(bundle.val_ids))
    test_ids = np.sort(np.asarray(bundle.test_ids))
    if args.quick:
        val_ids = common.subsample_ids(val_ids, common.QUICK_N_STAYS, args.seed)
        test_ids = common.subsample_ids(test_ids, common.QUICK_N_STAYS, args.seed)
        log(f"  QUICK: validation subsample n={len(val_ids):,}; test subsample n={len(test_ids):,}")
    log(f"  validation stays: {len(val_ids):,}; test stays: {len(test_ids):,}  [{time.time() - t0:.1f}s]")

    # ── 2. Datasets, model, decoder ──────────────────────────────────
    t0 = phase(2, N_PHASES, "Validation / test DTRDatasets, checkpoint, rule decoder check")
    device = common.resolve_device(args.device)
    ds_val = common.build_dataset(bundle, val_ids, vocab, label="validation", strict_vocab=False)
    ds = common.build_dataset(bundle, test_ids, vocab, label="test", strict_vocab=not args.quick)
    ckpt_path = common.resolve_checkpoint(root, args.checkpoint)
    model, info = common.load_model(ckpt_path, ds, bundle.config, device)
    decoder = common.RuleDecoder.from_bundle(bundle, vocab)
    rule_chk = common.rule_reproduction_check(bundle, ds, decoder)
    log(f"  [{time.time() - t0:.1f}s]")

    # ── 3. Validation rollout and raw calibration ────────────────────
    t0 = phase(3, N_PHASES, "Validation rollout with observed actions (p_val) and raw calibration")
    p_val = common.predict_fixed(model, ds_val, device, "action", args.batch_size,
                                 desc="validation usual-care rollout")["prob"]
    fval = common.per_stay_frame(ds_val, bundle, p=p_val)
    y_val = fval["mortality_28d"].to_numpy(dtype=np.float64)
    h_val = (fval["hf_subtype"].astype(str) == "hfpef").to_numpy(dtype=np.float64)
    val_strata = common.standard_strata(fval, extended=False)
    log(f"  validation: n={len(y_val):,}, deaths={int(y_val.sum())}, observed={100 * y_val.mean():.2f}%, "
        f"mean model risk={100 * p_val.mean():.2f}%; HFpEF n={int(h_val.sum())}")
    log(f"  [{time.time() - t0:.1f}s]")

    # ── 4. Fit recalibration maps ────────────────────────────────────
    t0 = phase(4, N_PHASES, "Fit recalibration maps on the validation set")
    recal = fit_recalibrators(p_val, y_val, h_val, args.eps)
    model_rows = []
    for meth, rc in recal.items():
        if meth == "raw":
            continue
        for term, est, se, lo, hi in rc.terms:
            model_rows.append({"method": meth, "term": term, "estimate": est, "se": se, "ci_lo": lo,
                               "ci_hi": hi, "n_fit": rc.n_fit, "events_fit": rc.events_fit,
                               "ok": rc.ok, "converged": rc.converged, "note": rc.note})
        log(f"  {meth:15s} ok={rc.ok} " + "; ".join(
            f"{t[0]}={common.fmt_num(t[1], 4)}" + (f" ({common.fmt_num(t[3], 3)} to {common.fmt_num(t[4], 3)})"
                                                  if np.isfinite(t[3]) else "") for t in rc.terms)
            + (f"  [{rc.note}]" if rc.note else ""))
    common.save_table(pd.DataFrame(model_rows), paths["tables"] / "BM5_recalibration_models.csv")
    b_fit = recal["logistic"].term("b_slope")[1]
    slope_positive = bool(np.isfinite(b_fit) and b_fit > 0)
    if not recal["logistic"].ok:
        log(f"WARNING: the logistic recalibration map could not be estimated ({recal['logistic'].note}); "
            "all logistic-recalibrated quantities are NA.")
    elif not slope_positive:
        log("WARNING: the logistic recalibration slope b is not positive: the outcome model does not rank "
            "validation patients in the right order, and the map REVERSES every contrast. Do not interpret "
            "the logistic-recalibrated RDs (expected only on the synthetic fixture).")
    val_pred = {m: rc.apply(p_val, h_val) for m, rc in recal.items()}
    calib_rows, val_cal = [], {}
    for s in tqdm(list(val_strata), desc="  validation calibration", mininterval=5, leave=False):
        msk = val_strata[s]
        for m, pv in val_pred.items():
            cm = calib_metrics(y_val[msk], pv[msk], calib_clip(m, args))
            val_cal[(s, m)] = cm
            calib_rows.append({"dataset": "validation", "stratum": s, "method": m, **cm})
    v = val_cal[("all", "raw")]
    log(f"  validation raw: CITL {common.fmt_num(v['citl'], 3)}, slope {common.fmt_num(v['slope'], 3)}, "
        f"Brier {common.fmt_num(v['brier'], 4)}, AUC {common.fmt_num(v['auc'], 3)}; after logistic: "
        f"CITL {common.fmt_num(val_cal[('all', 'logistic')]['citl'], 3)}, slope "
        f"{common.fmt_num(val_cal[('all', 'logistic')]['slope'], 3)} (1 by construction)")
    log(f"  [{time.time() - t0:.1f}s]")

    # ── 5. Test-set arms ─────────────────────────────────────────────
    t0 = phase(5, N_PHASES, f"Test-set arm predictions (arms source: {args.arms_source})")
    y_uc_new = common.predict_fixed(model, ds, device, "action", args.batch_size,
                                    desc="test usual-care rollout")["prob"]
    # the stored-rule arm is also recomputed (1 fixed rollout) to validate a reused BM1 file
    y_na_new = common.predict_fixed(model, ds, device, "rule_action", args.batch_size,
                                    desc="rule non-adaptive")["prob"]
    mort_test = np.asarray(ds.mortality_28d, dtype=np.float64)
    arms: dict[str, np.ndarray] = {}
    bm1_info = {"accepted": False, "reason": "not attempted (--arms-source recompute)",
                "uc_max_abs_diff": float("nan"), "na_max_abs_diff": float("nan"), "path": ""}
    if args.arms_source in ("auto", "bm1"):
        bm1_path = Path(args.bm1_perstay)
        bm1_path = bm1_path if bm1_path.is_absolute() else root / bm1_path
        bm1_df, bm1_info = load_bm1_perstay(bm1_path, np.asarray(ds.stay_ids), ckpt_path.name,
                                            y_uc_new, mort_test, args.uc_tol, y_na_new=y_na_new)
        if bm1_df is not None:
            for k, (c, _) in ARMS.items():
                arms[k] = bm1_df[c].to_numpy(dtype=np.float64)
            log(f"  using BM1 per-stay predictions: {bm1_path} (max |y_uc diff|="
                f"{bm1_info['uc_max_abs_diff']:.2e}; max |y_rule_nonadaptive diff|="
                f"{bm1_info['na_max_abs_diff']:.2e})" + (f"; {bm1_info['reason']}" if bm1_info["reason"] else ""))
        else:
            msg = f"BM1 per-stay predictions not used: {bm1_info['reason']}"
            if args.arms_source == "bm1":
                raise RuntimeError(msg + " (--arms-source bm1)")
            log(f"  {msg}; recomputing all arms")
    arms_source = "bm1_perstay" if arms else "recomputed"
    if not arms:
        arms["uc"] = y_uc_new
        arms["na"] = y_na_new
        arms["zero"] = common.predict_fixed(model, ds, device, common.legacy_zero_actions(ds), args.batch_size,
                                            desc="legacy zero-action arm")["prob"]
        fb_rec = np.nan_to_num(common.stay_bin_matrix(bundle.state_df, ds.stay_ids,
                                                      "cumulative_fluid_balance_L", bundle.n_bins), nan=0.0)
        urine = common.stay_bin_matrix(bundle.state_df, ds.stay_ids, "urine_output_ml", bundle.n_bins)
        flv = common.fluid_ml_by_level(bundle)
        arms["ad"] = common.predict_adaptive(model, ds, decoder, device, "recorded", fb_matrix=fb_rec,
                                             batch_size=args.batch_size, desc="adaptive rule (fb recorded)")["prob"]
        arms["adfb"] = common.predict_adaptive(model, ds, decoder, device, "approx", fluid_ml_by_level=flv,
                                               urine_matrix=urine, batch_size=args.batch_size,
                                               desc="adaptive rule (fb approx)")["prob"]
    for k, (c, lab) in ARMS.items():
        log(f"  {k:5s} {lab:55s} mean risk {100 * arms[k].mean():.3f}%")
    log(f"  [{time.time() - t0:.1f}s]")

    # ── 6. Apply recalibration; test calibration ─────────────────────
    t0 = phase(6, N_PHASES, "Apply recalibration maps; test-set calibration before / after")
    frame = common.per_stay_frame(ds, bundle, **{c: arms[k] for k, (c, _) in ARMS.items()})
    h_test = (frame["hf_subtype"].astype(str) == "hfpef").to_numpy(dtype=np.float64)
    y_test = frame["mortality_28d"].to_numpy(dtype=np.float64)
    for m, rc in recal.items():
        for k in ARMS:
            frame[pcol(k, m)] = rc.apply(arms[k], h_test)
        frame[f"se_uc_{m}"] = (y_test - frame[pcol("uc", m)]) ** 2
        for rd, (a, b) in RD_SPECS.items():
            if rd in ("rd_na", "rd_ad"):
                frame[f"d_{rd}_{m}"] = frame[pcol(a, m)] - frame[pcol(b, m)]
    strata = common.standard_strata(frame, extended=True)
    iso_ties = {}
    for k in ("na", "ad"):
        a, b = frame[pcol(k, "isotonic")].to_numpy(), frame[pcol("uc", "isotonic")].to_numpy()
        raw_diff = arms[k] != arms["uc"]
        # share among stays with a non-zero raw contrast (not among all stays)
        iso_ties[k] = (100 * float(np.mean(a[raw_diff] == b[raw_diff]))
                       if np.isfinite(a).all() and raw_diff.any() else np.nan)
    log(f"  isotonic: test stays whose rule-vs-usual-care difference is flattened to exactly 0: "
        f"na {common.fmt_num(iso_ties['na'], 1)}%, ad {common.fmt_num(iso_ties['ad'], 1)}%")

    test_cal = {}
    for s in tqdm(list(strata), desc="  test calibration", mininterval=5, leave=False):
        msk = strata[s]
        for m in recal:
            cm = calib_metrics(y_test[msk], frame.loc[msk, pcol("uc", m)].to_numpy(), calib_clip(m, args))
            test_cal[(s, m)] = cm
            calib_rows.append({"dataset": "test", "stratum": s, "method": m, **cm})
    common.save_table(pd.DataFrame(calib_rows), paths["tables"] / "BM5_calibration_metrics.csv")
    for s in CALIB_PASTE_STRATA:
        for m in recal:
            c = test_cal[(s, m)]
            log(f"  test {s:6s} {m:15s} pred {common.fmt_num(c['pred_pct'])}% vs obs {common.fmt_num(c['obs_pct'])}%; "
                f"CITL {common.fmt_num(c['citl'], 3)}; slope {common.fmt_num(c['slope'], 3)}; "
                f"Brier {common.fmt_num(c['brier'], 4)}")
    grp_rows = []
    for s, g in (("all", 10), ("hfpef", 5)):
        msk = strata[s]
        preds = {m: frame.loc[msk, pcol("uc", m)].to_numpy() for m in recal}
        for r in calib_groups(y_test[msk], arms["uc"][msk], preds, g):
            grp_rows.append({"dataset": "test", "stratum": s, **r})
    common.save_table(pd.DataFrame(grp_rows), paths["tables"] / "BM5_calibration_groups.csv")
    if not args.no_save_perstay:
        keep = ["stay_id", "mortality_28d", "hf_subtype", "rule_phenotype_2class"] + \
               [pcol(k, m) for m in recal for k in ARMS]
        common.save_interim(frame[keep], root, PERSTAY_FILE)
    log(f"  [{time.time() - t0:.1f}s]")

    # ── 7. Paired bootstrap ──────────────────────────────────────────
    t0 = phase(7, N_PHASES, f"Paired bootstrap of recalibrated risks and RDs ({args.n_boot} reps; "
                            f"recalibration fixed)")
    value_cols = ["mortality_28d"] + [pcol(k, m) for m in recal for k in ARMS] + [f"se_uc_{m}" for m in recal]
    contrasts = []
    for m in recal:
        for rd, (a, b) in RD_SPECS.items():
            contrasts.append((f"{rd}_{m}", pcol(a, m), pcol(b, m)))
        contrasts.append((f"gap_{m}", "mortality_28d", pcol("uc", m)))
        if m != "raw":
            contrasts.append((f"rdchange_na_{m}", f"d_rd_na_{m}", "d_rd_na_raw"))
            contrasts.append((f"rdchange_ad_{m}", f"d_rd_ad_{m}", "d_rd_ad_raw"))
            contrasts.append((f"brierchange_{m}", f"se_uc_{m}", "se_uc_raw"))
    res = common.paired_bootstrap(frame, value_cols, strata, n_boot=args.n_boot, seed=args.seed,
                                  contrasts=contrasts, scale=100.0, desc="paired bootstrap")
    common.save_table(res, paths["tables"] / "BM5_bootstrap_long.csv")

    def E(s, q, kind="diff"):
        return common.get_est(res, s, q, kind)

    wide_rows = []
    for s, msk in strata.items():
        for m in recal:
            row = OrderedDict(stratum=s, method=m, n=int(msk.sum()), deaths=int(y_test[msk].sum()),
                              obs_mort_pct=E(s, "mortality_28d", "mean")["estimate"])
            for k in ARMS:
                row[f"risk_{k}_pct"] = E(s, pcol(k, m), "mean")["estimate"]
            for q in [f"gap_{m}"] + [f"{rd}_{m}" for rd in RD_SPECS] + \
                     ([f"rdchange_na_{m}", f"rdchange_ad_{m}", f"brierchange_{m}"] if m != "raw" else []):
                r = E(s, q)
                base = q[: -len(m) - 1]
                row[f"{base}_pp"] = r["estimate"]
                row[f"{base}_lo"] = r["ci_lo"]
                row[f"{base}_hi"] = r["ci_hi"]
            row["rd_na_ratio_to_raw"] = ratio(row["rd_na_pp"], E(s, "rd_na_raw")["estimate"])
            row["rd_ad_ratio_to_raw"] = ratio(row["rd_ad_pp"], E(s, "rd_ad_raw")["estimate"])
            c = test_cal[(s, m)]
            row.update(citl=c["citl"], slope=c["slope"], brier=c["brier"], auc=c["auc"])
            wide_rows.append(row)
    wide = pd.DataFrame(wide_rows)
    common.save_table(wide, paths["tables"] / "BM5_strata_summary.csv")
    log(f"  logistic RD (rule non-adaptive - UC), all: {ci_str(E('all', 'rd_na_logistic'))} pp "
        f"(raw {ci_str(E('all', 'rd_na_raw'))}); adaptive: {ci_str(E('all', 'rd_ad_logistic'))} pp")
    log(f"  [{time.time() - t0:.1f}s]")

    # ── 8. Reproduction checks and report ────────────────────────────
    t0 = phase(8, N_PHASES, "Reproduction checks, report and paste-back block")
    rc_chk = common.repro_check_extended(
        E("all", pcol("uc", "raw"), "mean")["estimate"], E("hfpef", pcol("uc", "raw"), "mean")["estimate"],
        E("all", "rd_na_raw")["estimate"], E("hfpef", "rd_na_raw")["estimate"],
        E("all", "rd_zero_raw")["estimate"], E("hfpef", "rd_zero_raw")["estimate"])
    tgt = common.REPRO_TARGETS
    n_hf = int(strata["hfpef"].sum())
    n_ok = bool(len(ds) == tgt["n_test"] and n_hf == tgt["n_hfpef"])
    n_line = (f"REPRO_CHECK_N: {'PASS' if n_ok else 'FAIL'} (n_test={len(ds):,} vs {tgt['n_test']:,}; "
              f"n_hfpef={n_hf} vs {tgt['n_hfpef']})")
    print(n_line, flush=True)
    lg = recal["logistic"]
    a_t, b_t = lg.term("a_intercept"), lg.term("b_slope")
    i_t = recal["intercept"].term("a_intercept_offset")
    hf_rc = recal["logistic_hfpef"]
    runtime_min = (time.time() - t_start) / 60.0

    L = common.report_header("BM5 — Recalibration of the outcome model and the recalibrated contrast "
                             "(BMC Medicine revision)", root, args)
    L += [f"Script version {SCRIPT_VERSION}; device {device}; torch threads {torch.get_num_threads()}; "
          f"runtime {runtime_min:.1f} min" + ("  [QUICK SMOKE TEST — not for interpretation]" if args.quick else ""),
          f"Checkpoint: {ckpt_path.name} (epoch {info['epoch']}, val_loss {common.fmt_num(info['val_loss'], 4)}, "
          f"{info['n_params']:,} parameters); training action vocab ({len(vocab)}): {vocab}",
          f"Arms source: {arms_source}"
          + (f" ({bm1_info.get('path', '')}; max |y_uc(BM1) - y_uc(here)| = {bm1_info['uc_max_abs_diff']:.2e}; "
             f"max |y_rule_nonadaptive(BM1) - here| = {bm1_info['na_max_abs_diff']:.2e})"
             if arms_source == "bm1_perstay" else f" (BM1 file not used: {bm1_info['reason']})"),
          f"Rule decoder on recorded test states: {rule_chk['status']} (cumulative-FB agreement "
          f"{common.fmt_num(100 * rule_chk['cumulative_fluid_balance_L']['agreement'], 3)}%)",
          "",
          "Arms:"] + [f"  {k:5s} {c:26s} {lab}" for k, (c, lab) in ARMS.items()] + [
          "Recalibration methods:"] + [f"  {m:15s} {d}" for m, d in METHODS.items()] + [
          "",
          f"Validation set (fit of the maps): n={len(y_val):,}, deaths={int(y_val.sum())}, observed "
          f"{common.fmt_num(100 * y_val.mean())}%, mean model risk (usual-care rollout) "
          f"{common.fmt_num(100 * p_val.mean())}%; HFpEF n={int(h_val.sum())}, deaths "
          f"{int(y_val[h_val > 0].sum())}",
          f"  raw calibration on validation: CITL {common.fmt_ci(v['citl'], v['citl_lo'], v['citl_hi'], 3)}; "
          f"slope {common.fmt_ci(v['slope'], v['slope_lo'], v['slope_hi'], 3)}; Brier {common.fmt_num(v['brier'], 4)}; "
          f"AUC {common.fmt_num(v['auc'], 3)}",
          f"  logistic map: a = {common.fmt_ci(a_t[1], a_t[3], a_t[4], 4)}, b = {common.fmt_ci(b_t[1], b_t[3], b_t[4], 4)}"
          f" (ok={lg.ok}{'; ' + lg.note if lg.note else ''})",
          f"  intercept-only map: a = {common.fmt_ci(i_t[1], i_t[3], i_t[4], 4)}",
          "  logistic + HFpEF map: " + ", ".join(
              f"{t[0][0]} = {common.fmt_ci(t[1], t[3], t[4], 4)}" for t in hf_rc.terms)
          + f" (ok={hf_rc.ok}; HFpEF in validation n={hf_rc.extra.get('n_hfpef_val')}, deaths "
            f"{hf_rc.extra.get('events_hfpef_val')}{'; ' + hf_rc.note if hf_rc.note else ''})",
          f"  isotonic map: {recal['isotonic'].extra.get('n_levels', 'NA')} distinct levels, fitted range "
          f"{common.fmt_num(100 * recal['isotonic'].extra.get('min_fitted', np.nan))}-"
          f"{common.fmt_num(100 * recal['isotonic'].extra.get('max_fitted', np.nan))}%; test stays whose "
          f"rule-vs-UC difference is flattened to 0: na {common.fmt_num(iso_ties['na'], 1)}%, "
          f"ad {common.fmt_num(iso_ties['ad'], 1)}%",
          "",
          "Test-set calibration of the usual-care risk (observed actions) before / after recalibration:",
          f"  {'stratum':16s} {'method':15s} {'n':>6s} {'obs%':>6s} {'pred%':>6s} {'O-E pp':>7s} {'O/E':>5s} "
          f"{'CITL (95% CI)':>24s} {'slope (95% CI)':>24s} {'Brier':>7s} {'sBrier':>6s} {'AUC':>5s}"]
    for s in SPEC_STRATA:
        for m in recal:
            c = test_cal[(s, m)]
            L.append(f"  {s:16s} {m:15s} {c['n']:6d} {common.fmt_num(c['obs_pct']):>6s} "
                     f"{common.fmt_num(c['pred_pct']):>6s} {common.fmt_num(c['o_minus_e_pp']):>7s} "
                     f"{common.fmt_num(c['oe_ratio']):>5s} "
                     f"{common.fmt_ci(c['citl'], c['citl_lo'], c['citl_hi'], 3):>24s} "
                     f"{common.fmt_ci(c['slope'], c['slope_lo'], c['slope_hi'], 3):>24s} "
                     f"{common.fmt_num(c['brier'], 4):>7s} {common.fmt_num(c['brier_scaled'], 3):>6s} "
                     f"{common.fmt_num(c['auc'], 3):>5s}")
    L += ["", "Test-set calibration by decile of the raw usual-care risk (all; observed % | predicted % raw / "
          "logistic / isotonic):"]
    for r in [r for r in grp_rows if r["stratum"] == "all"]:
        L.append(f"  decile {r['group']:2d} n={r['n']:5d}: obs {common.fmt_num(r['obs_pct'])} | "
                 f"{common.fmt_num(r['pred_raw_pct'])} / {common.fmt_num(r['pred_logistic_pct'])} / "
                 f"{common.fmt_num(r['pred_isotonic_pct'])}")
    L += ["", "Risks (%) per stratum and method (obs = observed 28-day mortality):",
          f"  {'stratum':16s} {'method':15s} {'n':>6s} {'deaths':>6s} {'obs%':>6s} {'UC%':>6s} {'R-na%':>6s} "
          f"{'R-ad%':>6s} {'R-adFB%':>7s} {'zero%':>6s}"]
    for s in SPEC_STRATA:
        for m in recal:
            w = wide[(wide["stratum"] == s) & (wide["method"] == m)].iloc[0]
            L.append(f"  {s:16s} {m:15s} {int(w['n']):6d} {int(w['deaths']):6d} {common.fmt_num(w['obs_mort_pct']):>6s} "
                     f"{common.fmt_num(w['risk_uc_pct']):>6s} {common.fmt_num(w['risk_na_pct']):>6s} "
                     f"{common.fmt_num(w['risk_ad_pct']):>6s} {common.fmt_num(w['risk_adfb_pct']):>7s} "
                     f"{common.fmt_num(w['risk_zero_pct']):>6s}")
    L.append("  (extended strata: tables/bmc_medicine/BM5_strata_summary.csv)")
    L += ["", "Risk differences (pp, paired bootstrap 95% CI; negative favours the rule; gap = observed - "
          "predicted usual-care risk;", "rdchange = recalibrated RD - raw RD):"]
    for q in ["gap"] + list(RD_SPECS) + ["rdchange_na", "rdchange_ad"]:
        meths = [m for m in recal if not (q.startswith("rdchange") and m == "raw")]
        L.append(f"  {q:16s}" + "".join(f" {m:>29s}" for m in meths))
        for s in SPEC_STRATA:
            L.append(f"    {s:14s}" + "".join(f" {ci_str(E(s, f'{q}_{m}')):>29s}" for m in meths))
    L += ["", "Ratio of recalibrated to raw RD (point estimates; 1 = unchanged), rule non-adaptive / adaptive:",
          f"  {'stratum':16s}" + "".join(f"{m:>17s}" for m in recal if m != "raw")]
    for s in SPEC_STRATA:
        cells = []
        for m in recal:
            if m == "raw":
                continue
            w = wide[(wide["stratum"] == s) & (wide["method"] == m)].iloc[0]
            cells.append(f"{common.fmt_num(w['rd_na_ratio_to_raw'])} / {common.fmt_num(w['rd_ad_ratio_to_raw'])}")
        L.append(f"  {s:16s}" + "".join(f"{c:>17s}" for c in cells))
    L += ["", "HFpEF-specific signal (non-shock RD, HFpEF minus no-HF), per method:"]
    for m in recal:
        L.append(f"  {m:15s} rd_na {ci_str(E(DOD_NAME, f'rd_na_{m}', 'diff_of_diff'))}; "
                 f"rd_ad {ci_str(E(DOD_NAME, f'rd_ad_{m}', 'diff_of_diff'))}")
    L += ["", "Brier score change of the usual-care risk (x100; negative = better than raw):"]
    for s in CALIB_PASTE_STRATA:
        L.append(f"  {s:16s} " + "; ".join(f"{m} {ci_str(E(s, f'brierchange_{m}'), 4)}"
                                           for m in recal if m != "raw"))
    L += ["", "Reproduction of the published non-adaptive analysis (raw predictions):", "  " + n_line,
          "  " + rc_chk["line"], "  " + rc_chk["uc_line"]]
    if "legacy_line" in rc_chk:
        L.append("  " + rc_chk["legacy_line"])
    if rc_chk.get("legacy_status") == "PASS" and not rc_chk["pass"]:
        L.append("  REPRO_NOTE: " + common.LEGACY_RULE_NOTE)
    L += ["",
          "Notes:",
          "  * The maps are fitted ONCE on the validation set (usual-care rollout vs observed mortality) and",
          "    applied unchanged to every test arm; the bootstrap resamples test patients only (maps fixed),",
          "    so the CIs do not include the uncertainty of a, b (see the Wald CIs above).",
          "  * Increasing maps (logistic with b > 0, intercept-only, isotonic) cannot change the sign of a",
          "    per-patient contrast; they rescale absolute RDs. A logistic b <= 0 reverses every contrast.",
          "    The logistic map rescales an RD by about b * p'(1-p') / (p(1-p)) at the patient's risk level.",
          "  * Isotonic predictions are piecewise linear with flat segments; per-patient contrasts falling in",
          "    a flat segment become exactly 0, so isotonic RDs are attenuated by construction.",
          "  * Calibration intercept = calibration-in-the-large (GLM with offset logit p; 0 = ideal); slope =",
          "    GLM coefficient of logit p (1 = ideal; < 1 = predictions too extreme). Wald 95% CIs; p clipped",
          "    to [eps, 1 - eps] before logit (isotonic: [calib_eps, 1 - calib_eps], so isotonic slopes are",
          "    approximate).",
          "  * Calibration is assessable only for the usual-care arm (outcomes were observed under usual care);",
          "    applying the map to the counterfactual arms assumes the same miscalibration function.",
          f"  * Stay-level recalibrated predictions: data/interim/bmc_medicine/{PERSTAY_FILE} (never sync)."]

    # ── paste-back block ──
    P = OrderedDict()
    P["script"] = "BM5_recalibrated_contrast"
    P["version"] = SCRIPT_VERSION
    P["quick"] = bool(args.quick)
    P["checkpoint"] = ckpt_path.name
    P["ckpt_epoch"] = info["epoch"]
    P["ckpt_val_loss"] = info["val_loss"] if info["val_loss"] is not None else float("nan")
    P["device"] = str(device)
    P["n_boot"] = int(args.n_boot)
    P["seed"] = int(args.seed)
    P["arms_source"] = arms_source
    P["bm1_uc_max_abs_diff"] = f"{bm1_info['uc_max_abs_diff']:.2e}" if np.isfinite(bm1_info["uc_max_abs_diff"]) \
        else "NA"
    P["bm1_note"] = bm1_info["reason"] or "none"
    P["RULE_DECODER_CHECK"] = rule_chk["status"]
    # validation set and fitted maps
    P["n_val"] = int(len(y_val))
    P["val_deaths"] = int(y_val.sum())
    P["val_obs_mort_pct"] = 100 * float(y_val.mean())
    P["val_risk_uc_raw_pct"] = 100 * float(p_val.mean())
    P["val_n_hfpef"] = int(h_val.sum())
    P["val_hfpef_deaths"] = int(y_val[h_val > 0].sum())
    vh = val_cal[("hfpef", "raw")]
    P["val_hfpef_obs_mort_pct"] = vh["obs_pct"]
    P["val_hfpef_risk_uc_raw_pct"] = vh["pred_pct"]
    P["val_raw_citl"] = common.fmt_ci(v["citl"], v["citl_lo"], v["citl_hi"], 3)
    P["val_raw_slope"] = common.fmt_ci(v["slope"], v["slope_lo"], v["slope_hi"], 3)
    P["val_raw_brier"] = v["brier"]
    P["val_raw_auc"] = v["auc"]
    P["recal_logistic_ok"] = lg.ok
    P["recal_logistic_a"] = common.fmt_ci(a_t[1], a_t[3], a_t[4], 4)
    P["recal_logistic_b"] = common.fmt_ci(b_t[1], b_t[3], b_t[4], 4)
    P["flag_recal_logistic_slope_positive"] = slope_positive
    vl = val_cal[("all", "logistic")]
    P["val_after_logistic_citl_slope"] = f"{common.fmt_num(vl['citl'], 3)}/{common.fmt_num(vl['slope'], 3)}"
    P["recal_intercept_a"] = common.fmt_ci(i_t[1], i_t[3], i_t[4], 4)
    P["recal_logistic_hfpef_ok"] = hf_rc.ok
    for t in hf_rc.terms:
        P[f"recal_logistic_hfpef_{t[0][0]}"] = common.fmt_ci(t[1], t[3], t[4], 4)
    P["recal_logistic_hfpef_note"] = hf_rc.note or "none"
    P["recal_isotonic_n_levels"] = recal["isotonic"].extra.get("n_levels", "NA")
    P["recal_isotonic_flattened_na_pct"] = iso_ties["na"]
    P["recal_isotonic_flattened_ad_pct"] = iso_ties["ad"]
    # test calibration of the usual-care risk
    for s in CALIB_PASTE_STRATA:
        P[f"test_{s}_auc_raw"] = test_cal[(s, "raw")]["auc"]
        for m in recal:
            c = test_cal[(s, m)]
            P[f"test_{s}_{m}_pred_pct"] = c["pred_pct"]
            P[f"test_{s}_{m}_citl"] = common.fmt_ci(c["citl"], c["citl_lo"], c["citl_hi"], 3)
            P[f"test_{s}_{m}_slope"] = common.fmt_ci(c["slope"], c["slope_lo"], c["slope_hi"], 3)
            P[f"test_{s}_{m}_brier"] = f"{c['brier']:.5f}" if np.isfinite(c["brier"]) else "NA"
        for m in ("logistic", "isotonic"):
            P[f"test_{s}_brierchange_{m}_x100"] = ci_str(E(s, f"brierchange_{m}"), 4)
    # strata: n, deaths, observed mortality
    for s in SPEC_STRATA:
        w = wide[(wide["stratum"] == s) & (wide["method"] == "raw")].iloc[0]
        P[f"{s}_n"] = int(w["n"])
        P[f"{s}_deaths"] = int(w["deaths"])
        P[f"{s}_obs_mort_pct"] = float(w["obs_mort_pct"])
    # raw and primary (logistic) contrasts, all spec strata
    for m, qs in (("raw", ["gap", "rd_na", "rd_ad"]),
                  ("logistic", ["gap", "rd_na", "rd_ad", "ad_minus_na", "rd_zero", "rdchange_na", "rdchange_ad"])):
        for s in SPEC_STRATA:
            P[f"{s}_{m}_risk_uc_pct"] = float(E(s, pcol("uc", m), "mean")["estimate"])
            for q in qs:
                P[f"{s}_{m}_{q}_pp"] = ci_str(E(s, f"{q}_{m}"))
    for s in ("all", "hfpef"):
        P[f"{s}_logistic_rd_adfb_pp"] = ci_str(E(s, "rd_adfb_logistic"))
    # secondary methods, main strata
    for m in SECONDARY_METHODS:
        for s in SECONDARY_STRATA:
            for q in ("gap", "rd_na", "rd_ad"):
                P[f"{s}_{m}_{q}_pp"] = ci_str(E(s, f"{q}_{m}"))
    # HFpEF-specific signal
    for m in recal:
        for q in ("rd_na", "rd_ad"):
            P[f"dod_nonshock_hfpef_minus_nohf_{q}_{m}_pp"] = ci_str(E(DOD_NAME, f"{q}_{m}", "diff_of_diff"))
    # attenuation and interpretation flags
    for s in ("all", "hfpef"):
        for q in ("rd_na", "rd_ad"):
            P[f"ratio_{q}_logistic_to_raw_{s}"] = ratio(E(s, f"{q}_logistic")["estimate"], E(s, f"{q}_raw")["estimate"])
    for s in ("all", "hfpef", "hfpef_nonshock"):
        for q in ("rd_na", "rd_ad"):
            P[f"flag_{q}_logistic_same_sign_as_raw_{s}"] = same_sign(E(s, f"{q}_logistic")["estimate"],
                                                                      E(s, f"{q}_raw")["estimate"])
            P[f"flag_{q}_logistic_ci_excludes_0_{s}"] = ci_excludes_zero(E(s, f"{q}_logistic"))
    for q in ("rd_na", "rd_ad"):
        P[f"flag_{q}_isotonic_same_sign_as_raw_all"] = same_sign(E("all", f"{q}_isotonic")["estimate"],
                                                                  E("all", f"{q}_raw")["estimate"])
        P[f"flag_dod_{q}_logistic_ci_excludes_0"] = ci_excludes_zero(E(DOD_NAME, f"{q}_logistic", "diff_of_diff"))
    # reproduction
    P["REPRO_CHECK_N"] = "PASS" if n_ok else "FAIL"
    P["REPRO_CHECK"] = rc_chk["status"]
    P["REPRO_CHECK_UC"] = rc_chk["uc_status"]
    P["REPRO_CHECK_LEGACY"] = rc_chk.get("legacy_status", "NA")
    P["repro_risk_uc_all_pct"] = float(E("all", pcol("uc", "raw"), "mean")["estimate"])
    P["repro_risk_uc_hfpef_pct"] = float(E("hfpef", pcol("uc", "raw"), "mean")["estimate"])
    P["repro_rd_rule_na_all_pp"] = float(E("all", "rd_na_raw")["estimate"])
    P["repro_rd_rule_na_hfpef_pp"] = float(E("hfpef", "rd_na_raw")["estimate"])
    P["repro_rd_legacy_all_pp"] = float(E("all", "rd_zero_raw")["estimate"])
    P["repro_rd_legacy_hfpef_pp"] = float(E("hfpef", "rd_zero_raw")["estimate"])
    P["runtime_min"] = runtime_min

    common.write_report(paths["reports"] / "BM5_summary.txt", L, P, TAG)
    log(f"  [{time.time() - t0:.1f}s]")
    log(f"BM5 done in {(time.time() - t_start) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
