#!/usr/bin/env python3
"""
BM1_adaptive_rollout — Adaptive g-computation rollout of the 70c rule and a
negative-exposure control under both rollouts (BMC Medicine revision, BM1).

Purpose
-------
The published Step C contrast fed the G-Transformer a FIXED rule-action
sequence that had been computed from the RECORDED states (non-adaptive). A
dynamic treatment regime has to respond to the states that the regime itself
produces. BM1 re-estimates the contrast with an adaptive g-computation
rollout: at every 2-h interval the 70c rule is applied to the MODEL-SIMULATED
state (MAP, lactate and SpO2 de-normalised from the predicted z-states), the
chosen action is fed back, and the transition head simulates the next state.

Arms (test set, same checkpoint, same pooling as predict_counterfactual_rollout):
  y_uc                      usual care (observed contiguous actions)
  y_rule_nonadaptive        stored 70c rule actions (published design, corrected code)
  y_rule_adaptive           adaptive rule, cumulative fluid balance = recorded (PRIMARY)
  y_rule_adaptive_fbapprox  adaptive rule, fluid balance re-accumulated from the
                            adaptive fluid levels (training-set median mL per level)
                            minus recorded urine output (SENSITIVITY)
  y_legacy_zero             action 0 everywhere = the 'rule' arm the pre-v1.0.1
                            DTRDataset actually fed the model (diagnostic; = BM4 S0)

Reported per stratum (all; HFpEF; HFpEF non-shock; HFpEF shock; no-HF
non-shock; HFrEF; plus extended strata in the CSV): n, deaths, observed
mortality, model risk per arm, RD vs usual care and adaptive minus
non-adaptive (paired patient-level bootstrap, percentile 95% CI, models
fixed), and the difference of non-shock RDs (HFpEF minus no-HF) with its CI.

Diagnostics:
  - rule-decision agreement adaptive vs non-adaptive (category and raw action),
    overall, per bin and per stratum; also the rule applied to the simulated
    USUAL-CARE trajectory (isolates transition-model drift from the response to
    rule treatment);
  - rule-category mix (perfusion support P1-P3 / default / diuretic P4-P5);
  - mean simulated MAP / lactate / SpO2 (raw units) by bin per arm, share of
    intervals with MAP < 65 and lactate > 2;
  - transition-model calibration under usual care: one-step-ahead
    (teacher-forced) MAE in z units for map_z / lactate_z / spo2_z by target bin
    on measured bins (mask == 1) and on all active bins, against a
    last-value-carried-forward baseline, plus the free-running rollout MAE;
  - negative-exposure control (85e NCE1 procedure): observed merged actions
    permuted within each bin across test stays (10 reps, RandomState(42) with
    the 85e stream position), RD(rule - permuted usual care) for the
    non-adaptive, adaptive and legacy arms; mean, SD (ddof=0, as 85e) and
    retained fraction = mean RD_perm / RD_true.
  - REPRO_CHECK (spec), REPRO_CHECK_UC, REPRO_CHECK_LEGACY, REPRO_CHECK_NCE_LEGACY
    and RULE_DECODER_CHECK lines.

Protocol reference: BMC Medicine revision analysis BM1 (extends Step 80c DM
rollout and Step 85e NCE1); spec bmc_medicine_revision SPEC.md.
Compute: G1 (single GPU via env/sge/submit_g1.sh) or C3 direct on CPU. Measured on
a full-size synthetic fixture (39,000 stays, 5,850 test stays, same architecture,
118 temporal features): the complete run (1000 bootstrap reps, 10 NCE reps) took
2.0 min on 4 CPU threads, peak RSS about 2 GB; each 5,850-stay rollout takes about
4 s on CPU, so the run is dominated by data loading and the 12 DTRDataset builds
(training vocab, test, 10 NCE permutations; about 4 s each for the test set).

Inputs
------
- configs/80_step_c_training_v1.yaml (+ 45c normalisation, 70c rule, 70b merge map)
- data/interim/45c_state_tensors_v1_mimic.parquet
- data/interim/45d_outcomes_absorb_v1_mimic.parquet
- data/interim/70b_propensity_coarse3_v1_mimic.parquet
- data/interim/70c_rule_policy_actions_v1_mimic.parquet (attached by load_dtr_data)
- data/processed/phenotype_labels_frozen_v1_mimic.parquet
- models/80c_gtransformer_rollout_ow1_best.pt (default; the checkpoint behind the
  published 80g / 85e numbers) or --checkpoint models/80c_gtransformer_best.pt (P2 export)

Outputs
-------
- tables/bmc_medicine/BM1_strata_summary.csv          (wide, one row per stratum)
- tables/bmc_medicine/BM1_bootstrap_long.csv          (all bootstrap estimates)
- tables/bmc_medicine/BM1_rule_agreement_by_bin.csv
- tables/bmc_medicine/BM1_rule_category_distribution.csv
- tables/bmc_medicine/BM1_state_trajectory_by_bin.csv
- tables/bmc_medicine/BM1_transition_error_by_bin.csv
- tables/bmc_medicine/BM1_nce_reps.csv, BM1_nce_summary.csv
- reports/bmc_medicine/BM1_summary.txt (ends with the PASTE-BACK BLOCK)
- data/interim/bmc_medicine/BM1_perstay_predictions.parquet (+ .meta.json)
  STAY-LEVEL, gitignored, never synced; read by BM3b and BM5.

Usage
-----
    # Shirokane, GPU (recommended)
    qsub -N sBM1_adaptive -q gpua.q -l a100=1 env/sge/submit_g1.sh \\
        scripts/bmc_medicine/BM1_adaptive_rollout.py \\
        --checkpoint models/80c_gtransformer_rollout_ow1_best.pt
    # smoke test
    env/run_py.sh scripts/bmc_medicine/BM1_adaptive_rollout.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import log, phase  # noqa: E402

import torch  # noqa: E402

SCRIPT_VERSION = "1.0"
TAG = "BM1"
N_PHASES = 9
PERSTAY_FILE = "BM1_perstay_predictions.parquet"
NCE_TOL_RD_PP = 0.05          # REPRO_CHECK_NCE_LEGACY tolerance on mean RD_perm (pp)
NCE_TOL_RETAINED_PCT = 3.0    # ... and on the retained fraction (percentage points)

ARM_COLS = OrderedDict([
    ("y_uc", "usual care (observed actions)"),
    ("y_rule_nonadaptive", "rule, non-adaptive (stored 70c actions)"),
    ("y_rule_adaptive", "rule, adaptive (recorded cumulative fluid balance) [PRIMARY]"),
    ("y_rule_adaptive_fbapprox", "rule, adaptive (approximate fluid balance) [SENSITIVITY]"),
    ("y_legacy_zero", "legacy zero-action arm (pre-v1.0.1 'rule'; = BM4 S0)"),
])
CONTRASTS = [
    ("rd_rule_na", "y_rule_nonadaptive", "y_uc"),
    ("rd_rule_ad", "y_rule_adaptive", "y_uc"),
    ("rd_rule_ad_fbapprox", "y_rule_adaptive_fbapprox", "y_uc"),
    ("rd_ad_minus_na", "y_rule_adaptive", "y_rule_nonadaptive"),
    ("rd_adapprox_minus_na", "y_rule_adaptive_fbapprox", "y_rule_nonadaptive"),
    ("rd_legacy_zero", "y_legacy_zero", "y_uc"),
    ("obs_minus_uc", "mortality_28d", "y_uc"),
]
SPEC_STRATA = ["all", "hfpef", "hfpef_nonshock", "hfpef_shock", "nohf_nonshock", "hfref"]
DOD_NAME = common.STANDARD_STRATUM_CONTRASTS[0][0]          # hfpef_nonshock_minus_nohf_nonshock
CATEGORY_GROUPS = ["perfusion_support", "default", "diuretic"]
STATE_VARS = ("map", "lactate", "spo2")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BM1 adaptive g-computation rollout of the 70c rule + NCE")
    common.add_common_args(p, n_boot_default=1000, model_based=True)
    p.add_argument("--nce-reps", type=int, default=common.NCE_REPS,
                   help="negative-exposure permutation reps (0 = skip the NCE)")
    p.add_argument("--nce-seed", type=int, default=common.NCE_SEED)
    p.add_argument("--nce-rng", choices=["85e", "fresh"], default="85e",
                   help="'85e': advance RandomState(seed) by the 85e NCO3 outcome shuffle first "
                        "(identical permutations to 85e); 'fresh': plain RandomState(seed)")
    p.add_argument("--threads", type=int,
                   default=int(os.environ.get("TORCH_NUM_THREADS") or os.environ.get("NSLOTS") or 8),
                   help="torch CPU threads (default TORCH_NUM_THREADS / NSLOTS / 8)")
    return p.parse_args(argv)


# ═══════════════════════════════════════════════════════════════════════
# Small helpers
# ═══════════════════════════════════════════════════════════════════════
def share_equal(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    return float((a[mask] == b[mask]).mean()) if mask.any() else float("nan")


def pct(x: float) -> float:
    return 100.0 * float(x) if x is not None and np.isfinite(x) else float("nan")


def vec_str(values, d: int = 2) -> str:
    return ",".join(common.fmt_num(v, d) for v in values)


def ci_str(row: pd.Series, d: int = 3) -> str:
    return common.fmt_ci(row["estimate"], row["ci_lo"], row["ci_hi"], d)


def ci_excludes_zero(row: pd.Series) -> bool:
    lo, hi = float(row["ci_lo"]), float(row["ci_hi"])
    return bool(np.isfinite(lo) and np.isfinite(hi) and (hi < 0 or lo > 0))


def category_groups(cat: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Share (%) of active intervals per category group and per category."""
    c = cat[mask]
    out = {}
    n = c.size
    for g in CATEGORY_GROUPS:
        cats = [k for k, v in common.RULE_CATEGORY_GROUP.items() if v == g]
        out[g] = 100.0 * float(np.isin(c, cats).sum()) / n if n else float("nan")
    for k, name in common.RULE_CATEGORY_NAMES.items():
        out[name] = 100.0 * float((c == k).sum()) / n if n else float("nan")
    out["n_intervals"] = int(n)
    return out


def group_str(g: dict) -> str:
    return ";".join(f"{k}={common.fmt_num(g[k], 1)}" for k in CATEGORY_GROUPS)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main(argv=None) -> int:
    t_start = time.time()
    args = common.apply_quick(parse_args(argv))
    torch.set_num_threads(max(1, int(args.threads)))
    print("=" * 72, flush=True)
    print(f"BM1 — Adaptive g-computation rollout of the 70c rule (v{SCRIPT_VERSION})", flush=True)
    print("=" * 72, flush=True)

    # ── 1. Data ──────────────────────────────────────────────────────
    t0 = phase(1, N_PHASES, "Load data, split and training action vocabulary")
    root = common.find_root(__file__)
    paths = common.setup_paths(root)
    log(f"  PROJECT_ROOT={root}")
    bundle = common.load_all(root)
    vocab = common.training_vocab(bundle, method="dataset")
    test_ids = np.sort(np.asarray(bundle.test_ids))
    if args.quick:
        test_ids = common.subsample_ids(test_ids, common.QUICK_N_STAYS, args.seed)
        log(f"  QUICK: test subsample n={len(test_ids):,}")
    log(f"  test stays: {len(test_ids):,}  [{time.time() - t0:.1f}s]")

    # ── 2. Dataset, model, rule decoder ──────────────────────────────
    t0 = phase(2, N_PHASES, "Test DTRDataset, G-Transformer checkpoint, rule decoder check")
    device = common.resolve_device(args.device)
    ds = common.build_dataset(bundle, test_ids, vocab, label="test", strict_vocab=not args.quick)
    ckpt_path = common.resolve_checkpoint(root, args.checkpoint)
    model, info = common.load_model(ckpt_path, ds, bundle.config, device)
    decoder = common.RuleDecoder.from_bundle(bundle, vocab)
    log(f"  rule thresholds: {decoder.thresholds}")
    rule_chk = common.rule_reproduction_check(bundle, ds, decoder)
    cum_ag = rule_chk["cumulative_fluid_balance_L"]["agreement"]
    int_ag = (rule_chk.get("fluid_balance_L") or {}).get("agreement")
    # aggregate (stored->decoded raw action) disagreement counts; no patient-level content
    dis_pairs = ";".join(f"{k}:{v}" for k, v in
                         rule_chk["cumulative_fluid_balance_L"]["disagreement_pairs_stored_vs_decoded"].items())
    # A FAIL only means 'old 70c parquet' when the interval-FB definition fits better; a
    # near-100% FAIL with the cumulative FB fitting best is a near-threshold float edge case.
    if rule_chk["status"] == "PASS":
        rule_hint = ""
    elif int_ag is not None and int_ag > cum_ag:
        rule_hint = "legacy 70c parquet (interval FB fits better): regenerate it with the corrected 70c"
    else:
        rule_hint = "cumulative FB fits best: inspect the disagreement pairs (near-threshold / data mismatch)"
    if rule_chk["status"] != "PASS":
        log(f"WARNING: the attached 70c rule actions are NOT fully reproduced by the rule applied to the "
            f"recorded states ({rule_hint}; stored->decoded pairs {dis_pairs or 'none'}). The non-adaptive "
            "arm uses the stored actions; resolve this before interpreting adaptive vs non-adaptive.")
    act = common.active_mask(ds)
    n_active = int(act.sum())
    fb_rec = common.stay_bin_matrix(bundle.state_df, ds.stay_ids, "cumulative_fluid_balance_L", bundle.n_bins)
    urine = common.stay_bin_matrix(bundle.state_df, ds.stay_ids, "urine_output_ml", bundle.n_bins)
    n_fb_nan = int(np.isnan(fb_rec[act]).sum())
    n_ur_nan = int(np.isnan(urine[act]).sum())
    log(f"  active bins={n_active:,}; recorded cumulative FB missing on {n_fb_nan} active bins, "
        f"urine missing on {n_ur_nan} (both treated as 0)")
    fb_rec0 = np.nan_to_num(fb_rec, nan=0.0)
    flv = common.fluid_ml_by_level(bundle)
    log(f"  [{time.time() - t0:.1f}s]")

    # ── 3. Non-adaptive rollouts ─────────────────────────────────────
    t0 = phase(3, N_PHASES, "Non-adaptive rollouts: usual care, stored rule, legacy zero-action arm")
    uc = common.predict_fixed(model, ds, device, "action", args.batch_size, desc="usual care", keep_states=True)
    na = common.predict_fixed(model, ds, device, "rule_action", args.batch_size,
                              desc="rule non-adaptive", keep_states=True)
    lg = common.predict_fixed(model, ds, device, common.legacy_zero_actions(ds), args.batch_size,
                              desc="legacy zero-action arm")
    log(f"  [{time.time() - t0:.1f}s]")

    # ── 4. Adaptive rollouts ─────────────────────────────────────────
    t0 = phase(4, N_PHASES, "Adaptive rollouts: rule re-applied to simulated states")
    ad = common.predict_adaptive(model, ds, decoder, device, "recorded", fb_matrix=fb_rec0,
                                 batch_size=args.batch_size, desc="adaptive rule (fb recorded)",
                                 keep_states=True)
    ap = common.predict_adaptive(model, ds, decoder, device, "approx", fluid_ml_by_level=flv,
                                 urine_matrix=urine, batch_size=args.batch_size,
                                 desc="adaptive rule (fb approx)", keep_states=True)
    log(f"  [{time.time() - t0:.1f}s]")

    # ── 5. Per-stay frame and paired bootstrap ───────────────────────
    t0 = phase(5, N_PHASES, f"Per-stay frame and paired bootstrap ({args.n_boot} reps)")
    frame = common.per_stay_frame(ds, bundle, y_uc=uc["prob"], y_rule_nonadaptive=na["prob"],
                                  y_rule_adaptive=ad["prob"], y_rule_adaptive_fbapprox=ap["prob"],
                                  y_legacy_zero=lg["prob"], n_active_bins=act.sum(axis=1))
    perstay = frame[["stay_id", "y_uc", "y_rule_nonadaptive", "y_rule_adaptive", "y_rule_adaptive_fbapprox",
                     "mortality_28d", "hf_subtype", "rule_phenotype_2class", "y_legacy_zero", "n_active_bins"]]
    ps_path = common.save_interim(perstay, root, PERSTAY_FILE)
    meta = {"script": "BM1_adaptive_rollout.py", "version": SCRIPT_VERSION, "created": common.now(),
            "checkpoint": str(ckpt_path.relative_to(root)) if ckpt_path.is_relative_to(root) else str(ckpt_path),
            "ckpt_epoch": info["epoch"], "quick": bool(args.quick), "n_stays": int(len(perstay)),
            "split": "test", "columns": list(perstay.columns), "fluid_ml_by_level": flv}
    ps_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    if args.quick:
        log("  WARNING: QUICK run -> the per-stay file holds a SUBSAMPLE; rerun BM1 without --quick "
            "before BM3b / BM5 on real data")

    strata = common.standard_strata(frame, extended=True)
    value_cols = ["mortality_28d"] + list(ARM_COLS)
    res, dr = common.paired_bootstrap(frame, value_cols, strata, n_boot=args.n_boot, seed=args.seed,
                                      contrasts=CONTRASTS, scale=100.0, desc="paired bootstrap",
                                      return_draws=True)
    # ratio of non-shock RDs (HFpEF / no-HF) from the same draws
    s_idx = {s: i for i, s in enumerate(dr["strata"])}
    c_idx = {c: j for j, c in enumerate(dr["columns"])}
    ratios = {}
    for cname, a, b in CONTRASTS[:3] + [CONTRASTS[5]]:
        def rd(arr, s):
            return arr[..., s_idx[s], c_idx[a]] - arr[..., s_idx[s], c_idx[b]]
        with np.errstate(divide="ignore", invalid="ignore"):
            est = float(rd(dr["point"], "hfpef_nonshock") / rd(dr["point"], "nohf_nonshock"))
            draws = rd(dr["draws"], "hfpef_nonshock") / rd(dr["draws"], "nohf_nonshock")
        draws = draws[np.isfinite(draws)]
        lo, hi = (np.percentile(draws, [2.5, 97.5]) if draws.size else (np.nan, np.nan))
        ratios[cname] = (est, float(lo), float(hi))
        res = pd.concat([res, pd.DataFrame([{
            "stratum": "hfpef_nonshock_over_nohf_nonshock", "kind": "ratio", "quantity": cname,
            "n": int(strata["hfpef_nonshock"].sum() + strata["nohf_nonshock"].sum()),
            "estimate": est, "ci_lo": lo, "ci_hi": hi,
            "se_boot": float(np.std(draws, ddof=1)) if draws.size > 1 else np.nan,
            "n_boot_valid": int(draws.size)}])], ignore_index=True)
    common.save_table(res, paths["tables"] / "BM1_bootstrap_long.csv")

    wide_rows = []
    for s, m in strata.items():
        row = OrderedDict(stratum=s, n=int(m.sum()), deaths=int(frame.loc[m, "mortality_28d"].sum()))
        row["obs_mort_pct"] = common.get_est(res, s, "mortality_28d", "mean")["estimate"]
        for c in ARM_COLS:
            row[f"risk_{c[2:]}_pct"] = common.get_est(res, s, c, "mean")["estimate"]
        for cname, _, _ in CONTRASTS:
            r = common.get_est(res, s, cname, "diff")
            row[f"{cname}_pp"] = r["estimate"]
            row[f"{cname}_lo"] = r["ci_lo"]
            row[f"{cname}_hi"] = r["ci_hi"]
        wide_rows.append(row)
    wide = pd.DataFrame(wide_rows)
    common.save_table(wide, paths["tables"] / "BM1_strata_summary.csv")
    log(f"  [{time.time() - t0:.1f}s]")

    # ── 6. Rule-decision diagnostics ─────────────────────────────────
    t0 = phase(6, N_PHASES, "Rule-decision diagnostics (agreement, category mix)")
    temporal_t = torch.from_numpy(ds.temporal)
    raw_na_dec, cat_na = (x.numpy() for x in decoder.raw_rule_from_states(temporal_t, torch.from_numpy(fb_rec0)))
    stored = common.stored_rule_raw_matrix(bundle, ds.stay_ids)
    raw_ucsim, cat_ucsim = (x.numpy() for x in decoder.raw_rule_from_states(
        torch.from_numpy(uc["predicted_states"]), torch.from_numpy(fb_rec0)))
    cat_ad, raw_ad = ad["categories"], ad["raw_actions"]
    cat_ap, raw_ap = ap["categories"], ap["raw_actions"]
    cat_arms = OrderedDict([
        ("non_adaptive", cat_na), ("adaptive_fb_recorded", cat_ad), ("adaptive_fb_approx", cat_ap),
        ("rule_on_usual_care_simulation", cat_ucsim)])

    agree_rows = []
    for t in range(ds.n_bins):
        m = act[:, t]
        agree_rows.append({
            "bin": t, "n_active": int(m.sum()),
            "agree_cat_ad_vs_na_pct": pct(share_equal(cat_ad[:, t], cat_na[:, t], m)),
            "agree_cat_adapprox_vs_na_pct": pct(share_equal(cat_ap[:, t], cat_na[:, t], m)),
            "agree_raw_ad_vs_stored_pct": pct(share_equal(raw_ad[:, t], stored[:, t], m)),
            "agree_raw_adapprox_vs_stored_pct": pct(share_equal(raw_ap[:, t], stored[:, t], m)),
            "agree_cat_ucsim_vs_na_pct": pct(share_equal(cat_ucsim[:, t], cat_na[:, t], m)),
            "agree_raw_na_decoded_vs_stored_pct": pct(share_equal(raw_na_dec[:, t], stored[:, t], m)),
        })
    agree_df = pd.DataFrame(agree_rows)
    common.save_table(agree_df, paths["tables"] / "BM1_rule_agreement_by_bin.csv")

    agree = {}
    for s in SPEC_STRATA:
        m = act & strata[s][:, None]
        agree[s] = {"cat_ad": pct(share_equal(cat_ad, cat_na, m)),
                    "cat_ap": pct(share_equal(cat_ap, cat_na, m)),
                    "raw_ad": pct(share_equal(raw_ad, stored, m)),
                    "raw_ap": pct(share_equal(raw_ap, stored, m)),
                    "cat_ucsim": pct(share_equal(cat_ucsim, cat_na, m)),
                    "stays_all_bins_equal_ad": pct(np.mean([(cat_ad[i][act[i]] == cat_na[i][act[i]]).all()
                                                            for i in np.where(strata[s])[0]])
                                                   if strata[s].any() else np.nan)}
    log(f"  agreement adaptive vs non-adaptive (category, all active bins): {agree['all']['cat_ad']:.2f}% "
        f"(approx FB {agree['all']['cat_ap']:.2f}%; rule on usual-care simulation {agree['all']['cat_ucsim']:.2f}%)")

    dist_rows, dist = [], {}
    for s in strata:
        m = act & strata[s][:, None]
        for arm, cat in cat_arms.items():
            g = category_groups(cat, m)
            dist[(s, arm)] = g
            for k, v in g.items():
                if k == "n_intervals":
                    continue
                dist_rows.append({"stratum": s, "arm": arm,
                                  "level": "group" if k in CATEGORY_GROUPS else "category",
                                  "category": k, "n_intervals": g["n_intervals"], "pct": v})
    common.save_table(pd.DataFrame(dist_rows), paths["tables"] / "BM1_rule_category_distribution.csv")
    for s in ("all", "hfpef_nonshock"):
        for arm in cat_arms:
            log(f"  category mix [{s}] {arm:30s}: {group_str(dist[(s, arm)])}")
    log(f"  [{time.time() - t0:.1f}s]")

    # ── 7. Simulated trajectories and transition-model calibration ───
    t0 = phase(7, N_PHASES, "Simulated MAP/lactate by bin; transition-model calibration under usual care")
    traj_src = OrderedDict([
        ("recorded", ds.temporal), ("usual_care_sim", uc["predicted_states"]),
        ("rule_nonadaptive_sim", na["predicted_states"]), ("rule_adaptive_sim", ad["predicted_states"]),
        ("rule_adaptive_fbapprox_sim", ap["predicted_states"])])
    decoded = {arm: decoder.decode_np(st) for arm, st in traj_src.items()}
    traj_rows, traj = [], {}
    thr_map, thr_lac = decoder.thr_map, decoder.thr_lac
    for s in ("all", "hfpef", "hfpef_nonshock", "nohf_nonshock"):
        for arm, d in decoded.items():
            for t in range(ds.n_bins):
                m = act[:, t] & strata[s]
                row = {"stratum": s, "arm": arm, "bin": t, "n_active": int(m.sum())}
                for v in STATE_VARS:
                    row[f"{v}_mean"] = float(d[v][m, t].mean()) if m.any() else np.nan
                row["share_map_lt65_pct"] = pct((d["map"][m, t] < thr_map).mean()) if m.any() else np.nan
                row["share_lac_gt2_pct"] = pct((d["lactate"][m, t] > thr_lac).mean()) if m.any() else np.nan
                traj_rows.append(row)
            m_all = act & strata[s][:, None]
            traj[(s, arm)] = {
                "share_map_lt65": pct((d["map"][m_all] < thr_map).mean()) if m_all.any() else np.nan,
                "share_lac_gt2": pct((d["lactate"][m_all] > thr_lac).mean()) if m_all.any() else np.nan}
    traj_df = pd.DataFrame(traj_rows)
    # fluid-balance thresholds under recorded vs approximate accumulation
    fb_share = {}
    for name, fbm in (("recorded", fb_rec0), ("approx_adaptive", ap["fb_used"])):
        fb_share[name] = {"gt3": pct((fbm[act] > decoder.thr_fb3).mean()),
                          "gt5": pct((fbm[act] > decoder.thr_fb5).mean()),
                          "mean_last_bin": float(np.mean([fbm[i, np.where(act[i])[0][-1]]
                                                          for i in range(len(ds)) if act[i].any()]))}
    common.save_table(traj_df, paths["tables"] / "BM1_state_trajectory_by_bin.csv")

    tf = common.teacher_forced_next_state(model, ds, device, args.batch_size, desc="teacher-forced next state")
    target = ds.temporal[:, 1:, :]
    act_next = act[:, 1:]
    err_rows, err = [], {}
    for v in STATE_VARS:
        zi = bundle.temporal_cols.index(f"{v}_z")
        mcol = f"{v}_mask"
        has_mask = mcol in bundle.temporal_cols
        observed = (target[:, :, bundle.temporal_cols.index(mcol)] > 0.5) if has_mask else np.ones_like(act_next)
        subsets = OrderedDict([("measured", act_next & observed), ("all_active", act_next)]) if has_mask \
            else OrderedDict([("all_active", act_next)])
        e_tf = tf[:, :, zi].astype(np.float64) - target[:, :, zi]
        e_locf = ds.temporal[:, :-1, zi].astype(np.float64) - target[:, :, zi]
        e_free = uc["predicted_states"][:, 1:, zi].astype(np.float64) - target[:, :, zi]
        for sub, msk in subsets.items():
            for t in range(-1, ds.n_bins - 1):           # -1 = pooled over bins
                m = msk if t < 0 else np.zeros_like(msk)
                if t >= 0:
                    m[:, t] = msk[:, t]
                n_m = int(m.sum())
                row = {"variable": f"{v}_z", "subset": sub, "target_bin": "all" if t < 0 else t + 1,
                       "n": n_m,
                       "mae_one_step": float(np.abs(e_tf[m]).mean()) if n_m else np.nan,
                       "bias_one_step": float(e_tf[m].mean()) if n_m else np.nan,
                       "mae_locf": float(np.abs(e_locf[m]).mean()) if n_m else np.nan,
                       "mae_free_running_usual_care": float(np.abs(e_free[m]).mean()) if n_m else np.nan,
                       "bias_free_running_usual_care": float(e_free[m].mean()) if n_m else np.nan}
                err_rows.append(row)
                if t < 0:
                    err[(v, sub)] = row
            err[(v, sub, "by_bin")] = [r["mae_one_step"] for r in err_rows
                                       if r["variable"] == f"{v}_z" and r["subset"] == sub
                                       and r["target_bin"] != "all"]
    err_df = pd.DataFrame(err_rows)
    common.save_table(err_df, paths["tables"] / "BM1_transition_error_by_bin.csv")
    sub_main = "measured" if ("map", "measured") in err else "all_active"
    for v in STATE_VARS:
        e = err[(v, sub_main)]
        log(f"  {v}_z one-step MAE ({sub_main}, n={e['n']:,}): {e['mae_one_step']:.3f} "
            f"(LOCF {e['mae_locf']:.3f}; free-running usual care {e['mae_free_running_usual_care']:.3f})")
    log(f"  [{time.time() - t0:.1f}s]")

    # ── 8. Negative-exposure control ─────────────────────────────────
    t0 = phase(8, N_PHASES, f"Negative-exposure control ({args.nce_reps} within-bin permutations, 85e NCE1)")
    nce_rows, nce = [], {}
    if args.nce_reps > 0:
        prior = len(bundle.outcomes_df) if args.nce_rng == "85e" else None
        rng = common.make_nce_rng(args.nce_seed, prior_shuffle_len=prior)
        log(f"  RandomState({args.nce_seed}); stream advanced by the 85e outcome shuffle: {prior is not None}"
            + (f" (len={prior:,})" if prior else ""))
        m_hf = strata["hfpef"]
        arm_true = {"rule_nonadaptive": na["prob"], "rule_adaptive": ad["prob"],
                    "rule_adaptive_fbapprox": ap["prob"], "legacy_zero": lg["prob"]}
        for rep in tqdm(range(args.nce_reps), desc="  NCE permutations", mininterval=1):
            perm = common.permute_observed_actions_within_bin(bundle.propensity_df, ds.stay_ids,
                                                              bundle.n_bins, rng)
            ds_p = common.build_dataset(bundle, ds.stay_ids, vocab, label=f"nce{rep + 1}",
                                        strict_vocab=False, propensity_df=perm)
            share_changed = float((ds_p.action[act] != ds.action[act]).mean())
            y_perm = common.predict_fixed(model, ds_p, device, "action", args.batch_size,
                                          desc=f"permuted usual care rep {rep + 1}")["prob"]
            del ds_p, perm
            for arm, y in arm_true.items():
                nce_rows.append({
                    "rep": rep + 1, "arm": arm,
                    "risk_uc_perm_pct": 100 * float(y_perm.mean()),
                    "uc_perm_minus_uc_pp": 100 * float(y_perm.mean() - uc["prob"].mean()),
                    "rd_perm_all_pp": 100 * float(y.mean() - y_perm.mean()),
                    "rd_perm_hfpef_pp": (100 * float(y[m_hf].mean() - y_perm[m_hf].mean())
                                         if m_hf.any() else np.nan),
                    "share_active_actions_changed_pct": 100 * share_changed})
        nce_df = pd.DataFrame(nce_rows)
        common.save_table(nce_df, paths["tables"] / "BM1_nce_reps.csv")
        sum_rows = []
        for arm, y in arm_true.items():
            sub = nce_df[nce_df["arm"] == arm]
            rd_true = 100 * float(y.mean() - uc["prob"].mean())
            rd_true_hf = 100 * float(y[m_hf].mean() - uc["prob"][m_hf].mean()) if m_hf.any() else np.nan
            mean_all, sd_all = float(sub["rd_perm_all_pp"].mean()), float(np.std(sub["rd_perm_all_pp"]))
            mean_hf, sd_hf = float(sub["rd_perm_hfpef_pp"].mean()), float(np.std(sub["rd_perm_hfpef_pp"]))
            ret_all = 100 * mean_all / rd_true if np.isfinite(rd_true) and rd_true != 0 else np.nan
            ret_hf = 100 * mean_hf / rd_true_hf if np.isfinite(rd_true_hf) and rd_true_hf != 0 else np.nan
            nce[arm] = dict(rd_true=rd_true, mean=mean_all, sd=sd_all, retained=ret_all,
                            rd_true_hfpef=rd_true_hf, mean_hfpef=mean_hf, sd_hfpef=sd_hf, retained_hfpef=ret_hf)
            sum_rows.append({"arm": arm, "n_reps": len(sub), "rd_true_all_pp": rd_true,
                             "rd_perm_all_mean_pp": mean_all, "rd_perm_all_sd_pp": sd_all,
                             "retained_all_pct": ret_all, "rd_true_hfpef_pp": rd_true_hf,
                             "rd_perm_hfpef_mean_pp": mean_hf, "rd_perm_hfpef_sd_pp": sd_hf,
                             "retained_hfpef_pct": ret_hf})
            log(f"  NCE {arm:24s}: RD_true={rd_true:+.3f} pp; RD_perm={mean_all:+.3f} ± {sd_all:.3f} pp; "
                f"retained={common.fmt_num(ret_all, 1)}%")
        nce["uc_perm_minus_uc_mean"] = float(nce_df.loc[nce_df["arm"] == "rule_nonadaptive",
                                                        "uc_perm_minus_uc_pp"].mean())
        nce["share_changed"] = float(nce_df["share_active_actions_changed_pct"].mean())
        common.save_table(pd.DataFrame(sum_rows), paths["tables"] / "BM1_nce_summary.csv")
    else:
        log("  NCE skipped (--nce-reps 0)")
    log(f"  [{time.time() - t0:.1f}s]")

    # ── 9. Reproduction checks and report ────────────────────────────
    t0 = phase(9, N_PHASES, "Reproduction checks, report and paste-back block")

    def E(s, q, kind):
        return common.get_est(res, s, q, kind)

    rc = common.repro_check_extended(
        E("all", "y_uc", "mean")["estimate"], E("hfpef", "y_uc", "mean")["estimate"],
        E("all", "rd_rule_na", "diff")["estimate"], E("hfpef", "rd_rule_na", "diff")["estimate"],
        E("all", "rd_legacy_zero", "diff")["estimate"], E("hfpef", "rd_legacy_zero", "diff")["estimate"])
    tgt = common.REPRO_TARGETS
    n_ok = bool(len(ds) == tgt["n_test"] and int(strata["hfpef"].sum()) == tgt["n_hfpef"])
    n_line = (f"REPRO_CHECK_N: {'PASS' if n_ok else 'FAIL'} (n_test={len(ds):,} vs {tgt['n_test']:,}; "
              f"n_hfpef={int(strata['hfpef'].sum())} vs {tgt['n_hfpef']})")
    print(n_line, flush=True)
    nce_line = "REPRO_CHECK_NCE_LEGACY: NA (NCE skipped)"
    nce_status = "NA"
    if "legacy_zero" in nce:
        lgn = nce["legacy_zero"]
        ok = bool(abs(lgn["mean"] - tgt["nce_rd"]) <= NCE_TOL_RD_PP
                  and abs(lgn["retained"] - tgt["nce_retained_pct"]) <= NCE_TOL_RETAINED_PCT)
        nce_status = "PASS" if ok else "FAIL"
        nce_line = (f"REPRO_CHECK_NCE_LEGACY: {nce_status} (legacy zero-action NCE RD_perm="
                    f"{common.fmt_num(lgn['mean'])} pp vs {tgt['nce_rd']}; retained="
                    f"{common.fmt_num(lgn['retained'], 1)}% vs {tgt['nce_retained_pct']}%)")
        print(nce_line, flush=True)
    detail_targets = [("all", "rd_all"), ("hfpef", "rd_hfpef"), ("hfpef_nonshock", "rd_hfpef_nonshock"),
                      ("nohf_nonshock", "rd_nohf_nonshock"), ("hfpef_shock", "rd_hfpef_shock")]

    runtime_min = (time.time() - t_start) / 60.0
    L = common.report_header("BM1 — Adaptive g-computation rollout of the 70c rule (BMC Medicine revision)",
                             root, args)
    L += [f"Script version {SCRIPT_VERSION}; device {device}; torch threads {torch.get_num_threads()}; "
          f"runtime {runtime_min:.1f} min" + ("  [QUICK SMOKE TEST — not for interpretation]" if args.quick else ""),
          f"Checkpoint: {ckpt_path.name} (epoch {info['epoch']}, val_loss {common.fmt_num(info['val_loss'], 4)}, "
          f"{info['n_params']:,} parameters)",
          f"Test stays: {len(ds):,}; active 2-h intervals: {n_active:,}; training action vocab ({len(vocab)}): {vocab}",
          f"Fluid volume per adaptive fluid level (training median, mL / 2 h): "
          f"low={common.fmt_num(flv[1], 0)}, high={common.fmt_num(flv[2], 0)}",
          "",
          "Rule decoder on recorded states vs attached 70c actions (must be 100% with the corrected 70c):"
          + ("" if rule_chk["status"] == "PASS" else f"  ** FAIL: {rule_hint}; stored->decoded {dis_pairs} **"),
          f"  {rule_chk['status']}: cumulative-FB agreement "
          f"{common.fmt_num(100 * rule_chk['cumulative_fluid_balance_L']['agreement'], 3)}%"
          + (f"; interval-FB (legacy 70c) agreement "
             f"{common.fmt_num(100 * rule_chk['fluid_balance_L']['agreement'], 3)}%"
             if rule_chk.get("fluid_balance_L") else "")
          + f"; without float32 threshold snap {common.fmt_num(100 * rule_chk['agreement_without_snap'], 3)}%",
          "",
          "Arms:"] + [f"  {k:26s} {v}" for k, v in ARM_COLS.items()] + [
          "",
          "Model risk (%) and risk differences (pp, paired bootstrap 95% CI; negative favours the rule):"]
    hdr = (f"  {'stratum':18s} {'n':>6s} {'deaths':>6s} {'obs%':>6s} {'UC%':>6s} {'R-na%':>6s} {'R-ad%':>6s} "
           f"{'R-adFB%':>7s} {'zero%':>6s}")
    L.append(hdr)
    for s in strata:
        w = wide[wide["stratum"] == s].iloc[0]
        L.append(f"  {s:18s} {int(w['n']):6d} {int(w['deaths']):6d} {common.fmt_num(w['obs_mort_pct']):>6s} "
                 f"{common.fmt_num(w['risk_uc_pct']):>6s} {common.fmt_num(w['risk_rule_nonadaptive_pct']):>6s} "
                 f"{common.fmt_num(w['risk_rule_adaptive_pct']):>6s} "
                 f"{common.fmt_num(w['risk_rule_adaptive_fbapprox_pct']):>7s} "
                 f"{common.fmt_num(w['risk_legacy_zero_pct']):>6s}")
    L.append("")
    for cname, a, b in CONTRASTS:
        L.append(f"  {cname} = {a} - {b}:")
        for s in strata:
            L.append(f"    {s:18s} {ci_str(E(s, cname, 'diff'))}")
    L += ["", "HFpEF-specific signal (non-shock RD, HFpEF minus no-HF; ratio HFpEF / no-HF):"]
    for cname, _, _ in CONTRASTS[:6]:
        line = f"  {cname:22s} diff {ci_str(E(DOD_NAME, cname, 'diff_of_diff'))}"
        if cname in ratios:
            line += f"; ratio {common.fmt_ci(*ratios[cname], d=2)}"
        L.append(line)
    L += ["", "Rule-decision agreement on active intervals (%):",
          f"  {'stratum':18s} {'cat ad=na':>9s} {'cat apx=na':>10s} {'raw ad=stored':>13s} "
          f"{'ucsim=na':>9s} {'stays all-equal':>15s}"]
    for s in SPEC_STRATA:
        a = agree[s]
        L.append(f"  {s:18s} {common.fmt_num(a['cat_ad']):>9s} {common.fmt_num(a['cat_ap']):>10s} "
                 f"{common.fmt_num(a['raw_ad']):>13s} {common.fmt_num(a['cat_ucsim']):>9s} "
                 f"{common.fmt_num(a['stays_all_bins_equal_ad']):>15s}")
    L.append("  by bin (category, adaptive FB recorded vs non-adaptive): "
             + vec_str(agree_df["agree_cat_ad_vs_na_pct"], 1))
    L.append("  by bin (category, rule on usual-care simulation vs recorded): "
             + vec_str(agree_df["agree_cat_ucsim_vs_na_pct"], 1))
    L += ["", "Rule-category mix (% of active intervals: perfusion support P1-P3 / default / diuretic P4-P5):"]
    for s in ("all", "hfpef", "hfpef_nonshock", "nohf_nonshock"):
        for arm in cat_arms:
            L.append(f"  {s:16s} {arm:30s} {group_str(dist[(s, arm)])}")
    L += ["", "Simulated state by bin, all test stays (mean, raw units):"]
    for v, lab in (("map", "MAP mmHg"), ("lactate", "lactate mmol/L")):
        for arm in traj_src:
            sub = traj_df[(traj_df["stratum"] == "all") & (traj_df["arm"] == arm)]
            L.append(f"  {lab:15s} {arm:27s} {vec_str(sub[f'{v}_mean'], 2 if v == 'lactate' else 1)}")
    L.append("  share of active intervals with MAP<65 / lactate>2 (%): "
             + "; ".join(f"{arm}={common.fmt_num(traj[('all', arm)]['share_map_lt65'], 1)}/"
                         f"{common.fmt_num(traj[('all', arm)]['share_lac_gt2'], 1)}" for arm in traj_src))
    L.append(f"  cumulative FB >3 L / >5 L (% of active intervals): recorded "
             f"{common.fmt_num(fb_share['recorded']['gt3'], 1)}/{common.fmt_num(fb_share['recorded']['gt5'], 1)}; "
             f"approx under adaptive rule {common.fmt_num(fb_share['approx_adaptive']['gt3'], 1)}/"
             f"{common.fmt_num(fb_share['approx_adaptive']['gt5'], 1)}")
    L += ["", f"Transition-model calibration under usual care (z units; subset = {sub_main}):"]
    for v in STATE_VARS:
        e = err[(v, sub_main)]
        L.append(f"  {v}_z: one-step MAE {common.fmt_num(e['mae_one_step'], 3)} (bias "
                 f"{common.fmt_num(e['bias_one_step'], 3)}); LOCF MAE {common.fmt_num(e['mae_locf'], 3)}; "
                 f"free-running rollout MAE {common.fmt_num(e['mae_free_running_usual_care'], 3)}; n={e['n']:,}")
        L.append(f"    one-step MAE by target bin 1..{ds.n_bins - 1}: {vec_str(err[(v, sub_main, 'by_bin')], 3)}")
    L += ["", "Negative-exposure control (observed actions permuted within bin across test stays):"]
    if nce:
        L.append(f"  reps={args.nce_reps}; RandomState({args.nce_seed}), rng={args.nce_rng}; "
                 f"share of active actions changed {common.fmt_num(nce['share_changed'], 1)}%; "
                 f"mean V(uc_perm)-V(uc) = {common.fmt_num(nce['uc_perm_minus_uc_mean'], 3)} pp")
        for arm in ("rule_nonadaptive", "rule_adaptive", "rule_adaptive_fbapprox", "legacy_zero"):
            n_ = nce[arm]
            L.append(f"  {arm:24s} RD_true {common.fmt_num(n_['rd_true'], 3)}; RD_perm "
                     f"{common.fmt_num(n_['mean'], 3)} (SD {common.fmt_num(n_['sd'], 3)}); retained "
                     f"{common.fmt_num(n_['retained'], 1)}%; HFpEF retained {common.fmt_num(n_['retained_hfpef'], 1)}%")
    else:
        L.append("  skipped")
    L += ["", "Reproduction of the published non-adaptive analysis:", "  " + n_line, "  " + rc["line"],
          "  " + rc["uc_line"]]
    if "legacy_line" in rc:
        L.append("  " + rc["legacy_line"])
    L.append("  " + nce_line)
    L.append(f"  {'stratum':16s} {'target':>7s} {'rule_na':>8s} {'legacy0':>8s}")
    for s, k in detail_targets:
        L.append(f"  {s:16s} {tgt[k]:7.2f} {common.fmt_num(E(s, 'rd_rule_na', 'diff')['estimate']):>8s} "
                 f"{common.fmt_num(E(s, 'rd_legacy_zero', 'diff')['estimate']):>8s}")
    if rc.get("legacy_status") == "PASS" and not rc["pass"]:
        L.append("  REPRO_NOTE: " + common.LEGACY_RULE_NOTE)
    L += ["",
          "Notes:",
          "  * Bootstrap: paired patient-level resampling of per-stay predictions with the model fixed;",
          "    percentile 95% CI; model (transition and outcome) uncertainty is not propagated.",
          "  * Adaptive rule: a_t = 70c rule(decoded simulated state, fluid balance); t=0 uses the observed s_0.",
          "    'recorded' FB keeps the usual-care cumulative fluid balance (FB is not a model state);",
          "    'approx' re-accumulates FB from the adaptive fluid level (training median volume) minus",
          "    recorded urine output, starting at 0 at T0.",
          "  * The rollout is deterministic: the rule thresholds are applied to the model's conditional-MEAN",
          "    state, not to sampled trajectories, so threshold crossings (MAP<65, lactate>2) are under-",
          "    represented when predictions regress to the mean; read the category mix and the MAP<65 /",
          "    lactate>2 shares per arm (and rule_on_usual_care_simulation) before the adaptive RD.",
          "  * Ratios of non-shock RDs are unstable when the no-HF RD interval approaches 0.",
          "  * rule_on_usual_care_simulation: the rule applied to the states simulated under the OBSERVED",
          "    actions; its disagreement with the recorded-state rule measures transition-model drift.",
          "  * Stay-level predictions: data/interim/bmc_medicine/BM1_perstay_predictions.parquet (never sync)."]

    # ── paste-back block ──
    P = OrderedDict()
    P["script"] = "BM1_adaptive_rollout"
    P["version"] = SCRIPT_VERSION
    P["quick"] = bool(args.quick)
    P["checkpoint"] = ckpt_path.name
    P["ckpt_epoch"] = info["epoch"]
    P["ckpt_val_loss"] = info["val_loss"] if info["val_loss"] is not None else float("nan")
    P["device"] = str(device)
    P["n_boot"] = int(args.n_boot)
    P["seed"] = int(args.seed)
    P["n_test"] = int(len(ds))
    P["n_hfpef"] = int(strata["hfpef"].sum())
    P["n_active_intervals"] = n_active
    P["training_vocab"] = ",".join(map(str, vocab))
    P["fluid_ml_level1"] = float(flv[1])
    P["fluid_ml_level2"] = float(flv[2])
    P["fb_recorded_missing_active_bins"] = n_fb_nan
    P["RULE_DECODER_CHECK"] = rule_chk["status"]
    P["rule_decoder_agree_cumfb_pct"] = 100 * rule_chk["cumulative_fluid_balance_L"]["agreement"]
    if rule_chk.get("fluid_balance_L"):
        P["rule_decoder_agree_intervalfb_pct"] = 100 * rule_chk["fluid_balance_L"]["agreement"]
    P["rule_decoder_agree_nosnap_pct"] = 100 * rule_chk["agreement_without_snap"]
    P["rule_decoder_disagree_pairs_cumfb"] = dis_pairs or "none"
    for s in SPEC_STRATA:
        w = wide[wide["stratum"] == s].iloc[0]
        P[f"{s}_n"] = int(w["n"])
        P[f"{s}_deaths"] = int(w["deaths"])
        P[f"{s}_obs_mort_pct"] = float(w["obs_mort_pct"])
        P[f"{s}_risk_uc_pct"] = float(w["risk_uc_pct"])
        P[f"{s}_risk_rule_na_pct"] = float(w["risk_rule_nonadaptive_pct"])
        P[f"{s}_risk_rule_ad_pct"] = float(w["risk_rule_adaptive_pct"])
        P[f"{s}_risk_rule_ad_fbapprox_pct"] = float(w["risk_rule_adaptive_fbapprox_pct"])
        P[f"{s}_risk_legacy_zero_pct"] = float(w["risk_legacy_zero_pct"])
        for cname in ("rd_rule_na", "rd_rule_ad", "rd_rule_ad_fbapprox", "rd_ad_minus_na",
                      "rd_adapprox_minus_na", "rd_legacy_zero", "obs_minus_uc"):
            P[f"{s}_{cname}_pp"] = ci_str(E(s, cname, "diff"))
    for cname in ("rd_rule_na", "rd_rule_ad", "rd_rule_ad_fbapprox", "rd_ad_minus_na", "rd_legacy_zero"):
        P[f"dod_nonshock_hfpef_minus_nohf_{cname}_pp"] = ci_str(E(DOD_NAME, cname, "diff_of_diff"))
    for cname, v in ratios.items():
        P[f"ratio_nonshock_hfpef_over_nohf_{cname}"] = common.fmt_ci(*v, d=2)
    P["flag_rd_rule_ad_all_ci_excludes_0"] = ci_excludes_zero(E("all", "rd_rule_ad", "diff"))
    P["flag_rd_rule_ad_hfpef_ci_excludes_0"] = ci_excludes_zero(E("hfpef", "rd_rule_ad", "diff"))
    P["flag_rd_rule_ad_same_sign_as_na_all"] = bool(
        np.sign(E("all", "rd_rule_ad", "diff")["estimate"]) == np.sign(E("all", "rd_rule_na", "diff")["estimate"]))
    P["flag_rd_ad_minus_na_all_ci_excludes_0"] = ci_excludes_zero(E("all", "rd_ad_minus_na", "diff"))
    P["flag_dod_rd_rule_ad_ci_excludes_0"] = ci_excludes_zero(E(DOD_NAME, "rd_rule_ad", "diff_of_diff"))
    for s in ("all", "hfpef", "hfpef_nonshock", "nohf_nonshock"):
        P[f"agree_cat_ad_vs_na_{s}_pct"] = agree[s]["cat_ad"]
    P["agree_cat_adapprox_vs_na_all_pct"] = agree["all"]["cat_ap"]
    P["agree_raw_ad_vs_stored_all_pct"] = agree["all"]["raw_ad"]
    P["agree_cat_ucsim_vs_na_all_pct"] = agree["all"]["cat_ucsim"]
    P["stays_all_bins_equal_ad_all_pct"] = agree["all"]["stays_all_bins_equal_ad"]
    P["agree_cat_ad_vs_na_by_bin_pct"] = vec_str(agree_df["agree_cat_ad_vs_na_pct"], 1)
    P["agree_cat_ucsim_vs_na_by_bin_pct"] = vec_str(agree_df["agree_cat_ucsim_vs_na_pct"], 1)
    for s in ("all", "hfpef_nonshock"):
        for arm, short in (("non_adaptive", "na"), ("adaptive_fb_recorded", "ad"),
                           ("adaptive_fb_approx", "adapprox"), ("rule_on_usual_care_simulation", "ucsim")):
            P[f"catmix_{s}_{short}_pct"] = group_str(dist[(s, arm)])
    for arm, short in (("recorded", "recorded"), ("usual_care_sim", "uc"), ("rule_nonadaptive_sim", "na"),
                       ("rule_adaptive_sim", "ad")):
        sub = traj_df[(traj_df["stratum"] == "all") & (traj_df["arm"] == arm)]
        P[f"map_by_bin_{short}"] = vec_str(sub["map_mean"], 1)
        P[f"lactate_by_bin_{short}"] = vec_str(sub["lactate_mean"], 2)
        P[f"share_map_lt65_{short}_pct"] = traj[("all", arm)]["share_map_lt65"]
        P[f"share_lac_gt2_{short}_pct"] = traj[("all", arm)]["share_lac_gt2"]
    P["share_fb_gt3_recorded_pct"] = fb_share["recorded"]["gt3"]
    P["share_fb_gt3_approx_pct"] = fb_share["approx_adaptive"]["gt3"]
    P["mean_fb_last_bin_recorded_L"] = fb_share["recorded"]["mean_last_bin"]
    P["mean_fb_last_bin_approx_L"] = fb_share["approx_adaptive"]["mean_last_bin"]
    P["transition_error_subset"] = sub_main
    for v in STATE_VARS:
        e = err[(v, sub_main)]
        P[f"onestep_mae_{v}_z"] = e["mae_one_step"]
        P[f"onestep_bias_{v}_z"] = e["bias_one_step"]
        P[f"locf_mae_{v}_z"] = e["mae_locf"]
        P[f"freerun_mae_{v}_z"] = e["mae_free_running_usual_care"]
        P[f"onestep_mae_{v}_z_by_bin"] = vec_str(err[(v, sub_main, "by_bin")], 3)
    if nce:
        P["nce_reps"] = int(args.nce_reps)
        P["nce_rng"] = f"RandomState({args.nce_seed})/{args.nce_rng}"
        P["nce_share_actions_changed_pct"] = nce["share_changed"]
        P["nce_uc_perm_minus_uc_pp"] = nce["uc_perm_minus_uc_mean"]
        for arm, short in (("rule_nonadaptive", "na"), ("rule_adaptive", "ad"),
                           ("rule_adaptive_fbapprox", "adapprox"), ("legacy_zero", "legacy")):
            n_ = nce[arm]
            P[f"nce_{short}_rd_true_pp"] = n_["rd_true"]
            P[f"nce_{short}_rd_perm_mean_pp"] = n_["mean"]
            P[f"nce_{short}_rd_perm_sd_pp"] = n_["sd"]
            P[f"nce_{short}_retained_pct"] = n_["retained"]
            P[f"nce_{short}_retained_hfpef_pct"] = n_["retained_hfpef"]
    P["REPRO_CHECK_N"] = "PASS" if n_ok else "FAIL"
    P["REPRO_CHECK"] = rc["status"]
    P["REPRO_CHECK_UC"] = rc["uc_status"]
    P["REPRO_CHECK_LEGACY"] = rc.get("legacy_status", "NA")
    P["REPRO_CHECK_NCE_LEGACY"] = nce_status
    P["repro_risk_uc_all_pct"] = float(E("all", "y_uc", "mean")["estimate"])
    P["repro_risk_uc_hfpef_pct"] = float(E("hfpef", "y_uc", "mean")["estimate"])
    P["repro_rd_rule_na_all_pp"] = float(E("all", "rd_rule_na", "diff")["estimate"])
    P["repro_rd_rule_na_hfpef_pp"] = float(E("hfpef", "rd_rule_na", "diff")["estimate"])
    P["repro_rd_legacy_all_pp"] = float(E("all", "rd_legacy_zero", "diff")["estimate"])
    P["repro_rd_legacy_hfpef_pp"] = float(E("hfpef", "rd_legacy_zero", "diff")["estimate"])
    P["runtime_min"] = runtime_min

    common.write_report(paths["reports"] / "BM1_summary.txt", L, P, TAG)
    log(f"  [{time.time() - t0:.1f}s]")
    log(f"BM1 done in {(time.time() - t_start) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
