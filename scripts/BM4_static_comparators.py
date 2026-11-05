#!/usr/bin/env python3
"""
BM4_static_comparators — Static and partially-static comparator strategies under
the same G-Transformer counterfactual rollout (BMC Medicine revision, BM4).

Purpose
-------
The primary analysis contrasts the 70c rule with usual care under the 80c
G-Transformer DM rollout. Reviewers asked whether the predicted benefit comes
from STATE-RESPONSIVE assignment or merely from the action mix the rule favours,
and whether the model behaves sensibly for strategies whose direction of effect
is known a priori. BM4 evaluates, on the test set and with exactly the same
rollout (model.predict_counterfactual_rollout, observed s_0, model-simulated
s_1..s_T-1, masked mean pooling), the following strategies:

  UC    usual care: observed contiguous actions (batch['action']);
  S0    no treatment: raw 0 (no fluid, no vasopressor, no diuretic) in every
        interval -- a falsification probe: withholding all treatment is not
        expected to be beneficial. S0 is also, exactly, the arm the pre-v1.0.1
        DTRDataset fed to the model as "rule" (see common.LEGACY_RULE_NOTE);
  S1    default only: raw 6 (low/moderate fluid, no vasopressor, no diuretic)
        in every interval (= the rule's default category everywhere);
  S2    usual care with diuretics removed: observed raw action with the
        diuretic bit set to 0;
  S3    usual care with vasopressors removed: observed raw action with the
        vasopressor level set to 0;
  R_na  rule, non-adaptive: stored 70c rule actions (batch['rule_action'],
        categories from RECORDED states) -- the primary analysis arm;
  R_ad  rule, adaptive: the rule re-applied at every interval to the
        model-simulated state (common.predict_adaptive / rollout_adaptive;
        fluid balance per --fb-mode, default 'recorded').

All strategy actions are built from raw coarse_3 codes, merged with the 70b
merge map and mapped to the TRAINING contiguous vocabulary (common.ActionCodec);
inactive bins (padding / after the absorbing bin) keep the placeholder 0 as in
DTRDataset. For S2/S3, bins whose raw action is unchanged by the modification
keep the observed contiguous action, so S2/S3 differ from UC only where a
diuretic / vasopressor was actually given (and is visible after merging).

Reported per stratum (all; HFpEF; HFpEF non-shock; HFpEF shock; no-HF non-shock;
HFrEF; plus extended strata in the CSV): n, deaths, observed 28-day mortality,
model risk under every strategy, RD vs usual care (pp, paired patient-level
bootstrap 95% percentile CI, models fixed), and the paired differences
R_ad - S1 and R_na - S1 (does state-responsive assignment add anything beyond
the default action mix?), S2 - UC (effect attributed to withholding diuretics),
S0 - UC (falsification), S3 - UC, R_ad - R_na and R_na - S0; the
HFpEF-minus-no-HF non-shock difference of every contrast. S2 - UC and S3 - UC
are also reported among the patients whose model input actually changes
(diuretic- / vasopressor-exposed; the difference is 0 by construction elsewhere).

Diagnostics: action mix (share of active intervals with any / high fluid, any /
high vasopressor, diuretic, no treatment, default action) under every strategy;
mean model-predicted MAP / lactate / SpO2 (raw units) by bin under every
strategy (mechanistic face validity: removing vasopressors should not RAISE
predicted MAP), plus the within-patient change vs UC among the patients whose
input changes (S3 vasopressor-exposed, S2 diuretic-exposed, S0 all; the
undiluted version of that check); share of active intervals in which R_ad equals R_na; the 70c
rule-decoder check; the usual-care raw->merged consistency check.

Reproduction checks: REPRO_CHECK (R_na - UC vs -1.37 / -2.27 pp),
REPRO_CHECK_UC (usual-care risks 11.78 % / 15.77 %) and REPRO_CHECK_LEGACY
(S0 - UC vs -1.37 / -2.27 pp; S0 == legacy zero-action arm). On the synthetic
fixture they FAIL by design.

Protocol reference: BMC Medicine revision, analysis BM4 (static comparators).
Compute: G1 per spec (qsub env/sge/submit_g1.sh; ~2-5 min wall on the 5,848 test
stays, dominated by data loading / DTRDataset construction). The seven rollouts
are light (~4 s each per 5,848 stays on 4 CPU threads with the 609k-parameter
80c model), so direct CPU execution (C2, --device cpu, 8 threads) also works
(~3-6 min).

Inputs (through common.load_all)
------
- configs/80_step_c_training_v1.yaml, configs/45c_normalization_params_v1.yaml,
  configs/70b_action_merge_map.yaml, configs/70c_rule_policy_v1.yaml
- data/interim/45c_state_tensors_v1_mimic.parquet
- data/interim/45d_outcomes_absorb_v1_mimic.parquet
- data/interim/70b_propensity_coarse3_v1_mimic.parquet
- data/interim/70c_rule_policy_actions_v1_mimic.parquet
- data/processed/phenotype_labels_frozen_v1_mimic.parquet
- models/80c_gtransformer_rollout_ow1_best.pt (default; reproduces the published
  targets) or --checkpoint models/80c_gtransformer_best.pt (P2 export)

Outputs
-------
- tables/bmc_medicine/BM4_strategy_summary.csv      (wide, one row per stratum)
- tables/bmc_medicine/BM4_strategy_contrasts_long.csv (every mean / contrast / CI)
- tables/bmc_medicine/BM4_action_mix.csv            (strategy x stratum)
- tables/bmc_medicine/BM4_predicted_states_by_bin.csv (strategy x stratum x bin)
- reports/bmc_medicine/BM4_summary.txt              (ends with the PASTE-BACK block)
- data/interim/bmc_medicine/BM4_perstay_predictions.parquet (stay-level; never sync)

Usage
-----
    # Shirokane (project root), GPU:
    qsub -N sBM4_static -q gpua.q -l a100=1 env/sge/submit_g1.sh \\
        scripts/bmc_medicine/BM4_static_comparators.py \\
        --checkpoint models/80c_gtransformer_rollout_ow1_best.pt
    # smoke test:
    env/run_py.sh scripts/bmc_medicine/BM4_static_comparators.py --quick --device cpu
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

TAG = "BM4"
N_PHASES = 8

# strategy key -> (per-stay column, description)
STRATEGIES = OrderedDict([
    ("UC", ("y_uc", "usual care (observed actions)")),
    ("S0", ("y_s0", "no treatment (raw 0 every interval)")),
    ("S1", ("y_s1", "default only (raw 6 = low/moderate fluid every interval)")),
    ("S2", ("y_s2", "usual care, diuretics removed")),
    ("S3", ("y_s3", "usual care, vasopressors removed")),
    ("R_na", ("y_rna", "rule, non-adaptive (stored 70c actions)")),
    ("R_ad", ("y_rad", "rule, adaptive (re-decided on simulated states)")),
])
COL = {k: v[0] for k, v in STRATEGIES.items()}
VALUE_COLS = [COL[k] for k in STRATEGIES]

# (contrast name, strategy a, strategy b) -> mean(a) - mean(b)
CONTRAST_SPECS = [
    ("S0_minus_UC", "S0", "UC"),
    ("S1_minus_UC", "S1", "UC"),
    ("S2_minus_UC", "S2", "UC"),
    ("S3_minus_UC", "S3", "UC"),
    ("R_na_minus_UC", "R_na", "UC"),
    ("R_ad_minus_UC", "R_ad", "UC"),
    ("R_ad_minus_S1", "R_ad", "S1"),
    ("R_na_minus_S1", "R_na", "S1"),
    ("R_ad_minus_R_na", "R_ad", "R_na"),
    ("R_na_minus_S0", "R_na", "S0"),
]
CONTRASTS = [(name, COL[a], COL[b]) for name, a, b in CONTRAST_SPECS]

PASTE_STRATA = ["all", "hfpef", "hfpef_nonshock", "hfpef_shock", "nohf_nonshock", "hfref"]
MIX_STRATA = PASTE_STRATA
STATE_STRATA = ["all", "hfpef", "hfpef_nonshock", "hfpef_shock", "nohf_nonshock"]
DOD_NAME = "hfpef_nonshock_minus_nohf_nonshock"
EXPOSURE_STRATA = [("all_diuretic_exposed", "exp_diuretic", "all"),
                   ("hfpef_diuretic_exposed", "exp_diuretic", "hfpef"),
                   ("all_vaso_exposed", "exp_vaso", "all"),
                   ("hfpef_vaso_exposed", "exp_vaso", "hfpef")]


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(p, n_boot_default=1000, model_based=True)
    p.add_argument("--fb-mode", choices=("recorded", "approx"), default="recorded",
                   help="fluid balance used by the adaptive rule R_ad: 'recorded' (primary) = recorded "
                        "cumulative_fluid_balance_L; 'approx' = rebuilt from the adaptive fluid level and "
                        "recorded urine output")
    p.add_argument("--vocab-method", choices=("dataset", "fast"), default="dataset",
                   help="training action vocabulary: 'dataset' (spec / 85e: DTRDataset on train ids) or "
                        "'fast' (same rule from the frames)")
    p.add_argument("--no-save-perstay", action="store_true",
                   help="do not write data/interim/bmc_medicine/BM4_perstay_predictions.parquet")
    return p.parse_args(argv)


# ═══════════════════════════════════════════════════════════════════════
# Strategy construction
# ═══════════════════════════════════════════════════════════════════════
def stack_item_actions(ds, key: str) -> np.ndarray:
    """(N, T) contiguous actions exactly as DTRDataset.__getitem__ returns them."""
    return np.stack([ds[i][key].numpy() for i in range(len(ds))]).astype(np.int64)


def build_static_strategies(ds, codec: common.ActionCodec, act: np.ndarray,
                            uc_contig: np.ndarray) -> tuple[dict, dict]:
    """Contiguous (N, T) action matrices for S0..S3 (+ checks)."""
    raw_obs = np.asarray(ds.action_raw, dtype=np.int64)
    f, v, d = common.decode_raw(raw_obs)
    merged_from_raw = codec.raw_to_merged(np.clip(raw_obs, 0, common.N_RAW_ACTIONS - 1))
    consistency = float((merged_from_raw[act] == np.asarray(ds.action)[act]).mean()) if act.any() else float("nan")

    full = np.ones_like(raw_obs)
    strategies = {
        "S0": codec.raw_to_contiguous(0 * full, active=act),
        "S1": codec.raw_to_contiguous(6 * full, active=act),
    }
    raw_s2 = common.encode_raw(f, v, 0 * d)
    raw_s3 = common.encode_raw(f, 0 * v, d)
    for name, raw_mod in (("S2", raw_s2), ("S3", raw_s3)):
        changed = act & (raw_mod != raw_obs)
        mod = codec.raw_to_contiguous(raw_mod, active=changed)
        # unchanged raw action -> keep the observed contiguous action exactly
        strategies[name] = np.where(changed, mod, np.where(act, uc_contig, 0))
    checks = {
        "uc_raw_merge_consistency": consistency,
        "uc_raw_diuretic_bins_pct": 100 * float((d[act] == 1).mean()) if act.any() else float("nan"),
        "uc_raw_vaso_bins_pct": 100 * float((v[act] > 0).mean()) if act.any() else float("nan"),
    }
    legacy = common.legacy_zero_actions(ds)
    checks["S0_equals_legacy_zero_arm"] = bool(np.array_equal(strategies["S0"], legacy))
    return strategies, checks


# ═══════════════════════════════════════════════════════════════════════
# Diagnostics
# ═══════════════════════════════════════════════════════════════════════
def action_mix_table(merged: dict, uc_merged: np.ndarray, act: np.ndarray, strata: dict) -> pd.DataFrame:
    """Share (%) of active intervals by action component, per strategy x stratum."""
    rows = []
    for sname, m in merged.items():
        f, v, d = common.decode_raw(m)
        changed = (m != uc_merged)
        for st in MIX_STRATA:
            pm = strata[st]
            bm = act & pm[:, None]
            nb = int(bm.sum())

            def pct(x):
                return 100 * float(x[bm].mean()) if nb else float("nan")
            rows.append({
                "strategy": sname, "stratum": st, "n_patients": int(pm.sum()), "n_active_bins": nb,
                "fluid_any_pct": pct(f > 0), "fluid_high_pct": pct(f == 2),
                "vaso_any_pct": pct(v > 0), "vaso_high_pct": pct(v == 2),
                "diuretic_pct": pct(d == 1), "no_treatment_pct": pct(m == 0),
                "default_action6_pct": pct(m == 6),
                "changed_vs_uc_bins_pct": pct(changed),
                "changed_vs_uc_patients_pct": (100 * float((changed & bm).any(axis=1)[pm].mean())
                                               if pm.sum() else float("nan")),
            })
    return pd.DataFrame(rows)


def state_summary(values: dict, act: np.ndarray, strata: dict, sname: str) -> list[dict]:
    """Mean predicted MAP / lactate / SpO2 by bin (active bins) per stratum."""
    rows = []
    T = act.shape[1]
    for st in STATE_STRATA:
        pm = strata[st]
        for t in range(T):
            bm = act[:, t] & pm
            r = {"strategy": sname, "stratum": st, "bin": t, "n_active": int(bm.sum())}
            for k in ("map", "lactate", "spo2"):
                r[f"{k}_mean"] = float(values[k][bm, t].mean()) if bm.any() else float("nan")
            rows.append(r)
        bm = act.copy()
        bm[:, 0] = False
        bm &= pm[:, None]
        r = {"strategy": sname, "stratum": st, "bin": "t1plus", "n_active": int(bm.sum())}
        for k in ("map", "lactate", "spo2"):
            r[f"{k}_mean"] = float(values[k][bm].mean()) if bm.any() else float("nan")
        rows.append(r)
    return rows


STATE_DIFF_SPECS = [("S3", "all_vaso_exposed"), ("S3", "hfpef_vaso_exposed"),
                    ("S2", "all_diuretic_exposed"), ("S2", "hfpef_diuretic_exposed"),
                    ("S0", "all"), ("S0", "hfpef")]


def exposed_state_diffs(decoded: dict, act: np.ndarray, strata: dict) -> "OrderedDict[str, dict]":
    """Mean within-patient predicted-state change (strategy - UC) over active bins t>=1.

    Restricted to the patients whose model input changes, so the face-validity
    signal (e.g. removing vasopressors should not RAISE predicted MAP) is not
    diluted by unexposed patients, whose predicted states are identical by construction.
    """
    out = OrderedDict()
    for sname, st in STATE_DIFF_SPECS:
        bm = act.copy()
        bm[:, 0] = False
        bm &= strata[st][:, None]
        nb = int(bm.sum())
        r = {"n_patients": int(strata[st].sum()), "n_bins": nb}
        for k in ("map", "lactate"):
            r[k] = float((decoded[sname][k] - decoded["UC"][k])[bm].mean()) if nb else float("nan")
        out[f"{sname}_minus_UC_{st}"] = r
    return out


def ci_direction(row: pd.Series) -> str:
    lo, hi = float(row["ci_lo"]), float(row["ci_hi"])
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "NA"
    if hi < 0:
        return "negative"
    if lo > 0:
        return "positive"
    return "includes_0"


def falsification_flag(row: pd.Series) -> str:
    """S0 - UC: withholding all treatment should not be predicted beneficial."""
    return {"negative": "CONCERN", "positive": "OK", "includes_0": "INCONCLUSIVE"}.get(ci_direction(row), "NA")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main(argv=None) -> int:
    t_start = time.time()
    args = common.apply_quick(parse_args(argv))
    import torch
    torch.manual_seed(args.seed)

    # ── 1. data ──────────────────────────────────────────────────────────
    t0 = common.phase(1, N_PHASES, "Load Step C data, split and strata")
    root = common.find_root(__file__)
    paths = common.setup_paths(root)
    common.log(f"PROJECT_ROOT={root}")
    bundle = common.load_all(root)
    common.log(f"  phase 1 done [{time.time() - t0:.1f}s]")

    # ── 2. vocab + test dataset ──────────────────────────────────────────
    t0 = common.phase(2, N_PHASES, "Training vocabulary and test dataset")
    vocab = common.training_vocab(bundle, method=args.vocab_method)
    test_ids = bundle.test_ids
    if args.quick:
        test_ids = common.subsample_ids(test_ids, common.QUICK_N_STAYS, args.seed)
        common.log(f"  QUICK: {len(test_ids)} of {len(bundle.test_ids)} test stays")
    ds = common.build_dataset(bundle, test_ids, vocab, label="test", strict_vocab=not args.quick)
    test_auto_vocab = sorted(int(a) for a in np.unique(ds.action[ds.valid_mask.astype(bool)]))
    act = common.active_mask(ds)
    common.log(f"  test stays={len(ds):,}; active bins={int(act.sum()):,}; bins/stay={act.sum(1).mean():.2f}")
    common.log(f"  phase 2 done [{time.time() - t0:.1f}s]")

    # ── 3. model, decoder, checks ────────────────────────────────────────
    t0 = common.phase(3, N_PHASES, "Model, rule decoder and rule reproduction check")
    device = common.resolve_device(args.device)
    ckpt_path = common.resolve_checkpoint(root, args.checkpoint)
    model, info = common.load_model(ckpt_path, ds, bundle.config, device)
    decoder = common.RuleDecoder.from_bundle(bundle, vocab)
    codec = decoder.codec
    rule_chk = common.rule_reproduction_check(bundle, ds, decoder)
    common.log(f"  phase 3 done [{time.time() - t0:.1f}s]")

    # ── 4. strategy action matrices ──────────────────────────────────────
    t0 = common.phase(4, N_PHASES, "Build strategy action matrices (raw -> merged -> contiguous)")
    uc_contig = stack_item_actions(ds, "action")
    rna_contig = stack_item_actions(ds, "rule_action")
    uc_check = codec.merged_to_contiguous(np.asarray(ds.action), active=act)
    if not np.array_equal(uc_check, uc_contig):
        raise RuntimeError("usual-care contiguous actions differ between codec and DTRDataset")
    static_actions, checks = build_static_strategies(ds, codec, act, uc_contig)
    stored_raw = common.stored_rule_raw_matrix(bundle, ds.stay_ids)
    rna_from_raw = codec.raw_to_contiguous(np.where(act, stored_raw, 0), active=act)
    checks["R_na_codec_equals_dataset"] = bool(np.array_equal(rna_from_raw, rna_contig))
    if not checks["R_na_codec_equals_dataset"]:
        common.log("WARNING: stored rule actions mapped through the codec differ from DTRDataset rule_action")
    common.log(f"  usual-care raw->merged consistency on active bins: "
               f"{100 * checks['uc_raw_merge_consistency']:.3f}%")
    if not checks["uc_raw_merge_consistency"] >= 0.999:
        # S2/S3 are built from the raw codes; if raw->merged does not reproduce the model's UC input,
        # their modified bins are not "UC minus one component" and S2/S3 must not be interpreted.
        common.log("WARNING: observed raw actions (action_idx) do not reproduce the merged UC actions; "
                   "S2/S3 contrasts are NOT interpretable (check the 70b action_idx / merge map)")
    common.log(f"  S0 == legacy zero-action arm: {checks['S0_equals_legacy_zero_arm']}")
    for k in ("S2", "S3"):
        ch = act & (static_actions[k] != uc_contig)
        common.log(f"  {k}: model input changed in {100 * ch.sum() / max(act.sum(), 1):.2f}% of active bins, "
                   f"{100 * ch.any(1).mean():.1f}% of patients")
    common.log(f"  phase 4 done [{time.time() - t0:.1f}s]")

    # ── 5. rollouts ──────────────────────────────────────────────────────
    t0 = common.phase(5, N_PHASES, "Counterfactual rollouts (UC, S0-S3, R_na fixed; R_ad adaptive)")
    fixed_actions = OrderedDict([("UC", "action"), ("S0", static_actions["S0"]), ("S1", static_actions["S1"]),
                                 ("S2", static_actions["S2"]), ("S3", static_actions["S3"]),
                                 ("R_na", "rule_action")])
    probs: dict[str, np.ndarray] = {}
    state_rows: list[dict] = []
    contig_used: dict[str, np.ndarray] = {"UC": uc_contig, "R_na": rna_contig, **static_actions}
    decoded: dict[str, dict] = {}   # strategy -> predicted MAP / lactate (raw units, (N, T))
    timings = {}
    frame0 = common.per_stay_frame(ds, bundle)
    strata0 = common.standard_strata(frame0, extended=False)
    for sname in tqdm(list(STRATEGIES), desc="  strategies", mininterval=1):
        ts = time.time()
        if sname == "R_ad":
            fb = common.stay_bin_matrix(bundle.state_df, ds.stay_ids, "cumulative_fluid_balance_L", bundle.n_bins)
            n_nan = int(np.isnan(fb[act]).sum())
            if n_nan:
                common.log(f"WARNING: {n_nan} active bins without recorded cumulative fluid balance (set to 0)")
            kw = {}
            if args.fb_mode == "approx":
                kw = dict(fluid_ml_by_level=common.fluid_ml_by_level(bundle),
                          urine_matrix=common.stay_bin_matrix(bundle.state_df, ds.stay_ids, "urine_output_ml",
                                                              bundle.n_bins))
            out = common.predict_adaptive(model, ds, decoder, device, fb_mode=args.fb_mode,
                                          fb_matrix=fb if args.fb_mode == "recorded" else None,
                                          batch_size=args.batch_size, desc="rollout R_ad", **kw)
            probs[sname] = out["prob"]
            contig_used["R_ad"] = np.where(act, out["contig_actions"], 0)
            rad_raw = out["raw_actions"]
            vals = {k: out[k] for k in ("map", "lactate", "spo2")}
            checks["R_ad_equals_R_na_bins_pct"] = 100 * float((rad_raw[act] == stored_raw[act]).mean())
            checks["R_ad_equals_R_na_bins_t0_pct"] = (100 * float((rad_raw[act[:, 0], 0] ==
                                                                   stored_raw[act[:, 0], 0]).mean())
                                                      if act[:, 0].any() else float("nan"))
            cats = pd.Series(out["categories"][act]).map(common.RULE_CATEGORY_GROUP).value_counts(normalize=True)
            checks["R_ad_category_shares"] = {k: 100 * float(v) for k, v in cats.items()}
        else:
            out = common.predict_fixed(model, ds, device, fixed_actions[sname], batch_size=args.batch_size,
                                       desc=f"rollout {sname}", keep_states=True)
            probs[sname] = out["prob"]
            vals = decoder.decode_np(out["predicted_states"])
            del out
        state_rows += state_summary(vals, act, strata0, sname)
        decoded[sname] = {k: vals[k] for k in ("map", "lactate")}
        n_bad = int((~np.isfinite(probs[sname])).sum())
        if n_bad:
            common.log(f"WARNING: {sname}: {n_bad} stays with non-finite predicted risk (excluded per column "
                       "by the bootstrap; paired contrasts may then be unpaired)")
        timings[sname] = time.time() - ts
    stored_cats = pd.Series(np.select([np.isin(stored_raw, [16, 14, 8]), stored_raw == 1, stored_raw == 6],
                                      ["perfusion_support", "diuretic", "default"], "other")[act])
    checks["R_na_category_shares"] = {k: 100 * float(v) for k, v in stored_cats.value_counts(normalize=True).items()}
    common.log("  mean risks: " + ", ".join(f"{k}={100 * probs[k].mean():.2f}%" for k in STRATEGIES))
    common.log(f"  phase 5 done [{time.time() - t0:.1f}s]")

    # ── 6. per-stay frame + paired bootstrap ─────────────────────────────
    t0 = common.phase(6, N_PHASES, f"Paired patient-level bootstrap ({args.n_boot} reps, seed {args.seed})")
    exp_diur = (act & (static_actions["S2"] != uc_contig)).any(axis=1)
    exp_vaso = (act & (static_actions["S3"] != uc_contig)).any(axis=1)
    frame = common.per_stay_frame(ds, bundle, **{COL[k]: probs[k] for k in STRATEGIES},
                                  exp_diuretic=exp_diur.astype(int), exp_vaso=exp_vaso.astype(int))
    strata = common.standard_strata(frame, extended=True)
    for name, flag, base in EXPOSURE_STRATA:
        strata[name] = frame[flag].to_numpy().astype(bool) & strata[base]
    res = common.paired_bootstrap(frame, VALUE_COLS, strata, n_boot=args.n_boot, seed=args.seed,
                                  contrasts=CONTRASTS, scale=100.0, desc="paired bootstrap (BM4)")
    common.log(f"  phase 6 done [{time.time() - t0:.1f}s]")

    # ── 7. reproduction checks + flags ───────────────────────────────────
    t0 = common.phase(7, N_PHASES, "Reproduction checks and falsification flags")

    def est(stratum, qty, kind):
        return common.get_est(res, stratum, qty, kind)

    rep = common.repro_check_extended(
        est("all", COL["UC"], "mean")["estimate"], est("hfpef", COL["UC"], "mean")["estimate"],
        est("all", "R_na_minus_UC", "diff")["estimate"], est("hfpef", "R_na_minus_UC", "diff")["estimate"],
        est("all", "S0_minus_UC", "diff")["estimate"], est("hfpef", "S0_minus_UC", "diff")["estimate"])
    if args.quick:
        common.log("  (quick mode: reproduction checks are not meaningful on a subsample)")
    flags = {st: falsification_flag(est(st, "S0_minus_UC", "diff")) for st in ("all", "hfpef")}
    common.log(f"  S0 falsification: all={flags['all']} hfpef={flags['hfpef']}")
    common.log(f"  phase 7 done [{time.time() - t0:.1f}s]")

    # ── 8. tables, interim, report ───────────────────────────────────────
    t0 = common.phase(8, N_PHASES, "Write tables, stay-level interim and report")
    res_out = res.copy()
    res_out["quantity"] = res_out["quantity"].replace({COL[k]: f"risk_{k}" for k in STRATEGIES})
    common.save_table(res_out, paths["tables"] / "BM4_strategy_contrasts_long.csv")

    summ_rows = []
    for st, m in strata.items():
        r = OrderedDict(stratum=st, n=int(m.sum()), deaths=int(frame.loc[m, "mortality_28d"].sum()),
                        observed_mortality_pct=(100 * float(frame.loc[m, "mortality_28d"].mean())
                                                if m.sum() else float("nan")))
        for k in STRATEGIES:
            row = est(st, COL[k], "mean")
            r[f"risk_{k}_pct"] = row["estimate"]
        for cname, _, _ in CONTRASTS:
            row = est(st, cname, "diff")
            r[f"{cname}_pp"] = row["estimate"]
            r[f"{cname}_lo"] = row["ci_lo"]
            r[f"{cname}_hi"] = row["ci_hi"]
        summ_rows.append(r)
    dod = OrderedDict(stratum=DOD_NAME, n=int(est(DOD_NAME, CONTRASTS[0][0], "diff_of_diff")["n"]))
    for cname, _, _ in CONTRASTS:
        row = est(DOD_NAME, cname, "diff_of_diff")
        dod[f"{cname}_pp"], dod[f"{cname}_lo"], dod[f"{cname}_hi"] = row["estimate"], row["ci_lo"], row["ci_hi"]
    summ_rows.append(dod)
    summary = pd.DataFrame(summ_rows)
    common.save_table(summary, paths["tables"] / "BM4_strategy_summary.csv")

    vocab_arr = np.asarray(vocab, dtype=np.int64)
    merged = OrderedDict((k, vocab_arr[contig_used[k]]) for k in STRATEGIES)
    mix = action_mix_table(merged, merged["UC"], act, strata)
    common.save_table(mix, paths["tables"] / "BM4_action_mix.csv")
    states_tab = pd.DataFrame(state_rows)
    states_tab["bin"] = states_tab["bin"].astype(str)
    common.save_table(states_tab, paths["tables"] / "BM4_predicted_states_by_bin.csv")
    state_diffs = exposed_state_diffs(decoded, act, strata)

    if not args.no_save_perstay:
        common.save_interim(frame, root, "BM4_perstay_predictions.parquet")

    # ---- report text -------------------------------------------------------
    lines = common.report_header("BM4 — Static comparator strategies under the G-Transformer DM rollout", root, args)
    if args.quick:
        lines += ["*** QUICK SMOKE TEST (subsample, few bootstrap reps): numbers are NOT for interpretation ***", ""]
    lines += [f"Checkpoint: {ckpt_path.name} (epoch {info['epoch']}, val_loss {common.fmt_num(info['val_loss'], 4)}, "
              f"{info['n_params']:,} parameters); device {device}",
              f"Training vocab ({len(vocab)}): {vocab}; test auto-discovered vocab: {test_auto_vocab}",
              f"Test stays: {len(ds):,}; active intervals: {int(act.sum()):,}; bootstrap reps {args.n_boot}, "
              f"seed {args.seed}; adaptive fluid-balance mode: {args.fb_mode}", "",
              "Strategies:"]
    lines += [f"  {k:5s} {v[1]}" for k, v in STRATEGIES.items()]
    lines += ["", "Construction checks:",
              f"  usual-care raw->merged consistency (active bins): {100 * checks['uc_raw_merge_consistency']:.3f}%",
              f"  S0 identical to the legacy zero-action arm: {checks['S0_equals_legacy_zero_arm']}",
              f"  stored rule raw -> codec equals DTRDataset rule_action: {checks['R_na_codec_equals_dataset']}",
              f"  RULE_DECODER_CHECK: {rule_chk['status']} (agreement with cumulative fluid balance "
              f"{common.fmt_num(100 * (rule_chk.get('cumulative_fluid_balance_L') or {}).get('agreement', np.nan), 3)}%)",
              f"  R_ad equals R_na in {common.fmt_num(checks['R_ad_equals_R_na_bins_pct'], 2)}% of active intervals "
              f"(t=0: {common.fmt_num(checks['R_ad_equals_R_na_bins_t0_pct'], 2)}%)",
              f"  rule category shares (% of active intervals): R_na {fmt_shares(checks['R_na_category_shares'])}; "
              f"R_ad {fmt_shares(checks['R_ad_category_shares'])}",
              f"  observed raw diuretic in {common.fmt_num(checks['uc_raw_diuretic_bins_pct'])}% and vasopressor in "
              f"{common.fmt_num(checks['uc_raw_vaso_bins_pct'])}% of active intervals; the 70b merge map folds "
              "raw 3/5/11/15/17 (vasopressor + diuretic) into their no-diuretic cells, so S2 changes the model "
              "input only for merged diuretic cells 1/7/9/13 and S3 can expose a diuretic hidden by the merge.",
              ""]
    lines += ["Per-stratum results (risk %, RD pp with paired-bootstrap 95% CI; negative = lower predicted "
              "mortality than the comparator). CIs resample test patients with the model FIXED: they reflect "
              "test-sample variability only, not model-estimation uncertainty or confounding bias."]
    for st in PASTE_STRATA + ["no_hf", "hfref_or_hfmref"]:
        m = strata[st]
        lines.append(f"  [{st}] n={int(m.sum()):,} deaths={int(frame.loc[m, 'mortality_28d'].sum()):,} observed="
                     f"{common.fmt_num(100 * frame.loc[m, 'mortality_28d'].mean() if m.sum() else np.nan)}%")
        lines.append("      risk: " + "  ".join(f"{k}={common.fmt_num(est(st, COL[k], 'mean')['estimate'])}"
                                                for k in STRATEGIES))
        for cname, _, _ in CONTRASTS:
            lines.append(f"      {cname:16s} {common.fmt_row_ci(est(st, cname, 'diff'))}")
    lines.append(f"  [{DOD_NAME}] (difference of non-shock contrasts, HFpEF minus no-HF)")
    for cname, _, _ in CONTRASTS:
        lines.append(f"      {cname:16s} {common.fmt_row_ci(est(DOD_NAME, cname, 'diff_of_diff'))}")
    lines += ["", "Effect of removing a component among patients whose model input changes:"]
    for name, flag, base in EXPOSURE_STRATA:
        cname = "S2_minus_UC" if "diuretic" in name else "S3_minus_UC"
        r = est(name, cname, "diff")
        lines.append(f"  {name:24s} n={int(r['n']):,} ({common.fmt_num(100 * r['n'] / max(strata[base].sum(), 1), 1)}% "
                     f"of {base}) {cname}: {common.fmt_row_ci(r)}")
    lines += ["", "Action mix (% of active intervals; merged codes as fed to the model):"]
    mix_cols = ["fluid_any_pct", "fluid_high_pct", "vaso_any_pct", "vaso_high_pct", "diuretic_pct",
                "no_treatment_pct", "default_action6_pct", "changed_vs_uc_bins_pct"]
    for st in ("all", "hfpef"):
        lines.append(f"  [{st}] " + " | ".join(c.replace("_pct", "") for c in mix_cols))
        for k in STRATEGIES:
            r = mix[(mix["strategy"] == k) & (mix["stratum"] == st)].iloc[0]
            lines.append(f"     {k:5s} " + " ".join(f"{common.fmt_num(r[c], 1):>6s}" for c in mix_cols))
    lines += ["", "Model-predicted states (raw units, mean over active intervals t>=1; t=0 is observed):"]
    for st in ("all", "hfpef"):
        for k in STRATEGIES:
            r = states_tab[(states_tab["strategy"] == k) & (states_tab["stratum"] == st) &
                           (states_tab["bin"] == "t1plus")].iloc[0]
            lines.append(f"  [{st}] {k:5s} MAP={common.fmt_num(r['map_mean'])} mmHg  lactate="
                         f"{common.fmt_num(r['lactate_mean'], 3)} mmol/L  SpO2={common.fmt_num(r['spo2_mean'])}%")
    lines += ["  Within-patient predicted-state change vs UC among patients whose input changes (active t>=1):"]
    for key, r in state_diffs.items():
        lines.append(f"    {key:32s} MAP {common.fmt_num(r['map'], 3)} mmHg, lactate {common.fmt_num(r['lactate'], 4)} "
                     f"mmol/L (n={r['n_patients']:,} patients, {r['n_bins']:,} bins)")
    lines += ["", "Reproduction checks:", f"  {rep['line']}", f"  {rep['uc_line']}"]
    if "legacy_line" in rep:
        lines.append(f"  {rep['legacy_line']}")
    lines += ["  " + common.LEGACY_RULE_NOTE, ""]
    lines += interpretation_lines(est, flags)
    lines += ["", "Rollout timings (s): " + ", ".join(f"{k}={v:.1f}" for k, v in timings.items()),
              f"Total runtime: {time.time() - t_start:.1f}s"]

    # ---- paste-back block --------------------------------------------------
    pb = OrderedDict()
    pb["script"] = "BM4_static_comparators"
    pb["quick"] = bool(args.quick)
    pb["checkpoint"] = ckpt_path.name
    pb["ckpt_epoch"] = info["epoch"]
    pb["ckpt_val_loss"] = float(info["val_loss"]) if info["val_loss"] is not None else float("nan")
    pb["device"] = str(device)
    pb["n_boot"] = int(args.n_boot)
    pb["seed"] = int(args.seed)
    pb["fb_mode_R_ad"] = args.fb_mode
    pb["train_vocab"] = " ".join(map(str, vocab))
    pb["test_vocab_auto"] = " ".join(map(str, test_auto_vocab))
    pb["n_test"] = len(ds)
    pb["n_active_bins"] = int(act.sum())
    pb["RULE_DECODER_CHECK"] = rule_chk["status"]
    pb["rule_decoder_agreement_cumfb_pct"] = 100 * float(
        (rule_chk.get("cumulative_fluid_balance_L") or {}).get("agreement", np.nan))
    pb["rule_decoder_agreement_intervalfb_pct"] = 100 * float(
        (rule_chk.get("fluid_balance_L") or {}).get("agreement", np.nan))
    pb["uc_raw_merge_consistency_pct"] = 100 * checks["uc_raw_merge_consistency"]
    pb["S0_equals_legacy_zero_arm"] = checks["S0_equals_legacy_zero_arm"]
    pb["R_ad_equals_R_na_bins_pct"] = checks["R_ad_equals_R_na_bins_pct"]
    pb["R_na_category_shares_pct"] = fmt_shares(checks["R_na_category_shares"])
    pb["R_ad_category_shares_pct"] = fmt_shares(checks["R_ad_category_shares"])
    for st in PASTE_STRATA:
        m = strata[st]
        pb[f"n_{st}"] = int(m.sum())
        pb[f"deaths_{st}"] = int(frame.loc[m, "mortality_28d"].sum())
        pb[f"obs_mort_{st}_pct"] = 100 * float(frame.loc[m, "mortality_28d"].mean()) if m.sum() else float("nan")
        pb[f"risk_{st}_pct"] = " ".join(f"{k}={common.fmt_num(est(st, COL[k], 'mean')['estimate'])}"
                                        for k in STRATEGIES)
        for cname, _, _ in CONTRASTS:
            pb[f"rd_{st}_{cname}_pp"] = common.fmt_row_ci(est(st, cname, "diff"))
    for cname, _, _ in CONTRASTS:
        pb[f"dod_{DOD_NAME}_{cname}_pp"] = common.fmt_row_ci(est(DOD_NAME, cname, "diff_of_diff"))
    for name, flag, base in EXPOSURE_STRATA:
        cname = "S2_minus_UC" if "diuretic" in name else "S3_minus_UC"
        r = est(name, cname, "diff")
        pb[f"n_{name}"] = int(r["n"])
        pb[f"rd_{name}_{cname}_pp"] = common.fmt_row_ci(r)
    pb["uc_raw_diuretic_bins_pct"] = checks["uc_raw_diuretic_bins_pct"]
    pb["uc_raw_vaso_bins_pct"] = checks["uc_raw_vaso_bins_pct"]
    for st in ("all", "hfpef"):
        for k in STRATEGIES:
            r = mix[(mix["strategy"] == k) & (mix["stratum"] == st)].iloc[0]
            pb[f"mix_{st}_{k}"] = (f"fluid_any={common.fmt_num(r['fluid_any_pct'], 1)} "
                                   f"fluid_high={common.fmt_num(r['fluid_high_pct'], 1)} "
                                   f"vaso_any={common.fmt_num(r['vaso_any_pct'], 1)} "
                                   f"diuretic={common.fmt_num(r['diuretic_pct'], 1)} "
                                   f"none={common.fmt_num(r['no_treatment_pct'], 1)} "
                                   f"default6={common.fmt_num(r['default_action6_pct'], 1)} "
                                   f"changed_vs_uc={common.fmt_num(r['changed_vs_uc_bins_pct'], 1)}")
    for st in ("all", "hfpef"):
        for k in STRATEGIES:
            r = states_tab[(states_tab["strategy"] == k) & (states_tab["stratum"] == st) &
                           (states_tab["bin"] == "t1plus")].iloc[0]
            r11 = states_tab[(states_tab["strategy"] == k) & (states_tab["stratum"] == st) &
                             (states_tab["bin"] == str(bundle.n_bins - 1))].iloc[0]
            pb[f"predstate_{st}_{k}"] = (f"map_t1plus={common.fmt_num(r['map_mean'])} "
                                         f"lactate_t1plus={common.fmt_num(r['lactate_mean'], 3)} "
                                         f"map_t{bundle.n_bins - 1}={common.fmt_num(r11['map_mean'])} "
                                         f"lactate_t{bundle.n_bins - 1}={common.fmt_num(r11['lactate_mean'], 3)}")
    for key, r in state_diffs.items():
        pb[f"predstate_diff_{key}"] = (f"map={common.fmt_num(r['map'], 3)} lactate={common.fmt_num(r['lactate'], 4)} "
                                       f"n={r['n_patients']}")
    pb["S0_falsification_all"] = flags["all"]
    pb["S0_falsification_hfpef"] = flags["hfpef"]
    pb["REPRO_CHECK"] = rep["status"]
    pb["REPRO_CHECK_UC"] = rep["uc_status"]
    pb["REPRO_CHECK_LEGACY"] = rep.get("legacy_status", "NA")
    pb["repro_rd_all_R_na_pp"] = float(est("all", "R_na_minus_UC", "diff")["estimate"])
    pb["repro_rd_hfpef_R_na_pp"] = float(est("hfpef", "R_na_minus_UC", "diff")["estimate"])
    pb["repro_rd_all_S0_legacy_pp"] = float(est("all", "S0_minus_UC", "diff")["estimate"])
    pb["repro_rd_hfpef_S0_legacy_pp"] = float(est("hfpef", "S0_minus_UC", "diff")["estimate"])
    pb["repro_risk_uc_all_pct"] = float(est("all", COL["UC"], "mean")["estimate"])
    pb["repro_risk_uc_hfpef_pct"] = float(est("hfpef", COL["UC"], "mean")["estimate"])
    pb["runtime_s"] = float(time.time() - t_start)

    common.write_report(paths["reports"] / "BM4_summary.txt", lines, pb, TAG)
    common.log(f"  phase 8 done [{time.time() - t0:.1f}s]")
    common.log(f"BM4 complete in {time.time() - t_start:.1f}s")
    return 0


def fmt_shares(d: dict) -> str:
    order = ["perfusion_support", "default", "diuretic", "other"]
    return " ".join(f"{k}={common.fmt_num(d.get(k, 0.0), 1)}" for k in order if k in d or k != "other")


def interpretation_lines(est, flags: dict) -> list[str]:
    """Data-driven reading aids (directions only; numbers are in the tables above)."""
    out = ["Interpretation aids (CI direction: negative = CI entirely < 0, positive = CI entirely > 0):"]

    def d(st, c):
        r = est(st, c, "diff")
        return f"{common.fmt_row_ci(r)} [{ci_direction(r)}]"
    for st in ("all", "hfpef", "hfpef_nonshock"):
        out.append(f"  [{st}] R_ad - S1 {d(st, 'R_ad_minus_S1')}; R_na - S1 {d(st, 'R_na_minus_S1')}; "
                   f"S1 - UC {d(st, 'S1_minus_UC')}")
    out += [
        "  * R_ad - S1 / R_na - S1 isolate the rule's state-responsive deviations from its default action "
        "(raw 6): a negative CI means the deviations (perfusion support / diuretic) are predicted to add benefit "
        "beyond the default mix; a CI including 0 with S1 - UC similar to R - UC means the predicted rule benefit "
        "is carried by the default action mix itself; a positive CI means the deviations are predicted harmful.",
        f"  * S0 falsification (S0 - UC; withholding all treatment): all={flags['all']}, hfpef={flags['hfpef']}. "
        "OK = predicted harmful (expected); CONCERN = the model credits withholding ALL treatment with lower "
        "mortality, which suggests residual confounding by indication (treated patients are sicker) and that "
        "contrasts moving actions toward 'less treatment' are biased toward benefit. S0 is the arm the "
        "pre-v1.0.1 DTRDataset fed to the model as 'rule'; if REPRO_CHECK_LEGACY passes, S0 - UC is the "
        "quantity behind the published -1.37 pp (HFpEF -2.27 pp).",
        "  * S2 - UC: change in predicted risk attributed to withholding the observed diuretics (look at the "
        "diuretic-exposed rows; elsewhere the difference is 0 by construction). S3 - UC: same for vasopressors; "
        "removing vasopressors in hypotensive patients should not lower predicted mortality and should not RAISE "
        "predicted MAP (compare the predicted-state lines for S3 and S0 with UC).",
    ]
    return out


if __name__ == "__main__":
    sys.exit(main())
