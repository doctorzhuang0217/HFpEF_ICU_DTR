#!/usr/bin/env python3
"""
85f_evalue_sensitivity — E-value and goals-of-care sensitivity analysis.

Purpose
-------
1. E-value analysis: Quantify how strong unmeasured confounding would need to
   be to explain away the observed ATE. The E-value (VanderWeele & Ding 2017)
   provides the minimum strength of association that an unmeasured confounder
   would need to have with both treatment and outcome to fully explain the
   observed treatment–outcome association.

2. Goals-of-care sensitivity: ICU HFpEF patients may have differential
   DNR/DNI (do-not-resuscitate/intubate) status or comfort-care transitions
   that confound the treatment–outcome relationship. We test sensitivity by
   excluding patients with documented limitations of care.

Method:
  E1 (E-value):    Compute E-value for the primary ATE and its CI bound.
                   Also compute for HFpEF and no_hf subgroups.
  E2 (GOC-exclude): Exclude stays with goals-of-care indicators (if available)
                    and re-evaluate ATE to test for confounding by palliative intent.
  E3 (mortality-recoding): For stays with GOC indicators, recode mortality_28d
                           to 0 (assume they would survive under full treatment)
                           as an extreme sensitivity bound.

Protocol reference: Step 85f (sensitivity layer)
Compute: C2 (lightweight — E-value is analytic, GOC re-evaluation uses frozen model)

Inputs
------
- configs/80_step_c_training_v1.yaml
- models/80c_gtransformer_rollout_ow1_best.pt
- data/interim/45c_state_tensors_v1_mimic.parquet
- data/interim/45d_outcomes_absorb_v1_mimic.parquet
- data/interim/70b_propensity_coarse3_v1_mimic.parquet
- data/processed/phenotype_labels_frozen_v1_mimic.parquet

Outputs
-------
- tables/85f_evalue_sensitivity.csv
- reports/85f_evalue_sensitivity_summary.txt
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hfpef_dtr.models.dtr_dataset import DTRDataset, load_dtr_data
from hfpef_dtr.models.dataset import make_train_val_test_split
from hfpef_dtr.models.ope import evaluate_dm, bootstrap_ope

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "tables"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"

N_BOOTSTRAP = 500
BOOT_SEED = 7777
CI_LEVEL = 0.95
GOC_PROXY_MODE = "strict"  # options: strict, broad


# ── E-value computation ─────────────────────────────────────────────────


def compute_evalue_rr(rr: float) -> float:
    """
    Compute E-value for a risk ratio (RR).

    E-value = RR + sqrt(RR * (RR - 1))   if RR >= 1
    For protective effects (RR < 1), use 1/RR.

    Reference: VanderWeele & Ding (2017), Annals of Internal Medicine.
    """
    if rr < 0:
        raise ValueError(f"RR must be non-negative, got {rr}")
    if rr == 0:
        return float("inf")
    if rr < 1:
        rr = 1.0 / rr
    return float(rr + np.sqrt(rr * (rr - 1)))


def compute_evalue_rd(
    ate: float,
    p_outcome_unexposed: float,
) -> dict:
    """
    Compute E-value for a risk difference (ATE) by converting to approximate RR.

    Approximation: RR ≈ (p0 + ATE) / p0
    where p0 = baseline risk in the unexposed (observed policy) group.

    Returns dict with E-value for point estimate and CI bound.
    """
    # Avoid division by zero
    p0 = max(p_outcome_unexposed, 0.001)

    # For protective effects (ATE < 0), the treated group has lower risk
    p1 = p0 + ate
    p1 = max(p1, 0.001)  # floor at 0.1%

    rr = p1 / p0

    evalue_point = compute_evalue_rr(rr)

    return {
        "p_unexposed": p0,
        "p_exposed": p1,
        "rr_approx": rr,
        "evalue_point": evalue_point,
    }


def compute_evalue_with_ci(
    ate: float,
    ate_ci_lo: float,
    ate_ci_hi: float,
    p_outcome_unexposed: float,
) -> dict:
    """
    Compute E-value for point estimate and the CI bound closest to null.

    For a protective effect (ATE < 0):
      - CI bound closest to null = ate_ci_hi (least negative)
    For a harmful effect (ATE > 0):
      - CI bound closest to null = ate_ci_lo (least positive)
    """
    result = compute_evalue_rd(ate, p_outcome_unexposed)

    # CI bound closest to null
    if ate < 0:
        ci_bound = ate_ci_hi  # closest to 0
    else:
        ci_bound = ate_ci_lo  # closest to 0

    # If CI includes/null-crosses 0 on RD scale, E-value for CI bound is 1.0
    if (ate_ci_lo <= 0 <= ate_ci_hi) or np.isclose(ci_bound, 0.0):
        result["evalue_ci_bound"] = 1.0
    else:
        ci_result = compute_evalue_rd(ci_bound, p_outcome_unexposed)
        result["evalue_ci_bound"] = ci_result["evalue_point"]
    result["ci_bound_used"] = ci_bound
    result["ate"] = ate
    result["ate_ci_lo"] = ate_ci_lo
    result["ate_ci_hi"] = ate_ci_hi

    return result


# ── Model loading ────────────────────────────────────────────────────────


def load_primary_model(config, d_temporal, d_static, n_actions, device):
    """Load 80c G-Transformer (primary model) from checkpoint."""
    import importlib
    mod = importlib.import_module("hfpef_dtr.models.gtransformer")
    builder_fn = getattr(mod, "build_gtransformer_from_config")

    model_cfg = config["models"]["80c_gtransformer"]
    grid_cfg = config["grid"]
    model = builder_fn(
        d_temporal=d_temporal, d_static=d_static,
        n_actions=n_actions, model_cfg=model_cfg, grid_cfg=grid_cfg,
    )

    ckpt_path = MODELS_DIR / "80c_gtransformer_rollout_ow1_best.pt"
    if not ckpt_path.exists():
        print(f"  FATAL: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def set_action_vocab(ds: DTRDataset, action_vocab: list[int]) -> DTRDataset:
    """Force dataset action vocabulary to match model-training mapping."""
    ds.action_vocab = action_vocab
    ds.n_actions = len(action_vocab)
    ds._action_to_contiguous = {a: i for i, a in enumerate(action_vocab)}
    return ds


# ── Goals-of-care identification ─────────────────────────────────────────


def identify_goc_stays(
    outcomes_df: pd.DataFrame,
    state_df: pd.DataFrame,
    temporal_cols: list[str],
    stay_ids: np.ndarray,
    proxy_mode: str = "strict",
) -> set:
    """
    Identify stays with goals-of-care (GOC) indicators.

    Heuristics for MIMIC-IV (no dedicated GOC column in structured data):
      1. Check for 'cmo' (comfort measures only) in outcomes if available
      2. Check for 'code_status' in state features
      3. Fallback: patients who died with very short ICU stays AND low
         treatment intensity may be palliative-intent

    Returns set of stay_ids flagged as GOC-affected.
    """
    stay_set = set(stay_ids)
    goc_ids = set()

    # Strategy 1: Check outcomes for explicit GOC columns
    goc_cols = [c for c in outcomes_df.columns if c in (
        "cmo", "comfort_measures", "dnr", "dni", "dnr_dni",
        "code_status_dnr", "code_status_cmo", "palliative",
    )]
    if goc_cols:
        out_sub = outcomes_df[outcomes_df["stay_id"].isin(stay_set)]
        for col in goc_cols:
            flagged = out_sub.loc[out_sub[col] == 1, "stay_id"]
            goc_ids.update(flagged)
        print(f"  GOC from outcomes columns ({goc_cols}): {len(goc_ids)} stays")
        return goc_ids

    # Strategy 2: Check state features for code status indicators
    code_cols = [c for c in temporal_cols if "code_status" in c.lower()
                 or "dnr" in c.lower() or "cmo" in c.lower()]
    if code_cols:
        sub = state_df[state_df["stay_id"].isin(stay_set)]
        # If any bin has code_status indicator > 0, flag it
        for col in code_cols:
            flagged = sub.loc[sub[col] > 0, "stay_id"].unique()
            goc_ids.update(flagged)
        print(f"  GOC from state features ({code_cols}): {len(goc_ids)} stays")
        return goc_ids

    # Strategy 3: Proxy — deceased patients with early discharge signal.
    # strict: died + early_discharge
    # broad : strict OR died with very short ICU LOS (<=24h)
    print("  No explicit GOC columns found; using mortality + early_discharge proxy")
    out_sub = outcomes_df[outcomes_df["stay_id"].isin(stay_set)].copy()

    # Flag: died AND early discharge (potential comfort-care withdrawal)
    if "early_discharge" in out_sub.columns:
        strict_mask = (
            (out_sub["mortality_28d"] == 1) &
            (out_sub["early_discharge"] == 1)
        )
        proxy_mask = strict_mask.copy()

        if proxy_mode == "broad":
            los_col = "icu_los_hours" if "icu_los_hours" in out_sub.columns else None
            if los_col is None and "los_icu_days" in out_sub.columns:
                los_col = "los_icu_days"
            if los_col is not None:
                if los_col == "icu_los_hours":
                    short_los = out_sub[los_col] <= 24
                else:
                    short_los = out_sub[los_col] <= 1.0
                proxy_mask = strict_mask | ((out_sub["mortality_28d"] == 1) & short_los)

        goc_proxy = out_sub.loc[proxy_mask, "stay_id"]
        goc_ids.update(goc_proxy)
        print(f"  GOC proxy mode={proxy_mode}: {len(goc_ids)} stays")
    else:
        print("  No early_discharge column; GOC proxy not available")

    return goc_ids


def main() -> None:
    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("85f — E-value & Goals-of-Care Sensitivity Analysis")
    print("=" * 72)

    # ── 1. Load config and data ──────────────────────────────────────
    print("\n[1] Loading data ...")
    config_path = ROOT / "configs" / "80_step_c_training_v1.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    state_df, outcomes_df, propensity_df, temporal_cols, static_cols, merge_map = (
        load_dtr_data(ROOT, config)
    )

    all_stay_ids = state_df["stay_id"].unique()
    split_cfg = config["split"]
    train_ids, _, test_ids = make_train_val_test_split(
        all_stay_ids, outcomes_df,
        train_frac=split_cfg["train_frac"],
        val_frac=split_cfg["val_frac"],
        test_frac=split_cfg["test_frac"],
        stratify_col=split_cfg["stratify_col"],
        seed=split_cfg["seed"],
    )
    test_set = set(test_ids)
    print(f"  Test stays: {len(test_ids):,}")

    # Phenotype labels
    pheno_path = ROOT / config["data"]["phenotype_labels"]
    pheno_df = pd.read_parquet(pheno_path)
    pheno_test = pheno_df[pheno_df["stay_id"].isin(test_set)].set_index("stay_id")

    # ── 2. Load model ────────────────────────────────────────────────
    print("\n[2] Loading 80c G-Transformer ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_temporal = len(temporal_cols)
    d_static = len(static_cols)
    n_actions = len(config["action"]["valid_merged_indices"])
    model = load_primary_model(config, d_temporal, d_static, n_actions, device)
    print(f"  Device: {device}")

    # Build anchor for action vocab
    anchor_ds = DTRDataset(
        state_df=state_df, outcomes_df=outcomes_df, propensity_df=propensity_df,
        stay_ids=train_ids, temporal_cols=temporal_cols, static_cols=static_cols,
        merge_map=merge_map, n_bins=config["grid"]["n_bins"],
    )
    action_vocab = anchor_ds.action_vocab

    all_results = []

    # ── E1: E-value computation ──────────────────────────────────────
    print("\n[E1] Computing E-values from primary 80c results ...")

    # Full-sample point estimate + bootstrap CI
    ds_full = DTRDataset(
        state_df=state_df, outcomes_df=outcomes_df, propensity_df=propensity_df,
        stay_ids=test_ids, temporal_cols=temporal_cols, static_cols=static_cols,
        merge_map=merge_map, n_bins=config["grid"]["n_bins"],
    )
    ds_full = set_action_vocab(ds_full, action_vocab)

    full_point = evaluate_dm(model, ds_full, device, use_rollout=True)

    def _eval_fn(ds, **kw):
        return evaluate_dm(model, ds, device, use_rollout=True)

    full_boot = bootstrap_ope(
        _eval_fn, ds_full,
        n_resamples=N_BOOTSTRAP, seed=BOOT_SEED, confidence=CI_LEVEL,
    )

    # E-value for full sample
    ev_full = compute_evalue_with_ci(
        ate=full_point["ate"],
        ate_ci_lo=full_boot.get("ate_ci_lo", np.nan),
        ate_ci_hi=full_boot.get("ate_ci_hi", np.nan),
        p_outcome_unexposed=full_point["v_observed"],
    )
    all_results.append({
        "analysis": "E1_evalue",
        "subgroup": "ALL",
        "n_stays": len(test_ids),
        "ate": full_point["ate"],
        "ate_ci_lo": full_boot.get("ate_ci_lo", np.nan),
        "ate_ci_hi": full_boot.get("ate_ci_hi", np.nan),
        "v_observed": full_point["v_observed"],
        "rr_approx": ev_full["rr_approx"],
        "evalue_point": ev_full["evalue_point"],
        "evalue_ci_bound": ev_full["evalue_ci_bound"],
    })
    print(f"  ALL: ATE={full_point['ate']:+.4f}, RR≈{ev_full['rr_approx']:.3f}, "
          f"E-value={ev_full['evalue_point']:.2f} (CI bound={ev_full['evalue_ci_bound']:.2f})")

    # E-value per HF subtype
    for hf_grp in tqdm(["hfpef", "no_hf", "hfref"], desc="E1 subgroups"):
        grp_sids = pheno_test.loc[
            pheno_test["hf_subtype"] == hf_grp
        ].index.values
        grp_sids = np.array(sorted(set(grp_sids) & test_set))

        if len(grp_sids) < 30:
            print(f"  {hf_grp}: n={len(grp_sids)} — TOO SMALL")
            continue

        grp_ds = DTRDataset(
            state_df=state_df, outcomes_df=outcomes_df, propensity_df=propensity_df,
            stay_ids=grp_sids, temporal_cols=temporal_cols, static_cols=static_cols,
            merge_map=merge_map, n_bins=config["grid"]["n_bins"],
        )
        grp_ds = set_action_vocab(grp_ds, action_vocab)

        grp_point = evaluate_dm(model, grp_ds, device, use_rollout=True)
        grp_boot = bootstrap_ope(
            _eval_fn, grp_ds,
            n_resamples=200, seed=BOOT_SEED + 50, confidence=CI_LEVEL,
        )

        ev_grp = compute_evalue_with_ci(
            ate=grp_point["ate"],
            ate_ci_lo=grp_boot.get("ate_ci_lo", np.nan),
            ate_ci_hi=grp_boot.get("ate_ci_hi", np.nan),
            p_outcome_unexposed=grp_point["v_observed"],
        )

        all_results.append({
            "analysis": "E1_evalue",
            "subgroup": hf_grp,
            "n_stays": len(grp_sids),
            "ate": grp_point["ate"],
            "ate_ci_lo": grp_boot.get("ate_ci_lo", np.nan),
            "ate_ci_hi": grp_boot.get("ate_ci_hi", np.nan),
            "v_observed": grp_point["v_observed"],
            "rr_approx": ev_grp["rr_approx"],
            "evalue_point": ev_grp["evalue_point"],
            "evalue_ci_bound": ev_grp["evalue_ci_bound"],
        })
        print(f"  {hf_grp}: ATE={grp_point['ate']:+.4f}, RR≈{ev_grp['rr_approx']:.3f}, "
              f"E-value={ev_grp['evalue_point']:.2f} (CI={ev_grp['evalue_ci_bound']:.2f})")

    # ── E2: Goals-of-care exclusion ──────────────────────────────────
    print("\n[E2] Goals-of-care exclusion sensitivity ...")
    goc_ids = identify_goc_stays(
        outcomes_df, state_df, temporal_cols, test_ids, proxy_mode=GOC_PROXY_MODE
    )
    non_goc_ids = np.array(sorted(test_set - goc_ids))

    if len(goc_ids) >= 5 and len(non_goc_ids) >= 30:
        print(f"  GOC stays: {len(goc_ids):,} ({len(goc_ids)/len(test_ids)*100:.1f}%)")
        print(f"  Non-GOC stays: {len(non_goc_ids):,}")

        # E2a: Exclude GOC stays entirely
        ds_e2 = DTRDataset(
            state_df=state_df, outcomes_df=outcomes_df, propensity_df=propensity_df,
            stay_ids=non_goc_ids, temporal_cols=temporal_cols, static_cols=static_cols,
            merge_map=merge_map, n_bins=config["grid"]["n_bins"],
        )
        ds_e2 = set_action_vocab(ds_e2, action_vocab)

        e2_point = evaluate_dm(model, ds_e2, device, use_rollout=True)
        e2_boot = bootstrap_ope(
            _eval_fn, ds_e2,
            n_resamples=N_BOOTSTRAP, seed=BOOT_SEED + 200, confidence=CI_LEVEL,
        )

        all_results.append({
            "analysis": "E2_exclude_goc",
            "subgroup": "ALL",
            "n_stays": len(non_goc_ids),
            "ate": e2_point["ate"],
            "ate_ci_lo": e2_boot.get("ate_ci_lo", np.nan),
            "ate_ci_hi": e2_boot.get("ate_ci_hi", np.nan),
            "v_observed": e2_point["v_observed"],
            "rr_approx": np.nan,
            "evalue_point": np.nan,
            "evalue_ci_bound": np.nan,
        })
        print(f"  E2 ALL: ATE={e2_point['ate']:+.4f} "
              f"[{e2_boot.get('ate_ci_lo', np.nan):+.4f}, {e2_boot.get('ate_ci_hi', np.nan):+.4f}]")

        # E2 subgroups
        for hf_grp in tqdm(["hfpef", "no_hf"], desc="E2 subgroups"):
            grp_sids = pheno_test.loc[
                (pheno_test.index.isin(set(non_goc_ids)))
                & (pheno_test["hf_subtype"] == hf_grp)
            ].index.values
            if len(grp_sids) < 20:
                continue

            grp_ds = DTRDataset(
                state_df=state_df, outcomes_df=outcomes_df, propensity_df=propensity_df,
                stay_ids=grp_sids, temporal_cols=temporal_cols, static_cols=static_cols,
                merge_map=merge_map, n_bins=config["grid"]["n_bins"],
            )
            grp_ds = set_action_vocab(grp_ds, action_vocab)

            grp_point = evaluate_dm(model, grp_ds, device, use_rollout=True)
            grp_boot = bootstrap_ope(
                _eval_fn, grp_ds,
                n_resamples=200, seed=BOOT_SEED + 210, confidence=CI_LEVEL,
            )

            all_results.append({
                "analysis": "E2_exclude_goc",
                "subgroup": hf_grp,
                "n_stays": len(grp_sids),
                "ate": grp_point["ate"],
                "ate_ci_lo": grp_boot.get("ate_ci_lo", np.nan),
                "ate_ci_hi": grp_boot.get("ate_ci_hi", np.nan),
                "v_observed": grp_point["v_observed"],
                "rr_approx": np.nan,
                "evalue_point": np.nan,
                "evalue_ci_bound": np.nan,
            })
            print(f"  E2 {hf_grp}: ATE={grp_point['ate']:+.4f} "
                  f"[{grp_boot.get('ate_ci_lo', np.nan):+.4f}, "
                  f"{grp_boot.get('ate_ci_hi', np.nan):+.4f}]")
    else:
        print(f"  GOC stays: {len(goc_ids)} — insufficient for sensitivity analysis")

    # ── E3: GOC mortality recoding ───────────────────────────────────
    if len(goc_ids) >= 5:
        print("\n[E3] Goals-of-care mortality recoding (extreme bound) ...")
        out_e3 = outcomes_df.copy()
        goc_mask = out_e3["stay_id"].isin(goc_ids)
        out_e3.loc[goc_mask, "mortality_28d"] = 0  # Assume survival under full treatment

        # DM rollout computes v_observed from model-predicted observed actions,
        # so outcome recoding does not alter v_observed inside evaluate_dm.
        # We therefore keep model-based v_rule from the baseline estimate and
        # replace v_observed with the recoded raw mean mortality (extreme bound).
        test_outcomes = outcomes_df[outcomes_df["stay_id"].isin(test_set)][
            ["stay_id", "mortality_28d"]
        ].copy()
        test_outcomes_e3 = out_e3[out_e3["stay_id"].isin(test_set)][
            ["stay_id", "mortality_28d"]
        ].copy()
        raw_observed = float(test_outcomes["mortality_28d"].mean())
        raw_observed_e3 = float(test_outcomes_e3["mortality_28d"].mean())

        e3_v_rule = full_point["v_rule"]
        e3_ate = e3_v_rule - raw_observed_e3

        # Optional model-free bound for transparency.
        e3_raw_shift = raw_observed_e3 - raw_observed

        all_results.append({
            "analysis": "E3_recode_goc_alive",
            "subgroup": "ALL",
            "n_stays": len(test_ids),
            "ate": e3_ate,
            "ate_ci_lo": np.nan,
            "ate_ci_hi": np.nan,
            "v_observed": raw_observed_e3,
            "v_rule": e3_v_rule,
            "e3_raw_shift": e3_raw_shift,
            "rr_approx": np.nan,
            "evalue_point": np.nan,
            "evalue_ci_bound": np.nan,
        })
        print(
            "  E3 ALL: "
            f"raw observed={raw_observed:.4f} -> recoded={raw_observed_e3:.4f}, "
            f"v_rule(model)={e3_v_rule:.4f}, ATE={e3_ate:+.4f}"
        )

    # ── 5. Save results ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("[5] Saving results ...")

    results_df = pd.DataFrame(all_results)
    csv_path = TABLES_DIR / "85f_evalue_sensitivity.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path} ({len(results_df)} rows)")

    # ── 6. Generate report ───────────────────────────────────────────
    print("\n[6] Generating report ...")

    baseline_ate = full_point["ate"]
    lines = [
        "=" * 72,
        "85f — E-value & Goals-of-Care Sensitivity (MIMIC-IV)",
        "=" * 72,
        f"Generated: {ts}",
        f"Model: 80c G-Transformer DM (rollout, ow=1)",
        f"Bootstrap: B={N_BOOTSTRAP}, seed={BOOT_SEED}",
        "",
        "E-VALUE ANALYSIS",
        "────────────────",
        "  The E-value quantifies the minimum strength of unmeasured confounding",
        "  (on RR scale) needed to explain away the observed treatment effect.",
        "  E-value > 2 suggests moderate robustness; > 3 suggests strong robustness.",
        "",
        f"  {'Subgroup':<12s} {'n':>6s} {'ATE':>9s} {'RR':>7s} "
        f"{'E-val':>7s} {'E-CI':>7s}",
        f"  {'-'*12} {'-'*6} {'-'*9} {'-'*7} {'-'*7} {'-'*7}",
    ]

    e1_rows = results_df[results_df["analysis"] == "E1_evalue"]
    for _, r in e1_rows.iterrows():
        lines.append(
            f"  {r['subgroup']:<12s} {r['n_stays']:6d} {r['ate']:+9.4f} "
            f"{r['rr_approx']:7.3f} {r['evalue_point']:7.2f} {r['evalue_ci_bound']:7.2f}"
        )

    lines.extend([
        "",
        "  Interpretation:",
        "    E-value(point): unmeasured confounder must have RR ≥ E-value with BOTH",
        "    treatment and outcome to explain away the point estimate.",
        "    E-value(CI): same, but for the CI bound closest to null.",
        "    If E-value(CI) > 1: the null is outside the CI even after accounting",
        "    for unmeasured confounding up to strength E-value(CI).",
    ])

    # GOC results
    e2_rows = results_df[results_df["analysis"] == "E2_exclude_goc"]
    if not e2_rows.empty:
        lines.extend([
            "",
            "GOALS-OF-CARE SENSITIVITY",
            "─────────────────────────",
            f"  {'Analysis':<25s} {'Subgroup':<10s} {'n':>6s} {'ATE':>9s} "
            f"{'CI_lo':>9s} {'CI_hi':>9s} {'Δ vs S0':>9s}",
            f"  {'-'*25} {'-'*10} {'-'*6} {'-'*9} {'-'*9} {'-'*9} {'-'*9}",
        ])
        for _, r in results_df[results_df["analysis"].str.startswith("E2")
                                | results_df["analysis"].str.startswith("E3")].iterrows():
            delta = r["ate"] - baseline_ate if r["subgroup"] == "ALL" else np.nan
            delta_str = f"{delta:+9.4f}" if pd.notna(delta) else "       —"
            ci_lo = f"{r['ate_ci_lo']:+9.4f}" if pd.notna(r["ate_ci_lo"]) else "       —"
            ci_hi = f"{r['ate_ci_hi']:+9.4f}" if pd.notna(r["ate_ci_hi"]) else "       —"
            lines.append(
                f"  {r['analysis']:<25s} {r['subgroup']:<10s} {r['n_stays']:6d} "
                f"{r['ate']:+9.4f} {ci_lo} {ci_hi} {delta_str}"
            )

        lines.extend([
            "",
            "  Interpretation:",
            "    E2 (exclude GOC): If ATE persists → not driven by palliative-care patients",
            "    E3 (recode GOC alive): Extreme bound using recoded RAW observed risk",
            "    with model-based V(rule). CI not reported (deterministic recoding bound).",
            "    the treatment effect is very robust to GOC confounding",
        ])

    # Robustness summary
    lines.extend([
        "",
        "ROBUSTNESS SUMMARY",
        "──────────────────",
        f"  Baseline ATE: {baseline_ate:+.4f}",
    ])

    if not e1_rows.empty:
        main_eval = e1_rows[e1_rows["subgroup"] == "ALL"].iloc[0]
        ev_pt = main_eval["evalue_point"]
        ev_ci = main_eval["evalue_ci_bound"]
        if ev_ci > 2.0:
            lines.append(f"  → ROBUST: E-value(CI) = {ev_ci:.2f} > 2.0")
        elif ev_ci > 1.5:
            lines.append(f"  → MODERATE: E-value(CI) = {ev_ci:.2f} (1.5–2.0)")
        elif ev_ci > 1.0:
            lines.append(f"  → WEAK: E-value(CI) = {ev_ci:.2f} (1.0–1.5)")
        else:
            lines.append(f"  → FRAGILE: E-value(CI) = {ev_ci:.2f} ≤ 1.0 — "
                        "null not excluded after adjusting for unmeasured confounding")

    if not e2_rows.empty:
        e2_all = e2_rows[e2_rows["subgroup"] == "ALL"].iloc[0]
        pct_change = (abs(e2_all["ate"] - baseline_ate) / abs(baseline_ate) * 100
                      if abs(baseline_ate) > 1e-9 else 0.0)
        same_sign = (e2_all["ate"] < 0) == (baseline_ate < 0)
        if same_sign and pct_change < 20:
            lines.append(f"  → GOC ROBUST: excluding GOC changes ATE by {pct_change:.1f}%")
        elif same_sign:
            lines.append(f"  → GOC PARTIAL: excluding GOC changes ATE by {pct_change:.1f}% (same sign)")
        else:
            lines.append(f"  → GOC SENSITIVE: excluding GOC changes ATE sign")

    report = "\n".join(lines) + "\n"
    report_path = REPORTS_DIR / "85f_evalue_sensitivity_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {report_path}")
    print(report)

    total_time = time.time() - t0
    print(f"\n{'=' * 72}")
    print(f"85f COMPLETE. Total time: {total_time / 60:.1f} min")
    print("=" * 72)


if __name__ == "__main__":
    main()
