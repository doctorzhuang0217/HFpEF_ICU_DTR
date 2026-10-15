#!/usr/bin/env python3
"""
85j_informative_missingness_sensitivity — Test whether informative missingness
features and measurement intensity materially alter estimated treatment effects.

Purpose
-------
ICU missingness is informative: sicker patients receive more frequent
measurements. The 40c missingness features (_mask, _tsl_h, _freq_24h,
_never_measured) are included in state tensors and thus influence the
80c G-Transformer's outcome predictions. This sensitivity test evaluates
whether the ATE is driven by measurement intensity patterns rather than
treatment effects.

Approach:
  A1 (baseline):  Full state tensor as-is (includes missingness features)
  A2 (ablate):    Zero out all missingness features → re-evaluate ATE
  A3 (shuffle):   Randomly permute missingness features across stays
                  (destroys miss→outcome association while preserving marginals)
  A4 (intensity): Stratify by measurement intensity quartiles, evaluate
                  ATE per quartile to check for measurement-driven heterogeneity

For A2/A3, we modify the test DTRDataset's temporal tensor in-place before
evaluating DM. The outcome model (80c) is FROZEN — no retraining. This tests
whether the model's ATE estimate relies on missingness features.

Protocol reference: Step 85j (sensitivity layer)
Compute: C3 (modified forward passes + bootstrap)

Inputs
------
- configs/80_step_c_training_v1.yaml
- models/80c_gtransformer_rollout_ow1_best.pt
- data/interim/45c_state_tensors_v1_mimic.parquet
- data/interim/45d_outcomes_absorb_v1_mimic.parquet
- data/interim/70b_propensity_coarse3_v1_mimic.parquet
- data/processed/phenotype_labels_frozen_v1_mimic.parquet
- configs/45c_normalization_params_v1.yaml (to identify missingness feature columns)

Outputs
-------
- tables/85j_missingness_sensitivity.csv
- reports/85j_missingness_sensitivity_summary.txt
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
N_SHUFFLE_REPS = 5  # Average over multiple shuffles for A3


def set_action_vocab(ds: DTRDataset, action_vocab: list[int]) -> DTRDataset:
    """Force dataset action vocabulary to match model-training mapping."""
    ds.action_vocab = action_vocab
    ds.n_actions = len(action_vocab)
    ds._action_to_contiguous = {a: i for i, a in enumerate(action_vocab)}
    return ds


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


def identify_missingness_columns(temporal_cols: list[str]) -> list[int]:
    """
    Find indices of missingness-related features in temporal_cols.

    Missingness features from 40c have suffixes: _mask, _tsl_h, _freq_24h, _never_measured
    """
    miss_suffixes = ("_mask", "_tsl_h", "_freq_24h", "_never_measured")
    miss_indices = []
    for i, col in enumerate(temporal_cols):
        if any(col.endswith(s) for s in miss_suffixes):
            miss_indices.append(i)
    return miss_indices


def compute_measurement_intensity(
    state_df: pd.DataFrame,
    temporal_cols: list[str],
    stay_ids: np.ndarray,
) -> pd.DataFrame:
    """
    Compute per-stay measurement intensity as mean of _mask features.

    Higher intensity = more measurements = potentially sicker patient.
    """
    mask_cols = [c for c in temporal_cols if c.endswith("_mask")]
    if not mask_cols:
        # Fallback: count non-zero values per stay
        print("  WARNING: No _mask columns found; using raw obs count")
        sub = state_df[state_df["stay_id"].isin(set(stay_ids))]
        counts = sub.groupby("stay_id").size().reset_index(name="measurement_intensity")
        return counts

    sub = state_df[state_df["stay_id"].isin(set(stay_ids))].copy()
    # Mean observation rate across all variables and bins per stay
    intensity = sub.groupby("stay_id")[mask_cols].mean().mean(axis=1).reset_index()
    intensity.columns = ["stay_id", "measurement_intensity"]
    return intensity


def main() -> None:
    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("85j — Informative Missingness Sensitivity Analysis")
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

    anchor_ds = DTRDataset(
        state_df=state_df, outcomes_df=outcomes_df, propensity_df=propensity_df,
        stay_ids=train_ids, temporal_cols=temporal_cols,
        static_cols=static_cols, merge_map=merge_map, n_bins=config["grid"]["n_bins"],
    )
    action_vocab = anchor_ds.action_vocab

    # Identify missingness feature indices
    miss_indices = identify_missingness_columns(temporal_cols)
    miss_cols = [temporal_cols[i] for i in miss_indices]
    n_miss = len(miss_indices)
    n_total = len(temporal_cols)
    print(f"  Temporal features: {n_total} total, {n_miss} missingness ({n_miss/n_total*100:.0f}%)")
    if n_miss == 0:
        print("  WARNING: No missingness features found — A2/A3 ablation will have no effect")

    # Load phenotype labels for subgroup analysis
    pheno_path = ROOT / config["data"]["phenotype_labels"]
    pheno_df = pd.read_parquet(pheno_path)
    pheno_test = pheno_df[pheno_df["stay_id"].isin(test_set)].set_index("stay_id")

    # ── 2. Load primary model ────────────────────────────────────────
    print("\n[2] Loading 80c G-Transformer ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_temporal = len(temporal_cols)
    d_static = len(static_cols)
    n_actions = len(config["action"]["valid_merged_indices"])

    model = load_primary_model(config, d_temporal, d_static, n_actions, device)
    print(f"  Device: {device}")

    # ── 3. Build baseline dataset ────────────────────────────────────
    print("\n[3] Building baseline test dataset ...")
    ds_base = DTRDataset(
        state_df=state_df, outcomes_df=outcomes_df, propensity_df=propensity_df,
        stay_ids=test_ids, temporal_cols=temporal_cols, static_cols=static_cols,
        merge_map=merge_map, n_bins=config["grid"]["n_bins"],
    )
    ds_base = set_action_vocab(ds_base, action_vocab)

    all_results = []

    # ── A1: Baseline ─────────────────────────────────────────────────
    print("\n[A1] Baseline evaluation (full features) ...")
    a1_point = evaluate_dm(model, ds_base, device, use_rollout=True)

    def _eval_fn(ds, **kw):
        return evaluate_dm(model, ds, device, use_rollout=True)

    a1_boot = bootstrap_ope(
        _eval_fn, ds_base,
        n_resamples=N_BOOTSTRAP, seed=BOOT_SEED, confidence=CI_LEVEL,
    )

    all_results.append({
        "scenario": "A1_baseline",
        "subgroup": "ALL",
        "n_stays": len(test_ids),
        "ate": a1_point["ate"],
        "v_observed": a1_point["v_observed"],
        "v_rule": a1_point["v_rule"],
        "ate_ci_lo": a1_boot.get("ate_ci_lo", np.nan),
        "ate_ci_hi": a1_boot.get("ate_ci_hi", np.nan),
    })
    print(f"  A1 ALL: ATE={a1_point['ate']:+.4f} "
          f"[{a1_boot.get('ate_ci_lo', np.nan):+.4f}, {a1_boot.get('ate_ci_hi', np.nan):+.4f}]")

    # ── A2: Zero-out missingness features ─────────────────────────────
    if miss_indices:
        print(f"\n[A2] Zero-out ablation ({n_miss} missingness features) ...")
        # Deep copy temporal array and zero out missingness columns
        ds_a2 = DTRDataset(
            state_df=state_df, outcomes_df=outcomes_df, propensity_df=propensity_df,
            stay_ids=test_ids, temporal_cols=temporal_cols, static_cols=static_cols,
            merge_map=merge_map, n_bins=config["grid"]["n_bins"],
        )
        ds_a2 = set_action_vocab(ds_a2, action_vocab)
        # Zero out missingness features in the temporal tensor
        for idx in miss_indices:
            ds_a2.temporal[:, :, idx] = 0.0

        a2_point = evaluate_dm(model, ds_a2, device, use_rollout=True)

        def _eval_a2(ds, **kw):
            # Apply zero-ablation to each bootstrap subset view.
            for idx in miss_indices:
                ds._dataset.temporal[ds._indices, :, idx] = 0.0
            return evaluate_dm(model, ds, device, use_rollout=True)

        a2_boot = bootstrap_ope(
            _eval_a2, ds_a2,
            n_resamples=N_BOOTSTRAP, seed=BOOT_SEED, confidence=CI_LEVEL,
        )

        all_results.append({
            "scenario": "A2_zero_missingness",
            "subgroup": "ALL",
            "n_stays": len(test_ids),
            "ate": a2_point["ate"],
            "v_observed": a2_point["v_observed"],
            "v_rule": a2_point["v_rule"],
            "ate_ci_lo": a2_boot.get("ate_ci_lo", np.nan),
            "ate_ci_hi": a2_boot.get("ate_ci_hi", np.nan),
        })
        print(f"  A2 ALL: ATE={a2_point['ate']:+.4f} "
              f"[{a2_boot.get('ate_ci_lo', np.nan):+.4f}, {a2_boot.get('ate_ci_hi', np.nan):+.4f}]")
    else:
        print("\n[A2] SKIPPED — no missingness features to ablate")

    # ── A3: Shuffle missingness features across stays ─────────────────
    if miss_indices:
        print(f"\n[A3] Shuffle ablation ({N_SHUFFLE_REPS} reps) ...")
        rng = np.random.RandomState(42)
        shuffle_ates = []

        for rep in tqdm(range(N_SHUFFLE_REPS), desc="A3 shuffle reps"):
            ds_a3 = DTRDataset(
                state_df=state_df, outcomes_df=outcomes_df, propensity_df=propensity_df,
                stay_ids=test_ids, temporal_cols=temporal_cols, static_cols=static_cols,
                merge_map=merge_map, n_bins=config["grid"]["n_bins"],
            )
            ds_a3 = set_action_vocab(ds_a3, action_vocab)
            # Permute missingness features across stays (axis=0)
            n_stays = ds_a3.temporal.shape[0]
            perm_idx = rng.permutation(n_stays)
            for idx in miss_indices:
                ds_a3.temporal[:, :, idx] = ds_a3.temporal[perm_idx, :, idx]

            a3_point = evaluate_dm(model, ds_a3, device, use_rollout=True)
            shuffle_ates.append(a3_point["ate"])

        a3_mean_ate = float(np.mean(shuffle_ates))
        a3_std_ate = float(np.std(shuffle_ates))

        all_results.append({
            "scenario": "A3_shuffle_missingness",
            "subgroup": "ALL",
            "n_stays": len(test_ids),
            "ate": a3_mean_ate,
            "v_observed": np.nan,
            "v_rule": np.nan,
            "ate_ci_lo": a3_mean_ate - 1.96 * a3_std_ate,
            "ate_ci_hi": a3_mean_ate + 1.96 * a3_std_ate,
        })
        print(f"  A3 ALL: ATE={a3_mean_ate:+.4f} ± {a3_std_ate:.4f} "
              f"(mean over {N_SHUFFLE_REPS} shuffles)")
    else:
        print("\n[A3] SKIPPED — no missingness features to shuffle")

    # ── A4: Stratify by measurement intensity quartiles ───────────────
    print(f"\n[A4] Measurement intensity stratification ...")
    intensity_df = compute_measurement_intensity(state_df, temporal_cols, test_ids)
    intensity_df = intensity_df[intensity_df["stay_id"].isin(test_set)]

    # Quartile boundaries
    q_edges = intensity_df["measurement_intensity"].quantile([0.25, 0.5, 0.75]).values
    intensity_df["intensity_q4"] = np.digitize(
        intensity_df["measurement_intensity"].values, q_edges
    ) + 1
    int_indexed = intensity_df.set_index("stay_id")

    print(f"  Intensity quartile distribution: "
          f"{intensity_df['intensity_q4'].value_counts().sort_index().to_dict()}")

    for q in tqdm(range(1, 5), desc="A4 intensity quartiles"):
        q_sids = int_indexed.loc[int_indexed["intensity_q4"] == q].index.values
        q_sids = np.array(sorted(set(q_sids) & test_set))

        if len(q_sids) < 30:
            print(f"  Q{q}: n={len(q_sids)} — TOO SMALL")
            continue

        q_ds = DTRDataset(
            state_df=state_df, outcomes_df=outcomes_df, propensity_df=propensity_df,
            stay_ids=q_sids, temporal_cols=temporal_cols, static_cols=static_cols,
            merge_map=merge_map, n_bins=config["grid"]["n_bins"],
        )
        q_ds = set_action_vocab(q_ds, action_vocab)

        q_point = evaluate_dm(model, q_ds, device, use_rollout=True)

        def _eval_q(ds, **kw):
            return evaluate_dm(model, ds, device, use_rollout=True)

        q_boot = bootstrap_ope(
            _eval_q, q_ds,
            n_resamples=200, seed=BOOT_SEED + q, confidence=CI_LEVEL,
        )

        all_results.append({
            "scenario": f"A4_intensity_Q{q}",
            "subgroup": "ALL",
            "n_stays": len(q_sids),
            "ate": q_point["ate"],
            "v_observed": q_point["v_observed"],
            "v_rule": q_point["v_rule"],
            "ate_ci_lo": q_boot.get("ate_ci_lo", np.nan),
            "ate_ci_hi": q_boot.get("ate_ci_hi", np.nan),
        })
        print(f"  Q{q}: n={len(q_sids):,}, ATE={q_point['ate']:+.4f} "
              f"[{q_boot.get('ate_ci_lo', np.nan):+.4f}, {q_boot.get('ate_ci_hi', np.nan):+.4f}]")

    # ── A2/A3 subgroup: HFpEF vs no_hf under ablation ────────────────
    if miss_indices:
        print(f"\n[A2 subgroup] HFpEF vs no_hf under zero-ablation ...")
        for hf_grp in ["hfpef", "no_hf"]:
            grp_sids = pheno_test.loc[
                (pheno_test.index.isin(test_set))
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
            # Zero out missingness features
            for idx in miss_indices:
                grp_ds.temporal[:, :, idx] = 0.0

            grp_point = evaluate_dm(model, grp_ds, device, use_rollout=True)

            def _eval_grp(ds, **kw):
                for idx in miss_indices:
                    ds._dataset.temporal[ds._indices, :, idx] = 0.0
                return evaluate_dm(model, ds, device, use_rollout=True)

            grp_boot = bootstrap_ope(
                _eval_grp, grp_ds,
                n_resamples=200, seed=BOOT_SEED + (11 if hf_grp == "hfpef" else 29),
                confidence=CI_LEVEL,
            )

            all_results.append({
                "scenario": "A2_zero_missingness",
                "subgroup": hf_grp,
                "n_stays": len(grp_sids),
                "ate": grp_point["ate"],
                "v_observed": grp_point["v_observed"],
                "v_rule": grp_point["v_rule"],
                "ate_ci_lo": grp_boot.get("ate_ci_lo", np.nan),
                "ate_ci_hi": grp_boot.get("ate_ci_hi", np.nan),
            })
            print(
                f"  A2 {hf_grp}: ATE={grp_point['ate']:+.4f} "
                f"[{grp_boot.get('ate_ci_lo', np.nan):+.4f}, {grp_boot.get('ate_ci_hi', np.nan):+.4f}]"
            )

    # ── 5. Save results ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("[5] Saving results ...")

    results_df = pd.DataFrame(all_results)
    csv_path = TABLES_DIR / "85j_missingness_sensitivity.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path} ({len(results_df)} rows)")

    # ── 6. Generate report ───────────────────────────────────────────
    print("\n[6] Generating report ...")

    baseline_ate = a1_point["ate"]
    lines = [
        "=" * 72,
        "85j — Informative Missingness Sensitivity (MIMIC-IV)",
        "=" * 72,
        f"Generated: {ts}",
        f"Model: 80c G-Transformer DM (rollout, ow=1)",
        f"Missingness features: {n_miss} of {n_total} temporal features ({n_miss/n_total*100:.0f}%)",
        f"Bootstrap: B={N_BOOTSTRAP}, seed={BOOT_SEED}",
        "",
        "FEATURE ABLATION RESULTS",
        "────────────────────────",
        f"  {'Scenario':<25s} {'Subgroup':<10s} {'n':>6s} {'ATE':>9s} "
        f"{'CI_lo':>9s} {'CI_hi':>9s} {'Δ vs A1':>9s}",
        f"  {'-'*25} {'-'*10} {'-'*6} {'-'*9} {'-'*9} {'-'*9} {'-'*9}",
    ]

    for _, r in results_df.iterrows():
        delta = r["ate"] - baseline_ate if r["subgroup"] == "ALL" else np.nan
        delta_str = (
            f"{delta:+9.4f}"
            if pd.notna(delta) and r["scenario"] != "A1_baseline"
            else "       —"
        )
        ci_lo = f"{r['ate_ci_lo']:+9.4f}" if pd.notna(r["ate_ci_lo"]) else "       —"
        ci_hi = f"{r['ate_ci_hi']:+9.4f}" if pd.notna(r["ate_ci_hi"]) else "       —"
        lines.append(
            f"  {r['scenario']:<25s} {r['subgroup']:<10s} {r['n_stays']:6d} "
            f"{r['ate']:+9.4f} {ci_lo} {ci_hi} {delta_str}"
        )

    # Interpretation
    lines.extend([
        "",
        "INTERPRETATION",
        "──────────────",
        f"  Baseline ATE (A1): {baseline_ate:+.4f}",
        "",
        "  A2 (zero-out) tests: does the model NEED missingness features for ATE?",
        "    If A2 ≈ A1 → ATE is driven by state values, not missingness patterns",
        "    If A2 ≠ A1 → missingness features contribute to the treatment effect estimate",
        "",
        "  A3 (shuffle) tests: is the miss→outcome association causal or spurious?",
        "    If A3 ≈ A1 → marginal distribution alone is sufficient (not patient-specific)",
        "    If A3 ≠ A1 → patient-specific missingness patterns matter for outcome prediction",
        "",
        "  A4 (intensity quartiles) tests: is ATE heterogeneous by measurement frequency?",
        "    If ATE similar across quartiles → robust to measurement intensity",
        "    If ATE varies → treatment effect may be confounded by monitoring intensity",
    ])

    # Robustness assessment
    a2_results = results_df.loc[
        (results_df["scenario"] == "A2_zero_missingness") & (results_df["subgroup"] == "ALL")
    ]
    if not a2_results.empty:
        a2_ate = a2_results.iloc[0]["ate"]
        pct_change = abs(a2_ate - baseline_ate) / abs(baseline_ate) * 100
        if pct_change < 20:
            lines.append(f"\n  → ROBUST: A2 ATE change = {pct_change:.1f}% (< 20% threshold)")
        elif pct_change < 50:
            lines.append(f"\n  → PARTIALLY ROBUST: A2 ATE change = {pct_change:.1f}% (20-50%)")
        else:
            lines.append(f"\n  → SENSITIVE: A2 ATE change = {pct_change:.1f}% (> 50%) — "
                        "missingness features materially affect the estimate")

    report = "\n".join(lines) + "\n"
    report_path = REPORTS_DIR / "85j_missingness_sensitivity_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {report_path}")
    print(report)

    total_time = time.time() - t0
    print(f"\n{'=' * 72}")
    print(f"85j COMPLETE. Total time: {total_time / 60:.1f} min")
    print("=" * 72)


if __name__ == "__main__":
    main()
