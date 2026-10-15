#!/usr/bin/env python3
"""
85d_positivity_grid_stability — Positivity grid stability analysis.

Purpose
-------
Evaluate how DM ATE estimates change across the prespecified weight-trimming
grid from 70f. If ATE is stable across trimming brackets, the estimate is
robust to positivity violations. If ATE varies sharply, positivity is fragile
and the estimate is trimming-sensitive.

This analysis sweeps through 5 prespecified trimming brackets
([1,99], [2.5,97.5], [5,95], [10,90], [20,80]) and evaluates the 80c
G-Transformer DM ATE under each bracket. The trimming is applied to
propensity-based sample weights (reweighting, not exclusion), but since DM
does not use weights directly, we implement trimming as sample exclusion:
stays with propensity-weight outside the bracket bounds are excluded.

Additionally, we evaluate stability across HFpEF vs no_hf subgroups and
generate stability plots.

Protocol reference: Step 85d (sensitivity layer)
Compute: C3 (bootstrap across multiple brackets)

Inputs
------
- configs/80_step_c_training_v1.yaml
- configs/70f_trimming_grid_v1.yaml
- models/80c_gtransformer_rollout_ow1_best.pt
- data/interim/45c_state_tensors_v1_mimic.parquet
- data/interim/45d_outcomes_absorb_v1_mimic.parquet
- data/interim/70b_propensity_coarse3_v1_mimic.parquet
- data/processed/phenotype_labels_frozen_v1_mimic.parquet

Outputs
-------
- tables/85d_positivity_grid.csv
- figures/85d_positivity_grid_stability.png
- reports/85d_positivity_grid_summary.txt
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
FIGURES_DIR = ROOT / "figures"

N_BOOTSTRAP = 300
BOOT_SEED = 7777
CI_LEVEL = 0.95

# Prespecified trimming brackets from 70f
TRIMMING_BRACKETS = [
    (1.0, 99.0),
    (2.5, 97.5),
    (5.0, 95.0),   # primary
    (10.0, 90.0),
    (20.0, 80.0),
]


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


def compute_stay_level_weights(
    propensity_df: pd.DataFrame,
    stay_ids: np.ndarray,
    n_bins: int = 12,
    propensity_floor: float = 0.005,
) -> pd.DataFrame:
    """
    Compute per-stay cumulative IPTW weight from per-bin propensities.

    Returns DataFrame with columns: stay_id, cum_weight.
    """
    stay_set = set(stay_ids)
    prop_sub = propensity_df[propensity_df["stay_id"].isin(stay_set)].copy()

    prob_col = (
        "propensity_clipped" if "propensity_clipped" in prop_sub.columns
        else "propensity" if "propensity" in prop_sub.columns
        else "prob"
    )

    if prob_col not in prop_sub.columns:
        # Fallback: uniform propensity (no weight variation)
        print("  WARNING: No propensity column found; using uniform weights")
        return pd.DataFrame({
            "stay_id": stay_ids,
            "cum_weight": np.ones(len(stay_ids)),
        })

    # Clip propensity floor
    prop_sub[prob_col] = prop_sub[prob_col].clip(lower=propensity_floor)

    # Cumulative IPTW weight = product of 1/P(A_t|H_t) across bins
    # For numerical stability, work in log space
    prop_sub["log_inv_prop"] = -np.log(prop_sub[prob_col])

    cum_log_weights = prop_sub.groupby("stay_id")["log_inv_prop"].sum()
    cum_weights = np.exp(np.clip(cum_log_weights, -50, 50))

    result = cum_weights.reset_index()
    result.columns = ["stay_id", "cum_weight"]

    # Ensure all test stays are present (default weight=1 if missing)
    result = pd.DataFrame({"stay_id": stay_ids}).merge(result, on="stay_id", how="left")
    result["cum_weight"] = result["cum_weight"].fillna(1.0)
    return result


def main() -> None:
    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("85d — Positivity Grid Stability Analysis")
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

    # ── 2. Compute stay-level weights ────────────────────────────────
    print("\n[2] Computing stay-level IPTW weights ...")
    weight_df = compute_stay_level_weights(
        propensity_df, test_ids,
        n_bins=config["grid"]["n_bins"],
    )
    weight_indexed = weight_df.set_index("stay_id")

    w = weight_df["cum_weight"].values
    print(f"  Weight stats: median={np.median(w):.2f}, mean={np.mean(w):.2f}, "
          f"max={np.max(w):.2f}, min={np.min(w):.4f}")

    # ── 3. Load model ────────────────────────────────────────────────
    print("\n[3] Loading 80c G-Transformer ...")
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

    # ── 4. Evaluate across trimming grid ─────────────────────────────
    print("\n[4] Evaluating ATE across trimming brackets ...")
    all_results = []

    # Untrimmed baseline
    print("\n  ── No trimming (full sample) ──")
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

    all_results.append({
        "bracket": "none",
        "pct_lo": 0.0,
        "pct_hi": 100.0,
        "subgroup": "ALL",
        "n_stays": len(test_ids),
        "n_excluded": 0,
        "pct_excluded": 0.0,
        "ate": full_point["ate"],
        "ate_ci_lo": full_boot.get("ate_ci_lo", np.nan),
        "ate_ci_hi": full_boot.get("ate_ci_hi", np.nan),
        "v_observed": full_point["v_observed"],
        "v_rule": full_point["v_rule"],
    })
    print(f"    ALL: ATE={full_point['ate']:+.4f} n={len(test_ids):,}")

    for hf_grp in ["hfpef", "no_hf"]:
        grp_sids = pheno_test.loc[
            pheno_test["hf_subtype"] == hf_grp
        ].index.values

        if len(grp_sids) < 20:
            print(f"    {hf_grp}: n={len(grp_sids)} — TOO SMALL")
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
            n_resamples=200, seed=BOOT_SEED + 25,
            confidence=CI_LEVEL,
        )

        all_results.append({
            "bracket": "none",
            "pct_lo": 0.0,
            "pct_hi": 100.0,
            "subgroup": hf_grp,
            "n_stays": len(grp_sids),
            "n_excluded": np.nan,
            "pct_excluded": np.nan,
            "ate": grp_point["ate"],
            "ate_ci_lo": grp_boot.get("ate_ci_lo", np.nan),
            "ate_ci_hi": grp_boot.get("ate_ci_hi", np.nan),
            "v_observed": grp_point["v_observed"],
            "v_rule": grp_point["v_rule"],
        })
        print(f"    {hf_grp}: ATE={grp_point['ate']:+.4f} "
              f"[{grp_boot.get('ate_ci_lo', np.nan):+.4f}, "
              f"{grp_boot.get('ate_ci_hi', np.nan):+.4f}]")

    # Each trimming bracket
    for lo_pct, hi_pct in tqdm(TRIMMING_BRACKETS, desc="Trimming brackets"):
        bracket_label = f"[{lo_pct},{hi_pct}]"
        print(f"\n  ── Bracket {bracket_label} ──")

        # Compute weight percentile bounds on test set
        w_all = weight_indexed.reindex(test_ids)["cum_weight"].values
        w_lo = np.percentile(w_all, lo_pct)
        w_hi = np.percentile(w_all, hi_pct)

        # Keep stays within weight bounds
        keep_mask = (w_all >= w_lo) & (w_all <= w_hi)
        keep_ids = test_ids[keep_mask]
        n_excluded = len(test_ids) - len(keep_ids)
        pct_excluded = n_excluded / len(test_ids) * 100

        print(f"    Weight bounds: [{w_lo:.4f}, {w_hi:.4f}]")
        print(f"    Excluded: {n_excluded:,} ({pct_excluded:.1f}%)")

        if len(keep_ids) < 50:
            print(f"    TOO FEW stays ({len(keep_ids)}) — skipping")
            continue

        # Full sample under this bracket
        ds_bracket = DTRDataset(
            state_df=state_df, outcomes_df=outcomes_df, propensity_df=propensity_df,
            stay_ids=keep_ids, temporal_cols=temporal_cols, static_cols=static_cols,
            merge_map=merge_map, n_bins=config["grid"]["n_bins"],
        )
        ds_bracket = set_action_vocab(ds_bracket, action_vocab)

        b_point = evaluate_dm(model, ds_bracket, device, use_rollout=True)
        b_boot = bootstrap_ope(
            _eval_fn, ds_bracket,
            n_resamples=N_BOOTSTRAP, seed=BOOT_SEED + int(lo_pct * 10),
            confidence=CI_LEVEL,
        )

        all_results.append({
            "bracket": bracket_label,
            "pct_lo": lo_pct,
            "pct_hi": hi_pct,
            "subgroup": "ALL",
            "n_stays": len(keep_ids),
            "n_excluded": n_excluded,
            "pct_excluded": pct_excluded,
            "ate": b_point["ate"],
            "ate_ci_lo": b_boot.get("ate_ci_lo", np.nan),
            "ate_ci_hi": b_boot.get("ate_ci_hi", np.nan),
            "v_observed": b_point["v_observed"],
            "v_rule": b_point["v_rule"],
        })
        print(f"    ALL: ATE={b_point['ate']:+.4f} "
              f"[{b_boot.get('ate_ci_lo', np.nan):+.4f}, {b_boot.get('ate_ci_hi', np.nan):+.4f}]")

        # Subgroups: HFpEF vs no_hf
        for hf_grp in ["hfpef", "no_hf"]:
            grp_sids = pheno_test.loc[
                (pheno_test.index.isin(set(keep_ids)))
                & (pheno_test["hf_subtype"] == hf_grp)
            ].index.values

            if len(grp_sids) < 20:
                print(f"    {hf_grp}: n={len(grp_sids)} — TOO SMALL")
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
                n_resamples=200, seed=BOOT_SEED + int(lo_pct * 10) + 50,
                confidence=CI_LEVEL,
            )

            all_results.append({
                "bracket": bracket_label,
                "pct_lo": lo_pct,
                "pct_hi": hi_pct,
                "subgroup": hf_grp,
                "n_stays": len(grp_sids),
                "n_excluded": np.nan,
                "pct_excluded": np.nan,
                "ate": grp_point["ate"],
                "ate_ci_lo": grp_boot.get("ate_ci_lo", np.nan),
                "ate_ci_hi": grp_boot.get("ate_ci_hi", np.nan),
                "v_observed": grp_point["v_observed"],
                "v_rule": grp_point["v_rule"],
            })
            print(f"    {hf_grp}: ATE={grp_point['ate']:+.4f} "
                  f"[{grp_boot.get('ate_ci_lo', np.nan):+.4f}, "
                  f"{grp_boot.get('ate_ci_hi', np.nan):+.4f}]")

    # ── 5. Save results ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("[5] Saving results ...")

    results_df = pd.DataFrame(all_results)
    csv_path = TABLES_DIR / "85d_positivity_grid.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path} ({len(results_df)} rows)")

    # ── 6. Generate stability plot ───────────────────────────────────
    print("\n[6] Generating stability plot ...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        bracket_order = ["none"] + [f"[{lo},{hi}]" for lo, hi in TRIMMING_BRACKETS]
        x_map = {b: i for i, b in enumerate(bracket_order)}

        _legend_display = {"ALL": "ALL", "hfpef": "HFpEF", "no_hf": "No HF"}
        for subgroup, marker, color in [
            ("ALL", "o", "#2196F3"),
            ("hfpef", "s", "#E91E63"),
            ("no_hf", "^", "#4CAF50"),
        ]:
            sub = results_df[results_df["subgroup"] == subgroup].copy()
            if sub.empty:
                continue

            sub = sub[sub["bracket"].isin(x_map)].copy()
            sub["x_pos"] = sub["bracket"].map(x_map)
            sub = sub.sort_values("x_pos")

            x_pos = sub["x_pos"].values
            ates = sub["ate"].values
            ci_lo = sub["ate_ci_lo"].values
            ci_hi = sub["ate_ci_hi"].values

            yerr_lo = ates - ci_lo
            yerr_hi = ci_hi - ates
            yerr = np.array([yerr_lo, yerr_hi])
            # Replace NaN with 0 for error bars
            yerr = np.nan_to_num(yerr, nan=0.0)

            ax.errorbar(
                x_pos, ates, yerr=yerr,
                marker=marker, color=color, label=_legend_display.get(subgroup, subgroup),
                capsize=4, linewidth=1.5, markersize=8,
            )

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Trimming Bracket")
        ax.set_ylabel("ATE (mortality risk difference)")
        ax.set_title("Positivity Grid Stability: ATE by Trimming Bracket")

        ax.set_xticks(np.arange(len(bracket_order)))
        ax.set_xticklabels(bracket_order, rotation=45, ha="right")

        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        fig.tight_layout()

        fig_path = FIGURES_DIR / "85d_positivity_grid_stability.png"
        fig.savefig(fig_path, dpi=150)
        fig.savefig(str(fig_path).replace('.png', '.pdf'), format='pdf',
                    bbox_inches='tight', dpi=300, facecolor='white')
        plt.close(fig)
        print(f"  Saved: {fig_path} (+pdf)")
    except ImportError:
        print("  WARNING: matplotlib not available; skipping plot")

    # ── 7. Generate report ───────────────────────────────────────────
    print("\n[7] Generating report ...")

    baseline_ate = full_point["ate"]
    lines = [
        "=" * 72,
        "85d — Positivity Grid Stability (MIMIC-IV)",
        "=" * 72,
        f"Generated: {ts}",
        f"Model: 80c G-Transformer DM (rollout, ow=1)",
        f"Bootstrap: B={N_BOOTSTRAP}, seed={BOOT_SEED}",
        f"Trimming brackets: {len(TRIMMING_BRACKETS)} prespecified from 70f",
        "",
        "RESULTS",
        "───────",
        f"  {'Bracket':<14s} {'Subgroup':<10s} {'n':>6s} {'Excl%':>6s} "
        f"{'ATE':>9s} {'CI_lo':>9s} {'CI_hi':>9s}",
        f"  {'-'*14} {'-'*10} {'-'*6} {'-'*6} {'-'*9} {'-'*9} {'-'*9}",
    ]

    for _, r in results_df.iterrows():
        excl = f"{r['pct_excluded']:5.1f}%" if pd.notna(r["pct_excluded"]) else "    —"
        ci_lo = f"{r['ate_ci_lo']:+9.4f}" if pd.notna(r["ate_ci_lo"]) else "       —"
        ci_hi = f"{r['ate_ci_hi']:+9.4f}" if pd.notna(r["ate_ci_hi"]) else "       —"
        lines.append(
            f"  {r['bracket']:<14s} {r['subgroup']:<10s} {r['n_stays']:6d} "
            f"{excl} {r['ate']:+9.4f} {ci_lo} {ci_hi}"
        )

    # Stability assessment
    all_ates = results_df.loc[results_df["subgroup"] == "ALL", "ate"].values
    ate_range = np.ptp(all_ates)
    ate_cv = np.std(all_ates) / abs(np.mean(all_ates)) * 100 if np.mean(all_ates) != 0 else np.nan

    lines.extend([
        "",
        "STABILITY ASSESSMENT",
        "────────────────────",
        f"  ATE range across brackets (ALL): {ate_range:.4f}",
        f"  ATE coefficient of variation: {ate_cv:.1f}%",
        f"  Baseline ATE (no trimming): {baseline_ate:+.4f}",
        "",
        "  Criteria:",
        "    - STABLE if ATE range < 50% of |baseline ATE| and CV < 30%",
        "    - PARTIALLY STABLE if same sign across all brackets",
        "    - UNSTABLE if sign changes or range > 100% of |baseline ATE|",
    ])

    same_sign = all(all_ates < 0) if baseline_ate < 0 else all(all_ates > 0)
    if same_sign and ate_range < abs(baseline_ate) * 0.5 and ate_cv < 30:
        lines.append(f"  → STABLE: consistent sign, low variation")
    elif same_sign:
        lines.append(f"  → PARTIALLY STABLE: consistent sign, moderate variation")
    else:
        lines.append(f"  → UNSTABLE: ATE sign or magnitude changes across trimming brackets")

    report = "\n".join(lines) + "\n"
    report_path = REPORTS_DIR / "85d_positivity_grid_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {report_path}")
    print(report)

    total_time = time.time() - t0
    print(f"\n{'=' * 72}")
    print(f"85d COMPLETE. Total time: {total_time / 60:.1f} min")
    print("=" * 72)


if __name__ == "__main__":
    main()
