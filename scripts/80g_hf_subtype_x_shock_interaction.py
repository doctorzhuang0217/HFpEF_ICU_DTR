#!/usr/bin/env python3
"""
80g_hf_subtype_x_shock_interaction — HF-subtype × shock 2×2 interaction analysis.

Purpose
-------
Evaluate whether rule-policy treatment effects are HFpEF-specific or
driven solely by shock management. This is a critical validity check:
if ATE is similar across HF subtypes within each shock stratum, the
effect is shock-driven and not HFpEF-specific.

Design:
  Cross-stratification: hf_subtype × rule_phenotype_2class
  Primary estimator: DM (from 80c G-Transformer, rollout ow=1)
  Confirmatory: DM from 80a GNet; DR from 80b ACTIN
  Note: 80d/80e FQE models are reported as not subgroup-evaluable here
        (requires subgroup-specific fitted-Q retraining)

Prerequisites:
  - 65e must have been re-run with hf_subtype in frozen labels
  - Trained model checkpoints from 80a-80e must exist

Protocol reference: Step 80g (v3.1 § amendment)

Inputs
------
- models/80{a,b,c,d,e}_*_best.pt
- data/processed/phenotype_labels_frozen_v1_mimic.parquet (with hf_subtype)
- configs/80_step_c_training_v1.yaml

Outputs
-------
- tables/80g_hf_subtype_x_shock_interaction.csv
- tables/80g_hf_subtype_x_shock_interaction_summary.yaml
- reports/80g_hf_subtype_x_shock_interaction_summary.txt

Compute: C2 (CPU-heavy forward passes, no GPU training needed; GPU optional)
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hfpef_dtr.config import load_config, resolve_path
from hfpef_dtr.models.dtr_dataset import DTRDataset, load_dtr_data
from hfpef_dtr.models.dataset import discover_feature_columns, make_train_val_test_split
from hfpef_dtr.models.ope import evaluate_dm, evaluate_dr

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "tables"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"


# ── Model registry: what to load, what estimator to use ──────────────
MODEL_SPECS = [
    {
        "model_id": "80c_gtransformer_rollout_ow1",
        "checkpoint": "80c_gtransformer_rollout_ow1_best.pt",
        "module": "hfpef_dtr.models.gtransformer",
        "builder": "build_gtransformer_from_config",
        "config_key": "80c_gtransformer",
        "estimator": "DM",
        "use_rollout": True,
        "role": "primary",
    },
    {
        "model_id": "80a_gnet",
        "checkpoint": "80a_gnet_best.pt",
        "module": "hfpef_dtr.models.gnet",
        "builder": "build_gnet_from_config",
        "config_key": "80a_gnet",
        "estimator": "DM",
        "use_rollout": False,
        "role": "confirmatory",
    },
    {
        "model_id": "80b_actin",
        "checkpoint": "80b_actin_best.pt",
        "module": "hfpef_dtr.models.actin",
        "builder": "build_actin_from_config",
        "config_key": "80b_actin",
        "estimator": "DR",
        "use_rollout": False,
        "role": "confirmatory",
    },
    {
        "model_id": "80d_ct_legacy",
        "checkpoint": "80d_ct_legacy_best.pt",
        "module": "hfpef_dtr.models.ct_legacy",
        "builder": "build_ct_legacy_from_config",
        "config_key": "80d_ct_legacy",
        "estimator": "FQE",
        "use_rollout": False,
        "role": "ablation",
    },
    {
        "model_id": "80e_mamba_cdsp",
        "checkpoint": "80e_mamba_cdsp_best.pt",
        "module": "hfpef_dtr.models.mamba_cdsp",
        "builder": "build_mamba_cdsp_from_config",
        "config_key": "80e_mamba_cdsp",
        "estimator": "FQE",
        "use_rollout": False,
        "role": "optional",
    },
]


def load_model(spec: dict, config: dict, d_temporal: int, d_static: int,
               n_actions: int, device: torch.device):
    """Dynamically load a trained model from checkpoint."""
    import importlib
    mod = importlib.import_module(spec["module"])
    builder_fn = getattr(mod, spec["builder"])

    model_cfg = config["models"][spec["config_key"]]
    grid_cfg = config["grid"]
    model = builder_fn(
        d_temporal=d_temporal, d_static=d_static,
        n_actions=n_actions, model_cfg=model_cfg, grid_cfg=grid_cfg,
    )

    ckpt_path = MODELS_DIR / spec["checkpoint"]
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate_subgroup_dm(model, state_df, outcomes_df, propensity_df,
                         temporal_cols, static_cols, merge_map,
                         stay_ids, config, device, use_rollout=False):
    """Build DTRDataset for a subgroup and evaluate DM."""
    if len(stay_ids) < 20:
        return None

    sub_ds = DTRDataset(
        state_df=state_df,
        outcomes_df=outcomes_df,
        propensity_df=propensity_df,
        stay_ids=stay_ids,
        temporal_cols=temporal_cols,
        static_cols=static_cols,
        merge_map=merge_map,
        n_bins=config["grid"]["n_bins"],
        action_vocab=config["action"]["valid_merged_indices"],
    )
    try:
        result = evaluate_dm(model, sub_ds, device, use_rollout=use_rollout)
        return result
    except Exception as e:
        print(f"    WARNING: DM evaluation failed: {e}")
        return None


def evaluate_subgroup_dr(model, state_df, outcomes_df, propensity_df,
                         temporal_cols, static_cols, merge_map,
                         stay_ids, config, device):
    """Build DTRDataset for a subgroup and evaluate DR."""
    if len(stay_ids) < 20:
        return None

    sub_ds = DTRDataset(
        state_df=state_df,
        outcomes_df=outcomes_df,
        propensity_df=propensity_df,
        stay_ids=stay_ids,
        temporal_cols=temporal_cols,
        static_cols=static_cols,
        merge_map=merge_map,
        n_bins=config["grid"]["n_bins"],
        action_vocab=config["action"]["valid_merged_indices"],
    )
    try:
        result = evaluate_dr(model, sub_ds, device)
        return result
    except Exception as e:
        print(f"    WARNING: DR evaluation failed: {e}")
        return None


def main() -> None:
    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("80g — HF-Subtype × Shock 2×2 Interaction Analysis")
    print("=" * 72)

    # ── 1. Load config and data ──────────────────────────────────────
    print("\n[1] Loading data ...")
    config_path = ROOT / "configs" / "80_step_c_training_v1.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    state_df, outcomes_df, propensity_df, temporal_cols, static_cols, merge_map = (
        load_dtr_data(ROOT, config)
    )

    # Test set IDs
    all_stay_ids = state_df["stay_id"].unique()
    split_cfg = config["split"]
    _, _, test_ids = make_train_val_test_split(
        all_stay_ids, outcomes_df,
        train_frac=split_cfg["train_frac"],
        val_frac=split_cfg["val_frac"],
        test_frac=split_cfg["test_frac"],
        stratify_col=split_cfg["stratify_col"],
        seed=split_cfg["seed"],
    )
    print(f"  Test stays: {len(test_ids):,}")

    # ── 2. Load phenotype labels (must include hf_subtype) ───────────
    print("\n[2] Loading phenotype labels ...")
    pheno_path = ROOT / config["data"]["phenotype_labels"]
    pheno_df = pd.read_parquet(pheno_path)

    required_cols = ["hf_subtype", "rule_phenotype_2class"]
    missing = [c for c in required_cols if c not in pheno_df.columns]
    if missing:
        print(f"  FATAL: Missing columns in frozen labels: {missing}")
        print(f"  Available columns: {list(pheno_df.columns)}")
        print("  → Re-run 65e_freeze_phenotype_labels.py first")
        sys.exit(1)

    pheno_test = pheno_df[pheno_df["stay_id"].isin(set(test_ids))].copy()
    print(f"  Phenotype labels for test: {len(pheno_test):,}")

    # Build 2×2 strata
    hf_groups = sorted(pheno_test["hf_subtype"].dropna().unique())
    shock_groups = sorted(pheno_test["rule_phenotype_2class"].dropna().unique())
    print(f"  HF subtypes: {hf_groups}")
    print(f"  Shock strata: {shock_groups}")

    # Show cross-tab
    ct = pd.crosstab(pheno_test["hf_subtype"], pheno_test["rule_phenotype_2class"],
                     margins=True)
    print(f"\n  Cross-tabulation (test set):\n{ct.to_string()}")

    # ── 3. Device setup ──────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # Feature dimensions
    d_temporal = len(temporal_cols)
    d_static = len(static_cols)
    n_actions = len(config["action"]["valid_merged_indices"])
    print(f"  d_temporal={d_temporal}, d_static={d_static}, n_actions={n_actions}")

    # ── 4. Run 2×2 analysis for each model ───────────────────────────
    all_results = []

    for spec in tqdm(MODEL_SPECS, desc="Models", unit="model"):
        model_id = spec["model_id"]
        estimator = spec["estimator"]
        print(f"\n{'─' * 60}")
        print(f"[4] Model: {model_id} (estimator={estimator}, role={spec['role']})")

        # FQE subgroup analysis would require subgroup-specific fitted-Q re-estimation.
        # To avoid estimator mismatch, we skip FQE models in this script.
        if estimator == "FQE":
            print("  SKIP: FQE models require subgroup-specific fitted-Q estimation; "
                  "no DM fallback used to avoid methodological mismatch")
            continue

        if estimator in ("DM", "DR"):
            model = load_model(spec, config, d_temporal, d_static, n_actions, device)
            if model is None:
                print(f"  Checkpoint not found: {spec['checkpoint']} — SKIPPING")
                continue
        else:
            print(f"  Unknown estimator: {estimator} — SKIPPING")
            continue

        # Evaluate on each 2×2 cell
        cell_pairs = list(product(hf_groups, shock_groups))
        for hf_grp, shock_grp in tqdm(cell_pairs, desc=f"{model_id} cells", unit="cell", leave=False):
            mask = (
                (pheno_test["hf_subtype"] == hf_grp)
                & (pheno_test["rule_phenotype_2class"] == shock_grp)
            )
            cell_ids = pheno_test.loc[mask, "stay_id"].values
            n_cell = len(cell_ids)

            if n_cell < 20:
                print(f"  {hf_grp} × {shock_grp}: n={n_cell} — TOO SMALL, skipping")
                continue

            if estimator == "DR":
                result = evaluate_subgroup_dr(
                    model, state_df, outcomes_df, propensity_df,
                    temporal_cols, static_cols, merge_map,
                    cell_ids, config, device,
                )
            else:
                use_rollout = spec.get("use_rollout", False)
                result = evaluate_subgroup_dm(
                    model, state_df, outcomes_df, propensity_df,
                    temporal_cols, static_cols, merge_map,
                    cell_ids, config, device, use_rollout=use_rollout,
                )

            if result is not None:
                all_results.append({
                    "model": model_id,
                    "role": spec["role"],
                    "estimator": estimator,
                    "hf_subtype": hf_grp,
                    "shock_stratum": shock_grp,
                    "n_stays": n_cell,
                    "v_observed": result["v_observed"],
                    "v_rule": result["v_rule"],
                    "ate": result["ate"],
                })
                print(f"  {hf_grp:12s} × {shock_grp:20s}: n={n_cell:4d}, "
                      f"ATE={result['ate']:+.4f}")

        # Also evaluate marginal hf_subtype (collapsed across shock)
        print(f"\n  Marginal hf_subtype analysis:")
        for hf_grp in hf_groups:
            cell_ids = pheno_test.loc[
                pheno_test["hf_subtype"] == hf_grp, "stay_id"
            ].values
            n_cell = len(cell_ids)
            if n_cell < 20:
                continue

            if estimator == "DR":
                result = evaluate_subgroup_dr(
                    model, state_df, outcomes_df, propensity_df,
                    temporal_cols, static_cols, merge_map,
                    cell_ids, config, device,
                )
            else:
                result = evaluate_subgroup_dm(
                    model, state_df, outcomes_df, propensity_df,
                    temporal_cols, static_cols, merge_map,
                    cell_ids, config, device,
                    use_rollout=spec.get("use_rollout", False),
                )

            if result is not None:
                all_results.append({
                    "model": model_id,
                    "role": spec["role"],
                    "estimator": estimator,
                    "hf_subtype": hf_grp,
                    "shock_stratum": "ALL",
                    "n_stays": n_cell,
                    "v_observed": result["v_observed"],
                    "v_rule": result["v_rule"],
                    "ate": result["ate"],
                })
                print(f"  {hf_grp:12s} (marginal): n={n_cell:4d}, "
                      f"ATE={result['ate']:+.4f}")

    # ── 5. Save results ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("[5] Saving results ...")

    results_df = pd.DataFrame(all_results)
    csv_path = TABLES_DIR / "80g_hf_subtype_x_shock_interaction.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path} ({len(results_df)} rows)")

    # Summary YAML
    summary = {
        "step": "80g_hf_subtype_x_shock_interaction",
        "date": ts,
        "protocol_version": "v3.1",
        "database": "MIMIC-IV",
        "purpose": "Test whether rule-policy ATE is HFpEF-specific or shock-driven",
        "design": "hf_subtype x rule_phenotype_2class 2x2 cross-stratification",
        "n_test_stays": int(len(test_ids)),
        "hf_subtypes": hf_groups,
        "shock_strata": shock_groups,
        "n_models_evaluated": len(set(r["model"] for r in all_results)),
        "n_results": len(all_results),
    }

    # Extract primary model (80c) results for summary
    primary_rows = [r for r in all_results
                    if r["model"] == "80c_gtransformer_rollout_ow1"]
    if primary_rows:
        summary["primary_model"] = "80c_gtransformer_rollout_ow1"
        summary["primary_results"] = {}
        for r in primary_rows:
            key = f"{r['hf_subtype']}_x_{r['shock_stratum']}"
            summary["primary_results"][key] = {
                "n_stays": r["n_stays"],
                "ate": round(r["ate"], 6),
                "v_observed": round(r["v_observed"], 6),
                "v_rule": round(r["v_rule"], 6),
            }

    yaml_path = TABLES_DIR / "80g_hf_subtype_x_shock_interaction_summary.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
    print(f"  Saved: {yaml_path}")

    # ── 6. Generate report ───────────────────────────────────────────
    print("\n[6] Generating report ...")
    lines = [
        "=" * 72,
        "80g — HF-Subtype × Shock 2×2 Interaction Analysis (MIMIC-IV)",
        "=" * 72,
        f"Generated: {ts}",
        "",
        "PURPOSE",
        "───────",
        "  Test whether rule-policy treatment effect is HFpEF-specific",
        "  or explained entirely by shock management (confound check).",
        "",
        "  Key question: Is ATE(HFpEF, shock) > ATE(non-HF, shock)?",
        "  If yes → HFpEF background modifies treatment effect (specificity).",
        "  If no  → effect is shock-driven, not HFpEF-specific.",
        "",
        "CROSS-TABULATION (test set)",
        "───────────────────────────",
        ct.to_string(),
        "",
        "RESULTS",
        "───────",
    ]

    # Group by model
    for model_id in dict.fromkeys(r["model"] for r in all_results):
        model_rows = [r for r in all_results if r["model"] == model_id]
        role = model_rows[0]["role"]
        est = model_rows[0]["estimator"]
        lines.append(f"\n  {model_id} ({role}, {est}):")
        lines.append(f"  {'HF subtype':>15s}  {'Shock stratum':>20s}  {'n':>5s}  {'ATE':>8s}  {'V(obs)':>8s}  {'V(rule)':>8s}")
        lines.append(f"  {'-'*15}  {'-'*20}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}")
        for r in sorted(model_rows, key=lambda x: (x["shock_stratum"], x["hf_subtype"])):
            lines.append(
                f"  {r['hf_subtype']:>15s}  {r['shock_stratum']:>20s}  "
                f"{r['n_stays']:5d}  {r['ate']:+8.4f}  "
                f"{r['v_observed']:8.4f}  {r['v_rule']:8.4f}"
            )

    # Interpretation guide
    lines.extend([
        "",
        "INTERPRETATION GUIDE",
        "────────────────────",
        "  1. Compare ATE(hfpef, shock) vs ATE(non_hf, shock):",
        "     If similar → shock management drives effect, not HFpEF-specific",
        "     If HFpEF larger → HFpEF background modifies treatment response",
        "",
        "  2. Look at ATE(hfpef, non_shock):",
        "     If significantly negative → decongestion rules (P4-P5) provide",
        "     HFpEF-specific benefit independent of shock management",
        "",
        "  3. Compare across models for robustness of interaction pattern",
        "",
        "NEXT STEPS",
        "──────────",
        "  If HFpEF-specific signal confirmed: proceed to Step 85 sensitivity",
        "  If signal is shock-only: reframe as 'hemodynamic optimization benefits",
        "  all ICU patients; HFpEF as distinct substrate requires further evidence'",
    ])

    report_path = REPORTS_DIR / "80g_hf_subtype_x_shock_interaction_summary.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {report_path}")

    total_time = time.time() - t0
    print(f"\n{'=' * 72}")
    print(f"80g COMPLETE. Total time: {total_time / 60:.1f} min")
    print("=" * 72)


if __name__ == "__main__":
    main()
