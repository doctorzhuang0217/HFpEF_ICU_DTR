#!/usr/bin/env python3
"""
80f_medium4_granularity_sensitivity — 32-cell (medium_4) action-granularity check.

Purpose
-------
Enhancement Analysis F (supplementary-analysis specification, section F).

Addresses the compound-intervention / action-granularity question by
re-running the FULL DTR pipeline on a FINER action vocabulary (medium_4 = 4 fluid
× 4 vaso × 2 diuretic = 32 joint cells) and comparing the resulting DM-rollout ATE
(overall + by hf_subtype) against the frozen coarse_3 (18-cell) primary.

This script is the medium_4 ORCHESTRATOR + ATE reader. It does NOT modify the
frozen coarse_3 primary in any way: every medium_4 artifact lives on a distinct
``*_medium4`` path, produced by the ADDITIVE ``--scheme medium_4`` flag added to
45b / 45c / 70b / 80c.

medium_4 pipeline chain (run on the compute cluster, FROM main, GPU)
---------------------------------------------------------
    env/run_py.sh scripts/45b_encode_actions_2h.py        --scheme medium_4
    env/run_py.sh scripts/45c_build_state_tensors.py       --scheme medium_4
    env/run_py.sh scripts/70b_fit_behavior_policy.py       --scheme medium_4
    env/run_py.sh scripts/80c_train_gtransformer.py        --scheme medium_4 \
        --use-rollout --tag rollout_ow1
    env/run_py.sh scripts/80f_medium4_granularity_sensitivity.py

Each ``--scheme medium_4`` step writes ONLY ``*_medium4`` outputs:
  45b → data/interim/45b_actions_2h_v1_medium4_mimic.parquet
  45c → data/interim/45c_state_tensors_v1_medium4_mimic.parquet
  70b → configs/70b_action_merge_map_medium4.yaml,
        data/interim/70b_propensity_coarse3_medium4_v1_mimic.parquet
  80c → models/80c_gtransformer_medium4_<tag>_best.pt (+ *_medium4_* tables/reports)

Inputs
------
- configs/80_step_c_training_medium4_v1.yaml  (medium_4 Step C config)
- models/80c_gtransformer_medium4_<tag>_best.pt  (medium_4 G-Transformer)
- data/processed/phenotype_labels_frozen_v1_mimic.parquet  (hf_subtype)
- (transitively, via load_dtr_data) the 45c/45d/70b *_medium4 staged inputs.

Outputs
-------
- tables/80f_medium4_granularity_ate.csv
    cols: scheme, subgroup, estimator, n_stays, v_observed, v_rule,
          ate, ate_ci_lo, ate_ci_hi
- reports/80f_medium4_granularity_summary.txt

Compute: G1 (GPU). HEAVIEST analysis — requires the full medium_4 retrain to have
completed first. Run on the compute cluster (raw data is NOT local).
"""

from __future__ import annotations

import argparse
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
CONFIGS_DIR = ROOT / "configs"

DEFAULT_CONFIG = CONFIGS_DIR / "80_step_c_training_medium4_v1.yaml"
DEFAULT_TAG = "rollout_ow1"          # mirror the coarse_3 primary (rollout, ow=1)
HF_SUBTYPES = ["hfpef", "hfref", "hfmref", "hf_ef_unknown", "no_hf"]
MIN_CELL_N = 30
DEFAULT_B = 500
DEFAULT_BOOT_SEED = 7777


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="80f — medium_4 (32-cell) granularity sensitivity ATE reader.",
    )
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                   help="medium_4 Step C training config YAML.")
    p.add_argument("--tag", type=str, default=DEFAULT_TAG,
                   help="Tag suffix of the trained medium_4 80c checkpoint "
                        "(model_id = 80c_gtransformer_medium4_<tag>). Default "
                        "'rollout_ow1' mirrors the coarse_3 primary.")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Explicit checkpoint path; overrides --tag-derived path.")
    p.add_argument("--n-bootstrap", type=int, default=DEFAULT_B,
                   help="Bootstrap resamples for ATE CIs (default 500).")
    p.add_argument("--boot-seed", type=int, default=DEFAULT_BOOT_SEED,
                   help="Bootstrap RNG seed (default 7777).")
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"],
                   help="Force device; default auto-detect.")
    p.add_argument("--ack-rule-policy-gap", action="store_true", default=False,
                   help="REQUIRED to run. The medium_4 rule-policy action is now "
                        "RE-ENCODED into the 32-cell vocabulary by 70b --scheme "
                        "medium_4 (wave-2 §F: 70c decides categorically, then "
                        "ordinal-aligns the 3-level intent onto medium_4's 4-level "
                        "bins and applies the medium_4 merge map), so the DM-rollout "
                        "counterfactual is VALID. The flag is retained as an explicit "
                        "opt-in to this non-default finer-granularity sensitivity; "
                        "ensure the medium_4 chain (45b→45c→70b→80c --scheme "
                        "medium_4) has been run first.")
    return p.parse_args()


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint)
    tag = args.tag
    model_id = f"80c_gtransformer_medium4_{tag}" if tag else "80c_gtransformer_medium4"
    return MODELS_DIR / f"{model_id}_best.pt"


def load_gtransformer(config: dict, checkpoint: Path, d_temporal: int,
                      d_static: int, n_actions: int, device: torch.device):
    """Load the trained medium_4 80c G-Transformer (mirrors 80g2 loader)."""
    import importlib
    mod = importlib.import_module("hfpef_dtr.models.gtransformer")
    builder_fn = getattr(mod, "build_gtransformer_from_config")
    model_cfg = config["models"]["80c_gtransformer"]
    grid_cfg = config["grid"]
    model = builder_fn(
        d_temporal=d_temporal, d_static=d_static,
        n_actions=n_actions, model_cfg=model_cfg, grid_cfg=grid_cfg,
    )
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def build_subgroup_ds(state_df, outcomes_df, propensity_df,
                      temporal_cols, static_cols, merge_map,
                      stay_ids, config) -> DTRDataset:
    return DTRDataset(
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


def main() -> None:
    args = parse_args()
    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("80f — medium_4 (32-cell) granularity sensitivity (DM-rollout ATE)")
    print("=" * 72)

    # ── GUARD: explicit opt-in to the non-default medium_4 chain ──────
    # The DM-rollout counterfactual substitutes the RULE-POLICY action. Under
    # medium_4 the rule's 3-level fluid/vaso intent is RE-ENCODED onto the
    # 32-cell vocab by 70b --scheme medium_4 (wave-2 §F: ordinal alignment +
    # medium_4 merge map), so the medium_4 rule_action carried in the medium_4
    # 70b propensity parquet is a VALID medium_4 encoding. The flag is retained
    # as an explicit acknowledgement that this is the finer-granularity
    # sensitivity (not the frozen coarse_3 primary).
    if not getattr(args, "ack_rule_policy_gap", False):
        print(
            "\nABORT: --scheme medium_4 is the non-default finer-granularity "
            "sensitivity chain.\n"
            "  The medium_4 rule-policy action re-encoding IS now implemented "
            "(wave-2 §F, in 70b --scheme medium_4), so the DM-rollout "
            "counterfactual is VALID once the medium_4 chain has been run.\n"
            "  Run 45b→45c→70b→80c --scheme medium_4 first, then re-run this "
            "with --ack-rule-policy-gap to opt in explicitly."
        )
        raise SystemExit(2)

    # ── 1. Load medium_4 config ──────────────────────────────────────
    print("\n[1] Loading medium_4 config ...")
    with open(args.config) as f:
        config = yaml.safe_load(f)
    scheme = config.get("action", {}).get("scheme", "medium_4")

    # Sync the surviving merged vocab from the 70b medium_4 merge map (authoritative).
    mm_path = ROOT / config["data"]["merge_map"]
    if mm_path.exists():
        with open(mm_path) as f:
            mm_cfg = yaml.safe_load(f) or {}
        vmi = mm_cfg.get("valid_merged_indices")
        if vmi:
            config["action"]["valid_merged_indices"] = sorted(int(a) for a in vmi)
            print(f"  valid_merged_indices synced from {mm_path.name}: "
                  f"{config['action']['valid_merged_indices']}")
    else:
        print(f"  WARNING: medium_4 merge map {mm_path} not found; using config "
              f"placeholder vocab (run 70b --scheme medium_4 first).")

    # ── 2. Load medium_4 DTR data ────────────────────────────────────
    print("\n[2] Loading medium_4 DTR data ...")
    state_df, outcomes_df, propensity_df, temporal_cols, static_cols, merge_map = (
        load_dtr_data(ROOT, config)
    )
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

    # ── 3. Phenotype labels (hf_subtype) ─────────────────────────────
    print("\n[3] Loading phenotype labels ...")
    pheno_df = pd.read_parquet(ROOT / config["data"]["phenotype_labels"])
    if "hf_subtype" not in pheno_df.columns:
        print("  FATAL: 'hf_subtype' missing from frozen phenotype labels.")
        sys.exit(1)
    pheno_test = pheno_df[pheno_df["stay_id"].isin(set(test_ids))].copy()

    # ── 4. Load trained medium_4 model ───────────────────────────────
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = resolve_checkpoint(args)
    d_temporal = len(temporal_cols)
    d_static = len(static_cols)
    n_actions = len(config["action"]["valid_merged_indices"])
    print(f"\n[4] Loading medium_4 80c G-Transformer (device={device}) ...")
    print(f"    checkpoint: {ckpt_path}")
    print(f"    d_temporal={d_temporal}, d_static={d_static}, n_actions={n_actions}")
    if not ckpt_path.exists():
        print(f"  FATAL: medium_4 checkpoint not found: {ckpt_path}")
        print("  → Run the medium_4 chain first (45b→45c→70b→80c --scheme medium_4).")
        sys.exit(1)
    model = load_gtransformer(config, ckpt_path, d_temporal, d_static,
                              n_actions, device)

    # ── 5. DM-rollout ATE: overall + by hf_subtype ───────────────────
    print(f"\n[5] DM-rollout ATE (overall + by hf_subtype, B={args.n_bootstrap}) ...")

    def dm_eval(ds, **kw):
        return evaluate_dm(model, ds, device, use_rollout=True)

    rows: list[dict] = []

    # Build (subgroup_name, stay_ids) list: ALL first, then each hf_subtype.
    groups: list[tuple[str, np.ndarray]] = [
        ("ALL", np.asarray(test_ids)),
    ]
    for hf in HF_SUBTYPES:
        ids = pheno_test.loc[pheno_test["hf_subtype"] == hf, "stay_id"].values
        groups.append((hf, ids))

    for subgroup, ids in tqdm(groups, desc="Subgroups", unit="grp"):
        n_stays = int(len(ids))
        if n_stays < MIN_CELL_N:
            tqdm.write(f"  {subgroup:14s}: n={n_stays:4d}  SKIP (n<{MIN_CELL_N})")
            rows.append({
                "scheme": scheme, "subgroup": subgroup, "estimator": "DM_rollout",
                "n_stays": n_stays, "v_observed": np.nan, "v_rule": np.nan,
                "ate": np.nan, "ate_ci_lo": np.nan, "ate_ci_hi": np.nan,
            })
            continue
        sub_ds = build_subgroup_ds(
            state_df, outcomes_df, propensity_df,
            temporal_cols, static_cols, merge_map, ids, config,
        )
        point = evaluate_dm(model, sub_ds, device, use_rollout=True)
        boot = bootstrap_ope(
            dm_eval, sub_ds,
            n_resamples=args.n_bootstrap, seed=args.boot_seed,
        )
        ate_lo = boot.get("ate_ci_lo", np.nan)
        ate_hi = boot.get("ate_ci_hi", np.nan)
        rows.append({
            "scheme": scheme,
            "subgroup": subgroup,
            "estimator": "DM_rollout",
            "n_stays": n_stays,
            "v_observed": float(point["v_observed"]),
            "v_rule": float(point["v_rule"]),
            "ate": float(point["ate"]),
            "ate_ci_lo": float(ate_lo) if ate_lo == ate_lo else np.nan,
            "ate_ci_hi": float(ate_hi) if ate_hi == ate_hi else np.nan,
        })
        tqdm.write(
            f"  {subgroup:14s}: n={n_stays:4d}  "
            f"V(obs)={point['v_observed']:+.4f} V(rule)={point['v_rule']:+.4f}  "
            f"ATE={point['ate']:+.4f} [{ate_lo}, {ate_hi}]"
        )

    out_df = pd.DataFrame(rows, columns=[
        "scheme", "subgroup", "estimator", "n_stays",
        "v_observed", "v_rule", "ate", "ate_ci_lo", "ate_ci_hi",
    ])
    out_csv = TABLES_DIR / "80f_medium4_granularity_ate.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_csv} ({len(out_df)} rows)")

    # ── 6. Report ────────────────────────────────────────────────────
    print("\n[6] Generating report ...")
    elapsed = time.time() - t0

    def fmt(x, spec="+.4f"):
        return "NA" if (x is None or (isinstance(x, float) and np.isnan(x))) else format(x, spec)

    lines = [
        "=" * 72,
        "80f — medium_4 (32-cell) Granularity Sensitivity (MIMIC-IV)",
        "=" * 72,
        f"Generated: {ts}",
        f"Scheme: {scheme} (4 fluid × 4 vaso × 2 diuretic = 32 joint cells)",
        f"Estimator: 80c G-Transformer DM rollout",
        f"Checkpoint: {ckpt_path.name}",
        f"Action vocab (merged): {config['action']['valid_merged_indices']}",
        f"Bootstrap: B={args.n_bootstrap}, seed={args.boot_seed}",
        f"Wall time: {elapsed/60:.1f} min",
        "",
        "medium_4 PIPELINE CHAIN (additive, default-preserving)",
        "──────────────────────────────────────────────────────",
        "  45b_encode_actions_2h.py        --scheme medium_4",
        "  45c_build_state_tensors.py       --scheme medium_4",
        "  70b_fit_behavior_policy.py       --scheme medium_4",
        "  80c_train_gtransformer.py        --scheme medium_4 --use-rollout --tag rollout_ow1",
        "  80f_medium4_granularity_sensitivity.py   (this script)",
        "",
        "DM-ROLLOUT ATE (overall + by hf_subtype)",
        "────────────────────────────────────────",
        f"  {'subgroup':>14s}  {'n':>5s}  {'V(obs)':>9s}  {'V(rule)':>9s}  "
        f"{'ATE':>9s}  {'ATE_CI':>22s}",
    ]
    for r in rows:
        ci = f"[{fmt(r['ate_ci_lo'])},{fmt(r['ate_ci_hi'])}]"
        lines.append(
            f"  {r['subgroup']:>14s}  {r['n_stays']:5d}  "
            f"{fmt(r['v_observed']):>9s}  {fmt(r['v_rule']):>9s}  "
            f"{fmt(r['ate']):>9s}  {ci:>22s}"
        )

    lines.extend([
        "",
        "INTERPRETATION (spec §F)",
        "────────────────────────",
        "  - Compare the medium_4 (32-cell) DM-rollout ATE against the frozen",
        "    coarse_3 (18-cell) primary (overall ≈ -1.37pp test-set / -4.87pp",
        "    anchor; HFpEF ≈ 1.9× no_hf). Concordant sign + magnitude → the",
        "    primary finding is robust to action granularity.",
        "  - HFpEF medium_4 cells are sparse (≈8 of 32 cells <25 stays) → some",
        "    subgroup CIs WIDE; read as consistency, not independent confirmation.",
        "  - The coarse_3 primary artifacts are NEVER modified by this chain.",
    ])

    report_path = REPORTS_DIR / "80f_medium4_granularity_summary.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Report: {report_path}")
    print(f"\n80f medium_4 granularity sensitivity COMPLETE. ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
