#!/usr/bin/env python3
"""
P16b_sepsis_ards_gtransformer_subgroup — G-Transformer DM subgroup sensitivity.

Purpose
-------
Confirm P16a's IPTW-derived sepsis/ARDS subgroup signal under the primary
80c G-Transformer DM estimator. Loads the saved 80c checkpoint and re-
evaluates DM ATE within each P11 acute-driver subgroup on the held-out
test split (same split used in the manuscript primary results).

Compute: G1 (single GPU, ~20-40 min depending on subgroup count).

Usage
-----
    env/run_py.sh scripts/P16b_sepsis_ards_gtransformer_subgroup.py \\
        --tag rollout_ow1                # or whichever 80c variant

Inputs
------
- models/80c_gtransformer<_tag>_best.pt   (saved checkpoint)
- data/staging/P11_acute_illness_driver_refined_mimic.parquet
- (everything load_dtr_data needs)

Outputs
-------
- tables/P16b_subgroup_dm_<tag>.csv
- reports/P16b_subgroup_dm_<tag>_summary.txt
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── Project root ──
ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(ROOT / "src"))

from hfpef_dtr.models.dtr_dataset import DTRDataset, load_dtr_data
from hfpef_dtr.models.dataset import make_train_val_test_split
from hfpef_dtr.models.gtransformer import build_gtransformer_from_config
from hfpef_dtr.models.ope import evaluate_dm


# ── 80c-as-library loader (only for load_config + paths) ─────────────
def _load_80c_module():
    spec = importlib.util.spec_from_file_location(
        "_step80c",
        Path(__file__).resolve().parent / "80c_train_gtransformer.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Subgroups (boolean flags from P11) and exclusion sensitivities.
SUBGROUP_FLAGS = [
    "is_cardiogenic_shock",
    "is_septic_shock",
    "is_sepsis_no_shock",
    "is_ards",
    "is_resp_failure_other",
    "is_neurocritical",
    "is_trauma",
    "is_hemorrhage",
    "is_aki_primary",
    "has_cardiac_procedure",
    "has_noncardiac_procedure_only",
]
EXCLUSION_SETS = [
    ("excl_ards",          ["is_ards"]),
    ("excl_severe_resp",   ["is_ards", "is_resp_failure_other"]),
    ("excl_septic_shock",  ["is_septic_shock"]),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P16b G-Transformer subgroup sensitivity")
    p.add_argument("--tag", type=str, default="rollout_ow1",
                   help="80c training tag (matches saved checkpoint name)")
    p.add_argument("--use-rollout", action="store_true", default=True,
                   help="Use rollout-style DM evaluation (matches manuscript primary)")
    p.add_argument("--no-rollout", dest="use_rollout", action="store_false")
    p.add_argument("--n-boot", type=int, default=100,
                   help="Bootstrap reps per subgroup (default: 100)")
    return p.parse_args()


def _build_dataset(group_ids, state_df, outcomes_df, propensity_df,
                   temporal_cols, static_cols, merge_map, config) -> DTRDataset:
    return DTRDataset(
        state_df=state_df,
        outcomes_df=outcomes_df,
        propensity_df=propensity_df,
        stay_ids=group_ids,
        temporal_cols=temporal_cols,
        static_cols=static_cols,
        merge_map=merge_map,
        n_bins=config["grid"]["n_bins"],
        action_vocab=config["action"]["valid_merged_indices"],
    )


def evaluate_subgroup(
    label: str,
    group_ids: np.ndarray,
    model, device, config,
    state_df, outcomes_df, propensity_df,
    temporal_cols, static_cols, merge_map,
    use_rollout: bool, n_boot: int,
) -> dict:
    if len(group_ids) < 30:
        return {"label": label, "n_stays": len(group_ids),
                "v_observed": np.nan, "v_rule": np.nan, "ate": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan,
                "skipped_reason": f"n<30 (n={len(group_ids)})"}

    ds = _build_dataset(group_ids, state_df, outcomes_df, propensity_df,
                         temporal_cols, static_cols, merge_map, config)
    dm = evaluate_dm(model, ds, device, use_rollout=use_rollout)

    # Bootstrap CI by resampling stay_ids within the subgroup.
    rng = np.random.default_rng(20260510)
    ates = []
    try:
        from tqdm import tqdm
        outer = tqdm(range(n_boot), desc=f"    boot:{label}", unit="rep")
    except ImportError:
        outer = range(n_boot)
    for _ in outer:
        boot_ids = rng.choice(group_ids, size=len(group_ids), replace=True)
        try:
            ds_b = _build_dataset(boot_ids, state_df, outcomes_df, propensity_df,
                                   temporal_cols, static_cols, merge_map, config)
            dm_b = evaluate_dm(model, ds_b, device, use_rollout=use_rollout)
            ates.append(dm_b["ate"])
        except (RuntimeError, ValueError):
            continue
    if len(ates) >= n_boot * 0.5:
        ci_lo = float(np.percentile(ates, 2.5))
        ci_hi = float(np.percentile(ates, 97.5))
    else:
        ci_lo, ci_hi = np.nan, np.nan

    return {
        "label": label,
        "n_stays": int(len(group_ids)),
        "v_observed": float(dm["v_observed"]),
        "v_rule": float(dm["v_rule"]),
        "ate": float(dm["ate"]),
        "ci_lo": ci_lo, "ci_hi": ci_hi,
        "skipped_reason": "",
    }


def main() -> None:
    args = parse_args()
    mod80c = _load_80c_module()
    config = mod80c.load_config()
    device = mod80c.get_device()
    mod80c.set_seed(config["split"]["seed"])

    print("=" * 70)
    print(f"P16b_sepsis_ards_gtransformer_subgroup — tag={args.tag}")
    print(f"  use_rollout={args.use_rollout}  n_boot={args.n_boot}")
    print("=" * 70)

    print("\n[1/4] Loading data + checkpoint ...")
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
    test_id_set = set(test_ids)
    print(f"  Test set: {len(test_ids):,} stays")

    model_id = f"{mod80c.MODEL_ID}_{args.tag}" if args.tag else mod80c.MODEL_ID
    ckpt_path = mod80c.MODELS_DIR / f"{model_id}_best.pt"
    if not ckpt_path.exists():
        print(f"  ERROR: checkpoint not found at {ckpt_path}")
        sys.exit(1)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Build a reference dataset on the full test set to recover the model's
    # input dimensions (d_temporal / d_static / n_actions). This mirrors
    # 80c's training-time pattern, which reads these from train_ds.
    ref_ds = _build_dataset(
        test_ids, state_df, outcomes_df, propensity_df,
        temporal_cols, static_cols, merge_map, config,
    )
    model_cfg = config["models"][mod80c.MODEL_ID]
    model = build_gtransformer_from_config(
        d_temporal=ref_ds.d_temporal,
        d_static=ref_ds.d_static,
        n_actions=ref_ds.n_actions,
        model_cfg=model_cfg,
        grid_cfg=config["grid"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded checkpoint: {ckpt_path}")
    print(f"  Model dims: d_temporal={ref_ds.d_temporal}, "
          f"d_static={ref_ds.d_static}, n_actions={ref_ds.n_actions}")

    print("\n[2/4] Loading P11 subgroup flags ...")
    p11 = pd.read_parquet(
        ROOT / "data" / "staging" / "P11_acute_illness_driver_refined_mimic.parquet",
        columns=["stay_id"] + SUBGROUP_FLAGS,
    )
    p11 = p11[p11["stay_id"].isin(test_id_set)].copy()
    for c in SUBGROUP_FLAGS:
        p11[c] = p11[c].fillna(False).astype(bool)

    print("\n[3/4] Per-subgroup DM + bootstrap ...")
    rows = []
    for flag in SUBGROUP_FLAGS:
        group_ids = p11.loc[p11[flag], "stay_id"].to_numpy()
        res = evaluate_subgroup(
            f"{flag}=True", group_ids,
            model, device, config,
            state_df, outcomes_df, propensity_df,
            temporal_cols, static_cols, merge_map,
            args.use_rollout, args.n_boot,
        )
        print(f"  {flag}=True: {res}")
        rows.append(res)

    # Exclusion-sensitivity slices.
    for label, drop_flags in EXCLUSION_SETS:
        keep = p11.loc[~p11[drop_flags].any(axis=1), "stay_id"].to_numpy()
        res = evaluate_subgroup(
            label, keep,
            model, device, config,
            state_df, outcomes_df, propensity_df,
            temporal_cols, static_cols, merge_map,
            args.use_rollout, args.n_boot,
        )
        print(f"  {label}: {res}")
        rows.append(res)

    print("\n[4/4] Saving outputs ...")
    df = pd.DataFrame(rows)
    table_path = ROOT / "tables" / f"P16b_subgroup_dm_{args.tag}.csv"
    df.to_csv(table_path, index=False)
    print(f"  → {table_path}")

    df_sorted = df.sort_values("n_stays", ascending=False)
    report = textwrap.dedent(f"""\
    ================================================================
    P16b_sepsis_ards_gtransformer_subgroup — Summary Report
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ================================================================

    Model     : 80c G-Transformer (tag={args.tag})
    Test set  : {len(test_ids):,} stays
    Bootstrap : {args.n_boot} reps
    Rollout DM: {args.use_rollout}

    SUBGROUP TABLE
    ───────────────
    {textwrap.indent(df_sorted.to_string(index=False), '  ')}

    INTERPRETATION GUIDE
    ─────────────────────
    - ate < 0 → rule policy associated with lower predicted mortality vs
      observed in that subgroup. Compare 95 % CI to zero.
    - Compare row-by-row against P16a's IPTW Hájek estimates. Sign
      concordance across the two methods supports a robust subgroup signal.
      Sign disagreement on a heavy-tailed subgroup (low n_stays) most
      likely reflects estimator noise rather than a true reversal.
    - The exclusion-sensitivity rows (excl_*) re-estimate the cohort-wide
      ATE after dropping stays in the named flag(s). Stable ATE here
      supports robustness to the named driver class.

    OUTPUT
    ──────
    Table: {table_path}
    """)
    report_path = ROOT / "reports" / f"P16b_subgroup_dm_{args.tag}_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"  → Report: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
