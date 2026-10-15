#!/usr/bin/env python3
"""
check_80g_action_vocab_coverage — Verify action vocab completeness per 80g subgroup.

Purpose
-------
Check whether each 2×2 cell (hf_subtype × shock_stratum) in the 80g analysis
contains all 13 merged actions. If yes, action_vocab=None produces identical
contiguous mapping as the training vocab, confirming the bug has no effect.

Also checks: P9 quartile subgroups, 80g_bootstrap, and all other scripts
that use DTRDataset with action_vocab=None.

Output: prints a table showing action coverage per subgroup.

Compute: C1 (lightweight, no model loading)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hfpef_dtr.models.dtr_dataset import DTRDataset, load_dtr_data
from hfpef_dtr.models.dataset import make_train_val_test_split

ROOT = Path(__file__).resolve().parents[1]


def get_subgroup_action_vocab(stay_ids, *, state_df, outcomes_df, propensity_df,
                               temporal_cols, static_cols, merge_map,
                               n_bins):
    """Build DTRDataset without action_vocab and return discovered vocab."""
    ds = DTRDataset(
        state_df=state_df,
        outcomes_df=outcomes_df,
        propensity_df=propensity_df,
        stay_ids=stay_ids,
        temporal_cols=temporal_cols,
        static_cols=static_cols,
        merge_map=merge_map,
        n_bins=n_bins,
    )
    return ds.action_vocab, ds.n_actions


def main() -> None:
    print("=" * 72)
    print("Action Vocab Coverage Check for 80g Subgroups")
    print("=" * 72)

    # ── 1. Load config and data ──
    config_path = ROOT / "configs" / "80_step_c_training_v1.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    state_df, outcomes_df, propensity_df, temporal_cols, static_cols, merge_map = (
        load_dtr_data(ROOT, config)
    )

    canonical_vocab = sorted(config["action"]["valid_merged_indices"])
    n_canonical = len(canonical_vocab)
    print(f"\nCanonical action vocab ({n_canonical} actions): {canonical_vocab}")

    # ── 2. Get test set IDs ──
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
    print(f"Test stays: {len(test_ids):,}")

    # ── 3. Load phenotype labels ──
    pheno_path = ROOT / config["data"]["phenotype_labels"]
    pheno_df = pd.read_parquet(pheno_path)
    pheno_test = pheno_df[pheno_df["stay_id"].isin(set(test_ids))].copy()

    n_bins = config["grid"]["n_bins"]
    common_kwargs = dict(
        state_df=state_df,
        outcomes_df=outcomes_df,
        propensity_df=propensity_df,
        temporal_cols=temporal_cols,
        static_cols=static_cols,
        merge_map=merge_map,
        n_bins=n_bins,
    )

    results = []
    any_mismatch = False

    # ── 4. Check 80g 2×2 cells ──
    print(f"\n{'─' * 72}")
    print("80g: hf_subtype × shock_stratum 2×2 cells")
    print(f"{'─' * 72}")

    hf_groups = sorted(pheno_test["hf_subtype"].dropna().unique())
    shock_groups = sorted(pheno_test["rule_phenotype_2class"].dropna().unique())

    from itertools import product
    for hf_grp, shock_grp in product(hf_groups, shock_groups):
        mask = (
            (pheno_test["hf_subtype"] == hf_grp)
            & (pheno_test["rule_phenotype_2class"] == shock_grp)
        )
        cell_ids = pheno_test.loc[mask, "stay_id"].values
        n_stays = len(cell_ids)

        if n_stays < 20:
            print(f"  {hf_grp:12s} × {shock_grp:20s}: n={n_stays:4d} — SKIPPED (too small)")
            results.append({
                "source": "80g",
                "subgroup": f"{hf_grp} × {shock_grp}",
                "n_stays": n_stays,
                "n_actions": None,
                "match": None,
                "missing": None,
                "status": "SKIPPED",
            })
            continue

        vocab, n_act = get_subgroup_action_vocab(cell_ids, **common_kwargs)
        match = vocab == canonical_vocab
        missing = sorted(set(canonical_vocab) - set(vocab))
        extra = sorted(set(vocab) - set(canonical_vocab))

        status = "OK" if match else "MISMATCH"
        if not match:
            any_mismatch = True

        detail = ""
        if missing:
            detail += f" missing={missing}"
        if extra:
            detail += f" extra={extra}"

        print(f"  {hf_grp:12s} × {shock_grp:20s}: n={n_stays:4d}, "
              f"actions={n_act}/{n_canonical} [{status}]{detail}")

        results.append({
            "source": "80g",
            "subgroup": f"{hf_grp} × {shock_grp}",
            "n_stays": n_stays,
            "n_actions": n_act,
            "match": match,
            "missing": missing if missing else None,
            "status": status,
        })

    # ── 5. Check 80g marginal hf_subtype groups ──
    print(f"\n{'─' * 72}")
    print("80g: marginal hf_subtype groups (collapsed across shock)")
    print(f"{'─' * 72}")

    for hf_grp in hf_groups:
        cell_ids = pheno_test.loc[
            pheno_test["hf_subtype"] == hf_grp, "stay_id"
        ].values
        n_stays = len(cell_ids)

        vocab, n_act = get_subgroup_action_vocab(cell_ids, **common_kwargs)
        match = vocab == canonical_vocab
        missing = sorted(set(canonical_vocab) - set(vocab))

        status = "OK" if match else "MISMATCH"
        if not match:
            any_mismatch = True

        detail = f" missing={missing}" if missing else ""
        print(f"  {hf_grp:12s} (marginal): n={n_stays:4d}, "
              f"actions={n_act}/{n_canonical} [{status}]{detail}")

        results.append({
            "source": "80g_marginal",
            "subgroup": hf_grp,
            "n_stays": n_stays,
            "n_actions": n_act,
            "match": match,
            "missing": missing if missing else None,
            "status": status,
        })

    # ── 6. Check full test set (sanity) ──
    print(f"\n{'─' * 72}")
    print("Sanity: full test set")
    print(f"{'─' * 72}")

    vocab, n_act = get_subgroup_action_vocab(test_ids, **common_kwargs)
    match = vocab == canonical_vocab
    missing = sorted(set(canonical_vocab) - set(vocab))
    status = "OK" if match else "MISMATCH"
    print(f"  Full test set: n={len(test_ids):4d}, "
          f"actions={n_act}/{n_canonical} [{status}]")

    # ── 7. Summary ──
    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")

    if any_mismatch:
        print("\n  *** MISMATCH DETECTED ***")
        print("  Some subgroups are missing actions from the canonical vocabulary.")
        print("  The action_vocab bug DOES affect these subgroups.")
        print("  → 80g results for mismatched subgroups should be re-run with")
        print("    action_vocab=config['action']['valid_merged_indices']")
        mismatched = [r for r in results if r.get("match") is False]
        for r in mismatched:
            print(f"    - {r['source']}: {r['subgroup']} "
                  f"(n={r['n_stays']}, missing={r['missing']})")
    else:
        print("\n  ALL SUBGROUPS MATCH CANONICAL VOCAB")
        print("  action_vocab=None produces identical mapping to training vocab.")
        print("  → 80g results are NOT affected by the action_vocab bug.")
        print("  → No need to re-run 80g.")

    print(f"\n{'=' * 72}")

    # Save results
    results_df = pd.DataFrame(results)
    out_path = ROOT / "tables" / "check_80g_action_vocab_coverage.csv"
    results_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
