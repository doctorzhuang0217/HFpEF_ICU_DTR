#!/usr/bin/env python3
"""
70b_fit_behavior_policy — Estimate clinician behavior policy (propensity).

Purpose
-------
Fit a model for P(A_t | S_t, A_{<t}, phenotype) — the probability of each
joint action given history. This is the denominator of IPTW weights in Step 75a.

Key design:
  1. Auto-merge very-rare action cells (<0.1%) per 45g Rule 1.
  2. Fit multinomial logistic regression per bin (pooled across bins with
     bin index as feature) on z-scored state + missingness + static covariates.
  3. Compute and cache propensity P(A_t=a | H_t) for every obs.
  4. Report ESS per action cell × HF subtype stratum (45g Rule 2).
  5. Sensitivity: also fit on coarse_2 (9-cell, diuretic collapsed) for HFpEF.

Protocol reference: Step 70b (v3.1), 45g rare-cell handling note

Inputs
------
- data/interim/45c_state_tensors_v1_mimic.parquet
- data/interim/45d_outcomes_absorb_v1_mimic.parquet
- data/processed/phenotype_labels_frozen_v1_mimic.parquet
- configs/action_space_v1.yaml
- configs/70a_policy_objects_manifest_v1.json

Outputs
-------
- models/70b_behavior_policy_coarse3.pkl
- models/70b_behavior_policy_coarse2.pkl
- configs/70b_action_merge_map.yaml
- data/interim/70b_propensity_coarse3_v1_mimic.parquet
- data/interim/70b_propensity_coarse2_v1_mimic.parquet
- tables/70b_ess_per_cell_stratum.csv
- reports/70b_behavior_policy_summary.txt

Compute: C2-C3
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path


# ── Rare-cell thresholds (from 45g) ──────────────────────────────────────
VERY_RARE_THRESHOLD = 0.001  # 0.1% of observation-bins


def _build_merge_map(action_counts: pd.Series, n_cells: int = 18) -> dict:
    """
    Build auto-merge map for very-rare cells (45g Rule 1).

    Merge strategy: collapse diuretic dimension (odd idx → even idx).
    action_idx encoding: fluid * 6 + vaso * 2 + diuretic
    So odd indices have diuretic=1; merge odd→even in same fluid×vaso stratum.
    """
    total = action_counts.sum()
    merge_map = {}
    for aidx in range(n_cells):
        freq = action_counts.get(aidx, 0) / total
        if freq < VERY_RARE_THRESHOLD and aidx % 2 == 1:
            # Merge diuretic=any (odd) → combined with diuretic=none (even)
            merge_map[aidx] = aidx - 1
    return merge_map


def _apply_merge(action_idx: pd.Series, merge_map: dict) -> pd.Series:
    """Apply merge map to action indices."""
    if not merge_map:
        return action_idx.copy()
    return action_idx.replace(merge_map)


def _build_coarse2_action(state_df: pd.DataFrame) -> pd.Series:
    """
    Build coarse_2 action index (3×3=9 cells, diuretic collapsed).

    coarse_2 = fluid_level * 3 + vaso_level
    """
    # Decode original action_idx: fluid*6 + vaso*2 + diuretic
    aidx = state_df["action_idx"].values
    fluid_level = aidx // 6
    vaso_level = (aidx % 6) // 2
    return pd.Series(fluid_level * 3 + vaso_level, index=state_df.index,
                     name="action_idx_coarse2")


def _fit_behavior_policy(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    seed: int = 2024,
) -> tuple:
    """Fit multinomial logistic regression for P(A|S)."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        C=1.0,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_s, y)

    # Predict probabilities
    probs = clf.predict_proba(X_s)

    # Ensure all classes are represented in output
    known_classes = clf.classes_
    full_probs = np.zeros((len(y), n_classes), dtype=np.float32)
    for i, c in enumerate(known_classes):
        if c < n_classes:
            full_probs[:, c] = probs[:, i]

    return clf, scaler, full_probs


def _compute_ess(weights: np.ndarray) -> float:
    """Effective sample size: (sum w)^2 / sum(w^2)."""
    sw = np.sum(weights)
    sw2 = np.sum(weights ** 2)
    if sw2 == 0:
        return 0.0
    return (sw ** 2) / sw2


def _resolve_n_cells(scheme: str, configs_dir: Path) -> int:
    """Number of joint action cells for the given scheme.

    coarse_3 → 18 (frozen primary). Other schemes read joint_cells from
    29a_candidate_action_schemes.yaml.
    """
    if scheme == "coarse_3":
        return N_ACTION_CELLS_COARSE3
    with open(configs_dir / "29a_candidate_action_schemes.yaml") as f:
        cand = yaml.safe_load(f)
    return int(cand["schemes"][scheme]["joint_cells"])


def _load_rule_policy_module():
    """Load the 70c rule-policy script as a module (digit-prefixed filename).

    Returns the imported module exposing `apply_rule_policy`, `_load_norm_params`,
    and `reencode_rule_action`. Used ONLY on the non-default (medium_4) path to
    re-encode the rule action; the coarse_3 path never touches this.
    """
    import importlib.util

    rule_path = Path(__file__).resolve().parent / "70c_define_rule_policy_v1.py"
    spec = importlib.util.spec_from_file_location("_rule_policy_70c", rule_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_medium4_rule_action(
    state_df: pd.DataFrame,
    scheme: str,
    configs_dir: Path,
    merge_map: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute scheme-aware rule actions (raw + merged) for the medium_4 chain.

    The rule DECIDES categorically (via 70c.apply_rule_policy → coarse_3-encoded
    intent), then 70c.reencode_rule_action ordinal-aligns each decision onto the
    `scheme` action vocabulary. Finally the medium_4 merge map collapses very-rare
    cells (45g Rule 1) exactly as for observed actions, so rule and observed
    actions share one vocabulary in the downstream OPE.

    Returns (rule_action_raw, rule_action_merged) as int arrays aligned to
    `state_df` rows.
    """
    rp = _load_rule_policy_module()
    norm_params = rp._load_norm_params(configs_dir)
    # apply_rule_policy returns the frozen coarse_3-encoded rule intent.
    rule_coarse3 = rp.apply_rule_policy(state_df, norm_params).values.astype(int)
    # Re-encode each coarse_3 decision under the target scheme (ordinal align).
    reencode = np.vectorize(
        lambda a: rp.reencode_rule_action(int(a), scheme, configs_dir)
    )
    rule_raw = reencode(rule_coarse3).astype(int)
    # Apply the SAME rare-cell merge map used for observed actions.
    rule_merged = pd.Series(rule_raw).replace(merge_map).values.astype(int)
    return rule_raw, rule_merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="70b — fit behavior policy (clinician propensity)")
    parser.add_argument(
        "--scheme", choices=["coarse_3", "medium_4"], default="coarse_3",
        help="Action-discretization scheme. DEFAULT coarse_3 reproduces the "
             "frozen behavior-policy artifacts exactly; medium_4 reads "
             "*_medium4 inputs and writes *_medium4 outputs.",
    )
    parser.add_argument(
        "--ack-rule-policy-gap", action="store_true", default=False,
        help="REQUIRED for --scheme medium_4. The medium_4 rule-policy action is "
             "now RE-ENCODED into the 32-cell medium_4 vocab (wave-2 §F: 70c "
             "decides categorically, then ordinal-aligns the 3-level intent onto "
             "medium_4's 4-level bins and applies the medium_4 merge map), so the "
             "medium_4 propensity parquet carries a VALID rule_action. The flag is "
             "retained as an explicit acknowledgement that this is the "
             "non-default, finer-granularity sensitivity chain (not the frozen "
             "coarse_3 primary); pass it to opt into the medium_4 run.",
    )
    args = parser.parse_args()
    scheme = args.scheme
    if scheme == "medium_4" and not args.ack_rule_policy_gap:
        raise SystemExit(
            "ABORT (70b): --scheme medium_4 is the non-default finer-granularity "
            "sensitivity chain. The medium_4 rule-policy action re-encoding IS now "
            "implemented (wave-2 §F), but the run must be explicitly opted into "
            "with --ack-rule-policy-gap (it writes *_medium4 artifacts only; the "
            "frozen coarse_3 primary is untouched)."
        )
    suffix = "" if scheme == "coarse_3" else f"_{scheme.replace('_', '')}"

    proj_cfg = load_config()
    interim_dir = resolve_path(proj_cfg, "interim")
    processed_dir = resolve_path(proj_cfg, "processed")
    configs_dir = resolve_path(proj_cfg, "configs")
    models_dir = resolve_path(proj_cfg, "models")
    tables_dir = resolve_path(proj_cfg, "tables")
    reports_dir = resolve_path(proj_cfg, "reports")
    models_dir.mkdir(parents=True, exist_ok=True)

    n_cells = _resolve_n_cells(scheme, configs_dir)

    print("=" * 72)
    print(f"70b — Fit Behavior Policy (Clinician Propensity) [scheme={scheme}]")
    print("=" * 72)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 1. Load data ──────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    manifest_path = configs_dir / "70a_policy_objects_manifest_v1.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    state_df = pd.read_parquet(
        interim_dir / f"45c_state_tensors_v1{suffix}_mimic.parquet")
    outcomes_df = pd.read_parquet(interim_dir / "45d_outcomes_absorb_v1_mimic.parquet")
    labels_df = pd.read_parquet(processed_dir / "phenotype_labels_frozen_v1_mimic.parquet")

    # action_idx lives in 45b (not in 45c state tensors)
    actions_df = pd.read_parquet(
        interim_dir / f"45b_actions_2h_v1{suffix}_mimic.parquet",
        columns=["stay_id", "bin", "action_idx"],
    )
    state_df = state_df.merge(actions_df, on=["stay_id", "bin"], how="left")

    n_obs = len(state_df)
    n_stays = state_df["stay_id"].nunique()
    print(f"  {n_obs:,} obs, {n_stays:,} stays")

    # ── 2. Build feature matrix for behavior policy ───────────────────────
    print("\n[2] Building feature matrix...")

    temporal_cols = manifest["state_schema"]["temporal_cols"]
    static_cols = manifest["state_schema"]["static_cols"]

    # Merge static covariates (from labels or state first-row)
    # Add bin index as feature (treatment patterns vary by time)
    feature_cols = temporal_cols.copy()
    # Add bin as feature
    state_df["bin_normalized"] = state_df["bin"] / 11.0  # normalize to [0,1]
    feature_cols.append("bin_normalized")

    # Add static cols if present (only numeric ones)
    STRING_COLS = {"gender", "admission_type", "hf_subtype",
                   "comparator_group", "dominant_syndrome", "reason_for_icu"}
    for sc in static_cols:
        if sc in state_df.columns and sc not in STRING_COLS:
            feature_cols.append(sc)

    # Filter to valid feature columns
    feature_cols = [c for c in feature_cols if c in state_df.columns]
    print(f"  Feature columns: {len(feature_cols)}")

    X = state_df[feature_cols].values.astype(np.float32)
    np.nan_to_num(X, copy=False, nan=0.0)
    y = state_df["action_idx"].values.astype(int)

    # ── 3. Rare-cell merge (45g Rule 1) ───────────────────────────────────
    print("\n[3] Applying rare-cell merge rules (45g Rule 1)...")
    action_counts = state_df["action_idx"].value_counts().sort_index()
    merge_map = _build_merge_map(action_counts, n_cells=n_cells)
    y_merged = _apply_merge(pd.Series(y), merge_map).values

    n_merged = sum(1 for v in merge_map.values() if v != v)  # shouldn't happen
    print(f"  Merge map: {merge_map}")
    print(f"  Cells merged: {len(merge_map)}")
    n_actions_after = len(set(y_merged))
    print(f"  Action cells after merge: {n_actions_after} (from {n_cells})")

    # Surviving merged action indices (consumed by 80c subgroup / 80f as the
    # action vocab for the medium_4 chain). For coarse_3 this matches the frozen
    # configs/80_step_c_training_v1.yaml `valid_merged_indices`.
    valid_merged_indices = sorted(int(a) for a in set(y_merged))

    # Save merge map.
    # coarse_3 (DEFAULT) → frozen path 70b_action_merge_map.yaml (unchanged).
    # medium_4 → distinct 70b_action_merge_map_medium4.yaml; also records the
    # surviving vocab so the medium_4 trainer/orchestrator can pick it up.
    if scheme == "coarse_3":
        merge_map_path = configs_dir / "70b_action_merge_map.yaml"
        with open(merge_map_path, "w") as f:
            yaml.dump({
                "step": "70b",
                "date": ts,
                "rule": "45g_rule_1",
                "threshold": VERY_RARE_THRESHOLD,
                "merge_map": {int(k): int(v) for k, v in merge_map.items()},
                "description": "Odd action_idx (diuretic=any) merged to even "
                              "(diuretic=none/any combined) when <0.1% frequency",
            }, f, default_flow_style=False)
    else:
        merge_map_path = configs_dir / f"70b_action_merge_map{suffix}.yaml"
        with open(merge_map_path, "w") as f:
            yaml.dump({
                "step": "70b",
                "date": ts,
                "scheme": scheme,
                "n_raw_cells": int(n_cells),
                "rule": "45g_rule_1",
                "threshold": VERY_RARE_THRESHOLD,
                "merge_map": {int(k): int(v) for k, v in merge_map.items()},
                "valid_merged_indices": valid_merged_indices,
                "description": "Odd action_idx (diuretic=any) merged to even "
                              "(diuretic=none/any combined) when <0.1% frequency",
            }, f, default_flow_style=False)
    print(f"  Saved: {merge_map_path}")

    # ── 4. Fit behavior policy (merged) ──────────────────────────────────
    print(f"\n[4] Fitting behavior policy ({scheme}, merged)...")
    clf_c3, scaler_c3, probs_c3 = _fit_behavior_policy(
        X, y_merged, n_classes=n_cells
    )
    train_acc = (clf_c3.predict(scaler_c3.transform(X)) == y_merged).mean()
    print(f"  Training accuracy: {train_acc:.4f}")

    # Save model
    model_c3_path = models_dir / f"70b_behavior_policy_coarse3{suffix}.pkl"
    with open(model_c3_path, "wb") as f:
        pickle.dump({"model": clf_c3, "scaler": scaler_c3,
                      "merge_map": merge_map, "feature_cols": feature_cols}, f)
    print(f"  Saved: {model_c3_path}")

    # Save propensities
    prop_df_c3 = state_df[["stay_id", "bin", "action_idx"]].copy()
    prop_df_c3["action_idx_merged"] = y_merged
    # P(observed action | history)
    prop_df_c3["propensity"] = probs_c3[np.arange(len(y_merged)), y_merged]
    # Clip tiny propensities to avoid extreme weights
    prop_df_c3["propensity_clipped"] = prop_df_c3["propensity"].clip(lower=0.005)
    for aidx in range(n_cells):
        prop_df_c3[f"prob_a{aidx}"] = probs_c3[:, aidx]

    # ── medium_4 rule-policy action re-encoding (wave-2 §F) ──────────────
    # coarse_3 (DEFAULT) is byte-identical: it never enters this branch, so
    # the frozen coarse_3 propensity parquet is unchanged. For medium_4 the
    # rule decision (coarse_3-encoded) is re-encoded onto the 32-cell medium_4
    # vocab by ordinal alignment (70c.reencode_rule_action) and the SAME merge
    # map is applied, yielding a VALID rule_action for the medium_4 DM-rollout.
    if scheme != "coarse_3":
        print(f"\n  [scheme={scheme}] re-encoding rule-policy action "
              f"(70c ordinal-aligned + medium_4 merge map)...")
        rule_raw, rule_merged = _build_medium4_rule_action(
            state_df, scheme, configs_dir, merge_map
        )
        prop_df_c3["rule_action"] = rule_raw
        prop_df_c3["rule_action_merged"] = rule_merged
        rule_dist = pd.Series(rule_merged).value_counts().sort_index()
        n_distinct = int(rule_dist.size)
        print(f"    rule_action_merged: {n_distinct} distinct medium_4 cells")
        for aidx, cnt in rule_dist.items():
            pct = cnt / len(rule_merged) * 100
            print(f"      cell {int(aidx):2d}: {cnt:,} ({pct:.1f}%)")
        # Sanity: rule cells must live inside the observed merged vocab so the
        # downstream contiguous remap (dtr_dataset) does not silently zero them.
        unsupported = sorted(set(int(a) for a in rule_dist.index)
                             - set(valid_merged_indices))
        if unsupported:
            raise SystemExit(
                f"ABORT (70b medium_4): rule cells {unsupported} are NOT in the "
                f"observed merged vocab {valid_merged_indices}; the downstream "
                f"contiguous remap (dtr_dataset) would map them to action 0, making "
                f"the medium_4 DM-rollout counterfactual INVALID. The medium_4 "
                f"propensity parquet was NOT written. Coarsen the medium_4 "
                f"granularity / inspect support, or revisit the rule->cell mapping, "
                f"before re-running."
            )

    prop_c3_path = interim_dir / f"70b_propensity_coarse3{suffix}_v1_mimic.parquet"
    prop_df_c3.to_parquet(prop_c3_path, index=False)
    print(f"  Saved: {prop_c3_path}")

    # ── 5. Fit behavior policy (coarse_2, 9 cells) ───────────────────────
    # coarse_2 is a coarse_3-specific diuretic-collapsed sensitivity (its decode
    # formula assumes the 18-cell coarse_3 layout). It is NOT part of the
    # medium_4 chain → skip cleanly when scheme != coarse_3.
    if scheme == "coarse_3":
        print("\n[5] Fitting behavior policy (coarse_2, diuretic collapsed)...")
        y_c2 = _build_coarse2_action(state_df).values
        n_cells_c2 = 9

        clf_c2, scaler_c2, probs_c2 = _fit_behavior_policy(
            X, y_c2, n_classes=n_cells_c2
        )
        train_acc_c2 = (clf_c2.predict(scaler_c2.transform(X)) == y_c2).mean()
        print(f"  Training accuracy (coarse_2): {train_acc_c2:.4f}")

        model_c2_path = models_dir / "70b_behavior_policy_coarse2.pkl"
        with open(model_c2_path, "wb") as f:
            pickle.dump({"model": clf_c2, "scaler": scaler_c2,
                          "feature_cols": feature_cols}, f)
        print(f"  Saved: {model_c2_path}")

        prop_df_c2 = state_df[["stay_id", "bin"]].copy()
        prop_df_c2["action_idx_coarse2"] = y_c2
        prop_df_c2["propensity"] = probs_c2[np.arange(len(y_c2)), y_c2]
        prop_df_c2["propensity_clipped"] = prop_df_c2["propensity"].clip(lower=0.005)
        for aidx in range(n_cells_c2):
            prop_df_c2[f"prob_a{aidx}"] = probs_c2[:, aidx]

        prop_c2_path = interim_dir / "70b_propensity_coarse2_v1_mimic.parquet"
        prop_df_c2.to_parquet(prop_c2_path, index=False)
        print(f"  Saved: {prop_c2_path}")
    else:
        print(f"\n[5] coarse_2 diuretic-collapsed sensitivity SKIPPED "
              f"(coarse_3-only; scheme={scheme}).")
        train_acc_c2 = float("nan")
        n_cells_c2 = 0

    # ── 6. ESS monitoring per cell × stratum (45g Rule 2) ─────────────────
    print("\n[6] Computing ESS per action cell × HF stratum...")

    # Merge HF subtype for stratification
    hf_col = None
    for candidate in ["hf_subtype", "comparator_group"]:
        if candidate in state_df.columns:
            hf_col = candidate
            break
    if hf_col is None:
        # Get from labels or outcomes
        if "comparator_group" in labels_df.columns:
            hf_lookup = labels_df[["stay_id", "comparator_group"]]
            prop_df_c3 = prop_df_c3.merge(hf_lookup, on="stay_id", how="left")
            hf_col = "comparator_group"
        else:
            print("  WARNING: No HF subtype column found for ESS stratification")

    ess_rows = []
    if hf_col and hf_col in prop_df_c3.columns:
        strata = prop_df_c3[hf_col].unique()
    elif hf_col and hf_col in state_df.columns:
        prop_df_c3[hf_col] = state_df[hf_col].values
        strata = prop_df_c3[hf_col].unique()
    else:
        strata = ["all"]
        prop_df_c3["_stratum"] = "all"
        hf_col = "_stratum"

    for stratum in sorted(strata, key=str):
        mask_s = prop_df_c3[hf_col] == stratum
        sub = prop_df_c3.loc[mask_s]
        for aidx in sorted(sub["action_idx_merged"].unique()):
            mask_a = sub["action_idx_merged"] == aidx
            if mask_a.sum() == 0:
                continue
            w = 1.0 / sub.loc[mask_a, "propensity_clipped"].values
            ess = _compute_ess(w)
            ess_rows.append({
                "stratum": str(stratum),
                "action_idx": int(aidx),
                "n_obs": int(mask_a.sum()),
                "ess": round(ess, 1),
                "ess_flag": ess < 25,
            })

    ess_df = pd.DataFrame(ess_rows)
    ess_path = tables_dir / f"70b_ess_per_cell_stratum{suffix}.csv"
    ess_df.to_csv(ess_path, index=False)
    n_flagged = ess_df["ess_flag"].sum()
    print(f"  ESS table: {len(ess_df)} cell×stratum combos, "
          f"{n_flagged} flagged (ESS<25)")
    print(f"  Saved: {ess_path}")

    if n_flagged > 3:
        print(f"  ⚠ >3 cell×stratum combos with ESS<25 — "
              f"coarser granularity recommended (45g Rule 2)")

    # ── 7. Propensity diagnostics ─────────────────────────────────────────
    print("\n[7] Propensity diagnostics...")
    prop_vals = prop_df_c3["propensity"].values
    prop_clipped = prop_df_c3["propensity_clipped"].values
    print(f"  Raw propensity:     mean={prop_vals.mean():.4f}, "
          f"median={np.median(prop_vals):.4f}, "
          f"min={prop_vals.min():.6f}, max={prop_vals.max():.4f}")
    print(f"  Clipped propensity: mean={prop_clipped.mean():.4f}, "
          f"P1={np.percentile(prop_clipped, 1):.4f}, "
          f"P99={np.percentile(prop_clipped, 99):.4f}")

    # Crude IPTW weight summary
    w_iptw = 1.0 / prop_clipped
    print(f"  IPTW weights:       mean={w_iptw.mean():.2f}, "
          f"P95={np.percentile(w_iptw, 95):.2f}, "
          f"P99={np.percentile(w_iptw, 99):.2f}, "
          f"max={w_iptw.max():.2f}")

    # ── 8. Summary report ─────────────────────────────────────────────────
    report_path = reports_dir / f"70b_behavior_policy_summary{suffix}.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "=" * 72,
        "70b — Behavior Policy (Clinician Propensity Model)",
        "=" * 72,
        f"Generated: {ts}",
        "",
        "DATA",
        "────",
        f"  Observations: {n_obs:,} ({n_stays:,} stays × 12 bins)",
        f"  Features: {len(feature_cols)}",
        "",
        "RARE-CELL MERGE (45g Rule 1)",
        "────────────────────────────",
        f"  Threshold: <{VERY_RARE_THRESHOLD*100:.1f}% of obs",
        f"  Cells merged: {len(merge_map)}",
        f"  Merge map: {merge_map}",
        f"  Effective action cells: {n_actions_after}",
        "",
        "COARSE_3 BEHAVIOR POLICY (merged)",
        "──────────────────────────────────",
        f"  Model: Multinomial Logistic Regression",
        f"  Training accuracy: {train_acc:.4f}",
        f"  Propensity: mean={prop_vals.mean():.4f}, "
        f"P1={np.percentile(prop_vals, 1):.6f}, "
        f"min={prop_vals.min():.6f}, max={prop_vals.max():.4f}",
        f"  IPTW weights: mean={w_iptw.mean():.2f}, "
        f"P1={np.percentile(w_iptw, 1):.2f}, "
        f"P95={np.percentile(w_iptw, 95):.2f}, "
        f"P99={np.percentile(w_iptw, 99):.2f}, max={w_iptw.max():.2f}",
        "",
        "COARSE_2 BEHAVIOR POLICY (diuretic collapsed)",
        "───────────────────────────────────────────────",
        f"  Cells: {n_cells_c2}",
        f"  Training accuracy: {train_acc_c2:.4f}",
        "",
        "ESS MONITORING (45g Rule 2)",
        "───────────────────────────",
        f"  Cell×stratum combinations: {len(ess_df)}",
        f"  Flagged (ESS<25): {n_flagged}",
    ]
    if n_flagged > 3:
        lines.append("  ⚠ TRIGGER: >3 combos flagged → consider coarser granularity")
    lines.extend([
        "",
        "NEXT STEPS",
        "──────────",
        "  70c: define rule-based policy v1",
        "  75a: use these propensities for dynamic-MSM/IPTW",
    ])

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Report: {report_path}")
    print("\n70b COMPLETE.")


# Fix: _build_merge_map uses module-level constant
N_ACTION_CELLS_COARSE3 = 18

if __name__ == "__main__":
    main()
