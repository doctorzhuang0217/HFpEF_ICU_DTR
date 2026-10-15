#!/usr/bin/env python3
"""P3_behavior_policy_diagnostics — Behavior policy accuracy diagnostics.

Purpose
-------
Compute detailed behavior policy diagnostics: top-1 and top-3 accuracy,
per-bin accuracy trends across the 12 decision points, and accuracy
breakdowns by HFpEF vs no_hf subgroup.

Inputs
------
- data/interim/70b_propensity_coarse3_v1_mimic.parquet   (propensity predictions)
- data/interim/45c_state_tensors_v1_mimic.parquet        (state features, for bin info)
- data/interim/45b_actions_2h_v1_mimic.parquet           (observed actions)
- data/processed/phenotype_labels_frozen_v1_mimic.parquet (HF subtype labels)

Outputs
-------
- reports/P3_behavior_policy_diagnostics_summary.txt

Compute: C1 (lightweight CPU)
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path

# ── Constants ────────────────────────────────────────────────────────────
N_BINS = 12
N_ACTION_CELLS_COARSE3 = 18


def _find_propensity_file(interim_dir: Path) -> Path | None:
    """Look for propensity prediction files with several fallback patterns."""
    candidates = [
        interim_dir / "70b_propensity_coarse3_v1_mimic.parquet",
        interim_dir / "70b_propensity_coarse2_v1_mimic.parquet",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Glob fallback
    hits = sorted(interim_dir.glob("70b_propensity*mimic*.parquet"))
    if hits:
        return hits[0]
    hits = sorted(interim_dir.glob("*propensity*mimic*.parquet"))
    if hits:
        return hits[0]
    return None


def _compute_top_k_accuracy(
    prob_matrix: np.ndarray, true_actions: np.ndarray, k: int
) -> float:
    """Compute top-k accuracy from a probability matrix."""
    top_k_preds = np.argsort(prob_matrix, axis=1)[:, -k:]
    hits = np.array([
        true_actions[i] in top_k_preds[i] for i in range(len(true_actions))
    ])
    return float(hits.mean())


def main() -> None:
    cfg = load_config()
    interim_dir = resolve_path(cfg, "interim")
    processed_dir = resolve_path(cfg, "processed")
    reports_dir = resolve_path(cfg, "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 72)
    print("P3 — Behavior Policy Diagnostics")
    print("=" * 72)

    # ── 1. Locate and load propensity predictions ────────────────────────
    print("\n[1] Loading propensity predictions...")
    prop_path = _find_propensity_file(interim_dir)
    if prop_path is None:
        msg = (
            "No propensity prediction file found.\n"
            "Expected: data/interim/70b_propensity_coarse3_v1_mimic.parquet\n"
            "Run scripts/70b_fit_behavior_policy.py first to generate predictions."
        )
        print(f"  SKIP: {msg}")
        report_path = reports_dir / "P3_behavior_policy_diagnostics_summary.txt"
        with open(report_path, "w") as f:
            f.write(f"P3 — Behavior Policy Diagnostics (SKIPPED)\n")
            f.write(f"Generated: {ts}\n\n")
            f.write(f"{msg}\n")
        print(f"\n  Report (skipped): {report_path}")
        return

    print(f"  Found: {prop_path.name}")
    prop_df = pd.read_parquet(prop_path)
    print(f"  Rows: {len(prop_df):,}")

    # Detect which propensity variant (coarse3 vs coarse2)
    prob_cols = sorted([c for c in prop_df.columns if c.startswith("prob_a")])
    n_action_cells = len(prob_cols)
    is_coarse3 = "coarse3" in prop_path.name
    variant = "coarse3" if is_coarse3 else "coarse2"
    print(f"  Variant: {variant} ({n_action_cells} action cells)")

    # Determine action column
    if "action_idx_merged" in prop_df.columns:
        action_col = "action_idx_merged"
    elif "action_idx_coarse2" in prop_df.columns:
        action_col = "action_idx_coarse2"
    elif "action_idx" in prop_df.columns:
        action_col = "action_idx"
    else:
        print("  ERROR: No action index column found in propensity data.")
        return

    # ── 2. Load observed actions (for cross-check) ───────────────────────
    print("\n[2] Loading observed actions...")
    actions_path = interim_dir / "45b_actions_2h_v1_mimic.parquet"
    if actions_path.exists():
        actions_df = pd.read_parquet(
            actions_path, columns=["stay_id", "bin", "action_idx"]
        )
        print(f"  Loaded {len(actions_df):,} action rows")
    else:
        actions_df = None
        print("  WARNING: 45b actions file not found; using propensity df actions only.")

    # ── 3. Ensure bin column ─────────────────────────────────────────────
    if "bin" not in prop_df.columns:
        state_path = interim_dir / "45c_state_tensors_v1_mimic.parquet"
        if state_path.exists():
            print("  Loading bin info from state tensors...")
            state_bins = pd.read_parquet(state_path, columns=["stay_id", "bin"])
            # Assume same row order
            if len(state_bins) == len(prop_df):
                prop_df["bin"] = state_bins["bin"].values
            else:
                prop_df = prop_df.merge(
                    state_bins, on="stay_id", how="left", suffixes=("", "_state")
                )
        else:
            print("  WARNING: No bin column and no state tensors — "
                  "per-bin analysis will be skipped.")

    # ── 4. Build probability matrix ──────────────────────────────────────
    print("\n[3] Computing accuracy metrics...")
    prob_matrix = prop_df[prob_cols].values.astype(np.float32)
    true_actions = prop_df[action_col].values.astype(int)

    # Top-1 accuracy
    top1_preds = np.argmax(prob_matrix, axis=1)
    top1_acc = float((top1_preds == true_actions).mean())

    # Top-3 accuracy
    top3_acc = _compute_top_k_accuracy(prob_matrix, true_actions, k=3)

    # Top-5 accuracy
    top5_acc = _compute_top_k_accuracy(prob_matrix, true_actions, k=5)

    n_obs = len(true_actions)
    n_stays = prop_df["stay_id"].nunique()

    print(f"  Observations: {n_obs:,} ({n_stays:,} stays)")
    print(f"  Top-1 accuracy: {top1_acc:.4f} ({top1_acc*100:.1f}%)")
    print(f"  Top-3 accuracy: {top3_acc:.4f} ({top3_acc*100:.1f}%)")
    print(f"  Top-5 accuracy: {top5_acc:.4f} ({top5_acc*100:.1f}%)")

    # ── 5. Per-bin accuracy trend ────────────────────────────────────────
    print("\n[4] Per-bin accuracy trend...")
    bin_results = []
    has_bin = "bin" in prop_df.columns

    if has_bin:
        bins_present = sorted(prop_df["bin"].dropna().unique())
        for b in tqdm(bins_present, desc="Per-bin accuracy"):
            mask = prop_df["bin"] == b
            n_b = int(mask.sum())
            if n_b == 0:
                continue
            preds_b = top1_preds[mask]
            true_b = true_actions[mask]
            acc_b = float((preds_b == true_b).mean())

            # Top-3
            prob_b = prob_matrix[mask]
            top3_b = np.argsort(prob_b, axis=1)[:, -3:]
            t3_acc_b = float(np.mean([
                true_b[i] in top3_b[i] for i in range(len(true_b))
            ]))

            bin_results.append({
                "bin": int(b),
                "time_h": f"{int(b)*2}-{int(b)*2+2}h",
                "n_obs": n_b,
                "top1_acc": acc_b,
                "top3_acc": t3_acc_b,
            })
            print(f"  Bin {int(b):2d} ({int(b)*2:2d}-{int(b)*2+2:2d}h): "
                  f"n={n_b:,}  top1={acc_b:.3f}  top3={t3_acc_b:.3f}")
    else:
        print("  SKIP: no bin column available.")

    # ── 6. Subgroup analysis: HFpEF vs no_hf ────────────────────────────
    print("\n[5] Subgroup analysis by HF subtype...")
    subgroup_results = {}
    labels_path = processed_dir / "phenotype_labels_frozen_v1_mimic.parquet"

    if labels_path.exists():
        labels_df = pd.read_parquet(labels_path)
        # Find the HF subtype column
        hf_col = None
        for candidate in ["comparator_group", "hf_subtype"]:
            if candidate in labels_df.columns:
                hf_col = candidate
                break

        if hf_col is not None:
            print(f"  Using label column: {hf_col}")
            # Merge labels to propensity df
            hf_lookup = labels_df[["stay_id", hf_col]].drop_duplicates("stay_id")
            prop_merged = prop_df.merge(hf_lookup, on="stay_id", how="left")

            subgroups = prop_merged[hf_col].dropna().unique()
            for sg in tqdm(sorted(subgroups, key=str), desc="Subgroup accuracy"):
                mask_sg = (prop_merged[hf_col] == sg).values
                n_sg = int(mask_sg.sum())
                if n_sg == 0:
                    continue
                preds_sg = top1_preds[mask_sg]
                true_sg = true_actions[mask_sg]
                acc_sg = float((preds_sg == true_sg).mean())

                prob_sg = prob_matrix[mask_sg]
                top3_sg = np.argsort(prob_sg, axis=1)[:, -3:]
                t3_acc_sg = float(np.mean([
                    true_sg[i] in top3_sg[i] for i in range(len(true_sg))
                ]))

                n_stays_sg = prop_merged.loc[mask_sg, "stay_id"].nunique()
                subgroup_results[str(sg)] = {
                    "n_obs": n_sg,
                    "n_stays": n_stays_sg,
                    "top1_acc": acc_sg,
                    "top3_acc": t3_acc_sg,
                }
                print(f"  {str(sg):20s}: n_stays={n_stays_sg:,}  "
                      f"top1={acc_sg:.3f}  top3={t3_acc_sg:.3f}")

            # Per-bin × subgroup (HFpEF vs no_hf only)
            key_groups = [g for g in ["HFpEF", "no_hf"] if g in subgroup_results]
            bin_subgroup_results = {}
            if has_bin and len(key_groups) >= 2:
                print("\n  Per-bin accuracy by key subgroups (HFpEF vs no_hf):")
                for sg in key_groups:
                    mask_sg = (prop_merged[hf_col] == sg).values
                    bin_sg_list = []
                    for b in sorted(prop_df["bin"].dropna().unique()):
                        mask_b = (prop_df["bin"] == b).values
                        mask_both = mask_sg & mask_b
                        n_both = int(mask_both.sum())
                        if n_both == 0:
                            continue
                        acc_both = float(
                            (top1_preds[mask_both] == true_actions[mask_both]).mean()
                        )
                        bin_sg_list.append({
                            "bin": int(b),
                            "top1_acc": acc_both,
                            "n_obs": n_both,
                        })
                    bin_subgroup_results[sg] = bin_sg_list
        else:
            print("  WARNING: No HF subtype column found in phenotype labels.")
    else:
        print("  SKIP: phenotype labels file not found.")

    # ── 7. Propensity calibration summary ────────────────────────────────
    print("\n[6] Propensity calibration summary...")
    prop_col = "propensity" if "propensity" in prop_df.columns else None
    prop_stats = {}
    if prop_col:
        pvals = prop_df[prop_col].values
        prop_stats = {
            "mean": float(np.mean(pvals)),
            "median": float(np.median(pvals)),
            "p1": float(np.percentile(pvals, 1)),
            "p5": float(np.percentile(pvals, 5)),
            "p25": float(np.percentile(pvals, 25)),
            "p75": float(np.percentile(pvals, 75)),
            "p95": float(np.percentile(pvals, 95)),
            "p99": float(np.percentile(pvals, 99)),
            "min": float(np.min(pvals)),
            "max": float(np.max(pvals)),
            "pct_below_01": float((pvals < 0.01).mean() * 100),
            "pct_below_05": float((pvals < 0.05).mean() * 100),
        }
        print(f"  Mean propensity: {prop_stats['mean']:.4f}")
        print(f"  Median:          {prop_stats['median']:.4f}")
        print(f"  [P1, P99]:       [{prop_stats['p1']:.4f}, {prop_stats['p99']:.4f}]")
        print(f"  % below 0.01:    {prop_stats['pct_below_01']:.1f}%")
        print(f"  % below 0.05:    {prop_stats['pct_below_05']:.1f}%")

    # ── 8. Action distribution summary ───────────────────────────────────
    print("\n[7] Action distribution...")
    action_counts = pd.Series(true_actions).value_counts().sort_index()
    action_freqs = action_counts / action_counts.sum()
    print(f"  Unique actions observed: {len(action_counts)}")
    print(f"  Most common: action={action_counts.idxmax()} "
          f"({action_freqs.max()*100:.1f}%)")
    print(f"  Least common: action={action_counts.idxmin()} "
          f"({action_freqs.min()*100:.1f}%)")

    # ── 9. Write summary report ──────────────────────────────────────────
    print("\n[8] Writing summary report...")
    report_path = reports_dir / "P3_behavior_policy_diagnostics_summary.txt"

    lines = [
        "=" * 72,
        "BEHAVIOR POLICY DIAGNOSTICS",
        "=" * 72,
        f"Generated: {ts}",
        f"Source: {prop_path.name} (variant: {variant})",
        "",
        "OVERALL ACCURACY",
        "-" * 40,
        f"  Observations:  {n_obs:,} ({n_stays:,} stays)",
        f"  Action cells:  {n_action_cells}",
        f"  Top-1 accuracy: {top1_acc*100:.1f}%",
        f"  Top-3 accuracy: {top3_acc*100:.1f}%",
        f"  Top-5 accuracy: {top5_acc*100:.1f}%",
        "",
    ]

    # Per-bin trend
    if bin_results:
        lines.append("PER-BIN ACCURACY TREND")
        lines.append("-" * 40)
        lines.append(f"  {'Bin':>4s}  {'Time':>8s}  {'N':>8s}  "
                     f"{'Top-1':>7s}  {'Top-3':>7s}")
        lines.append(f"  {'----':>4s}  {'--------':>8s}  {'--------':>8s}  "
                     f"{'-------':>7s}  {'-------':>7s}")
        for r in bin_results:
            lines.append(
                f"  {r['bin']:4d}  {r['time_h']:>8s}  {r['n_obs']:8,}  "
                f"{r['top1_acc']:7.3f}  {r['top3_acc']:7.3f}"
            )
        # Trend summary
        accs = [r["top1_acc"] for r in bin_results]
        lines.append("")
        lines.append(f"  Best bin:  {bin_results[np.argmax(accs)]['bin']} "
                     f"({max(accs)*100:.1f}%)")
        lines.append(f"  Worst bin: {bin_results[np.argmin(accs)]['bin']} "
                     f"({min(accs)*100:.1f}%)")
        lines.append(f"  Range:     {(max(accs)-min(accs))*100:.1f}pp")
        lines.append("")

    # Subgroup results
    if subgroup_results:
        lines.append("ACCURACY BY HF SUBTYPE")
        lines.append("-" * 40)
        lines.append(f"  {'Subgroup':>20s}  {'N_stays':>8s}  {'N_obs':>8s}  "
                     f"{'Top-1':>7s}  {'Top-3':>7s}")
        lines.append(f"  {'--------------------':>20s}  {'--------':>8s}  "
                     f"{'--------':>8s}  {'-------':>7s}  {'-------':>7s}")
        for sg_name in sorted(subgroup_results.keys()):
            sg = subgroup_results[sg_name]
            lines.append(
                f"  {sg_name:>20s}  {sg['n_stays']:8,}  {sg['n_obs']:8,}  "
                f"{sg['top1_acc']:7.3f}  {sg['top3_acc']:7.3f}"
            )

        # HFpEF vs no_hf comparison
        if "HFpEF" in subgroup_results and "no_hf" in subgroup_results:
            diff = (subgroup_results["HFpEF"]["top1_acc"]
                    - subgroup_results["no_hf"]["top1_acc"])
            lines.append("")
            lines.append(f"  HFpEF - no_hf top-1 diff: {diff*100:+.1f}pp")
        lines.append("")

    # Per-bin × subgroup
    if bin_subgroup_results:
        lines.append("PER-BIN ACCURACY: HFpEF vs no_hf")
        lines.append("-" * 40)
        header_parts = [f"  {'Bin':>4s}"]
        for sg in sorted(bin_subgroup_results.keys()):
            header_parts.append(f"  {sg:>12s}")
        lines.append("".join(header_parts))
        lines.append("  " + "-" * (4 + 14 * len(bin_subgroup_results)))
        # Align by bin
        all_bins_sg = set()
        for sg_data in bin_subgroup_results.values():
            for entry in sg_data:
                all_bins_sg.add(entry["bin"])
        for b in sorted(all_bins_sg):
            row_parts = [f"  {b:4d}"]
            for sg in sorted(bin_subgroup_results.keys()):
                entries = [e for e in bin_subgroup_results[sg] if e["bin"] == b]
                if entries:
                    row_parts.append(f"  {entries[0]['top1_acc']:12.3f}")
                else:
                    row_parts.append(f"  {'n/a':>12s}")
            lines.append("".join(row_parts))
        lines.append("")

    # Propensity calibration
    if prop_stats:
        lines.append("PROPENSITY CALIBRATION")
        lines.append("-" * 40)
        lines.append(f"  Mean:             {prop_stats['mean']:.4f}")
        lines.append(f"  Median:           {prop_stats['median']:.4f}")
        lines.append(f"  [P1, P99]:        [{prop_stats['p1']:.4f}, "
                     f"{prop_stats['p99']:.4f}]")
        lines.append(f"  [Min, Max]:       [{prop_stats['min']:.6f}, "
                     f"{prop_stats['max']:.4f}]")
        lines.append(f"  % P(a|s) < 0.01:  {prop_stats['pct_below_01']:.1f}%")
        lines.append(f"  % P(a|s) < 0.05:  {prop_stats['pct_below_05']:.1f}%")
        lines.append("")

    # Action distribution
    lines.append("ACTION DISTRIBUTION")
    lines.append("-" * 40)
    lines.append(f"  Unique actions: {len(action_counts)}")
    for aidx in action_counts.index:
        lines.append(
            f"  action={int(aidx):2d}:  n={action_counts[aidx]:8,}  "
            f"freq={action_freqs[aidx]*100:5.1f}%"
        )
    lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {report_path}")

    print("\nP3 Behavior Policy Diagnostics COMPLETE.")


if __name__ == "__main__":
    main()
