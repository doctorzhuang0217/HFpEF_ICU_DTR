"""
Shared reporting module for Step C deep causal models.

Generates standardised output tables and summary text compatible
with 80f_core_model_comparison and the 75c classical baseline format.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def save_policy_values(
    results: dict,
    model_id: str,
    output_dir: str | Path,
) -> Path:
    """Save policy value estimates as CSV (one row per estimator)."""
    rows = []
    for est_name, est_result in results.items():
        if est_name.startswith("_"):
            continue  # skip internal diagnostics (_rollout_drift etc.)
        rows.append({
            "model": model_id,
            "estimator": est_name,
            "v_observed": est_result.get("v_observed"),
            "v_rule": est_result.get("v_rule"),
            "ate": est_result.get("ate"),
            "ess": est_result.get("ess"),
        })
    df = pd.DataFrame(rows)
    path = Path(output_dir) / f"{model_id}_policy_values.csv"
    df.to_csv(path, index=False)
    return path


def save_bootstrap_cis(
    bootstrap_results: dict,
    model_id: str,
    output_dir: str | Path,
) -> Path:
    """Save bootstrap confidence intervals."""
    rows = []
    for est_name, boot in bootstrap_results.items():
        rows.append({
            "model": model_id,
            "estimator": est_name,
            "n_success": boot.get("n_success"),
            "ate_mean": boot.get("ate_mean"),
            "ate_ci_lo": boot.get("ate_ci_lo"),
            "ate_ci_hi": boot.get("ate_ci_hi"),
            "v_rule_mean": boot.get("v_rule_mean"),
            "v_rule_ci_lo": boot.get("v_rule_ci_lo"),
            "v_rule_ci_hi": boot.get("v_rule_ci_hi"),
        })
    df = pd.DataFrame(rows)
    path = Path(output_dir) / f"{model_id}_bootstrap_ci.csv"
    df.to_csv(path, index=False)
    return path


def append_bootstrap_cis(
    bootstrap_results: dict,
    model_id: str,
    output_dir: str | Path,
) -> Path:
    """Append bootstrap-CI rows to an existing {model_id}_bootstrap_ci.csv.

    Additive helper for opt-in flags (e.g. --fqe-bootstrap). Uses the exact
    same column schema as ``save_bootstrap_cis``. IDEMPOTENT UPSERT: if the file
    exists, any existing rows for the same (model, estimator) being written are
    replaced (not duplicated), so re-running an opt-in flag like --fqe-bootstrap
    does NOT create duplicate CI rows; other rows (e.g. DM/DR) are preserved.
    """
    rows = []
    for est_name, boot in bootstrap_results.items():
        rows.append({
            "model": model_id,
            "estimator": est_name,
            "n_success": boot.get("n_success"),
            "ate_mean": boot.get("ate_mean"),
            "ate_ci_lo": boot.get("ate_ci_lo"),
            "ate_ci_hi": boot.get("ate_ci_hi"),
            "v_rule_mean": boot.get("v_rule_mean"),
            "v_rule_ci_lo": boot.get("v_rule_ci_lo"),
            "v_rule_ci_hi": boot.get("v_rule_ci_hi"),
        })
    new_df = pd.DataFrame(rows)
    path = Path(output_dir) / f"{model_id}_bootstrap_ci.csv"
    if path.exists():
        existing = pd.read_csv(path)
        # Idempotent upsert: drop any prior rows for the same (model, estimator)
        # pairs being written, so re-runs do not create duplicate CI rows.
        if {"model", "estimator"}.issubset(existing.columns) and len(new_df):
            keys = set(zip(new_df["model"].astype(str), new_df["estimator"].astype(str)))
            keep = ~existing.apply(
                lambda r: (str(r.get("model")), str(r.get("estimator"))) in keys,
                axis=1,
            )
            existing = existing[keep]
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(path, index=False)
    else:
        new_df.to_csv(path, index=False)
    return path


def save_subgroup_results(
    subgroup_results: list[dict],
    model_id: str,
    output_dir: str | Path,
) -> Path:
    """Save subgroup-stratified OPE results."""
    df = pd.DataFrame(subgroup_results)
    path = Path(output_dir) / f"{model_id}_subgroup_results.csv"
    df.to_csv(path, index=False)
    return path


def save_trimming_sensitivity(
    trim_results: list[dict],
    model_id: str,
    output_dir: str | Path,
) -> Path:
    """Save trimming sensitivity analysis results."""
    df = pd.DataFrame(trim_results)
    path = Path(output_dir) / f"{model_id}_trimming_sensitivity.csv"
    df.to_csv(path, index=False)
    return path


def generate_summary_report(
    model_id: str,
    model_label: str,
    results: dict,
    bootstrap_results: dict,
    subgroup_results: list[dict],
    training_info: dict,
    output_dir: str | Path,
) -> Path:
    """Generate human-readable summary report (matching 75a/75c format)."""
    lines = []

    def w(text=""):
        lines.append(text)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    w("=" * 72)
    w(f"{model_id} — {model_label} (MIMIC-IV)")
    w("=" * 72)
    w(f"Generated: {timestamp}")
    w(f"Protocol: v3.1, Step {model_id[:3]}")
    w()

    # Training info
    w("TRAINING")
    w("─" * 8)
    for k, v in training_info.items():
        w(f"  {k}: {v}")
    w()

    # Policy evaluation
    w("POLICY EVALUATION")
    w("─" * 17)
    for est_name, est_result in results.items():
        v_obs = est_result.get("v_observed", 0)
        v_rule = est_result.get("v_rule", 0)
        ate = est_result.get("ate", 0)
        ess = est_result.get("ess", "")
        ess_str = f", ESS={ess:.0f}" if isinstance(ess, (int, float)) and ess else ""
        w(f"  {est_name:6s}: V(obs)={v_obs:.4f}, V(rule)={v_rule:.4f}, ATE={ate:+.4f}{ess_str}")

        # Add bootstrap CI if available
        boot = bootstrap_results.get(est_name, {})
        if boot.get("n_success", 0) > 0:
            w(f"         CI: [{boot['ate_ci_lo']:.4f}, {boot['ate_ci_hi']:.4f}] (B={boot['n_success']})")
    w()

    # Subgroup results
    if subgroup_results:
        w("SUBGROUP ANALYSIS")
        w("─" * 17)
        for sg in subgroup_results:
            grp = f"{sg.get('stratifier', '')}={sg.get('group', '')}"
            n = sg.get("n_stays", 0)
            ate = sg.get("ate", 0)
            w(f"  {grp:45s}: n={n:>5}, ATE={ate:+.4f}")
        w()

    # Comparison with 75a
    w("COMPARISON WITH 75a CLASSICAL BASELINE")
    w("─" * 38)
    w("  75a ATE(WIS): -0.0487 [-0.0568, -0.0405]")
    for est_name, est_result in results.items():
        ate = est_result.get("ate", 0)
        boot = bootstrap_results.get(est_name, {})
        if boot.get("n_success", 0) > 0:
            w(f"  {model_id} ATE({est_name}): {ate:+.4f} [{boot['ate_ci_lo']:.4f}, {boot['ate_ci_hi']:.4f}]")
        else:
            w(f"  {model_id} ATE({est_name}): {ate:+.4f}")
    w()

    w("NEXT STEPS")
    w("─" * 10)
    w("  See 80f_core_model_comparison for cross-model analysis.")
    w()

    text = "\n".join(lines)
    path = Path(output_dir) / f"{model_id}_summary.txt"
    with open(path, "w") as f:
        f.write(text)
    return path


def save_structured_yaml(
    model_id: str,
    results: dict,
    bootstrap_results: dict,
    training_info: dict,
    output_dir: str | Path,
) -> Path:
    """Save machine-readable structured summary YAML."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    summary = {
        "step": model_id,
        "date": timestamp,
        "protocol_version": "v3.1",
        "database": "MIMIC-IV",
        "training": training_info,
        "results": {},
    }

    for est_name, est_result in results.items():
        entry = {
            "v_observed": _py(est_result.get("v_observed")),
            "v_rule": _py(est_result.get("v_rule")),
            "ate": _py(est_result.get("ate")),
        }
        boot = bootstrap_results.get(est_name, {})
        if boot.get("n_success", 0) > 0:
            entry["ate_ci"] = [_py(boot["ate_ci_lo"]), _py(boot["ate_ci_hi"])]
            entry["n_bootstrap"] = boot["n_success"]
        summary["results"][est_name] = entry

    path = Path(output_dir) / f"{model_id}_summary.yaml"
    with open(path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
    return path


def _py(x):
    """Convert numpy scalar to native Python type."""
    if x is None:
        return None
    if hasattr(x, "item"):
        return x.item()
    if isinstance(x, float):
        return round(x, 6)
    return x
