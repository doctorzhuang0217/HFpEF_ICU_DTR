#!/usr/bin/env python3
"""
70e_define_ope_config — Freeze Off-Policy Evaluation stack settings.

Purpose
-------
Pre-declare the OPE methods, their configurations, and the evaluation
protocol for all Step C models. This prevents post-hoc cherry-picking
of evaluation approaches.

OPE stack:
  1. IPTW / WIS — importance-sampling based (for 75a MSM)
  2. Doubly-Robust (DR) — combines IPTW with outcome model
  3. FQE — Fitted Q-Evaluation (for deep RL policies 80a-80e)
  4. Direct Method (DM) — g-computation / outcome regression only

Protocol reference: Step 70e (v3.1), SAP 03f

Inputs
------
- configs/70a_policy_objects_manifest_v1.json
- configs/70d_rewards_metrics_v1.yaml

Outputs
-------
- configs/70e_ope_config_v1.yaml
- reports/70e_ope_config_summary.txt

Compute: C1
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path


def main() -> None:
    proj_cfg = load_config()
    configs_dir = resolve_path(proj_cfg, "configs")
    reports_dir = resolve_path(proj_cfg, "reports")

    print("=" * 72)
    print("70e — Define OPE Configuration")
    print("=" * 72)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ope_cfg = {
        "step": "70e_define_ope_config",
        "date": ts,
        "protocol_version": "v3.1",
        "ope_methods": {
            "iptw_wis": {
                "name": "Weighted Importance Sampling",
                "abbreviation": "WIS",
                "applicable_to": ["75a_dynamic_msm_iptw"],
                "weight_source": "70b behavior policy propensities",
                "normalization": "self-normalized (WIS)",
                "variance_estimator": "bootstrap (B=500)",
                "confidence_level": 0.95,
            },
            "doubly_robust": {
                "name": "Doubly-Robust Estimator",
                "abbreviation": "DR",
                "applicable_to": [
                    "75a_dynamic_msm_iptw",
                    "80a_gnet",
                    "80c_gtransformer",
                ],
                "components": {
                    "propensity": "70b behavior policy",
                    "outcome_model": "fitted within each Step C method",
                },
                "variance_estimator": "influence-function + bootstrap",
                "confidence_level": 0.95,
            },
            "fqe": {
                "name": "Fitted Q-Evaluation",
                "abbreviation": "FQE",
                "applicable_to": [
                    "80a_gnet", "80b_actin", "80c_gtransformer",
                    "80d_ct_legacy", "80e_mamba_cdsp",
                ],
                "q_network": {
                    "architecture": "2-layer MLP (256, 128)",
                    "activation": "ReLU",
                    "training_epochs": 100,
                    "learning_rate": 1e-3,
                    "batch_size": 256,
                },
                "n_iterations": 50,
                "convergence_tol": 1e-4,
            },
            "direct_method": {
                "name": "Direct Method (G-computation)",
                "abbreviation": "DM",
                "applicable_to": ["80a_gnet", "80c_gtransformer"],
                "description": (
                    "Outcome model only, no importance weighting. "
                    "Valid under correct outcome model specification."
                ),
            },
        },
        "evaluation_protocol": {
            "primary_estimand": "ATE (28-day mortality)",
            "comparison_policies": [
                {
                    "name": "observed_clinician",
                    "type": "behavior_policy",
                    "source": "70b propensities",
                },
                {
                    "name": "rule_policy_v1",
                    "type": "rule_based",
                    "source": "70c rule policy",
                },
            ],
            "subgroup_analyses": [
                {
                    "stratifier": "rule_phenotype_2class",
                    "groups": ["shock_predominant", "non_shock"],
                    "note": "Primary subgroup analysis",
                },
                {
                    "stratifier": "hf_subtype",
                    "groups": ["hfpef", "hfref", "hfmref", "no_hf"],
                    "note": "HF subtype-specific policy evaluation",
                },
            ],
            "sensitivity_analyses": [
                {
                    "name": "phenotype_sensitivity",
                    "stratifier": "early_data_driven_k2",
                    "note": "Repeat with data-driven binary phenotype",
                },
                {
                    "name": "trimming_sensitivity",
                    "reference": "70f_trimming_grid_v1.yaml",
                },
            ],
            "bootstrap": {
                "n_resamples": 500,
                "seed": 7777,
                "ci_method": "percentile",
                "confidence_level": 0.95,
            },
        },
        "reporting_requirements": {
            "per_method": [
                "point_estimate",
                "ci_lower", "ci_upper",
                "n_effective",
                "weight_diagnostics",
            ],
            "comparison_table": (
                "All methods × all policies × all subgroups in a single table"
            ),
            "visualization": [
                "forest_plot_by_method",
                "policy_value_comparison",
                "weight_distribution_histogram",
            ],
        },
    }

    cfg_path = configs_dir / "70e_ope_config_v1.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(ope_cfg, f, default_flow_style=False, sort_keys=False)
    print(f"\n  Saved: {cfg_path}")

    # ── Summary report ────────────────────────────────────────────────────
    report_path = reports_dir / "70e_ope_config_summary.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "=" * 72,
        "70e — OPE Configuration (MIMIC-IV)",
        "=" * 72,
        f"Generated: {ts}",
        "",
        "OPE METHODS",
        "───────────",
        "  1. WIS  — Weighted Importance Sampling (for 75a MSM)",
        "  2. DR   — Doubly-Robust (75a, 80a, 80c)",
        "  3. FQE  — Fitted Q-Evaluation (80a-80e deep methods)",
        "  4. DM   — Direct Method / G-computation (80a, 80c)",
        "",
        "EVALUATION PROTOCOL",
        "───────────────────",
        "  Primary estimand: ATE (28-day mortality)",
        "  Comparison: evaluated policy vs observed clinician behavior",
        "  Additional: rule_policy_v1 as interpretable baseline",
        "",
        "SUBGROUP ANALYSES",
        "─────────────────",
        "  Primary:     rule_phenotype_2class (shock vs non-shock)",
        "  HF subtype:  hfpef / hfref / hfmref / no_hf",
        "  Sensitivity: early_data_driven_k2",
        "",
        "BOOTSTRAP",
        "─────────",
        "  Resamples: 500, seed: 7777",
        "  CI: percentile, 95%",
        "",
        "NEXT STEPS",
        "──────────",
        "  70f: define trimming and support grid",
        "  70g: freeze model registry",
    ]

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Report: {report_path}")
    print("\n70e COMPLETE.")


if __name__ == "__main__":
    main()
