#!/usr/bin/env python3
"""
70g_primary_vs_secondary_model_registry — Freeze Step C model roles.

Purpose
-------
Lock the model-role registry BEFORE inspecting any Step C results.
This prevents post-hoc cherry-picking and ensures the primary pair,
confirmatory models, and sensitivity methods are declared a priori.

Model hierarchy (from 00f global config, locked):
  Primary pair:
    - 75a_dynamic_msm_iptw (classical)
    - 80c_gtransformer (deep)
  Confirmatory:
    - 80a_gnet
    - 80b_actin
  Legacy ablation:
    - 80d_ct_legacy
  Optional:
    - 80e_mamba_cdsp
  Sensitivity/evaluation:
    - 85a_edq
    - 85b_seqdeconf
    - 85j_informative_missingness_sensitivity

Protocol reference: Step 70g (v3.1), SAP 03f

Inputs
------
- configs/00f_global_config_registry.yaml
- configs/70a_policy_objects_manifest_v1.json
- configs/70d_rewards_metrics_v1.yaml
- configs/70e_ope_config_v1.yaml
- configs/70f_trimming_grid_v1.yaml

Outputs
-------
- configs/70g_model_registry_v1.yaml
- reports/70g_model_registry_summary.txt

Compute: C1
"""

from __future__ import annotations

import json
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
    print("70g — Freeze Step C Model Registry")
    print("=" * 72)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 1. Pull model hierarchy from global config ────────────────────────
    print("\n[1] Loading model hierarchy from global config...")
    sc = proj_cfg.get("step_c_models", {})

    primary_pair = sc.get("primary_pair", [])
    confirmatory = sc.get("confirmatory", [])
    legacy = sc.get("legacy_ablation", [])
    optional = sc.get("optional", [])
    sensitivity = sc.get("sensitivity_evaluation", [])

    print(f"  Primary pair:  {[m['id'] for m in primary_pair]}")
    print(f"  Confirmatory:  {[m['id'] for m in confirmatory]}")
    print(f"  Legacy:        {[m['id'] for m in legacy]}")
    print(f"  Optional:      {[m['id'] for m in optional]}")
    print(f"  Sensitivity:   {[m['id'] for m in sensitivity]}")

    # ── 2. Validate locked constraints ────────────────────────────────────
    print("\n[2] Validating locked constraints...")
    pp_ids = {m["id"] for m in primary_pair}
    assert "75a_dynamic_msm_iptw" in pp_ids, "Missing 75a in primary pair"
    assert "80c_gtransformer" in pp_ids, "Missing 80c in primary pair"
    print("  Primary pair constraint: PASS")

    # ── 3. Cross-reference available configs ──────────────────────────────
    print("\n[3] Cross-referencing Step 70 configs...")
    config_deps = {
        "70a_policy_objects": "70a_policy_objects_manifest_v1.json",
        "70b_behavior_policy": "70b_action_merge_map.yaml",
        "70c_rule_policy": "70c_rule_policy_v1.yaml",
        "70d_rewards_metrics": "70d_rewards_metrics_v1.yaml",
        "70e_ope_config": "70e_ope_config_v1.yaml",
        "70f_trimming_grid": "70f_trimming_grid_v1.yaml",
    }
    config_status = {}
    for name, fname in config_deps.items():
        exists = (configs_dir / fname).exists()
        config_status[name] = "present" if exists else "missing"
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {fname}")

    # ── 4. Build comprehensive registry ───────────────────────────────────
    print("\n[4] Building model registry...")

    registry = {
        "step": "70g_primary_vs_secondary_model_registry",
        "date": ts,
        "protocol_version": "v3.1",
        "lock_status": "LOCKED — do NOT modify after Step 75 begins",
        "models": {
            "primary_pair": [
                {
                    "id": m["id"],
                    "label": m["label"],
                    "role": m.get("role", "primary"),
                    "ope_methods": ["WIS", "DR"] if "msm" in m["id"]
                                   else ["DR", "FQE", "DM"],
                    "subgroup_analysis": True,
                    "trimming_grid": True,
                    "phenotype_stratification": [
                        "rule_phenotype_4class",
                        "rule_phenotype_2class",
                        "early_data_driven_k2",
                    ],
                    "coarse2_sensitivity": True,
                }
                for m in primary_pair
            ],
            "confirmatory": [
                {
                    "id": m["id"],
                    "label": m["label"],
                    "role": "confirmatory",
                    "ope_methods": ["DR", "FQE"],
                    "subgroup_analysis": True,
                    "trimming_grid": True,
                    "phenotype_stratification": [
                        "rule_phenotype_4class",
                        "rule_phenotype_2class",
                    ],
                    "coarse2_sensitivity": False,
                }
                for m in confirmatory
            ],
            "legacy_ablation": [
                {
                    "id": m["id"],
                    "label": m["label"],
                    "role": "legacy_ablation",
                    "ope_methods": ["FQE"],
                    "subgroup_analysis": False,
                    "trimming_grid": True,
                    "phenotype_stratification": [],
                    "coarse2_sensitivity": False,
                }
                for m in legacy
            ],
            "optional": [
                {
                    "id": m["id"],
                    "label": m["label"],
                    "role": "optional",
                    "ope_methods": ["FQE"],
                    "subgroup_analysis": False,
                    "trimming_grid": False,
                    "phenotype_stratification": [],
                    "coarse2_sensitivity": False,
                }
                for m in optional
            ],
            "sensitivity_evaluation": [
                {
                    "id": m["id"],
                    "label": m["label"],
                    "role": "sensitivity",
                    "ope_methods": ["method-specific"],
                    "subgroup_analysis": False,
                    "trimming_grid": False,
                    "phenotype_stratification": [],
                    "coarse2_sensitivity": False,
                }
                for m in sensitivity
            ],
        },
        "execution_order": {
            "phase_1": [
                "75a_dynamic_msm_iptw",
                "80c_gtransformer",
            ],
            "phase_2": [
                "80a_gnet",
                "80b_actin",
            ],
            "phase_3": [
                "80d_ct_legacy",
                "80e_mamba_cdsp",
            ],
            "phase_4_sensitivity": [
                "85a_edq",
                "85b_seqdeconf",
                "85j_informative_missingness_sensitivity",
            ],
        },
        "interpretation_rules": {
            "primary_conclusion": (
                "Based on primary pair (75a + 80c). Both must agree "
                "for a primary claim. If they disagree, report both "
                "and discuss discrepancy."
            ),
            "confirmatory": (
                "80a + 80b provide robustness. Directional agreement "
                "strengthens conclusion; disagreement flags model "
                "sensitivity."
            ),
            "legacy_ablation": (
                "80d provides continuity with existing DTR literature. "
                "Not used for primary conclusion."
            ),
            "sensitivity": (
                "85a-85j test specific assumptions. Flagged results "
                "qualify, not invalidate, primary conclusions."
            ),
        },
        "config_dependencies": config_status,
    }

    reg_path = configs_dir / "70g_model_registry_v1.yaml"
    with open(reg_path, "w") as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)
    print(f"  Saved: {reg_path}")

    # ── 5. Summary report ─────────────────────────────────────────────────
    report_path = reports_dir / "70g_model_registry_summary.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    all_models = (
        primary_pair + confirmatory + legacy + optional + sensitivity
    )
    lines = [
        "=" * 72,
        "70g — Step C Model Registry (MIMIC-IV)",
        "=" * 72,
        f"Generated: {ts}",
        f"Lock status: LOCKED",
        "",
        "MODEL HIERARCHY",
        "───────────────",
        "",
        "  Primary pair (conclusion-bearing):",
    ]
    for m in primary_pair:
        lines.append(f"    {m['id']:35s} {m['label']}")
    lines.append("\n  Confirmatory:")
    for m in confirmatory:
        lines.append(f"    {m['id']:35s} {m['label']}")
    lines.append("\n  Legacy ablation:")
    for m in legacy:
        lines.append(f"    {m['id']:35s} {m['label']}")
    lines.append("\n  Optional:")
    for m in optional:
        lines.append(f"    {m['id']:35s} {m['label']}")
    lines.append("\n  Sensitivity:")
    for m in sensitivity:
        lines.append(f"    {m['id']:35s} {m['label']}")

    lines.extend([
        "",
        "EXECUTION ORDER",
        "───────────────",
        "  Phase 1: 75a + 80c (primary pair)",
        "  Phase 2: 80a + 80b (confirmatory)",
        "  Phase 3: 80d + 80e (legacy + optional)",
        "  Phase 4: 85a + 85b + 85j (sensitivity)",
        "",
        "CONFIG DEPENDENCIES",
        "───────────────────",
    ])
    for name, status in config_status.items():
        sym = "✓" if status == "present" else "✗"
        lines.append(f"  {sym} {name}")

    lines.extend([
        "",
        "INTERPRETATION RULES",
        "────────────────────",
        "  Primary: 75a + 80c must agree for primary claim",
        "  Confirmatory: 80a + 80b for robustness",
        "  Sensitivity: qualifies but does not invalidate primary",
        "",
        "STEP 70 COMPLETE",
        "────────────────",
        "  All 7 sub-steps (70a-70g) define the policy evaluation framework.",
        "  Next: Step 75a (dynamic-MSM/IPTW primary estimation).",
    ])

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Report: {report_path}")
    print("\n70g COMPLETE — Step 70 policy framework fully defined.")


if __name__ == "__main__":
    main()
