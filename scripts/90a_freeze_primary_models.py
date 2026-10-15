#!/usr/bin/env python3
"""90a — Freeze Primary Models for External Validation

Protocol: v3.1, Step 90a
Compute: C1

Purpose
-------
Create a frozen model registry (primary_model_registry.json) that captures
all information needed to reproduce and transport MIMIC-trained models to
external datasets (eICU, HiRID).

This must be run BEFORE any external validation step (90b/90c).

Frozen artifacts:
  1. Model checkpoint paths + integrity hashes (SHA-256)
  2. Architecture configs + hyperparameters
  3. Normalization parameters (45c — MIMIC z-score params applied to external data)
  4. Action space encoding + merge map
  5. Rule policy definition
  6. Feature column lists (temporal + static)
  7. Training metadata (best epoch, val loss, etc.)
  8. OPE configuration

Inputs
------
- models/80c_gtransformer_best.pt  (+ rollout variants)
- models/80a_gnet_best.pt
- models/80b_actin_best.pt
- models/80d_ct_legacy_best.pt
- models/70b_behavior_policy_coarse3.pkl
- configs/80_step_c_training_v1.yaml
- configs/45c_normalization_params_v1.yaml
- configs/70b_action_merge_map.yaml
- configs/70c_rule_policy_v1.yaml
- configs/70g_model_registry_v1.yaml
- configs/action_space_v1.yaml
- data/interim/45c_state_tensors_v1_mimic.parquet  (for feature columns)

Outputs
-------
- configs/90a_primary_model_registry.json
- reports/90a_freeze_primary_models_summary.txt
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# ── Project root ──
ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(ROOT / "src"))

from hfpef_dtr.models.dataset import discover_feature_columns


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):  # 1 MB chunks
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hash of bytes."""
    return hashlib.sha256(data).hexdigest()


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_checkpoint_metadata(ckpt_path: Path) -> dict:
    """Extract metadata from a PyTorch checkpoint without loading full weights."""
    if not ckpt_path.exists():
        return {"exists": False, "path": str(ckpt_path)}

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = {
        "exists": True,
        "path": str(ckpt_path.relative_to(ROOT)),
        "sha256": sha256_file(ckpt_path),
        "file_size_mb": round(ckpt_path.stat().st_size / 1e6, 2),
    }
    # Extract training metadata if available
    if "epoch" in ckpt:
        meta["best_epoch"] = ckpt["epoch"]
    if "val_loss" in ckpt:
        meta["best_val_loss"] = round(float(ckpt["val_loss"]), 6)
    if "config" in ckpt:
        meta["embedded_config"] = True
    # Count parameters
    if "model_state_dict" in ckpt:
        n_params = sum(v.numel() for v in ckpt["model_state_dict"].values())
        meta["n_parameters"] = n_params
    return meta


def load_sklearn_metadata(pkl_path: Path) -> dict:
    """Extract metadata from a sklearn behavior policy pickle."""
    if not pkl_path.exists():
        return {"exists": False, "path": str(pkl_path)}

    meta = {
        "exists": True,
        "path": str(pkl_path.relative_to(ROOT)),
        "sha256": sha256_file(pkl_path),
        "file_size_mb": round(pkl_path.stat().st_size / 1e6, 2),
    }
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    if "model" in obj:
        clf = obj["model"]
        meta["model_class"] = type(clf).__name__
        meta["n_classes"] = len(clf.classes_)
        meta["classes"] = clf.classes_.tolist()
        meta["C"] = clf.C
    if "feature_cols" in obj:
        meta["n_features"] = len(obj["feature_cols"])
        meta["feature_cols"] = obj["feature_cols"]
    if "merge_map" in obj:
        meta["merge_map"] = {str(k): int(v) for k, v in obj["merge_map"].items()}
    return meta


def main():
    print("=" * 72)
    print("90a — Freeze Primary Models for External Validation")
    print("=" * 72)
    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    registry = {
        "step": "90a_freeze_primary_models",
        "date": ts,
        "protocol_version": "v3.1",
        "database": "MIMIC-IV",
        "purpose": (
            "Frozen model registry for external validation. "
            "All external datasets must use these exact configs + checkpoints."
        ),
    }

    # ── 1. Training configuration ────────────────────────────────────────
    print("\n[1] Loading training configuration...")
    config_path = ROOT / "configs" / "80_step_c_training_v1.yaml"
    config = load_yaml(config_path)
    registry["training_config"] = {
        "path": str(config_path.relative_to(ROOT)),
        "sha256": sha256_file(config_path),
        "grid": config["grid"],
        "action": config["action"],
        "split": config["split"],
        "reward": config["reward"],
    }
    print(f"  Config: {config_path.name}")

    # ── 2. Normalization parameters (CRITICAL for external data) ─────────
    print("\n[2] Loading normalization parameters...")
    norm_path = ROOT / "configs" / "45c_normalization_params_v1.yaml"
    norm_cfg = load_yaml(norm_path)
    registry["normalization"] = {
        "path": str(norm_path.relative_to(ROOT)),
        "sha256": sha256_file(norm_path),
        "method": norm_cfg.get("normalization_method", "z-score"),
        "imputation": norm_cfg.get("imputation_method", "LOCF + population_median"),
        "n_active_vars": norm_cfg.get("n_active_vars"),
        "active_vars": norm_cfg.get("active_vars"),
        "dropped_vars": norm_cfg.get("dropped_vars"),
        "params": norm_cfg.get("params"),
    }
    print(f"  {norm_cfg['n_active_vars']} active variables, method={norm_cfg['normalization_method']}")
    print(f"  CRITICAL: External datasets must apply MIMIC normalization params, not re-estimate!")

    # ── 3. Action space + merge map ──────────────────────────────────────
    print("\n[3] Loading action space configuration...")
    action_path = ROOT / "configs" / "action_space_v1.yaml"
    merge_path = ROOT / "configs" / "70b_action_merge_map.yaml"
    rule_path = ROOT / "configs" / "70c_rule_policy_v1.yaml"

    action_cfg = load_yaml(action_path) if action_path.exists() else None
    merge_cfg = load_yaml(merge_path) if merge_path.exists() else None
    rule_cfg = load_yaml(rule_path)

    registry["action_space"] = {
        "encoding": config["action"]["encoding"],
        "scheme": config["action"]["scheme"],
        "n_raw_cells": config["action"]["n_raw_cells"],
        "n_merged_cells": config["action"]["n_merged_cells"],
        "valid_merged_indices": config["action"]["valid_merged_indices"],
        "merge_map": merge_cfg.get("merge_map") if merge_cfg else {},
        "merge_map_sha256": sha256_file(merge_path) if merge_path.exists() else None,
    }
    registry["rule_policy"] = {
        "path": str(rule_path.relative_to(ROOT)),
        "sha256": sha256_file(rule_path),
        "rules": rule_cfg.get("rules", []),
        "thresholds": rule_cfg.get("thresholds", {}),
        "normalization_used": rule_cfg.get("normalization_used", {}),
    }
    print(f"  Action scheme: {config['action']['scheme']}, "
          f"{config['action']['n_merged_cells']} merged cells")

    # ── 4. Feature columns from state tensors ────────────────────────────
    print("\n[4] Discovering feature columns...")
    state_path = ROOT / config["data"]["state_tensors"]
    if state_path.exists():
        # Read just the schema (first few rows)
        state_sample = pd.read_parquet(state_path, columns=None)
        # Use only first 100 rows for column discovery
        if len(state_sample) > 100:
            state_sample = state_sample.head(100)
        temporal_cols, static_cols = discover_feature_columns(state_sample)
        registry["features"] = {
            "temporal_cols": temporal_cols,
            "static_cols": static_cols,
            "n_temporal": len(temporal_cols),
            "n_static": len(static_cols),
            "state_tensors_path": str(state_path.relative_to(ROOT)),
        }
        print(f"  Temporal: {len(temporal_cols)}, Static: {len(static_cols)}")
    else:
        print(f"  WARNING: State tensors not found at {state_path}")
        registry["features"] = {"warning": "state tensors not found"}

    # ── 5. Model checkpoints ─────────────────────────────────────────────
    print("\n[5] Registering model checkpoints...")
    models_dir = ROOT / "models"

    # Define expected checkpoints
    checkpoint_specs = {
        "80c_gtransformer": {
            "base": "80c_gtransformer_best.pt",
            "variants": [
                "80c_gtransformer_rollout_ow1_best.pt",
            ],
            "role": "primary_deep",
            "ope_methods": ["DM", "DR", "FQE"],
        },
        "80a_gnet": {
            "base": "80a_gnet_best.pt",
            "variants": [],
            "role": "confirmatory",
            "ope_methods": ["DM", "DR", "FQE"],
        },
        "80b_actin": {
            "base": "80b_actin_best.pt",
            "variants": [],
            "role": "confirmatory",
            "ope_methods": ["DR", "FQE"],
        },
        "80d_ct_legacy": {
            "base": "80d_ct_legacy_best.pt",
            "variants": [],
            "role": "legacy_ablation",
            "ope_methods": ["FQE"],
        },
    }

    registry["checkpoints"] = {}
    for model_id, spec in checkpoint_specs.items():
        print(f"  {model_id}:")
        base_path = models_dir / spec["base"]
        meta = load_checkpoint_metadata(base_path)
        meta["role"] = spec["role"]
        meta["ope_methods"] = spec["ope_methods"]

        # Check for architecture config
        model_cfg_key = model_id.replace("80c_", "80c_").replace("80a_", "80a_")
        if model_id in config.get("models", {}):
            meta["architecture_config"] = config["models"][model_id]

        if meta["exists"]:
            print(f"    FOUND: {spec['base']} ({meta['file_size_mb']} MB, "
                  f"{meta.get('n_parameters', '?'):,} params, "
                  f"epoch={meta.get('best_epoch', '?')}, "
                  f"val_loss={meta.get('best_val_loss', '?')})")
        else:
            print(f"    MISSING: {spec['base']}")

        # Check variants
        variant_meta = {}
        for vname in spec["variants"]:
            vpath = models_dir / vname
            vmeta = load_checkpoint_metadata(vpath)
            variant_meta[vname] = vmeta
            status = "FOUND" if vmeta["exists"] else "MISSING"
            print(f"    variant {status}: {vname}")

        meta["variants"] = variant_meta
        registry["checkpoints"][model_id] = meta

    # ── 6. Behavior policy ───────────────────────────────────────────────
    print("\n[6] Registering behavior policy...")
    bp_path = models_dir / "70b_behavior_policy_coarse3.pkl"
    bp_meta = load_sklearn_metadata(bp_path)
    if bp_meta["exists"]:
        print(f"  FOUND: {bp_path.name} ({bp_meta['file_size_mb']} MB, "
              f"{bp_meta.get('n_features', '?')} features, "
              f"{bp_meta.get('n_classes', '?')} classes)")
    else:
        print(f"  MISSING: {bp_path.name}")
    registry["behavior_policy"] = bp_meta

    # ── 7. OPE configuration ─────────────────────────────────────────────
    print("\n[7] Recording OPE configuration...")
    ope_config_path = ROOT / "configs" / "70e_ope_config_v1.yaml"
    trim_config_path = ROOT / "configs" / "70f_trimming_grid_v1.yaml"

    registry["ope_config"] = {
        "path": str(ope_config_path.relative_to(ROOT)),
        "sha256": sha256_file(ope_config_path) if ope_config_path.exists() else None,
        "bootstrap": config["ope"]["bootstrap"],
        "fqe": config["ope"]["fqe"],
        "subgroup_stratifiers": config["ope"]["subgroup_stratifiers"],
        "trimming_brackets": config["ope"]["trimming_brackets"],
    }
    print(f"  Bootstrap: B={config['ope']['bootstrap']['n_resamples']}, "
          f"seed={config['ope']['bootstrap']['seed']}")

    # ── 8. MIMIC reference results (for comparison) ──────────────────────
    print("\n[8] Loading MIMIC reference results...")
    mimic_ref = {}

    # Try to load primary model results
    for model_id in ["80c_gtransformer", "80c_gtransformer_rollout_ow1"]:
        pv_path = ROOT / "tables" / f"{model_id}_policy_values.csv"
        ci_path = ROOT / "tables" / f"{model_id}_bootstrap_ci.csv"
        sg_path = ROOT / "tables" / f"{model_id}_subgroup_results.csv"

        ref = {}
        if pv_path.exists():
            pv = pd.read_csv(pv_path)
            ref["policy_values"] = pv.to_dict(orient="records")
        if ci_path.exists():
            ci = pd.read_csv(ci_path)
            ref["bootstrap_ci"] = ci.to_dict(orient="records")
        if sg_path.exists():
            sg = pd.read_csv(sg_path)
            ref["subgroup"] = sg.to_dict(orient="records")

        if ref:
            mimic_ref[model_id] = ref
            print(f"  {model_id}: loaded {', '.join(ref.keys())}")
        else:
            print(f"  {model_id}: no result files found")

    registry["mimic_reference_results"] = mimic_ref

    # ── 9. External validation instructions ──────────────────────────────
    registry["external_validation_instructions"] = {
        "normalization": (
            "Apply MIMIC normalization params to external data. "
            "Do NOT re-estimate mean/std from external data. "
            "Use params from configs/45c_normalization_params_v1.yaml."
        ),
        "imputation": (
            "Apply same LOCF + population_median strategy. "
            "Use MIMIC population medians from normalization params, "
            "NOT external dataset medians."
        ),
        "action_space": (
            "Use same action encoding (fluid*6 + vaso*2 + diuretic). "
            "Apply same merge map. "
            "Thresholds for treatment levels must match MIMIC definitions."
        ),
        "rule_policy": (
            "Apply same rule policy thresholds (MAP<65, lactate>2, etc.) "
            "using RAW (un-normalized) clinical values. "
            "Normalization info in rule_policy.normalization_used is for "
            "converting z-scores back to raw if needed."
        ),
        "model_loading": (
            "Load MIMIC-trained checkpoints and set model.eval(). "
            "Do NOT fine-tune on external data. "
            "Temporal/static feature dimensions must match exactly."
        ),
        "behavior_policy": (
            "Fit a NEW behavior policy on external data "
            "(same model class: LogisticRegression, same features). "
            "This captures external clinician treatment patterns. "
            "MIMIC behavior policy is NOT transported."
        ),
        "splits": (
            "External data is used as a FULL test set (no train/val split). "
            "The models are frozen from MIMIC training."
        ),
        "ascertainment": (
            "See docs/protocol/29d_external_ef_ascertainment_plan.md "
            "for per-database HFpEF ascertainment strategy. "
            "See docs/protocol/29e_dataset_role_downgrade_rules.md "
            "for sample size and positivity triggers."
        ),
    }

    # ── 10. Verification summary ─────────────────────────────────────────
    print("\n[9] Verification summary...")
    n_models_found = sum(
        1 for m in registry["checkpoints"].values() if m.get("exists", False)
    )
    n_models_expected = len(checkpoint_specs)
    bp_found = registry["behavior_policy"].get("exists", False)
    norm_found = norm_path.exists()

    registry["verification"] = {
        "models_found": n_models_found,
        "models_expected": n_models_expected,
        "behavior_policy_found": bp_found,
        "normalization_found": norm_found,
        "all_critical_artifacts_present": (
            n_models_found >= 2  # At least primary pair
            and norm_found
        ),
        "warnings": [],
    }

    if n_models_found < n_models_expected:
        missing = [
            k for k, v in registry["checkpoints"].items()
            if not v.get("exists", False)
        ]
        w = f"Missing {n_models_expected - n_models_found} model checkpoint(s): {missing}"
        registry["verification"]["warnings"].append(w)
        print(f"  WARNING: {w}")

    if not bp_found:
        w = "Behavior policy checkpoint not found"
        registry["verification"]["warnings"].append(w)
        print(f"  WARNING: {w}")

    # ── Save registry ────────────────────────────────────────────────────
    print("\n[10] Saving registry...")
    configs_dir = ROOT / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    registry_path = configs_dir / "90a_primary_model_registry.json"

    # Convert numpy types for JSON serialization
    def _json_default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2, default=_json_default)
    print(f"  Saved: {registry_path}")

    # ── Summary report ───────────────────────────────────────────────────
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "90a_freeze_primary_models_summary.txt"

    lines = [
        "=" * 72,
        "90a — Freeze Primary Models for External Validation",
        "=" * 72,
        f"Generated: {ts}",
        f"Protocol: v3.1",
        "",
        "MODEL REGISTRY",
        "──────────────",
    ]

    for model_id, meta in registry["checkpoints"].items():
        status = "FOUND" if meta.get("exists") else "MISSING"
        role = meta.get("role", "unknown")
        params = meta.get("n_parameters", "?")
        epoch = meta.get("best_epoch", "?")
        val_loss = meta.get("best_val_loss", "?")
        lines.append(
            f"  [{status}] {model_id:30s} role={role:20s} "
            f"params={params:>10} epoch={epoch} val_loss={val_loss}"
        )

    bp_status = "FOUND" if bp_found else "MISSING"
    lines.extend([
        "",
        f"  [{bp_status}] behavior_policy (LogisticRegression, coarse_3)",
        "",
        "NORMALIZATION",
        "─────────────",
        f"  Method: {norm_cfg.get('normalization_method')}",
        f"  Active vars: {norm_cfg.get('n_active_vars')} / {len(STATE_VARS_REF)}",
        f"  Dropped: {norm_cfg.get('dropped_vars')}",
        "",
        "ACTION SPACE",
        "────────────",
        f"  Scheme: {config['action']['scheme']}",
        f"  Raw cells: {config['action']['n_raw_cells']}",
        f"  Merged cells: {config['action']['n_merged_cells']}",
        f"  Merge map: {merge_cfg.get('merge_map') if merge_cfg else 'N/A'}",
        "",
        "EXTERNAL VALIDATION REQUIREMENTS",
        "─────────────────────────────────",
        "  1. Apply MIMIC normalization params (do NOT re-estimate)",
        "  2. Apply MIMIC population medians for imputation",
        "  3. Use same action encoding + merge map",
        "  4. Fit NEW behavior policy on external data",
        "  5. Load frozen MIMIC checkpoints (no fine-tuning)",
        "  6. Apply 29d ascertainment + 29e downgrade rules",
        "",
        "VERIFICATION",
        "────────────",
        f"  Models found: {n_models_found}/{n_models_expected}",
        f"  Behavior policy: {bp_status}",
        f"  Normalization: {'FOUND' if norm_found else 'MISSING'}",
    ])

    if registry["verification"]["warnings"]:
        lines.append("")
        lines.append("WARNINGS")
        lines.append("────────")
        for w in registry["verification"]["warnings"]:
            lines.append(f"  - {w}")

    lines.extend([
        "",
        "NEXT STEPS",
        "──────────",
        "  90b: Run eICU external validation with frozen models",
        "  90c: Run HiRID external validation (if eligible)",
    ])

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {report_path}")

    total_time = time.time() - t0
    print()
    print("=" * 72)
    print(f"90a COMPLETE. Registry: {registry_path}")
    print(f"Total time: {total_time:.1f}s")
    print("=" * 72)


# Reference: full original state variable list
STATE_VARS_REF = [
    "map", "sbp", "dbp", "heart_rate", "spo2", "cvp", "resp_rate",
    "temperature", "fio2",
    "lactate", "creatinine", "bun", "potassium", "sodium",
    "bicarbonate", "glucose", "ph",
    "hemoglobin", "wbc", "platelet", "albumin", "bilirubin",
    "troponin", "bnp",
]


if __name__ == "__main__":
    main()
