#!/usr/bin/env python3
"""
70a_define_policy_objects — Shared state / action / policy interface for Step C.

Purpose
-------
Define and validate the common data structures that all Step C methods will
consume: state representation, action encoding, phenotype labels, and a base
Policy class. This script:
  1. Loads frozen artifacts (45c state tensors, 45b action space, 65e phenotype
     labels, 45d outcomes) and validates schema integrity.
  2. Builds a canonical column manifest that downstream scripts import.
  3. Asserts that explanatory-only phenotype labels are excluded from
     stratification inputs.
  4. Emits a policy_objects_manifest_v1.json recording all validated schemas.

Protocol reference: Step 70a (v3.1)

Inputs
------
- data/interim/45c_state_tensors_v1_mimic.parquet
- data/interim/45d_outcomes_absorb_v1_mimic.parquet
- data/processed/phenotype_labels_frozen_v1_mimic.parquet
- configs/action_space_v1.yaml
- configs/phenotype_registry_v1.json  (if synced from the compute cluster)

Outputs
-------
- configs/70a_policy_objects_manifest_v1.json
- reports/70a_policy_objects_summary.txt

Compute: C1
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path


# ── Constants ─────────────────────────────────────────────────────────────
N_BINS = 12
BIN_HOURS = 2
HORIZON_HOURS = 24
N_ACTION_CELLS_COARSE3 = 18

# Phenotype roles that may enter Step C stratification
ALLOWED_STEPC_ROLES = {"primary", "sensitivity"}
# Roles that MUST NOT enter Step C.
FORBIDDEN_STEPC_ROLES = {"explanatory", "reference_only"}


def _load_action_space(configs_dir: Path) -> dict:
    """Load and validate the frozen action space config."""
    path = configs_dir / "action_space_v1.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    assert cfg["joint_cells"] == N_ACTION_CELLS_COARSE3, (
        f"Expected {N_ACTION_CELLS_COARSE3} joint cells, got {cfg['joint_cells']}"
    )
    return cfg


def _discover_state_columns(df: pd.DataFrame) -> dict:
    """Classify 45c state tensor columns into semantic groups."""
    skip = {"stay_id", "bin", "subject_id", "hadm_id"}
    z_cols, mask_cols, tsl_cols, freq_cols, never_cols = [], [], [], [], []
    action_cols, passthrough_cols, static_cols = [], [], []

    for col in sorted(df.columns):
        if col in skip:
            continue
        if col.endswith("_z"):
            z_cols.append(col)
        elif col.endswith("_mask"):
            mask_cols.append(col)
        elif col.endswith("_tsl_h"):
            tsl_cols.append(col)
        elif col.endswith("_freq_24h"):
            freq_cols.append(col)
        elif col.endswith("_never_measured"):
            never_cols.append(col)
        elif col in ("action_idx",):
            action_cols.append(col)
        elif col.startswith("static_") or col in {
            "age", "gender", "admission_type", "hf_subtype",
            "comparator_group", "dominant_syndrome", "reason_for_icu",
            "hemo_gate_passed", "include_in_phase05",
        }:
            static_cols.append(col)
        else:
            passthrough_cols.append(col)

    return {
        "z_scored": z_cols,
        "mask": mask_cols,
        "tsl_h": tsl_cols,
        "freq_24h": freq_cols,
        "never_measured": never_cols,
        "action": action_cols,
        "static": static_cols,
        "passthrough": passthrough_cols,
    }


def _validate_phenotype_roles(
    registry: dict, labels_df: pd.DataFrame
) -> list[str]:
    """Ensure explanatory/reference_only labels have used_in_step_c=False."""
    errors = []
    for lbl_name, lbl_info in registry.get("labels", {}).items():
        role = lbl_info.get("role", "unknown")
        used = lbl_info.get("used_in_step_c", None)
        if role in FORBIDDEN_STEPC_ROLES and used is True:
            errors.append(
                f"CRITICAL: label '{lbl_name}' has role={role} but "
                f"used_in_step_c=True — must be False"
            )
        if role in ALLOWED_STEPC_ROLES and used is not True:
            errors.append(
                f"WARNING: label '{lbl_name}' has role={role} but "
                f"used_in_step_c is not True"
            )
        # Check column exists in frozen labels
        if lbl_name not in labels_df.columns and lbl_name != "stay_id":
            # Try matching the column name pattern
            col_candidates = [
                c for c in labels_df.columns if lbl_name in c
            ]
            if not col_candidates:
                errors.append(
                    f"Label '{lbl_name}' in registry but missing from "
                    f"frozen label file columns: {list(labels_df.columns)}"
                )
    return errors


def main() -> None:
    proj_cfg = load_config()
    interim_dir = resolve_path(proj_cfg, "interim")
    processed_dir = resolve_path(proj_cfg, "processed")
    configs_dir = resolve_path(proj_cfg, "configs")
    reports_dir = resolve_path(proj_cfg, "reports")

    print("=" * 72)
    print("70a — Define Policy Objects (Step C Common Interface)")
    print("=" * 72)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 1. Load frozen artifacts ──────────────────────────────────────────
    print("\n[1] Loading frozen artifacts...")

    state_path = interim_dir / "45c_state_tensors_v1_mimic.parquet"
    state_df = pd.read_parquet(state_path)
    n_obs = len(state_df)
    n_stays_state = state_df["stay_id"].nunique()
    print(f"  State tensors: {n_obs:,} obs, {n_stays_state:,} stays")

    outcomes_path = interim_dir / "45d_outcomes_absorb_v1_mimic.parquet"
    outcomes_df = pd.read_parquet(outcomes_path)
    n_stays_out = outcomes_df["stay_id"].nunique()
    print(f"  Outcomes: {n_stays_out:,} stays")

    labels_path = processed_dir / "phenotype_labels_frozen_v1_mimic.parquet"
    labels_df = pd.read_parquet(labels_path)
    n_stays_lbl = labels_df["stay_id"].nunique()
    print(f"  Phenotype labels: {n_stays_lbl:,} stays, "
          f"{len(labels_df.columns)} columns")

    action_cfg = _load_action_space(configs_dir)
    print(f"  Action space: {action_cfg['primary_scheme']} "
          f"({action_cfg['joint_cells']} cells)")

    # ── 2. Validate stay_id alignment ─────────────────────────────────────
    print("\n[2] Validating stay_id alignment...")
    sids_state = set(state_df["stay_id"].unique())
    sids_out = set(outcomes_df["stay_id"].unique())
    sids_lbl = set(labels_df["stay_id"].unique())

    assert sids_state == sids_out, (
        f"State/outcome stay_id mismatch: "
        f"{len(sids_state - sids_out)} in state only, "
        f"{len(sids_out - sids_state)} in outcomes only"
    )
    assert sids_state == sids_lbl, (
        f"State/label stay_id mismatch: "
        f"{len(sids_state - sids_lbl)} in state only, "
        f"{len(sids_lbl - sids_state)} in labels only"
    )
    n_stays = len(sids_state)
    print(f"  All 3 artifacts aligned: {n_stays:,} stays")

    # ── 3. Discover state column schema ───────────────────────────────────
    print("\n[3] Discovering state column schema...")
    col_groups = _discover_state_columns(state_df)
    for grp, cols in col_groups.items():
        print(f"  {grp}: {len(cols)} columns")

    # Temporal features for model input = z-scored + missingness
    temporal_cols = (
        col_groups["z_scored"]
        + col_groups["mask"]
        + col_groups["tsl_h"]
        + col_groups["freq_24h"]
        + col_groups["never_measured"]
    )
    print(f"  Total temporal features per bin: {len(temporal_cols)}")

    # ── 4. Validate action_idx range ──────────────────────────────────────
    print("\n[4] Validating action_idx...")
    if "action_idx" in state_df.columns:
        aidx_min = int(state_df["action_idx"].min())
        aidx_max = int(state_df["action_idx"].max())
        n_actions_obs = int(state_df["action_idx"].nunique())
        print(f"  action_idx range: [{aidx_min}, {aidx_max}], "
              f"{n_actions_obs} unique values")
        assert aidx_max < N_ACTION_CELLS_COARSE3, (
            f"action_idx max={aidx_max} exceeds joint_cells={N_ACTION_CELLS_COARSE3}"
        )
    else:
        print("  WARNING: action_idx not found in state tensor")

    # ── 5. Validate phenotype registry roles ──────────────────────────────
    print("\n[5] Validating phenotype registry roles...")
    registry_path = configs_dir / "phenotype_registry_v1.json"
    if registry_path.exists():
        with open(registry_path) as f:
            pheno_registry = json.load(f)
        role_errors = _validate_phenotype_roles(pheno_registry, labels_df)
    else:
        # Fall back: reconstruct roles from 65e summary conventions
        print("  WARNING: phenotype_registry_v1.json not found, "
              "reconstructing from label columns")
        pheno_registry = {"labels": {}}
        # Define expected roles based on protocol
        expected_roles = {
            "rule_phenotype_4class": ("primary", True),
            "rule_phenotype_2class": ("primary", True),
            "early_data_driven_k2": ("sensitivity", True),
            "early_data_driven_k3_ref": ("reference_only", False),
            "trajectory_data_driven_k2": ("explanatory", False),
            "gmvae_cluster": ("explanatory", False),
        }
        for col in labels_df.columns:
            if col == "stay_id":
                continue
            role, used = expected_roles.get(col, ("unknown", False))
            pheno_registry["labels"][col] = {
                "role": role,
                "used_in_step_c": used,
                "n_classes": int(labels_df[col].nunique()),
            }
        role_errors = _validate_phenotype_roles(pheno_registry, labels_df)

    for err in role_errors:
        print(f"  {err}")
    critical_errors = [e for e in role_errors if e.startswith("CRITICAL")]
    assert not critical_errors, (
        f"Critical phenotype role violations: {critical_errors}"
    )

    # Identify which labels enter Step C
    stepc_labels = [
        name for name, info in pheno_registry["labels"].items()
        if info.get("used_in_step_c", False)
    ]
    explanatory_labels = [
        name for name, info in pheno_registry["labels"].items()
        if info.get("role") in FORBIDDEN_STEPC_ROLES
    ]
    print(f"  Step C labels: {stepc_labels}")
    print(f"  Explanatory-only labels (excluded): {explanatory_labels}")

    # ── 6. Build manifest ─────────────────────────────────────────────────
    print("\n[6] Building policy objects manifest...")

    manifest = {
        "step": "70a_define_policy_objects",
        "generated": ts,
        "protocol_version": "v3.1",
        "dataset": "mimic",
        "n_stays": n_stays,
        "n_bins": N_BINS,
        "bin_hours": BIN_HOURS,
        "horizon_hours": HORIZON_HOURS,
        "state_schema": {
            "n_obs": n_obs,
            "column_groups": {k: v for k, v in col_groups.items()},
            "n_temporal_features": len(temporal_cols),
            "temporal_cols": temporal_cols,
            "static_cols": col_groups["static"],
        },
        "action_space": {
            "scheme": action_cfg["primary_scheme"],
            "n_cells": action_cfg["joint_cells"],
            "dimensions": {
                dim: {
                    "n_levels": info["n_levels"],
                    "labels": info["labels"],
                }
                for dim, info in action_cfg["action_dimensions"].items()
            },
            "sensitivity_schemes": action_cfg.get("sensitivity_schemes", []),
        },
        "outcome_columns": {
            "primary": "mortality_28d",
            "secondary": ["mortality_7d"],
            "absorbing": "absorbing_state",
            "absorbing_bin": "absorbing_state_bin",
        },
        "phenotype_labels": {
            "step_c": stepc_labels,
            "explanatory_only": explanatory_labels,
            "primary_stratifier": "rule_phenotype_2class",
        },
        "artifacts": {
            "state_tensors": str(state_path),
            "outcomes": str(outcomes_path),
            "phenotype_labels": str(labels_path),
            "action_space_config": str(configs_dir / "action_space_v1.yaml"),
        },
    }

    manifest_path = configs_dir / "70a_policy_objects_manifest_v1.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  Saved: {manifest_path}")

    # ── 7. Summary report ─────────────────────────────────────────────────
    report_path = reports_dir / "70a_policy_objects_summary.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "=" * 72,
        "70a — Policy Objects Manifest (MIMIC-IV)",
        "=" * 72,
        f"Generated: {ts}",
        "",
        "FROZEN ARTIFACTS",
        "────────────────",
        f"  State tensors: {n_obs:,} obs ({n_stays:,} stays × {N_BINS} bins)",
        f"  Outcomes: {n_stays:,} stays",
        f"  Phenotype labels: {n_stays:,} stays, "
        f"{len(labels_df.columns) - 1} labels",
        f"  Action space: {action_cfg['primary_scheme']} "
        f"({N_ACTION_CELLS_COARSE3} cells)",
        "",
        "STATE SCHEMA",
        "────────────",
    ]
    for grp, cols in col_groups.items():
        lines.append(f"  {grp}: {len(cols)}")
    lines.append(f"  Total temporal features per bin: {len(temporal_cols)}")
    lines.extend([
        "",
        "ACTION SPACE",
        "────────────",
    ])
    for dim, info in action_cfg["action_dimensions"].items():
        lines.append(f"  {dim}: {info['n_levels']} levels {info['labels']}")
    lines.append(f"  Joint cells: {N_ACTION_CELLS_COARSE3}")
    lines.extend([
        "",
        "PHENOTYPE LABELS FOR STEP C",
        "───────────────────────────",
        f"  Primary stratifier: rule_phenotype_2class",
        f"  Step C labels: {stepc_labels}",
        f"  Excluded (explanatory): {explanatory_labels}",
        "",
        "PHENOTYPE ROLE VALIDATION",
        "─────────────────────────",
    ])
    for err in role_errors:
        lines.append(f"  {err}")
    if not role_errors:
        lines.append("  All roles validated — PASS")
    lines.extend([
        "",
        "OUTCOME DEFINITIONS",
        "───────────────────",
        "  Primary: mortality_28d (28-day all-cause mortality)",
        "  Secondary: mortality_7d",
        "  Absorbing states: death, discharge (during 24h horizon)",
        "",
        "NEXT STEPS",
        "──────────",
        "  70b: fit behavior policy (clinician propensity model)",
        "  70c: define rule-based policy v1",
    ])

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Report: {report_path}")

    print("\n70a COMPLETE.")


if __name__ == "__main__":
    main()
