#!/usr/bin/env python3
"""
45f_freeze_modeling_dataset_v1 — Freeze the first analysis-ready dataset manifest.

Purpose
-------
Validate all 45a–45e outputs exist and are consistent, then write a manifest
JSON documenting the frozen dataset for downstream modeling (Step 50+).

Protocol reference: Step 45f (C1)

Inputs
------
- configs/action_space_v1.yaml                          (45a)
- data/interim/45b_actions_2h_v1_mimic.parquet          (45b)
- data/interim/45c_state_tensors_v1_mimic.parquet       (45c)
- data/interim/45d_outcomes_absorb_v1_mimic.parquet     (45d)
- data/interim/45e_data_quality_flags_v1_mimic.parquet  (45e)
- configs/45c_normalization_params_v1.yaml              (45c)

Outputs
-------
- data/processed/dataset_v1_manifest_mimic.json
- reports/45f_freeze_dataset_v1_summary.txt
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path


def file_hash(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    cfg = load_config()
    interim_dir = resolve_path(cfg, "interim")
    configs_dir = resolve_path(cfg, "configs")
    processed_dir = resolve_path(cfg, "processed")
    reports_dir = resolve_path(cfg, "reports")
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("45f_freeze_modeling_dataset_v1 — Dataset manifest freeze")
    print("=" * 70)

    # ── Define expected artifacts ─────────────────────────────────────
    artifacts = {
        "action_space_config": configs_dir / "action_space_v1.yaml",
        "normalization_params": configs_dir / "45c_normalization_params_v1.yaml",
        "actions_2h": interim_dir / "45b_actions_2h_v1_mimic.parquet",
        "state_tensors": interim_dir / "45c_state_tensors_v1_mimic.parquet",
        "outcomes_absorb": interim_dir / "45d_outcomes_absorb_v1_mimic.parquet",
        "quality_flags": interim_dir / "45e_data_quality_flags_v1_mimic.parquet",
    }

    # ── Validate existence ────────────────────────────────────────────
    print("\n[1/4] Validating artifacts ...")
    missing = []
    for name, path in artifacts.items():
        if path.exists():
            print(f"  OK: {name} → {path.name}")
        else:
            print(f"  MISSING: {name} → {path}")
            missing.append(name)

    if missing:
        print(f"\n  ERROR: {len(missing)} artifact(s) missing: {missing}")
        print("  Run 45a–45e first.")
        sys.exit(1)

    # ── Load and cross-validate ───────────────────────────────────────
    print("\n[2/4] Cross-validating ...")
    actions = pd.read_parquet(artifacts["actions_2h"])
    states = pd.read_parquet(artifacts["state_tensors"])
    outcomes = pd.read_parquet(artifacts["outcomes_absorb"])
    flags = pd.read_parquet(artifacts["quality_flags"])

    # Stay-level consistency
    action_stays = set(actions["stay_id"].unique())
    state_stays = set(states["stay_id"].unique())
    outcome_stays = set(outcomes["stay_id"].unique())
    flag_stays = set(flags["stay_id"].unique())

    n_action = len(action_stays)
    n_state = len(state_stays)
    n_outcome = len(outcome_stays)
    n_flag = len(flag_stays)

    errors = []
    if action_stays != state_stays:
        diff = action_stays.symmetric_difference(state_stays)
        errors.append(f"actions vs states: {len(diff)} mismatched stays")
    if state_stays != outcome_stays:
        diff = state_stays.symmetric_difference(outcome_stays)
        errors.append(f"states vs outcomes: {len(diff)} mismatched stays")
    if state_stays != flag_stays:
        diff = state_stays.symmetric_difference(flag_stays)
        errors.append(f"states vs flags: {len(diff)} mismatched stays")

    # Row-level consistency (actions and states should match row-for-row)
    if len(actions) != len(states):
        errors.append(f"Row count mismatch: actions={len(actions):,} vs states={len(states):,}")

    # Key-level consistency (stay_id/bin pairs should be identical)
    action_keys = set(zip(actions["stay_id"], actions["bin"]))
    state_keys = set(zip(states["stay_id"], states["bin"]))
    if action_keys != state_keys:
        diff = action_keys.symmetric_difference(state_keys)
        errors.append(f"actions vs states: {len(diff):,} mismatched (stay_id, bin) keys")

    if errors:
        print("  VALIDATION ERRORS:")
        for e in errors:
            print(f"    - {e}")
        print("  ERROR: Freeze aborted due to validation failures.")
        sys.exit(1)
    else:
        print("  All cross-validations PASSED.")

    # ── Compute hashes ────────────────────────────────────────────────
    print("\n[3/4] Computing file hashes ...")
    hashes = {}
    for name, path in artifacts.items():
        h = file_hash(path)
        hashes[name] = h
        print(f"  {name}: {h[:16]}...")

    # ── Build manifest ────────────────────────────────────────────────
    print("\n[4/4] Writing manifest ...")

    # Load action space config for metadata
    with open(artifacts["action_space_config"]) as f:
        as_cfg = yaml.safe_load(f)

    manifest = {
        "manifest_version": "v1",
        "date_frozen": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "database": "mimic-iv",
        "protocol_version": "v3.1",
        "study_design": {
            "t0_hours": 6,
            "bin_hours": 2,
            "n_bins": 12,
            "horizon_hours": 24,
        },
        "cohort": {
            "n_stays": n_state,
            "n_rows": len(states),
        },
        "action_space": {
            "scheme": as_cfg.get("primary_scheme", "coarse_3"),
            "joint_cells": as_cfg.get("joint_cells", 18),
        },
        "state_tensor": {
            "n_columns": states.shape[1],
            "columns": list(states.columns),
        },
        "outcomes": {
            "n_stays": n_outcome,
            "columns": list(outcomes.columns),
        },
        "quality_flags": {
            "n_stays": n_flag,
            "columns": list(flags.columns),
        },
        "artifacts": {
            name: {
                "path": str(path.relative_to(Path(cfg["_project_root"]))),
                "sha256": hashes[name],
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
            }
            for name, path in artifacts.items()
        },
        "validation_errors": errors,
    }

    manifest_path = processed_dir / "dataset_v1_manifest_mimic.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  Saved: {manifest_path}")

    # ── Summary report ────────────────────────────────────────────────
    report_lines = [
        "=" * 72,
        "45f — Dataset v1 Freeze Summary (MIMIC-IV)",
        "=" * 72,
        f"Generated: {manifest['date_frozen']}",
        f"Protocol: {manifest['protocol_version']}",
        "",
        "COHORT",
        "───────",
        f"  Stays: {n_state:,}",
        f"  Rows (bins): {len(states):,}",
        f"  Bins per stay: {len(states) / max(n_state, 1):.1f}",
        "",
        "ACTION SPACE",
        "─────────────",
        f"  Scheme: {manifest['action_space']['scheme']}",
        f"  Joint cells: {manifest['action_space']['joint_cells']}",
        "",
        "ARTIFACTS",
        "──────────",
    ]
    for name, info in manifest["artifacts"].items():
        report_lines.append(f"  {name:<25}: {info['path']} ({info['size_mb']:.1f} MB)")

    if errors:
        report_lines.extend(["", "VALIDATION WARNINGS", "────────────────────"])
        for e in errors:
            report_lines.append(f"  - {e}")
    else:
        report_lines.extend(["", "VALIDATION: ALL PASSED"])

    report = "\n".join(report_lines) + "\n"
    report_path = reports_dir / "45f_freeze_dataset_v1_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {report_path}")
    print(report)
    print("[45f] Done — Dataset v1 frozen.")


if __name__ == "__main__":
    main()
