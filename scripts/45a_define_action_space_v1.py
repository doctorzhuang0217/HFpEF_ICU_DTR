#!/usr/bin/env python3
"""
45a_define_action_space_v1 — Freeze Phase-1 action space and thresholds.

Purpose
-------
Select the primary action-space scheme (coarse_3) from 29a candidates and
freeze it as the canonical action space for all downstream modeling.
Validates the scheme definition and writes action_space_v1.yaml.

Protocol reference: Step 45a

Inputs
------
- configs/29a_candidate_action_schemes.yaml

Outputs
-------
- configs/action_space_v1.yaml
- reports/45a_action_space_v1_summary.txt
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path


def main() -> None:
    cfg = load_config()
    configs_dir = resolve_path(cfg, "configs")
    reports_dir = resolve_path(cfg, "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("[45a] Freezing Phase-1 action space...")

    # ── Load 29a schemes ──────────────────────────────────────────────
    schemes_path = configs_dir / "29a_candidate_action_schemes.yaml"
    with open(schemes_path) as f:
        schemes_cfg = yaml.safe_load(f)

    # ── Select primary scheme: coarse_3 ───────────────────────────────
    # Decision from 29b: coarse_3 is CONDITIONAL (mitigated), medium_4 FAIL,
    # fine_5 FAIL. Phase 0.5 GO with C2 monitoring selects coarse_3.
    primary_scheme_name = "coarse_3"
    scheme = schemes_cfg["schemes"][primary_scheme_name]

    # ── Build frozen action space config ──────────────────────────────
    action_space = {
        "step": "45a_define_action_space_v1",
        "date_frozen": datetime.now().strftime("%Y-%m-%d"),
        "protocol_version": schemes_cfg.get("protocol_version", "v3.1"),
        "source": str(schemes_path),
        "primary_scheme": primary_scheme_name,
        "grid": schemes_cfg["grid"],
        "ne_equivalent_factors": schemes_cfg["ne_equivalent_factors"],
        "furo_equivalent_factors": schemes_cfg["furo_equivalent_factors"],
        "action_dimensions": {
            "iv_fluid_ml_per_2h": {
                "bins": scheme["iv_fluid_ml_per_2h"]["bins"],
                "labels": scheme["iv_fluid_ml_per_2h"]["labels"],
                "n_levels": len(scheme["iv_fluid_ml_per_2h"]["labels"]),
            },
            "vasopressor_ne_eq_mcg_kg_min": {
                "bins": scheme["vasopressor_ne_eq_mcg_kg_min"]["bins"],
                "labels": scheme["vasopressor_ne_eq_mcg_kg_min"]["labels"],
                "n_levels": len(scheme["vasopressor_ne_eq_mcg_kg_min"]["labels"]),
            },
            "diuretic_furo_eq_mg_per_2h": {
                "bins": scheme["diuretic_furo_eq_mg_per_2h"]["bins"],
                "labels": scheme["diuretic_furo_eq_mg_per_2h"]["labels"],
                "n_levels": len(scheme["diuretic_furo_eq_mg_per_2h"]["labels"]),
            },
        },
        "joint_cells": scheme["joint_cells"],
        "sensitivity_schemes": ["medium_4"],  # pre-registered for sensitivity
    }

    # ── Validate ──────────────────────────────────────────────────────
    n_fluid = action_space["action_dimensions"]["iv_fluid_ml_per_2h"]["n_levels"]
    n_pressor = action_space["action_dimensions"]["vasopressor_ne_eq_mcg_kg_min"]["n_levels"]
    n_diuretic = action_space["action_dimensions"]["diuretic_furo_eq_mg_per_2h"]["n_levels"]
    expected_cells = n_fluid * n_pressor * n_diuretic
    assert expected_cells == scheme["joint_cells"], (
        f"Joint cells mismatch: {n_fluid}×{n_pressor}×{n_diuretic}="
        f"{expected_cells} != {scheme['joint_cells']}"
    )

    # ── Save ──────────────────────────────────────────────────────────
    out_path = configs_dir / "action_space_v1.yaml"
    with open(out_path, "w") as f:
        yaml.dump(action_space, f, default_flow_style=False, sort_keys=False)
    print(f"  Saved: {out_path}")

    # ── Report ────────────────────────────────────────────────────────
    report = (
        f"{'=' * 72}\n"
        f"45a — Action Space v1 Summary\n"
        f"{'=' * 72}\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Primary scheme: {primary_scheme_name}\n"
        f"Grid: {schemes_cfg['grid']['bin_hours']}h bins × "
        f"{schemes_cfg['grid']['n_bins']} = "
        f"{schemes_cfg['grid']['primary_horizon_hours']}h horizon\n\n"
        f"ACTION DIMENSIONS\n"
        f"─────────────────\n"
        f"  IV fluid:     {n_fluid} levels {scheme['iv_fluid_ml_per_2h']['labels']}\n"
        f"  Vasopressor:  {n_pressor} levels {scheme['vasopressor_ne_eq_mcg_kg_min']['labels']}\n"
        f"  Diuretic:     {n_diuretic} levels {scheme['diuretic_furo_eq_mg_per_2h']['labels']}\n"
        f"  Joint cells:  {expected_cells}\n\n"
        f"SENSITIVITY\n"
        f"───────────\n"
        f"  medium_4 (32 cells) pre-registered for sensitivity analysis\n"
    )

    report_path = reports_dir / "45a_action_space_v1_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {report_path}")
    print(report)
    print("[45a] Done.")


if __name__ == "__main__":
    main()
