#!/usr/bin/env python3
"""
45b_encode_actions_2h — Encode fluid, pressor, and diuretic actions on the 2h grid.

Purpose
-------
Apply the frozen coarse_3 action-space discretization to the continuous
action variables in 40b grid. Produces integer action codes per bin and
a joint action index (0–17 for coarse_3).

Protocol reference: Step 45b

Inputs
------
- data/interim/40b_grid2h_states_mimic.parquet
- configs/action_space_v1.yaml

Outputs
-------
- data/interim/45b_actions_2h_v1_mimic.parquet
- reports/45b_actions_2h_v1_summary.txt
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path


def discretize(series: pd.Series, bins: list, labels: list) -> pd.Series:
    """Discretize a continuous series using bin edges."""
    # Replace inf with a large number for pd.cut
    edges = [float(b) if b != "inf" else np.inf for b in bins]
    codes = pd.cut(series, bins=edges, labels=labels, right=False, include_lowest=True)
    return codes.astype("category")


def _load_action_scheme(scheme: str, configs_dir: Path) -> dict:
    """Resolve the action-discretization config for a given scheme.

    coarse_3 (DEFAULT) → frozen ``action_space_v1.yaml`` (byte-for-byte the
    historical behaviour). For any other scheme, read the cell definition from
    ``29a_candidate_action_schemes.yaml`` and normalise it into the same
    ``action_dimensions`` / ``joint_cells`` shape used downstream.
    """
    if scheme == "coarse_3":
        with open(configs_dir / "action_space_v1.yaml") as f:
            return yaml.safe_load(f)

    # Sensitivity schemes (e.g. medium_4) read from the candidate-scheme catalog.
    with open(configs_dir / "29a_candidate_action_schemes.yaml") as f:
        cand = yaml.safe_load(f)
    if scheme not in cand["schemes"]:
        raise ValueError(
            f"scheme '{scheme}' not found in 29a_candidate_action_schemes.yaml "
            f"(available: {sorted(cand['schemes'])})"
        )
    sdef = cand["schemes"][scheme]
    dims = {}
    for dim_key in ("iv_fluid_ml_per_2h",
                    "vasopressor_ne_eq_mcg_kg_min",
                    "diuretic_furo_eq_mg_per_2h"):
        d = sdef[dim_key]
        dims[dim_key] = {
            "bins": d["bins"],
            "labels": d["labels"],
            "n_levels": len(d["labels"]),
        }
    return {
        "primary_scheme": scheme,
        "action_dimensions": dims,
        "joint_cells": int(sdef["joint_cells"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="45b — encode 2h actions")
    parser.add_argument(
        "--scheme", choices=["coarse_3", "medium_4"], default="coarse_3",
        help="Action-discretization scheme. DEFAULT coarse_3 reproduces the "
             "frozen primary behaviour exactly; medium_4 writes *_medium4 outputs.",
    )
    args = parser.parse_args()
    scheme = args.scheme
    suffix = "" if scheme == "coarse_3" else f"_{scheme.replace('_', '')}"

    cfg = load_config()
    interim_dir = resolve_path(cfg, "interim")
    configs_dir = resolve_path(cfg, "configs")
    reports_dir = resolve_path(cfg, "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"[45b] Encoding actions on 2h grid (scheme={scheme})...")

    # ── Load action space config ──────────────────────────────────────
    as_cfg = _load_action_scheme(scheme, configs_dir)

    dims = as_cfg["action_dimensions"]

    # ── Load 40b grid ─────────────────────────────────────────────────
    grid = pd.read_parquet(interim_dir / "40b_grid2h_states_mimic.parquet")
    n_rows = len(grid)
    n_stays = grid["stay_id"].nunique()
    print(f"  Grid: {n_rows:,} rows, {n_stays:,} stays")

    # ── Discretize each action dimension ──────────────────────────────
    # IV fluid
    fluid_cfg = dims["iv_fluid_ml_per_2h"]
    grid["fluid_cat"] = discretize(
        grid["iv_fluid_ml"].fillna(0), fluid_cfg["bins"], fluid_cfg["labels"]
    )

    # Vasopressor NE-equivalent
    pressor_cfg = dims["vasopressor_ne_eq_mcg_kg_min"]
    grid["pressor_cat"] = discretize(
        grid["vasopressor_ne_eq"].fillna(0), pressor_cfg["bins"], pressor_cfg["labels"]
    )

    # Diuretic furosemide-equivalent
    diuretic_cfg = dims["diuretic_furo_eq_mg_per_2h"]
    grid["diuretic_cat"] = discretize(
        grid["diuretic_furo_eq_mg"].fillna(0), diuretic_cfg["bins"], diuretic_cfg["labels"]
    )

    # ── Joint action index ────────────────────────────────────────────
    # Encode as integers for joint index computation
    fluid_codes = grid["fluid_cat"].cat.codes.astype(int)
    pressor_codes = grid["pressor_cat"].cat.codes.astype(int)
    diuretic_codes = grid["diuretic_cat"].cat.codes.astype(int)

    n_p = pressor_cfg["n_levels"]
    n_d = diuretic_cfg["n_levels"]
    grid["action_idx"] = fluid_codes * (n_p * n_d) + pressor_codes * n_d + diuretic_codes
    grid["action_idx"] = grid["action_idx"].astype(np.int16)

    assert grid["action_idx"].max() < as_cfg["joint_cells"], (
        f"action_idx max={grid['action_idx'].max()} >= {as_cfg['joint_cells']}"
    )

    # ── Select output columns ─────────────────────────────────────────
    out_cols = [
        "stay_id", "bin",
        "iv_fluid_ml", "vasopressor_ne_eq", "diuretic_furo_eq_mg",
        "fluid_cat", "pressor_cat", "diuretic_cat", "action_idx",
    ]
    out = grid[out_cols].copy()

    # ── Save ──────────────────────────────────────────────────────────
    out_path = interim_dir / f"45b_actions_2h_v1{suffix}_mimic.parquet"
    out.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path}")

    # ── Summary report ────────────────────────────────────────────────
    joint_dist = out["action_idx"].value_counts().sort_index()
    fluid_dist = out["fluid_cat"].value_counts()
    pressor_dist = out["pressor_cat"].value_counts()
    diuretic_dist = out["diuretic_cat"].value_counts()

    report_lines = [
        "=" * 72,
        "45b — Action Encoding Summary (MIMIC-IV)",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Scheme: {as_cfg['primary_scheme']}",
        f"Rows: {n_rows:,}, Stays: {n_stays:,}",
        "",
        "MARGINAL DISTRIBUTIONS",
        "──────────────────────",
        f"  IV fluid:     {fluid_dist.to_dict()}",
        f"  Vasopressor:  {pressor_dist.to_dict()}",
        f"  Diuretic:     {diuretic_dist.to_dict()}",
        "",
        "JOINT ACTION INDEX DISTRIBUTION",
        "────────────────────────────────",
    ]
    for idx, cnt in joint_dist.items():
        pct = cnt / n_rows * 100
        report_lines.append(f"  action_idx={idx:>2d}: {cnt:>8,} ({pct:5.1f}%)")

    report_lines.extend([
        "",
        f"Empty cells: {(joint_dist == 0).sum() if len(joint_dist) < as_cfg['joint_cells'] else as_cfg['joint_cells'] - len(joint_dist)}",
        f"Min cell count: {joint_dist.min():,}",
    ])

    report = "\n".join(report_lines) + "\n"
    report_path = reports_dir / f"45b_actions_2h_v1{suffix}_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {report_path}")
    print(report)
    print("[45b] Done.")


if __name__ == "__main__":
    main()
