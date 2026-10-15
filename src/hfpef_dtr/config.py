"""
00f_global_config_registry — loader and validator.

Usage
-----
    from hfpef_dtr.config import load_config
    cfg = load_config()                       # auto-detect PROJECT_ROOT
    cfg["paths"]["raw"]                       # "data/raw"
    cfg["datasets"]["mimic"]["status"]        # "ready"

    # Resolve an absolute path:
    from hfpef_dtr.config import resolve_path
    raw_dir = resolve_path(cfg, "raw")        # Path("<root>/data/raw")

Validation
----------
    python -m hfpef_dtr.config               # prints summary + runs checks
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml


# ── locate config ──────────────────────────────────────────────────────────

_CONFIG_RELPATH = "configs/00f_global_config_registry.yaml"


def _find_project_root() -> Path:
    """Resolve PROJECT_ROOT from env var or by walking up from this file."""
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root)
    # Fall back: walk up from src/hfpef_dtr/config.py → repo root
    return Path(__file__).resolve().parents[2]


def load_config(project_root: str | Path | None = None) -> dict[str, Any]:
    """Load and return the global config as a plain dict."""
    root = Path(project_root) if project_root else _find_project_root()
    config_path = root / _CONFIG_RELPATH
    if not config_path.exists():
        raise FileNotFoundError(
            f"Global config not found at {config_path}. "
            "Ensure you are running from the project root or "
            "set the PROJECT_ROOT environment variable."
        )
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_project_root"] = str(root)
    return cfg


# ── path helpers ───────────────────────────────────────────────────────────

def resolve_path(cfg: dict[str, Any], path_key: str) -> Path:
    """Return an absolute Path for a key in cfg['paths']."""
    root = Path(cfg["_project_root"])
    rel = cfg["paths"][path_key]
    return root / rel


def dataset_entrypoint(cfg: dict[str, Any], db: str) -> Path:
    """Return absolute Path to a dataset entrypoint."""
    root = Path(cfg["_project_root"])
    return root / cfg["datasets"][db]["entrypoint"]


# ── validation ─────────────────────────────────────────────────────────────

_REQUIRED_TOP_KEYS = [
    "project",
    "paths",
    "datasets",
    "dataset_priority",
    "study_design",
    "comparators",
    "step_c_models",
    "execution",
    "occupied_step_ids",
    "seeds",
]

_REQUIRED_DATASETS = ["mimic", "eicu", "hirid"]

_REQUIRED_STUDY_DESIGN_KEYS = [
    "time_zero",
    "decision_grid",
    "primary_outcome",
    "cohort",
    "trimming_grid",
]


def validate(cfg: dict[str, Any]) -> list[str]:
    """Return a list of validation errors (empty = OK)."""
    errors: list[str] = []

    # top-level keys
    for k in _REQUIRED_TOP_KEYS:
        if k not in cfg:
            errors.append(f"Missing top-level key: {k}")

    # datasets
    ds = cfg.get("datasets", {})
    for db in _REQUIRED_DATASETS:
        if db not in ds:
            errors.append(f"Missing dataset entry: {db}")
        else:
            for field in ("entrypoint", "status", "hfpef_ascertainment"):
                if field not in ds[db]:
                    errors.append(f"datasets.{db} missing field: {field}")

    # study_design
    sd = cfg.get("study_design", {})
    for k in _REQUIRED_STUDY_DESIGN_KEYS:
        if k not in sd:
            errors.append(f"study_design missing key: {k}")

    # time_zero sanity
    tz = sd.get("time_zero", {})
    if tz.get("primary_hours") != 6:
        errors.append(f"time_zero.primary_hours should be 6, got {tz.get('primary_hours')}")
    if tz.get("sensitivity_hours") != 12:
        errors.append(f"time_zero.sensitivity_hours should be 12, got {tz.get('sensitivity_hours')}")

    # decision_grid sanity
    dg = sd.get("decision_grid", {})
    if dg.get("bin_hours") != 2:
        errors.append(f"decision_grid.bin_hours should be 2, got {dg.get('bin_hours')}")
    if dg.get("primary_horizon_hours") != 24:
        errors.append(f"decision_grid.primary_horizon_hours should be 24, got {dg.get('primary_horizon_hours')}")

    # Step C primary pair
    sc = cfg.get("step_c_models", {})
    pp_ids = [m["id"] for m in sc.get("primary_pair", [])]
    if "75a_dynamic_msm_iptw" not in pp_ids:
        errors.append("Step C primary pair missing 75a_dynamic_msm_iptw")
    if "80c_gtransformer" not in pp_ids:
        errors.append("Step C primary pair missing 80c_gtransformer")

    # occupied step IDs: ensure bootstrap + completed steps present
    occ = cfg.get("occupied_step_ids", {})
    for sid in ["00a", "00b", "00c", "00d", "00e", "00f", "03f", "03g", "03h"]:
        if sid not in occ:
            errors.append(f"occupied_step_ids missing: {sid}")

    return errors


# ── CLI entrypoint ─────────────────────────────────────────────────────────

def _print_summary(cfg: dict[str, Any]) -> None:
    root = cfg["_project_root"]
    print(f"Project root : {root}")
    print(f"Protocol     : {cfg['project']['protocol_version']}")
    print()

    print("Datasets:")
    for db in cfg["dataset_priority"]:
        d = cfg["datasets"][db]
        print(f"  {d['label']:16s}  status={d['status']:40s}  ascertainment={d['hfpef_ascertainment']}")
    print()

    sd = cfg["study_design"]
    tz = sd["time_zero"]
    dg = sd["decision_grid"]
    print(f"Time zero    : primary {tz['primary_hours']}h / sensitivity {tz['sensitivity_hours']}h")
    print(f"Decision grid: {dg['bin_hours']}h bins, {dg['primary_horizon_hours']}h horizon "
          f"(sens: {dg['sensitivity_horizons_hours']})")
    print(f"Primary outcome: {sd['primary_outcome']}")
    print()

    print("Step C primary pair:")
    for m in cfg["step_c_models"]["primary_pair"]:
        print(f"  {m['id']:30s}  {m['label']}")
    print()

    print(f"Execution phase: {cfg['execution']['current_phase']}")
    print(f"Last completed : {cfg['execution']['last_completed_step']}")
    print(f"Next steps     : {', '.join(cfg['execution']['next_steps'][:3])} ...")
    print()

    print(f"Seeds: global={cfg['seeds']['global']}, "
          f"split={cfg['seeds']['split']}, "
          f"bootstrap={cfg['seeds']['bootstrap']}")


def main() -> None:
    cfg = load_config()
    _print_summary(cfg)

    errors = validate(cfg)
    if errors:
        print(f"\nValidation FAILED ({len(errors)} error(s)):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("\nValidation PASSED — all checks OK.")


if __name__ == "__main__":
    main()
