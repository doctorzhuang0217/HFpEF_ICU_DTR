#!/usr/bin/env python3
"""
40a_common_variable_map — Validate and report on the common variable map.

Purpose
-------
Load the frozen common_variable_map_v1.yaml, validate its structure,
and produce a summary report showing variable counts by role/domain,
cross-database coverage, and any QC warnings.

Protocol reference: Step 40a (C1).

Inputs
------
- configs/common_variable_map_v1.yaml

Outputs
-------
- reports/40a_common_variable_map_summary.txt
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    config_path = ROOT / "configs" / "common_variable_map_v1.yaml"
    print(f"[40a] Loading variable map from {config_path} ...")

    with open(config_path) as f:
        vmap = yaml.safe_load(f)

    print(f"  Version: {vmap.get('version', '?')}")
    print(f"  Grid: {vmap.get('grid_bin_hours', '?')}h bins, "
          f"T0={vmap.get('t0_hours', '?')}h, "
          f"horizon={vmap.get('primary_horizon_hours', '?')}h")

    # ── Collect all variables ──────────────────────────────────────────
    sections = [
        ("static_covariates", "static"),
        ("state_variables", "state"),
        ("action_variables", "action"),
        ("outcome_variables", "outcome"),
        ("context_variables", "context"),
    ]

    all_vars = []
    for section_key, default_role in sections:
        section = vmap.get(section_key, {})
        if not isinstance(section, dict):
            continue
        for var_name, var_def in section.items():
            if not isinstance(var_def, dict):
                continue
            var_def["_name"] = var_name
            var_def["_section"] = section_key
            var_def["_role"] = var_def.get("role", default_role)
            all_vars.append(var_def)

    n_total = len(all_vars)
    print(f"  Total variables defined: {n_total}")

    # ── Coverage analysis ──────────────────────────────────────────────
    databases = ["mimic", "eicu", "hirid"]
    coverage = {db: {"mapped": 0, "null": 0} for db in databases}

    for v in all_vars:
        for db in databases:
            col = v.get(db)
            if col and col != "null":
                coverage[db]["mapped"] += 1
            else:
                coverage[db]["null"] += 1

    # ── By-domain counts ───────────────────────────────────────────────
    domain_counts: dict[str, int] = {}
    for v in all_vars:
        d = v.get("domain", "unknown")
        domain_counts[d] = domain_counts.get(d, 0) + 1

    # ── QC checks ──────────────────────────────────────────────────────
    warnings = []

    for v in all_vars:
        name = v["_name"]
        # Check valid_range format
        vr = v.get("valid_range")
        if vr is not None:
            if not isinstance(vr, list) or len(vr) != 2:
                warnings.append(f"{name}: valid_range must be [lo, hi], got {vr}")
            elif vr[0] >= vr[1]:
                warnings.append(f"{name}: valid_range lo >= hi: {vr}")

        # Check state vars have grid_agg
        if v["_role"] == "state" and "grid_agg" not in v:
            warnings.append(f"{name}: state variable missing grid_agg")

        # Check unit is specified
        if "unit" not in v:
            warnings.append(f"{name}: missing unit")

    # ── Build report ───────────────────────────────────────────────────
    lines = [
        "=" * 72,
        "40a — Common Variable Map Summary",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Config: {config_path.relative_to(ROOT)}",
        f"Version: {vmap.get('version', '?')}",
        f"Grid: {vmap.get('grid_bin_hours', '?')}h bins, "
        f"T0={vmap.get('t0_hours', '?')}h, "
        f"horizon={vmap.get('primary_horizon_hours', '?')}h",
        "",
        "VARIABLE COUNTS BY ROLE",
        "───────────────────────",
    ]

    role_counts: dict[str, int] = {}
    for v in all_vars:
        r = v["_role"]
        role_counts[r] = role_counts.get(r, 0) + 1
    for role, cnt in sorted(role_counts.items()):
        lines.append(f"  {role:<20} {cnt:>3}")
    lines.append(f"  {'TOTAL':<20} {n_total:>3}")
    lines.append("")

    lines.append("VARIABLE COUNTS BY DOMAIN")
    lines.append("─────────────────────────")
    for domain, cnt in sorted(domain_counts.items()):
        lines.append(f"  {domain:<25} {cnt:>3}")
    lines.append("")

    lines.append("CROSS-DATABASE COVERAGE")
    lines.append("───────────────────────")
    for db in databases:
        m = coverage[db]["mapped"]
        n = coverage[db]["null"]
        pct = m / n_total * 100 if n_total > 0 else 0
        lines.append(f"  {db:<10} mapped: {m:>3}/{n_total}  ({pct:.0f}%)  unmapped: {n:>3}")
    lines.append("")

    # List unmapped eICU/HiRID vars
    for db in ["eicu", "hirid"]:
        unmapped = [v["_name"] for v in all_vars if not v.get(db) or v.get(db) == "null"]
        if unmapped:
            lines.append(f"  {db} unmapped: {', '.join(unmapped[:10])}"
                         f"{'...' if len(unmapped) > 10 else ''}")
    lines.append("")

    lines.append("QC WARNINGS")
    lines.append("───────────")
    if warnings:
        for w in warnings:
            lines.append(f"  ⚠ {w}")
    else:
        lines.append("  None")
    lines.append("")

    # Variable inventory table
    lines.append("VARIABLE INVENTORY")
    lines.append("──────────────────")
    lines.append(f"  {'Name':<30} {'Role':<10} {'Domain':<20} {'Unit':<12} {'MIMIC':<20} {'eICU':<18} {'HiRID':<10}")
    lines.append("  " + "─" * 118)
    for v in all_vars:
        name = v["_name"]
        role = v["_role"]
        domain = v.get("domain", "?")
        unit = v.get("unit", "?")
        mimic = v.get("mimic", "—") or "—"
        eicu = v.get("eicu", "—") or "—"
        hirid = v.get("hirid", "—") or "—"
        lines.append(f"  {name:<30} {role:<10} {domain:<20} {unit:<12} {mimic:<20} {eicu:<18} {hirid:<10}")

    # ── Save ───────────────────────────────────────────────────────────
    report_path = reports_dir / "40a_common_variable_map_summary.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {report_path}")

    # Print warnings to console
    if warnings:
        print(f"\n  ⚠ {len(warnings)} QC warning(s):")
        for w in warnings:
            print(f"    {w}")
    else:
        print("  QC: all checks passed")

    print(f"\n[40a] Done. {n_total} variables mapped.")


if __name__ == "__main__":
    main()
