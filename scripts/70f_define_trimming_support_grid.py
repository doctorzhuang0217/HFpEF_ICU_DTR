#!/usr/bin/env python3
"""
70f_define_trimming_support_grid — Freeze positivity trimming thresholds.

Purpose
-------
Pre-declare the propensity weight trimming grid and positivity-violation
handling rules. These apply uniformly across all Step C methods.

Grid (from SAP 03f + 45g):
  - Percentile-based: trim at 1/99, 2.5/97.5, 5/95, 20/80 of weight dist.
  - ESS threshold: flag if ESS < 25 per cell×stratum.
  - Sparse action coverage: pre-activate coarse_2 for HFpEF arm (45g Rule 3).

Protocol reference: Step 70f (v3.1), SAP 03f, 45g rare-cell handling note

Inputs
------
- data/interim/70b_propensity_coarse3_v1_mimic.parquet
- tables/70b_ess_per_cell_stratum.csv
- configs/00f_global_config_registry.yaml

Outputs
-------
- configs/70f_trimming_grid_v1.yaml
- reports/70f_trimming_support_summary.txt

Compute: C1
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path


def main() -> None:
    proj_cfg = load_config()
    interim_dir = resolve_path(proj_cfg, "interim")
    configs_dir = resolve_path(proj_cfg, "configs")
    tables_dir = resolve_path(proj_cfg, "tables")
    reports_dir = resolve_path(proj_cfg, "reports")

    print("=" * 72)
    print("70f — Define Trimming and Support Grid")
    print("=" * 72)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 1. Load propensities for weight distribution ──────────────────────
    print("\n[1] Loading propensity data...")
    prop_path = interim_dir / "70b_propensity_coarse3_v1_mimic.parquet"
    prop_df = pd.read_parquet(prop_path, columns=["propensity", "propensity_clipped"])
    n_obs = len(prop_df)

    w_raw = 1.0 / prop_df["propensity"].clip(lower=1e-6).values
    w_clipped = 1.0 / prop_df["propensity_clipped"].values

    print(f"  {n_obs:,} observations")
    print(f"  Raw weights:     P1={np.percentile(w_raw,1):.2f}, "
          f"P50={np.median(w_raw):.2f}, P99={np.percentile(w_raw,99):.2f}, "
          f"max={w_raw.max():.2f}")

    # ── 2. Compute trimming grid statistics ───────────────────────────────
    print("\n[2] Computing trimming grid...")
    trim_levels = [
        (1, 99),
        (2.5, 97.5),
        (5, 95),
        (10, 90),
        (20, 80),
    ]

    trim_results = []
    for lo, hi in trim_levels:
        lo_val = np.percentile(w_clipped, lo)
        hi_val = np.percentile(w_clipped, hi)
        trimmed = w_clipped.copy()
        trimmed = np.clip(trimmed, lo_val, hi_val)
        n_trimmed = int(np.sum((w_clipped < lo_val) | (w_clipped > hi_val)))
        ess = (np.sum(trimmed) ** 2) / np.sum(trimmed ** 2)
        trim_results.append({
            "percentile_lo": lo,
            "percentile_hi": hi,
            "weight_lo": round(float(lo_val), 4),
            "weight_hi": round(float(hi_val), 4),
            "n_trimmed": n_trimmed,
            "pct_trimmed": round(n_trimmed / n_obs * 100, 2),
            "ess_after_trim": round(float(ess), 1),
        })
        print(f"  [{lo:5.1f}, {hi:5.1f}]: trim=[{lo_val:.4f}, {hi_val:.4f}], "
              f"n_trimmed={n_trimmed:,} ({n_trimmed/n_obs*100:.1f}%), "
              f"ESS={ess:.0f}")

    # ── 3. ESS summary ───────────────────────────────────────────────────
    print("\n[3] ESS summary from 70b...")
    ess_path = tables_dir / "70b_ess_per_cell_stratum.csv"
    if ess_path.exists():
        ess_df = pd.read_csv(ess_path)
        n_flagged = int(ess_df["ess_flag"].sum())
        n_total = len(ess_df)
        print(f"  {n_flagged}/{n_total} cell×stratum combos with ESS<25")
        coarser_triggered = n_flagged > 3
    else:
        print("  ESS table not found (run 70b first)")
        n_flagged = 0
        n_total = 0
        coarser_triggered = False

    # ── 4. Build trimming config ──────────────────────────────────────────
    print("\n[4] Building trimming config...")
    trim_cfg = {
        "step": "70f_define_trimming_support_grid",
        "date": ts,
        "protocol_version": "v3.1",
        "propensity_clipping": {
            "method": "hard_floor",
            "floor": 0.005,
            "description": "Clip P(A|H) >= 0.005 before computing IPTW weights",
        },
        "weight_trimming_grid": {
            "method": "percentile_based",
            "primary": {"lo": 5, "hi": 95},
            "sensitivity": [
                {"lo": lo, "hi": hi} for lo, hi in trim_levels
            ],
            "results": trim_results,
        },
        "ess_monitoring": {
            "threshold": 25,
            "n_flagged": n_flagged,
            "n_total": n_total,
            "coarser_granularity_triggered": coarser_triggered,
            "trigger_rule": "45g Rule 2: >3 cell×stratum combos with ESS<25",
        },
        "positivity_support": {
            "minimum_cell_frequency": 0.001,
            "auto_merge_threshold": 0.001,
            "merge_map_reference": "configs/70b_action_merge_map.yaml",
        },
        "coarse2_activation": {
            "hfpef_sparse_coverage": 0.308,
            "threshold": 0.15,
            "activated": True,
            "note": (
                "HFpEF arm has 30.8% stays with sparse-action exposure, "
                "exceeding 15% threshold. coarse_2 (9-cell) activated "
                "as mandatory sensitivity for HFpEF-specific estimates."
            ),
        },
    }

    cfg_path = configs_dir / "70f_trimming_grid_v1.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(trim_cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  Saved: {cfg_path}")

    # ── 5. Summary report ─────────────────────────────────────────────────
    report_path = reports_dir / "70f_trimming_support_summary.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "=" * 72,
        "70f — Trimming and Support Grid (MIMIC-IV)",
        "=" * 72,
        f"Generated: {ts}",
        "",
        "PROPENSITY CLIPPING",
        "───────────────────",
        "  Floor: P(A|H) >= 0.005",
        "",
        "WEIGHT TRIMMING GRID",
        "────────────────────",
        "  Primary: [P5, P95]",
        "  Sensitivity levels:",
    ]
    for r in trim_results:
        lines.append(
            f"    [{r['percentile_lo']:5.1f}, {r['percentile_hi']:5.1f}]: "
            f"weight=[{r['weight_lo']:.4f}, {r['weight_hi']:.4f}], "
            f"trimmed={r['n_trimmed']:,} ({r['pct_trimmed']:.1f}%), "
            f"ESS={r['ess_after_trim']:.0f}"
        )
    lines.extend([
        "",
        "ESS MONITORING",
        "──────────────",
        f"  Flagged (ESS<25): {n_flagged}/{n_total} cell×stratum combos",
        f"  Coarser granularity triggered: {coarser_triggered}",
        "",
        "COARSE_2 ACTIVATION (45g Rule 3)",
        "─────────────────────────────────",
        f"  HFpEF sparse coverage: 30.8% (threshold: 15%)",
        f"  Status: ACTIVATED for HFpEF-specific estimates",
        "",
        "NEXT STEPS",
        "──────────",
        "  70g: freeze model registry",
        "  75a: apply trimming grid to IPTW estimation",
    ])

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Report: {report_path}")
    print("\n70f COMPLETE.")


if __name__ == "__main__":
    main()
