#!/usr/bin/env python3
"""
70d_define_rewards_and_metrics — Freeze rewards and evaluation metrics.

Purpose
-------
Freeze the reward definitions and evaluation metric configurations for
all Step C methods. This ensures consistent outcome handling across:
  - 75a (dynamic-MSM/IPTW)
  - 80a-80e (deep learning methods)
  - 85a-85j (sensitivity analyses)

Reward design:
  - Primary reward: -1 * mortality_28d (survival = 0, death = -1)
  - Per-step intermediate reward: 0 (sparse reward formulation)
  - Absorbing state handling: once absorbed (death/discharge), no further
    reward accumulates; terminal reward assigned at absorbing step.

Protocol reference: Step 70d (v3.1), SAP 03f

Inputs
------
- data/interim/45d_outcomes_absorb_v1_mimic.parquet
- configs/00f_global_config_registry.yaml

Outputs
-------
- configs/70d_rewards_metrics_v1.yaml
- reports/70d_rewards_metrics_summary.txt

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
    reports_dir = resolve_path(proj_cfg, "reports")

    print("=" * 72)
    print("70d — Define Rewards and Metrics")
    print("=" * 72)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 1. Load outcomes for distribution summary ─────────────────────────
    print("\n[1] Loading outcomes...")
    outcomes_df = pd.read_parquet(interim_dir / "45d_outcomes_absorb_v1_mimic.parquet")
    n_stays = outcomes_df["stay_id"].nunique()

    mort_28d = outcomes_df["mortality_28d"].mean()
    mort_7d = outcomes_df["mortality_7d"].mean()
    print(f"  {n_stays:,} stays")
    print(f"  28-day mortality: {mort_28d:.4f} ({mort_28d*100:.1f}%)")
    print(f"  7-day mortality:  {mort_7d:.4f} ({mort_7d*100:.1f}%)")

    # Absorbing state distribution
    if "absorbing_state" in outcomes_df.columns:
        abs_dist = outcomes_df["absorbing_state"].value_counts()
        print(f"\n  Absorbing states (within 24h horizon):")
        for state, count in abs_dist.items():
            pct = count / n_stays * 100
            print(f"    {state}: {count:,} ({pct:.1f}%)")

    # ── 2. Define reward config ───────────────────────────────────────────
    print("\n[2] Defining reward configuration...")

    reward_cfg = {
        "step": "70d_define_rewards_and_metrics",
        "date": ts,
        "protocol_version": "v3.1",
        "reward_formulation": {
            "type": "sparse_terminal",
            "description": (
                "Sparse reward: R_t=0 for non-terminal steps. "
                "Terminal reward at absorbing step or end of horizon."
            ),
            "primary": {
                "name": "neg_mortality_28d",
                "column": "mortality_28d",
                "transform": "R = -1 * mortality_28d",
                "range": [-1, 0],
                "interpretation": "0 = survival, -1 = death at 28 days",
            },
            "secondary": [
                {
                    "name": "neg_mortality_7d",
                    "column": "mortality_7d",
                    "transform": "R = -1 * mortality_7d",
                    "range": [-1, 0],
                },
            ],
            "discount_factor": 1.0,
            "discount_note": (
                "gamma=1.0 because horizon is finite (12 bins = 24h) "
                "and reward is terminal-only. No discounting needed."
            ),
        },
        "absorbing_state_handling": {
            "death": {
                "terminal": True,
                "reward": -1.0,
                "post_absorption": "no further actions or rewards",
            },
            "discharge": {
                "terminal": True,
                "reward": "determined by 28-day outcome (ascertained post-ICU)",
                "post_absorption": "no further actions",
            },
            "end_of_horizon": {
                "terminal": True,
                "reward": "determined by 28-day outcome",
            },
        },
        "evaluation_metrics": {
            "primary": {
                "name": "weighted_mean_outcome",
                "description": (
                    "Mean potential outcome under evaluated policy, "
                    "estimated via IPTW or g-computation."
                ),
                "estimands": [
                    {
                        "name": "ATE",
                        "description": (
                            "Average Treatment Effect: E[Y(pi)] - E[Y(pi_obs)]"
                        ),
                    },
                    {
                        "name": "policy_value",
                        "description": "V(pi) = E[sum_t gamma^t R_t | pi]",
                    },
                ],
            },
            "secondary": [
                {
                    "name": "mortality_risk_ratio",
                    "description": "RR of 28-day mortality: policy vs observed",
                },
                {
                    "name": "mortality_risk_difference",
                    "description": "RD of 28-day mortality: policy - observed",
                },
            ],
            "ope_metrics": [
                {
                    "name": "WIS",
                    "description": "Weighted Importance Sampling estimator",
                },
                {
                    "name": "DR",
                    "description": "Doubly-Robust estimator",
                },
                {
                    "name": "FQE",
                    "description": "Fitted Q-Evaluation (for deep RL policies)",
                },
            ],
        },
        "baseline_rates": {
            "mortality_28d": round(float(mort_28d), 4),
            "mortality_7d": round(float(mort_7d), 4),
            "n_stays": int(n_stays),
        },
    }

    cfg_path = configs_dir / "70d_rewards_metrics_v1.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(reward_cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  Saved: {cfg_path}")

    # ── 3. Summary report ─────────────────────────────────────────────────
    report_path = reports_dir / "70d_rewards_metrics_summary.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "=" * 72,
        "70d — Rewards and Evaluation Metrics (MIMIC-IV)",
        "=" * 72,
        f"Generated: {ts}",
        "",
        "REWARD FORMULATION",
        "──────────────────",
        "  Type: sparse terminal",
        "  Primary: R = -1 * mortality_28d (0=survival, -1=death)",
        "  Secondary: R = -1 * mortality_7d",
        "  Discount: gamma = 1.0 (finite horizon, terminal reward)",
        "",
        "ABSORBING STATES",
        "────────────────",
        "  Death:     terminal, R = -1.0",
        "  Discharge: terminal, outcome ascertained at 28 days",
        "  Horizon:   terminal at bin 12 (24h), outcome at 28 days",
        "",
        "BASELINE RATES",
        "──────────────",
        f"  28-day mortality: {mort_28d:.4f} ({mort_28d*100:.1f}%)",
        f"  7-day mortality:  {mort_7d:.4f} ({mort_7d*100:.1f}%)",
        f"  Stays: {n_stays:,}",
        "",
        "EVALUATION METRICS",
        "──────────────────",
        "  Primary: ATE (policy vs observed), V(pi) policy value",
        "  Secondary: mortality RR, mortality RD",
        "  OPE: WIS, Doubly-Robust, FQE",
        "",
        "NEXT STEPS",
        "──────────",
        "  70e: define OPE configuration",
        "  70f: define trimming/support grid",
    ]

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Report: {report_path}")
    print("\n70d COMPLETE.")


if __name__ == "__main__":
    main()
