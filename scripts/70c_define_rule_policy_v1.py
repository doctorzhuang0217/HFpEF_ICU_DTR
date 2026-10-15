#!/usr/bin/env python3
"""
70c_define_rule_policy_v1 — Reproducible rule-based hemodynamic policy.

Purpose
-------
Implement a clinically grounded rule-based treatment policy that serves as:
  1. An interpretable baseline for Step C policy evaluation.
  2. A sanity check for OPE methods (rule policy value should be plausible).
  3. A comparison target in Aim 2 (data-driven vs guideline-based DTR).

The rule policy maps current hemodynamic state to a deterministic action:

  Rule logic (per 2h bin):
    IF MAP < 65 AND lactate > 2.0:
        → fluid=high, vaso=high, diuretic=none  (action_idx=16)
    ELIF MAP < 65:
        → fluid=high, vaso=low, diuretic=none   (action_idx=14)
    ELIF lactate > 2.0:
        → fluid=low_moderate, vaso=low, diuretic=none (action_idx=8)
    ELIF fluid_balance > 3.0L AND SpO2 < 94:
        → fluid=none, vaso=none, diuretic=any    (action_idx=1)
    ELIF fluid_balance > 5.0L:
        → fluid=none, vaso=none, diuretic=any    (action_idx=1)
    ELSE:
        → fluid=low_moderate, vaso=none, diuretic=none (action_idx=6)

  Action encoding: fluid_level * 6 + vaso_level * 2 + diuretic_level

Protocol reference: Step 70c (v3.1)

Inputs
------
- data/interim/45c_state_tensors_v1_mimic.parquet
- configs/action_space_v1.yaml

Outputs
-------
- configs/70c_rule_policy_v1.yaml
- data/interim/70c_rule_policy_actions_v1_mimic.parquet
- reports/70c_rule_policy_summary.txt

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


# ── Action encoding: fluid * 6 + vaso * 2 + diuretic ─────────────────────
# fluid:    0=none, 1=low_moderate, 2=high
# vaso:     0=none, 1=low, 2=high
# diuretic: 0=none, 1=any
def _encode_action(fluid: int, vaso: int, diuretic: int) -> int:
    return fluid * 6 + vaso * 2 + diuretic


# Named actions for clarity
ACTION_RESUSCITATE_AGGRESSIVE = _encode_action(2, 2, 0)  # 16
ACTION_RESUSCITATE_FLUID      = _encode_action(2, 1, 0)  # 14  # noqa: E221
ACTION_PERFUSION_SUPPORT      = _encode_action(1, 1, 0)  # 8   # noqa: E221
ACTION_DECONGEST              = _encode_action(0, 0, 1)  # 1   # noqa: E221
ACTION_MAINTENANCE            = _encode_action(1, 0, 0)  # 6   # noqa: E221


# ── Scheme-aware action encoder (ADDITIVE; coarse_3 default unchanged) ────
#
# `apply_rule_policy()` decides categorically — it emits a (fluid_level,
# vaso_level, diuretic_level) intent on a 3-level fluid/vaso scale
# (0=none, 1=low/low_moderate, 2=high) and a binary diuretic (0/1). The
# rule encodes a clinical *intent level*, NOT a numeric dose, so encoding it
# under a finer scheme is an ordinal-alignment problem, not a re-binning of a
# measured dose.
#
# coarse_3 (DEFAULT, frozen): fluid*6 + vaso*2 + diuretic  — byte-identical to
#   the original `_encode_action`; produces the same 5 named actions
#   {16, 14, 8, 6, 1}. This is the encoding consumed by the frozen primary
#   coarse_3 chain and MUST NOT change.
#
# medium_4: the rule's 3-level categorical is mapped onto medium_4's 4-level
#   bins (read from configs/29a_candidate_action_schemes.yaml) by ORDINAL
#   ALIGNMENT of the intent level:
#     none            → bin 0
#     low/low_moderate → lowest NONZERO bin (bin 1)
#     high            → highest bin (last index)
#   for BOTH fluid and vaso; diuretic stays binary (0→0, 1→1). The encoded
#   index is then fluid_bin4*8 + vaso_bin4*2 + diuretic (matching the
#   medium_4 joint encoding used by 45b --scheme medium_4). With the medium_4
#   bins (none/minimal/moderate/aggressive for fluid, none/low/moderate/high
#   for vaso) the 5 named rule actions map to:
#     RESUSCITATE_AGGRESSIVE (f=2,v=2,d=0) → 3*8 + 3*2 + 0 = 30
#     RESUSCITATE_FLUID      (f=2,v=1,d=0) → 3*8 + 1*2 + 0 = 26
#     PERFUSION_SUPPORT      (f=1,v=1,d=0) → 1*8 + 1*2 + 0 = 10
#     MAINTENANCE            (f=1,v=0,d=0) → 1*8 + 0   + 0 = 8
#     DECONGEST              (f=0,v=0,d=1) → 0   + 0   + 1 = 1
#   i.e. 5 distinct medium_4 cells, preserving the ordinal ordering of the
#   rule intent (aggressive > fluid > perfusion ≈ maintenance > decongest).
#
# Rationale for "high → highest bin": the rule's "high" intent is the most
# aggressive level the rule can express; under a finer scheme it should map to
# the most aggressive available bin (not an intermediate one), so the
# counterfactual action under the rule is unambiguously the aggressive cell.
# Mapping "low" → lowest nonzero bin keeps a non-trivial but minimal exposure,
# matching the rule's "some, but not aggressive" intent.

# Number of cells along the fluid / vaso / diuretic axes per scheme. coarse_3
# is frozen at 3×3×2; others are read from 29a (length of each label list).
_COARSE3_AXES = {"fluid": 3, "vaso": 3, "diuretic": 2}


def _scheme_axis_bins(scheme: str, configs_dir: Path | None) -> dict:
    """Return {axis: n_bins} for fluid / vaso / diuretic for `scheme`.

    coarse_3 is frozen (3/3/2). Any other scheme is read from
    29a_candidate_action_schemes.yaml using the length of each label list.
    """
    if scheme == "coarse_3":
        return dict(_COARSE3_AXES)
    if configs_dir is None:
        raise ValueError(
            f"configs_dir required to resolve non-default scheme '{scheme}'")
    with open(configs_dir / "29a_candidate_action_schemes.yaml") as f:
        cand = yaml.safe_load(f)
    sch = cand["schemes"][scheme]
    return {
        "fluid": len(sch["iv_fluid_ml_per_2h"]["labels"]),
        "vaso": len(sch["vasopressor_ne_eq_mcg_kg_min"]["labels"]),
        "diuretic": len(sch["diuretic_furo_eq_mg_per_2h"]["labels"]),
    }


def _ordinal_align(level: int, n_bins: int) -> int:
    """Map a 3-level rule intent (0=none,1=low,2=high) to an `n_bins`-bin axis.

    none(0) → bin 0; low(1) → lowest nonzero bin (1); high(2) → highest bin
    (n_bins-1). For a 3-bin axis this is the identity (coarse_3-compatible).
    Diuretic axes (n_bins==2) keep the binary 0/1 mapping (1 → bin 1).
    """
    if level <= 0:
        return 0
    if level == 1:
        return 1  # lowest nonzero bin
    # level >= 2 ("high") → highest available bin
    return n_bins - 1


def encode_action(
    fluid_level: int,
    vaso_level: int,
    diuretic_level: int,
    scheme: str = "coarse_3",
    configs_dir: Path | None = None,
) -> int:
    """Scheme-aware encoder for a categorical rule decision.

    `apply_rule_policy()` decides categorically on the 3-level intent scale
    (fluid/vaso ∈ {0=none,1=low,2=high}, diuretic ∈ {0,1}); this maps that
    decision to the joint action index of `scheme`.

    coarse_3 (DEFAULT): fluid*6 + vaso*2 + diuretic — byte-identical to
      `_encode_action` (the frozen primary encoding; do not change).
    medium_4 (and any finer scheme): ordinal-aligns fluid/vaso onto the
      scheme's bin count (see module docstring) then encodes
      fluid_bin*(vaso_n*diuretic_n) + vaso_bin*diuretic_n + diuretic.
    """
    if scheme == "coarse_3":
        # Frozen path: identical to the original encoding (3/3/2 layout).
        return _encode_action(fluid_level, vaso_level, diuretic_level)

    axes = _scheme_axis_bins(scheme, configs_dir)
    n_fluid, n_vaso, n_diur = axes["fluid"], axes["vaso"], axes["diuretic"]
    f_bin = _ordinal_align(int(fluid_level), n_fluid)
    v_bin = _ordinal_align(int(vaso_level), n_vaso)
    d_bin = 0 if int(diuretic_level) <= 0 else min(1, n_diur - 1)
    # Joint encoding: fluid is most significant, then vaso, then diuretic —
    # consistent with the 45b --scheme <s> joint action layout.
    return f_bin * (n_vaso * n_diur) + v_bin * n_diur + d_bin


# ── Categorical decode of the frozen coarse_3 rule actions ───────────────
# `apply_rule_policy()` returns coarse_3-encoded ints (byte-identical to the
# original behaviour, so all existing coarse_3 callers are unaffected). To
# re-encode under another scheme we recover the categorical (fluid, vaso,
# diuretic) intent that produced each coarse_3 action via this fixed map.
_COARSE3_ACTION_TO_LEVELS = {
    ACTION_RESUSCITATE_AGGRESSIVE: (2, 2, 0),
    ACTION_RESUSCITATE_FLUID:      (2, 1, 0),  # noqa: E241
    ACTION_PERFUSION_SUPPORT:      (1, 1, 0),  # noqa: E241
    ACTION_DECONGEST:              (0, 0, 1),  # noqa: E241
    ACTION_MAINTENANCE:            (1, 0, 0),  # noqa: E241
}


def coarse3_action_to_levels(action_idx: int) -> tuple[int, int, int]:
    """Recover (fluid_level, vaso_level, diuretic_level) from a coarse_3 idx.

    Defaults to MAINTENANCE levels for any unexpected index (the rule only
    ever emits the 5 named actions). Used to re-encode the rule decision under
    a non-default scheme without changing `apply_rule_policy`'s coarse_3 output.
    """
    return _COARSE3_ACTION_TO_LEVELS.get(
        int(action_idx), _COARSE3_ACTION_TO_LEVELS[ACTION_MAINTENANCE])


def reencode_rule_action(action_idx: int, scheme: str,
                         configs_dir: Path | None = None) -> int:
    """Re-encode a coarse_3 rule action index under `scheme`.

    coarse_3 → identity (byte-identical). Other schemes → decode to the
    categorical intent then re-encode via `encode_action`.
    """
    if scheme == "coarse_3":
        return int(action_idx)
    f, v, d = coarse3_action_to_levels(action_idx)
    return encode_action(f, v, d, scheme=scheme, configs_dir=configs_dir)


def _load_norm_params(configs_dir: Path) -> dict:
    """Load exact frozen normalization parameters from Step 45c."""
    norm_path = configs_dir / "45c_normalization_params_v1.yaml"
    with open(norm_path) as f:
        cfg = yaml.safe_load(f)
    params = cfg.get("params", {})
    required = ["map", "lactate", "spo2"]
    missing = [k for k in required if k not in params]
    assert not missing, f"Missing normalization params for: {missing}"
    return params


def apply_rule_policy(state_df: pd.DataFrame, norm_params: dict) -> pd.Series:
    """
    Apply rule-based policy to each observation row.

    Expects columns: map_z (or map raw), lactate_z (or lactate raw),
    fluid_balance_cumulative_L, spo2_z (or spo2 raw).

    Falls back to z-scored columns with threshold conversion using
    frozen normalization parameters from 45c.
    """
    n = len(state_df)
    actions = np.full(n, ACTION_MAINTENANCE, dtype=int)

    # Try to use raw or z-scored columns
    # MAP thresholds
    if "map" in state_df.columns:
        map_vals = state_df["map"].values
    elif "map_z" in state_df.columns:
        map_mean = float(norm_params["map"]["mean"])
        map_std = float(norm_params["map"]["std"])
        map_vals = state_df["map_z"].values * map_std + map_mean
    else:
        map_vals = np.full(n, 80.0)  # default safe value

    if "lactate" in state_df.columns:
        lac_vals = state_df["lactate"].values
    elif "lactate_z" in state_df.columns:
        lac_mean = float(norm_params["lactate"]["mean"])
        lac_std = float(norm_params["lactate"]["std"])
        lac_vals = state_df["lactate_z"].values * lac_std + lac_mean
    else:
        lac_vals = np.full(n, 1.5)

    # Fluid balance (cumulative, in L)
    fb_col = None
    for candidate in ["fluid_balance_cumulative_L", "fluid_balance_L",
                       "cum_fluid_balance_L"]:
        if candidate in state_df.columns:
            fb_col = candidate
            break
    fb_vals = state_df[fb_col].values if fb_col else np.zeros(n)

    if "spo2" in state_df.columns:
        spo2_vals = state_df["spo2"].values
    elif "spo2_z" in state_df.columns:
        spo2_mean = float(norm_params["spo2"]["mean"])
        spo2_std = float(norm_params["spo2"]["std"])
        spo2_vals = state_df["spo2_z"].values * spo2_std + spo2_mean
    else:
        spo2_vals = np.full(n, 97.0)

    # Apply rules in priority order (highest priority first)
    # Rule 1: Shock + hypoperfusion → aggressive resuscitation
    mask_shock_perf = (map_vals < 65) & (lac_vals > 2.0)
    actions[mask_shock_perf] = ACTION_RESUSCITATE_AGGRESSIVE

    # Rule 2: Hypotension only → fluid + moderate vasopressor
    mask_hypo = (map_vals < 65) & ~mask_shock_perf
    actions[mask_hypo] = ACTION_RESUSCITATE_FLUID

    # Rule 3: Hypoperfusion without hypotension → perfusion support
    mask_perf = (lac_vals > 2.0) & (map_vals >= 65)
    actions[mask_perf] = ACTION_PERFUSION_SUPPORT

    # Rule 4: Volume overload + hypoxemia → decongest
    mask_overload_hypox = (fb_vals > 3.0) & (spo2_vals < 94) & \
                          (map_vals >= 65) & (lac_vals <= 2.0)
    actions[mask_overload_hypox] = ACTION_DECONGEST

    # Rule 5: Severe volume overload → decongest
    mask_severe_overload = (fb_vals > 5.0) & (map_vals >= 65) & \
                           (lac_vals <= 2.0) & ~mask_overload_hypox
    actions[mask_severe_overload] = ACTION_DECONGEST

    # Rule 6: Default → maintenance fluids
    # (already set as default)

    return pd.Series(actions, index=state_df.index, name="rule_policy_action")


def main() -> None:
    proj_cfg = load_config()
    interim_dir = resolve_path(proj_cfg, "interim")
    configs_dir = resolve_path(proj_cfg, "configs")
    reports_dir = resolve_path(proj_cfg, "reports")

    print("=" * 72)
    print("70c — Define Rule-Based Policy v1")
    print("=" * 72)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 1. Load state tensors + observed actions ────────────────────────
    print("\n[1] Loading state tensors...")
    state_df = pd.read_parquet(interim_dir / "45c_state_tensors_v1_mimic.parquet")
    n_obs = len(state_df)
    n_stays = state_df["stay_id"].nunique()
    print(f"  {n_obs:,} obs, {n_stays:,} stays")

    # action_idx lives in 45b (not in 45c state tensors)
    actions_df = pd.read_parquet(
        interim_dir / "45b_actions_2h_v1_mimic.parquet",
        columns=["stay_id", "bin", "action_idx"],
    )
    state_df = state_df.merge(actions_df, on=["stay_id", "bin"], how="left")
    print(f"  Merged action_idx: {state_df['action_idx'].notna().sum():,} / {n_obs:,}")

    # ── 2. Apply rule policy ──────────────────────────────────────────────
    print("\n[2] Applying rule policy...")
    norm_params = _load_norm_params(configs_dir)
    rule_actions = apply_rule_policy(state_df, norm_params)

    # ── 3. Compare with observed actions ──────────────────────────────────
    print("\n[3] Comparing rule vs observed actions...")
    obs_actions = state_df["action_idx"].values
    agreement = (rule_actions.values == obs_actions).mean()
    print(f"  Rule-observed agreement: {agreement:.4f} ({agreement*100:.1f}%)")

    # Distribution of rule policy actions
    rule_counts = rule_actions.value_counts().sort_index()
    print("\n  Rule policy action distribution:")
    action_names = {
        ACTION_RESUSCITATE_AGGRESSIVE: "resuscitate_aggressive (f=high,v=high,d=none)",
        ACTION_RESUSCITATE_FLUID: "resuscitate_fluid (f=high,v=low,d=none)",
        ACTION_PERFUSION_SUPPORT: "perfusion_support (f=low_mod,v=low,d=none)",
        ACTION_DECONGEST: "decongest (f=none,v=none,d=any)",
        ACTION_MAINTENANCE: "maintenance (f=low_mod,v=none,d=none)",
    }
    for aidx, count in rule_counts.items():
        name = action_names.get(aidx, f"action_{aidx}")
        pct = count / n_obs * 100
        print(f"    {aidx:2d} ({name}): {count:,} ({pct:.1f}%)")

    # ── 4. Save rule policy config ────────────────────────────────────────
    print("\n[4] Saving rule policy config...")
    rule_cfg = {
        "step": "70c_define_rule_policy_v1",
        "date": ts,
        "version": "v1",
        "description": "Hemodynamic-guided rule-based treatment policy",
        "action_encoding": "fluid_level * 6 + vaso_level * 2 + diuretic_level",
        "action_encoding_note": (
            "Encoding above is the frozen coarse_3 (3x3x2) scheme. The rule "
            "DECIDES categorically on a 3-level intent scale "
            "(fluid/vaso in {0=none,1=low,2=high}, diuretic in {0,1}); "
            "encode_action(..., scheme) in 70c re-encodes that decision for "
            "other schemes by ordinal alignment "
            "(none->bin0, low->lowest nonzero bin, high->highest bin). "
            "medium_4 maps the 5 named actions to cells {30,26,10,8,1}."
        ),
        "rules": [
            {
                "priority": 1,
                "name": "shock_plus_hypoperfusion",
                "condition": "MAP < 65 AND lactate > 2.0",
                "action": "fluid=high, vaso=high, diuretic=none",
                "action_idx": ACTION_RESUSCITATE_AGGRESSIVE,
            },
            {
                "priority": 2,
                "name": "hypotension_only",
                "condition": "MAP < 65 AND lactate <= 2.0",
                "action": "fluid=high, vaso=low, diuretic=none",
                "action_idx": ACTION_RESUSCITATE_FLUID,
            },
            {
                "priority": 3,
                "name": "hypoperfusion_only",
                "condition": "lactate > 2.0 AND MAP >= 65",
                "action": "fluid=low_moderate, vaso=low, diuretic=none",
                "action_idx": ACTION_PERFUSION_SUPPORT,
            },
            {
                "priority": 4,
                "name": "volume_overload_hypoxemia",
                "condition": "fluid_balance > 3.0L AND SpO2 < 94%",
                "action": "fluid=none, vaso=none, diuretic=any",
                "action_idx": ACTION_DECONGEST,
            },
            {
                "priority": 5,
                "name": "severe_volume_overload",
                "condition": "fluid_balance > 5.0L",
                "action": "fluid=none, vaso=none, diuretic=any",
                "action_idx": ACTION_DECONGEST,
            },
            {
                "priority": 6,
                "name": "default_maintenance",
                "condition": "otherwise",
                "action": "fluid=low_moderate, vaso=none, diuretic=none",
                "action_idx": ACTION_MAINTENANCE,
            },
        ],
        "thresholds": {
            "map_hypotension": 65,
            "lactate_hypoperfusion": 2.0,
            "fluid_balance_overload_L": 3.0,
            "fluid_balance_severe_overload_L": 5.0,
            "spo2_hypoxemia": 94,
        },
        "normalization_used": {
            "map": {
                "mean": float(norm_params["map"]["mean"]),
                "std": float(norm_params["map"]["std"]),
            },
            "lactate": {
                "mean": float(norm_params["lactate"]["mean"]),
                "std": float(norm_params["lactate"]["std"]),
            },
            "spo2": {
                "mean": float(norm_params["spo2"]["mean"]),
                "std": float(norm_params["spo2"]["std"]),
            },
        },
    }

    rule_cfg_path = configs_dir / "70c_rule_policy_v1.yaml"
    with open(rule_cfg_path, "w") as f:
        yaml.dump(rule_cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  Saved: {rule_cfg_path}")

    # ── 5. Save rule policy actions ───────────────────────────────────────
    print("\n[5] Saving rule policy actions...")
    out_df = state_df[["stay_id", "bin", "action_idx"]].copy()
    out_df["rule_policy_action"] = rule_actions.values
    out_df["rule_agrees_with_observed"] = (
        out_df["action_idx"] == out_df["rule_policy_action"]
    )

    out_path = interim_dir / "70c_rule_policy_actions_v1_mimic.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path}")

    # ── 6. Agreement by bin (time trend) ──────────────────────────────────
    print("\n[6] Agreement by bin:")
    bin_agreement = out_df.groupby("bin")["rule_agrees_with_observed"].mean()
    for b, ag in bin_agreement.items():
        print(f"    Bin {b:2d}: {ag:.4f} ({ag*100:.1f}%)")

    # ── 7. Summary report ─────────────────────────────────────────────────
    report_path = reports_dir / "70c_rule_policy_summary.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "=" * 72,
        "70c — Rule-Based Policy v1 (MIMIC-IV)",
        "=" * 72,
        f"Generated: {ts}",
        "",
        "DATA",
        "────",
        f"  Observations: {n_obs:,} ({n_stays:,} stays × 12 bins)",
        "",
        "RULE POLICY LOGIC",
        "─────────────────",
    ]
    for rule in rule_cfg["rules"]:
        lines.append(f"  P{rule['priority']}: {rule['name']}")
        lines.append(f"      IF {rule['condition']}")
        lines.append(f"      → {rule['action']} (idx={rule['action_idx']})")

    lines.extend([
        "",
        "ACTION DISTRIBUTION (rule policy)",
        "──────────────────────────────────",
    ])
    for aidx, count in rule_counts.items():
        name = action_names.get(aidx, f"action_{aidx}")
        pct = count / n_obs * 100
        lines.append(f"  {aidx:2d} ({name}): {count:,} ({pct:.1f}%)")

    lines.extend([
        "",
        "RULE vs OBSERVED AGREEMENT",
        "──────────────────────────",
        f"  Overall: {agreement:.4f} ({agreement*100:.1f}%)",
        "",
        "  By bin:",
    ])
    for b, ag in bin_agreement.items():
        lines.append(f"    Bin {b:2d}: {ag:.4f}")

    lines.extend([
        "",
        "NOTE",
        "────",
        "  This rule policy is a reproducible baseline for OPE comparison.",
        "  It is NOT intended as a clinical recommendation.",
        "  Step 75a/80c will evaluate data-driven policies against this baseline.",
        "",
        "NEXT STEPS",
        "──────────",
        "  70d: define rewards and metrics",
        "  75a: evaluate rule policy via IPTW (sanity check)",
    ])

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Report: {report_path}")
    print("\n70c COMPLETE.")


if __name__ == "__main__":
    main()
