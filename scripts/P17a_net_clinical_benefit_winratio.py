#!/usr/bin/env python3
"""P17a — Net-clinical-benefit / win-ratio (pattern-count rewrite).

Protocol: v3.1-patch, enhancement analysis B (see
the supplementary-analysis specification, section P17a; original framing in
row B of the same specification).
Compute tier: C2-C3 (CPU only).
Depends on: 45d outcomes (mortality_28d → dead_28d), 10h vent indicators
            (IMV escalation), 10e labs (creatinine → AKI progression),
            70c rule-policy concordance, phenotype_labels_frozen.

Objective
---------
Hierarchical win-ratio (Pocock-style) net-clinical-benefit comparison of the
rule-concordance TOP tertile vs the BOTTOM tertile, over a 3-component ordered
outcome (death > IMV escalation > AKI progression; lower event = better). The
directional tension between the favorable mortality signal and the adverse
IMV/AKI signals is reported honestly: the win-ratio aggregates all three under
a pre-specified clinical hierarchy, and a separate UNADJUSTED composite-event
(any-of-three) risk difference is reported alongside.

Algorithm (pattern-count, O(P^2); replaces the removed O(B·n^2) pairwise
version that was infeasible at ~10k stays)
------------------------------------------------------------------------------
1. Per-stay outcome vector:
     dead_28d        ∈ {0, 1}        (from 45d mortality_28d)
     imv_escalation  ∈ {0, 1}        (P1 build_imv_escalation)
     aki_progression ∈ {0, 1, NaN}   (P1 build_aki_progression; NaN = no paired
                                       creatinine → AKI tier is unknown)
2. Enumerate the 12 outcome patterns (2 death × 2 IMV × 3 AKI{0,1,missing});
   map each stay → pattern index; per-group pattern counts c[12].
3. Precompute a 12×12 hierarchical win matrix W (compare death first, then IMV,
   then AKI; lower event = better; AKI-missing → tie on the AKI tier).
   W[i, j] ∈ {+1 = i wins over j, −1 = i loses, 0 = tie}.
4. Win-ratio (group A = TOP tertile vs group B = BOTTOM tertile):
     wins   = Σ_ij c_a[i] · c_b[j] · 1[W[i,j] = +1]
     losses = Σ_ij c_a[i] · c_b[j] · 1[W[i,j] = −1]
     WR = wins / losses          (>1 ⇒ TOP-tertile favored)
   Composite-event (any-of-three) risk difference = UNADJUSTED top − bottom RD
   (clearly labeled; NO IPTW / WIS adjustment claim).
5. Bootstrap B=500: MULTINOMIAL resample of each group's pattern counts
   (rng.multinomial(n_g, c_g / n_g)), recompute WR + composite RD → percentile
   95% CIs. O(B·P^2) is trivial.
6. Strata: overall / HFpEF (hf_subtype.str.lower() == 'hfpef') / no_hf.

Honest labeling
---------------
The output reports a win-ratio (hierarchical, all three components) and an
UNADJUSTED composite RD. No IPTW / WIS adjustment is claimed for either. The
WIS rule-policy means remain available in P1; they are referenced here only in
an optional interpretive text note, not in the CSV columns.

Outputs
-------
- tables/P17a_net_clinical_benefit_winratio.csv
- reports/P17a_net_clinical_benefit_winratio_summary.txt
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Project root / src on path ──
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hfpef_dtr.config import load_config, resolve_path  # noqa: E402

# ── Constants ──
BOOTSTRAP_B = 500
BOOTSTRAP_SEED = 2024

# Outcome-pattern enumeration.
#   death ∈ {0, 1}            (2 levels)
#   imv   ∈ {0, 1}            (2 levels)
#   aki   ∈ {0, 1, missing}   (3 levels; missing encoded as the sentinel below)
AKI_LEVELS = (0, 1, "missing")          # ordered tiers for AKI
N_PATTERNS = 2 * 2 * len(AKI_LEVELS)    # = 12


# ── P1-as-library loader (mirrors P16a's _load_75a_module) ──────────────
def _load_p1_module():
    """Import P1_secondary_outcomes_iptw.py as a module to reuse its builders.

    Reuses build_imv_escalation / build_aki_progression / compute_rule_concordance.
    """
    spec = importlib.util.spec_from_file_location(
        "_step_p1_secondary",
        Path(__file__).resolve().parent / "P1_secondary_outcomes_iptw.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ══════════════════════════════════════════════════════════════════════════
# Pattern enumeration + hierarchical win matrix
# ══════════════════════════════════════════════════════════════════════════

def enumerate_patterns() -> list[tuple[int, int, object]]:
    """Return the 12 (death, imv, aki) outcome patterns in a fixed order.

    AKI tier is one of {0, 1, 'missing'}.
    """
    patterns: list[tuple[int, int, object]] = []
    for death in (0, 1):
        for imv in (0, 1):
            for aki in AKI_LEVELS:
                patterns.append((death, imv, aki))
    assert len(patterns) == N_PATTERNS
    return patterns


def pattern_index(death: int, imv: int, aki: object) -> int:
    """Map a (death, imv, aki) triple to its pattern index 0..11."""
    aki_pos = AKI_LEVELS.index(aki)
    return death * (2 * len(AKI_LEVELS)) + imv * len(AKI_LEVELS) + aki_pos


def _compare_tier(a: object, b: object) -> int:
    """Compare one ordered tier for two stays; lower event = better.

    Returns +1 if a is better (lower), −1 if a is worse (higher), 0 if a tie
    OR if the comparison cannot be resolved (missing on either side → tie on
    THIS tier, per spec: AKI-missing → tie on the AKI tier).
    """
    if a == "missing" or b == "missing":
        return 0  # unresolved on this tier → tie, fall through to next tier
    if a < b:
        return +1  # lower event = better
    if a > b:
        return -1
    return 0


def _win_outcome(pi: tuple[int, int, object], pj: tuple[int, int, object]) -> int:
    """Hierarchical win comparison of pattern i vs pattern j.

    Compare death first, then IMV, then AKI; lower event = better. The first
    tier that resolves (non-tie) decides; if all tiers tie, it is an overall tie.
    AKI-missing → tie on the AKI tier (handled in _compare_tier).
    Returns +1 (i wins), −1 (i loses), 0 (tie).
    """
    di, ii, ai = pi
    dj, ij, aj = pj
    # Tier 1: death (no missingness — always resolvable).
    r = _compare_tier(di, dj)
    if r != 0:
        return r
    # Tier 2: IMV escalation.
    r = _compare_tier(ii, ij)
    if r != 0:
        return r
    # Tier 3: AKI progression (missing → tie).
    return _compare_tier(ai, aj)


def build_win_matrix() -> tuple[np.ndarray, list[tuple[int, int, object]]]:
    """Precompute the 12×12 hierarchical win matrix W and the pattern list.

    W[i, j] ∈ {+1, −1, 0}: outcome of pattern i compared against pattern j.
    """
    patterns = enumerate_patterns()
    W = np.zeros((N_PATTERNS, N_PATTERNS), dtype=np.int8)
    for i in range(N_PATTERNS):
        for j in range(N_PATTERNS):
            W[i, j] = _win_outcome(patterns[i], patterns[j])
    return W, patterns


# ══════════════════════════════════════════════════════════════════════════
# Pattern counts + win-ratio + composite RD
# ══════════════════════════════════════════════════════════════════════════

def pattern_counts(
    sub: pd.DataFrame,
    patterns: list[tuple[int, int, object]],
) -> np.ndarray:
    """Per-group pattern counts c[12] for a stay-level DataFrame.

    `sub` must have integer columns 'dead_28d', 'imv_escalation' and an
    'aki_progression' column in {0.0, 1.0, NaN}. NaN AKI maps to the
    'missing' tier.
    """
    counts = np.zeros(N_PATTERNS, dtype=np.int64)
    for _, row in sub.iterrows():
        death = int(row["dead_28d"])
        imv = int(row["imv_escalation"])
        aki_raw = row["aki_progression"]
        if pd.isna(aki_raw):
            aki: object = "missing"
        else:
            aki = int(aki_raw)
        counts[pattern_index(death, imv, aki)] += 1
    return counts


def win_ratio_from_counts(
    c_a: np.ndarray,
    c_b: np.ndarray,
    W: np.ndarray,
) -> tuple[float, int, int, int]:
    """Win-ratio between group A (c_a) and group B (c_b) via pattern counts.

    wins   = Σ_ij c_a[i] c_b[j] 1[W=+1]
    losses = Σ_ij c_a[i] c_b[j] 1[W=−1]
    ties   = Σ_ij c_a[i] c_b[j] 1[W= 0]
    WR = wins / losses (NaN if losses == 0).
    O(P^2). Counts are computed with the full outer product, then masked.
    """
    outer = np.outer(c_a, c_b)  # outer[i, j] = c_a[i] * c_b[j]
    wins = int(outer[W == 1].sum())
    losses = int(outer[W == -1].sum())
    ties = int(outer[W == 0].sum())
    wr = float(wins / losses) if losses > 0 else float("nan")
    return wr, wins, losses, ties


def composite_event_rate_from_counts(
    c: np.ndarray,
    patterns: list[tuple[int, int, object]],
) -> float:
    """UNADJUSTED composite-event (any-of-three) rate from pattern counts.

    Composite event = death OR IMV escalation OR AKI progression.
    AKI-missing is NOT counted as an event (only known AKI==1 contributes); a
    stay with death==0, imv==0, aki=='missing' is therefore a non-event.
    Denominator = total stays in the group (all patterns).
    """
    n = int(c.sum())
    if n == 0:
        return float("nan")
    n_event = 0
    for idx, (death, imv, aki) in enumerate(patterns):
        is_event = (death == 1) or (imv == 1) or (aki == 1)
        if is_event:
            n_event += int(c[idx])
    return float(n_event / n)


# ══════════════════════════════════════════════════════════════════════════
# Bootstrap (multinomial resample of pattern counts)
# ══════════════════════════════════════════════════════════════════════════

def bootstrap_winratio_composite(
    c_a: np.ndarray,
    c_b: np.ndarray,
    W: np.ndarray,
    patterns: list[tuple[int, int, object]],
    B: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
    desc: str = "bootstrap",
) -> dict:
    """Multinomial-resample bootstrap of WR and composite RD (top − bottom).

    For each replicate, resample each group's pattern counts via
    rng.multinomial(n_g, c_g / n_g), then recompute WR and the composite RD.
    Returns percentile 95% CIs. O(B·P^2), trivial.
    """
    n_a = int(c_a.sum())
    n_b = int(c_b.sum())
    if n_a == 0 or n_b == 0:
        return {
            "wr_ci_lo": float("nan"), "wr_ci_hi": float("nan"),
            "composite_rd_ci_lo": float("nan"), "composite_rd_ci_hi": float("nan"),
        }
    rng = np.random.RandomState(seed)
    p_a = c_a / n_a
    p_b = c_b / n_b
    boot_wr = np.full(B, np.nan)
    boot_rd = np.full(B, np.nan)
    for b in tqdm(range(B), desc=f"    {desc}", mininterval=2.0, leave=False):
        cb_a = rng.multinomial(n_a, p_a)
        cb_b = rng.multinomial(n_b, p_b)
        wr_b, _, _, _ = win_ratio_from_counts(cb_a, cb_b, W)
        boot_wr[b] = wr_b
        rate_a = composite_event_rate_from_counts(cb_a, patterns)
        rate_b = composite_event_rate_from_counts(cb_b, patterns)
        boot_rd[b] = rate_a - rate_b
    return {
        "wr_ci_lo": float(np.nanpercentile(boot_wr, 2.5)),
        "wr_ci_hi": float(np.nanpercentile(boot_wr, 97.5)),
        "composite_rd_ci_lo": float(np.nanpercentile(boot_rd, 2.5)),
        "composite_rd_ci_hi": float(np.nanpercentile(boot_rd, 97.5)),
    }


# ══════════════════════════════════════════════════════════════════════════
# Per-stratum analysis
# ══════════════════════════════════════════════════════════════════════════

def analyze_stratum(
    subgroup_name: str,
    df: pd.DataFrame,
    W: np.ndarray,
    patterns: list[tuple[int, int, object]],
    B: int,
    seed: int,
) -> dict:
    """Win-ratio + composite RD for one stratum (TOP tertile vs BOTTOM tertile).

    `df` is the per-stay frame restricted to this stratum, with columns
    dead_28d, imv_escalation, aki_progression, concordance_tertile.
    """
    top = df[df["concordance_tertile"] == "top"]
    bot = df[df["concordance_tertile"] == "bottom"]
    n_top = len(top)
    n_bot = len(bot)

    base = {
        "comparison": "rule_concordance_top_vs_bottom_tertile",
        "subgroup": subgroup_name,
        "n": len(df),
        "n_top_tertile": n_top,
        "n_bot_tertile": n_bot,
    }

    if n_top == 0 or n_bot == 0:
        base.update({
            "win_ratio": np.nan, "wr_ci_lo": np.nan, "wr_ci_hi": np.nan,
            "n_wins": 0, "n_losses": 0, "n_ties": 0,
            "composite_unadj_rd": np.nan,
            "composite_unadj_rd_ci_lo": np.nan,
            "composite_unadj_rd_ci_hi": np.nan,
            "note": "empty top or bottom tertile",
        })
        return base

    c_top = pattern_counts(top, patterns)
    c_bot = pattern_counts(bot, patterns)

    wr, wins, losses, ties = win_ratio_from_counts(c_top, c_bot, W)
    rate_top = composite_event_rate_from_counts(c_top, patterns)
    rate_bot = composite_event_rate_from_counts(c_bot, patterns)
    composite_rd = rate_top - rate_bot

    boot = bootstrap_winratio_composite(
        c_top, c_bot, W, patterns, B=B, seed=seed,
        desc=f"{subgroup_name} WR/RD",
    )

    base.update({
        "win_ratio": round(wr, 4) if np.isfinite(wr) else np.nan,
        "wr_ci_lo": round(boot["wr_ci_lo"], 4) if np.isfinite(boot["wr_ci_lo"]) else np.nan,
        "wr_ci_hi": round(boot["wr_ci_hi"], 4) if np.isfinite(boot["wr_ci_hi"]) else np.nan,
        "n_wins": wins,
        "n_losses": losses,
        "n_ties": ties,
        "composite_unadj_rd": round(composite_rd, 4) if np.isfinite(composite_rd) else np.nan,
        "composite_unadj_rd_ci_lo": round(boot["composite_rd_ci_lo"], 4)
            if np.isfinite(boot["composite_rd_ci_lo"]) else np.nan,
        "composite_unadj_rd_ci_hi": round(boot["composite_rd_ci_hi"], 4)
            if np.isfinite(boot["composite_rd_ci_hi"]) else np.nan,
    })
    return base


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="P17a — pattern-count hierarchical win-ratio + UNADJUSTED "
                    "composite RD (rule-concordance top vs bottom tertile)."
    )
    parser.add_argument(
        "--bootstrap", type=int, default=BOOTSTRAP_B,
        help=f"Bootstrap replicates (default {BOOTSTRAP_B}).",
    )
    parser.add_argument(
        "--seed", type=int, default=BOOTSTRAP_SEED,
        help=f"Bootstrap RNG seed (default {BOOTSTRAP_SEED}).",
    )
    args = parser.parse_args()

    ts_start = datetime.datetime.now()
    cfg = load_config()
    interim_dir = resolve_path(cfg, "interim")
    staging_dir = resolve_path(cfg, "staging")
    processed_dir = resolve_path(cfg, "processed")
    tables_dir = resolve_path(cfg, "tables")
    reports_dir = resolve_path(cfg, "reports")
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("P17a — Net-clinical-benefit / win-ratio (pattern-count rewrite)")
    print("=" * 72)
    print(f"Started: {ts_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Bootstrap: B={args.bootstrap}, seed={args.seed}")

    # ── Reuse P1 builders ──
    p1 = _load_p1_module()

    # ══════════════════════════════════════════════════════════════════════
    # 1. Load shared data
    # ══════════════════════════════════════════════════════════════════════
    print("\n[1/5] Loading shared data ...")

    # 1a. Outcomes (45d) — mortality_28d → dead_28d.
    outcomes_path = interim_dir / "45d_outcomes_absorb_v1_mimic.parquet"
    print(f"  Loading outcomes: {outcomes_path}")
    outcomes_df = pd.read_parquet(outcomes_path)
    n_stays = len(outcomes_df)
    print(f"  Stays: {n_stays:,}")
    if "mortality_28d" not in outcomes_df.columns:
        raise KeyError("mortality_28d not found in 45d outcomes (needed for dead_28d)")

    # 1b. Rule policy actions (70c) → concordance tertiles.
    rule_path = interim_dir / "70c_rule_policy_actions_v1_mimic.parquet"
    print(f"  Loading rule policy: {rule_path}")
    rule_df = pd.read_parquet(rule_path)
    print(f"  Rule policy: {len(rule_df):,} obs")

    # 1c. Phenotype labels.
    labels_path = processed_dir / "phenotype_labels_frozen_v1_mimic.parquet"
    print(f"  Loading phenotype labels: {labels_path}")
    labels_df = pd.read_parquet(labels_path)
    print(f"  Labels: {len(labels_df):,} stays")

    # 1d. Vent indicators (10h) → IMV escalation.
    vent_path = staging_dir / "10h_vent_indicators_mimic.parquet"
    print(f"  Loading vent indicators: {vent_path}")
    vent_df = pd.read_parquet(vent_path)

    # 1e. Labs (10e) → creatinine → AKI progression.
    labs_path = staging_dir / "10e_labs_long_mimic.parquet"
    if not labs_path.exists():
        labs_path = staging_dir / "10e_labs_mimic.parquet"
    print(f"  Loading labs: {labs_path}")
    labs_df = pd.read_parquet(labs_path)

    # ══════════════════════════════════════════════════════════════════════
    # 2. Build the 3-component outcome vector (reuse P1 builders)
    # ══════════════════════════════════════════════════════════════════════
    print("\n[2/5] Building 3-component outcome vector ...")

    # dead_28d (from 45d mortality_28d).
    dead_28d = (
        outcomes_df.set_index("stay_id")["mortality_28d"].astype(int)
    )
    dead_28d.name = "dead_28d"
    print(f"  dead_28d: N={len(dead_28d):,}, mean={dead_28d.mean():.4f}")

    # imv_escalation (P1 builder).
    imv = p1.build_imv_escalation(vent_df, outcomes_df)
    del vent_df

    # aki_progression (P1 builder; {0,1,NaN}).
    aki = p1.build_aki_progression(labs_df, outcomes_df)
    del labs_df

    # Assemble per-stay outcome frame on 45d stay order.
    stay_ids = outcomes_df["stay_id"].values
    outcome_frame = pd.DataFrame(index=pd.Index(stay_ids, name="stay_id"))
    outcome_frame["dead_28d"] = dead_28d.reindex(stay_ids).astype("Int64")
    outcome_frame["imv_escalation"] = imv.reindex(stay_ids).fillna(0).astype("Int64")
    outcome_frame["aki_progression"] = aki.reindex(stay_ids).astype("Float64")

    # Drop stays with missing death or IMV (the two non-missing-allowed tiers).
    before = len(outcome_frame)
    outcome_frame = outcome_frame.dropna(subset=["dead_28d", "imv_escalation"])
    # Convert the two definite columns to plain int for the count loop.
    outcome_frame["dead_28d"] = outcome_frame["dead_28d"].astype(int)
    outcome_frame["imv_escalation"] = outcome_frame["imv_escalation"].astype(int)
    # aki_progression stays as float with NaN for the 'missing' tier.
    outcome_frame["aki_progression"] = outcome_frame["aki_progression"].astype(float)
    n_aki_known = int(outcome_frame["aki_progression"].notna().sum())
    print(f"  Outcome frame: {len(outcome_frame):,} stays "
          f"(dropped {before - len(outcome_frame):,} with missing death/IMV)")
    print(f"  AKI tier known (paired creatinine): {n_aki_known:,} "
          f"({n_aki_known / max(len(outcome_frame), 1) * 100:.1f}%); "
          f"remainder → AKI-missing tie tier")

    # ══════════════════════════════════════════════════════════════════════
    # 3. Rule-concordance tertiles (reuse P1) + HF subtype join
    # ══════════════════════════════════════════════════════════════════════
    print("\n[3/5] Computing rule-concordance tertiles + HF subtype ...")
    concordance_df = p1.compute_rule_concordance(rule_df)
    conc = concordance_df.set_index("stay_id")["concordance_tertile"]
    outcome_frame = outcome_frame.join(conc, how="left")
    n_with_tertile = int(outcome_frame["concordance_tertile"].notna().sum())
    print(f"  Stays with a concordance tertile: {n_with_tertile:,}")

    # HF subtype (lowercase aliases per spec).
    hf_col = None
    for candidate in ["hf_subtype", "comparator_group"]:
        if candidate in labels_df.columns:
            hf_col = candidate
            break
    if hf_col is None:
        print("  WARNING: no hf_subtype/comparator_group column — HFpEF/no_hf "
              "strata will be empty")
        outcome_frame["hf_subtype_l"] = pd.NA
    else:
        hf_map = (
            labels_df.set_index("stay_id")[hf_col]
            .astype("string").str.lower()
        )
        outcome_frame["hf_subtype_l"] = (
            outcome_frame.index.map(hf_map.to_dict())
        )
        print(f"  HF subtype column: {hf_col}; "
              f"value counts (lowercased): "
              f"{outcome_frame['hf_subtype_l'].value_counts(dropna=False).to_dict()}")

    # ══════════════════════════════════════════════════════════════════════
    # 4. Win matrix + per-stratum analysis
    # ══════════════════════════════════════════════════════════════════════
    print("\n[4/5] Win matrix + per-stratum win-ratio / composite RD ...")
    W, patterns = build_win_matrix()
    print(f"  Win matrix: {W.shape[0]}×{W.shape[1]} "
          f"(+1: {int((W == 1).sum())}, −1: {int((W == -1).sum())}, "
          f"0: {int((W == 0).sum())})")

    # Strata definitions (overall / HFpEF / no_hf), lowercase matching.
    strata = {
        "overall": outcome_frame,
        "HFpEF": outcome_frame[outcome_frame["hf_subtype_l"] == "hfpef"],
        "no_hf": outcome_frame[outcome_frame["hf_subtype_l"] == "no_hf"],
    }

    results = []
    for subgroup_name, sub in tqdm(strata.items(), desc="  Strata", leave=True):
        print(f"\n  --- {subgroup_name} (n={len(sub):,}) ---")
        # Restrict to stays that have a concordance tertile assigned.
        sub_t = sub.dropna(subset=["concordance_tertile"])
        if len(sub_t) == 0:
            results.append({
                "comparison": "rule_concordance_top_vs_bottom_tertile",
                "subgroup": subgroup_name, "n": 0,
                "n_top_tertile": 0, "n_bot_tertile": 0,
                "win_ratio": np.nan, "wr_ci_lo": np.nan, "wr_ci_hi": np.nan,
                "n_wins": 0, "n_losses": 0, "n_ties": 0,
                "composite_unadj_rd": np.nan,
                "composite_unadj_rd_ci_lo": np.nan,
                "composite_unadj_rd_ci_hi": np.nan,
                "note": "no stays with concordance tertile",
            })
            continue
        try:
            row = analyze_stratum(
                subgroup_name, sub_t, W, patterns,
                B=args.bootstrap, seed=args.seed,
            )
            print(f"    WR={row.get('win_ratio')} "
                  f"[{row.get('wr_ci_lo')}, {row.get('wr_ci_hi')}], "
                  f"composite_unadj_rd={row.get('composite_unadj_rd')} "
                  f"[{row.get('composite_unadj_rd_ci_lo')}, "
                  f"{row.get('composite_unadj_rd_ci_hi')}]")
            results.append(row)
        except Exception as e:  # noqa: BLE001
            print(f"    ERROR analyzing {subgroup_name}: {e}")
            traceback.print_exc()
            results.append({
                "comparison": "rule_concordance_top_vs_bottom_tertile",
                "subgroup": subgroup_name, "n": len(sub_t),
                "note": f"analysis failed: {e}",
            })

    # ══════════════════════════════════════════════════════════════════════
    # 5. Save outputs
    # ══════════════════════════════════════════════════════════════════════
    print("\n[5/5] Saving outputs ...")
    col_order = [
        "comparison", "subgroup", "n", "n_top_tertile", "n_bot_tertile",
        "win_ratio", "wr_ci_lo", "wr_ci_hi",
        "n_wins", "n_losses", "n_ties",
        "composite_unadj_rd", "composite_unadj_rd_ci_lo", "composite_unadj_rd_ci_hi",
        "note",
    ]
    results_df = pd.DataFrame(results)
    for c in col_order:
        if c not in results_df.columns:
            results_df[c] = np.nan
    results_df = results_df[col_order]
    csv_path = tables_dir / "P17a_net_clinical_benefit_winratio.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # ── Text report ──
    ts_end = datetime.datetime.now()
    report_lines = [
        "=" * 72,
        "P17a — Net-clinical-benefit / win-ratio (pattern-count rewrite)",
        "=" * 72,
        f"Generated: {ts_end.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Runtime: {(ts_end - ts_start).total_seconds():.0f}s",
        f"Bootstrap: B={args.bootstrap} (multinomial pattern-count resample), "
        f"seed={args.seed}",
        f"Stays (45d): {n_stays:,}",
        f"Outcome-frame stays (death & IMV non-missing): {len(outcome_frame):,}",
        f"AKI tier known (paired creatinine): {n_aki_known:,}",
        "",
        "METHOD",
        "──────",
        "  Hierarchical win-ratio over a 3-component ordered outcome:",
        "    death (28d) > IMV escalation > AKI progression; lower event = better.",
        "  AKI-missing → tie on the AKI tier (death & IMV always resolvable).",
        "  12 outcome patterns (2 death × 2 IMV × 3 AKI{0,1,missing}); per-group",
        "  pattern counts; win-ratio via the 12×12 hierarchical win matrix:",
        "    WR = Σ c_top[i]·c_bot[j]·1[W=+1] / Σ ...·1[W=−1].",
        "  Comparison: rule-concordance TOP tertile vs BOTTOM tertile.",
        "  Composite RD = UNADJUSTED (any-of-three) top − bottom risk difference.",
        "  CIs: percentile bootstrap (multinomial resample of pattern counts).",
        "",
        "HONEST LABELING",
        "───────────────",
        "  - WR aggregates ALL three components under a pre-specified hierarchy;",
        "    WR > 1 ⇒ TOP-concordance tertile favored. The favorable mortality",
        "    signal and any adverse IMV/AKI signals are aggregated, not hidden.",
        "  - composite_unadj_rd is UNADJUSTED (no IPTW / WIS). It carries NO",
        "    causal/adjusted claim; it is a descriptive top−bottom contrast.",
        "  - WIS rule-policy means (P1) are NOT reported here; reference P1's",
        "    tables/P1_secondary_outcomes_summary.csv for the WIS estimates.",
        "  - AKI-missing stays are non-events for the composite (only known",
        "    AKI==1 contributes) and tie on the AKI tier for the win-ratio.",
        "",
        "RESULTS BY SUBGROUP",
        "───────────────────",
    ]
    for _, row in results_df.iterrows():
        sg = row.get("subgroup", "?")
        n = row.get("n", 0)
        if pd.notna(row.get("note")):
            report_lines.append(f"\n  {sg} (N={n}): {row['note']}")
            continue
        report_lines.append(
            f"\n  {sg} (N={n}; top n={row.get('n_top_tertile')}, "
            f"bottom n={row.get('n_bot_tertile')})"
        )
        wr = row.get("win_ratio")
        if pd.notna(wr):
            report_lines.append(
                f"    Win-ratio (top vs bottom): {wr:.4f} "
                f"[{row.get('wr_ci_lo'):.4f}, {row.get('wr_ci_hi'):.4f}]"
            )
            report_lines.append(
                f"    Wins={row.get('n_wins'):,}, losses={row.get('n_losses'):,}, "
                f"ties={row.get('n_ties'):,}"
            )
        crd = row.get("composite_unadj_rd")
        if pd.notna(crd):
            report_lines.append(
                f"    Composite (any-of-3) UNADJUSTED RD: {crd:.4f} "
                f"[{row.get('composite_unadj_rd_ci_lo'):.4f}, "
                f"{row.get('composite_unadj_rd_ci_hi'):.4f}]"
            )

    report_lines.extend([
        "",
        "INTERPRETATION NOTES",
        "────────────────────",
        "  - Win-ratio > 1: the rule-concordance TOP tertile experiences more",
        "    hierarchical wins than losses vs the BOTTOM tertile. This is an",
        "    OBSERVATIONAL concordance contrast (not a randomized comparison).",
        "  - The hierarchy weights death most heavily, then IMV escalation,",
        "    then AKI progression; this is the pre-specified clinical priority.",
        "  - Composite RD < 0: TOP tertile has a lower any-of-three event rate.",
        "    Sign tension between components (favorable death vs adverse IMV/AKI)",
        "    is resolved hierarchically by the WR and shown descriptively by RD.",
    ])

    report = "\n".join(report_lines) + "\n"
    report_path = reports_dir / "P17a_net_clinical_benefit_winratio_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {report_path}")
    print(report)

    print(f"\n[P17a] COMPLETE. Runtime: {(ts_end - ts_start).total_seconds():.0f}s")


if __name__ == "__main__":
    main()
