#!/usr/bin/env python3
"""
80g2_subtype_difference_ci — Paired bootstrap subtype-DIFFERENCE CIs (5-subtype).

Purpose
-------
Generalize the 80g HFpEF×shock 2×2 interaction bootstrap (which only contrasted
hfpef vs no_hf) to the FULL 5-subtype × 2-stratum grid and quantify pairwise
*subtype-difference* CIs under the PRIMARY 80c G-Transformer DM-rollout estimator.

Enhancement Analysis D (supplementary-analysis specification, section D).

Design
------
  - PRIMARY estimator only: 80c G-Transformer DM rollout (ow=1).
    DR (80b ACTIN) and FQE cells are EXCLUDED — pathological weight blow-up
    makes their per-cell differences untrustworthy (spec §C/§D caveat).
  - Grid: hf_subtype {hfpef, hfref, hfmref, hf_ef_unknown, no_hf}
          × shock_stratum {shock_predominant, non_shock} + marginal "ALL".
  - PAIRED bootstrap: B patient-level resamples. Within each replicate b we use a
    SINGLE RNG draw per (subtype, stratum) cell, then form every contrast's
    DIFFERENCE element-wise across the SAME replicate index b
    (diff_b = ATE_a^(b) - ATE_b^(b)). This is the same paired machinery the base
    80g script uses for the hfpef−no_hf difference; here it is applied to the full
    pairwise matrix + each-vs-no_hf + a difference-in-differences (DiD) across the
    two shock strata.
  - Contrasts reported:
      * non_shock hfpef−hfref  (PRIMARY contrast of interest; expected ~null)
      * full pairwise matrix (all ordered a>b pairs) within each stratum + ALL
      * each subtype vs no_hf within each stratum + ALL
      * DiD = (diff in non_shock) − (diff in shock_predominant) for every pair
  - Percentile CIs + p(diff>=0) reuse the base 80g percentile / sign machinery.

Honest caveat (spec §D)
-----------------------
  non_shock hfref n≈112, hfmref n≈77 → WIDE difference CIs. The baseline point
  estimates (hfpef −1.44pp vs hfref −1.39pp) imply a likely-NULL hfpef−hfref
  difference (~−0.05pp). Surfacing this honest null is the intended output: it
  underpins the D1 calibration argument and the 4-subtype forest plot.

Inputs
------
- models/80c_gtransformer_rollout_ow1_best.pt
- data/processed/phenotype_labels_frozen_v1_mimic.parquet  (needs hf_subtype,
  rule_phenotype_2class)
- configs/80_step_c_training_v1.yaml
- (transitively, via load_dtr_data) 45c/45d/70b staged DTR inputs.

Outputs
-------
- tables/80g2_subtype_difference_ci.csv
    cols: contrast, shock_stratum, n_a, n_b, diff_point, diff_ci_lo, diff_ci_hi,
          ratio_point, ratio_ci_lo, ratio_ci_hi, did, did_ci_lo, did_ci_hi,
          p_diff_ge0
- tables/80g2_subtype_difference_ci_cells.csv   (per-cell ATE point + bootstrap CI)
- reports/80g2_subtype_difference_ci_summary.txt

Run on the compute cluster (raw data is NOT local):
    env/run_py.sh scripts/80g2_subtype_difference_ci.py
Compute: C3-C4 / GPU (~2-4h for B=500 paired patient-level resamples).
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hfpef_dtr.models.dtr_dataset import DTRDataset, load_dtr_data
from hfpef_dtr.models.dataset import make_train_val_test_split
from hfpef_dtr.models.ope import evaluate_dm

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "tables"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"

# Full 5-subtype universe (spec §D); ordered so no_hf is the canonical reference.
HF_SUBTYPES = ["hfpef", "hfref", "hfmref", "hf_ef_unknown", "no_hf"]
SHOCK_STRATA = ["shock_predominant", "non_shock"]
REFERENCE_SUBTYPE = "no_hf"

# Bootstrap defaults (mirror base 80g conventions).
DEFAULT_B = 500
DEFAULT_SEED = 42
DEFAULT_CI_LEVEL = 0.95
MIN_CELL_N = 20          # minimum stays to attempt an OPE on a cell
MIN_VALID_BOOT = 10      # minimum non-NaN bootstrap reps to report a CI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paired bootstrap subtype-difference CIs (5-subtype, 80c DM rollout).",
    )
    p.add_argument("--config", type=str,
                   default=str(ROOT / "configs" / "80_step_c_training_v1.yaml"),
                   help="Path to Step C training config YAML.")
    p.add_argument("--checkpoint", type=str,
                   default=str(MODELS_DIR / "80c_gtransformer_rollout_ow1_best.pt"),
                   help="80c G-Transformer DM-rollout checkpoint.")
    p.add_argument("--n-bootstrap", type=int, default=DEFAULT_B,
                   help="Number of paired bootstrap resamples (default 500).")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="Base RNG seed (default 42).")
    p.add_argument("--ci-level", type=float, default=DEFAULT_CI_LEVEL,
                   help="Confidence level for percentile CIs (default 0.95).")
    p.add_argument("--device", type=str, default=None,
                   choices=["cpu", "cuda"],
                   help="Force device; default auto-detect.")
    return p.parse_args()


def load_gtransformer(config: dict, checkpoint: Path, d_temporal: int,
                      d_static: int, n_actions: int, device: torch.device):
    """Load the primary 80c G-Transformer model (mirrors base 80g loader)."""
    import importlib
    mod = importlib.import_module("hfpef_dtr.models.gtransformer")
    builder_fn = getattr(mod, "build_gtransformer_from_config")
    model_cfg = config["models"]["80c_gtransformer"]
    grid_cfg = config["grid"]
    model = builder_fn(
        d_temporal=d_temporal, d_static=d_static,
        n_actions=n_actions, model_cfg=model_cfg, grid_cfg=grid_cfg,
    )
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate_subgroup(model, state_df, outcomes_df, propensity_df,
                      temporal_cols, static_cols, merge_map,
                      stay_ids, config, device):
    """Evaluate DM-rollout ATE for a subgroup; None if too small or failed."""
    if len(stay_ids) < MIN_CELL_N:
        return None
    sub_ds = DTRDataset(
        state_df=state_df,
        outcomes_df=outcomes_df,
        propensity_df=propensity_df,
        stay_ids=stay_ids,
        temporal_cols=temporal_cols,
        static_cols=static_cols,
        merge_map=merge_map,
        n_bins=config["grid"]["n_bins"],
        action_vocab=config["action"]["valid_merged_indices"],
    )
    try:
        result = evaluate_dm(model, sub_ds, device, use_rollout=True)
        return result["ate"]
    except Exception as e:
        print(f"    WARNING: DM evaluation failed: {e}")
        return None


def perstay_diff(model, state_df, outcomes_df, propensity_df,
                 temporal_cols, static_cols, merge_map,
                 stay_ids, config, device):
    """Per-stay DM-rollout difference (cf_prob - obs_prob) for a subgroup.

    Returns a 1-D np.ndarray of length = number of (unique) cell stays, or None
    if the cell is too small / evaluation failed. The MEAN of this array is the
    subgroup DM-rollout ATE; resampling THIS array WITH REPLACEMENT gives a
    correct patient-level nonparametric bootstrap.

    Why per-stay (not re-evaluating on resampled stay_ids): DTRDataset stores
    `np.array(sorted(set(stay_ids)))`, i.e. it DE-DUPLICATES stay_ids. Passing a
    with-replacement resample (which contains repeats) into a fresh DTRDataset
    would silently collapse repeats to unique stays and DESTROY the bootstrap
    resampling weights/multiplicity (CIs would be far too narrow / invalid).
    Computing per-stay contributions once and resampling them preserves
    multiplicity and is ~B× cheaper.
    """
    if len(stay_ids) < MIN_CELL_N:
        return None
    sub_ds = DTRDataset(
        state_df=state_df, outcomes_df=outcomes_df, propensity_df=propensity_df,
        stay_ids=stay_ids, temporal_cols=temporal_cols, static_cols=static_cols,
        merge_map=merge_map, n_bins=config["grid"]["n_bins"],
        action_vocab=config["action"]["valid_merged_indices"],
    )
    try:
        result = evaluate_dm(model, sub_ds, device, use_rollout=True)
        d = (np.asarray(result["cf_probs"], dtype=float)
             - np.asarray(result["obs_probs"], dtype=float))
        return d.ravel()
    except Exception as e:
        print(f"    WARNING: DM evaluation failed: {e}")
        return None


def percentile_ci(arr: np.ndarray, alpha: float):
    """Two-sided percentile CI from a bootstrap array (NaNs dropped).

    Mirrors the base 80g percentile machinery; returns (lo, hi, valid) with NaNs
    when there are too few valid replicates.
    """
    valid = arr[~np.isnan(arr)]
    if len(valid) < MIN_VALID_BOOT:
        return np.nan, np.nan, len(valid)
    lo = float(np.percentile(valid, 100 * alpha / 2))
    hi = float(np.percentile(valid, 100 * (1 - alpha / 2)))
    return lo, hi, len(valid)


def ratio_ci(arr_a: np.ndarray, arr_b: np.ndarray, alpha: float):
    """Paired ratio CI a/b across shared replicate index (inf/NaN dropped)."""
    ratio = arr_a / arr_b
    valid = ratio[~np.isnan(ratio) & ~np.isinf(ratio)]
    if len(valid) < MIN_VALID_BOOT:
        return np.nan, np.nan
    lo = float(np.percentile(valid, 100 * alpha / 2))
    hi = float(np.percentile(valid, 100 * (1 - alpha / 2)))
    return lo, hi


def main() -> None:
    args = parse_args()
    B = args.n_bootstrap
    SEED = args.seed
    CI_LEVEL = args.ci_level
    alpha = 1 - CI_LEVEL

    t0_wall = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("80g2 — Paired Bootstrap Subtype-Difference CIs (5-subtype, 80c DM rollout)")
    print(f"B={B}, CI={CI_LEVEL*100:.0f}%, seed={SEED}")
    print("=" * 72)

    # ── 1. Load config and data ──────────────────────────────────────
    print("\n[1] Loading data ...")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    state_df, outcomes_df, propensity_df, temporal_cols, static_cols, merge_map = (
        load_dtr_data(ROOT, config)
    )

    all_stay_ids = state_df["stay_id"].unique()
    split_cfg = config["split"]
    _, _, test_ids = make_train_val_test_split(
        all_stay_ids, outcomes_df,
        train_frac=split_cfg["train_frac"],
        val_frac=split_cfg["val_frac"],
        test_frac=split_cfg["test_frac"],
        stratify_col=split_cfg["stratify_col"],
        seed=split_cfg["seed"],
    )
    print(f"  Test stays: {len(test_ids):,}")

    # ── 2. Load phenotype labels ─────────────────────────────────────
    print("\n[2] Loading phenotype labels ...")
    pheno_path = ROOT / config["data"]["phenotype_labels"]
    pheno_df = pd.read_parquet(pheno_path)

    required_cols = ["hf_subtype", "rule_phenotype_2class"]
    missing = [c for c in required_cols if c not in pheno_df.columns]
    if missing:
        print(f"  FATAL: Missing columns in frozen labels: {missing}")
        print(f"  Available columns: {list(pheno_df.columns)}")
        print("  → Re-run 65e_freeze_phenotype_labels.py first")
        sys.exit(1)

    pheno_test = pheno_df[pheno_df["stay_id"].isin(set(test_ids))].copy()
    print(f"  Phenotype labels for test: {len(pheno_test):,}")

    # Cross-tab for transparency / report.
    ct = pd.crosstab(pheno_test["hf_subtype"], pheno_test["rule_phenotype_2class"],
                     margins=True)
    print(f"\n  Cross-tabulation (test set):\n{ct.to_string()}")

    # Build cell -> stay_ids for the FULL grid + marginal ALL.
    # cell key = (hf_subtype, shock_stratum) where shock_stratum may be "ALL".
    cell_ids: dict[tuple[str, str], np.ndarray] = {}
    for hf in HF_SUBTYPES:
        # marginal (collapsed across shock)
        ids_all = pheno_test.loc[pheno_test["hf_subtype"] == hf, "stay_id"].values
        cell_ids[(hf, "ALL")] = ids_all
        for shock in SHOCK_STRATA:
            mask = ((pheno_test["hf_subtype"] == hf)
                    & (pheno_test["rule_phenotype_2class"] == shock))
            cell_ids[(hf, shock)] = pheno_test.loc[mask, "stay_id"].values

    print("\n  Cell sizes (test set):")
    for hf in HF_SUBTYPES:
        sizes = " | ".join(
            f"{s}={len(cell_ids[(hf, s)])}" for s in (SHOCK_STRATA + ["ALL"])
        )
        print(f"    {hf:14s}: {sizes}")

    # ── 3. Load primary model (80c DM rollout ONLY) ──────────────────
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_temporal = len(temporal_cols)
    d_static = len(static_cols)
    n_actions = len(config["action"]["valid_merged_indices"])
    print(f"\n[3] Loading 80c G-Transformer (DM rollout, device={device}) ...")
    print(f"    d_temporal={d_temporal}, d_static={d_static}, n_actions={n_actions}")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"  FATAL: checkpoint not found: {ckpt_path}")
        sys.exit(1)
    model = load_gtransformer(config, ckpt_path, d_temporal, d_static,
                              n_actions, device)

    # ── 4. Per-stay DM contributions + point estimates (per cell) ────
    # Compute each cell's per-stay DM difference (cf_prob - obs_prob) ONCE via
    # evaluate_dm; the point ATE is its mean. The bootstrap (section 5) resamples
    # these per-stay contributions WITH REPLACEMENT — preserving resample
    # multiplicity — instead of rebuilding DTRDataset per replicate (which would
    # de-duplicate the resampled stay_ids and collapse the bootstrap). See
    # perstay_diff() docstring.
    print("\n[4] Per-stay DM contributions + point estimates (per cell) ...")
    cell_diffs: dict[tuple[str, str], np.ndarray] = {}
    point_ates: dict[tuple[str, str], float] = {}
    for key in tqdm(list(cell_ids.keys()), desc="Point cells", unit="cell"):
        hf, shock = key
        d = perstay_diff(
            model, state_df, outcomes_df, propensity_df,
            temporal_cols, static_cols, merge_map,
            cell_ids[key], config, device,
        )
        cell_diffs[key] = d
        point_ates[key] = float(np.mean(d)) if d is not None else None
        n = len(cell_ids[key])
        if d is None:
            tqdm.write(f"  {hf:14s} × {shock:18s}: n={n:4d}  SKIP (n<{MIN_CELL_N} or failed)")
        else:
            tqdm.write(f"  {hf:14s} × {shock:18s}: n={n:4d}  ATE={point_ates[key]:+.4f}")

    # ── 5. Paired bootstrap on per-stay contributions ────────────────
    # PAIRED: a single RNG drives all cells so every contrast's difference is
    # formed element-wise across the SHARED replicate index b. Each cell resamples
    # its OWN per-stay diff array WITH REPLACEMENT (size = number of cell stays),
    # which correctly preserves the with-replacement multiplicity.
    print(f"\n[5] Paired bootstrap ({B} resamples, per-stay) ...")
    rng = np.random.RandomState(SEED)
    boot_ates: dict[tuple[str, str], list[float]] = {k: [] for k in cell_ids}

    for b in tqdm(range(B), desc="Bootstrap", unit="rep"):
        for key in cell_ids:
            d = cell_diffs[key]
            if d is None:
                boot_ates[key].append(np.nan)
                continue
            idx = rng.choice(len(d), size=len(d), replace=True)
            boot_ates[key].append(float(np.mean(d[idx])))

    boot_arr = {k: np.array(v, dtype=float) for k, v in boot_ates.items()}

    # ── 6. Per-cell CIs (companion table) ────────────────────────────
    print("\n[6] Per-cell bootstrap CIs ...")
    cell_rows = []
    for key in cell_ids:
        hf, shock = key
        lo, hi, n_valid = percentile_ci(boot_arr[key], alpha)
        valid = boot_arr[key][~np.isnan(boot_arr[key])]
        cell_rows.append({
            "hf_subtype": hf,
            "shock_stratum": shock,
            "n_stays": int(len(cell_ids[key])),
            "ate_point": point_ates.get(key, np.nan),
            "ate_boot_mean": float(np.mean(valid)) if len(valid) else np.nan,
            "ate_ci_lo": lo,
            "ate_ci_hi": hi,
            "ate_se": float(np.std(valid, ddof=1)) if len(valid) > 1 else np.nan,
            "n_valid_boot": n_valid,
        })
    cell_df = pd.DataFrame(cell_rows)
    cell_csv = TABLES_DIR / "80g2_subtype_difference_ci_cells.csv"
    cell_df.to_csv(cell_csv, index=False)
    print(f"  Saved: {cell_csv} ({len(cell_df)} rows)")

    # ── 7. Contrast (difference) CIs + ratio + DiD ───────────────────
    print("\n[7] Computing paired difference / ratio / DiD CIs ...")

    def build_contrast(sub_a: str, sub_b: str, shock: str) -> dict | None:
        """Paired difference (a−b) within a stratum, with ratio and DiD."""
        key_a = (sub_a, shock)
        key_b = (sub_b, shock)
        n_a = int(len(cell_ids[key_a]))
        n_b = int(len(cell_ids[key_b]))

        pa = point_ates.get(key_a)
        pb = point_ates.get(key_b)
        diff_point = (pa - pb) if (pa is not None and pb is not None) else np.nan
        ratio_point = (pa / pb) if (pa is not None and pb is not None
                                    and pb not in (0.0, None)) else np.nan

        arr_a = boot_arr[key_a]
        arr_b = boot_arr[key_b]
        diff = arr_a - arr_b  # paired across shared replicate index
        diff_lo, diff_hi, _ = percentile_ci(diff, alpha)
        rat_lo, rat_hi = ratio_ci(arr_a, arr_b, alpha)

        diff_valid = diff[~np.isnan(diff)]
        # p(diff >= 0): reuse base 80g sign machinery.
        p_diff_ge0 = float(np.mean(diff_valid >= 0)) if len(diff_valid) >= MIN_VALID_BOOT else np.nan

        # DiD: only meaningful for stratum-level (non-ALL) contrasts.
        # DiD = (a-b | non_shock) - (a-b | shock_predominant)
        # DiD is always oriented (non_shock diff) - (shock_predominant diff),
        # independent of which stratum row we are on, so both stratum rows for a
        # given pair report the same DiD value/CI.
        did = did_lo = did_hi = np.nan
        if shock in SHOCK_STRATA:
            diff_ns = boot_arr[(sub_a, "non_shock")] - boot_arr[(sub_b, "non_shock")]
            diff_sh = boot_arr[(sub_a, "shock_predominant")] - boot_arr[(sub_b, "shock_predominant")]
            did_arr = diff_ns - diff_sh
            p_ns_a = point_ates.get((sub_a, "non_shock"))
            p_ns_b = point_ates.get((sub_b, "non_shock"))
            p_sh_a = point_ates.get((sub_a, "shock_predominant"))
            p_sh_b = point_ates.get((sub_b, "shock_predominant"))
            if None not in (p_ns_a, p_ns_b, p_sh_a, p_sh_b):
                did = (p_ns_a - p_ns_b) - (p_sh_a - p_sh_b)
            did_lo, did_hi, _ = percentile_ci(did_arr, alpha)

        return {
            "contrast": f"{sub_a}-{sub_b}",
            "shock_stratum": shock,
            "n_a": n_a,
            "n_b": n_b,
            "diff_point": diff_point,
            "diff_ci_lo": diff_lo,
            "diff_ci_hi": diff_hi,
            "ratio_point": ratio_point,
            "ratio_ci_lo": rat_lo,
            "ratio_ci_hi": rat_hi,
            "did": did,
            "did_ci_lo": did_lo,
            "did_ci_hi": did_hi,
            "p_diff_ge0": p_diff_ge0,
        }

    contrast_rows: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    def add_contrast(sub_a: str, sub_b: str, shock: str) -> None:
        ckey = (sub_a, sub_b, shock)
        if ckey in seen:
            return
        seen.add(ckey)
        contrast_rows.append(build_contrast(sub_a, sub_b, shock))

    # (a) PRIMARY contrast of interest first: non_shock hfpef − hfref.
    add_contrast("hfpef", "hfref", "non_shock")

    # (b) Full pairwise matrix (all ordered a>b pairs) across every stratum + ALL.
    pair_order = {s: i for i, s in enumerate(HF_SUBTYPES)}
    for shock in (SHOCK_STRATA + ["ALL"]):
        for sub_a, sub_b in combinations(HF_SUBTYPES, 2):
            # keep consistent orientation by HF_SUBTYPES ordering
            if pair_order[sub_a] > pair_order[sub_b]:
                sub_a, sub_b = sub_b, sub_a
            add_contrast(sub_a, sub_b, shock)

    # (c) Each subtype vs no_hf (reference) across every stratum + ALL.
    for shock in (SHOCK_STRATA + ["ALL"]):
        for sub in HF_SUBTYPES:
            if sub == REFERENCE_SUBTYPE:
                continue
            add_contrast(sub, REFERENCE_SUBTYPE, shock)

    contrast_df = pd.DataFrame(contrast_rows, columns=[
        "contrast", "shock_stratum", "n_a", "n_b",
        "diff_point", "diff_ci_lo", "diff_ci_hi",
        "ratio_point", "ratio_ci_lo", "ratio_ci_hi",
        "did", "did_ci_lo", "did_ci_hi", "p_diff_ge0",
    ])
    out_csv = TABLES_DIR / "80g2_subtype_difference_ci.csv"
    contrast_df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv} ({len(contrast_df)} rows)")

    # ── 8. Report ────────────────────────────────────────────────────
    print("\n[8] Generating report ...")
    elapsed = time.time() - t0_wall

    def fmt(x, spec="+.4f"):
        return "NA" if (x is None or (isinstance(x, float) and np.isnan(x))) else format(x, spec)

    lines = [
        "=" * 72,
        "80g2 — Paired Bootstrap Subtype-Difference CIs (5-subtype, MIMIC-IV)",
        "=" * 72,
        f"Generated: {ts}",
        "Estimator: 80c G-Transformer DM rollout (ow=1) — PRIMARY ONLY",
        "  (DR/ACTIN + FQE EXCLUDED: pathological weights → untrustworthy diffs)",
        f"Bootstrap: B={B}, seed={SEED}, CI={CI_LEVEL*100:.0f}% (paired-by-replicate)",
        f"Wall time: {elapsed/60:.1f} min",
        "",
        "CROSS-TABULATION (test set)",
        "───────────────────────────",
        ct.to_string(),
        "",
        "PER-CELL ATE (point + bootstrap CI)",
        "───────────────────────────────────",
    ]
    for r in cell_rows:
        lines.append(
            f"  {r['hf_subtype']:14s} × {r['shock_stratum']:18s}: "
            f"n={r['n_stays']:4d}  ATE={fmt(r['ate_point'])}  "
            f"CI=[{fmt(r['ate_ci_lo'])}, {fmt(r['ate_ci_hi'])}]  "
            f"(valid={r['n_valid_boot']})"
        )

    # Highlight primary contrast.
    prim = next((r for r in contrast_rows
                 if r["contrast"] == "hfpef-hfref" and r["shock_stratum"] == "non_shock"),
                None)
    lines.extend([
        "",
        "PRIMARY CONTRAST OF INTEREST",
        "────────────────────────────",
    ])
    if prim is not None:
        lines.extend([
            f"  non_shock hfpef − hfref:",
            f"    n_hfpef={prim['n_a']}, n_hfref={prim['n_b']}",
            f"    diff = {fmt(prim['diff_point'])}  "
            f"CI=[{fmt(prim['diff_ci_lo'])}, {fmt(prim['diff_ci_hi'])}]",
            f"    p(diff>=0) = {fmt(prim['p_diff_ge0'], '.4f')}",
            f"    ratio = {fmt(prim['ratio_point'], '.3f')}  "
            f"CI=[{fmt(prim['ratio_ci_lo'], '.3f')}, {fmt(prim['ratio_ci_hi'], '.3f')}]",
            "    NOTE: hfref non_shock is small-n (~112) → WIDE CI expected;",
            "          a diff near 0 (~-0.0005) is the honest, expected NULL.",
        ])
    else:
        lines.append("  (primary contrast unavailable — check cell sizes)")

    lines.extend([
        "",
        "ALL CONTRASTS (diff / ratio / DiD)",
        "──────────────────────────────────",
        f"  {'contrast':>22s}  {'stratum':>18s}  {'n_a':>5s} {'n_b':>5s}  "
        f"{'diff':>9s}  {'diff_CI':>20s}  {'p>=0':>6s}  {'DiD':>9s}",
    ])
    for r in contrast_rows:
        ci = f"[{fmt(r['diff_ci_lo'])},{fmt(r['diff_ci_hi'])}]"
        lines.append(
            f"  {r['contrast']:>22s}  {r['shock_stratum']:>18s}  "
            f"{r['n_a']:5d} {r['n_b']:5d}  {fmt(r['diff_point']):>9s}  "
            f"{ci:>20s}  {fmt(r['p_diff_ge0'], '.4f'):>6s}  {fmt(r['did']):>9s}"
        )

    lines.extend([
        "",
        "INTERPRETATION",
        "──────────────",
        "  - diff CI excluding 0 AND p(diff>=0) far from 0.5 → subtypes differ.",
        "  - Small-n strata (non_shock hfref/hfmref) → WIDE CIs; absence of",
        "    separation is EXPECTED, not evidence of equivalence.",
        "  - DiD CI excluding 0 → contrast differs across shock strata (interaction).",
        "  - The hfpef−hfref non_shock NULL underpins the D1 calibration + forest.",
    ])

    report_path = REPORTS_DIR / "80g2_subtype_difference_ci_summary.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Report: {report_path}")

    print(f"\n80g2 subtype-difference CI COMPLETE. ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
