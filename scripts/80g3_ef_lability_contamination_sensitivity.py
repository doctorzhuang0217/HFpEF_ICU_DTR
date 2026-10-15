#!/usr/bin/env python3
"""
80g3_ef_lability_contamination_sensitivity — EF-lability / contamination cuts.

Purpose
-------
Stress-test the non-shock HFpEF-vs-HFrEF bootstrap-DIFFERENCE (the contrast that
underpins the HFpEF-specific signal) against three plausible contamination
sources raised in review:

  (e1) EF LABILITY / TIMING. Restrict BOTH arms (hfpef, hfref) to stays whose
       CLASSIFYING EF is both localizable in time AND chart-derived, i.e.
         classification_ef_bucket ∈ {pre_icu, icu_pre_t0}
         AND classification_ef_source_family == "chartevents".
       This drops note-NLP / undated EF and any EF charted after T0, so the
       subtype label cannot have been driven by a transient post-T0 EF dip.

  (e2) ACUTE-ILLNESS CONTAMINATION OF THE HFrEF ARM. Drop from the hfref arm any
       stay flagged is_septic_shock OR is_sepsis_no_shock (P11). Reduced EF in a
       septic patient may be a reversible acute depression rather than chronic
       HFrEF; removing these sharpens the hfref comparator.

  (e3) BOUNDARY / HFmrEF CONTAMINATION. Exclude hfmref stays AND any stay whose
       classifying EF sits on the 45–50% boundary (ef_value ∈ [45, 50)). This
       guards against borderline-EF misclassification leaking across the
       hfpef / hfref divide.

Each sub-analysis RE-ESTIMATES the SAME non-shock hfpef−hfref paired bootstrap
difference used in 80g2, after applying its inclusion/exclusion cut. This is a
SELF-CONTAINED adaptation of the 80g/80g2 contrast+bootstrap engine (it does NOT
import 80g2); it reuses the shared library helpers (DTRDataset, load_dtr_data,
make_train_val_test_split, evaluate_dm) exactly as 80g2 does.

Honest expectation (spec §E)
----------------------------
The dominant constraint is N. MIMIC HFpEF EF is mostly note-NLP / undated, so the
e1 localizable subset collapses to a few dozen stays; e2 shrinks hfref further;
e3 trims both. These cuts are EXPECTED to be UNDERPOWERED → frame outputs as
CONSISTENCY CHECKS ("no large change / sign-stable"), NOT as confirmation. N
before/after each cut is reported LOUDLY in stdout, the report, and the CSV note.

Enhancement Analysis E (supplementary-analysis specification, section E).

Inputs
------
- models/80c_gtransformer_rollout_ow1_best.pt          (PRIMARY estimator)
- data/processed/phenotype_labels_frozen_v1_mimic.parquet
      (needs hf_subtype, rule_phenotype_2class)
- data/staging/P12_ef_timing_mimic.parquet
      (cols classification_ef_bucket, classification_ef_source_family, ef_value)
      Produced by scripts/P12_ef_timing_relative_to_t0.py.
      Assumed to EXIST; pass --rerun-p12 to regenerate it first.
- data/staging/P11_acute_illness_driver_refined_mimic.parquet
      (flags is_septic_shock, is_sepsis_no_shock)
      Produced by scripts/P11_acute_illness_driver_refined.py.
- configs/80_step_c_training_v1.yaml
- (transitively, via load_dtr_data) 45c/45d/70b staged DTR inputs.

Outputs
-------
- tables/80g3_ef_lability_contamination_sensitivity.csv
    cols: analysis, shock_stratum, n_hfpef, n_hfref, ate_hfpef, ate_hfref,
          diff_point, diff_ci_lo, diff_ci_hi, p_diff_ge0, note
- reports/80g3_ef_lability_contamination_sensitivity_summary.txt

Run on the compute cluster (raw data is NOT local; this needs the GPU node):
    env/run_py.sh scripts/80g3_ef_lability_contamination_sensitivity.py
    env/run_py.sh scripts/80g3_ef_lability_contamination_sensitivity.py --rerun-p12
Compute: G1 (GPU); +P12 rescan (~30-50 min chartevents scan) only if --rerun-p12.
"""

from __future__ import annotations

import argparse
import runpy
import sys
import time
from datetime import datetime
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
STAGING_DIR = ROOT / "data" / "staging"

# Contrast universe for E: the non-shock hfpef−hfref difference only.
SHOCK_STRATUM = "non_shock"
SUB_A = "hfpef"
SUB_B = "hfref"

# EF-lability / contamination cut parameters (spec §E).
E1_BUCKETS = {"pre_icu", "icu_pre_t0"}     # localizable early EF buckets
E1_SOURCE_FAMILY = "chartevents"           # chart-derived (not note-NLP / undated)
E3_BOUNDARY_LO = 45.0                       # boundary EF window [45, 50)
E3_BOUNDARY_HI = 50.0
E3_EXCLUDE_SUBTYPE = "hfmref"

# Bootstrap defaults (mirror base 80g / 80g2 conventions).
DEFAULT_B = 500
DEFAULT_SEED = 42
DEFAULT_CI_LEVEL = 0.95
MIN_CELL_N = 20          # minimum stays to attempt an OPE on a cell
MIN_VALID_BOOT = 10      # minimum non-NaN bootstrap reps to report a CI

# Staged artifacts consumed by the contamination cuts.
P12_PARQUET = STAGING_DIR / "P12_ef_timing_mimic.parquet"
P11_PARQUET = STAGING_DIR / "P11_acute_illness_driver_refined_mimic.parquet"
P12_SCRIPT = Path(__file__).resolve().parent / "P12_ef_timing_relative_to_t0.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("EF-lability / contamination sensitivity for the non-shock "
                     "hfpef-hfref bootstrap difference (80c DM rollout)."),
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
    p.add_argument("--rerun-p12", action="store_true", default=False,
                   help=("Regenerate data/staging/P12_ef_timing_mimic.parquet "
                         "via P12_ef_timing_relative_to_t0.py before running "
                         "(default OFF: assume the P12 parquet already exists). "
                         "Triggers a full chartevents EF scan (~30-50 min)."))
    return p.parse_args()


def load_gtransformer(config: dict, checkpoint: Path, d_temporal: int,
                      d_static: int, n_actions: int, device: torch.device):
    """Load the primary 80c G-Transformer model (mirrors base 80g/80g2 loader)."""
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
    """Per-stay DM-rollout difference (cf_prob - obs_prob) for an arm.

    Returns a 1-D np.ndarray (length = number of unique arm stays) or None. The
    MEAN is the arm's DM-rollout ATE; resampling THIS array WITH REPLACEMENT is a
    correct patient-level bootstrap. DTRDataset de-duplicates stay_ids
    (`np.array(sorted(set(stay_ids)))`), so re-evaluating on with-replacement
    resampled ids would collapse repeats and destroy the bootstrap multiplicity;
    resampling per-stay contributions avoids that and is ~B× cheaper.
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

    Mirrors the base 80g / 80g2 percentile machinery; returns (lo, hi, valid)
    with NaNs when there are too few valid replicates.
    """
    valid = arr[~np.isnan(arr)]
    if len(valid) < MIN_VALID_BOOT:
        return np.nan, np.nan, len(valid)
    lo = float(np.percentile(valid, 100 * alpha / 2))
    hi = float(np.percentile(valid, 100 * (1 - alpha / 2)))
    return lo, hi, len(valid)


def paired_bootstrap_diff(
    ids_a: np.ndarray, ids_b: np.ndarray,
    model, state_df, outcomes_df, propensity_df,
    temporal_cols, static_cols, merge_map, config, device,
    B: int, seed: int, alpha: float, desc: str,
) -> dict:
    """Paired patient-level bootstrap of the (arm_a − arm_b) ATE difference.

    Reuses the base 80g paired machinery: within each replicate b we draw ONE
    resample per arm, evaluate the DM-rollout ATE per arm, and form the
    difference element-wise across the SHARED replicate index. Returns point
    ATEs, paired difference point/CI, and p(diff>=0).
    """
    n_a = int(len(ids_a))
    n_b = int(len(ids_b))

    # Per-stay DM contributions, computed ONCE per arm (see perstay_diff()).
    d_a = perstay_diff(model, state_df, outcomes_df, propensity_df,
                       temporal_cols, static_cols, merge_map, ids_a, config, device)
    d_b = perstay_diff(model, state_df, outcomes_df, propensity_df,
                       temporal_cols, static_cols, merge_map, ids_b, config, device)
    ate_a = float(np.mean(d_a)) if d_a is not None else None
    ate_b = float(np.mean(d_b)) if d_b is not None else None
    diff_point = (ate_a - ate_b) if (ate_a is not None and ate_b is not None) else np.nan

    # If either arm is too small / failed, skip the bootstrap entirely.
    if d_a is None or d_b is None or n_a < MIN_CELL_N or n_b < MIN_CELL_N:
        return {
            "n_a": n_a, "n_b": n_b,
            "ate_a": ate_a if ate_a is not None else np.nan,
            "ate_b": ate_b if ate_b is not None else np.nan,
            "diff_point": diff_point,
            "diff_ci_lo": np.nan, "diff_ci_hi": np.nan,
            "p_diff_ge0": np.nan, "n_valid_boot": 0,
        }

    rng = np.random.RandomState(seed)
    boot_a: list[float] = []
    boot_b: list[float] = []
    for _ in tqdm(range(B), desc=desc, unit="rep"):
        # Resample each arm's per-stay contributions WITH REPLACEMENT (preserves
        # multiplicity); paired across the shared replicate index.
        ia = rng.choice(len(d_a), size=len(d_a), replace=True)
        ib = rng.choice(len(d_b), size=len(d_b), replace=True)
        boot_a.append(float(np.mean(d_a[ia])))
        boot_b.append(float(np.mean(d_b[ib])))

    arr_a = np.array(boot_a, dtype=float)
    arr_b = np.array(boot_b, dtype=float)
    diff = arr_a - arr_b  # paired across shared replicate index
    diff_lo, diff_hi, n_valid = percentile_ci(diff, alpha)
    diff_valid = diff[~np.isnan(diff)]
    p_diff_ge0 = (float(np.mean(diff_valid >= 0))
                  if len(diff_valid) >= MIN_VALID_BOOT else np.nan)

    return {
        "n_a": n_a, "n_b": n_b,
        "ate_a": ate_a if ate_a is not None else np.nan,
        "ate_b": ate_b if ate_b is not None else np.nan,
        "diff_point": diff_point,
        "diff_ci_lo": diff_lo, "diff_ci_hi": diff_hi,
        "p_diff_ge0": p_diff_ge0, "n_valid_boot": n_valid,
    }


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
    print("80g3 — EF-lability / contamination sensitivity (non-shock hfpef-hfref)")
    print(f"B={B}, CI={CI_LEVEL*100:.0f}%, seed={SEED}")
    print("Estimator: 80c G-Transformer DM rollout (ow=1) — PRIMARY ONLY")
    print("=" * 72)

    # ── 0. Optional P12 rescan (default OFF) ─────────────────────────
    if args.rerun_p12:
        print("\n[0] --rerun-p12 set: regenerating P12_ef_timing parquet ...")
        if not P12_SCRIPT.exists():
            print(f"  FATAL: P12 script not found: {P12_SCRIPT}")
            sys.exit(1)
        # Execute P12 as __main__ so its main() runs (full chartevents EF scan).
        runpy.run_path(str(P12_SCRIPT), run_name="__main__")
        print("  P12 rescan complete.")
    else:
        print("\n[0] --rerun-p12 NOT set: assuming P12 parquet already exists.")

    # ── 1. Load config and DTR data ──────────────────────────────────
    print("\n[1] Loading config + DTR data ...")
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
    test_id_set = set(test_ids)
    print(f"  Test stays: {len(test_ids):,}")

    # ── 2. Phenotype labels (subtype + shock stratum) ────────────────
    print("\n[2] Loading phenotype labels ...")
    pheno_path = ROOT / config["data"]["phenotype_labels"]
    pheno_df = pd.read_parquet(pheno_path)
    required_cols = ["hf_subtype", "rule_phenotype_2class"]
    missing = [c for c in required_cols if c not in pheno_df.columns]
    if missing:
        print(f"  FATAL: Missing columns in frozen labels: {missing}")
        print(f"  Available columns: {list(pheno_df.columns)}")
        sys.exit(1)
    pheno_test = pheno_df[pheno_df["stay_id"].isin(test_id_set)].copy()
    print(f"  Phenotype labels for test: {len(pheno_test):,}")

    # Baseline non-shock arm membership (no cut applied).
    def arm_ids(sub: str, frame: pd.DataFrame) -> np.ndarray:
        mask = ((frame["hf_subtype"] == sub)
                & (frame["rule_phenotype_2class"] == SHOCK_STRATUM))
        return frame.loc[mask, "stay_id"].values

    base_hfpef = arm_ids(SUB_A, pheno_test)
    base_hfref = arm_ids(SUB_B, pheno_test)
    print(f"\n  BASELINE non-shock arms (test set):")
    print(f"    {SUB_A:8s}: n={len(base_hfpef)}")
    print(f"    {SUB_B:8s}: n={len(base_hfref)}")

    # ── 3. Load contamination-cut staging artifacts ──────────────────
    print("\n[3] Loading P12 (EF timing) + P11 (sepsis flags) staging ...")
    if not P12_PARQUET.exists():
        print(f"  FATAL: P12 parquet not found: {P12_PARQUET}")
        print("  → re-run with --rerun-p12 or run P12_ef_timing_relative_to_t0.py first")
        sys.exit(1)
    p12 = pd.read_parquet(
        P12_PARQUET,
        columns=["stay_id", "classification_ef_bucket",
                 "classification_ef_source_family", "ef_value"],
    )
    print(f"  P12 rows: {len(p12):,}")

    if not P11_PARQUET.exists():
        print(f"  FATAL: P11 parquet not found: {P11_PARQUET}")
        print("  → run P11_acute_illness_driver_refined.py first")
        sys.exit(1)
    p11 = pd.read_parquet(
        P11_PARQUET,
        columns=["stay_id", "is_septic_shock", "is_sepsis_no_shock"],
    )
    print(f"  P11 rows: {len(p11):,}")

    # Per-stay lookups restricted to the test set for the cuts.
    p12_idx = p12.set_index("stay_id")
    p11_idx = p11.set_index("stay_id")

    def ef_bucket(sid) -> str:
        try:
            return str(p12_idx.at[sid, "classification_ef_bucket"])
        except KeyError:
            return "unknown"

    def ef_source_family(sid) -> str:
        try:
            return str(p12_idx.at[sid, "classification_ef_source_family"])
        except KeyError:
            return "no_ef"

    def ef_value(sid) -> float:
        try:
            return float(p12_idx.at[sid, "ef_value"])
        except (KeyError, TypeError, ValueError):
            return np.nan

    def is_sepsis(sid) -> bool:
        try:
            return bool(p11_idx.at[sid, "is_septic_shock"]) or \
                   bool(p11_idx.at[sid, "is_sepsis_no_shock"])
        except KeyError:
            return False

    # ── 4. Build the three contamination cuts ────────────────────────
    # Each cut returns (ids_hfpef, ids_hfref, note) for the non-shock stratum.
    print("\n[4] Building contamination cuts (N before/after reported LOUDLY) ...")

    def cut_e1() -> tuple[np.ndarray, np.ndarray, str]:
        """Restrict BOTH arms to localizable, chart-derived early EF."""
        def keep(ids: np.ndarray) -> np.ndarray:
            return np.array([
                sid for sid in ids
                if ef_bucket(sid) in E1_BUCKETS
                and ef_source_family(sid) == E1_SOURCE_FAMILY
            ])
        a = keep(base_hfpef)
        b = keep(base_hfref)
        note = (f"e1: localizable early EF (bucket in {sorted(E1_BUCKETS)} AND "
                f"source={E1_SOURCE_FAMILY}); hfpef {len(base_hfpef)}->{len(a)}, "
                f"hfref {len(base_hfref)}->{len(b)}")
        return a, b, note

    def cut_e2() -> tuple[np.ndarray, np.ndarray, str]:
        """Drop septic / sepsis stays from the HFrEF arm only."""
        a = base_hfpef  # hfpef arm unchanged
        b = np.array([sid for sid in base_hfref if not is_sepsis(sid)])
        note = (f"e2: drop is_septic_shock|is_sepsis_no_shock from hfref arm; "
                f"hfpef {len(base_hfpef)}->{len(a)} (unchanged), "
                f"hfref {len(base_hfref)}->{len(b)}")
        return a, b, note

    def cut_e3() -> tuple[np.ndarray, np.ndarray, str]:
        """Exclude hfmref + boundary-EF [45,50) stays from BOTH arms.

        hfmref stays never appear in the hfpef/hfref arms by construction, so the
        operative filter is the boundary-EF exclusion; hfmref is dropped at the
        cohort level for transparency (it cannot contaminate this contrast).
        """
        def keep(ids: np.ndarray) -> np.ndarray:
            out = []
            for sid in ids:
                v = ef_value(sid)
                # Drop boundary EF in [45, 50); keep NaN (cannot localize value).
                if not np.isnan(v) and (E3_BOUNDARY_LO <= v < E3_BOUNDARY_HI):
                    continue
                out.append(sid)
            return np.array(out)
        a = keep(base_hfpef)
        b = keep(base_hfref)
        n_hfmref = int((pheno_test["hf_subtype"] == E3_EXCLUDE_SUBTYPE).sum())
        note = (f"e3: exclude {E3_EXCLUDE_SUBTYPE} (n={n_hfmref}, not in this "
                f"contrast) + boundary EF in [{E3_BOUNDARY_LO:.0f},{E3_BOUNDARY_HI:.0f}); "
                f"hfpef {len(base_hfpef)}->{len(a)}, hfref {len(base_hfref)}->{len(b)}")
        return a, b, note

    cuts = {
        "e1_localizable_early_ef": cut_e1,
        "e2_drop_sepsis_hfref": cut_e2,
        "e3_exclude_hfmref_boundary": cut_e3,
    }

    cut_specs: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    print("\n" + "!" * 72)
    print("!! N BEFORE/AFTER EACH CUT (subsets EXPECTED to be UNDERPOWERED) !!")
    print("!" * 72)
    print(f"  baseline non-shock: {SUB_A}={len(base_hfpef)}  {SUB_B}={len(base_hfref)}")
    for name, fn in cuts.items():
        ids_a, ids_b, note = fn()
        cut_specs[name] = (ids_a, ids_b, note)
        flag_a = "" if len(ids_a) >= MIN_CELL_N else f"  <-- BELOW n={MIN_CELL_N} FLOOR (will SKIP OPE)"
        flag_b = "" if len(ids_b) >= MIN_CELL_N else f"  <-- BELOW n={MIN_CELL_N} FLOOR (will SKIP OPE)"
        print(f"  {name}:")
        print(f"      {SUB_A}: {len(base_hfpef)} -> {len(ids_a)}{flag_a}")
        print(f"      {SUB_B}: {len(base_hfref)} -> {len(ids_b)}{flag_b}")
    print("!" * 72)

    # ── 5. Load primary model (80c DM rollout ONLY) ──────────────────
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_temporal = len(temporal_cols)
    d_static = len(static_cols)
    n_actions = len(config["action"]["valid_merged_indices"])
    print(f"\n[5] Loading 80c G-Transformer (DM rollout, device={device}) ...")
    print(f"    d_temporal={d_temporal}, d_static={d_static}, n_actions={n_actions}")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"  FATAL: checkpoint not found: {ckpt_path}")
        sys.exit(1)
    model = load_gtransformer(config, ckpt_path, d_temporal, d_static,
                              n_actions, device)

    # ── 6. Baseline + per-cut paired bootstrap differences ───────────
    print("\n[6] Paired bootstrap differences (baseline + 3 cuts) ...")
    rows: list[dict] = []

    def make_row(analysis: str, ids_a: np.ndarray, ids_b: np.ndarray,
                 base_note: str, seed: int) -> dict:
        underpowered = (len(ids_a) < MIN_CELL_N) or (len(ids_b) < MIN_CELL_N)
        res = paired_bootstrap_diff(
            ids_a, ids_b, model, state_df, outcomes_df, propensity_df,
            temporal_cols, static_cols, merge_map, config, device,
            B=B, seed=seed, alpha=alpha,
            desc=f"boot[{analysis}]",
        )
        note = base_note
        if underpowered:
            note += (f" | UNDERPOWERED (n<{MIN_CELL_N} in an arm): "
                     "OPE skipped, CONSISTENCY-CHECK ONLY")
        elif res["n_valid_boot"] < MIN_VALID_BOOT:
            note += " | too few valid bootstrap reps -> CI unreliable"
        else:
            note += " | consistency check (underpowered small-n CI expected wide)"
        return {
            "analysis": analysis,
            "shock_stratum": SHOCK_STRATUM,
            "n_hfpef": res["n_a"],
            "n_hfref": res["n_b"],
            "ate_hfpef": res["ate_a"],
            "ate_hfref": res["ate_b"],
            "diff_point": res["diff_point"],
            "diff_ci_lo": res["diff_ci_lo"],
            "diff_ci_hi": res["diff_ci_hi"],
            "p_diff_ge0": res["p_diff_ge0"],
            "note": note,
        }

    # Baseline (no cut) — anchor for all consistency comparisons.
    base_note = (f"baseline: full non-shock {SUB_A} vs {SUB_B} (no contamination "
                 f"cut); hfpef={len(base_hfpef)}, hfref={len(base_hfref)}")
    rows.append(make_row("baseline_no_cut", base_hfpef, base_hfref, base_note, SEED))

    # Three contamination cuts (distinct seed offsets keep RNG streams independent).
    for i, (name, (ids_a, ids_b, note)) in enumerate(cut_specs.items(), start=1):
        rows.append(make_row(name, ids_a, ids_b, note, SEED + i))

    # ── 7. Save table ────────────────────────────────────────────────
    print("\n[7] Saving table ...")
    out_df = pd.DataFrame(rows, columns=[
        "analysis", "shock_stratum", "n_hfpef", "n_hfref",
        "ate_hfpef", "ate_hfref",
        "diff_point", "diff_ci_lo", "diff_ci_hi", "p_diff_ge0", "note",
    ])
    out_csv = TABLES_DIR / "80g3_ef_lability_contamination_sensitivity.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv} ({len(out_df)} rows)")

    # ── 8. Report ────────────────────────────────────────────────────
    print("\n[8] Generating report ...")
    elapsed = time.time() - t0_wall

    def fmt(x, spec="+.4f"):
        return ("NA" if (x is None or (isinstance(x, float) and np.isnan(x)))
                else format(x, spec))

    lines = [
        "=" * 72,
        "80g3 — EF-lability / contamination sensitivity (non-shock hfpef-hfref)",
        "=" * 72,
        f"Generated: {ts}",
        "Estimator: 80c G-Transformer DM rollout (ow=1) — PRIMARY ONLY",
        f"Bootstrap: B={B}, seed={SEED}, CI={CI_LEVEL*100:.0f}% (paired-by-replicate)",
        f"Wall time: {elapsed/60:.1f} min",
        "",
        "EXPECTATION (spec §E)",
        "─────────────────────",
        "  The dominant constraint is N. These cuts shrink the arms (especially e1",
        "  localizable-EF and e2 hfref-sepsis removal) and are EXPECTED to be",
        "  UNDERPOWERED. Read these as CONSISTENCY CHECKS (sign-stable / no large",
        "  change vs baseline), NOT as confirmation.",
        "",
        "N BEFORE/AFTER EACH CUT",
        "───────────────────────",
        f"  baseline non-shock: {SUB_A}={len(base_hfpef)}  {SUB_B}={len(base_hfref)}",
    ]
    for name, (ids_a, ids_b, note) in cut_specs.items():
        lines.append(f"  {name}:")
        lines.append(f"      {SUB_A}: {len(base_hfpef)} -> {len(ids_a)}")
        lines.append(f"      {SUB_B}: {len(base_hfref)} -> {len(ids_b)}")
        lines.append(f"      {note}")

    lines.extend([
        "",
        "RESULTS (non-shock hfpef − hfref difference)",
        "────────────────────────────────────────────",
        f"  {'analysis':>28s}  {'n_hfpef':>7s} {'n_hfref':>7s}  "
        f"{'ATE_hfpef':>10s} {'ATE_hfref':>10s}  {'diff':>9s}  "
        f"{'diff_CI':>20s}  {'p>=0':>6s}",
    ])
    for r in rows:
        ci = f"[{fmt(r['diff_ci_lo'])},{fmt(r['diff_ci_hi'])}]"
        lines.append(
            f"  {r['analysis']:>28s}  {r['n_hfpef']:7d} {r['n_hfref']:7d}  "
            f"{fmt(r['ate_hfpef']):>10s} {fmt(r['ate_hfref']):>10s}  "
            f"{fmt(r['diff_point']):>9s}  {ci:>20s}  "
            f"{fmt(r['p_diff_ge0'], '.4f'):>6s}"
        )

    lines.extend([
        "",
        "NOTES PER ANALYSIS",
        "──────────────────",
    ])
    for r in rows:
        lines.append(f"  [{r['analysis']}] {r['note']}")

    lines.extend([
        "",
        "INTERPRETATION",
        "──────────────",
        "  - Baseline non-shock hfpef−hfref is the anchor; the three cuts test",
        "    whether EF lability / timing (e1), acute-illness depression of the",
        "    hfref arm (e2), or boundary/hfmref misclassification (e3) materially",
        "    move that difference.",
        "  - A sign-stable diff with overlapping (wide) CIs across cuts = the",
        "    contamination hypotheses do NOT explain away the contrast.",
        "  - Cuts that fall below the n-floor report N only (OPE skipped); this is",
        "    a transparency / data-availability statement, not a null result.",
    ])

    report_path = REPORTS_DIR / "80g3_ef_lability_contamination_sensitivity_summary.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Report: {report_path}")

    print(f"\n80g3 EF-lability / contamination sensitivity COMPLETE. ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
