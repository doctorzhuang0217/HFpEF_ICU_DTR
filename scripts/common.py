#!/usr/bin/env python3
"""
common — Shared helpers for the BMC Medicine revision analyses (BM1–BM5).

Purpose
-------
One place for everything the revision scripts must do identically:

  * project-root resolution, config loading, output directories, timestamped
    phase logging, aggregate-only table writing, paste-back report blocks;
  * loading the Step C data exactly as the corrected code does
    (``hfpef_dtr.models.dtr_dataset.load_dtr_data`` + the frozen split), the
    phenotype strata, and DTRDatasets that share the TRAINING action vocabulary
    (85e convention) with an explicit check against the auto-discovered vocab;
  * loading the 80c G-Transformer checkpoint with shape checks;
  * the action codec (raw coarse_3 -> merged -> contiguous) and the 70c rule
    decoder (``RuleDecoder``) that de-normalises MAP / lactate / SpO2 from model
    states and applies the 70c priority logic with the same strict inequalities;
  * the non-adaptive rollout (``rollout_fixed`` = predict_counterfactual_rollout)
    and the ADAPTIVE g-computation rollout (``rollout_adaptive``) in which the
    rule is re-applied at every interval to the model-simulated state;
  * per-stay frames, the standard strata and a vectorised paired patient-level
    bootstrap with percentile CIs for means, paired differences and
    differences of differences between strata;
  * the 85e negative-exposure permutation and the REPRO_CHECK.

Placement
---------
Copy this file together with the BM scripts to
``<PROJECT_ROOT>/scripts/bmc_medicine/``; scripts do
``sys.path.insert(0, str(Path(__file__).resolve().parent)); import common``.

Inputs (read through load_all)
------------------------------
- configs/80_step_c_training_v1.yaml, configs/45c_normalization_params_v1.yaml,
  configs/70c_rule_policy_v1.yaml, configs/70b_action_merge_map.yaml
- data/interim/45c_state_tensors_v1_mimic.parquet
- data/interim/45d_outcomes_absorb_v1_mimic.parquet
- data/interim/70b_propensity_coarse3_v1_mimic.parquet
- data/interim/70c_rule_policy_actions_v1_mimic.parquet (attached by load_dtr_data)
- data/processed/phenotype_labels_frozen_v1_mimic.parquet
- models/80c_gtransformer*_best.pt

Outputs
-------
None by itself; helpers write to tables/bmc_medicine/, reports/bmc_medicine/
and (stay-level, never synced) data/interim/bmc_medicine/.

Compute: library (no standalone run).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

try:  # torch is needed for model-based helpers only (BM2/BM3 can import common without it)
    import torch
except ImportError:  # pragma: no cover
    torch = None

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════
SPEC_SEED = 20260907
NCE_SEED = 42
NCE_REPS = 10
N_BINS = 12
N_RAW_ACTIONS = 18
QUICK_N_STAYS = 400
QUICK_N_BOOT = 20
# 80c / 80g / 85e / 85j and the published reproduction targets (-1.37 / -2.27 pp)
# used the rollout_ow1 checkpoint, so it is the default. The untagged
# 80c_gtransformer_best.pt produced the P2 export (-1.41 / -2.31 pp).
PRIMARY_ROLLOUT_CHECKPOINT = "models/80c_gtransformer_rollout_ow1_best.pt"
DEFAULT_CHECKPOINT = PRIMARY_ROLLOUT_CHECKPOINT
CONFIG_REL = "configs/80_step_c_training_v1.yaml"
NORM_REL = "configs/45c_normalization_params_v1.yaml"
RULE_CFG_REL = "configs/70c_rule_policy_v1.yaml"
FLUID_BALANCE_CLIP_L = (-10.0, 15.0)          # 40b D1 (interval balance clip)
FORBIDDEN_TABLE_COLUMNS = {"stay_id", "subject_id", "hadm_id", "specimen_id",
                           "intime", "outtime", "charttime", "admittime", "dischtime"}

# 70c thresholds (configs/70c_rule_policy_v1.yaml 'thresholds')
DEFAULT_THRESHOLDS = {
    "map_hypotension": 65.0,
    "lactate_hypoperfusion": 2.0,
    "fluid_balance_overload_L": 3.0,
    "fluid_balance_severe_overload_L": 5.0,
    "spo2_hypoxemia": 94.0,
}
# rule categories (priority order of 70c apply_rule_policy) and their raw actions
RULE_CATEGORY_NAMES = OrderedDict([
    (1, "P1_shock_plus_hypoperfusion"),
    (2, "P2_hypotension_only"),
    (3, "P3_hypoperfusion_only"),
    (4, "P4_overload_hypoxaemia"),
    (5, "P5_severe_overload"),
    (6, "P6_default_maintenance"),
])
RULE_CATEGORY_TO_RAW = {1: 16, 2: 14, 3: 8, 4: 1, 5: 1, 6: 6}
RULE_CATEGORY_GROUP = {1: "perfusion_support", 2: "perfusion_support", 3: "perfusion_support",
                       4: "diuretic", 5: "diuretic", 6: "default"}

# Reproduction targets of the primary non-adaptive analysis (test set, pp)
REPRO_TARGETS = {
    "n_test": 5848, "n_hfpef": 304,
    "risk_uc_all": 11.78, "risk_rule_all": 10.41, "rd_all": -1.37, "rd_all_ci": (-1.49, -1.25),
    "risk_uc_hfpef": 15.77, "risk_rule_hfpef": 13.50, "rd_hfpef": -2.27, "rd_hfpef_ci": (-2.90, -1.70),
    "rd_hfpef_nonshock": -1.44, "rd_nohf_nonshock": -0.52, "rd_hfpef_shock": -3.23,
    "nce_rd": -1.04, "nce_retained_pct": 76.0,
}
REPRO_TOL_ALL_PP = 0.05
REPRO_TOL_HFPEF_PP = 0.10
# The EJHF figure pipeline (E01/E03/E04) ran from a separate root on Shirokane:
EJHF_ROOT_FALLBACK = "/home/drzhuang/HFpEF_ICU_DTR_ejhf"

LEGACY_RULE_NOTE = (
    "Pre-v1.0.1 DTRDataset read rule actions only from a 'rule_action' column of the 70b "
    "propensity table, which the coarse_3 70b never writes, so every 'rule' bin stayed at "
    "contiguous action 0 (= raw/merged 0: no fluid, no vasopressor, no diuretic). The published "
    "DM-rollout contrast (-1.37 pp; HFpEF -2.27 pp) is therefore expected to be reproduced by the "
    "legacy zero-action arm (BM4 S0), not by the corrected 70c rule arm. Usual-care risks "
    "(11.78 %, HFpEF 15.77 %) do not depend on rule actions and must reproduce either way.")


# ═══════════════════════════════════════════════════════════════════════
# Logging / paths / config
# ═══════════════════════════════════════════════════════════════════════
def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str = "") -> None:
    """Timestamped, flushed print (visible in UGE log files)."""
    print(f"[{now()}] {msg}", flush=True)


def phase(i: int | str, n: int | str, title: str) -> float:
    """Print a phase header; returns time.time() for elapsed-time logging."""
    log("-" * 72)
    log(f"[{i}/{n}] {title}")
    return time.time()


def find_root(script_path: str | Path | None = None) -> Path:
    """Resolve PROJECT_ROOT and put <ROOT>/src on sys.path.

    Order: env var PROJECT_ROOT; else walk up from Path.cwd(); else walk up from
    the calling script's directory (``script_path``, default this file) until a
    directory containing ``src/hfpef_dtr`` is found.
    """
    env = os.environ.get("PROJECT_ROOT")
    root: Path | None = None
    if env:
        root = Path(env).expanduser().resolve()
        if not (root / "src" / "hfpef_dtr").is_dir():
            raise FileNotFoundError(f"PROJECT_ROOT={root} does not contain src/hfpef_dtr")
    else:
        here = Path(script_path).resolve().parent if script_path else Path(__file__).resolve().parent
        cwd = Path.cwd().resolve()
        for cand in [cwd, *cwd.parents, here, *here.parents]:
            if (cand / "src" / "hfpef_dtr").is_dir():
                root = cand
                break
        if root is None:
            raise FileNotFoundError(
                "Could not locate the project root (a directory with src/hfpef_dtr). "
                "Run from /home/drzhuang/HFpEF_ICU_DTR or set PROJECT_ROOT.")
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return root


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def setup_paths(root: Path) -> dict[str, Path]:
    """Create and return the bmc_medicine output directories."""
    paths = {
        "root": root,
        "tables": root / "tables" / "bmc_medicine",
        "reports": root / "reports" / "bmc_medicine",
        "interim": root / "data" / "interim" / "bmc_medicine",
        "models": root / "models",
    }
    for key in ("tables", "reports", "interim"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def add_common_args(parser: argparse.ArgumentParser, n_boot_default: int = 1000,
                    model_based: bool = True) -> argparse.ArgumentParser:
    """Standard CLI flags shared by every BM script."""
    if model_based:
        parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                            help="G-Transformer checkpoint (relative to PROJECT_ROOT or absolute). "
                                 f"The published DM-rollout numbers used {PRIMARY_ROLLOUT_CHECKPOINT}.")
        parser.add_argument("--batch-size", type=int, default=256)
        parser.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0 ...")
    parser.add_argument("--n-boot", type=int, default=n_boot_default)
    parser.add_argument("--seed", type=int, default=SPEC_SEED)
    parser.add_argument("--quick", action="store_true",
                        help=f"smoke test: <= {QUICK_N_STAYS} stays and {QUICK_N_BOOT} bootstrap reps")
    return parser


def apply_quick(args: argparse.Namespace) -> argparse.Namespace:
    if getattr(args, "quick", False):
        args.n_boot = min(int(args.n_boot), QUICK_N_BOOT)
        log(f"QUICK MODE: n_boot={args.n_boot}, stays capped at {QUICK_N_STAYS}")
    return args


def resolve_device(name: str = "auto"):
    if torch is None:
        raise ImportError("torch is required for model-based helpers")
    if name == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(name)
    if dev.type == "cuda":
        log(f"Device: {dev} ({torch.cuda.get_device_name(dev)})")
    else:
        log("Device: cpu")
    return dev


def resolve_checkpoint(root: Path, arg: str | Path = DEFAULT_CHECKPOINT) -> Path:
    path = Path(arg)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        cands = sorted(p.name for p in (root / "models").glob("80c_gtransformer*_best.pt")) \
            if (root / "models").is_dir() else []
        raise FileNotFoundError(f"Checkpoint not found: {path}. Available 80c checkpoints: {cands}")
    primary = root / PRIMARY_ROLLOUT_CHECKPOINT
    if path.resolve() != primary.resolve() and primary.is_file():
        log(f"NOTE: using {path.name}; the published DM-rollout targets (-1.37 pp, HFpEF -2.27 pp) "
            f"were produced with {primary.name}. If REPRO_CHECK fails, rerun with "
            f"--checkpoint {PRIMARY_ROLLOUT_CHECKPOINT}")
    return path


def resolve_input_path(root: Path, path: str | Path, fallback_roots: Sequence[str | Path] = (EJHF_ROOT_FALLBACK,),
                       required: bool = True) -> Path | None:
    """Absolute path as given; a relative path is tried under ROOT, then under each fallback root
    (the EJHF interim files live under /home/drzhuang/HFpEF_ICU_DTR_ejhf on Shirokane)."""
    p = Path(path)
    cands = [p] if p.is_absolute() else [root / p] + [Path(fr) / p for fr in fallback_roots]
    for c in cands:
        if c.exists():
            if c != cands[0]:
                log(f"  input {p} not under PROJECT_ROOT; using {c}")
            return c
    if required:
        raise FileNotFoundError(f"Input not found; tried: {[str(c) for c in cands]}")
    return None


def subsample_ids(ids: Sequence, n: int, seed: int = SPEC_SEED) -> np.ndarray:
    ids = np.sort(np.asarray(ids))
    if len(ids) <= n:
        return ids
    rs = np.random.RandomState(seed)
    return np.sort(rs.choice(ids, size=n, replace=False))


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class Bundle:
    """Everything load_all() returns (frames are shared, never copied)."""
    root: Path
    config: dict
    state_df: pd.DataFrame
    outcomes_df: pd.DataFrame
    propensity_df: pd.DataFrame
    temporal_cols: list
    static_cols: list
    merge_map: dict
    train_ids: np.ndarray
    val_ids: np.ndarray
    test_ids: np.ndarray
    pheno_df: pd.DataFrame
    norm_params: dict
    thresholds: dict
    n_bins: int = N_BINS
    rule_col: str = "rule_action"
    extra: dict = field(default_factory=dict)

    @property
    def all_ids(self) -> np.ndarray:
        return np.sort(self.state_df["stay_id"].unique())


def load_all(root: Path, config: dict | None = None, verbose: bool = True) -> Bundle:
    """Load Step C frames exactly as the corrected code does, plus split and strata."""
    from hfpef_dtr.models import dtr_dataset as dd
    from hfpef_dtr.models.dataset import make_train_val_test_split

    if not hasattr(dd, "_attach_rule_actions"):
        raise RuntimeError(
            "hfpef_dtr.models.dtr_dataset has no _attach_rule_actions: this is the UNCORRECTED "
            "code base. Run from the corrected project root (public repo v1.0.1).")
    if config is None:
        config = load_yaml(root / CONFIG_REL)
    t0 = time.time()
    state_df, outcomes_df, propensity_df, temporal_cols, static_cols, merge_map = dd.load_dtr_data(root, config)
    rule_col = next((c for c in ("rule_action_merged", "rule_action", "rule_policy_action")
                     if c in propensity_df.columns), None)
    if rule_col is None:
        raise RuntimeError("No rule-action column attached to the propensity table (70c sidecar missing?)")
    split = config["split"]
    train_ids, val_ids, test_ids = make_train_val_test_split(
        state_df["stay_id"].unique(), outcomes_df,
        train_frac=split["train_frac"], val_frac=split["val_frac"], test_frac=split["test_frac"],
        stratify_col=split["stratify_col"], seed=split["seed"])
    pheno_path = root / config["data"]["phenotype_labels"]
    pheno = pd.read_parquet(pheno_path)
    keep = [c for c in ("stay_id", "hf_subtype", "rule_phenotype_2class", "rule_phenotype_4class",
                        "early_data_driven_k2") if c in pheno.columns]
    pheno = pheno[keep].drop_duplicates("stay_id")
    if "hf_subtype" not in pheno.columns:
        log("WARNING: hf_subtype missing from phenotype labels; taking it from the 45c static columns")
        hs = state_df[["stay_id", "hf_subtype"]].drop_duplicates("stay_id")
        pheno = pheno.merge(hs, on="stay_id", how="left")
    norm = load_yaml(root / NORM_REL)["params"]
    thr = dict(DEFAULT_THRESHOLDS)
    rule_cfg_path = root / config["data"].get("rule_policy", RULE_CFG_REL)
    if rule_cfg_path.is_file():
        cfg_thr = load_yaml(rule_cfg_path).get("thresholds", {}) or {}
        for k, v in cfg_thr.items():
            if k in thr and not math.isclose(float(v), thr[k]):
                log(f"WARNING: 70c threshold {k}={v} differs from the spec value {thr[k]}; using config value")
            if k in thr:
                thr[k] = float(v)
    n_bins = int(config["grid"]["n_bins"])
    b = Bundle(root=root, config=config, state_df=state_df, outcomes_df=outcomes_df,
               propensity_df=propensity_df, temporal_cols=list(temporal_cols),
               static_cols=list(static_cols), merge_map=dict(merge_map or {}),
               train_ids=np.asarray(train_ids), val_ids=np.asarray(val_ids), test_ids=np.asarray(test_ids),
               pheno_df=pheno, norm_params=norm, thresholds=thr, n_bins=n_bins, rule_col=rule_col)
    if verbose:
        log(f"  state rows={len(state_df):,} stays={state_df['stay_id'].nunique():,} "
            f"(load {time.time() - t0:.1f}s)")
        log(f"  temporal={len(temporal_cols)} static={len(static_cols)} {list(static_cols)}")
        log(f"  merge_map={b.merge_map}; rule column attached: {rule_col}")
        log(f"  split: train={len(train_ids):,} val={len(val_ids):,} test={len(test_ids):,}")
    return b


def training_vocab(bundle: Bundle, method: str = "dataset") -> list[int]:
    """Training action vocabulary.

    method='dataset' (spec / 85e): build a DTRDataset on the training ids and read
    ``action_vocab`` (auto-discovered). method='fast': same rule computed from the
    frames (unique merged observed action over valid training bins; bins with a
    state row but no action row count as action 0, as in DTRDataset).
    Both are logged; a mismatch raises.
    """
    fast = _fast_vocab(bundle, bundle.train_ids)
    if method == "fast":
        log(f"  training vocab (fast, {len(fast)}): {fast}")
        return fast
    from hfpef_dtr.models.dtr_dataset import DTRDataset
    t0 = time.time()
    log(f"  building training DTRDataset for the action vocabulary ({len(bundle.train_ids):,} stays) ...")
    ds = DTRDataset(state_df=bundle.state_df, outcomes_df=bundle.outcomes_df,
                    propensity_df=bundle.propensity_df, stay_ids=bundle.train_ids,
                    temporal_cols=bundle.temporal_cols, static_cols=bundle.static_cols,
                    merge_map=bundle.merge_map, n_bins=bundle.n_bins)
    vocab = [int(a) for a in ds.action_vocab]
    log(f"  training vocab (DTRDataset, {len(vocab)}): {vocab}  [{time.time() - t0:.1f}s]")
    if vocab != fast:
        raise ValueError(f"training vocab mismatch: DTRDataset {vocab} vs frame-based {fast}")
    cfg_v = sorted(int(a) for a in bundle.config.get("action", {}).get("valid_merged_indices", []))
    if cfg_v and cfg_v != vocab:
        log(f"WARNING: training vocab differs from config valid_merged_indices {cfg_v}")
    return vocab


def _fast_vocab(bundle: Bundle, ids: Sequence) -> list[int]:
    ids = set(np.asarray(ids).tolist())
    st = bundle.state_df.loc[bundle.state_df["stay_id"].isin(ids), ["stay_id", "bin"]]
    st = st[(st["bin"] >= 0) & (st["bin"] < bundle.n_bins)]
    pr = bundle.propensity_df
    act_col = "action_idx_merged" if "action_idx_merged" in pr.columns else "action_idx"
    m = st.merge(pr.loc[pr["stay_id"].isin(ids), ["stay_id", "bin", act_col]], on=["stay_id", "bin"], how="left")
    a = pd.to_numeric(m[act_col], errors="coerce").fillna(0).astype(np.int64)
    if act_col == "action_idx" and bundle.merge_map:
        a = a.replace(bundle.merge_map)
    return sorted(int(x) for x in np.unique(a.to_numpy()))


def build_dataset(bundle: Bundle, ids: Sequence, action_vocab: Sequence[int] | None = None,
                  label: str = "eval", strict_vocab: bool = True, propensity_df: pd.DataFrame | None = None):
    """DTRDataset for ``ids`` using the TRAINING vocabulary.

    The dataset is constructed with ``action_vocab`` (so the contiguous mapping
    equals the model's), and its auto-discovered vocabulary (unique merged
    observed action over valid bins, exactly what DTRDataset would discover with
    action_vocab=None) is logged and compared. strict_vocab=True raises on any
    difference; strict_vocab=False (quick mode / tiny subsets) only warns when
    the auto vocab is a subset.
    """
    from hfpef_dtr.models.dtr_dataset import DTRDataset
    t0 = time.time()
    ids = np.sort(np.asarray(ids))
    ds = DTRDataset(state_df=bundle.state_df, outcomes_df=bundle.outcomes_df,
                    propensity_df=bundle.propensity_df if propensity_df is None else propensity_df,
                    stay_ids=ids, temporal_cols=bundle.temporal_cols, static_cols=bundle.static_cols,
                    merge_map=bundle.merge_map, n_bins=bundle.n_bins,
                    action_vocab=None if action_vocab is None else list(action_vocab))
    auto = sorted(int(a) for a in np.unique(ds.action[ds.valid_mask.astype(bool)]))
    if action_vocab is not None:
        train = sorted(int(a) for a in action_vocab)
        log(f"  [{label}] n={len(ds):,}; training vocab {train}; auto-discovered vocab {auto}")
        if auto != train:
            msg = f"[{label}] auto-discovered vocab {auto} != training vocab {train}"
            if strict_vocab or not set(auto) <= set(train):
                raise ValueError(msg)
            log(f"WARNING: {msg} (subset; training vocab used)")
    else:
        log(f"  [{label}] n={len(ds):,}; auto-discovered vocab {auto}")
    log(f"  [{label}] built in {time.time() - t0:.1f}s; d_temporal={ds.d_temporal} d_static={ds.d_static}")
    return ds


def active_mask(ds) -> np.ndarray:
    """(N, T) bool: valid and non-absorbed bins (DTRDataset convention)."""
    return (ds.valid_mask > 0) & (ds.absorbing_mask > 0)


def stay_bin_matrix(frame: pd.DataFrame, stay_ids: Sequence, col: str, n_bins: int = N_BINS,
                    fill: float = np.nan) -> np.ndarray:
    """(N, T) float64 matrix of ``col`` aligned to ``stay_ids`` (dataset order)."""
    stay_ids = np.asarray(stay_ids)
    out = np.full((len(stay_ids), n_bins), fill, dtype=np.float64)
    sub = frame.loc[frame["stay_id"].isin(set(stay_ids.tolist())), ["stay_id", "bin", col]]
    sub = sub[(sub["bin"] >= 0) & (sub["bin"] < n_bins)]
    ri = pd.Index(stay_ids).get_indexer(sub["stay_id"].to_numpy())
    vals = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=np.float64)
    out[ri, sub["bin"].to_numpy().astype(int)] = vals
    return out


def stored_rule_raw_matrix(bundle: Bundle, stay_ids: Sequence) -> np.ndarray:
    """(N, T) stored 70c rule action (raw code unless only merged is available), -1 if absent."""
    m = stay_bin_matrix(bundle.propensity_df, stay_ids, bundle.rule_col, bundle.n_bins, fill=-1)
    return np.nan_to_num(m, nan=-1).astype(np.int64)


def fluid_ml_by_level(bundle: Bundle, ids: Sequence | None = None) -> dict[int, float]:
    """Median recorded iv_fluid_ml per observed coarse_3 fluid level (TRAINING set, active bins).

    Level 0 (none) is fixed at 0 mL. Used by the 'approx' fluid-balance mode.
    """
    ids = bundle.train_ids if ids is None else np.asarray(ids)
    ids_set = set(ids.tolist())
    st = bundle.state_df.loc[bundle.state_df["stay_id"].isin(ids_set), ["stay_id", "bin", "iv_fluid_ml"]]
    pr = bundle.propensity_df.loc[bundle.propensity_df["stay_id"].isin(ids_set), ["stay_id", "bin", "action_idx"]]
    m = st.merge(pr, on=["stay_id", "bin"], how="inner")
    ab = bundle.outcomes_df.set_index("stay_id")["absorbing_state_bin"]
    m["ab"] = m["stay_id"].map(ab)
    m = m[(m["bin"] < bundle.n_bins) & (m["ab"].isna() | (m["bin"] <= m["ab"]))]
    lvl = pd.to_numeric(m["action_idx"], errors="coerce") // 6
    out = {0: 0.0}
    for lv in (1, 2):
        v = m.loc[lvl == lv, "iv_fluid_ml"]
        out[lv] = float(v.median()) if len(v) else float("nan")
    log(f"  fluid_ml_by_level (training median, mL/2h): {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════
def load_model(path: str | Path, ds, config: dict, device):
    """Build the 80c G-Transformer from config and load ``path``; returns (model, info)."""
    from hfpef_dtr.models.gtransformer import build_gtransformer_from_config
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    n_actions = int(sd["action_embed.weight"].shape[0])
    d_temporal = int(sd["state_proj.weight"].shape[1])
    d_static = int(sd["static_embed.0.weight"].shape[1]) if "static_embed.0.weight" in sd else 0
    problems = []
    if d_temporal != ds.d_temporal:
        problems.append(f"d_temporal ckpt={d_temporal} vs data={ds.d_temporal}")
    if d_static != ds.d_static:
        problems.append(f"d_static ckpt={d_static} vs data={ds.d_static}")
    if n_actions != len(ds.action_vocab):
        problems.append(f"n_actions ckpt={n_actions} vs vocab={len(ds.action_vocab)}")
    if problems:
        raise ValueError("Checkpoint/data shape mismatch: " + "; ".join(problems))
    model = build_gtransformer_from_config(d_temporal=d_temporal, d_static=d_static, n_actions=n_actions,
                                           model_cfg=config["models"]["80c_gtransformer"],
                                           grid_cfg=config["grid"])
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    info = {"path": str(path), "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
            "val_loss": ckpt.get("val_loss") if isinstance(ckpt, dict) else None,
            "n_params": int(sum(p.numel() for p in model.parameters())),
            "d_temporal": d_temporal, "d_static": d_static, "n_actions": n_actions}
    vl = info["val_loss"]
    log(f"  checkpoint {Path(path).name}: epoch={info['epoch']} val_loss="
        f"{(f'{vl:.4f}' if isinstance(vl, (int, float)) else vl)} params={info['n_params']:,}")
    return model, info


# ═══════════════════════════════════════════════════════════════════════
# Action codec and rule decoder
# ═══════════════════════════════════════════════════════════════════════
def encode_raw(fluid, vaso, diuretic):
    """coarse_3 raw action = fluid*6 + vaso*2 + diuretic (numpy or torch)."""
    return fluid * 6 + vaso * 2 + diuretic


def decode_raw(raw):
    """raw -> (fluid_level, vaso_level, diuretic_level) (numpy or torch)."""
    return raw // 6, (raw % 6) // 2, raw % 2


class ActionCodec:
    """raw coarse_3 (0..17) -> merged (70b merge map) -> contiguous (training vocab)."""

    def __init__(self, merge_map: Mapping[int, int] | None, action_vocab: Sequence[int]):
        self.merge_map = {int(k): int(v) for k, v in (merge_map or {}).items()}
        self.vocab = sorted(int(a) for a in action_vocab)
        self.raw_to_merged_lut = np.arange(N_RAW_ACTIONS, dtype=np.int64)
        for old, new in self.merge_map.items():
            self.raw_to_merged_lut[old] = new
        size = max(N_RAW_ACTIONS, max(self.vocab) + 1)
        self.merged_to_contig_lut = np.full(size, -1, dtype=np.int64)
        for i, a in enumerate(self.vocab):
            self.merged_to_contig_lut[a] = i
        self.raw_to_contig_lut = self.merged_to_contig_lut[self.raw_to_merged_lut]
        self._torch_luts: dict = {}

    def _lut(self, name: str, like):
        if torch is not None and torch.is_tensor(like):
            key = (name, str(like.device))
            if key not in self._torch_luts:
                self._torch_luts[key] = torch.as_tensor(getattr(self, name), device=like.device)
            return self._torch_luts[key]
        return getattr(self, name)

    def raw_to_merged(self, raw):
        return self._lut("raw_to_merged_lut", raw)[raw]

    def merged_to_contiguous(self, merged, active=None):
        return self._map("merged_to_contig_lut", merged, active, "merged")

    def raw_to_contiguous(self, raw, active=None):
        """Contiguous index; inactive positions (active==False) get the placeholder 0."""
        return self._map("raw_to_contig_lut", raw, active, "raw")

    def _map(self, lut_name, codes, active, kind):
        lut = self._lut(lut_name, codes)
        is_t = torch is not None and torch.is_tensor(codes)
        if is_t:
            safe = codes.clamp(min=0, max=lut.shape[0] - 1)
            out = lut[safe]
            bad = (out < 0) | (codes < 0) | (codes >= lut.shape[0])
            if active is not None:
                bad = bad & active.bool()
                out = torch.where(active.bool(), out, torch.zeros_like(out))
            if bool(bad.any()):
                raise ValueError(f"{kind} action codes {sorted(set(codes[bad].tolist()))} not in vocab {self.vocab}")
            return out.clamp(min=0)
        codes = np.asarray(codes)
        safe = np.clip(codes, 0, lut.shape[0] - 1)
        out = lut[safe]
        bad = (out < 0) | (codes < 0) | (codes >= lut.shape[0])
        if active is not None:
            act = np.asarray(active, dtype=bool)
            bad = bad & act
            out = np.where(act, out, 0)
        if bad.any():
            raise ValueError(f"{kind} action codes {sorted(set(codes[bad].tolist()))} not in vocab {self.vocab}")
        return np.clip(out, 0, None)


class RuleDecoder:
    """70c rule applied to (possibly model-predicted) G-Transformer states.

    De-normalises map_z / lactate_z / spo2_z with the frozen 45c parameters
    (raw = z*std + mean, float64) and applies the 70c priority logic with the
    same strict inequalities. Because DTRDataset stores states as float32, a
    recorded value lying exactly on a threshold (MAP 65, lactate 2.0, SpO2 94)
    comes back as e.g. 64.9999990; values within ``snap_tol_z * std`` of a
    threshold are therefore snapped onto it (default 1e-5 z-units, i.e.
    1.4e-4 mmHg MAP, 1.4e-5 mmol/L lactate, 3e-5 % SpO2), which reproduces the
    stored 70c actions exactly. Set snap_tol_z=0 to disable.
    Fluid balance (cumulative, L) is not a model state and is supplied by the caller.
    """

    VARS = ("map", "lactate", "spo2")

    def __init__(self, temporal_cols: Sequence[str], norm_params: Mapping, merge_map: Mapping | None,
                 action_vocab: Sequence[int], thresholds: Mapping | None = None, snap_tol_z: float = 1e-5):
        self.temporal_cols = list(temporal_cols)
        self.idx = {}
        for v in self.VARS:
            col = f"{v}_z"
            if col not in self.temporal_cols:
                raise ValueError(f"{col} not among temporal columns")
            self.idx[v] = self.temporal_cols.index(col)
        self.mean = {v: float(norm_params[v]["mean"]) for v in self.VARS}
        self.std = {v: float(norm_params[v]["std"]) for v in self.VARS}
        thr = dict(DEFAULT_THRESHOLDS)
        thr.update({k: float(v) for k, v in (thresholds or {}).items() if k in thr})
        self.thr_map = thr["map_hypotension"]
        self.thr_lac = thr["lactate_hypoperfusion"]
        self.thr_fb3 = thr["fluid_balance_overload_L"]
        self.thr_fb5 = thr["fluid_balance_severe_overload_L"]
        self.thr_spo2 = thr["spo2_hypoxemia"]
        self.snap_tol_z = float(snap_tol_z)
        self.codec = ActionCodec(merge_map, action_vocab)
        self.thresholds = thr

    @classmethod
    def from_bundle(cls, bundle: Bundle, action_vocab: Sequence[int], snap_tol_z: float = 1e-5) -> "RuleDecoder":
        return cls(bundle.temporal_cols, bundle.norm_params, bundle.merge_map, action_vocab,
                   bundle.thresholds, snap_tol_z)

    # ---- de-normalisation --------------------------------------------------
    def _snap(self, x, thr: float, tol: float):
        if tol <= 0:
            return x
        return torch.where((x - thr).abs() <= tol, torch.full_like(x, thr), x)

    def decode(self, states):
        """states (..., D) -> dict map/lactate/spo2 raw units (float64 tensors, snapped)."""
        s = _as_tensor(states)
        out = {}
        for v, thr in (("map", self.thr_map), ("lactate", self.thr_lac), ("spo2", self.thr_spo2)):
            z = s[..., self.idx[v]].to(torch.float64)
            raw = z * self.std[v] + self.mean[v]
            out[v] = self._snap(raw, thr, self.snap_tol_z * self.std[v])
        return out

    def decode_np(self, states) -> dict[str, np.ndarray]:
        return {k: v.cpu().numpy() for k, v in self.decode(states).items()}

    # ---- rule ----------------------------------------------------------------
    def category_from_values(self, map_v, lac_v, spo2_v, fb_v):
        """Rule category 1..6 (P1..P5, 6=default), 70c priority and inequalities."""
        map_v, lac_v, spo2_v = _as_tensor(map_v), _as_tensor(lac_v), _as_tensor(spo2_v)
        fb_v = _as_tensor(fb_v).to(map_v.device)
        p1 = (map_v < self.thr_map) & (lac_v > self.thr_lac)
        p2 = (map_v < self.thr_map) & ~p1
        p3 = (lac_v > self.thr_lac) & (map_v >= self.thr_map)
        p4 = (fb_v > self.thr_fb3) & (spo2_v < self.thr_spo2) & (map_v >= self.thr_map) & (lac_v <= self.thr_lac)
        p5 = (fb_v > self.thr_fb5) & (map_v >= self.thr_map) & (lac_v <= self.thr_lac) & ~p4
        cat = torch.full(map_v.shape, 6, dtype=torch.long, device=map_v.device)
        cat[p1] = 1
        cat[p2] = 2
        cat[p3] = 3
        cat[p4] = 4
        cat[p5] = 5
        return cat

    def raw_from_category(self, cat):
        lut = torch.tensor([-1, 16, 14, 8, 1, 1, 6], dtype=torch.long, device=cat.device)
        return lut[cat]

    def raw_rule_from_values(self, map_v, lac_v, spo2_v, fb_v):
        """Vectorised 70c rule -> raw coarse_3 action (16/14/8/1/6). Same type as input."""
        is_np = not (torch is not None and torch.is_tensor(map_v))
        raw = self.raw_from_category(self.category_from_values(map_v, lac_v, spo2_v, fb_v))
        return raw.cpu().numpy() if is_np else raw

    def raw_rule_from_states(self, states, fb):
        """(raw action, category) from states (..., D) and cumulative fluid balance (...)."""
        d = self.decode(states)
        cat = self.category_from_values(d["map"], d["lactate"], d["spo2"], fb)
        return self.raw_from_category(cat), cat

    def to_contiguous(self, raw, active=None):
        return self.codec.raw_to_contiguous(raw, active)

    def decide(self, states_t, fb_t, t: int | None = None, batch: dict | None = None) -> dict:
        """Decoder protocol used by rollout_adaptive: decision at interval t."""
        raw, cat = self.raw_rule_from_states(states_t, fb_t)
        return {"raw": raw, "category": cat, "contig": self.codec.raw_to_contiguous(raw)}


class FixedActionDecoder:
    """Decoder that replays precomputed contiguous actions from the batch (validation / replay).

    With key='rule_action' rollout_adaptive must reproduce predict_counterfactual_rollout
    with the stored (non-adaptive) rule actions exactly.
    """

    def __init__(self, key: str = "rule_action", raw_key: str | None = None):
        self.key = key
        self.raw_key = raw_key

    def decide(self, states_t, fb_t, t: int, batch: dict) -> dict:
        contig = batch[self.key][:, t]
        raw = batch[self.raw_key][:, t] if self.raw_key else torch.full_like(contig, -1)
        return {"raw": raw, "category": torch.zeros_like(contig), "contig": contig}


def _as_tensor(x):
    if torch is not None and torch.is_tensor(x):
        return x
    return torch.as_tensor(np.asarray(x))


def rule_reproduction_check(bundle: Bundle, ds, decoder: RuleDecoder,
                            fb_cols: Sequence[str] = ("cumulative_fluid_balance_L", "fluid_balance_L")) -> dict:
    """Does RuleDecoder on the RECORDED states reproduce the attached 70c rule actions?

    For each candidate fluid-balance column (corrected 70c = cumulative; the pre-v1.0.1
    70c silently used the INTERVAL 'fluid_balance_L') report the agreement on active bins,
    plus the agreement without the float32 threshold snap and the disagreement pattern.
    An agreement of 1.0 with 'cumulative_fluid_balance_L' confirms that the attached 70c
    parquet was produced by the corrected 70c; 1.0 only with 'fluid_balance_L' means the
    sidecar is the legacy one (regenerate it with the corrected 70c).
    """
    act = active_mask(ds)
    stored = stored_rule_raw_matrix(bundle, ds.stay_ids)
    states = torch.from_numpy(ds.temporal)
    out = {"n_active_bins": int(act.sum())}
    for col in fb_cols:
        if col not in bundle.state_df.columns:
            out[col] = None
            continue
        fb = np.nan_to_num(stay_bin_matrix(bundle.state_df, ds.stay_ids, col, bundle.n_bins), nan=0.0)
        raw = decoder.raw_rule_from_states(states, torch.from_numpy(fb))[0].numpy()
        agree = float((raw[act] == stored[act]).mean())
        dis = pd.Series(list(zip(stored[act][raw[act] != stored[act]], raw[act][raw[act] != stored[act]])))
        out[col] = {"agreement": agree,
                    "disagreement_pairs_stored_vs_decoded": {f"{a}->{b}": int(c) for (a, b), c in
                                                             dis.value_counts().head(8).items()}}
    nos = RuleDecoder(decoder.temporal_cols, bundle.norm_params, bundle.merge_map, decoder.codec.vocab,
                      decoder.thresholds, snap_tol_z=0.0)
    fb = np.nan_to_num(stay_bin_matrix(bundle.state_df, ds.stay_ids, fb_cols[0], bundle.n_bins), nan=0.0)
    raw0 = nos.raw_rule_from_states(states, torch.from_numpy(fb))[0].numpy()
    out["agreement_without_snap"] = float((raw0[act] == stored[act]).mean())
    first = out.get(fb_cols[0]) or {}
    out["status"] = "PASS" if first.get("agreement") == 1.0 else "FAIL"
    log(f"RULE_DECODER_CHECK: {out['status']} (agreement with {fb_cols[0]}="
        f"{fmt_num(100 * first.get('agreement', float('nan')), 3)}%; "
        + "; ".join(f"{c}={fmt_num(100 * out[c]['agreement'], 3)}%" for c in fb_cols[1:] if out.get(c))
        + f"; without snap={fmt_num(100 * out['agreement_without_snap'], 3)}%)")
    return out


def legacy_zero_actions(ds) -> np.ndarray:
    """(N, T) contiguous zeros = the 'rule' arm actually fed to the model by the pre-v1.0.1
    DTRDataset (raw/merged action 0 everywhere; identical to BM4 strategy S0). See LEGACY_RULE_NOTE."""
    if ds.action_vocab[0] != 0:
        raise ValueError("contiguous 0 is not merged action 0 in this vocabulary")
    return np.zeros((len(ds), ds.n_bins), dtype=np.int64)


# ═══════════════════════════════════════════════════════════════════════
# Rollouts
# ═══════════════════════════════════════════════════════════════════════
def _pool_outcome(model, h, valid_mask, absorbing_mask):
    """Masked mean pooling + outcome head, exactly as predict_counterfactual_rollout."""
    active = (valid_mask * absorbing_mask).unsqueeze(-1)
    h_pooled = (h * active).sum(dim=1) / active.sum(dim=1).clamp(min=1)
    logit = model.outcome_head(h_pooled)
    return logit


def rollout_fixed(model, batch: dict, actions) -> dict:
    """Non-adaptive rollout = model.predict_counterfactual_rollout with fixed contiguous actions."""
    with torch.no_grad():
        out = model.predict_counterfactual_rollout(batch["temporal"], batch["static"], actions,
                                                   batch["valid_mask"], batch["absorbing_mask"])
    return {"prob": out["outcome_prob"].reshape(-1), "logit": out["outcome_logit"].reshape(-1),
            "predicted_states": out["predicted_states"]}


def rollout_adaptive(model, batch: dict, decoder, fb_mode: str = "recorded", fb_tensor=None,
                     fluid_ml_by_level: Mapping[int, float] | None = None, urine_ml=None) -> dict:
    """Adaptive g-computation rollout: the decoder re-decides a_t from the simulated s_t.

    For t = 0..T-1: s_t = observed s_0 if t == 0 else predicted s_t; the decoder
    decodes MAP / lactate / SpO2 from s_t and receives the cumulative fluid
    balance per ``fb_mode``:
      'recorded': fb_tensor[:, t] (recorded cumulative_fluid_balance_L at (stay, t));
      'approx'  : sum_{k<t} clip((fluid_ml(level of adaptive a_k) - urine_ml[:, k]) / 1000,
                  -10, 15) with fluid_ml_by_level = {0: 0, 1: median, 2: median} (training
                  set) and urine_ml = recorded urine_output_ml; the per-interval clip
                  mirrors 40b D1 and never binds in practice.
    a_t = decision (inactive bins get the placeholder action 0, as in DTRDataset);
    if t < T-1: h = model._encode(s_{0..t}, static, a_{0..t}, valid_mask[:, :t+1]) and
    s_{t+1} = model.next_state_head(h[:, t:t+1]). Finally all predicted states are
    encoded and pooled exactly as predict_counterfactual_rollout.

    Returns dict: prob (B,), logit (B,), raw_actions (B,T) long (-1 inactive),
    categories (B,T) long (0 inactive), contig_actions (B,T), predicted_states
    (B,T,D), fb_used (B,T) float64, decoded {'map','lactate','spo2'} (B,T) float64
    (raw-unit values of the states the decisions were based on).
    """
    if fb_mode not in ("recorded", "approx"):
        raise ValueError("fb_mode must be 'recorded' or 'approx'")
    temporal, static = batch["temporal"], batch["static"]
    valid_mask, absorbing_mask = batch["valid_mask"], batch["absorbing_mask"]
    B, T, D = temporal.shape
    dev = temporal.device
    active = (valid_mask > 0) & (absorbing_mask > 0)
    if fb_mode == "recorded":
        if fb_tensor is None:
            raise ValueError("fb_mode='recorded' needs fb_tensor (recorded cumulative fluid balance)")
        fb_rec = torch.nan_to_num(_as_tensor(fb_tensor).to(device=dev, dtype=torch.float64), nan=0.0)
    else:
        if urine_ml is None or fluid_ml_by_level is None:
            raise ValueError("fb_mode='approx' needs urine_ml and fluid_ml_by_level")
        urine = torch.nan_to_num(_as_tensor(urine_ml).to(device=dev, dtype=torch.float64), nan=0.0)
        fl_lut = torch.tensor([float(fluid_ml_by_level.get(i, 0.0)) for i in range(3)],
                              dtype=torch.float64, device=dev)
        fb_cum = torch.zeros(B, dtype=torch.float64, device=dev)

    pred = torch.zeros_like(temporal)
    pred[:, 0] = temporal[:, 0]
    actions = torch.zeros(B, T, dtype=torch.long, device=dev)
    raw_actions = torch.full((B, T), -1, dtype=torch.long, device=dev)
    categories = torch.zeros(B, T, dtype=torch.long, device=dev)
    fb_used = torch.zeros(B, T, dtype=torch.float64, device=dev)
    dec_vals = {k: torch.zeros(B, T, dtype=torch.float64, device=dev) for k in ("map", "lactate", "spo2")}
    has_decode = hasattr(decoder, "decode")
    with torch.no_grad():
        for t in range(T):
            s_t = pred[:, t]
            fb_t = fb_rec[:, t] if fb_mode == "recorded" else fb_cum.clone()
            fb_used[:, t] = fb_t
            if has_decode:
                d = decoder.decode(s_t)
                for k in dec_vals:
                    dec_vals[k][:, t] = d[k]
            dec = decoder.decide(s_t, fb_t, t, batch)
            act_t = active[:, t]
            contig = dec["contig"].to(dev).long()
            actions[:, t] = torch.where(act_t, contig, torch.zeros_like(contig))
            raw_actions[:, t] = torch.where(act_t, dec["raw"].to(dev).long(), torch.full_like(contig, -1))
            categories[:, t] = torch.where(act_t, dec["category"].to(dev).long(), torch.zeros_like(contig))
            if fb_mode == "approx":
                lvl = torch.clamp(raw_actions[:, t], min=0) // 6
                interval = (fl_lut[lvl] - urine[:, t]) / 1000.0
                interval = interval.clamp(FLUID_BALANCE_CLIP_L[0], FLUID_BALANCE_CLIP_L[1])
                fb_cum = fb_cum + torch.where(act_t, interval, torch.zeros_like(interval))
            if t < T - 1:
                h = model._encode(pred[:, : t + 1], static, actions[:, : t + 1], valid_mask[:, : t + 1])
                pred[:, t + 1] = model.next_state_head(h[:, t: t + 1]).squeeze(1)
        h = model._encode(pred, static, actions, valid_mask)
        logit = _pool_outcome(model, h, valid_mask, absorbing_mask)
    return {"prob": torch.sigmoid(logit).reshape(-1), "logit": logit.reshape(-1),
            "raw_actions": raw_actions, "categories": categories, "contig_actions": actions,
            "predicted_states": pred, "fb_used": fb_used, "decoded": dec_vals}


def iter_batches(ds, batch_size: int, device, extras: Mapping[str, np.ndarray] | None = None,
                 desc: str | None = None):
    """Yield (batch_on_device, extras_on_device, slice) in dataset order (shuffle=False)."""
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    start = 0
    for batch in tqdm(loader, desc=desc, mininterval=5, leave=False):
        b = int(batch["temporal"].shape[0])
        on = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        ex = {k: torch.as_tensor(np.asarray(arr)[start:start + b], device=device)
              for k, arr in (extras or {}).items()}
        yield on, ex, slice(start, start + b)
        start += b


def predict_fixed(model, ds, device, actions="action", batch_size: int = 256, desc: str = "rollout",
                  keep_states: bool = False) -> dict:
    """Non-adaptive rollout over a dataset.

    actions: 'action' (observed / usual care), 'rule_action' (stored rule), an
    (N,T) int array of CONTIGUOUS actions in dataset order, or a callable
    batch -> (B,T) contiguous tensor. Returns {'prob': (N,), 'predicted_states': (N,T,D)|None}.
    """
    n = len(ds)
    probs = np.zeros(n, dtype=np.float64)
    states = np.zeros((n, ds.n_bins, ds.d_temporal), dtype=np.float32) if keep_states else None
    extras = {"_act": actions} if isinstance(actions, np.ndarray) else None
    t0 = time.time()
    for batch, ex, sl in iter_batches(ds, batch_size, device, extras, desc=desc):
        if isinstance(actions, str):
            act = batch[actions]
        elif callable(actions):
            act = actions(batch)
        else:
            act = ex["_act"].long()
        out = rollout_fixed(model, batch, act)
        probs[sl] = out["prob"].double().cpu().numpy()
        if keep_states:
            states[sl] = out["predicted_states"].cpu().numpy()
    log(f"  {desc}: n={n:,} mean risk={probs.mean() * 100:.2f}% [{time.time() - t0:.1f}s]")
    return {"prob": probs, "predicted_states": states}


def predict_adaptive(model, ds, decoder, device, fb_mode: str = "recorded", fb_matrix=None,
                     fluid_ml_by_level=None, urine_matrix=None, batch_size: int = 256,
                     desc: str = "adaptive rollout", keep_states: bool = False) -> dict:
    """rollout_adaptive over a dataset; matrices (N,T) are in dataset order (stay_bin_matrix)."""
    n, T = len(ds), ds.n_bins
    out = {"prob": np.zeros(n), "raw_actions": np.full((n, T), -1, dtype=np.int64),
           "categories": np.zeros((n, T), dtype=np.int64), "contig_actions": np.zeros((n, T), dtype=np.int64),
           "fb_used": np.zeros((n, T)), "map": np.zeros((n, T)), "lactate": np.zeros((n, T)),
           "spo2": np.zeros((n, T)),
           "predicted_states": np.zeros((n, T, ds.d_temporal), dtype=np.float32) if keep_states else None}
    extras = {}
    if fb_matrix is not None:
        extras["fb"] = np.asarray(fb_matrix, dtype=np.float64)
    if urine_matrix is not None:
        extras["urine"] = np.asarray(urine_matrix, dtype=np.float64)
    t0 = time.time()
    for batch, ex, sl in iter_batches(ds, batch_size, device, extras, desc=desc):
        r = rollout_adaptive(model, batch, decoder, fb_mode=fb_mode, fb_tensor=ex.get("fb"),
                             fluid_ml_by_level=fluid_ml_by_level, urine_ml=ex.get("urine"))
        out["prob"][sl] = r["prob"].double().cpu().numpy()
        for k in ("raw_actions", "categories", "contig_actions", "fb_used"):
            out[k][sl] = r[k].cpu().numpy()
        for k in ("map", "lactate", "spo2"):
            out[k][sl] = r["decoded"][k].cpu().numpy()
        if keep_states:
            out["predicted_states"][sl] = r["predicted_states"].cpu().numpy()
    log(f"  {desc} ({fb_mode}): n={n:,} mean risk={out['prob'].mean() * 100:.2f}% [{time.time() - t0:.1f}s]")
    return out


def teacher_forced_next_state(model, ds, device, batch_size: int = 256, desc: str = "teacher-forced") -> np.ndarray:
    """One-step-ahead predictions s_hat_{t+1} | observed history and actions -> (N, T-1, D)."""
    n = len(ds)
    out = np.zeros((n, ds.n_bins - 1, ds.d_temporal), dtype=np.float32)
    for batch, _, sl in iter_batches(ds, batch_size, device, None, desc=desc):
        with torch.no_grad():
            o = model(batch["temporal"], batch["static"], batch["action"], batch["valid_mask"],
                      batch["absorbing_mask"])
        out[sl] = o["next_state_pred"].cpu().numpy()
    return out


# ═══════════════════════════════════════════════════════════════════════
# Negative-exposure control (85e procedure)
# ═══════════════════════════════════════════════════════════════════════
def make_nce_rng(seed: int = NCE_SEED, prior_shuffle_len: int | None = None) -> np.random.RandomState:
    """RandomState(seed). 85e shuffled the full outcome vector (NCO3) with the SAME
    RandomState before the NCE loop; pass prior_shuffle_len=len(outcomes_df) to
    advance the stream identically (exact 85e permutations)."""
    rng = np.random.RandomState(seed)
    if prior_shuffle_len:
        rng.shuffle(np.zeros(int(prior_shuffle_len)))
    return rng


def permute_observed_actions_within_bin(propensity_df: pd.DataFrame, ids: Sequence, n_bins: int,
                                        rng: np.random.RandomState, act_col: str | None = None) -> pd.DataFrame:
    """85e NCE1: shuffle the observed action column across ``ids`` within each bin
    (propensity-table row order, column action_idx_merged if present). Returns a copy."""
    prop = propensity_df.copy()
    if act_col is None:
        act_col = "action_idx_merged" if "action_idx_merged" in prop.columns else "action_idx"
    id_mask = prop["stay_id"].isin(set(np.asarray(ids).tolist()))
    for b in range(n_bins):
        m = id_mask & (prop["bin"] == b)
        vals = prop.loc[m, act_col].values.copy()
        rng.shuffle(vals)
        prop.loc[m, act_col] = vals
    return prop


# ═══════════════════════════════════════════════════════════════════════
# Per-stay frames, strata, paired bootstrap
# ═══════════════════════════════════════════════════════════════════════
def per_stay_frame(ds, bundle: Bundle, **cols) -> pd.DataFrame:
    """stay_id, mortality_28d, hf_subtype, rule_phenotype_2class + given (N,) arrays (dataset order)."""
    n = len(ds)
    frame = pd.DataFrame({"stay_id": np.asarray(ds.stay_ids), "mortality_28d": np.asarray(ds.mortality_28d, float)})
    for k, v in cols.items():
        v = np.asarray(v).reshape(-1)
        if len(v) != n:
            raise ValueError(f"column {k} has length {len(v)} != {n}")
        frame[k] = v
    ph = bundle.pheno_df[[c for c in ("stay_id", "hf_subtype", "rule_phenotype_2class") if c in bundle.pheno_df]]
    frame = frame.merge(ph, on="stay_id", how="left", validate="one_to_one")
    n_miss = int(frame["hf_subtype"].isna().sum()) if "hf_subtype" in frame else n
    if n_miss:
        log(f"WARNING: {n_miss} stays without phenotype labels")
    return frame


def standard_strata(frame: pd.DataFrame, extended: bool = False) -> "OrderedDict[str, np.ndarray]":
    """Spec strata: all; hfpef; hfpef non-shock; hfpef shock; no-HF non-shock; hfref."""
    hf = frame["hf_subtype"].astype(str).to_numpy()
    sh = frame["rule_phenotype_2class"].astype(str).to_numpy()
    s = OrderedDict()
    s["all"] = np.ones(len(frame), dtype=bool)
    s["hfpef"] = hf == "hfpef"
    s["hfpef_nonshock"] = (hf == "hfpef") & (sh == "non_shock")
    s["hfpef_shock"] = (hf == "hfpef") & (sh == "shock_predominant")
    s["nohf_nonshock"] = (hf == "no_hf") & (sh == "non_shock")
    s["hfref"] = hf == "hfref"
    if extended:
        s["no_hf"] = hf == "no_hf"
        s["nohf_shock"] = (hf == "no_hf") & (sh == "shock_predominant")
        s["hfmref"] = hf == "hfmref"
        s["hf_ef_unknown"] = hf == "hf_ef_unknown"
        s["hfref_or_hfmref"] = np.isin(hf, ["hfref", "hfmref"])
    return s


STANDARD_STRATUM_CONTRASTS = [("hfpef_nonshock_minus_nohf_nonshock", "hfpef_nonshock", "nohf_nonshock")]


def paired_bootstrap(frame: pd.DataFrame, value_cols: Sequence[str],
                     strata_fn: Callable | Mapping | None = None, n_boot: int = 1000, seed: int = SPEC_SEED,
                     contrasts: Sequence[tuple] | None = None,
                     stratum_contrasts: Sequence[tuple] | None = None, ci: float = 0.95,
                     scale: float = 1.0, desc: str = "paired bootstrap",
                     return_draws: bool = False):
    """Paired patient-level nonparametric bootstrap (models fixed, percentile CI).

    Each replicate resamples patients with replacement from the WHOLE frame; every
    stratum statistic is then recomputed within the replicate (so per-stratum
    means, paired within-patient differences and between-stratum differences of
    differences share the same resample).

    value_cols: per-stay columns whose stratum means are reported (e.g. y_uc, y_rule).
    contrasts:  [(name, col_a, col_b)] -> mean(col_a) - mean(col_b) (paired);
                default: every col in value_cols[1:] minus value_cols[0].
    stratum_contrasts: [(name, stratum_a, stratum_b)] -> for every contrast,
                contrast[stratum_a] - contrast[stratum_b] (default STANDARD_STRATUM_CONTRASTS
                when both strata exist, else none).
    scale: multiply estimates (100 -> % and pp).
    Returns a long DataFrame [stratum, kind ('mean'|'diff'|'diff_of_diff'), quantity, n,
    estimate, ci_lo, ci_hi, se_boot, n_boot_valid] (and the draws dict if return_draws).
    NaN values are ignored per column; empty strata give NaN.
    """
    value_cols = list(value_cols)
    if strata_fn is None:
        strata = standard_strata(frame)
    elif callable(strata_fn):
        strata = strata_fn(frame)
    else:
        strata = strata_fn
    strata = OrderedDict((k, np.asarray(v, dtype=bool)) for k, v in strata.items())
    if contrasts is None:
        contrasts = [(f"{c}_minus_{value_cols[0]}", c, value_cols[0]) for c in value_cols[1:]]
    contrasts = list(contrasts)
    need = list(dict.fromkeys(value_cols + [c for _, a, b in contrasts for c in (a, b)]))
    if stratum_contrasts is None:
        stratum_contrasts = [sc for sc in STANDARD_STRATUM_CONTRASTS if sc[1] in strata and sc[2] in strata]
    names = list(strata)
    S, K, n = len(names), len(need), len(frame)
    V = frame[need].to_numpy(dtype=np.float64)
    valid = ~np.isnan(V)
    Vf = np.where(valid, V, 0.0)
    M = np.stack([strata[k] for k in names]).astype(np.float64)       # (S, n)
    col = {c: j for j, c in enumerate(need)}

    def stats(w):  # w: (n,) weights -> means (S, K)
        W = M * w[None, :]
        num = W @ Vf
        den = W @ valid.astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            return num / den

    point = stats(np.ones(n))
    rng = np.random.default_rng(seed)
    draws = np.full((n_boot, S, K), np.nan)
    for b in tqdm(range(n_boot), desc=f"  {desc}", mininterval=5, leave=False):
        idx = rng.integers(0, n, size=n)
        draws[b] = stats(np.bincount(idx, minlength=n).astype(np.float64))

    alpha = (1.0 - ci) / 2.0
    rows = []

    def summarise(stratum, kind, qty, nn, est, dr):
        dr = dr[np.isfinite(dr)]
        lo, hi = (np.percentile(dr, [100 * alpha, 100 * (1 - alpha)]) if dr.size else (np.nan, np.nan))
        rows.append({"stratum": stratum, "kind": kind, "quantity": qty, "n": int(nn),
                     "estimate": est * scale, "ci_lo": lo * scale, "ci_hi": hi * scale,
                     "se_boot": (float(np.std(dr, ddof=1)) * scale if dr.size > 1 else np.nan),
                     "n_boot_valid": int(dr.size)})

    contrast_draws = {}
    for si, s in enumerate(names):
        nn = int(strata[s].sum())
        for c in value_cols:
            summarise(s, "mean", c, nn, point[si, col[c]], draws[:, si, col[c]])
        for cname, a, b in contrasts:
            est = point[si, col[a]] - point[si, col[b]]
            dr = draws[:, si, col[a]] - draws[:, si, col[b]]
            contrast_draws[(s, cname)] = (est, dr)
            summarise(s, "diff", cname, nn, est, dr)
    for scname, sa, sb in stratum_contrasts:
        for cname, _, _ in contrasts:
            ea, da = contrast_draws[(sa, cname)]
            eb, db = contrast_draws[(sb, cname)]
            summarise(scname, "diff_of_diff", cname, int(strata[sa].sum() + strata[sb].sum()), ea - eb, da - db)
    res = pd.DataFrame(rows)
    if return_draws:
        return res, {"strata": names, "columns": need, "draws": draws, "point": point}
    return res


def get_est(res: pd.DataFrame, stratum: str, quantity: str, kind: str | None = None) -> pd.Series:
    """Row of a paired_bootstrap result (raises if absent)."""
    m = (res["stratum"] == stratum) & (res["quantity"] == quantity)
    if kind is not None:
        m &= res["kind"] == kind
    sub = res[m]
    if sub.empty:
        raise KeyError(f"{stratum}/{quantity}/{kind} not in bootstrap result")
    return sub.iloc[0]


# ═══════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════
def fmt_num(x, d: int = 2) -> str:
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "NA" if not np.isfinite(xf) else f"{xf:.{d}f}"


def fmt_ci(est, lo, hi, d: int = 2) -> str:
    """'-1.37 (-1.49 to -1.25)'."""
    return f"{fmt_num(est, d)} ({fmt_num(lo, d)} to {fmt_num(hi, d)})"


def fmt_row_ci(row: pd.Series, d: int = 2) -> str:
    return fmt_ci(row["estimate"], row["ci_lo"], row["ci_hi"], d)


def repro_check(rd_all_pp: float, rd_hfpef_pp: float, tol_all: float = REPRO_TOL_ALL_PP,
                tol_hfpef: float = REPRO_TOL_HFPEF_PP) -> dict:
    """REPRO_CHECK against the published non-adaptive contrast (pp)."""
    d_all = abs(float(rd_all_pp) - REPRO_TARGETS["rd_all"])
    d_hf = abs(float(rd_hfpef_pp) - REPRO_TARGETS["rd_hfpef"])
    ok = bool(np.isfinite(d_all) and np.isfinite(d_hf) and d_all <= tol_all and d_hf <= tol_hfpef)
    status = "PASS" if ok else "FAIL"
    line = (f"REPRO_CHECK: {status} (RD_all={fmt_num(rd_all_pp)} pp vs {REPRO_TARGETS['rd_all']} "
            f"[|diff|={fmt_num(d_all, 3)} <= {tol_all}]; RD_hfpef={fmt_num(rd_hfpef_pp)} pp vs "
            f"{REPRO_TARGETS['rd_hfpef']} [|diff|={fmt_num(d_hf, 3)} <= {tol_hfpef}])")
    print(line, flush=True)
    return {"status": status, "pass": ok, "line": line, "abs_diff_all": d_all, "abs_diff_hfpef": d_hf}


def repro_check_extended(risk_uc_all_pct: float, risk_uc_hfpef_pct: float,
                         rd_rule_all_pp: float, rd_rule_hfpef_pp: float,
                         rd_legacy_all_pp: float | None = None, rd_legacy_hfpef_pp: float | None = None) -> dict:
    """Spec REPRO_CHECK on the corrected rule arm plus two diagnostics that tell the
    author WHY it passes or fails:
      REPRO_CHECK_UC      usual-care risk all / HFpEF vs 11.78 % / 15.77 % (tolerances 0.05 / 0.10 pp);
                          fails only if checkpoint / data / split differ;
      REPRO_CHECK_LEGACY  legacy zero-action arm (pre-v1.0.1 'rule' = action 0) RD vs -1.37 / -2.27.
    """
    main = repro_check(rd_rule_all_pp, rd_rule_hfpef_pp)
    d1 = abs(float(risk_uc_all_pct) - REPRO_TARGETS["risk_uc_all"])
    d2 = abs(float(risk_uc_hfpef_pct) - REPRO_TARGETS["risk_uc_hfpef"])
    uc_ok = bool(d1 <= REPRO_TOL_ALL_PP and d2 <= REPRO_TOL_HFPEF_PP)
    line_uc = (f"REPRO_CHECK_UC: {'PASS' if uc_ok else 'FAIL'} (usual-care risk all={fmt_num(risk_uc_all_pct)}% "
               f"vs {REPRO_TARGETS['risk_uc_all']}; HFpEF={fmt_num(risk_uc_hfpef_pct)}% vs "
               f"{REPRO_TARGETS['risk_uc_hfpef']})")
    print(line_uc, flush=True)
    out = {"status": main["status"], "pass": main["pass"], "line": main["line"],
           "uc_status": "PASS" if uc_ok else "FAIL", "uc_line": line_uc}
    if rd_legacy_all_pp is not None and rd_legacy_hfpef_pp is not None:
        e1 = abs(float(rd_legacy_all_pp) - REPRO_TARGETS["rd_all"])
        e2 = abs(float(rd_legacy_hfpef_pp) - REPRO_TARGETS["rd_hfpef"])
        lg_ok = bool(e1 <= REPRO_TOL_ALL_PP and e2 <= REPRO_TOL_HFPEF_PP)
        line_lg = (f"REPRO_CHECK_LEGACY: {'PASS' if lg_ok else 'FAIL'} (zero-action arm RD all="
                   f"{fmt_num(rd_legacy_all_pp)} pp vs {REPRO_TARGETS['rd_all']}; HFpEF="
                   f"{fmt_num(rd_legacy_hfpef_pp)} pp vs {REPRO_TARGETS['rd_hfpef']})")
        print(line_lg, flush=True)
        out.update(legacy_status="PASS" if lg_ok else "FAIL", legacy_line=line_lg)
        if lg_ok and not main["pass"]:
            print("REPRO_NOTE: " + LEGACY_RULE_NOTE, flush=True)
    return out


def assert_aggregate_only(df: pd.DataFrame, label: str = "table") -> None:
    """Refuse to write patient identifiers or timestamps to tables/reports."""
    bad = [c for c in df.columns if str(c).lower() in FORBIDDEN_TABLE_COLUMNS]
    bad += [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if bad:
        raise ValueError(f"{label}: refusing to write identifier/time columns {sorted(set(map(str, bad)))}")


def save_table(df: pd.DataFrame, path: str | Path, float_format: str = "%.6g") -> Path:
    """Write an AGGREGATE table (CSV) after the identifier guard."""
    path = Path(path)
    assert_aggregate_only(df, path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format=float_format)
    log(f"  table -> {path} ({len(df)} rows)")
    return path


def save_interim(df: pd.DataFrame, root: Path, filename: str) -> Path:
    """Stay-level intermediate: ONLY under data/interim/bmc_medicine/ (gitignored, never synced)."""
    path = root / "data" / "interim" / "bmc_medicine" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log(f"  stay-level interim -> {path} ({len(df):,} rows; do not sync)")
    return path


def _fmt_paste(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "NA" if not np.isfinite(v) else f"{float(v):.4f}"
    return str(v)


def write_report(path: str | Path, lines: Iterable[str], paste_block_dict: Mapping, tag: str) -> str:
    """Write the human-readable report ending with the PASTE-BACK block; also prints it."""
    path = Path(path)
    body = [str(x) for x in lines]
    body += ["", f"===== PASTE-BACK BLOCK START ({tag}) ====="]
    for k, v in paste_block_dict.items():
        key = str(k).replace(":", "_").strip()
        body.append(f"{key}: {_fmt_paste(v)}")
    body.append(f"===== PASTE-BACK BLOCK END ({tag}) =====")
    text = "\n".join(body) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(text, flush=True)
    log(f"  report -> {path}")
    return text


def report_header(title: str, root: Path, args: argparse.Namespace | None = None) -> list[str]:
    lines = ["=" * 72, title, "=" * 72, f"Generated: {now()}", f"Project root: {root}"]
    if args is not None:
        lines.append("Arguments: " + " ".join(f"--{k.replace('_', '-')}={v}" for k, v in sorted(vars(args).items())))
    lines.append("")
    return lines
