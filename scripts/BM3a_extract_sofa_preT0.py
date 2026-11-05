#!/usr/bin/env python3
"""
BM3a_extract_sofa_preT0 — Baseline (pre-T0) SOFA score for the Step C analytic
cohort (BMC Medicine revision, BM3a).

Purpose
-------
Reviewers asked whether the HFpEF contrast is explained by baseline organ
dysfunction. BM3a builds a standard SOFA score (Vincent 1996 / Ferreira 2001
points) for every stay in the 45c state tensors from information recorded
between ICU admission and T0 (intime + 6 h; optional --lookback-hours before
intime), taking the worst value of each component in that window:

  Respiration     PaO2/FiO2: arterial pO2 (labevents, d_labitems 'pO2' / Blood /
                  Blood Gas, specimen type 52033 'ART.') paired with the most
                  recent FiO2 within the preceding 4 h (10d FiO2 + raw chartevents
                  223835, normalised to a fraction; 0.21 if none and neither
                  invasive ventilation nor NIV in the window). Without PaO2:
                  SpO2/FiO2 on SpO2 <= 97 % mapped with the Rice/Pandharipande
                  thresholds 512/357/214/89 <-> P/F
                  400/300/200/100. Points 3-4 need invasive mechanical
                  ventilation in the window (10h indicators/procedures); else
                  capped at 2.
  Coagulation     minimum platelets (10e 'platelet').
  Liver           maximum total bilirubin (10e 'bilirubin_total').
  Cardiovascular  minimum MAP (10d invasive or cuff) and maximum vasoactive
                  rates (10f) in ug/kg/min (per-minute rates divided by weight;
                  vasopressin in U/min). MAP < 70 -> 1; dopamine <= 5 or any
                  dobutamine -> 2; dopamine > 5, epinephrine <= 0.1 or
                  norepinephrine <= 0.1 -> 3; dopamine > 15, epinephrine > 0.1 or
                  norepinephrine > 0.1 -> 4. CONVENTION: any vasopressin or
                  phenylephrine infusion scores (at least) 3.
  CNS             minimum total GCS (chartevents 220739 + 223900 + 223901,
                  summed at each charttime, a missing component carried forward
                  from <= 1 h earlier). CONVENTION (--gcs-ett, default 'normal' =
                  MIMIC-code): an assessment whose verbal component is
                  'No Response-ETT' counts as GCS 15 (pre-sedation GCS unknown);
                  the alternative 'verbal1' (charted valuenum 1) is always stored
                  as well (gcs_min_ett_verbal1, sofa_total_ett_verbal1).
  Renal           maximum creatinine (10e) and urine output in the window
                  (10g 'urine', mL; outputevents are ICU-only, so the total is
                  scaled to 24 h over the ICU part intime -> T0 even with
                  --lookback-hours; < 500 mL/day -> 3, < 200 -> 4); the larger
                  score is used.

A component without any usable measurement scores 0 (convention) and is
flagged (sofa_<component>_missing); missingness is reported by HF subtype
and shock stratum. The script also stores the NE-equivalent maximum
(configs/action_space_v1.yaml factors; vasopressin factor per U/min) and the
underlying raw values so that alternative conventions can be recomputed.

Protocol reference: BMC Medicine revision BM3a (spec bmc_medicine_revision
SPEC.md); mirrors the chunked itemid-filtered extraction of 10d / 10e.
Compute: C4 (qsub env/sge/submit_c4.sh; C3 submit_c3.sh also fine). One chunked pass over
labevents (~158 M rows) and one over chartevents (~433 M rows) dominate
(pandas C parser ~0.4 M rows/s with usecols: chartevents ~20 min, labevents
~7-12 min); total about 30-60 min, budget 2 h. Rows are filtered by itemid,
cohort and the stay's time window INSIDE each 5 M-row chunk, so peak memory
stays at a few GB (one chunk plus the 10d/10e/10f staging subsets).

Inputs
------
- configs/80_step_c_training_v1.yaml (data.state_tensors, data.phenotype_labels)
- configs/action_space_v1.yaml (ne_equivalent_factors; optional)
- data/interim/45c_state_tensors_v1_mimic.parquet        (cohort stay_ids)
- data/processed/phenotype_labels_frozen_v1_mimic.parquet (strata for the report)
- data/staging/10c_timestamps_mimic.parquet              (intime, time_zero_6h)
- data/staging/10d_vitals_mimic.parquet                  (MAP, SpO2, FiO2)
- data/staging/10e_labs_mimic.parquet                    (platelet, bilirubin, creatinine)
- data/staging/10f_meds_infusions_mimic.parquet          (vasoactive infusions)
- data/staging/10g_output_events_mimic.parquet, 10g_weight_mimic.parquet
- data/staging/10h_vent_indicators_mimic.parquet, 10h_procedures_mimic.parquet
- data/raw/mimiciv/hosp/labevents.csv.gz, d_labitems.csv.gz   (pO2 + specimen type)
- data/raw/mimiciv/icu/chartevents.csv.gz, d_items.csv.gz      (GCS, FiO2)

Outputs
-------
- data/interim/bmc_medicine/BM3a_sofa_preT0.parquet (+ .meta.json)
  STAY-LEVEL, gitignored, never synced; read by BM3b and BM2 --sofa.
  Columns: stay_id, sofa_total, sofa_{resp,coag,liver,cardio,cns,renal},
  sofa_<component>_missing, n_components_missing, sofa_noncv, pf_ratio,
  sf_ratio, resp_source, vent_invasive_window, niv_window, gcs_min,
  gcs_min_ett_verbal1, sofa_cns_ett_verbal1, sofa_total_ett_verbal1, map_min,
  ne_eq_max, <drug>_max, platelet_min, bilirubin_max, creatinine_max,
  urine_ml_window, urine_ml_24h_scaled, sofa_renal_creat, sofa_renal_uo,
  window_hours.
- tables/bmc_medicine/BM3a_sofa_by_stratum.csv        (aggregate)
- tables/bmc_medicine/BM3a_component_distribution.csv (aggregate)
- tables/bmc_medicine/BM3a_source_coverage.csv        (aggregate)
- reports/bmc_medicine/BM3a_summary.txt (ends with the PASTE-BACK BLOCK)
  (a non-default --out, e.g. BM3a_sofa_preT0_lookback6.parquet, writes
  BM3a_summary_lookback6.txt and BM3a_*_lookback6.csv with paste tag BM3a_lookback6)

Usage
-----
    # Shirokane (qsub; large raw-file pass). C4 per the revision plan; C3
    # (env/sge/submit_c3.sh, 16 slots) is also enough. The C3/C4 templates set no
    # thread limits; qsub -V forwards the prefix (pyarrow honours OMP_NUM_THREADS).
    OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \\
        qsub -N sBM3a_sofa env/sge/submit_c4.sh scripts/bmc_medicine/BM3a_extract_sofa_preT0.py
    # smoke test (400 stays, first 3 chunks of each raw file)
    env/run_py.sh scripts/bmc_medicine/BM3a_extract_sofa_preT0.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import log, phase  # noqa: E402

SCRIPT_VERSION = "1.1"   # 1.1: review fixes (point-record NE-eq, window-edge overlap, UO scaling with lookback, NaT windows, NIV FiO2)
TAG = "BM3a"
N_PHASES = 7
OUT_FILE = "BM3a_sofa_preT0.parquet"


def output_suffix(out_name: str) -> str:
    """'' for the default --out; otherwise a suffix (e.g. '_lookback6' for
    BM3a_sofa_preT0_lookback6.parquet) used for the report, the tables and the paste tag,
    so that a sensitivity run never overwrites the primary BM3a outputs."""
    stem, base = Path(out_name).stem, Path(OUT_FILE).stem
    if stem == base:
        return ""
    rest = stem[len(base):] if stem.startswith(base) else stem
    return "_" + rest.lstrip("_") if rest.strip("_") else "_alt"
CHUNK_SIZE = 5_000_000
QUICK_MAX_CHUNKS = 3

# chartevents (d_items namespace)
GCS_ITEMIDS = OrderedDict([("eye", 220739), ("verbal", 223900), ("motor", 223901)])
GCS_RANGE = {"eye": (1, 4), "verbal": (1, 5), "motor": (1, 6)}
ETT_VERBAL_VALUES = {"no response-ett"}
FIO2_ITEMID = 223835
# labevents (d_labitems namespace) — resolved at runtime; fallbacks:
PO2_ITEMID_FALLBACK = [50821]
SPECIMEN_ITEMID_FALLBACK = [52033]
PO2_RANGE = (10.0, 600.0)                # 10e range filter for pO2 (mmHg)
FIO2_FRAC_RANGE = (0.21, 1.0)
ROOM_AIR_FIO2 = 0.21
SPO2_SF_MAX = 97.0
PF_THRESHOLDS = (400.0, 300.0, 200.0, 100.0)
SF_THRESHOLDS = (512.0, 357.0, 214.0, 89.0)   # Rice 2007 / Pandharipande 2009

# ventilation definitions (10h staging)
ETT_DEVICE_PATTERN = r"endotracheal|tracheostomy tube"
NIV_DEVICE_PATTERN = r"bipap|cpap"
NIV_VENT_MODES = {"duopap", "niv", "niv-st", "ncpap", "ncpap-ps"}   # Hamilton non-invasive modes
NONVENT_VENT_MODES = {"standby", "", "nan", "none", "no", "off", "0", "n", "false"}
PROC_INVASIVE_ITEMIDS = {225792, 224385}       # Invasive Ventilation, Intubation
PROC_NIV_ITEMIDS = {225794}                    # Non-invasive Ventilation
PROC_NOT_VENT_ITEMIDS = {227194, 225468, 225477}  # extubation events

VASO_DRUGS = ["norepinephrine", "epinephrine", "dopamine", "dobutamine", "vasopressin", "phenylephrine"]
NE_FACTORS_DEFAULT = {"norepinephrine": 1.0, "epinephrine": 1.0, "phenylephrine": 0.1, "dopamine": 0.01,
                      "vasopressin": 2.5, "dobutamine": 0.0}
COMPONENTS = ["resp", "coag", "liver", "cardio", "cns", "renal"]
COMPONENT_LABEL = {"resp": "respiration", "coag": "coagulation", "liver": "liver",
                   "cardio": "cardiovascular", "cns": "CNS", "renal": "renal"}


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BM3a: baseline (ICU admission to T0) SOFA score")
    p.add_argument("--seed", type=int, default=common.SPEC_SEED, help="seed for the --quick subsample")
    p.add_argument("--quick", action="store_true",
                   help=f"smoke test: {common.QUICK_N_STAYS} stays and the first {QUICK_MAX_CHUNKS} chunks "
                        "of each raw file (output flagged quick)")
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="rows per raw CSV chunk")
    p.add_argument("--max-chunks", type=int, default=0,
                   help=f"stop each raw pass after N chunks (0 = all; --quick sets {QUICK_MAX_CHUNKS})")
    p.add_argument("--lookback-hours", type=float, default=0.0,
                   help="start the window this many hours BEFORE ICU admission (default 0 = intime)")
    p.add_argument("--po2-specimen", choices=["art", "art_or_unknown", "any"], default="art",
                   help="pO2 specimen filter on labevents 52033 'Specimen Type' (default ART. only)")
    p.add_argument("--fio2-pair-hours", type=float, default=4.0,
                   help="FiO2 must be charted within this many hours before the PaO2/SpO2")
    p.add_argument("--gcs-carry-hours", type=float, default=1.0,
                   help="carry a GCS component forward for at most this many hours")
    p.add_argument("--gcs-ett", choices=["normal", "verbal1"], default="normal",
                   help="primary convention for 'No Response-ETT' verbal: normal = GCS 15 (MIMIC-code), "
                        "verbal1 = charted verbal 1 (the other one is stored as *_ett_verbal1 / *_ett_normal)")
    p.add_argument("--niv-as-vent", action="store_true",
                   help="count non-invasive ventilation as respiratory support for respiration 3-4")
    p.add_argument("--default-weight-kg", type=float, default=80.0,
                   help="weight used for per-minute vasoactive rates when no weight is recorded")
    p.add_argument("--mimic-root", default=None,
                   help="MIMIC-IV root (default: configs/00f datasets.mimic.entrypoint = data/raw/mimiciv)")
    p.add_argument("--out", default=OUT_FILE,
                   help="file name under data/interim/bmc_medicine/; a non-default name also suffixes the "
                        "report, tables and paste tag (e.g. BM3a_sofa_preT0_lookback6.parquet -> "
                        "BM3a_summary_lookback6.txt, tag BM3a_lookback6) so the primary outputs are kept")
    args = p.parse_args(argv)
    if args.quick and args.max_chunks == 0:
        args.max_chunks = QUICK_MAX_CHUNKS
    return args


# ═══════════════════════════════════════════════════════════════════════
# I/O helpers
# ═══════════════════════════════════════════════════════════════════════
def mimic_root(root: Path, override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    try:
        from hfpef_dtr.config import dataset_entrypoint, load_config
        return dataset_entrypoint(load_config(root), "mimic")
    except Exception as exc:  # pragma: no cover - fallback when 00f config is absent
        log(f"  NOTE: hfpef_dtr.config unavailable ({type(exc).__name__}); using data/raw/mimiciv")
        return root / "data" / "raw" / "mimiciv"


def hrs(x: float) -> pd.Timedelta:
    """Timedelta of x hours (explicit unit; avoids numpy's generic-unit deprecation path)."""
    return pd.to_timedelta(float(x), unit="h")


def as_int64(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("int64")


def read_staging(path: Path, columns: list[str], filters: list | None, label: str) -> pd.DataFrame:
    """Parquet read with pyarrow predicate push-down (falls back to a pandas filter)."""
    if not path.is_file():
        raise FileNotFoundError(f"{label}: staging file not found: {path}")
    t0 = time.time()
    try:
        df = pd.read_parquet(path, columns=columns, filters=filters)
    except Exception as exc:  # pragma: no cover - old pyarrow / dtype mismatch
        log(f"  {label}: push-down filter failed ({type(exc).__name__}: {exc}); filtering in pandas")
        fcols = [c for c, _, _ in (filters or []) if c not in columns]   # filter-only columns must be read too
        df = pd.read_parquet(path, columns=columns + list(dict.fromkeys(fcols)))
        for col, op, val in filters or []:
            if op == "in":
                df = df[df[col].isin(list(val))]
            elif op == ">=":
                df = df[df[col] >= val]
            elif op == "<=":
                df = df[df[col] <= val]
            elif op == "==":
                df = df[df[col] == val]
        df = df[columns]
    df = df.dropna(subset=["stay_id"]).copy()
    df["stay_id"] = as_int64(df["stay_id"])
    log(f"  {label}: {len(df):,} rows after push-down filters [{time.time() - t0:.1f}s]")
    return df


def chunked_csv(path: Path, usecols: list[str], dtype: dict, chunk_size: int, filter_fn,
                desc: str, max_chunks: int = 0, log_every: int = 10) -> tuple[pd.DataFrame, int, int, bool]:
    """Chunked gz-CSV pass with a byte-level tqdm bar (exact % / ETA) and timestamped chunk logs.

    Returns (kept rows, rows read, chunks read, truncated?)."""
    if not path.is_file():
        raise FileNotFoundError(f"raw file not found: {path}")
    size = path.stat().st_size
    compression = "gzip" if path.suffix == ".gz" else None
    kept, n_read, n_kept, n_chunks, truncated = [], 0, 0, 0, False
    t0 = time.time()
    with open(path, "rb") as fh:
        bar = tqdm(total=size, desc=f"  {desc}", unit="B", unit_scale=True, unit_divisor=1024,
                   mininterval=15, file=sys.stdout)
        wrapped = CallbackIOWrapper(bar.update, fh, "read")
        reader = pd.read_csv(wrapped, compression=compression, chunksize=chunk_size,
                             usecols=usecols, dtype=dtype)
        for chunk in reader:
            n_chunks += 1
            n_read += len(chunk)
            sub = filter_fn(chunk)
            if len(sub):
                kept.append(sub)
                n_kept += len(sub)
            bar.set_postfix_str(f"chunk {n_chunks}, read {n_read:,}, kept {n_kept:,}", refresh=False)
            if n_chunks % log_every == 0:
                log(f"    {desc}: chunk {n_chunks}: read {n_read:,}, kept {n_kept:,} "
                    f"[{(time.time() - t0) / 60:.1f} min]")
            if max_chunks and n_chunks >= max_chunks:
                truncated = True
                log(f"    {desc}: stopping after {n_chunks} chunks (--max-chunks / --quick)")
                break
        bar.close()
    out = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=usecols)
    log(f"  {desc}: read {n_read:,} rows in {n_chunks} chunks, kept {len(out):,} "
        f"[{(time.time() - t0) / 60:.1f} min]")
    return out, n_read, n_chunks, truncated


def time_filter(ch: pd.DataFrame, key: str, lo: pd.Series, hi: pd.Series) -> pd.DataFrame:
    """In-chunk filter: parse charttime only for the (few) itemid/cohort-matched rows and keep
    those inside [lo, hi) of their stay/subject, so a full raw pass keeps memory small."""
    if ch.empty:
        return ch
    t = pd.to_datetime(ch["charttime"], errors="coerce")
    k = pd.to_numeric(ch[key], errors="coerce")
    keep = (t >= k.map(lo)) & (t < k.map(hi))
    out = ch.loc[keep].copy()
    out["charttime"] = t.loc[keep]
    return out


def in_window(df: pd.DataFrame, win: pd.DataFrame, time_col: str, pad_before_h: float = 0.0) -> pd.DataFrame:
    """Rows whose time_col lies in [win_start - pad, win_end) of their stay."""
    if df.empty:
        return df.iloc[0:0].copy()
    m = df.merge(win[["stay_id", "win_start", "win_end"]], on="stay_id", how="inner")
    lo = m["win_start"] - hrs(pad_before_h)
    keep = m[time_col].notna() & (m[time_col] >= lo) & (m[time_col] < m["win_end"])
    return m.loc[keep].drop(columns=["win_start", "win_end"]).reset_index(drop=True)


def per_stay(series_by_stay: pd.Series, ids: np.ndarray) -> np.ndarray:
    return series_by_stay.reindex(ids).to_numpy(dtype=np.float64)


def overlaps_window(start: pd.Series, end: pd.Series, ws: pd.Series, we: pd.Series) -> pd.Series:
    """Interval [start, end) overlaps [ws, we) with positive length; a point record (end missing or
    end <= start) counts only if start lies in [ws, we). (An infusion/procedure that ended exactly
    at the window start does not count.)"""
    end = end.fillna(start)
    point = end <= start
    return start.notna() & (start < we) & ((end > ws) | (point & (start >= ws)))


# ═══════════════════════════════════════════════════════════════════════
# Scoring (SOFA 2001 points; strict inequalities as in Vincent et al.)
# ═══════════════════════════════════════════════════════════════════════
def score_ratio(ratio: np.ndarray, thresholds, support: np.ndarray) -> np.ndarray:
    t1, t2, t3, t4 = thresholds
    s = np.full(ratio.shape, np.nan)
    ok = np.isfinite(ratio)
    r = ratio[ok]
    v = np.zeros(r.shape)
    v[r < t1] = 1
    v[r < t2] = 2
    v[r < t3] = 3
    v[r < t4] = 4
    s[ok] = v
    return np.where(support | ~np.isfinite(s), s, np.minimum(s, 2.0))


def score_platelets(x):
    return np.select([x < 20, x < 50, x < 100, x < 150, np.isfinite(x)], [4, 3, 2, 1, 0], np.nan)


def score_bilirubin(x):
    return np.select([x >= 12.0, x >= 6.0, x >= 2.0, x >= 1.2, np.isfinite(x)], [4, 3, 2, 1, 0], np.nan)


def score_creatinine(x):
    return np.select([x >= 5.0, x >= 3.5, x >= 2.0, x >= 1.2, np.isfinite(x)], [4, 3, 2, 1, 0], np.nan)


def score_urine(uo24):
    return np.select([uo24 < 200, uo24 < 500, np.isfinite(uo24)], [4, 3, 0], np.nan)


def score_gcs(g):
    return np.select([g < 6, g <= 9, g <= 12, g <= 14, np.isfinite(g)], [4, 3, 2, 1, 0], np.nan)


def score_cardio(map_min, dmax: dict[str, np.ndarray]) -> np.ndarray:
    z = np.zeros_like(map_min)
    dopa = np.nan_to_num(dmax["dopamine"], nan=0.0)
    dobu = np.nan_to_num(dmax["dobutamine"], nan=0.0)
    epi = np.nan_to_num(dmax["epinephrine"], nan=0.0)
    ne = np.nan_to_num(dmax["norepinephrine"], nan=0.0)
    vp = np.nan_to_num(dmax["vasopressin"], nan=0.0)
    phe = np.nan_to_num(dmax["phenylephrine"], nan=0.0)
    s = np.where(np.nan_to_num(map_min, nan=np.inf) < 70.0, 1.0, z)
    s = np.where(((dopa > 0) & (dopa <= 5)) | (dobu > 0), np.maximum(s, 2.0), s)
    s = np.where((dopa > 5) | ((epi > 0) & (epi <= 0.1)) | ((ne > 0) & (ne <= 0.1)) | (vp > 0) | (phe > 0),
                 np.maximum(s, 3.0), s)
    s = np.where((dopa > 15) | (epi > 0.1) | (ne > 0.1), np.maximum(s, 4.0), s)
    any_info = np.isfinite(map_min) | (dopa > 0) | (dobu > 0) | (epi > 0) | (ne > 0) | (vp > 0) | (phe > 0)
    return np.where(any_info, s, np.nan)


# ═══════════════════════════════════════════════════════════════════════
# Component builders
# ═══════════════════════════════════════════════════════════════════════
def normalise_fio2(v: pd.Series) -> pd.Series:
    v = pd.to_numeric(v, errors="coerce")
    f = v.where(v <= 1.0, v / 100.0)
    return f.where((f >= FIO2_FRAC_RANGE[0]) & (f <= FIO2_FRAC_RANGE[1]))


def ventilation_flags(vent: pd.DataFrame, procs: pd.DataFrame, win: pd.DataFrame) -> tuple[set, set, dict]:
    """Invasive ventilation / NIV in the window from 10h chartevents indicators and procedures."""
    diag = {}
    v = in_window(vent, win, "charttime")
    val = v["value"].astype(str).str.strip().str.lower()
    is_dev = v["indicator_type"] == "o2_device"
    is_mode = v["indicator_type"].isin(["mech_vent", "vent_mode"])
    inv_ce = (is_dev & val.str.contains(ETT_DEVICE_PATTERN, regex=True)) | \
             (is_mode & v["value"].notna() & ~val.isin(NIV_VENT_MODES | NONVENT_VENT_MODES))
    niv_ce = (is_dev & val.str.contains(NIV_DEVICE_PATTERN, regex=True)) | (is_mode & val.isin(NIV_VENT_MODES))
    inv_ids = set(v.loc[inv_ce, "stay_id"].tolist())
    niv_ids = set(v.loc[niv_ce, "stay_id"].tolist())
    diag["vent_rows_window"] = int(len(v))
    diag["inv_from_chartevents"] = len(inv_ids)
    # aggregate value counts so the author can audit the mode / device classification on real data
    vc = lambda m: "; ".join(f"{k}={n}" for k, n in val[m].value_counts().head(25).items())
    diag["modes_invasive"] = vc(is_mode & inv_ce)
    diag["modes_not_invasive"] = vc(is_mode & ~inv_ce)
    diag["devices_window"] = vc(is_dev)
    p = procs.merge(win[["stay_id", "win_start", "win_end"]], on="stay_id", how="inner")
    if len(p):
        overlap = overlaps_window(p["starttime"], p["endtime"], p["win_start"], p["win_end"])
        iid = pd.to_numeric(p["itemid"], errors="coerce")
        p_niv = overlap & iid.isin(PROC_NIV_ITEMIDS)
        p_inv = overlap & (iid.isin(PROC_INVASIVE_ITEMIDS) |
                           ~iid.isin(PROC_NIV_ITEMIDS | PROC_NOT_VENT_ITEMIDS))
        proc_inv = set(p.loc[p_inv, "stay_id"].tolist())
        inv_ids |= proc_inv
        niv_ids |= set(p.loc[p_niv, "stay_id"].tolist())
        diag["inv_from_procedures"] = len(proc_inv)
        diag["proc_itemids_window"] = "; ".join(f"{int(k)}={n}" for k, n in
                                                iid[overlap].value_counts().items() if pd.notna(k))
    else:
        diag["inv_from_procedures"] = 0
    return inv_ids, niv_ids, diag


def pair_with_fio2(meas: pd.DataFrame, fio2: pd.DataFrame, support_ids: set, pair_h: float) -> pd.DataFrame:
    """Most recent FiO2 within pair_h hours before each measurement; 0.21 when none and not supported."""
    if meas.empty:
        return meas.assign(fio2=pd.Series(dtype=float), fio2_imputed=pd.Series(dtype=bool))
    left = meas.sort_values("charttime").reset_index(drop=True)
    right = fio2[["stay_id", "charttime", "fio2"]].dropna().sort_values("charttime").reset_index(drop=True)
    out = pd.merge_asof(left, right, on="charttime", by="stay_id", direction="backward",
                        tolerance=hrs(pair_h), allow_exact_matches=True)
    supported = out["stay_id"].isin(support_ids)
    out["fio2_imputed"] = out["fio2"].isna() & ~supported
    out.loc[out["fio2_imputed"], "fio2"] = ROOM_AIR_FIO2
    return out


def gcs_by_stay(ce_gcs: pd.DataFrame, win: pd.DataFrame, carry_h: float) -> tuple[pd.DataFrame, dict]:
    """Minimum total GCS per stay under both ETT conventions."""
    diag = {}
    inv = {v: k for k, v in GCS_ITEMIDS.items()}
    g = ce_gcs.copy()
    g["comp"] = g["itemid"].map(inv)
    g["val"] = pd.to_numeric(g["valuenum"], errors="coerce")
    g["ett"] = (g["comp"] == "verbal") & g["value"].astype(str).str.strip().str.lower().isin(ETT_VERBAL_VALUES)
    g.loc[g["ett"] & g["val"].isna(), "val"] = 1.0
    for c, (lo, hi) in GCS_RANGE.items():
        bad = (g["comp"] == c) & ~g["val"].between(lo, hi)
        g.loc[bad, "val"] = np.nan
    g = g.dropna(subset=["val"])
    diag["gcs_rows"] = int(len(g))
    diag["gcs_ett_rows"] = int(g["ett"].sum())
    g = in_window(g, win, "charttime", pad_before_h=carry_h)
    times = in_window(g, win, "charttime")[["stay_id", "charttime"]].drop_duplicates()
    times = times.sort_values("charttime").reset_index(drop=True)
    if times.empty:
        return pd.DataFrame(columns=["gcs_min_ett_normal", "gcs_min_ett_verbal1"]), diag
    for c in GCS_ITEMIDS:
        cc = (g[g["comp"] == c].groupby(["stay_id", "charttime"], as_index=False)
              .agg(**{c: ("val", "min"), f"{c}_ett": ("ett", "max")})
              .sort_values("charttime").reset_index(drop=True))
        times = pd.merge_asof(times, cc, on="charttime", by="stay_id", direction="backward",
                              tolerance=hrs(carry_h), allow_exact_matches=True)
    complete = times[["eye", "verbal", "motor"]].notna().all(axis=1)
    total = (times["eye"] + times["verbal"] + times["motor"]).where(complete)
    ett = times["verbal_ett"].eq(True)
    times["tot_verbal1"] = total
    times["tot_normal"] = np.where(ett, 15.0, total)
    diag["gcs_assessments_window"] = int(len(times))
    diag["gcs_assessments_complete"] = int(complete.sum())
    diag["gcs_assessments_ett"] = int(ett.sum())
    out = times.groupby("stay_id").agg(gcs_min_ett_normal=("tot_normal", "min"),
                                       gcs_min_ett_verbal1=("tot_verbal1", "min"))
    return out, diag


def parse_rate_units(uom: pd.Series) -> pd.DataFrame:
    u = uom.fillna("").astype(str).str.lower().str.replace(" ", "", regex=False)
    mass = np.select([u.str.match(r"^(mcg|ug|µg)"), u.str.match(r"^mg"), u.str.match(r"^ng")],
                     [1.0, 1000.0, 0.001], np.nan)
    units = np.select([u.str.match(r"^(units|unit|u/)"), u.str.match(r"^(mu|milliunits)")], [1.0, 0.001], np.nan)
    per_kg = u.str.contains("/kg", regex=False).to_numpy()
    tdiv = np.select([u.str.match(r".*/(min|minute)$"), u.str.match(r".*/(hour|hr|h)$")], [1.0, 60.0], np.nan)
    return pd.DataFrame({"mass_mcg": mass, "units": units, "per_kg": per_kg, "tdiv": tdiv}, index=uom.index)


def vasoactive_by_stay(inf: pd.DataFrame, win: pd.DataFrame, weight_by_stay: pd.Series, default_w: float,
                       ne_factors: dict) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Maximum standardised rate per drug and maximum concurrent NE-equivalent per stay."""
    diag = {"vaso_records_window": 0, "vaso_weight_default": 0, "vaso_unparsed_units": 0}
    cols = [f"{d}_max" for d in VASO_DRUGS]
    if inf.empty:
        return pd.DataFrame(columns=cols), pd.Series(dtype=float), diag
    p = inf.merge(win[["stay_id", "win_start", "win_end"]], on="stay_id", how="inner")
    overlap = overlaps_window(p["starttime"], p["endtime"], p["win_start"], p["win_end"])
    p = p.loc[overlap & (pd.to_numeric(p["rate"], errors="coerce") > 0)].copy()
    diag["vaso_records_window"] = int(len(p))
    if p.empty:
        return pd.DataFrame(columns=cols), pd.Series(dtype=float), diag
    w = pd.to_numeric(p["patientweight"], errors="coerce")
    w = w.where(w > 0, p["stay_id"].map(weight_by_stay))
    diag["vaso_weight_default"] = int(w.isna().sum())
    w = w.fillna(default_w)
    pu = parse_rate_units(p["rateuom"])
    rate = pd.to_numeric(p["rate"], errors="coerce")
    is_vp = p["med_name"].eq("vasopressin")
    adr = rate * pu["mass_mcg"] / pu["tdiv"] / np.where(pu["per_kg"], 1.0, w)
    vp = rate * pu["units"] / pu["tdiv"] * np.where(pu["per_kg"], w, 1.0)
    p["rate_std"] = np.where(is_vp, vp, adr)
    diag["vaso_unparsed_units"] = int(p["rate_std"].isna().sum())
    diag["vaso_units_seen"] = "; ".join(f"{m}:{u}" for (m, u), _ in
                                        p.groupby(["med_name", p["rateuom"].fillna("NA")]).size().items())
    p = p.dropna(subset=["rate_std"])
    dmax = p.groupby(["stay_id", "med_name"])["rate_std"].max().unstack()
    dmax = dmax.reindex(columns=VASO_DRUGS)
    dmax.columns = cols
    # maximum concurrent NE-equivalent (sum of overlapping infusions at each infusion start)
    p["nee"] = p["rate_std"] * p["med_name"].map(ne_factors).fillna(0.0)
    p["s"] = p[["starttime", "win_start"]].max(axis=1)
    p["e"] = pd.concat([p["endtime"].fillna(p["starttime"]), p["win_end"]], axis=1).min(axis=1)
    q = p.loc[p["nee"] > 0, ["stay_id", "s", "e", "nee"]]
    if len(q):
        cand = q[["stay_id", "s"]].drop_duplicates().rename(columns={"s": "t"})
        cart = cand.merge(q, on="stay_id", how="inner")
        # a zero-length (point) record is "on" only at its own start, not at every later t
        on = (cart["s"] <= cart["t"]) & ((cart["t"] < cart["e"]) | ((cart["s"] == cart["e"]) & (cart["t"] == cart["s"])))
        nee = cart.loc[on].groupby(["stay_id", "t"])["nee"].sum().groupby("stay_id").max()
    else:
        nee = pd.Series(dtype=float)
    return dmax, nee, diag


# ═══════════════════════════════════════════════════════════════════════
# Reporting helpers
# ═══════════════════════════════════════════════════════════════════════
def strata_masks(frame: pd.DataFrame) -> "OrderedDict[str, np.ndarray]":
    hf = frame["hf_subtype"].astype(str).to_numpy()
    sh = frame["rule_phenotype_2class"].astype(str).to_numpy()
    s = OrderedDict()
    s["all"] = np.ones(len(frame), dtype=bool)
    s["hfpef"] = hf == "hfpef"
    s["hfref_or_hfmref"] = np.isin(hf, ["hfref", "hfmref"])
    s["hfref"] = hf == "hfref"
    s["hfmref"] = hf == "hfmref"
    s["hf_ef_unknown"] = hf == "hf_ef_unknown"
    s["no_hf"] = hf == "no_hf"
    s["non_shock"] = sh == "non_shock"
    s["shock_predominant"] = sh == "shock_predominant"
    s["hfpef_nonshock"] = s["hfpef"] & s["non_shock"]
    s["hfpef_shock"] = s["hfpef"] & s["shock_predominant"]
    s["nohf_nonshock"] = s["no_hf"] & s["non_shock"]
    return s


def med_iqr(x: np.ndarray, d: int = 0) -> str:
    x = x[np.isfinite(x)]
    if not x.size:
        return "NA"
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return f"{q2:.{d}f} ({q1:.{d}f}-{q3:.{d}f})"


def summarise_stratum(name: str, sub: pd.DataFrame) -> dict:
    n = len(sub)
    row = {"stratum": name, "n": n}
    x = sub["sofa_total"].to_numpy(float)
    nc = sub["sofa_noncv"].to_numpy(float)
    alt = sub["sofa_total_ett_alt"].to_numpy(float)
    if n:
        q = np.percentile(x, [25, 50, 75])
        row.update(sofa_median=q[1], sofa_q1=q[0], sofa_q3=q[2], sofa_mean=float(x.mean()),
                   sofa_sd=float(x.std(ddof=1)) if n > 1 else np.nan,
                   sofa_ge2_pct=100 * float((x >= 2).mean()), sofa_ge6_pct=100 * float((x >= 6).mean()),
                   sofa_noncv_median=float(np.median(nc)), sofa_noncv_q1=float(np.percentile(nc, 25)),
                   sofa_noncv_q3=float(np.percentile(nc, 75)), sofa_noncv_mean=float(nc.mean()),
                   sofa_alt_ett_median=float(np.median(alt)), sofa_alt_ett_mean=float(alt.mean()),
                   any_component_missing_pct=100 * float((sub["n_components_missing"] > 0).mean()),
                   n_components_missing_mean=float(sub["n_components_missing"].mean()))
        for c in COMPONENTS:
            row[f"{c}_mean"] = float(sub[f"sofa_{c}"].mean())
            row[f"{c}_ge1_pct"] = 100 * float((sub[f"sofa_{c}"] >= 1).mean())
            row[f"{c}_missing_pct"] = 100 * float(sub[f"sofa_{c}_missing"].mean())
        row["vent_invasive_pct"] = 100 * float(sub["vent_invasive_window"].mean())
    return row


def miss_str(row: dict) -> str:
    return ";".join(f"{c}={common.fmt_num(row.get(f'{c}_missing_pct', np.nan), 1)}" for c in COMPONENTS)


def dist_str(x: pd.Series) -> str:
    n = len(x)
    return ";".join(f"{k}:{common.fmt_num(100 * float((x == k).sum()) / n if n else np.nan, 1)}" for k in range(5))


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main(argv=None) -> int:
    t_start = time.time()
    args = parse_args(argv)
    print("=" * 72, flush=True)
    print(f"BM3a — Baseline (ICU admission to T0) SOFA score (v{SCRIPT_VERSION})", flush=True)
    print("=" * 72, flush=True)
    if args.quick:
        log(f"QUICK MODE: <= {common.QUICK_N_STAYS} stays, first {args.max_chunks} chunks of each raw file")

    # ── 1. Cohort and windows ────────────────────────────────────────
    t0 = phase(1, N_PHASES, "Cohort (45c stays), ICU-admission-to-T0 windows, strata")
    root = common.find_root(__file__)
    paths = common.setup_paths(root)
    sfx = output_suffix(args.out)
    log(f"  PROJECT_ROOT={root}")
    if sfx:
        log(f"  non-default --out {args.out}: report / tables / paste tag get the suffix '{sfx}'")
    cfg = common.load_yaml(root / common.CONFIG_REL)
    state_path = root / cfg["data"]["state_tensors"]
    cohort = np.sort(pd.read_parquet(state_path, columns=["stay_id"])["stay_id"].unique().astype("int64"))
    log(f"  45c cohort: {len(cohort):,} stays ({state_path.name})")
    n_cohort_full = len(cohort)
    if args.quick:
        cohort = common.subsample_ids(cohort, common.QUICK_N_STAYS, args.seed)
        log(f"  QUICK: subsample n={len(cohort):,}")
    ts = pd.read_parquet(root / "data/staging/10c_timestamps_mimic.parquet",
                         columns=["subject_id", "hadm_id", "stay_id", "intime", "time_zero_6h"])
    ts["stay_id"] = as_int64(ts["stay_id"])
    win = ts[ts["stay_id"].isin(cohort)].drop_duplicates("stay_id").copy()
    win = win.dropna(subset=["intime", "time_zero_6h"])
    # counted AFTER dropping NaT windows, so such stays are kept (all-missing) instead of silently lost
    n_no_ts = int(len(cohort) - len(win))
    if n_no_ts:
        log(f"  WARNING: {n_no_ts} cohort stays have no usable 10c timestamps (scored as all-missing)")
    win["win_start"] = win["intime"] - hrs(args.lookback_hours)
    win["win_end"] = win["time_zero_6h"]
    win["window_hours"] = (win["win_end"] - win["win_start"]).dt.total_seconds() / 3600.0
    t0_off = (win["time_zero_6h"] - win["intime"]).dt.total_seconds() / 3600.0
    log(f"  windows: {len(win):,} stays; T0 - intime: min {t0_off.min():.2f} h, max {t0_off.max():.2f} h; "
        f"lookback {args.lookback_hours:g} h; window length median {win['window_hours'].median():.2f} h")
    win["subject_id"] = as_int64(win["subject_id"])
    cohort_subjects = set(win["subject_id"].tolist())
    cohort_set = set(cohort.tolist())
    # in-chunk time filter bounds (generous: the exact [win_start, win_end) filter is applied later)
    pad_h = max(args.fio2_pair_hours, args.gcs_carry_hours) + 1.0
    stay_lo = (win.set_index("stay_id")["win_start"] - hrs(pad_h))
    stay_hi = win.set_index("stay_id")["win_end"]
    subj_lo = (win.groupby("subject_id")["win_start"].min() - hrs(pad_h))
    subj_hi = win.groupby("subject_id")["win_end"].max()
    cohort_list = cohort.tolist()
    ph_path = root / cfg["data"]["phenotype_labels"]
    pheno = pd.read_parquet(ph_path, columns=["stay_id", "hf_subtype", "rule_phenotype_2class"])
    pheno["stay_id"] = as_int64(pheno["stay_id"])
    hrs_lo = -float(args.lookback_hours) - max(args.fio2_pair_hours, args.gcs_carry_hours) - 1.0
    hrs_hi = max(6.0, float(t0_off.max()) if len(t0_off) else 6.0) + 1e-6
    mroot = mimic_root(root, args.mimic_root)
    log(f"  MIMIC-IV root: {mroot}  [{time.time() - t0:.1f}s]")
    stg = root / "data" / "staging"
    coverage = []

    # ── 2. Staging sources ───────────────────────────────────────────
    t0 = phase(2, N_PHASES, "Staging sources: 10d vitals, 10e labs, 10f infusions, 10g urine/weight, 10h ventilation")
    hwin = [("hours_since_icu_admit", ">=", hrs_lo), ("hours_since_icu_admit", "<=", hrs_hi)]
    vit = read_staging(stg / "10d_vitals_mimic.parquet", ["stay_id", "charttime", "vital_name", "valuenum"],
                       [("stay_id", "in", cohort_list),
                        ("vital_name", "in", ["map_invasive", "map_noninvasive", "spo2", "fio2"])] + hwin, "10d")
    labs = read_staging(stg / "10e_labs_mimic.parquet", ["stay_id", "charttime", "lab_name", "valuenum"],
                        [("stay_id", "in", cohort_list),
                         ("lab_name", "in", ["platelet", "bilirubin_total", "bilirubin", "creatinine"])] + hwin,
                        "10e")
    inf = read_staging(stg / "10f_meds_infusions_mimic.parquet",
                       ["stay_id", "starttime", "endtime", "med_name", "rate", "rateuom", "patientweight"],
                       [("stay_id", "in", cohort_list), ("med_name", "in", VASO_DRUGS),
                        ("hours_since_icu_admit", "<=", hrs_hi)], "10f")
    uo = read_staging(stg / "10g_output_events_mimic.parquet", ["stay_id", "charttime", "output_type", "value"],
                      [("stay_id", "in", cohort_list), ("output_type", "==", "urine")] + hwin, "10g urine")
    wt = read_staging(stg / "10g_weight_mimic.parquet", ["stay_id", "charttime", "valuenum"],
                      [("stay_id", "in", cohort_list)], "10g weight")
    vent = read_staging(stg / "10h_vent_indicators_mimic.parquet",
                        ["stay_id", "charttime", "indicator_type", "value"],
                        [("stay_id", "in", cohort_list),
                         ("indicator_type", "in", ["mech_vent", "vent_mode", "o2_device"])] + hwin, "10h vent")
    procs = read_staging(stg / "10h_procedures_mimic.parquet",
                         ["stay_id", "starttime", "endtime", "itemid", "procedure_type"],
                         [("stay_id", "in", cohort_list), ("procedure_type", "==", "ventilation")], "10h proc")
    for df_, tc in ((vit, "charttime"), (labs, "charttime"), (uo, "charttime"), (wt, "charttime"),
                    (vent, "charttime"), (inf, "starttime"), (inf, "endtime"), (procs, "starttime"),
                    (procs, "endtime")):
        df_[tc] = pd.to_datetime(df_[tc], errors="coerce")
    wt = wt.sort_values("charttime")
    weight_by_stay = (wt[pd.to_numeric(wt["valuenum"], errors="coerce") > 0]
                      .groupby("stay_id")["valuenum"].first())
    log(f"  [{time.time() - t0:.1f}s]")

    # ── 3. Raw labevents: arterial pO2 ───────────────────────────────
    t0 = phase(3, N_PHASES, "Raw labevents pass: pO2 and specimen type (chunked, itemid + subject filter)")
    d_lab = pd.read_csv(mroot / "hosp" / "d_labitems.csv.gz")
    lab_lbl = d_lab["label"].fillna("").str.strip()
    fluid = d_lab["fluid"].fillna("").str.strip()
    cat = d_lab["category"].fillna("").str.strip()
    is_bg = fluid.str.fullmatch("Blood", case=False) & cat.str.contains("Blood Gas", case=False)
    po2_ids = d_lab.loc[is_bg & lab_lbl.str.fullmatch("pO2", case=False), "itemid"].astype(int).tolist()
    spec_ids = d_lab.loc[is_bg & lab_lbl.str.fullmatch("Specimen Type", case=False), "itemid"].astype(int).tolist()
    if not po2_ids:
        po2_ids = PO2_ITEMID_FALLBACK
        log(f"  WARNING: no pO2 item resolved in d_labitems; using {po2_ids}")
    if not spec_ids:
        spec_ids = SPECIMEN_ITEMID_FALLBACK
        log(f"  WARNING: no 'Specimen Type' item resolved; using {spec_ids}")
    for iid in po2_ids + spec_ids:
        r = d_lab[d_lab["itemid"] == iid]
        lab = f"{r['label'].iloc[0]} / {r['fluid'].iloc[0]} / {r['category'].iloc[0]}" if len(r) else "not in d_labitems"
        log(f"    {iid:>6d} -> {'pO2' if iid in po2_ids else 'specimen_type'} ({lab})")
    lab_targets = set(po2_ids) | set(spec_ids)

    def le_filter(ch):
        ch = ch.loc[ch["itemid"].isin(lab_targets)]
        ch = ch.loc[ch["subject_id"].isin(cohort_subjects)]
        return time_filter(ch, "subject_id", subj_lo, subj_hi)

    le, le_read, le_chunks, le_trunc = chunked_csv(
        mroot / "hosp" / "labevents.csv.gz",
        ["subject_id", "specimen_id", "itemid", "charttime", "value", "valuenum"],
        {"subject_id": "Int64", "specimen_id": "Int64", "itemid": "Int64", "value": str, "valuenum": "float64"},
        args.chunk_size, le_filter, "labevents", args.max_chunks, log_every=5)
    le = le.dropna(subset=["subject_id"]).copy()
    le["subject_id"] = as_int64(le["subject_id"])
    le["itemid"] = pd.to_numeric(le["itemid"], errors="coerce")
    spec = le[le["itemid"].isin(spec_ids)].dropna(subset=["specimen_id"])
    spec_map = (spec.assign(t=spec["value"].astype(str).str.strip().str.upper())
                .drop_duplicates("specimen_id").set_index("specimen_id")["t"])
    po2 = le[le["itemid"].isin(po2_ids)].dropna(subset=["valuenum"]).copy()
    po2["charttime"] = pd.to_datetime(po2["charttime"], errors="coerce")
    po2["spec_type"] = po2["specimen_id"].map(spec_map)
    po2 = po2[po2["valuenum"].between(*PO2_RANGE)]
    po2 = po2.merge(win[["subject_id", "stay_id"]], on="subject_id", how="inner")
    po2 = in_window(po2, win, "charttime")
    spec_counts = po2["spec_type"].fillna("UNKNOWN").value_counts()
    log(f"  pO2 in window: {len(po2):,} rows, {po2['stay_id'].nunique():,} stays; specimen types: "
        + ", ".join(f"{k}={v:,}" for k, v in spec_counts.items()))
    po2_mode = args.po2_specimen
    if po2_mode != "any" and spec.empty:
        log("  WARNING: no specimen-type rows found; falling back to --po2-specimen any")
        po2_mode = "any"
    if po2_mode == "art":
        po2 = po2[po2["spec_type"] == "ART."]
    elif po2_mode == "art_or_unknown":
        po2 = po2[po2["spec_type"].isna() | (po2["spec_type"] == "ART.")]
    log(f"  pO2 used ({po2_mode}): {len(po2):,} rows, {po2['stay_id'].nunique():,} stays  "
        f"[{time.time() - t0:.1f}s]")
    coverage.append({"source": "labevents_pO2_window_used", "rows": len(po2), "stays": po2["stay_id"].nunique()})

    # ── 4. Raw chartevents: GCS and FiO2 ─────────────────────────────
    t0 = phase(4, N_PHASES, "Raw chartevents pass: GCS components and FiO2 223835 (chunked)")
    d_items = pd.read_csv(mroot / "icu" / "d_items.csv.gz", usecols=["itemid", "label"])
    ce_targets = set(GCS_ITEMIDS.values()) | {FIO2_ITEMID}
    for iid in sorted(ce_targets):
        r = d_items[d_items["itemid"] == iid]
        log(f"    {iid:>6d} -> {r['label'].iloc[0] if len(r) else 'NOT IN d_items'}")

    def ce_filter(ch):
        ch = ch.loc[ch["itemid"].isin(ce_targets)]
        ch = ch.loc[ch["stay_id"].isin(cohort_set)]
        return time_filter(ch, "stay_id", stay_lo, stay_hi)

    ce, ce_read, ce_chunks, ce_trunc = chunked_csv(
        mroot / "icu" / "chartevents.csv.gz", ["stay_id", "charttime", "itemid", "value", "valuenum"],
        {"stay_id": "Int64", "itemid": "Int64", "value": str, "valuenum": "float64"},
        args.chunk_size, ce_filter, "chartevents", args.max_chunks, log_every=10)
    ce = ce.dropna(subset=["stay_id"]).copy()
    ce["stay_id"] = as_int64(ce["stay_id"])
    ce["itemid"] = pd.to_numeric(ce["itemid"], errors="coerce").astype("int64")
    ce["charttime"] = pd.to_datetime(ce["charttime"], errors="coerce")
    ce_gcs = ce[ce["itemid"].isin(list(GCS_ITEMIDS.values()))]
    ce_fio2 = ce[ce["itemid"] == FIO2_ITEMID]
    log(f"  GCS rows {len(ce_gcs):,}; FiO2 rows {len(ce_fio2):,}  [{time.time() - t0:.1f}s]")

    # ── 5. Component values ──────────────────────────────────────────
    t0 = phase(5, N_PHASES, "Component values in the window and SOFA points")
    ids = win["stay_id"].to_numpy()
    inv_ids, niv_ids, vdiag = ventilation_flags(vent, procs, win)
    support_ids = inv_ids | niv_ids if args.niv_as_vent else inv_ids
    # room-air FiO2 (0.21) is imputed only for stays with neither invasive ventilation nor NIV in the
    # window, whatever --niv-as-vent says (that flag only decides which stays may score resp 3-4)
    no_impute_ids = inv_ids | niv_ids
    log(f"  invasive ventilation in window: {len(inv_ids):,} stays (chartevents {vdiag['inv_from_chartevents']:,}, "
        f"procedures {vdiag['inv_from_procedures']:,}); NIV {len(niv_ids):,}; support used for resp 3-4: "
        f"{'invasive + NIV' if args.niv_as_vent else 'invasive only'}")
    for k in ("modes_invasive", "modes_not_invasive", "devices_window", "proc_itemids_window"):
        log(f"    {k} (rows in window): {vdiag.get(k) or '-'}")

    # FiO2: 10d (21-100 %, range-filtered) + raw 223835 (recovers fraction-coded values)
    f10d = vit[vit["vital_name"] == "fio2"][["stay_id", "charttime", "valuenum"]]
    fce = ce_fio2[["stay_id", "charttime", "valuenum"]]
    fio2 = pd.concat([fce.assign(src="chartevents"), f10d.assign(src="10d")], ignore_index=True)
    fio2["fio2"] = normalise_fio2(fio2["valuenum"])
    n_frac_coded = int((pd.to_numeric(fce["valuenum"], errors="coerce") <= 1.0).sum())
    fio2 = fio2.dropna(subset=["fio2", "charttime"]).drop_duplicates(["stay_id", "charttime"], keep="first")
    fio2 = in_window(fio2, win, "charttime", pad_before_h=args.fio2_pair_hours)
    log(f"  FiO2 rows usable (window - {args.fio2_pair_hours:g} h): {len(fio2):,} "
        f"(raw 223835 coded as fraction <= 1: {n_frac_coded:,})")

    # Respiration
    po2m = po2[["stay_id", "charttime", "valuenum"]].rename(columns={"valuenum": "po2"})
    pf = pair_with_fio2(po2m, fio2, no_impute_ids, args.fio2_pair_hours)
    n_po2_unpaired_vent = int(pf["fio2"].isna().sum())
    pf = pf.dropna(subset=["fio2"])
    pf["ratio"] = pf["po2"] / pf["fio2"]
    pf_min = pf.groupby("stay_id")["ratio"].min()
    spo2_all = in_window(vit[vit["vital_name"] == "spo2"][["stay_id", "charttime", "valuenum"]], win, "charttime")
    spo2 = spo2_all[spo2_all["valuenum"] <= SPO2_SF_MAX].rename(columns={"valuenum": "spo2"})
    sf = pair_with_fio2(spo2, fio2, no_impute_ids, args.fio2_pair_hours).dropna(subset=["fio2"])
    sf["ratio"] = sf["spo2"] / sf["fio2"]
    sf_min = sf.groupby("stay_id")["ratio"].min()
    support = np.isin(ids, list(support_ids))
    pf_v = per_stay(pf_min, ids)
    sf_v = per_stay(sf_min, ids)
    s_pf = score_ratio(pf_v, PF_THRESHOLDS, support)
    s_sf = score_ratio(sf_v, SF_THRESHOLDS, support)
    s_resp = np.where(np.isfinite(s_pf), s_pf, s_sf)
    has_spo2 = np.isin(ids, spo2_all["stay_id"].unique())
    resp_source = np.where(np.isfinite(pf_v), "pf", np.where(np.isfinite(sf_v), "sf",
                                                         np.where(has_spo2, "none_spo2_gt97", "none")))
    log(f"  respiration source: " + ", ".join(f"{k}={v:,}" for k, v in
                                               pd.Series(resp_source).value_counts().items())
        + f"; PaO2 dropped (invasive ventilation / NIV, no FiO2 <= {args.fio2_pair_hours:g} h): {n_po2_unpaired_vent:,}; "
          f"FiO2 imputed 0.21: P/F {int(pf['fio2_imputed'].sum()):,}, S/F {int(sf['fio2_imputed'].sum()):,}")

    # Coagulation / liver / renal labs
    lw = in_window(labs, win, "charttime")
    lname = lw["lab_name"].replace({"bilirubin": "bilirubin_total"})
    plt_min = lw[lname == "platelet"].groupby("stay_id")["valuenum"].min()
    bili_max = lw[lname == "bilirubin_total"].groupby("stay_id")["valuenum"].max()
    creat_max = lw[lname == "creatinine"].groupby("stay_id")["valuenum"].max()
    plt_v, bili_v, cr_v = per_stay(plt_min, ids), per_stay(bili_max, ids), per_stay(creat_max, ids)
    s_coag, s_liver, s_cr = score_platelets(plt_v), score_bilirubin(bili_v), score_creatinine(cr_v)

    # Urine output (mL in window, scaled to 24 h). outputevents are ICU-only, so the scaling uses the
    # ICU part of the window (intime -> T0); dividing by a window that includes --lookback-hours
    # before intime would understate UO and inflate the renal score.
    uw = in_window(uo, win, "charttime")
    uw["value"] = pd.to_numeric(uw["value"], errors="coerce")
    uo_sum = uw.dropna(subset=["value"]).groupby("stay_id")["value"].sum()
    uo_v = per_stay(uo_sum, ids)
    whrs = win["window_hours"].to_numpy(float)
    uo_hrs = ((win["win_end"] - win[["win_start", "intime"]].max(axis=1)).dt.total_seconds() / 3600.0).to_numpy(float)
    uo24 = np.where(uo_hrs > 0, uo_v * 24.0 / np.where(uo_hrs > 0, uo_hrs, 1.0), np.nan)
    s_uo = score_urine(uo24)
    s_renal = np.where(np.isfinite(s_cr) | np.isfinite(s_uo), np.fmax(s_cr, s_uo), np.nan)

    # Cardiovascular
    mw = in_window(vit[vit["vital_name"].isin(["map_invasive", "map_noninvasive"])], win, "charttime")
    map_v = per_stay(mw.groupby("stay_id")["valuenum"].min(), ids)
    ne_factors = dict(NE_FACTORS_DEFAULT)
    asp = root / "configs" / "action_space_v1.yaml"
    if asp.is_file():
        ne_factors.update({k: float(v) for k, v in (common.load_yaml(asp).get("ne_equivalent_factors") or {}).items()})
    dmax, nee, cdiag = vasoactive_by_stay(inf, win, weight_by_stay, args.default_weight_kg, ne_factors)
    dmax_v = {d: per_stay(dmax[f"{d}_max"], ids) if f"{d}_max" in dmax else np.full(len(ids), np.nan)
              for d in VASO_DRUGS}
    nee_v = per_stay(nee, ids) if len(nee) else np.full(len(ids), np.nan)
    s_cardio = score_cardio(map_v, dmax_v)
    log(f"  vasoactive records overlapping the window: {cdiag['vaso_records_window']:,}; weight defaulted "
        f"({args.default_weight_kg:g} kg): {cdiag['vaso_weight_default']:,}; unparsed rate units: "
        f"{cdiag['vaso_unparsed_units']:,}")
    if cdiag.get("vaso_units_seen"):
        log(f"  units seen: {cdiag['vaso_units_seen']}")

    # CNS
    gcs, gdiag = gcs_by_stay(ce_gcs, win, args.gcs_carry_hours)
    g_norm = per_stay(gcs["gcs_min_ett_normal"], ids) if len(gcs) else np.full(len(ids), np.nan)
    g_v1 = per_stay(gcs["gcs_min_ett_verbal1"], ids) if len(gcs) else np.full(len(ids), np.nan)
    g_prim, g_alt = (g_norm, g_v1) if args.gcs_ett == "normal" else (g_v1, g_norm)
    s_cns, s_cns_alt = score_gcs(g_prim), score_gcs(g_alt)
    log(f"  GCS assessments in window: {gdiag.get('gcs_assessments_window', 0):,} (complete "
        f"{gdiag.get('gcs_assessments_complete', 0):,}, ETT verbal {gdiag.get('gcs_assessments_ett', 0):,})")

    raw_scores = {"resp": s_resp, "coag": s_coag, "liver": s_liver, "cardio": s_cardio, "cns": s_cns,
                  "renal": s_renal}
    out = pd.DataFrame({"stay_id": ids})
    for c in COMPONENTS:
        out[f"sofa_{c}"] = np.nan_to_num(raw_scores[c], nan=0.0).astype(int)
        out[f"sofa_{c}_missing"] = ~np.isfinite(raw_scores[c])
    out["sofa_total"] = out[[f"sofa_{c}" for c in COMPONENTS]].sum(axis=1).astype(int)
    out["n_components_missing"] = out[[f"sofa_{c}_missing" for c in COMPONENTS]].sum(axis=1).astype(int)
    out["sofa_noncv"] = (out["sofa_total"] - out["sofa_cardio"]).astype(int)
    alt_name = "verbal1" if args.gcs_ett == "normal" else "normal"
    out[f"sofa_cns_ett_{alt_name}"] = np.nan_to_num(s_cns_alt, nan=0.0).astype(int)
    out[f"sofa_total_ett_{alt_name}"] = (out["sofa_total"] - out["sofa_cns"] + out[f"sofa_cns_ett_{alt_name}"]).astype(int)
    out["pf_ratio"] = pf_v
    out["sf_ratio"] = sf_v
    out["resp_source"] = resp_source
    out["vent_invasive_window"] = np.isin(ids, list(inv_ids))
    out["niv_window"] = np.isin(ids, list(niv_ids))
    out["gcs_min"] = g_prim
    out[f"gcs_min_ett_{alt_name}"] = g_alt
    out["map_min"] = map_v
    out["ne_eq_max"] = np.where(np.isfinite(nee_v), nee_v, 0.0)
    for d in VASO_DRUGS:
        out[f"{d}_max"] = dmax_v[d]
    out["platelet_min"] = plt_v
    out["bilirubin_max"] = bili_v
    out["creatinine_max"] = cr_v
    out["urine_ml_window"] = uo_v
    out["urine_ml_24h_scaled"] = uo24
    out["sofa_renal_creat"] = s_cr
    out["sofa_renal_uo"] = s_uo
    out["window_hours"] = whrs
    if n_no_ts:   # stays without timestamps: all components missing
        miss = pd.DataFrame({"stay_id": np.setdiff1d(cohort, ids)})
        for c in COMPONENTS:
            miss[f"sofa_{c}"] = 0
            miss[f"sofa_{c}_missing"] = True
        miss["sofa_total"], miss["n_components_missing"], miss["sofa_noncv"] = 0, 6, 0
        # keep the alternative-ETT totals, flags and source defined (BM3b / BM2 read them)
        miss[f"sofa_cns_ett_{alt_name}"], miss[f"sofa_total_ett_{alt_name}"] = 0, 0
        miss["resp_source"], miss["vent_invasive_window"], miss["niv_window"] = "none", False, False
        miss["ne_eq_max"] = 0.0
        out = pd.concat([out, miss], ignore_index=True)
    out = out.sort_values("stay_id").reset_index(drop=True)
    for c, src, n_rows in (("platelet", "10e platelet", int((lname == "platelet").sum())),
                           ("bilirubin", "10e bilirubin_total", int((lname == "bilirubin_total").sum())),
                           ("creatinine", "10e creatinine", int((lname == "creatinine").sum()))):
        coverage.append({"source": src, "rows": n_rows, "stays": int(np.isfinite(out[f"{c}_{'min' if c == 'platelet' else 'max'}"]).sum())})
    coverage += [
        {"source": "10d MAP (invasive+cuff)", "rows": len(mw), "stays": int(np.isfinite(out["map_min"]).sum())},
        {"source": "10d SpO2", "rows": len(spo2_all), "stays": int(spo2_all["stay_id"].nunique())},
        {"source": "FiO2 (10d + chartevents 223835, incl. 4 h pre-window)", "rows": len(fio2),
         "stays": int(fio2["stay_id"].nunique())},
        {"source": "P/F pairs", "rows": len(pf), "stays": int(np.isfinite(out["pf_ratio"]).sum())},
        {"source": "S/F pairs (SpO2<=97)", "rows": len(sf), "stays": int(np.isfinite(out["sf_ratio"]).sum())},
        {"source": "10f vasoactive infusions", "rows": cdiag["vaso_records_window"],
         "stays": int((out[[f"{d}_max" for d in VASO_DRUGS]].fillna(0) > 0).any(axis=1).sum())},
        {"source": "10g urine", "rows": len(uw), "stays": int(np.isfinite(out["urine_ml_window"]).sum())},
        {"source": "chartevents GCS (complete assessments)", "rows": gdiag.get("gcs_assessments_complete", 0),
         "stays": int(np.isfinite(out["gcs_min"]).sum())},
        {"source": "10h invasive ventilation", "rows": vdiag["vent_rows_window"], "stays": len(inv_ids)},
    ]
    log(f"  SOFA total median {np.median(out['sofa_total']):.0f}, mean {out['sofa_total'].mean():.2f}  "
        f"[{time.time() - t0:.1f}s]")

    # ── 6. Stay-level output ─────────────────────────────────────────
    t0 = phase(6, N_PHASES, "Stay-level output (data/interim/bmc_medicine only)")
    out_path = common.save_interim(out, root, args.out)
    meta = {"script": "BM3a_extract_sofa_preT0.py", "version": SCRIPT_VERSION, "created": common.now(),
            "quick": bool(args.quick), "max_chunks": int(args.max_chunks), "n_stays": int(len(out)),
            "n_cohort_full": int(n_cohort_full), "lookback_hours": float(args.lookback_hours),
            "po2_specimen": po2_mode, "gcs_ett_primary": args.gcs_ett, "niv_as_vent": bool(args.niv_as_vent),
            "fio2_pair_hours": args.fio2_pair_hours, "gcs_carry_hours": args.gcs_carry_hours,
            "ne_factors": ne_factors, "columns": list(out.columns),
            "raw_truncated": {"labevents": bool(le_trunc), "chartevents": bool(ce_trunc)}}
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    if args.quick or le_trunc or ce_trunc:
        log("  WARNING: QUICK / truncated run -> the SOFA file is NOT complete; rerun BM3a without --quick "
            "before BM3b / BM2 --sofa on real data")
    log(f"  [{time.time() - t0:.1f}s]")

    # ── 7. Aggregate tables and report ───────────────────────────────
    t0 = phase(7, N_PHASES, "Aggregate tables and report")
    alt_total = f"sofa_total_ett_{alt_name}"
    fr = out.merge(pheno, on="stay_id", how="left")
    n_no_ph = int(fr["hf_subtype"].isna().sum())
    if n_no_ph:
        log(f"  WARNING: {n_no_ph} stays without phenotype labels (counted in 'all' only)")
    fr["sofa_total_ett_alt"] = fr[alt_total]
    strata = strata_masks(fr)
    rows = [summarise_stratum(s, fr.loc[m]) for s, m in strata.items()]
    by_stratum = pd.DataFrame(rows)
    common.save_table(by_stratum, paths["tables"] / f"BM3a_sofa_by_stratum{sfx}.csv")
    drows = []
    for s in ("all", "hfpef", "hfref_or_hfmref", "hf_ef_unknown", "no_hf", "non_shock", "shock_predominant"):
        sub = fr.loc[strata[s]]
        for c in COMPONENTS + ["total"]:
            col = f"sofa_{c}"
            vals = range(5) if c != "total" else range(int(fr["sofa_total"].max()) + 1)
            for k in vals:
                drows.append({"stratum": s, "component": c, "score": k, "n": int((sub[col] == k).sum()),
                              "pct": 100 * float((sub[col] == k).mean()) if len(sub) else np.nan})
    common.save_table(pd.DataFrame(drows), paths["tables"] / f"BM3a_component_distribution{sfx}.csv")
    cov = pd.DataFrame(coverage)
    cov["stays_pct"] = 100.0 * cov["stays"] / max(len(out), 1)
    common.save_table(cov, paths["tables"] / f"BM3a_source_coverage{sfx}.csv")

    R = {r["stratum"]: r for r in rows}
    runtime_min = (time.time() - t_start) / 60.0
    rs_counts = pd.Series(resp_source).value_counts()
    L = common.report_header("BM3a — Baseline (ICU admission to T0) SOFA score (BMC Medicine revision)", root, args)
    L += [f"Script version {SCRIPT_VERSION}; runtime {runtime_min:.1f} min"
          + ("  [QUICK SMOKE TEST — subsample / truncated raw passes, not for interpretation]"
             if (args.quick or le_trunc or ce_trunc) else ""),
          f"Cohort: {len(out):,} stays scored (45c cohort {n_cohort_full:,}); window = ICU admission"
          f"{f' - {args.lookback_hours:g} h' if args.lookback_hours else ''} to T0 (intime + 6 h), "
          f"[start, T0); median window {np.median(whrs):.2f} h",
          f"Raw passes: labevents {le_read:,} rows / {le_chunks} chunks; chartevents {ce_read:,} rows / "
          f"{ce_chunks} chunks (chunk size {args.chunk_size:,})",
          "",
          "Conventions:",
          f"  * pO2 specimen filter: {po2_mode} (labevents 52033 Specimen Type); pO2 in window by specimen: "
          + ", ".join(f"{k}={v:,}" for k, v in spec_counts.items()),
          f"  * FiO2 = most recent within {args.fio2_pair_hours:g} h before the PaO2/SpO2 (10d + raw 223835, "
          f"% or fraction -> fraction, kept 0.21-1.0); 0.21 when none and neither invasive ventilation nor NIV; "
          f"PaO2 of ventilated / NIV stays without FiO2 dropped ({n_po2_unpaired_vent:,} rows)",
          f"  * S/F only when no PaO2: SpO2 <= {SPO2_SF_MAX:g} %, thresholds 512/357/214/89 (P/F 400/300/200/100)",
          f"  * respiratory support for points 3-4: {'invasive ventilation or NIV' if args.niv_as_vent else 'invasive ventilation'} "
          "in the window (10h: ETT/tracheostomy tube, ventilator mode other than standby/NIV modes, "
          "Invasive Ventilation / Intubation procedures)",
          "  * cardiovascular: vasopressin or phenylephrine infusion alone scores 3; rates per minute divided by "
          f"weight (10f patientweight, else first 10g weight, else {args.default_weight_kg:g} kg); "
          "no minimum infusion duration",
          f"  * GCS: components summed per charttime, carried forward <= {args.gcs_carry_hours:g} h; "
          f"'No Response-ETT' primary convention = {args.gcs_ett} "
          f"({'GCS 15, MIMIC-code' if args.gcs_ett == 'normal' else 'verbal 1 as charted'}); alternative stored",
          "  * renal: worse of creatinine and urine output (ICU-part window total scaled to 24 h; stays without "
          "urine charting = UO missing, not anuric)",
          "  * a component without data scores 0 and is flagged missing",
          "",
          "Respiration source (stays): " + ", ".join(f"{k}={v:,}" for k, v in rs_counts.items()),
          f"Invasive ventilation in window: {len(inv_ids):,} stays ({100 * len(inv_ids) / max(len(ids), 1):.1f}%); "
          f"NIV {len(niv_ids):,}",
          f"  ventilator-mode values counted invasive (rows): {vdiag.get('modes_invasive') or '-'}",
          f"  ventilator-mode values NOT counted invasive (rows): {vdiag.get('modes_not_invasive') or '-'}",
          f"  O2 delivery devices in window (rows): {vdiag.get('devices_window') or '-'}",
          f"  10h ventilation procedure itemids overlapping the window (rows): {vdiag.get('proc_itemids_window') or '-'}",
          "",
          "SOFA total by stratum: n, median (IQR), mean, % >= 2, % >= 6, non-CV median (IQR), "
          f"alternative-ETT ({alt_name}) mean:"]
    for s, r in R.items():
        if not r["n"]:
            L.append(f"  {s:18s} n=0")
            continue
        L.append(f"  {s:18s} n={r['n']:>6,}  {common.fmt_num(r['sofa_median'], 0)} "
                 f"({common.fmt_num(r['sofa_q1'], 0)}-{common.fmt_num(r['sofa_q3'], 0)})  "
                 f"mean {common.fmt_num(r['sofa_mean'])}  >=2 {common.fmt_num(r['sofa_ge2_pct'], 1)}%  "
                 f">=6 {common.fmt_num(r['sofa_ge6_pct'], 1)}%  nonCV {common.fmt_num(r['sofa_noncv_median'], 0)} "
                 f"({common.fmt_num(r['sofa_noncv_q1'], 0)}-{common.fmt_num(r['sofa_noncv_q3'], 0)})  "
                 f"alt mean {common.fmt_num(r['sofa_alt_ett_mean'])}")
    L += ["", "Component mean points / % >= 1 point, by stratum:"]
    for s in ("all", "hfpef", "hfref_or_hfmref", "hf_ef_unknown", "no_hf", "non_shock", "shock_predominant"):
        r = R[s]
        if r["n"]:
            L.append(f"  {s:18s} " + "  ".join(f"{c} {common.fmt_num(r[f'{c}_mean'])}/{common.fmt_num(r[f'{c}_ge1_pct'], 0)}%"
                                                for c in COMPONENTS))
    L += ["", "Component missingness (% of stays scored 0 by convention), by stratum:"]
    for s, r in R.items():
        if r["n"]:
            L.append(f"  {s:18s} {miss_str(r)}  any={common.fmt_num(r['any_component_missing_pct'], 1)}")
    L += ["", "Component score distribution, all stays (% with 0/1/2/3/4 points):"]
    for c in COMPONENTS:
        L.append(f"  {COMPONENT_LABEL[c]:15s} {dist_str(out[f'sofa_{c}'])}")
    L += ["", "Source coverage (rows in window / stays with data):"]
    for _, r in cov.iterrows():
        L.append(f"  {r['source']:55s} {int(r['rows']):>10,}  {int(r['stays']):>7,} ({r['stays_pct']:.1f}%)")
    L += ["", "Stay-level SOFA: " + str(out_path.relative_to(root)) + " (never sync); used by BM3b and "
          "BM2 --sofa."]

    P = OrderedDict()
    P["script"] = "BM3a_extract_sofa_preT0"
    P["version"] = SCRIPT_VERSION
    P["quick"] = bool(args.quick)
    P["raw_truncated"] = bool(le_trunc or ce_trunc)
    P["n_cohort_45c"] = int(n_cohort_full)
    P["n_scored"] = int(len(out))
    P["n_without_timestamps"] = n_no_ts
    P["lookback_hours"] = float(args.lookback_hours)
    P["window_hours_median"] = float(np.median(whrs))
    P["po2_specimen_filter"] = po2_mode
    P["gcs_ett_primary"] = args.gcs_ett
    P["niv_as_vent"] = bool(args.niv_as_vent)
    P["labevents_rows_read"] = int(le_read)
    P["chartevents_rows_read"] = int(ce_read)
    P["po2_rows_window_by_specimen"] = ";".join(f"{k}={v}" for k, v in spec_counts.items())
    P["po2_rows_used"] = int(len(po2))
    P["fio2_rows_raw_fraction_coded"] = n_frac_coded
    P["pao2_dropped_vent_no_fio2"] = n_po2_unpaired_vent
    P["resp_source_counts"] = ";".join(f"{k}={v}" for k, v in rs_counts.items())
    P["vent_invasive_window_pct"] = 100 * len(inv_ids) / max(len(ids), 1)
    P["niv_window_pct"] = 100 * len(niv_ids) / max(len(ids), 1)
    P["vaso_weight_defaulted_records"] = cdiag["vaso_weight_default"]
    P["vaso_unparsed_unit_records"] = cdiag["vaso_unparsed_units"]
    P["gcs_assessments_complete"] = gdiag.get("gcs_assessments_complete", 0)
    P["gcs_assessments_ett"] = gdiag.get("gcs_assessments_ett", 0)
    for _, r in cov.iterrows():
        key = (r["source"].split(" (")[0].lower().replace(" ", "_").replace("/", "").replace("+", "_")
               .replace("<=", "le").replace(",", ""))
        P[f"coverage_{key}_stays_pct"] = float(r["stays_pct"])
    for s, r in R.items():
        P[f"{s}_n"] = int(r["n"])
        if not r["n"]:
            continue
        P[f"{s}_sofa_median_iqr"] = (f"{common.fmt_num(r['sofa_median'], 0)} ({common.fmt_num(r['sofa_q1'], 0)}-"
                                     f"{common.fmt_num(r['sofa_q3'], 0)})")
        P[f"{s}_sofa_mean_sd"] = f"{common.fmt_num(r['sofa_mean'])} ({common.fmt_num(r['sofa_sd'])})"
        P[f"{s}_sofa_ge2_pct"] = float(r["sofa_ge2_pct"])
        P[f"{s}_sofa_ge6_pct"] = float(r["sofa_ge6_pct"])
        P[f"{s}_sofa_noncv_median_iqr"] = (f"{common.fmt_num(r['sofa_noncv_median'], 0)} "
                                           f"({common.fmt_num(r['sofa_noncv_q1'], 0)}-{common.fmt_num(r['sofa_noncv_q3'], 0)})")
        P[f"{s}_sofa_alt_ett_{alt_name}_mean"] = float(r["sofa_alt_ett_mean"])
        P[f"{s}_component_mean"] = ";".join(f"{c}={common.fmt_num(r[f'{c}_mean'])}" for c in COMPONENTS)
        P[f"{s}_missing_pct"] = miss_str(r)
        P[f"{s}_any_missing_pct"] = float(r["any_component_missing_pct"])
    for c in COMPONENTS:
        P[f"dist_all_{c}_pct_0to4"] = dist_str(out[f"sofa_{c}"])
    P["runtime_min"] = runtime_min
    common.write_report(paths["reports"] / f"BM3a_summary{sfx}.txt", L, P, TAG + sfx)
    log(f"  [{time.time() - t0:.1f}s]")
    log(f"Done in {runtime_min:.1f} min.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
