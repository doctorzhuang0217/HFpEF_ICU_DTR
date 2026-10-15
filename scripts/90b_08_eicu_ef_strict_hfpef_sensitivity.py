#!/usr/bin/env python3
"""90b_08 — eICU EF-strict HFpEF Transportability Sensitivity (Analysis A)

Protocol: v3.1, Step 90b (sub-step 6); enhancement spec §A.
Compute: G1 (GPU for frozen-model inference) + C3 (bootstrap)

Purpose
-------
The eICU external-validation HFpEF subgroup used in 90b_05 is built from a
BROADER-COMPATIBLE ICD ascertainment (HF dx + NOT known systolic HF). A key
concern is EF-misclassification: roughly a third of these
ICD-HFpEF stays carry HFrEF-range EF when EF is actually documented. This script
answers that directly by re-running the protocol-PRIMARY external estimator on
the EF-STRICT HFpEF subset:

    hfpef_ef_ge50 = (ICD-HFpEF, i.e. hf_subtype == "hfpef") AND (ejectfx >= 50)

eICU EF is read from raw apachePredVar.ejectfx, where the overwhelming majority
of stays carry the -1 "not assessed" sentinel. Only stays with a genuine
documented EF (ejectfx > 0) are EF-evaluable, so the EF-strict HFpEF subset is
expected to be VERY small (~50-60 stays). This is therefore a DESCRIPTIVE-ONLY
sensitivity: we report how few qualify and a wide-CI point estimate. The
sub-30 subgroup-skip floor used elsewhere is intentionally REMOVED here; if the
subset is below the floor we still report N + the point estimate (CI reported if
the bootstrap succeeds).

The PRIMARY external estimator is the non-rollout (teacher-forced) Direct Method
(DM), identical to 90b_05's external-validation hierarchy: real eICU states with
only the action substituted (rollout's MIMIC-trained next_state_head does not
transport across databases). Frozen-model loading mirrors 90b_05 exactly.

Inputs
------
- data/raw/eicu/2.0/apachePredVar.csv.gz          (column `ejectfx`; -1 sentinel)
- data/interim/90b_eicu_state_tensors_v1.parquet  (stay_id == patientunitstayid)
- data/interim/90b_eicu_outcomes_absorb.parquet
- data/interim/90b_eicu_propensity_coarse3.parquet
- data/staging/90b_eicu_cohort_v1.parquet         (hf_subtype, patientunitstayid)
- models/80c_gtransformer_<tag>_best.pt           (frozen MIMIC G-Transformer)
- configs/80_step_c_training_v1.yaml
- configs/90a_primary_model_registry.json
- configs/70b_action_merge_map.yaml

Outputs
-------
- tables/90b_08_eicu_ef_strict_hfpef_sensitivity.csv
    cols: subset, n_stays, n_ef_documented, v_observed, v_rule,
          ate, ate_ci_lo, ate_ci_hi, database
- reports/90b_08_eicu_ef_strict_hfpef_sensitivity.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(ROOT / "src"))

from hfpef_dtr.models.dtr_dataset import DTRDataset
from hfpef_dtr.models.gtransformer import build_gtransformer_from_config
from hfpef_dtr.models.gnet import build_gnet_from_config
from hfpef_dtr.models.actin import build_actin_from_config
from hfpef_dtr.models.ope import evaluate_dm, bootstrap_ope

MODELS_DIR = ROOT / "models"
TABLES_DIR = ROOT / "tables"
REPORTS_DIR = ROOT / "reports"
INTERIM_DIR = ROOT / "data" / "interim"
STAGING_DIR = ROOT / "data" / "staging"
EICU_ROOT = ROOT / "data" / "raw" / "eicu" / "2.0"

for d in [TABLES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(
        description="90b_08: eICU EF-strict HFpEF transportability sensitivity"
    )
    p.add_argument("--model", type=str, default="80c_gtransformer",
                   help="Frozen model ID (default: 80c_gtransformer)")
    p.add_argument("--tag", type=str, default="rollout_ow1",
                   help="Checkpoint tag (e.g. 'rollout_ow1' for best 80c variant)")
    p.add_argument("--ef-threshold", type=float, default=50.0,
                   help="EF threshold for strict HFpEF (default: 50%%)")
    p.add_argument("--bootstrap", type=int, default=500,
                   help="Number of bootstrap resamples (default: 500)")
    p.add_argument("--skip-bootstrap", action="store_true",
                   help="Skip bootstrap CIs (faster for debugging)")
    p.add_argument("--min-n", type=int, default=0,
                   help="Subgroup-skip floor; DESCRIPTIVE subset uses 0 "
                        "(report N-only if below). Default: 0 (no floor).")
    return p.parse_args()


def load_mimic_feature_spec() -> dict:
    """Load MIMIC feature specification from the frozen model registry."""
    registry_path = ROOT / "configs" / "90a_primary_model_registry.json"
    with open(registry_path) as f:
        registry = json.load(f)
    features = registry["features"]
    training = registry["training_config"]
    return {
        "temporal_cols": features["temporal_cols"],   # 118 columns
        "static_cols": features["static_cols"],        # 3 columns
        "n_temporal": features["n_temporal"],          # 118
        "n_static": features["n_static"],              # 3
        "n_actions": training["action"]["n_merged_cells"],  # 13
    }


def load_eicu_ef() -> pd.DataFrame:
    """
    Load documented EF from raw apachePredVar (one APACHE record per ICU stay).

    eICU codes 'not assessed' EF as -1. We keep ONLY genuinely documented EF
    (ejectfx > 0) and drop the -1 sentinel (and any other non-positive value).
    If multiple records exist per stay (rare), keep the maximum documented EF.

    Returns a DataFrame with columns: patientunitstayid, ejectfx.
    """
    ef_path = EICU_ROOT / "apachePredVar.csv.gz"
    if not ef_path.exists():
        raise FileNotFoundError(f"eICU EF source not found: {ef_path}")

    print(f"  Loading eICU EF from {ef_path.name} ...")
    apd = pd.read_csv(ef_path, usecols=["patientunitstayid", "ejectfx"])
    n_raw = len(apd)
    apd["ejectfx"] = pd.to_numeric(apd["ejectfx"], errors="coerce")

    n_sentinel = int((apd["ejectfx"] <= 0).sum() + apd["ejectfx"].isna().sum())
    ef = apd[apd["ejectfx"] > 0].copy()  # drop the -1 sentinel + non-positive
    # Collapse to one EF per stay (keep max documented EF)
    ef = (
        ef.groupby("patientunitstayid", as_index=False)["ejectfx"].max()
    )
    print(f"    apachePredVar rows: {n_raw:,}")
    print(f"    -1 / non-positive / NaN sentinel (dropped): {n_sentinel:,}")
    print(f"    Stays with documented EF (ejectfx > 0): {len(ef):,}")
    if len(ef) > 0:
        print(f"    Documented EF: mean={ef['ejectfx'].mean():.1f}, "
              f"median={ef['ejectfx'].median():.1f}, "
              f"min={ef['ejectfx'].min():.0f}, max={ef['ejectfx'].max():.0f}")
    return ef


def load_merge_map() -> dict[int, int] | None:
    merge_path = ROOT / "configs" / "70b_action_merge_map.yaml"
    if not merge_path.exists():
        return None
    with open(merge_path) as f:
        mm = yaml.safe_load(f)
    if "merge_map" in mm:
        return {int(k): int(v) for k, v in mm["merge_map"].items()}
    return None


def align_eicu_to_mimic_features(
    state_df: pd.DataFrame,
    mimic_spec: dict,
) -> pd.DataFrame:
    """Ensure eICU state tensor has exactly the MIMIC feature columns."""
    all_mimic_cols = mimic_spec["temporal_cols"] + mimic_spec["static_cols"]
    missing_cols = [c for c in all_mimic_cols if c not in state_df.columns]
    if missing_cols:
        print(f"    WARNING: {len(missing_cols)} MIMIC columns missing in eICU, "
              f"filling with 0: {missing_cols[:5]}...")
        for c in missing_cols:
            state_df[c] = 0.0
    return state_df


def load_frozen_model(
    model_id: str,
    config: dict,
    d_temporal: int,
    d_static: int,
    n_actions: int,
    device: torch.device,
    tag: str | None = None,
) -> torch.nn.Module:
    """Load a frozen MIMIC-trained model checkpoint (mirrors 90b_05)."""
    model_cfg = config["models"].get(model_id, {})
    grid_cfg = config["grid"]

    base_name = f"{model_id}_{tag}_best.pt" if tag else f"{model_id}_best.pt"
    ckpt_path = MODELS_DIR / base_name

    if not ckpt_path.exists() and tag:
        ckpt_path = MODELS_DIR / f"{model_id}_best.pt"
        print(f"  WARNING: {base_name} not found, falling back to {ckpt_path.name}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if "gtransformer" in model_id:
        model = build_gtransformer_from_config(
            d_temporal, d_static, n_actions, model_cfg, grid_cfg
        )
    elif "gnet" in model_id:
        model = build_gnet_from_config(
            d_temporal, d_static, n_actions, model_cfg, grid_cfg
        )
    elif "actin" in model_id:
        model = build_actin_from_config(
            d_temporal, d_static, n_actions, model_cfg, grid_cfg
        )
    else:
        raise ValueError(f"Unknown model: {model_id}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", "?")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {model_id}: {n_params:,} params, "
          f"epoch={epoch}, val_loss={val_loss}")
    return model


def run_subgroup_analysis(
    model,
    model_id: str,
    state_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    propensity_df: pd.DataFrame,
    subset_stay_ids: np.ndarray,
    subset_name: str,
    n_ef_documented: int,
    config: dict,
    device: torch.device,
    mimic_spec: dict,
    merge_map: dict[int, int] | None,
    args,
) -> dict:
    """
    Run protocol-PRIMARY external DM (non-rollout, teacher-forced) on a subset.

    Mirrors 90b_05.run_subgroup_analysis()/save_subgroup_results conventions but
    REMOVES the <30 subgroup-skip floor (descriptive subset). If the subset is
    below args.min_n it is still reported (N-only when empty / below floor; CI
    only reported when bootstrap succeeds).
    """
    print(f"\n  Subgroup analysis: {subset_name} (model={model_id})...")
    temporal_cols = mimic_spec["temporal_cols"]
    static_cols = mimic_spec["static_cols"]

    n = len(subset_stay_ids)
    row = {
        "subset": subset_name,
        "n_stays": int(n),
        "n_ef_documented": int(n_ef_documented),
        "v_observed": np.nan,
        "v_rule": np.nan,
        "ate": np.nan,
        "ate_ci_lo": np.nan,
        "ate_ci_hi": np.nan,
        "database": "eICU",
    }

    if n == 0:
        print(f"    {subset_name}: n=0 — NO qualifying stays (reporting N-only)")
        return row
    if n < args.min_n:
        print(f"    {subset_name}: n={n} < min_n={args.min_n} "
              f"(floor lowered for descriptive subset; proceeding anyway)")

    sub_ds = DTRDataset(
        state_df=state_df,
        outcomes_df=outcomes_df,
        propensity_df=propensity_df,
        stay_ids=subset_stay_ids,
        temporal_cols=temporal_cols,
        static_cols=static_cols,
        merge_map=merge_map,
        n_bins=config["grid"]["n_bins"],
        action_vocab=config["action"]["valid_merged_indices"],
    )
    print(f"    DTR dataset: {len(sub_ds)} stays, "
          f"{sub_ds.d_temporal} temporal, {sub_ds.d_static} static, "
          f"{sub_ds.n_actions} actions (MIMIC vocab forced)")

    if not hasattr(model, "predict_counterfactual"):
        print("    WARNING: model lacks predict_counterfactual; cannot run DM")
        return row

    # PRIMARY external estimator: non-rollout (teacher-forced) DM
    print("    DM (rollout=False) [PRIMARY for external validation]...")
    dm = evaluate_dm(model, sub_ds, device, use_rollout=False)
    row["v_observed"] = float(dm["v_observed"])
    row["v_rule"] = float(dm["v_rule"])
    row["ate"] = float(dm["ate"])
    print(f"      V(obs)={dm['v_observed']:.4f}, V(rule)={dm['v_rule']:.4f}, "
          f"ATE={dm['ate']:+.4f}")

    if not args.skip_bootstrap:
        n_boot = args.bootstrap
        seed = config["ope"]["bootstrap"]["seed"]
        print(f"    Bootstrap DM non-rollout (B={n_boot}) [PRIMARY]...")

        def dm_fn(ds, **kw):
            return evaluate_dm(model, ds, device, use_rollout=False)

        boot_dm = bootstrap_ope(dm_fn, sub_ds, n_resamples=n_boot, seed=seed)
        if boot_dm.get("n_success", 0) > 0:
            row["ate_ci_lo"] = float(boot_dm["ate_ci_lo"])
            row["ate_ci_hi"] = float(boot_dm["ate_ci_hi"])
            print(f"      ATE={boot_dm['ate_mean']:+.4f} "
                  f"[{boot_dm['ate_ci_lo']:.4f}, {boot_dm['ate_ci_hi']:.4f}] "
                  f"(B_success={boot_dm['n_success']})")
        else:
            print("      Bootstrap produced no successful resamples "
                  "(too few stays?) — CI left blank")

    return row


def main():
    args = parse_args()

    print("=" * 72)
    print("90b_08 — eICU EF-strict HFpEF Transportability Sensitivity (A)")
    print("=" * 72)
    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 1. Setup ─────────────────────────────────────────────────────────
    print("\n[1] Setup...")
    config_path = ROOT / "configs" / "80_step_c_training_v1.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  WARNING: No GPU, using CPU (will be slow)")

    mimic_spec = load_mimic_feature_spec()
    print(f"  MIMIC spec: {mimic_spec['n_temporal']} temporal, "
          f"{mimic_spec['n_static']} static, {mimic_spec['n_actions']} actions")
    merge_map = load_merge_map()

    # ── 2. Load eICU data (state tensors + outcomes + propensity + cohort) ─
    print("\n[2] Loading eICU data...")
    state_df = pd.read_parquet(INTERIM_DIR / "90b_eicu_state_tensors_v1.parquet")
    outcomes_df = pd.read_parquet(INTERIM_DIR / "90b_eicu_outcomes_absorb.parquet")
    propensity_df = pd.read_parquet(INTERIM_DIR / "90b_eicu_propensity_coarse3.parquet")
    cohort = pd.read_parquet(STAGING_DIR / "90b_eicu_cohort_v1.parquet")
    state_df = align_eicu_to_mimic_features(state_df, mimic_spec)

    cohort_stays = set(state_df["stay_id"].unique())
    print(f"  eICU cohort stays (in state tensors): {len(cohort_stays):,}")
    if "hf_subtype" not in cohort.columns:
        raise KeyError("cohort missing 'hf_subtype' — cannot build HFpEF subset")

    # ICD-based broader HFpEF (the 90b_05 subgroup)
    icd_hfpef = cohort.loc[cohort["hf_subtype"] == "hfpef", "patientunitstayid"]
    icd_hfpef_stays = set(icd_hfpef.unique()) & cohort_stays
    print(f"  ICD-based broader HFpEF stays (in cohort+tensors): "
          f"{len(icd_hfpef_stays):,}")

    # ── 3. Join raw EF (apachePredVar.ejectfx) ───────────────────────────
    print("\n[3] Loading + joining eICU EF (apachePredVar.ejectfx)...")
    ef = load_eicu_ef()
    ef_documented_stays = set(ef["patientunitstayid"].unique())
    ef_ge_thr_stays = set(
        ef.loc[ef["ejectfx"] >= args.ef_threshold, "patientunitstayid"].unique()
    )

    # EF documentation within the ICD-HFpEF group
    icd_hfpef_ef_documented = icd_hfpef_stays & ef_documented_stays
    n_ef_documented = len(icd_hfpef_ef_documented)

    # EF-strict HFpEF = ICD-HFpEF AND ejectfx >= threshold (e.g. 50)
    ef_strict_hfpef_stays = icd_hfpef_stays & ef_ge_thr_stays & cohort_stays
    n_strict = len(ef_strict_hfpef_stays)

    # ── LOUD reporting of qualifying N ───────────────────────────────────
    print("\n" + "!" * 72)
    print("!!  EF-STRICT HFpEF QUALIFYING COUNTS (LOUD)")
    print("!" * 72)
    print(f"!!  ICD-based broader HFpEF stays                : {len(icd_hfpef_stays):,}")
    print(f"!!  ... with DOCUMENTED EF (ejectfx > 0)         : {n_ef_documented:,}")
    print(f"!!  ... with EF >= {args.ef_threshold:.0f} (EF-STRICT HFpEF)        : {n_strict:,}")
    if n_ef_documented > 0:
        print(f"!!  EF>= {args.ef_threshold:.0f} fraction among EF-documented   : "
              f"{100 * n_strict / n_ef_documented:.1f}%")
    print(f"!!  EXPECTED (spec §A): ~50-60 qualifying stays  → DESCRIPTIVE-ONLY")
    print("!" * 72 + "\n")

    # ── 4. Load frozen model ─────────────────────────────────────────────
    print("\n[4] Loading frozen MIMIC model...")
    model = load_frozen_model(
        args.model, config,
        d_temporal=len(mimic_spec["temporal_cols"]),
        d_static=len(mimic_spec["static_cols"]),
        n_actions=mimic_spec["n_actions"],
        device=device,
        tag=args.tag if "gtransformer" in args.model else None,
    )

    # ── 5. Run PRIMARY external DM on the EF-strict HFpEF subset ──────────
    print("\n[5] EF-strict HFpEF subset evaluation (PRIMARY non-rollout DM)...")
    rows = []

    # Reference subset: full ICD-HFpEF (90b_05 broader subgroup) for context
    ref_row = run_subgroup_analysis(
        model, args.model, state_df, outcomes_df, propensity_df,
        subset_stay_ids=np.array(sorted(icd_hfpef_stays)),
        subset_name="hfpef_icd_broader",
        n_ef_documented=n_ef_documented,
        config=config, device=device, mimic_spec=mimic_spec,
        merge_map=merge_map, args=args,
    )
    rows.append(ref_row)

    # Primary descriptive subset: EF-strict HFpEF (ICD-HFpEF AND ejectfx>=thr)
    strict_row = run_subgroup_analysis(
        model, args.model, state_df, outcomes_df, propensity_df,
        subset_stay_ids=np.array(sorted(ef_strict_hfpef_stays)),
        subset_name=f"hfpef_ef_ge{int(args.ef_threshold)}",
        n_ef_documented=n_ef_documented,
        config=config, device=device, mimic_spec=mimic_spec,
        merge_map=merge_map, args=args,
    )
    rows.append(strict_row)

    # ── 6. Save table ────────────────────────────────────────────────────
    print("\n[6] Saving results table...")
    out_cols = [
        "subset", "n_stays", "n_ef_documented", "v_observed", "v_rule",
        "ate", "ate_ci_lo", "ate_ci_hi", "database",
    ]
    out_df = pd.DataFrame(rows)[out_cols]
    out_path = TABLES_DIR / "90b_08_eicu_ef_strict_hfpef_sensitivity.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    print(out_df.to_string(index=False))

    # ── 7. Report ────────────────────────────────────────────────────────
    print("\n[7] Generating report...")
    report_lines = [
        "=" * 72,
        "90b_08 — eICU EF-strict HFpEF Transportability Sensitivity (A)",
        "=" * 72,
        f"Generated: {ts}",
        f"Protocol: v3.1, Step 90b (sub-step 6); enhancement spec §A",
        "",
        "DESIGN",
        "──────",
        f"  Frozen model: {args.model} (tag={args.tag})",
        f"  PRIMARY estimator: DM non-rollout (teacher-forced, external-validation)",
        f"  EF source: data/raw/eicu/2.0/apachePredVar.csv.gz (ejectfx; -1 sentinel)",
        f"  EF-strict definition: ICD-HFpEF (hf_subtype=='hfpef') AND ejectfx >= "
        f"{args.ef_threshold:.0f}",
        f"  Bootstrap: B={args.bootstrap}"
        f"{' (SKIPPED)' if args.skip_bootstrap else ''}",
        f"  Subgroup-skip floor: min_n={args.min_n} (descriptive; floor removed)",
        "",
        "QUALIFYING COUNTS",
        "─────────────────",
        f"  ICD-based broader HFpEF stays          : {len(icd_hfpef_stays):,}",
        f"  ... with documented EF (ejectfx > 0)   : {n_ef_documented:,}",
        f"  ... with EF >= {args.ef_threshold:.0f} (EF-STRICT HFpEF)  : {n_strict:,}",
        "",
        "RESULTS",
        "───────",
    ]
    for _, r in out_df.iterrows():
        line = (f"  {r['subset']:24s}: n={int(r['n_stays']):>5}, "
                f"n_ef_doc={int(r['n_ef_documented']):>4}")
        if pd.notna(r["ate"]):
            line += f", V(obs)={r['v_observed']:.4f}, V(rule)={r['v_rule']:.4f}, "
            line += f"ATE={r['ate']:+.4f}"
            if pd.notna(r["ate_ci_lo"]) and pd.notna(r["ate_ci_hi"]):
                line += f" [{r['ate_ci_lo']:.4f}, {r['ate_ci_hi']:.4f}]"
            else:
                line += " [CI unavailable — too few stays]"
        else:
            line += " — NO qualifying stays (N-only)"
        report_lines.append(line)

    report_lines.extend([
        "",
        "INTERPRETATION",
        "──────────────",
        "  This is a DESCRIPTIVE-ONLY sensitivity. eICU EF is documented for only a",
        "  tiny minority of stays (apachePredVar.ejectfx is -1 'not assessed' for the",
        "  vast majority), so the EF-strict HFpEF subset is very small (expected",
        "  ~50-60). The point estimate carries a wide CI and should NOT be over-",
        "  interpreted; its value is to show (a) how few ICD-HFpEF stays carry a",
        "  documented EF >= 50 and (b) that, on that EF-confirmed sliver, the",
        "  external non-rollout DM sign/magnitude is consistent (or not) with the",
        "  broader-ascertainment HFpEF result. This directly addresses the EF-misclassification",
        "  EF-misclassification critique.",
        "",
        "NEXT STEPS",
        "──────────",
        "  Feed 90b_08 table into the transportability narrative + supplement; pair",
        "  with the 90b_05 broader-HFpEF subgroup for the misclassification discussion.",
    ])

    report_path = REPORTS_DIR / "90b_08_eicu_ef_strict_hfpef_sensitivity.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"  Report: {report_path}")

    total_time = time.time() - t0
    print()
    print("=" * 72)
    print(f"90b_08 COMPLETE. EF-strict HFpEF n={n_strict}, "
          f"n_ef_documented={n_ef_documented}. Time: {total_time/60:.1f} min")
    print("=" * 72)


if __name__ == "__main__":
    main()
