#!/usr/bin/env python3
"""P5 trajectory data export for R figure redrawing.

Extracts the 6 representative patient trajectories (from P5 main analysis)
into a flat CSV that the R plotting script can read directly.

Run AFTER P5_counterfactual_trajectories.py has completed.

Compute tier: C1 (lightweight)

Outputs:
  - tables/P5_trajectory_data.csv  (per-patient × per-bin: state + actions)
  - tables/P5_case_metadata.csv    (per-patient: label, CATE, HF, pheno, mortality)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(ROOT / "src"))

from hfpef_dtr.models.dtr_dataset import DTRDataset, load_dtr_data
from hfpef_dtr.models.dataset import make_train_val_test_split

CONFIG_PATH = ROOT / "configs" / "80_step_c_training_v1.yaml"
NORM_PATH = ROOT / "configs" / "45c_normalization_params_v1.yaml"
TABLES_DIR = ROOT / "tables"


def decode_action_components(action_idx: int) -> tuple[int, int, int]:
    """Decode action_idx -> (fluid_level, vaso_level, diuretic_level)."""
    diuretic = action_idx % 2
    vaso = (action_idx // 2) % 3
    fluid = action_idx // 6
    return fluid, vaso, diuretic


def main():
    print("=" * 60)
    print("P5 trajectory export for R figure redrawing")
    print("=" * 60)

    # Load config
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    with open(NORM_PATH) as f:
        norm_cfg = yaml.safe_load(f)
    norm_params = norm_cfg.get("params", {})

    n_bins = config["grid"]["n_bins"]
    bin_hours = config["grid"]["bin_hours"]

    # Load data
    print("[1] Loading data...")
    state_df, outcomes_df, propensity_df, temporal_cols, static_cols, merge_map = (
        load_dtr_data(ROOT, config)
    )
    split_cfg = config["split"]
    all_stay_ids = state_df["stay_id"].unique()
    _, _, test_ids = make_train_val_test_split(
        all_stay_ids, outcomes_df,
        train_frac=split_cfg["train_frac"],
        val_frac=split_cfg["val_frac"],
        test_frac=split_cfg["test_frac"],
        stratify_col=split_cfg["stratify_col"],
        seed=split_cfg["seed"],
    )

    test_ds = DTRDataset(
        state_df=state_df, outcomes_df=outcomes_df,
        propensity_df=propensity_df, stay_ids=test_ids,
        temporal_cols=temporal_cols, static_cols=static_cols,
        merge_map=merge_map, n_bins=n_bins,
    )

    pheno_df = pd.read_parquet(ROOT / config["data"]["phenotype_labels"])
    propensity_df_raw = pd.read_parquet(ROOT / config["data"]["propensities_coarse3"])
    rule_policy_df = pd.read_parquet(
        ROOT / "data" / "interim" / "70c_rule_policy_actions_v1_mimic.parquet"
    )
    cate_df = pd.read_csv(TABLES_DIR / "P2_cate_distribution.csv")

    # Read the P5 summary to get the 6 selected stay_ids
    print("[2] Reading P5 case selections from report...")
    report_path = ROOT / "reports" / "P5_trajectory_summary.txt"
    selected_cases = []
    with open(report_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("stay_id="):
                # Parse: stay_id=<id>, CATE=<value>pp, HF=<subtype>, ...
                parts = line.split(", ")
                info = {}
                for p in parts:
                    k, v = p.split("=", 1)
                    info[k.strip()] = v.strip()
                selected_cases.append(info)
            elif "Case " in line and "stay_id=" in line:
                # Parse: Case 1: HFpEF × shock\n    stay_id=...
                pass  # handled by next line
            elif line.startswith("Case "):
                # "Case 1: HFpEF × shock"
                parts = line.split(": ", 1)
                case_num = int(parts[0].replace("Case ", ""))
                case_label = parts[1] if len(parts) > 1 else ""
                # Next non-empty line has stay_id
                if len(selected_cases) >= case_num:
                    selected_cases[case_num - 1]["case"] = case_num
                    selected_cases[case_num - 1]["label"] = case_label

    # Re-parse more robustly from the report
    selected_cases = []
    with open(report_path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Case "):
            parts = line.split(": ", 1)
            case_num = int(parts[0].replace("Case ", ""))
            case_label = parts[1] if len(parts) > 1 else ""
            # Next line has details
            if i + 1 < len(lines):
                detail = lines[i + 1].strip()
                info = {"case": case_num, "label": case_label}
                for token in detail.split(", "):
                    if "=" in token:
                        k, v = token.split("=", 1)
                        info[k.strip()] = v.strip()
                selected_cases.append(info)
                i += 2
                continue
        i += 1

    print(f"  Found {len(selected_cases)} cases")
    for c in selected_cases:
        print(f"    Case {c['case']}: {c['label']} (stay_id={c['stay_id']})")

    # Build lookup maps
    sid_to_idx = {sid: i for i, sid in enumerate(test_ds.stay_ids)}

    # Key temporal variable indices
    col_map = {name: i for i, name in enumerate(temporal_cols)}
    key_vars = ["map", "heart_rate", "spo2", "lactate"]
    key_z_names = [f"{v}_z" if f"{v}_z" in col_map else v for v in key_vars]
    key_indices = [col_map.get(n) for n in key_z_names]

    # ── 3. Export trajectory data ──
    print("[3] Extracting trajectory data...")
    rows = []
    meta_rows = []

    for case_info in selected_cases:
        sid = int(case_info["stay_id"])
        case_num = case_info["case"]
        case_label = case_info["label"]

        if sid not in sid_to_idx:
            print(f"  WARNING: stay_id {sid} not in test set, skipping")
            continue

        idx = sid_to_idx[sid]
        temporal = test_ds.temporal[idx]   # (T, D) numpy
        valid = test_ds.valid_mask[idx]    # (T,)
        mortality = int(test_ds.mortality_28d[idx])

        # Get CATE
        cate_row = cate_df[cate_df["stay_id"] == sid]
        cate_val = float(cate_row["cate"].iloc[0]) if len(cate_row) > 0 else 0.0

        # Get phenotype info
        pheno_row = pheno_df[pheno_df["stay_id"] == sid]
        hf_subtype = pheno_row.iloc[0]["hf_subtype"] if len(pheno_row) > 0 else ""
        phenotype = pheno_row.iloc[0].get("rule_phenotype_2class", "") if len(pheno_row) > 0 else ""

        # Get actions
        obs_col = "action_idx" if "action_idx" in propensity_df_raw.columns else "action_idx_merged"
        sub_prop = propensity_df_raw[propensity_df_raw["stay_id"] == sid]
        sub_rule = rule_policy_df[rule_policy_df["stay_id"] == sid]

        obs_actions = np.zeros(n_bins, dtype=int)
        rule_actions = np.zeros(n_bins, dtype=int)
        for _, row in sub_prop.iterrows():
            b = int(row["bin"])
            if 0 <= b < n_bins:
                obs_actions[b] = int(row[obs_col])
        for _, row in sub_rule.iterrows():
            b = int(row["bin"])
            if 0 <= b < n_bins:
                rule_actions[b] = int(row["rule_policy_action"])

        # Export per-bin data
        for t in range(n_bins):
            hour = t * bin_hours
            row_data = {
                "case": case_num,
                "stay_id": sid,
                "bin": t,
                "hour": hour,
                "valid": int(valid[t]),
            }
            # Denormalize key variables
            for var_name, z_name, col_idx in zip(key_vars, key_z_names, key_indices):
                if col_idx is not None:
                    z_val = float(temporal[t, col_idx])
                    if var_name in norm_params:
                        raw_val = z_val * norm_params[var_name]["std"] + norm_params[var_name]["mean"]
                    else:
                        raw_val = z_val
                    row_data[f"{var_name}_z"] = z_val
                    row_data[var_name] = raw_val
                else:
                    row_data[f"{var_name}_z"] = np.nan
                    row_data[var_name] = np.nan

            # Actions
            row_data["obs_action"] = int(obs_actions[t])
            row_data["rule_action"] = int(rule_actions[t])

            # Decode action components
            of, ov, od = decode_action_components(int(obs_actions[t]))
            rf, rv, rd = decode_action_components(int(rule_actions[t]))
            row_data["obs_fluid"] = of
            row_data["obs_vaso"] = ov
            row_data["obs_diur"] = od
            row_data["rule_fluid"] = rf
            row_data["rule_vaso"] = rv
            row_data["rule_diur"] = rd
            row_data["agree"] = 1 if obs_actions[t] == rule_actions[t] else 0

            rows.append(row_data)

        # Metadata
        agree_count = sum(
            1 for t in range(n_bins)
            if valid[t] > 0 and obs_actions[t] == rule_actions[t]
        )
        valid_count = sum(1 for t in range(n_bins) if valid[t] > 0)
        meta_rows.append({
            "case": case_num,
            "label": case_label,
            "stay_id": sid,
            "cate": cate_val,
            "cate_pp": cate_val * 100,
            "hf_subtype": hf_subtype,
            "phenotype": phenotype,
            "mortality_28d": mortality,
            "agreement": f"{agree_count}/{valid_count}",
            "agreement_pct": agree_count / valid_count * 100 if valid_count > 0 else 0,
        })

    # Save
    traj_df = pd.DataFrame(rows)
    meta_df = pd.DataFrame(meta_rows)

    traj_path = TABLES_DIR / "P5_trajectory_data.csv"
    meta_path = TABLES_DIR / "P5_case_metadata.csv"
    traj_df.to_csv(traj_path, index=False)
    meta_df.to_csv(meta_path, index=False)

    print(f"\n  Saved: {traj_path} ({len(traj_df)} rows)")
    print(f"  Saved: {meta_path} ({len(meta_df)} rows)")
    print(f"  Columns: {list(traj_df.columns)}")
    print("\nP5 export COMPLETE.")


if __name__ == "__main__":
    main()
