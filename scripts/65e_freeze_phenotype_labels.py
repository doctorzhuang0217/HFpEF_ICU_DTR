#!/usr/bin/env python3
"""
65e_freeze_phenotype_labels — Freeze phenotype labels for Step C.

Purpose
-------
Consolidate all phenotype labels (early + trajectory) into a single
frozen registry and label file for downstream Step C causal models.

Frozen labels include:
  Primary (for Step C stratification):
    - rule_phenotype_2class: shock_predominant vs non_shock (from 60d)
    - rule_phenotype_4class: shock/congestion/mixed/neither (from 60d)
  Sensitivity (for Step C sensitivity analysis):
    - early_data_driven_k2: K=2 K-means early clusters (from 60b)
  Explanatory (for Aim 3, NOT used in Step C):
    - trajectory_data_driven: K-means trajectory clusters (from 65b)
    - trajectory_gmvae: GMVAE trajectory clusters (from 65b, if available)

Protocol reference: Step 65e (v3.1 § 12)

Inputs
------
- data/interim/60b_early_clusters_v1_mimic.parquet
- data/interim/60d_rule_phenotype_v1_mimic.parquet
- data/interim/65b_trajectory_clusters_v1_mimic.parquet
- tables/60c_early_stability.csv
- tables/65c_trajectory_stability.csv

Outputs
-------
- data/processed/phenotype_labels_frozen_v1_mimic.parquet
- configs/phenotype_registry_v1.json
- reports/65e_freeze_phenotype_labels_summary.txt

Compute: C1
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path


def main() -> None:
    proj_cfg = load_config()
    project_root = Path(proj_cfg.get("project_root", "."))
    interim_dir = resolve_path(proj_cfg, "interim")
    processed_dir = resolve_path(proj_cfg, "processed")
    configs_dir = resolve_path(proj_cfg, "configs")
    tables_dir = resolve_path(proj_cfg, "tables")
    reports_dir = resolve_path(proj_cfg, "reports")
    for d in [processed_dir, configs_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("65e_freeze_phenotype_labels — Freeze phenotype labels for Step C")
    print("=" * 72)

    registry = {
        "version": "v1",
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "v3.1",
        "database": "mimic",
        "labels": {},
    }

    # ── Load HF subtype from 25d (for Step C interaction analysis) ──────
    print("\n[0/4] Loading hf_subtype from 25d ...")
    staging_dir = resolve_path(proj_cfg, "staging")
    hf_subtype_map = None
    syn_path = staging_dir / "25d_dominant_syndrome_flags_mimic.parquet"
    if syn_path.exists():
        full_cols = pq.ParquetFile(syn_path).schema.names
        if "hf_subtype" in full_cols:
            syn_df = pd.read_parquet(syn_path, columns=["stay_id", "hf_subtype"])
            syn_df = syn_df.drop_duplicates(subset=["stay_id"])
            hf_subtype_map = syn_df
            print(
                f"  hf_subtype loaded: {len(syn_df):,} stays, "
                f"{syn_df['hf_subtype'].nunique()} subtypes"
            )
        else:
            print("  WARNING: 25d exists but hf_subtype column is missing")
    else:
        # Fallback: try 25a
        alt_path = staging_dir / "25a_time_zero_flags_mimic.parquet"
        if alt_path.exists():
            alt_cols = pq.ParquetFile(alt_path).schema.names
            if "hf_subtype" in alt_cols:
                alt_df = pd.read_parquet(alt_path, columns=["stay_id", "hf_subtype"])
                alt_df = alt_df.drop_duplicates(subset=["stay_id"])
                hf_subtype_map = alt_df
                print(f"  hf_subtype loaded from 25a: {len(alt_df):,} stays")
            else:
                print("  WARNING: 25a exists but hf_subtype column is missing")
        else:
            print("  WARNING: hf_subtype not found in 25d or 25a")

    # ── Load early phenotype labels (60b, 60d) ─────────────────────────
    print("\n[1/4] Loading early phenotype labels ...")
    frozen = None

    # Rule-based (primary)
    rule_path = interim_dir / "60d_rule_phenotype_v1_mimic.parquet"
    if rule_path.exists():
        rule_df = pd.read_parquet(rule_path)
        rule_df = rule_df.drop_duplicates(subset=["stay_id"])
        frozen = rule_df[["stay_id"]].copy()

        if "rule_phenotype_4class" in rule_df.columns:
            frozen["rule_phenotype_4class"] = rule_df["rule_phenotype_4class"].values
            registry["labels"]["rule_phenotype_4class"] = {
                "role": "primary",
                "source_step": "60d",
                "description": "Rule-based 4-class: shock_dominant, congestion_dominant, mixed, neither",
                "used_in_step_c": True,
                "n_classes": int(rule_df["rule_phenotype_4class"].nunique()),
            }
            print(f"  rule_phenotype_4class: {rule_df['rule_phenotype_4class'].nunique()} classes")

        if "rule_phenotype_2class" in rule_df.columns:
            frozen["rule_phenotype_2class"] = rule_df["rule_phenotype_2class"].values
            registry["labels"]["rule_phenotype_2class"] = {
                "role": "primary",
                "source_step": "60d",
                "description": "Rule-based 2-class: shock_predominant vs non_shock",
                "used_in_step_c": True,
                "n_classes": int(rule_df["rule_phenotype_2class"].nunique()),
            }
            print(f"  rule_phenotype_2class: {rule_df['rule_phenotype_2class'].nunique()} classes")
    else:
        print("  WARNING: 60d rule phenotype not found")

    # Data-driven early (sensitivity)
    early_cluster_path = interim_dir / "60b_early_clusters_v1_mimic.parquet"
    if early_cluster_path.exists():
        ec_df = pd.read_parquet(early_cluster_path)
        ec_df = ec_df.drop_duplicates(subset=["stay_id"])

        # K=2 as sensitivity (stable, ARI=0.986)
        if "kmeans_k2" in ec_df.columns:
            k2_labels = ec_df[["stay_id", "kmeans_k2"]].copy()
            if frozen is not None:
                frozen = frozen.merge(k2_labels, on="stay_id", how="left")
            else:
                frozen = k2_labels.copy()
            frozen.rename(
                columns={"kmeans_k2": "early_data_driven_k2"}, inplace=True
            )
            registry["labels"]["early_data_driven_k2"] = {
                "role": "sensitivity",
                "source_step": "60b",
                "description": "Data-driven K=2 early clusters (stable, ARI=0.986)",
                "used_in_step_c": True,
                "n_classes": 2,
                "note": "Primary K=3 failed stability (ARI_P5=0.164); K=2 used as sensitivity",
            }
            print(f"  early_data_driven_k2: 2 classes (sensitivity)")

        # Selected K for reference (even if unstable)
        selected_col = [c for c in ec_df.columns if c.startswith("selected_k")]
        if selected_col:
            sel_col = selected_col[0]
            sel_k = int(sel_col.replace("selected_k", ""))
            ref_labels = ec_df[["stay_id", sel_col]].copy()
            ref_labels.rename(
                columns={sel_col: f"early_data_driven_k{sel_k}_ref"}, inplace=True
            )
            frozen = frozen.merge(ref_labels, on="stay_id", how="left")
            registry["labels"][f"early_data_driven_k{sel_k}_ref"] = {
                "role": "reference_only",
                "source_step": "60b",
                "description": f"Data-driven K={sel_k} (60b selected, failed stability 60c)",
                "used_in_step_c": False,
                "n_classes": sel_k,
                "note": "Reference only — do NOT use for stratification",
            }
            print(f"  early_data_driven_k{sel_k}_ref: reference only")
    else:
        print("  WARNING: 60b early clusters not found")

    # ── Load trajectory phenotype labels (65b) ─────────────────────────
    print("\n[2/4] Loading trajectory phenotype labels ...")
    traj_path = interim_dir / "65b_trajectory_clusters_v1_mimic.parquet"
    if traj_path.exists():
        traj_df = pd.read_parquet(traj_path)
        traj_df = traj_df.drop_duplicates(subset=["stay_id"])

        # Selected trajectory K-means
        traj_selected_col = [
            c for c in traj_df.columns if c.startswith("traj_selected_k")
        ]
        if traj_selected_col:
            tsel = traj_selected_col[0]
            traj_k = int(tsel.replace("traj_selected_k", ""))
            traj_labels = traj_df[["stay_id", tsel]].copy()
            traj_labels.rename(
                columns={tsel: f"trajectory_data_driven_k{traj_k}"}, inplace=True
            )
            if frozen is not None:
                frozen = frozen.merge(traj_labels, on="stay_id", how="left")
            else:
                frozen = traj_labels.copy()
            registry["labels"][f"trajectory_data_driven_k{traj_k}"] = {
                "role": "explanatory",
                "source_step": "65b",
                "description": f"Data-driven trajectory K={traj_k} clusters",
                "used_in_step_c": False,
                "n_classes": traj_k,
                "note": "Explanatory only (Aim 3) — does NOT feed primary estimand",
            }
            print(f"  trajectory_data_driven_k{traj_k}: {traj_k} classes (explanatory)")

        # GMVAE clusters
        if "gmvae_cluster" in traj_df.columns:
            gmvae_labels = traj_df[["stay_id", "gmvae_cluster"]].copy()
            frozen = frozen.merge(gmvae_labels, on="stay_id", how="left")
            n_gmvae = int(traj_df["gmvae_cluster"].nunique())
            registry["labels"]["trajectory_gmvae"] = {
                "role": "explanatory",
                "source_step": "65b",
                "description": f"GMVAE trajectory clusters ({n_gmvae} components)",
                "used_in_step_c": False,
                "n_classes": n_gmvae,
                "note": "Explanatory only (Aim 3)",
            }
            print(f"  trajectory_gmvae: {n_gmvae} classes (explanatory)")
    else:
        print("  WARNING: 65b trajectory clusters not found")

    if frozen is None:
        print("\n  ERROR: no phenotype labels found to freeze")
        sys.exit(1)

    # ── Merge hf_subtype into frozen labels ──────────────────────────
    if hf_subtype_map is not None:
        frozen = frozen.merge(hf_subtype_map, on="stay_id", how="left")
        n_matched = frozen["hf_subtype"].notna().sum()
        registry["labels"]["hf_subtype"] = {
            "role": "interaction",
            "source_step": "25d",
            "description": "HF subtype (hfpef, hfref, hfmref, non_hf) for Step C interaction analysis",
            "used_in_step_c": True,
            "n_classes": int(frozen["hf_subtype"].nunique()),
            "note": "Required for hf_subtype x shock 2x2 interaction analysis (80f)",
        }
        print(f"\n  hf_subtype merged: {n_matched:,}/{len(frozen):,} matched")
        print(f"  Distribution: {frozen['hf_subtype'].value_counts().to_dict()}")

    # ── Load stability results ─────────────────────────────────────────
    print("\n[3/4] Loading stability results ...")
    for stab_file, prefix in [
        ("60c_early_stability.csv", "early"),
        ("65c_trajectory_stability.csv", "trajectory"),
    ]:
        stab_path = tables_dir / stab_file
        if stab_path.exists():
            stab = pd.read_csv(stab_path)
            registry[f"{prefix}_stability"] = stab.to_dict(orient="records")
            print(f"  {prefix}: {len(stab)} K values assessed")
        else:
            print(f"  {prefix}: stability file not found")

    # ── Save frozen labels ─────────────────────────────────────────────
    print("\n[4/4] Saving frozen phenotype labels ...")
    out_path = processed_dir / "phenotype_labels_frozen_v1_mimic.parquet"
    frozen.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"  Saved: {out_path} ({len(frozen):,} stays, {frozen.shape[1]} columns)")

    # Registry JSON
    registry_path = configs_dir / "phenotype_registry_v1.json"
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2, default=str)
    print(f"  Saved: {registry_path}")

    # ── Summary report ────────────────────────────────────────────────
    report_lines = [
        "=" * 72,
        "65e — Frozen Phenotype Label Registry (MIMIC-IV)",
        "=" * 72,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "LABEL FILE",
        "──────────",
        f"  Path: {out_path}",
        f"  Stays: {len(frozen):,}",
        f"  Columns: {list(frozen.columns)}",
        "",
        "REGISTERED LABELS",
        "─────────────────",
    ]

    for label_id, meta in registry["labels"].items():
        report_lines.extend([
            f"  {label_id}:",
            f"    Role: {meta['role']}",
            f"    Source: Step {meta['source_step']}",
            f"    Classes: {meta['n_classes']}",
            f"    Used in Step C: {meta['used_in_step_c']}",
            f"    Description: {meta['description']}",
        ])
        if "note" in meta:
            report_lines.append(f"    Note: {meta['note']}")

    # Distribution summary
    report_lines.extend(["", "LABEL DISTRIBUTIONS", "───────────────────"])
    for col in frozen.columns:
        if col == "stay_id":
            continue
        vc = frozen[col].value_counts()
        report_lines.append(f"\n  {col}:")
        for val, cnt in vc.head(10).items():
            report_lines.append(
                f"    {val}: {cnt:,} ({cnt / len(frozen) * 100:.1f}%)"
            )
        n_na = int(frozen[col].isna().sum())
        if n_na > 0:
            report_lines.append(f"    NaN: {n_na:,}")

    report_lines.extend([
        "",
        "STEP C USAGE GUIDE",
        "──────────────────",
        "  Primary stratification: rule_phenotype_2class",
        "    → use for phenotype-specific DTR analysis",
        "    → split: shock_predominant vs non_shock",
        "",
        "  Sensitivity analysis: early_data_driven_k2",
        "    → repeat Step C with data-driven binary phenotype",
        "    → compare conclusions with rule-based primary",
        "",
        "  Explanatory enrichment: trajectory_data_driven_k*",
        "    → post-hoc trajectory-outcome association analysis",
        "    → does NOT enter Step C model fitting",
        "",
        "NEXT STEPS",
        "──────────",
        "  Step 70: policy framework setup",
        "  Step 75: primary DTR estimation",
    ])

    report = "\n".join(report_lines) + "\n"
    report_path = reports_dir / "65e_freeze_phenotype_labels_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n  Saved: {report_path}")
    print(report)
    print("[65e] Done.")


if __name__ == "__main__":
    main()
