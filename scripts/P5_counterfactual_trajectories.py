#!/usr/bin/env python3
"""P5 — Counterfactual Trajectory Visualization

Protocol: v3.1-patch, Step P5
Compute tier: C2 (lightweight)
Depends on: 80c G-Transformer trained model, patient-level data

Objective: Create individual-level counterfactual trajectory visualizations
for representative patients, showing how the rule policy would have altered
hemodynamic management and predicted outcomes.

Patient selection:
  1. HFpEF × shock_predominant (large predicted benefit)
  2. HFpEF × non_shock (moderate predicted benefit)
  3. Non-HF × shock_predominant (reference)
  4. Non-HF × non_shock (minimal predicted benefit)
  5. Patient where rule predicts harm (if any)
  6. Patient with concordant observed-rule treatment

Outputs:
  - figures/P5_trajectory_case_*.png (individual cases)
  - figures/P5_trajectory_panel.png (combined multi-panel)
  - reports/P5_trajectory_summary.txt
"""

from __future__ import annotations

import datetime
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import yaml

# ── Project root ──
ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(ROOT / "src"))

from hfpef_dtr.models.dtr_dataset import DTRDataset, load_dtr_data
from hfpef_dtr.models.dataset import make_train_val_test_split

# ── Paths ──
CONFIG_PATH = ROOT / "configs" / "80_step_c_training_v1.yaml"
NORM_PATH = ROOT / "configs" / "45c_normalization_params_v1.yaml"
MODEL_CKPT = ROOT / "models" / "80c_gtransformer_best.pt"
TABLES_DIR = ROOT / "tables"
FIGURES_DIR = ROOT / "figures"
REPORTS_DIR = ROOT / "reports"

for d in [TABLES_DIR, FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Action decoding
ACTION_LABELS = {
    0: "none/none/none",
    1: "none/none/diur",
    2: "none/low_v/none",
    4: "none/high_v/none",
    6: "low_fl/none/none",
    7: "low_fl/none/diur",
    8: "low_fl/low_v/none",
    9: "low_fl/low_v/diur",
    10: "low_fl/high_v/none",
    12: "high_fl/none/none",
    13: "high_fl/none/diur",
    14: "high_fl/low_v/none",
    16: "high_fl/high_v/none",
}


def decode_action_components(action_idx: int) -> tuple[int, int, int]:
    """Decode action_idx → (fluid_level, vaso_level, diuretic_level)."""
    diuretic = action_idx % 2
    vaso = (action_idx // 2) % 3
    fluid = action_idx // 6
    return fluid, vaso, diuretic


def extract_raw_action_trajectories(
    propensity_df: pd.DataFrame,
    rule_policy_df: pd.DataFrame,
    stay_id: int,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract raw observed/rule actions aligned to 70c coding (0-17).

    Observed actions come from the propensity file (column ``action_idx``).
    Rule actions come from the 70c rule policy file (column
    ``rule_policy_action``).
    """
    obs_col = "action_idx" if "action_idx" in propensity_df.columns else "action_idx_merged"

    obs = np.zeros(n_bins, dtype=np.int64)
    rule = np.zeros(n_bins, dtype=np.int64)

    # Observed actions from propensity table
    sub_prop = propensity_df[propensity_df["stay_id"] == stay_id]
    for _, row in sub_prop.iterrows():
        b = int(row["bin"])
        if 0 <= b < n_bins:
            obs[b] = int(row[obs_col])

    # Rule actions from 70c rule policy table
    sub_rule = rule_policy_df[rule_policy_df["stay_id"] == stay_id]
    rule_col = "rule_policy_action"
    for _, row in sub_rule.iterrows():
        b = int(row["bin"])
        if 0 <= b < n_bins:
            rule[b] = int(row[rule_col])

    return obs, rule


def select_representative_patients(
    cate_df: pd.DataFrame,
    pheno_df: pd.DataFrame,
) -> list[dict]:
    """Select representative patients for each category."""
    merged = cate_df.merge(
        pheno_df[["stay_id", "hf_subtype", "rule_phenotype_2class"]].drop_duplicates("stay_id"),
        on="stay_id", how="inner",
    )

    selections = []
    target_profiles = [
        ("HFpEF × shock", "hfpef", "shock_predominant", "benefit"),
        ("HFpEF × non-shock", "hfpef", "non_shock", "benefit"),
        ("Non-HF × shock", "no_hf", "shock_predominant", "benefit"),
        ("Non-HF × non-shock", "no_hf", "non_shock", "minimal"),
        ("Predicted harm", None, None, "harm"),
        ("Concordant treatment", None, None, "concordant"),
    ]

    for label, hf, pheno, selection_type in target_profiles:
        if selection_type == "harm":
            # Patient with largest positive CATE
            candidates = merged[merged["cate"] > 0].nlargest(10, "cate")
            if len(candidates) > 0:
                selected = candidates.iloc[len(candidates) // 2]  # median of top-10
                selections.append({"label": label, "stay_id": selected["stay_id"],
                                   "cate": selected["cate"], "hf_subtype": selected.get("hf_subtype"),
                                   "phenotype": selected.get("rule_phenotype_2class")})
            continue

        if selection_type == "concordant":
            # Patient with highest observed-rule action agreement
            # (true concordance, not just CATE≈0)
            # Requires propensity + rule_policy data to compute agreement
            # — defer to main() which has access to these; store placeholder
            selections.append({"label": label, "stay_id": None,
                               "cate": None, "hf_subtype": None,
                               "phenotype": None,
                               "_concordant_deferred": True})
            continue

        # Filter by HF subtype and phenotype
        mask = pd.Series(True, index=merged.index)
        if hf:
            mask &= merged["hf_subtype"] == hf
        if pheno:
            mask &= merged["rule_phenotype_2class"] == pheno

        candidates = merged[mask]
        if len(candidates) == 0:
            continue

        if selection_type == "benefit":
            # Median-benefit patient (not extreme)
            sorted_cand = candidates.sort_values("cate")
            idx = len(sorted_cand) // 4  # 25th percentile for moderate benefit
            selected = sorted_cand.iloc[idx]
        else:
            selected = candidates.iloc[len(candidates) // 2]

        selections.append({
            "label": label,
            "stay_id": selected["stay_id"],
            "cate": selected["cate"],
            "hf_subtype": selected.get("hf_subtype"),
            "phenotype": selected.get("rule_phenotype_2class"),
        })

    return selections


def plot_single_trajectory(
    stay_data: dict,
    norm_params: dict,
    patient_info: dict,
    output_path: Path,
    n_bins: int = 12,
    bin_hours: int = 2,
) -> None:
    """Plot observed vs rule-policy trajectory for a single patient."""
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 0.8], hspace=0.35)

    hours = np.arange(n_bins) * bin_hours
    valid = stay_data["valid_mask"]
    temporal = stay_data["temporal"]  # (T, D)
    obs_action_raw = stay_data["action_raw"]
    rule_action_raw = stay_data["rule_action_raw"]

    # Denormalize key state variables
    def denorm(col_name, z_col_idx):
        if col_name in norm_params and z_col_idx is not None:
            vals = temporal[:, z_col_idx]
            return vals * float(norm_params[col_name]["std"]) + float(norm_params[col_name]["mean"])
        return None

    # Try to find column indices (based on typical ordering)
    col_names = patient_info.get("temporal_cols", [])
    col_map = {name: i for i, name in enumerate(col_names)}

    map_idx = col_map.get("map_z") or col_map.get("map")
    lac_idx = col_map.get("lactate_z") or col_map.get("lactate")
    hr_idx = col_map.get("heart_rate_z") or col_map.get("heart_rate")
    spo2_idx = col_map.get("spo2_z") or col_map.get("spo2")

    # Panel 1: Hemodynamic state (MAP, HR)
    ax1 = fig.add_subplot(gs[0])
    if map_idx is not None:
        map_vals = denorm("map", map_idx) if "map" in norm_params else temporal[:, map_idx]
        if map_vals is not None:
            ax1.plot(hours[valid > 0], map_vals[valid > 0], "o-", color="#1976D2",
                     label="MAP (mmHg)", linewidth=1.5)
            ax1.axhline(y=65, color="red", linestyle="--", alpha=0.4, label="MAP=65 threshold")
    if hr_idx is not None:
        hr_vals = denorm("heart_rate", hr_idx) if "heart_rate" in norm_params else temporal[:, hr_idx]
        if hr_vals is not None:
            ax1t = ax1.twinx()
            ax1t.plot(hours[valid > 0], hr_vals[valid > 0], "s-", color="#FF9800",
                      label="HR (bpm)", linewidth=1, alpha=0.7)
            ax1t.set_ylabel("HR (bpm)", fontsize=9, color="#FF9800")
    ax1.set_ylabel("MAP (mmHg)", fontsize=9)
    _hf_d = {"hfpef": "HFpEF", "no_hf": "No HF", "hfref": "HFrEF",
              "hfmref": "HFmrEF", "hf_ef_unknown": "HF (EF unknown)"}
    _ph_d = {"shock_predominant": "Shock", "non_shock": "Non-shock"}
    ax1.set_title(f"Case: {patient_info['label']} | "
                  f"CATE={patient_info['cate']*100:+.1f}pp | "
                  f"HF={_hf_d.get(patient_info.get('hf_subtype',''), patient_info.get('hf_subtype',''))} | "
                  f"Pheno={_ph_d.get(patient_info.get('phenotype',''), patient_info.get('phenotype',''))} | "
                  f"Mortality={patient_info.get('mortality', '?')}",
                  fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.2)

    # Panel 2: Lactate + SpO2
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    if lac_idx is not None:
        lac_vals = denorm("lactate", lac_idx) if "lactate" in norm_params else temporal[:, lac_idx]
        if lac_vals is not None:
            ax2.plot(hours[valid > 0], lac_vals[valid > 0], "D-", color="#F44336",
                     label="Lactate (mmol/L)", linewidth=1.5)
            ax2.axhline(y=2.0, color="#F44336", linestyle="--", alpha=0.3, label="Lactate=2.0")
    if spo2_idx is not None:
        spo2_vals = denorm("spo2", spo2_idx) if "spo2" in norm_params else temporal[:, spo2_idx]
        if spo2_vals is not None:
            ax2t = ax2.twinx()
            ax2t.plot(hours[valid > 0], spo2_vals[valid > 0], "^-", color="#4CAF50",
                      label="SpO2 (%)", linewidth=1, alpha=0.7)
            ax2t.set_ylabel("SpO2 (%)", fontsize=9, color="#4CAF50")
    ax2.set_ylabel("Lactate (mmol/L)", fontsize=9)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(alpha=0.2)

    # Panel 3: Treatment actions comparison
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    width = 0.35
    obs_fluid, obs_vaso, obs_diur = [], [], []
    rule_fluid, rule_vaso, rule_diur = [], [], []

    for t in range(n_bins):
        if valid[t] > 0:
            of, ov, od = decode_action_components(int(obs_action_raw[t]))
            rf, rv, rd = decode_action_components(int(rule_action_raw[t]))
        else:
            of = ov = od = rf = rv = rd = 0
        obs_fluid.append(of)
        obs_vaso.append(ov)
        obs_diur.append(od)
        rule_fluid.append(rf)
        rule_vaso.append(rv)
        rule_diur.append(rd)

    x = hours
    # Stacked bar: fluid + vaso + diuretic
    ax3.bar(x - width / 2, obs_fluid, width, label="Obs: fluid", color="#90CAF9", alpha=0.8)
    ax3.bar(x - width / 2, obs_vaso, width, bottom=obs_fluid, label="Obs: vaso", color="#EF9A9A", alpha=0.8)
    ax3.bar(x + width / 2, rule_fluid, width, label="Rule: fluid", color="#1565C0", alpha=0.8)
    ax3.bar(x + width / 2, rule_vaso, width, bottom=rule_fluid, label="Rule: vaso", color="#C62828", alpha=0.8)

    # Mark diuretics with triangles
    for t in range(n_bins):
        if obs_diur[t] > 0:
            ax3.plot(hours[t] - width / 2, obs_fluid[t] + obs_vaso[t] + 0.15,
                     "v", color="#4CAF50", markersize=6)
        if rule_diur[t] > 0:
            ax3.plot(hours[t] + width / 2, rule_fluid[t] + rule_vaso[t] + 0.15,
                     "v", color="#2E7D32", markersize=6)

    # Mark divergence points
    for t in range(n_bins):
        if valid[t] > 0 and obs_action_raw[t] != rule_action_raw[t]:
            ax3.axvline(x=hours[t], color="orange", alpha=0.15, linewidth=8)

    ax3.set_ylabel("Action level", fontsize=9)
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(["none", "low/mod", "high"], fontsize=8)
    ax3.legend(fontsize=7, ncol=4, loc="upper right")
    ax3.grid(alpha=0.2, axis="y")

    # Panel 4: Agreement timeline
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    agreement = [1 if obs_action_raw[t] == rule_action_raw[t] else 0 for t in range(n_bins)]
    colors_agree = ["#4CAF50" if a else "#F44336" for a in agreement]
    ax4.bar(hours, [1] * n_bins, color=colors_agree, alpha=0.6, width=1.5)
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(["Disagree", "Agree"], fontsize=8)
    ax4.set_xlabel("Hours post-T0", fontsize=10)
    n_agree = sum(agreement)
    ax4.set_title(f"Policy agreement: {n_agree}/{n_bins} bins ({n_agree/n_bins*100:.0f}%)",
                  fontsize=9)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(str(output_path).replace('.png', '.pdf'), format='pdf',
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"    Saved: {output_path} (+pdf)")


def main():
    print("=" * 72)
    print("P5 — Counterfactual Trajectory Visualization")
    print("=" * 72)
    t0 = time.time()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 1. Load everything ──
    print("[1] Loading config and data...")
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    with open(NORM_PATH) as f:
        norm_cfg = yaml.safe_load(f)
    norm_params = norm_cfg.get("params", {})

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
        merge_map=merge_map, n_bins=config["grid"]["n_bins"],
    )

    pheno_df = pd.read_parquet(ROOT / config["data"]["phenotype_labels"])
    propensity_df_raw = pd.read_parquet(ROOT / config["data"]["propensities_coarse3"])
    rule_policy_df = pd.read_parquet(ROOT / "data" / "interim" / "70c_rule_policy_actions_v1_mimic.parquet")

    # ── 2. Load CATE from P2 ──
    print("[2] Loading P2 CATE distribution...")
    cate_path = TABLES_DIR / "P2_cate_distribution.csv"
    if not cate_path.exists():
        print("  ERROR: P2 CATE distribution not found. Run P2 first.")
        return
    cate_df = pd.read_csv(cate_path)

    # ── 3. Select representative patients ──
    print("[3] Selecting representative patients...")
    selections = select_representative_patients(cate_df, pheno_df)
    for s in selections:
        if s.get("_concordant_deferred"):
            print(f"  {s['label']}: (deferred — resolved after action agreement scan)")
        else:
            print(f"  {s['label']}: stay_id={s['stay_id']}, CATE={s['cate']*100:+.1f}pp, "
                  f"HF={s.get('hf_subtype','')}, pheno={s.get('phenotype','')}")

    # ── 3b. Resolve deferred concordant case ──
    n_bins = config["grid"]["n_bins"]
    bin_hours = config["grid"]["bin_hours"]
    sid_to_idx = {sid: i for i, sid in enumerate(test_ds.stay_ids)}

    for sel in selections:
        if sel.get("_concordant_deferred"):
            print("  Resolving concordant case by action agreement...")
            best_sid, best_agree, best_cate = None, -1, 0.0
            cate_lookup = dict(zip(cate_df["stay_id"], cate_df["cate"]))
            for sid in test_ids:
                obs_r, rule_r = extract_raw_action_trajectories(
                    propensity_df_raw, rule_policy_df, int(sid), n_bins)
                idx = sid_to_idx.get(sid)
                if idx is None:
                    continue
                valid = test_ds.valid_mask[idx]
                agree = sum(1 for t in range(n_bins)
                            if valid[t] > 0 and obs_r[t] == rule_r[t])
                if agree > best_agree:
                    best_agree = agree
                    best_sid = sid
                    best_cate = cate_lookup.get(int(sid), 0.0)
            if best_sid is not None:
                sel["stay_id"] = int(best_sid)
                sel["cate"] = best_cate
                sel.pop("_concordant_deferred", None)
                pheno_row = pheno_df[pheno_df["stay_id"] == int(best_sid)]
                if len(pheno_row) > 0:
                    sel["hf_subtype"] = pheno_row.iloc[0].get("hf_subtype")
                    sel["phenotype"] = pheno_row.iloc[0].get("rule_phenotype_2class")
                print(f"    Selected stay_id={best_sid}, agreement={best_agree}/{n_bins}, "
                      f"CATE={best_cate*100:+.1f}pp")

    # Remove any unresolved placeholders
    selections = [s for s in selections if s.get("stay_id") is not None]

    # ── 4. Get raw state data and generate plots ──
    print("[4] Generating trajectory plots...")

    case_summaries = []
    for i, sel in enumerate(selections):
        sid = sel["stay_id"]
        if sid not in sid_to_idx:
            print(f"    SKIP: stay_id {sid} not in test set")
            continue

        idx = sid_to_idx[sid]
        # Get raw action indices (pre-contiguous mapping)
        obs_raw, rule_raw = extract_raw_action_trajectories(propensity_df_raw, rule_policy_df, sid, n_bins)
        stay_data = {
            "temporal": test_ds.temporal[idx],
            "valid_mask": test_ds.valid_mask[idx],
            "action_raw": obs_raw,
            "rule_action_raw": rule_raw,
        }

        # Get mortality
        mortality = test_ds.mortality_28d[idx]
        sel["mortality"] = int(mortality)

        patient_info = {
            **sel,
            "temporal_cols": temporal_cols,
        }

        case_num = i + 1
        filename = f"P5_trajectory_case_{case_num}_{sel['label'].replace(' ', '_').replace('×', 'x').lower()}.png"
        plot_single_trajectory(
            stay_data, norm_params, patient_info,
            FIGURES_DIR / filename,
            n_bins=n_bins, bin_hours=bin_hours,
        )

        # Compute agreement stats
        agreement = sum(
            1 for t in range(n_bins)
            if stay_data["valid_mask"][t] > 0 and
            stay_data["action_raw"][t] == stay_data["rule_action_raw"][t]
        )
        valid_bins = sum(1 for t in range(n_bins) if stay_data["valid_mask"][t] > 0)

        case_summaries.append({
            "case": case_num,
            "label": sel["label"],
            "stay_id": sid,
            "cate_pp": sel["cate"] * 100,
            "hf_subtype": sel.get("hf_subtype", ""),
            "phenotype": sel.get("phenotype", ""),
            "mortality_28d": int(mortality),
            "agreement_bins": f"{agreement}/{valid_bins}",
        })

    # ── 5. Combined panel figure ──
    print("[5] Creating combined panel figure...")
    n_cases = len(case_summaries)
    if n_cases > 0:
        fig, axes = plt.subplots(n_cases, 1, figsize=(14, 3 * n_cases))
        if n_cases == 1:
            axes = [axes]

        for ax_i, cs in enumerate(case_summaries):
            ax = axes[ax_i]
            sid = cs["stay_id"]
            idx = sid_to_idx[sid]

            obs_raw, rule_raw = extract_raw_action_trajectories(propensity_df_raw, rule_policy_df, sid, n_bins)
            valid = test_ds.valid_mask[idx]
            hours = np.arange(n_bins) * bin_hours

            # Simple comparison bar chart
            # Fixed colors: light = observed, dark = rule-policy
            width = 0.35
            first_bar = True
            for t in range(n_bins):
                if valid[t] > 0:
                    kw_obs = {"label": "Observed"} if first_bar else {}
                    kw_rule = {"label": "Rule policy"} if first_bar else {}
                    ax.bar(hours[t] - width / 2, obs_raw[t], width,
                           color="#90CAF9", alpha=0.8, **kw_obs)
                    ax.bar(hours[t] + width / 2, rule_raw[t], width,
                           color="#C62828", alpha=0.8, **kw_rule)
                    first_bar = False
            if ax_i == 0:
                ax.legend(fontsize=7, loc="upper right")

            ax.set_ylabel("Action idx", fontsize=8)
            ax.set_title(f"Case {cs['case']}: {cs['label']} | CATE={cs['cate_pp']:+.1f}pp | "
                         f"Mort={cs['mortality_28d']} | Agree={cs['agreement_bins']}",
                         fontsize=9)
            if ax_i == n_cases - 1:
                ax.set_xlabel("Hours post-T0", fontsize=9)

        plt.tight_layout()
        fig_path = FIGURES_DIR / "P5_trajectory_panel.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.savefig(str(fig_path).replace('.png', '.pdf'), format='pdf',
                    bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        print(f"  Saved: {FIGURES_DIR / 'P5_trajectory_panel.png'}")

    # ── 6. Summary report ──
    print("[6] Generating summary report...")
    report_lines = [
        "=" * 72,
        "P5 — Counterfactual Trajectory Visualization",
        "=" * 72,
        f"Generated: {ts}",
        f"Protocol: v3.1-patch, Step P5",
        f"Database: MIMIC-IV",
        f"Test set: {len(test_ds)} stays",
        "",
        "SELECTED CASES",
        "─" * 14,
    ]
    for cs in case_summaries:
        report_lines.append(
            f"  Case {cs['case']}: {cs['label']}"
        )
        report_lines.append(
            f"    stay_id={cs['stay_id']}, CATE={cs['cate_pp']:+.1f}pp, "
            f"HF={cs['hf_subtype']}, pheno={cs['phenotype']}, "
            f"mortality={cs['mortality_28d']}, agreement={cs['agreement_bins']}"
        )

    report_lines.extend([
        "",
        "INTERPRETATION",
        "─" * 14,
        "  Each panel shows observed vs rule-policy treatment actions.",
        "  Yellow bands = divergence points where rule differs from observed.",
        "  Green/red bars = bins where policies agree/disagree.",
        "  CATE < 0 = predicted mortality benefit from switching to rule policy.",
        "",
        "MANUSCRIPT PLACEMENT",
        "─" * 20,
        "  Figure 3 or 4 (high-impact visual); makes abstract treatment effects",
        "  concrete and clinically interpretable.",
        "",
    ])

    with open(REPORTS_DIR / "P5_trajectory_summary.txt", "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Saved: {REPORTS_DIR / 'P5_trajectory_summary.txt'}")

    total_time = time.time() - t0
    print()
    print("=" * 72)
    print(f"P5 COMPLETE. Total time: {total_time / 60:.1f} min")
    print("=" * 72)


if __name__ == "__main__":
    main()
