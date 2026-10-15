#!/usr/bin/env python3
"""
P16a_sepsis_ards_iptw_subgroup — IPTW subgroup sensitivity (lightweight).

Purpose
-------
Test whether the 75a dynamic-MSM rule-policy ATE is consistent across acute-
illness driver subgroups defined by P11 (septic shock, sepsis-no-shock,
ARDS, other respiratory failure, neurocritical, trauma_hemorrhage,
post-op cardiac/non-cardiac, AKI primary, cardiogenic shock).

Reuses 75a's panel-building + rule-policy-weighting + Hájek WIS estimator
machinery via importlib. For each subgroup we report:

    n_stays · observed mortality · WIS rule-policy value · ESS · ATE
    bootstrap 95% CI on ATE

Companion to P16b (G-Transformer DM subgroup, heavyweight). P16a runs on
CPU in 5-10 minutes; if the directional sign is interesting per subgroup,
P16b confirms it under the primary G-Transformer model on GPU.

Inputs
------
- data/staging/P11_acute_illness_driver_refined_mimic.parquet
- (everything 75a needs to rebuild its panel — see 75a docstring)

Outputs
-------
- tables/P16a_subgroup_sensitivity_iptw.csv
- reports/P16a_subgroup_sensitivity_iptw_summary.txt
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path


# ── 75a-as-library loader ────────────────────────────────────────────
def _load_75a_module():
    spec = importlib.util.spec_from_file_location(
        "_step75a",
        Path(__file__).resolve().parent / "75a_dynamic_msm_iptw.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Subgroups: P11 binary flags worth probing in sensitivity.
SUBGROUP_FLAGS = [
    "is_cardiogenic_shock",
    "is_septic_shock",
    "is_sepsis_no_shock",
    "is_ards",
    "is_resp_failure_other",
    "is_neurocritical",
    "is_trauma",
    "is_hemorrhage",
    "is_aki_primary",
    "has_cardiac_procedure",
    "has_noncardiac_procedure_only",
]

# Exclusion sensitivities: drop the flagged stays and re-estimate.
EXCLUSION_SETS = [
    ("excl_ards",          ["is_ards"]),
    ("excl_severe_resp",   ["is_ards", "is_resp_failure_other"]),
    ("excl_septic_shock",  ["is_septic_shock"]),
]


def hajek_subgroup(
    panel: pd.DataFrame,
    mod75a,
    prob_cols: list[str],
    merge_map: dict,
    trim_lo: float,
    trim_hi: float,
    label: str,
) -> dict:
    """Run Hájek-WIS estimator on the supplied panel slice."""
    n_stays = panel["stay_id"].nunique()
    if n_stays < 30:
        return {"label": label, "n_stays": n_stays,
                "observed_mortality": np.nan, "wis_rule_value": np.nan,
                "wis_ess": np.nan, "ate_wis": np.nan,
                "bootstrap_ci_lo": np.nan, "bootstrap_ci_hi": np.nan,
                "skipped_reason": f"n<30 (n={n_stays})"}

    terminal = panel[panel["is_terminal"]].dropna(subset=["mortality_28d"])
    obs_value = float(terminal["mortality_28d"].mean())

    wis_t = mod75a.compute_policy_weights(panel, prob_cols, merge_map, verbose=False)
    wis_t = wis_t.join(
        terminal.set_index("stay_id")[["mortality_28d"]], how="inner",
    )
    if len(wis_t) <= 10:
        return {"label": label, "n_stays": n_stays,
                "observed_mortality": obs_value, "wis_rule_value": np.nan,
                "wis_ess": 0.0, "ate_wis": np.nan,
                "bootstrap_ci_lo": np.nan, "bootstrap_ci_hi": np.nan,
                "skipped_reason": f"weight rows<=10"}

    w_arr = wis_t["wis_raw"].to_numpy()
    lo_v = np.percentile(w_arr, trim_lo)
    hi_v = np.percentile(w_arr, trim_hi)
    wis_t["wis_trimmed"] = np.clip(w_arr, lo_v, hi_v)
    wis_res = mod75a.wis_estimator(wis_t["wis_trimmed"], wis_t["mortality_28d"])

    return {
        "label": label,
        "n_stays": n_stays,
        "observed_mortality": obs_value,
        "wis_rule_value": float(wis_res["value"]),
        "wis_ess": float(wis_res["ess"]),
        "ate_wis": float(wis_res["value"] - obs_value),
        "skipped_reason": "",
    }


def bootstrap_ate(
    panel: pd.DataFrame,
    mod75a,
    prob_cols: list[str],
    merge_map: dict,
    trim_lo: float,
    trim_hi: float,
    n_boot: int = 200,
    seed: int = 20260510,
) -> tuple[float, float]:
    """Stay-level cluster bootstrap of ATE = WIS - observed."""
    stays = panel["stay_id"].unique()
    if len(stays) < 30:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    ates = []
    try:
        from tqdm import tqdm
        outer = tqdm(range(n_boot), desc="    bootstrap", unit="rep")
    except ImportError:
        outer = range(n_boot)
    for _ in outer:
        boot_stays = rng.choice(stays, size=len(stays), replace=True)
        boot_counts = pd.Series(boot_stays).value_counts()
        boot_panel = panel[panel["stay_id"].isin(boot_counts.index)]
        # Replicate stays per resample count for proper bootstrap weighting.
        rep_idx = np.repeat(
            np.arange(len(boot_counts)), boot_counts.values,
        )
        order = pd.DataFrame({
            "stay_id": boot_counts.index.values[rep_idx],
            "_rep": np.arange(len(rep_idx)),
        })
        bp = boot_panel.merge(order, on="stay_id", how="left")
        bp["stay_id"] = bp["stay_id"].astype(str) + "_b" + bp["_rep"].astype(str)
        try:
            res = hajek_subgroup(bp, mod75a, prob_cols, merge_map,
                                 trim_lo, trim_hi, "boot")
            if not np.isnan(res["ate_wis"]):
                ates.append(res["ate_wis"])
        except (ValueError, KeyError):
            continue
    if len(ates) < n_boot * 0.5:
        return np.nan, np.nan
    arr = np.array(ates)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def main() -> None:
    cfg = load_config()
    staging_dir = resolve_path(cfg, "staging")
    reports_dir = resolve_path(cfg, "reports")
    tables_dir = resolve_path(cfg, "tables")
    for d in (reports_dir, tables_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("P16a_sepsis_ards_iptw_subgroup — subgroup IPTW sensitivity")
    print("=" * 70)

    print("\n[1/5] Loading 75a as a library + rebuilding panel ...")
    mod75a = _load_75a_module()
    data = mod75a.load_all_data(cfg)
    panel = mod75a.build_panel(data)
    numer_probs = mod75a.fit_numerator_model(panel)
    panel = mod75a.compute_stabilised_weights(panel, numer_probs)
    primary_trim = data["trim_cfg"]["weight_trimming_grid"]["primary"]
    trim_lo = primary_trim["lo"]
    trim_hi = primary_trim["hi"]
    panel = mod75a.apply_trimming(panel, trim_lo, trim_hi)
    prob_cols = data["prob_cols"]
    merge_map = data["merge_map"]
    print(f"  Panel: {len(panel):,} rows · {panel['stay_id'].nunique():,} stays")

    print("\n[2/5] Joining P11 subgroup flags ...")
    p11 = pd.read_parquet(
        staging_dir / "P11_acute_illness_driver_refined_mimic.parquet",
        columns=["stay_id"] + SUBGROUP_FLAGS,
    )
    panel = panel.merge(p11, on="stay_id", how="left")
    for c in SUBGROUP_FLAGS:
        panel[c] = panel[c].fillna(False).astype(bool)

    print("\n[3/5] Per-subgroup Hájek WIS + bootstrap CI ...")
    rows = []
    for flag in SUBGROUP_FLAGS:
        # Affirmative slice (flag == True).
        sub = panel[panel[flag]]
        res = hajek_subgroup(sub, mod75a, prob_cols, merge_map,
                              trim_lo, trim_hi, label=f"{flag}=True")
        if res["n_stays"] >= 30:
            ci_lo, ci_hi = bootstrap_ate(sub, mod75a, prob_cols, merge_map,
                                          trim_lo, trim_hi)
            res["bootstrap_ci_lo"] = ci_lo
            res["bootstrap_ci_hi"] = ci_hi
        else:
            res["bootstrap_ci_lo"] = np.nan
            res["bootstrap_ci_hi"] = np.nan
        print(f"  {flag}=True: {res}")
        rows.append(res)

    print("\n[4/5] Exclusion-sensitivity slices ...")
    for label, drop_flags in EXCLUSION_SETS:
        keep_mask = ~panel[drop_flags].any(axis=1)
        sub = panel[keep_mask]
        res = hajek_subgroup(sub, mod75a, prob_cols, merge_map,
                              trim_lo, trim_hi, label=label)
        if res["n_stays"] >= 30:
            ci_lo, ci_hi = bootstrap_ate(sub, mod75a, prob_cols, merge_map,
                                          trim_lo, trim_hi)
            res["bootstrap_ci_lo"] = ci_lo
            res["bootstrap_ci_hi"] = ci_hi
        else:
            res["bootstrap_ci_lo"] = np.nan
            res["bootstrap_ci_hi"] = np.nan
        print(f"  {label}: {res}")
        rows.append(res)

    print("\n[5/5] Saving outputs ...")
    df = pd.DataFrame(rows)
    table_path = tables_dir / "P16a_subgroup_sensitivity_iptw.csv"
    df.to_csv(table_path, index=False)
    print(f"  → {table_path}")

    # Sort for readability.
    df_sorted = df.sort_values(["n_stays"], ascending=False)
    report = textwrap.dedent(f"""\
    ================================================================
    P16a_sepsis_ards_iptw_subgroup — Summary Report
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ================================================================

    SUBGROUP TABLE
    ───────────────
    {textwrap.indent(df_sorted.to_string(index=False), '  ')}

    INTERPRETATION GUIDE
    ─────────────────────
    - ate_wis < 0 → rule policy associated with lower mortality vs observed
      in that subgroup. Compare the 95 % CI to zero.
    - Subgroups with low ESS (<300) are noisy; treat as exploratory only.
    - The exclusion-sensitivity rows (excl_*) re-estimate the cohort-wide
      ATE after dropping stays in the named flag(s). Stable ATE under
      exclusion supports robustness of the primary 75a result.
    - Heavy-tail subgroups whose sign disagrees with the cohort-wide
      result should be confirmed under the G-Transformer DM (see P16b).

    OUTPUT
    ──────
    Table: {table_path}
    """)
    report_path = reports_dir / "P16a_subgroup_sensitivity_iptw_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"  → Report: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
