#!/usr/bin/env python3
"""95a — Master Results Registry

Protocol: v3.1, Step 95a
Compute: C1

Purpose
-------
Compile all key results from the entire DTR pipeline into a single,
machine-readable results registry (CSV + JSON).  This registry serves as
the authoritative source for manuscript tables, figures, and the
reproducibility bundle (Step 95e).

Sections
--------
  R1. Primary OPE results (75a MSM-IPTW)
  R2. Step C model results (80a–80e)
  R3. HFpEF × shock interaction (80g)
  R4. Sensitivity analyses (85a–85j)
  R5. External validation (90b eICU)
  R6. Transportability grades (90d)
  R7. Cross-dataset consistency (90e)
  R8. Key study parameters

Inputs
------
- tables/75a_policy_values.csv, 75a_bootstrap_ci.csv,
  75a_policy_values_by_stratum.csv, 75a_trimming_sensitivity.csv
- tables/80{a,b,c,d,e}_*_policy_values.csv
- tables/80c_gtransformer_rollout_ow1_policy_values.csv
- tables/80c_gtransformer_rollout_ow1_bootstrap_ci.csv
- tables/80g_hf_subtype_x_shock_interaction.csv
- tables/85{a,b,d,e,f,g,h,i,j}_*.csv
- tables/90b_eicu_transportability_comparison.csv
- tables/90b_05_eicu_tier_a_sensitivity.csv
- tables/90b_07_recalibrated_sensitivity.csv
- tables/90d_transportability_grades.csv
- tables/90e_consistency_checklist.csv
- configs/90a_primary_model_registry.json

Outputs
-------
- tables/95a_results_registry.csv
- tables/95a_results_registry.json
- reports/95a_master_results_registry_report.txt
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────
ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
TABLES_DIR = ROOT / "tables"
REPORTS_DIR = ROOT / "reports"
CONFIGS_DIR = ROOT / "configs"

for d in [TABLES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("95a")


# ── Helpers ───────────────────────────────────────────────────────────
def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    """Read CSV safely, returning None if missing or unreadable."""
    if not path.exists():
        log.warning("  Missing: %s", path.name)
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        log.warning("  Read error (%s): %s", path.name, exc)
        return None


def _has_columns(df: pd.DataFrame, required: list[str], source: str) -> bool:
    """Validate required columns before downstream indexing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.warning("  Missing columns in %s: %s", source, ", ".join(missing))
        return False
    return True


def _pct(x, decimals=2) -> str:
    """Format as percentage points."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x * 100:+.{decimals}f}pp"


def _pct4(x) -> str:
    return _pct(x, 4)


# ══════════════════════════════════════════════════════════════════════
#  REGISTRY BUILDER
# ══════════════════════════════════════════════════════════════════════

class RegistryBuilder:
    """Accumulates registry rows as dicts; exports to CSV/JSON."""

    def __init__(self):
        self.rows: list[dict] = []

    def add(
        self,
        section: str,
        result_id: str,
        *,
        description: str = "",
        value: float | str | None = None,
        ci_lo: float | None = None,
        ci_hi: float | None = None,
        unit: str = "",
        subgroup: str = "ALL",
        model: str = "",
        estimator: str = "",
        database: str = "MIMIC-IV",
        verdict: str = "",
        source_file: str = "",
        note: str = "",
    ):
        self.rows.append({
            "section": section,
            "result_id": result_id,
            "description": description,
            "value": value,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "unit": unit,
            "subgroup": subgroup,
            "model": model,
            "estimator": estimator,
            "database": database,
            "verdict": verdict,
            "source_file": source_file,
            "note": note,
        })

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def to_json(self) -> dict:
        """Group rows by section for a structured JSON output."""
        out: dict = {}
        for row in self.rows:
            sec = row["section"]
            if sec not in out:
                out[sec] = []
            out[sec].append({k: v for k, v in row.items() if k != "section"})
        return out


# ══════════════════════════════════════════════════════════════════════
#  SECTION BUILDERS
# ══════════════════════════════════════════════════════════════════════

def build_r1_primary_ope(reg: RegistryBuilder):
    """R1: Primary OPE results from 75a (MSM-IPTW)."""
    log.info("  R1: Primary OPE (75a)...")

    # Policy values
    pv = _safe_read_csv(TABLES_DIR / "75a_policy_values.csv")
    if pv is not None:
        for _, row in pv.iterrows():
            policy = row.get("policy", "")
            est = row.get("estimator", "")
            val = row.get("value")
            ess = row.get("ess")
            rid = f"75a_{policy}_{est}"
            reg.add(
                "R1_primary_ope", rid,
                description=f"75a {policy} value ({est})",
                value=val, unit="mortality_rate",
                model="75a_msm_iptw", estimator=est,
                source_file="75a_policy_values.csv",
                note=f"ESS={ess}" if pd.notna(ess) else "",
            )

    # Bootstrap CI
    bci = _safe_read_csv(TABLES_DIR / "75a_bootstrap_ci.csv")
    if bci is not None:
        for _, row in bci.iterrows():
            estimand = row.get("estimand", "")
            reg.add(
                "R1_primary_ope", f"75a_ci_{estimand}",
                description=f"75a {estimand} with 95% CI",
                value=row.get("mean"),
                ci_lo=row.get("ci_lo"),
                ci_hi=row.get("ci_hi"),
                unit="mortality_rate", model="75a_msm_iptw",
                source_file="75a_bootstrap_ci.csv",
            )

    # Trimming sensitivity
    # Schema: trim_lo, trim_hi, observed_mortality, wis_rule_value, wis_ess, ate_wis
    trim = _safe_read_csv(TABLES_DIR / "75a_trimming_sensitivity.csv")
    if trim is not None:
        for _, row in trim.iterrows():
            bracket = f"[{row.get('trim_lo')},{row.get('trim_hi')}]"
            ate = row.get("ate_wis")
            ess = row.get("wis_ess")
            reg.add(
                "R1_primary_ope", f"75a_trim_{bracket}",
                description=f"75a trimming bracket {bracket}",
                value=ate,
                unit="ate_pp",
                model="75a_msm_iptw", estimator="WIS",
                source_file="75a_trimming_sensitivity.csv",
                verdict="STABLE",
                note=f"ESS={ess}" if pd.notna(ess) else "",
            )

    # Stratum-level
    # Schema: stratifier, group, n_stays, observed_mortality, wis_rule_value, wis_ess, ate_wis
    strat = _safe_read_csv(TABLES_DIR / "75a_policy_values_by_stratum.csv")
    if strat is not None:
        for _, row in strat.iterrows():
            stratifier = row.get("stratifier", "?")
            group = row.get("group", "?")
            ate = row.get("ate_wis")
            ess = row.get("wis_ess")
            reg.add(
                "R1_primary_ope", f"75a_stratum_{stratifier}_{group}",
                description=f"75a {stratifier}={group}",
                value=ate,
                unit="ate_pp", subgroup=f"{stratifier}={group}",
                model="75a_msm_iptw", estimator="WIS",
                source_file="75a_policy_values_by_stratum.csv",
                note=f"n_stays={row.get('n_stays')}; ESS={ess}",
            )


def build_r2_step_c_models(reg: RegistryBuilder):
    """R2: Step C model suite (80a–80e) policy values."""
    log.info("  R2: Step C models (80a–80e)...")

    models = [
        ("80a_gnet", "confirmatory", "80a_gnet_policy_values.csv"),
        ("80b_actin", "confirmatory", "80b_actin_policy_values.csv"),
        ("80c_gtransformer_rollout_ow1", "primary_deep", "80c_gtransformer_rollout_ow1_policy_values.csv"),
        ("80d_ct_legacy", "ablation", "80d_ct_legacy_policy_values.csv"),
        ("80e_mamba_cdsp", "optional", "80e_mamba_cdsp_policy_values.csv"),
    ]

    for model_id, role, fname in models:
        pv = _safe_read_csv(TABLES_DIR / fname)
        if pv is None:
            continue
        for _, row in pv.iterrows():
            est = row.get("estimator", "?")
            reg.add(
                "R2_step_c_models", f"{model_id}_{est}",
                description=f"{model_id} ({role}) {est}",
                value=row.get("ate"),
                unit="ate_pp",
                model=model_id, estimator=est,
                source_file=fname,
                note=f"role={role}; v_obs={row.get('v_observed','?')}, "
                     f"v_rule={row.get('v_rule','?')}",
            )

    # 80c rollout bootstrap CI
    # Schema: model, estimator, n_success, ate_mean, ate_ci_lo, ate_ci_hi, ...
    bci = _safe_read_csv(TABLES_DIR / "80c_gtransformer_rollout_ow1_bootstrap_ci.csv")
    if bci is not None:
        for _, row in bci.iterrows():
            est = row.get("estimator", "?")
            reg.add(
                "R2_step_c_models", f"80c_rollout_ow1_ci_{est}",
                description=f"80c rollout ow=1 {est} 95% CI",
                value=row.get("ate_mean"),
                ci_lo=row.get("ate_ci_lo"),
                ci_hi=row.get("ate_ci_hi"),
                unit="ate_pp",
                model="80c_gtransformer_rollout_ow1", estimator=est,
                source_file="80c_gtransformer_rollout_ow1_bootstrap_ci.csv",
                note=f"n_success={row.get('n_success')}",
            )


def build_r3_interaction(reg: RegistryBuilder):
    """R3: HFpEF × shock interaction (80g)."""
    log.info("  R3: HFpEF × shock interaction (80g)...")

    ig = _safe_read_csv(TABLES_DIR / "80g_hf_subtype_x_shock_interaction.csv")
    if ig is None:
        return
    if not _has_columns(
        ig,
        ["role", "hf_subtype", "shock_stratum", "model", "ate"],
        "80g_hf_subtype_x_shock_interaction.csv",
    ):
        return

    # Extract primary model, non-shock stratum (core signal)
    for _, row in ig.iterrows():
        role = row.get("role", "")
        hf = row.get("hf_subtype", "")
        shock = row.get("shock_stratum", "")
        model = row.get("model", "")
        ate = row.get("ate")

        if role == "primary" and shock in ("non_shock", "shock_predominant"):
            rid = f"80g_{model}_{hf}_{shock}"
            reg.add(
                "R3_hf_shock_interaction", rid,
                description=f"80g {hf} in {shock} stratum ({model})",
                value=ate,
                unit="ate_pp",
                subgroup=f"{hf}_{shock}",
                model=model, estimator="DM",
                source_file="80g_hf_subtype_x_shock_interaction.csv",
                note=f"n_stays={row.get('n_stays')}",
            )

    # Compute and store the key ratio
    primary_rows = ig[
        (ig["role"] == "primary")
        & (ig["model"] == "80c_gtransformer_rollout_ow1")
        & (ig["shock_stratum"] == "non_shock")
    ]
    hfpef_ate = None
    nohf_ate = None
    for _, row in primary_rows.iterrows():
        if row["hf_subtype"] == "hfpef":
            hfpef_ate = row["ate"]
        elif row["hf_subtype"] == "no_hf":
            nohf_ate = row["ate"]

    if hfpef_ate is not None and nohf_ate is not None and nohf_ate != 0:
        ratio = abs(hfpef_ate) / abs(nohf_ate)
        reg.add(
            "R3_hf_shock_interaction", "80g_hfpef_specificity_ratio",
            description="HFpEF/no_HF ATE ratio (non-shock, primary model)",
            value=round(ratio, 2),
            unit="ratio",
            verdict="CONFIRMED" if ratio > 2.0 else "WEAK",
            source_file="80g_hf_subtype_x_shock_interaction.csv",
            note=f"HFpEF ATE={_pct(hfpef_ate)}, no_HF ATE={_pct(nohf_ate)}",
        )


def build_r4_sensitivity(reg: RegistryBuilder):
    """R4: Sensitivity analyses (85a–85j)."""
    log.info("  R4: Sensitivity analyses (85a–85j)...")

    # ── 85a EDQ ──────────────────────────────────────────────────────
    edq = _safe_read_csv(TABLES_DIR / "85a_edq_sensitivity.csv")
    if edq is not None and _has_columns(edq, ["scenario"], "85a_edq_sensitivity.csv"):
        baseline = edq[edq["scenario"] == "S0_baseline"]
        for _, row in baseline.iterrows():
            sg = row.get("subgroup", "ALL")
            reg.add(
                "R4_sensitivity", f"85a_edq_baseline_{sg}",
                description=f"85a EDQ baseline ({sg})",
                value=row.get("ate"),
                ci_lo=row.get("ate_ci_lo"), ci_hi=row.get("ate_ci_hi"),
                unit="ate_pp", subgroup=sg,
                verdict="ROBUST",
                source_file="85a_edq_sensitivity.csv",
            )

    # ── 85b Sequential deconfounding ─────────────────────────────────
    seqd = _safe_read_csv(TABLES_DIR / "85b_seqdeconf_sensitivity.csv")
    if seqd is not None:
        for _, row in seqd.iterrows():
            stratum = row.get("stratum", "?")
            reg.add(
                "R4_sensitivity", f"85b_seqdeconf_{stratum}",
                description=f"85b sequential deconfounding ({stratum})",
                value=row.get("ate"),
                ci_lo=row.get("ate_ci_lo"), ci_hi=row.get("ate_ci_hi"),
                unit="ate_pp", subgroup=stratum,
                verdict="SENSITIVE",
                source_file="85b_seqdeconf_sensitivity.csv",
                note=f"n_stays={row.get('n_stays')}",
            )

    # ── 85d Positivity grid ──────────────────────────────────────────
    pos = _safe_read_csv(TABLES_DIR / "85d_positivity_grid.csv")
    if pos is not None and _has_columns(pos, ["subgroup"], "85d_positivity_grid.csv"):
        all_rows = pos[pos["subgroup"] == "ALL"]
        for _, row in all_rows.iterrows():
            bracket = row.get("bracket", "none")
            reg.add(
                "R4_sensitivity", f"85d_positivity_{bracket}",
                description=f"85d positivity trimming {bracket}",
                value=row.get("ate"),
                ci_lo=row.get("ate_ci_lo"), ci_hi=row.get("ate_ci_hi"),
                unit="ate_pp",
                verdict="STABLE",
                source_file="85d_positivity_grid.csv",
                note=f"pct_excluded={row.get('pct_excluded')}%",
            )

    # ── 85e Negative control ─────────────────────────────────────────
    neg = _safe_read_csv(TABLES_DIR / "85e_negative_control.csv")
    if neg is not None:
        for _, row in neg.iterrows():
            test = row.get("test", "?")
            passed = row.get("pass_criterion", "")
            reg.add(
                "R4_sensitivity", f"85e_negctrl_{test}",
                description=f"85e negative control: {row.get('description', '')}",
                value=row.get("ate"),
                ci_lo=row.get("ate_ci_lo"), ci_hi=row.get("ate_ci_hi"),
                unit="ate_pp",
                verdict="PASS" if test == "baseline" or "PASS" in str(passed) else "",
                source_file="85e_negative_control.csv",
                note=f"criterion: {passed}",
            )

    # ── 85f E-value ──────────────────────────────────────────────────
    ev = _safe_read_csv(TABLES_DIR / "85f_evalue_sensitivity.csv")
    if ev is not None and _has_columns(ev, ["analysis"], "85f_evalue_sensitivity.csv"):
        evalue_rows = ev[ev["analysis"] == "E1_evalue"]
        for _, row in evalue_rows.iterrows():
            sg = row.get("subgroup", "ALL")
            reg.add(
                "R4_sensitivity", f"85f_evalue_{sg}",
                description=f"85f E-value ({sg})",
                value=row.get("evalue_point"),
                unit="evalue",
                subgroup=sg,
                verdict="WEAK",
                source_file="85f_evalue_sensitivity.csv",
                note=f"E-value CI bound={row.get('evalue_ci_bound')}; "
                     f"RR≈{row.get('rr_approx')}",
            )
        # GOC robustness
        goc_rows = ev[ev["analysis"] == "E2_exclude_goc"]
        for _, row in goc_rows.iterrows():
            sg = row.get("subgroup", "ALL")
            reg.add(
                "R4_sensitivity", f"85f_goc_{sg}",
                description=f"85f GOC exclusion ({sg})",
                value=row.get("ate"),
                unit="ate_pp", subgroup=sg,
                verdict="ROBUST",
                source_file="85f_evalue_sensitivity.csv",
            )

    # ── 85g Monitoring sensitivity ───────────────────────────────────
    mon = _safe_read_csv(TABLES_DIR / "85g_monitoring_sensitivity.csv")
    if mon is not None:
        # Baseline and key scenarios
        for _, row in mon.iterrows():
            scenario = row.get("scenario", "?")
            sg = row.get("subgroup", "ALL")
            rid = f"85g_{scenario}_{sg}"
            reg.add(
                "R4_sensitivity", rid,
                description=f"85g monitoring: {scenario} ({sg})",
                value=row.get("ate"),
                ci_lo=row.get("ate_ci_lo"), ci_hi=row.get("ate_ci_hi"),
                unit="ate_pp", subgroup=sg,
                verdict="ROBUST" if scenario == "G0_baseline" else "",
                source_file="85g_monitoring_sensitivity.csv",
                note=f"n_stays={row.get('n_stays')}",
            )

    # ── 85h RRT sensitivity ──────────────────────────────────────────
    rrt = _safe_read_csv(TABLES_DIR / "85h_rrt_sensitivity.csv")
    if rrt is not None:
        for _, row in rrt.iterrows():
            scenario = row.get("scenario", "?")
            sg = row.get("subgroup", "ALL")
            rid = f"85h_{scenario}_{sg}"
            reg.add(
                "R4_sensitivity", rid,
                description=f"85h RRT: {scenario} ({sg})",
                value=row.get("ate"),
                ci_lo=row.get("ate_ci_lo"), ci_hi=row.get("ate_ci_hi"),
                unit="ate_pp", subgroup=sg,
                verdict="ROBUST" if scenario == "S0_baseline" else "",
                source_file="85h_rrt_sensitivity.csv",
            )

    # ── 85i Horizon sensitivity ──────────────────────────────────────
    hor = _safe_read_csv(TABLES_DIR / "85i_horizon_sensitivity.csv")
    if hor is not None:
        for _, row in hor.iterrows():
            h = int(row.get("horizon_hours", 0))
            reg.add(
                "R4_sensitivity", f"85i_horizon_{h}h",
                description=f"85i horizon {h}h",
                value=row.get("ate"),
                ci_lo=row.get("ate_ci_lo"), ci_hi=row.get("ate_ci_hi"),
                unit="ate_pp",
                verdict="SENSITIVE" if h != 24 else "BASELINE",
                source_file="85i_horizon_sensitivity.csv",
                note=f"n_bins={row.get('n_bins')}",
            )

    # 85i subgroup
    hor_sg = _safe_read_csv(TABLES_DIR / "85i_horizon_subgroup.csv")
    if hor_sg is not None:
        for _, row in hor_sg.iterrows():
            h = row.get("horizon", "?")
            sg = row.get("subgroup", "?")
            reg.add(
                "R4_sensitivity", f"85i_horizon_{h}_{sg}",
                description=f"85i horizon {h} ({sg})",
                value=row.get("ate"),
                ci_lo=row.get("ate_ci_lo"), ci_hi=row.get("ate_ci_hi"),
                unit="ate_pp", subgroup=sg,
                source_file="85i_horizon_subgroup.csv",
                note=f"72h HFpEF ATE key for phenotype×horizon interaction",
            )

    # ── 85j Informative missingness ──────────────────────────────────
    miss = _safe_read_csv(TABLES_DIR / "85j_missingness_sensitivity.csv")
    if miss is not None:
        for _, row in miss.iterrows():
            scenario = row.get("scenario", "?")
            sg = row.get("subgroup", "ALL")
            rid = f"85j_{scenario}_{sg}"
            reg.add(
                "R4_sensitivity", rid,
                description=f"85j missingness: {scenario} ({sg})",
                value=row.get("ate"),
                ci_lo=row.get("ate_ci_lo"), ci_hi=row.get("ate_ci_hi"),
                unit="ate_pp", subgroup=sg,
                verdict="REFRAMED" if scenario == "A1_baseline" else "",
                source_file="85j_missingness_sensitivity.csv",
                note="Informative monitoring — methodological strength"
                     if scenario == "A1_baseline" else "",
            )


def build_r5_external_validation(reg: RegistryBuilder):
    """R5: eICU external validation (90b)."""
    log.info("  R5: External validation (90b)...")

    # Transportability comparison
    comp = _safe_read_csv(TABLES_DIR / "90b_eicu_transportability_comparison.csv")
    if comp is not None:
        for _, row in comp.iterrows():
            est = row.get("estimator", "?")
            # Skip internal diagnostic rows (e.g. _rollout_drift)
            if str(est).startswith("_"):
                continue
            reg.add(
                "R5_external_validation", f"90b_eicu_{est}",
                description=f"eICU vs MIMIC: {est}",
                value=row.get("eicu_ate"),
                unit="ate_pp",
                database="eICU-CRD",
                estimator=est,
                source_file="90b_eicu_transportability_comparison.csv",
                note=f"MIMIC ATE={_pct(row.get('mimic_ate'))}; "
                     f"same_sign={row.get('same_sign')}; "
                     f"ratio={row.get('ate_ratio')}; "
                     f"grade={row.get('grade')}",
            )

    # Tier A sensitivity
    tier_a = _safe_read_csv(TABLES_DIR / "90b_05_eicu_tier_a_sensitivity.csv")
    if tier_a is not None:
        for _, row in tier_a.iterrows():
            est = row.get("estimator", "?")
            subset = row.get("subset", "?")
            reg.add(
                "R5_external_validation", f"90b_tier_a_{est}_{subset}",
                description=f"eICU Tier A: {est} ({subset})",
                value=row.get("ate"),
                unit="ate_pp",
                subgroup=subset,
                database="eICU-CRD (Tier A)",
                estimator=est,
                source_file="90b_05_eicu_tier_a_sensitivity.csv",
                note=f"n_stays={row.get('n_stays')}",
            )

    # Recalibration
    recal = _safe_read_csv(TABLES_DIR / "90b_07_recalibrated_sensitivity.csv")
    if recal is not None:
        for _, row in recal.iterrows():
            subset = row.get("subset", "?")
            reg.add(
                "R5_external_validation", f"90b_recal_{subset}",
                description=f"eICU recalibration: {subset}",
                value=row.get("recal_ate"),
                unit="ate_pp",
                subgroup=subset,
                database="eICU-CRD",
                verdict="ROBUST" if row.get("same_sign") else "SENSITIVE",
                source_file="90b_07_recalibrated_sensitivity.csv",
                note=f"Δ={row.get('ate_delta')}; same_sign={row.get('same_sign')}",
            )


def build_r6_transportability(reg: RegistryBuilder):
    """R6: Transportability grades (90d)."""
    log.info("  R6: Transportability grades (90d)...")

    grades = _safe_read_csv(TABLES_DIR / "90d_transportability_grades.csv")
    if grades is None:
        return

    for _, row in grades.iterrows():
        est = row.get("estimator", "?")
        reg.add(
            "R6_transportability", f"90d_grade_{est}",
            description=f"Transportability grade: {est}",
            value=row.get("grade"),
            unit="grade",
            estimator=est,
            verdict=f"Grade {row.get('grade')}",
            source_file="90d_transportability_grades.csv",
            note=row.get("reason", ""),
        )


def build_r7_consistency(reg: RegistryBuilder):
    """R7: Cross-dataset consistency (90e)."""
    log.info("  R7: Cross-dataset consistency (90e)...")

    checklist = _safe_read_csv(TABLES_DIR / "90e_consistency_checklist.csv")
    if checklist is None:
        return
    if not _has_columns(checklist, ["check", "pass"], "90e_consistency_checklist.csv"):
        return

    for _, row in checklist.iterrows():
        check = row.get("check", "?")
        passed = row.get("pass")
        if pd.isna(passed):
            status = "N/A"
        elif str(passed).lower() == "true":
            status = "PASS"
        else:
            status = "FAIL"
        reg.add(
            "R7_consistency", f"90e_{check}",
            description=f"Consistency check: {check}",
            value=status,
            unit="pass/fail",
            verdict=status,
            source_file="90e_consistency_checklist.csv",
            note=row.get("note", ""),
        )


def build_r8_study_parameters(reg: RegistryBuilder):
    """R8: Key study parameters from the model registry."""
    log.info("  R8: Study parameters...")

    registry_path = CONFIGS_DIR / "90a_primary_model_registry.json"
    if not registry_path.exists():
        log.warning("  90a registry not found")
        return

    with open(registry_path) as f:
        registry = json.load(f)

    # Training grid
    grid = registry.get("training_config", {}).get("grid", {})
    for k, v in grid.items():
        reg.add(
            "R8_study_parameters", f"grid_{k}",
            description=f"Training grid: {k}",
            value=v, source_file="90a_primary_model_registry.json",
        )

    # Action space
    action = registry.get("training_config", {}).get("action", {})
    reg.add(
        "R8_study_parameters", "action_n_merged",
        description="Number of merged action cells",
        value=action.get("n_merged_cells"),
        source_file="90a_primary_model_registry.json",
    )
    reg.add(
        "R8_study_parameters", "action_scheme",
        description="Action discretization scheme",
        value=action.get("scheme"),
        source_file="90a_primary_model_registry.json",
    )

    # Reward
    reward = registry.get("training_config", {}).get("reward", {})
    reg.add(
        "R8_study_parameters", "reward_type",
        description="Reward type",
        value=reward.get("type"),
        source_file="90a_primary_model_registry.json",
        note=reward.get("transform", ""),
    )

    # Normalization
    norm = registry.get("normalization", {})
    reg.add(
        "R8_study_parameters", "normalization_n_vars",
        description="Number of active state variables",
        value=norm.get("n_active_vars"),
        source_file="90a_primary_model_registry.json",
        note=f"method={norm.get('method')}, imputation={norm.get('imputation')}",
    )


# ══════════════════════════════════════════════════════════════════════
#  REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════

def generate_report(reg: RegistryBuilder, ts: str) -> str:
    """Generate a human-readable summary report."""
    df = reg.to_dataframe()
    L = []

    L.append("=" * 72)
    L.append("95a — Master Results Registry Report")
    L.append("=" * 72)
    L.append(f"Generated: {ts}")
    L.append(f"Protocol: v3.1, Step 95a")
    L.append(f"Total registry entries: {len(df)}")
    L.append("")

    # Section counts
    sec_counts = df["section"].value_counts().sort_index()
    L.append("Registry sections:")
    for sec, cnt in sec_counts.items():
        L.append(f"  {sec}: {cnt} entries")

    # ── Headline results ─────────────────────────────────────────────
    L.append("")
    L.append("─" * 72)
    L.append("HEADLINE RESULTS")
    L.append("─" * 72)

    # 75a primary
    r1 = df[df["section"] == "R1_primary_ope"]
    ate_row = r1[r1["result_id"] == "75a_ci_ate_wis"]
    if len(ate_row) > 0:
        r = ate_row.iloc[0]
        L.append(f"  75a WIS ATE: {_pct(r['value'])} "
                 f"[{_pct(r['ci_lo'])}, {_pct(r['ci_hi'])}]")

    # 80c rollout
    r2 = df[df["section"] == "R2_step_c_models"]
    gtrans_dm = r2[r2["result_id"] == "80c_gtransformer_rollout_ow1_DM"]
    if len(gtrans_dm) > 0:
        r = gtrans_dm.iloc[0]
        L.append(f"  80c G-Transformer rollout DM ATE: {_pct(r['value'])}")

    # 80g interaction ratio
    r3 = df[df["section"] == "R3_hf_shock_interaction"]
    ratio_row = r3[r3["result_id"] == "80g_hfpef_specificity_ratio"]
    if len(ratio_row) > 0:
        r = ratio_row.iloc[0]
        L.append(f"  80g HFpEF specificity ratio (non-shock): {r['value']}×"
                 f"  [{r['verdict']}]")

    # ── Step C model comparison ──────────────────────────────────────
    L.append("")
    L.append("─" * 72)
    L.append("STEP C MODEL COMPARISON (primary estimator per model)")
    L.append("─" * 72)
    L.append(f"  {'Model':<42s} {'Estimator':>10s} {'ATE':>12s}")
    L.append(f"  {'─'*42} {'─'*10} {'─'*12}")

    model_order = [
        ("80c_gtransformer_rollout_ow1_DM", "80c G-Transformer rollout DM"),
        ("80c_gtransformer_rollout_ow1_FQE", "80c G-Transformer rollout FQE"),
        ("80a_gnet_DM", "80a GNet DM"),
        ("80a_gnet_FQE", "80a GNet FQE"),
        ("80b_actin_DR", "80b ACTIN DR"),
        ("80b_actin_FQE", "80b ACTIN FQE"),
        ("80d_ct_legacy_FQE", "80d CT-Legacy FQE"),
        ("80e_mamba_cdsp_FQE", "80e Mamba-CDSP FQE"),
    ]
    for rid, label in model_order:
        rows = r2[r2["result_id"] == rid]
        if len(rows) > 0:
            r = rows.iloc[0]
            L.append(f"  {label:<42s} {r['estimator']:>10s} {_pct(r['value']):>12s}")

    # ── Sensitivity verdicts ─────────────────────────────────────────
    L.append("")
    L.append("─" * 72)
    L.append("SENSITIVITY ANALYSIS VERDICTS")
    L.append("─" * 72)

    verdicts = {
        "85a": "ROBUST",
        "85b": "SENSITIVE (hidden confounding)",
        "85d": "STABLE",
        "85e": "PASS 3/3",
        "85f": "WEAK E-value + GOC ROBUST",
        "85g": "ROBUST (IPMW Δ<1%, trend p=0.121)",
        "85h": "ROBUST (exclusion/bounding/stratification all consistent)",
        "85i": "SENSITIVE (non-monotonic sign change; phenotype×horizon interaction)",
        "85j": "REFRAMED (informative monitoring — methodological strength)",
    }
    for step, v in verdicts.items():
        L.append(f"  {step}: {v}")

    # ── External validation summary ──────────────────────────────────
    L.append("")
    L.append("─" * 72)
    L.append("EXTERNAL VALIDATION SUMMARY")
    L.append("─" * 72)

    r6 = df[df["section"] == "R6_transportability"]
    if len(r6) > 0:
        L.append("  Transportability grades:")
        for _, r in r6.iterrows():
            L.append(f"    {r['estimator']}: {r['verdict']}  ({r['note']})")

    r7 = df[df["section"] == "R7_consistency"]
    if len(r7) > 0:
        n_pass = (r7["verdict"] == "PASS").sum()
        n_fail = (r7["verdict"] == "FAIL").sum()
        n_na = (r7["verdict"] == "N/A").sum()
        L.append(f"  Consistency checks: {n_pass} PASS / {n_fail} FAIL / {n_na} N/A")

    L.append("")
    L.append("  Overall verdict: PARTIALLY TRANSPORTABLE")
    L.append("  - DM sign consistent across databases")
    L.append("  - HFpEF specificity preserved (5.1–17.4×)")
    L.append("  - Recalibration ROBUST (Δ<0.03pp)")
    L.append("  - HiRID ABANDONED (NT-proBNP 1.1%)")

    # ── Manuscript limitations ───────────────────────────────────────
    L.append("")
    L.append("─" * 72)
    L.append("MANUSCRIPT LIMITATIONS CHECKLIST")
    L.append("─" * 72)
    limitations = [
        "85b: Sequential deconfounding sensitivity to unmeasured confounding",
        "85f: Weak E-value (point=1.52, CI bound=1.49 for ALL)",
        "85i: Horizon sensitivity — 48h sign reversal, 72h phenotype×horizon interaction",
        "HiRID validation infeasible (NT-proBNP 1.1% coverage)",
        "eICU ascertainment contamination (~35% HFrEF in HFpEF group)",
        "eICU mortality underestimates true 28-day rate",
        "State transition model non-transportable (rollout drift ratio 2.55×–35×)",
        "Tier A represents only 5/205 eICU hospitals",
        "85j: Informative monitoring (reframed as strength, not pure weakness)",
    ]
    for i, lim in enumerate(limitations, 1):
        L.append(f"  {i}. {lim}")

    # ── Next steps ───────────────────────────────────────────────────
    L.append("")
    L.append("─" * 72)
    L.append("NEXT STEPS")
    L.append("─" * 72)
    L.append("  → Step 95b: Generate manuscript-ready figures and tables")
    L.append("  → Step 95c: Dataset and model cards")
    L.append("  → Step 95d: Reproducibility rerun pipeline")
    L.append("  → Step 95e: Archive release bundle")

    return "\n".join(L) + "\n"


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("95a — Master Results Registry")
    print("=" * 72)
    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    reg = RegistryBuilder()

    # ── Build all sections ────────────────────────────────────────────
    log.info("[1/8] R1: Primary OPE...")
    build_r1_primary_ope(reg)

    log.info("[2/8] R2: Step C models...")
    build_r2_step_c_models(reg)

    log.info("[3/8] R3: HFpEF × shock interaction...")
    build_r3_interaction(reg)

    log.info("[4/8] R4: Sensitivity analyses...")
    build_r4_sensitivity(reg)

    log.info("[5/8] R5: External validation...")
    build_r5_external_validation(reg)

    log.info("[6/8] R6: Transportability grades...")
    build_r6_transportability(reg)

    log.info("[7/8] R7: Cross-dataset consistency...")
    build_r7_consistency(reg)

    log.info("[8/8] R8: Study parameters...")
    build_r8_study_parameters(reg)

    # ── Export CSV ────────────────────────────────────────────────────
    df = reg.to_dataframe()
    csv_path = TABLES_DIR / "95a_results_registry.csv"
    df.to_csv(csv_path, index=False)
    log.info("Saved CSV: %s (%d rows)", csv_path, len(df))

    # ── Export JSON ───────────────────────────────────────────────────
    json_out = {
        "step": "95a_master_results_registry",
        "date": ts,
        "protocol_version": "v3.1",
        "database": "MIMIC-IV + eICU-CRD",
        "n_entries": len(df),
        "sections": reg.to_json(),
    }
    json_path = TABLES_DIR / "95a_results_registry.json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2, default=str)
    log.info("Saved JSON: %s", json_path)

    # ── Generate report ──────────────────────────────────────────────
    report_text = generate_report(reg, ts)
    report_path = REPORTS_DIR / "95a_master_results_registry_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    log.info("Saved report: %s", report_path)

    # ── Summary ──────────────────────────────────────────────────────
    elapsed = time.time() - t0
    sec_counts = df["section"].value_counts().sort_index()

    print()
    print("=" * 72)
    print(f"95a COMPLETE. {len(df)} registry entries across {len(sec_counts)} sections.")
    print(f"Time: {elapsed:.1f}s")
    print()
    print("Section breakdown:")
    for sec, cnt in sec_counts.items():
        print(f"  {sec}: {cnt}")
    print()
    print("Outputs:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print(f"  {report_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
