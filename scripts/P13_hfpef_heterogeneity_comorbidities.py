#!/usr/bin/env python3
"""
P13_hfpef_heterogeneity_comorbidities — HFpEF subtype heterogeneity table.

Purpose
-------
This script asks whether the HFpEF cohort is contaminated by/dominated
by specific cardiac substrates that behave hemodynamically differently from
"classic" HFpEF (e.g. atrial fibrillation, pulmonary hypertension, severe
valvular disease, restrictive/infiltrative cardiomyopathy, hypertrophic
cardiomyopathy / LVOTO). This script builds an ICD-based descriptive
comorbidity table for the heterogeneity discussion.

Comorbidity domains
-------------------
1. Atrial fibrillation / flutter      (AF)
2. Pulmonary hypertension              (PH)
3. Valvular heart disease              (VHD) — overall + AS / MR / MS / AR /
                                              tricuspid / multi-valve subtypes
4. Restrictive / infiltrative CM       (incl. systemic amyloidosis)
5. Hypertrophic cardiomyopathy         (HCM, obstructive vs non-obstructive)

Limitations (documented in report)
----------------------------------
- ICD codes do NOT encode severity (the table reports "any coded VHD";
  moderate/severe stratification requires echo-derived gradients).
- SAM and dynamic LVOTO require echo phenotyping (not codable from ICD).
- Cardiac amyloidosis is frequently undercoded; this is a lower bound.
- is_restrictive_cm uses ICD-10 I42.5 / I42.3 only; ICD-9 425.4 is excluded
  per P13b audit (it functions as a generic CM-NOS code, not restrictive).
  425.4 is exposed separately as is_cardiomyopathy_icd9_4254_unspecified.
- is_amyloidosis ICD-9 codes restricted to 277.30 / 277.39; 277.31
  (familial Mediterranean fever) is excluded.

Inputs
------
- data/staging/25a_time_zero_flags_mimic.parquet
- data/staging/10b_diagnoses_mimic.parquet

Outputs
-------
- data/staging/P13_hfpef_heterogeneity_comorbidities_mimic.parquet
- reports/P13_hfpef_heterogeneity_comorbidities_summary.txt
- tables/P13_hfpef_heterogeneity_table.csv
"""

from __future__ import annotations

import sys
import textwrap
from datetime import datetime
from pathlib import Path  # noqa: F401  (Path used by resolve_path return type)

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, resolve_path


def _norm(code: str) -> str:
    return str(code).replace(".", "").strip().upper()


# ── ICD code sets ────────────────────────────────────────────────────
# 1. Atrial fibrillation / flutter
AF_ICD10_PREFIX = ["I48"]
AF_ICD9 = {"42731", "42732"}

# 2. Pulmonary hypertension
PH_ICD10 = {"I270", "I2720", "I2721", "I2722", "I2723", "I2724", "I2729",
            "I2781", "I2782", "I2783", "I2789", "I279"}
PH_ICD9_PREFIX = ["4160", "4161", "4162", "4168", "4169"]

# 3. Valvular heart disease (any) — broken into subtypes for finer reporting.
#    ICD-10 nonrheumatic primary codes: I34 (mitral), I35 (aortic),
#    I36 (tricuspid), I37 (pulmonary).
#    ICD-10 rheumatic: I05, I06, I07, I08, I09 (heart-related rheumatic).
#    ICD-9: 394–397 rheumatic, 424.x nonrheumatic.
AORTIC_STENOSIS_ICD10 = {"I350", "I352", "I060", "I062"}
AORTIC_STENOSIS_ICD9 = {"3950", "3952"}
AORTIC_REGURG_ICD10 = {"I351", "I352", "I061", "I062"}
AORTIC_REGURG_ICD9 = {"3951", "3952", "4241"}
MITRAL_STENOSIS_ICD10 = {"I050", "I052"}
MITRAL_STENOSIS_ICD9 = {"3940", "3942"}
MITRAL_REGURG_ICD10 = {"I340", "I341", "I342", "I348", "I349",
                       "I051", "I052", "I058", "I059"}
MITRAL_REGURG_ICD9 = {"3941", "3949", "4240"}
TRICUSPID_ICD10_PREFIX = ["I07", "I36"]
TRICUSPID_ICD9_PREFIX = ["397", "4242"]
MULTI_VALVE_ICD10_PREFIX = ["I08"]
MULTI_VALVE_ICD9_PREFIX = ["3961", "3962", "3963", "3968", "3969"]
ANY_VHD_ICD10_PREFIX = ["I05", "I06", "I07", "I08", "I34", "I35", "I36", "I37"]
ANY_VHD_ICD9_PREFIX = ["394", "395", "396", "397", "4240", "4241", "4242", "4243"]

# 4. Restrictive / infiltrative cardiomyopathy & systemic amyloidosis.
#    ICD-10: I42.5 restrictive CM; I42.3 endomyocardial (eosinophilic).
#    ICD-9 425.4 ("Other primary cardiomyopathies") was originally included
#    here, but the P13b audit (commit 77f6a8d) showed that 425.4 is used in
#    practice as a generic "cardiomyopathy NOS" code: 99.6 % of HFrEF/HFmrEF
#    flag carriers were 425.4 with median EF 25 % (IQR 20-35), i.e. dilated
#    HFrEF mis-coded as "other primary CM". We therefore EXCLUDE 425.4 from
#    the restrictive flag and expose it separately as
#    is_cardiomyopathy_icd9_4254_unspecified for transparency.
#
#    Amyloidosis: E85.x (E85.4 organ-limited, E85.81/82 light/heavy chain,
#    E85.9 unspecified). ICD-9 277.30 amyloidosis-unspecified, 277.39 other
#    amyloidosis. (277.31 is FAMILIAL MEDITERRANEAN FEVER, NOT amyloidosis,
#    and is excluded.)
RESTRICTIVE_CM_ICD10 = {"I425", "I423"}
RESTRICTIVE_CM_ICD9: set[str] = set()   # 425.4 excluded post-audit (P13b)
UNSPEC_CM_ICD9 = {"4254"}                # captured separately for transparency
AMYLOID_ICD10_PREFIX = ["E85"]
AMYLOID_ICD9 = {"27730", "27739"}        # exact codes; excludes 277.31 (FMF)

# 5. Hypertrophic cardiomyopathy (obstructive vs non-obstructive).
HCM_OBSTRUCTIVE_ICD10 = {"I421"}
HCM_OBSTRUCTIVE_ICD9 = {"4251"}    # 425.1 hypertrophic obstructive CM
HCM_NONOBSTRUCTIVE_ICD10 = {"I422"}
HCM_NONOBSTRUCTIVE_ICD9 = {"42518"}  # 425.18 = "other hypertrophic CM"


def _has_prefix(codes: pd.Series, prefixes: list[str]) -> pd.Series:
    if not prefixes:
        return pd.Series(False, index=codes.index)
    pat = "|".join(f"^{p}" for p in prefixes)
    return codes.str.contains(pat, regex=True, na=False)


def classify(hadm_ids: set, dx: pd.DataFrame) -> pd.DataFrame:
    dx = dx.loc[dx["hadm_id"].isin(hadm_ids), ["hadm_id", "icd_code", "icd_version"]]
    dx_norm = (
        dx["icd_code"].astype(str)
        .str.replace(".", "", regex=False).str.upper().str.strip()
    )
    v10 = (dx["icd_version"] == 10).to_numpy()
    v9 = (dx["icd_version"] == 9).to_numpy()
    hadm_arr = dx["hadm_id"].to_numpy()

    def _by_prefix(prefixes: list[str], is_v10: bool) -> set:
        if not prefixes:
            return set()
        ver = v10 if is_v10 else v9
        match = _has_prefix(dx_norm, prefixes).to_numpy()
        return set(hadm_arr[ver & match])

    def _by_exact(code_set: set, is_v10: bool) -> set:
        ver = v10 if is_v10 else v9
        match = dx_norm.isin(code_set).to_numpy()
        return set(hadm_arr[ver & match])

    sets = {
        "is_af": (
            _by_prefix(AF_ICD10_PREFIX, True)
            | _by_exact(AF_ICD9, False)
        ),
        "is_pulm_hypertension": (
            _by_exact(PH_ICD10, True)
            | _by_prefix(PH_ICD9_PREFIX, False)
        ),
        "is_aortic_stenosis": (
            _by_exact(AORTIC_STENOSIS_ICD10, True)
            | _by_exact(AORTIC_STENOSIS_ICD9, False)
        ),
        "is_aortic_regurg": (
            _by_exact(AORTIC_REGURG_ICD10, True)
            | _by_exact(AORTIC_REGURG_ICD9, False)
        ),
        "is_mitral_stenosis": (
            _by_exact(MITRAL_STENOSIS_ICD10, True)
            | _by_exact(MITRAL_STENOSIS_ICD9, False)
        ),
        "is_mitral_regurg": (
            _by_exact(MITRAL_REGURG_ICD10, True)
            | _by_exact(MITRAL_REGURG_ICD9, False)
        ),
        "is_tricuspid_disease": (
            _by_prefix(TRICUSPID_ICD10_PREFIX, True)
            | _by_prefix(TRICUSPID_ICD9_PREFIX, False)
        ),
        "is_multivalve_disease": (
            _by_prefix(MULTI_VALVE_ICD10_PREFIX, True)
            | _by_prefix(MULTI_VALVE_ICD9_PREFIX, False)
        ),
        "is_any_vhd": (
            _by_prefix(ANY_VHD_ICD10_PREFIX, True)
            | _by_prefix(ANY_VHD_ICD9_PREFIX, False)
        ),
        "is_restrictive_cm": (
            _by_exact(RESTRICTIVE_CM_ICD10, True)
            | _by_exact(RESTRICTIVE_CM_ICD9, False)
        ),
        "is_cardiomyopathy_icd9_4254_unspecified": (
            _by_exact(UNSPEC_CM_ICD9, False)
        ),
        "is_amyloidosis": (
            _by_prefix(AMYLOID_ICD10_PREFIX, True)
            | _by_exact(AMYLOID_ICD9, False)
        ),
        "is_hcm_obstructive": (
            _by_exact(HCM_OBSTRUCTIVE_ICD10, True)
            | _by_exact(HCM_OBSTRUCTIVE_ICD9, False)
        ),
        "is_hcm_nonobstructive": (
            _by_exact(HCM_NONOBSTRUCTIVE_ICD10, True)
            | _by_exact(HCM_NONOBSTRUCTIVE_ICD9, False)
        ),
    }
    # Composite flags.
    result = pd.DataFrame({"hadm_id": sorted(hadm_ids)})
    for col, hadm_set in sets.items():
        result[col] = result["hadm_id"].isin(hadm_set)

    result["is_hcm_any"] = result["is_hcm_obstructive"] | result["is_hcm_nonobstructive"]
    result["is_restrictive_or_amyloid"] = (
        result["is_restrictive_cm"] | result["is_amyloidosis"]
    )
    return result


def _table_by_subtype(out: pd.DataFrame, flag_cols: list[str]) -> pd.DataFrame:
    """Produce comorbidity prevalence table: rows = comorbidities, cols = HF subtype."""
    groups = out.groupby("comparator_group")
    n_per_group = groups.size()
    rows = []
    for col in flag_cols:
        counts = groups[col].sum()
        pcts = (counts / n_per_group * 100).round(1)
        row = {"comorbidity": col}
        for grp in n_per_group.index:
            n = int(counts.get(grp, 0))
            p = float(pcts.get(grp, 0.0))
            row[f"{grp}_n"] = n
            row[f"{grp}_pct"] = p
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    staging_dir = resolve_path(cfg, "staging")
    reports_dir = resolve_path(cfg, "reports")
    tables_dir = resolve_path(cfg, "tables")
    for d in (staging_dir, reports_dir, tables_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("P13_hfpef_heterogeneity_comorbidities — ICD-based heterogeneity")
    print("=" * 70)

    print("\n[1/4] Loading inputs ...")
    cohort = pd.read_parquet(
        staging_dir / "25a_time_zero_flags_mimic.parquet",
        columns=["stay_id", "subject_id", "hadm_id",
                 "hf_subtype", "comparator_group"],
    )
    n_index = len(cohort)
    hadm_ids = set(cohort["hadm_id"].unique())
    print(f"  Index stays: {n_index:,}  ({len(hadm_ids):,} admissions)")

    dx = pd.read_parquet(
        staging_dir / "10b_diagnoses_mimic.parquet",
        columns=["hadm_id", "icd_code", "icd_version"],
    )
    print(f"  Diagnoses table: {len(dx):,} rows")

    print("\n[2/4] Classifying heterogeneity comorbidities ...")
    flags = classify(hadm_ids, dx)
    out = cohort.merge(flags, on="hadm_id", how="left")
    flag_cols = [c for c in flags.columns if c != "hadm_id"]
    for c in flag_cols:
        out[c] = out[c].fillna(False)

    print("\n[3/4] Writing per-stay flags ...")
    out = out.sort_values("stay_id").reset_index(drop=True)
    out_path = staging_dir / "P13_hfpef_heterogeneity_comorbidities_mimic.parquet"
    out.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"  → {out_path}  ({len(out):,} rows, {out.shape[1]} cols)")

    print("\n[4/4] Generating heterogeneity table + report ...")
    table = _table_by_subtype(out, flag_cols)
    table_path = tables_dir / "P13_hfpef_heterogeneity_table.csv"
    table.to_csv(table_path, index=False)
    print(f"  → {table_path}")

    # Per-comparator-group summary lines.
    n_per_group = out.groupby("comparator_group").size()
    n_lines = "\n".join(
        f"    {grp:25s}: N = {n:>8,}"
        for grp, n in n_per_group.items()
    )

    # Pretty table block (comorbidity × group, percentages).
    pivot = (
        out.groupby("comparator_group")[flag_cols]
        .mean()
        .mul(100)
        .round(1)
        .T
    )

    report = textwrap.dedent(f"""\
    ================================================================
    P13_hfpef_heterogeneity_comorbidities — Summary Report
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ================================================================

    GROUP SIZES
    ───────────
{n_lines}

    PREVALENCE (%) BY COMPARATOR GROUP
    ────────────────────────────────────
    {textwrap.indent(pivot.to_string(), '  ')}

    NOTES / LIMITATIONS
    ────────────────────
    - All flags derive from administrative ICD-9/ICD-10 diagnosis codes
      attached to the index hospital admission (hadm_id), not the ICU stay
      itself. They reflect coded comorbidities at any point during the
      admission, including on-admission and during-stay codes.
    - ICD codes do NOT encode severity; "is_any_vhd" should be read as
      "any coded valvular heart disease" rather than moderate/severe VHD.
      A future patch could add echo-derived gradient/regurgitation severity.
    - SAM (systolic anterior motion) and dynamic LVOTO are echo-defined
      and are NOT captured here; HCM flags only proxy the structural
      substrate.
    - Cardiac amyloidosis is well-known to be undercoded in administrative
      data; the amyloidosis flag should be treated as a lower bound.
    - is_restrictive_cm is intentionally STRICT: only ICD-10 I42.5 and
      I42.3 are counted. ICD-9 425.4 ("Other primary cardiomyopathies")
      was excluded after the P13b audit (commit 77f6a8d) showed it is
      used as a generic "cardiomyopathy NOS" code (99.6 % of HFrEF/HFmrEF
      425.4 carriers had median EF 25 %, i.e. dilated HFrEF). 425.4 is
      retained separately as is_cardiomyopathy_icd9_4254_unspecified for
      transparency, but should not be interpreted as restrictive CM.
    - is_amyloidosis ICD-9 codes are restricted to 277.30 / 277.39
      (amyloidosis specific). 277.31 (familial Mediterranean fever) is
      excluded.

    OUTPUTS
    ───────
    Per-stay parquet : {out_path}
    Heterogeneity CSV: {table_path}
    """)

    report_path = reports_dir / "P13_hfpef_heterogeneity_comorbidities_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"  → Report: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
