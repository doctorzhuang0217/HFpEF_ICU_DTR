#!/usr/bin/env python3
"""02_build_stays_eicu — Build eICU-CRD ICU stays base table.

Purpose
-------
Load the eICU-CRD patient table and build a stays-level table with
demographics, LOS, index-stay flag, and hospital/ICU mortality.  This mirrors
01_build_stays_mimic but adapted for eICU-CRD's schema.

Inputs
------
- data/raw/eicu/2.0/patient.csv.gz

Outputs
-------
- data/staging/02_stays_eicu.parquet
- reports/02_stays_eicu_summary.txt

Notes
-----
- eICU-CRD stores all stay-level data in a single patient table (no
  separate patients/admissions tables).
- Timestamps are expressed as minute offsets from ICU admission (unitadmit
  is the anchor point, offset = 0).  We compute LOS from these offsets
  rather than absolute timestamps.
- Age is provided directly but may contain the string "> 89" for patients
  older than 89.  We recode this to 90 (convention in eICU literature).
- Negative LOS values (rare data quality issues) are flagged and excluded
  from the main LOS columns (set to NaN).
- Index ICU stay = first visit per uniquepid (by unitvisitnumber).
- This step does NOT apply HFpEF ascertainment, T0 filtering, or the
  hemodynamic-management gate.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

BANNER = "=" * 70


def _parse_age_to_years(series: pd.Series) -> pd.Series:
    age_str = series.astype("string").str.strip()
    over_89 = age_str.str.contains(r"^>\s*89$", case=False, regex=True, na=False)
    as_num = pd.to_numeric(age_str, errors="coerce")
    as_num.loc[over_89] = 90
    return as_num


def _flag_expired(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .isin(["expired", "dead", "deceased"])
        .astype("int8")
    )


def _compute_index_flag(df: pd.DataFrame) -> pd.Series:
    visit_num = pd.to_numeric(df.get("unitvisitnumber"), errors="coerce")
    tie_breaker = pd.to_numeric(df.get("patientunitstayid"), errors="coerce")
    sorter = pd.DataFrame(
        {
            "uniquepid": df["uniquepid"],
            "visit_num": visit_num,
            "patientunitstayid": tie_breaker,
        }
    )

    first_idx = (
        sorter.sort_values(
            by=["uniquepid", "visit_num", "patientunitstayid"],
            ascending=[True, True, True],
            na_position="last",
        )
        .groupby("uniquepid", dropna=False)
        .head(1)
        .index
    )

    out = pd.Series(0, index=df.index, dtype="int8")
    out.loc[first_idx] = 1
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    input_csv = root / "data/raw/eicu/2.0/patient.csv.gz"
    output_parquet = root / "data/staging/02_stays_eicu.parquet"
    report_txt = root / "reports/02_stays_eicu_summary.txt"
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    report_txt.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{BANNER}\n\n02_build_stays_eicu — Build eICU ICU stays base table\n\n{BANNER}\n")
    if not input_csv.exists():
        raise FileNotFoundError(f"Input table not found: {input_csv}")

    stays = pd.read_csv(input_csv, low_memory=False)
    stays["age_years"] = _parse_age_to_years(stays.get("age"))
    stays["adult_18plus"] = (stays["age_years"] >= 18).fillna(False).astype("int8")

    stays["hospital_mortality"] = _flag_expired(stays.get("hospitaldischargestatus"))
    stays["icu_mortality"] = _flag_expired(stays.get("unitdischargestatus"))

    stays["unitdischargeoffset_min"] = pd.to_numeric(
        stays.get("unitdischargeoffset"), errors="coerce"
    )
    stays["hospitaladmitoffset_min"] = pd.to_numeric(
        stays.get("hospitaladmitoffset"), errors="coerce"
    )
    stays["hospitaldischargeoffset_min"] = pd.to_numeric(
        stays.get("hospitaldischargeoffset"), errors="coerce"
    )

    stays["icu_los_hours_raw"] = stays["unitdischargeoffset_min"] / 60.0
    stays["icu_los_negative"] = (stays["icu_los_hours_raw"] < 0).astype("int8")
    stays["icu_los_hours"] = stays["icu_los_hours_raw"].where(
        stays["icu_los_hours_raw"] >= 0
    )

    stays["hospital_los_hours_raw"] = (
        stays["hospitaldischargeoffset_min"] - stays["hospitaladmitoffset_min"]
    ) / 60.0
    stays["hospital_los_negative"] = (stays["hospital_los_hours_raw"] < 0).astype("int8")
    stays["hospital_los_hours"] = stays["hospital_los_hours_raw"].where(
        stays["hospital_los_hours_raw"] >= 0
    )

    stays["index_icu_stay"] = _compute_index_flag(stays)
    stays["icu_los_ge_6h"] = (stays["icu_los_hours"] >= 6).fillna(False).astype("int8")
    stays["index_stay_ge_6h"] = (
        (stays["index_icu_stay"] == 1) & (stays["icu_los_ge_6h"] == 1)
    ).astype("int8")

    keep_cols = [
        "patientunitstayid",
        "patienthealthsystemstayid",
        "uniquepid",
        "gender",
        "ethnicity",
        "age",
        "age_years",
        "adult_18plus",
        "hospitalid",
        "wardid",
        "apacheadmissiondx",
        "admissionweight",
        "dischargeweight",
        "unittype",
        "unitvisitnumber",
        "unitstaytype",
        "hospitaladmitsource",
        "unitadmitsource",
        "unitdischargelocation",
        "unitdischargestatus",
        "hospitaldischargelocation",
        "hospitaldischargestatus",
        "hospitaldischargeyear",
        "unitdischargeoffset_min",
        "hospitaladmitoffset_min",
        "hospitaldischargeoffset_min",
        "icu_los_hours",
        "hospital_los_hours",
        "icu_los_negative",
        "hospital_los_negative",
        "icu_mortality",
        "hospital_mortality",
        "index_icu_stay",
        "icu_los_ge_6h",
        "index_stay_ge_6h",
    ]
    keep_cols = [c for c in keep_cols if c in stays.columns]

    out = stays.loc[:, keep_cols].copy()
    out.to_parquet(output_parquet, index=False)

    total = len(out)
    adult_n = int(out["adult_18plus"].sum()) if "adult_18plus" in out else 0
    pediatric_n = total - adult_n if total else 0
    neg_icu_los_n = int(out["icu_los_negative"].sum()) if "icu_los_negative" in out else 0
    neg_hosp_los_n = (
        int(out["hospital_los_negative"].sum()) if "hospital_los_negative" in out else 0
    )
    unique_hospitals = out["hospitalid"].nunique(dropna=True) if "hospitalid" in out else 0
    icu_deaths = int(out["icu_mortality"].sum()) if "icu_mortality" in out else 0
    hosp_deaths = int(out["hospital_mortality"].sum()) if "hospital_mortality" in out else 0
    icu_mortality_rate = (100.0 * icu_deaths / total) if total else float("nan")
    hosp_mortality_rate = (100.0 * hosp_deaths / total) if total else float("nan")

    lines = [
        BANNER,
        "02_build_stays_eicu — Summary Report",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        BANNER,
        "",
        f"Total ICU stays         : {total:10,}",
        f"Unique patients         : {out['uniquepid'].nunique(dropna=True):10,}",
        f"Unique hospital stays   : {out['patienthealthsystemstayid'].nunique(dropna=True):10,}",
        f"Contributing hospitals  : {unique_hospitals:10,}",
        f"Adult stays (>=18y)     : {adult_n:10,}",
        f"<18y stays              : {pediatric_n:10,}",
        f"Index ICU stays         : {int(out['index_icu_stay'].sum()):10,}",
        f"ICU LOS >= 6h           : {int(out['icu_los_ge_6h'].sum()):10,}",
        f"Index stays with >= 6h  : {int(out['index_stay_ge_6h'].sum()):10,}",
        f"ICU deaths              : {icu_deaths:10,}",
        f"Hospital deaths         : {hosp_deaths:10,}",
        f"ICU mortality rate      : {icu_mortality_rate:9.1f}%",
        f"Hospital mortality rate : {hosp_mortality_rate:9.1f}%",
        f"Negative ICU LOS rows   : {neg_icu_los_n:10,}",
        f"Negative Hosp LOS rows  : {neg_hosp_los_n:10,}",
        "",
        "Age distribution:",
        str(out["age_years"].describe()) if "age_years" in out else "<missing>",
        "",
        "ICU LOS (hours) distribution (non-negative only):",
        str(out["icu_los_hours"].describe()) if "icu_los_hours" in out else "<missing>",
        "",
        "Top 10 unit types:",
        out["unittype"].value_counts(dropna=False).head(10).to_string()
        if "unittype" in out
        else "<missing>",
        "",
        f"Output: {output_parquet}",
    ]
    report_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Written: {output_parquet}")
    print(f"Report : {report_txt}")
    print("Done.")


if __name__ == "__main__":
    main()
