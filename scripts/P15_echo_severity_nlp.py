#!/usr/bin/env python3
"""
P15_echo_severity_nlp — Echo-derived VHD severity, SAM, and LVOTO flags.

Purpose
-------
P13 reports any-VHD presence by ICD code, which does not encode severity
and cannot capture echo-only phenomena like
systolic anterior motion (SAM) or dynamic LVOT obstruction (LVOTO).
P15 extracts these from MIMIC-IV-Note 2.2 free-text reports:

  - discharge.csv.gz  (discharge summaries — usually contain echo summary)
  - radiology.csv.gz  (radiology reports — some TTE/TEE filed here)

Per-valve severity (mild / moderate / severe) is taken via three
strategies in priority order:

  1. Adjective adjacent to valve phrase: "moderate aortic stenosis",
     "aortic stenosis is severe", "mild-to-moderate MR"
  2. Numeric thresholds (AHA/ESC guideline cutoffs):
        AS  : peak gradient ≥64 / mean gradient ≥40 / AVA <1.0   → severe
              peak 40-63 / mean 20-39 / AVA 1.0-1.5              → moderate
              peak <40 / mean <20 / AVA >1.5                     → mild
        MR  : EROA ≥0.4 cm² / regurg vol ≥60 mL / VC ≥0.7 cm     → severe
              EROA 0.2-0.39 / regurg vol 30-59 / VC 0.3-0.69     → moderate
              EROA <0.2 / regurg vol <30 / VC <0.3               → mild
  3. None found → severity_known = False.

SAM and LVOTO:
  - SAM    : "SAM" / "systolic anterior motion" mentioned
  - LVOTO  : "LVOTO" / "dynamic LVOT obstruction" / "LVOT obstruction" /
             "LVOT gradient ≥30 mmHg" extracted numeric

Inputs
------
- data/raw/mimiciv/note/2.2/note/discharge.csv.gz
- data/raw/mimiciv/note/2.2/note/radiology.csv.gz
- data/staging/25a_time_zero_flags_mimic.parquet

Outputs
-------
- data/staging/P15_echo_severity_mimic.parquet
- tables/P15_echo_severity_table.csv
- reports/P15_echo_severity_summary.txt
"""

from __future__ import annotations

import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hfpef_dtr.config import load_config, dataset_entrypoint, resolve_path

# ── Severity adjective patterns ────────────────────────────────────────
# We accept compound severities and pick the upper bound:
#   "mild to moderate" -> moderate; "moderate to severe" -> severe.
SEVERITY_RANK = {"mild": 1, "moderate": 2, "severe": 3, "critical": 3}
SEVERITY_PHRASE = (
    r"(?:mild(?:[\s-]+to[\s-]+moderate)?|moderate(?:[\s-]+to[\s-]+severe)?|severe|critical)"
)

VALVE_PATTERNS = {
    # Full-form-only valve patterns; 2-letter abbreviations (AS/MR/AI/PI/MS/TR)
    # were removed after the v1 run showed massive over-capture (HFpEF severe
    # AS 27 % — implausible). Bare 'AS' matches 'AS otherwise stable',
    # 'AS noted above', 'AS per cardiology'; 'MR' matches 'MR-resistant',
    # 'MR scan', patient initials; 'AI' matches 'AI' (artificial intelligence)
    # mentions; 'PI' matches 'PI' (principal investigator) etc.
    "aortic_stenosis":   r"aortic\s*(?:valve)?\s*stenosis",
    "aortic_regurg":     r"aortic\s*(?:valve)?\s*(?:regurgitation|insufficiency)",
    "mitral_stenosis":   r"mitral\s*(?:valve)?\s*stenosis",
    "mitral_regurg":     r"mitral\s*(?:valve)?\s*(?:regurgitation|insufficiency)",
    "tricuspid_regurg":  r"tricuspid\s*(?:valve)?\s*(?:regurgitation|insufficiency)",
    "pulmonic_regurg":   r"pulmon(?:ic|ary)\s*(?:valve)?\s*(?:regurgitation|insufficiency)",
}

# Sentence boundary characters used to scope severity adjacency.
_SENTENCE_BREAK_RE = re.compile(r"[.;:\n]")

# Negation cues looked for in the ~15 chars BEFORE a severity adjective.
_NEGATION_RE = re.compile(
    r"\b(?:no|not|without|denied?|denies|absent|free\s+of|"
    r"rule[ds]?\s+out|negative\s+for|w/o)\b\s*$",
    re.IGNORECASE,
)


def _resolve_compound_sev(s: str) -> str:
    """Map a severity phrase like 'mild to moderate' to upper bound."""
    s = s.lower().strip()
    if "to severe" in s or "-to-severe" in s:
        return "severe"
    if "to moderate" in s or "-to-moderate" in s:
        return "moderate"
    if "critical" in s:
        return "severe"
    if "severe" in s:
        return "severe"
    if "moderate" in s:
        return "moderate"
    if "mild" in s:
        return "mild"
    return ""


def _sentence_bounds(text: str, pos: int) -> tuple[int, int]:
    """Return (start, end) of the sentence containing position `pos`.

    Sentence = span between two sentence-break characters (.;:\\n).
    Capped at ±200 chars to avoid runaway matches across very long
    paragraphs lacking punctuation.
    """
    cap = 200
    left = max(0, pos - cap)
    right = min(len(text), pos + cap)
    # Walk left: stop at most-recent break.
    s = pos
    for i in range(pos - 1, left - 1, -1):
        if _SENTENCE_BREAK_RE.match(text[i]):
            s = i + 1
            break
    else:
        s = left
    # Walk right: stop at next break.
    e = pos
    for i in range(pos, right):
        if _SENTENCE_BREAK_RE.match(text[i]):
            e = i
            break
    else:
        e = right
    return s, e


def _adjective_severity(text: str, valve_re: str) -> str:
    """Severity adjective in the SAME SENTENCE as the valve phrase.

    Tighter than the v1 ±60-char window (which let 'severe' from a nearby
    clause leak across — e.g. 'moderate aortic stenosis with severe MR'
    used to assign severe to AS as well as MR).
    """
    matches = list(re.finditer(valve_re, text, re.IGNORECASE))
    best_rank = 0
    best = ""
    sev_rx = re.compile(SEVERITY_PHRASE, re.IGNORECASE)
    for m in matches:
        s, e = _sentence_bounds(text, m.start())
        # Severity must appear within the same sentence AND within ±35
        # chars of the valve mention (to avoid 'severe sepsis ... mild
        # aortic stenosis' all in one run-on sentence).
        local_start = max(s, m.start() - 35)
        local = text[local_start:min(e, m.end() + 35)]
        for sm in sev_rx.finditer(local):
            # Skip if a negation token appears in the 15 chars immediately
            # before this severity word.
            sm_global_start = local_start + sm.start()
            preceding = text[max(0, sm_global_start - 15):sm_global_start]
            if _NEGATION_RE.search(preceding):
                continue
            sev = _resolve_compound_sev(sm.group(0))
            if sev and SEVERITY_RANK[sev] > best_rank:
                best_rank = SEVERITY_RANK[sev]
                best = sev
    return best


# ── Numeric thresholds for AS / MR ──────────────────────────────────────
# AS peak gradient
RX_AS_PEAK = re.compile(
    r"(?:aortic|AV)\s*(?:peak\s*)?(?:transvalvular|valve)?\s*"
    r"(?:peak\s*)?gradient[:\s=]+(\d+(?:\.\d+)?)\s*mmHg",
    re.IGNORECASE,
)
RX_AS_MEAN = re.compile(
    r"(?:aortic|AV)\s*mean\s*gradient[:\s=]+(\d+(?:\.\d+)?)\s*mmHg",
    re.IGNORECASE,
)
RX_AVA = re.compile(
    r"aortic\s*valve\s*area[:\s=]+(\d+(?:\.\d+)?)\s*cm",
    re.IGNORECASE,
)
RX_MR_EROA = re.compile(
    r"(?:MR\s*)?EROA[:\s=]+(\d+(?:\.\d+)?)\s*cm",
    re.IGNORECASE,
)
RX_MR_REGVOL = re.compile(
    r"regurgitant\s*volume[:\s=]+(\d+(?:\.\d+)?)\s*mL",
    re.IGNORECASE,
)
RX_MR_VC = re.compile(
    r"vena\s*contracta[:\s=]+(\d+(?:\.\d+)?)\s*cm",
    re.IGNORECASE,
)

# LVOT gradient (any context)
RX_LVOT_GRAD = re.compile(
    r"LVOT\s*(?:peak\s*)?gradient[:\s=]+(\d+(?:\.\d+)?)\s*mmHg",
    re.IGNORECASE,
)

# Keyword anchors
RX_SAM = re.compile(
    r"\bSAM\b|systolic\s+anterior\s+motion",
    re.IGNORECASE,
)
RX_LVOTO_KW = re.compile(
    r"\bLVOTO\b|(?:dynamic\s+)?LVOT\s+obstruction|outflow\s+tract\s+obstruction",
    re.IGNORECASE,
)


def _max_float(rx, text: str) -> float | None:
    vals = [float(m.group(1)) for m in rx.finditer(text)]
    return max(vals) if vals else None


def _min_float(rx, text: str) -> float | None:
    vals = [float(m.group(1)) for m in rx.finditer(text)]
    return min(vals) if vals else None


def _as_severity_numeric(text: str) -> str:
    """Aortic stenosis severity from numeric thresholds (severe wins)."""
    peak = _max_float(RX_AS_PEAK, text)
    mean = _max_float(RX_AS_MEAN, text)
    ava = _min_float(RX_AVA, text)
    is_severe = (
        (peak is not None and peak >= 64)
        or (mean is not None and mean >= 40)
        or (ava is not None and ava < 1.0)
    )
    is_moderate = (
        (peak is not None and 40 <= peak < 64)
        or (mean is not None and 20 <= mean < 40)
        or (ava is not None and 1.0 <= ava < 1.5)
    )
    is_mild = (
        (peak is not None and peak < 40)
        or (mean is not None and mean < 20)
        or (ava is not None and ava >= 1.5)
    )
    if is_severe:
        return "severe"
    if is_moderate:
        return "moderate"
    if is_mild:
        return "mild"
    return ""


def _mr_severity_numeric(text: str) -> str:
    eroa = _max_float(RX_MR_EROA, text)
    rv = _max_float(RX_MR_REGVOL, text)
    vc = _max_float(RX_MR_VC, text)
    is_severe = (
        (eroa is not None and eroa >= 0.4)
        or (rv is not None and rv >= 60)
        or (vc is not None and vc >= 0.7)
    )
    is_moderate = (
        (eroa is not None and 0.2 <= eroa < 0.4)
        or (rv is not None and 30 <= rv < 60)
        or (vc is not None and 0.3 <= vc < 0.7)
    )
    is_mild = (
        (eroa is not None and eroa < 0.2)
        or (rv is not None and rv < 30)
        or (vc is not None and vc < 0.3)
    )
    if is_severe:
        return "severe"
    if is_moderate:
        return "moderate"
    if is_mild:
        return "mild"
    return ""


def extract_per_note(text: str) -> dict:
    """Pull every severity / SAM / LVOTO signal from a single note text."""
    if not isinstance(text, str) or len(text) < 20:
        return {}

    out: dict[str, object] = {}
    # Per-valve adjective severity.
    for valve, pat in VALVE_PATTERNS.items():
        sev = _adjective_severity(text, pat)
        out[f"{valve}_sev_adj"] = sev

    # AS numeric.
    out["as_sev_numeric"] = _as_severity_numeric(text)
    # MR numeric.
    out["mr_sev_numeric"] = _mr_severity_numeric(text)

    # LVOT gradient + SAM/LVOTO keywords.
    lvot = _max_float(RX_LVOT_GRAD, text)
    out["lvot_gradient_mmHg"] = lvot
    out["sam_keyword"] = bool(RX_SAM.search(text))
    out["lvoto_keyword"] = bool(RX_LVOTO_KW.search(text))
    return out


def _merge_severity(adj: str, num: str) -> str:
    """Combine adjective and numeric severities — take the more severe."""
    a = SEVERITY_RANK.get(adj, 0)
    n = SEVERITY_RANK.get(num, 0)
    if max(a, n) == 0:
        return ""
    return {1: "mild", 2: "moderate", 3: "severe"}[max(a, n)]


def scan_notes(
    path: Path,
    target_hadms: set,
    label: str,
    chunksize: int = 5_000,
) -> list[dict]:
    """Scan a CSV.gz of notes, return one dict per matching note."""
    rows: list[dict] = []
    if not path.exists():
        print(f"    [skip] {path} not found")
        return rows

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda it, **kw: it  # noqa: E731

    pbar = tqdm(
        pd.read_csv(
            path,
            usecols=["subject_id", "hadm_id", "charttime", "text"],
            chunksize=chunksize,
        ),
        desc=f"    {label}",
        unit="chunk",
    )
    for chunk in pbar:
        sub = chunk[chunk["hadm_id"].isin(target_hadms)]
        for r in sub.itertuples(index=False):
            features = extract_per_note(r.text)
            if any(features.get(k) for k in features):
                features.update({
                    "subject_id": r.subject_id,
                    "hadm_id": r.hadm_id,
                    "note_charttime": r.charttime,
                    "source": label,
                })
                rows.append(features)
    return rows


def aggregate_per_admission(rows: list[dict]) -> pd.DataFrame:
    """Collapse multiple notes per hadm_id into a single severity row by
    taking the most severe adjective+numeric per valve, max LVOT gradient,
    and OR over keyword flags."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    # Final per-valve severity = max(adjective, numeric).
    out_rows: list[dict] = []
    for hadm, sub in df.groupby("hadm_id"):
        rec: dict[str, object] = {"hadm_id": hadm}
        for valve in VALVE_PATTERNS:
            adj_col = f"{valve}_sev_adj"
            adj = ""
            if adj_col in sub.columns:
                ranks = [SEVERITY_RANK.get(v, 0) for v in sub[adj_col].fillna("")]
                if any(r > 0 for r in ranks):
                    adj = {1: "mild", 2: "moderate", 3: "severe"}[max(ranks)]
            num = ""
            if valve == "aortic_stenosis":
                ranks = [SEVERITY_RANK.get(v, 0) for v in sub.get("as_sev_numeric", pd.Series(dtype=object)).fillna("")]
                if any(r > 0 for r in ranks):
                    num = {1: "mild", 2: "moderate", 3: "severe"}[max(ranks)]
            elif valve == "mitral_regurg":
                ranks = [SEVERITY_RANK.get(v, 0) for v in sub.get("mr_sev_numeric", pd.Series(dtype=object)).fillna("")]
                if any(r > 0 for r in ranks):
                    num = {1: "mild", 2: "moderate", 3: "severe"}[max(ranks)]
            rec[f"{valve}_severity"] = _merge_severity(adj, num)

        rec["lvot_gradient_max_mmHg"] = (
            float(sub["lvot_gradient_mmHg"].dropna().max())
            if sub["lvot_gradient_mmHg"].notna().any()
            else np.nan
        )
        rec["sam_present"] = bool(sub["sam_keyword"].any()) if "sam_keyword" in sub.columns else False
        rec["lvoto_present"] = (
            bool(sub["lvoto_keyword"].any() if "lvoto_keyword" in sub.columns else False)
            or bool(np.nansum([rec["lvot_gradient_max_mmHg"]]) >= 30)
        )
        rec["any_severe_vhd"] = any(
            rec[f"{v}_severity"] == "severe" for v in VALVE_PATTERNS
        )
        rec["any_moderate_or_severe_vhd"] = any(
            rec[f"{v}_severity"] in ("moderate", "severe") for v in VALVE_PATTERNS
        )
        rec["sources"] = ";".join(sorted(sub["source"].unique()))
        out_rows.append(rec)
    return pd.DataFrame(out_rows)


def main() -> None:
    cfg = load_config()
    mimic_root = dataset_entrypoint(cfg, "mimic")
    staging_dir = resolve_path(cfg, "staging")
    reports_dir = resolve_path(cfg, "reports")
    tables_dir = resolve_path(cfg, "tables")
    for d in (staging_dir, reports_dir, tables_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("P15_echo_severity_nlp — VHD severity / SAM / LVOTO from notes")
    print("=" * 70)

    print("\n[1/4] Loading cohort ...")
    cohort = pd.read_parquet(
        staging_dir / "25a_time_zero_flags_mimic.parquet",
        columns=["stay_id", "subject_id", "hadm_id",
                 "hf_subtype", "comparator_group"],
    )
    target_hadms = set(cohort["hadm_id"].unique())
    print(f"  Target admissions: {len(target_hadms):,}")

    note_subdir = cfg["datasets"]["mimic"].get("subdirs", {}).get("note")
    note_dir = mimic_root / note_subdir if note_subdir else mimic_root / "note" / "2.2" / "note"

    print("\n[2/4] Scanning discharge.csv.gz ...")
    rows = scan_notes(note_dir / "discharge.csv.gz", target_hadms, "discharge")
    print(f"  Matched discharge notes: {len(rows):,}")

    print("\n[3/4] Scanning radiology.csv.gz ...")
    rad_rows = scan_notes(note_dir / "radiology.csv.gz", target_hadms,
                          "radiology", chunksize=10_000)
    print(f"  Matched radiology notes: {len(rad_rows):,}")
    rows.extend(rad_rows)

    print("\n[4/4] Aggregating per admission + writing outputs ...")
    agg = aggregate_per_admission(rows)
    if agg.empty:
        print("  WARNING: no echo severity signal extracted.")
        agg = pd.DataFrame({"hadm_id": []})

    out = cohort.merge(agg, on="hadm_id", how="left")
    for col in [c for c in agg.columns if c.endswith("_severity")]:
        out[col] = out[col].fillna("")
    for col in ("sam_present", "lvoto_present",
                "any_severe_vhd", "any_moderate_or_severe_vhd"):
        if col in out.columns:
            out[col] = out[col].fillna(False).astype(bool)
    if "lvot_gradient_max_mmHg" in out.columns:
        out["lvot_gradient_max_mmHg"] = out["lvot_gradient_max_mmHg"].astype(float)

    out = out.sort_values("stay_id").reset_index(drop=True)
    out_path = staging_dir / "P15_echo_severity_mimic.parquet"
    out.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"  → {out_path}  ({len(out):,} rows, {out.shape[1]} cols)")

    # Build summary table: severity prevalence per HF subtype.
    flag_cols = [f"{v}_severity" for v in VALVE_PATTERNS] + [
        "sam_present", "lvoto_present",
        "any_severe_vhd", "any_moderate_or_severe_vhd",
    ]
    rows_out: list[dict] = []
    for grp, sg in out.groupby("comparator_group"):
        n = len(sg)
        rec: dict[str, object] = {"comparator_group": grp, "n": n}
        for valve in VALVE_PATTERNS:
            col = f"{valve}_severity"
            if col not in sg.columns:
                continue
            for sev in ("mild", "moderate", "severe"):
                cnt = int((sg[col] == sev).sum())
                rec[f"{valve}_{sev}_pct"] = round(cnt / n * 100, 1) if n else 0.0
        for col in ("sam_present", "lvoto_present",
                    "any_severe_vhd", "any_moderate_or_severe_vhd"):
            if col in sg.columns:
                rec[f"{col}_pct"] = round(sg[col].mean() * 100, 1) if n else 0.0
        rows_out.append(rec)
    table = pd.DataFrame(rows_out)

    # Count stays with at least one extracted signal (any non-empty severity
    # adjective, SAM, LVOTO, or LVOT gradient).
    severity_cols = [f"{v}_severity" for v in VALVE_PATTERNS]
    has_severity = pd.Series(False, index=out.index)
    for c in severity_cols:
        if c in out.columns:
            has_severity |= out[c].fillna("").astype(str).ne("")
    has_other = pd.Series(False, index=out.index)
    for c in ("sam_present", "lvoto_present"):
        if c in out.columns:
            has_other |= out[c].fillna(False).astype(bool)
    if "lvot_gradient_max_mmHg" in out.columns:
        has_other |= out["lvot_gradient_max_mmHg"].notna()
    n_with_signal = int((has_severity | has_other).sum())

    table_path = tables_dir / "P15_echo_severity_table.csv"
    table.to_csv(table_path, index=False)
    print(f"  → {table_path}")

    report = textwrap.dedent(f"""\
    ================================================================
    P15_echo_severity_nlp — Summary Report
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ================================================================

    Total cohort stays                       : {len(cohort):>8,}
    Stays with any echo signal extracted     : {n_with_signal:>8,}

    SEVERITY PREVALENCE TABLE
    ──────────────────────────
    {textwrap.indent(table.to_string(index=False), '  ')}

    NOTES / LIMITATIONS
    ────────────────────
    - Severity is the MAX of adjective- and numeric-derived classes per
      valve, taken over all notes for the admission. Compound phrases
      like 'mild-to-moderate' are mapped to the upper bound (moderate).
    - AS thresholds: peak ≥64 / mean ≥40 / AVA <1.0 → severe (AHA/ESC).
      MR thresholds: EROA ≥0.4 / regurg vol ≥60 / VC ≥0.7 → severe.
    - SAM and LVOTO use both keyword anchors and an LVOT gradient
      ≥30 mmHg numeric trigger.
    - Recall is bounded by free-text quality. Discharge summaries vary
      widely in echo-section detail; radiology echoes are inconsistent
      between admissions.
    - Single-rater NLP, no human-validation pass yet. For a formal
      revision a 100-note manual audit would be advisable.

    OUTPUTS
    ───────
    Per-stay parquet : {out_path}
    Severity table   : {table_path}
    """)

    report_path = reports_dir / "P15_echo_severity_summary.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"  → Report: {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
