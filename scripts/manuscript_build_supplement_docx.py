#!/usr/bin/env python3
"""Build submission-ready supplementary materials DOCX.

Output: manuscript/HFpEF_ICU_DTR_supplement.docx

Formatting matches the main manuscript (Times New Roman, double-spaced,
line-numbered, 1-inch margins). sFigure numbering is consistent with
the main manuscript text references (sFigure 1-10, sTable 1-11).
Each table and figure includes abbreviation footnotes.
"""
import csv
import os
from pathlib import Path

from docx import Document
from docx.shared import Pt, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
FIGURES_PDF = ROOT / "manuscript" / "figures_pdf"
TABLES = ROOT / "tables"
OUTPUT = ROOT / "manuscript" / "HFpEF_ICU_DTR_supplement.docx"


# ── helpers ─────────────────────────────────────────────────────────────
def set_margins(section, top=2.54, bottom=2.54, left=2.54, right=2.54):
    section.top_margin = Cm(top)
    section.bottom_margin = Cm(bottom)
    section.left_margin = Cm(left)
    section.right_margin = Cm(right)


def set_line_numbering(section):
    sectPr = section._sectPr
    ln_num = parse_xml(
        '<w:lnNumType {} w:countBy="1" w:restart="continuous"/>'.format(
            nsdecls('w')
        )
    )
    sectPr.append(ln_num)


def set_font(run, name='Times New Roman', size=12, bold=False, italic=False):
    run.font.name = name
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = parse_xml(
            '<w:rFonts {} w:eastAsia="{}"/>'.format(nsdecls('w'), name)
        )
        rPr.append(rFonts)
    else:
        rFonts.set(qn('w:eastAsia'), name)


def add_para(doc, text, bold=False, italic=False, font_size=12,
             alignment=None, space_after=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    set_font(run, size=font_size, bold=bold, italic=italic)
    pf = p.paragraph_format
    pf.line_spacing = Pt(24)
    pf.space_before = Pt(0)
    if alignment:
        pf.alignment = alignment
    if space_after is not None:
        pf.space_after = Pt(space_after)
    else:
        pf.space_after = Pt(0)
    return p


def add_heading(doc, text, level=1):
    sizes = {1: 14, 2: 13, 3: 12}
    p = add_para(doc, text, bold=True, font_size=sizes.get(level, 12),
                 space_after=6)
    if level == 1:
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(12)
    return p


def add_csv_table(doc, csv_path, font_size=9):
    """Read CSV and add as a formatted Word table."""
    if not csv_path.exists():
        add_para(doc, f'[Data file not found: {csv_path.name}]', italic=True)
        return None

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return None

    headers = rows[0]
    data = rows[1:]
    return add_table_from_rows(doc, headers, data, font_size)


def add_table_from_rows(doc, headers, rows, font_size=9):
    """Add a formatted table with header shading."""
    ncols = len(headers)
    table = doc.add_table(rows=1 + len(rows), cols=ncols)
    table.style = 'Table Grid'

    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ''
        p = cell.paragraphs[0]
        run = p.add_run(h)
        set_font(run, size=font_size, bold=True)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.space_before = Pt(0)
        shading = parse_xml(
            '<w:shd {} w:fill="D9E2F3" w:val="clear"/>'.format(nsdecls('w'))
        )
        cell._element.get_or_add_tcPr().append(shading)

    for ri, row_data in enumerate(rows):
        for ci in range(min(len(row_data), ncols)):
            cell = table.rows[ri + 1].cells[ci]
            cell.text = ''
            p = cell.paragraphs[0]
            run = p.add_run(str(row_data[ci]))
            set_font(run, size=font_size)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.space_before = Pt(0)

    add_para(doc, '', space_after=6)
    return table


def add_table_footnote(doc, text, font_size=8):
    """Add an abbreviation/footnote paragraph below a table or figure."""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = Pt(18)
    pf.space_before = Pt(4)
    pf.space_after = Pt(0)
    run = p.add_run(text)
    set_font(run, size=font_size, italic=True)
    return p


def find_figure(name_variants):
    """Try multiple paths for a figure file, return first that exists."""
    for v in name_variants:
        if v.exists():
            return v
    return None


def add_figure(doc, img_path_variants, label, title, legend, abbreviations,
               width=5.5):
    """Add a figure with label, legend, and abbreviation footnote."""
    if isinstance(img_path_variants, (str, Path)):
        img_path_variants = [Path(img_path_variants)]
    else:
        img_path_variants = [Path(p) for p in img_path_variants]

    img_path = find_figure(img_path_variants)

    if img_path is None:
        names = ', '.join(str(p.name) for p in img_path_variants)
        add_para(doc, f'[Figure not found: {names}]', italic=True)
    else:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(str(img_path), width=Inches(width))

    # Legend with bold label
    p2 = doc.add_paragraph()
    p2.paragraph_format.line_spacing = Pt(24)
    p2.paragraph_format.space_after = Pt(2)
    run_label = p2.add_run(f'{label}. {title}. ')
    set_font(run_label, size=10, bold=True)
    run_text = p2.add_run(legend)
    set_font(run_text, size=10)

    # Abbreviation footnote
    if abbreviations:
        p3 = doc.add_paragraph()
        p3.paragraph_format.line_spacing = Pt(16)
        p3.paragraph_format.space_before = Pt(2)
        p3.paragraph_format.space_after = Pt(0)
        run_a = p3.add_run(f'Abbreviations: {abbreviations}')
        set_font(run_a, size=8, italic=True)


def add_page_number_header(section, running_title=None):
    """Add running title (left) and page number (right) to section header."""
    header = section.header
    header.is_linked_to_previous = False
    hp = header.paragraphs[0]
    if running_title:
        run_title = hp.add_run(running_title)
        set_font(run_title, size=9, italic=True)
        run_tab = hp.add_run('\t\t')
        set_font(run_tab, size=9)
    else:
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = hp.add_run()
    fldChar1 = parse_xml(
        '<w:fldChar {} w:fldCharType="begin"/>'.format(nsdecls('w'))
    )
    run._element.append(fldChar1)
    instrText = parse_xml(
        '<w:instrText {} xml:space="preserve"> PAGE </w:instrText>'.format(
            nsdecls('w')
        )
    )
    run._element.append(instrText)
    fldChar2 = parse_xml(
        '<w:fldChar {} w:fldCharType="end"/>'.format(nsdecls('w'))
    )
    run._element.append(fldChar2)
    set_font(run, size=9)


# ── Build ───────────────────────────────────────────────────────────────
def build_supplement():
    doc = Document()

    # Default style
    style = doc.styles['Normal']
    style.font.name = 'Times New Roman'
    style.font.size = Pt(12)
    style.paragraph_format.line_spacing = Pt(24)
    style.paragraph_format.space_after = Pt(0)
    style.paragraph_format.space_before = Pt(0)

    section = doc.sections[0]
    set_margins(section)
    section.page_width = Cm(21.59)
    section.page_height = Cm(27.94)
    set_line_numbering(section)
    add_page_number_header(
        section,
        running_title='Supplementary Materials \u2014 '
                       'Dynamic Treatment Regimes for HFpEF in the ICU'
    )

    # ═══════════════════════════════════════════════════════════════════
    # TITLE PAGE
    # ═══════════════════════════════════════════════════════════════════
    add_para(doc, '', space_after=48)
    add_para(
        doc,
        'Supplementary Materials',
        bold=True, font_size=16,
        alignment=WD_ALIGN_PARAGRAPH.CENTER, space_after=12
    )
    add_para(doc, '', space_after=6)
    add_para(
        doc,
        'Causal Evaluation of Dynamic Treatment Regimes for Critically Ill '
        'Patients with Heart Failure with Preserved Ejection Fraction: '
        'Primary Analysis in MIMIC-IV and External Validation in eICU-CRD',
        bold=True, font_size=13,
        alignment=WD_ALIGN_PARAGRAPH.CENTER, space_after=24
    )

    # ── Table of Contents ────────────────────────────────────────────
    add_para(doc, '', space_after=12)
    add_heading(doc, 'TABLE OF CONTENTS', level=2)
    toc_items = [
        'Supplementary Figures',
        '  sFigure 1: Hemodynamic Phenotype Characteristics',
        '  sFigure 2: IPTW Weight Distribution',
        '  sFigure 3: Cumulative ATE Time Curve (A, B)',
        '  sFigure 4: CATE Distribution and Benefit Profile (A, B)',
        '  sFigure 5: Guideline Strategy Comparison Forest Plot',
        '  sFigure 6: Counterfactual Trajectory Panel',
        '  sFigure 7: Informative Monitoring Dose\u2013Response',
        '  sFigure 8: Action Mechanism Decomposition',
        '  sFigure 9: Hemodynamic Fragility Metrics',
        '  sFigure 10: Continuous HFI \u00d7 HF-Phenotype Interaction',
        'Supplementary Tables',
        '  sTable 1: Guideline Strategy Comparison',
        '  sTable 2: CATE Heterogeneity Analysis',
        '  sTable 3: Time-Varying Cumulative ATE',
        '  sTable 4: NNT Subgroup Analysis',
        '  sTable 5: RCT Sample Size Estimation',
        '  sTable 6: Informative Monitoring Calibration Comparison',
        '  sTable 7: MSM/IPTW Policy Values by Trimming Scheme',
        '  sTable 8: Target Trial Emulation Specification',
        '  sTable 9: Complete Covariate Definitions and Cross-Database Mapping',
        '  sTable 10: Secondary Outcome Directional Analysis',
        '  sTable 11: HFpEF \u00d7 Shock 2\u00d72 Interaction Test Details',
    ]
    for item in toc_items:
        add_para(doc, f'  \u2022  {item}', font_size=11)

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════
    # SUPPLEMENTARY FIGURES
    # ═══════════════════════════════════════════════════════════════════
    add_heading(doc, 'SUPPLEMENTARY FIGURES', level=1)

    # ── sFigure 1 ─────────────────────────────────────────────────────
    doc.add_page_break()
    add_figure(
        doc,
        [FIGURES / '60e_phenotype_radar.png'],
        'sFigure 1',
        'Hemodynamic Phenotype Characteristics',
        'Rule-based four-class hemodynamic phenotype (shock-predominant, '
        'congestion-predominant, mixed, neither) radar plot showing the '
        'characteristic profile of each phenotype across blood pressure, '
        'lactate, fluid balance, and other dimensions.',
        'MAP, mean arterial pressure; SBP, systolic blood pressure; '
        'DBP, diastolic blood pressure.'
    )

    # ── sFigure 2 ─────────────────────────────────────────────────────
    doc.add_page_break()
    add_figure(
        doc,
        [FIGURES / '75a_weight_distribution.png'],
        'sFigure 2',
        'IPTW Weight Distribution',
        'Histogram of stabilized inverse probability of treatment weights '
        'from the 75a MSM/IPTW analysis. Primary trimming scheme [5, 95] '
        'percentile trimmed 10% of observations; effective sample size '
        '(ESS) = 2,492.',
        'ESS, effective sample size; IPTW, inverse probability of treatment '
        'weighting; MSM, marginal structural model.'
    )

    # ── sFigure 3A ────────────────────────────────────────────────────
    doc.add_page_break()
    add_figure(
        doc,
        [FIGURES / 'P3_cumulative_ate_curve.png'],
        'sFigure 3A',
        'Cumulative ATE Time Curve',
        'Marginal ATE increment and cumulative ATE curve at each 2-hour '
        'decision point after T0. Maximum marginal benefit at 0\u20132 h '
        '(\u0394 = \u22120.49 pp, 35% of 24-h total). 82% of total effect '
        'accumulated by 8 h.',
        'ATE, average treatment effect; pp, percentage points; '
        'T0, time zero (ICU admission + 6 hours).'
    )

    # ── sFigure 3B ────────────────────────────────────────────────────
    doc.add_page_break()
    add_figure(
        doc,
        [FIGURES / 'P3_cumulative_ate_by_hf_subtype.png'],
        'sFigure 3B',
        'Cumulative ATE by HF Subtype',
        'Stratified curves showing the cumulative ATE for overall, HFpEF, '
        'HFrEF/HFmrEF, no-HF, shock-predominant, and non-shock subgroups. '
        'HFpEF-to-no-HF effect ratio remained consistent across decision '
        'points (1.89-fold).',
        'ATE, average treatment effect; HF, heart failure; '
        'HFmrEF, heart failure with mildly reduced ejection fraction; '
        'HFpEF, heart failure with preserved ejection fraction; '
        'HFrEF, heart failure with reduced ejection fraction; '
        'pp, percentage points.'
    )

    # ── sFigure 4 ─────────────────────────────────────────────────────
    doc.add_page_break()
    add_figure(
        doc,
        [FIGURES / 'manuscript' / 'sFigure4_composite.png',
         FIGURES / 'manuscript' / 'sFigure4.png',
         FIGURES / 'P2_cate_histogram.png'],
        'sFigure 4',
        'CATE Distribution and Benefit Profile',
        '(A) Histogram of conditional average treatment effects for the '
        'overall cohort and HFpEF subgroup. Overall 71.3% predicted to '
        'benefit (CATE < 0); HFpEF 78.3%. Left panel: overall distribution '
        'with benefit (green) and harm (red) regions. Right panel: '
        'distribution by HF subtype. '
        '(B) Mean CATE (percentage points) by HF subtype and hemodynamic '
        'phenotype. Shock-predominant patients showed larger estimated '
        'effects across all HF subtypes. HFrEF showed the largest estimated '
        'effect in the shock stratum (\u22123.89 pp), while HFpEF showed a '
        'comparable effect to HFmrEF in the non-shock stratum '
        '(\u22121.42 pp each).',
        'CATE, conditional average treatment effect; EF, ejection fraction; '
        'HF, heart failure; HFmrEF, heart failure with mildly reduced '
        'ejection fraction; HFpEF, heart failure with preserved ejection '
        'fraction; HFrEF, heart failure with reduced ejection fraction; '
        'pp, percentage points.',
        width=6.5
    )

    # ── sFigure 5 ─────────────────────────────────────────────────────
    doc.add_page_break()
    add_figure(
        doc,
        [FIGURES / 'P1_guideline_policy_forest_plot.png'],
        'sFigure 5',
        'Guideline Strategy Comparison Forest Plot',
        'Comparison of the study rule policy, ESC-2023 conservative strategy, '
        'and SSC-2021 aggressive fluid resuscitation strategy in the overall '
        'cohort and HFpEF subgroup. In the HFpEF subgroup, SSC-2021 showed '
        'the smallest benefit (ATE = \u22120.25 pp), compared with ESC-2023 '
        '(\u22121.66 pp) and the study rule policy (\u22122.31 pp).',
        'ATE, average treatment effect; ESC, European Society of Cardiology; '
        'HFpEF, heart failure with preserved ejection fraction; '
        'pp, percentage points; SSC, Surviving Sepsis Campaign.'
    )

    # ── sFigure 6 ─────────────────────────────────────────────────────
    doc.add_page_break()
    add_figure(
        doc,
        [FIGURES / 'P5_trajectory_panel.png',
         FIGURES / 'manuscript' / 'sFigure6.png'],
        'sFigure 6',
        'Counterfactual Trajectory Panel',
        'Six representative patient trajectories illustrating observed '
        'versus rule-policy counterfactual treatment sequences. Cases span '
        'the CATE spectrum: (1) HFpEF \u00d7 shock, large benefit; '
        '(2) HFpEF \u00d7 non-shock, moderate benefit; (3) non-HF \u00d7 '
        'shock; (4) non-HF \u00d7 non-shock, minimal benefit; (5) predicted '
        'harm; (6) concordant treatment. '
        'Each panel shows the observed action index (dark bars) and '
        'rule-policy action index (light bars) across 24-hour time bins '
        'post-T0.',
        'CATE, conditional average treatment effect; '
        'HFpEF, heart failure with preserved ejection fraction; '
        'pp, percentage points; T0, time zero (ICU admission + 6 hours).'
    )

    # ── sFigure 7 ─────────────────────────────────────────────────────
    doc.add_page_break()
    add_figure(
        doc,
        [FIGURES / 'P7_monitoring_dose_response.png'],
        'sFigure 7',
        'Informative Monitoring Dose\u2013Response',
        '(A) Calibration comparison between monitoring-aware model (A1) and '
        'monitoring-naive model (A2): 2.03 pp vs 18.97 pp. '
        '(B) Dose\u2013response relationship of ATE by monitoring intensity '
        'quartile (Q1 = \u22120.87 pp to Q4 = \u22122.27 pp, monotonically '
        'increasing).',
        'ATE, average treatment effect; IPMW, inverse probability of '
        'monitoring weighting; pp, percentage points; Q, quartile.'
    )

    # ── sFigure 8 ─────────────────────────────────────────────────────
    doc.add_page_break()
    add_figure(
        doc,
        [FIGURES_PDF / 'sFigure8.png',
         FIGURES / 'P8_action_distribution_comparison.png'],
        'sFigure 8',
        'Action Mechanism Decomposition in the Non-Shock Stratum',
        'Comparison of rule policy versus observed (clinician) action '
        'distributions for HFpEF and no-HF patients in the non-shock '
        'stratum. Left panel: rule policy trigger classification by '
        'priority (P1: MAP < 65 mmHg, P2: lactate > 2, P3\u2013P6: other '
        'hemodynamic criteria). Right panel: observed clinical action '
        'pathway classification (shock resuscitation, decongestion, '
        'maintenance, other). The rule policy triggered hypotension-related '
        'interventions more frequently for HFpEF (20.6% vs 14.0%), while '
        'clinicians administered diuretics at higher rates for HFpEF '
        '(7.8% vs 0.8%), suggesting complementary but distinct approaches '
        'to HFpEF hemodynamic management.',
        'HFpEF, heart failure with preserved ejection fraction; '
        'MAP, mean arterial pressure.',
        width=6.0
    )

    # ── sFigure 9 ─────────────────────────────────────────────────────
    doc.add_page_break()
    add_figure(
        doc,
        [FIGURES_PDF / 'sFigure9.png',
         FIGURES / 'P9_hfi_distribution.png'],
        'sFigure 9',
        'Hemodynamic Fragility Metrics by Heart Failure Subtype',
        'Individual components of the hemodynamic fragility index (HFI) '
        'across heart failure subtypes: (A) number of hypotensive bins '
        '(MAP < 65 mmHg); (B) MAP coefficient of variation; (C) MAP range '
        '(max\u2013min); (D) MAP minimum. HFpEF patients showed consistently '
        'higher fragility across all individual metrics compared with '
        'no-HF patients.',
        'CV, coefficient of variation; HF, heart failure; '
        'HFI, hemodynamic fragility index; HFpEF, heart failure with '
        'preserved ejection fraction; HFrEF, heart failure with reduced '
        'ejection fraction; MAP, mean arterial pressure.'
    )

    # ── sFigure 10 ────────────────────────────────────────────────────
    doc.add_page_break()
    add_figure(
        doc,
        [FIGURES / 'P10_continuous_dose_response.png'],
        'sFigure 10',
        'Continuous HFI \u00d7 HF-Phenotype Interaction Analysis',
        '(A) LOWESS-smoothed estimated ATE as a continuous function of the '
        'hemodynamic fragility index for HFpEF (red) and no-HF (blue) '
        'patients, with 95% bootstrap confidence bands (B = 500). The '
        'HFpEF curve lies consistently below the no-HF curve across the '
        'entire HFI spectrum. Inset: raw mean ITE difference = \u22121.09 pp '
        '(Welch P = 0.0007); WLS \u03b2(HFpEF) = \u22120.79 pp (P < 0.001); '
        'WLS interaction P = 0.556. (B) Sliding-window bin-level ATE '
        'estimates (overlapping windows, overlap factor = 2.5) with 95% '
        'confidence intervals. The approximately parallel trajectories of '
        'both curves confirm that the HFpEF advantage is a constant '
        'intercept shift (~0.8 pp) rather than a divergence at any '
        'particular vulnerability level.',
        'ATE, average treatment effect; CI, confidence interval; '
        'HF, heart failure; HFI, hemodynamic fragility index; '
        'HFpEF, heart failure with preserved ejection fraction; '
        'ITE, individual treatment effect; LOWESS, locally weighted '
        'scatterplot smoothing; pp, percentage points; '
        'WLS, weighted least squares.',
        width=6.0
    )

    # ═══════════════════════════════════════════════════════════════════
    # SUPPLEMENTARY TABLES
    # ═══════════════════════════════════════════════════════════════════
    doc.add_page_break()
    add_heading(doc, 'SUPPLEMENTARY TABLES', level=1)

    # ── sTable 1 ──────────────────────────────────────────────────────
    doc.add_page_break()
    add_heading(doc, 'sTable 1. Guideline Strategy Comparison', level=2)
    add_para(doc, 'Detailed results comparing three strategies (study rule, '
             'ESC-2023, SSC-2021) across overall, HFpEF, HFrEF/HFmrEF, and '
             'no-HF subgroups.', font_size=10, italic=True, space_after=6)
    add_csv_table(doc, TABLES / 'P1_guideline_policy_comparison.csv')
    add_table_footnote(
        doc,
        'Abbreviations: ATE, average treatment effect; '
        'ESC, European Society of Cardiology; HFmrEF, heart failure with '
        'mildly reduced ejection fraction; HFpEF, heart failure with '
        'preserved ejection fraction; HFrEF, heart failure with reduced '
        'ejection fraction; pp, percentage points; SSC, Surviving Sepsis '
        'Campaign.'
    )

    # ── sTable 2 ──────────────────────────────────────────────────────
    doc.add_page_break()
    add_heading(doc, 'sTable 2. CATE Heterogeneity Analysis', level=2)
    add_para(doc, 'Mean/median CATE, benefit/harm proportions, and predictor '
             'importance from gradient boosted regression.',
             font_size=10, italic=True, space_after=6)

    cate_csv = TABLES / 'P2_cate_distribution.csv'
    if cate_csv.exists():
        import pandas as pd
        cate_df = pd.read_csv(cate_csv)
        n_total = len(cate_df)
        n_benefit = (cate_df['cate'] < 0).sum()
        pct_benefit = 100 * n_benefit / n_total
        mean_cate = cate_df['cate_pp'].mean()
        median_cate = cate_df['cate_pp'].median()
        q25 = cate_df['cate_pp'].quantile(0.25)
        q75 = cate_df['cate_pp'].quantile(0.75)

        cate_headers = ['Metric', 'Overall']
        cate_rows = [
            ['N', str(n_total)],
            ['Mean CATE (pp)', f'{mean_cate:.2f}'],
            ['Median CATE (pp)', f'{median_cate:.2f}'],
            ['IQR (pp)', f'[{q25:.2f}, {q75:.2f}]'],
            ['Predicted to benefit (CATE < 0)', f'{n_benefit} ({pct_benefit:.1f}%)'],
            ['Predicted harm (CATE > 0)', f'{n_total - n_benefit} ({100 - pct_benefit:.1f}%)'],
        ]
        add_table_from_rows(doc, cate_headers, cate_rows, font_size=9)
    else:
        add_para(doc, '[P2_cate_distribution.csv not found]', italic=True)
    add_table_footnote(
        doc,
        'Abbreviations: CATE, conditional average treatment effect; '
        'IQR, interquartile range; pp, percentage points.'
    )

    # ── sTable 3 ──────────────────────────────────────────────────────
    doc.add_page_break()
    add_heading(doc, 'sTable 3. Time-Varying Cumulative ATE', level=2)
    add_para(doc, 'Marginal ATE increment and cumulative ATE at each 2-hour '
             'decision point, with subgroup stratification.',
             font_size=10, italic=True, space_after=6)
    add_csv_table(doc, TABLES / 'P3_time_varying_ate.csv')
    add_table_footnote(
        doc,
        'Abbreviations: ATE, average treatment effect; HF, heart failure; '
        'HFpEF, heart failure with preserved ejection fraction; '
        'pp, percentage points; T0, time zero (ICU admission + 6 hours).'
    )

    # ── sTable 4 ──────────────────────────────────────────────────────
    doc.add_page_break()
    add_heading(doc, 'sTable 4. NNT Subgroup Analysis', level=2)
    add_para(doc, 'Number needed to treat by estimator (DM/FQE) and subgroup '
             '(overall/HFpEF/shock/non-shock), with reference ICU RCT '
             'benchmarks.',
             font_size=10, italic=True, space_after=6)
    add_csv_table(doc, TABLES / 'P4_nnt_by_subgroup.csv')
    add_table_footnote(
        doc,
        'Abbreviations: CI, confidence interval; DM, direct method; '
        'FQE, fitted Q-evaluation; HFpEF, heart failure with preserved '
        'ejection fraction; NNT, number needed to treat; '
        'pp, percentage points; RCT, randomized controlled trial.'
    )

    # ── sTable 5 ──────────────────────────────────────────────────────
    doc.add_page_break()
    add_heading(doc, 'sTable 5. Pilot RCT Sample Size Estimation', level=2)
    add_para(doc, 'Sample size requirements under conservative, moderate, and '
             'optimistic scenarios at 80% and 90% power.',
             font_size=10, italic=True, space_after=6)
    add_csv_table(doc, TABLES / 'P6_sample_size_table.csv')
    add_table_footnote(
        doc,
        'Abbreviations: RCT, randomized controlled trial. '
        'Power calculations assume a two-sided \u03b1 = 0.05 and 28-day '
        'all-cause mortality as the primary endpoint.'
    )

    # ── sTable 6 ──────────────────────────────────────────────────────
    doc.add_page_break()
    add_heading(doc, 'sTable 6. Informative Monitoring Calibration '
                'Comparison', level=2)
    add_para(doc, 'Calibration gap between monitoring-aware (A1) and '
             'monitoring-naive (A2) models, monitoring intensity correlates, '
             'and dose\u2013response gradient.',
             font_size=10, italic=True, space_after=6)
    add_csv_table(doc, TABLES / 'P7_calibration_comparison.csv')
    add_table_footnote(
        doc,
        'Abbreviations: ATE, average treatment effect; IPMW, inverse '
        'probability of monitoring weighting; pp, percentage points; '
        'Q, quartile.'
    )

    # ── sTable 7 ──────────────────────────────────────────────────────
    doc.add_page_break()
    add_heading(doc, 'sTable 7. MSM/IPTW Policy Values by Trimming Scheme',
                level=2)
    add_para(doc, 'WIS-estimated rule policy mortality, observed policy '
             'mortality, ATE, and ESS across five trimming brackets.',
             font_size=10, italic=True, space_after=6)
    add_csv_table(doc, TABLES / '75a_trimming_sensitivity.csv')
    add_table_footnote(
        doc,
        'Abbreviations: ATE, average treatment effect; ESS, effective sample '
        'size; IPTW, inverse probability of treatment weighting; '
        'MSM, marginal structural model; pp, percentage points; '
        'WIS, weighted importance sampling.'
    )

    # ── sTable 8 ──────────────────────────────────────────────────────
    doc.add_page_break()
    add_heading(doc, 'sTable 8. Target Trial Emulation Specification',
                level=2)
    add_para(doc, 'Complete specification of the emulated target trial, '
             'including eligibility, treatment strategies, time zero, '
             'decision grid, outcomes, and statistical analysis plan.',
             font_size=10, italic=True, space_after=6)

    tte_rows = [
        ['Eligibility criteria',
         'Adults \u2265 18 years, ICU admission, alive at T0 '
         '(ICU + 6 h), received \u2265 1 hemodynamic intervention '
         'pre-T0. Excludes: comfort care/withdrawal, LVAD/ECMO/'
         'transplant, pregnancy, chronic dialysis/ESRD, active '
         'hemorrhage.'],
        ['Treatment strategies',
         'Strategy A (rule policy): deterministic, priority-ordered '
         'clinical decision algorithm based on MAP, lactate, fluid '
         'balance, and SpO2 thresholds. Strategy B (observed): actual '
         'clinical practice as observed in the database.'],
        ['Treatment assignment',
         'Observational: treatment determined by clinical decision-making. '
         'Estimated via behavior policy classifier.'],
        ['Time zero',
         'ICU admission + 6 hours. Sensitivity: +12 hours.'],
        ['Decision grid',
         '2-hour intervals \u00d7 12 decision points = 24-hour primary '
         'window. Action space: 3 (fluid) \u00d7 3 (vasopressor) \u00d7 '
         '2 (diuretic) = 18 cells, merged to 13 effective cells.'],
        ['Outcome',
         'Primary: 28-day all-cause mortality. Secondary: 7-day '
         'mortality, 72-h IMV escalation, VFD-28, AKI progression, '
         '72-h net fluid balance, ICU-free days.'],
        ['Causal contrast',
         'ATE = E[Y(rule)] \u2013 E[Y(observed)], estimated via DM '
         'rollout (primary deep), WIS (classical anchor), DR, and FQE.'],
        ['Absorbing states',
         'Death and ICU discharge are absorbing states. Post-absorbing '
         'actions set to maintenance (low fluid, no vasopressor, '
         'no diuretic).'],
        ['Censoring',
         'Administrative censoring at 24 h (primary), 48 h and 72 h '
         '(sensitivity). Early ICU discharge handled via exclusion, '
         'bounding, and stratification sensitivity analyses.'],
        ['Statistical analysis',
         'Six model families in parallel. 500 bootstrap resamples. '
         '\u03b1 = 0.05, two-sided.'],
    ]
    table = doc.add_table(rows=len(tte_rows) + 1, cols=2)
    table.style = 'Table Grid'
    for ci, h in enumerate(['Component', 'Specification']):
        cell = table.rows[0].cells[ci]
        cell.text = ''
        p = cell.paragraphs[0]
        run = p.add_run(h)
        set_font(run, size=9, bold=True)
        p.paragraph_format.space_after = Pt(0)
        shading = parse_xml(
            '<w:shd {} w:fill="D9E2F3" w:val="clear"/>'.format(nsdecls('w'))
        )
        cell._element.get_or_add_tcPr().append(shading)
    for ri, (comp, spec) in enumerate(tte_rows):
        for ci, val in enumerate([comp, spec]):
            cell = table.rows[ri + 1].cells[ci]
            cell.text = ''
            p = cell.paragraphs[0]
            run = p.add_run(val)
            set_font(run, size=9)
            p.paragraph_format.space_after = Pt(0)
    add_para(doc, '', space_after=6)
    add_table_footnote(
        doc,
        'Abbreviations: AKI, acute kidney injury; ATE, average treatment '
        'effect; DM, direct method; DR, doubly robust; ECMO, extracorporeal '
        'membrane oxygenation; ESRD, end-stage renal disease; '
        'FQE, fitted Q-evaluation; ICU, intensive care unit; '
        'IMV, invasive mechanical ventilation; LVAD, left ventricular assist '
        'device; MAP, mean arterial pressure; SpO2, peripheral oxygen '
        'saturation; T0, time zero; VFD-28, 28-day ventilator-free days; '
        'WIS, weighted importance sampling.'
    )

    # ── sTable 9 ──────────────────────────────────────────────────────
    doc.add_page_break()
    add_heading(doc, 'sTable 9. Complete Covariate Definitions and '
                'Cross-Database Mapping', level=2)
    add_para(doc, 'Time-varying state variables (L_t): 21 variables across '
             '6 domains, each augmented with 4 missingness features '
             '(observation mask, time since last observation, 24-h frequency, '
             'never-measured indicator) = 105 time-varying features. '
             'Baseline covariates (V): 8 static variables.',
             font_size=10, italic=True, space_after=6)

    cov_rows = [
        ['Hemodynamic', 'MAP', 'mmHg', '20\u2013200', 'Mean per 2-h bin',
         'chartevents / vitalperiodic'],
        ['Hemodynamic', 'SBP', 'mmHg', '40\u2013300', 'Mean per 2-h bin',
         'chartevents / vitalperiodic'],
        ['Hemodynamic', 'DBP', 'mmHg', '10\u2013200', 'Mean per 2-h bin',
         'chartevents / vitalperiodic'],
        ['Hemodynamic', 'Heart rate', 'bpm', '20\u2013250', 'Mean per 2-h bin',
         'chartevents / vitalperiodic'],
        ['Hemodynamic', 'SpO2', '%', '50\u2013100', 'Mean per 2-h bin',
         'chartevents / vitalperiodic'],
        ['Hemodynamic', 'CVP', 'mmHg', '\u22125 to 40', 'Mean per 2-h bin',
         'chartevents / vitalperiodic'],
        ['Perfusion', 'Lactate', 'mmol/L', '0.1\u201330', 'Last per 2-h bin',
         'labevents / lab'],
        ['Perfusion', 'ScvO2', '%', '30\u2013100', 'Last per 2-h bin',
         'chartevents / vitalperiodic'],
        ['Renal', 'Creatinine', 'mg/dL', '0.1\u201325', 'Last per 2-h bin',
         'labevents / lab'],
        ['Renal', 'BUN', 'mg/dL', '1\u2013200', 'Last per 2-h bin',
         'labevents / lab'],
        ['Renal', 'Urine output', 'mL/2 h', '0\u20135000', 'Sum per 2-h bin',
         'outputevents / intakeoutput'],
        ['Renal', 'Fluid balance', 'mL', '\u221210 000 to 20 000', 'Cumulative',
         'Derived'],
        ['Respiratory', 'RR', 'breaths/min', '4\u201360', 'Mean per 2-h bin',
         'chartevents / vitalperiodic'],
        ['Respiratory', 'FiO2', '%', '21\u2013100', 'Max per 2-h bin',
         'chartevents / respiratorycharting'],
        ['Respiratory', 'P/F ratio', 'mmHg', '30\u2013600', 'Derived',
         'Derived'],
        ['Respiratory', 'Vent mode', 'categorical', '0\u20134',
         'Last per 2-h bin', 'chartevents / respiratorycharting'],
        ['Metabolic', 'pH', 'units', '6.8\u20137.7', 'Last per 2-h bin',
         'labevents / lab'],
        ['Metabolic', 'Bicarbonate', 'mEq/L', '5\u201350', 'Last per 2-h bin',
         'labevents / lab'],
        ['Metabolic', 'Potassium', 'mEq/L', '1.5\u20139', 'Last per 2-h bin',
         'labevents / lab'],
        ['Metabolic', 'Sodium', 'mEq/L', '110\u2013170', 'Last per 2-h bin',
         'labevents / lab'],
        ['Metabolic', 'Glucose', 'mg/dL', '20\u20131000', 'Last per 2-h bin',
         'labevents / lab'],
    ]
    cov_headers = ['Domain', 'Variable', 'Unit', 'Valid Range',
                   'Aggregation', 'MIMIC / eICU Source']
    add_table_from_rows(doc, cov_headers, cov_rows, font_size=8)

    add_para(doc, '', space_after=6)
    add_para(doc, 'Baseline Covariates (V):', bold=True, font_size=10,
             space_after=3)
    add_para(doc, 'Age (years), Sex (M/F), Weight (kg), Height (cm), '
             'BMI (kg/m\u00b2), HFpEF ascertainment method (ICD+Echo / '
             'ICD-only), Ejection fraction (%), Comorbidity burden '
             '(Charlson score).', font_size=10)
    add_table_footnote(
        doc,
        'Abbreviations: BMI, body mass index; BUN, blood urea nitrogen; '
        'CVP, central venous pressure; DBP, diastolic blood pressure; '
        'eICU, eICU Collaborative Research Database; FiO2, fraction of '
        'inspired oxygen; ICD, International Classification of Diseases; '
        'MAP, mean arterial pressure; MIMIC, Medical Information Mart for '
        'Intensive Care; P/F, PaO2/FiO2 ratio; RR, respiratory rate; '
        'SBP, systolic blood pressure; ScvO2, central venous oxygen '
        'saturation; SpO2, peripheral oxygen saturation.'
    )

    # ── sTable 10 ─────────────────────────────────────────────────────
    doc.add_page_break()
    add_heading(doc, 'sTable 10. Secondary Outcome Directional Analysis',
                level=2)
    add_para(doc, 'IPTW-weighted analysis of 6 secondary outcomes stratified '
             'by rule policy concordance rate. Four outcomes directionally '
             'consistent with primary analysis; 2 showed adverse direction '
             '(IMV escalation, AKI progression).',
             font_size=10, italic=True, space_after=6)

    sec_headers = ['Outcome', 'High Concordance', 'Low Concordance',
                   'Difference (pp)', 'Direction']
    sec_rows = [
        ['7-day mortality', '5.1%', '10.2%', '\u22125.16', 'Favorable'],
        ['ICU-free days (28 d)', '+1.50 days', 'Reference', '+1.50',
         'Favorable'],
        ['24-h fluid balance', '\u22120.14 L', 'Reference', '\u22120.14 L',
         'Favorable'],
        ['VFD-28', '+1.57 days', 'Reference', '+1.57', 'Favorable'],
        ['IMV escalation (72 h)', '+2.14 pp', 'Reference', '+2.14',
         'Adverse'],
        ['AKI progression', '+0.99 pp', 'Reference', '+0.99', 'Adverse'],
    ]
    add_table_from_rows(doc, sec_headers, sec_rows, font_size=9)
    add_table_footnote(
        doc,
        'Abbreviations: AKI, acute kidney injury; ICU, intensive care unit; '
        'IMV, invasive mechanical ventilation; IPTW, inverse probability of '
        'treatment weighting; pp, percentage points; VFD-28, 28-day '
        'ventilator-free days. "Favorable" indicates the direction is '
        'consistent with the primary analysis (lower mortality); "Adverse" '
        'indicates the opposite direction.'
    )

    # ── sTable 11 ─────────────────────────────────────────────────────
    doc.add_page_break()
    add_heading(doc, 'sTable 11. HFpEF \u00d7 Shock 2\u00d72 Interaction '
                'Test Details', level=2)
    add_para(doc, 'ATE point estimates, bootstrap 95% CI, SE for four '
             'stratified cells, non-shock difference, effect ratio, and '
             'difference-in-differences (DiD).',
             font_size=10, italic=True, space_after=6)

    int_headers = ['Cell', 'n (test set)', 'ATE (pp)', '95% CI', 'SE']
    int_rows = [
        ['HFpEF \u00d7 shock', '141', '\u22123.23', '[\u22124.00, \u22122.45]', '0.38'],
        ['HFpEF \u00d7 non-shock', '163', '\u22121.44', '[\u22121.88, \u22120.90]', '0.26'],
        ['no-HF \u00d7 shock', '1,539', '\u22122.48', '[\u22122.71, \u22122.23]', '0.13'],
        ['no-HF \u00d7 non-shock', '3,108', '\u22120.52', '[\u22120.59, \u22120.46]', '0.03'],
        ['Non-shock diff (HFpEF \u2212 no-HF)', '\u2014', '\u22120.92', '[\u22121.37, \u22120.36]', '\u2014'],
        ['Effect ratio (non-shock)', '\u2014', '2.76\u00d7', '[1.67\u00d7, 3.76\u00d7]', '\u2014'],
        ['DiD', '\u2014', '\u2014', '[\u22121.08, +0.84]', 'p = 0.412'],
        ['Interaction p (non-shock)', '\u2014', '\u2014', '\u2014', 'p < 0.001'],
    ]
    add_table_from_rows(doc, int_headers, int_rows, font_size=9)
    add_table_footnote(
        doc,
        'Abbreviations: ATE, average treatment effect; CI, confidence '
        'interval; DiD, difference-in-differences; HF, heart failure; '
        'HFpEF, heart failure with preserved ejection fraction; '
        'pp, percentage points; SE, standard error. '
        'Bootstrap resamples B = 500. The interaction p-value tests '
        'whether the HFpEF vs no-HF difference is significant within '
        'the non-shock stratum.'
    )

    # ── Save ─────────────────────────────────────────────────────────
    doc.save(str(OUTPUT))
    print(f"Supplement saved to: {OUTPUT}")
    print(f"File size: {OUTPUT.stat().st_size / 1024:.1f} KB")


if __name__ == '__main__':
    build_supplement()
