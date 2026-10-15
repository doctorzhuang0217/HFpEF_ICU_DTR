#!/usr/bin/env python3
"""Build submission-ready main manuscript DOCX from English markdown parts.

Output: manuscript/HFpEF_ICU_DTR_manuscript.docx

Formatting follows common high-impact medical journal requirements
(journal style):
  - Times New Roman 12pt, double-spaced
  - 1-inch (2.54 cm) margins on all sides
  - Continuous line numbering
  - Page numbers in header (right-aligned)
  - Structured abstract
  - Tables after references, each on a new page with abbreviation footnotes
  - Figure legends on separate page(s) with abbreviation footnotes
  - Figures at end, one per page with legend below
  - Abbreviations list
  - Word count on title page
"""
import csv
import os
import re
from pathlib import Path

from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml

ROOT = Path(__file__).resolve().parent.parent
MANUSCRIPT_DIR = ROOT / "manuscript"
FIGURES_PDF_DIR = MANUSCRIPT_DIR / "figures_pdf"
FIGURES_PNG_DIR = ROOT / "figures" / "submission"
FIGURES_SRC = ROOT / "figures"
TABLES_DIR = ROOT / "tables"
OUTPUT = MANUSCRIPT_DIR / "HFpEF_ICU_DTR_manuscript.docx"


# ── helpers ─────────────────────────────────────────────────────────────
def set_line_numbering(section):
    """Add continuous line numbering to a section."""
    sectPr = section._sectPr
    ln_num = parse_xml(
        '<w:lnNumType {} w:countBy="1" w:restart="continuous"/>'.format(
            nsdecls('w')
        )
    )
    sectPr.append(ln_num)


def set_margins(section, top=2.54, bottom=2.54, left=2.54, right=2.54):
    section.top_margin = Cm(top)
    section.bottom_margin = Cm(bottom)
    section.left_margin = Cm(left)
    section.right_margin = Cm(right)


def set_font(run, name='Times New Roman', size=12, bold=False, italic=False):
    """Set font properties on a run, including East Asian fallback."""
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


def add_paragraph(doc, text, bold=False, italic=False,
                  alignment=None, space_after=None, font_size=12,
                  keep_with_next=False, first_line_indent=None,
                  left_indent=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    set_font(run, size=font_size, bold=bold, italic=italic)
    pf = p.paragraph_format
    pf.line_spacing = Pt(24)  # double spacing
    pf.space_before = Pt(0)
    if alignment:
        pf.alignment = alignment
    if space_after is not None:
        pf.space_after = Pt(space_after)
    else:
        pf.space_after = Pt(0)
    if keep_with_next:
        pf.keep_with_next = True
    if first_line_indent is not None:
        pf.first_line_indent = Cm(first_line_indent)
    if left_indent is not None:
        pf.left_indent = Cm(left_indent)
    return p


def add_heading_custom(doc, text, level=1):
    """Add a heading with Times New Roman formatting."""
    sizes = {1: 14, 2: 13, 3: 12}
    fs = sizes.get(level, 12)
    p = add_paragraph(doc, text, bold=True, font_size=fs,
                      space_after=6, keep_with_next=True)
    if level == 1:
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(12)
    return p


def add_table_from_rows(doc, headers, rows, font_size=9):
    """Add a formatted table with header shading."""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Table Grid'
    # Header row
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
    # Data rows
    for ri, row_data in enumerate(rows):
        for ci, val in enumerate(row_data):
            if ci >= len(headers):
                break
            cell = table.rows[ri + 1].cells[ci]
            cell.text = ''
            p = cell.paragraphs[0]
            run = p.add_run(str(val))
            set_font(run, size=font_size)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.space_before = Pt(0)
    return table


def add_table_footnote(doc, text, font_size=8):
    """Add an abbreviation/footnote paragraph below a table."""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = Pt(18)
    pf.space_before = Pt(4)
    pf.space_after = Pt(0)
    run = p.add_run(text)
    set_font(run, size=font_size, italic=True)
    return p


def add_csv_table(doc, csv_path, font_size=9):
    """Read CSV and add as a formatted Word table."""
    if not csv_path.exists():
        add_paragraph(doc, f'[Data file not found: {csv_path.name}]',
                      italic=True)
        return
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return
    headers = rows[0]
    data = rows[1:]
    return add_table_from_rows(doc, headers, data, font_size)


def parse_md_table(text_block):
    """Parse a markdown table into headers and rows."""
    lines = [l.strip() for l in text_block.strip().split('\n') if l.strip()]
    if len(lines) < 3:
        return None, None
    headers = [c.strip() for c in lines[0].split('|') if c.strip()]
    rows = []
    for line in lines[2:]:  # skip separator
        cols = [c.strip() for c in line.split('|') if c.strip()]
        if cols:
            rows.append(cols)
    return headers, rows


# ── read all parts ──────────────────────────────────────────────────────
def read_part(filename):
    path = MANUSCRIPT_DIR / filename
    return path.read_text(encoding='utf-8')


def process_inline_formatting(paragraph, text):
    """Process bold and italic markdown inline formatting."""
    parts = re.split(r'\*\*(.+?)\*\*', text)
    for i, part in enumerate(parts):
        if not part:
            continue
        if i % 2 == 1:  # bold part
            run = paragraph.add_run(part)
            set_font(run, size=12, bold=True)
        else:
            italic_parts = re.split(r'\*(.+?)\*', part)
            for j, ip in enumerate(italic_parts):
                if not ip:
                    continue
                run = paragraph.add_run(ip)
                if j % 2 == 1:
                    set_font(run, size=12, italic=True)
                else:
                    set_font(run, size=12)
    paragraph.paragraph_format.line_spacing = Pt(24)


def build_body_from_md(doc, md_text, skip_title=False):
    """Parse markdown and add to document with proper formatting."""
    lines = md_text.split('\n')
    i = 0
    in_table = False
    table_lines = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            i += 1
            continue

        # Skip title if requested
        if skip_title and stripped.startswith('# ') and not stripped.startswith('## '):
            i += 1
            continue

        # Skip horizontal rules
        if stripped == '---':
            i += 1
            continue

        # Detect table start
        if '|' in stripped and not in_table:
            if i + 1 < len(lines) and re.match(r'^[\s|:-]+$', lines[i + 1].strip()):
                in_table = True
                table_lines = [stripped]
                i += 1
                continue

        if in_table:
            if '|' in stripped:
                table_lines.append(stripped)
                i += 1
                continue
            else:
                in_table = False
                table_text = '\n'.join(table_lines)
                headers, rows = parse_md_table(table_text)
                if headers and rows:
                    add_table_from_rows(doc, headers, rows)
                    add_paragraph(doc, '', space_after=6)
                table_lines = []
                continue

        # Headings
        if stripped.startswith('### '):
            add_heading_custom(doc, stripped[4:], level=3)
            i += 1
            continue
        if stripped.startswith('## '):
            add_heading_custom(doc, stripped[3:], level=2)
            i += 1
            continue
        if stripped.startswith('# '):
            add_heading_custom(doc, stripped[2:], level=1)
            i += 1
            continue

        # Bullet points
        if stripped.startswith('- **') or stripped.startswith('- '):
            p = doc.add_paragraph()
            p.paragraph_format.line_spacing = Pt(24)
            p.paragraph_format.left_indent = Cm(1.27)
            p.paragraph_format.first_line_indent = Cm(-0.63)
            p.paragraph_format.space_after = Pt(0)
            process_inline_formatting(p, '\u2022 ' + stripped[2:])
            i += 1
            continue

        # Regular paragraph - collect continuation lines
        para_text = stripped
        while i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if (not next_line or next_line.startswith('#') or
                next_line.startswith('---') or next_line.startswith('- ') or
                next_line.startswith('|') or next_line.startswith('```')):
                break
            para_text += ' ' + next_line
            i += 1

        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        process_inline_formatting(p, para_text)
        i += 1

    # Flush remaining table
    if table_lines:
        table_text = '\n'.join(table_lines)
        headers, rows = parse_md_table(table_text)
        if headers and rows:
            add_table_from_rows(doc, headers, rows)


def count_words(text):
    """Count words in markdown text, excluding formatting markers."""
    clean = re.sub(r'\*+', '', text)
    clean = re.sub(r'#+ ', '', clean)
    clean = re.sub(r'\|[^\n]*\|', '', clean)
    clean = re.sub(r'---', '', clean)
    words = clean.split()
    return len(words)


def add_page_number_header(section, running_title=None):
    """Add running title (left) and page number (right) to section header."""
    header = section.header
    header.is_linked_to_previous = False
    hp = header.paragraphs[0]
    # Add running title on the left
    if running_title:
        run_title = hp.add_run(running_title)
        set_font(run_title, size=9, italic=True)
        # Add tab to right-align page number
        run_tab = hp.add_run('\t\t')
        set_font(run_tab, size=9)
    else:
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    # Add page number field
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


def find_figure(name_variants):
    """Try multiple paths for a figure file, return first that exists."""
    for v in name_variants:
        if v.exists():
            return v
    return None


# ── main build ──────────────────────────────────────────────────────────
def build_manuscript():
    doc = Document()

    # ── Default style ───────────────────────────────────────────────────
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(12)
    pf = style.paragraph_format
    pf.line_spacing = Pt(24)
    pf.space_after = Pt(0)
    pf.space_before = Pt(0)

    # ── Section setup ───────────────────────────────────────────────────
    section = doc.sections[0]
    set_margins(section)
    section.page_width = Cm(21.59)   # US Letter
    section.page_height = Cm(27.94)
    set_line_numbering(section)
    add_page_number_header(
        section,
        running_title='Dynamic Treatment Regimes for HFpEF in the ICU'
    )

    # ── Read all parts for word count ────────────────────────────────
    part1 = read_part('en_part1_title_abstract_intro.md')
    part2 = read_part('en_part2_methods.md')
    part3 = read_part('en_part3_results.md')
    part4 = read_part('en_part4_discussion_refs.md')

    # Count words (abstract + body, excluding references/legends)
    # Separate abstract body from keywords
    abstract_text = re.search(r'## ABSTRACT\n(.+?)(?=\n### Keywords)', part1, re.DOTALL)
    if not abstract_text:
        abstract_text = re.search(r'## ABSTRACT\n(.+?)(?=\n## INTRODUCTION)', part1, re.DOTALL)
    keywords_text = re.search(r'### Keywords\n\n(.+?)(?=\n---)', part1, re.DOTALL)
    intro_text = re.search(r'## INTRODUCTION\n(.+)', part1, re.DOTALL)
    methods_text = re.search(r'## METHODS\n(.+)', part2, re.DOTALL)
    results_text = re.search(r'## RESULTS\n(.+)', part3, re.DOTALL)
    disc_text = re.search(r'## DISCUSSION\n(.+?)(?=\n---\n\n## DATA AVAILABILITY)', part4, re.DOTALL)

    abstract_wc = count_words(abstract_text.group(1)) if abstract_text else 0
    body_wc = sum(
        count_words(m.group(1)) if m else 0
        for m in [intro_text, methods_text, results_text, disc_text]
    )
    total_wc = abstract_wc + body_wc

    # ═══════════════════════════════════════════════════════════════════
    # TITLE PAGE
    # ═══════════════════════════════════════════════════════════════════
    add_paragraph(doc, '', space_after=24)

    # Manuscript type
    add_paragraph(
        doc, 'Original Article',
        bold=True, font_size=12,
        alignment=WD_ALIGN_PARAGRAPH.CENTER, space_after=24
    )

    # Title
    add_paragraph(
        doc,
        'Causal Evaluation of Dynamic Treatment Regimes for Critically Ill '
        'Patients with Heart Failure with Preserved Ejection Fraction: '
        'Primary Analysis in MIMIC-IV and External Validation in eICU-CRD',
        bold=True, font_size=16,
        alignment=WD_ALIGN_PARAGRAPH.CENTER, space_after=12
    )

    add_paragraph(doc, '', space_after=6)

    # Running title
    add_paragraph(
        doc,
        'Short Title: Dynamic Treatment Regimes for HFpEF in the ICU',
        italic=True, font_size=11,
        alignment=WD_ALIGN_PARAGRAPH.CENTER, space_after=18
    )

    # Authors
    add_paragraph(
        doc, '[Author names to be inserted]',
        bold=True, font_size=12,
        alignment=WD_ALIGN_PARAGRAPH.CENTER, space_after=6
    )

    add_paragraph(
        doc, '[Author affiliations to be inserted]',
        font_size=11,
        alignment=WD_ALIGN_PARAGRAPH.CENTER, space_after=18
    )

    # Corresponding author
    add_paragraph(
        doc,
        'Corresponding Author:',
        bold=True, font_size=11,
        alignment=WD_ALIGN_PARAGRAPH.CENTER, space_after=2
    )
    add_paragraph(
        doc,
        '[Name, degree(s)]\n'
        '[Department, Institution]\n'
        '[Address]\n'
        '[Email] | [Telephone]',
        font_size=11,
        alignment=WD_ALIGN_PARAGRAPH.CENTER, space_after=18
    )

    # Word counts
    add_paragraph(
        doc, f'Abstract Word Count: {abstract_wc}',
        alignment=WD_ALIGN_PARAGRAPH.CENTER, space_after=2
    )
    add_paragraph(
        doc, f'Manuscript Word Count: {body_wc}',
        alignment=WD_ALIGN_PARAGRAPH.CENTER, space_after=2
    )
    add_paragraph(
        doc, 'Tables: 4  |  Figures: 5  |  Supplementary Figures: 10  |  '
             'Supplementary Tables: 11',
        alignment=WD_ALIGN_PARAGRAPH.CENTER, space_after=2
    )
    add_paragraph(
        doc, 'References: 45',
        alignment=WD_ALIGN_PARAGRAPH.CENTER, space_after=6
    )

    # Subject codes
    add_paragraph(
        doc,
        'Subject Codes: Heart Failure, Critical Care, '
        'Epidemiology, Clinical Studies',
        italic=True, font_size=10,
        alignment=WD_ALIGN_PARAGRAPH.CENTER, space_after=12
    )

    # Page break
    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════
    # ABSTRACT
    # ═══════════════════════════════════════════════════════════════════
    if abstract_text:
        add_heading_custom(doc, 'ABSTRACT', level=1)
        build_body_from_md(doc, abstract_text.group(1))
        # Keywords (separate from abstract body)
        if keywords_text:
            add_paragraph(doc, '', space_after=6)
            p = doc.add_paragraph()
            p.paragraph_format.line_spacing = Pt(24)
            p.paragraph_format.space_after = Pt(0)
            run_label = p.add_run('Key Words: ')
            set_font(run_label, size=12, bold=True)
            run_kw = p.add_run(keywords_text.group(1).strip())
            set_font(run_kw, size=12, italic=True)
        doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════
    # INTRODUCTION
    # ═══════════════════════════════════════════════════════════════════
    if intro_text:
        add_heading_custom(doc, 'INTRODUCTION', level=1)
        build_body_from_md(doc, intro_text.group(1))

    # ═══════════════════════════════════════════════════════════════════
    # METHODS
    # ═══════════════════════════════════════════════════════════════════
    if methods_text:
        add_heading_custom(doc, 'METHODS', level=1)
        build_body_from_md(doc, methods_text.group(1))

    # ═══════════════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════════════
    if results_text:
        add_heading_custom(doc, 'RESULTS', level=1)
        build_body_from_md(doc, results_text.group(1))

    # ═══════════════════════════════════════════════════════════════════
    # DISCUSSION
    # ═══════════════════════════════════════════════════════════════════
    if disc_text:
        add_heading_custom(doc, 'DISCUSSION', level=1)
        build_body_from_md(doc, disc_text.group(1))

    # ── Data Availability ────────────────────────────────────────────
    da_match = re.search(
        r'## DATA AVAILABILITY STATEMENT\n(.+?)(?=\n## ETHICS STATEMENT)',
        part4, re.DOTALL
    )
    if da_match:
        add_heading_custom(doc, 'DATA AVAILABILITY STATEMENT', level=1)
        build_body_from_md(doc, da_match.group(1))

    # ── Ethics ───────────────────────────────────────────────────────
    eth_match = re.search(
        r'## ETHICS STATEMENT\n(.+?)(?=\n## CONFLICT OF INTEREST)',
        part4, re.DOTALL
    )
    if eth_match:
        add_heading_custom(doc, 'ETHICS STATEMENT', level=1)
        build_body_from_md(doc, eth_match.group(1))

    # ── COI ──────────────────────────────────────────────────────────
    coi_match = re.search(
        r'## CONFLICT OF INTEREST\n(.+?)(?=\n## FUNDING)',
        part4, re.DOTALL
    )
    if coi_match:
        add_heading_custom(doc, 'CONFLICT OF INTEREST', level=1)
        build_body_from_md(doc, coi_match.group(1))

    # ── Funding ──────────────────────────────────────────────────────
    fund_match = re.search(
        r'## FUNDING\n(.+?)(?=\n## AUTHOR CONTRIBUTIONS)',
        part4, re.DOTALL
    )
    if fund_match:
        add_heading_custom(doc, 'FUNDING', level=1)
        build_body_from_md(doc, fund_match.group(1))

    # ── Author Contributions ─────────────────────────────────────────
    auth_match = re.search(
        r'## AUTHOR CONTRIBUTIONS.*?\n(.+?)(?=\n---)',
        part4, re.DOTALL
    )
    if auth_match:
        add_heading_custom(doc, 'AUTHOR CONTRIBUTIONS', level=1)
        build_body_from_md(doc, auth_match.group(1))

    # ═══════════════════════════════════════════════════════════════════
    # CLINICAL PERSPECTIVES
    # ═══════════════════════════════════════════════════════════════════
    doc.add_page_break()
    add_heading_custom(doc, 'CLINICAL PERSPECTIVES', level=1)

    # Competency in Medical Knowledge
    p_cmk = doc.add_paragraph()
    p_cmk.paragraph_format.line_spacing = Pt(24)
    p_cmk.paragraph_format.space_after = Pt(6)
    run_h1 = p_cmk.add_run('Competency in Medical Knowledge: ')
    set_font(run_h1, size=12, bold=True)
    run_t1 = p_cmk.add_run(
        'HFpEF in critically ill patients is associated with distinct '
        'hemodynamic vulnerability, including a higher prevalence of '
        'subclinical hypotension and greater hemodynamic lability compared '
        'with non-heart-failure patients. A phenotype-aware dynamic treatment '
        'regime that jointly optimizes fluids, vasopressors, and diuretics '
        'was associated with lower 28-day mortality, with a 2.8-fold '
        'treatment effect ratio for HFpEF versus non-HF in the non-shock '
        'stratum.'
    )
    set_font(run_t1, size=12)

    # Translational Outlook
    p_to = doc.add_paragraph()
    p_to.paragraph_format.line_spacing = Pt(24)
    p_to.paragraph_format.space_after = Pt(6)
    run_h2 = p_to.add_run('Translational Outlook: ')
    set_font(run_h2, size=12, bold=True)
    run_t2 = p_to.add_run(
        'These observational findings support prospective evaluation of '
        'phenotype-aware hemodynamic management strategies in ICU patients '
        'with HFpEF. The identified "golden time window" (first 2 hours '
        'contributing 35% of effect) and the estimated sample size for a '
        'confirmatory trial (2,500 patients at 80% power) provide a '
        'foundation for designing targeted randomized controlled trials. '
        'The informative monitoring framework developed in this study may '
        'serve as a methodological template for future ICU causal inference '
        'studies.'
    )
    set_font(run_t2, size=12)

    # ═══════════════════════════════════════════════════════════════════
    # REFERENCES
    # ═══════════════════════════════════════════════════════════════════
    doc.add_page_break()
    add_heading_custom(doc, 'REFERENCES', level=1)

    ref_match = re.search(
        r'## REFERENCES\n(.+?)(?=\n---|\n## FIGURE)',
        part4, re.DOTALL
    )
    if ref_match:
        ref_text = ref_match.group(1).strip()
        refs = re.split(r'\n(?=\d+\.)', ref_text)
        for ref in refs:
            ref = ref.strip()
            if ref:
                p = doc.add_paragraph()
                p.paragraph_format.line_spacing = Pt(24)
                p.paragraph_format.left_indent = Cm(1.27)
                p.paragraph_format.first_line_indent = Cm(-1.27)
                p.paragraph_format.space_after = Pt(0)
                run = p.add_run(ref)
                set_font(run, size=11)

    # ═══════════════════════════════════════════════════════════════════
    # TABLES (each on new page, with title, footnote, and abbreviations)
    # ═══════════════════════════════════════════════════════════════════

    # ── Table 1: Baseline Characteristics ────────────────────────────
    doc.add_page_break()
    add_heading_custom(doc, 'Table 1. Baseline Characteristics of the Study Cohort', level=2)
    add_paragraph(doc, 'Baseline characteristics of the overall cohort '
                  '(N = 38,981) and subgroups. Values are median [IQR] for '
                  'continuous variables and n (%) for categorical variables.',
                  font_size=10, italic=True, space_after=6)

    t1_csv = TABLES_DIR / '35c_case_mix_compare.csv'
    if t1_csv.exists():
        import pandas as pd
        df = pd.read_csv(t1_csv)
        nohf = df[df['comparator'] == 'no_hf']
        hfref = df[df['comparator'] == 'hfref_hfmref']

        t1_headers = ['Variable', 'HFpEF (n=2,080)', 'HFrEF/HFmrEF (n=2,211)',
                      'No HF (n=31,184)', 'SMD (vs No HF)']
        t1_rows = []
        key_vars = [
            'Age (years)', 'Female (%)', 'Ejection fraction (%)',
            'ICU LOS (hours)', 'Pre-T0 min MAP (mmHg)',
            'Pre-T0 mean MAP (mmHg)', 'Pre-T0 max lactate (mmol/L)',
            'Pre-T0 fluid balance (L)', 'Hospital mortality (%)',
        ]
        for var in key_vars:
            hfpef_val = ''
            hfref_val = ''
            nohf_val = ''
            smd = ''
            row_nohf = nohf[nohf['variable'] == var]
            row_hfref = hfref[hfref['variable'] == var]
            if len(row_nohf) > 0:
                hfpef_val = row_nohf.iloc[0]['hfpef_value']
                nohf_val = row_nohf.iloc[0]['comp_value']
                smd = row_nohf.iloc[0]['smd']
            if len(row_hfref) > 0:
                hfref_val = row_hfref.iloc[0]['comp_value']
            t1_rows.append([var, str(hfpef_val), str(hfref_val),
                            str(nohf_val), str(smd)])

        for reason in ['sepsis', 'respiratory', 'post_op', 'aki', 'other']:
            var = f'Reason for ICU: {reason}'
            row_nohf = nohf[nohf['variable'] == var]
            row_hfref = hfref[hfref['variable'] == var]
            hfpef_val = row_nohf.iloc[0]['hfpef_value'] if len(row_nohf) > 0 else ''
            nohf_val = row_nohf.iloc[0]['comp_value'] if len(row_nohf) > 0 else ''
            hfref_val = row_hfref.iloc[0]['comp_value'] if len(row_hfref) > 0 else ''
            smd = row_nohf.iloc[0]['smd'] if len(row_nohf) > 0 else ''
            display_name = f'  {reason.replace("_", " ").title()}'
            t1_rows.append([display_name, str(hfpef_val), str(hfref_val),
                            str(nohf_val), str(smd)])

        for syn in ['shock', 'congestion', 'mixed', 'neither']:
            var = f'Dominant syndrome: {syn}'
            row_nohf = nohf[nohf['variable'] == var]
            row_hfref = hfref[hfref['variable'] == var]
            hfpef_val = row_nohf.iloc[0]['hfpef_value'] if len(row_nohf) > 0 else ''
            nohf_val = row_nohf.iloc[0]['comp_value'] if len(row_nohf) > 0 else ''
            hfref_val = row_hfref.iloc[0]['comp_value'] if len(row_hfref) > 0 else ''
            smd = row_nohf.iloc[0]['smd'] if len(row_nohf) > 0 else ''
            display_name = f'  {syn.title()}'
            t1_rows.append([display_name, str(hfpef_val), str(hfref_val),
                            str(nohf_val), str(smd)])

        add_table_from_rows(doc, t1_headers, t1_rows, font_size=8)
    else:
        add_paragraph(doc, '[Table 1 data not found \u2014 to be populated]',
                      italic=True)

    add_table_footnote(
        doc,
        'Abbreviations: AKI, acute kidney injury; HF, heart failure; '
        'HFmrEF, heart failure with mildly reduced ejection fraction; '
        'HFpEF, heart failure with preserved ejection fraction; '
        'HFrEF, heart failure with reduced ejection fraction; '
        'ICU, intensive care unit; IQR, interquartile range; '
        'LOS, length of stay; MAP, mean arterial pressure; '
        'SMD, standardized mean difference; T0, time zero '
        '(ICU admission + 6 hours).'
    )

    # ── Table 2: Cross-Model ATE Comparison ──────────────────────────
    doc.add_page_break()
    add_heading_custom(doc, 'Table 2. Cross-Model Average Treatment Effect Comparison', level=2)
    add_paragraph(doc, 'ATE estimates (percentage points) for 28-day all-cause '
                  'mortality across six model families with pre-registered role '
                  'hierarchy. CI = 95% bootstrap confidence interval (B = 500).',
                  font_size=10, italic=True, space_after=6)

    t2_headers = ['Model', 'Role', 'Estimator', 'ATE (pp)', '95% CI']
    t2_rows = [
        ['MSM/IPTW', 'Classical anchor', 'WIS', '\u22124.87',
         '[\u22125.68, \u22124.05]'],
        ['G-Transformer', 'Primary deep', 'DM rollout', '\u22121.37',
         '[\u22121.49, \u22121.26]'],
        ['G-Transformer', 'Primary deep', 'DR', '\u22120.68', '\u2014'],
        ['G-Transformer', 'Primary deep', 'FQE', '\u22123.64', '\u2014'],
        ['G-Net', 'Confirmatory', 'DM', '\u22120.22',
         '[\u22120.31, \u22120.13]'],
        ['G-Net', 'Confirmatory', 'DR', '\u22120.43', '\u2014'],
        ['G-Net', 'Confirmatory', 'FQE', '\u22121.80', '\u2014'],
        ['ACTIN', 'Confirmatory', 'DR', '\u22120.79', '\u2014'],
        ['ACTIN', 'Confirmatory', 'FQE', '\u22121.97', '\u2014'],
        ['CT-Legacy', 'Ablation', 'FQE', '\u22122.43', '\u2014'],
        ['Mamba-CDSP', 'Optional', 'FQE', '\u22122.16', '\u2014'],
    ]
    add_table_from_rows(doc, t2_headers, t2_rows, font_size=9)

    add_table_footnote(
        doc,
        'Abbreviations: ACTIN, adversarial counterfactual temporal inference '
        'network; ATE, average treatment effect; CI, confidence interval; '
        'CT-Legacy, causal Transformer (legacy architecture); '
        'DM, direct method; DR, doubly robust; FQE, fitted Q-evaluation; '
        'IPTW, inverse probability of treatment weighting; '
        'Mamba-CDSP, Mamba counterfactual dynamic state prediction; '
        'MSM, marginal structural model; pp, percentage points; '
        'WIS, weighted importance sampling. '
        'The em dash (\u2014) indicates that bootstrap CIs were not computed '
        'for non-primary estimators.'
    )

    # ── Table 3: Sensitivity Analysis Summary ────────────────────────
    doc.add_page_break()
    add_heading_custom(doc, 'Table 3. Summary of Pre-Specified Sensitivity Analyses', level=2)
    add_paragraph(doc, 'Methods, key findings, and final determinations for '
                  'nine pre-specified sensitivity analyses.',
                  font_size=10, italic=True, space_after=6)

    t3_headers = ['Analysis', 'Key Finding', 'Determination']
    t3_rows = [
        ['Early discharge (85a)', 'ATE stable across exclusion/bounding/stratification',
         'Robust'],
        ['Sequential deconfounding (85b)',
         'ATE attenuates Q1 (\u22122.43 pp) to Q4 (\u22120.54 pp)', 'Sensitive'],
        ['Positivity grid (85d)', 'ATE stable across 5 trimming brackets; CV = 8.4%',
         'Stable'],
        ['Negative controls (85e)', '3/3 NCO pass; 1 NCE non-null (DM-specific)',
         'Pass (3/3 NCO)'],
        ['E-value + GOC (85f)',
         'E-value = 1.52 (CI bound 1.49); GOC exclusion ATE unchanged',
         'Moderate E-value'],
        ['Monitoring intensity (85g)',
         'IPMW adjustment \u0394 < 1%; trend p = 0.121', 'Robust'],
        ['RRT sensitivity (85h)', 'Exclusion/bounding/stratification all consistent',
         'Robust'],
        ['Extended time window (85i)',
         '48h sign reversal (unstable); 24h sign-stable across seeds', 'Sensitive'],
        ['Informative monitoring (85j)',
         'Monitoring-aware calibration 2.03 pp vs 18.97 pp naive',
         'Reframed'],
    ]
    add_table_from_rows(doc, t3_headers, t3_rows, font_size=9)

    add_table_footnote(
        doc,
        'Abbreviations: ATE, average treatment effect; CI, confidence interval; '
        'CV, coefficient of variation; DM, direct method; '
        'GOC, goals of care; IPMW, inverse probability of monitoring weighting; '
        'NCE, negative control exposure; NCO, negative control outcome; '
        'pp, percentage points; RRT, renal replacement therapy. '
        '"Robust" indicates that the primary finding was insensitive to the '
        'analysis; "Sensitive" indicates attenuation or instability under the '
        'specified condition; "Reframed" indicates reinterpretation as a '
        'methodological strength rather than a limitation.'
    )

    # ── Table 4: External Validation ─────────────────────────────────
    doc.add_page_break()
    add_heading_custom(doc, 'Table 4. External Validation Consistency Checks '
                       '(eICU-CRD)', level=2)
    add_paragraph(doc, 'Results of 10 pre-registered consistency checks '
                  '(C1\u2013C10) in the eICU-CRD external validation cohort.',
                  font_size=10, italic=True, space_after=6)

    t4_csv = TABLES_DIR / '95b_t4_consistency_checklist.csv'
    if t4_csv.exists():
        add_csv_table(doc, t4_csv, font_size=8)
    else:
        t4_headers = ['Check', 'Category', 'Pass', 'Note']
        t4_rows = [
            ['C1: Sign consistency', 'Pre-registered', 'Yes',
             'MIMIC DM = \u22120.18 pp; eICU DM = \u22120.11 pp'],
            ['C2: CI overlap', 'Pre-registered', 'No', 'CI do not overlap'],
            ['C3: Magnitude ratio', 'Pre-registered', 'Yes', 'Ratio = 0.63; within [0.2, 5.0]'],
            ['C4: Subgroup pattern', 'Pre-registered', 'Yes',
             'HFpEF > no-HF in both (2.8\u00d7 vs 5.1\u00d7)'],
            ['C5: Transport grade', 'Pre-registered', 'No', 'Primary DM grade C'],
            ['C6: Prevalence', 'Additional', 'Yes', 'eICU HFpEF 9.8% (broader ascertainment)'],
            ['C7: Treatment TVD', 'Additional', 'Yes', 'TVD = 0.252 < 0.5'],
            ['C8: Mortality', 'Additional', 'Yes', 'Descriptive comparison'],
            ['C9: Recalibration', 'Additional', 'Yes', '\u0394 < 0.03 pp'],
            ['C10: Tier A', 'Additional', 'Yes', 'HFpEF/no-HF ratio = 17.4\u00d7'],
        ]
        add_table_from_rows(doc, t4_headers, t4_rows, font_size=9)

    add_table_footnote(
        doc,
        'Abbreviations: CI, confidence interval; DM, direct method; '
        'eICU-CRD, eICU Collaborative Research Database; '
        'HFpEF, heart failure with preserved ejection fraction; '
        'MIMIC, Medical Information Mart for Intensive Care; '
        'pp, percentage points; TVD, total variation distance. '
        'C1\u2013C5 are pre-registered consistency checks; '
        'C6\u2013C10 are additional supporting checks.'
    )

    # ═══════════════════════════════════════════════════════════════════
    # FIGURE LEGENDS (separate page, with abbreviation footnotes)
    # ═══════════════════════════════════════════════════════════════════
    doc.add_page_break()
    add_heading_custom(doc, 'FIGURE LEGENDS', level=1)

    figure_legends = [
        ('Figure 1', 'Study Flow Diagram',
         'CONSORT-style cohort construction flowchart for the MIMIC-IV '
         'database. A total of 38,981 patients were included: HFpEF '
         '(n = 2,080), HFrEF/HFmrEF (n = 2,211), HF-EF-unknown '
         '(n = 3,506), and no HF (n = 31,184).',
         'EF, ejection fraction; ECMO, extracorporeal membrane '
         'oxygenation; ESRD, end-stage renal disease; HF, heart failure; '
         'HFmrEF, heart failure with mildly reduced ejection fraction; '
         'HFpEF, heart failure with preserved ejection fraction; '
         'HFrEF, heart failure with reduced ejection fraction; '
         'ICU, intensive care unit; LVAD, left ventricular assist device; '
         'MIMIC-IV, Medical Information Mart for Intensive Care IV; '
         'T0, time zero (ICU admission + 6 hours).'),
        ('Figure 2', 'Core Findings: Cross-Model ATE Estimates and '
         'HFpEF-Specific Signal',
         '(A) ATE estimates and 95% confidence intervals for 28-day '
         'mortality across six model families. All estimates were negative '
         '(100% directional consistency). (B) ATE comparison by HF subtype '
         'in the non-shock stratum: HFpEF ATE = \u22121.44 pp versus '
         'no HF = \u22120.52 pp (2.8-fold ratio), demonstrating the '
         'HFpEF-specific treatment response signal.',
         'ACTIN, adversarial counterfactual temporal inference network; '
         'ATE, average treatment effect; CI, confidence interval; '
         'CT-Legacy, causal Transformer (legacy architecture); '
         'DM, direct method; DR, doubly robust; FQE, fitted Q-evaluation; '
         'HF, heart failure; HFpEF, heart failure with preserved ejection '
         'fraction; IPTW, inverse probability of treatment weighting; '
         'Mamba-CDSP, Mamba counterfactual dynamic state prediction; '
         'MSM, marginal structural model; pp, percentage points; '
         'WIS, weighted importance sampling.'),
        ('Figure 3', 'Robustness Evidence: Sensitivity Analyses and '
         'Positivity Diagnostics',
         '(A) Summary of nine pre-specified sensitivity analysis '
         'determinations: 5 robust/pass, 2 sensitive, 1 reframed, E-value '
         'moderate. (B) Positivity grid stability analysis: ATEs remained '
         'negative across six trimming schemes, with the HFpEF subgroup '
         'showing consistently larger effects than the no-HF group under '
         'all trimming conditions.',
         'ATE, average treatment effect; E-value, expected value for '
         'unmeasured confounding; HFpEF, heart failure with preserved '
         'ejection fraction; pp, percentage points.'),
        ('Figure 4', 'External Validation Results',
         'Summary of eICU-CRD external validation. HFpEF/no-HF effect '
         'ratios were consistent across databases (MIMIC, 2.8-fold; eICU, '
         '5.1\u201317.4-fold). Of 10 consistency checks, 8 passed and '
         '2 failed.',
         'ATE, average treatment effect; CI, confidence interval; '
         'DM, direct method; eICU-CRD, eICU Collaborative Research '
         'Database; FQE, fitted Q-evaluation; HFpEF, heart failure with '
         'preserved ejection fraction; MIMIC-IV, Medical Information Mart '
         'for Intensive Care IV; pp, percentage points.'),
        ('Figure 5', 'Hemodynamic Vulnerability Characterization',
         '(A) Distribution of the hemodynamic fragility index (HFI) by '
         'heart failure subtype. HFpEF patients showed significantly '
         'higher vulnerability (mean HFI = +0.224) compared with no-HF '
         'patients (\u22120.024; P = 9.2 \u00d7 10\u207b\u2078). '
         '(B) ATE dose\u2013response across HFI quartiles: estimated '
         'treatment effect increased monotonically from Q1 (stable, '
         '\u22121.02 pp) to Q4 (vulnerable, \u22122.03 pp; Spearman '
         '\u03c1 = \u22121.00). (C) HFI-matched specificity test: within '
         'each vulnerability quartile, HFpEF patients showed larger '
         'estimated treatment effects than vulnerability-matched no-HF '
         'patients (absolute differences \u22120.62 to \u22121.71 pp; '
         'ratios 1.35\u20132.60-fold), suggesting residual phenotype-specific '
         'treatment responsiveness beyond hemodynamic vulnerability.',
         'ATE, average treatment effect; HF, heart failure; '
         'HFI, hemodynamic fragility index; HFmrEF, heart failure with '
         'mildly reduced ejection fraction; HFpEF, heart failure with '
         'preserved ejection fraction; HFrEF, heart failure with reduced '
         'ejection fraction; MAP, mean arterial pressure; '
         'pp, percentage points; Q, quartile.'),
    ]

    for fig_label, fig_title, fig_text, fig_abbr in figure_legends:
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = Pt(24)
        p.paragraph_format.space_after = Pt(4)
        run_label = p.add_run(f'{fig_label}. {fig_title}. ')
        set_font(run_label, size=11, bold=True)
        run_text = p.add_run(fig_text)
        set_font(run_text, size=11)
        # Abbreviation footnote for this figure
        p_abbr = doc.add_paragraph()
        p_abbr.paragraph_format.line_spacing = Pt(18)
        p_abbr.paragraph_format.space_before = Pt(2)
        p_abbr.paragraph_format.space_after = Pt(12)
        run_a = p_abbr.add_run(f'Abbreviations: {fig_abbr}')
        set_font(run_a, size=8, italic=True)

    # ═══════════════════════════════════════════════════════════════════
    # ABBREVIATIONS
    # ═══════════════════════════════════════════════════════════════════
    doc.add_page_break()
    add_heading_custom(doc, 'ABBREVIATIONS', level=1)

    abbr_match = re.search(r'## ABBREVIATIONS\n(.+)', part4, re.DOTALL)
    if abbr_match:
        build_body_from_md(doc, abbr_match.group(1))

    # ═══════════════════════════════════════════════════════════════════
    # FIGURES (at end, one per page with legend and abbreviations below)
    # ═══════════════════════════════════════════════════════════════════
    FIGURES_MANU = FIGURES_SRC / 'manuscript'
    figure_files = [
        ('Figure 1', [
            FIGURES_PDF_DIR / 'Figure1.png',
            FIGURES_MANU / 'Figure1.png',
        ], 6.0),
        ('Figure 2', [
            FIGURES_PNG_DIR / 'Figure2_core_findings.png',
            FIGURES_PDF_DIR / 'Figure2.png',
            FIGURES_MANU / 'Figure2.png',
            FIGURES_SRC / '95b_f2_hfpef_specificity.png',
        ], 6.0),
        ('Figure 3', [
            FIGURES_PNG_DIR / 'Figure3_robustness.png',
            FIGURES_PDF_DIR / 'Figure3.png',
            FIGURES_MANU / 'Figure3.png',
            FIGURES_SRC / '95b_f3_sensitivity_heatmap.png',
        ], 6.0),
        ('Figure 4', [
            FIGURES_PNG_DIR / 'Figure4_external_validation.png',
            FIGURES_PDF_DIR / 'Figure4.png',
            FIGURES_MANU / 'Figure4.png',
            FIGURES_SRC / '95b_f4_external_validation.png',
        ], 6.0),
        ('Figure 5', [
            FIGURES_PDF_DIR / 'Figure5.png',
            FIGURES_MANU / 'Figure5.png',
            FIGURES_SRC / '95b_f5_trimming_stability.png',
        ], 6.0),
    ]

    for fig_idx, (fig_label, path_candidates, width) in enumerate(figure_files):
        doc.add_page_break()
        add_paragraph(doc, fig_label, bold=True, font_size=14,
                      alignment=WD_ALIGN_PARAGRAPH.CENTER, space_after=12)

        # Try to insert image
        img_path = find_figure(path_candidates)
        if img_path:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run()
            run.add_picture(str(img_path), width=Inches(width))
        else:
            names = ' or '.join(str(c.name) for c in path_candidates)
            add_paragraph(
                doc,
                f'[{fig_label} \u2014 image file not found: {names}]',
                italic=True,
                alignment=WD_ALIGN_PARAGRAPH.CENTER, space_after=12
            )

        # Add legend below figure
        if fig_idx < len(figure_legends):
            fl = figure_legends[fig_idx]
            p = doc.add_paragraph()
            p.paragraph_format.line_spacing = Pt(24)
            p.paragraph_format.space_after = Pt(2)
            run_label = p.add_run(f'{fl[0]}. {fl[1]}. ')
            set_font(run_label, size=10, bold=True)
            run_text = p.add_run(fl[2])
            set_font(run_text, size=10)
            # Abbreviations below figure legend
            p_abbr = doc.add_paragraph()
            p_abbr.paragraph_format.line_spacing = Pt(16)
            p_abbr.paragraph_format.space_before = Pt(2)
            p_abbr.paragraph_format.space_after = Pt(0)
            run_a = p_abbr.add_run(f'Abbreviations: {fl[3]}')
            set_font(run_a, size=8, italic=True)

    # ── Save ────────────────────────────────────────────────────────
    doc.save(str(OUTPUT))
    print(f"Main manuscript saved to: {OUTPUT}")
    print(f"File size: {OUTPUT.stat().st_size / 1024:.1f} KB")
    print(f"Abstract word count: {abstract_wc}")
    print(f"Body word count: {body_wc}")
    print(f"Total word count: {total_wc}")


if __name__ == '__main__':
    build_manuscript()
