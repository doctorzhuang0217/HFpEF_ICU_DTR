#!/usr/bin/env python3
"""
Generate submission-ready PDF from manuscript markdown.

Converts 11_merged_manuscript.md → manuscript_submission.pdf
with embedded figures, proper formatting, and CJK font support.
"""

from pathlib import Path
import markdown
import weasyprint
import re

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript" / "11_merged_manuscript.md"
FIG_DIR = ROOT / "figures" / "submission"
ALL_FIG_DIR = ROOT / "figures"
OUT_PDF = ROOT / "manuscript" / "manuscript_submission.pdf"

# Map figure references to actual image files
FIGURE_MAP = {
    "Figure 1": None,  # CONSORT — placeholder text
    "Figure 2": FIG_DIR / "Figure2_core_findings.png",
    "Figure 3": FIG_DIR / "Figure3_robustness.png",
    "Figure 4": FIG_DIR / "Figure4_external_validation.png",
}


def inject_figures(html: str) -> str:
    """Insert figure images after their caption paragraphs."""
    for fig_label, fig_path in FIGURE_MAP.items():
        if fig_path and fig_path.exists():
            img_tag = (
                f'<div class="figure-container">'
                f'<img src="file://{fig_path}" class="figure-img" />'
                f'</div>'
            )
            # Find the h3 containing the figure label and the next </p> after it
            # Use DOTALL to match across newlines
            escaped = re.escape(fig_label)
            pattern = rf'<h3[^>]*>{escaped}\..+?</h3>.+?</p>'
            match = re.search(pattern, html, re.DOTALL)
            if match:
                insert_pos = match.end()
                html = html[:insert_pos] + '\n' + img_tag + html[insert_pos:]
                print(f"  Injected {fig_label} image")
            else:
                print(f"  WARNING: Could not find insertion point for {fig_label}")
    return html


def md_to_html(md_text: str) -> str:
    """Convert markdown to HTML with extensions."""
    extensions = ["tables", "fenced_code", "toc", "smarty"]
    return markdown.markdown(md_text, extensions=extensions)


CSS = """
@page {
    size: A4;
    margin: 2.2cm 2cm 2.2cm 2cm;
    @bottom-center {
        content: counter(page);
        font-size: 9pt;
        color: #666;
    }
}

body {
    font-family: "Noto Serif CJK SC", "Source Han Serif SC", "SimSun",
                 "Times New Roman", "Noto Serif", serif;
    font-size: 10.5pt;
    line-height: 1.65;
    color: #1a1a1a;
    text-align: justify;
    hyphens: auto;
}

h1 {
    font-size: 15pt;
    font-weight: bold;
    text-align: center;
    margin-top: 0;
    margin-bottom: 6pt;
    line-height: 1.4;
    page-break-after: avoid;
}

h2 {
    font-size: 13pt;
    font-weight: bold;
    margin-top: 18pt;
    margin-bottom: 8pt;
    border-bottom: 1px solid #ccc;
    padding-bottom: 4pt;
    page-break-after: avoid;
}

h3 {
    font-size: 11.5pt;
    font-weight: bold;
    margin-top: 14pt;
    margin-bottom: 6pt;
    page-break-after: avoid;
}

p {
    margin-top: 3pt;
    margin-bottom: 6pt;
    text-indent: 0;
    orphans: 3;
    widows: 3;
}

strong {
    font-weight: bold;
}

em {
    font-style: italic;
}

hr {
    border: none;
    border-top: 1px solid #ccc;
    margin: 16pt 0;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 10pt 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
}

th, td {
    border: 1px solid #999;
    padding: 4pt 6pt;
    text-align: left;
}

th {
    background-color: #f0f0f0;
    font-weight: bold;
}

code {
    font-family: "Courier New", monospace;
    font-size: 9pt;
    background-color: #f5f5f5;
    padding: 1pt 3pt;
    border-radius: 2pt;
}

pre {
    font-family: "Courier New", monospace;
    font-size: 8.5pt;
    background-color: #f5f5f5;
    padding: 8pt;
    border: 1px solid #ddd;
    border-radius: 3pt;
    overflow-wrap: break-word;
    white-space: pre-wrap;
    page-break-inside: avoid;
}

ul, ol {
    margin-top: 4pt;
    margin-bottom: 8pt;
    padding-left: 20pt;
}

li {
    margin-bottom: 3pt;
}

blockquote {
    border-left: 3pt solid #ccc;
    margin-left: 0;
    padding-left: 12pt;
    color: #555;
}

.figure-container {
    text-align: center;
    margin: 12pt 0;
    page-break-inside: avoid;
}

.figure-img {
    max-width: 100%;
    height: auto;
    border: 1px solid #eee;
}

/* Title page styling */
h1 + p {
    text-align: center;
}

/* Keep figure captions with figures */
h3 + p + .figure-container {
    page-break-before: avoid;
}
"""


def main():
    print("=" * 60)
    print("Generating submission PDF")
    print("=" * 60)

    # Read markdown
    md_text = MANUSCRIPT.read_text(encoding="utf-8")
    print(f"  Read manuscript: {len(md_text):,} chars")

    # Convert to HTML
    html_body = md_to_html(md_text)
    print(f"  Converted to HTML: {len(html_body):,} chars")

    # Inject figures
    html_body = inject_figures(html_body)

    # Wrap in full HTML document
    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<style>
{CSS}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

    # Generate PDF
    print("  Generating PDF (this may take a moment)...")
    wp = weasyprint.HTML(string=html_doc)
    wp.write_pdf(str(OUT_PDF))

    size_mb = OUT_PDF.stat().st_size / (1024 * 1024)
    print(f"  Saved: {OUT_PDF} ({size_mb:.1f} MB)")
    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
