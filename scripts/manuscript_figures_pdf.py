#!/usr/bin/env python3
"""
Generate a figures-only PDF for journal submission.
Each figure on its own page with caption.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")

# CJK font support
import matplotlib.font_manager as fm
fm._load_fontmanager(try_read_cache=False)
matplotlib.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "Noto Sans SC", "DejaVu Sans", "Arial"
]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures" / "submission"
OUT_PDF = ROOT / "manuscript" / "submission_figures.pdf"

FIGURES = [
    {
        "label": "Figure 1",
        "title": "Study Flow Diagram (CONSORT)",
        "caption": (
            "CONSORT-style cohort construction in MIMIC-IV.\n"
            "Final cohort: 38,981 patients (HFpEF n=2,080; HFrEF/HFmrEF n=2,211;\n"
            "HF-EF-unknown n=3,506; no HF n=31,184)."
        ),
        "image": None,  # CONSORT — placeholder
    },
    {
        "label": "Figure 2",
        "title": "Core Findings: Cross-Model ATE and HFpEF-Specific Signal",
        "caption": (
            "(A) 28-day mortality ATE estimates across six model families with 95% CI.\n"
            "All estimates are negative (100% directional consistency).\n"
            "(B) ATE by HF subtype in non-shock stratum: HFpEF ATE = -1.44 pp\n"
            "vs no-HF = -0.52 pp (2.8-fold ratio)."
        ),
        "image": FIG_DIR / "Figure2_core_findings.png",
    },
    {
        "label": "Figure 3",
        "title": "Robustness: Sensitivity Analyses and Positivity Diagnostics",
        "caption": (
            "(A) Verdicts for 9 pre-specified sensitivity analyses: 5 robust/pass,\n"
            "2 sensitive, 1 reframed, E-value moderate.\n"
            "(B) Positivity grid stability: ATE remains negative across 6 trimming\n"
            "brackets; HFpEF effect exceeds no-HF under all conditions."
        ),
        "image": FIG_DIR / "Figure3_robustness.png",
    },
    {
        "label": "Figure 4",
        "title": "External Validation Results (eICU-CRD)",
        "caption": (
            "Cross-database ATE comparison (MIMIC-IV vs eICU-CRD).\n"
            "HFpEF/no-HF effect ratio consistent across databases\n"
            "(MIMIC 2.8x; eICU 5.1-17.4x). 8/10 consistency checks PASS."
        ),
        "image": FIG_DIR / "Figure4_external_validation.png",
    },
]


def main():
    print("=" * 60)
    print("Generating submission figures PDF")
    print("=" * 60)

    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(str(OUT_PDF)) as pdf:
        for fig_info in FIGURES:
            label = fig_info["label"]
            img_path = fig_info["image"]

            if img_path and img_path.exists():
                img = mpimg.imread(str(img_path))
                h, w = img.shape[:2]
                aspect = h / w

                # Size: full width, height proportional + space for caption
                fig_w = 10
                fig_h = fig_w * aspect + 2.5  # extra for title+caption
                fig_h = min(fig_h, 14)  # cap at ~A4 height

                fig = plt.figure(figsize=(fig_w, fig_h))

                # Title at top
                fig.text(0.5, 0.97, f"{label}",
                         fontsize=16, fontweight="bold",
                         ha="center", va="top")
                fig.text(0.5, 0.94, fig_info["title"],
                         fontsize=11, ha="center", va="top",
                         style="italic")

                # Image
                img_bottom = 0.15
                img_top = 0.91
                ax = fig.add_axes([0.03, img_bottom, 0.94, img_top - img_bottom])
                ax.imshow(img)
                ax.axis("off")

                # Caption at bottom
                fig.text(0.05, 0.10, fig_info["caption"],
                         fontsize=9, ha="left", va="top",
                         wrap=True,
                         fontstyle="italic",
                         color="#333333")

                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                print(f"  Added {label} ({img_path.name})")

            else:
                # Placeholder page for Figure 1 (CONSORT)
                fig = plt.figure(figsize=(10, 7))
                fig.text(0.5, 0.95, f"{label}",
                         fontsize=16, fontweight="bold",
                         ha="center", va="top")
                fig.text(0.5, 0.90, fig_info["title"],
                         fontsize=11, ha="center", va="top",
                         style="italic")
                fig.text(0.5, 0.50,
                         "[CONSORT Flow Diagram — to be inserted]",
                         fontsize=14, ha="center", va="center",
                         color="#999999",
                         bbox=dict(boxstyle="round,pad=1",
                                   facecolor="#f5f5f5",
                                   edgecolor="#cccccc"))
                fig.text(0.05, 0.15, fig_info["caption"],
                         fontsize=9, ha="left", va="top",
                         fontstyle="italic", color="#333333")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                print(f"  Added {label} (placeholder)")

    size_kb = OUT_PDF.stat().st_size / 1024
    print(f"\n  Output: {OUT_PDF} ({size_kb:.0f} KB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
