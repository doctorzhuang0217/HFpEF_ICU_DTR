#!/usr/bin/env python3
"""
Generate Figure 1: CONSORT-style flow diagram for journal submission.
All numbers from verified pipeline reports.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

OUT_DIR = Path(__file__).resolve().parents[1] / "manuscript" / "figures_pdf"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_box(ax, x, y, w, h, text, fontsize=8.5, bold_first=False,
             facecolor="#EBF5FB", edgecolor="#2C3E50", lw=1.2):
    """Draw a rounded box with centered text. (x,y) = center."""
    box = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.015",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
    )
    ax.add_patch(box)

    if bold_first and "\n" in text:
        first, rest = text.split("\n", 1)
        ax.text(x, y + h * 0.18, first, ha="center", va="bottom",
                fontsize=fontsize, fontweight="bold", color="#1a1a1a")
        ax.text(x, y - h * 0.05, rest, ha="center", va="top",
                fontsize=fontsize - 0.5, color="#333333", linespacing=1.25)
    else:
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, color="#1a1a1a", linespacing=1.25)


def arrow_v(ax, x, y_from, y_to, color="#2C3E50"):
    """Vertical arrow."""
    ax.annotate("", xy=(x, y_to), xytext=(x, y_from),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2))


def arrow_L(ax, x_from, y_from, x_to, y_to, color="#2C3E50"):
    """L-shaped arrow: horizontal then vertical (right-side exclusion boxes)."""
    ax.plot([x_from, x_to], [y_from, y_from], color=color, lw=1.0)
    ax.annotate("", xy=(x_to, y_to), xytext=(x_to, y_from),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0))


def main():
    # Taller canvas to prevent overlap
    fig, ax = plt.subplots(1, 1, figsize=(11, 17))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── Coordinates ──────────────────────────────────────────
    # Center column for main flow
    cx = 0.42
    # Right column for exclusion boxes
    rx = 0.80
    # Main box dimensions
    mw, mh = 0.48, 0.038

    # Title
    ax.text(0.5, 0.985, "Figure 1. Study Flow Diagram",
            ha="center", va="top", fontsize=15, fontweight="bold")

    # ════════════════════════════════════════════════════════
    # Level 1 — Screening
    # ════════════════════════════════════════════════════════
    L1 = 0.935
    draw_box(ax, cx, L1, mw, 0.042,
             "ICU admissions assessed for eligibility\n"
             "(MIMIC-IV, 2008–2019)\n"
             "N = 94,458 stays (65,366 subjects)",
             fontsize=9, bold_first=True,
             facecolor="#D5E8D4", edgecolor="#82B366")

    # ════════════════════════════════════════════════════════
    # Level 2 — Hard exclusions → Eligible
    # ════════════════════════════════════════════════════════
    L2 = 0.845
    arrow_v(ax, cx, L1 - 0.021, L2 + 0.019)

    # Exclusion box — right side, vertically centered between L1 and L2
    ex_y = (L1 + L2) / 2 + 0.005
    draw_box(ax, rx, ex_y, 0.34, 0.078,
             "Excluded (n = 21,084)\n"
             "Chronic dialysis / ESRD: 14,770\n"
             "Pregnancy-related: 3,837\n"
             "Hemorrhagic shock: 2,174\n"
             "Died / discharged < 6 h: 1,612\n"
             "LVAD / ECMO / transplant: 203\n"
             "Comfort care pre-T0: 89",
             fontsize=7, bold_first=True,
             facecolor="#F8D7DA", edgecolor="#C0392B")
    # Horizontal connector from main arrow to exclusion box
    mid_y = (L1 - 0.021 + L2 + 0.019) / 2
    arrow_L(ax, cx + mw / 2, mid_y, rx - 0.17, ex_y + 0.033)

    draw_box(ax, cx, L2, mw, mh,
             "Eligible after hard exclusions\nn = 73,374 stays",
             fontsize=9, bold_first=True)

    # ════════════════════════════════════════════════════════
    # Level 3 — Index stay selection
    # ════════════════════════════════════════════════════════
    L3 = 0.765
    arrow_v(ax, cx, L2 - mh / 2, L3 + mh / 2)

    # Index exclusion — right side
    ix_y = (L2 + L3) / 2
    draw_box(ax, rx, ix_y, 0.30, 0.032,
             "Multiple stays per subject excluded\n(1 index stay retained; n = 19,075 removed)",
             fontsize=7, bold_first=True,
             facecolor="#F8D7DA", edgecolor="#C0392B")
    mid_y2 = (L2 - mh / 2 + L3 + mh / 2) / 2
    arrow_L(ax, cx + mw / 2, mid_y2, rx - 0.15, ix_y + 0.016)

    draw_box(ax, cx, L3, mw, mh,
             "Index ICU stays (unique patients)\nn = 54,299",
             fontsize=9, bold_first=True)

    # ════════════════════════════════════════════════════════
    # Level 4 — Pre-T0 hemodynamic gate
    # ════════════════════════════════════════════════════════
    L4 = 0.68
    arrow_v(ax, cx, L3 - mh / 2, L4 + mh / 2)

    # Gate failure — right side
    gx_y = (L3 + L4) / 2
    draw_box(ax, rx, gx_y, 0.34, 0.040,
             "Pre-T0 gate failed (n = 15,318)\n"
             "No hemodynamic intervention in\n"
             "ICU admission → T0 (6 h) window",
             fontsize=7, bold_first=True,
             facecolor="#F8D7DA", edgecolor="#C0392B")
    mid_y3 = (L3 - mh / 2 + L4 + mh / 2) / 2
    arrow_L(ax, cx + mw / 2, mid_y3, rx - 0.17, gx_y + 0.020)

    draw_box(ax, cx, L4, mw, 0.042,
             "Final analytic cohort\nN = 38,981",
             fontsize=10, bold_first=True,
             facecolor="#D5E8D4", edgecolor="#27AE60", lw=2.0)

    # ════════════════════════════════════════════════════════
    # Level 5 — HF classification (4 subgroups)
    # ════════════════════════════════════════════════════════
    L5_label = 0.625
    L5 = 0.575

    arrow_v(ax, cx, L4 - 0.021, L5_label + 0.012)

    ax.text(cx, L5_label, "Heart failure classification  (ICD-9/10 + echocardiography EF)",
            ha="center", va="center", fontsize=8, style="italic", color="#555")

    bw = 0.215
    bh = 0.048
    bxs = [0.125, 0.365, 0.605, 0.855]
    blabels = [
        "HFpEF\n(EF ≥ 50%)\nn = 2,080 (5.3%)",
        "HFrEF / HFmrEF\n(EF < 50%)\nn = 2,211 (5.7%)",
        "HF-EF-unknown\n(HF dx, no EF data)\nn = 3,506 (9.0%)",
        "No HF\n\nn = 31,184 (80.0%)",
    ]
    bcolors = ["#FADBD8", "#F5CBA7", "#FCF3CF", "#D4E6F1"]
    bedges = ["#E74C3C", "#E67E22", "#F1C40F", "#3498DB"]

    for xc, lab, fc, ec in zip(bxs, blabels, bcolors, bedges):
        arrow_v(ax, xc, L5_label - 0.012, L5 + bh / 2)
        draw_box(ax, xc, L5, bw, bh, lab,
                 fontsize=7.5, bold_first=True, facecolor=fc, edgecolor=ec)

    # ════════════════════════════════════════════════════════
    # Level 6 — Shock stratification (under HFpEF only)
    # ════════════════════════════════════════════════════════
    L6_label = 0.505
    L6 = 0.465

    hfpef_x = bxs[0]  # 0.125
    arrow_v(ax, hfpef_x, L5 - bh / 2, L6_label + 0.012)

    ax.text(hfpef_x, L6_label,
            "Shock phenotype stratification",
            ha="center", va="center", fontsize=7, style="italic", color="#555")

    sw, sh = 0.115, 0.035
    sx_l, sx_r = 0.10, 0.26
    draw_box(ax, sx_l, L6, sw, sh,
             "Shock-predominant\nn = 891 (42.8%)",
             fontsize=7, facecolor="#F5B7B1", edgecolor="#C0392B")
    draw_box(ax, sx_r, L6, sw, sh,
             "Non-shock\nn = 1,189 (57.2%)",
             fontsize=7, facecolor="#ABEBC6", edgecolor="#27AE60")

    arrow_v(ax, sx_l, L6_label - 0.012, L6 + sh / 2)
    arrow_v(ax, sx_r, L6_label - 0.012, L6 + sh / 2)
    # Horizontal bar connecting the two arrows
    ax.plot([sx_l, sx_r], [L6_label - 0.012, L6_label - 0.012],
            color="#2C3E50", lw=1.0)

    # ════════════════════════════════════════════════════════
    # Level 7 — Train / Validation / Test split
    # ════════════════════════════════════════════════════════
    L7_label = 0.39
    L7 = 0.35

    # Arrow from center of HF subgroups down
    arrow_v(ax, cx, L5 - bh / 2, L7_label + 0.012)

    ax.text(cx, L7_label,
            "Stratified train / validation / test split",
            ha="center", va="center", fontsize=8, style="italic", color="#555")

    tw, th = 0.27, 0.038
    splits = [
        ("Training set\nn = 27,287 (70%)", 0.18, "#E8F8F5", "#1ABC9C"),
        ("Validation set\nn = 3,898 (10%)", 0.50, "#FEF9E7", "#F39C12"),
        ("Test set\nn = 7,796 (20%)", 0.82, "#FDEDEC", "#E74C3C"),
    ]
    for lab, xc, fc, ec in splits:
        arrow_v(ax, xc, L7_label - 0.012, L7 + th / 2)
        draw_box(ax, xc, L7, tw, th, lab,
                 fontsize=8, bold_first=True, facecolor=fc, edgecolor=ec)
    # Horizontal bar connecting
    ax.plot([0.18, 0.82], [L7_label - 0.012, L7_label - 0.012],
            color="#2C3E50", lw=1.0)

    # ════════════════════════════════════════════════════════
    # Level 8 — Test-set interaction cells
    # ════════════════════════════════════════════════════════
    L8_label = 0.285
    L8 = 0.25

    test_x = 0.82
    arrow_v(ax, test_x, L7 - th / 2, L8_label + 0.012)

    ax.text(test_x, L8_label,
            "HFpEF × shock interaction analysis (test set)",
            ha="center", va="center", fontsize=7, style="italic", color="#555")

    cw, ch = 0.155, 0.032
    cells = [
        ("HFpEF × non-shock\nn = 163", 0.735, "#FADBD8", "#E74C3C"),
        ("HFpEF × shock\nn = 141", 0.905, "#F5B7B1", "#C0392B"),
    ]
    for lab, xc, fc, ec in cells:
        arrow_v(ax, xc, L8_label - 0.010, L8 + ch / 2)
        draw_box(ax, xc, L8, cw, ch, lab,
                 fontsize=7, facecolor=fc, edgecolor=ec)
    ax.plot([0.735, 0.905], [L8_label - 0.010, L8_label - 0.010],
            color="#2C3E50", lw=1.0)

    # ════════════════════════════════════════════════════════
    # Abbreviations
    # ════════════════════════════════════════════════════════
    ax.text(0.5, 0.185,
            "Abbreviations: HFpEF = heart failure with preserved ejection fraction; "
            "HFrEF = HF with reduced EF; HFmrEF = HF with mildly reduced EF;\n"
            "EF = ejection fraction; T0 = ICU admission + 6 hours; "
            "LVAD = left ventricular assist device; ECMO = extracorporeal membrane oxygenation;\n"
            "ESRD = end-stage renal disease; dx = diagnosis.",
            ha="center", va="top", fontsize=7.5, color="#666666",
            linespacing=1.5, style="italic")

    # ── Save ─────────────────────────────────────────────────
    out_path = OUT_DIR / "Figure1.pdf"
    out_png = OUT_DIR / "Figure1.png"
    with PdfPages(str(out_path)) as pdf:
        pdf.savefig(fig, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=200, facecolor="white")
    plt.close(fig)

    print(f"  Saved: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")
    print(f"  Saved: {out_png} ({out_png.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
