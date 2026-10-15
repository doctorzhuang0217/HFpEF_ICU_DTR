#!/usr/bin/env python3
"""
Regenerate ALL manuscript figures as vector PDFs for journal submission.

Two-phase strategy:
  Phase 1 (CSV-only): Run manuscript_replot_all_figures.py, which reads pre-computed
    CSV tables and outputs true vector PDFs (matplotlib-native) for Figures 2-5
    and sFigures 3-5, 7-10.  No GPU or pipeline data required.

  Phase 2 (pipeline-dependent): Handle remaining figures that need pipeline
    outputs or dedicated scripts:
      Figure 1  — CONSORT (manuscript_figure1_consort.py)
      sFigure 1 — Phenotype radar (60e_phenotype_cards_early.py output)
      sFigure 2 — IPTW weight distribution (75a output)
      sFigure 6 — Counterfactual trajectories (P5 output)

    For Phase 2, we prefer existing vector PDFs in figures/ and fall back to
    high-resolution PNG embedding when vector sources are unavailable.

Outputs → manuscript/figures_pdf/
  Figure1.pdf  through Figure5.pdf
  sFigure1.pdf through sFigure10.pdf

Compute: C1 (< 30 seconds on any machine)

Usage
-----
  env/run_py.sh scripts/manuscript_regenerate_all_vector_figures.py
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType fonts in PDF (editable text)
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "manuscript" / "figures_pdf"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = ROOT / "figures"
MANUSCRIPT_FIG = FIG_DIR / "manuscript"

sys.path.insert(0, str(ROOT / "src"))


# ══════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════

def _save_pdf(fig, name: str):
    """Save figure as vector PDF in the output directory."""
    out = OUT_DIR / f"{name}.pdf"
    fig.savefig(str(out), format="pdf", bbox_inches="tight",
                dpi=300, facecolor="white", edgecolor="none")
    print(f"    {name}.pdf ({out.stat().st_size / 1024:.0f} KB)")
    plt.close(fig)


def _copy_pdf(src: Path, name: str):
    """Copy an existing vector PDF to the output directory."""
    out = OUT_DIR / f"{name}.pdf"
    shutil.copy2(str(src), str(out))
    print(f"    {name}.pdf (copy, {out.stat().st_size / 1024:.0f} KB)")


def _embed_png_as_pdf(png_path: Path, name: str):
    """Embed a PNG in a PDF page at high resolution (raster fallback)."""
    if not png_path.exists():
        print(f"    SKIPPED {name} — {png_path.name} not found")
        return
    img = mpimg.imread(str(png_path))
    h, w = img.shape[:2]
    aspect = h / w
    fig_w = 10
    fig = plt.figure(figsize=(fig_w, fig_w * aspect))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, interpolation="lanczos")
    ax.axis("off")
    _save_pdf(fig, name)


def _find_best_source(base_name: str, fig_name: str):
    """Find the best available source for a figure: vector PDF > manuscript PDF > PNG."""
    # 1. Vector PDF from replot script
    pdf_in_pdf_dir = OUT_DIR / f"{fig_name}.pdf"
    if pdf_in_pdf_dir.exists():
        return pdf_in_pdf_dir, "replot"

    # 2. Vector PDF in figures/
    pdf_in_fig = FIG_DIR / f"{base_name}.pdf"
    if pdf_in_fig.exists():
        return pdf_in_fig, "figures_pdf"

    # 3. Manuscript PNG
    png_in_manu = MANUSCRIPT_FIG / f"{fig_name}.png"
    if png_in_manu.exists():
        return png_in_manu, "manuscript_png"

    # 4. Direct PNG in figures/
    png_in_fig = FIG_DIR / f"{base_name}.png"
    if png_in_fig.exists():
        return png_in_fig, "figures_png"

    return None, None


# ══════════════════════════════════════════════════════════════════════
#  PHASE 1: Run CSV-only replot script
# ══════════════════════════════════════════════════════════════════════

def run_phase1():
    """Run manuscript_replot_all_figures.py to generate true vector PDFs."""
    print("\n" + "=" * 65)
    print("  Phase 1: CSV-only figure generation (manuscript_replot_all_figures.py)")
    print("=" * 65)

    replot_script = ROOT / "scripts" / "manuscript_replot_all_figures.py"
    if not replot_script.exists():
        print("  ERROR: manuscript_replot_all_figures.py not found!")
        return False

    spec = importlib.util.spec_from_file_location("replot", str(replot_script))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        mod.main()
        return True
    except Exception as e:
        print(f"  ERROR in Phase 1: {e}")
        return False


def _clean_stale_pngs():
    """Remove stale PNGs in manuscript/figures_pdf/ that have label issues.

    The replot script outputs PNGs to figures/manuscript/ and PDFs to
    manuscript/figures_pdf/. Any PNGs lingering in manuscript/figures_pdf/
    from older scripts may have incorrect labels (e.g. 'no_HF' instead of
    'No HF'). Remove them to avoid confusion.
    """
    stale = list(OUT_DIR.glob("sFigure*.png")) + list(OUT_DIR.glob("Figure*.png"))
    # Keep Figure1.png (CONSORT — generated by dedicated script)
    stale = [p for p in stale if p.name != "Figure1.png"]
    if stale:
        print(f"\n  Cleaning {len(stale)} stale PNG(s) in manuscript/figures_pdf/:")
        for p in stale:
            p.unlink()
            print(f"    removed {p.name}")


# ══════════════════════════════════════════════════════════════════════
#  PHASE 2: Pipeline-dependent figures
# ══════════════════════════════════════════════════════════════════════

def gen_figure1():
    """Figure 1: CONSORT flow diagram — run dedicated script."""
    print("\n  [Figure 1] CONSORT flow diagram")
    consort_script = ROOT / "scripts" / "manuscript_figure1_consort.py"
    if not consort_script.exists():
        print("    SKIPPED — manuscript_figure1_consort.py not found")
        return

    spec = importlib.util.spec_from_file_location("consort", str(consort_script))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        mod.main()
        # Check if it produced the PDF
        consort_pdf = OUT_DIR / "Figure1.pdf"
        if consort_pdf.exists():
            print(f"    Figure1.pdf ({consort_pdf.stat().st_size / 1024:.0f} KB)")
        else:
            print("    NOTE: CONSORT script ran but Figure1.pdf not found in output")
    except Exception as e:
        print(f"    ERROR: {e}")


def gen_pipeline_sfigures():
    """Generate sFigures that require pipeline outputs (not CSV-only)."""
    # Mapping: (output_name, search_bases)
    pipeline_figs = [
        ("sFigure1", ["60e_phenotype_radar"]),
        ("sFigure2", ["75a_weight_distribution"]),
        ("sFigure6", ["P5_trajectory_panel", "P5_counterfactual_trajectories"]),
    ]

    for fig_name, base_candidates in pipeline_figs:
        # Check if Phase 1 already produced this (shouldn't, but just in case)
        existing = OUT_DIR / f"{fig_name}.pdf"
        if existing.exists():
            print(f"\n  [{fig_name}] already generated by Phase 1")
            continue

        found = False
        for base_name in base_candidates:
            # Try vector PDF first
            pdf_path = FIG_DIR / f"{base_name}.pdf"
            if pdf_path.exists():
                print(f"\n  [{fig_name}] {base_name}.pdf (vector)")
                _copy_pdf(pdf_path, fig_name)
                found = True
                break

            # Try PNG fallback
            png_path = FIG_DIR / f"{base_name}.png"
            if png_path.exists():
                print(f"\n  [{fig_name}] {base_name}.png (raster fallback)")
                _embed_png_as_pdf(png_path, fig_name)
                found = True
                break

        if not found:
            print(f"\n  [{fig_name}] SKIPPED — no source found")


def run_phase2():
    """Generate pipeline-dependent figures."""
    print("\n" + "=" * 65)
    print("  Phase 2: Pipeline-dependent figures")
    print("=" * 65)

    gen_figure1()
    gen_pipeline_sfigures()


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 65)
    print("  Manuscript Vector PDF Generator (comprehensive)")
    print(f"  Output: {OUT_DIR}")
    print("=" * 65)

    # Phase 1: CSV-only figures (Figures 2-5, sFigures 3-5, 7-10)
    run_phase1()

    # Clean stale PNGs from previous script runs
    _clean_stale_pngs()

    # Phase 2: Pipeline-dependent figures (Figure 1, sFigures 1, 2, 6)
    run_phase2()

    # ── Summary ──────────────────────────────────────────────
    elapsed = time.time() - t0
    pdfs = sorted(OUT_DIR.glob("*.pdf"))
    print("\n" + "=" * 65)
    print(f"  Generated {len(pdfs)} vector PDFs in {elapsed:.1f}s:")
    print()

    # Expected figures
    expected = (
        ["Figure1", "Figure2", "Figure3", "Figure4", "Figure5"]
        + [f"sFigure{i}" for i in range(1, 11)]
    )

    for name in expected:
        pdf = OUT_DIR / f"{name}.pdf"
        if pdf.exists():
            size_kb = pdf.stat().st_size / 1024
            marker = "OK" if size_kb > 5 else "TINY"
            print(f"    {name:15s} {size_kb:7.0f} KB  [{marker}]")
        else:
            print(f"    {name:15s}   MISSING")

    # Report any unexpected files
    expected_set = {f"{n}.pdf" for n in expected}
    extras = [p for p in pdfs if p.name not in expected_set]
    if extras:
        print(f"\n  Extra files: {', '.join(p.name for p in extras)}")

    missing = [n for n in expected if not (OUT_DIR / f"{n}.pdf").exists()]
    if missing:
        print(f"\n  Missing: {', '.join(missing)}")
        print("\n  To generate missing figures, run these on the compute cluster:")
        hints = {
            "Figure1": "  manuscript_figure1_consort.py",
            "sFigure1": "  60e_phenotype_cards_early.py",
            "sFigure2": "  75a_dynamic_msm_iptw.py",
            "sFigure6": "  P5_counterfactual_trajectories.py",
        }
        for m in missing:
            if m in hints:
                print(f"    {m}: env/run_py.sh scripts/{hints[m].strip()}")

    print("=" * 65)


if __name__ == "__main__":
    main()
