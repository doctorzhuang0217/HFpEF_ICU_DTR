#!/usr/bin/env python3
"""
Merge manuscript figures into composite panels for journal submission.

Figure 1: CONSORT flowchart (standalone, no merge needed)
Figure 2: (A) Cross-model ATE forest plot + (B) HFpEF-specific signal
Figure 3: (A) Sensitivity analysis heatmap + (B) Positivity grid stability
Figure 4: External validation (standalone)
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")

# Force CJK font
import matplotlib.font_manager as fm
fm._load_fontmanager(try_read_cache=False)
matplotlib.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "Noto Sans SC", "DejaVu Sans", "Arial"
]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "figures" / "submission"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def merge_vertical(img_a_path, img_b_path, out_path, label_a="A", label_b="B"):
    """Merge two images vertically with panel labels OUTSIDE the image area."""
    img_a = mpimg.imread(str(img_a_path))
    img_b = mpimg.imread(str(img_b_path))

    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]
    aspect_a = h_a / w_a
    aspect_b = h_b / w_b

    # Figure dimensions
    fig_w = 11
    # Height for each panel proportional to aspect
    panel_a_h = fig_w * aspect_a
    panel_b_h = fig_w * aspect_b
    label_h = 0.35  # height for each label row
    gap = 0.3
    total_h = panel_a_h + panel_b_h + 2 * label_h + gap + 0.3

    fig = plt.figure(figsize=(fig_w, total_h))

    # Compute positions (bottom-up)
    bottom_margin = 0.15 / total_h
    top_margin = 0.15 / total_h

    usable = 1 - bottom_margin - top_margin
    label_frac = label_h / total_h
    gap_frac = gap / total_h
    panel_a_frac = (panel_a_h / total_h) * usable / (usable)
    panel_b_frac = (panel_b_h / total_h) * usable / (usable)

    # Normalize
    total_content = panel_a_h + panel_b_h + 2 * label_h + gap
    b_label_bottom = bottom_margin
    b_img_bottom = b_label_bottom + label_h / total_h
    a_label_bottom = b_img_bottom + panel_b_h / total_h + gap / total_h
    a_img_bottom = a_label_bottom + label_h / total_h

    # Panel A label
    fig.text(0.02, a_img_bottom + panel_a_h / total_h - 0.01,
             label_a, fontsize=24, fontweight="bold", va="top", ha="left",
             fontfamily="sans-serif")

    # Panel A image
    ax_a = fig.add_axes([0.05, a_img_bottom, 0.93, panel_a_h / total_h])
    ax_a.imshow(img_a)
    ax_a.axis("off")

    # Panel B label
    fig.text(0.02, b_img_bottom + panel_b_h / total_h - 0.01,
             label_b, fontsize=24, fontweight="bold", va="top", ha="left",
             fontfamily="sans-serif")

    # Panel B image
    ax_b = fig.add_axes([0.05, b_img_bottom, 0.93, panel_b_h / total_h])
    ax_b.imshow(img_b)
    ax_b.axis("off")

    fig.savefig(str(out_path), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


def copy_standalone(src_path, out_path):
    import shutil
    shutil.copy2(str(src_path), str(out_path))
    print(f"  Copied: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


def gen_sfigure4b_heatmap(output_path: Path, try_csv: bool = True):
    """Regenerate sFigure 4B (CATE benefit profile heatmap) with publication labels.

    Attempts to load from P2 CATE data + phenotype data on the compute cluster;
    falls back to manuscript-reported values if data files are unavailable.
    """
    import numpy as np

    # ── Label mapping ────────────────────────────────────────────────
    HF_ORDER = ["HF (EF Unknown)", "HFmrEF", "HFpEF", "HFrEF", "No HF"]
    PHENO_ORDER = ["Non-Shock", "Shock-Predominant"]
    _HF_MAP = {
        "hfpef": "HFpEF", "hfref": "HFrEF", "hfmref": "HFmrEF",
        "no_hf": "No HF", "hf_ef_unknown": "HF (EF Unknown)",
    }
    _PHENO_MAP = {
        "non_shock": "Non-Shock", "shock_predominant": "Shock-Predominant",
    }

    data = None

    # ── Try loading from source data (compute cluster) ─────────────────────
    if try_csv:
        cate_csv = ROOT / "tables" / "P2_cate_distribution.csv"
        pheno_candidates = [
            ROOT / "data" / "processed" / "phenotype_labels_frozen_v1_mimic.parquet",
            ROOT / "data" / "interim" / "60d_rule_phenotype.parquet",
        ]
        try:
            import pandas as pd
            if cate_csv.exists():
                cate_df = pd.read_csv(cate_csv)
                pheno_df = None
                for p in pheno_candidates:
                    if p.exists():
                        pheno_df = pd.read_parquet(p)
                        break
                if pheno_df is not None and "hf_subtype" in pheno_df.columns:
                    merged = cate_df.merge(
                        pheno_df.drop_duplicates("stay_id"),
                        on="stay_id", how="left",
                    )
                    if ("hf_subtype" in merged.columns and
                            "rule_phenotype_2class" in merged.columns):
                        merged["hf_label"] = merged["hf_subtype"].map(
                            lambda v: _HF_MAP.get(v, v))
                        merged["pheno_label"] = merged["rule_phenotype_2class"].map(
                            lambda v: _PHENO_MAP.get(v, v))
                        pivot = merged.pivot_table(
                            values="cate", index="hf_label",
                            columns="pheno_label", aggfunc="mean",
                        ) * 100  # to pp
                        pivot = pivot.reindex(index=HF_ORDER, columns=PHENO_ORDER)
                        data = pivot.values
                        print("  Loaded heatmap data from source CSV + phenotype")
        except Exception as exc:
            print(f"  CSV load failed ({exc}); using manuscript values")

    # ── Fallback: manuscript-reported values ──────────────────────────
    if data is None:
        data = np.array([
            [-1.20, -2.75],   # HF (EF Unknown)
            [-1.42, -2.18],   # HFmrEF
            [-1.42, -3.33],   # HFpEF
            [-1.53, -3.89],   # HFrEF
            [-0.59, -2.48],   # No HF
        ])
        print("  Using manuscript-reported values for heatmap")

    # ── Plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 5))
    vmax = max(abs(data.min()), abs(data.max()))
    im = ax.imshow(data, cmap="RdBu_r", aspect="auto",
                   vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(PHENO_ORDER)))
    ax.set_xticklabels(PHENO_ORDER, fontsize=11, rotation=0, ha="center")
    ax.set_yticks(range(len(HF_ORDER)))
    ax.set_yticklabels(HF_ORDER, fontsize=11)
    ax.set_xlabel("Hemodynamic Phenotype", fontsize=12, labelpad=8)
    ax.set_ylabel("HF Subtype", fontsize=12, labelpad=8)
    ax.set_title("Mean CATE (pp)", fontsize=13, pad=10)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            color = "white" if abs(val) > vmax * 0.45 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    pdf_path = str(output_path).replace(".png", ".pdf")
    fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")
    print(f"  Saved: {pdf_path}")


def merge_sfigure4(panel_a_path: Path, panel_b_path: Path,
                   out_png: Path, out_pdf: Path):
    """Merge sFigure 4A (CATE histogram) + 4B (benefit profile heatmap)."""
    img_a = mpimg.imread(str(panel_a_path))
    img_b = mpimg.imread(str(panel_b_path))

    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]

    # Scale panel B to match panel A height
    scale = h_a / h_b
    new_w_b = int(w_b * scale)

    from PIL import Image
    import numpy as np
    pil_b = Image.fromarray((img_b * 255).astype(np.uint8) if img_b.max() <= 1.0
                            else img_b)
    pil_b = pil_b.resize((new_w_b, h_a), Image.LANCZOS)
    img_b_resized = np.array(pil_b).astype(np.float32) / 255.0

    # Side-by-side layout
    total_w = w_a + new_w_b
    fig_w = 16
    fig_h = fig_w * h_a / total_w + 0.6

    fig = plt.figure(figsize=(fig_w, fig_h))
    frac_a = w_a / total_w
    frac_b = new_w_b / total_w

    label_h = 0.4 / fig_h
    img_bottom = 0.02
    img_top = 1 - label_h - 0.02

    # Panel A
    ax_a = fig.add_axes([0.01, img_bottom, frac_a * 0.96, img_top - img_bottom])
    ax_a.imshow(img_a, interpolation="lanczos")
    ax_a.axis("off")
    fig.text(0.01, img_top + 0.01, "A", fontsize=22, fontweight="bold",
             va="bottom", ha="left", fontfamily="sans-serif")

    # Panel B
    left_b = frac_a * 0.96 + 0.02
    ax_b = fig.add_axes([left_b, img_bottom, frac_b * 0.96, img_top - img_bottom])
    ax_b.imshow(img_b_resized, interpolation="lanczos")
    ax_b.axis("off")
    fig.text(left_b - 0.005, img_top + 0.01, "B", fontsize=22, fontweight="bold",
             va="bottom", ha="left", fontfamily="sans-serif")

    fig.savefig(str(out_png), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    fig.savefig(str(out_pdf), format="pdf", dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {out_png} ({out_png.stat().st_size / 1024:.0f} KB)")
    print(f"  Saved: {out_pdf}")


def main():
    print("=" * 60)
    print("Merging manuscript figures for submission")
    print("=" * 60)

    # ═══════════════════════════════════════════════════════════
    # ICM medicine-first layout (4 main figures)
    # ═══════════════════════════════════════════════════════════

    print("\n[Figure 1] Clinical schematic — TBD (manual/Illustrator)")

    print("\n[Figure 2] Primary result (R-generated composite)")
    r_fig2 = FIG_DIR / "R" / "newF2_primary_result.png"
    if r_fig2.exists():
        copy_standalone(r_fig2, OUT_DIR / "Figure2_primary_result.png")
    else:
        print("  WARNING: R figure not found, falling back to legacy merge")
        merge_vertical(
            FIG_DIR / "95b_f1_cross_model_ate_forest.png",
            FIG_DIR / "95b_f2_hfpef_specificity.png",
            OUT_DIR / "Figure2_primary_result.png",
        )

    print("\n[Figure 3] External validation — standalone copy")
    copy_standalone(
        FIG_DIR / "R" / "F4_external_validation.png"
        if (FIG_DIR / "R" / "F4_external_validation.png").exists()
        else FIG_DIR / "95b_f4_external_validation.png",
        OUT_DIR / "Figure3_external_validation.png",
    )

    print("\n[Figure 4] Mechanism (R-generated 2-panel)")
    r_fig4 = FIG_DIR / "R" / "newF4_mechanism.png"
    if r_fig4.exists():
        copy_standalone(r_fig4, OUT_DIR / "Figure4_mechanism.png")
    else:
        print("  WARNING: R figure not found, skipping")

    # ── Supplementary: demoted main figures ──
    print("\n[sFigure 1] CONSORT flowchart (demoted from old Figure 1)")

    print("\n[sFigure 2] Full cross-model forest (demoted from old Figure 2A)")
    r_full_forest = FIG_DIR / "R" / "F1_cross_model_ate_forest.png"
    if r_full_forest.exists():
        copy_standalone(r_full_forest, OUT_DIR / "sFigure2_full_forest.png")

    print("\n[sFigure 3] Robustness evidence (demoted from old Figure 3)")
    merge_vertical(
        FIG_DIR / "R" / "F3_sensitivity_heatmap.png"
        if (FIG_DIR / "R" / "F3_sensitivity_heatmap.png").exists()
        else FIG_DIR / "95b_f3_sensitivity_heatmap.png",
        FIG_DIR / "R" / "85d_positivity_grid_stability.png"
        if (FIG_DIR / "R" / "85d_positivity_grid_stability.png").exists()
        else FIG_DIR / "85d_positivity_grid_stability.png",
        OUT_DIR / "sFigure3_robustness.png",
    )

    print("\n[sFigure 4] Trimming stability (demoted from old Figure 3B)")
    r_trim = FIG_DIR / "R" / "F5_trimming_stability.png"
    if r_trim.exists():
        copy_standalone(r_trim, OUT_DIR / "sFigure4_trimming.png")

    print("\n[sFigure 5] HFI dose-response (demoted from old Figure 5B)")
    r_hfi = FIG_DIR / "R" / "P9_hfi_distribution.png"
    if r_hfi.exists():
        copy_standalone(r_hfi, OUT_DIR / "sFigure5_hfi_dose_response.png")

    # ── sFigure 4: CATE distribution (A) + benefit profile heatmap (B) ──
    print("\n[sFigure 4] Regenerating 4B heatmap + merging 4A/4B")
    sfig4b_path = FIG_DIR / "manuscript" / "sFigure4B_benefit_profile.png"
    sfig4b_path.parent.mkdir(parents=True, exist_ok=True)
    gen_sfigure4b_heatmap(sfig4b_path)

    # Panel A: use the cleaned manuscript version
    sfig4a_path = FIG_DIR / "manuscript" / "sFigure4.png"
    if not sfig4a_path.exists():
        sfig4a_path = FIG_DIR / "P2_cate_histogram.png"

    sfig4_out_png = FIG_DIR / "manuscript" / "sFigure4_composite.png"
    sfig4_out_pdf = ROOT / "manuscript" / "figures_pdf" / "sFigure4.pdf"
    merge_sfigure4(sfig4a_path, sfig4b_path, sfig4_out_png, sfig4_out_pdf)

    print("\n" + "=" * 60)
    print("Done. Submission figures in:", OUT_DIR)
    print("sFigure 4 composite in:", sfig4_out_png.parent)


if __name__ == "__main__":
    main()
