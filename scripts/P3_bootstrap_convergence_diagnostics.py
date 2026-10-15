#!/usr/bin/env python3
"""P3 — Bootstrap Convergence Diagnostics

Purpose:
    Assess whether the bootstrap confidence intervals for the primary
    DM estimator (Step 80c) have converged, by computing running mean
    and running CI width as a function of bootstrap iteration count.

Inputs:
    - tables/80c_bootstrap_dm_ci.csv  (bootstrap replicate-level results)
      Expected columns: replicate_id (or row index), ate_estimate
      Falls back to synthetic data if the file is absent.

Outputs:
    - reports/P3_bootstrap_convergence_summary.txt
    - figures/P3_bootstrap_convergence.png

Compute tier: C1 (lightweight CPU, direct execution)
Protocol: v3.1-patch, Step P3 (supplementary)
"""

from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Project root ──
ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(ROOT / "src"))


# ── Known point estimates from Step 80c ──
KNOWN_ATE = -0.0137
KNOWN_CI_LO = -0.0149
KNOWN_CI_HI = -0.0126


def load_bootstrap_samples(tables_dir: Path) -> tuple[np.ndarray, bool]:
    """Load bootstrap ATE replicates from CSV, or generate synthetic ones.

    Returns
    -------
    samples : np.ndarray, shape (B,)
        Bootstrap replicate ATE estimates.
    is_synthetic : bool
        True if data was synthetically generated.
    """
    candidates = [
        tables_dir / "80c_bootstrap_dm_ci.csv",
        tables_dir / "80c_bootstrap_results.csv",
        tables_dir / "80c_dm_bootstrap.csv",
    ]

    for fpath in candidates:
        if fpath.exists():
            print(f"[INFO] Loading bootstrap replicates from {fpath}")
            df = pd.read_csv(fpath)
            # Try common column names
            for col in ["ate_estimate", "ate", "ATE", "estimate", "value"]:
                if col in df.columns:
                    samples = df[col].dropna().values.astype(float)
                    if len(samples) >= 50:
                        print(f"[INFO] Loaded {len(samples)} bootstrap replicates (column: {col})")
                        return samples, False
            # If no named column found, try first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                samples = df[numeric_cols[0]].dropna().values.astype(float)
                if len(samples) >= 50:
                    print(f"[INFO] Loaded {len(samples)} replicates from column '{numeric_cols[0]}'")
                    return samples, False
            print(f"[WARN] File {fpath} found but no usable ATE column with >=50 rows.")

    # ── Synthetic fallback ──
    print("[WARN] No bootstrap CSV found. Generating synthetic convergence data.")
    print(f"       Using known estimates: ATE={KNOWN_ATE}, CI=[{KNOWN_CI_LO}, {KNOWN_CI_HI}]")

    rng = np.random.default_rng(seed=42)
    ci_half_width = (KNOWN_CI_HI - KNOWN_CI_LO) / 2.0
    # Approximate bootstrap SE from CI (assuming normal, 95% CI ≈ 1.96 SE)
    bootstrap_se = ci_half_width / 1.96
    samples = rng.normal(loc=KNOWN_ATE, scale=bootstrap_se, size=500)
    return samples, True


def compute_running_statistics(
    samples: np.ndarray,
    checkpoints: list[int] | None = None,
) -> pd.DataFrame:
    """Compute running mean and CI width at each checkpoint.

    Parameters
    ----------
    samples : np.ndarray
        Bootstrap replicate ATE values.
    checkpoints : list[int], optional
        Bootstrap iteration counts to evaluate. Defaults to 50, 100, ..., len(samples).

    Returns
    -------
    pd.DataFrame with columns: B, running_mean, ci_lo, ci_hi, ci_width
    """
    n = len(samples)
    if checkpoints is None:
        step = 50
        checkpoints = list(range(step, n + 1, step))
        if checkpoints[-1] != n:
            checkpoints.append(n)

    records = []
    for b in tqdm(checkpoints, desc="Running statistics", unit="checkpoint"):
        subset = samples[:b]
        mean_val = np.mean(subset)
        ci_lo = np.percentile(subset, 2.5)
        ci_hi = np.percentile(subset, 97.5)
        records.append({
            "B": b,
            "running_mean": mean_val,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "ci_width": ci_hi - ci_lo,
        })

    return pd.DataFrame(records)


def check_convergence(df: pd.DataFrame, threshold: float = 0.05) -> dict:
    """Check if CI width converged within threshold of final width by B=300.

    Returns a dict with convergence diagnostics.
    """
    final_row = df.iloc[-1]
    final_width = final_row["ci_width"]
    final_B = int(final_row["B"])

    # Find B=300 row (or closest)
    b300_rows = df[df["B"] <= 300]
    if len(b300_rows) == 0:
        b300_row = df.iloc[0]
    else:
        b300_row = b300_rows.iloc[-1]

    width_at_300 = b300_row["ci_width"]
    b_actual = int(b300_row["B"])
    pct_of_final = width_at_300 / final_width if final_width != 0 else float("inf")
    converged = abs(1.0 - pct_of_final) <= threshold

    # Running mean stability: CV over last 100 samples
    # Use last ~100 bootstrap replicates worth of running means
    tail_rows = df[df["B"] >= max(df["B"].max() - 100, df["B"].min())]
    running_means = tail_rows["running_mean"].values
    if len(running_means) > 1 and np.mean(np.abs(running_means)) > 0:
        cv = np.std(running_means) / np.abs(np.mean(running_means))
    else:
        cv = 0.0

    return {
        "final_B": final_B,
        "final_width": final_width,
        "final_mean": final_row["running_mean"],
        "final_ci_lo": final_row["ci_lo"],
        "final_ci_hi": final_row["ci_hi"],
        "b300_actual": b_actual,
        "width_at_300": width_at_300,
        "pct_of_final": pct_of_final,
        "converged": converged,
        "cv_tail": cv,
    }


def write_report(
    report_path: Path,
    diag: dict,
    is_synthetic: bool,
    n_samples: int,
) -> None:
    """Write convergence summary report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "=" * 60,
        "BOOTSTRAP CONVERGENCE DIAGNOSTICS",
        f"Generated: {timestamp}",
        "=" * 60,
        "",
    ]

    if is_synthetic:
        lines += [
            "*** NOTE: Using SYNTHETIC bootstrap data ***",
            f"    Known ATE = {KNOWN_ATE}, CI = [{KNOWN_CI_LO}, {KNOWN_CI_HI}]",
            "    Real bootstrap CSV not found; this demonstrates the analysis framework.",
            "",
        ]

    lines += [
        f"Total bootstrap replicates: {n_samples}",
        "",
        "── Convergence Summary ──",
        f"- Final CI width at B={diag['final_B']}: {diag['final_width']:.4f} pp",
        f"- CI width at B={diag['b300_actual']}: {diag['width_at_300']:.4f} pp "
        f"({diag['pct_of_final'] * 100:.1f}% of final)",
        f"- Convergence criterion (±5% by B=300): {'PASS' if diag['converged'] else 'FAIL'}",
        f"- Running mean stability: coefficient of variation over last 100 samples = {diag['cv_tail']:.6f}",
        "",
        "── Final Estimates ──",
        f"- Running mean at B={diag['final_B']}: {diag['final_mean']:.6f}",
        f"- 95% CI: [{diag['final_ci_lo']:.6f}, {diag['final_ci_hi']:.6f}]",
        "",
    ]

    if diag["converged"]:
        lines.append("Conclusion: Bootstrap CIs have converged adequately.")
    else:
        lines.append(
            "Conclusion: Bootstrap CIs have NOT converged by B=300. "
            "Consider increasing the number of bootstrap replicates."
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] Report written to {report_path}")


def make_convergence_plot(
    df: pd.DataFrame,
    fig_path: Path,
    is_synthetic: bool,
) -> None:
    """Produce a two-panel convergence plot: running mean + CI width."""
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # ── Panel 1: Running mean ──
    ax1 = axes[0]
    ax1.plot(df["B"], df["running_mean"] * 100, "o-", color="steelblue", markersize=4)
    ax1.fill_between(
        df["B"],
        df["ci_lo"] * 100,
        df["ci_hi"] * 100,
        alpha=0.2,
        color="steelblue",
        label="95% CI",
    )
    ax1.axhline(KNOWN_ATE * 100, ls="--", color="grey", lw=0.8, label=f"Known ATE ({KNOWN_ATE*100:.2f} pp)")
    ax1.set_ylabel("Running Mean ATE (pp)")
    ax1.legend(fontsize=8)
    title_suffix = " [SYNTHETIC]" if is_synthetic else ""
    ax1.set_title(f"Bootstrap Convergence — Running Mean{title_suffix}")

    # ── Panel 2: CI width ──
    ax2 = axes[1]
    ax2.plot(df["B"], df["ci_width"] * 100, "s-", color="darkorange", markersize=4)
    final_width = df["ci_width"].iloc[-1] * 100
    ax2.axhline(final_width, ls="--", color="grey", lw=0.8, label=f"Final width ({final_width:.3f} pp)")
    ax2.axhline(final_width * 1.05, ls=":", color="red", lw=0.7, alpha=0.6, label="±5% band")
    ax2.axhline(final_width * 0.95, ls=":", color="red", lw=0.7, alpha=0.6)
    ax2.axvline(300, ls="--", color="green", lw=0.7, alpha=0.6, label="B=300 checkpoint")
    ax2.set_xlabel("Bootstrap Replicates (B)")
    ax2.set_ylabel("CI Width (pp)")
    ax2.legend(fontsize=8)
    ax2.set_title("Bootstrap Convergence — CI Width")

    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Figure saved to {fig_path}")


def main() -> None:
    """Entry point for bootstrap convergence diagnostics."""
    print("=" * 60)
    print("P3 Bootstrap Convergence Diagnostics")
    print("=" * 60)

    tables_dir = ROOT / "tables"
    reports_dir = ROOT / "reports"
    figures_dir = ROOT / "figures"

    # ── Load data ──
    samples, is_synthetic = load_bootstrap_samples(tables_dir)
    n_samples = len(samples)
    print(f"[INFO] Working with {n_samples} bootstrap replicates "
          f"({'synthetic' if is_synthetic else 'empirical'})")

    # ── Compute running statistics ──
    df = compute_running_statistics(samples)
    print(f"\n[INFO] Running statistics computed at {len(df)} checkpoints")
    print(df.to_string(index=False, float_format="%.6f"))

    # ── Convergence check ──
    diag = check_convergence(df)
    status = "PASS" if diag["converged"] else "FAIL"
    print(f"\n[RESULT] Convergence criterion (±5% by B=300): {status}")

    # ── Write report ──
    report_path = reports_dir / "P3_bootstrap_convergence_summary.txt"
    write_report(report_path, diag, is_synthetic, n_samples)

    # ── Plot ──
    fig_path = figures_dir / "P3_bootstrap_convergence.png"
    make_convergence_plot(df, fig_path, is_synthetic)

    print("\n[DONE] P3 bootstrap convergence diagnostics complete.")


if __name__ == "__main__":
    main()
