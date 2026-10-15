#!/usr/bin/env python3
"""95e — Archive Release Bundle

Protocol: v3.1, Step 95e
Compute: C1

Purpose
-------
Archive configs, manifests, registries, model metadata, and final reports
into a self-contained release bundle directory. This bundle contains
everything needed to understand and reproduce the study results (excluding
raw patient data and model checkpoints).

Outputs
-------
- release/hfpef_icu_dtr_v3_1_release/  (directory)
  ├── configs/          (all YAML/JSON configs)
  ├── tables/           (key result tables)
  ├── figures/          (manuscript figures)
  ├── reports/          (summary reports and cards)
  ├── scripts/          (rerun pipeline)
  ├── protocol/         (protocol document)
  └── MANIFEST.txt      (file listing with checksums)
- reports/95e_release_bundle_summary.txt
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────
ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
RELEASE_DIR = ROOT / "release" / "hfpef_icu_dtr_v3_1_release"
REPORTS_DIR = ROOT / "reports"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("95e")


# ── File lists to include ─────────────────────────────────────────────
# (source_dir, glob_pattern, target_subdir)
BUNDLE_SPECS = [
    # Configs
    ("configs", "*.yaml", "configs"),
    ("configs", "*.json", "configs"),

    # Key result tables (95a registry + 95b manuscript tables)
    ("tables", "95a_results_registry.*", "tables"),
    ("tables", "95b_*.csv", "tables"),
    ("tables", "90d_transportability_grades.csv", "tables"),
    ("tables", "90e_consistency_checklist.csv", "tables"),

    # Figures
    ("figures", "95b_*.png", "figures"),
    ("figures", "75a_*.png", "figures"),
    ("figures", "85d_*.png", "figures"),

    # Reports
    ("reports", "95a_*.txt", "reports"),
    ("reports", "95b_*.txt", "reports"),
    ("reports", "95c_*.md", "reports"),
    ("reports", "95d_*.txt", "reports"),
    ("reports", "90d_transportability_summary_memo.txt", "reports"),
    ("reports", "90e_cross_dataset_consistency_report.txt", "reports"),
    ("reports", "80f_step_c_cross_model_review.md", "reports"),

    # Rerun script
    ("scripts", "95d_rerun_from_frozen.sh", "scripts"),

    # Protocol
    ("docs/protocol", "HFpEF_ICU_DTR_protocol_v3_1_handoff_packet.md", "protocol"),
    ("docs/protocol", "v3_1_patch_supplementary_analyses.md", "protocol"),
]


def sha256_file(path: Path) -> str:
    """Compute SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 72)
    print("95e — Archive Release Bundle")
    print("=" * 72)
    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Clean and create release directory
    if RELEASE_DIR.exists():
        shutil.rmtree(RELEASE_DIR)
    RELEASE_DIR.mkdir(parents=True)
    log.info("Release directory: %s", RELEASE_DIR)

    # Copy files
    log.info("[1/3] Copying files to release bundle...")
    copied_files: list[tuple[str, str, int]] = []  # (rel_path, sha256, size)
    missing_files: list[str] = []

    for src_dir, pattern, tgt_dir in BUNDLE_SPECS:
        src_path = ROOT / src_dir
        tgt_path = RELEASE_DIR / tgt_dir
        tgt_path.mkdir(parents=True, exist_ok=True)

        matches = list(src_path.glob(pattern))
        if not matches:
            missing_files.append(f"{src_dir}/{pattern}")
            continue

        for f in matches:
            if f.is_file():
                dest = tgt_path / f.name
                shutil.copy2(f, dest)
                sha = sha256_file(dest)
                size = dest.stat().st_size
                rel = str(dest.relative_to(RELEASE_DIR))
                copied_files.append((rel, sha, size))
                log.info("  %s (%d bytes)", rel, size)

    # Generate MANIFEST
    log.info("[2/3] Generating MANIFEST.txt...")
    manifest_lines = [
        f"HFpEF ICU DTR v3.1 Release Bundle",
        f"Generated: {ts}",
        f"Total files: {len(copied_files)}",
        f"Total size: {sum(s for _, _, s in copied_files):,} bytes",
        "",
        f"{'File':<60s} {'SHA256':<66s} {'Size':>10s}",
        f"{'─'*60} {'─'*66} {'─'*10}",
    ]
    for rel, sha, size in sorted(copied_files):
        manifest_lines.append(f"{rel:<60s} {sha:<66s} {size:>10,d}")

    if missing_files:
        manifest_lines.extend([
            "",
            "MISSING (not found, non-critical):",
        ])
        for mf in missing_files:
            manifest_lines.append(f"  {mf}")

    manifest_text = "\n".join(manifest_lines) + "\n"
    manifest_path = RELEASE_DIR / "MANIFEST.txt"
    with open(manifest_path, "w") as f:
        f.write(manifest_text)
    log.info("  Saved: %s", manifest_path)

    # Summary report
    log.info("[3/3] Generating summary report...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_lines = [
        "=" * 72,
        "95e — Release Bundle Summary",
        "=" * 72,
        f"Generated: {ts}",
        f"Bundle path: {RELEASE_DIR}",
        f"Files copied: {len(copied_files)}",
        f"Total size: {sum(s for _, _, s in copied_files):,} bytes",
        f"Missing patterns: {len(missing_files)}",
        "",
        "Contents by directory:",
    ]

    dir_counts: dict[str, int] = {}
    for rel, _, _ in copied_files:
        d = rel.split("/")[0]
        dir_counts[d] = dir_counts.get(d, 0) + 1
    for d, cnt in sorted(dir_counts.items()):
        summary_lines.append(f"  {d}/: {cnt} files")

    if missing_files:
        summary_lines.extend(["", "Missing (non-critical):"])
        for mf in missing_files:
            summary_lines.append(f"  {mf}")

    summary_lines.extend([
        "",
        "Note: Raw patient data and model checkpoints are NOT included.",
        "      See MANIFEST.txt in the bundle for file checksums.",
    ])

    summary_path = REPORTS_DIR / "95e_release_bundle_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    log.info("  Saved: %s", summary_path)

    elapsed = time.time() - t0
    print()
    print("=" * 72)
    print(f"95e COMPLETE. {len(copied_files)} files bundled in {elapsed:.1f}s.")
    print(f"Bundle: {RELEASE_DIR}")
    print(f"Summary: {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
