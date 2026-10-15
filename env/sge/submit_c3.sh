#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log
#
# ── HFpEF ICU DTR: UGE Job Template ──────────────────────────────────
# Compute tier: C3 (intensive CPU)
# Resources: 16 cores, 64–128 GB RAM
#
# Usage:
#   qsub -N step40b env/sge/submit_c3.sh scripts/40b_grid_2h_builder_mimic.py
# ──────────────────────────────────────────────────────────────────────

#$ -pe def_slot 16
#$ -l s_vmem=8G
#$ -l h_vmem=10G
#$ -l h_rt=12:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SGE_O_WORKDIR:-$PWD}}"  # run qsub from the project root
cd "${PROJECT_ROOT}"
mkdir -p logs

echo "=== Job: ${JOB_NAME:-unknown} (ID: ${JOB_ID:-local}) ==="
echo "=== Tier: C3 | Slots: ${NSLOTS:-16} | Start: $(date) ==="
echo "=== Script: $1 ==="

source env/boot.sh
python "$@"

echo "=== Done: $(date) ==="
