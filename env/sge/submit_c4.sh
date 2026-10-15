#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log
#
# ── HFpEF ICU DTR: UGE Job Template ──────────────────────────────────
# Compute tier: C4 (very large-scale CPU)
# Resources: 32 cores, 128–256 GB RAM
# Use case: HiRID extraction/IO, massive sparse matrices
#
# Usage:
#   qsub -N step40b_hirid env/sge/submit_c4.sh scripts/40b_grid_2h_builder_hirid.py
# ──────────────────────────────────────────────────────────────────────

#$ -pe def_slot 32
#$ -l s_vmem=8G
#$ -l h_vmem=10G
#$ -l h_rt=24:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SGE_O_WORKDIR:-$PWD}}"  # run qsub from the project root
cd "${PROJECT_ROOT}"
mkdir -p logs

echo "=== Job: ${JOB_NAME:-unknown} (ID: ${JOB_ID:-local}) ==="
echo "=== Tier: C4 | Slots: ${NSLOTS:-32} | Start: $(date) ==="
echo "=== Script: $1 ==="

source env/boot.sh
python "$@"

echo "=== Done: $(date) ==="
