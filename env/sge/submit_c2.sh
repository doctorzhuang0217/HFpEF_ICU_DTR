#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log
#
# ── HFpEF ICU DTR: UGE Job Template ──────────────────────────────────
# Compute tier: C2 (standard CPU)
# Resources: 8 cores, 32–64 GB RAM
#
# Usage:
#   qsub -N step35a env/sge/submit_c2.sh scripts/35a_build_comparator_cohorts.py
# ──────────────────────────────────────────────────────────────────────

#$ -pe def_slot 8
#$ -l s_vmem=8G
#$ -l h_vmem=10G
#$ -l h_rt=04:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SGE_O_WORKDIR:-$PWD}}"  # run qsub from the project root
cd "${PROJECT_ROOT}"
mkdir -p logs

echo "=== Job: ${JOB_NAME:-unknown} (ID: ${JOB_ID:-local}) ==="
echo "=== Tier: C2 | Slots: ${NSLOTS:-8} | Start: $(date) ==="
echo "=== Script: $1 ==="

source env/boot.sh
python "$@"

echo "=== Done: $(date) ==="
