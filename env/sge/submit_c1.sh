#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log
#
# ── HFpEF ICU DTR: UGE Job Template ──────────────────────────────────
# Compute tier: C1 (lightweight CPU)
# Resources: 4 cores, 16 GB RAM, ~30 min wall time
#
# Usage:
#   qsub -N step35e env/sge/submit_c1.sh scripts/35e_phase05b_comparator_pack.py
#
# Or override resources inline:
#   qsub -N step35e -l s_vmem=32G -pe def_slot 8 env/sge/submit_c1.sh scripts/35e.py
# ──────────────────────────────────────────────────────────────────────

#$ -pe def_slot 4
#$ -l s_vmem=4G
#$ -l h_vmem=6G
#$ -l h_rt=01:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SGE_O_WORKDIR:-$PWD}}"  # run qsub from the project root
cd "${PROJECT_ROOT}"
mkdir -p logs

echo "=== Job: ${JOB_NAME:-unknown} (ID: ${JOB_ID:-local}) ==="
echo "=== Tier: C1 | Slots: ${NSLOTS:-4} | Start: $(date) ==="
echo "=== Script: $1 ==="

source env/boot.sh
python "$@"

echo "=== Done: $(date) ==="
