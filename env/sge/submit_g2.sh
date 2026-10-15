#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log
#
# ── HFpEF ICU DTR: UGE Job Template ──────────────────────────────────
# Compute tier: G2 (dual GPU)
# Resources: 16 cores, 2× GPU
# Note: s_vmem/h_vmem are per-slot VIRTUAL memory limits.
#   CUDA maps GPU HBM into virtual address space (~40-80G),
#   so vmem must be high even when physical RSS is <1G.
# Use case: G-Transformer, DANN, ACTIN, Mamba-CDSP (primary Step C)
#
# Usage:
#   qsub -N step80c env/sge/submit_g2.sh scripts/80c_gtransformer.py
#
# Cluster GPU queues:
#   gpuv.q = V100  |  gpua.q = A100  |  gpuh.q = H100
# Default: gpua.q (A100). Override with: qsub -q gpuh.q ...
# ──────────────────────────────────────────────────────────────────────

#$ -pe def_slot 16
#$ -l s_vmem=64G
#$ -l h_vmem=80G
#$ -l h_rt=48:00:00
#$ -l m_gpu=2
#$ -q gpua.q

PROJECT_ROOT="${PROJECT_ROOT:-${SGE_O_WORKDIR:-$PWD}}"  # run qsub from the project root
cd "${PROJECT_ROOT}"
mkdir -p logs

echo "=== Job: ${JOB_NAME:-unknown} (ID: ${JOB_ID:-local}) ==="
echo "=== Tier: G2 | Slots: ${NSLOTS:-16} | GPU: 2 | Start: $(date) ==="
echo "=== Script: $1 ==="

# Show GPU allocation
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# Source boot.sh WITHOUT set -e (micromamba init may return non-zero spuriously)
source env/boot.sh
if [ $? -ne 0 ]; then
    echo "FATAL: env/boot.sh failed" >&2
    exit 1
fi
echo "=== Environment: $(which python) ==="

# Unbuffered Python output so logs survive OOM-kill / SIGKILL
export PYTHONUNBUFFERED=1

# Enable strict mode AFTER boot.sh; disable -e around python so we capture RC
set -uo pipefail
python "$@"
RC=$?

echo "=== Done: $(date) | exit=$RC ==="
exit $RC
