#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.log
#
# ── HFpEF ICU DTR: UGE Job Template ──────────────────────────────────
# Compute tier: G1 (single GPU)
# Resources: 8 cores, 1× GPU
# Note: s_vmem/h_vmem are per-slot VIRTUAL memory limits.
#   CUDA maps GPU HBM into virtual address space (~40-80G),
#   so vmem must be high even when physical RSS is <1G.
# Use case: baseline encoders, probe tasks, small confirmatory models
#
# Usage:
#   qsub -N step50a env/sge/submit_g1.sh scripts/50a_encoder_baseline.py
#
# Cluster GPU queues & h_rt limits:
#   gpua.q = A100  (max h_rt=96:00:00)
#   gpuh.q = H100  (max h_rt=96:00:00)
#   gpul.q = L4    (max h_rt=96:00:00, 8×L4 per node, 96 cores, 1.5TB RAM)
# Default: gpua.q (A100). Override with: qsub -q gpul.q ...
# ──────────────────────────────────────────────────────────────────────

#$ -pe def_slot 8
#$ -l s_vmem=64G
#$ -l h_vmem=80G
#$ -l h_rt=96:00:00
#$ -l m_gpu=1
#$ -q gpua.q

PROJECT_ROOT="${PROJECT_ROOT:-${SGE_O_WORKDIR:-$PWD}}"  # run qsub from the project root
cd "${PROJECT_ROOT}"
mkdir -p logs

echo "=== Job: ${JOB_NAME:-unknown} (ID: ${JOB_ID:-local}) ==="
echo "=== Tier: G1 | Slots: ${NSLOTS:-8} | GPU: 1 | Start: $(date) ==="
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

# ── Thread control: prevent CPU oversubscription ──
# NSLOTS is set by UGE to match -pe def_slot value.
# Without these, PyTorch/NumPy will use ALL node cores and trigger the
# Cluster CPU-overuse detector.
_SLOTS="${NSLOTS:-8}"
export OMP_NUM_THREADS="${_SLOTS}"
export MKL_NUM_THREADS="${_SLOTS}"
export OPENBLAS_NUM_THREADS="${_SLOTS}"
export NUMEXPR_NUM_THREADS="${_SLOTS}"
export TORCH_NUM_THREADS="${_SLOTS}"
echo "=== Thread limits: OMP/MKL/TORCH=${_SLOTS} ==="

# Enable strict mode AFTER boot.sh; disable -e around python so we capture RC
set -uo pipefail

# Resolve the env's python by absolute path to avoid PATH-order issues that
# can appear on GPU compute nodes when /etc/profile.d module init runs after
# boot.sh and pushes a system python ahead of the micromamba env's python.
# (Symptom: P16b qsub log 2026-05-10 — `python` resolved but `import torch`
# failed.)
ENV_PY="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}/envs/hfpef-icu-dtr/bin/python"
if [ ! -x "$ENV_PY" ]; then
    echo "FATAL: env python not found at $ENV_PY" >&2
    exit 1
fi
echo "=== Python: $ENV_PY ==="
"$ENV_PY" "$@"
RC=$?

echo "=== Done: $(date) | exit=$RC ==="
exit $RC
