#!/usr/bin/env bash
# ── env/install_torch.sh ──────────────────────────────────────────────
# Install PyTorch (CUDA 12.1 wheel) into the hfpef-icu-dtr micromamba
# env. Run this AFTER every (re)build of the env. The base recipe in
# environment.base.yml does NOT list torch — see the comment block at
# the bottom of that file for rationale.
#
# Verifies the install on completion and (when run on a GPU node)
# checks torch.cuda.is_available().
#
# Usage:
#   bash env/install_torch.sh                  # install torch>=2.4
#   bash env/install_torch.sh 2.5.1            # pin specific version
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

VERSION_SPEC="${1:->=2.4}"
INDEX_URL="https://download.pytorch.org/whl/cu121"

source env/boot.sh

echo "=== Installing torch ${VERSION_SPEC} from ${INDEX_URL} ==="
python -m pip install --index-url "${INDEX_URL}" "torch${VERSION_SPEC}"

echo "=== Verifying ==="
python -c "
import torch
print(f'torch        : {torch.__version__}')
print(f'cuda built   : {torch.version.cuda}')
print(f'cuda runtime : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'device count : {torch.cuda.device_count()}')
    print(f'device 0     : {torch.cuda.get_device_name(0)}')
"
echo "=== Done. cuda.is_available() will be False on the login node. ==="
echo "=== Re-verify on a GPU compute node before G-tier jobs.       ==="
