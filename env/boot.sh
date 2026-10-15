#!/usr/bin/env bash
# NOTE: Do NOT use "set -euo pipefail" here.
# This file is meant to be sourced into the current shell.
# Strict mode would leak into the caller and cause the shell to exit
# on any subsequent non-zero return (e.g. a Python script failing),
# which is the root cause of "getting kicked back to login node".

_HFPEF_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_HFPEF_ENV_DIR}/env.paths" || { return 1 2>/dev/null || exit 1; }

# Defensive: clear inherited PYTHONHOME before activation
unset PYTHONHOME
source "${_HFPEF_ENV_DIR}/mamba_init.sh" || { return 1 2>/dev/null || exit 1; }
# Sanitize MAMBA_ROOT_PREFIX: repoint deprecated scratch roots
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX%/}"
case "${MAMBA_ROOT_PREFIX:-}" in
  ""|/swork/*/mamba-root|/swork/*/*/mamba-root|/swork/*/*/*/mamba-root)
    export MAMBA_ROOT_PREFIX="$HOME/micromamba"
    ;;
esac



micromamba activate hfpef-icu-dtr || {
  echo "micromamba env 'hfpef-icu-dtr' not found or activation failed." >&2
  echo "Create it with:" >&2
  echo "  micromamba create -y -f ${_HFPEF_ENV_DIR}/environment.base.yml" >&2
  return 1 2>/dev/null || exit 1
}
