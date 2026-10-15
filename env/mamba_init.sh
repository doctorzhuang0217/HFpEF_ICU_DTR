#!/usr/bin/env bash

export PATH="$HOME/.local/bin:$PATH"

if ! command -v micromamba >/dev/null 2>&1; then
  echo "micromamba not found in PATH. Expected it around ~/.local/bin/micromamba" >&2
  return 1 2>/dev/null || exit 1
fi

eval "$(micromamba shell hook -s bash)"
