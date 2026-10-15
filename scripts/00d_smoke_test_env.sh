#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env/boot.sh"

python - <<'PY'
import os, sys
print("python:", sys.executable)
for k in ["PROJECT_ROOT", "MIMIC_ROOT", "EICU_ROOT", "HIRID_ROOT", "STAGING_ROOT", "PROCESSED_ROOT"]:
    print(f"{k}={os.environ.get(k)}")
import pandas, polars, duckdb, pyarrow, sklearn
print("imports: ok")
PY
