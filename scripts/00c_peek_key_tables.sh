#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env/env.paths"

echo "== MIMIC icustays header =="
gzip -dc "$MIMIC_ROOT/icu/icustays.csv.gz" | sed -n '1p'
echo

echo "== eICU patient header =="
gzip -dc "$EICU_ROOT/patient.csv.gz" | sed -n '1p'
echo

echo "== HiRID inventory (maxdepth 3) =="
find "$HIRID_ROOT" -maxdepth 3 -type f | sed "s#^$HIRID_ROOT/##" | sort | head -n 100