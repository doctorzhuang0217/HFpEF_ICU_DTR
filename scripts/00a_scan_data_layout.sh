#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/path/to/data}"

echo "== DATA_ROOT =="
echo "$DATA_ROOT"
echo

echo "== Top-level directories (size) =="
du -sh "$DATA_ROOT"/* 2>/dev/null | sort -h || true
echo

echo "== Top-level directories only =="
find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d | sort
echo

for d in "$DATA_ROOT"/*; do
  [ -d "$d" ] || continue
  name="$(basename "$d")"
  echo "===== $name ====="
  echo "-- level<=2 file count --"
  find "$d" -maxdepth 2 -type f | wc -l
  echo "-- first 40 files within level<=2 --"
  find "$d" -maxdepth 2 -type f | sed "s#^$d/##" | sort | head -n 40
  echo
done
