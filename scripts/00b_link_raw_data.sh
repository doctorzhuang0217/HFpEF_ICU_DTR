#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/path/to/data}"
RAW_ROOT="$PROJECT_ROOT/data/raw"

pick_one() {
  local label="$1"; shift
  local -a pats=("$@")
  local -a hits=()
  local pat

  for pat in "${pats[@]}"; do
    while IFS= read -r line; do
      [ -n "$line" ] && hits+=("$line")
    done < <(find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d -iname "$pat" | sort)
  done

  if [ "${#hits[@]}" -gt 0 ]; then
    mapfile -t hits < <(printf "%s\n" "${hits[@]}" | awk '!seen[$0]++')
  fi

  if [ "${#hits[@]}" -ne 1 ]; then
    echo "ERROR: could not uniquely resolve $label under $DATA_ROOT" >&2
    echo "Candidates:" >&2
    if [ "${#hits[@]}" -eq 0 ]; then
      echo "  <none>" >&2
    else
      printf '  %s\n' "${hits[@]}" >&2
    fi
    echo "Please rerun with manual override, e.g.:" >&2
    echo "  MIMIC_SRC=/path/to/mimic EICU_SRC=/path/to/eicu HIRID_SRC=/path/to/hirid bash scripts/00b_link_raw_data.sh" >&2
    return 1
  fi

  printf "%s\n" "${hits[0]}"
}

mkdir -p "$RAW_ROOT"

MIMIC_SRC="${MIMIC_SRC:-$(pick_one MIMIC 'mimic*')}"
EICU_SRC="${EICU_SRC:-$(pick_one eICU 'eicu*')}"
HIRID_SRC="${HIRID_SRC:-$(pick_one HiRID 'hirid*')}"

ln -sfn "$MIMIC_SRC" "$RAW_ROOT/mimiciv"
ln -sfn "$EICU_SRC"  "$RAW_ROOT/eicu"
ln -sfn "$HIRID_SRC" "$RAW_ROOT/hirid"

echo "== Symlinks created =="
ls -l "$RAW_ROOT"
echo

echo "== Resolved paths =="
readlink -f "$RAW_ROOT/mimiciv"
readlink -f "$RAW_ROOT/eicu"
readlink -f "$RAW_ROOT/hirid"
