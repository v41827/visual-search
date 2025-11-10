#!/usr/bin/env bash
set -euo pipefail

# Refresh the master evaluation report by re‑scanning all
# results under results/eval and writing a single CSV.
#
# Usage:
#   bash refresh_report.sh                # writes results/eval/REPORT_all_summary.csv
#   bash refresh_report.sh --out path.csv # custom output path
#   bash refresh_report.sh --run-id ID    # only rows matching run_id

OUT="results/eval/REPORT_all_summary.csv"
RUNID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      OUT="$2"; shift 2;;
    --run-id)
      RUNID="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -n "$RUNID" ]]; then
  python tools/aggregate_evals.py --eval-root results/eval --out "$OUT" --run-id "$RUNID"
else
  python tools/aggregate_evals.py --eval-root results/eval --out "$OUT"
fi

echo "Wrote $OUT"

