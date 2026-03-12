#!/usr/bin/env bash
set -ue

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT="NPU_Profiller.py"
OUTDIR="./results"
JSONL="./en-ko_KR_seg1-100.jsonl"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"

MODELS=(
  "rbln-Llama-3-1-8B-Instruct"
)

MONITOR_FLAGS=( --monitor --monitor-devices "0-7" --plot-min-watt "5" )

for MODEL in "${MODELS[@]}"; do
  echo "================ BENCH: ${MODEL} ================"
  "${PYTHON_BIN}" "${SCRIPT}" \
    --model "./${MODEL}" \
    --jsonl-file "${JSONL}" \
    --outdir "${OUTDIR}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    "${MONITOR_FLAGS[@]}" \
    || echo "[WARN] ${MODEL} failed. Continue..."
  echo
done

echo "All done."
