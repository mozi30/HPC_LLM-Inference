#!/bin/bash
# Usage:
# ./run_llama_bench.sh -m model.gguf -b /path/to/llama-cli -p prompts.json -o results/experiments

set -euo pipefail

MODEL="../models/gemma-2-2b-it-Q4_K_M.gguf"
BINARY="/eb/x86_64/software/llama.cpp/20260415-foss-2023a/bin/llama-cli"
PROMPTS="../data/prompts.json"
OUT_DIR="results/experiments-x86"
BENCH_SCRIPT="bench_llamacpp_cli_separate.py"
EXP_SCRIPT="run_llamacpp_experiments.py"

while getopts "m:b:p:o:" opt; do
  case "$opt" in
    m) MODEL="$OPTARG" ;;
    b) BINARY="$OPTARG" ;;
    p) PROMPTS="$OPTARG" ;;
    o) OUT_DIR="$OPTARG" ;;
    *) echo "Usage: $0 -m model.gguf [-b llama-cli] [-p prompts.json] [-o out_dir]"; exit 1 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "Error: model path required with -m"
  exit 1
fi

module load llama.cpp

export FLEXIBLAS=OPENBLAS
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=128
export OMP_PROC_BIND=close
export OMP_PLACES=cores

cmd=(
  python run_llamacpp_experiments.py
  --bench-script "$BENCH_SCRIPT"
  --binary "$BINARY"
  --model "$MODEL"
  --prompt-file "$PROMPTS"
  --out-dir "$OUT_DIR"
  --runs 3
  --warmup-runs 0
  --monitor
  --thread-values 4,16,64,128
  --concurrency-values 2,4,16
  --context-values 32,128,512,2048
  --decode-values 32,128,512,2048
  --workload-max-prompts 0
  --threads-for-sweeps 16
)

echo "Running: ${cmd[*]}"

if command -v numactl >/dev/null 2>&1; then
  exec numactl --localalloc "${cmd[@]}"
else
  exec "${cmd[@]}"
fi
