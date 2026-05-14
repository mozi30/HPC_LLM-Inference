#!/bin/bash
# Usage:
# ./llama_bench_sweep.sh -m model.gguf -t 48 -r 3
#
# Runs:
#   1) TTFT tests: p = 32,128,512,2048 and n = 1
#   2) Generation tests: p = 128 and n = 32,128,512,2048
#
# All output is written to a timestamped log file.

set -euo pipefail

SCRIPT="/projects/F202500010HPCVLABUMINHO/e13105/LLM_Inference/scripts/llama-bench.sh"

THREADS=48
REPEATS=3
OUTDIR="bench_logs"

while getopts "m:t:r:s:o:h" opt; do
  case $opt in
    m) MODEL="$OPTARG" ;;
    t) THREADS="$OPTARG" ;;
    r) REPEATS="$OPTARG" ;;
    s) SCRIPT="$OPTARG" ;;
    o) OUTDIR="$OPTARG" ;;
    h)
      echo "Usage: $0 -m model.gguf [-t threads] [-r repeats] [-s run_script] [-o output_dir]"
      exit 0
      ;;
    *)
      echo "Invalid option. Use -h for help."
      exit 1
      ;;
  esac
done

if [[ -z "${MODEL:-}" ]]; then
  echo "Error: model file is required."
  echo "Usage: $0 -m model.gguf [-t threads] [-r repeats]"
  exit 1
fi

if [[ ! -x "$SCRIPT" ]]; then
  echo "Error: '$SCRIPT' not found or not executable."
  echo "Run: chmod +x $SCRIPT"
  exit 1
fi

mkdir -p "$OUTDIR"

MODEL_BASENAME=$(basename "$MODEL")
MODEL_NAME="${MODEL_BASENAME%.gguf}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="${OUTDIR}/${MODEL_NAME}_t${THREADS}_r${REPEATS}_${TIMESTAMP}.log"

PROMPTS=(32 128 512 2048)
NGENS=(32 128 512 2048)
NTHREADS=(8 16 48)

# Log everything to both terminal and file
exec > >(tee -a "$LOGFILE") 2>&1

echo "========================================================"
echo "llama-bench sweep"
echo "Model:      $MODEL"
echo "Threads:    $THREADS"
echo "Repeats:    $REPEATS"
echo "Script:     $SCRIPT"
echo "Log file:   $LOGFILE"
echo "Started:    $(date)"
echo "========================================================"

echo
echo "=== TTFT benchmark: varying prompt, n=1 ==="
for p in "${PROMPTS[@]}"; do
  echo
  echo "--------------------------------------------------------"
  echo "TTFT: prompt=$p, n=1"
  echo "--------------------------------------------------------"
  "$SCRIPT" -m "$MODEL" -p "$p" -n 1 -t "$THREADS" -r "$REPEATS"
done

echo
echo "=== Generation benchmark: prompt=128, varying n ==="
for n in "${NGENS[@]}"; do
  echo
  echo "--------------------------------------------------------"
  echo "Generation: prompt=128, n=$n"
  echo "--------------------------------------------------------"
  "$SCRIPT" -m "$MODEL" -p 128 -n "$n" -t "$THREADS" -r "$REPEATS"
done

echo
echo "=== Generation benchmark: prompt=512, n=128 ==="
for t in "${NTHREADS[@]}"; do
  echo
  echo "--------------------------------------------------------"
  echo "Generation: prompt=128, n=128 threads=$t"
  echo "--------------------------------------------------------"
  "$SCRIPT" -m "$MODEL" -p 512 -n 128 -t "$t" -r 1
done


echo
echo "========================================================"
echo "Completed: $(date)"
echo "Results saved to: $LOGFILE"
echo "========================================================"
