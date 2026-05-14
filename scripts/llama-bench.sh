#!/bin/bash
# Usage:
# ./run_llama_bench.sh -m model.gguf -p 512 -n 128 -t 48 -r 3

set -e

# Load module
module load llama.cpp

# Optimized environment
export FLEXIBLAS=OPENBLAS
export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Parse command-line options
while getopts "m:p:n:t:r:h" opt; do
  case $opt in
    m) MODEL="$OPTARG" ;;
    p) PROMPT="$OPTARG" ;;
    n) NGEN="$OPTARG" ;;
    t) THREADS="$OPTARG" ;;
    r) REPEATS="$OPTARG" ;;
    h)
      echo "Usage: $0 -m model.gguf -p prompt_tokens -n gen_tokens -t threads -r repeats"
      exit 0
      ;;
    *)
      echo "Invalid option. Use -h for help."
      exit 1
      ;;
  esac
done

# Build command as an array
cmd=(llama-bench)

[[ -n "$MODEL"   ]] && cmd+=(-m "$MODEL")
[[ -n "$PROMPT"  ]] && cmd+=(-p "$PROMPT")
[[ -n "$NGEN"    ]] && cmd+=(-n "$NGEN")
[[ -n "$THREADS" ]] && cmd+=(-t "$THREADS")
[[ -n "$REPEATS" ]] && cmd+=(-r "$REPEATS")

# Show and run
echo "Running: ${cmd[*]}"

if command -v numactl >/dev/null 2>&1; then
    exec numactl --localalloc "${cmd[@]}"
else
    exec "${cmd[@]}"
fi