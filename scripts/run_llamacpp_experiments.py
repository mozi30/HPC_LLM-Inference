#!/usr/bin/env python3
"""
Experiment runner for llama.cpp CLI benchmarking.

This script uses the benchmark script `bench_llamacpp_cli_separate.py` and a
model path to run controlled experiments for:

1. Threading / parallelism:
   - Varies inference threads, e.g. 1, 4, 8, 16, 32, 48
   - Identifies the point of diminishing returns

2. Concurrent requests:
   - Runs multiple simultaneous benchmark processes, e.g. 1, 2, 4, 8, 16
   - Observes total throughput and per-request latency

3. Context length:
   - Uses prompt categories / prompt subsets as a proxy for prompt length
   - Tests target context sizes, e.g. 128, 512, 1024, 2048
   - Observes TTFT and memory impact

4. Decode length:
   - Varies maximum generated tokens, e.g. 64, 128, 256, 512
   - Observes TPOT stability

Assumptions:
- `bench_llamacpp_cli_separate.py` exists in the same directory or is passed via
  --bench-script.
- `prompts.json` exists and contains short, medium, and long prompt categories.
- llama.cpp `llama-cli` exists at --binary.
- The GGUF model exists at --model.

Example:
python run_llamacpp_experiments.py \
  --bench-script ./bench_llamacpp_cli_separate.py \
  --binary ./build/bin/llama-cli \
  --model ./models/model.gguf \
  --prompt-file ./prompts.json \
  --out-dir results/experiments \
  --monitor
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# -----------------------------------------------------------------------------
# Basic I/O helpers
# -----------------------------------------------------------------------------


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)



def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)



def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# -----------------------------------------------------------------------------
# Prompt subset generation
# -----------------------------------------------------------------------------


def filter_prompt_file(
    source_prompt_file: Path,
    target_prompt_file: Path,
    categories: Optional[List[str]] = None,
    mandatory_only: bool = False,
    max_prompts: Optional[int] = None,
) -> None:
    """Create a smaller prompt file for one experiment condition."""
    data = read_json(source_prompt_file)
    prompts = data["prompts"]

    if categories is not None:
        prompts = [p for p in prompts if p.get("category") in categories]

    if mandatory_only:
        prompts = [p for p in prompts if p.get("mandatory")]

    if max_prompts is not None:
        prompts = prompts[:max_prompts]

    if not prompts:
        raise ValueError(
            f"No prompts selected from {source_prompt_file} with categories={categories} "
            f"mandatory_only={mandatory_only}."
        )

    out = {
        "metadata": data.get("metadata", {}),
        "prompts": prompts,
    }
    write_json(target_prompt_file, out)



def make_context_prompt_file(
    source_prompt_file: Path,
    target_prompt_file: Path,
    target_tokens: int,
) -> None:
    """
    Build a synthetic one-prompt workload for context-length experiments.

    This does not tokenize with the model tokenizer. It approximates token count
    using repeated text. For exact token counts, pre-tokenize with llama.cpp or a
    tokenizer matched to the model.
    """
    base_sentence = (
        "Transformer inference on CPU requires repeated access to model weights, "
        "attention state, and runtime buffers. Prompt length affects the prefill "
        "phase, while generated length affects the decoding phase. "
    )

    # Rough English approximation: one token ~= 0.75 words.
    target_words = max(16, int(target_tokens * 0.75))
    words = []
    base_words = base_sentence.split()
    while len(words) < target_words:
        words.extend(base_words)

    context = " ".join(words[:target_words])
    prompt = (
        f"Context:\n{context}\n\n"
        "Question: Based on the context, explain how prompt length affects TTFT, "
        "memory usage, and inference performance."
    )

    out = {
        "metadata": {
            "description": f"Synthetic context-length workload targeting about {target_tokens} tokens.",
            "target_context_tokens_approx": target_tokens,
        },
        "prompts": [
            {
                "id": f"context_{target_tokens}",
                "category": "long" if target_tokens >= 1024 else "medium" if target_tokens >= 512 else "short",
                "mandatory": False,
                "prompt": prompt,
            }
        ],
    }
    write_json(target_prompt_file, out)


# -----------------------------------------------------------------------------
# Benchmark invocation
# -----------------------------------------------------------------------------


def build_benchmark_command(
    python_bin: str,
    bench_script: Path,
    binary: Path,
    model: Path,
    prompt_file: Path,
    out_dir: Path,
    n_predict: int,
    runs: int,
    warmup_runs: int,
    seed: int,
    threads: Optional[int],
    ctx_size: Optional[int],
    monitor: bool,
    rapl: bool,
    only_mandatory: bool,
    extra_args: List[str],
) -> List[str]:
    cmd = [
        python_bin,
        str(bench_script),
        "--binary",
        str(binary),
        "--model",
        str(model),
        "--prompt-file",
        str(prompt_file),
        "--out-dir",
        str(out_dir),
        "--n-predict",
        str(n_predict),
        "--runs",
        str(runs),
        "--warmup-runs",
        str(warmup_runs),
        "--seed",
        str(seed),
    ]

    if threads is not None:
        cmd += ["--threads", str(threads)]

    if ctx_size is not None:
        cmd += ["--ctx-size", str(ctx_size)]

    if monitor:
        cmd.append("--monitor")

    if rapl:
        cmd.append("--rapl")

    if only_mandatory:
        cmd.append("--only-mandatory")

    if extra_args:
        cmd.append("--extra-args")
        cmd.extend(extra_args)

    return cmd



def run_command(cmd: List[str], log_file: Path) -> Dict[str, Any]:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    end = time.perf_counter()

    return {
        "returncode": proc.returncode,
        "wall_time_s": end - start,
        "log_file": str(log_file),
        "command": cmd,
    }



def load_aggregate(out_dir: Path) -> List[Dict[str, Any]]:
    aggregate_path = out_dir / "aggregate.json"
    if not aggregate_path.exists():
        return []
    data = read_json(aggregate_path)
    return data.get("results", [])



def metric_mean(item: Dict[str, Any], metric: str) -> Optional[float]:
    value = item.get(metric)
    if isinstance(value, dict):
        return value.get("mean")
    return None



def summarize_aggregate(out_dir: Path) -> Dict[str, Optional[float]]:
    rows = load_aggregate(out_dir)
    if not rows:
        return {
            "ttft_mean_s": None,
            "tpot_mean_s": None,
            "throughput_mean_events_per_s": None,
            "memory_peak_mean_mb": None,
            "energy_mean_j": None,
        }

    def avg_metric(metric: str) -> Optional[float]:
        values = [metric_mean(r, metric) for r in rows]
        values = [v for v in values if v is not None]
        if not values:
            return None
        return sum(values) / len(values)

    return {
        "ttft_mean_s": avg_metric("ttft_s"),
        "tpot_mean_s": avg_metric("tpot_s"),
        "throughput_mean_events_per_s": avg_metric("throughput_events_per_s"),
        "memory_peak_mean_mb": avg_metric("memory_peak_mb"),
        "energy_mean_j": avg_metric("energy_j"),
    }


# -----------------------------------------------------------------------------
# Diminishing returns analysis
# -----------------------------------------------------------------------------


def identify_diminishing_returns(
    rows: List[Dict[str, Any]],
    x_key: str,
    y_key: str,
    threshold: float,
) -> Optional[Dict[str, Any]]:
    """
    Identify the first point where relative improvement falls below threshold.

    Example:
    threshold=0.10 means the first setting where improvement over previous
    setting is less than 10%.
    """
    sorted_rows = sorted(
        [r for r in rows if r.get(y_key) is not None],
        key=lambda r: float(r[x_key]),
    )

    if len(sorted_rows) < 2:
        return None

    previous = sorted_rows[0]

    for current in sorted_rows[1:]:
        prev_y = float(previous[y_key])
        cur_y = float(current[y_key])

        if prev_y <= 0:
            improvement = None
        else:
            improvement = (cur_y - prev_y) / prev_y

        current["relative_improvement_from_previous"] = improvement

        if improvement is not None and improvement < threshold:
            return {
                "diminishing_returns_at": current[x_key],
                "previous_setting": previous[x_key],
                "metric": y_key,
                "relative_improvement": improvement,
                "threshold": threshold,
            }

        previous = current

    return None


# -----------------------------------------------------------------------------
# Experiment groups
# -----------------------------------------------------------------------------


def run_thread_experiment(args: argparse.Namespace, workload_file: Path) -> List[Dict[str, Any]]:
    print("\n=== Threading / parallelism experiment ===")

    rows = []

    for threads in args.thread_values:
        out_dir = args.out_dir / "threads" / f"threads_{threads}"
        log_file = out_dir / "run.log"

        cmd = build_benchmark_command(
            python_bin=args.python_bin,
            bench_script=args.bench_script,
            binary=args.binary,
            model=args.model,
            prompt_file=workload_file,
            out_dir=out_dir,
            n_predict=args.n_predict,
            runs=args.runs,
            warmup_runs=args.warmup_runs,
            seed=args.seed,
            threads=threads,
            ctx_size=args.ctx_size,
            monitor=args.monitor,
            rapl=args.rapl,
            only_mandatory=False,
            extra_args=args.extra_args,
        )

        result = run_command(cmd, log_file)
        summary = summarize_aggregate(out_dir)

        row = {
            "experiment": "threads",
            "threads": threads,
            "returncode": result["returncode"],
            "wall_time_s": result["wall_time_s"],
            **summary,
            "out_dir": str(out_dir),
            "log_file": str(log_file),
        }
        rows.append(row)
        print(row)

    analysis = identify_diminishing_returns(
        rows,
        x_key="threads",
        y_key="throughput_mean_events_per_s",
        threshold=args.diminishing_threshold,
    )

    write_csv(args.out_dir / "threads_summary.csv", rows)
    write_json(args.out_dir / "threads_diminishing_returns.json", analysis)

    return rows



def run_single_concurrent_worker(
    args: argparse.Namespace,
    workload_file: Path,
    concurrency: int,
    worker_id: int,
) -> Dict[str, Any]:
    out_dir = args.out_dir / "concurrency" / f"concurrency_{concurrency}" / f"worker_{worker_id}"
    log_file = out_dir / "run.log"

    cmd = build_benchmark_command(
        python_bin=args.python_bin,
        bench_script=args.bench_script,
        binary=args.binary,
        model=args.model,
        prompt_file=workload_file,
        out_dir=out_dir,
        n_predict=args.n_predict,
        runs=args.concurrent_runs,
        warmup_runs=args.warmup_runs,
        seed=args.seed + worker_id,
        threads=args.concurrent_threads,
        ctx_size=args.ctx_size,
        monitor=args.monitor,
        rapl=False,
        only_mandatory=False,
        extra_args=args.extra_args,
    )

    result = run_command(cmd, log_file)
    summary = summarize_aggregate(out_dir)

    return {
        "worker_id": worker_id,
        "returncode": result["returncode"],
        "wall_time_s": result["wall_time_s"],
        **summary,
        "out_dir": str(out_dir),
        "log_file": str(log_file),
    }



def run_concurrency_experiment(args: argparse.Namespace, workload_file: Path) -> List[Dict[str, Any]]:
    print("\n=== Concurrent requests experiment ===")

    rows = []

    for concurrency in args.concurrency_values:
        start = time.perf_counter()
        worker_results = []

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(run_single_concurrent_worker, args, workload_file, concurrency, i)
                for i in range(concurrency)
            ]
            for future in as_completed(futures):
                worker_results.append(future.result())

        end = time.perf_counter()
        wall_time_s = end - start

        throughput_values = [
            r["throughput_mean_events_per_s"]
            for r in worker_results
            if r.get("throughput_mean_events_per_s") is not None
        ]
        ttft_values = [r["ttft_mean_s"] for r in worker_results if r.get("ttft_mean_s") is not None]
        tpot_values = [r["tpot_mean_s"] for r in worker_results if r.get("tpot_mean_s") is not None]

        total_throughput = sum(throughput_values) if throughput_values else None
        mean_ttft = sum(ttft_values) / len(ttft_values) if ttft_values else None
        mean_tpot = sum(tpot_values) / len(tpot_values) if tpot_values else None

        row = {
            "experiment": "concurrency",
            "concurrency": concurrency,
            "threads_per_process": args.concurrent_threads,
            "workers_completed": len(worker_results),
            "failed_workers": sum(1 for r in worker_results if r["returncode"] != 0),
            "wall_time_s": wall_time_s,
            "total_throughput_events_per_s": total_throughput,
            "mean_per_request_ttft_s": mean_ttft,
            "mean_per_request_tpot_s": mean_tpot,
            "out_dir": str(args.out_dir / "concurrency" / f"concurrency_{concurrency}"),
        }
        rows.append(row)
        print(row)

        write_json(
            args.out_dir / "concurrency" / f"concurrency_{concurrency}" / "worker_results.json",
            worker_results,
        )

    write_csv(args.out_dir / "concurrency_summary.csv", rows)
    return rows



def run_context_experiment(args: argparse.Namespace) -> List[Dict[str, Any]]:
    print("\n=== Context length experiment ===")

    rows = []
    context_dir = args.out_dir / "context_prompt_files"
    context_dir.mkdir(parents=True, exist_ok=True)

    for context_tokens in args.context_values:
        prompt_file = context_dir / f"context_{context_tokens}.json"
        make_context_prompt_file(args.prompt_file, prompt_file, context_tokens)

        out_dir = args.out_dir / "context" / f"context_{context_tokens}"
        log_file = out_dir / "run.log"

        # ctx_size must be at least prompt target plus room for generated tokens.
        ctx_size = max(context_tokens + args.n_predict + 64, args.ctx_size or 0)

        cmd = build_benchmark_command(
            python_bin=args.python_bin,
            bench_script=args.bench_script,
            binary=args.binary,
            model=args.model,
            prompt_file=prompt_file,
            out_dir=out_dir,
            n_predict=args.n_predict,
            runs=args.runs,
            warmup_runs=args.warmup_runs,
            seed=args.seed,
            threads=args.threads_for_sweeps,
            ctx_size=ctx_size,
            monitor=args.monitor,
            rapl=args.rapl,
            only_mandatory=False,
            extra_args=args.extra_args,
        )

        result = run_command(cmd, log_file)
        summary = summarize_aggregate(out_dir)

        row = {
            "experiment": "context_length",
            "target_context_tokens_approx": context_tokens,
            "ctx_size": ctx_size,
            "threads": args.threads_for_sweeps,
            "returncode": result["returncode"],
            "wall_time_s": result["wall_time_s"],
            **summary,
            "out_dir": str(out_dir),
            "log_file": str(log_file),
        }
        rows.append(row)
        print(row)

    write_csv(args.out_dir / "context_summary.csv", rows)
    return rows



def run_decode_experiment(args: argparse.Namespace, workload_file: Path) -> List[Dict[str, Any]]:
    print("\n=== Decode length experiment ===")

    rows = []

    for n_predict in args.decode_values:
        out_dir = args.out_dir / "decode" / f"decode_{n_predict}"
        log_file = out_dir / "run.log"

        cmd = build_benchmark_command(
            python_bin=args.python_bin,
            bench_script=args.bench_script,
            binary=args.binary,
            model=args.model,
            prompt_file=workload_file,
            out_dir=out_dir,
            n_predict=n_predict,
            runs=args.runs,
            warmup_runs=args.warmup_runs,
            seed=args.seed,
            threads=args.threads_for_sweeps,
            ctx_size=args.ctx_size,
            monitor=args.monitor,
            rapl=args.rapl,
            only_mandatory=False,
            extra_args=args.extra_args,
        )

        result = run_command(cmd, log_file)
        summary = summarize_aggregate(out_dir)

        row = {
            "experiment": "decode_length",
            "n_predict": n_predict,
            "threads": args.threads_for_sweeps,
            "returncode": result["returncode"],
            "wall_time_s": result["wall_time_s"],
            **summary,
            "out_dir": str(out_dir),
            "log_file": str(log_file),
        }
        rows.append(row)
        print(row)

    write_csv(args.out_dir / "decode_summary.csv", rows)
    return rows


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]



def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--bench-script", type=Path, default=Path("bench_llamacpp_cli_separate.py"))
    parser.add_argument("--binary", type=Path, required=True, help="Path to llama-cli")
    parser.add_argument("--model", type=Path, required=True, help="Path to GGUF model")
    parser.add_argument("--prompt-file", type=Path, default=Path("prompts.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/experiments"))
    parser.add_argument("--python-bin", default=sys.executable)

    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-predict", type=int, default=128)
    parser.add_argument("--ctx-size", type=int, default=None)

    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--rapl", action="store_true")

    parser.add_argument("--thread-values", type=parse_int_list, default="1,4,8,16,32,48")
    parser.add_argument("--concurrency-values", type=parse_int_list, default="1,2,4,8,16")
    parser.add_argument("--context-values", type=parse_int_list, default="128,512,1024,2048")
    parser.add_argument("--decode-values", type=parse_int_list, default="64,128,256,512")

    parser.add_argument(
        "--threads-for-sweeps",
        type=int,
        default=8,
        help="Thread count used for context-length and decode-length sweeps.",
    )
    parser.add_argument(
        "--concurrent-threads",
        type=int,
        default=4,
        help="Threads per llama-cli process during concurrency experiment.",
    )
    parser.add_argument(
        "--concurrent-runs",
        type=int,
        default=1,
        help="Measured runs per worker during concurrency experiment.",
    )
    parser.add_argument(
        "--diminishing-threshold",
        type=float,
        default=0.10,
        help="Relative throughput improvement threshold for diminishing returns.",
    )

    parser.add_argument(
        "--workload-max-prompts",
        type=int,
        default=3,
        help="Use a small fixed subset for sweep speed. Set 0 to use all prompts.",
    )
    parser.add_argument(
        "--skip-threads",
        action="store_true",
        help="Skip threading experiment.",
    )
    parser.add_argument(
        "--skip-concurrency",
        action="store_true",
        help="Skip concurrency experiment.",
    )
    parser.add_argument(
        "--skip-context",
        action="store_true",
        help="Skip context length experiment.",
    )
    parser.add_argument(
        "--skip-decode",
        action="store_true",
        help="Skip decode length experiment.",
    )

    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments passed to bench script, then llama-cli via --extra-args.",
    )

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.bench_script.exists():
        raise FileNotFoundError(f"Benchmark script not found: {args.bench_script}")
    if not args.prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {args.prompt_file}")
    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not args.binary.exists():
        raise FileNotFoundError(f"llama-cli binary not found: {args.binary}")

    workload_dir = args.out_dir / "workloads"
    workload_dir.mkdir(parents=True, exist_ok=True)

    # Use a smaller representative workload for sweeps by default:
    # mandatory prompts include one short, one medium, and one long prompt.
    workload_file = workload_dir / "sweep_workload.json"
    max_prompts = None if args.workload_max_prompts == 0 else args.workload_max_prompts
    filter_prompt_file(
        source_prompt_file=args.prompt_file,
        target_prompt_file=workload_file,
        mandatory_only=False,
        max_prompts=max_prompts,
    )

    experiment_config = {
        "bench_script": str(args.bench_script),
        "binary": str(args.binary),
        "model": str(args.model),
        "prompt_file": str(args.prompt_file),
        "sweep_workload": str(workload_file),
        "runs": args.runs,
        "warmup_runs": args.warmup_runs,
        "thread_values": args.thread_values,
        "concurrency_values": args.concurrency_values,
        "context_values": args.context_values,
        "decode_values": args.decode_values,
        "threads_for_sweeps": args.threads_for_sweeps,
        "concurrent_threads": args.concurrent_threads,
        "diminishing_threshold": args.diminishing_threshold,
    }
    write_json(args.out_dir / "experiment_config.json", experiment_config)

    all_results = {}

    if not args.skip_threads:
        all_results["threads"] = run_thread_experiment(args, workload_file)

    if not args.skip_concurrency:
        all_results["concurrency"] = run_concurrency_experiment(args, workload_file)

    if not args.skip_context:
        all_results["context"] = run_context_experiment(args)

    if not args.skip_decode:
        all_results["decode"] = run_decode_experiment(args, workload_file)

    write_json(args.out_dir / "all_experiment_summaries.json", all_results)

    print("\nAll experiments finished.")
    print(f"Experiment directory: {args.out_dir}")
    print(f"Combined summary: {args.out_dir / 'all_experiment_summaries.json'}")


if __name__ == "__main__":
    main()
