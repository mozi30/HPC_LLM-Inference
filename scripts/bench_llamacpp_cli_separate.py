#!/usr/bin/env python3
"""
Benchmark llama.cpp via CLI using a separate prompts.json file.

This script:
- reads a fixed prompt workload from prompts.json
- runs llama-cli directly, not HTTP
- supports warmup runs that are discarded from aggregate reporting
- runs three measured trials per prompt by default
- records TTFT, TPOT, throughput, peak memory, generated output
- logs raw per-output-event timestamps to JSON
- logs per-run summaries to CSV
- optionally samples CPU and memory with dstat
- optionally records Intel RAPL energy

Important measurement note:
llama-cli stdout does not expose exact internal token boundaries. This script
measures user-visible output events by timestamping stdout bytes as they arrive.
Therefore:
- TTFT = process launch to first visible output byte
- TPOT = mean interval between consecutive visible output byte events
- throughput = visible output byte events per second after first output

For exact token-level timing, llama.cpp itself must be instrumented or a token
streaming API should be used.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


MANDATORY_PROMPT_IDS = {
    "mandatory_short_capital_france",
    "mandatory_medium_ml_vs_dl",
    "mandatory_long_transformer_cpu_quantization",
}


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def now() -> float:
    return time.perf_counter()



def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)



def append_csv(path: Path, row: Dict[str, Any], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)



def sanitize_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_")[:120]

def get_rss_kb(pid: int) -> Optional[int]:
    try:
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])  # kB
    except FileNotFoundError:
        return None
    return None


# -----------------------------------------------------------------------------
# Prompt loading and validation
# -----------------------------------------------------------------------------


def load_prompts(prompt_file: Path) -> Dict[str, Any]:
    with prompt_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    validate_prompt_file(data)
    return data



def validate_prompt_file(data: Dict[str, Any]) -> None:
    prompts = data.get("prompts")
    if not isinstance(prompts, list):
        raise ValueError("prompts.json must contain a list field named 'prompts'.")

    counts = {"short": 0, "medium": 0, "long": 0}
    found_mandatory_ids = set()
    seen_ids = set()

    for p in prompts:
        if not isinstance(p, dict):
            raise ValueError("Each prompt entry must be a JSON object.")

        for key in ["id", "category", "prompt"]:
            if key not in p:
                raise ValueError(f"Prompt entry missing required key: {key}")

        prompt_id = p["id"]
        category = p["category"]

        if prompt_id in seen_ids:
            raise ValueError(f"Duplicate prompt id: {prompt_id}")
        seen_ids.add(prompt_id)

        if category not in counts:
            raise ValueError(
                f"Invalid category for prompt {prompt_id}: {category}. "
                "Expected one of short, medium, long."
            )

        counts[category] += 1

        if prompt_id in MANDATORY_PROMPT_IDS:
            found_mandatory_ids.add(prompt_id)

    if len(prompts) >= 30:
        for category, count in counts.items():
            if count < 10:
                raise ValueError(
                    f"Category '{category}' contains {count} prompts; expected at least 10."
                )

        missing = MANDATORY_PROMPT_IDS - found_mandatory_ids
        if missing:
            raise ValueError(
                f"Missing mandatory prompt ids: {sorted(missing)}"
            )


# -----------------------------------------------------------------------------
# Resource monitoring with dstat
# -----------------------------------------------------------------------------


def start_dstat(dstat_bin: str, path: Path) -> Optional[subprocess.Popen]:
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("w", encoding="utf-8")

    proc = subprocess.Popen(
        ["vmstat", "1"],
        stdout=f,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
        text=True,
    )
    proc._output_file = f
    return proc



def stop_process_group(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=5)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    finally:
        if hasattr(proc, "_output_file"):
            proc._output_file.close()



def parse_number(value: str) -> Optional[float]:
    """Parse dstat-style values. Returns bytes for B/K/M/G-style suffixes."""
    value = value.strip()
    if not value:
        return None

    multipliers = {
        "B": 1,
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
        "T": 1024**4,
    }

    match = re.match(r"^([-+]?\d+(?:\.\d+)?)([BKMGTP]?)$", value, re.I)
    if not match:
        try:
            return float(value)
        except ValueError:
            return None

    number = float(match.group(1))
    suffix = match.group(2).upper()
    return number * multipliers.get(suffix, 1)



def parse_dstat_peak_memory_mb(path: Path) -> Optional[float]:
    """
    Best-effort parser for dstat --cpu --mem --output CSV.

    dstat CSV output varies between versions. This function searches for the
    memory 'used' column and returns its peak value in MB.
    """
    if not path.exists():
        return None

    lines = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

    header = None
    data_start = None

    for i, line in enumerate(lines):
        cols = [c.strip().lower() for c in line.split(",")]
        if "used" in cols and ("free" in cols or "buff" in cols or "cach" in cols):
            header = cols
            data_start = i + 1

    if header is None or data_start is None:
        return None

    used_indices = [i for i, col in enumerate(header) if col == "used"]
    if not used_indices:
        return None

    # Usually the first 'used' column after CPU fields is memory used.
    used_idx = used_indices[0]
    values_mb = []

    for line in lines[data_start:]:
        cols = [c.strip() for c in line.split(",")]
        if len(cols) <= used_idx:
            continue

        parsed = parse_number(cols[used_idx])
        if parsed is not None:
            values_mb.append(parsed / (1024**2))

    return max(values_mb) if values_mb else None


# -----------------------------------------------------------------------------
# Optional Intel RAPL energy measurement
# -----------------------------------------------------------------------------


def find_rapl_energy_file() -> Optional[Path]:
    candidates = [
        Path("/sys/class/powercap/intel-rapl:0/energy_uj"),
        Path("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"),
    ]

    for path in candidates:
        if path.exists():
            return path

    return None



def read_rapl_energy_j(path: Path) -> float:
    return int(path.read_text().strip()) / 1_000_000.0


# -----------------------------------------------------------------------------
# llama.cpp CLI execution
# -----------------------------------------------------------------------------


def build_llama_command(
    binary: str,
    model: str,
    prompt: str,
    n_predict: int,
    seed: int,
    threads: Optional[int],
    ctx_size: Optional[int],
    extra_args: List[str],
) -> List[str]:
    cmd = [
        binary,
        "-m",
        model,
        "-p",
        prompt,
        "-n",
        str(n_predict),
        "--no-display-prompt",
	"--simple-io",
	"--single-turn",
        "--temp",
        "0",
        "--seed",
        str(seed),
    ]

    if threads is not None:
        cmd += ["-t", str(threads)]

    if ctx_size is not None:
        cmd += ["-c", str(ctx_size)]

    return cmd + extra_args



def run_llama_cli(cmd: List[str]) -> Dict[str, Any]:
    start_time = now()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        text=False,
    )

    events = []
    generated_bytes = bytearray()
    first_output_time = None
    max_rss_kb = 0

    assert proc.stdout is not None

    while True:
        rss_kb = get_rss_kb(proc.pid)
        if rss_kb is not None:
            max_rss_kb = max(max_rss_kb, rss_kb)

        chunk = proc.stdout.read(1)
        if not chunk:
            break

        t = now()
        generated_bytes.extend(chunk)

        events.append(
            {
                "index": len(events),
                "timestamp_s": t - start_time,
                "byte": chunk.decode("utf-8", errors="ignore"),
            }
        )

        if first_output_time is None:
            first_output_time = t

    proc.wait()
    end_time = now()

    stderr = ""
    if proc.stderr is not None:
        stderr = proc.stderr.read().decode("utf-8", errors="ignore")

    output_text = generated_bytes.decode("utf-8", errors="replace")

    if proc.returncode != 0:
        raise RuntimeError(
            "llama-cli failed\n"
            f"Exit code: {proc.returncode}\n"
            f"Command: {' '.join(cmd)}\n\n"
            f"STDERR:\n{stderr}"
        )

    if first_output_time is None:
        return {
            "ttft_s": None,
            "tpot_s": None,
            "throughput_events_per_s": 0.0,
            "memory_peak_mb": max_rss_kb / 1024 if max_rss_kb else None,
            "output_events_count": 0,
            "total_time_s": end_time - start_time,
            "output_text": output_text,
            "stderr": stderr,
            "events": events,
        }

    ttft_s = first_output_time - start_time

    deltas = [
        events[i]["timestamp_s"] - events[i - 1]["timestamp_s"]
        for i in range(1, len(events))
    ]

    tpot_s = statistics.mean(deltas) if deltas else 0.0
    generation_time_s = max(end_time - first_output_time, 1e-12)
    throughput_events_per_s = len(events) / generation_time_s

    return {
        "ttft_s": ttft_s,
        "tpot_s": tpot_s,
        "throughput_events_per_s": throughput_events_per_s,
        "memory_peak_mb": max_rss_kb / 1024 if max_rss_kb else None,
        "output_events_count": len(events),
        "total_time_s": end_time - start_time,
        "output_text": output_text,
        "stderr": stderr,
        "events": events,
    }

# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------


def mean_std(values: List[Optional[float]]) -> Dict[str, Optional[float]]:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"mean": None, "std": None}
    if len(clean) == 1:
        return {"mean": clean[0], "std": 0.0}
    return {"mean": statistics.mean(clean), "std": statistics.stdev(clean)}



def write_aggregate(rows: List[Dict[str, Any]], path: Path) -> None:
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for row in rows:
        grouped.setdefault(row["prompt_id"], []).append(row)

    aggregate_rows = []

    for prompt_id, group in sorted(grouped.items()):
        aggregate_rows.append(
            {
                "prompt_id": prompt_id,
                "category": group[0]["category"],
                "mandatory": group[0]["mandatory"],
                "trials": len(group),
                "ttft_s": mean_std([r["ttft_s"] for r in group]),
                "tpot_s": mean_std([r["tpot_s"] for r in group]),
                "throughput_events_per_s": mean_std(
                    [r["throughput_events_per_s"] for r in group]
                ),
                "memory_peak_mb": mean_std([r["memory_peak_mb"] for r in group]),
                "energy_j": mean_std([r["energy_j"] for r in group]),
            }
        )

    write_json(path, {"results": aggregate_rows})


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--binary", required=True, help="Path to llama-cli")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--prompt-file", type=Path, default=Path("prompts.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))

    parser.add_argument("--n-predict", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3, help="Measured trials per prompt")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Discarded warmup runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--ctx-size", type=int, default=None)

    parser.add_argument(
        "--only-mandatory",
        action="store_true",
        help="Run only the three mandatory standardized prompts.",
    )

    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Collect CPU and memory samples with dstat.",
    )
    parser.add_argument("--dstat-bin", default="dstat")

    parser.add_argument(
        "--rapl",
        action="store_true",
        help="Collect Intel RAPL package energy if available.",
    )

    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments passed directly to llama-cli.",
    )

    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    prompt_data = load_prompts(args.prompt_file)
    prompts = prompt_data["prompts"]

    if args.only_mandatory:
        prompts = [p for p in prompts if p.get("mandatory", False)]

    rapl_path = None
    if args.rapl:
        rapl_path = find_rapl_energy_file()
        if rapl_path is None:
            print("Warning: RAPL energy file not found; disabling --rapl", file=sys.stderr)
            args.rapl = False

    summary_csv = args.out_dir / "summary.csv"
    aggregate_json = args.out_dir / "aggregate.json"
    config_json = args.out_dir / "benchmark_config.json"

    config = {
        "binary": args.binary,
        "model": args.model,
        "prompt_file": str(args.prompt_file),
        "out_dir": str(args.out_dir),
        "n_predict": args.n_predict,
        "runs": args.runs,
        "warmup_runs": args.warmup_runs,
        "seed": args.seed,
        "threads": args.threads,
        "ctx_size": args.ctx_size,
        "monitor": args.monitor,
        "rapl": args.rapl,
        "extra_args": args.extra_args,
        "fixed_generation_parameters": {
            "temperature": 0.0,
            "seed": args.seed,
            "n_predict": args.n_predict,
        },
        "measurement_definitions": {
            "TTFT": "Time from llama-cli process launch to first visible stdout byte.",
            "TPOT": "Mean interval between consecutive visible stdout byte events.",
            "throughput": "Visible stdout byte events per second after first output event.",
            "memory_peak": "Best-effort peak used memory parsed from dstat CSV.",
        },
    }
    write_json(config_json, config)

    fieldnames = [
        "prompt_id",
        "category",
        "mandatory",
        "trial",
        "warmup",
        "ttft_s",
        "tpot_s",
        "throughput_events_per_s",
        "output_events_count",
        "total_time_s",
        "memory_peak_mb",
        "energy_j",
        "generated_output",
        "raw_json",
        "resource_csv",
    ]

    measured_rows = []

    for prompt in prompts:
        prompt_id = prompt["id"]
        category = prompt["category"]
        mandatory = bool(prompt.get("mandatory", False))
        prompt_text = prompt["prompt"]

        print(f"\nPrompt: {prompt_id} ({category})", file=sys.stderr)

        total_runs = args.warmup_runs + args.runs

        for run_index in range(total_runs):
            warmup = run_index < args.warmup_runs
            measured_trial = run_index - args.warmup_runs
            trial_label = "warmup" if warmup else f"trial{measured_trial}"

            print(f"  Running {trial_label}...", file=sys.stderr)

            base_name = f"{sanitize_filename(prompt_id)}_{trial_label}"
            raw_json = args.out_dir / f"{base_name}.json"
            resource_csv = args.out_dir / f"{base_name}_resources.csv"

            cmd = build_llama_command(
                binary=args.binary,
                model=args.model,
                prompt=prompt_text,
                n_predict=args.n_predict,
                seed=args.seed,
                threads=args.threads,
                ctx_size=args.ctx_size,
                extra_args=args.extra_args,
            )

            monitor_proc = None
            energy_start = None
            energy_j = None
            memory_peak_mb = None

            try:
                if args.monitor:
                    monitor_proc = start_dstat(args.dstat_bin, resource_csv)
                    time.sleep(0.2)

                if args.rapl and rapl_path is not None:
                    energy_start = read_rapl_energy_j(rapl_path)

                result = run_llama_cli(cmd)

                if args.rapl and rapl_path is not None and energy_start is not None:
                    energy_end = read_rapl_energy_j(rapl_path)
                    energy_j = energy_end - energy_start
                    if energy_j < 0:
                        # RAPL counter wrapped; leave unavailable.
                        energy_j = None

            finally:
                stop_process_group(monitor_proc)

            if args.monitor:
                memory_peak_mb = result.get("memory_peak_mb")

            raw_record = {
                "prompt_id": prompt_id,
                "category": category,
                "mandatory": mandatory,
                "trial": None if warmup else measured_trial,
                "warmup": warmup,
                "command": cmd,
                "prompt": prompt_text,
                "metrics": {
                    "ttft_s": result["ttft_s"],
                    "tpot_s": result["tpot_s"],
                    "throughput_events_per_s": result["throughput_events_per_s"],
                    "output_events_count": result["output_events_count"],
                    "total_time_s": result["total_time_s"],
                    "memory_peak_mb": memory_peak_mb,
                    "energy_j": energy_j,
                },
                "generated_output": result["output_text"],
                "stderr": result["stderr"],
                "events": result["events"],
                "resource_csv": str(resource_csv) if args.monitor else "",
            }
            write_json(raw_json, raw_record)

            row = {
                "prompt_id": prompt_id,
                "category": category,
                "mandatory": mandatory,
                "trial": "" if warmup else measured_trial,
                "warmup": warmup,
                "ttft_s": result["ttft_s"],
                "tpot_s": result["tpot_s"],
                "throughput_events_per_s": result["throughput_events_per_s"],
                "output_events_count": result["output_events_count"],
                "total_time_s": result["total_time_s"],
                "memory_peak_mb": memory_peak_mb,
                "energy_j": energy_j,
                "generated_output": result["output_text"] if mandatory and not warmup else "",
                "raw_json": str(raw_json),
                "resource_csv": str(resource_csv) if args.monitor else "",
            }

            append_csv(summary_csv, row, fieldnames)

            if not warmup:
                measured_rows.append(row)

            print(
                "    "
                f"TTFT={result['ttft_s']} | "
                f"TPOT={result['tpot_s']} | "
                f"throughput={result['throughput_events_per_s']:.2f} events/s | "
                f"peak_mem={memory_peak_mb} MB | "
                f"energy={energy_j} J",
                file=sys.stderr,
            )

    write_aggregate(measured_rows, aggregate_json)

    print("\nDone.")
    print(f"Summary CSV: {summary_csv}")
    print(f"Aggregate JSON: {aggregate_json}")
    print(f"Benchmark config: {config_json}")


if __name__ == "__main__":
    main()
