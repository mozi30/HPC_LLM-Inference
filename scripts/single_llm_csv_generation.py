from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT_DEFAULT = Path("data/experiments-x86")

PROMPT_LINE_RE = re.compile(r"^>\s", re.MULTILINE)
PROMPT_STATS_RE = re.compile(r"^\[ Prompt: .* \]$")
PROMPT_TPS_RE = re.compile(
    r"\[\s*Prompt:\s*([0-9]*\.?[0-9]+)\s*t/s\s*\|\s*Generation:\s*([0-9]*\.?[0-9]+)\s*t/s\s*\]"
)

NUMERIC_METRICS = ["ttft_s", "tpot_s", "prompt_tps", "generation_tps", "peak_rss_mb"]


def _parse_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text == "":
        return None
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_int_from_name(value: str) -> Optional[int]:
    match = re.search(r"(-?\d+)", value)
    return int(match.group(1)) if match else None


def _infer_configuration(summary_csv: Path, root_dir: Path) -> Dict[str, Any]:
    rel = summary_csv.relative_to(root_dir)
    parts = rel.parts[:-1]

    info: Dict[str, Any] = {
        "configuration_group": "unknown",
        "configuration_name": "unknown",
        "configuration_value": None,
        "configuration_label": "unknown",
        "worker_id": None,
    }

    if len(parts) >= 2 and parts[0] == "threads" and parts[1].startswith("threads_"):
        value = _parse_int_from_name(parts[1])
        info.update({
            "configuration_group": "threads",
            "configuration_name": "threads",
            "configuration_value": value,
            "configuration_label": f"threads={value}",
        })
    elif len(parts) >= 2 and parts[0] == "decode" and parts[1].startswith("decode_"):
        value = _parse_int_from_name(parts[1])
        info.update({
            "configuration_group": "decode_length",
            "configuration_name": "n_predict",
            "configuration_value": value,
            "configuration_label": f"n_predict={value}",
        })
    elif len(parts) >= 2 and parts[0] == "context" and parts[1].startswith("context_"):
        value = _parse_int_from_name(parts[1])
        info.update({
            "configuration_group": "context_length",
            "configuration_name": "target_context_tokens_approx",
            "configuration_value": value,
            "configuration_label": f"target_context_tokens_approx={value}",
        })
    elif len(parts) >= 2 and parts[0] == "concurrency" and parts[1].startswith("concurrency_"):
        value = _parse_int_from_name(parts[1])
        worker_id = None
        if len(parts) >= 3 and parts[2].startswith("worker_"):
            worker_id = _parse_int_from_name(parts[2])
        info.update({
            "configuration_group": "concurrency",
            "configuration_name": "concurrency",
            "configuration_value": value,
            "configuration_label": f"concurrency={value}",
            "worker_id": worker_id,
        })
    elif len(parts) >= 1:
        group = parts[0]
        value_name = parts[1] if len(parts) > 1 else group
        value = _parse_int_from_name(value_name)
        info.update({
            "configuration_group": group,
            "configuration_name": value_name,
            "configuration_value": value,
            "configuration_label": "/".join(parts),
        })

    return info


def _extract_response_span(text: str) -> Tuple[str, int, int]:
    if not text:
        return "", 0, 0
    work = text.replace("\r\n", "\n").replace("\r", "\n")
    match = PROMPT_LINE_RE.search(work)
    if match:
        line_end = work.find("\n", match.end())
        start = line_end + 1 if line_end != -1 else len(work)
    else:
        marker = "available commands:"
        idx = work.find(marker)
        if idx != -1:
            after = work.find("\n\n", idx)
            start = after + 2 if after != -1 else idx
        else:
            start = 0
    while start < len(work) and work[start] == "\n":
        start += 1

    trimmed = work[start:]
    lines = trimmed.splitlines(keepends=True)
    while lines and lines[-1].strip() == "":
        lines.pop()

    def _is_trailing_line(line: str) -> bool:
        stripped = line.strip()
        return stripped == "Exiting..." or bool(PROMPT_STATS_RE.match(stripped))

    while lines and _is_trailing_line(lines[-1]):
        lines.pop()
        while lines and lines[-1].strip() == "":
            lines.pop()

    cleaned = "".join(lines).rstrip("\n")
    end = start + len(cleaned)
    return cleaned, start, end


def _extract_prompt_generation_tps(text: str) -> Tuple[float, float]:
    if not text:
        return np.nan, np.nan
    matches = list(PROMPT_TPS_RE.finditer(text))
    if not matches:
        return np.nan, np.nan
    last = matches[-1]
    try:
        prompt_tps = float(last.group(1))
        generation_tps = float(last.group(2))
    except ValueError:
        return np.nan, np.nan
    return prompt_tps, generation_tps


def _resolve_raw_json(raw_value: Any, summary_dir: Path, root: Path) -> Optional[Path]:
    if raw_value is None:
        return None
    raw_text = str(raw_value).strip()
    if raw_text == "":
        return None

    candidates: List[Path] = []
    raw_path = Path(raw_text)
    candidates.append(raw_path)
    if "results/experiments/" in raw_text:
        candidates.append(Path(raw_text.replace("results/experiments/", "data/experiments/")))
    candidates.append(summary_dir / raw_path.name)
    candidates.append(root / raw_path.name)

    for candidate in candidates:
        resolved = candidate
        if not resolved.is_absolute():
            resolved = (Path.cwd() / resolved).resolve()
        if resolved.exists():
            return resolved
    return None


def _make_prompt_uid(prompt_id: Optional[str], prompt_text: Optional[str]) -> str:
    if prompt_id:
        return str(prompt_id)
    if prompt_text:
        digest = hashlib.sha1(prompt_text.encode("utf-8")).hexdigest()[:12]
        return f"prompt_{digest}"
    return "prompt_unknown"


def _make_run_id(
    configuration_group: str,
    configuration_value: Optional[int],
    worker_id: Optional[int],
    prompt_uid: str,
    trial: Optional[int],
    warmup: Optional[bool],
    raw_json_path: Optional[Path],
    row_index: int,
) -> str:
    value_part = configuration_value if configuration_value is not None else "na"
    worker_part = worker_id if worker_id is not None else "na"
    trial_part = trial if trial is not None else "na"
    warmup_part = "warmup" if warmup else "run"
    raw_part = raw_json_path.stem if raw_json_path else f"row{row_index}"
    return f"{configuration_group}:{value_part}:{worker_part}:{prompt_uid}:{trial_part}:{warmup_part}:{raw_part}"


def _iter_summary_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("summary.csv"):
        if "analysis_exports" in path.parts:
            continue
        yield path


def _std_sample_zero_for_single(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) <= 1:
        return 0.0 if len(x) == 1 else np.nan
    return float(x.std(ddof=1))


def _mean_value(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return float(x.mean()) if len(x) else np.nan


def _format_mean_std(mean: Any, std: Any, decimals: int = 6) -> str:
    if pd.isna(mean):
        return ""
    std = 0.0 if pd.isna(std) else std
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def _summarize_by(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    base = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_observations=("prompt_id", "size"),
            n_prompts=("prompt_id", "nunique"),
            n_workers=("worker_id", lambda s: s.dropna().nunique()),
        )
        .reset_index()
    )

    metric_parts = []
    for metric in NUMERIC_METRICS:
        if metric not in df.columns:
            continue
        part = (
            df.groupby(group_cols, dropna=False)[metric]
            .agg(**{f"{metric}_mean": _mean_value, f"{metric}_std": _std_sample_zero_for_single})
            .reset_index()
        )
        part[f"{metric}_mean_pm_std"] = [
            _format_mean_std(m, s) for m, s in zip(part[f"{metric}_mean"], part[f"{metric}_std"])
        ]
        metric_parts.append(part)

    out = base
    for part in metric_parts:
        out = out.merge(part, on=group_cols, how="left")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Export per-run responses and response events.")
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT, help="Root experiments directory")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT_DEFAULT / "analysis_exports",
        help="Output directory for exports",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "jsonl"],
        default="csv",
        help="Output format for runs and events (csv or jsonl)",
    )
    parser.add_argument(
        "--runs-out",
        type=str,
        default=None,
        help="Filename for run-level output (defaults based on format)",
    )
    parser.add_argument(
        "--events-out",
        type=str,
        default=None,
        help="Filename for response events output (defaults based on format)",
    )
    args = parser.parse_args()

    root = args.root
    output_format = args.format
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_out_name = args.runs_out or (
        "all_runs_with_response.csv" if output_format == "csv" else "all_runs_with_response.jsonl"
    )
    events_out_name = args.events_out or (
        "all_runs_response_events.csv"
        if output_format == "csv"
        else "all_runs_response_events.jsonl"
    )
    runs_out_path = out_dir / runs_out_name
    events_out_path = out_dir / events_out_name

    csv.field_size_limit(10_000_000)

    summary_files = list(_iter_summary_files(root))
    runs_written = 0
    events_written = 0
    missing_raw_json: List[str] = []

    run_fieldnames = [
        "run_id",
        "configuration_group",
        "configuration_name",
        "configuration_value",
        "configuration_label",
        "worker_id",
        "prompt_id",
        "prompt_uid",
        "category",
        "mandatory",
        "trial",
        "warmup",
        "ttft_s",
        "tpot_s",
        "prompt_tps",
        "generation_tps",
        "throughput_events_per_s",
        "output_events_count",
        "total_time_s",
        "memory_peak_mb",
        "energy_j",
        "command",
        "prompt",
        "response_text",
        "response_char_count",
        "response_event_count",
        "summary_csv",
        "raw_json",
        "resource_csv",
        "response_events_file",
    ]
    event_fieldnames = [
        "run_id",
        "prompt_uid",
        "prompt_id",
        "event_index",
        "timestamp_s",
        "byte",
    ]

    def _to_csv_value(value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=True)
        return value

    if output_format == "csv":
        runs_handle = runs_out_path.open("w", encoding="utf-8", newline="")
        events_handle = events_out_path.open("w", encoding="utf-8", newline="")
        runs_writer = csv.DictWriter(runs_handle, fieldnames=run_fieldnames)
        events_writer = csv.DictWriter(events_handle, fieldnames=event_fieldnames)
        runs_writer.writeheader()
        events_writer.writeheader()

        def write_run(record: Dict[str, Any]) -> None:
            runs_writer.writerow({k: _to_csv_value(record.get(k)) for k in run_fieldnames})

        def write_event(record: Dict[str, Any]) -> None:
            events_writer.writerow({k: _to_csv_value(record.get(k)) for k in event_fieldnames})

    else:
        runs_handle = runs_out_path.open("w", encoding="utf-8", newline="\n")
        events_handle = events_out_path.open("w", encoding="utf-8", newline="\n")

        def write_run(record: Dict[str, Any]) -> None:
            runs_handle.write(json.dumps(record, ensure_ascii=True))
            runs_handle.write("\n")

        def write_event(record: Dict[str, Any]) -> None:
            events_handle.write(json.dumps(record, ensure_ascii=True))
            events_handle.write("\n")

    summary_rows: List[Dict[str, Any]] = []

    try:
        for summary_path in summary_files:
            config_info = _infer_configuration(summary_path, root)
            configuration_group = config_info["configuration_group"]
            configuration_name = config_info["configuration_name"]
            configuration_value = config_info["configuration_value"]
            configuration_label = config_info["configuration_label"]
            worker_id = config_info["worker_id"]

            with summary_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row_index, row in enumerate(reader):
                    raw_json_path = _resolve_raw_json(row.get("raw_json"), summary_path.parent, root)
                    run_data: Dict[str, Any] = {}
                    if raw_json_path is not None:
                        try:
                            run_data = json.loads(raw_json_path.read_text(encoding="utf-8"))
                        except Exception:
                            run_data = {}
                    else:
                        if row.get("raw_json"):
                            missing_raw_json.append(str(row.get("raw_json")))

                    prompt_id = row.get("prompt_id") or run_data.get("prompt_id")
                    prompt_text = run_data.get("prompt")
                    prompt_uid = _make_prompt_uid(prompt_id, prompt_text)
                    trial = _parse_int(row.get("trial"))
                    warmup = _parse_bool(row.get("warmup"))

                    run_id = _make_run_id(
                        configuration_group,
                        configuration_value,
                        worker_id,
                        prompt_uid,
                        trial,
                        warmup,
                        raw_json_path,
                        row_index,
                    )

                    generated_output = run_data.get("generated_output")
                    if not generated_output:
                        generated_output = row.get("generated_output", "")

                    generated_text = str(generated_output or "")
                    response_text, _, _ = _extract_response_span(generated_text)
                    prompt_tps, generation_tps = _extract_prompt_generation_tps(generated_text)
                    tpot_s = np.nan
                    if not pd.isna(generation_tps) and generation_tps > 0:
                        tpot_s = 1.0 / generation_tps

                    events = run_data.get("events") or []
                    event_output = "".join(
                        str(evt.get("byte") or "") for evt in events if isinstance(evt, dict)
                    )
                    _, ev_start, ev_end = _extract_response_span(event_output)

                    pos = 0
                    response_event_count = 0
                    for evt in events:
                        if not isinstance(evt, dict):
                            continue
                        byte_text = str(evt.get("byte") or "")
                        length = len(byte_text)
                        if length > 0 and pos < ev_end and (pos + length) > ev_start:
                            event_record = {
                                "run_id": run_id,
                                "prompt_uid": prompt_uid,
                                "prompt_id": prompt_id,
                                "event_index": evt.get("index"),
                                "timestamp_s": evt.get("timestamp_s"),
                                "byte": evt.get("byte"),
                            }
                            write_event(event_record)
                            events_written += 1
                            response_event_count += 1
                        pos += length

                    ttft_s = _parse_float(row.get("ttft_s"))
                    throughput_events_per_s = _parse_float(row.get("throughput_events_per_s"))
                    output_events_count = _parse_int(row.get("output_events_count"))
                    total_time_s = _parse_float(row.get("total_time_s"))
                    memory_peak_mb = _parse_float(row.get("memory_peak_mb"))
                    energy_j = _parse_float(row.get("energy_j"))
                    peak_rss_mb = memory_peak_mb

                    run_record: Dict[str, Any] = {
                        "run_id": run_id,
                        "configuration_group": configuration_group,
                        "configuration_name": configuration_name,
                        "configuration_value": configuration_value,
                        "configuration_label": configuration_label,
                        "worker_id": worker_id,
                        "prompt_id": prompt_id,
                        "prompt_uid": prompt_uid,
                        "category": row.get("category") or run_data.get("category"),
                        "mandatory": _parse_bool(row.get("mandatory")),
                        "trial": trial,
                        "warmup": warmup,
                        "ttft_s": ttft_s,
                        "tpot_s": tpot_s,
                        "prompt_tps": prompt_tps,
                        "generation_tps": generation_tps,
                        "throughput_events_per_s": throughput_events_per_s,
                        "output_events_count": output_events_count,
                        "total_time_s": total_time_s,
                        "memory_peak_mb": memory_peak_mb,
                        "energy_j": energy_j,
                        "command": run_data.get("command"),
                        "prompt": prompt_text,
                        "response_text": response_text,
                        "response_char_count": len(response_text),
                        "response_event_count": response_event_count,
                        "summary_csv": str(summary_path.as_posix()),
                        "raw_json": str(raw_json_path.as_posix()) if raw_json_path else row.get("raw_json"),
                        "resource_csv": row.get("resource_csv"),
                        "response_events_file": str(events_out_path.as_posix()),
                    }
                    write_run(run_record)
                    runs_written += 1

                    summary_rows.append({
                        "configuration_group": configuration_group,
                        "configuration_name": configuration_name,
                        "configuration_value": configuration_value,
                        "configuration_label": configuration_label,
                        "worker_id": worker_id,
                        "prompt_id": prompt_id,
                        "category": row.get("category") or run_data.get("category"),
                        "warmup": warmup,
                        "ttft_s": ttft_s,
                        "tpot_s": tpot_s,
                        "prompt_tps": prompt_tps,
                        "generation_tps": generation_tps,
                        "throughput_events_per_s": throughput_events_per_s,
                        "peak_rss_mb": peak_rss_mb,
                        "memory_peak_mb": memory_peak_mb,
                    })
    finally:
        runs_handle.close()
        events_handle.close()

    if summary_rows:
        grouped_df = pd.DataFrame(summary_rows)
        if "category" not in grouped_df.columns:
            grouped_df["category"] = "unknown"
        if "warmup" in grouped_df.columns:
            grouped_df = grouped_df[grouped_df["warmup"] != True].copy()
        if "peak_rss_mb" not in grouped_df.columns and "memory_peak_mb" in grouped_df.columns:
            grouped_df["peak_rss_mb"] = grouped_df["memory_peak_mb"]

        config_cols = [
            "configuration_group",
            "configuration_name",
            "configuration_value",
            "configuration_label",
        ]
        by_category = _summarize_by(grouped_df, config_cols + ["category"])
        overall_input = grouped_df.copy()
        overall_input["category"] = "overall"
        overall = _summarize_by(overall_input, config_cols + ["category"])

        clustered = pd.concat([by_category, overall], ignore_index=True, sort=False)
        cat_order = {"short": 0, "medium": 1, "long": 2, "overall": 3}
        clustered["_cat_order"] = clustered["category"].map(cat_order).fillna(99)
        clustered = clustered.sort_values(
            ["configuration_group", "configuration_value", "_cat_order", "category"],
            kind="stable",
        ).drop(columns=["_cat_order"])

        grouped_out_path = out_dir / "configuration_category_and_overall_mean_std.csv"
        clustered.to_csv(grouped_out_path, index=False)
        print(f"Wrote {grouped_out_path}")

    print(f"Summary files: {len(summary_files)}")
    print(f"Runs exported: {runs_written}")
    print(f"Response events exported: {events_written}")
    print(f"Missing raw_json paths: {len(missing_raw_json)}")
    print(f"Wrote runs: {runs_out_path.as_posix()}")
    print(f"Wrote events: {events_out_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
