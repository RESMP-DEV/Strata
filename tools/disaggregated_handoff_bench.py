#!/usr/bin/env python3
"""Measure Strata's real cross-engine prefill/decode session-file handoff.

The split is deliberately narrow and honest: session files refuse ``--batch`` and
``--layer-split``, so each lane is serial.  The prefill lane reads a prompt and
writes a model-bound conversation snapshot; the decode lane restores it and
generates the reply.  The two lanes run concurrently, which measures whether
decode overlaps the next prompt without claiming intra-lane parallelism.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import queue
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

TOPICS = (
    "Inspect this module and report the smallest correctness risk.",
    "Add one failing boundary test, then repair only that failure.",
    "Explain the rollback path and the state it must preserve.",
    "Review the diff for hidden cross-process state.",
    "Draft the next agent's handoff with exact evidence paths.",
)


def shared_context(words: int) -> str:
    """Deterministic common user document pinned with ``strata_prefix``."""
    paragraphs: list[str] = []
    remaining = words
    index = 0
    while remaining > 0:
        count = min(96, remaining)
        paragraphs.append(
            f"Shared repository note {index}: admission owns request ordering, "
            "the prefill lane owns prompt state publication, and the decode lane "
            "owns generation after restore. Interfaces and hashes stay explicit. " * 3
        )
        remaining -= count
        index += 1
    return (
        "You are serving a bounded engineering handoff. Preserve unrelated work, "
        "name the exact state boundary, and cite live evidence.\n\n" + "\n\n".join(paragraphs)
    )


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def request_messages(context: str, worker: int, round_index: int) -> list[dict[str, str]]:
    turn = TOPICS[round_index % len(TOPICS)]
    return [
        {"role": "user", "content": context},
        {"role": "user", "content": f"Worker {worker}, round {round_index}: {turn}"},
    ]


def stream_chat(
    url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: float,
    reasoning_effort: str,
) -> dict[str, Any]:
    """Stream one OpenAI chat request and retain usage and Strata timings."""
    body: dict[str, Any] = {
        "model": model,
        "stream": True,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": messages,
        "strata_prefix": {"messages": 1},
    }
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
        body["chat_template_kwargs"] = {"enable_thinking": False}
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token = None
    pieces: list[str] = []
    usage: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    error = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except ValueError:
                    continue
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                    if piece:
                        pieces.append(piece)
                        if first_token is None:
                            first_token = time.perf_counter() - started
                if event.get("usage"):
                    usage = event["usage"]
                if event.get("timings"):
                    timings = event["timings"]
    except Exception as exc:  # noqa: BLE001 - a benchmark reports every request failure
        error = f"{type(exc).__name__}: {exc}"
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    return {
        "wall_s": time.perf_counter() - started,
        "first_token_s": first_token,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached,
        "fresh_tokens": max(0, prompt_tokens - cached),
        "cache_ratio": cached / prompt_tokens if prompt_tokens else 0.0,
        "completion_tokens": completion,
        "timings": timings,
        "text": "".join(pieces),
        "error": error,
    }


def slot_action(url: str, action: str, filename: str, timeout: float) -> dict[str, Any]:
    """Invoke the single-slot session save/restore HTTP API."""
    request = urllib.request.Request(
        url.rstrip("/") + f"/slots/0?action={action}",
        data=json.dumps({"filename": filename}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.load(response)
    except Exception as exc:  # noqa: BLE001 - a benchmark reports every handoff failure
        return {
            "wall_s": time.perf_counter() - started,
            "action": action,
            "filename": filename,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "wall_s": time.perf_counter() - started,
        "action": action,
        "filename": filename,
        "response": body,
        "error": None,
    }


def summarize_requests(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if not record.get("error")]
    prompt_tokens = sum(record["prompt_tokens"] for record in valid)
    cached = sum(record["cached_tokens"] for record in valid)
    completion = sum(record["completion_tokens"] for record in valid)
    first = [
        float(record["first_token_s"]) for record in valid if record["first_token_s"] is not None
    ]
    return {
        "requests": len(records),
        "errors": len(records) - len(valid),
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached,
        "fresh_tokens": prompt_tokens - cached,
        "cache_ratio": cached / prompt_tokens if prompt_tokens else 0.0,
        "completion_tokens": completion,
        "first_token_p50_s": percentile(first, 0.50),
        "first_token_p95_s": percentile(first, 0.95),
    }


def summarize_slot(records: list[dict[str, Any]], action: str) -> dict[str, Any]:
    valid = [record for record in records if not record["error"]]
    times = [record["wall_s"] for record in valid]
    engine = [
        float(record["response"]["timings"][f"{action}_ms"]) / 1000.0
        for record in valid
        if record.get("response", {}).get("timings", {}).get(f"{action}_ms") is not None
    ]
    tokens = [int(record["response"].get(f"n_{action}d", 0)) for record in valid]
    bytes_ = [
        int(record["response"].get("n_written" if action == "save" else "n_read", 0))
        for record in valid
    ]
    return {
        "operations": len(records),
        "errors": len(records) - len(valid),
        "client_wall_s_sum": sum(times),
        "client_wall_s_p50": percentile(times, 0.50),
        "client_wall_s_p95": percentile(times, 0.95),
        "engine_s_sum": sum(engine),
        "engine_s_p50": percentile(engine, 0.50),
        "tokens_sum": sum(tokens),
        "bytes_sum": sum(bytes_),
    }


def summarize_split(
    records: list[dict[str, Any]],
    started: float,
    ended: float,
    prefill_busy: float,
    decode_busy: float,
) -> dict[str, Any]:
    saves = [record["save"] for record in records]
    restores = [record["restore"] for record in records]
    prefill_requests = [record["prefill"] for record in records]
    decode_requests = [record["decode"] for record in records]
    decode_summary = summarize_requests(decode_requests)
    makespan = ended - started
    completion = decode_summary["completion_tokens"]
    return {
        "makespan_s": makespan,
        "aggregate_completion_tok_s": completion / makespan if makespan else 0.0,
        "prefill_summary": summarize_requests(prefill_requests),
        "decode_summary": decode_summary,
        "save_summary": summarize_slot(saves, "save"),
        "restore_summary": summarize_slot(restores, "restore"),
        "prefill_lane_busy_s": prefill_busy,
        "decode_lane_busy_s": decode_busy,
        "lane_overlap_s": max(0.0, prefill_busy + decode_busy - makespan),
    }


def _prefill_lane(
    args: argparse.Namespace,
    requests: list[dict[str, Any]],
    output: queue.Queue[dict[str, Any] | None],
) -> tuple[float, list[dict[str, Any]]]:
    busy = 0.0
    all_requests: list[dict[str, Any]] = []
    for index, item in enumerate(requests):
        filename = f"{args.session_prefix}-{index:05d}.bin"
        started = time.perf_counter()
        prefill = stream_chat(
            args.prefill_url,
            args.model,
            item["messages"],
            1,
            args.timeout,
            args.reasoning_effort,
        )
        save = slot_action(args.prefill_url, "save", filename, args.timeout)
        busy += time.perf_counter() - started
        record = {**item, "index": index, "filename": filename, "prefill": prefill, "save": save}
        all_requests.append(record)
        output.put(record)
    output.put(None)
    return busy, all_requests


def _decode_lane(
    args: argparse.Namespace,
    input_queue: queue.Queue[dict[str, Any] | None],
) -> tuple[float, list[dict[str, Any]]]:
    busy = 0.0
    records: list[dict[str, Any]] = []
    while True:
        record = input_queue.get()
        if record is None:
            break
        started = time.perf_counter()
        restore = slot_action(args.decode_url, "restore", record["filename"], args.timeout)
        decode = stream_chat(
            args.decode_url,
            args.model,
            record["messages"],
            args.max_tokens,
            args.timeout,
            args.reasoning_effort,
        )
        busy += time.perf_counter() - started
        records.append({**record, "restore": restore, "decode": decode})
    return busy, records


def run_split(args: argparse.Namespace) -> dict[str, Any]:
    requests = [
        {
            "worker": worker,
            "round": round_index,
            "messages": request_messages(shared_context(args.context_words), worker, round_index),
        }
        for round_index in range(args.rounds)
        for worker in range(args.workers)
    ]
    handoffs: queue.Queue[dict[str, Any] | None] = queue.Queue()
    started = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="strata-handoff") as executor:
        prefill_future = executor.submit(_prefill_lane, args, requests, handoffs)
        decode_future = executor.submit(_decode_lane, args, handoffs)
        prefill_busy, _ = prefill_future.result()
        decode_busy, decode_records = decode_future.result()
    ended = time.perf_counter()
    result = {
        "format": "strata-disaggregated-handoff-bench-v1",
        "mode": "split",
        "url": args.prefill_url,
        "decode_url": args.decode_url,
        "model": args.model,
        "workers": args.workers,
        "rounds": args.rounds,
        "summary": summarize_split(decode_records, started, ended, prefill_busy, decode_busy),
        "records": decode_records,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_control(args: argparse.Namespace) -> dict[str, Any]:
    requests = [
        {
            "worker": worker,
            "round": round_index,
            "messages": request_messages(shared_context(args.context_words), worker, round_index),
        }
        for round_index in range(args.rounds)
        for worker in range(args.workers)
    ]
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    for item in requests:
        reply = stream_chat(
            args.control_url,
            args.model,
            item["messages"],
            args.max_tokens,
            args.timeout,
            args.reasoning_effort,
        )
        records.append({**item, "reply": reply})
    ended = time.perf_counter()
    summary = summarize_requests([record["reply"] for record in records])
    summary["makespan_s"] = ended - started
    summary["aggregate_completion_tok_s"] = (
        summary["completion_tokens"] / summary["makespan_s"] if summary["makespan_s"] else 0.0
    )
    result = {
        "format": "strata-disaggregated-handoff-bench-v1",
        "mode": "control",
        "url": args.control_url,
        "model": args.model,
        "workers": args.workers,
        "rounds": args.rounds,
        "summary": summary,
        "records": records,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def compare_control(split: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    split_rows = split["records"]
    control_rows = control["records"]
    if len(split_rows) != len(control_rows):
        return {
            "exact_text_matches": 0,
            "requests": len(split_rows),
            "error": "record count mismatch",
        }
    matches = [
        left["decode"]["text"] == right["reply"]["text"]
        for left, right in zip(split_rows, control_rows, strict=True)
    ]
    return {"exact_text_matches": sum(matches), "requests": len(matches), "all_exact": all(matches)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("split", "control"), required=True)
    parser.add_argument("--prefill-url", help="single-GPU prefill/save server")
    parser.add_argument("--decode-url", help="single-GPU restore/decode server")
    parser.add_argument("--control-url", help="single-lane full prefill+decode server")
    parser.add_argument("--model", default="strata")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--context-words", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--session-prefix", default="handoff")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "split" and (not args.prefill_url or not args.decode_url):
        parser.error("--mode split needs --prefill-url and --decode-url")
    if args.mode == "control" and not args.control_url:
        parser.error("--mode control needs --control-url")
    if args.workers < 1 or args.rounds < 1 or args.context_words < 32 or args.max_tokens < 1:
        parser.error("workers/rounds/context/max tokens are too small")
    return args


def main() -> int:
    args = parse_args()
    result = run_split(args) if args.mode == "split" else run_control(args)
    summary = result["summary"]
    if args.mode == "split":
        print(
            f"split requests={summary['decode_summary']['requests']} "
            f"errors={summary['decode_summary']['errors']} "
            f"makespan={summary['makespan_s']:.3f}s "
            f"completion_tok_s={summary['aggregate_completion_tok_s']:.2f} "
            f"save_p50={summary['save_summary']['client_wall_s_p50']:.3f}s "
            f"restore_p50={summary['restore_summary']['client_wall_s_p50']:.3f}s"
        )
    else:
        print(
            f"control requests={summary['requests']} errors={summary['errors']} "
            f"makespan={summary['makespan_s']:.3f}s "
            f"completion_tok_s={summary['aggregate_completion_tok_s']:.2f} "
            f"ttft_p50/p95={summary['first_token_p50_s']:.3f}/"
            f"{summary['first_token_p95_s']:.3f}s"
        )
    errors = summary["decode_summary"]["errors"] if args.mode == "split" else summary["errors"]
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
