#!/usr/bin/env python3
"""Measure parallel agent turns when conversation cache reuse is the common case.

The workload is deliberately different from ``serve_load.py``: every worker keeps
a long conversation, and later rounds send only a short new user turn.  A round
submits all workers concurrently, records OpenAI usage (including cached prompt
tokens) and Strata's timings, then advances that worker's history with the model
reply.  Round zero is the cold prime; the reported warm summary excludes it.

Standard library only, so the same driver can run from the Mac against a remote
loopback SSH tunnel or directly on a compute host.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

TOPICS = (
    "Refactor this module without changing its public API.",
    "Add a failing test for the boundary condition, then repair it.",
    "Explain the failure and the smallest safe rollback.",
    "Review the diff for correctness and maintainability.",
    "Draft a short implementation note for the next agent.",
)

FIXED_REPLIES = (
    "Boundary acknowledged. No unrelated files changed.",
    "Regression test added; the boundary condition is pinned.",
    "Root cause isolated and the minimal safe repair applied.",
    "Diff reviewed; correctness and rollback path are documented.",
    "Implementation note prepared for the next agent.",
)


def long_context(worker: int, words: int) -> str:
    """Deterministic, worker-specific context large enough to dominate a turn."""
    paragraphs = []
    remaining = words
    index = 0
    while remaining > 0:
        count = min(90, remaining)
        paragraphs.append(
            f"Worker {worker} repository note {index}: the service owns request "
            f"admission, cache restoration, and graceful shutdown. Interface "
            f"contracts remain stable while agents edit isolated files. "
            f"Verification uses focused tests plus one integration receipt. " * 3
        )
        remaining -= count
        index += 1
    return (
        f"You are worker {worker} in a shared engineering session. Follow the "
        "repository contracts, preserve unrelated work, and cite live evidence.\n\n"
        + "\n\n".join(paragraphs)
    )


def shared_context(words: int) -> str:
    """The common repository prefix every parallel worker reads."""
    paragraphs = []
    remaining = words
    index = 0
    while remaining > 0:
        count = min(90, remaining)
        paragraphs.append(
            f"Shared repository note {index}: the service owns request admission, "
            "cache restoration, and graceful shutdown. Interface contracts remain "
            "stable while agents edit isolated files. Verification uses focused "
            "tests plus one integration receipt. " * 3
        )
        remaining -= count
        index += 1
    return (
        "You are one of several workers in a shared engineering session. Follow "
        "the repository contracts, preserve unrelated work, and cite live evidence.\n\n"
        + "\n\n".join(paragraphs)
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


def one_request(
    url: str,
    model: str,
    worker: int,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: float,
    reasoning_effort: str,
    prefix_messages: int | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "stream": True,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if prefix_messages:
        body["strata_prefix"] = {"messages": prefix_messages}
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
    except Exception as exc:  # noqa: BLE001 - a load driver reports every request failure
        error = f"{type(exc).__name__}: {exc}"
    wall = time.perf_counter() - started
    text = "".join(pieces)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    completion = int(usage.get("completion_tokens") or len(pieces))
    return {
        "worker": worker,
        "wall_s": wall,
        "first_token_s": first_token,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached,
        "fresh_tokens": max(0, prompt_tokens - cached),
        "cache_ratio": cached / prompt_tokens if prompt_tokens else 0.0,
        "completion_tokens": completion,
        "prompt_per_second": timings.get("prompt_per_second"),
        "predicted_per_second": timings.get("predicted_per_second"),
        "prompt_ms": timings.get("prompt_ms"),
        "predicted_ms": timings.get("predicted_ms"),
        "draft_n": timings.get("draft_n"),
        "draft_n_accepted": timings.get("draft_n_accepted"),
        "text": text,
        "error": error,
    }


def summarize_round(
    round_index: int, records: list[dict[str, Any]], wall_s: float
) -> dict[str, Any]:
    valid = [record for record in records if not record["error"]]
    prompt_tokens = sum(record["prompt_tokens"] for record in valid)
    cached = sum(record["cached_tokens"] for record in valid)
    fresh = sum(record["fresh_tokens"] for record in valid)
    completion = sum(record["completion_tokens"] for record in valid)
    prompt_ms = [
        float(record["prompt_ms"]) for record in valid if record.get("prompt_ms") is not None
    ]
    predicted = [
        float(record["predicted_per_second"])
        for record in valid
        if record.get("predicted_per_second") is not None
    ]
    first = [
        float(record["first_token_s"]) for record in valid if record["first_token_s"] is not None
    ]
    return {
        "round": round_index,
        "requests": len(records),
        "errors": len(records) - len(valid),
        "wall_s": wall_s,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached,
        "fresh_tokens": fresh,
        "cache_ratio": cached / prompt_tokens if prompt_tokens else 0.0,
        "completion_tokens": completion,
        "aggregate_completion_tok_s": completion / wall_s if wall_s else 0.0,
        "client_fresh_prompt_tok_s": fresh / wall_s if wall_s else 0.0,
        "engine_fresh_prompt_tok_s": fresh / (sum(prompt_ms) / 1000) if sum(prompt_ms) else 0.0,
        "first_token_p50_s": percentile(first, 0.50),
        "first_token_p95_s": percentile(first, 0.95),
        "predicted_per_request_p50": median(predicted),
        "records": records,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    histories: list[list[dict[str, str]]] = []
    for worker in range(args.workers):
        system = (
            shared_context(args.context_words)
            if args.shared_prefix
            else long_context(worker, args.context_words)
        )
        histories.append(
            [
                {"role": "user" if args.shared_prefix else "system", "content": system},
                {
                    "role": "user",
                    "content": f"Worker {worker}: start the task and acknowledge your boundary.",
                },
            ]
        )

    rounds: list[dict[str, Any]] = []
    for round_index in range(args.rounds + 1):
        requests: list[tuple[int, list[dict[str, str]]]] = []
        for worker, history in enumerate(histories):
            turn = TOPICS[round_index % len(TOPICS)]
            messages = (
                history if round_index == 0 else [*history, {"role": "user", "content": turn}]
            )
            requests.append((worker, messages))

        started = time.perf_counter()
        with cf.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    one_request,
                    args.url,
                    args.model,
                    worker,
                    messages,
                    args.max_tokens,
                    args.timeout,
                    args.reasoning_effort,
                    1 if args.shared_prefix else None,
                ): worker
                for worker, messages in requests
            }
            by_worker = {}
            for future in cf.as_completed(futures):
                record = future.result()
                by_worker[record["worker"]] = record
        wall = time.perf_counter() - started
        records = [by_worker[worker] for worker in range(args.workers)]
        rounds.append(summarize_round(round_index, records, wall))

        if round_index < args.rounds:
            for worker, record in enumerate(records):
                reply = (
                    FIXED_REPLIES[round_index % len(FIXED_REPLIES)]
                    if args.fixed_replies
                    else record["text"].strip() or "(no reply)"
                )
                histories[worker].append({"role": "assistant", "content": reply})
            # A slot's held prefix is published after its BDONE bookkeeping.  Without
            # this pause a next burst can race that release and measure cold reads.
            time.sleep(args.inter_round_delay)

    warm = rounds[1:]
    warm_prompt = sum(row["prompt_tokens"] for row in warm)
    warm_cached = sum(row["cached_tokens"] for row in warm)
    warm_completion = sum(row["completion_tokens"] for row in warm)
    warm_wall = sum(row["wall_s"] for row in warm)
    result = {
        "format": "strata-agent-cache-bench-v1",
        "url": args.url,
        "model": args.model,
        "workers": args.workers,
        "cold_prime": rounds[0],
        "warm_rounds": warm,
        "warm_summary": {
            "rounds": len(warm),
            "requests": sum(row["requests"] for row in warm),
            "errors": sum(row["errors"] for row in warm),
            "prompt_tokens": warm_prompt,
            "cached_tokens": warm_cached,
            "cache_ratio": warm_cached / warm_prompt if warm_prompt else 0.0,
            "fresh_tokens": warm_prompt - warm_cached,
            "completion_tokens": warm_completion,
            "round_wall_s_sum": warm_wall,
            "aggregate_completion_tok_s": warm_completion / warm_wall if warm_wall else 0.0,
            "first_token_p50_s": percentile(
                [
                    float(r["first_token_s"])
                    for row in warm
                    for r in row["records"]
                    if r["first_token_s"] is not None
                ],
                0.50,
            ),
            "first_token_p95_s": percentile(
                [
                    float(r["first_token_s"])
                    for row in warm
                    for r in row["records"]
                    if r["first_token_s"] is not None
                ],
                0.95,
            ),
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.transcripts:
        transcript_dir = Path(args.transcripts)
        transcript_dir.mkdir(parents=True, exist_ok=True)
        for row in rounds:
            for record in row["records"]:
                path = transcript_dir / f"r{row['round']}-w{record['worker']}.txt"
                path.write_text(record["text"], encoding="utf-8")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url")
    parser.add_argument("--model", default="strata")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=4, help="warm rounds after the cold prime")
    parser.add_argument("--context-words", type=int, default=1200)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument(
        "--inter-round-delay", type=float, default=0.5, help="seconds between admission bursts"
    )
    parser.add_argument(
        "--fixed-replies",
        action="store_true",
        help="advance histories with deterministic replies for a fair cross-model control",
    )
    parser.add_argument(
        "--shared-prefix",
        action="store_true",
        help="pin the common first user document with strata_prefix for every worker",
    )
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--transcripts", type=Path)
    args = parser.parse_args(argv)
    if (
        args.workers < 1
        or args.rounds < 1
        or args.context_words < 32
        or args.max_tokens < 1
        or args.inter_round_delay < 0
    ):
        parser.error("workers/rounds/context/max tokens are too small for an agent-cache benchmark")
    return args


def main() -> int:
    result = run_benchmark(parse_args())
    summary = result["warm_summary"]
    print(
        f"warm workers={result['workers']} rounds={summary['rounds']} "
        f"cache={summary['cache_ratio']:.4f} fresh={summary['fresh_tokens']} "
        f"completion_tok_s={summary['aggregate_completion_tok_s']:.2f} "
        f"ttft_p50/p95={summary['first_token_p50_s']:.3f}/{summary['first_token_p95_s']:.3f}s "
        f"errors={summary['errors']}"
    )
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
