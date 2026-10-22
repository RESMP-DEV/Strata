#!/usr/bin/env python3
"""Build a deterministic REAP-style expert-pruning control from routing traces.

This is the cheap control arm for the GSQ/RCO track.  It consumes the same
`--dump-routing` files as `tools/make_profile.py`, ranks each layer's experts by
observed router weight (or activation count with --unweighted), keeps exactly
`--keep` experts per layer, and writes both an explicit JSON mask and an optional
Strata expert profile.  It does not load a model or claim RCO's joint KL search.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_profile  # noqa: E402


def read_trace_scores(
    paths: list[Path], n_layers: int, n_expert: int, weighted: bool
) -> tuple[dict[tuple[int, int], float], int]:
    """Return `{(layer, expert): score}` and the number of usable records.

    A record is `layer, k, k int32 ids, k float32 weights`, exactly as the engine's
    `--dump-routing` writer emits it.  Out-of-range pairs are ignored, matching
    `make_profile.read_trace`; malformed files are rejected rather than truncated.
    """
    scores: dict[tuple[int, int], float] = defaultdict(float)
    records = 0
    for path in paths:
        blob = path.read_bytes()
        offset = 0
        while offset < len(blob):
            if offset + 8 > len(blob):
                raise ValueError(f"{path}: truncated routing header at byte {offset}")
            layer, count = struct.unpack_from("<ii", blob, offset)
            offset += 8
            if count < 0 or offset + 8 * count > len(blob):
                raise ValueError(f"{path}: invalid routing count {count} at layer {layer}")
            if not 0 <= layer < n_layers:
                offset += 8 * count
                continue
            ids = struct.unpack_from(f"<{count}i", blob, offset)
            weights = struct.unpack_from(f"<{count}f", blob, offset + 4 * count)
            offset += 8 * count
            records += 1
            for expert, weight in zip(ids, weights, strict=False):
                if 0 <= expert < n_expert:
                    # Serve-mode traces emit zero weights by contract and represent
                    # unit activations; one-shot traces preserve real router weights.
                    contribution = 1.0 if not weighted or weight == 0.0 else float(weight)
                    scores[(layer, expert)] += contribution
    return scores, records


def choose_mask(
    scores: dict[tuple[int, int], float],
    n_layers: int,
    n_expert: int,
    keep: int,
    seeds: list[set[int]] | None = None,
) -> tuple[list[list[int]], list[list[int]], list[list[float]]]:
    """Choose exactly `keep` experts per layer.

    Seed experts are selected first, in ascending source-id order. Remaining
    experts use descending score, with ties preferring the lower original id.
    Layers without enough observations are filled by ascending id; fill entries
    are returned separately rather than presented as trace-supported choices.
    """
    seed_rows = seeds if seeds is not None else [set() for _ in range(n_layers)]
    keep_ids: list[list[int]] = []
    fill_ids: list[list[int]] = []
    selected_scores: list[list[float]] = []
    for layer in range(n_layers):
        seed = seed_rows[layer]
        if len(seed) > keep:
            raise ValueError(f"layer {layer} seed has {len(seed)} experts, above budget {keep}")
        chosen_ids = sorted(seed)
        used = set(chosen_ids)
        chosen_scores = [scores.get((layer, expert), 0.0) for expert in chosen_ids]
        rows = [(pair[1], score) for pair, score in scores.items() if pair[0] == layer]
        rows.sort(key=lambda row: (-row[1], row[0]))
        for expert, score in rows:
            if len(chosen_ids) == keep:
                break
            if expert not in used:
                chosen_ids.append(expert)
                chosen_scores.append(score)
                used.add(expert)
        fills: list[int] = []
        for expert in range(n_expert):
            if len(chosen_ids) == keep:
                break
            if expert not in used:
                chosen_ids.append(expert)
                chosen_scores.append(0.0)
                fills.append(expert)
                used.add(expert)
        if len(chosen_ids) != keep:
            raise AssertionError("internal mask fill error")
        keep_ids.append(chosen_ids)
        fill_ids.append(fills)
        selected_scores.append(chosen_scores)
    return keep_ids, fill_ids, selected_scores


def parse_seed_allocation(path: Path, n_layers: int, n_expert: int) -> list[set[int]]:
    pattern = re.compile(r"^blk\.(\d+)\.ffn_down_exps\.weight\[(\d+)\]:\s*(\S+)\s*$")
    result: list[set[int]] = [set() for _ in range(n_layers)]
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match is None:
            continue
        layer, expert, quant = int(match[1]), int(match[2]), match[3]
        if quant and 0 <= layer < n_layers and 0 <= expert < n_expert:
            result[layer].add(expert)
    if not all(len(row) == 256 for row in result):
        sizes = sorted({len(row) for row in result})
        raise ValueError(f"{path}: seed allocation is not uniform 256/layer: {sizes}")
    return result


def write_profile(
    path: Path, keep_ids: list[list[int]], scores: dict[tuple[int, int], float], n_expert: int
) -> None:
    ranked = sorted(
        ((layer, expert) for layer, experts in enumerate(keep_ids) for expert in experts),
        key=lambda pair: (-scores.get(pair, 0.0), pair[0], pair[1]),
    )
    make_profile.write_profile(path, ranked, n_expert)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("traces", nargs="+", type=Path, help="--dump-routing files")
    parser.add_argument("--out", required=True, type=Path, help="JSON mask output")
    parser.add_argument("--profile", type=Path, help="optional Strata expert profile output")
    parser.add_argument(
        "--keep", type=int, default=256, help="experts kept per layer (default 256)"
    )
    parser.add_argument("--layers", type=int, default=make_profile.N_LAYER)
    parser.add_argument("--n-expert", type=int, default=make_profile.N_EXPERT)
    parser.add_argument(
        "--unweighted",
        action="store_true",
        help="count activations instead of router weights",
    )
    parser.add_argument(
        "--seed-allocation",
        type=Path,
        help="RCO allocation text whose retained experts are selected before trace scores",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < args.keep < args.n_expert:
        raise SystemExit(f"--keep must be in (0, {args.n_expert})")
    scores, records = read_trace_scores(
        args.traces, args.layers, args.n_expert, weighted=not args.unweighted
    )
    seeds = (
        parse_seed_allocation(args.seed_allocation, args.layers, args.n_expert)
        if args.seed_allocation is not None
        else None
    )
    keep_ids, fills, selected_scores = choose_mask(
        scores, args.layers, args.n_expert, args.keep, seeds=seeds
    )

    trace_supported = sum(
        (layer, expert) in scores for layer, experts in enumerate(keep_ids) for expert in experts
    )
    seed_selected = sum(len(row) for row in (seeds or []))
    filled_selected = sum(len(row) for row in fills)
    total_weight = sum(max(0.0, score) for score in scores.values())
    selected_weight = sum(
        max(0.0, scores.get((layer, expert), 0.0))
        for layer, experts in enumerate(keep_ids)
        for expert in experts
    )
    result = {
        "format": "strata-reap-mask-v1",
        "weighted": not args.unweighted,
        "n_layers": args.layers,
        "n_expert": args.n_expert,
        "keep_per_layer": args.keep,
        "trace_files": [str(path) for path in args.traces],
        "trace_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in args.traces],
        "trace_records": records,
        "distinct_observed_pairs": len(scores),
        "trace_supported_selected": trace_supported,
        "filled_selected": filled_selected,
        "unobserved_selected": args.layers * args.keep - trace_supported,
        "seed_selected": seed_selected,
        "seed_and_trace_selected": trace_supported,
        "positive_trace_weight_covered": selected_weight / total_weight if total_weight else 0.0,
        "keep_ids": keep_ids,
        "fill_ids": fills,
        "selected_scores": selected_scores,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.profile is not None:
        args.profile.parent.mkdir(parents=True, exist_ok=True)
        write_profile(args.profile, keep_ids, scores, args.n_expert)
    print(
        f"wrote {args.out}: {args.layers}x{args.keep} mask, "
        f"{trace_supported} trace-supported, {result['filled_selected']} filled, "
        f"positive-weight coverage={result['positive_trace_weight_covered']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
