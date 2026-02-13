#!/usr/bin/env python3
"""
Count comment threads and signals by signal_type from a labeled threads JSON file.
Reusable: pass any *_labeled.json path.

Usage:
  python count_signal_types.py [path_to_labeled.json]
  python count_signal_types.py threads_rPckDuTaSnw_with_likes.txt_labeled.json

Output:
  - Count of THREADS that have at least one signal of each type (thread-level)
  - Count of SIGNALS per type (signal-level; one thread can have multiple signals)
  - Optional: write summary to a JSON file (--out).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_labeled(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_by_signal_type(data: dict) -> tuple[dict, dict]:
    """
    Returns (signals_per_type, threads_per_type).
    - signals_per_type: for each signal_type, number of signals (can be multiple per thread).
    - threads_per_type: for each signal_type, number of threads that have at least one signal of that type.
    """
    signals_per_type: dict[str, int] = defaultdict(int)
    threads_per_type: dict[str, set[str]] = defaultdict(set)  # type -> set of thread_ids

    for t in data.get("threads", []):
        thread_id = t.get("thread_id") or ""
        for sig in t.get("signals", []):
            st = sig.get("signal_type")
            if not st:
                continue
            st = str(st).strip()
            signals_per_type[st] += 1
            threads_per_type[st].add(thread_id)

    # Convert thread sets to counts
    thread_counts = {k: len(v) for k, v in threads_per_type.items()}
    return dict(signals_per_type), thread_counts


def count_threads_by_signal_count(data: dict) -> dict[int, int]:
    """
    Returns mapping: number_of_signals -> count of threads with that many signals.
    E.g. {1: 3500, 2: 800, 3: 150} = 3500 threads with 1 signal, 800 with 2, 150 with 3.
    """
    out: dict[int, int] = defaultdict(int)
    for t in data.get("threads", []):
        n = len(t.get("signals") or [])
        out[n] += 1
    return dict(out)


def unique_topics(data: dict) -> set[str]:
    """Collect all unique topic strings across all signals (each signal has topics: [...])."""
    topics: set[str] = set()
    for t in data.get("threads", []):
        for sig in t.get("signals", []):
            for topic in sig.get("topics") or []:
                if topic and isinstance(topic, str):
                    topics.add(topic.strip())
    return topics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Count threads and signals by signal_type from a labeled threads JSON."
    )
    parser.add_argument(
        "labeled_json",
        nargs="?",
        default=None,
        help="Path to *_labeled.json (e.g. threads_rPckDuTaSnw_with_likes.txt_labeled.json)",
    )
    parser.add_argument(
        "--out", "-o",
        metavar="FILE",
        help="Write summary (counts by signal_type) to this JSON file.",
    )
    parser.add_argument(
        "--sort",
        choices=("count", "name"),
        default="count",
        help="Sort signal types by count (desc) or name (default: count).",
    )
    parser.add_argument(
        "--list-topics",
        action="store_true",
        help="Print all unique topics (one per line) after the count.",
    )
    args = parser.parse_args()

    path = args.labeled_json
    if not path:
        parser.print_help()
        print("\nError: provide a labeled JSON file path.", file=sys.stderr)
        return 1
    if not Path(path).is_file():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    data = load_labeled(path)
    signals_per_type, threads_per_type = count_by_signal_type(data)
    threads_by_signal_count = count_threads_by_signal_count(data)
    topics_set = unique_topics(data)

    # Sort keys
    keys = list(signals_per_type.keys())
    if args.sort == "count":
        keys.sort(key=lambda k: -threads_per_type.get(k, 0))
    else:
        keys.sort(key=str.lower)

    total_threads = sum(1 for t in data.get("threads", []) if t.get("thread_id"))
    total_signals = sum(signals_per_type.values())

    print(f"File: {path}")
    print(f"Total threads: {total_threads}  |  Total signals: {total_signals}")
    print(f"Unique topics: {len(topics_set)}")
    print()
    # Threads by number of signals (single vs multiple)
    print("Threads by number of signals:")
    print("-" * 50)
    print(f"{'# signals':>10} {'threads':>10}  (threads with multiple signals)")
    print("-" * 50)
    for n in sorted(threads_by_signal_count.keys()):
        c = threads_by_signal_count[n]
        label = " (multiple)" if n > 1 else ""
        print(f"{n:>10} {c:>10}{label}")
    print("-" * 50)
    multi = sum(c for n, c in threads_by_signal_count.items() if n > 1)
    print(f"  Total threads with multiple signals: {multi}")
    print()
    print("By signal_type:")
    print("-" * 60)
    print(f"{'signal_type':<24} {'threads':>10} {'signals':>10}")
    print("-" * 60)
    for k in keys:
        tc = threads_per_type.get(k, 0)
        sc = signals_per_type.get(k, 0)
        print(f"{k:<24} {tc:>10} {sc:>10}")
    print("-" * 60)

    if args.list_topics:
        print("All unique topics:")
        print("-" * 50)
        for topic in sorted(topics_set, key=str.lower):
            print(topic)
        print("-" * 50)
        print(f"Total unique topics: {len(topics_set)}")
        print()

    summary = {
        "file": path,
        "total_threads": total_threads,
        "total_signals": total_signals,
        "unique_topics": len(topics_set),
        "threads_with_multiple_signals": multi,
        "threads_by_signal_count": {str(k): v for k, v in sorted(threads_by_signal_count.items())},
        "by_signal_type": {
            k: {"threads": threads_per_type.get(k, 0), "signals": signals_per_type.get(k, 0)}
            for k in keys
        },
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote summary to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
