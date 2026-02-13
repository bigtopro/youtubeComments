#!/usr/bin/env python3
"""
Build initial subgrouping (buckets) of signals by:
  1) signal_type
  2) topic

A signal can belong to multiple topic buckets if it has multiple topics.

Usage:
  python group_signals.py threads_rPckDuTaSnw_with_likes.txt_labeled.json
  python group_signals.py threads_rPckDuTaSnw_with_likes.txt_labeled.json --out buckets.json

Output (stdout):
  - For each signal_type, the top N topics by signal count (default N=10).

If --out is provided, writes a JSON file with structure:
{
  "file": "...",
  "total_threads": ...,
  "total_signals": ...,
  "buckets": {
    "<signal_type>": {
      "<topic>": {
        "signal_count": <int>,
        "thread_count": <int>,
        "thread_ids": [<str>, ...]
      },
      ...
    },
    ...
  }
}
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any


def load_labeled(path: str) -> Dict[str, Any]:
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)


def build_buckets(data: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
  """
  Build nested buckets:
    buckets[signal_type][topic] -> {
      "signal_count": int,
      "thread_count": int,
      "thread_ids": [<thread_id>, ...],
    }

  - signal_count: how many signals (across all threads) have this signal_type+topic.
  - thread_count: how many distinct threads have at least one signal with this signal_type+topic.
  - thread_ids: the concrete list of those distinct thread_ids (for downstream graph building).
  """
  # signal_type -> topic -> signal_count
  signal_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
  # signal_type -> topic -> set(thread_id)
  threads_for_bucket: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))

  for t in data.get("threads", []):
    thread_id = t.get("thread_id")
    signals = t.get("signals") or []
    for sig in signals:
      st = sig.get("signal_type")
      if not st:
        continue
      st = str(st).strip()
      topics = sig.get("topics") or []
      if not topics:
        # Optionally bucket signals with no topics under a special key
        topics = ["__NO_TOPIC__"]
      for topic in topics:
        if not isinstance(topic, str):
          continue
        topic_key = topic.strip()
        if not topic_key:
          continue
        signal_counts[st][topic_key] += 1
        if thread_id:
          threads_for_bucket[st][topic_key].add(thread_id)

  buckets: Dict[str, Dict[str, Dict[str, Any]]] = {}
  for st, topic_map in signal_counts.items():
    buckets[st] = {}
    for topic, scount in topic_map.items():
      tset = threads_for_bucket[st].get(topic, set())
      buckets[st][topic] = {
        "signal_count": scount,
        "thread_count": len(tset),
        "thread_ids": sorted(tset),
      }
  return buckets


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Group signals into buckets by signal_type and topic."
  )
  parser.add_argument(
    "labeled_json",
    help="Path to *_labeled.json (e.g. threads_rPckDuTaSnw_with_likes.txt_labeled.json)",
  )
  parser.add_argument(
    "--out",
    "-o",
    metavar="FILE",
    help="Write full buckets structure to this JSON file.",
  )
  parser.add_argument(
    "--top-n",
    type=int,
    default=10,
    help="Show top N topics per signal_type (by signal_count) on stdout (default: 10).",
  )
  args = parser.parse_args()

  path = args.labeled_json
  if not Path(path).is_file():
    print(f"Error: file not found: {path}", file=sys.stderr)
    return 1

  data = load_labeled(path)
  buckets = build_buckets(data)

  total_threads = len(data.get("threads") or [])
  total_signals = sum(
    len(t.get("signals") or []) for t in data.get("threads", [])
  )

  print(f"File: {path}")
  print(f"Total threads: {total_threads}  |  Total signals: {total_signals}")
  print()
  # Short summary: number of topics per signal_type
  topics_per_type = {st: len(topic_map) for st, topic_map in buckets.items()}
  print("Topics per signal_type:")
  print("-" * 40)
  print(f"{'signal_type':<24} {'# topics':>10}")
  print("-" * 40)
  for st in sorted(topics_per_type.keys(), key=lambda s: -topics_per_type[s]):
    print(f"{st:<24} {topics_per_type[st]:>10}")
  print("-" * 40)
  print(f"{'TOTAL (unique topic keys)':<24} {sum(topics_per_type.values()):>10}")
  print()
  print(f"Top {args.top_n} topics per signal_type (by signal_count):")
  print("-" * 80)

  for st in sorted(buckets.keys()):
    topic_map = buckets[st]
    # sort topics by signal_count desc, then name
    sorted_topics = sorted(
      topic_map.items(),
      key=lambda kv: (-kv[1]["signal_count"], kv[0].lower()),
    )
    print(f"\n[signal_type = {st}]")
    print(f"{'topic':<60} {'signals':>8} {'threads':>8}")
    print("-" * 80)
    for topic, stats in sorted_topics[: args.top_n]:
      print(
        f"{topic[:57] + '...' if len(topic) > 60 else topic:<60} "
        f"{stats['signal_count']:>8} {stats['thread_count']:>8}"
      )

  if args.out:
    summary = {
      "file": path,
      "total_threads": total_threads,
      "total_signals": total_signals,
      "topics_per_signal_type": topics_per_type,
      "buckets": buckets,
    }
    with open(args.out, "w", encoding="utf-8") as f:
      json.dump(summary, f, indent=2)
    print(f"\nWrote buckets JSON to {args.out}")

  return 0


if __name__ == "__main__":
  sys.exit(main())

