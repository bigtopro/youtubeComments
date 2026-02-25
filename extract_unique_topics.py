#!/usr/bin/env python3
"""
Extract unique topics from a buckets JSON.

Outputs a JSON with:
- all_unique_topics: sorted list of all unique topic strings (across all signal types)
- topics_by_signal_type: { signal_type: sorted list of unique topics }
- counts: total unique topics, and per-signal-type counts

Usage:
  python extract_unique_topics.py buckets.json --out unique_topics.json
  python extract_unique_topics.py revised_buckets.json -o unique_topics_revised.json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

from consolidate_topics import load_buckets


def extract_unique_topics(buckets: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    From buckets[signal_type][topic] -> {...}, extract:
    - all_unique_topics: union of all topic strings, sorted
    - topics_by_signal_type: { signal_type: sorted list of topics }
    """
    all_topics: set[str] = set()
    topics_by_signal_type: Dict[str, List[str]] = {}

    for signal_type, topic_map in buckets.items():
        if not isinstance(topic_map, dict):
            continue
        topics = sorted(topic_map.keys())
        topics_by_signal_type[signal_type] = topics
        all_topics.update(topics)

    return {
        "all_unique_topics": sorted(all_topics),
        "topics_by_signal_type": topics_by_signal_type,
        "counts": {
            "total_unique_topics": len(all_topics),
            "by_signal_type": {
                st: len(topics) for st, topics in topics_by_signal_type.items()
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract unique topics (all and per signal type) from buckets JSON"
    )
    parser.add_argument("buckets_json", help="Path to buckets JSON (e.g. from group_signals or revised_buckets)")
    parser.add_argument("--out", "-o", required=True, help="Output JSON path")
    parser.add_argument(
        "--source-metadata",
        action="store_true",
        help="Include source file path in output metadata",
    )
    args = parser.parse_args()

    print(f"Loading buckets from {args.buckets_json}...")
    buckets = load_buckets(args.buckets_json)
    if not buckets:
        print("Error: No buckets found.", file=sys.stderr)
        return 1

    print(f"Extracting unique topics from {len(buckets)} signal types...")
    result = extract_unique_topics(buckets)

    output = {
        "source": args.buckets_json if args.source_metadata else None,
        "all_unique_topics": result["all_unique_topics"],
        "topics_by_signal_type": result["topics_by_signal_type"],
        "counts": result["counts"],
    }
    if not args.source_metadata:
        output.pop("source", None)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Total unique topics: {result['counts']['total_unique_topics']}")
    for st, count in result["counts"]["by_signal_type"].items():
        print(f"  {st}: {count}")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
