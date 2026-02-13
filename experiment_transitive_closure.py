#!/usr/bin/env python3
"""
Compare grouping WITH vs WITHOUT transitive closure at threshold 0.75.

Shows how transitive closure affects:
- Number of groups
- Group size distribution
- Examples of groups that differ

Usage:
  python experiment_transitive_closure.py buckets_threads_rPckDuTaSnw_with_likes.json --out transitive_comparison.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Set

from dotenv import load_dotenv

from consolidate_topics import (
    build_candidate_groups,
    build_candidate_groups_with_transitive_closure,
    build_faiss_index,
    compute_group_size_stats,
    flatten_all_topics,
    load_buckets,
    embed_topics,
    HAS_EMBED,
)

THRESHOLD = 0.75
TOP_K = 25
MAX_PATH_LENGTH = 3  # For transitive closure


def groups_to_sets(groups: List[List[str]]) -> List[Set[str]]:
    """Convert groups to sets for easier comparison."""
    return [set(g) for g in groups]


def find_group_containing(topic: str, groups: List[Set[str]]) -> int:
    """Find index of group containing topic, or -1."""
    for i, g in enumerate(groups):
        if topic in g:
            return i
    return -1


def compare_groupings(
    groups_no_tc: List[List[str]],
    groups_with_tc: List[List[str]],
) -> Dict[str, Any]:
    """Compare two grouping results and return statistics."""
    sets_no_tc = groups_to_sets(groups_no_tc)
    sets_with_tc = groups_to_sets(groups_with_tc)
    
    stats_no_tc = compute_group_size_stats(groups_no_tc)
    stats_with_tc = compute_group_size_stats(groups_with_tc)
    
    # Find groups that merged (multiple no-TC groups → one with-TC group)
    merged_examples: List[Dict[str, Any]] = []
    used_no_tc = set()
    
    for tc_group in sets_with_tc:
        if len(tc_group) < 2:
            continue
        # Find which no-TC groups contributed to this TC group
        contributing_no_tc = []
        for i, no_tc_group in enumerate(sets_no_tc):
            if i in used_no_tc:
                continue
            if no_tc_group & tc_group:  # Intersection
                contributing_no_tc.append(i)
                used_no_tc.add(i)
        
        if len(contributing_no_tc) > 1:
            merged_examples.append({
                "tc_group_size": len(tc_group),
                "merged_from": len(contributing_no_tc),
                "tc_group": sorted(list(tc_group)),
                "no_tc_groups": [
                    sorted(list(sets_no_tc[i])) for i in contributing_no_tc
                ],
            })
            if len(merged_examples) >= 10:  # Limit examples
                break
    
    # Find groups that are larger in TC version
    size_increase_examples: List[Dict[str, Any]] = []
    topic_to_no_tc_size: Dict[str, int] = {}
    for g in sets_no_tc:
        size = len(g)
        for topic in g:
            topic_to_no_tc_size[topic] = size
    
    for tc_group in sets_with_tc:
        if len(tc_group) < 2:
            continue
        # Check if any topic in this group had a smaller group in no-TC
        for topic in tc_group:
            no_tc_size = topic_to_no_tc_size.get(topic, 1)
            if len(tc_group) > no_tc_size:
                size_increase_examples.append({
                    "topic": topic,
                    "no_tc_group_size": no_tc_size,
                    "tc_group_size": len(tc_group),
                    "tc_group": sorted(list(tc_group)),
                })
                break
        if len(size_increase_examples) >= 10:
            break
    
    return {
        "without_transitive_closure": stats_no_tc,
        "with_transitive_closure": stats_with_tc,
        "comparison": {
            "total_groups_diff": stats_with_tc["total_groups"] - stats_no_tc["total_groups"],
            "singletons_diff": stats_with_tc["singletons"] - stats_no_tc["singletons"],
            "multi_topic_diff": stats_with_tc["multi_topic"] - stats_no_tc["multi_topic"],
            "max_size_no_tc": stats_no_tc["max_size"],
            "max_size_with_tc": stats_with_tc["max_size"],
            "avg_size_no_tc": stats_no_tc["avg_size"],
            "avg_size_with_tc": stats_with_tc["avg_size"],
        },
        "merged_examples": merged_examples,
        "size_increase_examples": size_increase_examples,
    }


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Compare grouping WITH vs WITHOUT transitive closure at threshold 0.75",
    )
    parser.add_argument("buckets_json", help="Path to buckets JSON")
    parser.add_argument("--out", "-o", required=True, help="Output JSON")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help=f"Similarity threshold (default {THRESHOLD})")
    parser.add_argument("--top-k", type=int, default=TOP_K, help=f"Neighbors per topic (default {TOP_K})")
    parser.add_argument("--max-path-length", type=int, default=MAX_PATH_LENGTH, help=f"Max path length for transitive closure (default {MAX_PATH_LENGTH})")
    args = parser.parse_args()

    if not HAS_EMBED:
        print("Error: Install sentence-transformers and faiss-cpu.", file=sys.stderr)
        return 1

    print(f"Loading buckets from {args.buckets_json}...")
    buckets = load_buckets(args.buckets_json)
    topics = flatten_all_topics(buckets)
    print(f"Flattened {len(topics)} unique topics")
    
    print("Embedding and building FAISS index...")
    vectors, _ = embed_topics(topics)
    index = build_faiss_index(vectors)
    print("Done.\n")

    print(f"Threshold: {args.threshold}, top_k: {args.top_k}")
    print("=" * 60)
    
    print("\n1. Building groups WITHOUT transitive closure...")
    groups_no_tc = build_candidate_groups(
        topics,
        vectors,
        index,
        top_k=args.top_k,
        min_similarity=args.threshold,
    )
    stats_no_tc = compute_group_size_stats(groups_no_tc)
    print(f"   Groups: {stats_no_tc['total_groups']} total ({stats_no_tc['singletons']} singletons, {stats_no_tc['multi_topic']} multi-topic)")
    print(f"   Size: min={stats_no_tc['min_size']}, max={stats_no_tc['max_size']}, avg={stats_no_tc['avg_size']:.2f}, median={stats_no_tc['median_size']}")
    
    print("\n2. Building groups WITH transitive closure (max_path_length={})...".format(args.max_path_length))
    groups_with_tc = build_candidate_groups_with_transitive_closure(
        topics,
        vectors,
        index,
        top_k=args.top_k,
        min_similarity=args.threshold,
        max_path_length=args.max_path_length,
    )
    stats_with_tc = compute_group_size_stats(groups_with_tc)
    print(f"   Groups: {stats_with_tc['total_groups']} total ({stats_with_tc['singletons']} singletons, {stats_with_tc['multi_topic']} multi-topic)")
    print(f"   Size: min={stats_with_tc['min_size']}, max={stats_with_tc['max_size']}, avg={stats_with_tc['avg_size']:.2f}, median={stats_with_tc['median_size']}")
    
    print("\n3. Comparing...")
    comparison = compare_groupings(groups_no_tc, groups_with_tc)
    
    print(f"\n   Difference:")
    print(f"   - Total groups: {comparison['comparison']['total_groups_diff']:+d}")
    print(f"   - Singletons: {comparison['comparison']['singletons_diff']:+d}")
    print(f"   - Multi-topic: {comparison['comparison']['multi_topic_diff']:+d}")
    print(f"   - Max size: {comparison['comparison']['max_size_no_tc']} → {comparison['comparison']['max_size_with_tc']}")
    print(f"   - Avg size: {comparison['comparison']['avg_size_no_tc']:.2f} → {comparison['comparison']['avg_size_with_tc']:.2f}")
    
    if comparison["merged_examples"]:
        print(f"\n   Found {len(comparison['merged_examples'])} examples of groups that merged:")
        for i, ex in enumerate(comparison["merged_examples"][:3], 1):
            print(f"   Example {i}: {ex['merged_from']} groups → 1 group (size {ex['tc_group_size']})")
            print(f"      TC group: {ex['tc_group'][:5]}{'...' if len(ex['tc_group']) > 5 else ''}")
    
    if comparison["size_increase_examples"]:
        print(f"\n   Found {len(comparison['size_increase_examples'])} examples of groups that grew:")
        for i, ex in enumerate(comparison["size_increase_examples"][:3], 1):
            print(f"   Example {i}: '{ex['topic'][:40]}...'")
            print(f"      Size: {ex['no_tc_group_size']} → {ex['tc_group_size']}")
    
    out_data = {
        "source": args.buckets_json,
        "threshold": args.threshold,
        "top_k": args.top_k,
        "max_path_length": args.max_path_length,
        "num_topics": len(topics),
        "comparison": comparison,
        "groups_without_tc": groups_no_tc,
        "groups_with_tc": groups_with_tc,
    }
    
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"\nWrote comparison to {args.out}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
