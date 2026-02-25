#!/usr/bin/env python3
"""
Apply canonical topic map to buckets (batching.md Step 7).

Takes the original buckets and a canonical map, then creates revised buckets
where each original topic is replaced with its canonical topic and stats are aggregated.

Usage:
  python apply_canonical_map.py buckets.json canonical_map.json --out revised_buckets.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Any, Dict, List, Set

from consolidate_topics import load_buckets


def load_canonical_map(path: str) -> Dict[str, str]:
    """Load canonical map from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle different formats:
    # 1. {"canonical_map": {...}}
    # 2. Direct mapping {...}
    if "canonical_map" in data:
        return data["canonical_map"]
    return data


def extract_canonical_map_from_batch_results(batch_results: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Extract canonical map from experiment batch results.
    
    Batch results format:
      [
        {
          "clusters": [
            {"canonical_label": "...", "members": ["...", ...]},
            ...
          ],
          ...
        },
        ...
      ]
    """
    canonical_map: Dict[str, str] = {}
    
    for batch in batch_results:
        clusters = batch.get("clusters", [])
        for cluster in clusters:
            canonical_label = cluster.get("canonical_label", "")
            members = cluster.get("members", [])
            if not canonical_label or not members:
                continue
            # Map each member to its canonical label
            for member in members:
                if isinstance(member, str) and member.strip():
                    canonical_map[member.strip()] = canonical_label.strip()
    
    return canonical_map


def apply_canonical_map_to_buckets(
    buckets: Dict[str, Dict[str, Dict[str, Any]]],
    canonical_map: Dict[str, str],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Apply canonical map to buckets. Update topic names to canonical and keep thread_ids from all merged topics.
    
    Input buckets structure:
      buckets[signal_type][topic] -> {
        "signal_count": int,
        "thread_count": int,
        "thread_ids": [str, ...]
      }
    
    Output revised buckets:
      revised_buckets[signal_type][canonical_topic] -> {
        "signal_count": int,  # aggregated sum
        "thread_count": int,   # count of unique thread_ids (after concatenating all)
        "thread_ids": [str, ...]  # thread_ids from ALL original topics that map to this canonical topic
      }
    
    Note: When multiple original topics map to the same canonical topic, we:
    - Aggregate signal_count (sum)
    - Concatenate thread_ids from all original topics (keeping all, including duplicates)
    - Compute thread_count as unique count of the concatenated thread_ids
    """
    revised_buckets: Dict[str, Dict[str, Dict[str, Any]]] = {}
    
    for signal_type, topic_map in buckets.items():
        if not isinstance(topic_map, dict):
            continue
        
        # Track canonical topics and their aggregates
        canonical_aggregates: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "signal_count": 0,
            "thread_ids": [],  # Keep thread_ids from ALL original topics that map to this canonical
            "thread_count": 0,
        })
        
        for original_topic, stats in topic_map.items():
            canonical_topic = canonical_map.get(original_topic, original_topic)
            
            # Aggregate signal_count
            canonical_aggregates[canonical_topic]["signal_count"] += stats.get("signal_count", 0)
            
            # Append thread_ids from this original topic (keep all, don't deduplicate yet)
            thread_ids = stats.get("thread_ids", [])
            if isinstance(thread_ids, list):
                canonical_aggregates[canonical_topic]["thread_ids"].extend(thread_ids)
            elif isinstance(thread_ids, set):
                canonical_aggregates[canonical_topic]["thread_ids"].extend(sorted(thread_ids))
        
        # Convert to final structure: keep all thread_ids, compute unique count
        revised_buckets[signal_type] = {}
        for canonical_topic, agg in canonical_aggregates.items():
            # Keep all thread_ids (including duplicates if any)
            thread_ids_list = agg["thread_ids"]
            # Compute unique count for thread_count
            unique_thread_ids = set(thread_ids_list) if thread_ids_list else set()
            revised_buckets[signal_type][canonical_topic] = {
                "signal_count": agg["signal_count"],
                "thread_count": len(unique_thread_ids),
                "thread_ids": thread_ids_list,  # Keep all thread_ids, including duplicates
            }
    
    return revised_buckets


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply canonical topic map to buckets (batching.md Step 7)"
    )
    parser.add_argument("buckets_json", help="Path to original buckets JSON")
    parser.add_argument(
        "canonical_map_json",
        help="Path to canonical map JSON (or experiment results JSON with batch results)",
    )
    parser.add_argument("--out", "-o", required=True, help="Output path for revised buckets JSON")
    parser.add_argument(
        "--preserve-metadata",
        action="store_true",
        help="Preserve metadata fields (file, total_threads, etc.) from original buckets",
    )
    parser.add_argument(
        "--from-batch-results",
        action="store_true",
        help="Extract canonical map from experiment batch results (looks for 'batch_results' or 'details' key)",
    )
    parser.add_argument(
        "--method",
        choices=["no_tc", "with_tc"],
        help="If extracting from batch results, which method to use (for tc_merge_split_results.json)",
    )
    
    args = parser.parse_args()
    
    print(f"Loading buckets from {args.buckets_json}...")
    # Load full buckets file (may have metadata)
    with open(args.buckets_json, "r", encoding="utf-8") as f:
        buckets_data = json.load(f)
    
    buckets = load_buckets(args.buckets_json)
    print(f"Loaded {len(buckets)} signal types")
    
    # Count original topics
    original_topic_count = sum(len(topic_map) for topic_map in buckets.values())
    print(f"Original topics: {original_topic_count}")
    
    print(f"Loading canonical map from {args.canonical_map_json}...")
    if args.from_batch_results:
        # Extract from experiment batch results
        with open(args.canonical_map_json, "r", encoding="utf-8") as f:
            exp_data = json.load(f)
        
        # Handle different experiment result formats
        batch_results = None
        if "batch_results" in exp_data:
            batch_results_data = exp_data["batch_results"]
            # Check if batch_results is a dict with method keys (e.g., {"without_tc": [...], "with_tc": [...]})
            if isinstance(batch_results_data, dict):
                if args.method:
                    # Map method names: "no_tc" -> "without_tc", "with_tc" -> "with_tc"
                    method_key = "without_tc" if args.method == "no_tc" else args.method
                    if method_key in batch_results_data:
                        batch_results = batch_results_data[method_key]
                    else:
                        # Try direct method name
                        if args.method in batch_results_data:
                            batch_results = batch_results_data[args.method]
                        else:
                            # Fallback: use first available list
                            for key, value in batch_results_data.items():
                                if isinstance(value, list):
                                    batch_results = value
                                    print(f"  Using batch_results['{key}'] for batch results")
                                    break
                else:
                    # No method specified, use first available list
                    for key, value in batch_results_data.items():
                        if isinstance(value, list):
                            batch_results = value
                            print(f"  Using batch_results['{key}'] for batch results")
                            break
            else:
                # batch_results is directly a list
                batch_results = batch_results_data
        elif "details" in exp_data:
            batch_results = exp_data["details"]
        elif args.method and f"{args.method}_results" in exp_data:
            batch_results = exp_data[f"{args.method}_results"]
        elif "results" in exp_data:
            # Try to find results for the specified method
            if args.method:
                results_key = f"{args.method}_results"
                if results_key in exp_data:
                    batch_results = exp_data[results_key]
                else:
                    # Fallback: use first available results
                    for key in exp_data.keys():
                        if key.endswith("_results") and isinstance(exp_data[key], list):
                            batch_results = exp_data[key]
                            print(f"  Using {key} for batch results")
                            break
            else:
                batch_results = exp_data["results"]
        
        if batch_results is None or not isinstance(batch_results, list):
            print(
                f"Error: Could not find batch results in {args.canonical_map_json}",
                file=sys.stderr,
            )
            print("  Expected keys: 'batch_results', 'details', 'results', or '{method}_results'", file=sys.stderr)
            return 1
        
        canonical_map = extract_canonical_map_from_batch_results(batch_results)
        print(f"Extracted {len(canonical_map)} canonical mappings from {len(batch_results)} batches")
    else:
        canonical_map = load_canonical_map(args.canonical_map_json)
        print(f"Loaded {len(canonical_map)} canonical mappings")
    
    print("Applying canonical map to buckets...")
    revised_buckets = apply_canonical_map_to_buckets(buckets, canonical_map)
    
    # Count revised topics
    revised_topic_count = sum(len(topic_map) for topic_map in revised_buckets.values())
    print(f"Revised topics: {revised_topic_count}")
    print(f"Reduction: {original_topic_count} -> {revised_topic_count} ({100.0 * (1 - revised_topic_count / original_topic_count):.1f}% reduction)")
    
    # Prepare output
    if args.preserve_metadata and isinstance(buckets_data, dict):
        # Preserve metadata fields
        output_data = {
            **{k: v for k, v in buckets_data.items() if k != "buckets"},
            "buckets": revised_buckets,
            "canonical_map_source": args.canonical_map_json,
            "original_topic_count": original_topic_count,
            "revised_topic_count": revised_topic_count,
        }
    else:
        output_data = {
            "source": args.buckets_json,
            "canonical_map_source": args.canonical_map_json,
            "original_topic_count": original_topic_count,
            "revised_topic_count": revised_topic_count,
            "buckets": revised_buckets,
        }
    
    print(f"Writing revised buckets to {args.out}...")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    
    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
