#!/usr/bin/env python3
"""
Compare merge/split rates WITH vs WITHOUT transitive closure.

For each approach:
1. Build groups (with/without TC)
2. Send batches to LLM
3. Measure merge vs split rates

This shows if transitive closure creates groups that LLM splits more (bad) or merges more (good).

Usage:
  python experiment_tc_merge_split.py buckets_threads_rPckDuTaSnw_with_likes.json --out tc_merge_split_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from consolidate_topics import (
    build_candidate_groups,
    build_candidate_groups_with_transitive_closure,
    build_faiss_index,
    call_deepseek_consolidation,
    compute_group_size_stats,
    flatten_all_topics,
    format_llm_user_content,
    load_buckets,
    parse_consolidation_response,
    embed_topics,
    HAS_EMBED,
)

MIN_GROUP_SIZE_FOR_LLM = 2
MAX_PARALLEL_DEFAULT = 40
THRESHOLD = 0.75

# Stricter system prompt - more conservative about merging
STRICT_CONSOLIDATION_SYSTEM_PROMPT = """You are consolidating topic labels.

Given a list of topic phrases that may be near-duplicates or semantically equivalent, your task is to:
1. Group them into semantically equivalent clusters ONLY if they are truly equivalent or very close paraphrases.
2. If phrases have different meanings or nuances, SPLIT them into separate clusters.
3. Assign one canonical label per cluster.
4. Return valid JSON only, no markdown or explanation.

Output format (strict JSON):
{
  "clusters": [
    {
      "canonical_label": "<single canonical topic string>",
      "members": ["<phrase1>", "<phrase2>", ...]
    }
  ]
}

Rules:
- Each input phrase must appear in exactly one cluster's "members".
- Keep canonical_label short and clear (a few words).
- BE CONSERVATIVE: Only merge phrases that are truly equivalent or very close paraphrases.
- If phrases have different meanings, contexts, or nuances, put them in separate clusters.
- If a phrase does not fit any cluster, put it in a cluster of one with itself as canonical_label.
- Preserve the exact input phrase strings in "members" (copy-paste from the input list).
- When in doubt, SPLIT rather than merge.
"""


def call_deepseek_strict(
    api_key: str,
    user_content: str,
    batch_index: int,
    retry: int = 0,
) -> Dict[str, Any]:
    """Call DeepSeek API with strict prompt."""
    import requests
    from consolidate_topics import DEEPSEEK_API_URL, DEEPSEEK_MODEL, MAX_RETRIES, RETRY_BASE_DELAY, REQUEST_TIMEOUT
    import time
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": STRICT_CONSOLIDATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "max_tokens": 4096,
    }
    try:
        resp = requests.post(
            DEEPSEEK_API_URL,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(data["error"].get("message", "Unknown error"))
        return data
    except (requests.exceptions.RequestException, RuntimeError) as e:
        if retry < MAX_RETRIES:
            time.sleep(RETRY_BASE_DELAY * (2 ** retry))
            return call_deepseek_strict(api_key, user_content, batch_index, retry + 1)
        raise


def run_batch(
    api_key: str,
    batch_index: int,
    phrases: List[str],
    method: str,  # "no_tc" or "with_tc"
    log_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Call LLM for one batch; return outcome (merged/split) and cluster info."""
    user_content = format_llm_user_content(phrases)
    raw_content = ""
    try:
        response = call_deepseek_strict(api_key, user_content, batch_index)
        content = (response.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        raw_content = content
        if not content.strip():
            result = {
                "method": method,
                "batch_index": batch_index,
                "input_size": len(phrases),
                "num_clusters": 0,
                "outcome": "error",
                "error": "empty response",
                "clusters": [],
            }
            if log_dir:
                write_batch_log(log_dir, method, batch_index, phrases, user_content, raw_content, result)
            return result
        clusters = parse_consolidation_response(content)
        num_clusters = len(clusters)
        outcome = "merged" if num_clusters == 1 else "split"
        result = {
            "method": method,
            "batch_index": batch_index,
            "input_size": len(phrases),
            "num_clusters": num_clusters,
            "outcome": outcome,
            "clusters": [
                {"canonical_label": c, "members": m, "size": len(m)}
                for c, m in clusters
            ],
        }
        if log_dir:
            write_batch_log(log_dir, method, batch_index, phrases, user_content, raw_content, result)
        return result
    except Exception as e:
        result = {
            "method": method,
            "batch_index": batch_index,
            "input_size": len(phrases),
            "num_clusters": 0,
            "outcome": "error",
            "error": str(e),
            "clusters": [],
        }
        if log_dir:
            write_batch_log(log_dir, method, batch_index, phrases, user_content, raw_content, result)
        return result


def write_batch_log(
    log_dir: str,
    method: str,
    batch_index: int,
    phrases: List[str],
    user_content: str,
    raw_response: str,
    result: Dict[str, Any],
) -> None:
    """Write one log file per batch."""
    subdir = os.path.join(log_dir, method)
    os.makedirs(subdir, exist_ok=True)
    path = os.path.join(subdir, f"batch_{batch_index:04d}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== META ===\n")
        f.write(f"method: {method}\n")
        f.write(f"batch_index: {batch_index}\n")
        f.write(f"input_size: {len(phrases)}\n")
        f.write(f"outcome: {result.get('outcome', '?')}\n")
        f.write(f"num_clusters: {result.get('num_clusters', 0)}\n")
        if result.get("error"):
            f.write(f"error: {result['error']}\n")
        f.write("\n=== INPUT (user message sent to LLM) ===\n")
        f.write(user_content)
        f.write("\n\n=== RAW LLM RESPONSE ===\n")
        f.write(raw_response if raw_response else "(empty)")
        f.write("\n\n=== PARSED ===\n")
        f.write(json.dumps({"outcome": result.get("outcome"), "num_clusters": result.get("num_clusters"), "clusters": result.get("clusters", [])}, indent=2))
        f.write("\n")


def run_threshold(
    api_key: str,
    groups: List[List[str]],
    method: str,
    max_workers: int,
    log_dir: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run all batches for one method in parallel."""
    batches = [(i, g) for i, g in enumerate(groups) if len(g) >= MIN_GROUP_SIZE_FOR_LLM]
    if not batches:
        return (
            {
                "method": method,
                "total_batches": 0,
                "merged": 0,
                "split": 0,
                "error": 0,
                "pct_merged": 0.0,
                "pct_split": 0.0,
                "pct_error": 0.0,
            },
            [],
        )

    results: List[Dict[str, Any]] = []
    total_batches = len(batches)
    completed = 0
    merged_so_far = 0
    split_so_far = 0
    error_so_far = 0
    progress_lock = threading.Lock()

    def print_progress(force_newline: bool = False) -> None:
        with progress_lock:
            msg = (
                f"  [{method}] {completed}/{total_batches} batches | "
                f"merged: {merged_so_far}  split: {split_so_far}  error: {error_so_far}"
            )
            if force_newline:
                print(msg, flush=True)
            else:
                print(f"\r{msg}   ", end="", flush=True)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(run_batch, api_key, idx, phrases, method, log_dir): (idx, phrases)
            for idx, phrases in batches
        }
        for future in as_completed(futures):
            try:
                r = future.result()
                results.append(r)
                completed += 1
                if r.get("outcome") == "merged":
                    merged_so_far += 1
                elif r.get("outcome") == "split":
                    split_so_far += 1
                else:
                    error_so_far += 1
                print_progress()
            except Exception as e:
                idx, phrases = futures[future]
                results.append({
                    "method": method,
                    "batch_index": idx,
                    "input_size": len(phrases),
                    "num_clusters": 0,
                    "outcome": "error",
                    "error": str(e),
                    "clusters": [],
                })
                completed += 1
                error_so_far += 1
                print_progress()
    print_progress(force_newline=True)

    merged = sum(1 for r in results if r.get("outcome") == "merged")
    split = sum(1 for r in results if r.get("outcome") == "split")
    err = sum(1 for r in results if r.get("outcome") == "error")
    total = len(results)
    summary = {
        "method": method,
        "total_batches": total,
        "merged": merged,
        "split": split,
        "error": err,
        "pct_merged": round(100.0 * merged / total, 2) if total else 0.0,
        "pct_split": round(100.0 * split / total, 2) if total else 0.0,
        "pct_error": round(100.0 * err / total, 2) if total else 0.0,
    }
    return summary, results


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Compare merge/split rates WITH vs WITHOUT transitive closure",
    )
    parser.add_argument("buckets_json", help="Path to buckets JSON")
    parser.add_argument("--out", "-o", required=True, help="Output JSON")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help=f"Similarity threshold (default {THRESHOLD})")
    parser.add_argument("--top-k", type=int, default=25, help="Neighbors per topic")
    parser.add_argument("--max-path-length", type=int, default=3, help="Max path length for transitive closure")
    parser.add_argument("--max-parallel", type=int, default=MAX_PARALLEL_DEFAULT, help="Parallel API requests")
    parser.add_argument("--api-key", help="DeepSeek API key (default: DEEPSEEK_API_KEY)")
    parser.add_argument("--log-dir", metavar="DIR", default=None, help="Log LLM I/O to DIR")
    parser.add_argument("--no-details", action="store_true", help="Do not save per-batch details")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY not set and --api-key not provided.", file=sys.stderr)
        return 1
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

    # WITHOUT transitive closure
    print("\n1. Building groups WITHOUT transitive closure...")
    groups_no_tc = build_candidate_groups(
        topics,
        vectors,
        index,
        top_k=args.top_k,
        min_similarity=args.threshold,
    )
    stats_no_tc = compute_group_size_stats(groups_no_tc)
    n_for_llm_no_tc = stats_no_tc["multi_topic"]
    print(f"   Groups: {stats_no_tc['total_groups']} total ({stats_no_tc['singletons']} singletons, {n_for_llm_no_tc} for LLM)")
    print(f"   Size: min={stats_no_tc['min_size']}, max={stats_no_tc['max_size']}, avg={stats_no_tc['avg_size']:.2f}")

    if n_for_llm_no_tc == 0:
        print("   No batches to send to LLM. Exiting.")
        return 0

    print(f"\n2. Calling LLM for WITHOUT transitive closure batches (max_workers={args.max_parallel})...")
    if args.log_dir:
        print(f"   Logging to {args.log_dir}/no_tc/")
    summary_no_tc, results_no_tc = run_threshold(
        api_key, groups_no_tc, "no_tc", args.max_parallel, log_dir=args.log_dir
    )

    # WITH transitive closure
    print("\n3. Building groups WITH transitive closure (max_path_length={})...".format(args.max_path_length))
    groups_with_tc = build_candidate_groups_with_transitive_closure(
        topics,
        vectors,
        index,
        top_k=args.top_k,
        min_similarity=args.threshold,
        max_path_length=args.max_path_length,
    )
    stats_with_tc = compute_group_size_stats(groups_with_tc)
    n_for_llm_with_tc = stats_with_tc["multi_topic"]
    print(f"   Groups: {stats_with_tc['total_groups']} total ({stats_with_tc['singletons']} singletons, {n_for_llm_with_tc} for LLM)")
    print(f"   Size: min={stats_with_tc['min_size']}, max={stats_with_tc['max_size']}, avg={stats_with_tc['avg_size']:.2f}")

    if n_for_llm_with_tc == 0:
        print("   No batches to send to LLM.")
        summary_with_tc = {
            "method": "with_tc",
            "total_batches": 0,
            "merged": 0,
            "split": 0,
            "error": 0,
            "pct_merged": 0.0,
            "pct_split": 0.0,
            "pct_error": 0.0,
        }
        results_with_tc = []
    else:
        print(f"\n4. Calling LLM for WITH transitive closure batches (max_workers={args.max_parallel})...")
        if args.log_dir:
            print(f"   Logging to {args.log_dir}/with_tc/")
        summary_with_tc, results_with_tc = run_threshold(
            api_key, groups_with_tc, "with_tc", args.max_parallel, log_dir=args.log_dir
        )

    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON:")
    print(f"{'Method':<20} {'Batches':<10} {'Merged':<10} {'Split':<10} {'%Merged':<12} {'%Split':<12}")
    print("-" * 60)
    print(f"{'NO transitive closure':<20} {summary_no_tc['total_batches']:<10} {summary_no_tc['merged']:<10} {summary_no_tc['split']:<10} {summary_no_tc['pct_merged']:<12} {summary_no_tc['pct_split']:<12}")
    print(f"{'WITH transitive closure':<20} {summary_with_tc['total_batches']:<10} {summary_with_tc['merged']:<10} {summary_with_tc['split']:<10} {summary_with_tc['pct_merged']:<12} {summary_with_tc['pct_split']:<12}")
    
    diff_merged = summary_with_tc['pct_merged'] - summary_no_tc['pct_merged']
    diff_split = summary_with_tc['pct_split'] - summary_no_tc['pct_split']
    print(f"\nDifference: %Merged {diff_merged:+.2f}%, %Split {diff_split:+.2f}%")
    
    if diff_split > 0:
        print("⚠️  Transitive closure INCREASES split rate → creates groups that LLM splits more (bad groupings)")
    elif diff_merged > 0:
        print("✓ Transitive closure INCREASES merge rate → creates groups that LLM merges more (good groupings)")
    else:
        print("→ Transitive closure has minimal effect on merge/split rates")

    out_data = {
        "source": args.buckets_json,
        "threshold": args.threshold,
        "top_k": args.top_k,
        "max_path_length": args.max_path_length,
        "num_topics": len(topics),
        "group_stats": {
            "without_tc": stats_no_tc,
            "with_tc": stats_with_tc,
        },
        "llm_results": {
            "without_tc": summary_no_tc,
            "with_tc": summary_with_tc,
        },
        "comparison": {
            "pct_merged_diff": diff_merged,
            "pct_split_diff": diff_split,
            "interpretation": "TC increases split rate (bad)" if diff_split > 0 else ("TC increases merge rate (good)" if diff_merged > 0 else "TC has minimal effect"),
        },
    }
    if not args.no_details:
        out_data["batch_results"] = {
            "without_tc": results_no_tc,
            "with_tc": results_with_tc,
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"\nWrote results to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
