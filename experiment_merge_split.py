#!/usr/bin/env python3
"""
Merge vs split experiment: at multiple similarity thresholds, send topic batches
to the LLM and measure how often the model merges (1 cluster) vs splits (2+ clusters).
Uses parallel API calls like label_comments.py for speed.

Usage:
  python experiment_merge_split.py buckets_threads_rPckDuTaSnw_with_likes.json --out merge_split_results.json
  python experiment_merge_split.py buckets.json -o results.json --max-parallel 20
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

# Reuse consolidation pipeline (embedding, grouping, API, parsing)
from consolidate_topics import (
    build_candidate_groups,
    build_faiss_index,
    call_deepseek_consolidation,
    flatten_all_topics,
    format_llm_user_content,
    load_buckets,
    parse_consolidation_response,
    embed_topics,
    HAS_EMBED,
)

# All thresholds: lower (0.6, 0.65, 0.7) + previous sweep
EXPERIMENT_THRESHOLDS = [0.6, 0.65, 0.7, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.89, 0.90, 0.92]
MIN_GROUP_SIZE_FOR_LLM = 2
MAX_PARALLEL_DEFAULT = 40


def write_batch_log(
    log_dir: str,
    threshold: float,
    batch_index: int,
    phrases: List[str],
    user_content: str,
    raw_response: str,
    result: Dict[str, Any],
) -> None:
    """Write one log file per batch: input sent to LLM and raw output received."""
    subdir = os.path.join(log_dir, f"{threshold:.2f}")
    os.makedirs(subdir, exist_ok=True)
    path = os.path.join(subdir, f"batch_{batch_index:04d}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== META ===\n")
        f.write(f"threshold: {threshold}\n")
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


def run_batch(
    api_key: str,
    batch_index: int,
    phrases: List[str],
    threshold: float,
    log_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Call LLM for one batch; return outcome (merged/split) and cluster info."""
    user_content = format_llm_user_content(phrases)
    raw_content = ""
    try:
        response = call_deepseek_consolidation(api_key, user_content, batch_index)
        content = (response.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        raw_content = content
        if not content.strip():
            result = {
                "threshold": threshold,
                "batch_index": batch_index,
                "input_size": len(phrases),
                "num_clusters": 0,
                "outcome": "error",
                "error": "empty response",
                "clusters": [],
            }
            if log_dir:
                write_batch_log(log_dir, threshold, batch_index, phrases, user_content, raw_content, result)
            return result
        clusters = parse_consolidation_response(content)
        num_clusters = len(clusters)
        outcome = "merged" if num_clusters == 1 else "split"
        result = {
            "threshold": threshold,
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
            write_batch_log(log_dir, threshold, batch_index, phrases, user_content, raw_content, result)
        return result
    except Exception as e:
        result = {
            "threshold": threshold,
            "batch_index": batch_index,
            "input_size": len(phrases),
            "num_clusters": 0,
            "outcome": "error",
            "error": str(e),
            "clusters": [],
        }
        if log_dir:
            write_batch_log(log_dir, threshold, batch_index, phrases, user_content, raw_content, result)
        return result


def run_threshold(
    api_key: str,
    groups: List[List[str]],
    threshold: float,
    max_workers: int,
    limit: Optional[int] = None,
    log_dir: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run all batches for one threshold in parallel. Returns (summary, batch_results)."""
    batches = [(i, g) for i, g in enumerate(groups) if len(g) >= MIN_GROUP_SIZE_FOR_LLM]
    if limit is not None and limit > 0:
        batches = batches[:limit]
    if not batches:
        return (
            {
                "threshold": threshold,
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
                f"  [{threshold:.2f}] {completed}/{total_batches} batches | "
                f"merged: {merged_so_far}  split: {split_so_far}  error: {error_so_far}"
            )
            if force_newline:
                print(msg, flush=True)
            else:
                print(f"\r{msg}   ", end="", flush=True)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(run_batch, api_key, idx, phrases, threshold, log_dir): (idx, phrases)
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
                    "threshold": threshold,
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
        "threshold": threshold,
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
        description="Experiment: LLM merge vs split at multiple similarity thresholds",
    )
    parser.add_argument("buckets_json", help="Path to buckets JSON (group_signals output)")
    parser.add_argument("--out", "-o", required=True, help="Output JSON for summary + details")
    parser.add_argument("--max-parallel", type=int, default=MAX_PARALLEL_DEFAULT, help="Parallel API requests")
    parser.add_argument("--api-key", help="DeepSeek API key (default: DEEPSEEK_API_KEY)")
    parser.add_argument("--thresholds", nargs="+", type=float, default=EXPERIMENT_THRESHOLDS, help="Thresholds to test")
    parser.add_argument("--top-k", type=int, default=25, help="Neighbors per topic for grouping")
    parser.add_argument("--max-group-size", type=int, default=25, help="Max topics per batch")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Sentence transformer model")
    parser.add_argument("--no-details", action="store_true", help="Do not save per-batch details (smaller file)")
    parser.add_argument("--limit", type=int, default=None, metavar="N", help="Limit to first N batches per threshold (for quick test)")
    parser.add_argument("--log-dir", metavar="DIR", default=None, help="Log each LLM input and raw output to DIR (one file per batch, grouped by threshold)")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY not set and --api-key not provided.", file=sys.stderr)
        return 1
    if not HAS_EMBED:
        print("Error: Install sentence-transformers and faiss-cpu.", file=sys.stderr)
        return 1

    thresholds_sorted = sorted(set(args.thresholds))
    print(f"Loading buckets from {args.buckets_json}...")
    buckets = load_buckets(args.buckets_json)
    topics = flatten_all_topics(buckets)
    print(f"Flattened {len(topics)} unique topics")
    print("Embedding and building FAISS index...")
    vectors, _ = embed_topics(topics, model_name=args.embedding_model)
    index = build_faiss_index(vectors)
    print("Done.")

    summaries: List[Dict[str, Any]] = []
    all_details: Dict[str, List[Dict[str, Any]]] = {}
    n_thresholds = len(thresholds_sorted)

    for ti, th in enumerate(thresholds_sorted, 1):
        print(f"\n{'='*60}")
        print(f"Threshold {ti}/{n_thresholds}: {th} — building groups...")
        groups = build_candidate_groups(
            topics,
            vectors,
            index,
            top_k=args.top_k,
            min_similarity=th,
            max_group_size=args.max_group_size,
        )
        n_for_llm = sum(1 for g in groups if len(g) >= MIN_GROUP_SIZE_FOR_LLM)
        print(f"  Groups: {len(groups)} total, {n_for_llm} batches to send to LLM")
        if n_for_llm == 0:
            summaries.append({
                "threshold": th,
                "total_batches": 0,
                "merged": 0,
                "split": 0,
                "error": 0,
                "pct_merged": 0.0,
                "pct_split": 0.0,
                "pct_error": 0.0,
            })
            all_details[f"{th:.2f}"] = []
            continue
        if getattr(args, "log_dir", None):
            print(f"  Logging LLM I/O to {args.log_dir}/")
        print(f"  Calling API (max_workers={args.max_parallel})...")
        sys.stdout.flush()
        summary, batch_results = run_threshold(
            api_key, groups, th, args.max_parallel,
            limit=getattr(args, "limit", None),
            log_dir=getattr(args, "log_dir", None),
        )
        summaries.append(summary)
        all_details[f"{th:.2f}"] = batch_results
        print(f"  -> merged: {summary['merged']} ({summary['pct_merged']}%), split: {summary['split']} ({summary['pct_split']}%), error: {summary['error']}")

    out_data = {
        "source": args.buckets_json,
        "num_topics": len(topics),
        "top_k": args.top_k,
        "max_group_size": args.max_group_size,
        "summaries": summaries,
        "summary_table": [
            {
                "threshold": s["threshold"],
                "total_batches": s["total_batches"],
                "merged": s["merged"],
                "split": s["split"],
                "error": s["error"],
                "pct_merged": s["pct_merged"],
                "pct_split": s["pct_split"],
                "pct_error": s["pct_error"],
            }
            for s in summaries
        ],
    }
    if not args.no_details:
        out_data["details_by_threshold"] = all_details

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"\nWrote results to {args.out}")
    print("\nSummary:")
    print(f"{'Threshold':<10} {'Batches':<8} {'Merged':<8} {'Split':<8} {'Error':<6} {'%Merged':<10} {'%Split':<10}")
    print("-" * 64)
    for s in summaries:
        print(f"{s['threshold']:<10} {s['total_batches']:<8} {s['merged']:<8} {s['split']:<8} {s['error']:<6} {s['pct_merged']:<10} {s['pct_split']:<10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
