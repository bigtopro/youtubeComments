#!/usr/bin/env python
"""
Consistency experiment for the comment labeling pipeline.
---------------------------------------------------------
Runs the labeler multiple times on the same input (e.g. test.txt) and measures:
1. Token throughput (tokens/sec) and timing per run
2. Output consistency: how similar are the labels for the same thread across runs?

Usage:
  python experiment_consistency.py --threads-file test.txt --runs 3 --out-dir experiment_runs
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv


def load_labeled_output(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load a labeled output JSON and return thread_id -> list of signals."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for t in data.get("threads", []):
        tid = t.get("thread_id")
        if tid is not None:
            out[tid] = t.get("signals") or []
    return out


def signal_fingerprint(s: Dict[str, Any]) -> Tuple[Any, ...]:
    """Canonical fingerprint for one signal (for equality / set membership)."""
    st = s.get("signal_type") or ""
    topics = tuple(sorted((s.get("topics") or [])))
    emotion = s.get("emotion") or ""
    return (st, topics, emotion)


def thread_fingerprints(signals: List[Dict[str, Any]]) -> frozenset:
    """Set of fingerprints for a thread's signals."""
    return frozenset(signal_fingerprint(s) for s in signals)


def jaccard_sets(a: frozenset, b: frozenset) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def consistency_two_runs(
    run_a: Dict[str, List[Dict[str, Any]]],
    run_b: Dict[str, List[Dict[str, Any]]],
    thread_ids: Optional[List[str]] = None,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Compare two runs. Returns:
    - mean_jaccard: mean over threads of Jaccard(signals_a, signals_b)
    - exact_match_ratio: fraction of threads with identical signal sets
    - per_thread_jaccard: thread_id -> jaccard for that thread
    """
    if thread_ids is None:
        thread_ids = sorted(set(run_a.keys()) | set(run_b.keys()))
    per_thread: Dict[str, float] = {}
    exact = 0
    for tid in thread_ids:
        sigs_a = run_a.get(tid) or []
        sigs_b = run_b.get(tid) or []
        fp_a = thread_fingerprints(sigs_a)
        fp_b = thread_fingerprints(sigs_b)
        j = jaccard_sets(fp_a, fp_b)
        per_thread[tid] = j
        if fp_a == fp_b:
            exact += 1
    n = len(thread_ids) or 1
    mean_j = sum(per_thread.values()) / n
    exact_ratio = exact / n
    return mean_j, exact_ratio, per_thread


def run_labeler(
    threads_file: str,
    output_file: str,
    stats_file: str,
    video_description: Optional[str],
    llm_log_dir: Optional[str],
    max_parallel: int,
) -> Tuple[bool, float, Optional[Dict[str, Any]]]:
    """
    Run label_comments.py. Returns (success, wall_seconds, stats_dict).
    """
    cmd = [
        sys.executable,
        "label_comments.py",
        "--threads-file", threads_file,
        "--output", output_file,
        "--stats-file", stats_file,
        "--max-parallel", str(max_parallel),
    ]
    if video_description:
        cmd.extend(["--video-description", video_description])
    if llm_log_dir:
        cmd.extend(["--llm-log-dir", llm_log_dir])

    start = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
            capture_output=True,
            text=True,
            timeout=3600,
        )
        elapsed = time.perf_counter() - start
        if result.returncode != 0:
            print(result.stderr or result.stdout, file=sys.stderr)
            return False, elapsed, None
        stats = None
        if os.path.exists(stats_file):
            with open(stats_file, "r", encoding="utf-8") as f:
                stats = json.load(f)
        return True, elapsed, stats
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        return False, elapsed, None
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"Run failed: {e}", file=sys.stderr)
        return False, elapsed, None


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Run labeling pipeline multiple times and measure consistency and token throughput."
    )
    parser.add_argument("--threads-file", default="test.txt", help="Input threads file")
    parser.add_argument("--runs", type=int, default=3, help="Number of labeling runs (default 3)")
    parser.add_argument("--out-dir", default="experiment_runs", help="Output directory for run outputs and report")
    parser.add_argument("--video-description", default=None, help="Optional video description")
    parser.add_argument("--llm-log-dir", default=None, help="Optional LLM log dir (only for first run to avoid clutter)")
    parser.add_argument("--max-parallel", type=int, default=5, help="Max parallel API requests")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Run labeler N times with fresh output each time (no idempotency reuse)
    run_outputs: List[str] = []
    run_stats: List[Dict[str, Any]] = []
    run_elapsed: List[float] = []

    video_desc = args.video_description
    if not video_desc and os.path.exists(args.threads_file):
        # Default description for the fantasy video
        video_desc = (
            "This video discusses how modern fantasy shows like Game of Thrones and The Witcher "
            "have adopted a nihilistic tone, losing the heroic and optimistic spirit traditionally "
            "associated with the genre. The speaker suggests that newer shows like A Knight of the "
            "Seven Kingdoms and One Piece are bringing back the sense of adventure and clear moral compass."
        )

    for i in range(args.runs):
        output_file = os.path.join(args.out_dir, f"run_{i+1:03d}.json")
        stats_file = os.path.join(args.out_dir, f"run_{i+1:03d}_stats.json")
        # Remove previous output so this run is not skipped by idempotency
        if os.path.exists(output_file):
            os.remove(output_file)
        if os.path.exists(output_file + ".progress.json"):
            os.remove(output_file + ".progress.json")

        llm_log = (args.llm_log_dir if i == 0 else None)  # Log only first run
        print(f"\n--- Run {i+1}/{args.runs} ---")
        ok, elapsed, stats = run_labeler(
            args.threads_file,
            output_file,
            stats_file,
            video_desc,
            llm_log,
            args.max_parallel,
        )
        run_elapsed.append(elapsed)
        if stats:
            run_stats.append(stats)
            print(
                f"  tokens: {stats.get('total_tokens', 0):,} | "
                f"elapsed: {stats.get('elapsed_sec', 0):.1f}s | "
                f"throughput: {stats.get('tokens_per_sec') or 0:.1f} tokens/s"
            )
        if not ok:
            print(f"  Run {i+1} failed.", file=sys.stderr)
            return 1
        run_outputs.append(output_file)

    # Load all run outputs
    runs_data: List[Dict[str, List[Dict[str, Any]]]] = []
    for path in run_outputs:
        if not os.path.exists(path):
            print(f"Missing output: {path}", file=sys.stderr)
            return 1
        runs_data.append(load_labeled_output(path))

    # Common thread IDs (all runs should have same set for same input)
    thread_ids = sorted(runs_data[0].keys()) if runs_data else []
    n_threads = len(thread_ids)

    # Pairwise consistency
    pairwise_jaccard: List[float] = []
    pairwise_exact: List[float] = []
    for i in range(len(runs_data)):
        for j in range(i + 1, len(runs_data)):
            mean_j, exact_ratio, _ = consistency_two_runs(runs_data[i], runs_data[j], thread_ids)
            pairwise_jaccard.append(mean_j)
            pairwise_exact.append(exact_ratio)

    avg_jaccard = sum(pairwise_jaccard) / len(pairwise_jaccard) if pairwise_jaccard else 0.0
    avg_exact = sum(pairwise_exact) / len(pairwise_exact) if pairwise_exact else 0.0

    # Report
    report = {
        "threads_file": args.threads_file,
        "num_runs": args.runs,
        "num_threads": n_threads,
        "consistency": {
            "mean_pairwise_jaccard": round(avg_jaccard, 4),
            "mean_pairwise_exact_match_ratio": round(avg_exact, 4),
            "description": "Jaccard = similarity of signal sets per thread; exact = fraction of threads with identical signals across run pair.",
        },
        "runs": [
            {
                "output": run_outputs[i],
                "elapsed_sec": round(run_elapsed[i], 2),
                "stats": run_stats[i] if i < len(run_stats) else None,
            }
            for i in range(args.runs)
        ],
        "throughput_summary": {
            "tokens_per_sec_min": min((s.get("tokens_per_sec") or 0) for s in run_stats) if run_stats else None,
            "tokens_per_sec_max": max((s.get("tokens_per_sec") or 0) for s in run_stats) if run_stats else None,
            "tokens_per_sec_mean": (
                sum(s.get("tokens_per_sec") or 0 for s in run_stats) / len(run_stats) if run_stats else None
            ),
        },
    }

    report_path = os.path.join(args.out_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("EXPERIMENT REPORT")
    print("=" * 60)
    print(f"Runs: {args.runs}  |  Threads: {n_threads}")
    print(f"Consistency (mean pairwise Jaccard): {avg_jaccard:.4f}")
    print(f"Consistency (mean exact-match ratio): {avg_exact:.4f}")
    if report["throughput_summary"]["tokens_per_sec_mean"] is not None:
        tps = report["throughput_summary"]
        print(f"Throughput: {tps['tokens_per_sec_mean']:.1f} tokens/s (min={tps['tokens_per_sec_min']:.1f}, max={tps['tokens_per_sec_max']:.1f})")
    print(f"Full report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
