#!/usr/bin/env python3
"""
Analyze final topic count after consolidation from existing LLM logs.

Reads logs from:
- llm_experiment_logs/ (different thresholds)
- tc_llm_logs/ (with/without transitive closure)

Extracts canonical labels from each batch and counts unique ones.

Usage:
  python analyze_final_topic_count.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import glob


def parse_log_file(log_path: str) -> List[str]:
    """Parse one log file and extract canonical labels from clusters."""
    canonical_labels: List[str] = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Find PARSED section
        parsed_section = re.search(r"=== PARSED ===\s*(.*)", content, re.DOTALL)
        if not parsed_section:
            return []
        
        parsed_text = parsed_section.group(1).strip()
        try:
            parsed_json = json.loads(parsed_text)
            clusters = parsed_json.get("clusters", [])
            for cluster in clusters:
                canonical = cluster.get("canonical_label", "").strip()
                if canonical:
                    canonical_labels.append(canonical)
        except json.JSONDecodeError:
            # Try to extract from raw response
            raw_section = re.search(r"=== RAW LLM RESPONSE ===\s*(.*?)(?=\n\n===|$)", content, re.DOTALL)
            if raw_section:
                raw_text = raw_section.group(1).strip()
                try:
                    raw_json = json.loads(raw_text)
                    clusters = raw_json.get("clusters", [])
                    for cluster in clusters:
                        canonical = cluster.get("canonical_label", "").strip()
                        if canonical:
                            canonical_labels.append(canonical)
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        print(f"Warning: Error parsing {log_path}: {e}", file=sys.stderr)
    
    return canonical_labels


def analyze_logs_directory(log_dir: str) -> Dict[str, Any]:
    """Analyze all logs in a directory and return statistics."""
    if not os.path.exists(log_dir):
        return {"error": f"Directory not found: {log_dir}"}
    
    all_canonicals: List[str] = []
    batch_count = 0
    error_count = 0
    
    # Find all batch_*.txt files
    log_files = sorted(glob.glob(os.path.join(log_dir, "**", "batch_*.txt"), recursive=True))
    
    for log_file in log_files:
        batch_count += 1
        canonicals = parse_log_file(log_file)
        if not canonicals:
            error_count += 1
        all_canonicals.extend(canonicals)
    
    unique_canonicals = set(all_canonicals)
    
    return {
        "total_batches": batch_count,
        "error_batches": error_count,
        "total_canonical_labels": len(all_canonicals),
        "unique_canonical_labels": len(unique_canonicals),
        "canonical_labels": sorted(list(unique_canonicals)),
    }


def load_singletons_from_sweep(sweep_file: str, threshold: str) -> Tuple[int, bool]:
    """Load number of singletons for a threshold from threshold_sweep_batches.json.
    Returns (singletons_count, found_in_file)."""
    # Known singleton counts for thresholds not in sweep file (from recent run)
    known_singletons = {
        "0.60": 461,
        "0.65": 719,
        "0.70": 1114,
    }
    
    if threshold in known_singletons:
        return known_singletons[threshold], True
    
    try:
        with open(sweep_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        thresholds = data.get("thresholds", {})
        th_data = thresholds.get(threshold, {})
        if th_data:
            # Can get from num_singletons field or compute from groups
            if "num_singletons" in th_data:
                return th_data["num_singletons"], True
            elif "groups" in th_data:
                # Count singletons from groups
                groups = th_data["groups"]
                singletons = sum(1 for g in groups if len(g) == 1)
                return singletons, True
        return 0, False
    except Exception:
        return 0, False


def main() -> int:
    base_dir = "/Users/venuvamsi/youtubeComments"
    
    print("Analyzing final topic counts from LLM logs (including singletons)...")
    print("=" * 70)
    
    # Load singletons from threshold sweep
    sweep_file = os.path.join(base_dir, "threshold_sweep_batches.json")
    
    # 1. Analyze threshold experiment logs (llm_experiment_logs/)
    print("\n1. Threshold experiment (llm_experiment_logs/):")
    print("-" * 70)
    threshold_logs_dir = os.path.join(base_dir, "llm_experiment_logs")
    threshold_results: Dict[str, Dict[str, Any]] = {}
    
    if os.path.exists(threshold_logs_dir):
        # Find threshold subdirectories (0.60, 0.65, etc.)
        for threshold_dir in sorted(os.listdir(threshold_logs_dir)):
            threshold_path = os.path.join(threshold_logs_dir, threshold_dir)
            if os.path.isdir(threshold_path):
                stats = analyze_logs_directory(threshold_path)
                if "error" not in stats:
                    # Add singletons
                    singletons, found = load_singletons_from_sweep(sweep_file, threshold_dir)
                    if not found:
                        # For thresholds not in sweep file, we can't get singletons from grouping phase
                        # But we know: total_topics = canonical_from_llm + singletons
                        # And we know original num_topics = 4691
                        # So singletons = 4691 - (topics_in_multi_groups)
                        # But we don't know how many topics were in multi-groups that got consolidated
                        # So we'll mark as unknown
                        singletons = None
                        total_unique = None
                        stats['singletons'] = None
                        stats['total_unique_topics'] = None
                        stats['singletons_unknown'] = True
                    else:
                        total_unique = stats['unique_canonical_labels'] + singletons
                        stats['singletons'] = singletons
                        stats['total_unique_topics'] = total_unique
                        stats['singletons_unknown'] = False
                    threshold_results[threshold_dir] = stats
                    if total_unique is not None:
                        print(f"  {threshold_dir}: {total_unique} total unique topics "
                              f"({stats['unique_canonical_labels']} from LLM + {singletons} singletons, "
                              f"{stats['total_batches']} batches)")
                    else:
                        print(f"  {threshold_dir}: {stats['unique_canonical_labels']} from LLM "
                              f"(singletons unknown - not in sweep file, {stats['total_batches']} batches)")
    else:
        print(f"  Directory not found: {threshold_logs_dir}")
    
    # 2. Analyze transitive closure experiment logs (tc_llm_logs/)
    print("\n2. Transitive closure experiment (tc_llm_logs/):")
    print("-" * 70)
    tc_logs_dir = os.path.join(base_dir, "tc_llm_logs")
    tc_comparison_file = os.path.join(base_dir, "transitive_comparison.json")
    tc_results: Dict[str, Dict[str, Any]] = {}
    
    # Load singletons from transitive_comparison.json
    tc_singletons: Dict[str, int] = {}
    if os.path.exists(tc_comparison_file):
        try:
            with open(tc_comparison_file, "r", encoding="utf-8") as f:
                tc_data = json.load(f)
            tc_singletons["no_tc"] = tc_data.get("comparison", {}).get("without_transitive_closure", {}).get("singletons", 0)
            tc_singletons["with_tc"] = tc_data.get("comparison", {}).get("with_transitive_closure", {}).get("singletons", 0)
        except Exception:
            pass
    
    if os.path.exists(tc_logs_dir):
        for method_dir in ["no_tc", "with_tc"]:
            method_path = os.path.join(tc_logs_dir, method_dir)
            if os.path.isdir(method_path):
                stats = analyze_logs_directory(method_path)
                if "error" not in stats:
                    # Add singletons
                    singletons = tc_singletons.get(method_dir, 0)
                    total_unique = stats['unique_canonical_labels'] + singletons
                    stats['singletons'] = singletons
                    stats['total_unique_topics'] = total_unique
                    tc_results[method_dir] = stats
                    print(f"  {method_dir}: {total_unique} total unique topics "
                          f"({stats['unique_canonical_labels']} from LLM + {singletons} singletons, "
                          f"{stats['total_batches']} batches)")
    else:
        print(f"  Directory not found: {tc_logs_dir}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Final unique canonical topic counts")
    print("=" * 70)
    
    if threshold_results:
        print("\nBy Threshold (TOTAL unique topics = LLM canonical + singletons):")
        print(f"{'Threshold':<12} {'Total Unique':<15} {'From LLM':<12} {'Singletons':<12} {'Batches':<10}")
        print("-" * 65)
        for th in sorted(threshold_results.keys()):
            s = threshold_results[th]
            total = s.get('total_unique_topics', '?')
            singletons = s.get('singletons', '?')
            if s.get('singletons_unknown'):
                total = f"{s['unique_canonical_labels']}+?"
                singletons = "?"
            print(f"{th:<12} {str(total):<15} {s['unique_canonical_labels']:<12} "
                  f"{str(singletons):<12} {s['total_batches']:<10}")
    
    if tc_results:
        print("\nWith vs Without Transitive Closure (TOTAL unique topics = LLM canonical + singletons):")
        print(f"{'Method':<20} {'Total Unique':<15} {'From LLM':<12} {'Singletons':<12} {'Batches':<10}")
        print("-" * 70)
        for method in ["no_tc", "with_tc"]:
            if method in tc_results:
                s = tc_results[method]
                print(f"{method:<20} {s['total_unique_topics']:<15} {s['unique_canonical_labels']:<12} "
                      f"{s['singletons']:<12} {s['total_batches']:<10}")
        
        if "no_tc" in tc_results and "with_tc" in tc_results:
            diff = tc_results["with_tc"]["total_unique_topics"] - tc_results["no_tc"]["total_unique_topics"]
            print(f"\nDifference: {diff:+d} topics ({diff/tc_results['no_tc']['total_unique_topics']*100:+.1f}%)")
    
    # Save results
    output = {
        "threshold_results": threshold_results,
        "tc_results": tc_results,
    }
    output_file = os.path.join(base_dir, "final_topic_counts.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved detailed results to {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
