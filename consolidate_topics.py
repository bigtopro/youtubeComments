#!/usr/bin/env python3
"""
Topic consolidation pipeline (batching.md).

Reads buckets from group_signals.py output, flattens topics across signal types,
embeds them, builds FAISS index, forms candidate groups by similarity (mutual
neighbors, high threshold), then sends small batches to DeepSeek for canonical
label consolidation. Outputs original_topic -> canonical_topic mapping.

Usage:
  python consolidate_topics.py buckets_threads_rPckDuTaSnw_with_likes.json --out canonical_map.json
  python consolidate_topics.py buckets.json --out map.json --max-group-size 25 --similarity 0.89
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import requests
from dotenv import load_dotenv

# Optional: embeddings + FAISS
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    HAS_EMBED = True
except ImportError:
    HAS_EMBED = False

# DeepSeek API (same as label_comments.py)
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2
REQUEST_TIMEOUT = 300
MAX_PARALLEL_REQUESTS = 10

# Consolidation defaults (batching.md)
SIMILARITY_THRESHOLD_DEFAULT = 0.75  # Changed from 0.89 based on experiment: 0.75 has highest absolute merged count (688)
TOP_K_NEIGHBORS = 25
MAX_GROUP_SIZE_DEFAULT = 25  # No longer enforced in grouping; kept for backward compatibility
MIN_GROUP_SIZE_FOR_LLM = 2  # Skip singletons

# LLM consolidation system prompt (STRICTER - more conservative about merging)
CONSOLIDATION_SYSTEM_PROMPT = """You are consolidating topic labels.

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


def load_buckets(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    buckets = data.get("buckets") or data
    if "buckets" in data:
        buckets = data["buckets"]
    return buckets if isinstance(buckets, dict) else {}


def flatten_all_topics(buckets: Dict[str, Dict[str, Any]]) -> List[str]:
    """Union of all topic strings across signal types (batching.md Step 1)."""
    seen: Set[str] = set()
    out: List[str] = []
    for signal_type, topic_map in buckets.items():
        if not isinstance(topic_map, dict):
            continue
        for topic in topic_map.keys():
            if isinstance(topic, str) and topic.strip() and topic not in seen:
                seen.add(topic)
                out.append(topic)
    return out


def embed_topics(
    topics: List[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> Tuple[np.ndarray, Optional[Any]]:
    """Embed topic strings and L2-normalize (Step 2). Returns (vectors, model)."""
    if not HAS_EMBED:
        raise RuntimeError("Install sentence-transformers and faiss-cpu to run embedding step.")
    model = SentenceTransformer(model_name)
    vecs = model.encode(topics, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32), model


def build_faiss_index(vectors: np.ndarray):
    """Build FAISS index for fast similarity search (Step 3)."""
    if not HAS_EMBED:
        raise RuntimeError("Install faiss-cpu to run FAISS step.")
    d = vectors.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product = cosine for normalized vectors
    index.add(vectors)
    return index


def get_top_k_neighbors(
    index: faiss.Index,
    vectors: np.ndarray,
    topic_id: int,
    k: int = TOP_K_NEIGHBORS,
    min_similarity: float = SIMILARITY_THRESHOLD_DEFAULT,
) -> List[Tuple[int, float]]:
    """Returns list of (neighbor_id, similarity) with similarity >= min_similarity."""
    q = vectors[topic_id : topic_id + 1]
    scores, ids = index.search(q, min(k + 1, vectors.shape[0]))  # +1 to drop self
    out = []
    for i, (idx, sc) in enumerate(zip(ids[0], scores[0])):
        if idx < 0 or idx == topic_id:
            continue
        if sc >= min_similarity:
            out.append((int(idx), float(sc)))
    return out


def is_mutual_neighbor(
    index: faiss.Index,
    vectors: np.ndarray,
    i: int,
    j: int,
    k: int,
    min_sim: float,
) -> bool:
    """True if j is in i's top-k above threshold and i is in j's top-k above threshold."""
    ni = get_top_k_neighbors(index, vectors, i, k=k, min_similarity=min_sim)
    nj = get_top_k_neighbors(index, vectors, j, k=k, min_similarity=min_sim)
    j_in_i = any(idx == j for idx, _ in ni)
    i_in_j = any(idx == i for idx, _ in nj)
    return j_in_i and i_in_j


def compute_group_size_stats(groups: List[List[str]]) -> Dict[str, Any]:
    """Compute statistics about group sizes."""
    sizes = [len(g) for g in groups]
    if not sizes:
        return {
            "total_groups": 0,
            "singletons": 0,
            "multi_topic": 0,
            "min_size": 0,
            "max_size": 0,
            "avg_size": 0.0,
            "median_size": 0,
            "size_distribution": {},
        }
    sizes_sorted = sorted(sizes)
    singletons = sum(1 for s in sizes if s == 1)
    multi_topic = len(sizes) - singletons
    size_dist = {}
    for s in sizes:
        size_dist[s] = size_dist.get(s, 0) + 1
    return {
        "total_groups": len(groups),
        "singletons": singletons,
        "multi_topic": multi_topic,
        "min_size": min(sizes),
        "max_size": max(sizes),
        "avg_size": round(sum(sizes) / len(sizes), 2),
        "median_size": sizes_sorted[len(sizes_sorted) // 2],
        "size_distribution": dict(sorted(size_dist.items())),
    }


def build_candidate_groups(
    topics: List[str],
    vectors: np.ndarray,
    index: faiss.Index,
    top_k: int = TOP_K_NEIGHBORS,
    min_similarity: float = SIMILARITY_THRESHOLD_DEFAULT,
    max_group_size: int = MAX_GROUP_SIZE_DEFAULT,  # Kept for backward compat, no longer enforced
) -> List[List[str]]:
    """
    Build candidate topic groups: for each topic, get high-sim mutual neighbors,
    form groups (NO SIZE CAP - groups can be as large as mutual neighbors allow).
    Returns list of groups (each group = list of topic strings).
    """
    n = len(topics)
    used = set()

    def collect_group(seed_id: int) -> List[int]:
        """Collect all mutual neighbors of seed - no size cap."""
        group_ids = {seed_id}
        candidates = get_top_k_neighbors(index, vectors, seed_id, k=top_k, min_similarity=min_similarity)
        for neighbor_id, _ in candidates:
            if is_mutual_neighbor(index, vectors, seed_id, neighbor_id, top_k, min_similarity):
                group_ids.add(neighbor_id)
        # No size cap - return all mutual neighbors
        return list(group_ids)

    groups: List[List[str]] = []
    for i in range(n):
        if i in used:
            continue
        group_ids = collect_group(i)
        for j in group_ids:
            used.add(j)
        group_strs = [topics[j] for j in group_ids]
        groups.append(group_strs)
    # Singletons that were never in any mutual group
    for j in range(n):
        if j not in used:
            used.add(j)
            groups.append([topics[j]])

    return groups


def build_candidate_groups_with_transitive_closure(
    topics: List[str],
    vectors: np.ndarray,
    index: faiss.Index,
    top_k: int = TOP_K_NEIGHBORS,
    min_similarity: float = SIMILARITY_THRESHOLD_DEFAULT,
    max_path_length: int = 3,
) -> List[List[str]]:
    """
    Build candidate groups WITH transitive closure: if A↔B and B↔C (mutual neighbors),
    then A, B, C are in the same group even if A↔C isn't direct.
    
    Uses BFS to find connected components in the mutual neighbor graph.
    max_path_length limits how many hops we traverse (safeguard against chains).
    """
    n = len(topics)
    
    # Build adjacency list: for each topic, list its mutual neighbors
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i in range(n):
        candidates = get_top_k_neighbors(index, vectors, i, k=top_k, min_similarity=min_similarity)
        for neighbor_id, _ in candidates:
            if is_mutual_neighbor(index, vectors, i, neighbor_id, top_k, min_similarity):
                adj[i].append(neighbor_id)
    
    # BFS to find connected components (transitive closure)
    visited = set()
    groups: List[List[int]] = []
    
    def bfs_component(start: int) -> List[int]:
        """BFS from start, collecting all reachable nodes within max_path_length."""
        component = []
        queue = [(start, 0)]  # (node, path_length)
        seen = {start}
        
        while queue:
            node, path_len = queue.pop(0)
            if path_len > max_path_length:
                continue
            component.append(node)
            for neighbor in adj[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, path_len + 1))
        
        return component
    
    for i in range(n):
        if i in visited:
            continue
        component = bfs_component(i)
        for node in component:
            visited.add(node)
        groups.append(component)
    
    # Convert to topic strings
    return [[topics[j] for j in group] for group in groups]


def format_llm_user_content(topic_phrases: List[str]) -> str:
    """Format topic list for LLM (Step 5)."""
    lines = ["Topic phrases to consolidate:", ""]
    for i, p in enumerate(topic_phrases, 1):
        lines.append(f"{i}. {p}")
    return "\n".join(lines)


def call_deepseek_consolidation(
    api_key: str,
    user_content: str,
    batch_index: int,
    retry: int = 0,
) -> Dict[str, Any]:
    """Single DeepSeek API call for consolidation (same pattern as label_comments.py)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": CONSOLIDATION_SYSTEM_PROMPT},
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
            return call_deepseek_consolidation(api_key, user_content, batch_index, retry + 1)
        raise


def parse_consolidation_response(content: str) -> List[Tuple[str, List[str]]]:
    """Parse LLM JSON to list of (canonical_label, members)."""
    raw = content.strip()
    if "```" in raw:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if m:
            raw = m.group(1).strip()
    raw = re.sub(r"^\s*[^{\[]*", "", raw)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        obj = json.loads(re.sub(r",\s*([}\]])", r"\1", raw))
    clusters = obj.get("clusters") or []
    out = []
    for c in clusters:
        canonical = (c.get("canonical_label") or "").strip() or "unknown"
        members = c.get("members") or []
        if isinstance(members, list):
            members = [str(m).strip() for m in members if m]
        out.append((canonical, members))
    return out


def extract_content(response: Dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if not choices:
        raise ValueError("No choices in response")
    msg = choices[0].get("message", {})
    content = (msg.get("content") or "").strip()
    if not content:
        raise ValueError("Empty content")
    return content


def build_canonical_map(
    groups: List[List[str]],
    api_key: str,
    max_workers: int = MAX_PARALLEL_REQUESTS,
) -> Dict[str, str]:
    """
    Send each group (size >= 2) to LLM; parse response and build
    original_topic -> canonical_topic (Step 6).
    """
    batches = [(i, g) for i, g in enumerate(groups) if len(g) >= MIN_GROUP_SIZE_FOR_LLM]
    canonical_map: Dict[str, str] = {}

    def process_one(item: Tuple[int, List[str]]) -> Tuple[int, Dict[str, str]]:
        idx, phrases = item
        user_content = format_llm_user_content(phrases)
        response = call_deepseek_consolidation(api_key, user_content, idx)
        content = response.get("choices", [{}])[0].get("message", {}).get("content") or ""
        if not content.strip():
            return idx, {p: p for p in phrases}
        try:
            clusters = parse_consolidation_response(content)
        except Exception:
            return idx, {p: p for p in phrases}
        # canonical by normalized (stripped) member string
        strip_to_canonical: Dict[str, str] = {}
        for canonical, members in clusters:
            for m in members:
                if m:
                    strip_to_canonical[m.strip()] = canonical
        # Every input phrase -> canonical (use strip for lookup so LLM echo matches)
        local_map = {p: strip_to_canonical.get(p.strip(), p) for p in phrases}
        return idx, local_map

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_one, (i, g)): (i, g) for i, g in batches}
        for future in as_completed(futures):
            try:
                idx, local_map = future.result()
                canonical_map.update(local_map)
                print(f"  Consolidated batch {idx + 1}/{len(batches)} -> {len(local_map)} mappings")
            except Exception as e:
                print(f"  Batch failed: {e}", file=sys.stderr)

    # Singletons: identity map
    for g in groups:
        if len(g) < MIN_GROUP_SIZE_FOR_LLM:
            for t in g:
                canonical_map.setdefault(t, t)
    return canonical_map


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Topic consolidation from group_signals buckets (batching.md)")
    parser.add_argument("buckets_json", help="Path to buckets JSON (output of group_signals.py --out)")
    parser.add_argument("--out", "-o", required=True, help="Output path for canonical topic map JSON")
    parser.add_argument("--similarity", type=float, default=SIMILARITY_THRESHOLD_DEFAULT, help=f"Min similarity (default {SIMILARITY_THRESHOLD_DEFAULT})")
    parser.add_argument("--top-k", type=int, default=TOP_K_NEIGHBORS, help=f"Neighbors per topic (default {TOP_K_NEIGHBORS})")
    parser.add_argument("--max-group-size", type=int, default=MAX_GROUP_SIZE_DEFAULT, help=f"Max topics per LLM batch (default {MAX_GROUP_SIZE_DEFAULT})")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Sentence transformer model name")
    parser.add_argument("--max-parallel", type=int, default=MAX_PARALLEL_REQUESTS, help="Parallel LLM requests")
    parser.add_argument("--api-key", help="DeepSeek API key (default: DEEPSEEK_API_KEY env)")
    parser.add_argument("--skip-llm", action="store_true", help="Only run embedding + grouping; write groups JSON, no canonical map")
    parser.add_argument(
        "--threshold-sweep",
        metavar="T",
        nargs="+",
        type=float,
        help="Test multiple similarity thresholds; write batches per threshold to --out JSON (no LLM). e.g. --threshold-sweep 0.85 0.88 0.89 0.90 0.92",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key and not args.skip_llm and not args.threshold_sweep:
        print("Error: DEEPSEEK_API_KEY not set and --api-key not provided.", file=sys.stderr)
        return 1

    if not HAS_EMBED:
        print("Error: Install sentence-transformers and faiss-cpu.", file=sys.stderr)
        return 1

    buckets = load_buckets(args.buckets_json)
    topics = flatten_all_topics(buckets)
    print(f"Flattened {len(topics)} unique topics from buckets")

    vectors, _ = embed_topics(topics, model_name=args.embedding_model)
    index = build_faiss_index(vectors)
    print("Built FAISS index")

    # Threshold sweep: run grouping at multiple thresholds, save all batches to one JSON
    if args.threshold_sweep:
        thresholds_sorted = sorted(set(args.threshold_sweep))
        print(f"Threshold sweep: {thresholds_sorted}")
        by_threshold: Dict[str, Any] = {}
        for th in thresholds_sorted:
            groups = build_candidate_groups(
                topics,
                vectors,
                index,
                top_k=args.top_k,
                min_similarity=th,
                max_group_size=args.max_group_size,
            )
            stats = compute_group_size_stats(groups)
            n_singleton = stats["singletons"]
            n_for_llm = stats["multi_topic"]
            key = f"{th:.2f}"
            by_threshold[key] = {
                "num_groups": len(groups),
                "num_singletons": n_singleton,
                "num_for_llm": n_for_llm,
                "group_size_stats": stats,
                "groups": groups,
            }
            print(f"  {key}: {len(groups)} groups ({n_singleton} singletons, {n_for_llm} for LLM)")
            print(f"      Size: min={stats['min_size']}, max={stats['max_size']}, avg={stats['avg_size']:.2f}, median={stats['median_size']}")
            # Show top group sizes
            top_sizes = sorted(stats["size_distribution"].items(), key=lambda x: -x[0])[:10]
            if top_sizes:
                print(f"      Top sizes: {', '.join(f'{size}×{count}' for size, count in top_sizes)}")
        out_data = {
            "source": args.buckets_json,
            "num_topics": len(topics),
            "top_k": args.top_k,
            "max_group_size": args.max_group_size,
            "thresholds": by_threshold,
        }
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)
        print(f"Wrote threshold sweep to {args.out}")
        return 0

    groups = build_candidate_groups(
        topics,
        vectors,
        index,
        top_k=args.top_k,
        min_similarity=args.similarity,
        max_group_size=args.max_group_size,
    )
    stats = compute_group_size_stats(groups)
    n_singleton = stats["singletons"]
    n_multi = stats["multi_topic"]
    print(f"Candidate groups: {len(groups)} total ({n_singleton} singletons, {n_multi} for LLM)")
    print(f"Group size stats: min={stats['min_size']}, max={stats['max_size']}, avg={stats['avg_size']:.2f}, median={stats['median_size']}")
    # Show distribution of group sizes
    top_sizes = sorted(stats["size_distribution"].items(), key=lambda x: -x[0])[:15]
    if top_sizes:
        print(f"Size distribution (top 15): {', '.join(f'{size}×{count}' for size, count in top_sizes)}")

    if args.skip_llm:
        out_data = {
            "source": args.buckets_json,
            "num_topics": len(topics),
            "num_groups": len(groups),
            "groups": groups,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)
        print(f"Wrote groups to {args.out}")
        return 0

    print("Calling DeepSeek for consolidation...")
    canonical_map = build_canonical_map(groups, api_key, max_workers=args.max_parallel)

    out_data = {
        "source": args.buckets_json,
        "num_topics": len(topics),
        "canonical_map": canonical_map,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"Wrote canonical map to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
