#!/usr/bin/env python3
"""
Macro Topic Labeller pipeline (macroPrompt.md).

Maps canonical topics → global macro topics in two phases:
  Phase 1: Batch canonical topics (200 per batch), LLM groups into macro labels.
  Phase 2: Deduplicate/merge macro labels across batches into global macros.

Uses DeepSeek API (OpenAI-compatible). Output: canonical_to_macro, macro_to_canonicals, global_macro_list.

Usage:
  python macro_labeller.py unique_topics.json --out macro_mapping.json --title "Video" --description "..."
  python macro_labeller.py revised_buckets.json -o macro.json --from-buckets   # revised_buckets = merged/canonical topics (apply_canonical_map output)
  With --log-dir: each LLM call logs full request (system + user prompt, params) and response to a file.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from dotenv import load_dotenv

# API (same as consolidate_topics / label_comments)
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2
REQUEST_TIMEOUT = 600  # 10 min per API docs (label_comments.py)
PHASE1_MAX_TOKENS = 3500
PHASE2_MAX_TOKENS = 4000
CHUNK_SIZE = 200


class DeepSeekAPIError(RuntimeError):
    """API error or request failure (same as label_comments.py)."""

# Phase 1: group canonical topics into macro topics
PHASE1_SYSTEM_PROMPT = """You are a semantic taxonomy builder.

You will receive a numbered list of canonical topic labels extracted from YouTube comments for a single video.

Your task is to group them into broader macro topics.

Definition of a macro topic:
- A stable, reusable semantic anchor.
- May be a person, product, feature, concept, event, organization, country, technology, or clearly defined domain.
- Must be concrete and meaningful.
- Must not be vague or overly abstract.
- Must not merge clearly unrelated subjects.

Rules:
1. Each topic number must appear exactly once—in one and only one macro topic.
2. Do not assign the same topic number to multiple macro labels. Pick the single best macro for each topic.
3. When a topic is itself a clear macro-level concept (e.g. a place name like "Arunachal Pradesh", a person, or a well-defined entity), assign it only to that macro. Do not also include it in a broader grouping (e.g. do not put "Arunachal Pradesh" under both "Arunachal Pradesh" and "Sino-Indian border disputes"). Prefer keeping it under the specific macro that matches the topic name when applicable.
4. Do not invent subjects not grounded in the input.
5. Do not create vague labels like "General discussion", "Various topics", "Other issues".
6. Keep macro labels concise (1–4 words).
7. If a topic is already broad and cannot logically group with others, it may form its own macro topic.

Output format (strict):

macro_label: n1,n2,n3
macro_label: n4,n5
...

Only output macro_label: numbers.
No explanations."""

# Phase 2: merge redundant macro labels into global macros
PHASE2_SYSTEM_PROMPT = """You are consolidating MACRO TOPIC LABELS (not individual canonical topics).

You will receive a numbered list of macro topic labels. Each item is a broader grouping label produced in Phase 1 (e.g. "Sino-Indian border disputes", "Arunachal Pradesh", "China-India Relations"). Your task is to find redundant or near-duplicate macro labels and merge them into a single canonical macro per group.

Rules:
1. Input is macro labels only. Merge only when two or more macro labels clearly refer to the same underlying subject or entity (e.g. "Aksai Chin" and "Aksai Chin dispute" → one canonical macro).
2. Do not over-merge distinct subjects. Keep separate macros for distinct entities or domains.
3. For each merged group, keep the most stable, reusable label as the canonical macro.
4. Every number must appear exactly once—each macro label is assigned to exactly one canonical macro (possibly a group of one).
5. If a macro label has no clear duplicate, leave it as its own canonical macro (one number in one group).

Output format:

canonical_macro: n1,n2
canonical_macro: n3
...

Only output canonical_macro: numbers.
No explanations."""


def chunk_topics(topics: List[str], chunk_size: int = CHUNK_SIZE) -> List[List[str]]:
    """Split topics into batches of at most chunk_size."""
    out: List[List[str]] = []
    for i in range(0, len(topics), chunk_size):
        out.append(topics[i : i + chunk_size])
    return out


def build_phase1_prompt(
    batch_topics: List[str],
    video_title: str = "",
    video_description: str = "",
) -> str:
    """Build user prompt for Phase 1: numbered topics + optional title/description."""
    parts: List[str] = []
    if video_title.strip():
        parts.append(f"Video Title: {video_title.strip()}")
    if video_description.strip():
        parts.append(f"Video Description: {video_description.strip()}")
    if parts:
        parts.append("")
    parts.append("#TOPICS")
    for i, t in enumerate(batch_topics, 1):
        parts.append(f"{i}. {t}")
    parts.append("#END")
    return "\n".join(parts)


def _write_llm_detailed_log(
    log_path: str,
    *,
    call_label: str,
    system_prompt: str,
    user_prompt: str,
    payload: Dict[str, Any],
    response_content: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """Write one file per LLM call with full request and response (or error)."""
    ts = datetime.now(timezone.utc).isoformat()
    lines = [
        "=== MACRO LLM REQUEST ===",
        f"call_label: {call_label}",
        f"timestamp: {ts}",
        f"model: {payload.get('model', '')}",
        f"temperature: {payload.get('temperature', '')}",
        f"max_tokens: {payload.get('max_tokens', '')}",
        "---",
        "SYSTEM PROMPT:",
        system_prompt,
        "---",
        "USER PROMPT:",
        user_prompt,
        "",
    ]
    if error is not None:
        lines.extend([
            "=== MACRO LLM RESPONSE (ERROR) ===",
            error,
        ])
    else:
        lines.extend([
            "=== MACRO LLM RESPONSE ===",
            f"(length: {len(response_content or '')} chars)",
            "---",
            response_content or "",
        ])
    # Ensure we don't pass None to join
    lines = [str(x) for x in lines]
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def call_llm(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = PHASE1_MAX_TOKENS,
    temperature: float = 0.2,
    retry: int = 0,
    log_path: Optional[str] = None,
    call_label: str = "macro_llm",
) -> str:
    """
    Call DeepSeek API (same practices as label_comments.py / consolidate_topics.py).
    Returns raw text content (no response_format so we get "macro_label: 1,2,3" plain text).
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": min(max_tokens, 8192),  # API cap (label_comments.py)
    }
    try:
        resp = requests.post(
            DEEPSEEK_API_URL,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
            stream=False,
        )
        resp.raise_for_status()
        try:
            data = resp.json()
        except json.JSONDecodeError:
            text = resp.text.strip()
            if not text or text.startswith(": keep-alive"):
                raise DeepSeekAPIError(
                    f"Received empty/keep-alive response for {call_label}. Retrying..."
                )
            raise
        if "error" in data:
            raise DeepSeekAPIError(
                data["error"].get("message", "Unknown error")
            )
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        if log_path:
            _write_llm_detailed_log(
                log_path,
                call_label=call_label,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                payload={"model": DEEPSEEK_MODEL, "temperature": temperature, "max_tokens": min(max_tokens, 8192)},
                response_content=content,
                error=None,
            )
        return content.strip()
    except (requests.exceptions.RequestException, DeepSeekAPIError) as e:
        if retry < MAX_RETRIES:
            delay = RETRY_BASE_DELAY * (2 ** retry)
            print(
                f"Retrying {call_label} after {delay}s "
                f"(attempt {retry + 1}/{MAX_RETRIES})...",
                file=sys.stderr,
            )
            time.sleep(delay)
            return call_llm(
                api_key, system_prompt, user_prompt,
                max_tokens=max_tokens, temperature=temperature, retry=retry + 1,
                log_path=log_path, call_label=call_label,
            )
        err_msg = f"Request failed for {call_label} after {MAX_RETRIES} retries: {e}"
        if log_path:
            _write_llm_detailed_log(
                log_path,
                call_label=call_label,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                payload={"model": DEEPSEEK_MODEL, "temperature": temperature, "max_tokens": min(max_tokens, 8192)},
                response_content=None,
                error=err_msg,
            )
        raise DeepSeekAPIError(err_msg)


def _parse_label_line(line: str) -> Optional[Tuple[str, List[int]]]:
    """Parse a single line like 'macro_label: 1,2,3' or 'CIA: 1,2,5'. Returns (label, [1,2,5])."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    # Match "label: n1,n2,n3" (label may contain spaces/colons; we take last colon as separator for numbers)
    m = re.match(r"^(.+?):\s*([\d,\s]+)$", line)
    if not m:
        return None
    label = m.group(1).strip()
    num_str = m.group(2).replace(" ", "").strip()
    if not num_str:
        return None
    try:
        numbers = [int(x) for x in num_str.split(",") if x.strip()]
    except ValueError:
        return None
    return (label, numbers)


def parse_phase1_output(output: str, num_topics: int) -> Dict[str, List[int]]:
    """
    Parse Phase 1 LLM output into macro_label -> list of topic numbers.
    Does not validate coverage; use validate_number_coverage for that.
    """
    mapping: Dict[str, List[int]] = {}
    for line in output.splitlines():
        parsed = _parse_label_line(line)
        if not parsed:
            continue
        label, nums = parsed
        if not label:
            continue
        mapping[label] = nums
    return mapping


def validate_number_coverage(mapping: Dict[str, List[int]], expected_count: int) -> Tuple[bool, str]:
    """
    Check that all numbers 1..expected_count appear exactly once in the mapping.
    Returns (valid, error_message).
    """
    if expected_count <= 0:
        return True, ""
    seen: Set[int] = set()
    for nums in mapping.values():
        for n in nums:
            if n < 1 or n > expected_count:
                return False, f"Out-of-range number {n} (expected 1..{expected_count})"
            if n in seen:
                return False, f"Duplicate number {n}"
            seen.add(n)
    missing = set(range(1, expected_count + 1)) - seen
    if missing:
        return False, f"Missing numbers: {sorted(missing)[:20]}{'...' if len(missing) > 20 else ''}"
    return True, ""


def build_phase2_prompt(macro_labels: List[str]) -> str:
    """Build user prompt for Phase 2: numbered macro topic labels (merge redundant ones)."""
    lines = [
        "#MACRO TOPIC LABELS",
        "The following are macro topic labels from Phase 1. Merge labels that refer to the same subject.",
        "",
    ]
    for i, label in enumerate(macro_labels, 1):
        lines.append(f"{i}. {label}")
    lines.append("#END")
    return "\n".join(lines)


def parse_phase2_output(output: str, num_macros: int) -> Dict[str, List[int]]:
    """Parse Phase 2 output: canonical_label -> list of macro numbers. Same format as Phase 1."""
    return parse_phase1_output(output, num_macros)


def build_final_macro_mapping(
    canonical_topics: List[str],
    batch_canonical_to_macro: Dict[str, str],
    batch_macro_labels: List[str],
    macro_to_numbers: Dict[str, List[int]],
) -> Tuple[Dict[str, str], Dict[str, List[str]], List[str]]:
    """
    Build batch_macro -> global_macro from Phase 2, then:
      canonical_to_macro[c] = global_macro for c in canonical_topics
      macro_to_canonicals[g] = list of canonical topics
      global_macro_list = sorted unique global macros
    """
    # macro_to_numbers: global_label -> [batch_macro indices 1..N]
    # Build batch_macro_index -> global_label
    batch_index_to_global: Dict[int, str] = {}
    for global_label, indices in macro_to_numbers.items():
        for idx in indices:
            batch_index_to_global[idx] = global_label

    # batch_macro_labels[i] is the batch macro for index i+1
    batch_macro_to_global: Dict[str, str] = {}
    for i, batch_label in enumerate(batch_macro_labels):
        one_index = i + 1
        batch_macro_to_global[batch_label] = batch_index_to_global.get(one_index, batch_label)

    # canonical_to_macro
    canonical_to_macro: Dict[str, str] = {}
    for c in canonical_topics:
        batch_macro = batch_canonical_to_macro.get(c, c)
        canonical_to_macro[c] = batch_macro_to_global.get(batch_macro, batch_macro)

    # macro_to_canonicals, global_macro_list
    macro_to_canonicals: Dict[str, List[str]] = {}
    for canon, macro in canonical_to_macro.items():
        macro_to_canonicals.setdefault(macro, []).append(canon)
    global_macro_list = sorted(macro_to_canonicals.keys())

    return canonical_to_macro, macro_to_canonicals, global_macro_list


def _run_phase1_batch(
    batch_index: int,
    batch_topics: List[str],
    api_key: str,
    video_title: str,
    video_description: str,
    log_dir: Optional[str],
) -> Tuple[int, Dict[str, str], List[str], Optional[str]]:
    """
    Run Phase 1 for one batch. Returns (batch_index, canonical_to_batch_macro, batch_macro_labels, error).
    On validation failure after retries, returns identity mapping and error message.
    """
    user_prompt = build_phase1_prompt(batch_topics, video_title, video_description)
    log_path = os.path.join(log_dir, f"phase1_batch_{batch_index}.txt") if log_dir else None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    for attempt in range(MAX_RETRIES):
        try:
            raw = call_llm(
                api_key,
                PHASE1_SYSTEM_PROMPT,
                user_prompt,
                max_tokens=PHASE1_MAX_TOKENS,
                temperature=0.2,
                log_path=log_path,
                call_label=f"phase1_batch_{batch_index}",
            )
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                continue
            identity = {t: t for t in batch_topics}
            return batch_index, identity, list(identity.keys()), str(e)

        mapping = parse_phase1_output(raw, len(batch_topics))
        valid, err = validate_number_coverage(mapping, len(batch_topics))
        if valid:
            num_to_label: Dict[int, str] = {}
            for label, nums in mapping.items():
                for n in nums:
                    num_to_label.setdefault(n, label)
            batch_macro_labels = []
            seen = set()
            for i in range(1, len(batch_topics) + 1):
                macro = num_to_label.get(i, batch_topics[i - 1])
                if macro not in seen:
                    seen.add(macro)
                    batch_macro_labels.append(macro)
            canonical_to_batch_macro = {
                batch_topics[i - 1]: num_to_label.get(i, batch_topics[i - 1])
                for i in range(1, len(batch_topics) + 1)
            }
            return batch_index, canonical_to_batch_macro, batch_macro_labels, None
        if log_dir:
            fail_path = os.path.join(log_dir, f"phase1_batch_{batch_index}_validation_fail.txt")
            with open(fail_path, "w", encoding="utf-8") as f:
                f.write(f"Attempt {attempt + 1}\nError: {err}\n\nRaw output:\n{raw}\n")
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
            continue

    identity = {t: t for t in batch_topics}
    return batch_index, identity, list(identity.keys()), f"Validation failed: {err}"


def run_macro_labeller(
    topics: List[str],
    video_title: str = "",
    video_description: str = "",
    api_key: str = "",
    chunk_size: int = CHUNK_SIZE,
    max_parallel: int = 5,
    log_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Orchestrator: Phase 1 (parallel batches) then Phase 2 (single call).
    Returns {
      "canonical_to_macro": {...},
      "macro_to_canonicals": {...},
      "global_macro_list": [...],
    }
    """
    if not topics:
        return {
            "canonical_to_macro": {},
            "macro_to_canonicals": {},
            "global_macro_list": [],
        }

    batches = chunk_topics(topics, chunk_size)
    all_canonical_to_batch_macro: Dict[str, str] = {}
    all_batch_macro_labels: List[str] = []

    # Phase 1: parallel batch calls
    def run_batch(idx_batch: Tuple[int, List[str]]) -> Tuple[int, Dict[str, str], List[str], Optional[str]]:
        idx, batch = idx_batch
        return _run_phase1_batch(idx, batch, api_key, video_title, video_description, log_dir)

    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        futures = {ex.submit(run_batch, (i, b)): i for i, b in enumerate(batches)}
        results_by_index: Dict[int, Tuple[Dict[str, str], List[str], Optional[str]]] = {}
        for future in as_completed(futures):
            batch_index, canonical_to_macro, batch_macros, err = future.result()
            results_by_index[batch_index] = (canonical_to_macro, batch_macros, err)
            if err:
                logging.warning("Phase 1 batch %s: %s (identity fallback used)", batch_index, err)
            all_canonical_to_batch_macro.update(canonical_to_macro)
            all_batch_macro_labels.extend(batch_macros)

    # Unique macro labels for Phase 2 (preserve order for stable numbering)
    seen_macros: Set[str] = set()
    unique_macro_labels: List[str] = []
    for m in all_batch_macro_labels:
        if m not in seen_macros:
            seen_macros.add(m)
            unique_macro_labels.append(m)

    if not unique_macro_labels:
        return {
            "canonical_to_macro": all_canonical_to_batch_macro,
            "macro_to_canonicals": {m: [c for c, m2 in all_canonical_to_batch_macro.items() if m2 == m] for m in set(all_canonical_to_batch_macro.values())},
            "global_macro_list": sorted(set(all_canonical_to_batch_macro.values())),
        }

    # Phase 2: single LLM call to merge macro labels
    phase2_prompt = build_phase2_prompt(unique_macro_labels)
    phase2_log = os.path.join(log_dir, "phase2.txt") if log_dir else None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    macro_to_numbers: Dict[str, List[int]] = {}
    phase2_error: Optional[str] = None
    for attempt in range(MAX_RETRIES):
        try:
            raw2 = call_llm(
                api_key,
                PHASE2_SYSTEM_PROMPT,
                phase2_prompt,
                max_tokens=PHASE2_MAX_TOKENS,
                temperature=0.2,
                log_path=phase2_log,
                call_label="phase2",
            )
        except Exception as e:
            phase2_error = str(e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                continue
            macro_to_numbers = {m: [i + 1] for i, m in enumerate(unique_macro_labels)}
            break

        macro_to_numbers = parse_phase2_output(raw2, len(unique_macro_labels))
        valid, err = validate_number_coverage(macro_to_numbers, len(unique_macro_labels))
        if valid:
            phase2_error = None
            break
        phase2_error = err
        if log_dir:
            with open(os.path.join(log_dir, "phase2_validation_fail.txt"), "w", encoding="utf-8") as f:
                f.write(f"Attempt {attempt + 1}\nError: {err}\n\nRaw:\n{raw2}\n")
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
            continue
        macro_to_numbers = {m: [i + 1] for i, m in enumerate(unique_macro_labels)}

    canonical_to_macro, macro_to_canonicals, global_macro_list = build_final_macro_mapping(
        topics,
        all_canonical_to_batch_macro,
        unique_macro_labels,
        macro_to_numbers,
    )
    return {
        "canonical_to_macro": canonical_to_macro,
        "macro_to_canonicals": macro_to_canonicals,
        "global_macro_list": global_macro_list,
    }


def load_canonical_topics(path: str, key: str = "all_unique_topics") -> List[str]:
    """Load list of canonical topics from JSON (e.g. unique_topics.json)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    topics = data.get(key)
    if topics is None:
        raise KeyError(f"Key '{key}' not found in {path}. Keys: {list(data.keys())}")
    if not isinstance(topics, list):
        raise TypeError(f"Expected list for '{key}', got {type(topics)}")
    return [str(t).strip() for t in topics if str(t).strip()]


def load_topics_from_buckets(path: str) -> List[str]:
    """Load all unique topic keys from a buckets JSON (e.g. revised_buckets.json), sorted alphabetically."""
    from consolidate_topics import load_buckets, flatten_all_topics
    buckets = load_buckets(path)
    if not buckets:
        return []
    topics = flatten_all_topics(buckets)
    return sorted(topics)  # alphabetical order for stable, deterministic LLM input


def main() -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Macro Topic Labeller: map canonical topics to global macro topics (macroPrompt.md)",
    )
    parser.add_argument(
        "input_json",
        help="JSON with canonical topics: unique_topics.json or buckets (e.g. revised_buckets.json) if --from-buckets",
    )
    parser.add_argument("--out", "-o", required=True, help="Output JSON path")
    parser.add_argument(
        "--from-buckets",
        action="store_true",
        help="Input is a buckets JSON (e.g. revised_buckets.json from apply_canonical_map.py); topic keys are canonical/merged topics",
    )
    parser.add_argument(
        "--topics-key",
        default="all_unique_topics",
        help="Key in input JSON for topic list when not using --from-buckets (default: all_unique_topics)",
    )
    parser.add_argument("--title", default="", help="Video title for Phase 1 context")
    parser.add_argument("--description", default="", help="Video description for Phase 1 context")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help=f"Topics per Phase 1 batch (default {CHUNK_SIZE})")
    parser.add_argument("--max-parallel", type=int, default=5, help="Parallel Phase 1 batches")
    parser.add_argument("--log-dir", help="Directory to log raw LLM responses and validation failures")
    parser.add_argument("--max-topics", type=int, metavar="N", help="Use only first N topics (after sorting) for a quick test (e.g. 400 for 2 batches of 200)")
    parser.add_argument("--api-key", help="DeepSeek API key (default: DEEPSEEK_API_KEY env)")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY not set and --api-key not provided.", file=sys.stderr)
        return 1

    try:
        if args.from_buckets:
            topics = load_topics_from_buckets(args.input_json)
            print(f"Loaded {len(topics)} unique canonical topics from buckets: {args.input_json}")
        else:
            topics = load_canonical_topics(args.input_json, args.topics_key)
    except (FileNotFoundError, KeyError, TypeError) as e:
        print(f"Error loading topics: {e}", file=sys.stderr)
        return 1

    # Always send topics to the LLM in alphabetical order (stable, deterministic)
    topics = sorted(topics)
    if args.max_topics is not None and args.max_topics > 0:
        topics = topics[: args.max_topics]
        print(f"Limited to first {len(topics)} topics (--max-topics={args.max_topics})")
    num_batches = (len(topics) + args.chunk_size - 1) // args.chunk_size if topics else 0
    print(f"Sending {len(topics)} topics in alphabetical order ({num_batches} Phase 1 batch(es), chunk_size={args.chunk_size}). max_parallel={args.max_parallel}")
    result = run_macro_labeller(
        topics,
        video_title=args.title,
        video_description=args.description,
        api_key=api_key,
        chunk_size=args.chunk_size,
        max_parallel=args.max_parallel,
        log_dir=args.log_dir,
    )
    print(f"Global macros: {len(result['global_macro_list'])}")

    out_data = {
        "source": args.input_json,
        "num_canonical_topics": len(topics),
        "canonical_to_macro": result["canonical_to_macro"],
        "macro_to_canonicals": result["macro_to_canonicals"],
        "global_macro_list": result["global_macro_list"],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"Wrote macro mapping to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
