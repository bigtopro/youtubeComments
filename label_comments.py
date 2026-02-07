#!/usr/bin/env python
"""
YouTube Comments Data Labeling Pipeline
----------------------------------------

Labels YouTube comment threads using DeepSeek-V3.2 API.
Processes threads in batches respecting token limits and never splits threads.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from queue import Queue
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv


# Constants
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"  # Non-thinking mode
MAX_CONTEXT_TOKENS = 7000
TOKENS_PER_CHAR = 0.3  # 1 English character ≈ 0.3 tokens
MAX_PARALLEL_REQUESTS = 20  # Adjust based on API capacity
MAX_RETRIES = 3  # Maximum retry attempts for failed requests
RETRY_BASE_DELAY = 2  # Base delay in seconds for exponential backoff
REQUEST_TIMEOUT = 600  # 10 minutes timeout (per API docs: connection closes after 10 min)
PROGRESS_SAVE_INTERVAL = 10  # Save progress every N batches
MAX_PARSE_RETRIES = 2  # Retry API call up to this many times when JSON parsing fails
LLM_LOG_DIR_DEFAULT = "llm_outputs"  # Default directory for logging raw LLM responses

# Output token limits (DeepSeek API: default 4K, max 8K for deepseek-chat)
# Batching must limit threads per batch so estimated output fits (avoid truncation → empty signals)
MAX_OUTPUT_TOKENS_DEFAULT = 8000  # Request up to 8K when API allows
AVG_OUTPUT_TOKENS_PER_THREAD = 150  # Conservative: ~140+ tokens per thread for signals JSON
MAX_EMPTY_SIGNAL_RETRIES = 2  # Retry batches for threads that got empty signals (e.g. due to truncation)


@dataclass
class Thread:
    """Represents a single comment thread."""
    thread_id: str
    lines: List[str]  # All lines including THREAD header and comment lines


@dataclass
class Batch:
    """Represents a batch of threads for API processing."""
    threads: List[Thread]
    thread_ids: List[str]  # For mapping outputs back
    batch_index: int
    estimated_tokens: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert batch to dictionary for JSON serialization."""
        return {
            "batch_index": self.batch_index,
            "thread_ids": self.thread_ids,
            "estimated_tokens": self.estimated_tokens,
            "num_threads": len(self.threads)
        }


class DeepSeekAPIError(RuntimeError):
    """Represents an error response from the DeepSeek API."""


def estimate_tokens(text: str) -> int:
    """
    Estimate token count from text.
    1 English character ≈ 0.3 tokens
    """
    return int(len(text) * TOKENS_PER_CHAR)


def parse_threads_file(file_path: str) -> List[Thread]:
    """
    Parse the threads file and extract individual threads.
    
    Format per schema:
    - THREAD <thread_id> (header line)
    - local_id|parent_local_id|like_count|text (comment lines)
    - (blank line separates threads)
    
    Schema rules:
    - Each thread represents one root comment
    - local_id starts at 1 per thread (restarts for each thread)
    - Threads separated by blank lines
    - Depth-first pre-order traversal
    """
    threads: List[Thread] = []
    current_thread_id: Optional[str] = None
    current_lines: List[str] = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            
            # Check if this is a THREAD header
            if line.startswith('THREAD '):
                # Save previous thread if exists (before starting new one)
                if current_thread_id is not None and current_lines:
                    threads.append(Thread(
                        thread_id=current_thread_id,
                        lines=current_lines.copy()
                    ))
                
                # Start new thread
                current_thread_id = line.replace('THREAD ', '').strip()
                current_lines = [line]  # Include THREAD header in lines
            elif line.strip() == '':
                # Empty line - marks end of current thread (per schema: threads separated by blank lines)
                if current_thread_id is not None and current_lines:
                    threads.append(Thread(
                        thread_id=current_thread_id,
                        lines=current_lines.copy()
                    ))
                    current_thread_id = None
                    current_lines = []
                # If we see blank line without active thread, ignore it (could be multiple blank lines)
            elif current_thread_id is not None:
                # Comment line: local_id|parent_local_id|like_count|text
                # Only add if we have an active thread
                current_lines.append(line)
            # If line doesn't start with THREAD and we don't have active thread, skip it
            # (shouldn't happen per schema, but be defensive)
        
        # Don't forget the last thread (if file doesn't end with blank line)
        if current_thread_id is not None and current_lines:
            threads.append(Thread(
                thread_id=current_thread_id,
                lines=current_lines.copy()
            ))
    
    return threads


def create_batches(
    threads: List[Thread],
    system_prompt: str,
    max_tokens: int = MAX_CONTEXT_TOKENS,
    max_output_tokens: int = MAX_OUTPUT_TOKENS_DEFAULT,
    avg_output_tokens_per_thread: int = AVG_OUTPUT_TOKENS_PER_THREAD,
) -> List[Batch]:
    """
    Create batches of threads ensuring:
    - No thread is split across batches (per schema: each thread is atomic)
    - INPUT: total tokens (system prompt + threads) <= max_tokens (context limit)
    - OUTPUT: estimated output tokens (threads * avg per thread) <= max_output_tokens,
      so the API response is not truncated (which causes empty signals for many threads).
    - Threads are never mixed (preserves thread integrity)
    
    Schema compliance:
    - Respects that each thread represents one root comment
    - Preserves thread boundaries (blank line separators)
    - Maintains local_id sequences within each thread (restart at 1 per thread)
    
    Returns list of Batch objects with thread_id mappings for result correlation.
    """
    batches: List[Batch] = []
    system_tokens = estimate_tokens(system_prompt)
    available_input_tokens = max_tokens - system_tokens
    max_threads_per_batch = max(1, max_output_tokens // avg_output_tokens_per_thread)

    if available_input_tokens <= 0:
        raise ValueError(
            f"System prompt is too large ({system_tokens} tokens). "
            f"Max context is {max_tokens} tokens."
        )

    current_batch_threads: List[Thread] = []
    current_batch_ids: List[str] = []
    current_batch_tokens = 0
    batch_index = 0

    for thread in threads:
        thread_text = "\n".join(thread.lines)
        thread_tokens = estimate_tokens(thread_text)

        # Output cap: don't exceed max_threads_per_batch (avoids response truncation)
        would_exceed_output = len(current_batch_threads) >= max_threads_per_batch
        # Input cap: don't exceed context limit
        would_exceed_input = current_batch_tokens + thread_tokens > available_input_tokens

        if thread_tokens > available_input_tokens:
            print(
                f"Warning: Thread {thread.thread_id} exceeds input token limit "
                f"({thread_tokens} > {available_input_tokens}). Processing in its own batch.",
                file=sys.stderr,
            )
            if current_batch_threads:
                batches.append(
                    Batch(
                        threads=current_batch_threads.copy(),
                        thread_ids=current_batch_ids.copy(),
                        batch_index=batch_index,
                        estimated_tokens=current_batch_tokens + system_tokens,
                    )
                )
                batch_index += 1
            current_batch_threads = []
            current_batch_ids = []
            current_batch_tokens = 0
        elif would_exceed_output or would_exceed_input:
            if current_batch_threads:
                batches.append(
                    Batch(
                        threads=current_batch_threads.copy(),
                        thread_ids=current_batch_ids.copy(),
                        batch_index=batch_index,
                        estimated_tokens=current_batch_tokens + system_tokens,
                    )
                )
                batch_index += 1
            current_batch_threads = []
            current_batch_ids = []
            current_batch_tokens = 0

        current_batch_threads.append(thread)
        current_batch_ids.append(thread.thread_id)
        current_batch_tokens += thread_tokens

    if current_batch_threads:
        batches.append(
            Batch(
                threads=current_batch_threads.copy(),
                thread_ids=current_batch_ids.copy(),
                batch_index=batch_index,
                estimated_tokens=current_batch_tokens + system_tokens,
            )
        )

    return batches


def format_batch_for_api(batch: Batch, video_description: Optional[str] = None) -> str:
    """
    Format a batch of threads into the text format expected by the API.
    
    Preserves schema structure:
    - Each thread starts with THREAD <thread_id> header
    - Comment lines follow: local_id|parent_local_id|like_count|text
    - Threads separated by blank lines (per schema)
    
    If video_description is provided, it will be included at the start.
    Each thread is prefixed with a short REF (T1, T2, ...) so the model can echo it
    in the output for reliable matching; we map ref -> thread_id when parsing.
    """
    lines: List[str] = []
    if video_description:
        lines.append(f"Video Description: {video_description}")
        lines.append("")

    for i, thread in enumerate(batch.threads):
        # First line must be THREAD <id>; insert REF T1, T2, ... right after for matching
        ref = f"T{i + 1}"
        if thread.lines:
            lines.append(thread.lines[0])  # THREAD <thread_id>
            lines.append(f"REF {ref}")
            lines.extend(thread.lines[1:])  # comment lines
        else:
            lines.append(f"THREAD {thread.thread_id}")
            lines.append(f"REF {ref}")
        if i < len(batch.threads) - 1:
            lines.append("")
    return "\n".join(lines)


def call_deepseek_api(
    api_key: str,
    system_prompt: str,
    user_content: str,
    batch_index: int,
    retry_count: int = 0,
    max_output_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Call DeepSeek API for a single batch with retry logic.
    Uses max_output_tokens (default 8K per API docs) so responses are not truncated.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    # API max_tokens: controls max response length (see https://api-docs.deepseek.com/guides/json_mode).
    # deepseek-chat: default 4K, maximum 8K. We request 8K by default to avoid truncation.
    out_tokens = max_output_tokens if max_output_tokens is not None else MAX_OUTPUT_TOKENS_DEFAULT
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.3,  # Sweet spot: stable/repeatable, schema compliance, edge cases preserved (max 0.3 for this task)
        "max_tokens": min(out_tokens, 8192),  # Requested response token limit (configurable via --max-output-tokens)
    }
    
    try:
        # Use longer timeout per API docs (10 minutes max)
        response = requests.post(
            DEEPSEEK_API_URL,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
            stream=False  # Non-streaming for simplicity
        )
        response.raise_for_status()
        
        # Handle empty lines/keep-alive comments per API docs
        # The requests library handles this automatically, but we verify JSON
        try:
            data = response.json()
        except json.JSONDecodeError:
            # If response isn't JSON, check if it's empty lines/keep-alive
            text = response.text.strip()
            if not text or text.startswith(': keep-alive'):
                raise DeepSeekAPIError(
                    f"Received empty/keep-alive response for batch {batch_index}. "
                    f"Server may be under high traffic. Retrying..."
                )
            raise
        
        # Check for API errors
        if "error" in data:
            error_msg = data["error"].get("message", "Unknown error")
            raise DeepSeekAPIError(f"DeepSeek API error: {error_msg}")
        
        return data
    
    except (requests.exceptions.RequestException, DeepSeekAPIError) as e:
        # Retry logic with exponential backoff
        if retry_count < MAX_RETRIES:
            delay = RETRY_BASE_DELAY * (2 ** retry_count)
            print(
                f"Retrying batch {batch_index} after {delay}s "
                f"(attempt {retry_count + 1}/{MAX_RETRIES})...",
                file=sys.stderr
            )
            time.sleep(delay)
            return call_deepseek_api(
                api_key, system_prompt, user_content, batch_index, retry_count + 1,
                max_output_tokens=max_output_tokens,
            )
        else:
            raise DeepSeekAPIError(
                f"Request failed for batch {batch_index} after {MAX_RETRIES} retries: {e}"
            )


def process_batch(
    api_key: str,
    system_prompt: str,
    batch: Batch,
    video_description: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
) -> Tuple[int, Dict[str, Any], List[str]]:
    """Process a single batch and return (batch_index, response, thread_ids)."""
    user_content = format_batch_for_api(batch, video_description)
    response = call_deepseek_api(
        api_key, system_prompt, user_content, batch.batch_index,
        max_output_tokens=max_output_tokens,
    )
    return (batch.batch_index, response, batch.thread_ids)


def process_batch_with_parse_retry(
    api_key: str,
    system_prompt: str,
    batch: Batch,
    video_description: Optional[str] = None,
    llm_log_dir: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
) -> Tuple[int, Dict[str, Any], List[str]]:
    """
    Process a single batch with robust parsing and up to 2 API retries on parse failure.
    Returns (batch_index, parsed_signals_dict, thread_ids).
    When llm_log_dir is set, logs every raw LLM response to files for debugging.
    """
    user_content = format_batch_for_api(batch, video_description)
    last_error: Optional[Exception] = None

    for attempt in range(1 + MAX_PARSE_RETRIES):
        try:
            response = call_deepseek_api(
                api_key, system_prompt, user_content, batch.batch_index,
                max_output_tokens=max_output_tokens,
            )
            content = _extract_content_from_response(response)
            parsed = parse_json_robust(content)
            # Normalize: ensure {"signals": [...]}
            if "signals" not in parsed:
                parsed = {"signals": parsed if isinstance(parsed, list) else []}

            usage = response.get("usage") or {}
            if llm_log_dir:
                log_llm_output(
                    llm_log_dir,
                    batch.batch_index,
                    attempt + 1,
                    batch.thread_ids,
                    content,
                    parse_ok=True,
                    completion_tokens=usage.get("completion_tokens"),
                    prompt_tokens=usage.get("prompt_tokens"),
                    total_tokens=usage.get("total_tokens"),
                    system_prompt=system_prompt,
                    user_content=user_content,
                )
            return (batch.batch_index, parsed, batch.thread_ids, usage)
        except (ValueError, json.JSONDecodeError) as e:
            last_error = e
            if llm_log_dir:
                raw_for_log = ""
                try:
                    raw_for_log = _extract_content_from_response(response)
                except Exception:
                    pass
                usage = response.get("usage") or {}
                log_llm_output(
                    llm_log_dir,
                    batch.batch_index,
                    attempt + 1,
                    batch.thread_ids,
                    raw_for_log,
                    parse_ok=False,
                    error_message=str(e),
                    completion_tokens=usage.get("completion_tokens"),
                    prompt_tokens=usage.get("prompt_tokens"),
                    total_tokens=usage.get("total_tokens"),
                    system_prompt=system_prompt,
                    user_content=user_content,
                )
            if attempt < MAX_PARSE_RETRIES:
                print(
                    f"Parse failed for batch {batch.batch_index + 1} (attempt {attempt + 1}), "
                    f"retrying API call ({attempt + 1}/{MAX_PARSE_RETRIES})...",
                    file=sys.stderr,
                )
                time.sleep(RETRY_BASE_DELAY * (attempt + 1))
            else:
                raise last_error

    raise last_error or ValueError("Parse failed after retries")


def _empty_usage() -> Dict[str, Any]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _extract_content_from_response(response: Dict[str, Any]) -> str:
    """Extract raw content string from API response."""
    choices = response.get("choices", [])
    if not choices:
        raise ValueError("No choices in API response")
    message = choices[0].get("message", {})
    content = message.get("content", "") or ""
    if not content:
        raise ValueError("Empty content in API response")
    return content


def log_llm_output(
    llm_log_dir: str,
    batch_index: int,
    attempt: int,
    thread_ids: List[str],
    raw_content: str,
    parse_ok: bool,
    error_message: Optional[str] = None,
    *,
    completion_tokens: Optional[int] = None,
    prompt_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
    user_content: Optional[str] = None,
) -> None:
    """
    Write a full trace log: what we sent to the LLM (system + user) and what we received.
    Filename includes timestamp for tracing. One file per batch per attempt.
    """
    os.makedirs(llm_log_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_batch = batch_index  # 0-based
    filename = f"batch_{safe_batch:04d}_attempt_{attempt}_{ts}.txt"
    filepath = os.path.join(llm_log_dir, filename)
    meta = {
        "batch_index": batch_index,
        "attempt": attempt,
        "thread_count": len(thread_ids),
        "thread_ids": thread_ids,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "parse_ok": parse_ok,
        "content_length": len(raw_content),
    }
    if completion_tokens is not None:
        meta["completion_tokens"] = completion_tokens
    if prompt_tokens is not None:
        meta["prompt_tokens"] = prompt_tokens
    if total_tokens is not None:
        meta["total_tokens"] = total_tokens
    if error_message:
        meta["parse_error"] = error_message
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(json.dumps(meta, indent=2) + "\n")
        f.write("\n---SENT_TO_LLM---\n")
        f.write("---system_prompt---\n")
        f.write((system_prompt or "") + "\n")
        f.write("---user_content---\n")
        f.write((user_content or "") + "\n")
        f.write("\n---RECEIVED_FROM_LLM---\n")
        f.write(raw_content)
        if not raw_content.endswith("\n"):
            f.write("\n")


def parse_api_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and parse JSON content from API response.
    Uses robust parsing to handle malformed LLM output.
    """
    content = _extract_content_from_response(response)
    return parse_json_robust(content)


def parse_json_robust(content: str) -> Dict[str, Any]:
    """
    Robustly parse JSON from LLM output. Handles common issues:
    - Markdown code blocks (```json ... ```)
    - Unterminated strings / truncated output
    - Trailing commas
    - Control characters and stray text
    """
    if not content or not content.strip():
        raise ValueError("Empty content")

    raw = content.strip()

    # 1) Extract from markdown code block if present
    if "```" in raw:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if match:
            raw = match.group(1).strip()
        else:
            # Take after first ``` 
            idx = raw.find("```")
            if idx >= 0:
                rest = raw[idx + 3:].strip()
                if rest.startswith("json"):
                    rest = rest[4:].strip()
                raw = rest

    # 2) Strip leading non-JSON (e.g. "Here is the JSON:"); only strip trailing whitespace/markdown
    raw = re.sub(r"^\s*[^{\[]*", "", raw)
    raw = raw.rstrip()
    raw = re.sub(r"\s*```\s*$", "", raw)

    # 3) Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 4) Remove trailing commas before ] or }
    repaired = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # 5) Truncation repair: find last complete signal object (ending with }\s*) and close
    # Match } followed by optional comma/whitespace - end of a complete object
    for end_m in list(re.finditer(r"\}\s*,?\s*", raw))[::-1]:
        end_pos = end_m.end()
        trial = raw[:end_pos].rstrip().rstrip(",")
        open_b = trial.count("{") - trial.count("}")
        open_s = trial.count("[") - trial.count("]")
        trial += "]" * open_s + "}" * open_b
        try:
            obj = json.loads(trial)
            if isinstance(obj, dict) and "signals" in obj:
                return obj
        except json.JSONDecodeError:
            continue

    # 6) Last resort: close any open brackets (handles truncated at random position)
    # If we ended inside a string, strip back to last complete token
    trimmed = raw.rstrip().rstrip(",")
    if '"' in trimmed:
        last_quote = trimmed.rfind('"')
        if last_quote > 0:
            before = trimmed[: last_quote]
            if before.count('"') % 2 == 1:
                trimmed = trimmed[: last_quote].rstrip().rstrip(",")
    open_brackets = trimmed.count("[") - trimmed.count("]")
    open_braces = trimmed.count("{") - trimmed.count("}")
    suffix = "]" * open_brackets + "}" * open_braces
    try:
        return json.loads(trimmed + suffix)
    except json.JSONDecodeError:
        pass

    raise ValueError("Could not parse JSON from API response (robust parsing exhausted)")


def process_all_batches(
    api_key: str,
    system_prompt: str,
    batches: List[Batch],
    max_workers: int = MAX_PARALLEL_REQUESTS,
    video_description: Optional[str] = None,
    progress_file: Optional[str] = None,
    output_file: Optional[str] = None,
    existing_results: Optional[Dict[str, Dict[str, Any]]] = None,
    llm_log_dir: Optional[str] = None,
    stats_file: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Process all batches in parallel with idempotency and incremental saving.
    
    Features:
    - Idempotency: Skips batches for threads already in existing_results
    - Queue-based processing for better async handling
    - Incremental saving every N batches
    - Progress tracking for resume capability
    
    Returns:
        Dictionary mapping thread_id to the signals JSON for that thread.
    """
    if existing_results is None:
        existing_results = {}
    
    # Filter out batches for already-processed threads (idempotency)
    batches_to_process: List[Batch] = []
    completed_batch_indices: List[int] = []
    
    for batch in batches:
        # Check if all threads in this batch are already processed
        all_processed = all(
            thread_id in existing_results 
            for thread_id in batch.thread_ids
        )
        
        if all_processed:
            completed_batch_indices.append(batch.batch_index)
            print(
                f"⊘ Skipping batch {batch.batch_index + 1}/{len(batches)} "
                f"({len(batch.thread_ids)} threads already processed)"
            )
        else:
            batches_to_process.append(batch)
    
    if not batches_to_process:
        print("All batches already processed. Nothing to do.")
        return existing_results.copy()

    if llm_log_dir:
        print(
            f"LLM trace logging enabled: request + response written to '{llm_log_dir}/' "
            f"(filenames include timestamp, e.g. batch_0000_attempt_1_YYYYMMDD_HHMMSS.txt)"
        )

    print(
        f"Processing {len(batches_to_process)}/{len(batches)} batches "
        f"({len(completed_batch_indices)} already completed) "
        f"with up to {max_workers} parallel requests..."
    )
    
    results: Dict[str, Dict[str, Any]] = existing_results.copy()
    batch_results: Dict[int, Tuple[Dict[str, Any], List[str]]] = {}
    results_lock = Lock()  # Thread-safe results updates

    # Token usage and timing (for throughput stats)
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    usage_lock = Lock()
    start_wall_time: Optional[float] = None

    # Queue for incremental saving
    save_queue: Queue = Queue()
    completed_count = len(completed_batch_indices)
    
    def save_incremental_results():
        """Save results incrementally from queue."""
        while True:
            try:
                batch_result = save_queue.get(timeout=1)
                if batch_result is None:  # Sentinel to stop
                    break
                
                batch_index, thread_results = batch_result
                
                with results_lock:
                    results.update(thread_results)
                
                # Save progress checkpoint
                if progress_file:
                    save_progress(
                        progress_file,
                        completed_batch_indices + list(batch_results.keys()),
                        batches
                    )
                
                # Incremental save to output file
                if output_file:
                    with results_lock:
                        save_results(results, output_file, append=False)
                
                save_queue.task_done()
            except:
                continue
    
    # Start incremental save thread
    import threading
    save_thread = threading.Thread(target=save_incremental_results, daemon=True)
    save_thread.start()
    
    try:
        start_wall_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit batches to process (with parse retry: up to 2 API retries on parse failure)
            future_to_batch = {
                executor.submit(
                    process_batch_with_parse_retry,
                    api_key,
                    system_prompt,
                    batch,
                    video_description,
                    llm_log_dir,
                    max_output_tokens,
                ): batch
                for batch in batches_to_process
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                completed_count += 1
                
                try:
                    result = future.result()
                    # result is (batch_index, parsed_response, thread_ids, usage)
                    if len(result) == 4:
                        batch_index, parsed_response, thread_ids, usage = result
                        with usage_lock:
                            total_prompt_tokens += usage.get("prompt_tokens") or 0
                            total_completion_tokens += usage.get("completion_tokens") or 0
                            total_tokens += usage.get("total_tokens") or 0
                    else:
                        batch_index, parsed_response, thread_ids = result[0], result[1], result[2]
                    batch_results[batch_index] = (parsed_response, thread_ids)
                    
                    # Map results to threads
                    thread_results = map_batch_results_to_threads(
                        batch, parsed_response, thread_ids
                    )
                    
                    # Queue for incremental save
                    save_queue.put((batch_index, thread_results))
                    
                    out_tokens = (usage.get("completion_tokens") or 0) if len(result) == 4 else None
                    if out_tokens is not None:
                        print(
                            f"✓ Completed batch {batch_index + 1}/{len(batches)} "
                            f"({len(thread_ids)} threads, {out_tokens:,} out tokens) "
                            f"[{completed_count}/{len(batches)} total]"
                        )
                    else:
                        print(
                            f"✓ Completed batch {batch_index + 1}/{len(batches)} "
                            f"({len(thread_ids)} threads) "
                            f"[{completed_count}/{len(batches)} total]"
                        )
                
                except Exception as e:
                    print(
                        f"✗ Error processing batch {batch.batch_index}: {e}",
                        file=sys.stderr
                    )
                    # Store error result
                    error_results = {
                        tid: {"signals": [], "error": str(e)}
                        for tid in batch.thread_ids
                    }
                    save_queue.put((batch.batch_index, error_results))
        
        # Wait for all saves to complete
        save_queue.put(None)  # Sentinel
        save_thread.join(timeout=30)
        
    finally:
        # Ensure final save
        if output_file:
            with results_lock:
                save_results(results, output_file, append=False)

    # Report token usage and throughput
    elapsed = time.perf_counter() - start_wall_time if start_wall_time else 0
    if total_tokens > 0 and elapsed > 0:
        tokens_per_sec = total_tokens / elapsed
        print(
            f"\nToken usage: prompt={total_prompt_tokens:,} completion={total_completion_tokens:,} "
            f"total={total_tokens:,} | elapsed={elapsed:.1f}s | throughput={tokens_per_sec:.1f} tokens/s"
        )
    if stats_file:
        d = os.path.dirname(stats_file)
        if d:
            os.makedirs(d, exist_ok=True)
        stats = {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "elapsed_sec": round(elapsed, 2),
            "tokens_per_sec": round(total_tokens / elapsed, 2) if elapsed > 0 else None,
            "batches_processed": len(batches_to_process),
            "threads_processed": sum(len(b.thread_ids) for b in batches_to_process),
        }
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Wrote stats to {stats_file}")
    
    return results


def map_batch_results_to_threads(
    batch: Batch,
    signals_data: Dict[str, Any],
    thread_ids: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Map batch API response to individual thread results.
    Prefer thread_ref (T1, T2, ...) from the model when present; fall back to
    source_comments matching for backward compatibility.
    """
    ref_to_thread_id: Dict[str, str] = {
        f"T{i + 1}": tid for i, tid in enumerate(thread_ids)
    }

    # Build a mapping: for each thread, extract all local_ids it contains (fallback)
    thread_local_ids: Dict[str, set] = {tid: set() for tid in thread_ids}
    for thread in batch.threads:
        for line in thread.lines:
            if "|" in line and not line.startswith("THREAD") and not line.startswith("REF "):
                parts = line.split("|", 1)
                if parts:
                    try:
                        local_id = int(parts[0].strip())
                        thread_local_ids[thread.thread_id].add(local_id)
                    except ValueError:
                        continue

    all_signals = signals_data.get("signals", [])
    thread_signals: Dict[str, List[Dict[str, Any]]] = {tid: [] for tid in thread_ids}

    for signal in all_signals:
        # Prefer thread_ref for matching when present and valid
        thread_ref = signal.get("thread_ref")
        if thread_ref and isinstance(thread_ref, str):
            ref = str(thread_ref).strip()
            if ref in ref_to_thread_id:
                thread_signals[ref_to_thread_id[ref]].append(signal)
                continue

        # Fallback: distribute by source_comments
        source_comments = signal.get("source_comments", [])
        if not source_comments:
            if thread_ids:
                thread_signals[thread_ids[0]].append(signal)
            continue
        assigned = False
        for thread_id in thread_ids:
            if any(local_id in thread_local_ids[thread_id] for local_id in source_comments):
                thread_signals[thread_id].append(signal)
                assigned = True
                break
        if not assigned and thread_ids:
            thread_signals[thread_ids[0]].append(signal)

    thread_results: Dict[str, Dict[str, Any]] = {}
    for thread_id in thread_ids:
        thread_results[thread_id] = {
            "signals": thread_signals.get(thread_id, [])
        }
    return thread_results
    


def load_existing_results(output_file: str) -> Dict[str, Dict[str, Any]]:
    """
    Load existing results from output file for idempotency.
    Returns dictionary mapping thread_id -> signals data.
    """
    if not os.path.exists(output_file):
        return {}
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert list format to dict for easy lookup
        results = {}
        for thread_data in data.get("threads", []):
            thread_id = thread_data.get("thread_id")
            if thread_id:
                results[thread_id] = {
                    "signals": thread_data.get("signals", [])
                }
        
        return results
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(
            f"Warning: Could not load existing results from '{output_file}': {e}. "
            f"Starting fresh.",
            file=sys.stderr
        )
        return {}


def save_results(
    results: Dict[str, Dict[str, Any]],
    output_file: str,
    append: bool = False
) -> None:
    """
    Save results to JSON file.
    
    Args:
        results: Dictionary mapping thread_id -> signals data
        output_file: Path to output file
        append: If True, merge with existing results. If False, overwrite.
    """
    if append:
        # Load existing results and merge
        existing = load_existing_results(output_file)
        existing.update(results)
        results = existing
    
    # Attach thread_id to every signal so labels are explicitly tied to their comment thread
    def signals_with_thread_id(thread_id: str, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {**sig, "thread_id": thread_id}
            for sig in (signals or [])
        ]

    output_data = {
        "threads": [
            {
                "thread_id": thread_id,
                "signals": signals_with_thread_id(thread_id, signals_data.get("signals", []))
            }
            for thread_id, signals_data in sorted(results.items())
        ]
    }
    
    # Atomic write: write to temp file then rename
    temp_file = output_file + ".tmp"
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    os.replace(temp_file, output_file)
    
    print(f"Saved results for {len(results)} threads to '{output_file}'")


def save_progress(
    progress_file: str,
    completed_batches: List[int],
    batches: List[Batch]
) -> None:
    """
    Save progress to allow resuming from checkpoint.
    """
    progress_data = {
        "completed_batches": sorted(completed_batches),
        "total_batches": len(batches),
        "batches": [batch.to_dict() for batch in batches]
    }
    
    temp_file = progress_file + ".tmp"
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, indent=2)
    
    os.replace(temp_file, progress_file)


def load_progress(progress_file: str) -> Optional[Dict[str, Any]]:
    """
    Load progress from checkpoint file.
    """
    if not os.path.exists(progress_file):
        return None
    
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def load_system_prompt(prompt_file: str) -> str:
    """Load system prompt from XML file."""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read()


def retry_empty_signals(
    api_key: str,
    system_prompt: str,
    results: Dict[str, Dict[str, Any]],
    thread_by_id: Dict[str, Thread],
    max_tokens: int,
    max_output_tokens: int,
    avg_output_tokens_per_thread: int,
    video_description: Optional[str],
    max_workers: int,
    llm_log_dir: Optional[str],
    output_file: Optional[str],
    max_retries: int = MAX_EMPTY_SIGNAL_RETRIES,
) -> Dict[str, Dict[str, Any]]:
    """
    Retry labeling for threads that ended up with empty signals (e.g. due to output truncation).
    Re-batches only those threads (respecting output token cap) and merges results.
    """
    for attempt in range(max_retries):
        empty_ids = [
            tid for tid, data in results.items()
            if not (data.get("signals"))
        ]
        if not empty_ids:
            break
        threads_to_retry = [thread_by_id[tid] for tid in empty_ids if tid in thread_by_id]
        if not threads_to_retry:
            break
        print(
            f"\nRetry {attempt + 1}/{max_retries}: {len(threads_to_retry)} threads with empty signals"
        )
        retry_batches = create_batches(
            threads_to_retry,
            system_prompt,
            max_tokens=max_tokens,
            max_output_tokens=max_output_tokens,
            avg_output_tokens_per_thread=avg_output_tokens_per_thread,
        )
        # So process_all_batches doesn't skip these, pass existing_results without empty-signal threads
        completed_only = {
            tid: data for tid, data in results.items()
            if data.get("signals")
        }
        more = process_all_batches(
            api_key,
            system_prompt,
            retry_batches,
            max_workers=max_workers,
            video_description=video_description,
            progress_file=None,
            output_file=output_file,
            existing_results=completed_only,
            llm_log_dir=llm_log_dir,
            stats_file=None,
            max_output_tokens=max_output_tokens,
        )
        results.update(more)
        if output_file:
            save_results(results, output_file, append=False)
    return results


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Label YouTube comment threads using DeepSeek-V3.2 API. "
            "Processes threads in batches respecting token limits."
        )
    )
    parser.add_argument(
        "--threads-file",
        required=True,
        help="Path to the formatted threads file (e.g., threads_JkwMMpz699Y.txt)"
    )
    parser.add_argument(
        "--system-prompt",
        default="systemPrompt.xml",
        help="Path to system prompt XML file (default: systemPrompt.xml)"
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output JSON file path for labeled results"
    )
    parser.add_argument(
        "--api-key",
        help="DeepSeek API key (default: from DEEPSEEK_API_KEY env var)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_CONTEXT_TOKENS,
        help=f"Maximum context tokens (default: {MAX_CONTEXT_TOKENS})"
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=MAX_PARALLEL_REQUESTS,
        help=f"Maximum parallel API requests (default: {MAX_PARALLEL_REQUESTS})"
    )
    parser.add_argument(
        "--video-description",
        help="Optional video description to include in the prompt"
    )
    parser.add_argument(
        "--progress-file",
        help="Path to progress checkpoint file (enables resume capability)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from progress checkpoint if available"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=PROGRESS_SAVE_INTERVAL,
        help=f"Save progress every N batches (default: {PROGRESS_SAVE_INTERVAL})"
    )
    parser.add_argument(
        "--llm-log-dir",
        default=None,
        metavar="DIR",
        help=(
            "Log every raw LLM response to this directory (one file per batch per attempt) "
            "for debugging parsing issues. Default: not set (no logging)."
        ),
    )
    parser.add_argument(
        "--stats-file",
        default=None,
        metavar="FILE",
        help="Write token usage and timing stats (JSON) to this file.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=MAX_OUTPUT_TOKENS_DEFAULT,
        help=f"Max response tokens per request (API default 4K, max 8K). Batches are capped so output fits (default: {MAX_OUTPUT_TOKENS_DEFAULT}).",
    )
    parser.add_argument(
        "--avg-output-tokens-per-thread",
        type=int,
        default=AVG_OUTPUT_TOKENS_PER_THREAD,
        help=f"Conservative estimate of output tokens per thread for batch sizing (default: {AVG_OUTPUT_TOKENS_PER_THREAD}).",
    )
    parser.add_argument(
        "--retry-empty-signals",
        action="store_true",
        help="Retry labeling for threads that got empty signals (e.g. due to truncation), in smaller batches.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    load_dotenv()
    
    args = parse_args(argv)
    
    # Get API key
    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        sys.stderr.write(
            "Error: DeepSeek API key not found. "
            "Provide via --api-key or set DEEPSEEK_API_KEY environment variable.\n"
        )
        return 1
    
    # Load system prompt
    try:
        system_prompt = load_system_prompt(args.system_prompt)
        print(f"Loaded system prompt ({estimate_tokens(system_prompt)} tokens)")
    except Exception as e:
        sys.stderr.write(f"Error loading system prompt: {e}\n")
        return 1
    
    # Load existing results for idempotency
    existing_results = load_existing_results(args.output)
    if existing_results:
        print(f"Found {len(existing_results)} already-processed threads in output file")
    
    # Load progress if resuming
    progress_file = args.progress_file or (args.output + ".progress.json" if args.resume else None)
    progress_data = None
    if args.resume and progress_file and os.path.exists(progress_file):
        progress_data = load_progress(progress_file)
        if progress_data:
            print(f"Resuming from progress checkpoint: {len(progress_data.get('completed_batches', []))} batches completed")
    
    # Parse threads
    try:
        print(f"Parsing threads from '{args.threads_file}'...")
        threads = parse_threads_file(args.threads_file)
        print(f"Parsed {len(threads)} threads")
    except Exception as e:
        sys.stderr.write(f"Error parsing threads file: {e}\n")
        return 1

    thread_by_id = {t.thread_id: t for t in threads}

    # Create batches (capped by both input context and output token limit to avoid truncation)
    try:
        max_threads_per_batch = max(1, args.max_output_tokens // args.avg_output_tokens_per_thread)
        print(
            f"\nCreating batches (input ≤{args.max_tokens} tokens, "
            f"output ≤{args.max_output_tokens} tokens, max {max_threads_per_batch} threads/batch)..."
        )
        batches = create_batches(
            threads,
            system_prompt,
            max_tokens=args.max_tokens,
            max_output_tokens=args.max_output_tokens,
            avg_output_tokens_per_thread=args.avg_output_tokens_per_thread,
        )
        print(f"Created {len(batches)} batches")

        total_tokens = sum(batch.estimated_tokens for batch in batches)
        print(f"Total estimated input tokens: ~{total_tokens:,}")

        for i, batch in enumerate(batches[:5]):
            print(f"  Batch {i + 1}: {len(batch.threads)} threads, ~{batch.estimated_tokens} tokens")
        if len(batches) > 5:
            print(f"  ... and {len(batches) - 5} more batches")
    except Exception as e:
        sys.stderr.write(f"Error creating batches: {e}\n")
        return 1

    # Process batches with idempotency and incremental saving
    try:
        results = process_all_batches(
            api_key,
            system_prompt,
            batches,
            max_workers=args.max_parallel,
            video_description=args.video_description,
            progress_file=progress_file,
            output_file=args.output,
            existing_results=existing_results,
            llm_log_dir=args.llm_log_dir,
            stats_file=args.stats_file,
            max_output_tokens=args.max_output_tokens,
        )
    except Exception as e:
        sys.stderr.write(f"Error processing batches: {e}\n")
        return 1

    empty_count = sum(1 for d in results.values() if not d.get("signals"))
    if empty_count > 0:
        print(f"\nNote: {empty_count} thread(s) have empty signals (may be due to output truncation).")
        if not args.retry_empty_signals:
            print("  Run with --retry-empty-signals to retry those threads in smaller batches.")

    # Retry threads that got empty signals (e.g. due to truncation)
    if args.retry_empty_signals:
        try:
            results = retry_empty_signals(
                api_key,
                system_prompt,
                results,
                thread_by_id,
                max_tokens=args.max_tokens,
                max_output_tokens=args.max_output_tokens,
                avg_output_tokens_per_thread=args.avg_output_tokens_per_thread,
                video_description=args.video_description,
                max_workers=args.max_parallel,
                llm_log_dir=args.llm_log_dir,
                output_file=args.output,
                max_retries=MAX_EMPTY_SIGNAL_RETRIES,
            )
        except Exception as e:
            sys.stderr.write(f"Error during empty-signal retry: {e}\n")
            # Keep existing results

    # Final save
    try:
        save_results(results, args.output, append=False)
        if progress_file and os.path.exists(progress_file):
            os.remove(progress_file)
            print(f"Cleaned up progress file: {progress_file}")
    except Exception as e:
        sys.stderr.write(f"Error saving final results: {e}\n")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
