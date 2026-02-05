#!/usr/bin/env python
"""
YouTube Comment Threads Extractor
---------------------------------

Extracts all comment threads (top-level comments and their replies) for a
single YouTube video using the YouTube Data API v3.

Output format (JSON):
[
  {
    "id": "<top_level_comment_id>",
    "videoId": "<video_id>",
    "author": "<author_display_name>",
    "text": "<comment_text>",
    "likeCount": 0,
    "publishedAt": "<iso8601>",
    "updatedAt": "<iso8601>",
    "parentId": null,
    "replies": [
      {
        "id": "<reply_comment_id>",
        "videoId": "<video_id>",
        "author": "<author_display_name>",
        "text": "<reply_text>",
        "likeCount": 0,
        "publishedAt": "<iso8601>",
        "updatedAt": "<iso8601>",
        "parentId": "<top_level_comment_id>",
        "replies": []
      }
    ]
  },
  ...
]

Notes:
- The official YouTube comments model only supports one level of replies
  (replies to top-level comments). The code is written in a recursive style
  so that, if deeper nesting ever appears, the structure can represent it,
  but today you should expect at most two levels.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv


YOUTUBE_COMMENTS_BASE_URL = "https://www.googleapis.com/youtube/v3"


class YouTubeAPIError(RuntimeError):
    """Represents an error response from the YouTube Data API."""


class APIKeyPool:
    """
    Thread-safe pool of API keys for basic round-robin distribution.
    """

    def __init__(self, keys: List[str]) -> None:
        if not keys:
            raise ValueError("APIKeyPool requires at least one key.")
        self._keys = keys
        self._lock = threading.Lock()
        self._idx = 0

    def next_key(self) -> str:
        with self._lock:
            key = self._keys[self._idx]
            self._idx = (self._idx + 1) % len(self._keys)
            return key


def _perform_request(endpoint: str, params: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """
    Perform a GET request to a YouTube Data API endpoint and return parsed JSON.

    Raises:
        YouTubeAPIError: if the response is not HTTP 200 or contains an error.
    """
    url = f"{YOUTUBE_COMMENTS_BASE_URL}/{endpoint}"
    full_params = dict(params)
    full_params["key"] = api_key
    resp = requests.get(url, params=full_params, timeout=30)

    try:
        data = resp.json()
    except ValueError:
        resp.raise_for_status()
        raise YouTubeAPIError(f"Non-JSON response from YouTube API: {resp.text[:200]}")  # pragma: no cover

    if resp.status_code != 200:
        message = data.get("error", {}).get("message") if isinstance(data, dict) else None
        raise YouTubeAPIError(
            f"YouTube API error (status={resp.status_code}): {message or resp.text[:200]}"
        )

    if isinstance(data, dict) and "error" in data:
        error = data["error"]
        raise YouTubeAPIError(
            f"YouTube API error: {error.get('message') or json.dumps(error)[:200]}"
        )

    return data


def build_comment_object(snippet: Dict[str, Any], comment_id: str, parent_id: Optional[str]) -> Dict[str, Any]:
    """
    Normalize a YouTube comment snippet into a consistent dict structure.
    """
    return {
        "id": comment_id,
        "videoId": snippet.get("videoId"),
        "author": snippet.get("authorDisplayName"),
        "text": snippet.get("textDisplay") or snippet.get("textOriginal"),
        "likeCount": snippet.get("likeCount", 0),
        "publishedAt": snippet.get("publishedAt"),
        "updatedAt": snippet.get("updatedAt"),
        "parentId": parent_id,
        "replies": [],
    }


MENTION_RE = re.compile(r"@([^\s,.:;!?()]+)")


def _normalize_author_name(name: Optional[str]) -> str:
    """
    Normalize an author name or handle for matching with @mentions.

    Strategy:
    - Lowercase
    - Strip all non-alphanumeric characters
    """
    if not name:
        return ""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _extract_mention_tokens(text: str) -> List[str]:
    """
    Extract raw @mention tokens from comment text.
    """
    if not text:
        return []
    return [m.group(1) for m in MENTION_RE.finditer(text)]


def _infer_parent_node_for_reply(
    existing_nodes: List[Dict[str, Any]],
    root_node: Dict[str, Any],
    reply_comment: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Infer the most likely parent node for a reply.

    Heuristics:
    1. Look for @mentions in the reply text.
    2. Map mention tokens to authors of earlier comments in this thread.
    3. Choose the most recent earlier comment whose normalized author matches
       the normalized mention token.
    4. Fallback: attach directly to the root.

    This constructs a *virtual* reply tree that can have depth > 2 even though
    the raw YouTube data only has root + direct replies.
    """
    text = reply_comment.get("text") or ""
    mention_tokens = _extract_mention_tokens(text)
    if mention_tokens:
        norm_mentions = [_normalize_author_name(tok) for tok in mention_tokens]

        # Search backwards through existing nodes (root + earlier replies)
        for node in reversed(existing_nodes):
            author = node["comment"].get("author")
            norm_author = _normalize_author_name(author)
            if not norm_author:
                continue
            for m in norm_mentions:
                if not m:
                    continue
                # Match if the normalized mention is contained in the normalized author.
                # This handles cases like "raining_macondo" vs "Raining Macondo".
                if m in norm_author or norm_author.endswith(m):
                    return node

    # Default: reply to root
    return root_node


def _format_single_thread_as_schema(thread_root: Dict[str, Any]) -> List[str]:
    """
    Convert a single thread (root + replies) into the compact THREAD schema.

    Output:
        [
          "THREAD <root_id>",
          "1|0|<root_text>",
          "2|1|<reply_text>",
          ...
        ]
    """
    root_id = thread_root.get("id") or ""
    root_text = thread_root.get("text") or ""
    replies = sorted(
        thread_root.get("replies", []),
        key=lambda r: r.get("publishedAt") or "",
    )

    # Build virtual tree with unlimited depth via inferred parents.
    root_node: Dict[str, Any] = {"comment": thread_root, "children": []}
    nodes: List[Dict[str, Any]] = [root_node]

    for reply in replies:
        parent_node = _infer_parent_node_for_reply(nodes, root_node, reply)
        node: Dict[str, Any] = {"comment": reply, "children": []}
        node["parent"] = parent_node
        parent_node["children"].append(node)
        nodes.append(node)

    lines: List[str] = [f"THREAD {root_id}"]

    # Depth-first pre-order traversal to assign local_ids and parent_local_ids.
    counter = 1

    def dfs(node: Dict[str, Any], parent_local_id: int) -> None:
        nonlocal counter
        local_id = counter
        counter += 1

        comment_text = node["comment"].get("text") or ""
        # Ensure single-line text as per schema.
        comment_text_one_line = " ".join(comment_text.split())

        lines.append(f"{local_id}|{parent_local_id}|{comment_text_one_line}")

        for child in node["children"]:
            dfs(child, local_id)

    dfs(root_node, 0)
    return lines


def format_threads_as_compact_schema(threads: List[Dict[str, Any]]) -> str:
    """
    Convert all threads into the compact THREAD format defined in `schema`.
    """
    all_lines: List[str] = []
    first = True

    for root in threads:
        if not first:
            all_lines.append("")
        first = False
        all_lines.extend(_format_single_thread_as_schema(root))

    return "\n".join(all_lines)


def fetch_replies_for_comment(
    api_keys: APIKeyPool, parent_comment_id: str, *, page_size: int = 100
) -> List[Dict[str, Any]]:
    """
    Fetch all direct replies for a given top-level comment.

    Uses `comments.list` with `parentId=<top_level_comment_id>` and paginates
    through all results.
    """
    all_replies: List[Dict[str, Any]] = []
    page_token: Optional[str] = None

    while True:
        params = {
            "part": "snippet",
            "parentId": parent_comment_id,
            "maxResults": page_size,
            "textFormat": "plainText",
        }
        if page_token:
            params["pageToken"] = page_token

        data = _perform_request("comments", params, api_key=api_keys.next_key())

        for item in data.get("items", []):
            comment_id = item.get("id")
            snippet = item.get("snippet", {}) or {}
            reply_obj = build_comment_object(snippet, comment_id, parent_comment_id)
            # YouTube currently does not support replies to replies, but we
            # keep the "replies" field for structural consistency.
            all_replies.append(reply_obj)

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return all_replies


def fetch_comment_threads_for_video(
    api_keys: APIKeyPool, video_id: str, *, page_size: int = 100
) -> List[Dict[str, Any]]:
    """
    Fetch all comment threads (top-level comments + replies) for a video.

    Returns:
        List of normalized comment objects representing the roots of each
        discussion thread (top-level comments) with nested `replies`.
    """
    all_threads: List[Dict[str, Any]] = []
    roots_needing_replies: List[Dict[str, Any]] = []
    page_token: Optional[str] = None

    while True:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": page_size,
            "textFormat": "plainText",
            "order": "time",
        }
        if page_token:
            params["pageToken"] = page_token

        data = _perform_request("commentThreads", params, api_key=api_keys.next_key())

        for item in data.get("items", []):
            snippet = item.get("snippet", {}) or {}
            top_comment = snippet.get("topLevelComment", {}) or {}
            top_comment_id = top_comment.get("id")
            top_snippet = top_comment.get("snippet", {}) or {}

            if not top_comment_id or not top_snippet:
                # Skip malformed items
                continue

            root_obj = build_comment_object(
                top_snippet,
                top_comment_id,
                parent_id=None,
            )

            total_reply_count = snippet.get("totalReplyCount", 0) or 0

            if total_reply_count > 0:
                # Defer fetching replies so we can parallelise later.
                roots_needing_replies.append(root_obj)

            all_threads.append(root_obj)

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    # Fetch replies for each root in parallel to maximise throughput.
    if roots_needing_replies:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        max_workers = min(32, max(4, len(roots_needing_replies)))

        def _load_replies(root: Dict[str, Any]) -> None:
            root_id = root.get("id")
            if not root_id:
                root["replies"] = []
                return
            root["replies"] = fetch_replies_for_comment(api_keys, root_id, page_size=page_size)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_load_replies, root): root for root in roots_needing_replies}
            for _ in as_completed(futures):
                # We don't need individual results; errors will be raised here if any.
                pass

    return all_threads


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract all YouTube comment threads (top-level comments and replies) "
            "for a single video using the YouTube Data API v3."
        )
    )
    parser.add_argument(
        "--video-id",
        required=True,
        help="YouTube video ID (the `v` parameter from the video URL).",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("YOUTUBE_API_KEY"),
        help=(
            "Primary YouTube Data API key. "
            "If omitted, keys are taken from environment variables named "
            "YOUTUBE_API_KEY, YOUTUBE_API_KEY_1, YOUTUBE_API_KEY_2, ..."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Optional output file path (JSON). If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "threads"],
        default="json",
        help=(
            "Output format: 'json' for full JSON (default), or 'threads' for the "
            "compact THREAD schema suitable for LLM input."
        ),
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indent level for pretty-printed JSON (default: 2).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Number of items per API page (1-100, default: 100).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    # Load .env so that YOUTUBE_API_KEY_* variables are available.
    load_dotenv()

    args = parse_args(argv)

    # Collect all available keys for the pool.
    keys: List[str] = []

    if args.api_key:
        keys.append(args.api_key)

    # Collect additional keys from environment (including numbered variants).
    for name, value in os.environ.items():
        if not name.startswith("YOUTUBE_API_KEY"):
            continue
        if not value:
            continue
        if value in keys:
            continue
        keys.append(value)

    if not keys:
        sys.stderr.write(
            "Error: No YouTube Data API keys found. Provide one via --api-key or "
            "set environment variables like YOUTUBE_API_KEY_1, YOUTUBE_API_KEY_2, ...\n"
        )
        return 1

    api_key_pool = APIKeyPool(keys)

    try:
        threads = fetch_comment_threads_for_video(
            api_keys=api_key_pool,
            video_id=args.video_id,
            page_size=args.page_size,
        )
    except YouTubeAPIError as exc:
        sys.stderr.write(f"Failed to fetch comment threads: {exc}\n")
        return 1

    if args.format == "json":
        output = json.dumps(threads, indent=args.indent, ensure_ascii=False)
    else:
        # Compact THREAD schema format (text, not JSON)
        output = format_threads_as_compact_schema(threads)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Wrote {len(threads)} comment threads to '{args.output}' in {args.format} format.")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

