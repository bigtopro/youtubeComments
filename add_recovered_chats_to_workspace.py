#!/usr/bin/env python3
"""
Add the 4 recovered workspace chats to this Cursor workspace so they appear
in the Chat history UI. Run this with Cursor closed to avoid state corruption.

- Copies transcripts to this project's agent-transcripts (if not already).
- Updates workspace state so composer.composerData.allComposers includes
  the 4 conversation IDs and they show in the sidebar.

Usage: python add_recovered_chats_to_workspace.py
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from pathlib import Path

# Recovered chat IDs and display names (from first user message)
RECOVERED = [
    ("636f94f0-fb4b-4b26-ab00-5ba880886e1b", "Data labelling pipeline (DeepSeek)"),
    ("5cb9f0b8-805b-40bb-9206-365c9f257833", "Batching from group_signals"),
    ("cdd42b49-cfc2-4652-9ae1-819c35643678", "Revised buckets, merged topics"),
    ("1c0be21e-efd4-4d65-bea2-21286a4788e3", "YouTube comments extractor"),
]

WORKSPACE_ROOT = Path(__file__).resolve().parent
RECOVERED_SRC = WORKSPACE_ROOT / "recovered_chats" / "workspace_1770306443713"
CURSOR_PROJECT = Path(os.path.expanduser("~/.cursor/projects/Users-venuvamsi-youtubeComments-youtubeComments-code-workspace/agent-transcripts"))
APP_SUPPORT = Path(os.path.expanduser("~/Library/Application Support/Cursor/User/workspaceStorage"))


def find_workspace_hashes() -> list[str]:
    """Find workspaceStorage hashes for youtubeComments (folder and code-workspace)."""
    hashes = []
    for entry in APP_SUPPORT.iterdir():
        if not entry.is_dir():
            continue
        ws_json = entry / "workspace.json"
        if not ws_json.is_file():
            continue
        try:
            data = json.loads(ws_json.read_text())
            folder = (data.get("folder") or "").lower()
            workspace = (data.get("workspace") or "").lower()
            if "youtubecomments" in folder and folder.endswith("youtubecomments"):
                hashes.append(entry.name)
            elif "youtubecomments" in workspace and "code-workspace" in workspace:
                hashes.append(entry.name)
        except Exception:
            continue
    return list(dict.fromkeys(hashes))  # preserve order, dedupe


def ensure_transcripts_copied() -> None:
    # Copy to code-workspace project
    CURSOR_PROJECT.mkdir(parents=True, exist_ok=True)
    folder_project = Path(os.path.expanduser("~/.cursor/projects/Users-venuvamsi-youtubeComments/agent-transcripts"))
    folder_project.mkdir(parents=True, exist_ok=True)
    for chat_id, _ in RECOVERED:
        src = RECOVERED_SRC / f"{chat_id}.txt"
        for dst_dir in (CURSOR_PROJECT, folder_project):
            dst = dst_dir / f"{chat_id}.txt"
            if src.is_file() and (not dst.exists() or dst.stat().st_size != src.stat().st_size):
                shutil.copy2(src, dst)
                print("Copied transcript:", chat_id, "->", dst_dir.parent.name)


def add_composers_to_workspace_state(ws_hash: str) -> None:
    db_path = APP_SUPPORT / ws_hash / "state.vscdb"
    if not db_path.is_file():
        raise FileNotFoundError(f"Workspace state not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT value FROM ItemTable WHERE key = 'composer.composerData'"
        ).fetchone()
        if not row:
            raise ValueError("composer.composerData not found in workspace state")

        data = json.loads(row[0])
        existing_ids = {c["composerId"] for c in data.get("allComposers", [])}

        # Base composer object (minimal fields Cursor might expect)
        base = {
            "type": "head",
            "unifiedMode": "agent",
            "forceMode": "edit",
            "hasUnreadMessages": False,
            "totalLinesAdded": 0,
            "totalLinesRemoved": 0,
            "isArchived": False,
            "isDraft": False,
            "isWorktree": False,
            "isSpec": False,
            "isProject": False,
            "isBestOfNSubcomposer": False,
            "numSubComposers": 0,
            "referencedPlans": [],
            "worktreeStartedReadOnly": False,
            "hasBlockingPendingActions": False,
            "contextUsagePercent": 0,
            "filesChangedCount": 0,
        }

        added = []
        for chat_id, name in RECOVERED:
            if chat_id in existing_ids:
                continue
            import time
            ts = int(time.time() * 1000)
            composer = {
                **base,
                "composerId": chat_id,
                "name": name,
                "subtitle": "Recovered chat",
                "createdAt": ts,
                "lastUpdatedAt": ts,
            }
            data.setdefault("allComposers", []).append(composer)
            added.append(chat_id)

        if not added:
            print("All recovered chats already in workspace state.")
            return

        conn.execute(
            "UPDATE ItemTable SET value = ? WHERE key = 'composer.composerData'",
            (json.dumps(data),),
        )
        conn.commit()
        print("Added to chat list:", ", ".join(added))
    finally:
        conn.close()


def main() -> None:
    print("Ensure transcripts are in project agent-transcripts...")
    ensure_transcripts_copied()

    ws_hashes = find_workspace_hashes()
    if not ws_hashes:
        print("Could not find workspace storage for youtubeComments (folder or code-workspace).")
        return
    for ws_hash in ws_hashes:
        print("Updating workspace state", ws_hash[:8], "...")
        add_composers_to_workspace_state(ws_hash)
    print("Done. Reload the window (Cmd+Shift+P > Developer: Reload Window) to see the chats.")


if __name__ == "__main__":
    main()
