#!/usr/bin/env python3
"""
Find Cursor agent chat transcripts under ~/.cursor/projects and copy them
into this workspace so you can browse them after the in-app history is empty.

Usage:
  python load_cursor_chats.py [--output-dir RECOVERED_CHATS]

Default output: ./recovered_chats in the current directory.
Creates: <output_dir>/<workspace_slug>/<chat_id>.(jsonl|txt) and INDEX.md.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path


CURSOR_PROJECTS = Path(os.path.expanduser("~/.cursor/projects"))
DEFAULT_OUT = Path("recovered_chats")


def workspace_slug(project_path: Path) -> str:
    """Derive a short workspace name from the Cursor project path."""
    name = project_path.name
    # e.g. Users-venuvamsi-youtubeComments-youtubeComments-code-workspace -> youtubeComments
    if "youtubeComments" in name:
        return "youtubeComments"
    if "ecoAssign" in name:
        return "ecoAssign"
    if "Library-Application-Support-Cursor-Workspaces" in name or "1770306443713" in name:
        return "workspace_1770306443713"
    if "1770305000909" in name:
        return "workspace_1770305000909"
    # fallback: take last meaningful part
    parts = re.sub(r"^Users-\w+-", "", name).split("-")
    for p in reversed(parts):
        if p and p not in ("code", "workspace", "json"):
            return p
    return name[:32] if len(name) > 32 else name


def first_user_preview_jsonl(file_path: Path, max_len: int = 120) -> str:
    """First user message text from a JSONL transcript."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("role") != "user":
                    continue
                content = obj.get("message", {}).get("content") or []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text") or ""
                        # strip <user_query> wrapper
                        if "<user_query>" in text:
                            text = text.split("<user_query>", 1)[-1].split("</user_query>", 1)[0]
                        text = text.strip().replace("\n", " ")[:max_len]
                        return text or "(no text)"
                break
    except Exception:
        pass
    return "(could not read)"


def first_user_preview_txt(file_path: Path, max_len: int = 120) -> str:
    """First user query from a .txt transcript (user: ... <user_query>...)."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(max_len * 5)
        if "<user_query>" in content:
            start = content.index("<user_query>") + len("<user_query>")
            end = content.find("</user_query>", start)
            if end == -1:
                end = start + max_len
            text = content[start:end].strip().replace("\n", " ")[:max_len]
            return text or "(no text)"
        # fallback: first line after "user:"
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("<user_query>") or (line and not line.startswith("user:") and not line.startswith("assistant:")):
                return line[:max_len] or "(no text)"
    except Exception:
        pass
    return "(could not read)"


def discover_transcripts() -> list[tuple[Path, Path]]:
    """Return list of (source_path, project_path) under CURSOR_PROJECTS/.../agent-transcripts."""
    found: list[tuple[Path, Path]] = []
    if not CURSOR_PROJECTS.is_dir():
        return found
    for project_dir in CURSOR_PROJECTS.iterdir():
        if not project_dir.is_dir():
            continue
        at = project_dir / "agent-transcripts"
        if not at.is_dir():
            continue
        for item in at.iterdir():
            if item.is_file() and item.suffix.lower() in (".txt", ".jsonl"):
                found.append((item, project_dir))
            elif item.is_dir():
                # UUID folder with .jsonl inside (e.g. fda21778.../fda21778....jsonl)
                for sub in item.iterdir():
                    if sub.is_file() and sub.suffix.lower() == ".jsonl":
                        found.append((sub, project_dir))
                        break
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description="Find and load Cursor chat transcripts into this workspace.")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT, help="Output directory (default: recovered_chats)")
    args = ap.parse_args()
    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    pairs = discover_transcripts()
    if not pairs:
        print("No agent transcripts found under", CURSOR_PROJECTS)
        return

    index_lines: list[str] = [
        "# Recovered Cursor chats",
        "",
        "Chats copied from `~/.cursor/projects/.../agent-transcripts` into this workspace.",
        "",
        "| Workspace | Chat ID | Preview | File |",
        "|----------|---------|---------|------|",
    ]

    for src, project_dir in sorted(pairs, key=lambda x: (x[1].name, str(x[0]))):
        slug = workspace_slug(project_dir)
        chat_id = src.stem
        if src.suffix.lower() == ".jsonl":
            preview = first_user_preview_jsonl(src)
        else:
            preview = first_user_preview_txt(src)

        dest_dir = out_root / slug
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        shutil.copy2(src, dest)
        rel = dest.relative_to(out_root)
        preview_esc = preview.replace("|", "\\|").replace("\n", " ")
        index_lines.append(f"| {slug} | `{chat_id}` | {preview_esc} | [{rel}]({rel}) |")

    index_path = out_root / "INDEX.md"
    index_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print("Wrote", index_path)
    print("Chats copied to:", out_root)
    for slug in sorted({workspace_slug(p) for _, p in pairs}):
        count = sum(1 for _, p in pairs if workspace_slug(p) == slug)
        print(f"  - {slug}: {count} chat(s)")


if __name__ == "__main__":
    main()
