#!/usr/bin/env bash
# Open the Cursor workspace that has your recovered chats (1770306443713).
# Usage: ./open_workspace_with_chats.sh

WORKSPACE_FILE="$HOME/Library/Application Support/Cursor/Workspaces/1770306443713/workspace.json"
if [[ ! -f "$WORKSPACE_FILE" ]]; then
  echo "Workspace file not found: $WORKSPACE_FILE"
  exit 1
fi

# Try 1: cursor CLI (new window)
if command -v cursor &>/dev/null; then
  cursor -n "$WORKSPACE_FILE" 2>/dev/null && exit 0
  cursor "$WORKSPACE_FILE" 2>/dev/null && exit 0
fi

# Try 2: macOS open (opens file with Cursor app)
open -a Cursor "$WORKSPACE_FILE" 2>/dev/null && exit 0

# Try 3: direct path to Cursor app
OPEN_WITH="/Applications/Cursor.app/Contents/MacOS/Cursor"
if [[ -x "$OPEN_WITH" ]]; then
  "$OPEN_WITH" -n "$WORKSPACE_FILE" 2>/dev/null && exit 0
fi

echo "Could not launch Cursor. Open manually:"
echo "  1. In Cursor: File → Open Workspace from File..."
echo "  2. Press Cmd+Shift+G and paste this path:"
echo "     $WORKSPACE_FILE"
# Copy path to clipboard on macOS so user can paste
if command -v pbcopy &>/dev/null; then
  echo -n "$WORKSPACE_FILE" | pbcopy
  echo "  (Path copied to clipboard — paste in the dialog.)"
fi
exit 1
