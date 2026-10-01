#!/usr/bin/env python3
"""PreToolUse guard for Edit/Write/NotebookEdit: agents may only change files inside this repository.

Two exceptions belong to Claude Code itself, not to the machine: the session scratchpad (/tmp/claude-<uid>) and this
project's auto-memory folder (~/.claude/projects/<project>/memory). Everything else is denied.
"""
import json
import os
import re
import sys

event = json.load(sys.stdin)
tool_input = event.get("tool_input") or {}
path = tool_input.get("file_path") or tool_input.get("notebook_path")
if not path:
    sys.exit(0)

root = os.path.realpath(os.environ.get("CLAUDE_PROJECT_DIR") or os.path.join(os.path.dirname(__file__), "..", ".."))
target = os.path.realpath(os.path.join(event.get("cwd") or root, os.path.expanduser(path)))
allowed = [
    root,
    os.path.realpath(f"/tmp/claude-{os.getuid()}"),
    os.path.realpath(os.path.expanduser(f"~/.claude/projects/{re.sub(r'[^A-Za-z0-9]', '-', root)}/memory")),
]
if any(target == base or target.startswith(base + os.sep) for base in allowed):
    sys.exit(0)

print(json.dumps({"hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "deny",
    "permissionDecisionReason": f"{target} is outside the repository {root}; agents may only change files inside it "
                                "(rule in .claude/hooks/restrict_to_repo.py).",
}}))
