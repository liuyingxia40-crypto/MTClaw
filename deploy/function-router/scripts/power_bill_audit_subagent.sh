#!/usr/bin/env bash
set -euo pipefail

PYTHON_FILE="$HOME/moor/MTClaw/vertical_subagents/power_bill_audit_subagent.py"

if [ ! -f "$PYTHON_FILE" ]; then
  echo "{\"result\":\"error\",\"message\":\"未找到主程序：$PYTHON_FILE\"}"
  exit 1
fi

INPUT="${1:-}"

if [ -n "$INPUT" ]; then
  python3 "$PYTHON_FILE" "$INPUT"
else
  python3 "$PYTHON_FILE"
fi
