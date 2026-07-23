"""Function Router service.

This module implements a FastAPI service that accepts OpenAI-compatible
``/v1/chat/completions`` requests, uses a local Qwen model to detect tool calls
for system control actions, executes those actions via local shell scripts, and
falls back to transparently proxying requests to an upstream model provider.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

try:
    from .builtin_tools import execute_builtin_tool, get_builtin_tools, is_builtin_tool
except ImportError:  # pragma: no cover - direct script execution fallback
    from builtin_tools import execute_builtin_tool, get_builtin_tools, is_builtin_tool


#SYSTEM_PROMPT = (
#    "You are a system and filesystem assistant. Use the provided tools to handle "
#    "user requests about system settings, wallpaper, volume, brightness, file "
#    "search, directory listing, file reading, text search, and short waits. If a "
#    "user request does not match any available tool, respond with a brief text "
#    "saying you cannot handle it. Always respond in the same language as the user."
#)

#SYSTEM_PROMPT = (
#    "You are a system and filesystem assistant. Use the provided tools to handle "
#    "user requests about system settings, wallpaper, volume, brightness, file "
#    "search, directory listing, file reading, text search, and short waits. If a "
#    "user request does not match any available tool, respond with a brief text "
#    "saying you cannot handle it. Always respond in the same language as the user."
#)

SYSTEM_PROMPT = (
    "You are a system and filesystem assistant. Use only the provided tools to handle "
    "user requests about system settings, wallpaper, volume, brightness, file search, "
    "directory listing, file reading, text search, and short waits. "
    "If a request does not match any available tool, reply briefly that you cannot handle it. "
    "You must always reply in Chinese. "
    "Do not use emojis, emoticons, or decorative symbols. "
    "You must strictly base your reply on tool outputs only. Do not add any information "
    "that is not directly supported by the tool results. Do not speculate, infer, expand, "
    "or provide extra commentary. "
    "When issuing a tool call that uses any exact value from a previous tool result, such as a "
    "file path, file name, URL, ID, command output field, or other identifier, you must copy that "
    "value verbatim exactly as it appears in the tool result. "
    "Never shorten, summarize, normalize, translate, rename, or rewrite such values. "
    "Never replace any part of an exact value with ellipsis like '...' or similar placeholders. "
    "Never substitute visually similar Unicode characters, homoglyphs, confusable characters, or "
    "characters from another script such as Cyrillic or Greek letters for any part of an exact value. "
    "For example, if a tool result contains an ASCII file path, you must preserve the exact original "
    "ASCII characters and must not replace Latin letters with look-alike Unicode letters. "
    "If the exact value is missing or ambiguous, do not guess; call another tool to retrieve it first. "
    "Replies must be concise, accurate, reliable, and focused on the core result only."
)

SYSTEM_PROMPT_REVIEW = '''You are a task completion judge.

Return TASK_COMPLETE if:
- the assistant completed the request, or
- the assistant successfully moved the task forward and is now waiting for the user to choose, confirm, or provide the next input.

Return TASK_INCOMPLETE only if:
- a necessary tool call failed,
- the assistant was blocked,
- the assistant did not meaningfully address the request,
- or the task did not reach a valid stopping point.

Do not require the user's ultimate real-world goal to be fully finished.
If the workflow has reached a natural handoff point after successful tool use, return TASK_COMPLETE.

Use only the shown conversation and tool results.

Reply with exactly one of:
TASK_COMPLETE
TASK_INCOMPLETE'''

ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")
DEFAULT_CONFIG_PATH = Path.home() / ".function-router" / "config.json"
DEFAULT_ROOT_DIR = Path.home() / ".function-router"


class RecreatingRotatingFileHandler(RotatingFileHandler):
    """Rotating file handler that recreates the target file if deleted.

    If the log file path is removed while the process is still running,
    the next emit reopens a fresh file at the original path. Old deleted-file
    contents are not recovered.
    """

    def emit(self, record: logging.LogRecord) -> None:
        if self.stream is not None and not os.path.exists(self.baseFilename):
            self.acquire()
            try:
                if self.stream is not None:
                    self.stream.close()
                    self.stream = self._open()
            finally:
                self.release()
        super().emit(record)
DEBUG_LOGGER_NAME = "function_router.debug"
REQUEST_LOGGER_NAME = "function_router.request"


@dataclass(slots=True)
class ModelConfig:
    """Connection details for a model endpoint."""

    base_url: str
    model: str
    api_key: str


@dataclass(slots=True)
class AppConfig:
    """Runtime configuration loaded from disk."""

    listen_host: str
    listen_port: int
    routing: ModelConfig
    upstream: ModelConfig
    functions_file: str
    scripts_dir: str
    max_tool_rounds: int
    tool_exec_timeout_s: int
    root_dir: Path
    config_path: Path
    tools_base_dir: str | None = None
    fr_completion_check: bool = True
    fr_completion_check_mode: str = "permissive"
    fr_completion_check_always_true: bool = False
    fr_context_history: bool = True
    fr_context_preserve: bool = False
    debug_logging: bool = False
    routing_timeout_s: float = 10.0
    delegate_to_openclaw: bool = True
    delegate_tools: list[str] | None = None

    @property
    def functions_path(self) -> Path:
        """Return the resolved functions file path."""

        path = Path(self.functions_file)
        return path if path.is_absolute() else self.root_dir / path

    @property
    def resolved_scripts_dir(self) -> Path:
        """Return the resolved scripts directory."""

        path = Path(self.scripts_dir)
        return path if path.is_absolute() else self.root_dir / path


@dataclass(slots=True)
class AppStateData:
    """Mutable application state populated during startup."""

    config_path: Path
    config: AppConfig | None = None
    tools: list[dict[str, Any]] | None = None
    logger: logging.Logger | None = None
    http_client: httpx.AsyncClient | None = None
    warmup_ok: bool = False


STATE = AppStateData(config_path=DEFAULT_CONFIG_PATH)

# Ring buffer for recent tool executions (thread-safe via deque).
TOOL_HISTORY: deque[dict[str, Any]] = deque(maxlen=200)

# Qwen internal context buckets keyed by caller-provided session id.
# Each value contains messages[1:] from the last successful tool loop
# (everything after system prompt).
_QWEN_SAVED_CONTEXTS: dict[str, list[dict[str, Any]]] = {}

# Pending Qwen-completed plain-text turns to expose to upstream later,
# keyed by caller-provided session id. Each item is a dict with
# user_text/assistant_text only; tool traces remain in Qwen internal history.
_QWEN_PENDING_UPSTREAM_TURNS: dict[str, list[dict[str, str]]] = {}
_SESSION_PENDING_DELEGATED_TOOL_IDS: dict[str, set[str]] = {}
_LAST_DEBUG_SESSION_KEY: str | None = None


def _get_pending_upstream_turns(session_key: str) -> list[dict[str, str]]:
    """Return pending plain-text completed turns for one session key."""

    return _QWEN_PENDING_UPSTREAM_TURNS.get(session_key, [])


def _append_pending_upstream_turn(session_key: str, user_text: str, assistant_text: str) -> None:
    """Queue one completed Qwen turn for future upstream visibility."""

    turns = _QWEN_PENDING_UPSTREAM_TURNS.setdefault(session_key, [])
    turns.append({"user_text": user_text, "assistant_text": assistant_text})


def _clear_pending_upstream_turns(session_key: str) -> None:
    """Clear pending upstream-visible turns for one session key."""

    _QWEN_PENDING_UPSTREAM_TURNS[session_key] = []


def _mark_pending_delegated_tool_calls(
    session_key: str,
    tool_calls: list[dict[str, Any]],
) -> None:
    """Remember delegated tool call ids that should return as OpenClaw continuations."""

    if not session_key or not tool_calls:
        return
    ids = {
        tool_call.get("id")
        for tool_call in tool_calls
        if isinstance(tool_call.get("id"), str) and tool_call.get("id")
    }
    if not ids:
        return
    bucket = _SESSION_PENDING_DELEGATED_TOOL_IDS.setdefault(session_key, set())
    bucket.update(ids)
    _debug_log(
        "delegated_tool_pending_add",
        session_key=session_key,
        tool_call_ids=sorted(ids),
        pending=len(bucket),
    )


def _observed_tool_call_ids(tool_call_ids: Any = None) -> list[str]:
    if isinstance(tool_call_ids, str):
        return [tool_call_ids] if tool_call_ids else []
    if isinstance(tool_call_ids, (list, tuple, set)):
        return [item for item in tool_call_ids if isinstance(item, str) and item]
    return []


def _consume_pending_delegated_tool_turn(
    session_key: str,
    tool_call_ids: Any = None,
) -> bool:
    """Consume pending delegated id(s) for one OpenClaw tool-result continuation."""

    if not session_key:
        return False
    bucket = _SESSION_PENDING_DELEGATED_TOOL_IDS.get(session_key)
    if not bucket:
        return False

    observed_ids = _observed_tool_call_ids(tool_call_ids)
    matched_ids = {tool_call_id for tool_call_id in observed_ids if tool_call_id in bucket}
    if matched_ids:
        consumed_ids = sorted(matched_ids)
        bucket.difference_update(matched_ids)
    else:
        consumed_ids = [sorted(bucket)[0]]
        bucket.remove(consumed_ids[0])

    if not bucket:
        _SESSION_PENDING_DELEGATED_TOOL_IDS.pop(session_key, None)
    _debug_log(
        "delegated_tool_pending_consume",
        session_key=session_key,
        expected_tool_call_ids=consumed_ids,
        observed_tool_call_ids=observed_ids,
        pending=len(bucket),
    )
    return True


def _clear_pending_delegated_tool_ids(session_key: str) -> None:
    """Clear remembered delegated tool call ids for one session key."""

    previous = len(_SESSION_PENDING_DELEGATED_TOOL_IDS.get(session_key, set()))
    _SESSION_PENDING_DELEGATED_TOOL_IDS.pop(session_key, None)
    if previous:
        _debug_log(
            "delegated_tool_pending_clear",
            session_key=session_key,
            previous_ids=previous,
        )


def _render_pending_upstream_messages(session_key: str) -> list[dict[str, str]]:
    """Render pending completed turns as plain OpenAI-style chat messages."""

    rendered: list[dict[str, str]] = []
    for turn in _get_pending_upstream_turns(session_key):
        rendered.append({"role": "user", "content": turn["user_text"]})
        rendered.append({"role": "assistant", "content": turn["assistant_text"]})
    return rendered


def _get_saved_context(session_key: str) -> list[dict[str, Any]]:
    """Return saved Qwen context for one session key."""

    return _QWEN_SAVED_CONTEXTS.get(session_key, [])


def _set_saved_context(session_key: str, messages: list[dict[str, Any]]) -> None:
    """Store saved Qwen context for one session key."""

    _QWEN_SAVED_CONTEXTS[session_key] = list(messages)


def _clear_saved_context(session_key: str) -> None:
    """Clear saved Qwen context for one session key."""

    _QWEN_SAVED_CONTEXTS.pop(session_key, None)


def derive_session_key(
    original_request: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> str:
    """Return a stable session key from request headers/body if present."""

    if headers:
        header_value = headers.get("x-openclaw-session-key")
        if isinstance(header_value, str):
            header_value = header_value.strip()
            if header_value:
                return header_value

        header_value = headers.get("x-openclaw-session-id")
        if isinstance(header_value, str):
            header_value = header_value.strip()
            if header_value:
                return header_value

    direct_key_candidates = (
        "sessionKey",
        "session_key",
        "sessionId",
        "session_id",
        "conversationId",
        "conversation_id",
        "chatId",
        "chat_id",
    )
    nested_key_candidates = ("metadata", "extra_body")

    for key in direct_key_candidates:
        value = original_request.get(key)
        if value not in (None, ""):
            return str(value)

    for container_key in nested_key_candidates:
        container = original_request.get(container_key)
        if not isinstance(container, dict):
            continue
        for key in direct_key_candidates:
            value = container.get(key)
            if value not in (None, ""):
                return str(value)

    return "default"


def substitute_env_vars(value: Any) -> Any:
    """Recursively substitute ``${VAR_NAME}`` placeholders from the environment."""

    if isinstance(value, str):
        return ENV_PATTERN.sub(lambda match: os.environ.get(match.group(1), ""), value)
    if isinstance(value, list):
        return [substitute_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {key: substitute_env_vars(item) for key, item in value.items()}
    return value


def setup_logging(root_dir: Path, debug_logging: bool = False) -> logging.Logger:
    """Configure rotating file and stderr logging."""
    global _LAST_DEBUG_SESSION_KEY

    logs_dir = root_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    _LAST_DEBUG_SESSION_KEY = None

    logger = logging.getLogger("function_router")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    file_handler = RotatingFileHandler(
        logs_dir / "router.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)

    request_logger = logging.getLogger(REQUEST_LOGGER_NAME)
    request_logger.setLevel(logging.INFO)
    request_logger.handlers.clear()
    request_logger.propagate = False
    request_logger.addHandler(file_handler)
    request_logger.addHandler(stderr_handler)

    debug_logger = logging.getLogger(DEBUG_LOGGER_NAME)
    debug_logger.setLevel(logging.DEBUG)
    debug_logger.handlers.clear()
    debug_logger.propagate = False
    if debug_logging:
        debug_handler = RecreatingRotatingFileHandler(
            logs_dir / "router.debug.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        debug_handler.setFormatter(logging.Formatter(fmt="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        debug_logger.addHandler(debug_handler)

    return logger


def load_config(config_path: Path) -> AppConfig:
    """Load and validate configuration from JSON."""

    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(f"config file not found: {config_path}") from exc
    except OSError as exc:
        raise RuntimeError(f"failed to read config file: {config_path}") from exc

    try:
        raw_data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid config JSON in {config_path}: {exc}") from exc

    data = substitute_env_vars(raw_data)
    root_dir = config_path.expanduser().resolve().parent

    try:
        routing_data = data.get("routing", data.get("qwen"))
        if routing_data is None:
            raise KeyError("routing")
        routing = ModelConfig(**routing_data)
        upstream = ModelConfig(**data["upstream"])
        completion_cfg = data.get("fr_completion_check", {})
        completion_mode = str(completion_cfg.get("mode", "permissive"))
        if completion_mode not in {"permissive", "strict"}:
            raise RuntimeError(
                f"invalid fr_completion_check.mode in {config_path}: {completion_mode}"
            )
        delegation_cfg = data.get("delegate_tools_to_openclaw", {"enabled": True})
        delegate_to_openclaw = True
        delegate_tools: list[str] | None = None
        if isinstance(delegation_cfg, bool):
            delegate_to_openclaw = delegation_cfg
        elif isinstance(delegation_cfg, dict):
            delegate_to_openclaw = bool(delegation_cfg.get("enabled", True))
            configured_tools = delegation_cfg.get("tools")
            if configured_tools is None:
                delegate_tools = None
            elif isinstance(configured_tools, list):
                delegate_tools = [
                    tool_name
                    for tool_name in configured_tools
                    if isinstance(tool_name, str) and tool_name
                ] or None
            else:
                raise RuntimeError(
                    f"invalid delegate_tools_to_openclaw.tools in {config_path}"
                )
        else:
            raise RuntimeError(
                f"invalid delegate_tools_to_openclaw in {config_path}"
            )
        return AppConfig(
            listen_host=data["listen_host"],
            listen_port=int(data["listen_port"]),
            routing=routing,
            upstream=upstream,
            functions_file=data["functions_file"],
            scripts_dir=data["scripts_dir"],
            max_tool_rounds=int(data["max_tool_rounds"]),
            tool_exec_timeout_s=int(data["tool_exec_timeout_s"]),
            root_dir=root_dir,
            config_path=config_path,
            tools_base_dir=data.get("tools_base_dir"),
            fr_completion_check=bool(
                data.get("fr_completion_check", {}).get("enabled", True)
                or data.get("qwen_completion_check", {}).get("enabled", False)
            ),
            fr_completion_check_mode=completion_mode,
            fr_completion_check_always_true=bool(
                data.get("fr_completion_check", {}).get("always_true", False)
            ),
            fr_context_history=bool(
                data.get("fr_context_history", {}).get("enabled", True)
                or data.get("qwen_context_history", {}).get("enabled", False)
            ),
            fr_context_preserve=bool(
                data.get("fr_context_preserve", {}).get("enabled", False)
                or data.get("qwen_context_preserve", {}).get("enabled", False)
            ),
            debug_logging=bool(
                data.get("debug_logging", {}).get("enabled", False)
            ),
            routing_timeout_s=float(data.get("routing_timeout_s", 10.0)),
            delegate_to_openclaw=delegate_to_openclaw,
            delegate_tools=delegate_tools,
        )
    except KeyError as exc:
        raise RuntimeError(f"missing config key: {exc.args[0]}") from exc
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid config structure in {config_path}: {exc}") from exc


def load_tools(functions_path: Path) -> list[dict[str, Any]]:
    """Load JSONL functions and convert them to OpenAI tools format."""

    if not functions_path.exists():
        raise RuntimeError(f"functions file not found: {functions_path}")

    tools: list[dict[str, Any]] = []
    try:
        with functions_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    function_obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"failed parsing {functions_path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(function_obj, dict):
                    raise RuntimeError(
                        f"invalid function object at {functions_path}:{line_number}"
                    )
                tools.append({"type": "function", "function": function_obj})
    except OSError as exc:
        raise RuntimeError(f"failed reading functions file: {functions_path}") from exc

    seen_names = {
        tool.get("function", {}).get("name")
        for tool in tools
        if isinstance(tool.get("function"), dict)
    }
    for builtin_tool in get_builtin_tools():
        builtin_name = builtin_tool["function"].get("name")
        if builtin_name in seen_names:
            continue
        tools.append(builtin_tool)
        seen_names.add(builtin_name)

    return tools


def now_iso() -> str:
    """Return a UTC timestamp string for request logs."""

    return datetime.now(timezone.utc).isoformat()


def _truncate(text: str, max_len: int = 1024) -> str:
    """Return text capped to max_len with an explicit truncation marker."""

    if not isinstance(text, str):
        text = str(text)
    if len(text) > max_len:
        return text[:max_len] + "...[truncated]"
    return text


def _debug_log(event: str, **fields: Any) -> None:
    """Deprecated metadata logger kept as a no-op for transcript-only debug logs."""

    return


def _append_debug_entry(lines: list[str], prefix: str, label: str, content: Any) -> None:
    text = "" if content is None else str(content)
    text_lines = text.splitlines() or [""]
    lines.append(f"{prefix}{label}: {text_lines[0]}")
    for continuation in text_lines[1:]:
        lines.append(f"{prefix}  {continuation}")


def _debug_log_messages(
    label: str,
    messages: list[dict[str, Any]],
    *,
    offset: int = 0,
) -> None:
    """Write current-turn debug logs for non-upstream routing."""

    if not (STATE.config and STATE.config.debug_logging):
        return

    subset = messages[offset:]
    if not subset:
        return

    logger = logging.getLogger(DEBUG_LOGGER_NAME)
    lines: list[str] = []

    for msg in subset:
        role = (msg.get("role") or "unknown").upper()
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls")

        if role == "USER":
            lines.append(f"USER: {content}")
        elif role == "ASSISTANT":
            if tool_calls:
                for tool_call in tool_calls:
                    function_meta = tool_call.get("function", {})
                    name = function_meta.get("name", "?")
                    arguments = function_meta.get("arguments", "")
                    lines.append(f"TOOL: {name}({arguments})")
            elif content:
                lines.append(f"ASSISTANT: {content}")
        elif role == "TOOL":
            name = msg.get("name") or msg.get("tool_call_id") or "?"
            lines.append(f"TOOL RESULT [{name}]: {content}")

    if lines:
        logger.debug("\n".join(lines))


def _debug_message_content(msg: dict[str, Any]) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                value = item.get("text")
                if isinstance(value, str):
                    text_parts.append(value)
        text = "".join(text_parts)
    else:
        text = ""

    if (msg.get("role") or "").lower() == "user":
        return _strip_openclaw_metadata(text)
    return text


def _debug_log_upstream_context(
    pending_messages: list[dict[str, Any]],
    current_user_message: dict[str, Any] | None,
    assistant_content: str,
    *,
    pending_before: int = 0,
    pending_injected: int = 0,
    pending_after: int | None = None,
) -> None:
    """Write only FR pending context plus the current user and upstream response."""

    if not (STATE.config and STATE.config.debug_logging):
        return

    logger = logging.getLogger(DEBUG_LOGGER_NAME)
    lines: list[str] = ["*** START UPSTREAM ***"]
    lines.append(f"\tPENDING_UPSTREAM_TURNS before: {pending_before}")
    lines.append(f"\tPENDING_UPSTREAM_TURNS injected: {pending_injected}")
    if pending_after is not None:
        lines.append(f"\tPENDING_UPSTREAM_TURNS after_clear: {pending_after}")

    user_index = 0
    assistant_index = 0
    for msg in pending_messages:
        role = (msg.get("role") or "unknown").upper()
        content = _debug_message_content(msg)
        if role == "USER":
            user_index += 1
            lines.append(f"\tUSER{user_index}: {content}")
        elif role == "ASSISTANT" and content:
            assistant_index += 1
            lines.append(f"\tASSISTANT{assistant_index}: {content}")

    if current_user_message is not None:
        user_index += 1
        lines.append(f"\tUSER{user_index}: {_debug_message_content(current_user_message)}")

    lines.append(f"\tASSISTANT last: {assistant_content}")
    lines.append("*** FINISHED UPSTREAM ***")
    logger.debug("\n".join(lines))


def _extract_upstream_assistant_content(response_bytes: bytes, content_type: str) -> str:
    text = response_bytes.decode("utf-8", errors="replace")
    content_parts: list[str] = []

    if "text/event-stream" in content_type.lower():
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            data_text = line[5:].strip()
            if not data_text or data_text == "[DONE]":
                continue
            try:
                data = json.loads(data_text)
            except json.JSONDecodeError:
                continue
            for choice in data.get("choices") or []:
                delta = choice.get("delta") or {}
                message = choice.get("message") or {}
                content = delta.get("content") or message.get("content")
                if isinstance(content, str):
                    content_parts.append(content)
        return "".join(content_parts)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text
    for choice in data.get("choices") or []:
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            content_parts.append(content)
    return "".join(content_parts)


def _has_visible_assistant_reply(content: str) -> bool:
    visible = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
    return bool(visible.strip())


def _debug_log_session(session_key: str) -> None:
    """Write a transcript section header when the active session changes."""
    global _LAST_DEBUG_SESSION_KEY

    if not (STATE.config and STATE.config.debug_logging):
        return

    logger = logging.getLogger(DEBUG_LOGGER_NAME)
    file_empty = True
    for handler in logger.handlers:
        base_filename = getattr(handler, "baseFilename", None)
        if base_filename and os.path.exists(base_filename):
            file_empty = os.path.getsize(base_filename) == 0
            break

    if session_key == _LAST_DEBUG_SESSION_KEY and not file_empty:
        return

    if not file_empty:
        logger.debug("")
    logger.debug("===== SESSION_KEY ======")
    logger.debug(session_key)
    _LAST_DEBUG_SESSION_KEY = session_key

_MEMORIES_RE = re.compile(r"<relevant-memories>.*?</relevant-memories>", re.DOTALL)
_INGEST_REPLY_ASSIST_RE = re.compile(
    r"<ingest-reply-assist\b[^>]*>.*?</ingest-reply-assist>", re.DOTALL | re.IGNORECASE
)
_SENDER_RE = re.compile(
    r"Sender \(untrusted metadata\):\s*```json\s*\{.*?\}\s*```", re.DOTALL
)
_CONVERSATION_INFO_RE = re.compile(
    r"Conversation info \(untrusted metadata\):\s*```json\s*\{.*?\}\s*```",
    re.DOTALL,
)
_TIMESTAMP_RE = re.compile(r"^\[.*?\]\s*", re.MULTILINE)
_TRANSCRIPT_SPEAKER_PREFIX_RE = re.compile(
    r"^\s*(?:System|User|Assistant)\s*:\s*\[[^\]]+\]\s*.*?\b(?:message|said|says)\b(?:\s+from\s+session\s+[^:]+)?\s*:\s*",
    re.IGNORECASE,
)
_SESSION_ECHO_RE = re.compile(
    r"(?:^|\s+)(?:session\s+)?[A-Za-z0-9_-]{6,}\s*:\s*"
)
_BRACKET_WRAPPER_RE = re.compile(r"\[[^\[\]]*\]|\{[^{}]*\}|<[^<>]*>")
_SINGLE_SESSION_ECHO_RE = re.compile(
    r"^(?:session\s+)?([A-Za-z0-9_-]{6,})\s*:\s*(.+)$",
    re.DOTALL,
)
_DUPLICATE_SESSION_ECHO_RE = re.compile(
    r"^([A-Za-z0-9_-]{6,})\s*:\s*(.+?)\s+\1\s*:\s*\2$",
    re.DOTALL,
)
_WRAPPER_KEYWORDS = (
    "Conversation info (untrusted metadata)",
    "Queued messages while agent was busy",
    "Xiaomai message from session",
)
_LAST_SESSION_ECHO_RE = re.compile(
    r"(?:^|\s)(?:session\s+)?([A-Za-z0-9_-]{6,})\s*:\s*([^\n]+?)\s*$",
    re.DOTALL,
)
_WORKSPACE_BOOTSTRAP_RE = re.compile(
    r"\n*Some workspace bootstrap files were truncated before injection\..*$",
    re.DOTALL,
)


def _extract_last_session_echo_for_wrappers(text: str) -> str | None:
    """For known wrappers, extract the trailing `session_id: message` payload."""

    if not any(keyword in text for keyword in _WRAPPER_KEYWORDS):
        return None
    match = _LAST_SESSION_ECHO_RE.search(text)
    if not match:
        return None
    return match.group(2).strip()


def _drop_bracket_wrappers(text: str) -> str:
    """Remove simple bracketed wrappers like [x], {x}, <x>."""

    previous = None
    while text != previous:
        previous = text
        text = _BRACKET_WRAPPER_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_transcript_message(text: str) -> str:
    """Extract the raw spoken message from transcript-style wrappers."""

    transcript_match = _TRANSCRIPT_SPEAKER_PREFIX_RE.match(text)
    if transcript_match:
        text = text[transcript_match.end() :].strip()
        if not text:
            return text

    duplicate_match = _DUPLICATE_SESSION_ECHO_RE.match(text)
    if duplicate_match:
        return duplicate_match.group(2).strip()

    single_match = _SINGLE_SESSION_ECHO_RE.match(text)
    if single_match:
        return single_match.group(2).strip()

    parts = _SESSION_ECHO_RE.split(text)
    normalized_parts = [part.strip() for part in parts if part.strip()]
    if len(normalized_parts) >= 2 and len(set(normalized_parts)) == 1:
        return normalized_parts[0]

    return text


def _strip_openclaw_metadata(text: str) -> str:
    """Strip OpenClaw-injected metadata, returning only the raw user input."""

    text = _MEMORIES_RE.sub("", text)
    text = _INGEST_REPLY_ASSIST_RE.sub(" ", text)
    text = _SENDER_RE.sub("", text)
    text = _CONVERSATION_INFO_RE.sub("", text)
    text = _WORKSPACE_BOOTSTRAP_RE.sub("", text)
    text = _extract_transcript_message(text.strip())
    wrapped_text = _extract_last_session_echo_for_wrappers(text)
    if wrapped_text is not None:
        text = wrapped_text
    text = _TIMESTAMP_RE.sub("", text)
    text = _drop_bracket_wrappers(text)
    text = re.sub(r"^\s*:\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return _extract_transcript_message(text)




def extract_user_text(messages: list[dict[str, Any]]) -> str | None:
    """Extract the latest user message text from OpenAI chat messages."""

    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return _strip_openclaw_metadata(content)
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_value = item.get("text")
                    if isinstance(text_value, str):
                        text_parts.append(text_value)
            return _strip_openclaw_metadata("".join(text_parts))
        return None
    return None


async def build_http_client() -> httpx.AsyncClient:
    """Create a shared HTTP client."""

    return httpx.AsyncClient(follow_redirects=True)


async def qwen_health_check() -> bool:
    """Check whether the local Qwen endpoint is reachable."""

    if STATE.http_client is None or STATE.config is None:
        return False

    url = f"{STATE.config.routing.base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {STATE.config.routing.api_key}"}
    try:
        response = await STATE.http_client.get(url, headers=headers, timeout=5.0)
        return response.status_code < 500
    except httpx.HTTPError:
        return False


def _schema_tool_name(tool: dict[str, Any]) -> str | None:
    """从OpenAI格式或普通格式的工具定义中读取工具名称。"""

    if not isinstance(tool, dict):
        return None

    function_meta = tool.get("function")

    if isinstance(function_meta, dict):
        function_name = function_meta.get("name")

        if isinstance(function_name, str) and function_name:
            return function_name

    direct_name = tool.get("name")

    if isinstance(direct_name, str) and direct_name:
        return direct_name

    return None


def _deterministic_employee_tool(user_text: str) -> str | None:
    # official-subagent-route-aliases
    official_subagent_text = (user_text or "").lower()

    official_subagent_routes = (
        (
            (
                "能碳诊断报告 subagent",
                "能碳报告 subagent",
                "energy carbon delivery subagent",
            ),
            "energy_carbon_delivery_subagent",
        ),
        (
            (
                "电费账单稽核 subagent",
                "电费稽核 subagent",
                "power bill audit subagent",
            ),
            "power_bill_audit_subagent",
        ),
        (
            (
                "节能改造项目管理 subagent",
                "改造项目管理 subagent",
                "retrofit project manager subagent",
            ),
            "retrofit_project_manager_subagent",
        ),
        (
            (
                "碳资产运营 subagent",
                "碳资产管理 subagent",
                "carbon asset operations subagent",
            ),
            "carbon_asset_operations_subagent",
        ),
    )

    for aliases, tool_name in official_subagent_routes:
        if any(
            alias in official_subagent_text
            for alias in aliases
        ):
            return tool_name

    # energy-carbon-delivery-deterministic-route
    delivery_route_text = (user_text or "").lower()

    delivery_keywords = (
        "ai能碳诊断报告 Subagent",
        "能碳诊断报告 Subagent",
        "能碳诊断报告",
        "能碳报告",
        "能源碳排放报告",
        "生成并交付",
        "生成报告并交付",
        "报告下载地址",
        "报告下载链接",
        "导出pdf",
        "pdf报告",
        "客户交付邮件",
        "交付执行日志",
    )

    power_exclusion_keywords = (
        "电费账单",
        "电费稽核",
        "电费核验",
        "电度电费",
        "基本电费",
        "合同容量",
        "最大需量",
        "功率因数",
        "峰平谷电价",
    )

    if (
        any(
            keyword in delivery_route_text
            for keyword in delivery_keywords
        )
        and not any(
            keyword in delivery_route_text
            for keyword in power_exclusion_keywords
        )
    ):
        return "energy_carbon_delivery_subagent"

    # carbon-asset-operations-deterministic-route
    carbon_route_text = (user_text or "").lower()

    carbon_route_keywords = (
        "碳资产运营 Subagent",
        "碳资产",
        "ccer",
        "碳配额",
        "履约注销",
        "碳资产注销",
        "碳资产登记",
        "碳资产组合",
        "组合估值",
        "碳资产余额",
        "碳资产交易",
        "碳交易记录",
        "买入碳资产",
        "卖出碳资产",
        "转入碳资产",
        "转出碳资产",
        "ca-",
        "ct-",
    )

    if any(
        keyword in carbon_route_text
        for keyword in carbon_route_keywords
    ):
        return "carbon_asset_operations_subagent"

    # retrofit-project-manager-deterministic-route
    retrofit_route_text = (user_text or "").lower()

    retrofit_route_keywords = (
        "改造项目",
        "项目管理员工",
        "项目管理",
        "项目进度",
        "项目任务",
        "项目预算",
        "项目里程碑",
        "项目风险",
        "创建项目",
        "项目立项",
        "新增任务",
        "添加任务",
        "更新任务",
        "查询项目",
        "retrofit",
        "rp-",
        "task-",
    )

    if any(
        keyword in retrofit_route_text
        for keyword in retrofit_route_keywords
    ):
        return "retrofit_project_manager_subagent"
    """为核心AI业务 Subagent提供稳定、可扩展的确定性路由。"""

    text = (user_text or "").strip().lower()

    if not text:
        return None

    power_business_keywords = (
        "电费稽核",
        "电费账单",
        "电费单",
        "电度电费",
        "基本电费",
        "合同容量",
        "最大需量",
        "功率因数",
        "峰平谷",
        "综合电价",
        "容量单价",
        "容量优化",
        "稽核工单",
        "电费台账",
    )

    power_action_keywords = (
        "核验",
        "稽核",
        "审计",
        "复核",
        "检查",
        "测算",
        "创建工单",
        "登记台账",
    )

    matched_business_keywords = sum(
        1
        for keyword in power_business_keywords
        if keyword in text
    )

    has_power_action = any(
        keyword in text
        for keyword in power_action_keywords
    )

    # 至少命中两个电费业务特征，并且用户要求执行相关动作，
    # 才确定性分发，避免普通聊天中的偶然关键词误触发。
    if matched_business_keywords >= 2 and has_power_action:
        return "power_bill_audit_subagent"

    return None


async def call_qwen(
    messages: list[dict[str, Any]],
    *,
    forced_tool_name: str | None = None,
) -> dict[str, Any]:
    """Send a non-streaming chat completion request to the local Qwen endpoint.

    Retries once on timeout with a small random jitter to ride out brief
    routing-model latency spikes (e.g. KV cache warmup). After the retry is
    exhausted the timeout propagates so the caller can fall back to upstream.
    """

    if STATE.http_client is None or STATE.config is None or STATE.tools is None:
        raise RuntimeError("application state is not initialized")

    selected_tools = STATE.tools

    if forced_tool_name:
        selected_tools = [
            tool
            for tool in STATE.tools
            if _schema_tool_name(tool) == forced_tool_name
        ]

        if not selected_tools:
            raise RuntimeError(
                f"deterministic route tool not loaded: "
                f"{forced_tool_name}"
            )

    payload = {
        "model": STATE.config.routing.model,
        "messages": messages,
        "tools": selected_tools,
        "stream": False,
        "temperature": 0.0,
        "repetition_penalty": 1.2,
        "frequency_penalty": 0.2,
        "parallel_tool_calls": False,
        "thinking": {"type": "disabled"},
    }

    if forced_tool_name:
        payload["tool_choice"] = {
            "type": "function",
            "function": {
                "name": forced_tool_name
            },
        }

    headers = {
        "Authorization": f"Bearer {STATE.config.routing.api_key}",
        "Content-Type": "application/json",
    }
    url = f"{STATE.config.routing.base_url.rstrip('/')}/chat/completions"
    timeout_s = STATE.config.routing_timeout_s

    try:
        response = await STATE.http_client.post(
            url, json=payload, headers=headers, timeout=timeout_s,
        )
    except httpx.TimeoutException as exc:
        if STATE.logger is not None:
            STATE.logger.warning("routing model timeout (%.1fs), retrying once", timeout_s)
        await asyncio.sleep(random.uniform(0.1, 0.5))
        try:
            response = await STATE.http_client.post(
                url, json=payload, headers=headers, timeout=timeout_s,
            )
        except httpx.TimeoutException as retry_exc:
            if STATE.logger is not None:
                STATE.logger.warning("routing model timeout on retry, giving up")
            raise retry_exc from exc

    response.raise_for_status()
    return response.json()


async def warmup_qwen() -> bool:
    """Warm the Qwen KV cache with a deterministic one-token request."""

    if STATE.http_client is None or STATE.config is None or STATE.tools is None:
        return False

    headers = {
        "Authorization": f"Bearer {STATE.config.routing.api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": STATE.config.routing.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "hello"},
        ],
        "tools": STATE.tools,
        "stream": False,
        "max_tokens": 1,
        "temperature": 0.0,
        "repetition_penalty": 1.2,
        "frequency_penalty": 0.2,
        "parallel_tool_calls": False,
        "thinking": {"type": "disabled"},
    }
    try:
        response = await STATE.http_client.post(
            f"{STATE.config.routing.base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
            timeout=STATE.config.routing_timeout_s,
        )
        response.raise_for_status()
        return True
    except httpx.HTTPError as exc:
        if STATE.logger is not None:
            STATE.logger.warning("qwen warmup failed: %s", exc)
        return False


def _validate_function_name(name: str) -> bool:
    """Validate function_name contains only safe characters (letters, digits, underscores)."""
    return bool(re.match(r"^[a-zA-Z0-9_]+$", name))


async def execute_tool(function_name: str, arguments_json: str) -> dict[str, Any]:
    """Execute a shell script for a tool call and parse its JSON stdout."""

    if STATE.config is None:
        raise RuntimeError("application config not initialized")

    if not _validate_function_name(function_name):
        return {"error": f"invalid function name: {function_name}"}

    if is_builtin_tool(function_name):
        return await asyncio.to_thread(
            execute_builtin_tool,
            function_name,
            arguments_json,
            STATE.config.tool_exec_timeout_s,
        )

    script_path = (STATE.config.resolved_scripts_dir / f"{function_name}.sh").resolve()
    # Ensure script is within the expected directory (prevent directory traversal)
    if not str(script_path).startswith(str(STATE.config.resolved_scripts_dir.resolve())):
        return {"error": f"script path outside scripts directory: {function_name}"}

    if not script_path.exists():
        return {"error": f"script not found: {function_name}.sh"}

    # Build env with FR_TOOLS_BASE_DIR if configured
    env = dict(os.environ)
    if STATE.config.tools_base_dir:
        env["FR_TOOLS_BASE_DIR"] = STATE.config.tools_base_dir

    # subprocess.run inside a worker thread keeps the event loop free and,
    # unlike asyncio.create_subprocess_exec, also works when the loop runs in a
    # non-main thread (e.g. under Starlette's TestClient).
    def _run_script() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(script_path)],
            input=arguments_json,
            text=True,
            capture_output=True,
            timeout=STATE.config.tool_exec_timeout_s,
            env=env,
        )

    try:
        completed = await asyncio.to_thread(_run_script)
    except subprocess.TimeoutExpired:
        return {"error": "execution timeout"}

    stderr_text = completed.stderr.strip()
    stdout_text = completed.stdout.strip()

    parsed_output: Any | None = None
    if stdout_text:
        try:
            parsed_output = json.loads(stdout_text)
        except json.JSONDecodeError:
            parsed_output = None

    if completed.returncode != 0:
        logger = logging.getLogger("function_router")
        logger.warning(
            "tool %s failed (rc=%s) stdout=%r stderr=%r",
            function_name, completed.returncode, stdout_text[:500], stderr_text[:500],
        )
        if isinstance(parsed_output, dict):
            return parsed_output
        return {
            "error": stderr_text or "script execution failed",
            "returncode": completed.returncode,
            **({"stdout": stdout_text} if stdout_text else {}),
        }

    if not stdout_text:
        return {}

    if parsed_output is None:
        return {"error": "invalid JSON output", "stdout": stdout_text}

    if isinstance(parsed_output, dict):
        return parsed_output
    return {"result": parsed_output}


def _tool_call_function_name(tool_call: dict[str, Any]) -> str:
    function_meta = tool_call.get("function") or {}
    name = function_meta.get("name")
    return name if isinstance(name, str) else ""


def _normalize_tool_call_for_response(
    tool_call: dict[str, Any],
    *,
    fallback_id: str,
) -> dict[str, Any]:
    """Return an OpenAI assistant.tool_calls item without rewriting arguments."""

    function_meta = dict(tool_call.get("function") or {})
    arguments = function_meta.get("arguments")
    if arguments is None:
        function_meta["arguments"] = "{}"
    elif not isinstance(arguments, str):
        function_meta["arguments"] = json.dumps(
            arguments,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    return {
        "id": tool_call.get("id") or fallback_id,
        "type": tool_call.get("type") or "function",
        "function": function_meta,
    }


def _tool_calls_are_delegated(
    tool_calls: list[dict[str, Any]],
    delegated_tool_names: set[str],
) -> bool:
    if not tool_calls or not delegated_tool_names:
        return False
    return all(_tool_call_function_name(tool_call) in delegated_tool_names for tool_call in tool_calls)


def _delegated_tool_names() -> set[str]:
    """Return the configured set of tool names to delegate to OpenClaw."""

    if STATE.config is None or not STATE.config.delegate_to_openclaw:
        return set()
    if STATE.config.delegate_tools:
        return set(STATE.config.delegate_tools)

    names: set[str] = set()
    for tool in STATE.tools or []:
        if not isinstance(tool, dict):
            continue
        function_meta = tool.get("function")
        if not isinstance(function_meta, dict):
            continue
        name = function_meta.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


def _find_delegated_tool_continuation(
    messages: list[dict[str, Any]],
    delegated_names: set[str],
) -> tuple[list[dict[str, Any]], list[str]] | None:
    """Return assistant tool call(s) plus trailing OpenClaw tool result messages."""

    if not messages or not delegated_names:
        return None

    index = len(messages) - 1
    if not isinstance(messages[index], dict) or messages[index].get("role") != "tool":
        return None
    while index >= 0 and isinstance(messages[index], dict) and messages[index].get("role") == "tool":
        index -= 1
    tool_messages = messages[index + 1 :]
    if not tool_messages:
        return None

    observed_ids = [
        message.get("tool_call_id")
        for message in tool_messages
        if isinstance(message.get("tool_call_id"), str) and message.get("tool_call_id")
    ]
    observed_id_set = set(observed_ids)
    observed_names = {
        message.get("name")
        for message in tool_messages
        if isinstance(message.get("name"), str) and message.get("name")
    }

    for previous in reversed(messages[: index + 1]):
        if not isinstance(previous, dict) or previous.get("role") != "assistant":
            continue
        previous_tool_calls = previous.get("tool_calls") or []
        matched_calls: list[dict[str, Any]] = []
        for call_index, tool_call in enumerate(previous_tool_calls):
            if not isinstance(tool_call, dict):
                continue
            function_name = _tool_call_function_name(tool_call)
            if function_name not in delegated_names:
                continue
            tool_call_id = tool_call.get("id")
            if (
                observed_id_set
                and isinstance(tool_call_id, str)
                and tool_call_id
                and tool_call_id not in observed_id_set
            ):
                continue
            if not observed_id_set and observed_names and function_name not in observed_names:
                continue
            matched_calls.append(
                _normalize_tool_call_for_response(
                    tool_call,
                    fallback_id=f"call_{function_name or 'tool'}_{call_index}",
                )
            )

        if not matched_calls:
            continue

        name_by_id = {
            tool_call.get("id"): _tool_call_function_name(tool_call)
            for tool_call in matched_calls
            if isinstance(tool_call.get("id"), str)
        }
        matched_names = [
            _tool_call_function_name(tool_call)
            for tool_call in matched_calls
            if _tool_call_function_name(tool_call)
        ]
        normalized_tools: list[dict[str, Any]] = []
        for tool_message in tool_messages:
            normalized_tool = dict(tool_message)
            if not normalized_tool.get("name"):
                tool_call_id = normalized_tool.get("tool_call_id")
                if isinstance(tool_call_id, str) and tool_call_id in name_by_id:
                    normalized_tool["name"] = name_by_id[tool_call_id]
                elif len(matched_names) == 1:
                    normalized_tool["name"] = matched_names[0]
            normalized_tools.append(normalized_tool)

        assistant_message = {
            "role": "assistant",
            "content": previous.get("content") or "",
            "tool_calls": matched_calls,
        }
        return [assistant_message, *normalized_tools], observed_ids

    return None



def _extract_business_number(
    text: str,
    patterns: tuple[str, ...],
) -> float | None:
    """从中文业务描述中提取数字。"""

    normalized = text.replace(",", "").replace("，", "")

    for pattern in patterns:
        match = re.search(
            pattern,
            normalized,
            flags=re.IGNORECASE,
        )

        if match:
            try:
                return float(match.group(1))
            except (TypeError, ValueError):
                continue

    return None





def _extract_energy_carbon_delivery_arguments(
    user_text: str,
) -> dict[str, Any]:
    """提取能碳诊断报告 Subagent的项目查询名称。"""

    import re as _re

    text = (user_text or "").strip()

    def clean_name(value: str) -> str:
        name = value.strip(
            " \t\r\n：:，,。；;“”\"'"
        )

        name = _re.sub(
            r"^(?:请|给|针对)\s*",
            "",
            name,
        )

        name = _re.sub(
            r"\s*20\d{2}年度?$",
            "",
            name,
        )

        return name.strip()

    project_query = ""

    # 1. 项目名称：朝阳云庭酒店
    match = _re.search(
        r"(?:项目名称|项目名)[：:]\s*"
        r"([^，。；;\n]+)",
        text,
    )

    if match:
        project_query = clean_name(
            match.group(1)
        )

    # 2. 为朝阳云庭酒店生成报告
    # (?<!作) 避免匹配“作为能碳诊断报告 Subagent”
    if not project_query:
        match = _re.search(
            r"(?<!作)为\s*"
            r"([^，。；;\n]{2,60}?)"
            r"(?=\s*(?:生成|制作|导出|交付))",
            text,
        )

        if match:
            project_query = clean_name(
                match.group(1)
            )

    # 3. 调取/读取/查找某项目
    if not project_query:
        match = _re.search(
            r"(?:调取|读取|查找)\s*[“\"']?"
            r"([^”\"'，。；;\n]{2,60}?)"
            r"(?=\s*(?:项目|数据|报告|，|。|$))",
            text,
        )

        if match:
            project_query = clean_name(
                match.group(1)
            )

    # 4. 根据酒店、大厦、园区等名称兜底
    if not project_query:
        candidates = _re.findall(
            r"[\u4e00-\u9fffA-Za-z0-9·_-]{2,30}?"
            r"(?:酒店|大厦|园区|写字楼)",
            text,
        )

        for candidate in candidates:
            candidate = clean_name(candidate)

            if (
                "员工" not in candidate
                and "报告" not in candidate
                and "AI" not in candidate
            ):
                project_query = candidate
                break

    invalid_names = {
        "",
        "项目",
        "能碳报告",
        "AI能碳报告",
        "能碳诊断报告 Subagent",
        "能碳诊断报告 Subagent",
    }

    if project_query in invalid_names:
        raise ValueError(
            "未能从请求中识别有效项目名称"
        )

    return {
        "project_query": project_query,
    }


def _extract_carbon_asset_arguments(
    user_text: str,
) -> dict[str, Any]:
    """从自然语言中提取碳资产运营参数。"""

    import re as _re

    text = (user_text or "").strip()
    arguments: dict[str, Any] = {}

    asset_id_match = _re.search(
        r"\bCA-\d{14}-[A-Za-z0-9]{6}\b",
        text,
        flags=_re.IGNORECASE,
    )

    if asset_id_match:
        arguments["asset_id"] = (
            asset_id_match.group(0).upper()
        )

    if any(
        keyword in text
        for keyword in (
            "交易记录",
            "交易台账",
            "操作记录",
            "历史交易",
        )
    ):
        arguments["action"] = "list_transactions"

    elif any(
        keyword in text
        for keyword in (
            "资产组合",
            "组合估值",
            "资产汇总",
            "全部碳资产",
            "所有碳资产",
            "碳资产总量",
        )
    ):
        arguments["action"] = "get_portfolio"

    elif any(
        keyword in text
        for keyword in (
            "履约注销",
            "碳资产注销",
            "注销碳资产",
            "注销",
        )
    ):
        arguments["action"] = "retire_asset"

    elif any(
        keyword in text
        for keyword in (
            "登记碳资产",
            "注册碳资产",
            "新增碳资产",
            "碳资产登记",
        )
    ):
        arguments["action"] = "register_asset"

    elif any(
        keyword in text
        for keyword in (
            "买入",
            "卖出",
            "转入",
            "转出",
            "记录交易",
            "碳资产交易",
        )
    ):
        arguments["action"] = "record_transaction"

    elif asset_id_match:
        arguments["action"] = "get_asset"

    else:
        arguments["action"] = "get_portfolio"

    transaction_types = (
        ("履约注销", "retire"),
        ("注销", "retire"),
        ("买入", "buy"),
        ("卖出", "sell"),
        ("转入", "transfer_in"),
        ("转出", "transfer_out"),
    )

    for keyword, transaction_type in transaction_types:
        if keyword in text:
            arguments["transaction_type"] = transaction_type
            break

    asset_name_match = _re.search(
        r"(?:资产名称|资产名)[：:]\s*"
        r"([^，。；;\n]+)",
        text,
    )

    if asset_name_match:
        arguments["asset_name"] = (
            asset_name_match.group(1).strip()
        )

    owner_match = _re.search(
        r"(?:所有者|资产所有者|持有人|持有主体)"
        r"[：:]\s*"
        r"([^，。；;\n]+)",
        text,
    )

    if owner_match:
        arguments["owner"] = (
            owner_match.group(1).strip()
        )

    project_match = _re.search(
        r"(?:所属项目|项目名称)[：:]\s*"
        r"([^，。；;\n]+)",
        text,
    )

    if project_match:
        arguments["project_name"] = (
            project_match.group(1).strip()
        )

    year_match = _re.search(
        r"(?:签发年份|资产年份|年份)"
        r"[：:]?\s*(20\d{2})",
        text,
    )

    if year_match:
        arguments["vintage_year"] = (
            year_match.group(1)
        )

    if "CCER" in text.upper():
        arguments["asset_type"] = "CCER"
    elif "碳配额" in text:
        arguments["asset_type"] = "碳配额"

    quantity_match = _re.search(
        r"(?:登记数量|交易数量|注销数量|操作数量|数量)"
        r"[：:]?\s*"
        r"([\d,.]+)\s*"
        r"(?:tCO2e|tco2e|吨二氧化碳当量|吨)?",
        text,
        flags=_re.IGNORECASE,
    )

    if quantity_match:
        arguments["quantity_tco2e"] = float(
            quantity_match.group(1).replace(",", "")
        )

    market_price_match = _re.search(
        r"(?:市场单价|估值单价)"
        r"[：:]?\s*"
        r"([\d,.]+)\s*(?:元)?",
        text,
    )

    if market_price_match:
        arguments[
            "estimated_market_unit_price_cny"
        ] = float(
            market_price_match.group(1).replace(",", "")
        )

    acquisition_price_match = _re.search(
        r"(?:取得单价|购入单价|登记单价)"
        r"[：:]?\s*"
        r"([\d,.]+)\s*(?:元)?",
        text,
    )

    if acquisition_price_match:
        arguments[
            "acquisition_unit_price_cny"
        ] = float(
            acquisition_price_match.group(1).replace(",", "")
        )

    transaction_price_match = _re.search(
        r"(?:交易单价|注销单价|操作单价|单价)"
        r"[：:]?\s*"
        r"([\d,.]+)\s*(?:元)?",
        text,
    )

    if transaction_price_match:
        arguments["unit_price_cny"] = float(
            transaction_price_match.group(1).replace(",", "")
        )

    counterparty_match = _re.search(
        r"(?:交易对手|对手方)[：:]\s*"
        r"([^，。；;\n]+)",
        text,
    )

    if counterparty_match:
        arguments["counterparty"] = (
            counterparty_match.group(1).strip()
        )

    purpose_match = _re.search(
        r"(?:用途|注销用途|履约用途)[：:]\s*"
        r"([^，。；;\n]+)",
        text,
    )

    if purpose_match:
        arguments["purpose"] = (
            purpose_match.group(1).strip()
        )

    date_match = _re.search(
        r"\b(20\d{2})[-/年](\d{1,2})[-/月](\d{1,2})日?\b",
        text,
    )

    if date_match:
        arguments["transaction_date"] = (
            f"{int(date_match.group(1)):04d}-"
            f"{int(date_match.group(2)):02d}-"
            f"{int(date_match.group(3)):02d}"
        )

    limit_match = _re.search(
        r"(?:最近|前)\s*(\d+)\s*条",
        text,
    )

    if limit_match:
        arguments["limit"] = int(
            limit_match.group(1)
        )

    return arguments


def _extract_retrofit_project_arguments(
    user_text: str,
) -> dict[str, Any]:
    """从自然语言中提取节能改造项目管理 Subagent参数。"""

    import re as _re

    text = (user_text or "").strip()
    lower_text = text.lower()
    arguments: dict[str, Any] = {}

    project_id_match = _re.search(
        r"\bRP-\d{14}-[A-Za-z0-9]{6}\b",
        text,
        flags=_re.IGNORECASE,
    )

    task_id_match = _re.search(
        r"\bTASK-\d{14}-[A-Za-z0-9]{4}\b",
        text,
        flags=_re.IGNORECASE,
    )

    if project_id_match:
        arguments["project_id"] = (
            project_id_match.group(0).upper()
        )

    if task_id_match:
        arguments["task_id"] = (
            task_id_match.group(0).upper()
        )

    if any(
        keyword in text
        for keyword in (
            "项目列表",
            "所有项目",
            "全部项目",
            "列出项目",
        )
    ):
        arguments["action"] = "list_projects"

    elif any(
        keyword in text
        for keyword in (
            "新增任务",
            "添加任务",
            "创建任务",
        )
    ):
        arguments["action"] = "add_task"

    elif (
        any(
            keyword in text
            for keyword in (
                "更新任务",
                "任务进度",
                "完成任务",
                "任务状态",
                "登记费用",
                "实际费用",
            )
        )
        and task_id_match
    ):
        arguments["action"] = "update_task"

    elif any(
        keyword in text
        for keyword in (
            "创建改造项目",
            "创建项目",
            "建立改造项目",
            "项目立项",
            "立项",
        )
    ):
        arguments["action"] = "create_project"

    else:
        arguments["action"] = "get_project"

    labelled_project_name = _re.search(
        r"(?:项目名称|项目名)[：:]\s*"
        r"([^，。；;\n]+)",
        text,
    )

    if labelled_project_name:
        arguments["project_name"] = (
            labelled_project_name.group(1).strip()
        )
    elif arguments["action"] == "create_project":
        create_name_match = _re.search(
            r"(?:为|给)?"
            r"([^，。；;\n]{2,60}?)"
            r"(?:创建|建立|发起|立项)"
            r"(?:一个|一项)?"
            r"(?:节能降碳|节能|降碳|综合)?"
            r"改造项目",
            text,
        )

        if create_name_match:
            name = create_name_match.group(1).strip()
            name = _re.sub(
                r"^(请|作为节能改造项目管理 Subagent)",
                "",
                name,
            ).strip()

            arguments["project_name"] = name

    task_name_match = _re.search(
        r"(?:任务名称|任务名|新增任务|添加任务|创建任务)"
        r"[：:]?\s*"
        r"([^，。；;\n]+)",
        text,
    )

    if (
        task_name_match
        and arguments["action"] == "add_task"
    ):
        arguments["task_name"] = (
            task_name_match.group(1).strip()
        )

    manager_match = _re.search(
        r"(?:项目负责人|项目经理|负责人)"
        r"[：:]?\s*"
        r"([^，。；;\n]+)",
        text,
    )

    if manager_match:
        manager_value = manager_match.group(1).strip()

        if arguments["action"] == "create_project":
            arguments["project_manager"] = manager_value
        else:
            arguments["owner"] = manager_value

    budget_match = _re.search(
        r"(?:总预算|项目预算)"
        r"[：:]?\s*"
        r"([\d,.]+)\s*(万元|元)?",
        text,
    )

    if budget_match:
        value = float(
            budget_match.group(1).replace(",", "")
        )

        if budget_match.group(2) == "万元":
            value *= 10000

        arguments["budget_total_cny"] = value

    task_budget_match = _re.search(
        r"(?:任务预算)"
        r"[：:]?\s*"
        r"([\d,.]+)\s*(万元|元)?",
        text,
    )

    if task_budget_match:
        value = float(
            task_budget_match.group(1).replace(",", "")
        )

        if task_budget_match.group(2) == "万元":
            value *= 10000

        arguments["task_budget_cny"] = value

    actual_cost_match = _re.search(
        r"(?:实际费用|实际成本|已发生费用|登记费用)"
        r"[：:]?\s*"
        r"([\d,.]+)\s*(万元|元)?",
        text,
    )

    if actual_cost_match:
        value = float(
            actual_cost_match.group(1).replace(",", "")
        )

        if actual_cost_match.group(2) == "万元":
            value *= 10000

        arguments["actual_cost_cny"] = value

    progress_match = _re.search(
        r"(?:进度|完成度)"
        r"[：:]?\s*"
        r"(\d+(?:\.\d+)?)\s*%",
        text,
    )

    if progress_match:
        arguments["progress_percent"] = float(
            progress_match.group(1)
        )

    status_values = (
        "已完成",
        "进行中",
        "待开始",
        "暂停",
        "已取消",
    )

    for status_value in status_values:
        if status_value in text:
            arguments["status"] = status_value
            break

    if "高优先级" in text or "优先级高" in text:
        arguments["priority"] = "高"
    elif "低优先级" in text or "优先级低" in text:
        arguments["priority"] = "低"
    elif "中优先级" in text or "优先级中" in text:
        arguments["priority"] = "中"

    date_values = _re.findall(
        r"\b20\d{2}[-/年]\d{1,2}[-/月]\d{1,2}日?\b",
        text,
    )

    normalized_dates = []

    for date_value in date_values:
        normalized = (
            date_value
            .replace("年", "-")
            .replace("月", "-")
            .replace("日", "")
            .replace("/", "-")
        )

        parts = normalized.split("-")

        if len(parts) == 3:
            normalized_dates.append(
                f"{int(parts[0]):04d}-"
                f"{int(parts[1]):02d}-"
                f"{int(parts[2]):02d}"
            )

    if arguments["action"] == "create_project":
        if normalized_dates:
            arguments["start_date"] = normalized_dates[0]

        if len(normalized_dates) >= 2:
            arguments["planned_end_date"] = (
                normalized_dates[1]
            )

    elif arguments["action"] == "add_task":
        if normalized_dates:
            arguments["due_date"] = normalized_dates[-1]

    limit_match = _re.search(
        r"(?:最近|前)\s*(\d+)\s*个项目",
        text,
    )

    if limit_match:
        arguments["limit"] = int(
            limit_match.group(1)
        )

    return arguments


def _extract_deterministic_employee_arguments(
    tool_name: str,
    user_text: str,
) -> dict[str, Any]:
    if tool_name == "energy_carbon_delivery_subagent":
        return _extract_energy_carbon_delivery_arguments(
            user_text
        )

    if tool_name == "carbon_asset_operations_subagent":
        return _extract_carbon_asset_arguments(
            user_text
        )

    if tool_name == "retrofit_project_manager_subagent":
        return _extract_retrofit_project_arguments(
            user_text
        )

    if tool_name == "power_bill_audit_subagent":
        return _extract_power_bill_audit_arguments(
            user_text
        )

    return {}


def _extract_power_bill_audit_arguments(
    user_text: str,
) -> dict[str, Any] | None:
    """从自然语言中提取电费稽核工具参数。"""

    text = (user_text or "").strip()

    if not text:
        return None

    # power-list-deterministic-extraction-v1
    # 识别“查询最近N条稽核台账”等只读查询，
    # 避免返回OpenAI风格tool_calls。
    import re as _power_re

    list_intent = any(
        keyword in text
        for keyword in (
            "稽核台账",
            "电费台账",
            "历史稽核",
            "历史记录",
            "稽核记录",
            "最近记录",
            "记录列表",
        )
    )

    if _power_re.search(
        r"(?:最近|前)\s*\d+\s*条",
        text,
    ):
        list_intent = True

    if _power_re.search(
        r"\blist\b",
        text,
        flags=_power_re.IGNORECASE,
    ):
        list_intent = True

    if list_intent:
        limit_match = _power_re.search(
            r"(?:最近|前)?\s*(\d+)\s*条",
            text,
        )

        limit = (
            int(limit_match.group(1))
            if limit_match
            else 20
        )

        limit = max(1, min(limit, 100))

        return {
            "action": "list",
            "limit": limit,
        }

    arguments: dict[str, Any] = {
        "action": "audit",
    }

    # 项目名称：优先识别“核验 + 项目名 + 年月”
    project_patterns = (
        r"(?:核验|稽核|复核|检查)\s*"
        r"([^，。；:：]{2,50}?)"
        r"(?=20\d{2}年\d{1,2}月)",
        r"(?:项目名称|企业名称|客户名称)"
        r"\s*[:：]\s*([^，。；:：]{2,50})",
    )

    for pattern in project_patterns:
        match = re.search(pattern, text)

        if match:
            project_name = match.group(1).strip()

            # 清理可能夹带的动作词
            project_name = re.sub(
                r"^(?:请|为|对|执行|进行)",
                "",
                project_name,
            ).strip()

            if project_name:
                arguments["project_name"] = project_name
                break

    # 账单月份
    month_match = re.search(
        r"(20\d{2})年(\d{1,2})月",
        text,
    )

    if not month_match:
        month_match = re.search(
            r"(20\d{2})[-/](\d{1,2})",
            text,
        )

    if month_match:
        year = int(month_match.group(1))
        month = int(month_match.group(2))
        arguments["billing_month"] = f"{year:04d}-{month:02d}"

    field_patterns: dict[str, tuple[str, ...]] = {
        "electricity_kwh": (
            r"用电量\s*([0-9.]+)\s*(?:kwh|千瓦时)",
        ),
        "energy_charge_cny": (
            r"电度电费\s*([0-9.]+)\s*元",
        ),
        "basic_charge_cny": (
            r"基本电费\s*([0-9.]+)\s*元",
        ),
        "power_factor_adjustment_cny": (
            r"功率因数(?:调整)?电费\s*([0-9.]+)\s*元",
        ),
        "other_charge_cny": (
            r"其他费用\s*([0-9.]+)\s*元",
        ),
        "total_charge_cny": (
            r"(?:总电费|账单总额|总额)\s*([0-9.]+)\s*元",
        ),
        "contract_capacity_kva": (
            r"合同容量\s*([0-9.]+)\s*kva",
        ),
        "max_demand_kw": (
            r"最大需量\s*([0-9.]+)\s*kw",
        ),
        "basic_capacity_rate_cny_per_kva_month": (
            r"(?:容量基本电费单价|容量单价)"
            r"\s*([0-9.]+)\s*元",
        ),
        "baseline_unit_cost_cny_per_kwh": (
            r"(?:历史基准综合电价|基准综合电价)"
            r"\s*([0-9.]+)\s*元",
        ),
    }

    for field_name, patterns in field_patterns.items():
        value = _extract_business_number(
            text,
            patterns,
        )

        if value is not None:
            arguments[field_name] = value

    # 两个主程序必填项必须能提取出来；
    # 否则继续使用原模型路由，避免错误执行。
    if not arguments.get("project_name"):
        return None

    if not arguments.get("billing_month"):
        return None

    return arguments



def _format_internal_employee_result(
    function_name: str,
    tool_result: dict,
) -> str:
    """将垂直业务Subagent结果转换为正式业务输出。"""

    subagent_names = {
        "energy_carbon_delivery_subagent":
            "能碳诊断报告 Subagent",
        "power_bill_audit_subagent":
            "电费账单稽核 Subagent",
        "retrofit_project_manager_subagent":
            "节能改造项目管理 Subagent",
        "carbon_asset_operations_subagent":
            "碳资产运营 Subagent",
    }

    result_titles = {
        "energy_carbon_delivery_subagent":
            "能碳诊断报告生成完成",
        "power_bill_audit_subagent":
            "电费账单稽核完成",
        "retrofit_project_manager_subagent":
            "节能改造项目查询完成",
        "carbon_asset_operations_subagent":
            "碳资产组合汇总完成",
    }

    subagent_name = subagent_names.get(
        function_name,
        function_name,
    )

    result_title = result_titles.get(
        function_name,
        "业务任务处理完成",
    )

    def format_value(value):
        if isinstance(value, bool):
            return "是" if value else "否"

        if isinstance(value, float):
            if value.is_integer():
                return str(int(value))

            return (
                f"{value:.4f}"
                .rstrip("0")
                .rstrip(".")
            )

        if value is None:
            return ""

        return str(value)

    if not isinstance(tool_result, dict):
        return (
            "## 业务任务处理失败\n\n"
            f"- 执行模块：{subagent_name}\n"
            "- 错误信息：返回数据格式异常"
        )

    status = str(
        tool_result.get("result", "ok")
    ).lower()

    if status not in {
        "ok",
        "success",
        "completed",
    }:
        error_message = (
            tool_result.get("message")
            or tool_result.get("error")
            or "业务任务执行失败"
        )

        return (
            "## 业务任务处理失败\n\n"
            f"- 执行模块：{subagent_name}\n"
            f"- 错误信息：{error_message}"
        )

    lines = [
        f"## {result_title}",
        "",
        f"- 执行模块：{subagent_name}",
    ]

    message = str(
        tool_result.get("message", "")
    ).strip()

    if message and message != result_title:
        lines.extend([
            "",
            message,
        ])

    visible_result = tool_result.get(
        "visible_result"
    )

    if isinstance(visible_result, dict):
        visible_lines = []

        for key, value in visible_result.items():
            if value in (None, ""):
                continue

            if isinstance(value, (dict, list)):
                continue

            visible_lines.append(
                f"- {key}：{format_value(value)}"
            )

        if visible_lines:
            lines.extend([
                "",
                "### 执行结果",
                *visible_lines,
            ])

    if (
        function_name
        == "energy_carbon_delivery_subagent"
    ):
        lines.extend([
            "",
            "### 报告交付",
        ])

        report_fields = [
            ("project_name", "项目名称"),
            ("project_id", "项目编号"),
            ("delivery_batch_id", "交付批次"),
            ("executed_at", "生成时间"),
        ]

        for key, label in report_fields:
            value = tool_result.get(key)

            if value not in (None, ""):
                lines.append(
                    f"- {label}：{format_value(value)}"
                )

        validation = tool_result.get(
            "validation"
        )

        if isinstance(validation, dict):
            score = validation.get(
                "completeness_score"
            )

            if score is not None:
                lines.append(
                    "- 数据完整度："
                    f"{format_value(score)}%"
                )

        download_url = tool_result.get(
            "download_url"
        )

        if download_url:
            lines.append(
                "- PDF报告："
                f"[点击下载报告]({download_url})"
            )

        calculation = tool_result.get(
            "calculation_results"
        )

        if isinstance(calculation, dict):
            metric_fields = [
                (
                    "building_area_m2",
                    "建筑面积",
                    "m²",
                ),
                (
                    "electricity_kwh",
                    "年度用电量",
                    "kWh",
                ),
                (
                    "natural_gas_m3",
                    "年度天然气用量",
                    "m³",
                ),
                (
                    "total_emissions_tco2e",
                    "年度碳排放量",
                    "tCO₂e",
                ),
                (
                    "carbon_intensity_kgco2e_per_m2",
                    "单位面积碳排放强度",
                    "kgCO₂e/m²",
                ),
            ]

            metric_lines = []

            for key, label, unit in metric_fields:
                value = calculation.get(key)

                if value is None:
                    continue

                metric_lines.append(
                    f"- {label}："
                    f"{format_value(value)} {unit}"
                )

            if metric_lines:
                lines.extend([
                    "",
                    "### 核心诊断指标",
                    *metric_lines,
                ])

    # power-list-result-format-v1
    if (
        function_name
        == "power_bill_audit_subagent"
    ):
        records = tool_result.get("records")
        record_count = tool_result.get(
            "record_count"
        )

        if isinstance(records, list):
            lines.extend([
                "",
                "### 电费稽核台账",
            ])

            if record_count is not None:
                lines.append(
                    "- 返回记录数："
                    f"{format_value(record_count)}"
                )

            if not records:
                lines.append("- 暂无稽核台账记录")

            for index, record in enumerate(
                records,
                start=1,
            ):
                if not isinstance(record, dict):
                    lines.append(
                        f"{index}. {record}"
                    )
                    continue

                project_name = record.get(
                    "project_name",
                    "未命名项目",
                )
                billing_month = record.get(
                    "billing_month",
                    "",
                )
                status_text = record.get(
                    "verification_status",
                    "",
                )

                heading = str(project_name)

                if billing_month:
                    heading += f"｜{billing_month}"

                if status_text:
                    heading += f"｜{status_text}"

                lines.append(
                    f"{index}. {heading}"
                )

                details = [
                    (
                        "稽核编号",
                        record.get("audit_id"),
                        "",
                    ),
                    (
                        "总电费",
                        record.get(
                            "total_charge_cny"
                        ),
                        "元",
                    ),
                    (
                        "用电量",
                        record.get(
                            "electricity_kwh"
                        ),
                        "kWh",
                    ),
                    (
                        "综合电价",
                        record.get(
                            "unit_cost_cny_per_kwh"
                        ),
                        "元/kWh",
                    ),
                    (
                        "异常数量",
                        record.get(
                            "anomaly_count"
                        ),
                        "",
                    ),
                    (
                        "工单编号",
                        record.get(
                            "work_order_id"
                        ),
                        "",
                    ),
                    (
                        "预计年度节省",
                        record.get(
                            "estimated_annual_saving_cny"
                        ),
                        "元",
                    ),
                ]

                for label, value, unit in details:
                    if value in (None, ""):
                        continue

                    lines.append(
                        f"   - {label}："
                        f"{format_value(value)}{unit}"
                    )

    anomalies = tool_result.get("anomalies")

    if isinstance(anomalies, list) and anomalies:
        lines.extend([
            "",
            "### 发现的异常",
        ])

        for index, anomaly in enumerate(
            anomalies,
            start=1,
        ):
            if isinstance(anomaly, dict):
                severity = anomaly.get(
                    "severity",
                    "",
                )
                anomaly_message = anomaly.get(
                    "message",
                    "",
                )
                code = anomaly.get(
                    "code",
                    "",
                )

                item = f"{index}. "

                if severity:
                    item += f"【{severity}】"

                item += str(anomaly_message)

                if code:
                    item += f"（{code}）"

                lines.append(item)
            else:
                lines.append(
                    f"{index}. {anomaly}"
                )

    project = tool_result.get("project")

    if (
        function_name
        == "retrofit_project_manager_subagent"
        and isinstance(project, dict)
    ):
        tasks = project.get("tasks")

        if isinstance(tasks, list) and tasks:
            lines.extend([
                "",
                "### 项目任务",
            ])

            for index, task in enumerate(
                tasks,
                start=1,
            ):
                if not isinstance(task, dict):
                    lines.append(
                        f"{index}. {task}"
                    )
                    continue

                task_name = task.get(
                    "task_name",
                    "未命名任务",
                )
                task_status = task.get(
                    "status",
                    "",
                )
                progress = task.get(
                    "progress_percent",
                    "",
                )
                owner = task.get(
                    "owner",
                    "",
                )

                lines.append(
                    f"{index}. {task_name}"
                )

                details = []

                if task_status:
                    details.append(
                        f"状态：{task_status}"
                    )

                if progress != "":
                    details.append(
                        "进度："
                        f"{format_value(progress)}%"
                    )

                if owner:
                    details.append(
                        f"负责人：{owner}"
                    )

                if details:
                    lines.append(
                        "   - "
                        + "；".join(details)
                    )

    portfolio = tool_result.get(
        "portfolio_by_type"
    )

    if (
        function_name
        == "carbon_asset_operations_subagent"
        and isinstance(portfolio, dict)
        and portfolio
    ):
        lines.extend([
            "",
            "### 资产分类",
        ])

        for asset_type, summary in portfolio.items():
            if not isinstance(summary, dict):
                continue

            count = summary.get(
                "asset_count",
                0,
            )
            available = summary.get(
                "available_tco2e",
                0,
            )
            estimated_value = summary.get(
                "estimated_value_cny",
                0,
            )

            lines.append(
                f"- {asset_type}："
                f"{format_value(count)}项，"
                f"可用{format_value(available)} tCO₂e，"
                f"估值{format_value(estimated_value)}元"
            )

    business_changes = tool_result.get(
        "business_state_changes"
    )

    if (
        isinstance(business_changes, list)
        and business_changes
    ):
        lines.extend([
            "",
            "### 业务状态变化",
        ])

        for change in business_changes:
            if not isinstance(change, dict):
                continue

            business_object = change.get(
                "business_object",
                "业务对象",
            )
            before = change.get(
                "before",
                "",
            )
            after = change.get(
                "after",
                "",
            )

            lines.append(
                f"- {business_object}："
                f"{before} → {after}"
            )

    recommendations = (
        tool_result.get("recommendations")
        or tool_result.get("suggestions")
        or tool_result.get("advice")
    )

    if (
        isinstance(recommendations, list)
        and recommendations
    ):
        lines.extend([
            "",
            "### 后续建议",
        ])

        for index, recommendation in enumerate(
            recommendations,
            start=1,
        ):
            if isinstance(recommendation, dict):
                priority = str(
                    recommendation.get(
                        "priority",
                        "",
                    )
                ).strip()

                title = str(
                    recommendation.get(
                        "title",
                        "优化建议",
                    )
                ).strip()

                heading = f"{index}. "

                if priority:
                    heading += f"【{priority}】"

                heading += title
                lines.append(heading)

                reason = recommendation.get(
                    "reason"
                )
                measure = recommendation.get(
                    "measure"
                )

                if reason:
                    lines.append(
                        f"   - 诊断依据：{reason}"
                    )

                if measure:
                    lines.append(
                        f"   - 建议措施：{measure}"
                    )
            else:
                lines.append(
                    f"{index}. {recommendation}"
                )

    lines.extend([
        "",
        "> 本结果由对应业务 Subagent "
        "基于真实项目数据和业务规则生成。",
    ])

    return "\n".join(lines)




@dataclass(slots=True)
class ToolLoopResult:
    """Result from the Qwen tool selection and execution loop."""

    used_any_tool: bool
    last_function_name: str | None
    tool_rounds: int
    # Tool context messages to inject into upstream request (excludes system prompt
    # and the final assistant text reply, keeps: user, assistant+tool_calls, tool results)
    tool_context: list[dict[str, Any]]
    # Whether the loop ended because max rounds were exhausted (still had tool_calls)
    max_rounds_exhausted: bool
    # Qwen's text reply after tool execution (normally stripped); preserved for
    # completion-check so it can be returned directly on short-circuit.
    qwen_reply: str | None = None
    # Internal messages list from the tool loop (system + user + rounds);
    # used by completion-check to append the judgment prompt in-context.
    _loop_messages: list[dict[str, Any]] = field(default_factory=list)
    # Per-LLM-call timing records (one entry per call_qwen invocation).
    # Each entry: {kind, round, request_timestamp, response_timestamp, model}
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    delegated_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    direct_response: str | None = None


async def run_tool_loop(
    user_text: str,
    *,
    history: list[dict[str, Any]] | None = None,
    delegated_tool_names: set[str] | None = None,
    resume_tool_context: list[dict[str, Any]] | None = None,
) -> ToolLoopResult:
    """Run the Qwen tool selection and execution loop.

    Returns a ToolLoopResult containing the tool interaction context.
    The final assistant text reply (if any) is stripped — the upstream model
    will generate the user-facing response based on the tool context.

    If *history* is provided (from _QWEN_SAVED_CONTEXT), prior Qwen-internal
    turns are prepended before the current user message.
    """

    if STATE.tools is None:
        raise RuntimeError("tools not loaded")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    current_turn_offset = len(messages) - 1
    last_function_name: str | None = None
    used_any_tool = False
    max_rounds = STATE.config.max_tool_rounds if STATE.config else 0
    routing_model_name = STATE.config.routing.model if STATE.config else ""
    llm_calls: list[dict[str, Any]] = []
    delegated_tool_names = set(delegated_tool_names or ())
    forced_tool_name = _deterministic_employee_tool(user_text)

    if forced_tool_name and STATE.logger is not None:
        STATE.logger.info(
            "deterministic employee route selected: %s",
            forced_tool_name,
        )

    # 核心业务 Subagent由Function Router直接执行。
    # OpenClaw只负责接收并展示最终结果，避免委托工具调用反复循环。
    if (
        forced_tool_name in {
            "power_bill_audit_subagent",
            "retrofit_project_manager_subagent",
            "carbon_asset_operations_subagent",
            "energy_carbon_delivery_subagent",
        
        }
        and resume_tool_context is None
    ):
        deterministic_arguments = (
            _extract_deterministic_employee_arguments(forced_tool_name, user_text)
        )

        if deterministic_arguments is not None:
            arguments_json = json.dumps(
                deterministic_arguments,
                ensure_ascii=False,
            )

            tool_call_id = (
                f"call_{forced_tool_name}_"
                f"{int(time.time() * 1000)}"
            )

            deterministic_tool_call = (
                _normalize_tool_call_for_response(
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": forced_tool_name,
                            "arguments": arguments_json,
                        },
                    },
                    fallback_id=tool_call_id,
                )
            )

            assistant_message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    deterministic_tool_call,
                ],
            }

            # 在Router服务器内部真实执行员工脚本
            deterministic_tool_result = await execute_tool(
                forced_tool_name,
                arguments_json,
            )

            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": forced_tool_name,
                "content": json.dumps(
                    deterministic_tool_result,
                    ensure_ascii=False,
                ),
            }

            if STATE.logger is not None:
                STATE.logger.info(
                    "deterministic employee executed internally: "
                    "%s result=%s",
                    forced_tool_name,
                    json.dumps(
                        deterministic_tool_result,
                        ensure_ascii=False,
                    )[:2000],
                )

            direct_response = _format_internal_employee_result(
                forced_tool_name,
                deterministic_tool_result,
            )

            return ToolLoopResult(
                used_any_tool=True,
                last_function_name=forced_tool_name,
                tool_rounds=1,
                tool_context=[
                    *messages[1:],
                    assistant_message,
                    tool_message,
                ],
                max_rounds_exhausted=False,
                qwen_reply=None,
                _loop_messages=[
                    *messages,
                    assistant_message,
                    tool_message,
                ],
                llm_calls=[],
                delegated_tool_calls=[],
                direct_response=direct_response,
            )

    if resume_tool_context:
        messages.extend(dict(message) for message in resume_tool_context)
        last_tool_message = messages[-1] if messages and messages[-1].get("role") == "tool" else None
        if isinstance(last_tool_message, dict):
            tool_name = last_tool_message.get("name")
            if isinstance(tool_name, str) and tool_name:
                last_function_name = tool_name
        for message in reversed(resume_tool_context):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                last_function_name = _tool_call_function_name(tool_calls[-1]) or last_function_name
                break
        used_any_tool = True

    _debug_log_messages("QWEN", messages, offset=current_turn_offset)

    for round_index in range(1, max_rounds + 1):
        llm_req_ts = now_iso()
        response_json = await call_qwen(
            messages,
            forced_tool_name=(
                forced_tool_name
                if round_index == 1
                else None
            ),
        )
        llm_resp_ts = now_iso()
        llm_calls.append({
            "kind": "qwen_tool_loop",
            "round": round_index,
            "model": routing_model_name,
            "request_timestamp": llm_req_ts,
            "response_timestamp": llm_resp_ts,
        })
        choices = response_json.get("choices") or []
        if not choices:
            # Qwen returned nothing — treat as no-tool
            return ToolLoopResult(
                used_any_tool=used_any_tool,
                last_function_name=last_function_name,
                tool_rounds=round_index,
                tool_context=[],
                max_rounds_exhausted=False,
                llm_calls=llm_calls,
            )

        message = choices[0].get("message") or {}
        tool_calls = message.get("tool_calls") or []
        content = message.get("content")
        if tool_calls or used_any_tool:
            _debug_log_messages("QWEN", [message])

        assistant_message: dict[str, Any] = {"role": "assistant"}
        if content is not None:
            assistant_message["content"] = content
        else:
            assistant_message["content"] = ""
        if tool_calls:
            tool_calls_with_ts = []
            for tool_call in tool_calls:
                tool_call_copy = dict(tool_call)
                tool_call_copy["timestamp"] = now_iso()
                tool_call_copy["llm_request_timestamp"] = llm_req_ts
                tool_call_copy["llm_response_timestamp"] = llm_resp_ts
                tool_calls_with_ts.append(tool_call_copy)
            assistant_message["tool_calls"] = tool_calls_with_ts
        messages.append(assistant_message)

        if not tool_calls:
            # Qwen finished with a text reply. If we used tools, strip this
            # final assistant reply and return the tool context for upstream.
            # If no tools were used, return empty context (pure non-function query).
            if used_any_tool:
                # Context = everything after system prompt; explicitly strip the
                # trailing assistant text-only message (the one we just appended).
                tool_context = messages[1:]  # skip system[0]
                qwen_reply: str | None = None
                if (
                    tool_context
                    and tool_context[-1].get("role") == "assistant"
                    and not tool_context[-1].get("tool_calls")
                ):
                    qwen_reply = tool_context[-1].get("content") or None
                    tool_context = tool_context[:-1]
                return ToolLoopResult(
                    used_any_tool=True,
                    last_function_name=last_function_name,
                    tool_rounds=round_index,
                    tool_context=tool_context,
                    max_rounds_exhausted=False,
                    qwen_reply=qwen_reply,
                    _loop_messages=messages,
                    llm_calls=llm_calls,
                )
            return ToolLoopResult(
                used_any_tool=False,
                last_function_name=last_function_name,
                tool_rounds=round_index,
                tool_context=[],
                max_rounds_exhausted=False,
                qwen_reply=content or None,
                _loop_messages=messages,
                llm_calls=llm_calls,
            )

        if _tool_calls_are_delegated(tool_calls, delegated_tool_names):
            delegated_tool_calls = [
                _normalize_tool_call_for_response(
                    tool_call,
                    fallback_id=f"call_{_tool_call_function_name(tool_call) or 'tool'}_{round_index}_{index}",
                )
                for index, tool_call in enumerate(tool_calls)
            ]
            assistant_message["tool_calls"] = delegated_tool_calls
            messages[-1] = assistant_message
            last_function_name = _tool_call_function_name(delegated_tool_calls[-1]) or last_function_name
            return ToolLoopResult(
                used_any_tool=True,
                last_function_name=last_function_name,
                tool_rounds=round_index,
                tool_context=messages[1:],
                max_rounds_exhausted=False,
                qwen_reply=None,
                _loop_messages=messages,
                llm_calls=llm_calls,
                delegated_tool_calls=delegated_tool_calls,
            )

        used_any_tool = True
        for tool_call in tool_calls:
            function_meta = tool_call.get("function") or {}
            function_name = function_meta.get("name", "")
            arguments_json = function_meta.get("arguments") or "{}"
            last_function_name = function_name or last_function_name
            tool_result = await execute_tool(function_name, arguments_json)
            tool_result_ts = now_iso()
            _debug_log_messages(
                "QWEN",
                [{
                    "role": "tool",
                    "name": function_name,
                    "tool_call_id": tool_call.get("id") or f"call_{function_name}_{round_index}",
                    "content": json.dumps(tool_result, ensure_ascii=False),
                }],
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id") or f"call_{function_name}_{round_index}",
                    "name": function_name,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                    "timestamp": tool_result_ts,
                }
            )

    # Max rounds exhausted — return full context (everything after system prompt)
    # including the last assistant message (which may have tool_calls)
    tool_context = messages[1:]  # skip system[0]
    return ToolLoopResult(
        used_any_tool=used_any_tool,
        last_function_name=last_function_name,
        tool_rounds=max_rounds,
        tool_context=tool_context,
        max_rounds_exhausted=True,
        llm_calls=llm_calls,
    )

#COMPLETION_CHECK_PROMPT = (
#    "根据上面的对话，用户的请求是否已经被完全满足？只根据以下标准判断：\n"
#    "- 只要工具调用成功，且当前流程已经推进到等待用户挑选、确认或补充信息的阶段，输出 TASK_COMPLETE\n"
#    "- 只有工具调用失败、报错、或流程根本没有推进，输出 TASK_INCOMPLETE，不要根据‘是否已经最终下单’来判断。”\n"
#    "只输出上述两个标记之一，不要输出任何其他内容。"
#)

# COMPLETION_CHECK_PROMPT = (
#     "根据上面的对话，用户的请求是否已经被完全满足？\n"
#     "- 如果已完成，仅回复: TASK_COMPLETE\n"
#     "- 如果需要用户输入更多信息，或者希望用户进行挑选和确认，仅回复: TASK_COMPLETE\n"
#     "- 如果工具调用失败或不满足上述两个情况，仅回复: TASK_INCOMPLETE\n"
#     "只输出上述两个标记之一，不要输出任何其他内容。"
# )

COMPLETION_CHECK_PROMPT_PERMISSIVE = (
    "根据上面的对话，用户的请求是否已经被完全满足？只根据以下标准判断：\n"
    "- 只要工具调用成功，且当前流程已经推进到等待用户挑选、确认或补充信息的阶段，输出 TASK_COMPLETE\n"
    "- 只有工具调用失败、报错、或流程根本没有推进，输出 TASK_INCOMPLETE，不要根据‘是否已经最终下单’来判断。\n"
    "只输出上述两个标记之一，不要输出任何其他内容。"
)

COMPLETION_CHECK_PROMPT_STRICT = (
    "根据上面的对话，用户的请求是否已经被完全满足？\n"
    "- 如果已完成，仅回复: TASK_COMPLETE\n"
    "- 如果未完成（工具失败、信息不足、用户还需要更多操作等），仅回复: TASK_INCOMPLETE\n"
    "只输出上述两个标记之一，不要输出任何其他内容。"
)


def get_completion_check_prompt(mode: str) -> str:
    """Return the completion-check prompt for the configured mode."""

    if mode == "strict":
        return COMPLETION_CHECK_PROMPT_STRICT
    return COMPLETION_CHECK_PROMPT_PERMISSIVE



async def call_qwen_completion_check(
    messages: list[dict[str, Any]],
    out_llm_calls: list[dict[str, Any]] | None = None,
) -> bool:
    """Ask Qwen whether the user's task is complete.

    Appends a user message to the existing tool-loop *messages* (which already
    contain system + user + assistant+tool_calls + tool_results + assistant reply)
    and asks Qwen to judge.  This round does **not** send tools so Qwen cannot
    issue new tool calls.

    Returns True if Qwen judges the task complete, False otherwise (including
    on any error).  The appended judgment messages are **not** kept — callers
    should treat *messages* as consumed.
    """

    if STATE.http_client is None or STATE.config is None:
        return False

    completion_prompt = get_completion_check_prompt(STATE.config.fr_completion_check_mode)

    # Build a new list so callers can still use the original (e.g. context buffer).
    check_messages = [*messages, {"role": "user", "content": completion_prompt}]
    for i in range(len(check_messages)):
        if check_messages[i]["role"].lower() == "system":
            check_messages[i]["content"] = SYSTEM_PROMPT_REVIEW
            break

    payload = {
        "model": STATE.config.routing.model,
        "messages": check_messages,
        # No tools — prevent Qwen from issuing new tool calls in this round.
        "stream": False,
        "max_tokens": 128,
        "temperature": 0.0,
        "repetition_penalty": 1.2,
        "frequency_penalty": 0.2,
        "thinking": {"type": "disabled"},
    }
    headers = {
        "Authorization": f"Bearer {STATE.config.routing.api_key}",
        "Content-Type": "application/json",
    }
    llm_req_ts = now_iso()
    try:
        response = await STATE.http_client.post(
            f"{STATE.config.routing.base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
            timeout=5.0,
        )
        llm_resp_ts = now_iso()
        if out_llm_calls is not None:
            out_llm_calls.append({
                "kind": "qwen_completion_check",
                "model": STATE.config.routing.model,
                "request_timestamp": llm_req_ts,
                "response_timestamp": llm_resp_ts,
            })
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            _debug_log_messages("QWEN", [{"role": "user", "content": completion_prompt}])
            _debug_log_messages("QWEN", [{"role": "assistant", "content": ""}])
            return False
        msg_obj = choices[0].get("message") or {}
        text = msg_obj.get("content") or ""
        is_complete = "TASK_COMPLETE" in text.upper().strip()
        _debug_log_messages("QWEN", [{"role": "user", "content": completion_prompt}])
        _debug_log_messages("QWEN", [{"role": "assistant", "content": text}])
        return is_complete
    except (httpx.HTTPError, Exception) as exc:
        if out_llm_calls is not None:
            out_llm_calls.append({
                "kind": "qwen_completion_check",
                "model": STATE.config.routing.model,
                "request_timestamp": llm_req_ts,
                "response_timestamp": now_iso(),
                "error": type(exc).__name__,
            })
        if STATE.logger is not None:
            STATE.logger.warning("qwen completion check failed, treating as incomplete")
        _debug_log_messages("QWEN", [{"role": "user", "content": completion_prompt}])
        _debug_log_messages("QWEN", [{"role": "assistant", "content": f"error:{type(exc).__name__}"}])
        return False



def _fr_only_mode() -> bool:
    return bool(STATE.config and STATE.config.fr_completion_check_always_true)


def _last_assistant_text(messages: list[dict[str, Any]]) -> str:
    """Return the last assistant text from a message list."""

    for message in reversed(messages):
        if message.get("role") == "assistant":
            content = message.get("content")
            return content if isinstance(content, str) else ""
    return ""


def _build_completion_response(
    content: str,
    *,
    stream: bool = False,
) -> JSONResponse | StreamingResponse:
    """Build an OpenAI-compatible chat completion response from plain text.

    Supports both non-streaming (JSONResponse) and streaming (SSE) modes so
    the short-circuit reply is compatible with whatever the caller requested.
    """

    completion_id = f"fr-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if not stream:
        return JSONResponse({
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": "function-router",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    # SSE streaming: one chunk with the full content, then [DONE].
    chunk = json.dumps({
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": "function-router",
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
    }, ensure_ascii=False)

    async def sse_stream():
        yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        sse_stream(),
        media_type="text/event-stream",
    )


def _build_tool_calls_response(
    tool_calls: list[dict[str, Any]],
    *,
    stream: bool = False,
) -> JSONResponse | StreamingResponse:
    """Build an OpenAI-compatible assistant.tool_calls response."""

    completion_id = f"fr-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    normalized_tool_calls = [
        _normalize_tool_call_for_response(
            tool_call,
            fallback_id=f"call_{_tool_call_function_name(tool_call) or 'tool'}_{index}",
        )
        for index, tool_call in enumerate(tool_calls)
    ]

    if not stream:
        return JSONResponse({
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": "function-router",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": normalized_tool_calls,
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    delta_tool_calls = [
        {"index": index, **tool_call}
        for index, tool_call in enumerate(normalized_tool_calls)
    ]
    first_chunk = json.dumps({
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": "function-router",
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
                "tool_calls": delta_tool_calls,
            },
            "finish_reason": None,
        }],
    }, ensure_ascii=False)
    finish_chunk = json.dumps({
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": "function-router",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "tool_calls",
        }],
    }, ensure_ascii=False)

    async def sse_stream():
        yield f"data: {first_chunk}\n\n"
        yield f"data: {finish_chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        sse_stream(),
        media_type="text/event-stream",
    )


def _build_upstream_request(
    original_request: dict[str, Any],
    *,
    session_key: str,
    tool_context: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build upstream request body with pending plain-text history injection."""

    proxied_request = json.loads(json.dumps(original_request))

    pending_messages = _render_pending_upstream_messages(session_key)
    if pending_messages:
        msgs = list(proxied_request.get("messages", []))
        split_index = len(msgs)
        if msgs and msgs[-1].get("role") == "user":
            split_index -= 1
        proxied_request["messages"] = [
            *msgs[:split_index],
            *pending_messages,
            *msgs[split_index:],
        ]

    if tool_context:
        msgs = proxied_request.get("messages", [])
        if msgs and msgs[-1].get("role") == "user":
            msgs.pop()
        msgs.extend(tool_context)
        proxied_request["messages"] = msgs

    return proxied_request


async def proxy_upstream(
    original_request: dict[str, Any],
    tool_context: list[dict[str, Any]] | None = None,
    session_key: str | None = None,
    out_timing: dict[str, Any] | None = None,
    on_stream_end: Callable[[], None] | None = None,
) -> StreamingResponse:
    """Proxy the request to the configured upstream model endpoint.

    This is the single place that applies pending Qwen-completed turns to the
    upstream payload so we do not inject the same backlog twice.
    """

    if STATE.http_client is None or STATE.config is None:
        raise RuntimeError("application state not initialized")

    pending_messages: list[dict[str, Any]] = []
    pending_before = 0
    if session_key:
        pending_messages = _render_pending_upstream_messages(session_key)
        pending_before = len(_get_pending_upstream_turns(session_key))
        proxied_request = _build_upstream_request(
            original_request,
            session_key=session_key,
            tool_context=tool_context,
        )
    else:
        proxied_request = json.loads(json.dumps(original_request))
        if tool_context:
            msgs = proxied_request.get("messages", [])
            if msgs and msgs[-1].get("role") == "user":
                msgs.pop()
            msgs.extend(tool_context)
            proxied_request["messages"] = msgs

    proxied_request["model"] = STATE.config.upstream.model

    headers = {
        "Authorization": f"Bearer {STATE.config.upstream.api_key}",
        "Content-Type": "application/json",
    }
    target_url = f"{STATE.config.upstream.base_url.rstrip('/')}/chat/completions"
    upstream_req_ts = now_iso()
    stream_context = STATE.http_client.stream(
        "POST",
        target_url,
        json=proxied_request,
        headers=headers,
        timeout=120.0,
    )
    upstream_response = await stream_context.__aenter__()
    upstream_resp_ts = now_iso()
    if out_timing is not None:
        out_timing.update({
            "kind": "upstream_proxy",
            "model": STATE.config.upstream.model,
            "request_timestamp": upstream_req_ts,
            "response_timestamp": upstream_resp_ts,
        })

    if upstream_response.status_code >= 400:
        body = await upstream_response.aread()
        await stream_context.__aexit__(None, None, None)
        if STATE.logger is not None:
            STATE.logger.warning(
                "upstream returned %d: %s", upstream_response.status_code, body[:500]
            )
        raise HTTPException(
            status_code=upstream_response.status_code,
            detail=f"upstream error: {body.decode('utf-8', errors='replace')[:200]}",
        )

    async def stream_bytes() -> Any:
        first_chunk_seen = False
        response_chunks: list[bytes] = []
        try:
            async for chunk in upstream_response.aiter_bytes():
                response_chunks.append(chunk)
                if not first_chunk_seen:
                    first_chunk_seen = True
                    if out_timing is not None:
                        out_timing["first_chunk_timestamp"] = now_iso()
                yield chunk
        finally:
            if out_timing is not None:
                out_timing["stream_end_timestamp"] = now_iso()
            await stream_context.__aexit__(None, None, None)
            pending_after = None
            if session_key and response_chunks:
                _clear_pending_upstream_turns(session_key)
                pending_after = len(_get_pending_upstream_turns(session_key))
            if response_chunks:
                assistant_content = _extract_upstream_assistant_content(
                    b"".join(response_chunks),
                    response_content_type,
                )
                original_messages = original_request.get("messages")
                current_user_message = None
                if isinstance(original_messages, list):
                    for message in reversed(original_messages):
                        if isinstance(message, dict) and message.get("role") == "user":
                            current_user_message = message
                            break
                if assistant_content and _has_visible_assistant_reply(assistant_content):
                    _debug_log_upstream_context(
                        pending_messages,
                        current_user_message,
                        assistant_content,
                        pending_before=pending_before,
                        pending_injected=len(pending_messages) // 2,
                        pending_after=pending_after,
                    )
            if on_stream_end is not None:
                try:
                    on_stream_end()
                except Exception:
                    if STATE.logger is not None:
                        STATE.logger.exception("proxy_upstream on_stream_end callback failed")

    response_content_type = upstream_response.headers.get("content-type", "text/event-stream")

    return StreamingResponse(
        stream_bytes(),
        status_code=upstream_response.status_code,
        media_type=response_content_type,
        headers={"x-function-router-route": "upstream"},
    )


def log_request(
    *,
    user_message: str | None,
    route: str,
    function_name: str | None,
    tool_rounds: int,
    latency_ms: float,
    status: str,
) -> None:
    """Log a structured per-request record."""

    payload = {
        "timestamp": now_iso(),
        "user_message": (user_message or "")[:100],
        "route": route,
        "function_name": function_name,
        "tool_rounds": tool_rounds,
        "latency_ms": round(latency_ms, 2),
        "status": status,
    }
    logging.getLogger(REQUEST_LOGGER_NAME).info(json.dumps(payload, ensure_ascii=False))



def _record_tool_history(
    user_message: str | None,
    tool_context: list[dict[str, Any]],
    tool_rounds: int,
    session_key: str,
    llm_calls: list[dict[str, Any]] | None = None,
) -> None:
    """Parse tool_context (OpenAI messages) and append to TOOL_HISTORY ring buffer.

    *llm_calls* carries per-LLM-call timing records (FR Qwen rounds, completion
    check, upstream proxy). They are emitted into ordered_events as
    type="llm_call" entries so consumers can render request/response timestamps.
    """

    tool_calls: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    ordered_events: list[dict[str, Any]] = []

    for msg in tool_context:
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function") or {}
                entry_ts = tc.get("timestamp") or now_iso()
                entry = {
                    "id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments", "{}"),
                    "timestamp": entry_ts,
                }
                llm_req_ts = tc.get("llm_request_timestamp")
                llm_resp_ts = tc.get("llm_response_timestamp")
                if llm_req_ts:
                    entry["llm_request_timestamp"] = llm_req_ts
                if llm_resp_ts:
                    entry["llm_response_timestamp"] = llm_resp_ts
                tool_calls.append(entry)
                ordered_events.append({
                    "type": "tool_call",
                    **entry,
                })
        elif role == "tool":
            content_raw = msg.get("content", "")
            is_error = False
            try:
                parsed = json.loads(content_raw) if isinstance(content_raw, str) else content_raw
                is_error = "error" in parsed and "result" not in parsed
            except (json.JSONDecodeError, TypeError):
                pass
            entry_ts = msg.get("timestamp") or now_iso()
            entry = {
                "tool_call_id": msg.get("tool_call_id", ""),
                "name": msg.get("name", ""),
                "content": content_raw,
                "is_error": is_error,
                "timestamp": entry_ts,
            }
            tool_results.append(entry)
            ordered_events.append({
                "type": "tool_result",
                **entry,
            })

    if llm_calls:
        for call in llm_calls:
            ordered_events.append({
                "type": "llm_call",
                **call,
            })

    if not tool_calls and not tool_results and not llm_calls:
        return

    def _ev_sort_key(ev: dict[str, Any]) -> str:
        return ev.get("request_timestamp") or ev.get("timestamp") or ""

    ordered_events.sort(key=_ev_sort_key)

    entry_timestamp = ordered_events[-1].get("timestamp") or ordered_events[-1].get("response_timestamp") or now_iso()
    TOOL_HISTORY.append({
        "timestamp": entry_timestamp,
        "session_key": session_key,
        "user_message": user_message or "",
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "ordered_events": ordered_events,
        "tool_rounds": tool_rounds,
        "llm_calls": list(llm_calls or []),
    })


app = FastAPI(title="Function Router", version="1.0.0")


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize config, tools, logging, and the HTTP client."""

    config = load_config(STATE.config_path)
    logger = setup_logging(config.root_dir, debug_logging=config.debug_logging)
    STATE.logger = logger
    STATE.config = config
    STATE.tools = load_tools(config.functions_path)
    try:
        (config.root_dir / "openclaw-tools.json").write_text(
            json.dumps({"tools": STATE.tools}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("failed to write openclaw tools snapshot: %s", exc)
    STATE.http_client = await build_http_client()
    STATE.warmup_ok = await warmup_qwen()

    logger.info("config loaded from %s", config.config_path)
    logger.info("loaded %d tools from %s", len(STATE.tools), config.functions_path)
    logger.info("warmup result: %s", "success" if STATE.warmup_ok else "failure")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Close the shared HTTP client."""

    if STATE.http_client is not None:
        await STATE.http_client.aclose()
        STATE.http_client = None


@app.get("/health")
async def health() -> JSONResponse:
    """Return a basic health response."""

    tools_loaded = len(STATE.tools or [])
    return JSONResponse({"status": "ok", "tools_loaded": tools_loaded})


@app.get("/ready")
async def ready() -> JSONResponse:
    """Check readiness based on Qwen reachability."""

    ready_ok = await qwen_health_check()
    return JSONResponse({"status": "ok" if ready_ok else "unavailable"}, status_code=200 if ready_ok else 503)


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    """List available models in OpenAI-compatible format."""

    return JSONResponse({
        "object": "list",
        "data": [
            {
                "id": "function-router",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "function-router"
            }
        ]
    })


@app.get("/v1/tool_history")
async def get_tool_history(since: str | None = None, limit: int = 50) -> JSONResponse:
    """Return recent tool execution history.

    Query params:
        since: ISO timestamp — only return entries after this time.
        limit: max entries to return (default 50, max 200).
    """

    limit = min(max(limit, 1), 200)
    entries = list(TOOL_HISTORY)  # snapshot

    if since:
        entries = [e for e in entries if e["timestamp"] > since]

    # Most recent first, apply limit.
    entries = entries[-limit:]
    entries.reverse()

    return JSONResponse({"entries": entries})


@app.get("/v1/tools")
async def list_tools() -> JSONResponse:
    """Return loaded OpenAI-compatible tool definitions."""

    return JSONResponse({"tools": STATE.tools or []})


@app.post("/v1/execute_tool")
async def execute_tool_endpoint(payload: dict[str, Any]) -> JSONResponse:
    """Execute one loaded Function Router tool for OpenClaw-side delegation."""

    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise HTTPException(status_code=400, detail="name must be a non-empty string")
    name = name.strip()

    arguments = payload.get("arguments", {})
    if isinstance(arguments, str):
        arguments_json = arguments
    elif isinstance(arguments, dict):
        arguments_json = json.dumps(arguments, ensure_ascii=False)
    else:
        raise HTTPException(status_code=400, detail="arguments must be an object or JSON string")

    result = await execute_tool(name, arguments_json)
    return JSONResponse({"name": name, "result": result})


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> StreamingResponse:
    """Handle OpenAI-compatible chat completion requests."""

    started_at = time.perf_counter()
    function_name: str | None = None
    tool_rounds = 0
    user_text: str | None = None

    try:
        body_bytes = await request.body()
        original_request = json.loads(body_bytes)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}") from exc

    messages = original_request.get("messages")
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="request body must include a messages array")

    try:
        user_text = extract_user_text(messages)
        request_headers = dict(request.headers)
        session_key = derive_session_key(original_request, request_headers)
        _debug_log_session(session_key)
        if STATE.logger:
            STATE.logger.info(
                "header x-openclaw-session-key=%s x-openclaw-session-id=%s session_key=%s",
                request_headers.get("x-openclaw-session-key", ""),
                request_headers.get("x-openclaw-session-id", ""),
                session_key,
            )
        _debug_log(
            "request_entry",
            session_key=session_key,
            user_text=user_text or "",
            message_count=len(messages),
            has_stream=original_request.get("stream", False),
        )
        delegated_names = _delegated_tool_names()
        parsed_delegated_continuation = _find_delegated_tool_continuation(
            messages,
            delegated_names,
        )
        delegated_continuation = None
        if parsed_delegated_continuation is not None:
            _, tool_call_ids = parsed_delegated_continuation
            if _consume_pending_delegated_tool_turn(session_key, tool_call_ids):
                delegated_continuation = parsed_delegated_continuation
        if not user_text:
            if _fr_only_mode():
                return _build_completion_response(
                    _last_assistant_text(messages),
                    stream=original_request.get("stream", False),
                )
            upstream_timing: dict[str, Any] = {}
            _user_text_no_user = user_text

            def _finalize_no_user() -> None:
                if upstream_timing:
                    _record_tool_history(
                        _user_text_no_user,
                        [],
                        0,
                        session_key,
                        llm_calls=[upstream_timing],
                    )

            response = await proxy_upstream(
                original_request,
                session_key=session_key,
                out_timing=upstream_timing,
                on_stream_end=_finalize_no_user,
            )
            log_request(
                user_message=user_text,
                route="upstream",
                function_name=None,
                tool_rounds=0,
                latency_ms=(time.perf_counter() - started_at) * 1000,
                status="forwarded_no_user",
            )
            _debug_log(
                "route_decision",
                session_key=session_key,
                route="upstream",
                status="forwarded_no_user",
                function_name=function_name,
                tool_rounds=tool_rounds,
                latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
            )
            return response

        # Check for HEARTBEAT keyword to bypass Qwen routing
        if ("HEARTBEAT" in user_text) or \
            ("Conversation summary" in user_text) or \
            ("A new session was started" in user_text):
            if _fr_only_mode():
                return _build_completion_response(
                    _last_assistant_text(messages),
                    stream=original_request.get("stream", False),
                )
            upstream_timing: dict[str, Any] = {}
            _user_text_heartbeat = user_text

            def _finalize_heartbeat() -> None:
                if upstream_timing:
                    _record_tool_history(
                        _user_text_heartbeat,
                        [],
                        0,
                        session_key,
                        llm_calls=[upstream_timing],
                    )

            response = await proxy_upstream(
                original_request,
                session_key=session_key,
                out_timing=upstream_timing,
                on_stream_end=_finalize_heartbeat,
            )
            log_request(
                user_message=user_text,
                route="upstream",
                function_name=None,
                tool_rounds=0,
                latency_ms=(time.perf_counter() - started_at) * 1000,
                status="skipped_qwen_heartbeat",
            )
            _debug_log(
                "route_decision",
                session_key=session_key,
                route="upstream",
                status="skipped_qwen_heartbeat",
                function_name=function_name,
                tool_rounds=tool_rounds,
                latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
            )
            return response


        # 检查是否需要调用Qwen：只有最后一条是user消息时才需要
        # 如果最后一条是assistant或tool，说明是豆包自己在处理或工具在返回，
        # 不应该再调用Qwen（否则会用之前的user消息重复调用）
        last_role = messages[-1].get("role") if messages else None
        if last_role and last_role != "user" and delegated_continuation is None:
            if _fr_only_mode():
                return _build_completion_response(
                    _last_assistant_text(messages),
                    stream=original_request.get("stream", False),
                )
            upstream_timing: dict[str, Any] = {}
            _user_text_continuation = user_text

            def _finalize_continuation() -> None:
                if upstream_timing:
                    _record_tool_history(
                        _user_text_continuation,
                        [],
                        1,
                        session_key,
                        llm_calls=[upstream_timing],
                    )

            response = await proxy_upstream(
                original_request,
                session_key=session_key,
                out_timing=upstream_timing,
                on_stream_end=_finalize_continuation,
            )
            log_request(
                user_message=user_text,
                route="upstream",
                function_name=None,
                tool_rounds=1,
                latency_ms=(time.perf_counter() - started_at) * 1000,
                status="skipped_qwen_continuation",
            )
            _debug_log(
                "route_decision",
                session_key=session_key,
                route="upstream",
                status="skipped_qwen_continuation",
                function_name=function_name,
                tool_rounds=tool_rounds,
                latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
            )
            return response

        ctx_enabled = bool(
            STATE.config and STATE.config.fr_context_history
        )
        ctx_preserve = bool(
            STATE.config and STATE.config.fr_context_preserve
        )
        if STATE.logger:
            STATE.logger.info(
                "ctx_enabled=%s, ctx_preserve=%s, saved_context_len=%d",
                ctx_enabled, ctx_preserve, len(_get_saved_context(session_key)),
            )

        try:
            result = await run_tool_loop(
                user_text,
                history=(list(_get_saved_context(session_key)) or None) if ctx_enabled else None,
                delegated_tool_names=(delegated_names or None),
                resume_tool_context=(
                    delegated_continuation[0]
                    if delegated_continuation is not None
                    else None
                ),
            )
            function_name = result.last_function_name
            tool_rounds = result.tool_rounds
        except (httpx.HTTPError, RuntimeError, asyncio.TimeoutError) as exc:
            if not ctx_preserve:
                _clear_saved_context(session_key)
            if STATE.logger is not None:
                STATE.logger.warning("qwen routing failed, falling back upstream: %s", exc)
            if _fr_only_mode():
                log_request(
                    user_message=user_text,
                    route="function",
                    function_name=function_name,
                    tool_rounds=tool_rounds,
                    latency_ms=(time.perf_counter() - started_at) * 1000,
                    status=f"qwen_error_always_true:{type(exc).__name__}",
                )
                return _build_completion_response(
                    "",
                    stream=original_request.get("stream", False),
                )
            upstream_timing: dict[str, Any] = {}
            _user_text_fallback = user_text
            _fallback_rounds = tool_rounds

            def _finalize_fallback() -> None:
                if upstream_timing:
                    _record_tool_history(
                        _user_text_fallback,
                        [],
                        _fallback_rounds,
                        session_key,
                        llm_calls=[upstream_timing],
                    )

            response = await proxy_upstream(
                original_request,
                session_key=session_key,
                out_timing=upstream_timing,
                on_stream_end=_finalize_fallback,
            )
            log_request(
                user_message=user_text,
                route="upstream",
                function_name=function_name,
                tool_rounds=tool_rounds,
                latency_ms=(time.perf_counter() - started_at) * 1000,
                status="fallback_upstream",
            )
            _debug_log(
                "route_decision",
                session_key=session_key,
                route="upstream",
                status="fallback_upstream",
                function_name=function_name,
                tool_rounds=tool_rounds,
                latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
            )
            return response

        if result.direct_response is not None:
            _record_tool_history(
                user_text,
                result.tool_context,
                tool_rounds,
                session_key,
                llm_calls=result.llm_calls,
            )

            log_request(
                user_message=user_text,
                route="function",
                function_name=function_name,
                tool_rounds=tool_rounds,
                latency_ms=(
                    time.perf_counter() - started_at
                ) * 1000,
                status="internal_tool_completed",
            )

            _debug_log(
                "route_decision",
                session_key=session_key,
                route="function",
                status="internal_tool_completed",
                function_name=function_name,
                tool_rounds=tool_rounds,
                latency_ms=round(
                    (
                        time.perf_counter()
                        - started_at
                    ) * 1000,
                    2,
                ),
            )

            return _build_completion_response(
                result.direct_response,
                stream=original_request.get(
                    "stream",
                    False,
                ),
            )

        if not result.used_any_tool:
            if _fr_only_mode():
                if ctx_enabled and result._loop_messages:
                    _set_saved_context(session_key, result._loop_messages[1:])
                _record_tool_history(
                    user_text,
                    result.tool_context,
                    tool_rounds,
                    session_key,
                    llm_calls=result.llm_calls,
                )
                log_request(
                    user_message=user_text,
                    route="function",
                    function_name=function_name,
                    tool_rounds=tool_rounds,
                    latency_ms=(time.perf_counter() - started_at) * 1000,
                    status="qwen_completed_always_true",
                )
                return _build_completion_response(
                    result.qwen_reply or _last_assistant_text(result._loop_messages),
                    stream=original_request.get("stream", False),
                )
            if not ctx_preserve:
                _clear_saved_context(session_key)
            upstream_timing: dict[str, Any] = {}
            _user_text_after_routing = user_text
            _after_routing_rounds = tool_rounds
            _after_routing_tool_context = result.tool_context
            _after_routing_llm_calls = result.llm_calls

            def _finalize_after_routing() -> None:
                if upstream_timing:
                    _after_routing_llm_calls.append(upstream_timing)
                _record_tool_history(
                    _user_text_after_routing,
                    _after_routing_tool_context,
                    _after_routing_rounds,
                    session_key,
                    llm_calls=_after_routing_llm_calls,
                )

            response = await proxy_upstream(
                original_request,
                session_key=session_key,
                out_timing=upstream_timing,
                on_stream_end=_finalize_after_routing,
            )
            log_request(
                user_message=user_text,
                route="upstream",
                function_name=function_name,
                tool_rounds=tool_rounds,
                latency_ms=(time.perf_counter() - started_at) * 1000,
                status="forwarded_after_routing",
            )
            _debug_log(
                "route_decision",
                session_key=session_key,
                route="upstream",
                status="forwarded_after_routing",
                function_name=function_name,
                tool_rounds=tool_rounds,
                latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
            )
            return response
        if result.delegated_tool_calls:
            _mark_pending_delegated_tool_calls(session_key, result.delegated_tool_calls)
            log_request(
                user_message=user_text,
                route="function",
                function_name=function_name,
                tool_rounds=tool_rounds,
                latency_ms=(time.perf_counter() - started_at) * 1000,
                status="delegated_tool_call",
            )
            _debug_log(
                "route_decision",
                session_key=session_key,
                route="function",
                status="delegated_tool_call",
                function_name=function_name,
                tool_rounds=tool_rounds,
                latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
            )
            return _build_tool_calls_response(
                result.delegated_tool_calls,
                stream=original_request.get("stream", False),
            )
        # Tools were used — check if Qwen judges the task complete (short-circuit)
        # or fall through to upstream (Doubao) for the final response.
        if (
            STATE.config
            and STATE.config.fr_completion_check
            and not result.max_rounds_exhausted
            and result.qwen_reply
        ):
            task_complete = _fr_only_mode() or await call_qwen_completion_check(
                result._loop_messages,
                out_llm_calls=result.llm_calls,
            )
            if task_complete:
                if ctx_enabled:
                    _set_saved_context(session_key, result._loop_messages[1:])
                    if STATE.logger:
                        STATE.logger.info(
                            "saved context[%s]: %d messages", session_key, len(_get_saved_context(session_key)),
                        )
                # Delegated continuation turns are already fully persisted by
                # OpenClaw (user, assistant.tool_calls, tool result, and this
                # reply) — queueing a pending plain-text turn would inject a
                # duplicate copy into future upstream requests.
                if user_text and result.qwen_reply and delegated_continuation is None:
                    _append_pending_upstream_turn(session_key, user_text, result.qwen_reply)
                _record_tool_history(
                    user_text,
                    result.tool_context,
                    tool_rounds,
                    session_key,
                    llm_calls=result.llm_calls,
                )
                log_request(
                    user_message=user_text,
                    route="function",
                    function_name=function_name,
                    tool_rounds=tool_rounds,
                    latency_ms=(time.perf_counter() - started_at) * 1000,
                    status="qwen_completed",
                )
                _debug_log(
                    "route_decision",
                    session_key=session_key,
                    route="function",
                    status="qwen_completed",
                    function_name=function_name,
                    tool_rounds=tool_rounds,
                    latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
                )
                return _build_completion_response(
                    result.qwen_reply,
                    stream=original_request.get("stream", False),
                )

        if _fr_only_mode():
            if ctx_enabled and result._loop_messages:
                _set_saved_context(session_key, result._loop_messages[1:])
            if user_text and result.qwen_reply and delegated_continuation is None:
                _append_pending_upstream_turn(session_key, user_text, result.qwen_reply)
            _record_tool_history(
                user_text,
                result.tool_context,
                tool_rounds,
                session_key,
                llm_calls=result.llm_calls,
            )
            log_request(
                user_message=user_text,
                route="function",
                function_name=function_name,
                tool_rounds=tool_rounds,
                latency_ms=(time.perf_counter() - started_at) * 1000,
                status="qwen_completed_always_true",
            )
            return _build_completion_response(
                result.qwen_reply or _last_assistant_text(result._loop_messages),
                stream=original_request.get("stream", False),
            )

        # Fall through: forward to upstream without injecting tool context.
        # This prevents upstream models (Doubao) from hallucinating Qwen's local tools.
        # Only clear if not preserving context.
        if not ctx_preserve:
            _clear_saved_context(session_key)
        elif STATE.logger:
            STATE.logger.info("ctx_preserve[%s]: keeping %d messages", session_key, len(_get_saved_context(session_key)))
        if result.max_rounds_exhausted:
            status = "tool_max_rounds_to_upstream"
        else:
            status = "tool_result_to_upstream"
        upstream_timing: dict[str, Any] = {}
        _user_text_final = user_text
        _final_rounds = tool_rounds
        _final_tool_context = result.tool_context
        _final_llm_calls = result.llm_calls

        def _finalize_final() -> None:
            if upstream_timing:
                _final_llm_calls.append(upstream_timing)
            _record_tool_history(
                _user_text_final,
                _final_tool_context,
                _final_rounds,
                session_key,
                llm_calls=_final_llm_calls,
            )

        response = await proxy_upstream(
            original_request,
            tool_context=None,
            session_key=session_key,
            out_timing=upstream_timing,
            on_stream_end=_finalize_final,
        )
        log_request(
            user_message=user_text,
            route="function",
            function_name=function_name,
            tool_rounds=tool_rounds,
            latency_ms=(time.perf_counter() - started_at) * 1000,
            status=status,
        )
        _debug_log(
            "route_decision",
            session_key=session_key,
            route="function",
            status=status,
            function_name=function_name,
            tool_rounds=tool_rounds,
            latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
        )
        return response
    except httpx.HTTPError as exc:
        if STATE.logger is not None:
            STATE.logger.exception("upstream proxy failure")
        log_request(
            user_message=user_text,
            route="upstream",
            function_name=function_name,
            tool_rounds=tool_rounds,
            latency_ms=(time.perf_counter() - started_at) * 1000,
            status=f"error:{type(exc).__name__}",
        )
        _debug_log(
            "route_decision",
            session_key=session_key,
            route="upstream",
            status=f"error:{type(exc).__name__}",
            function_name=function_name,
            tool_rounds=tool_rounds,
            latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
        )
        raise HTTPException(status_code=502, detail="upstream request failed") from exc


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Function Router service")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to config.json (default: ~/.function-router/config.json)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    STATE.config_path = Path(args.config).expanduser().resolve()
    STATE.config_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        config = load_config(STATE.config_path)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    setup_logging(config.root_dir, debug_logging=config.debug_logging)

    try:
        load_tools(config.functions_path)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    uvicorn.run(
        app,
        host=config.listen_host,
        port=config.listen_port,
        log_config=None,
    )


if __name__ == "__main__":
    main()
