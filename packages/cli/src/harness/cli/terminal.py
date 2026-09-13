"""Async terminal editing with paste, persistent history and slash completion."""

from __future__ import annotations

import os
import sys

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings

from harness.core.paths import user_home

COMMANDS = [
    "/help",
    "/quit",
    "/new",
    "/switch",
    "/sessions",
    "/session",
    "/send",
    "/cancel",
    "/steer",
    "/retry",
    "/undo",
    "/compress",
    "/usage",
    "/attach",
    "/model",
    "/skill",
    "/tools",
    "/diff",
    "/clear",
]


def interactive_terminal() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty() and os.environ.get("TERM") != "dumb"


def make_prompt_session() -> PromptSession[str]:
    history = user_home() / "history"
    history.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(history, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    os.close(fd)
    history.chmod(0o600)
    keys = KeyBindings()

    @keys.add("enter")
    def submit(event):
        event.current_buffer.validate_and_handle()

    @keys.add("escape", "enter")
    def newline(event):
        event.current_buffer.insert_text("\n")

    return PromptSession(
        history=FileHistory(str(history)),
        completer=WordCompleter(COMMANDS),
        complete_while_typing=False,
        enable_history_search=True,
        multiline=True,
        key_bindings=keys,
        bottom_toolbar="Enter: send · Alt+Enter: newline · Ctrl+R: history · Ctrl+C: cancel turn",
    )
