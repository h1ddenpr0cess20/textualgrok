"""Chat log rendering and transcript tracking."""

from __future__ import annotations

from typing import Any

from pygments.styles import get_style_by_name
from pygments.util import ClassNotFound
from rich.console import Group
from rich.markdown import Markdown
from rich.text import Text
from textual.widgets import RichLog


def write_chat_log_entry(entry: dict[str, Any], chat_log: RichLog, *, code_theme: str) -> None:
    """Write a single chat log entry to the RichLog widget."""
    kind = entry.get("kind")
    message = str(entry.get("message", ""))

    if kind == "assistant":
        chat_log.write(Group(Text("Grok:"), render_markdown(message, code_theme=code_theme)))
        return
    if kind == "error":
        chat_log.write(Text(f"Error: {message}", style="bold red"))
        return

    attachment_count = entry.get("attachment_count", 0)
    prefix = "You:"
    if isinstance(attachment_count, int) and attachment_count > 0:
        prefix = f"You (+{attachment_count} attachment{'s' if attachment_count != 1 else ''}):"
    chat_log.write(Text(f"{prefix} {message}"))


def render_markdown(message: str, *, code_theme: str) -> Markdown:
    """Render a message as Rich Markdown with a code theme."""
    try:
        return Markdown(message, code_theme=code_theme)
    except Exception:
        return Markdown(message)


def code_theme_for_app_theme(theme_name: str, *, is_dark: bool) -> str:
    """Determine the best Pygments code theme for the current app theme."""
    clean_name = theme_name.strip() if isinstance(theme_name, str) else ""
    if is_valid_code_theme(clean_name):
        return clean_name
    return "ansi_dark" if is_dark else "ansi_light"


def is_valid_code_theme(style_name: str) -> bool:
    """Return True if the style name is a valid Pygments style."""
    if not style_name:
        return False
    try:
        get_style_by_name(style_name)
    except ClassNotFound:
        return False
    return True
