"""Load and save UISettings to disk as JSON."""

from __future__ import annotations

import json
from pathlib import Path

from textualgrok.ui_types import UISettings


def settings_file_path() -> Path:
    """Return the path to the settings file."""
    return Path.cwd() / ".textual-grok-settings.json"


def legacy_settings_file_path() -> Path:
    """Return the path to the legacy settings file."""
    return Path.cwd() / ".textualbot_settings.json"


def load_ui_settings(default: UISettings) -> UISettings:
    """Load UISettings from disk, falling back to legacy path or default."""
    path = settings_file_path()
    if not path.exists():
        legacy_path = legacy_settings_file_path()
        if legacy_path.exists():
            path = legacy_path
        else:
            return default

    try:
        data = path.read_text(encoding="utf-8")
        parsed = json.loads(data)
    except (OSError, json.JSONDecodeError):
        return default
    if not isinstance(parsed, dict):
        return default

    settings_data = default.__dict__.copy()
    for key in settings_data.keys():
        if key not in parsed:
            continue
        value = parsed[key]
        if key == "mcp_servers":
            if isinstance(value, list):
                normalized_servers: list[dict[str, object]] = []
                for item in value:
                    if isinstance(item, dict):
                        normalized_servers.append(item)
                settings_data[key] = normalized_servers
        else:
            settings_data[key] = value

    try:
        return UISettings(**settings_data)
    except TypeError:
        return default


def save_ui_settings(settings: UISettings) -> bool:
    """Save UISettings to disk. Returns True on success, False on failure."""
    path = settings_file_path()
    try:
        path.write_text(json.dumps(settings.__dict__, indent=2), encoding="utf-8")
        return True
    except OSError:
        return False
