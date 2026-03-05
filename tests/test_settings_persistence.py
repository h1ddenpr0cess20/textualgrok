"""Tests for settings_persistence module."""

import json
import pytest
from pathlib import Path

from textualgrok.ui_types import UISettings
from textualgrok.settings_persistence import load_ui_settings, save_ui_settings


def default_settings():
    return UISettings(chat_model="grok-3", system_prompt="Be helpful.")


class TestSaveAndLoadRoundtrip:
    def test_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        settings = default_settings()
        assert save_ui_settings(settings) is True
        loaded = load_ui_settings(default_settings())
        assert loaded.chat_model == settings.chat_model
        assert loaded.system_prompt == settings.system_prompt
        assert loaded.theme == settings.theme

    def test_modified_field_persists(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        settings = UISettings(chat_model="grok-2", system_prompt="Be concise.", theme="nord")
        save_ui_settings(settings)
        loaded = load_ui_settings(default_settings())
        assert loaded.theme == "nord"
        assert loaded.chat_model == "grok-2"


class TestLoadFallbacks:
    def test_returns_default_if_no_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = load_ui_settings(default_settings())
        assert result.chat_model == "grok-3"

    def test_returns_default_on_invalid_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".textual-grok-settings.json").write_text("not json")
        result = load_ui_settings(default_settings())
        assert result.chat_model == "grok-3"

    def test_returns_default_on_non_dict_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".textual-grok-settings.json").write_text("[1, 2, 3]")
        result = load_ui_settings(default_settings())
        assert result.chat_model == "grok-3"

    def test_loads_legacy_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data = default_settings().__dict__.copy()
        data["theme"] = "solarized-light"
        (tmp_path / ".textualbot_settings.json").write_text(json.dumps(data))
        result = load_ui_settings(default_settings())
        assert result.theme == "solarized-light"

    def test_primary_takes_precedence_over_legacy(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        primary_data = default_settings().__dict__.copy()
        primary_data["theme"] = "primary-theme"
        legacy_data = default_settings().__dict__.copy()
        legacy_data["theme"] = "legacy-theme"
        (tmp_path / ".textual-grok-settings.json").write_text(json.dumps(primary_data))
        (tmp_path / ".textualbot_settings.json").write_text(json.dumps(legacy_data))
        result = load_ui_settings(default_settings())
        assert result.theme == "primary-theme"

    def test_unknown_keys_ignored(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data = default_settings().__dict__.copy()
        data["future_unknown_key"] = "ignored"
        (tmp_path / ".textual-grok-settings.json").write_text(json.dumps(data))
        result = load_ui_settings(default_settings())
        assert not hasattr(result, "future_unknown_key")

    def test_mcp_servers_normalized(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data = default_settings().__dict__.copy()
        data["mcp_servers"] = [{"server_url": "http://x"}, "invalid_string"]
        (tmp_path / ".textual-grok-settings.json").write_text(json.dumps(data))
        result = load_ui_settings(default_settings())
        # Non-dict items should be filtered out
        assert all(isinstance(s, dict) for s in result.mcp_servers)
        assert len(result.mcp_servers) == 1
