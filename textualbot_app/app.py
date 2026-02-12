from dataclasses import asdict, dataclass, field
import base64
from datetime import datetime
import json
import textwrap
from typing import Any, Optional
import tempfile
from pathlib import Path
import re

from rich.console import Group
from rich.markdown import Markdown
from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widget import Widget
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Static,
    Switch,
    TabPane,
    TabbedContent,
    TextArea,
)
from textual.worker import Worker, WorkerState

from textualbot_app.config import AppConfig
from textualbot_app.conversation import ConversationState
from textualbot_app.models import ChatMessage, ChatResult
from textualbot_app.options import RequestOptions, build_request_options
from textualbot_app.xai_client import XAIResponsesClient

from textual_image.widget import Image as TextualImageWidget


@dataclass
class UISettings:
    chat_model: str
    system_prompt: str
    include_history: bool = True
    web_search: bool = True
    x_search: bool = True
    x_search_image_understanding: bool = False
    x_search_video_understanding: bool = False
    code_interpreter: bool = True
    file_search: bool = False
    vector_store_ids_raw: str = ""
    file_search_max_results_raw: str = "10"
    image_generation: bool = False
    image_as_base64: bool = False
    image_model: str = "grok-imagine-image"
    image_count_raw: str = "1"
    image_source_url_raw: str = ""
    image_use_last: bool = True
    image_aspect_ratio_raw: str = "1:1"
    mcp_enabled: bool = False
    mcp_servers: list[dict[str, Any]] = field(default_factory=list)


class SettingsScreen(ModalScreen[Optional[UISettings]]):
    VALID_IMAGE_ASPECT_RATIOS: tuple[str, ...] = (
        "1:1",
        "16:9",
        "9:16",
        "4:3",
        "3:4",
        "3:2",
        "2:3",
        "2:1",
        "1:2",
        "19.5:9",
        "9:19.5",
        "20:9",
        "9:20",
    )

    CSS = """
    SettingsScreen {
        align: center middle;
    }

    #settings-dialog {
        width: 92;
        height: 42;
        border: round $accent;
        background: $panel;
        padding: 1 2;
    }

    #settings-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #settings-tabs {
        height: 1fr;
    }

    #mcp-scroll {
        height: 1fr;
    }

    .settings-input {
        width: 100%;
        margin-bottom: 1;
    }

    .settings-row {
        height: auto;
        margin-bottom: 1;
        align: left middle;
    }

    .settings-row Label {
        width: 1fr;
    }

    #model-tools {
        height: auto;
        margin-bottom: 1;
    }

    #models-status {
        color: $text-muted;
        margin-top: 1;
    }

    #settings-actions {
        height: auto;
        margin-top: 1;
        align: right middle;
    }
    """

    def __init__(self, initial: UISettings, client: XAIResponsesClient) -> None:
        super().__init__()
        self.initial = initial
        self.client = client
        self._mcp_servers: list[dict[str, Any]] = [
            self._copy_mcp_server_config(server) for server in initial.mcp_servers
        ]

    def compose(self) -> ComposeResult:
        selected_aspect_ratio = (
            self.initial.image_aspect_ratio_raw
            if self.initial.image_aspect_ratio_raw in self.VALID_IMAGE_ASPECT_RATIOS
            else "1:1"
        )
        aspect_ratio_options = [
            (ratio, ratio) for ratio in self.VALID_IMAGE_ASPECT_RATIOS
        ]
        with Vertical(id="settings-dialog"):
            yield Static("Settings", id="settings-title")
            with TabbedContent(id="settings-tabs"):
                with TabPane("Chat"):
                    yield Label("Model")
                    yield Select(
                        [(self.initial.chat_model, self.initial.chat_model)],
                        value=self.initial.chat_model,
                        allow_blank=False,
                        id="select-chat-model",
                        classes="settings-input",
                    )
                    with Horizontal(id="model-tools"):
                        yield Button("Refresh Models", id="btn-refresh-models")
                    yield Static("Loading models from server...", id="models-status")
                    yield Label("System Prompt")
                    yield Input(
                        value=self.initial.system_prompt,
                        id="input-system-prompt",
                        classes="settings-input",
                    )
                    with Horizontal(classes="settings-row"):
                        yield Label("Use Conversation History")
                        yield Switch(value=self.initial.include_history, id="switch-include-history")

                with TabPane("Tools"):
                    with Horizontal(classes="settings-row"):
                        yield Label("Web Search")
                        yield Switch(value=self.initial.web_search, id="switch-web-search")
                    with Horizontal(classes="settings-row"):
                        yield Label("X Search")
                        yield Switch(value=self.initial.x_search, id="switch-x-search")
                    with Horizontal(classes="settings-row"):
                        yield Label("X Search: Image Understanding")
                        yield Switch(
                            value=self.initial.x_search_image_understanding,
                            id="switch-x-search-image",
                        )
                    with Horizontal(classes="settings-row"):
                        yield Label("X Search: Video Understanding")
                        yield Switch(
                            value=self.initial.x_search_video_understanding,
                            id="switch-x-search-video",
                        )
                    with Horizontal(classes="settings-row"):
                        yield Label("Code Interpreter")
                        yield Switch(value=self.initial.code_interpreter, id="switch-code-interpreter")
                    with Horizontal(classes="settings-row"):
                        yield Label("File Search")
                        yield Switch(value=self.initial.file_search, id="switch-file-search")

                with TabPane("File Search"):
                    yield Label("Vector Store IDs (comma-separated)")
                    yield Input(
                        value=self.initial.vector_store_ids_raw,
                        placeholder="vs_123,vs_456",
                        id="input-vector-store-ids",
                        classes="settings-input",
                    )
                    yield Label("File Search Max Results")
                    yield Input(
                        value=self.initial.file_search_max_results_raw,
                        id="input-file-search-max",
                        classes="settings-input",
                    )

                with TabPane("Image"):
                    with Horizontal(classes="settings-row"):
                        yield Label("Enable Grok Imagine Tool")
                        yield Switch(value=self.initial.image_generation, id="switch-image-generation")
                    with Horizontal(classes="settings-row"):
                        yield Label("Return Image As Base64")
                        yield Switch(value=self.initial.image_as_base64, id="switch-image-b64")
                    with Horizontal(classes="settings-row"):
                        yield Label("Use Last Generated Image For Edits")
                        yield Switch(value=self.initial.image_use_last, id="switch-image-use-last")
                    yield Label("Grok Imagine Model")
                    yield Select(
                        [(self.initial.image_model, self.initial.image_model)],
                        value=self.initial.image_model,
                        allow_blank=False,
                        id="input-image-model",
                        classes="settings-input",
                    )
                    yield Label("Image Count (1-10)")
                    yield Input(
                        value=self.initial.image_count_raw,
                        id="input-image-count",
                        classes="settings-input",
                    )
                    yield Label("Aspect Ratio")
                    yield Select(
                        aspect_ratio_options,
                        value=selected_aspect_ratio,
                        allow_blank=False,
                        id="select-image-aspect-ratio",
                        classes="settings-input",
                    )
                    yield Label("Source Image URL (optional, for edits)")
                    yield Input(
                        value=self.initial.image_source_url_raw,
                        id="input-image-source-url",
                        classes="settings-input",
                    )

                with TabPane("MCP"):
                    with VerticalScroll(id="mcp-scroll"):
                        with Horizontal(classes="settings-row"):
                            yield Label("Enable MCP Tools")
                            yield Switch(value=self.initial.mcp_enabled, id="switch-mcp-enabled")
                        with Horizontal(classes="settings-row"):
                            yield Button("Add Server", id="btn-mcp-add")
                            yield Button("Remove Selected", id="btn-mcp-remove")
                        yield Label("Configured MCP Servers")
                        yield Select(
                            [],
                            allow_blank=True,
                            id="select-mcp-servers",
                            classes="settings-input",
                        )
                        yield Static("No MCP servers configured.", id="mcp-status")
                        yield Label("Server Label (optional)")
                        yield Input(
                            value="",
                            placeholder="deepwiki",
                            id="input-mcp-label",
                            classes="settings-input",
                        )
                        yield Label("Server URL")
                        yield Input(
                            value="",
                            placeholder="https://example.com/mcp",
                            id="input-mcp-url",
                            classes="settings-input",
                        )
                        yield Label("Server Description (optional)")
                        yield Input(
                            value="",
                            id="input-mcp-description",
                            classes="settings-input",
                        )
                        yield Label("Allowed Tool Names (comma-separated, optional)")
                        yield Input(
                            value="",
                            id="input-mcp-allowed-tools",
                            classes="settings-input",
                        )
                        yield Label("Authorization Header Value (optional)")
                        yield Input(
                            value="",
                            id="input-mcp-authorization",
                            classes="settings-input",
                        )
                        yield Label("Extra Headers JSON (optional)")
                        yield Input(
                            value="",
                            placeholder='{"X-Org":"demo"}',
                            id="input-mcp-extra-headers",
                            classes="settings-input",
                        )

            with Horizontal(id="settings-actions"):
                yield Button("Cancel", variant="default", id="btn-settings-cancel")
                yield Button("Save", variant="primary", id="btn-settings-save")

    def on_mount(self) -> None:
        self._sync_inputs()
        self._refresh_mcp_servers_ui()
        self._refresh_models()

    @on(Button.Pressed, "#btn-refresh-models")
    def on_refresh_models_clicked(self, _: Button.Pressed) -> None:
        self._refresh_models()

    @on(Button.Pressed, "#btn-settings-cancel")
    def on_cancel_clicked(self, _: Button.Pressed) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#btn-settings-save")
    def on_save_clicked(self, _: Button.Pressed) -> None:
        try:
            self._upsert_mcp_form_on_save()
            settings = self._collect_settings()
            build_request_options(
                web_search=settings.web_search,
                x_search=settings.x_search,
                x_search_image_understanding=settings.x_search_image_understanding,
                x_search_video_understanding=settings.x_search_video_understanding,
                code_interpreter=settings.code_interpreter,
                file_search=settings.file_search,
                vector_store_ids_raw=settings.vector_store_ids_raw,
                file_search_max_results_raw=settings.file_search_max_results_raw,
                image_generation=settings.image_generation,
                image_model=settings.image_model,
                image_count_raw=settings.image_count_raw,
                image_as_base64=settings.image_as_base64,
                image_source_url_raw=settings.image_source_url_raw,
                image_use_last=settings.image_use_last,
                image_aspect_ratio_raw=settings.image_aspect_ratio_raw,
                mcp_enabled=settings.mcp_enabled,
                mcp_servers=settings.mcp_servers,
            )
        except ValueError as exc:
            message = f"Validation error: {exc}"
            if "MCP" in str(exc):
                self.query_one("#mcp-status", Static).update(message)
            else:
                self.query_one("#models-status", Static).update(message)
            return

        self.dismiss(settings)

    @on(Button.Pressed, "#btn-mcp-add")
    def on_mcp_add_clicked(self, _: Button.Pressed) -> None:
        try:
            server_config = self._read_mcp_server_form()
        except ValueError as exc:
            self.query_one("#mcp-status", Static).update(f"Validation error: {exc}")
            return

        self._mcp_servers.append(server_config)
        selected_index = len(self._mcp_servers) - 1
        self.query_one("#mcp-status", Static).update("MCP server added.")
        self._refresh_mcp_servers_ui(selected_index=selected_index)

    @on(Button.Pressed, "#btn-mcp-remove")
    def on_mcp_remove_clicked(self, _: Button.Pressed) -> None:
        select = self.query_one("#select-mcp-servers", Select)
        selected = select.value
        if not isinstance(selected, str) or not selected.strip():
            self.query_one("#mcp-status", Static).update("Select an MCP server to remove.")
            return

        try:
            index = int(selected)
        except ValueError:
            self.query_one("#mcp-status", Static).update("Select an MCP server to remove.")
            return

        if index < 0 or index >= len(self._mcp_servers):
            self.query_one("#mcp-status", Static).update("Selected MCP server no longer exists.")
            self._refresh_mcp_servers_ui()
            return

        self._mcp_servers.pop(index)
        self.query_one("#mcp-status", Static).update("MCP server removed.")
        self._refresh_mcp_servers_ui()

    @on(Select.Changed, "#select-mcp-servers")
    def on_mcp_server_selected(self, event: Select.Changed) -> None:
        value = event.value
        if not isinstance(value, str) or not value.strip():
            return
        try:
            index = int(value)
        except ValueError:
            return
        if index < 0 or index >= len(self._mcp_servers):
            return
        self._fill_mcp_server_form(self._mcp_servers[index])

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if event.switch.id in {"switch-x-search-image", "switch-x-search-video"} and event.value:
            x_search_switch = self.query_one("#switch-x-search", Switch)
            if not x_search_switch.value:
                x_search_switch.value = True
        self._sync_inputs()

    def _refresh_models(self) -> None:
        self.query_one("#models-status", Static).update("Loading models from server...")
        refresh_button = self.query_one("#btn-refresh-models", Button)
        refresh_button.disabled = True
        self.run_worker(self._load_models, thread=True, exclusive=True, name="load-models")

    def _load_models(self) -> list[str]:
        return self.client.list_models()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name != "load-models":
            return
        if event.state != WorkerState.SUCCESS and event.state != WorkerState.ERROR:
            return

        refresh_button = self.query_one("#btn-refresh-models", Button)
        refresh_button.disabled = False

        if event.state == WorkerState.SUCCESS:
            models = event.worker.result
            if not isinstance(models, list) or not models:
                self.query_one("#models-status", Static).update("No models returned by server.")
                return
            self._set_model_options(models)
            self.query_one("#models-status", Static).update(
                f"Loaded {len(models)} model(s) from server."
            )
        else:
            self.query_one("#models-status", Static).update(f"Model list load failed: {event.worker.error}")

    def _set_model_options(self, models: list[str]) -> None:
        select_map = {
            "#select-chat-model": self.initial.chat_model,
            "#input-image-model": self.initial.image_model,
        }
        options = [(model, model) for model in models]

        for selector, fallback_default in select_map.items():
            select = self.query_one(selector, Select)
            current_value = select.value
            fallback = (
                current_value
                if isinstance(current_value, str) and current_value
                else fallback_default
            )
            select.set_options(options)
            if fallback in models:
                select.value = fallback
            else:
                select.value = models[0]

    def _refresh_mcp_servers_ui(self, selected_index: int | None = None) -> None:
        select = self.query_one("#select-mcp-servers", Select)
        remove_button = self.query_one("#btn-mcp-remove", Button)

        options: list[tuple[str, str]] = []
        for index, server in enumerate(self._mcp_servers):
            label = str(server.get("server_label", "")).strip() or f"server-{index + 1}"
            url = str(server.get("server_url", "")).strip()
            options.append((f"{label} ({url})", str(index)))

        select.set_options(options)

        if options:
            if selected_index is not None and 0 <= selected_index < len(options):
                select.value = str(selected_index)
                self._fill_mcp_server_form(self._mcp_servers[selected_index])
            else:
                current_value = select.value
                if isinstance(current_value, str) and current_value in {value for _, value in options}:
                    try:
                        current_index = int(current_value)
                        self._fill_mcp_server_form(self._mcp_servers[current_index])
                    except (ValueError, IndexError):
                        select.value = options[0][1]
                        self._fill_mcp_server_form(self._mcp_servers[0])
                else:
                    select.value = options[0][1]
                    self._fill_mcp_server_form(self._mcp_servers[0])
            remove_button.disabled = False
            self.query_one("#mcp-status", Static).update(f"{len(options)} MCP server(s) configured.")
        else:
            select.value = Select.BLANK
            remove_button.disabled = True
            self.query_one("#mcp-status", Static).update("No MCP servers configured.")
        self._sync_inputs()

    def _fill_mcp_server_form(self, server: dict[str, Any]) -> None:
        self.query_one("#input-mcp-label", Input).value = str(server.get("server_label", "")).strip()
        self.query_one("#input-mcp-url", Input).value = str(server.get("server_url", "")).strip()
        self.query_one("#input-mcp-description", Input).value = str(
            server.get("server_description", "")
        ).strip()

        allowed = server.get("allowed_tool_names")
        allowed_names: list[str] = []
        if isinstance(allowed, list):
            for item in allowed:
                name = str(item).strip()
                if name:
                    allowed_names.append(name)
        self.query_one("#input-mcp-allowed-tools", Input).value = ",".join(allowed_names)

        self.query_one("#input-mcp-authorization", Input).value = str(
            server.get("authorization", "")
        ).strip()

        headers = server.get("extra_headers")
        if isinstance(headers, dict) and headers:
            self.query_one("#input-mcp-extra-headers", Input).value = json.dumps(headers)
        else:
            self.query_one("#input-mcp-extra-headers", Input).value = ""

    def _read_mcp_server_form(self) -> dict[str, Any]:
        label = self.query_one("#input-mcp-label", Input).value.strip()
        server_url = self.query_one("#input-mcp-url", Input).value.strip()
        description = self.query_one("#input-mcp-description", Input).value.strip()
        allowed_raw = self.query_one("#input-mcp-allowed-tools", Input).value.strip()
        authorization = self.query_one("#input-mcp-authorization", Input).value.strip()
        extra_headers_raw = self.query_one("#input-mcp-extra-headers", Input).value.strip()

        if not server_url:
            raise ValueError("Server URL is required.")

        server: dict[str, Any] = {"server_url": server_url}
        if label:
            server["server_label"] = label
        if description:
            server["server_description"] = description
        if authorization:
            server["authorization"] = authorization

        allowed_tool_names = [item.strip() for item in allowed_raw.split(",") if item.strip()]
        if allowed_tool_names:
            server["allowed_tool_names"] = allowed_tool_names

        if extra_headers_raw:
            try:
                parsed_headers = json.loads(extra_headers_raw)
            except json.JSONDecodeError as exc:
                raise ValueError("Extra headers must be valid JSON object.") from exc
            if not isinstance(parsed_headers, dict):
                raise ValueError("Extra headers must be a JSON object.")
            normalized_headers: dict[str, str] = {}
            for key, value in parsed_headers.items():
                header_key = str(key).strip()
                header_value = str(value).strip()
                if header_key and header_value:
                    normalized_headers[header_key] = header_value
            if normalized_headers:
                server["extra_headers"] = normalized_headers

        return server

    @staticmethod
    def _copy_mcp_server_config(server: dict[str, Any]) -> dict[str, Any]:
        copied: dict[str, Any] = {}

        server_url = str(server.get("server_url", "")).strip()
        if server_url:
            copied["server_url"] = server_url

        server_label = str(server.get("server_label", "")).strip()
        if server_label:
            copied["server_label"] = server_label

        server_description = str(server.get("server_description", "")).strip()
        if server_description:
            copied["server_description"] = server_description

        authorization = str(server.get("authorization", "")).strip()
        if authorization:
            copied["authorization"] = authorization

        raw_allowed = server.get("allowed_tool_names")
        if isinstance(raw_allowed, list):
            allowed = [str(item).strip() for item in raw_allowed if str(item).strip()]
            if allowed:
                copied["allowed_tool_names"] = allowed

        raw_headers = server.get("extra_headers")
        if isinstance(raw_headers, dict):
            headers: dict[str, str] = {}
            for key, value in raw_headers.items():
                header_key = str(key).strip()
                header_value = str(value).strip()
                if header_key and header_value:
                    headers[header_key] = header_value
            if headers:
                copied["extra_headers"] = headers

        return copied

    def _upsert_mcp_form_on_save(self) -> None:
        label = self.query_one("#input-mcp-label", Input).value.strip()
        server_url = self.query_one("#input-mcp-url", Input).value.strip()
        description = self.query_one("#input-mcp-description", Input).value.strip()
        allowed = self.query_one("#input-mcp-allowed-tools", Input).value.strip()
        authorization = self.query_one("#input-mcp-authorization", Input).value.strip()
        headers = self.query_one("#input-mcp-extra-headers", Input).value.strip()

        if not any([label, server_url, description, allowed, authorization, headers]):
            return

        server_config = self._read_mcp_server_form()
        existing_index = next(
            (
                idx
                for idx, server in enumerate(self._mcp_servers)
                if str(server.get("server_url", "")).strip()
                == str(server_config.get("server_url", "")).strip()
            ),
            None,
        )
        if existing_index is None:
            self._mcp_servers.append(server_config)
            selected_index = len(self._mcp_servers) - 1
        else:
            self._mcp_servers[existing_index] = server_config
            selected_index = existing_index
        self._refresh_mcp_servers_ui(selected_index=selected_index)

    def _collect_settings(self) -> UISettings:
        model_value = self.query_one("#select-chat-model", Select).value
        if not isinstance(model_value, str) or not model_value:
            raise ValueError("Please choose a model.")

        system_prompt = self.query_one("#input-system-prompt", Input).value.strip()
        if not system_prompt:
            raise ValueError("System prompt cannot be empty.")

        image_model_value = self.query_one("#input-image-model", Select).value
        image_model = image_model_value.strip() if isinstance(image_model_value, str) else ""
        if not image_model:
            raise ValueError("Image model cannot be empty.")

        image_aspect_ratio_value = self.query_one("#select-image-aspect-ratio", Select).value
        image_aspect_ratio = (
            image_aspect_ratio_value.strip()
            if isinstance(image_aspect_ratio_value, str)
            else ""
        )
        if not image_aspect_ratio:
            raise ValueError("Image aspect ratio cannot be empty.")

        return UISettings(
            chat_model=model_value,
            system_prompt=system_prompt,
            include_history=self.query_one("#switch-include-history", Switch).value,
            web_search=self.query_one("#switch-web-search", Switch).value,
            x_search=self.query_one("#switch-x-search", Switch).value,
            x_search_image_understanding=self.query_one("#switch-x-search-image", Switch).value,
            x_search_video_understanding=self.query_one("#switch-x-search-video", Switch).value,
            code_interpreter=self.query_one("#switch-code-interpreter", Switch).value,
            file_search=self.query_one("#switch-file-search", Switch).value,
            vector_store_ids_raw=self.query_one("#input-vector-store-ids", Input).value,
            file_search_max_results_raw=self.query_one("#input-file-search-max", Input).value,
            image_generation=self.query_one("#switch-image-generation", Switch).value,
            image_as_base64=self.query_one("#switch-image-b64", Switch).value,
            image_model=image_model,
            image_count_raw=self.query_one("#input-image-count", Input).value,
            image_source_url_raw=self.query_one("#input-image-source-url", Input).value,
            image_use_last=self.query_one("#switch-image-use-last", Switch).value,
            image_aspect_ratio_raw=image_aspect_ratio,
            mcp_enabled=self.query_one("#switch-mcp-enabled", Switch).value,
            mcp_servers=[self._copy_mcp_server_config(server) for server in self._mcp_servers],
        )

    def _sync_inputs(self) -> None:
        file_search_enabled = self.query_one("#switch-file-search", Switch).value
        self.query_one("#input-vector-store-ids", Input).disabled = not file_search_enabled
        self.query_one("#input-file-search-max", Input).disabled = not file_search_enabled

        x_search_enabled = self.query_one("#switch-x-search", Switch).value
        x_image_switch = self.query_one("#switch-x-search-image", Switch)
        x_video_switch = self.query_one("#switch-x-search-video", Switch)
        x_image_switch.disabled = not x_search_enabled
        x_video_switch.disabled = not x_search_enabled
        if not x_search_enabled:
            x_image_switch.value = False
            x_video_switch.value = False

        image_generation_enabled = self.query_one("#switch-image-generation", Switch).value
        self.query_one("#switch-image-b64", Switch).disabled = not image_generation_enabled
        self.query_one("#switch-image-use-last", Switch).disabled = not image_generation_enabled
        self.query_one("#input-image-model", Select).disabled = not image_generation_enabled
        self.query_one("#input-image-count", Input).disabled = not image_generation_enabled
        self.query_one("#input-image-source-url", Input).disabled = not image_generation_enabled
        self.query_one("#select-image-aspect-ratio", Select).disabled = not image_generation_enabled

        self.query_one("#input-mcp-label", Input).disabled = False
        self.query_one("#input-mcp-url", Input).disabled = False
        self.query_one("#input-mcp-description", Input).disabled = False
        self.query_one("#input-mcp-allowed-tools", Input).disabled = False
        self.query_one("#input-mcp-authorization", Input).disabled = False
        self.query_one("#input-mcp-extra-headers", Input).disabled = False
        self.query_one("#btn-mcp-add", Button).disabled = False
        self.query_one("#btn-mcp-remove", Button).disabled = len(self._mcp_servers) == 0
        self.query_one("#select-mcp-servers", Select).disabled = len(self._mcp_servers) == 0


@dataclass
class SessionImageItem:
    path: Path
    source: str


class ImageGalleryScreen(Screen[None]):
    CSS = """
    ImageGalleryScreen {
        layout: vertical;
        background: $background;
        overflow-x: hidden;
        overflow-y: hidden;
    }

    #gallery-dialog {
        width: 100%;
        height: 100%;
        background: $panel;
        padding: 1 2;
        overflow-x: hidden;
        overflow-y: hidden;
    }

    #gallery-header {
        height: auto;
        align: left middle;
        margin-bottom: 1;
    }

    #gallery-counter {
        text-style: bold;
    }

    #gallery-image {
        width: 100%;
        height: 1fr;
        overflow-x: hidden;
        overflow-y: hidden;
    }

    #gallery-source {
        height: 2;
        color: $text-muted;
        margin-top: 1;
        overflow-x: hidden;
        overflow-y: hidden;
    }

    #gallery-actions {
        height: auto;
        margin-top: 1;
        align: center middle;
    }
    """

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("left", "prev_image", "Prev"),
        Binding("right", "next_image", "Next"),
    ]

    def __init__(self, images: list[SessionImageItem], index: int) -> None:
        super().__init__()
        self.images = images
        self.index = max(0, min(index, len(images) - 1)) if images else 0

    def compose(self) -> ComposeResult:
        with Vertical(id="gallery-dialog"):
            with Horizontal(id="gallery-header"):
                yield Static("", id="gallery-counter")
            yield TextualImageWidget(id="gallery-image")
            yield Static("", id="gallery-source")
            with Horizontal(id="gallery-actions"):
                yield Button("Prev", id="gallery-prev-btn")
                yield Button("Next", id="gallery-next-btn")
                yield Button("Save", id="gallery-save-btn", variant="primary")
                yield Button("Close", id="gallery-close-btn")

    def on_mount(self) -> None:
        self._refresh_gallery()

    def action_close(self) -> None:
        self.dismiss()

    def action_prev_image(self) -> None:
        if self.index <= 0:
            return
        self.index -= 1
        self._refresh_gallery()

    def action_next_image(self) -> None:
        if self.index >= len(self.images) - 1:
            return
        self.index += 1
        self._refresh_gallery()

    @on(Button.Pressed, "#gallery-prev-btn")
    def on_prev_pressed(self, _: Button.Pressed) -> None:
        self.action_prev_image()

    @on(Button.Pressed, "#gallery-next-btn")
    def on_next_pressed(self, _: Button.Pressed) -> None:
        self.action_next_image()

    @on(Button.Pressed, "#gallery-save-btn")
    def on_save_pressed(self, _: Button.Pressed) -> None:
        if not self.images:
            return
        app = self.app
        if not hasattr(app, "save_image_item"):
            return
        current = self.images[self.index]
        try:
            destination = app.save_image_item(current)  # type: ignore[attr-defined]
        except RuntimeError as exc:
            self.query_one("#gallery-source", Static).update(self._format_source_text(str(exc)))
            return
        self.query_one("#gallery-source", Static).update(
            self._format_source_text(f"{current.source}\nSaved: {destination}")
        )

    @on(Button.Pressed, "#gallery-close-btn")
    def on_close_pressed(self, _: Button.Pressed) -> None:
        self.dismiss()

    def _refresh_gallery(self) -> None:
        if not self.images:
            self.dismiss()
            return

        current = self.images[self.index]
        image_widget = self.query_one("#gallery-image", TextualImageWidget)
        image_widget.image = current.path
        self.query_one("#gallery-counter", Static).update(f"Image {self.index + 1} / {len(self.images)}")
        self.query_one("#gallery-source", Static).update(self._format_source_text(current.source))
        self.query_one("#gallery-prev-btn", Button).disabled = self.index <= 0
        self.query_one("#gallery-next-btn", Button).disabled = self.index >= len(self.images) - 1

    @staticmethod
    def _format_source_text(value: str) -> str:
        wrapped_lines: list[str] = []
        for line in value.splitlines():
            parts = textwrap.wrap(line, width=72, break_long_words=True, break_on_hyphens=False)
            wrapped_lines.extend(parts or [""])
        if len(wrapped_lines) > 2:
            wrapped_lines = wrapped_lines[:2]
            if len(wrapped_lines[1]) >= 3:
                wrapped_lines[1] = f"{wrapped_lines[1][:-3]}..."
            else:
                wrapped_lines[1] = "..."
        return "\n".join(wrapped_lines)


class ChatApp(App[None]):
    TITLE = "xAI Textual Chatbot"
    CSS = """
    Screen {
        layout: vertical;
    }

    #chat-column {
        height: 1fr;
        margin: 0 1;
    }

    #chat-toolbar {
        height: auto;
        margin: 0 0 1 0;
    }

    #chat-log {
        height: 2fr;
        border: round $accent;
        padding: 1;
    }

    #image-panel {
        height: 1fr;
        border: round $secondary;
        margin-top: 1;
        padding: 1;
    }

    #image-panel.hidden {
        display: none;
    }

    #image-panel-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #image-grid-scroll {
        height: 1fr;
    }

    #image-grid {
        layout: grid;
        grid-size: 1;
        grid-gutter: 1 1;
        height: auto;
    }

    .image-tile {
        border: round $secondary;
        padding: 0 1;
        height: auto;
    }

    .image-thumb {
        width: auto;
        height: 6;
    }

    .image-caption {
        height: auto;
        margin-top: 1;
        color: $text-muted;
    }

    .image-open-btn {
        width: 100%;
        margin-top: 1;
    }

    #image-grid-empty {
        color: $text-muted;
    }

    #status {
        margin-top: 1;
        color: $text-muted;
    }

    #prompt-row {
        height: auto;
        margin: 0 1 1 1;
    }

    #prompt {
        width: 1fr;
        height: 5;
    }

    #send-btn {
        width: 12;
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("enter", "send_prompt", "Send", show=False, priority=True),
        Binding("ctrl+s", "send_prompt", "Send"),
        Binding("ctrl+p", "prompt_history_prev", "Prev Prompt"),
        Binding("ctrl+n", "prompt_history_next", "Next Prompt"),
        Binding("alt+up", "prompt_history_prev", "Prev Prompt"),
        Binding("alt+down", "prompt_history_next", "Next Prompt"),
    ]

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.client = XAIResponsesClient(api_key=config.api_key, model=config.model)
        self.conversation = ConversationState(config.system_prompt)
        default_settings = UISettings(
            chat_model=config.model,
            system_prompt=config.system_prompt,
            image_model=config.image_model,
        )
        self.settings = self._load_ui_settings(default_settings)
        self.pending_prompt: Optional[str] = None
        self.pending_input: Optional[list[ChatMessage]] = None
        self.pending_options: Optional[RequestOptions] = None
        self.last_image_url: Optional[str] = None
        self._session_images: list[SessionImageItem] = []
        self._temp_image_paths: list[Path] = []
        self._gallery_open = False
        self._transcript_lines: list[str] = []
        self._prompt_history: list[str] = []
        self._prompt_history_index: Optional[int] = None
        self._prompt_history_draft: str = ""

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="chat-column"):
            with Horizontal(id="chat-toolbar"):
                yield Button("Settings", id="open-settings-btn")
                yield Button("Clear Chat", id="clear-chat-btn")
                yield Button("Save Chat", id="save-chat-btn")
            yield RichLog(id="chat-log", auto_scroll=True, wrap=True, highlight=False, markup=False)
            with Vertical(id="image-panel", classes="hidden"):
                yield Static("Session Images", id="image-panel-title")
                with VerticalScroll(id="image-grid-scroll"):
                    with Vertical(id="image-grid"):
                        yield Static("No images generated yet.", id="image-grid-empty")
            yield Static("Ready. Click Settings to configure tools and model.", id="status")
        with Horizontal(id="prompt-row"):
            yield TextArea(
                text="",
                soft_wrap=True,
                show_line_numbers=False,
                placeholder=(
                    "Ask Grok something... "
                    "(Enter send, Shift+Enter newline, Ctrl+S send, Alt+Up/Down history)"
                ),
                id="prompt",
            )
            yield Button("Send", id="send-btn", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#prompt", TextArea).focus()
        self._update_image_grid_columns()

    def on_unmount(self) -> None:
        self._cleanup_temp_images()
        self.client.close()

    def on_resize(self, event: events.Resize) -> None:
        self._update_image_grid_columns(event.size.width)

    @on(Button.Pressed, "#open-settings-btn")
    def on_open_settings_clicked(self, _: Button.Pressed) -> None:
        self.push_screen(SettingsScreen(self.settings, self.client), self._on_settings_closed)

    def _on_settings_closed(self, settings: Optional[UISettings]) -> None:
        if settings is None:
            self._set_status("Settings unchanged.")
        else:
            self.settings = settings
            self._save_ui_settings(settings)
            self._set_status("Settings saved.")
        self.query_one("#prompt", TextArea).focus()

    @on(Button.Pressed, "#clear-chat-btn")
    def on_clear_chat_clicked(self, _: Button.Pressed) -> None:
        self.conversation.clear_history()
        self.query_one(RichLog).clear()
        self._transcript_lines.clear()
        self.last_image_url = None
        self._session_images.clear()
        self._cleanup_temp_images()
        self._refresh_image_grid()
        self._set_status("Chat log and conversation history cleared.")
        self.query_one("#prompt", TextArea).focus()

    @on(Button.Pressed, "#save-chat-btn")
    def on_save_chat_clicked(self, _: Button.Pressed) -> None:
        if not self._transcript_lines:
            self._set_status("No chat content to save yet.")
            return

        exports_dir = self._ensure_exports_dir()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        destination = exports_dir / f"chat-{timestamp}.md"
        destination.write_text("\n\n".join(self._transcript_lines).strip() + "\n", encoding="utf-8")
        self._set_status(f"Saved chat: {destination}")

    @on(Button.Pressed, "#send-btn")
    def on_send_clicked(self, _: Button.Pressed) -> None:
        self._submit_prompt_from_ui()

    def on_click(self, event: events.Click) -> None:
        if self._gallery_open:
            return
        index = self._resolve_image_index_from_click(event)
        if index is None:
            return
        self._open_image_gallery(index)

    @on(Button.Pressed, ".image-open-btn")
    def on_image_open_button_pressed(self, event: Button.Pressed) -> None:
        if self._gallery_open:
            return
        index = getattr(event.button, "_image_index", None)
        if not isinstance(index, int):
            return
        if index is None:
            return
        self._open_image_gallery(index)

    def action_send_prompt(self) -> None:
        prompt_widget = self.query_one("#prompt", TextArea)
        if self.focused is not prompt_widget:
            return
        self._submit_prompt_from_ui()

    def action_prompt_history_prev(self) -> None:
        if not self._prompt_history:
            return

        prompt_widget = self.query_one("#prompt", TextArea)
        if self._prompt_history_index is None:
            self._prompt_history_draft = prompt_widget.text
            self._prompt_history_index = len(self._prompt_history) - 1
        elif self._prompt_history_index > 0:
            self._prompt_history_index -= 1

        self._set_prompt_text(self._prompt_history[self._prompt_history_index])

    def action_prompt_history_next(self) -> None:
        if self._prompt_history_index is None:
            return

        if self._prompt_history_index < len(self._prompt_history) - 1:
            self._prompt_history_index += 1
            self._set_prompt_text(self._prompt_history[self._prompt_history_index])
            return

        self._prompt_history_index = None
        self._set_prompt_text(self._prompt_history_draft)

    def _submit_prompt_from_ui(self) -> None:
        prompt_widget = self.query_one("#prompt", TextArea)
        send_button = self.query_one("#send-btn", Button)

        prompt = prompt_widget.text.strip()
        if not prompt:
            return
        if self.pending_prompt is not None:
            self._set_status("A request is already in progress.")
            return

        self.client.model = self.settings.chat_model
        self.conversation.system_prompt = self.settings.system_prompt

        try:
            options = build_request_options(
                web_search=self.settings.web_search,
                x_search=self.settings.x_search,
                x_search_image_understanding=self.settings.x_search_image_understanding,
                x_search_video_understanding=self.settings.x_search_video_understanding,
                code_interpreter=self.settings.code_interpreter,
                file_search=self.settings.file_search,
                vector_store_ids_raw=self.settings.vector_store_ids_raw,
                file_search_max_results_raw=self.settings.file_search_max_results_raw,
                image_generation=self.settings.image_generation,
                image_model=self.settings.image_model,
                image_count_raw=self.settings.image_count_raw,
                image_as_base64=self.settings.image_as_base64,
                image_source_url_raw=self.settings.image_source_url_raw,
                image_use_last=self.settings.image_use_last,
                image_aspect_ratio_raw=self.settings.image_aspect_ratio_raw,
                mcp_enabled=self.settings.mcp_enabled,
                mcp_servers=self.settings.mcp_servers,
            )
        except ValueError as exc:
            self._log_error(str(exc))
            self._set_status("Invalid settings. Open Settings and fix values.")
            return

        prompt_widget.clear()
        self._prompt_history_index = None
        self._prompt_history_draft = ""
        if not self._prompt_history or self._prompt_history[-1] != prompt:
            self._prompt_history.append(prompt)

        self.pending_prompt = prompt
        self.pending_options = options
        self.pending_input = self.conversation.build_request(
            prompt, include_history=self.settings.include_history
        )
        self._log_user(prompt)
        self._set_status("Waiting for xAI response...")
        prompt_widget.disabled = True
        send_button.disabled = True
        self.run_worker(self._ask_in_background, thread=True, exclusive=True, name="ask")

    def _ask_in_background(self) -> ChatResult:
        assert self.pending_prompt is not None
        assert self.pending_options is not None
        assert self.pending_input is not None
        return self.client.ask(
            messages=self.pending_input,
            options=self.pending_options,
            last_image_url=self.last_image_url,
        )

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.state != WorkerState.SUCCESS and event.state != WorkerState.ERROR:
            return

        if event.worker.name == "load-session-images":
            if event.state == WorkerState.SUCCESS:
                payload = event.worker.result
                if (
                    isinstance(payload, tuple)
                    and len(payload) == 2
                    and isinstance(payload[0], list)
                    and isinstance(payload[1], list)
                ):
                    images = payload[0]
                    errors = payload[1]
                    for item in images:
                        if isinstance(item, SessionImageItem):
                            self._session_images.append(item)
                    self._refresh_image_grid()
                    if errors:
                        for error in errors:
                            self._log_error(error)
                        self._set_status("Ready (some images failed to render).")
            else:
                self._log_error(f"Image load failed: {event.worker.error}")
            return

        if event.worker.name != "ask":
            return

        prompt_box = self.query_one("#prompt", TextArea)
        send_button = self.query_one("#send-btn", Button)
        prompt_box.disabled = False
        send_button.disabled = False
        prompt_box.focus()

        if event.state == WorkerState.SUCCESS:
            result = event.worker.result
            assert isinstance(result, ChatResult)
            assert self.pending_prompt is not None
            if self.settings.include_history:
                self.conversation.commit_turn(self.pending_prompt, result.text)
            if result.image_urls:
                self.last_image_url = result.image_urls[0]
            if result.image_urls or result.image_b64:
                self.run_worker(
                    lambda: self._materialize_session_images(result),
                    thread=True,
                    exclusive=False,
                    name="load-session-images",
                )
            self._log_assistant(result.text)
            self._set_status("Ready.")
        else:
            self._log_error(str(event.worker.error))
            self._set_status("Request failed.")

        self.pending_prompt = None
        self.pending_input = None
        self.pending_options = None

    def _set_prompt_text(self, text: str) -> None:
        prompt_widget = self.query_one("#prompt", TextArea)
        prompt_widget.load_text(text)
        lines = text.splitlines() or [""]
        prompt_widget.move_cursor((len(lines) - 1, len(lines[-1])))

    def _materialize_session_images(
        self,
        result: ChatResult,
    ) -> tuple[list[SessionImageItem], list[str]]:
        images: list[SessionImageItem] = []
        errors: list[str] = []

        urls = result.image_urls or []
        for image_url in urls:
            try:
                image_bytes = self.client.fetch_bytes(image_url)
                suffix = self._guess_image_suffix_from_url(image_url)
                image_path = self._write_temp_image_bytes(image_bytes, suffix=suffix)
                images.append(SessionImageItem(path=image_path, source=image_url))
            except Exception as exc:
                errors.append(f"Failed to load image URL {image_url}: {exc}")

        base64_images = result.image_b64 or []
        for index, image_b64 in enumerate(base64_images, start=1):
            try:
                image_bytes = base64.b64decode(image_b64, validate=True)
                image_path = self._write_temp_image_bytes(image_bytes, suffix=".png")
                images.append(
                    SessionImageItem(
                        path=image_path,
                        source=f"(embedded base64 image {index})",
                    )
                )
            except Exception as exc:
                errors.append(f"Failed to decode base64 image {index}: {exc}")

        return images, errors

    def _write_temp_image_bytes(self, image_bytes: bytes, *, suffix: str) -> Path:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(image_bytes)
            path = Path(temp.name)
        self._temp_image_paths.append(path)
        return path

    def _refresh_image_grid(self) -> None:
        panel = self.query_one("#image-panel", Vertical)
        grid = self.query_one("#image-grid", Vertical)
        self._update_image_grid_columns()

        grid.remove_children()

        if not self._session_images:
            panel.add_class("hidden")
            grid.mount(Static("No images generated yet.", id="image-grid-empty"))
            return

        panel.remove_class("hidden")
        for index, item in enumerate(self._session_images):
            thumb = TextualImageWidget(
                classes="image-thumb",
            )
            thumb.image = item.path
            setattr(thumb, "_image_index", index)
            caption_source = item.source
            if len(caption_source) > 42:
                caption_source = f"{caption_source[:39]}..."
            caption = Static(
                f"{index + 1}. Click image to open\n{caption_source}",
                classes="image-caption",
            )
            setattr(caption, "_image_index", index)
            open_button = Button(
                "Open",
                classes="image-open-btn",
            )
            setattr(open_button, "_image_index", index)
            tile = Vertical(
                thumb,
                caption,
                open_button,
                classes="image-tile",
            )
            setattr(tile, "_image_index", index)
            grid.mount(tile)

    def _update_image_grid_columns(self, width: Optional[int] = None) -> None:
        try:
            grid = self.query_one("#image-grid", Vertical)
        except Exception:
            return

        if width is None:
            width = self.size.width

        if width < 90:
            columns = 1
        elif width < 140:
            columns = 2
        elif width < 190:
            columns = 3
        else:
            columns = 4

        grid.styles.grid_size_columns = columns

    def _resolve_image_index_from_click(self, event: events.Click) -> Optional[int]:
        candidates: list[Widget] = []
        if isinstance(event.widget, Widget):
            candidates.append(event.widget)

        control = event.control
        if isinstance(control, Widget) and control not in candidates:
            candidates.append(control)

        for node in candidates:
            current: Optional[Widget] = node
            while current is not None:
                image_index = getattr(current, "_image_index", None)
                if isinstance(image_index, int):
                    return image_index
                parent = current.parent
                current = parent if isinstance(parent, Widget) else None
        return None

    def _open_image_gallery(self, index: int) -> None:
        if self._gallery_open:
            return
        if index < 0 or index >= len(self._session_images):
            return
        self._gallery_open = True
        self.push_screen(
            ImageGalleryScreen(images=self._session_images.copy(), index=index),
            self._on_image_gallery_closed,
        )

    def _on_image_gallery_closed(self, _: None) -> None:
        self._gallery_open = False
        self._refresh_image_grid()
        self.query_one("#prompt", TextArea).focus()

    @staticmethod
    def _guess_image_suffix_from_url(image_url: str) -> str:
        match = re.search(r"\.(png|jpg|jpeg|webp|gif)(?:\?|$)", image_url, flags=re.IGNORECASE)
        if not match:
            return ".jpg"
        ext = match.group(1).lower()
        if ext == "jpeg":
            ext = "jpg"
        return f".{ext}"

    def _cleanup_temp_images(self) -> None:
        for image_path in self._temp_image_paths:
            try:
                image_path.unlink(missing_ok=True)
            except OSError:
                pass
        self._temp_image_paths.clear()

    def _log_user(self, prompt: str) -> None:
        self.query_one(RichLog).write(Text(f"You: {prompt}"))
        self._transcript_lines.append(f"You: {prompt}")

    def _log_assistant(self, message: str) -> None:
        self.query_one(RichLog).write(
            Group(
                Text("Grok:", style="bold green"),
                Markdown(message),
            )
        )
        self._transcript_lines.append(f"Grok:\n{message}")

    def _log_error(self, message: str) -> None:
        self.query_one(RichLog).write(Text(f"Error: {message}", style="bold red"))
        self._transcript_lines.append(f"Error: {message}")

    def _set_status(self, message: str) -> None:
        self.query_one("#status", Static).update(message)

    @staticmethod
    def _settings_file_path() -> Path:
        return Path.cwd() / ".textualbot_settings.json"

    def _load_ui_settings(self, default: UISettings) -> UISettings:
        path = self._settings_file_path()
        if not path.exists():
            return default

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default
        if not isinstance(data, dict):
            return default

        settings_data = asdict(default)
        for key in settings_data.keys():
            if key not in data:
                continue
            value = data[key]
            if key == "mcp_servers":
                if isinstance(value, list):
                    normalized_servers: list[dict[str, Any]] = []
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

    def _save_ui_settings(self, settings: UISettings) -> None:
        path = self._settings_file_path()
        try:
            path.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
        except OSError:
            self._set_status("Warning: failed to persist settings to disk.")

    @staticmethod
    def _ensure_exports_dir() -> Path:
        exports_dir = Path.cwd() / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        return exports_dir

    def save_image_item(self, image: SessionImageItem) -> Path:
        exports_dir = self._ensure_exports_dir() / "images"
        exports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        if not image.path.exists():
            raise RuntimeError("Image file is no longer available.")

        suffix = image.path.suffix or ".png"
        destination = exports_dir / f"image-{timestamp}{suffix}"
        destination.write_bytes(image.path.read_bytes())
        self._set_status(f"Saved image: {destination}")
        return destination
