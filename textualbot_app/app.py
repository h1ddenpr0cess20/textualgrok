from dataclasses import asdict, dataclass, field
import base64
from datetime import datetime
import json
import mimetypes
import textwrap
import time
from typing import Any, Iterable, Optional
import tempfile
from pathlib import Path
import re

from rich.console import Group
from rich.markdown import Markdown
from rich.text import Text
from textual.command import DiscoveryHit, Hit, Hits, Provider
from textual import events, on
from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.containers import Horizontal, HorizontalScroll, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen, Screen
from textual.widget import Widget
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    OptionList,
    RichLog,
    Select,
    Static,
    Switch,
    TabPane,
    TabbedContent,
    TextArea,
)
from textual.widgets.select import InvalidSelectValueError
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
        "auto",
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
        aspect_ratio_options = [("auto (model default)", "auto")] + [
            (ratio, ratio) for ratio in self.VALID_IMAGE_ASPECT_RATIOS if ratio != "auto"
        ]
        with Vertical(id="settings-dialog"):
            yield Static("Settings", id="settings-title")
            with TabbedContent(id="settings-tabs"):
                with TabPane("Chat"):
                    with VerticalScroll(classes="settings-tab-scroll"):
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
                    with VerticalScroll(classes="settings-tab-scroll"):
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
                    with VerticalScroll(classes="settings-tab-scroll"):
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
                    with VerticalScroll(classes="settings-tab-scroll"):
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
                    with VerticalScroll(id="mcp-scroll", classes="settings-tab-scroll"):
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


@dataclass
class PendingAttachment:
    label: str
    content_part: dict[str, str]
    preview_path: Path | None = None


@dataclass
class BrowseEntry:
    path: Path
    kind: str
    is_image: bool


class OrderedSystemCommandsProvider(Provider):
    async def discover(self) -> Hits:
        for title, help_text, callback, discover in self.app.get_system_commands(self.screen):
            if discover:
                yield DiscoveryHit(title, callback, help=help_text)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for title, help_text, callback, *_ in self.app.get_system_commands(self.screen):
            if (match := matcher.match(title)) > 0:
                yield Hit(match, matcher.highlight(title), callback, help=help_text)


class BrowseOptionList(OptionList):
    DOUBLE_CLICK_SECONDS = 0.45

    def __init__(self, *content: object, **kwargs: object) -> None:
        super().__init__(*content, **kwargs)
        self._last_click_option: int | None = None
        self._last_click_time: float = 0.0

    class OpenRequested(Message):
        def __init__(self, option_list: "BrowseOptionList", option_index: int) -> None:
            super().__init__()
            self.option_list = option_list
            self.option_index = option_index

        @property
        def control(self) -> "BrowseOptionList":
            return self.option_list

    def reset_click_chain(self) -> None:
        self._last_click_option = None
        self._last_click_time = 0.0

    def _resolve_clicked_option(self, event: events.Click) -> int | None:
        clicked_option = event.style.meta.get("option")
        if isinstance(clicked_option, int):
            return clicked_option
        try:
            line_number = self.scroll_offset.y + int(event.y)
            if 0 <= line_number < len(self._lines):
                return self._lines[line_number][0]
        except (TypeError, ValueError):
            return None
        return None

    async def _on_click(self, event: events.Click) -> None:
        # Single click selects. Double click opens (via OptionSelected handler).
        clicked_option = self._resolve_clicked_option(event)
        if clicked_option is None:
            return

        self.highlighted = clicked_option
        now = time.monotonic()
        is_timed_double = self._last_click_option == clicked_option and (
            now - self._last_click_time <= self.DOUBLE_CLICK_SECONDS
        )
        if is_timed_double:
            self.post_message(self.OpenRequested(self, clicked_option))
            self.reset_click_chain()
        else:
            self._last_click_option = clicked_option
            self._last_click_time = now
        event.stop()


class BrowseAttachmentFileScreen(ModalScreen[Optional[str]]):
    def __init__(self, start_path: Path | None = None) -> None:
        super().__init__()
        self.start_path = (start_path or Path.home()).resolve()
        self._current_dir = self.start_path
        self._entries: list[BrowseEntry] = []
        self._selected_path: Optional[Path] = None

    def compose(self) -> ComposeResult:
        with Vertical(id="browse-dialog"):
            yield Static("Browse File or Folder")
            with Horizontal(id="browse-nav"):
                yield Button("Home", id="btn-browse-home")
                yield Button("CWD", id="btn-browse-cwd")
                yield Select(
                    [],
                    allow_blank=True,
                    id="browse-drive-select",
                )
            with Horizontal(id="browse-main"):
                yield BrowseOptionList(id="browse-list")
                with Vertical(id="browse-preview-panel"):
                    yield Static("Preview", id="browse-preview-title")
                    yield TextualImageWidget(id="browse-preview-image")
                    yield Static("Select a file or folder.", id="browse-preview-path")
            yield Static("No file or folder selected.", id="browse-selected")
            with Horizontal(id="browse-actions"):
                yield Button("Cancel", id="btn-browse-cancel")
                yield Button("Use Selected", id="btn-browse-use", variant="primary", disabled=True)

    def on_mount(self) -> None:
        self._populate_drive_select()
        self._set_current_directory(self.start_path)
        self.query_one("#browse-list", OptionList).focus()

    def action_open_highlighted(self) -> None:
        option_list = self.query_one("#browse-list", OptionList)
        highlighted = option_list.highlighted
        if highlighted is None:
            return
        self._activate_entry(highlighted)

    @on(Button.Pressed, "#btn-browse-cancel")
    def on_cancel_clicked(self, _: Button.Pressed) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#btn-browse-home")
    def on_home_clicked(self, _: Button.Pressed) -> None:
        self._set_current_directory(Path.home())

    @on(Button.Pressed, "#btn-browse-cwd")
    def on_cwd_clicked(self, _: Button.Pressed) -> None:
        self._set_current_directory(Path.cwd())

    @on(Select.Changed, "#browse-drive-select")
    def on_drive_changed(self, event: Select.Changed) -> None:
        value = event.value
        if not isinstance(value, str) or not value:
            return
        drive_path = Path(value)
        if drive_path.exists():
            self._set_current_directory(drive_path)

    @on(OptionList.OptionHighlighted, "#browse-list")
    def on_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        self._apply_highlight(event.option_index)

    @on(OptionList.OptionSelected, "#browse-list")
    def on_option_selected(self, event: OptionList.OptionSelected) -> None:
        self._apply_highlight(event.option_index)

    @on(BrowseOptionList.OpenRequested, "#browse-list")
    def on_open_requested(self, event: BrowseOptionList.OpenRequested) -> None:
        self._activate_entry(event.option_index)

    @on(Button.Pressed, "#btn-browse-use")
    def on_use_clicked(self, _: Button.Pressed) -> None:
        if self._selected_path is None:
            option_list = self.query_one("#browse-list", OptionList)
            highlighted = option_list.highlighted
            if highlighted is not None:
                entry = self._entry_at_index(highlighted)
                if entry is not None and entry.kind in {"dir", "file"}:
                    self._selected_path = entry.path
        if self._selected_path is None:
            return
        self.dismiss(str(self._selected_path))

    def _set_current_directory(self, path: Path) -> None:
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            return
        if resolved.is_file():
            resolved = resolved.parent
        if not resolved.is_dir():
            return
        self._current_dir = resolved
        self._selected_path = None
        self.query_one("#browse-list", BrowseOptionList).reset_click_chain()
        self._sync_drive_select_with_path(resolved)
        self._rebuild_browse_list()

    def _rebuild_browse_list(self) -> None:
        option_list = self.query_one("#browse-list", OptionList)
        preview_image = self.query_one("#browse-preview-image", TextualImageWidget)
        preview_path = self.query_one("#browse-preview-path", Static)
        use_button = self.query_one("#btn-browse-use", Button)
        selected_status = self.query_one("#browse-selected", Static)

        self._entries.clear()
        options: list[str] = []

        parent = self._current_dir.parent
        if parent != self._current_dir:
            self._entries.append(BrowseEntry(path=parent, kind="parent", is_image=False))
            options.append("..")

        try:
            children = list(self._current_dir.iterdir())
        except OSError as exc:
            option_list.set_options([])
            selected_status.update(f"Cannot open directory: {exc}")
            preview_image.image = None
            preview_path.update("Select a file or folder.")
            use_button.disabled = True
            return

        directories = sorted(
            [item for item in children if item.is_dir()],
            key=lambda p: p.name.lower(),
        )
        files = sorted(
            [item for item in children if item.is_file()],
            key=lambda p: p.name.lower(),
        )
        supported_files: list[Path] = []
        unsupported_files_count = 0
        for file_path in files:
            if ChatApp.is_supported_attachment_file(file_path):
                supported_files.append(file_path)
            else:
                unsupported_files_count += 1

        for directory in directories:
            self._entries.append(BrowseEntry(path=directory, kind="dir", is_image=False))
            options.append(f"[DIR] {directory.name}")
        for file_path in supported_files:
            is_image = ChatApp.is_likely_image_file(file_path)
            self._entries.append(BrowseEntry(path=file_path, kind="file", is_image=is_image))
            marker = "[IMG]" if is_image else "[FILE]"
            options.append(f"{marker} {file_path.name}")

        option_list.set_options(options)
        selected_message = f"Browsing: {self._current_dir}"
        if unsupported_files_count > 0:
            selected_message = (
                f"{selected_message} ({unsupported_files_count} unsupported file"
                f"{'s' if unsupported_files_count != 1 else ''} hidden)"
            )
        selected_status.update(selected_message)
        preview_image.image = None
        preview_path.update("Select a file or folder.")
        use_button.disabled = True

        if self._entries:
            option_list.highlighted = 0
            self._apply_highlight(0)

    def _entry_at_index(self, index: int) -> Optional[BrowseEntry]:
        if index < 0 or index >= len(self._entries):
            return None
        return self._entries[index]

    def _activate_entry(self, index: int) -> None:
        entry = self._entry_at_index(index)
        if entry is None:
            return
        if entry.kind in {"parent", "dir"}:
            self._set_current_directory(entry.path)
            return
        self._apply_highlight(index)

    def _apply_highlight(self, index: int) -> None:
        entry = self._entry_at_index(index)
        if entry is None:
            return

        preview_image = self.query_one("#browse-preview-image", TextualImageWidget)
        preview_path = self.query_one("#browse-preview-path", Static)
        use_button = self.query_one("#btn-browse-use", Button)
        selected_status = self.query_one("#browse-selected", Static)

        if entry.kind in {"parent", "dir"}:
            self._selected_path = entry.path if entry.kind == "dir" else None
            preview_image.image = None
            if entry.kind == "parent":
                selected_status.update(f"Up to: {entry.path}")
                preview_path.update("Click .. to go up.")
                use_button.disabled = True
            else:
                selected_status.update(f"Directory: {entry.path}")
                preview_path.update("Use Selected to attach this folder, or press Enter to open it.")
                use_button.disabled = False
            return

        if entry.is_image:
            self._selected_path = entry.path
            selected_status.update(str(entry.path))
            preview_image.image = entry.path
            preview_path.update(str(entry.path))
            use_button.disabled = False
            return

        self._selected_path = entry.path
        selected_status.update(str(entry.path))
        preview_image.image = None
        preview_path.update(str(entry.path))
        use_button.disabled = False

    def _populate_drive_select(self) -> None:
        select = self.query_one("#browse-drive-select", Select)
        drives = self._available_drive_roots()
        if not drives:
            select.set_options([("/", "/")])
            select.value = "/"
            return
        options = [(drive, drive) for drive in drives]
        select.set_options(options)
        self._sync_drive_select_with_path(self._current_dir)

    def _sync_drive_select_with_path(self, path: Path) -> None:
        select = self.query_one("#browse-drive-select", Select)
        value = self._matching_drive_value(path)
        if value is not None:
            try:
                select.value = value
            except InvalidSelectValueError:
                # Can happen transiently before options are fully synced.
                return

    def _matching_drive_value(self, path: Path) -> Optional[str]:
        path_str = str(path).lower()
        for drive in self._available_drive_roots():
            if path_str.startswith(drive.lower()):
                return drive
        return None

    @staticmethod
    def _available_drive_roots() -> list[str]:
        roots: list[str] = []
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            candidate = Path(f"{letter}:\\")
            if candidate.exists():
                roots.append(str(candidate))
        if roots:
            return roots
        root = Path("/").resolve()
        return [str(root)] if root.exists() else []


class AddAttachmentScreen(ModalScreen[Optional[str]]):
    def compose(self) -> ComposeResult:
        with Vertical(id="attach-dialog"):
            yield Static("Attach File or Folder", id="attach-title")
            yield Label("File path, folder path, or image URL")
            with Horizontal(id="attach-source-row"):
                yield Input(
                    value="",
                    placeholder=r"C:\path\to\file-or-folder or https://example.com/image.jpg",
                    id="input-attach-source",
                    classes="settings-input",
                )
                yield Button("Browse...", id="btn-attach-browse")
            with Horizontal(id="attach-actions"):
                yield Button("Cancel", id="btn-attach-cancel")
                yield Button("Attach", id="btn-attach-confirm", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#input-attach-source", Input).focus()

    @on(Button.Pressed, "#btn-attach-cancel")
    def on_cancel_clicked(self, _: Button.Pressed) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#btn-attach-browse")
    def on_browse_clicked(self, _: Button.Pressed) -> None:
        current_value = self.query_one("#input-attach-source", Input).value.strip()
        start_path = Path.home()
        if current_value and not (current_value.startswith("http://") or current_value.startswith("https://")):
            candidate = Path(current_value).expanduser()
            if candidate.is_file():
                start_path = candidate.parent.resolve()
            elif candidate.is_dir():
                start_path = candidate.resolve()
            elif candidate.parent.exists():
                start_path = candidate.parent.resolve()
        self.app.push_screen(BrowseAttachmentFileScreen(start_path=start_path), self._on_browse_closed)

    def _on_browse_closed(self, selected_path: Optional[str]) -> None:
        if not selected_path:
            self.query_one("#input-attach-source", Input).focus()
            return
        self.query_one("#input-attach-source", Input).value = selected_path
        self.query_one("#input-attach-source", Input).focus()

    @on(Button.Pressed, "#btn-attach-confirm")
    def on_confirm_clicked(self, _: Button.Pressed) -> None:
        source = self.query_one("#input-attach-source", Input).value.strip()
        if not source:
            return
        self.dismiss(source)


class ImageGalleryScreen(Screen[None]):
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
    CSS_PATH = "app.tcss"
    COMMANDS = {OrderedSystemCommandsProvider}
    # Based on Context7 xAI docs: files support many text-based formats and common office docs.
    SUPPORTED_ATTACHMENT_EXTENSIONS = {
        ".txt",
        ".md",
        ".markdown",
        ".rst",
        ".rtf",
        ".csv",
        ".tsv",
        ".json",
        ".jsonl",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
        ".log",
        ".xml",
        ".html",
        ".htm",
        ".pdf",
        ".doc",
        ".docx",
        ".ppt",
        ".pptx",
        ".xls",
        ".xlsx",
        ".py",
        ".pyi",
        ".js",
        ".jsx",
        ".mjs",
        ".cjs",
        ".ts",
        ".tsx",
        ".java",
        ".c",
        ".h",
        ".cpp",
        ".hpp",
        ".cc",
        ".cs",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".kts",
        ".scala",
        ".sql",
        ".sh",
        ".bash",
        ".zsh",
        ".ps1",
        ".bat",
        ".cmd",
    }
    SUPPORTED_ATTACHMENT_FILENAMES = {
        "dockerfile",
        "makefile",
        "readme",
        "license",
    }
    SUPPORTED_IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
    }
    SUPPORTED_IMAGE_MIME_TYPES = {
        "image/jpeg",
        "image/png",
    }
    SUPPORTED_ATTACHMENT_MIME_TYPES = {
        "application/pdf",
        "application/json",
        "application/xml",
        "application/yaml",
        "application/x-yaml",
        "application/rtf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "text/csv",
        "text/markdown",
    }

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("f1", "command_palette", "☰ Menu"),
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
        self.pending_user_content: Optional[object] = None
        self.pending_input: Optional[list[ChatMessage]] = None
        self.pending_options: Optional[RequestOptions] = None
        self.last_image_url: Optional[str] = None
        self._pending_attachments: list[PendingAttachment] = []
        self._session_images: list[SessionImageItem] = []
        self._temp_image_paths: list[Path] = []
        self._gallery_open = False
        self._transcript_lines: list[str] = []
        self._prompt_history: list[str] = []
        self._prompt_history_index: Optional[int] = None
        self._prompt_history_draft: str = ""

    def compose(self) -> ComposeResult:
        yield Header(icon="☰")
        with Vertical(id="chat-column"):
            yield RichLog(id="chat-log", auto_scroll=True, wrap=True, highlight=False, markup=False)
            with Vertical(id="image-panel", classes="hidden"):
                yield Static("Session Images", id="image-panel-title")
                with VerticalScroll(id="image-grid-scroll"):
                    with Vertical(id="image-grid"):
                        yield Static("No images generated yet.", classes="image-grid-empty")
            yield Static("Ready. Press F1 for menu.", id="status")
        with Horizontal(id="attachments-bar"):
            yield Static("Attachments: none", id="attachments-status")
            with Horizontal(id="attachments-actions"):
                yield Button("Attach File/Folder", id="attach-file-btn", variant="default")
                yield Button("Clear Attach", id="clear-attachments-btn", variant="default", disabled=True)
        with HorizontalScroll(id="attachments-thumbs", classes="hidden"):
            pass
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
        self._refresh_pending_attachments_ui()

    def on_unmount(self) -> None:
        self._cleanup_temp_images()
        self.client.close()

    def on_resize(self, event: events.Resize) -> None:
        self._update_image_grid_columns(event.size.width)

    def action_open_settings(self) -> None:
        self.push_screen(SettingsScreen(self.settings, self.client), self._on_settings_closed)

    def _on_settings_closed(self, settings: Optional[UISettings]) -> None:
        if settings is None:
            self._set_status("Settings unchanged.")
        else:
            self.settings = settings
            self._save_ui_settings(settings)
            self._set_status("Settings saved.")
        self.query_one("#prompt", TextArea).focus()

    def action_clear_chat(self) -> None:
        self.conversation.clear_history()
        self.query_one(RichLog).clear()
        self._transcript_lines.clear()
        self.last_image_url = None
        self._pending_attachments.clear()
        self._session_images.clear()
        self._cleanup_temp_images()
        self._refresh_image_grid()
        self._refresh_pending_attachments_ui()
        self._set_status("Chat log and conversation history cleared.")
        self.query_one("#prompt", TextArea).focus()

    def action_save_chat(self) -> None:
        if not self._transcript_lines:
            self._set_status("No chat content to save yet.")
            return

        exports_dir = self._ensure_exports_dir()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        destination = exports_dir / f"chat-{timestamp}.md"
        destination.write_text("\n\n".join(self._transcript_lines).strip() + "\n", encoding="utf-8")
        self._set_status(f"Saved chat: {destination}")

    def get_system_commands(self, screen: Screen) -> Iterable[SystemCommand]:
        yield SystemCommand("Settings", "Open settings dialog", self.action_open_settings)
        yield SystemCommand("Clear Chat", "Clear chat log and conversation history", self.action_clear_chat)
        yield SystemCommand("Save Chat", "Save chat transcript to exports", self.action_save_chat)
        yield from super().get_system_commands(screen)

    @on(Button.Pressed, "#attach-file-btn")
    def on_attach_file_clicked(self, _: Button.Pressed) -> None:
        self.push_screen(AddAttachmentScreen(), self._on_attachment_screen_closed)

    @on(Button.Pressed, "#clear-attachments-btn")
    def on_clear_attachments_clicked(self, _: Button.Pressed) -> None:
        self._pending_attachments.clear()
        self._refresh_pending_attachments_ui()
        self._set_status("Cleared pending attachments.")
        self.query_one("#prompt", TextArea).focus()

    @on(Button.Pressed, "#send-btn")
    def on_send_clicked(self, _: Button.Pressed) -> None:
        self._submit_prompt_from_ui()

    def _on_attachment_screen_closed(self, result: Optional[str]) -> None:
        if result is None:
            self.query_one("#prompt", TextArea).focus()
            return

        source = result
        try:
            attachments, skipped_unsupported = self._build_pending_attachments(source)
        except ValueError as exc:
            self._set_status(str(exc))
            self.query_one("#prompt", TextArea).focus()
            return

        self._pending_attachments.extend(attachments)
        self._refresh_pending_attachments_ui()
        is_folder_source = False
        if not (source.startswith("http://") or source.startswith("https://")):
            source_path = Path(source).expanduser()
            if not source_path.is_absolute():
                source_path = (Path.cwd() / source_path).resolve()
            is_folder_source = source_path.exists() and source_path.is_dir()
        if is_folder_source:
            source_path = Path(source).expanduser()
            if not source_path.is_absolute():
                source_path = (Path.cwd() / source_path).resolve()
            message = f"Attached folder: {source_path} ({len(attachments)} files)"
            if skipped_unsupported > 0:
                message = (
                    f"{message}; skipped {skipped_unsupported} unsupported file"
                    f"{'s' if skipped_unsupported != 1 else ''}"
                )
            self._set_status(message)
        else:
            self._set_status(f"Attached: {attachments[0].label}")
        self.query_one("#prompt", TextArea).focus()

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
        if isinstance(self.screen, BrowseAttachmentFileScreen):
            self.screen.action_open_highlighted()
            return
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
        attach_button = self.query_one("#attach-file-btn", Button)
        clear_attachments_button = self.query_one("#clear-attachments-btn", Button)

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

        user_content_parts: list[dict[str, str]] = []
        for attachment in self._pending_attachments:
            user_content_parts.append(dict(attachment.content_part))

        self.pending_prompt = prompt
        self.pending_user_content = (
            [*user_content_parts, {"type": "input_text", "text": prompt}]
            if user_content_parts
            else prompt
        )
        self.pending_options = options
        self.pending_input = self.conversation.build_request(
            prompt,
            include_history=self.settings.include_history,
            user_content_parts=user_content_parts,
        )
        self._log_user(prompt, attachment_count=len(user_content_parts))
        self._pending_attachments.clear()
        self._refresh_pending_attachments_ui()
        self._set_status("Waiting for xAI response...")
        prompt_widget.disabled = True
        send_button.disabled = True
        attach_button.disabled = True
        clear_attachments_button.disabled = True
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
        attach_button = self.query_one("#attach-file-btn", Button)
        clear_attachments_button = self.query_one("#clear-attachments-btn", Button)
        prompt_box.disabled = False
        send_button.disabled = False
        attach_button.disabled = False
        clear_attachments_button.disabled = len(self._pending_attachments) == 0
        prompt_box.focus()

        if event.state == WorkerState.SUCCESS:
            result = event.worker.result
            assert isinstance(result, ChatResult)
            assert self.pending_prompt is not None
            assert self.pending_user_content is not None
            if self.settings.include_history:
                self.conversation.commit_turn(self.pending_user_content, result.text)
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
        self.pending_user_content = None
        self.pending_input = None
        self.pending_options = None

    def _set_prompt_text(self, text: str) -> None:
        prompt_widget = self.query_one("#prompt", TextArea)
        prompt_widget.load_text(text)
        lines = text.splitlines() or [""]
        prompt_widget.move_cursor((len(lines) - 1, len(lines[-1])))

    def _refresh_pending_attachments_ui(self) -> None:
        summary_widget = self.query_one("#attachments-status", Static)
        clear_button = self.query_one("#clear-attachments-btn", Button)
        thumbs = self.query_one("#attachments-thumbs", HorizontalScroll)
        thumbs.remove_children()

        if not self._pending_attachments:
            summary_widget.update("Attachments: none")
            thumbs.add_class("hidden")
            if self.pending_prompt is None:
                clear_button.disabled = True
            return

        thumbs.remove_class("hidden")
        labels: list[str] = []
        for attachment in self._pending_attachments:
            label = attachment.label
            if label.startswith("http://") or label.startswith("https://"):
                labels.append(label if len(label) <= 50 else f"{label[:47]}...")
                chip_label = label if len(label) <= 36 else f"{label[:33]}..."
                thumbs.mount(Static(f"URL: {chip_label}", classes="attachment-url-chip"))
            else:
                labels.append(Path(label).name)
                if attachment.preview_path and attachment.preview_path.exists():
                    thumb = TextualImageWidget(classes="attachment-thumb")
                    thumb.image = attachment.preview_path
                    thumbs.mount(thumb)
                else:
                    file_name = Path(label).name
                    chip_label = file_name if len(file_name) <= 36 else f"{file_name[:33]}..."
                    thumbs.mount(Static(chip_label, classes="attachment-url-chip"))

        preview = ", ".join(labels[:3])
        if len(labels) > 3:
            preview = f"{preview}, +{len(labels) - 3} more"
        summary_widget.update(
            f"Attachments ({len(self._pending_attachments)}): {preview}"
        )
        if self.pending_prompt is None:
            clear_button.disabled = False

    def _build_pending_attachments(self, source: str) -> tuple[list[PendingAttachment], int]:
        if source.startswith("http://") or source.startswith("https://"):
            if not self.is_likely_image_url(source):
                raise ValueError(
                    "Only image URLs are supported. Use a local file path for other file types."
                )
            return (
                [
                    PendingAttachment(
                        label=source,
                        content_part={"type": "input_image", "image_url": source},
                        preview_path=None,
                    )
                ],
                0,
            )

        path = Path(source).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            raise ValueError(f"Path not found: {path}")

        if path.is_dir():
            files = sorted(
                [item for item in path.rglob("*") if item.is_file()],
                key=lambda item: str(item).lower(),
            )
            if not files:
                raise ValueError(f"Folder has no files: {path}")
            supported_files: list[Path] = []
            unsupported_files_count = 0
            for file_path in files:
                if self.is_supported_attachment_file(file_path):
                    supported_files.append(file_path)
                else:
                    unsupported_files_count += 1
            if not supported_files:
                raise ValueError(f"Folder has no supported files: {path}")
            attachments: list[PendingAttachment] = []
            for file_path in supported_files:
                relative_name = file_path.relative_to(path).as_posix()
                attachments.append(
                    self._build_pending_attachment_from_file(
                        file_path,
                        filename=relative_name,
                    )
                )
            return attachments, unsupported_files_count

        if not path.is_file():
            raise ValueError(f"Not a file or folder: {path}")
        return [self._build_pending_attachment_from_file(path)], 0

    def _build_pending_attachment_from_file(
        self,
        path: Path,
        *,
        filename: Optional[str] = None,
    ) -> PendingAttachment:
        if not path.exists() or not path.is_file():
            raise ValueError(f"Not a file: {path}")
        if not self.is_supported_attachment_file(path):
            raise ValueError(f"Unsupported file type: {path}")

        mime_type, _ = mimetypes.guess_type(path.name)
        if not mime_type:
            mime_type = "application/octet-stream"

        try:
            file_bytes = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"Could not read file: {exc}") from exc
        if not file_bytes:
            raise ValueError(f"File is empty: {path}")

        encoded = base64.b64encode(file_bytes).decode("ascii")
        data_url = f"data:{mime_type};base64,{encoded}"
        if mime_type.startswith("image/"):
            return PendingAttachment(
                label=str(path),
                content_part={"type": "input_image", "image_url": data_url},
                preview_path=path,
            )
        return PendingAttachment(
            label=str(path),
            content_part={
                "type": "input_file",
                "filename": filename or path.name,
                "file_data": data_url,
            },
            preview_path=None,
        )

    @staticmethod
    def is_likely_image_file(path: Path) -> bool:
        mime_type, _ = mimetypes.guess_type(path.name)
        if mime_type in ChatApp.SUPPORTED_IMAGE_MIME_TYPES:
            return True
        return path.suffix.lower() in ChatApp.SUPPORTED_IMAGE_EXTENSIONS

    @classmethod
    def is_supported_attachment_file(cls, path: Path) -> bool:
        extension = path.suffix.lower()
        if extension in cls.SUPPORTED_IMAGE_EXTENSIONS:
            return True
        file_name = path.name.lower()
        if file_name in cls.SUPPORTED_ATTACHMENT_FILENAMES:
            return True
        if extension in cls.SUPPORTED_ATTACHMENT_EXTENSIONS:
            return True
        mime_type, _ = mimetypes.guess_type(path.name)
        if not mime_type:
            return False
        if mime_type.startswith("text/"):
            return True
        return mime_type in cls.SUPPORTED_ATTACHMENT_MIME_TYPES

    @staticmethod
    def is_likely_image_url(url: str) -> bool:
        mime_type, _ = mimetypes.guess_type(url)
        if mime_type in ChatApp.SUPPORTED_IMAGE_MIME_TYPES:
            return True
        return Path(url).suffix.lower() in ChatApp.SUPPORTED_IMAGE_EXTENSIONS

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
            grid.mount(Static("No images generated yet.", classes="image-grid-empty"))
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

    def _log_user(self, prompt: str, *, attachment_count: int = 0) -> None:
        prefix = "You:"
        if attachment_count > 0:
            prefix = f"You (+{attachment_count} attachment{'s' if attachment_count != 1 else ''}):"
        self.query_one(RichLog).write(Text(f"{prefix} {prompt}"))
        self._transcript_lines.append(f"{prefix} {prompt}")

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
