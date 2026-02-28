"""Settings modal screen."""

import json
from typing import Any, Optional

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static, Switch, TabPane, TabbedContent
from textual.widgets.select import InvalidSelectValueError
from textual.worker import Worker, WorkerState

from textualgrok.options import build_request_options
from textualgrok.ui_types import UISettings
from textualgrok.xai_client import XAIResponsesClient


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

    BINDINGS = [Binding("escape", "dismiss_none", "Close")]

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

                with TabPane("MCP"):
                    with VerticalScroll(id="mcp-scroll", classes="settings-tab-scroll"):
                        with Horizontal(classes="settings-row"):
                            yield Label("Enable MCP Tools")
                            yield Switch(value=self.initial.mcp_enabled, id="switch-mcp-enabled")
                        with Horizontal(classes="settings-row"):
                            yield Button("Add Server", id="btn-mcp-add")
                            yield Button("Remove Selected", id="btn-mcp-remove")
                        yield Label("Configured MCP Servers")
                        yield Select([], allow_blank=True, id="select-mcp-servers", classes="settings-input")
                        yield Static("No MCP servers configured.", id="mcp-status")
                        yield Label("Server Label (optional)")
                        yield Input(value="", placeholder="deepwiki", id="input-mcp-label", classes="settings-input")
                        yield Label("Server URL")
                        yield Input(value="", placeholder="https://example.com/mcp", id="input-mcp-url", classes="settings-input")
                        yield Label("Server Description (optional)")
                        yield Input(value="", id="input-mcp-description", classes="settings-input")
                        yield Label("Allowed Tool Names (comma-separated, optional)")
                        yield Input(value="", id="input-mcp-allowed-tools", classes="settings-input")
                        yield Label("Authorization Header Value (optional)")
                        yield Input(value="", id="input-mcp-authorization", classes="settings-input")
                        yield Label("Extra Headers JSON (optional)")
                        yield Input(value="", placeholder='{"X-Org":"demo"}', id="input-mcp-extra-headers", classes="settings-input")

            with Horizontal(id="settings-actions"):
                yield Button("Cancel", variant="default", id="btn-settings-cancel")
                yield Button("Save", variant="primary", id="btn-settings-save")

    def on_mount(self) -> None:
        self._sync_inputs()
        self._refresh_mcp_servers_ui()
        self._refresh_models()

    # Events

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

    # Model loading

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
            chat_count, image_count = self._set_model_options(models)
            status_parts = [f"Loaded {len(models)} model(s) from server."]
            if chat_count == 0:
                status_parts.append("No non-grok-imagine models were returned for chat.")
            if image_count == 0:
                status_parts.append("No grok-imagine models were returned for image generation.")
            self.query_one("#models-status", Static).update(" ".join(status_parts))
        else:
            self.query_one("#models-status", Static).update(f"Model list load failed: {event.worker.error}")

    @staticmethod
    def _is_imagine_model(model: str) -> bool:
        return model.casefold().startswith("grok-imagine")

    def _set_model_options(self, models: list[str]) -> tuple[int, int]:
        chat_models = [model for model in models if not self._is_imagine_model(model)]
        image_models = [model for model in models if self._is_imagine_model(model)]

        self._set_filtered_model_options("#select-chat-model", chat_models, self.initial.chat_model)
        self._set_filtered_model_options("#input-image-model", image_models, self.initial.image_model)

        return len(chat_models), len(image_models)

    def _set_filtered_model_options(
        self,
        selector: str,
        models: list[str],
        fallback_default: str,
    ) -> None:
        if not models:
            return

        select = self.query_one(selector, Select)
        current_value = select.value
        fallback = current_value if isinstance(current_value, str) and current_value else fallback_default
        options = [(model, model) for model in models]

        select.set_options(options)
        if fallback in models:
            select.value = fallback
        else:
            select.value = models[0]

    # MCP helpers

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
            select.clear()
            remove_button.disabled = True
            self.query_one("#mcp-status", Static).update("No MCP servers configured.")
        self._sync_inputs()

    def _fill_mcp_server_form(self, server: dict[str, Any]) -> None:
        self.query_one("#input-mcp-label", Input).value = str(server.get("server_label", "")).strip()
        self.query_one("#input-mcp-url", Input).value = str(server.get("server_url", "")).strip()
        self.query_one("#input-mcp-description", Input).value = str(server.get("server_description", "")).strip()

        allowed = server.get("allowed_tool_names")
        allowed_names: list[str] = []
        if isinstance(allowed, list):
            for item in allowed:
                name = str(item).strip()
                if name:
                    allowed_names.append(name)
        self.query_one("#input-mcp-allowed-tools", Input).value = ",".join(allowed_names)

        self.query_one("#input-mcp-authorization", Input).value = str(server.get("authorization", "")).strip()

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

    # Input helpers

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
            image_source_url_raw="",
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

