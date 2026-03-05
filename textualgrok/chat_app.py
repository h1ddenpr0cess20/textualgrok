"""Main Textual chat application."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from textual import events, on
from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.containers import Horizontal, HorizontalScroll, Vertical
from textual.widgets import Button, Footer, Header, RichLog, Static, TextArea
from textual.worker import Worker, WorkerState

from textualgrok.attachment_handler import (
    build_pending_attachments,
    refresh_pending_attachments_ui,
)
from textualgrok.attachment_screens import AddAttachmentScreen, BrowseAttachmentFileScreen
from textualgrok.chat_log import code_theme_for_app_theme, write_chat_log_entry
from textualgrok.command_provider import OrderedSystemCommandsProvider
from textualgrok.config import AppConfig
from textualgrok.conversation import ConversationState
from textualgrok.image_gallery_screen import ImageGalleryScreen
from textualgrok.image_manager import (
    cleanup_temp_images,
    materialize_session_images,
    refresh_image_grid,
    resolve_image_index_from_click,
)
from textualgrok.models import ChatResult
from textualgrok.options import RequestOptions, build_request_options
from textualgrok.settings_persistence import load_ui_settings, save_ui_settings
from textualgrok.settings_screen import SettingsScreen
from textualgrok.ui_types import PendingAttachment, SessionImageItem, UISettings
from textualgrok.xai_client import XAIResponsesClient


class ChatApp(App[None]):
    TITLE = "Textual Grok"
    CSS_PATH = "app.tcss"
    COMMANDS = {OrderedSystemCommandsProvider}

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("enter", "send_prompt", "Send", show=False, priority=True),
        Binding("ctrl+s", "send_prompt", "Send"),
        Binding("ctrl+t", "cycle_theme", "Cycle Theme", priority=True),
        Binding("ctrl+up", "prompt_history_prev", "Prev Prompt"),
        Binding("ctrl+down", "prompt_history_next", "Next Prompt"),
    ]
    DEFAULT_THEME_NAME = "dracula"

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.client = XAIResponsesClient(api_key=config.api_key, model=config.model)
        self.conversation = ConversationState(config.system_prompt)
        default_settings = UISettings(
            chat_model=config.model,
            system_prompt=config.system_prompt,
            image_model=config.image_model,
        )
        self.settings = load_ui_settings(default_settings)
        self.pending_prompt: Optional[str] = None
        self.pending_user_content: Optional[object] = None
        self.pending_input: Optional[list] = None
        self.pending_options: Optional[RequestOptions] = None
        self.last_image_url: Optional[str] = None
        self._pending_attachments: list[PendingAttachment] = []
        self._session_images: list[SessionImageItem] = []
        self._temp_image_paths: list[Path] = []
        self._gallery_open = False
        self._transcript_lines: list[str] = []
        self._chat_log_entries: list[dict[str, Any]] = []
        self._prompt_history: list[str] = []
        self._prompt_history_index: Optional[int] = None
        self._prompt_history_draft: str = ""

    def compose(self) -> ComposeResult:
        yield Header(icon="☰")
        with Vertical(id="chat-column"):
            yield RichLog(id="chat-log", auto_scroll=True, wrap=False, highlight=False, markup=False)
            with Vertical(id="image-panel", classes="hidden"):
                yield Static("Session Images", id="image-panel-title")
                with HorizontalScroll(id="image-grid"):
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
                    "(Enter send, Shift+Enter newline, Ctrl+S send, Ctrl+Up/Down history)"
                ),
                id="prompt",
            )
            yield Button("Send", id="send-btn", variant="primary")
        yield Footer()

    # Lifecycle

    def on_mount(self) -> None:
        self.theme_changed_signal.subscribe(self, self._on_theme_changed)
        theme_name = self.settings.theme.strip() if isinstance(self.settings.theme, str) else ""
        if not theme_name:
            theme_name = self.DEFAULT_THEME_NAME
        self._set_theme(theme_name)
        self.query_one("#prompt", TextArea).focus()
        self._refresh_pending_attachments_ui()

    def watch_theme(self, theme_name: str) -> None:
        self._persist_theme(theme_name)

    def _on_theme_changed(self, theme: object) -> None:
        theme_name = getattr(theme, "name", None)
        if isinstance(theme_name, str):
            self._persist_theme(theme_name)
            self._rerender_chat_log()

    def on_unmount(self) -> None:
        self._persist_theme(self.theme)
        cleanup_temp_images(self._temp_image_paths)
        self.client.close()

    # Commands / palette

    def get_system_commands(self, screen) -> Iterable[SystemCommand]:  # type: ignore[override]
        yield SystemCommand("Settings", "Open settings dialog", self.action_open_settings)
        yield SystemCommand("Clear Chat", "Clear chat log and conversation history", self.action_clear_chat)
        yield SystemCommand("Save Chat", "Save chat transcript to exports", self.action_save_chat)
        yield from super().get_system_commands(screen)

    # Actions

    def action_open_settings(self) -> None:
        self.push_screen(SettingsScreen(self.settings, self.client), self._on_settings_closed)

    def _on_settings_closed(self, settings: Optional[UISettings]) -> None:
        if settings is None:
            self._set_status("Settings unchanged.")
        else:
            self.settings = settings
            save_ui_settings(settings)
            self._set_status("Settings saved.")
        self.query_one("#prompt", TextArea).focus()

    def action_clear_chat(self) -> None:
        self.conversation.clear_history()
        self.query_one(RichLog).clear()
        self._transcript_lines.clear()
        self._chat_log_entries.clear()
        self.last_image_url = None
        self._pending_attachments.clear()
        self._session_images.clear()
        cleanup_temp_images(self._temp_image_paths)
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

    # UI events

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
        try:
            attachments, skipped_unsupported = build_pending_attachments(result)
        except ValueError as exc:
            self._set_status(str(exc))
            self.query_one("#prompt", TextArea).focus()
            return

        self._pending_attachments.extend(attachments)
        self._refresh_pending_attachments_ui()
        is_folder_source = False
        source_path: Optional[Path] = None
        if not (result.startswith("http://") or result.startswith("https://")):
            source_path = Path(result).expanduser()
            if not source_path.is_absolute():
                source_path = (Path.cwd() / source_path).resolve()
            is_folder_source = source_path.exists() and source_path.is_dir()
        if is_folder_source:
            assert source_path is not None
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
        index = resolve_image_index_from_click(event)
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
        self._open_image_gallery(index)

    @on(events.MouseScrollDown, "#image-grid")
    def on_image_grid_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        grid = self.query_one("#image-grid", HorizontalScroll)
        grid.scroll_relative(x=8, y=0, animate=False)
        event.stop()

    @on(events.MouseScrollUp, "#image-grid")
    def on_image_grid_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        grid = self.query_one("#image-grid", HorizontalScroll)
        grid.scroll_relative(x=-8, y=0, animate=False)
        event.stop()

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

    def action_cycle_theme(self) -> None:
        theme_names = list(self.available_themes.keys())
        if not theme_names:
            return
        current_theme = self.theme if isinstance(self.theme, str) else ""
        try:
            current_index = theme_names.index(current_theme)
        except ValueError:
            current_index = -1
        next_theme = theme_names[(current_index + 1) % len(theme_names)]
        self._set_theme(next_theme)
        self._set_status(f"Theme: {next_theme}")

    # Prompt handling

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
                    and len(payload) == 3
                    and isinstance(payload[0], list)
                    and isinstance(payload[1], list)
                    and isinstance(payload[2], list)
                ):
                    images, errors, new_temp_paths = payload
                    for item in images:
                        if isinstance(item, SessionImageItem):
                            self._session_images.append(item)
                    self._temp_image_paths.extend(new_temp_paths)
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
                    lambda: materialize_session_images(
                        result, fetch_bytes_fn=self.client.fetch_bytes
                    ),
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

    # Attachment UI

    def _refresh_pending_attachments_ui(self) -> None:
        refresh_pending_attachments_ui(
            self._pending_attachments,
            self.query_one("#attachments-status", Static),
            self.query_one("#clear-attachments-btn", Button),
            self.query_one("#attachments-thumbs", HorizontalScroll),
            is_request_in_progress=self.pending_prompt is not None,
        )

    # Image grid and gallery

    def _refresh_image_grid(self) -> None:
        refresh_image_grid(
            self._session_images,
            self.query_one("#image-panel", Vertical),
            self.query_one("#image-grid", HorizontalScroll),
        )

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

    # Logging

    def _set_prompt_text(self, text: str) -> None:
        prompt_widget = self.query_one("#prompt", TextArea)
        prompt_widget.load_text(text)
        lines = text.splitlines() or [""]
        prompt_widget.move_cursor((len(lines) - 1, len(lines[-1])))

    def _log_user(self, prompt: str, *, attachment_count: int = 0) -> None:
        prefix = "You:"
        if attachment_count > 0:
            prefix = f"You (+{attachment_count} attachment{'s' if attachment_count != 1 else ''}):"
        entry: dict[str, Any] = {
            "kind": "user",
            "message": prompt,
            "attachment_count": attachment_count,
        }
        self._chat_log_entries.append(entry)
        self._write_entry(entry)
        self._transcript_lines.append(f"{prefix} {prompt}")

    def _log_assistant(self, message: str) -> None:
        entry: dict[str, Any] = {"kind": "assistant", "message": message}
        self._chat_log_entries.append(entry)
        self._write_entry(entry)
        self._transcript_lines.append(f"Grok:\n{message}")

    def _log_error(self, message: str) -> None:
        entry: dict[str, Any] = {"kind": "error", "message": message}
        self._chat_log_entries.append(entry)
        self._write_entry(entry)
        self._transcript_lines.append(f"Error: {message}")

    def _set_status(self, message: str) -> None:
        self.query_one("#status", Static).update(message)

    def _write_entry(self, entry: dict[str, Any]) -> None:
        write_chat_log_entry(
            entry,
            self.query_one(RichLog),
            code_theme=self._current_code_theme(),
        )

    def _rerender_chat_log(self) -> None:
        chat_log = self.query_one(RichLog)
        chat_log.clear()
        code_theme = self._current_code_theme()
        for entry in self._chat_log_entries:
            write_chat_log_entry(entry, chat_log, code_theme=code_theme)

    def _current_code_theme(self) -> str:
        theme_name = self.theme.strip() if isinstance(self.theme, str) else ""
        return code_theme_for_app_theme(theme_name, is_dark=self.current_theme.dark)

    # Theme persistence

    def _set_theme(self, theme_name: str) -> None:
        try:
            self.theme = theme_name
        except Exception:
            self.theme = self.DEFAULT_THEME_NAME

    def _persist_theme(self, theme_name: object) -> None:
        if not hasattr(self, "settings"):
            return
        if not isinstance(theme_name, str):
            return
        normalized_theme = theme_name.strip()
        if not normalized_theme:
            return
        if self.settings.theme == normalized_theme:
            return
        self.settings.theme = normalized_theme
        save_ui_settings(self.settings)

    # Export

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
