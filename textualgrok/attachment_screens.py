"""UI screens for selecting and attaching files/images."""

from pathlib import Path
import time
from typing import Optional

from textual import events, on
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, OptionList, Select, Static

from textual_image.widget import Image as TextualImageWidget

from textualgrok.attachments import is_likely_image_file, is_supported_attachment_file
from textualgrok.ui_types import BrowseEntry


class BrowseOptionList(OptionList):
    """OptionList with double-click detection."""

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
    """Modal to browse the filesystem and pick a file/folder."""

    def __init__(self, start_path: Path | None = None) -> None:
        super().__init__()
        self.start_path = (start_path or Path.home()).resolve()
        self._current_dir = self.start_path
        self._entries: list[BrowseEntry] = []
        self._selected_path: Optional[Path] = None

    def compose(self):
        with Vertical(id="browse-dialog"):
            yield Static("Browse File or Folder")
            with Horizontal(id="browse-nav"):
                yield Button("Home", id="btn-browse-home")
                yield Button("CWD", id="btn-browse-cwd")
                yield Select([], allow_blank=True, id="browse-drive-select")
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

    # Internal helpers

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

        directories = sorted([item for item in children if item.is_dir()], key=lambda p: p.name.lower())
        files = sorted([item for item in children if item.is_file()], key=lambda p: p.name.lower())
        supported_files: list[Path] = []
        unsupported_files_count = 0
        for file_path in files:
            if is_supported_attachment_file(file_path):
                supported_files.append(file_path)
            else:
                unsupported_files_count += 1

        for directory in directories:
            self._entries.append(BrowseEntry(path=directory, kind="dir", is_image=False))
            options.append(f"[DIR] {directory.name}")
        for file_path in supported_files:
            is_image = is_likely_image_file(file_path)
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
            select.set_options([( "/", "/" )])
            select.value = "/"
            return
        options = [(drive, drive) for drive in drives]
        select.set_options(options)
        self._sync_drive_select_with_path(self._current_dir)

    def _sync_drive_select_with_path(self, path: Path) -> None:
        select = self.query_one("#browse-drive-select", Select)
        value = self._matching_drive_value(path)
        if value is not None:
            select.value = value

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
    """Modal that accepts a path/URL or opens the browse dialog."""

    def compose(self):
        with Vertical(id="attach-dialog"):
            yield Static("Attach File or Folder", id="attach-title")
            yield Label("File path, folder path, or image URL")
            with Horizontal(id="attach-source-row"):
                yield Input(
                    value="",
                    placeholder=r"C:\\path\\to\\file-or-folder or https://example.com/image.jpg",
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
