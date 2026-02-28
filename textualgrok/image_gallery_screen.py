"""Image gallery screen for generated session images."""

from typing import Optional

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, HorizontalScroll, Vertical
from textual.screen import Screen
from textual.widgets import Button, Static

from textual_image.widget import Image as TextualImageWidget

from textualgrok.ui_types import SessionImageItem


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
            with HorizontalScroll(id="gallery-source-scroll"):
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
        lines = [line.strip() for line in value.splitlines() if line.strip()]
        if not lines:
            return value.strip()
        return " | ".join(lines)
