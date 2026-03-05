"""Session image materialization, temp file management, and grid UI."""

from __future__ import annotations

import base64
import re
import tempfile
from pathlib import Path

from textual.containers import Vertical
from textual.widgets import Button, Static

from textual_image.widget import Image as TextualImageWidget

from textualgrok.models import ChatResult
from textualgrok.ui_types import SessionImageItem


def materialize_session_images(
    result: ChatResult,
    *,
    fetch_bytes_fn: object,
) -> tuple[list[SessionImageItem], list[str], list[Path]]:
    """Download/decode images from a ChatResult into temp files.

    Args:
        result: The chat result containing image URLs and/or base64 data.
        fetch_bytes_fn: Callable(url: str) -> bytes for downloading URLs.

    Returns:
        Tuple of (images, errors, temp_paths).
    """
    images: list[SessionImageItem] = []
    errors: list[str] = []
    temp_paths: list[Path] = []

    urls = result.image_urls or []
    for image_url in urls:
        try:
            image_bytes = fetch_bytes_fn(image_url)
            suffix = guess_image_suffix_from_url(image_url)
            image_path = write_temp_image_bytes(image_bytes, suffix=suffix)
            temp_paths.append(image_path)
            images.append(SessionImageItem(path=image_path, source=image_url))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Failed to load image URL {image_url}: {exc}")

    base64_images = result.image_b64 or []
    for index, image_b64 in enumerate(base64_images, start=1):
        try:
            image_bytes = base64.b64decode(image_b64, validate=True)
            image_path = write_temp_image_bytes(image_bytes, suffix=".png")
            temp_paths.append(image_path)
            images.append(SessionImageItem(path=image_path, source=f"(embedded base64 image {index})"))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Failed to decode base64 image {index}: {exc}")

    return images, errors, temp_paths


def write_temp_image_bytes(image_bytes: bytes, *, suffix: str) -> Path:
    """Write image bytes to a temp file and return the path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(image_bytes)
        return Path(temp.name)


def guess_image_suffix_from_url(image_url: str) -> str:
    """Guess the file extension from an image URL."""
    match = re.search(r"\.(png|jpg|jpeg|webp|gif)(?:\?|$)", image_url, flags=re.IGNORECASE)
    if not match:
        return ".jpg"
    ext = match.group(1).lower()
    if ext == "jpeg":
        ext = "jpg"
    return f".{ext}"


def cleanup_temp_images(temp_paths: list[Path]) -> None:
    """Delete all temp image files and clear the list."""
    for image_path in temp_paths:
        try:
            image_path.unlink(missing_ok=True)
        except OSError:
            pass
    temp_paths.clear()


def refresh_image_grid(
    session_images: list[SessionImageItem],
    panel: Vertical,
    grid: object,
) -> None:
    """Rebuild the image grid UI from the current session images."""
    from textual.containers import HorizontalScroll

    grid_widget: HorizontalScroll = grid  # type: ignore[assignment]
    grid_widget.remove_children()

    if not session_images:
        panel.add_class("hidden")
        grid_widget.mount(Static("No images generated yet.", classes="image-grid-empty"))
        return

    panel.remove_class("hidden")
    for index, item in enumerate(session_images):
        thumb = TextualImageWidget(classes="image-thumb")
        thumb.image = item.path
        setattr(thumb, "_image_index", index)
        caption_source = item.source
        if len(caption_source) > 42:
            caption_source = f"{caption_source[:39]}..."
        caption = Static(f"{index + 1}. Click image to open\n{caption_source}", classes="image-caption")
        setattr(caption, "_image_index", index)
        open_button = Button("Open", classes="image-open-btn")
        setattr(open_button, "_image_index", index)
        tile = Vertical(thumb, caption, open_button, classes="image-tile")
        setattr(tile, "_image_index", index)
        grid_widget.mount(tile)


def resolve_image_index_from_click(event: object) -> int | None:
    """Walk the widget tree from a click event to find an _image_index attribute."""
    from textual import events

    click_event: events.Click = event  # type: ignore[assignment]
    candidates: list = []
    if isinstance(click_event.widget, object):
        candidates.append(click_event.widget)

    control = click_event.control
    if isinstance(control, object) and control not in candidates:
        candidates.append(control)

    for node in candidates:
        current = node
        while hasattr(current, "parent"):
            image_index = getattr(current, "_image_index", None)
            if isinstance(image_index, int):
                return image_index
            current = current.parent  # type: ignore[assignment]
            if current is None:
                break
    return None
