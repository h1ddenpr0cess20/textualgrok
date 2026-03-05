"""Build PendingAttachments from file paths and URLs, manage attachment UI."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Optional

from textual.containers import HorizontalScroll
from textual.widgets import Button, Static

from textual_image.widget import Image as TextualImageWidget

from textualgrok.attachments import (
    is_likely_image_url,
    is_supported_attachment_file,
)
from textualgrok.ui_types import PendingAttachment


def build_pending_attachments(source: str) -> tuple[list[PendingAttachment], int]:
    """Build PendingAttachments from a source path or URL.

    Args:
        source: A file path, folder path, or image URL.

    Returns:
        Tuple of (attachments, skipped_unsupported_count).

    Raises:
        ValueError: If the source is invalid or unsupported.
    """
    if source.startswith("http://") or source.startswith("https://"):
        if not is_likely_image_url(source):
            raise ValueError("Only image URLs are supported. Use a local file path for other file types.")
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
        files = sorted([item for item in path.rglob("*") if item.is_file()], key=lambda item: str(item).lower())
        if not files:
            raise ValueError(f"Folder has no files: {path}")
        supported_files: list[Path] = []
        unsupported_files_count = 0
        for file_path in files:
            if is_supported_attachment_file(file_path):
                supported_files.append(file_path)
            else:
                unsupported_files_count += 1
        if not supported_files:
            raise ValueError(f"Folder has no supported files: {path}")
        attachments: list[PendingAttachment] = []
        for file_path in supported_files:
            relative_name = file_path.relative_to(path).as_posix()
            attachments.append(build_pending_attachment_from_file(file_path, filename=relative_name))
        return attachments, unsupported_files_count

    if not path.is_file():
        raise ValueError(f"Not a file or folder: {path}")
    return [build_pending_attachment_from_file(path)], 0


def build_pending_attachment_from_file(path: Path, *, filename: Optional[str] = None) -> PendingAttachment:
    """Build a PendingAttachment from a single file path.

    Raises:
        ValueError: If the file is missing, unsupported, or unreadable.
    """
    if not path.exists() or not path.is_file():
        raise ValueError(f"Not a file: {path}")
    if not is_supported_attachment_file(path):
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
        content_part={"type": "input_file", "filename": filename or path.name, "file_data": data_url},
        preview_path=None,
    )


def refresh_pending_attachments_ui(
    pending_attachments: list[PendingAttachment],
    summary_widget: Static,
    clear_button: Button,
    thumbs: HorizontalScroll,
    *,
    is_request_in_progress: bool,
) -> None:
    """Update the attachment bar UI to reflect current pending attachments."""
    thumbs.remove_children()

    if not pending_attachments:
        summary_widget.update("Attachments: none")
        thumbs.add_class("hidden")
        if not is_request_in_progress:
            clear_button.disabled = True
        return

    thumbs.remove_class("hidden")
    labels: list[str] = []
    for attachment in pending_attachments:
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
    summary_widget.update(f"Attachments ({len(pending_attachments)}): {preview}")
    if not is_request_in_progress:
        clear_button.disabled = False
