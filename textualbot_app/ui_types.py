"""Shared UI-related dataclasses used across the Textual app."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


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

