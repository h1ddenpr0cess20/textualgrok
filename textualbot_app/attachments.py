"""Attachment helpers and supported file type detection."""

from __future__ import annotations

import mimetypes
from pathlib import Path

# Text/office/code file types the backend accepts.
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

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

SUPPORTED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png"}

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


def is_likely_image_file(path: Path) -> bool:
    """Return True if a file path looks like an image we support."""
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type in SUPPORTED_IMAGE_MIME_TYPES:
        return True
    return path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def is_likely_image_url(url: str) -> bool:
    """Return True if a URL looks like a supported image."""
    mime_type, _ = mimetypes.guess_type(url)
    if mime_type in SUPPORTED_IMAGE_MIME_TYPES:
        return True
    return Path(url).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def is_supported_attachment_file(path: Path) -> bool:
    """Return True if the file matches known supported types."""
    extension = path.suffix.lower()
    if extension in SUPPORTED_IMAGE_EXTENSIONS:
        return True
    file_name = path.name.lower()
    if file_name in SUPPORTED_ATTACHMENT_FILENAMES:
        return True
    if extension in SUPPORTED_ATTACHMENT_EXTENSIONS:
        return True
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type:
        return False
    if mime_type.startswith("text/"):
        return True
    return mime_type in SUPPORTED_ATTACHMENT_MIME_TYPES

