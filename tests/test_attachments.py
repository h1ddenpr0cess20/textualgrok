"""Tests for attachment file-type detection helpers."""

from pathlib import Path
import pytest
from textualgrok.attachments import (
    is_likely_image_file,
    is_likely_image_url,
    is_supported_attachment_file,
)


class TestIsLikelyImageFile:
    def test_jpeg(self):
        assert is_likely_image_file(Path("photo.jpg"))

    def test_jpeg_upper(self):
        assert is_likely_image_file(Path("photo.JPG"))

    def test_png(self):
        assert is_likely_image_file(Path("image.png"))

    def test_gif_not_supported(self):
        assert not is_likely_image_file(Path("anim.gif"))

    def test_text_file(self):
        assert not is_likely_image_file(Path("readme.txt"))


class TestIsLikelyImageUrl:
    def test_jpg_url(self):
        assert is_likely_image_url("https://example.com/photo.jpg")

    def test_png_url(self):
        assert is_likely_image_url("https://example.com/image.png")

    def test_non_image_url(self):
        assert not is_likely_image_url("https://example.com/doc.pdf")


class TestIsSupportedAttachmentFile:
    def test_python_file(self):
        assert is_supported_attachment_file(Path("script.py"))

    def test_pdf(self):
        assert is_supported_attachment_file(Path("doc.pdf"))

    def test_markdown(self):
        assert is_supported_attachment_file(Path("notes.md"))

    def test_json(self):
        assert is_supported_attachment_file(Path("data.json"))

    def test_image_also_supported(self):
        assert is_supported_attachment_file(Path("image.png"))

    def test_dockerfile_no_extension(self):
        assert is_supported_attachment_file(Path("Dockerfile"))

    def test_makefile_no_extension(self):
        assert is_supported_attachment_file(Path("Makefile"))

    def test_unknown_binary_not_supported(self):
        assert not is_supported_attachment_file(Path("binary.exe"))

    def test_mp4_not_supported(self):
        assert not is_supported_attachment_file(Path("video.mp4"))

    def test_csv(self):
        assert is_supported_attachment_file(Path("data.csv"))

    def test_shell_script(self):
        assert is_supported_attachment_file(Path("setup.sh"))

    def test_toml(self):
        assert is_supported_attachment_file(Path("pyproject.toml"))
