"""Tests for render_tags module."""

import pytest
from textualgrok.render_tags import (
    parse_tag_attributes,
    parse_render_body,
    looks_like_structured_payload,
    handle_render_tags,
    parse_function_arguments,
)


class TestParseTagAttributes:
    def test_double_quoted(self):
        attrs = parse_tag_attributes('type="generate_image" prompt="a cat"')
        assert attrs["type"] == "generate_image"
        assert attrs["prompt"] == "a cat"

    def test_single_quoted(self):
        attrs = parse_tag_attributes("type='generate_image'")
        assert attrs["type"] == "generate_image"

    def test_empty_string(self):
        assert parse_tag_attributes("") == {}

    def test_no_attrs(self):
        assert parse_tag_attributes("   ") == {}

    def test_multiple_attrs(self):
        attrs = parse_tag_attributes('n="2" response_format="url" aspect_ratio="16:9"')
        assert attrs["n"] == "2"
        assert attrs["response_format"] == "url"
        assert attrs["aspect_ratio"] == "16:9"


class TestParseRenderBody:
    def test_json_body(self):
        body = '{"prompt": "a dog", "n": 2}'
        result = parse_render_body(body)
        assert result["prompt"] == "a dog"
        assert result["n"] == 2

    def test_xml_tags(self):
        body = "<prompt>a cat</prompt><n>3</n>"
        result = parse_render_body(body)
        assert result["prompt"] == "a cat"
        assert result["n"] == "3"

    def test_line_pairs(self):
        body = "prompt: a sunset\nn: 1\nresponse_format: url"
        result = parse_render_body(body)
        assert result["prompt"] == "a sunset"
        assert result["n"] == "1"
        assert result["response_format"] == "url"

    def test_empty_body(self):
        assert parse_render_body("") == {}

    def test_fenced_json(self):
        body = "```json\n{\"prompt\": \"mountain\"}\n```"
        result = parse_render_body(body)
        assert result["prompt"] == "mountain"

    def test_plain_text_not_structured(self):
        result = parse_render_body("a beautiful landscape")
        assert result == {}


class TestLooksLikeStructuredPayload:
    def test_json_braces(self):
        assert looks_like_structured_payload('{"prompt": "x", "n": 1}')

    def test_prompt_colon(self):
        assert looks_like_structured_payload("prompt: something\nn: 1")

    def test_plain_text(self):
        assert not looks_like_structured_payload("a beautiful landscape painting")

    def test_single_marker_not_structured(self):
        assert not looks_like_structured_payload("response_format only")


class TestHandleRenderTags:
    def _make_tool(self, urls=None, b64=None, error=None):
        result = {}
        if urls is not None:
            result["images"] = urls
        if b64 is not None:
            result["images_b64"] = b64
        if error is not None:
            result["error"] = error
        return lambda arguments, options, fallback_source_image_url: result

    def test_no_tags_returns_unchanged(self):
        text, urls, b64 = handle_render_tags(
            text="Hello world",
            execute_image_tool=self._make_tool(),
            options=None,
            fallback_source_image_url=None,
        )
        assert text == "Hello world"
        assert urls == []
        assert b64 == []

    def test_empty_text(self):
        text, urls, b64 = handle_render_tags(
            text="",
            execute_image_tool=self._make_tool(),
            options=None,
            fallback_source_image_url=None,
        )
        assert text == ""
        assert urls == []

    def test_tag_with_json_body_generates_url(self):
        execute = self._make_tool(urls=["http://img.example.com/1.png"])
        text = '<grok:render type="generate_image">{"prompt": "a cat"}</grok:render>'
        cleaned, urls, b64 = handle_render_tags(
            text=text,
            execute_image_tool=execute,
            options=None,
            fallback_source_image_url=None,
        )
        assert "http://img.example.com/1.png" in cleaned
        assert urls == ["http://img.example.com/1.png"]

    def test_tag_with_error_shows_failure(self):
        execute = self._make_tool(error="quota exceeded")
        text = '<grok:render type="generate_image">{"prompt": "x"}</grok:render>'
        cleaned, urls, b64 = handle_render_tags(
            text=text,
            execute_image_tool=execute,
            options=None,
            fallback_source_image_url=None,
        )
        assert "Image generation failed" in cleaned
        assert "quota exceeded" in cleaned

    def test_tag_with_prompt_attr(self):
        execute = self._make_tool(urls=["http://result.png"])
        text = '<grok:render type="generate_image" prompt="a sunny day"></grok:render>'
        cleaned, urls, _ = handle_render_tags(
            text=text,
            execute_image_tool=execute,
            options=None,
            fallback_source_image_url=None,
        )
        assert urls == ["http://result.png"]

    def test_unrecognized_type_skipped(self):
        execute = self._make_tool(urls=["http://img.png"])
        text = '<grok:render type="some_other_type">ignored</grok:render>'
        cleaned, urls, _ = handle_render_tags(
            text=text,
            execute_image_tool=execute,
            options=None,
            fallback_source_image_url=None,
        )
        assert urls == []
        assert "ignored" in cleaned  # tag not stripped since it was skipped

    def test_tag_no_prompt_skipped(self):
        calls = []
        def execute(arguments, options, fallback_source_image_url):
            calls.append(arguments)
            return {}
        # structured payload with no extractable prompt
        text = '<grok:render type="generate_image">{"n": 1, "response_format": "url"}</grok:render>'
        cleaned, urls, _ = handle_render_tags(
            text=text,
            execute_image_tool=execute,
            options=None,
            fallback_source_image_url=None,
        )
        assert calls == []  # tool not called when no prompt


class TestParseFunctionArguments:
    def test_dict_passthrough(self):
        result = parse_function_arguments({"prompt": "x"})
        assert result == {"prompt": "x"}

    def test_json_string(self):
        result = parse_function_arguments('{"prompt": "test", "n": 2}')
        assert result["prompt"] == "test"
        assert result["n"] == 2

    def test_empty_string(self):
        assert parse_function_arguments("") == {}

    def test_non_string_non_dict(self):
        assert parse_function_arguments(42) == {}

    def test_trailing_comma_repaired(self):
        result = parse_function_arguments('{"prompt": "x",}')
        assert result.get("prompt") == "x"

    def test_line_pairs_fallback(self):
        result = parse_function_arguments("prompt: a dog\nn: 1")
        assert result.get("prompt") == "a dog"
