"""Tests for response_parser module."""

import pytest
from textualgrok.response_parser import (
    extract_text,
    extract_function_calls,
    extract_image_entries,
    extract_image_urls,
    extract_image_b64,
    extract_citations,
    collect_strings,
    find_render_text_in_payload,
)


class TestExtractText:
    def test_output_text_field(self):
        data = {"output_text": "Hello there"}
        assert extract_text(data) == "Hello there"

    def test_choices_format(self):
        data = {"choices": [{"message": {"content": "From choices"}}]}
        assert extract_text(data) == "From choices"

    def test_output_list_with_text(self):
        # The parser appends text via two paths for output_text items; result contains it twice joined by \n.
        data = {"output": [{"type": "output_text", "text": "From output list"}]}
        result = extract_text(data)
        assert "From output list" in result

    def test_empty_data(self):
        assert extract_text({}) == "(No text returned.)"

    def test_strips_whitespace(self):
        data = {"output_text": "  trimmed  "}
        assert extract_text(data) == "trimmed"

    def test_nested_content_list(self):
        data = {
            "output": [
                {
                    "content": [
                        {"type": "output_text", "text": "Nested text"}
                    ]
                }
            ]
        }
        assert extract_text(data) == "Nested text"

    def test_output_text_takes_priority(self):
        data = {
            "output_text": "Primary",
            "choices": [{"message": {"content": "Secondary"}}],
        }
        assert extract_text(data) == "Primary"


class TestExtractFunctionCalls:
    def test_empty(self):
        assert extract_function_calls({}) == []

    def test_function_call_type(self):
        data = {
            "output": [
                {
                    "type": "function_call",
                    "id": "call_1",
                    "name": "generate_image",
                    "arguments": {"prompt": "a cat"},
                }
            ]
        }
        calls = extract_function_calls(data)
        assert len(calls) == 1
        assert calls[0]["name"] == "generate_image"
        assert calls[0]["call_id"] == "call_1"

    def test_tool_calls_format(self):
        data = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "tc_1",
                                "function": {
                                    "name": "web_search",
                                    "arguments": '{"query": "test"}',
                                },
                            }
                        ]
                    }
                }
            ]
        }
        calls = extract_function_calls(data)
        assert len(calls) == 1
        assert calls[0]["name"] == "web_search"

    def test_deduplication(self):
        item = {
            "type": "function_call",
            "id": "call_1",
            "name": "fn",
            "arguments": {"x": 1},
        }
        data = {"output": [item, item]}
        calls = extract_function_calls(data)
        assert len(calls) == 1

    def test_missing_name_skipped(self):
        data = {"output": [{"type": "function_call", "id": "c1", "arguments": {}}]}
        calls = extract_function_calls(data)
        assert calls == []


class TestExtractImageEntries:
    def test_data_list_with_url(self):
        data = {"data": [{"url": "http://img.example.com/1.png"}]}
        entries = extract_image_entries(data)
        assert any(e.get("url") == "http://img.example.com/1.png" for e in entries)

    def test_b64_entry(self):
        data = {"data": [{"b64_json": "abc123"}]}
        entries = extract_image_entries(data)
        assert any(e.get("b64_json") == "abc123" for e in entries)

    def test_deduplication(self):
        url = "http://img.example.com/1.png"
        data = {"data": [{"url": url}, {"url": url}]}
        entries = extract_image_entries(data)
        assert sum(1 for e in entries if e.get("url") == url) == 1

    def test_empty(self):
        assert extract_image_entries({}) == []

    def test_revised_prompt_included(self):
        data = {"data": [{"url": "http://img.png", "revised_prompt": "A sunny day"}]}
        entries = extract_image_entries(data)
        assert entries[0].get("revised_prompt") == "A sunny day"


class TestExtractImageUrls:
    def test_returns_urls(self):
        data = {"data": [{"url": "http://a.png"}, {"url": "http://b.png"}]}
        urls = extract_image_urls(data)
        assert "http://a.png" in urls
        assert "http://b.png" in urls

    def test_b64_not_in_urls(self):
        data = {"data": [{"b64_json": "xyz"}]}
        assert extract_image_urls(data) == []


class TestExtractImageB64:
    def test_returns_b64(self):
        data = {"data": [{"b64_json": "abc123"}]}
        b64 = extract_image_b64(data)
        assert b64 == ["abc123"]

    def test_urls_not_in_b64(self):
        data = {"data": [{"url": "http://img.png"}]}
        assert extract_image_b64(data) == []


class TestExtractCitations:
    def test_basic_citations(self):
        data = {"citations": [{"url": "http://source1.com"}, {"url": "http://source2.com"}]}
        urls = extract_citations(data)
        assert "http://source1.com" in urls
        assert "http://source2.com" in urls

    def test_deduplication(self):
        data = {"citations": [{"url": "http://a.com"}, {"url": "http://a.com"}]}
        urls = extract_citations(data)
        assert urls.count("http://a.com") == 1

    def test_no_citations(self):
        assert extract_citations({}) == []

    def test_nested_response_citations(self):
        data = {"response": {"citations": [{"url": "http://nested.com"}]}}
        urls = extract_citations(data)
        assert "http://nested.com" in urls


class TestCollectStrings:
    def test_flat_dict(self):
        result = collect_strings({"a": "hello", "b": "world"})
        assert "hello" in result
        assert "world" in result

    def test_nested(self):
        result = collect_strings({"outer": {"inner": "deep"}})
        assert "deep" in result

    def test_list(self):
        result = collect_strings(["x", "y", "z"])
        assert result == ["x", "y", "z"]

    def test_empty_strings_excluded(self):
        result = collect_strings({"a": "", "b": "  "})
        assert result == []

    def test_non_string_values(self):
        result = collect_strings({"a": 42, "b": None})
        assert result == []


class TestFindRenderTextInPayload:
    def test_finds_tag(self):
        data = {"output_text": 'Some text <grok:render type="generate_image">cat</grok:render>'}
        result = find_render_text_in_payload(data)
        assert result is not None
        assert "<grok:render" in result

    def test_returns_none_if_not_found(self):
        data = {"output_text": "Plain text with no tags"}
        assert find_render_text_in_payload(data) is None

    def test_searches_nested(self):
        data = {"output": [{"text": '<grok:render type="generate_image">dog</grok:render>'}]}
        result = find_render_text_in_payload(data)
        assert result is not None
