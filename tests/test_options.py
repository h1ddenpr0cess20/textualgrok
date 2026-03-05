"""Tests for build_request_options and helpers."""

import pytest
from textualgrok.options import build_request_options, _parse_positive_int


# Minimal valid kwargs for build_request_options with nothing enabled.
def base_kwargs(**overrides):
    defaults = dict(
        web_search=False,
        x_search=False,
        x_search_image_understanding=False,
        x_search_video_understanding=False,
        code_interpreter=False,
        file_search=False,
        vector_store_ids_raw="",
        file_search_max_results_raw="10",
        image_generation=False,
        image_model="grok-imagine-image",
        image_count_raw="1",
        image_as_base64=False,
        image_source_url_raw="",
        image_use_last=True,
        image_aspect_ratio_raw="",
        mcp_enabled=False,
        mcp_servers=[],
    )
    defaults.update(overrides)
    return defaults


class TestParsePositiveInt:
    def test_valid(self):
        assert _parse_positive_int("5", field_name="X") == 5

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="greater than 0"):
            _parse_positive_int("0", field_name="X")

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="greater than 0"):
            _parse_positive_int("-1", field_name="X")

    def test_non_integer_raises(self):
        with pytest.raises(ValueError, match="must be an integer"):
            _parse_positive_int("abc", field_name="X")


class TestBuildRequestOptionsTools:
    def test_no_tools_by_default(self):
        opts = build_request_options(**base_kwargs())
        assert opts.tools == []

    def test_web_search_adds_tool(self):
        opts = build_request_options(**base_kwargs(web_search=True))
        assert any(t["type"] == "web_search" for t in opts.tools)

    def test_code_interpreter_adds_tool(self):
        opts = build_request_options(**base_kwargs(code_interpreter=True))
        assert any(t["type"] == "code_interpreter" for t in opts.tools)

    def test_x_search_basic(self):
        opts = build_request_options(**base_kwargs(x_search=True))
        x = next(t for t in opts.tools if t["type"] == "x_search")
        assert "filters" not in x

    def test_x_search_image_understanding_sets_filter(self):
        opts = build_request_options(**base_kwargs(x_search_image_understanding=True))
        x = next(t for t in opts.tools if t["type"] == "x_search")
        assert x["filters"]["enable_image_understanding"] is True

    def test_x_search_video_understanding_sets_filter(self):
        opts = build_request_options(**base_kwargs(x_search_video_understanding=True))
        x = next(t for t in opts.tools if t["type"] == "x_search")
        assert x["filters"]["enable_video_understanding"] is True

    def test_file_search_no_ids_raises(self):
        with pytest.raises(ValueError, match="no vector store IDs"):
            build_request_options(**base_kwargs(file_search=True, vector_store_ids_raw=""))

    def test_file_search_with_ids(self):
        opts = build_request_options(**base_kwargs(file_search=True, vector_store_ids_raw="vs1,vs2"))
        fs = next(t for t in opts.tools if t["type"] == "file_search")
        assert fs["vector_store_ids"] == ["vs1", "vs2"]
        assert fs["max_num_results"] == 10

    def test_file_search_custom_max_results(self):
        opts = build_request_options(**base_kwargs(file_search=True, vector_store_ids_raw="vs1", file_search_max_results_raw="5"))
        fs = next(t for t in opts.tools if t["type"] == "file_search")
        assert fs["max_num_results"] == 5

    def test_mcp_no_servers_raises(self):
        with pytest.raises(ValueError, match="no MCP servers"):
            build_request_options(**base_kwargs(mcp_enabled=True, mcp_servers=[]))

    def test_mcp_missing_server_url_raises(self):
        with pytest.raises(ValueError, match="missing server URL"):
            build_request_options(**base_kwargs(mcp_enabled=True, mcp_servers=[{"server_label": "no-url"}]))

    def test_mcp_valid_server(self):
        servers = [{"server_url": "http://mcp-server", "server_label": "myserver"}]
        opts = build_request_options(**base_kwargs(mcp_enabled=True, mcp_servers=servers))
        mcp = next(t for t in opts.tools if t["type"] == "mcp")
        assert mcp["server_url"] == "http://mcp-server"
        assert mcp["server_label"] == "myserver"

    def test_mcp_allowed_tool_names_from_list(self):
        servers = [{"server_url": "http://x", "allowed_tool_names": ["tool_a", "tool_b"]}]
        opts = build_request_options(**base_kwargs(mcp_enabled=True, mcp_servers=servers))
        mcp = next(t for t in opts.tools if t["type"] == "mcp")
        assert mcp["allowed_tool_names"] == ["tool_a", "tool_b"]

    def test_mcp_allowed_tool_names_from_string(self):
        servers = [{"server_url": "http://x", "allowed_tool_names": "tool_a, tool_b"}]
        opts = build_request_options(**base_kwargs(mcp_enabled=True, mcp_servers=servers))
        mcp = next(t for t in opts.tools if t["type"] == "mcp")
        assert mcp["allowed_tool_names"] == ["tool_a", "tool_b"]

    def test_image_generation_adds_function_tool(self):
        opts = build_request_options(**base_kwargs(image_generation=True))
        fn = next(t for t in opts.tools if t.get("name") == "generate_image")
        assert fn["type"] == "function"

    def test_image_count_too_high_raises(self):
        with pytest.raises(ValueError, match="between 1 and 10"):
            build_request_options(**base_kwargs(image_generation=True, image_count_raw="11"))

    def test_invalid_aspect_ratio_raises(self):
        with pytest.raises(ValueError, match="Invalid aspect ratio"):
            build_request_options(**base_kwargs(image_aspect_ratio_raw="7:3"))

    def test_valid_aspect_ratio(self):
        opts = build_request_options(**base_kwargs(image_aspect_ratio_raw="16:9"))
        assert opts.imagine_aspect_ratio == "16:9"

    def test_aspect_ratio_auto_becomes_none(self):
        opts = build_request_options(**base_kwargs(image_aspect_ratio_raw="auto"))
        assert opts.imagine_aspect_ratio is None

    def test_empty_aspect_ratio_becomes_none(self):
        opts = build_request_options(**base_kwargs(image_aspect_ratio_raw=""))
        assert opts.imagine_aspect_ratio is None

    def test_image_as_base64_sets_format(self):
        opts = build_request_options(**base_kwargs(image_as_base64=True))
        assert opts.imagine_response_format == "b64_json"

    def test_image_url_format_by_default(self):
        opts = build_request_options(**base_kwargs())
        assert opts.imagine_response_format == "url"
