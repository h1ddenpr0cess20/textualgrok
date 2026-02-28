from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RequestOptions:
    tools: list[dict[str, Any]] = field(default_factory=list)
    imagine_enabled: bool = False
    imagine_model: str = "grok-imagine-image"
    imagine_count: int = 1
    imagine_response_format: str = "url"
    imagine_source_image_url: str | None = None
    imagine_use_last_image: bool = True
    imagine_aspect_ratio: str | None = None


def build_request_options(
    *,
    web_search: bool,
    x_search: bool,
    x_search_image_understanding: bool,
    x_search_video_understanding: bool,
    code_interpreter: bool,
    file_search: bool,
    vector_store_ids_raw: str,
    file_search_max_results_raw: str,
    image_generation: bool,
    image_model: str,
    image_count_raw: str,
    image_as_base64: bool,
    image_source_url_raw: str,
    image_use_last: bool,
    image_aspect_ratio_raw: str,
    mcp_enabled: bool,
    mcp_servers: list[dict[str, Any]],
) -> RequestOptions:
    tools: list[dict[str, Any]] = []

    if web_search:
        tools.append({"type": "web_search"})

    use_x_search = x_search or x_search_image_understanding or x_search_video_understanding
    if use_x_search:
        x_search_tool: dict[str, Any] = {"type": "x_search"}
        filters: dict[str, Any] = {}
        if x_search_image_understanding:
            filters["enable_image_understanding"] = True
        if x_search_video_understanding:
            filters["enable_video_understanding"] = True
        if filters:
            x_search_tool["filters"] = filters
        tools.append(x_search_tool)

    if code_interpreter:
        tools.append({"type": "code_interpreter"})

    if file_search:
        vector_store_ids = [item.strip() for item in vector_store_ids_raw.split(",") if item.strip()]
        if not vector_store_ids:
            raise ValueError("File search is enabled but no vector store IDs were provided.")

        max_results = _parse_positive_int(
            file_search_max_results_raw.strip() or "10",
            field_name="File search max results",
        )
        tools.append(
            {
                "type": "file_search",
                "vector_store_ids": vector_store_ids,
                "max_num_results": max_results,
            }
        )

    if mcp_enabled:
        if not mcp_servers:
            raise ValueError("MCP is enabled but no MCP servers were provided.")

        for index, raw_server in enumerate(mcp_servers, start=1):
            if not isinstance(raw_server, dict):
                raise ValueError(f"MCP server {index} is invalid.")

            server_url = str(raw_server.get("server_url", "")).strip()
            if not server_url:
                raise ValueError(f"MCP server {index} is missing server URL.")

            tool: dict[str, Any] = {
                "type": "mcp",
                "server_url": server_url,
            }

            server_label = str(raw_server.get("server_label", "")).strip()
            if server_label:
                tool["server_label"] = server_label

            server_description = str(raw_server.get("server_description", "")).strip()
            if server_description:
                tool["server_description"] = server_description

            authorization = str(raw_server.get("authorization", "")).strip()
            if authorization:
                tool["authorization"] = authorization

            allowed_tool_names: list[str] = []
            raw_allowed = raw_server.get("allowed_tool_names")
            if isinstance(raw_allowed, list):
                for item in raw_allowed:
                    name = str(item).strip()
                    if name:
                        allowed_tool_names.append(name)
            elif isinstance(raw_allowed, str):
                allowed_tool_names = [item.strip() for item in raw_allowed.split(",") if item.strip()]
            if allowed_tool_names:
                tool["allowed_tool_names"] = allowed_tool_names

            raw_headers = raw_server.get("extra_headers")
            if isinstance(raw_headers, dict):
                headers: dict[str, str] = {}
                for key, value in raw_headers.items():
                    header_key = str(key).strip()
                    header_value = str(value).strip()
                    if header_key and header_value:
                        headers[header_key] = header_value
                if headers:
                    tool["extra_headers"] = headers

            tools.append(tool)

    if image_generation:
        tools.append(
            {
                "type": "function",
                "name": "generate_image",
                "description": (
                    "Generate or edit an image using Grok Imagine when the user asks for visual output."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Detailed image prompt.",
                        },
                        "source_image_url": {
                            "type": "string",
                            "description": "Optional source image URL for edit/variation requests.",
                        },
                        "n": {
                            "type": "integer",
                            "description": "Number of images to generate (1-10).",
                        },
                        "response_format": {
                            "type": "string",
                            "enum": ["url", "b64_json"],
                            "description": "Preferred output format.",
                        },
                        "aspect_ratio": {
                            "type": "string",
                            "description": "Aspect ratio such as 1:1 or 16:9. Use auto for model default.",
                        },
                        "model": {
                            "type": "string",
                            "description": "Image model override.",
                        },
                    },
                    "required": ["prompt"],
                },
            }
        )

    image_count = 1
    if image_generation:
        image_count = _parse_positive_int(image_count_raw.strip() or "1", field_name="Image count")
        if image_count > 10:
            raise ValueError("Image count must be between 1 and 10.")

    response_format = "b64_json" if image_as_base64 else "url"
    source_url = image_source_url_raw.strip() or None
    aspect_ratio = image_aspect_ratio_raw.strip() or None
    if isinstance(aspect_ratio, str) and aspect_ratio.lower() == "auto":
        aspect_ratio = None
    if aspect_ratio:
        allowed_aspect_ratios = {
            "1:1",
            "16:9",
            "9:16",
            "4:3",
            "3:4",
            "3:2",
            "2:3",
            "2:1",
            "1:2",
            "19.5:9",
            "9:19.5",
            "20:9",
            "9:20",
        }
        if aspect_ratio not in allowed_aspect_ratios:
            raise ValueError("Invalid aspect ratio.")

    return RequestOptions(
        tools=tools,
        imagine_enabled=image_generation,
        imagine_model=image_model.strip() or "grok-imagine-image",
        imagine_count=image_count,
        imagine_response_format=response_format,
        imagine_source_image_url=source_url,
        imagine_use_last_image=image_use_last,
        imagine_aspect_ratio=aspect_ratio,
    )


def _parse_positive_int(value: str, *, field_name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be greater than 0.")
    return parsed
