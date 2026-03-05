"""Extract text, images, and citations from xAI API response payloads."""

from __future__ import annotations

import json


def extract_text(data: dict) -> str:
    """Extract the assistant's text from an API response."""
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    chunks: list[str] = []

    output_items = data.get("output")
    if isinstance(output_items, list):
        for item in output_items:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("text"), str) and item.get("text", "").strip():
                chunks.append(item["text"].strip())
            if item.get("type") == "output_text":
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                chunks.append(content.strip())
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") in {"output_text", "text"}:
                        text = part.get("text")
                        if isinstance(text, str) and text.strip():
                            chunks.append(text.strip())
                        elif isinstance(text, dict):
                            value = text.get("value")
                            if isinstance(value, str) and value.strip():
                                chunks.append(value.strip())
                    value = part.get("value")
                    if isinstance(value, str) and value.strip():
                        chunks.append(value.strip())

    choices = data.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                chunks.append(content.strip())
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    text = part.get("text")
                    if isinstance(text, str) and text.strip():
                        chunks.append(text.strip())
                    elif isinstance(text, dict):
                        value = text.get("value")
                        if isinstance(value, str) and value.strip():
                            chunks.append(value.strip())
                    value = part.get("value")
                    if isinstance(value, str) and value.strip():
                        chunks.append(value.strip())

    if chunks:
        return "\n".join(chunks)
    return "(No text returned.)"


def extract_function_calls(data: dict) -> list[dict]:
    """Extract deduplicated function/tool calls from an API response."""
    calls: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    def normalize_call_id(value: object) -> str | None:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    def append_call(call_id: object, name: object, arguments: object) -> None:
        normalized_call_id = normalize_call_id(call_id)
        normalized_name = name.strip() if isinstance(name, str) else ""
        if not normalized_call_id or not normalized_name:
            return

        if isinstance(arguments, str):
            signature = arguments
        elif isinstance(arguments, dict):
            try:
                signature = json.dumps(arguments, sort_keys=True)
            except TypeError:
                signature = str(arguments)
        else:
            signature = ""

        key = (normalized_call_id, normalized_name, signature)
        if key in seen:
            return
        seen.add(key)
        calls.append(
            {
                "call_id": normalized_call_id,
                "name": normalized_name,
                "arguments": arguments,
            }
        )

    def scan(node: object, inherited_call_id: str | None = None) -> None:
        if isinstance(node, list):
            for item in node:
                scan(item, inherited_call_id)
            return

        if not isinstance(node, dict):
            return

        local_call_id = (
            normalize_call_id(node.get("call_id"))
            or normalize_call_id(node.get("tool_call_id"))
            or normalize_call_id(node.get("id"))
            or inherited_call_id
        )
        node_type = str(node.get("type", "")).strip().lower()
        node_name = node.get("name")
        node_arguments = node.get("arguments")

        if node_type in {"function_call", "tool_call"}:
            append_call(local_call_id, node_name, node_arguments)

        if isinstance(node.get("function_call"), dict):
            nested = node["function_call"]
            append_call(
                nested.get("call_id") or nested.get("id") or local_call_id,
                nested.get("name") or node_name,
                nested.get("arguments") if "arguments" in nested else node_arguments,
            )

        if isinstance(node.get("function"), dict):
            nested = node["function"]
            append_call(
                nested.get("call_id") or nested.get("id") or local_call_id,
                nested.get("name") or node_name,
                nested.get("arguments") if "arguments" in nested else node_arguments,
            )

        tool_calls = node.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function_block = tool_call.get("function")
                if isinstance(function_block, dict):
                    append_call(
                        tool_call.get("id") or tool_call.get("call_id") or local_call_id,
                        function_block.get("name"),
                        function_block.get("arguments"),
                    )
                else:
                    append_call(
                        tool_call.get("id") or tool_call.get("call_id") or local_call_id,
                        tool_call.get("name"),
                        tool_call.get("arguments"),
                    )

        for value in node.values():
            if isinstance(value, (dict, list)):
                scan(value, local_call_id)

    scan(data)
    return calls


def extract_image_entries(data: dict) -> list[dict[str, str]]:
    """Extract unique image entries (URL or b64) from an API response."""
    entries: list[dict[str, str]] = []

    def append_entry(node: object) -> None:
        if isinstance(node, list):
            for item in node:
                append_entry(item)
            return

        if not isinstance(node, dict):
            return

        url = node.get("url")
        if not isinstance(url, str):
            url = node.get("image_url")
        b64_data = node.get("b64_json")
        revised_prompt = node.get("revised_prompt")

        if isinstance(url, str) and url:
            entry: dict[str, str] = {"url": url}
            if isinstance(revised_prompt, str) and revised_prompt:
                entry["revised_prompt"] = revised_prompt
            entries.append(entry)
            return

        if isinstance(b64_data, str) and b64_data:
            entry = {"b64_json": b64_data}
            if isinstance(revised_prompt, str) and revised_prompt:
                entry["revised_prompt"] = revised_prompt
            entries.append(entry)

        nested_images = node.get("images")
        if isinstance(nested_images, list):
            for image_item in nested_images:
                append_entry(image_item)
        elif isinstance(nested_images, dict):
            append_entry(nested_images)

        result_block = node.get("result")
        if isinstance(result_block, (dict, list)):
            append_entry(result_block)

        output_block = node.get("output")
        if isinstance(output_block, (dict, list)):
            append_entry(output_block)

        data_block = node.get("data")
        if isinstance(data_block, dict):
            append_entry(data_block)
        elif isinstance(data_block, list):
            append_entry(data_block)

        image_block = node.get("image")
        if isinstance(image_block, dict):
            append_entry(image_block)
        elif isinstance(image_block, str) and image_block:
            if image_block.startswith("http://") or image_block.startswith("https://"):
                entries.append({"url": image_block})
            else:
                entries.append({"b64_json": image_block})

    image_data = data.get("data")
    if isinstance(image_data, list):
        for item in image_data:
            append_entry(item)
    elif isinstance(image_data, dict):
        append_entry(image_data)

    append_entry(data)

    unique: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        if "url" in entry:
            key = ("url", entry["url"])
        elif "b64_json" in entry:
            key = ("b64_json", entry["b64_json"])
        else:
            continue
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    return unique


def extract_image_urls(data: dict) -> list[str]:
    """Extract image URLs from an API response."""
    urls: list[str] = []
    for item in extract_image_entries(data):
        url = item.get("url")
        if isinstance(url, str) and url:
            urls.append(url)
    return urls


def extract_image_b64(data: dict) -> list[str]:
    """Extract base64-encoded image data from an API response."""
    images_b64: list[str] = []
    for item in extract_image_entries(data):
        b64_data = item.get("b64_json")
        if isinstance(b64_data, str) and b64_data:
            images_b64.append(b64_data)
    return images_b64


def extract_first_revised_prompt(data: dict) -> str | None:
    """Extract the first revised prompt from image generation data."""
    for item in extract_image_entries(data):
        revised_prompt = item.get("revised_prompt")
        if isinstance(revised_prompt, str) and revised_prompt:
            return revised_prompt
    return None


def extract_image_text(data: dict) -> str:
    """Build a human-readable summary of image generation results."""
    image_entries = extract_image_entries(data)
    if not image_entries:
        return "(No images returned.)"

    lines: list[str] = []
    for index, item in enumerate(image_entries, start=1):
        url = item.get("url")
        b64_data = item.get("b64_json")
        revised_prompt = item.get("revised_prompt")
        if isinstance(url, str) and url:
            lines.append(f"Image {index}: {url}")
        elif isinstance(b64_data, str):
            lines.append(f"Image {index}: base64 data returned (length={len(b64_data)})")
        else:
            lines.append(f"Image {index}: generated")
        if isinstance(revised_prompt, str) and revised_prompt:
            lines.append(f"Revised prompt {index}: {revised_prompt}")

    return "\n".join(lines) if lines else "(No images returned.)"


def extract_citations(data: dict) -> list[str]:
    """Extract unique citation URLs from an API response."""
    urls: list[str] = []

    citations = data.get("citations")
    if isinstance(citations, list):
        for item in citations:
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            if isinstance(url, str) and url:
                urls.append(url)

    response_block = data.get("response")
    if isinstance(response_block, dict):
        nested = response_block.get("citations")
        if isinstance(nested, list):
            for item in nested:
                if not isinstance(item, dict):
                    continue
                url = item.get("url")
                if isinstance(url, str) and url:
                    urls.append(url)

    unique: list[str] = []
    for url in urls:
        if url not in unique:
            unique.append(url)
    return unique


def collect_strings(node: object) -> list[str]:
    """Recursively collect all non-empty strings from a nested data structure."""
    found: list[str] = []

    if isinstance(node, str):
        if node.strip():
            found.append(node)
        return found

    if isinstance(node, dict):
        for value in node.values():
            found.extend(collect_strings(value))
        return found

    if isinstance(node, list):
        for value in node:
            found.extend(collect_strings(value))
        return found

    return found


def find_render_text_in_payload(data: dict) -> str | None:
    """Search the payload for any string containing a <grok:render> tag."""
    strings = collect_strings(data)
    for value in strings:
        if "<grok:render" in value.lower():
            return value
    return None
