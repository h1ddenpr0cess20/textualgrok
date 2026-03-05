"""Parse and process <grok:render> tags for inline image generation."""

from __future__ import annotations

import json
import re


def handle_render_tags(
    *,
    text: str,
    execute_image_tool: object,
    options: object,
    fallback_source_image_url: str | None,
) -> tuple[str, list[str], list[str]]:
    """Process <grok:render> tags in text, executing image generation for each.

    Args:
        text: The response text potentially containing render tags.
        execute_image_tool: Callable(arguments, options, fallback_source_image_url) -> dict.
        options: RequestOptions passed through to the image tool.
        fallback_source_image_url: Last known image URL for edit operations.

    Returns:
        Tuple of (cleaned_text, generated_urls, generated_b64).
    """
    if not text:
        return text, [], []

    pattern = re.compile(
        r"<grok:render\b(?P<attrs>[^>]*)>(?P<body>.*?)</grok:render>",
        re.IGNORECASE | re.DOTALL,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return text, [], []

    generated_urls: list[str] = []
    generated_b64: list[str] = []
    replacement_blocks: list[tuple[tuple[int, int], str]] = []
    working_source = fallback_source_image_url

    for match in matches:
        attrs = parse_tag_attributes(match.group("attrs") or "")
        render_type = attrs.get("type", "").strip().lower()
        if render_type and render_type != "generate_image":
            continue

        body = (match.group("body") or "").strip()
        parsed_body = parse_render_body(body)

        call_arguments: dict = {}
        for key in ("model", "response_format", "aspect_ratio", "source_image_url", "n"):
            value = attrs.get(key)
            if value:
                call_arguments[key] = value

        for key in ("model", "response_format", "aspect_ratio", "source_image_url", "n"):
            value = parsed_body.get(key)
            if value is not None and value != "":
                call_arguments[key] = value

        prompt = str(parsed_body.get("prompt", "")).strip() or attrs.get("prompt", "").strip()
        if not prompt:
            if body and not looks_like_structured_payload(body):
                prompt = body
        if not prompt:
            replacement_blocks.append((match.span(), ""))
            continue

        call_arguments["prompt"] = prompt

        tool_result = execute_image_tool(
            arguments=call_arguments,
            options=options,
            fallback_source_image_url=working_source,
        )
        urls = tool_result.get("images", [])
        b64_images = tool_result.get("images_b64", [])
        error_message = tool_result.get("error")
        if isinstance(urls, list):
            for url in urls:
                if isinstance(url, str) and url:
                    generated_urls.append(url)
                    if not working_source:
                        working_source = url
        if isinstance(b64_images, list):
            for b64_data in b64_images:
                if isinstance(b64_data, str) and b64_data:
                    generated_b64.append(b64_data)

        replacement_text = "Generated image:\n"
        if isinstance(error_message, str) and error_message.strip():
            replacement_text = "Image generation failed:\n"
            replacement_text += f"- {error_message.strip()}"
        elif urls:
            replacement_text += "\n".join(f"- {url}" for url in urls if isinstance(url, str))
        elif b64_images:
            replacement_text += "- base64 image returned"
        else:
            replacement_text += "- (no URL returned)"
        replacement_blocks.append((match.span(), replacement_text))

    if not replacement_blocks:
        return text, generated_urls, generated_b64

    rebuilt: list[str] = []
    cursor = 0
    for (start, end), replacement in replacement_blocks:
        rebuilt.append(text[cursor:start])
        rebuilt.append(replacement)
        cursor = end
    rebuilt.append(text[cursor:])
    cleaned_text = "".join(rebuilt).strip()
    return (cleaned_text or "(No text returned.)"), generated_urls, generated_b64


def parse_tag_attributes(raw_attrs: str) -> dict[str, str]:
    """Parse HTML-style key="value" or key='value' attributes."""
    attrs: dict[str, str] = {}
    for key, value in re.findall(r'([A-Za-z_:][A-Za-z0-9_:\-]*)\s*=\s*"([^"]*)"', raw_attrs):
        attrs[key] = value
    for key, value in re.findall(r"([A-Za-z_:][A-Za-z0-9_:\-]*)\s*=\s*'([^']*)'", raw_attrs):
        attrs[key] = value
    return attrs


def parse_render_body(body: str) -> dict:
    """Parse the body content of a <grok:render> tag into a dict."""
    cleaned = body.strip()
    if not cleaned:
        return {}

    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.splitlines()[1:-1]).strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]
    except json.JSONDecodeError:
        pass

    extracted: dict[str, str] = {}
    for key in ("prompt", "model", "response_format", "aspect_ratio", "source_image_url", "n"):
        match = re.search(rf"<{key}>\s*(.*?)\s*</{key}>", cleaned, flags=re.IGNORECASE | re.DOTALL)
        if match:
            extracted[key] = match.group(1).strip()
    if extracted:
        return extracted

    # Parse simple line-based payloads:
    # prompt: ...
    # n: 1
    # response_format: url
    line_pairs: dict[str, str] = {}
    key_value_pattern = re.compile(
        r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)\s*$",
        flags=re.MULTILINE,
    )
    for key, value in key_value_pattern.findall(cleaned):
        lowered = key.lower()
        if lowered in {"prompt", "model", "response_format", "aspect_ratio", "source_image_url", "n"}:
            line_pairs[lowered] = value.strip().strip('"').strip("'")
    if line_pairs:
        return line_pairs
    return extracted


def looks_like_structured_payload(body: str) -> bool:
    """Return True if the body looks like structured data rather than a plain prompt."""
    lowered = body.lower()
    if re.search(r"^\s*(prompt|model|response_format|aspect_ratio|source_image_url|n)\s*:", lowered, re.MULTILINE):
        return True
    structured_markers = (
        "response_format",
        "source_image_url",
        "aspect_ratio",
        "prompt:",
        "n:",
        "\"prompt\"",
        "'prompt'",
        "{",
        "}",
    )
    marker_count = sum(1 for marker in structured_markers if marker in lowered)
    return marker_count >= 2


def parse_function_arguments(arguments_raw: object) -> dict:
    """Parse function call arguments from string or dict form."""
    if isinstance(arguments_raw, dict):
        return arguments_raw

    if not isinstance(arguments_raw, str):
        return {}

    cleaned = arguments_raw.strip()
    if not cleaned:
        return {}

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    repaired = cleaned
    if repaired.startswith("{") and repaired.endswith("}"):
        repaired = repaired.replace("\n", " ")
        repaired = re.sub(r",\s*}", "}", repaired)
        try:
            parsed = json.loads(repaired)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    parsed_body = parse_render_body(cleaned)
    if isinstance(parsed_body, dict) and parsed_body:
        return parsed_body
    return {}
