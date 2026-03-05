"""Image generation API calls and payload construction for xAI."""

from __future__ import annotations

from textualgrok.options import RequestOptions
from textualgrok.response_parser import extract_image_b64, extract_image_urls, extract_first_revised_prompt


def build_image_payload(
    *,
    prompt: str,
    model: str,
    image_count: int,
    response_format: str,
    source_image_url: str | None,
    aspect_ratio: str | None,
) -> dict:
    """Build the JSON payload for /images/generations."""
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "n": image_count,
        "response_format": response_format,
    }
    if aspect_ratio:
        payload["aspect_ratio"] = aspect_ratio
    if source_image_url:
        payload["image_url"] = source_image_url
    return payload


def execute_generate_image_tool(
    *,
    arguments: dict,
    options: RequestOptions,
    fallback_source_image_url: str | None,
    post_json_fn: object,
) -> dict:
    """Execute a generate_image tool call, returning images and metadata.

    Args:
        arguments: Parsed function call arguments.
        options: Current request options with image defaults.
        fallback_source_image_url: Last image URL for edit operations.
        post_json_fn: Callable(endpoint, payload, label, *, timeout, retries) -> dict.

    Returns:
        Dict with keys: images, images_b64, revised_prompt, count, and optionally error.
    """
    prompt = str(arguments.get("prompt", "")).strip()
    if not prompt:
        return {"error": "prompt is required for generate_image"}

    model = str(arguments.get("model", "")).strip() or options.imagine_model

    n = options.imagine_count
    arg_n = arguments.get("n")
    if isinstance(arg_n, int):
        n = arg_n
    elif isinstance(arg_n, str) and arg_n.strip().isdigit():
        n = int(arg_n.strip())
    if n < 1:
        n = 1
    if n > 10:
        n = 10

    # Respect user's configured preference first. If URL mode is selected,
    # force URL output to ensure links/preview can be shown reliably.
    response_format = options.imagine_response_format
    arg_response_format = str(arguments.get("response_format", "")).strip()
    if response_format == "b64_json":
        if arg_response_format in {"url", "b64_json"}:
            response_format = arg_response_format
    else:
        response_format = "url"

    aspect_ratio = str(arguments.get("aspect_ratio", "")).strip() or options.imagine_aspect_ratio
    if isinstance(aspect_ratio, str) and aspect_ratio.lower() == "auto":
        aspect_ratio = None

    source_image_url = str(arguments.get("source_image_url", "")).strip() or options.imagine_source_image_url
    if not source_image_url and options.imagine_use_last_image:
        source_image_url = fallback_source_image_url

    payload = build_image_payload(
        prompt=prompt,
        model=model,
        image_count=n,
        response_format=response_format,
        source_image_url=source_image_url,
        aspect_ratio=aspect_ratio,
    )
    try:
        data = post_json_fn(
            "/images/generations",
            payload,
            "xAI image API",
            timeout=180.0,
            retries=1,
        )
    except RuntimeError as exc:
        return {
            "images": [],
            "images_b64": [],
            "revised_prompt": None,
            "count": 0,
            "error": str(exc),
        }
    image_urls = extract_image_urls(data)
    image_b64 = extract_image_b64(data)
    return {
        "images": image_urls,
        "images_b64": image_b64,
        "revised_prompt": extract_first_revised_prompt(data),
        "count": len(image_urls) + len(image_b64),
    }
