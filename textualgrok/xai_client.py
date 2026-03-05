"""HTTP client for xAI's Responses API."""

from __future__ import annotations

import json
import time

import httpx

from textualgrok.image_api import build_image_payload, execute_generate_image_tool
from textualgrok.models import ChatMessage, ChatResult
from textualgrok.options import RequestOptions
from textualgrok.render_tags import handle_render_tags, parse_function_arguments
from textualgrok.response_parser import (
    extract_citations,
    extract_function_calls,
    extract_image_b64,
    extract_image_text,
    extract_image_urls,
    extract_text,
    find_render_text_in_payload,
)


class XAIResponsesClient:
    def __init__(self, api_key: str, model: str) -> None:
        self.model = model
        self._http = httpx.Client(
            base_url="https://api.x.ai/v1",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    def close(self) -> None:
        self._http.close()

    def fetch_bytes(self, url: str) -> bytes:
        try:
            response = self._http.get(url)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(self._build_api_error(exc)) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"Network error fetching media: {exc}") from exc
        return response.content

    def list_models(self) -> list[str]:
        try:
            response = self._http.get("/models")
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(self._build_api_error(exc)) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"Network error calling xAI API: {exc}") from exc

        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("Unexpected xAI models API response format.")

        models_block = data.get("data")
        if not isinstance(models_block, list):
            raise RuntimeError("xAI models API response did not include a data list.")

        models: list[str] = []
        for item in models_block:
            if not isinstance(item, dict):
                continue
            model_id = item.get("id")
            if isinstance(model_id, str) and model_id:
                models.append(model_id)

        if not models:
            raise RuntimeError("No models returned by xAI API.")
        return sorted(set(models))

    def ask(
        self,
        messages: list[ChatMessage],
        options: RequestOptions,
        *,
        last_image_url: str | None = None,
    ) -> ChatResult:
        payload = {"model": self.model, "input": messages}
        if options.tools:
            payload["tools"] = options.tools
        data = self._post_json("/responses", payload, "xAI API")

        aggregated_image_urls: list[str] = []
        aggregated_image_b64: list[str] = []
        tool_errors: list[str] = []
        attempted_image_tool = False
        working_last_image_url = last_image_url
        max_tool_loops = 8

        for _ in range(max_tool_loops):
            function_calls = extract_function_calls(data)
            if not function_calls:
                break

            tool_outputs: list[dict[str, str]] = []
            for call in function_calls:
                call_id = call.get("call_id")
                if not isinstance(call_id, str) or not call_id:
                    continue

                name = call.get("name")
                arguments_raw = call.get("arguments")
                arguments = parse_function_arguments(arguments_raw)

                if name == "generate_image":
                    attempted_image_tool = True
                    tool_result = execute_generate_image_tool(
                        arguments=arguments,
                        options=options,
                        fallback_source_image_url=working_last_image_url,
                        post_json_fn=self._post_json,
                    )
                    generated_urls = tool_result.get("images", [])
                    generated_b64 = tool_result.get("images_b64", [])
                    if isinstance(generated_urls, list):
                        for url in generated_urls:
                            if isinstance(url, str) and url:
                                aggregated_image_urls.append(url)
                                if not working_last_image_url:
                                    working_last_image_url = url
                    if isinstance(generated_b64, list):
                        for b64_data in generated_b64:
                            if isinstance(b64_data, str) and b64_data:
                                aggregated_image_b64.append(b64_data)
                    error_message = tool_result.get("error")
                    if isinstance(error_message, str) and error_message.strip():
                        tool_errors.append(error_message.strip())
                    elif not generated_urls and not generated_b64:
                        tool_errors.append("Image tool returned no image data.")
                    output_text = json.dumps(tool_result)
                else:
                    output_text = json.dumps({"error": f"Unsupported tool: {name}"})

                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": output_text,
                    }
                )

            if not tool_outputs:
                break

            continuation_payload = {
                "model": self.model,
                "input": tool_outputs,
                "previous_response_id": data.get("id"),
            }
            if options.tools:
                continuation_payload["tools"] = options.tools
            data = self._post_json("/responses", continuation_payload, "xAI API")

        response_id = data.get("id") if isinstance(data.get("id"), str) else None
        text = extract_text(data)
        if not text or text == "(No text returned.)":
            render_fallback = find_render_text_in_payload(data)
            if render_fallback:
                text = render_fallback

        text, tag_image_urls, tag_image_b64 = handle_render_tags(
            text=text,
            execute_image_tool=lambda **kwargs: execute_generate_image_tool(
                **kwargs, post_json_fn=self._post_json
            ),
            options=options,
            fallback_source_image_url=working_last_image_url,
        )
        for url in tag_image_urls:
            aggregated_image_urls.append(url)
            if not working_last_image_url:
                working_last_image_url = url
        for b64_data in tag_image_b64:
            aggregated_image_b64.append(b64_data)

        if aggregated_image_urls:
            unique_urls = []
            for url in aggregated_image_urls:
                if url not in unique_urls:
                    unique_urls.append(url)
            aggregated_image_urls = unique_urls

            if not text or text == "(No text returned.)":
                text = "Generated image(s):\n" + "\n".join(aggregated_image_urls)
            else:
                missing_urls = [url for url in aggregated_image_urls if url not in text]
                if missing_urls:
                    text = f"{text}\n\nImage output:\n" + "\n".join(f"- {url}" for url in missing_urls)

        if aggregated_image_b64:
            unique_b64 = []
            for b64_data in aggregated_image_b64:
                if b64_data not in unique_b64:
                    unique_b64.append(b64_data)
            aggregated_image_b64 = unique_b64
            if not text or text == "(No text returned.)":
                text = f"Generated {len(aggregated_image_b64)} image(s) as base64 data."
            elif "base64" not in text.lower():
                text = f"{text}\n\nImage output:\n- base64 image data returned"

        if (not text or text == "(No text returned.)") and aggregated_image_urls:
            text = "Generated image(s):\n" + "\n".join(aggregated_image_urls)
        if (not text or text == "(No text returned.)") and attempted_image_tool:
            if tool_errors:
                unique_errors: list[str] = []
                for error in tool_errors:
                    if error not in unique_errors:
                        unique_errors.append(error)
                text = "Image generation failed or produced no content:\n" + "\n".join(
                    f"- {error}" for error in unique_errors
                )
            else:
                text = "Image generation returned no assistant text or image output."
        citations = extract_citations(data)
        if citations:
            text = f"{text}\n\nCitations:\n" + "\n".join(f"- {url}" for url in citations)
        return ChatResult(
            text=text,
            response_id=response_id,
            image_urls=aggregated_image_urls,
            image_b64=aggregated_image_b64,
        )

    def generate_images(self, prompt: str, options: RequestOptions, source_image_url: str | None = None) -> ChatResult:
        payload = build_image_payload(
            prompt=prompt,
            model=options.imagine_model,
            image_count=options.imagine_count,
            response_format=options.imagine_response_format,
            source_image_url=source_image_url,
            aspect_ratio=options.imagine_aspect_ratio,
        )
        data = self._post_json(
            "/images/generations",
            payload,
            "xAI image API",
            timeout=180.0,
            retries=1,
        )
        return ChatResult(
            text=extract_image_text(data),
            response_id=data.get("id") if isinstance(data.get("id"), str) else None,
            image_urls=extract_image_urls(data),
            image_b64=extract_image_b64(data),
        )

    def _post_json(
        self,
        endpoint: str,
        payload: dict,
        label: str,
        *,
        timeout: float | None = None,
        retries: int = 0,
    ) -> dict:
        last_request_error: httpx.RequestError | None = None
        for attempt in range(retries + 1):
            try:
                response = self._http.post(endpoint, json=payload, timeout=timeout)
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise RuntimeError(f"Unexpected {label} response format.")
                return data
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                retryable_status = {408, 429, 500, 502, 503, 504}
                if attempt < retries and status in retryable_status:
                    time.sleep(min(2.0, 0.5 * (attempt + 1)))
                    continue
                raise RuntimeError(self._build_api_error(exc)) from exc
            except httpx.RequestError as exc:
                last_request_error = exc
                if attempt < retries:
                    time.sleep(min(2.0, 0.5 * (attempt + 1)))
                    continue
                raise RuntimeError(f"Network error calling {label}: {exc}") from exc

        if last_request_error is not None:
            raise RuntimeError(f"Network error calling {label}: {last_request_error}")
        raise RuntimeError(f"Unknown error calling {label}.")

    @staticmethod
    def _build_api_error(exc: httpx.HTTPStatusError) -> str:
        detail = ""
        try:
            body = exc.response.json()
            if isinstance(body, dict):
                error_block = body.get("error", {})
                if isinstance(error_block, dict):
                    detail = str(error_block.get("message", ""))
                if not detail and "message" in body:
                    detail = str(body["message"])
        except ValueError:
            detail = exc.response.text

        if not detail:
            detail = exc.response.text or "Unknown API error"
        return f"xAI API error {exc.response.status_code}: {detail}"
