import httpx
import json
import re

from textualbot_app.models import ChatMessage, ChatResult
from textualbot_app.options import RequestOptions


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
        working_last_image_url = last_image_url
        max_tool_loops = 8

        for _ in range(max_tool_loops):
            function_calls = self._extract_function_calls(data)
            if not function_calls:
                break

            tool_outputs: list[dict[str, str]] = []
            for call in function_calls:
                call_id = call.get("call_id")
                if not isinstance(call_id, str) or not call_id:
                    continue

                name = call.get("name")
                arguments_raw = call.get("arguments")
                arguments: dict = {}
                if isinstance(arguments_raw, str) and arguments_raw.strip():
                    try:
                        parsed = json.loads(arguments_raw)
                        if isinstance(parsed, dict):
                            arguments = parsed
                    except json.JSONDecodeError:
                        arguments = {}
                elif isinstance(arguments_raw, dict):
                    arguments = arguments_raw

                if name == "generate_image":
                    tool_result = self._execute_generate_image_tool(
                        arguments=arguments,
                        options=options,
                        fallback_source_image_url=working_last_image_url,
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
        text = self._extract_text(data)
        if not text or text == "(No text returned.)":
            render_fallback = self._find_render_text_in_payload(data)
            if render_fallback:
                text = render_fallback

        text, tag_image_urls, tag_image_b64 = self._handle_render_tags_in_text(
            text=text,
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
        citations = self._extract_citations(data)
        if citations:
            text = f"{text}\n\nCitations:\n" + "\n".join(f"- {url}" for url in citations)
        return ChatResult(
            text=text,
            response_id=response_id,
            image_urls=aggregated_image_urls,
            image_b64=aggregated_image_b64,
        )

    def generate_images(self, prompt: str, options: RequestOptions, source_image_url: str | None = None) -> ChatResult:
        payload = self._build_image_payload(
            prompt=prompt,
            model=options.imagine_model,
            image_count=options.imagine_count,
            response_format=options.imagine_response_format,
            source_image_url=source_image_url,
            aspect_ratio=options.imagine_aspect_ratio,
        )
        data = self._post_json("/images/generations", payload, "xAI image API")
        return ChatResult(
            text=self._extract_image_text(data),
            response_id=data.get("id") if isinstance(data.get("id"), str) else None,
            image_urls=self._extract_image_urls(data),
            image_b64=self._extract_image_b64(data),
        )

    @staticmethod
    def _build_image_payload(
        *,
        prompt: str,
        model: str,
        image_count: int,
        response_format: str,
        source_image_url: str | None,
        aspect_ratio: str | None,
    ) -> dict:
        payload = {
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

    def _execute_generate_image_tool(
        self,
        *,
        arguments: dict,
        options: RequestOptions,
        fallback_source_image_url: str | None,
    ) -> dict:
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

        source_image_url = str(arguments.get("source_image_url", "")).strip() or options.imagine_source_image_url
        if not source_image_url and options.imagine_use_last_image:
            source_image_url = fallback_source_image_url

        payload = self._build_image_payload(
            prompt=prompt,
            model=model,
            image_count=n,
            response_format=response_format,
            source_image_url=source_image_url,
            aspect_ratio=aspect_ratio,
        )
        data = self._post_json("/images/generations", payload, "xAI image API")
        image_urls = self._extract_image_urls(data)
        image_b64 = self._extract_image_b64(data)
        return {
            "images": image_urls,
            "images_b64": image_b64,
            "revised_prompt": self._extract_first_revised_prompt(data),
            "count": len(image_urls) + len(image_b64),
        }

    def _post_json(self, endpoint: str, payload: dict, label: str) -> dict:
        try:
            response = self._http.post(endpoint, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(self._build_api_error(exc)) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"Network error calling {label}: {exc}") from exc

        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected {label} response format.")
        return data

    @staticmethod
    def _extract_function_calls(data: dict) -> list[dict]:
        output = data.get("output")
        if not isinstance(output, list):
            return []

        calls: list[dict] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "function_call":
                continue
            call = {
                "call_id": item.get("call_id"),
                "name": item.get("name"),
                "arguments": item.get("arguments"),
            }
            if not call["name"] and isinstance(item.get("function_call"), dict):
                nested = item["function_call"]
                call["name"] = nested.get("name")
                call["arguments"] = nested.get("arguments")
            if not call["call_id"] and isinstance(item.get("id"), str):
                call["call_id"] = item["id"]
            calls.append(call)
        return calls

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

    @staticmethod
    def _extract_text(data: dict) -> str:
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

    @staticmethod
    def _extract_image_text(data: dict) -> str:
        image_entries = XAIResponsesClient._extract_image_entries(data)
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

    @staticmethod
    def _extract_image_urls(data: dict) -> list[str]:
        urls: list[str] = []
        for item in XAIResponsesClient._extract_image_entries(data):
            url = item.get("url")
            if isinstance(url, str) and url:
                urls.append(url)
        return urls

    @staticmethod
    def _extract_image_b64(data: dict) -> list[str]:
        images_b64: list[str] = []
        for item in XAIResponsesClient._extract_image_entries(data):
            b64_data = item.get("b64_json")
            if isinstance(b64_data, str) and b64_data:
                images_b64.append(b64_data)
        return images_b64

    @staticmethod
    def _extract_first_revised_prompt(data: dict) -> str | None:
        for item in XAIResponsesClient._extract_image_entries(data):
            revised_prompt = item.get("revised_prompt")
            if isinstance(revised_prompt, str) and revised_prompt:
                return revised_prompt
        return None

    @staticmethod
    def _extract_image_entries(data: dict) -> list[dict[str, str]]:
        entries: list[dict[str, str]] = []

        def append_entry(node: object) -> None:
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

        image_data = data.get("data")
        if isinstance(image_data, list):
            for item in image_data:
                append_entry(item)
        elif isinstance(image_data, dict):
            append_entry(image_data)

        append_entry(data)

        top_level_image = data.get("image")
        if isinstance(top_level_image, dict):
            append_entry(top_level_image)
        elif isinstance(top_level_image, str) and top_level_image:
            entries.append({"b64_json": top_level_image})

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

    @staticmethod
    def _extract_citations(data: dict) -> list[str]:
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

    def _handle_render_tags_in_text(
        self,
        *,
        text: str,
        options: RequestOptions,
        fallback_source_image_url: str | None,
    ) -> tuple[str, list[str], list[str]]:
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
            attrs = self._parse_tag_attributes(match.group("attrs") or "")
            render_type = attrs.get("type", "").strip().lower()
            if render_type and render_type != "generate_image":
                continue

            body = (match.group("body") or "").strip()
            parsed_body = self._parse_render_body(body)

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
                if body and not self._looks_like_structured_payload(body):
                    prompt = body
            if not prompt:
                replacement_blocks.append((match.span(), ""))
                continue

            call_arguments["prompt"] = prompt

            tool_result = self._execute_generate_image_tool(
                arguments=call_arguments,
                options=options,
                fallback_source_image_url=working_source,
            )
            urls = tool_result.get("images", [])
            b64_images = tool_result.get("images_b64", [])
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
            if urls:
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

    @staticmethod
    def _parse_tag_attributes(raw_attrs: str) -> dict[str, str]:
        attrs: dict[str, str] = {}
        for key, value in re.findall(r'([A-Za-z_:][A-Za-z0-9_:\-]*)\s*=\s*"([^"]*)"', raw_attrs):
            attrs[key] = value
        for key, value in re.findall(r"([A-Za-z_:][A-Za-z0-9_:\-]*)\s*=\s*'([^']*)'", raw_attrs):
            attrs[key] = value
        return attrs

    @staticmethod
    def _parse_render_body(body: str) -> dict:
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

    @staticmethod
    def _looks_like_structured_payload(body: str) -> bool:
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

    @staticmethod
    def _find_render_text_in_payload(data: dict) -> str | None:
        strings = XAIResponsesClient._collect_strings(data)
        for value in strings:
            if "<grok:render" in value.lower():
                return value
        return None

    @staticmethod
    def _collect_strings(node: object) -> list[str]:
        found: list[str] = []

        if isinstance(node, str):
            if node.strip():
                found.append(node)
            return found

        if isinstance(node, dict):
            for value in node.values():
                found.extend(XAIResponsesClient._collect_strings(value))
            return found

        if isinstance(node, list):
            for value in node:
                found.extend(XAIResponsesClient._collect_strings(value))
            return found

        return found
