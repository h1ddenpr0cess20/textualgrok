# Textual xAI Chatbot (Responses API)

Terminal chatbot built with Textual and xAI's Responses API (`POST https://api.x.ai/v1/responses`) using direct HTTP calls.
Includes a redesigned settings dialog with tabbed sections and live model loading.

## Requirements

- Python 3.10+
- xAI API key

## Setup

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create your env file:

```powershell
Copy-Item .env.example .env
```

Then edit `.env` and set your key:

```dotenv
XAI_API_KEY=your_key_here
XAI_MODEL=grok-4-1-fast-non-reasoning
XAI_SYSTEM_PROMPT=You are a concise, helpful assistant.
XAI_IMAGE_MODEL=grok-imagine-image
```

## Run

```powershell
python textualbot.py
```

## Project layout

- `textualbot.py`: entrypoint wiring config + UI app.
- `textualbot_app/config.py`: environment configuration loading.
- `textualbot_app/xai_client.py`: xAI Responses API transport layer.
- `textualbot_app/conversation.py`: conversation state and request assembly.
- `textualbot_app/options.py`: feature toggle validation and request option builder.
- `textualbot_app/app.py`: Textual UI and event handling.
- `textualbot_app/models.py`: shared data types.

## Notes

- `Settings` opens a dedicated modal with separate tabs for:
- Chat (model dropdown + system prompt + history mode)
- Tools (web/x/code/file toggles)
- File Search (vector stores + max results)
- Image (Grok Imagine tool, edit source image, aspect ratio, format, model, count)
- MCP (enable MCP + add/remove remote MCP server configs)
- Chat model and Grok Imagine model dropdowns are populated from xAI at runtime via `GET /v1/models` (Refresh Models).
- `Clear Chat` clears both displayed chat log and stored conversation history.
- The app keeps conversation context in memory (when history mode is enabled) and sends it on each request.
- Grok Imagine is executed as a client-side function tool in the Responses tool loop (`function_call` -> `function_call_output`), not as a separate heuristic pass.
- Generated images are shown in the app preview panel (requires `textual-image` support in the terminal).
- If image format is `url`, the link is shown in chat output and in the preview panel label. If image format is `b64_json`, the app decodes and previews the image locally and labels it as embedded base64 output.
- Tools currently support `web_search`, `x_search`, `code_interpreter`, `file_search`, and remote `mcp` servers.
- No `openai` Python package is used.
