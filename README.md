# Textual Grok

Terminal-first chatbot built with [Textual](https://textual.textualize.io/) and xAI's Responses API (`/v1/responses`) using direct `httpx` calls.

## What It Supports

- Chat with xAI models from a Textual UI.
- Live model refresh from xAI (`/v1/models`) inside Settings.
- Tool toggles: `web_search`, `x_search`, `code_interpreter`, `file_search`, and remote `mcp`.
- File and image attachments (single files, folders, or image URLs).
- Optional image generation/editing via Grok Imagine (`/v1/images/generations`).
- Saved UI settings in `.textual-grok-settings.json`.
- Chat transcript and generated image export to `exports/`.

## Requirements

- Python `3.10+`
- xAI API key

## Install

### Windows (PowerShell)

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

### macOS / Linux (bash/zsh)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

```dotenv
XAI_API_KEY=your_key_here
XAI_MODEL=grok-4-1-fast-non-reasoning
XAI_SYSTEM_PROMPT=You are a concise, helpful assistant.
XAI_IMAGE_MODEL=grok-imagine-image
```

## Run

```bash
python textualgrok.py
```

## Serve In Browser (Textual Serve)

```bash
python textualgrokserve.py
```

## Controls

- `Enter`: send prompt
- `Shift+Enter`: newline in prompt
- `Ctrl+S`: send prompt
- `Ctrl+T`: cycle app theme
- `Ctrl+Up` / `Ctrl+Down`: prompt history
- `Ctrl+P`: command palette (`Settings`, `Clear Chat`, `Save Chat`)
- `Ctrl+C`: quit

## Attachments

- Local files and folders are supported.
- URL attachments are supported for images only.
- Supported image types: `.jpg`, `.jpeg`, `.png`.
- Supported document/code types include common text, markdown, JSON/YAML/XML, CSV/TSV, PDF, Office docs, and many code extensions.

Local images are sent as `input_image` data URLs. Other files are sent as `input_file` data URLs with filename metadata.

## Settings

Settings are available from the command palette and saved to:

- `.textual-grok-settings.json`

Notes:

- Older installs may still use `.textualbot_settings.json` (legacy fallback).
- The saved `system_prompt` is user-configurable. If the current tone is not what you want, change `system_prompt` in your settings file or in `Settings -> Chat`.

Tabs include:

- `Chat`: model, system prompt, history mode
- `Tools`: web/x/code/file toggles
- `File Search`: vector store IDs and max results
- `Image`: image model/count/format/aspect ratio/edit source
- `MCP`: enable and configure remote MCP servers

## Exports

- `Save Chat` writes markdown transcripts to `exports/chat-YYYYMMDD-HHMMSS.md`.
- Saved images are written under `exports/images/`.

## Project Layout

- `textualgrok.py`: local terminal entrypoint
- `textualgrokserve.py`: browser serve entrypoint
- `textualgrok/chat_app.py`: main Textual app
- `textualgrok/settings_screen.py`: settings modal UI
- `textualgrok/options.py`: tool option validation/building
- `textualgrok/xai_client.py`: xAI HTTP client
- `textualgrok/attachments.py`: attachment file-type support logic
- `textualgrok/conversation.py`: conversation state/history assembly
- `textualgrok/config.py`: environment config loader
