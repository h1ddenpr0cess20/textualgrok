# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

TextualGrok is a terminal-first chatbot UI built with [Textual](https://textual.textualize.io/) that talks to [xAI's Responses API](https://docs.x.ai/api). It supports tool use (web search, code interpreter, file search, MCP), file/image attachments, and image generation via Grok Imagine.

## Setup & Running

```bash
# Install dependencies
pip install -r requirements.txt

# Run (requires XAI_API_KEY in .env or environment)
python textualgrok.py

# Run in browser via textual-serve
python textualgrokserve.py
```

There are no build steps and no linter configuration. A test suite for the pure-logic modules exists (run with `pytest`).

## Architecture

### Entry Points
- `textualgrok.py` — loads `AppConfig`, launches `ChatApp`
- `textualgrokserve.py` — wraps same app for browser via `textual-serve`

### Package: `textualgrok/`

| File | Role |
|---|---|
| `app.py` | Backward-compatible `ChatApp` re-export (one-liner shim) |
| `chat_app.py` | Main Textual `App` subclass; all UI logic, workers, and keybindings |
| `xai_client.py` | `XAIResponsesClient` — direct `httpx` calls to `https://api.x.ai/v1`; tool execution loop (up to 8 iterations) |
| `config.py` | `AppConfig` dataclass loaded from `.env` via `python-dotenv` |
| `ui_types.py` | Pure data types: `UISettings` (30+ fields), `PendingAttachment`, `SessionImageItem`, `BrowseEntry` |
| `models.py` | API response types: `ChatMessage` (TypedDict), `ChatResult` (dataclass) |
| `conversation.py` | `ConversationState` — manages history, builds request message list, commits turns |
| `options.py` | `build_request_options()` — validates and assembles the `tools` list for each API call |
| `response_parser.py` | Extracts text, image URLs/b64, citations, and function calls from raw API response payloads |
| `render_tags.py` | Parses `<grok:render>` tags in response text and dispatches inline image generation calls |
| `image_api.py` | Builds `/images/generations` payloads and executes `generate_image` tool calls |
| `image_manager.py` | Materializes session images from URLs and base64 data into temp files; rebuilds the image grid UI |
| `attachment_handler.py` | Builds `PendingAttachment`s from file paths, folders, and image URLs; updates the attachment bar UI |
| `chat_log.py` | Writes entries to the `RichLog` widget; renders markdown with code highlighting; selects Pygments theme |
| `settings_persistence.py` | Loads and saves `UISettings` to `.textual-grok-settings.json`; handles legacy filename fallback |
| `settings_screen.py` | 5-tab settings modal (Chat, Tools, File Search, Image, MCP) |
| `attachment_screens.py` | File browser modal + URL/path input modal |
| `attachments.py` | File type detection helpers |
| `image_gallery_screen.py` | Full-screen image viewer with navigation |
| `command_provider.py` | Custom Textual command palette provider |
| `app.tcss` | Textual CSS for all screens |

### Data Flow

1. User types a message → `ChatApp` uses `attachment_handler` to assemble `PendingAttachment`s + prompt
2. `ConversationState.build_request()` builds the message list
3. `build_request_options()` validates and builds the tools array
4. `XAIResponsesClient.ask()` posts to xAI, loops over tool calls until a final response
5. `response_parser` extracts text, images, citations, and function calls from the raw response
6. `render_tags` processes any `<grok:render>` tags, dispatching to `image_api` for inline generation
7. `ChatResult` (text + image URLs/b64) returned to `ChatApp` worker
8. `chat_log` renders the result in `RichLog`; `image_manager` materializes images into the grid

### Settings Persistence

Settings saved to `.textual-grok-settings.json` in the **current working directory** (not `~/.config/`). The app also checks the legacy filename `.textualbot_settings.json` on first load.

### Key Keybindings (in `ChatApp`)
- `Ctrl+P` — command palette
- `Ctrl+T` — cycle theme
- `Ctrl+Up/Down` — prompt history navigation
- Export functions in command palette: save chat transcript, export images to `exports/`
