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

There are no tests, no build steps, and no linter configuration.

## Architecture

### Entry Points
- `textualgrok.py` — loads `AppConfig`, launches `ChatApp`
- `textualgrokserve.py` — wraps same app for browser via `textual-serve`

### Package: `textualgrok/`

| File | Role |
|---|---|
| `chat_app.py` | Main Textual `App` subclass (~900 lines); all UI logic, workers, keybindings, settings persistence |
| `xai_client.py` | `XAIResponsesClient` — direct `httpx` calls to `https://api.x.ai/v1`; tool execution loop (up to 8 iterations); parses `<grok:render>` tags for inline image generation |
| `config.py` | `AppConfig` dataclass loaded from `.env` via `python-dotenv` |
| `ui_types.py` | Pure data types: `UISettings` (30+ fields), `PendingAttachment`, `SessionImageItem`, `BrowseEntry` |
| `models.py` | API response types: `ChatMessage` (TypedDict), `ChatResult` (dataclass) |
| `conversation.py` | `ConversationState` — manages history, builds request message list, commits turns |
| `options.py` | `build_request_options()` — validates and assembles the `tools` list for each API call |
| `settings_screen.py` | 5-tab settings modal (Chat, Tools, File Search, Image, MCP) |
| `attachment_screens.py` | File browser modal + URL/path input modal |
| `attachments.py` | File type detection helpers |
| `image_gallery_screen.py` | Full-screen image viewer with navigation |
| `command_provider.py` | Custom Textual command palette provider |
| `app.tcss` | Textual CSS for all screens |

### Data Flow

1. User types a message → `ChatApp` assembles `PendingAttachment`s + prompt
2. `ConversationState.build_request()` builds the message list
3. `build_request_options()` validates and builds the tools array
4. `XAIResponsesClient.ask()` posts to xAI, loops over tool calls until a final response
5. `ChatResult` (text + image URLs/b64) returned to `ChatApp` worker → rendered in `RichLog`

### Settings Persistence

Settings saved to `.textual-grok-settings.json` in the **current working directory** (not `~/.config/`). The app also checks the legacy filename `.textualbot_settings.json` on first load.

### Key Keybindings (in `ChatApp`)
- `Ctrl+P` — command palette
- `Ctrl+T` — cycle theme
- `Ctrl+Up/Down` — prompt history navigation
- Export functions in command palette: save chat transcript, export images to `exports/`
