# Architecture

This document describes how TextualGrok is structured internally. It is intended for contributors and developers who want to understand the codebase, add features, or debug issues.

---

## High-level overview

TextualGrok is a [Textual](https://textual.textualize.io/) application that makes direct HTTP calls to xAI's Responses API (`/v1/responses`) and Images API (`/v1/images/generations`) using `httpx`. There is no intermediate server or SDK — the app builds and sends JSON payloads directly.

The codebase is organized as a Python package (`textualgrok/`) with two thin entry points at the project root.

---

## Entry points

| File | Role |
|---|---|
| `textualgrok.py` | Loads `AppConfig` from the environment, instantiates `ChatApp`, and calls `.run()`. |
| `textualgrokserve.py` | Instantiates the same `ChatApp` and wraps it with a `textual-serve` `Server` for browser access. |

Both entry points are identical in behavior from the user's perspective. The serve entrypoint proxies the terminal app through a WebSocket-based browser renderer.

---

## Package structure

```
textualgrok/
  app.py                  # Re-exports ChatApp for backward compatibility
  chat_app.py             # Main Textual App subclass
  xai_client.py           # HTTP client for xAI APIs
  config.py               # AppConfig: environment variable loading
  ui_types.py             # Shared data types: UISettings, PendingAttachment, ...
  models.py               # API response types: ChatMessage, ChatResult
  conversation.py         # ConversationState: history management
  options.py              # build_request_options(): tool list assembly and validation
  response_parser.py      # Extract text, images, citations, function calls from API responses
  render_tags.py          # Parse and execute <grok:render> tags
  image_api.py            # /images/generations payload construction and execution
  image_manager.py        # Temp file management, session image grid UI
  attachment_handler.py   # Build PendingAttachments from paths/URLs, attachment bar UI
  attachments.py          # File type detection helpers
  settings_screen.py      # 5-tab settings modal
  settings_persistence.py # Load/save UISettings as JSON
  attachment_screens.py   # File browser and URL input modals
  image_gallery_screen.py # Full-screen image viewer
  command_provider.py     # Custom command palette provider
  chat_log.py             # RichLog rendering and code theme selection
  app.tcss                # Textual CSS for all screens
```

---

## Module responsibilities

### `config.py` — `AppConfig`

A frozen dataclass loaded by `AppConfig.from_env()`. Reads `XAI_API_KEY`, `XAI_MODEL`, `XAI_SYSTEM_PROMPT`, and `XAI_IMAGE_MODEL` from the environment (or `.env` file via `python-dotenv`). Raises `RuntimeError` if `XAI_API_KEY` is missing.

`AppConfig` is immutable after creation and passed into `ChatApp` at construction time.

### `ui_types.py` — Data classes

Pure data containers with no external dependencies:

- `UISettings` — all 22+ configurable fields that can be changed through the Settings dialog and persisted to disk.
- `PendingAttachment` — a file or URL queued to be sent with the next message. Holds the label (shown in the UI), the API-ready content part dict, and an optional preview path for images.
- `SessionImageItem` — a generated image stored as a temp file path plus its source reference string.
- `BrowseEntry` — a single entry in the file browser (path, kind, and whether it is an image).

### `models.py` — API types

- `ChatMessage` — a `TypedDict` with `role` and `content`. Used to build the message list for `/v1/responses`.
- `ChatResult` — a dataclass returned by `XAIResponsesClient.ask()` and `generate_images()`. Contains `text`, `response_id`, `image_urls`, and `image_b64`.

### `conversation.py` — `ConversationState`

Manages the in-memory conversation history:

- `build_request()` — assembles the full message list (system message + history + new user message) for the API payload. Accepts optional `user_content_parts` for attachments.
- `commit_turn()` — appends the user and assistant messages to history after a successful response.
- `clear_history()` — resets history.

### `options.py` — `build_request_options()`

Validates all tool-related settings and assembles the `tools` list for each API request. Returns a frozen `RequestOptions` dataclass. Raises `ValueError` for invalid configurations (missing vector store IDs, invalid count, invalid aspect ratio, etc.).

The function is called immediately before each message is sent and also when the user clicks **Save** in Settings (to catch errors early).

### `xai_client.py` — `XAIResponsesClient`

The only file that makes HTTP calls to xAI. Key methods:

- `ask(messages, options, *, last_image_url)` — sends to `/v1/responses` and runs the tool execution loop (up to 8 iterations). Handles `generate_image` function calls, `<grok:render>` tags, citation extraction, and deduplication of image outputs. Returns a `ChatResult`.
- `generate_images(prompt, options, source_image_url)` — calls `/v1/images/generations` directly (used for standalone image requests, not tool calls).
- `list_models()` — calls `/v1/models` and returns a sorted list of model IDs.
- `fetch_bytes(url)` — downloads bytes from a URL (used to download generated images to temp files).
- `_post_json()` — the internal POST helper. Supports configurable timeout and retry count with exponential backoff for transient HTTP errors (408, 429, 500-504).

### `response_parser.py`

Pure functions that extract structured data from raw xAI API response dicts. Handles multiple response formats that xAI may return:

- `extract_text()` — pulls the assistant's text from `output_text`, `output` items, or `choices`.
- `extract_function_calls()` — recursively scans the response for function/tool call descriptors, deduplicates them, and returns a normalized list.
- `extract_image_entries()` — finds all image entries (URL or base64) across multiple possible response shapes.
- `extract_citations()` — finds citation URLs in the response.
- `find_render_text_in_payload()` — searches the full payload for any string containing a `<grok:render>` tag, used as a fallback when text extraction returns nothing.

### `render_tags.py`

Detects and processes `<grok:render>` tags in response text. The main entry point is `handle_render_tags()`, which:

1. Finds all `<grok:render>...</grok:render>` blocks with a regex.
2. Parses attributes and body (supports JSON, XML sub-elements, line-based key-value pairs, and plain text).
3. Calls `execute_generate_image_tool()` for each valid tag.
4. Replaces the tag in the text with a summary of what was generated.

### `image_api.py`

- `build_image_payload()` — constructs the JSON payload for `/v1/images/generations`.
- `execute_generate_image_tool()` — the shared callable used by both `xai_client.py` (for tool calls) and `render_tags.py` (for render tags). Merges per-call arguments with the user's configured defaults, calls `_post_json`, and returns a dict with `images`, `images_b64`, `revised_prompt`, `count`, and optionally `error`.

### `image_manager.py`

Handles the lifecycle of session images:

- `materialize_session_images()` — takes a `ChatResult` and downloads or decodes all images into temp files. Returns `(images, errors, temp_paths)`.
- `refresh_image_grid()` — rebuilds the horizontal image grid UI widget from the current list of `SessionImageItem` objects.
- `cleanup_temp_images()` — deletes temp files on quit.
- `resolve_image_index_from_click()` — walks the Textual widget tree from a click event to find which image was clicked (by `_image_index` attribute).

### `attachment_handler.py`

- `build_pending_attachments(source)` — takes a file path, folder path, or URL and returns a list of `PendingAttachment` objects. For folders, recurses through all files and skips unsupported types. Images are encoded as `input_image` data URLs; other files as `input_file` data URLs with filename metadata.
- `refresh_pending_attachments_ui()` — updates the attachment bar widgets (status text, clear button state, thumbnail strip) to reflect the current pending attachments.

### `attachments.py`

Defines sets of supported file extensions, MIME types, and filenames. Provides three predicates: `is_likely_image_file()`, `is_likely_image_url()`, and `is_supported_attachment_file()`.

### `settings_persistence.py`

- `load_ui_settings(default)` — reads `.textual-grok-settings.json` from `Path.cwd()`, falls back to `.textualbot_settings.json` (legacy), then to the provided default `UISettings`. Merges JSON keys onto the default's fields so added fields in future versions always have a fallback.
- `save_ui_settings(settings)` — writes `UISettings.__dict__` as JSON to `.textual-grok-settings.json`.

### `chat_app.py` — `ChatApp`

The main `textual.App` subclass (~580 lines). Responsibilities:

- **UI composition** — header, chat log, image panel, status bar, attachments bar, prompt input, footer.
- **Lifecycle** — subscribes to theme-change signals on mount; closes the HTTP client and cleans up temp files on unmount.
- **Command palette** — exposes Settings, Clear Chat, and Save Chat commands.
- **Prompt submission** — validates settings, assembles `RequestOptions`, builds the message list via `ConversationState`, and dispatches a background worker (`run_worker(thread=True)`).
- **Worker result handling** — `on_worker_state_changed` dispatches on worker name: the `ask` worker delivers `ChatResult` to the UI; the `load-session-images` worker delivers materialized image items.
- **Theme persistence** — saves the active theme to `UISettings` and to disk whenever it changes.
- **Export** — saves the transcript and images to `exports/`.

### `settings_screen.py` — `SettingsScreen`

A `ModalScreen[Optional[UISettings]]` with five `TabPane` sections. Dismisses with `None` (cancel) or with a new `UISettings` instance (save). Validates settings by calling `build_request_options()` before dismissing. Loads the model list from the API in a background worker when opened.

### `attachment_screens.py`

- `AddAttachmentScreen` — a modal with a text input for a path or URL and a Browse button. Dismisses with a path/URL string or `None`.
- `BrowseAttachmentFileScreen` — a full file browser modal. Lists directories and supported files. Supports double-click to open directories. Has Home, CWD, and drive-selector navigation controls.
- `BrowseOptionList` — an `OptionList` subclass with double-click detection (0.45-second window).

### `chat_log.py`

Thin rendering helpers used by `ChatApp`:

- `write_chat_log_entry()` — writes a single entry (user, assistant, or error) to a `RichLog` widget.
- `render_markdown()` — wraps `rich.markdown.Markdown` with error handling.
- `code_theme_for_app_theme()` — maps the Textual theme name to a valid Pygments style for code blocks.

### `command_provider.py` — `OrderedSystemCommandsProvider`

A `textual.command.Provider` that yields the app's system commands in definition order (Settings, Clear Chat, Save Chat, then Textual built-ins) rather than alphabetically.

---

## Data flow

```
User types prompt
        |
ChatApp._submit_prompt_from_ui()
        |
        +-- build_request_options()     -- validates settings, builds tools list
        |
        +-- ConversationState.build_request()  -- assembles message list
        |
        +-- run_worker(thread=True, name="ask")
                |
                XAIResponsesClient.ask()
                        |
                        POST /v1/responses
                        |
                        Tool loop (up to 8x):
                          extract_function_calls()
                          execute_generate_image_tool()  or  built-in tools
                          POST /v1/responses (continuation)
                        |
                        handle_render_tags()  -- process <grok:render> tags
                        |
                        extract_citations()
                        |
                        return ChatResult
                |
        on_worker_state_changed(name="ask")
                |
                ConversationState.commit_turn()
                |
                write_chat_log_entry()  (renders in RichLog)
                |
                run_worker(thread=True, name="load-session-images")
                        |
                        materialize_session_images()
                                |
                                fetch_bytes()  -- downloads URL images
                                write_temp_image_bytes()
                                return (images, errors, temp_paths)
                        |
                on_worker_state_changed(name="load-session-images")
                        |
                        refresh_image_grid()
```

---

## Extension points

### Adding a new tool type

1. Add a new field to `UISettings` in `ui_types.py`.
2. Add the field to the `build_request_options()` signature and body in `options.py`. Append the tool dict to `tools` when the field is truthy.
3. Add a UI control in the **Tools** tab of `SettingsScreen` in `settings_screen.py`.
4. Connect the UI control in `_collect_settings()` in `settings_screen.py`.
5. Update `_sync_inputs()` if the new tool has dependent fields.
6. The tool will be passed automatically in API requests. If the tool requires an execution loop (like `generate_image`), add handling in `XAIResponsesClient.ask()`.

### Adding a new UI screen

Create a `ModalScreen` subclass. Push it from `ChatApp` using `self.push_screen()` and handle the return value in a callback. Follow the pattern in `SettingsScreen` or `AddAttachmentScreen`.

### Modifying response parsing

All response parsing is isolated in `response_parser.py`. The functions are pure (no side effects, no imports from other app modules) and can be tested independently.

### Changing where settings are saved

Edit `settings_file_path()` in `settings_persistence.py`. The current implementation uses `Path.cwd()`. A common change would be to use `Path.home() / ".config" / "textualgrok"`.

---

## Threading model

TextualGrok uses Textual's built-in worker system for all blocking I/O:

- `run_worker(thread=True, exclusive=True, name="ask")` — runs the API call in a thread. `exclusive=True` cancels any previous worker with the same exclusivity group.
- `run_worker(thread=True, exclusive=False, name="load-session-images")` — runs image materialization in a thread. Not exclusive so multiple image loads can overlap.
- `run_worker(thread=True, exclusive=True, name="load-models")` — runs inside `SettingsScreen` to fetch the model list.

All UI updates from workers happen in `on_worker_state_changed`, which is called on the main thread by Textual.

---

## CSS and styling

`app.tcss` is a Textual CSS file that styles all screens. Textual CSS is a subset of CSS with custom widget selectors. The file is loaded automatically by `ChatApp` via `CSS_PATH = "app.tcss"`.

Theme switching (`Ctrl+T`) cycles through Textual's built-in themes. Code block highlighting in the chat log uses Pygments and adapts to the current theme's dark/light state.
