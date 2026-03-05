# Usage Guide

## TL;DR

Type your message, press `Enter` to send. Use `Ctrl+P` to open the command palette. Attach files with the "Attach File/Folder" button. Press `Ctrl+C` to quit.

---

## UI layout

When the application starts, the screen is divided into several regions from top to bottom:

1. **Header** — displays the application title.
2. **Chat log** — scrollable history of the conversation. Your messages appear as `You:` and Grok's replies appear as `Grok:` with markdown formatting.
3. **Session image panel** — appears automatically below the chat log when Grok generates images. Shows thumbnails of all images produced in the current session.
4. **Status bar** — a one-line area below the chat log that shows state messages ("Ready.", "Waiting for xAI response...", error messages, and so on).
5. **Attachments bar** — shows which files are currently queued to be sent with your next message, along with buttons to add or clear attachments.
6. **Prompt row** — a multi-line text input on the left and a Send button on the right.
7. **Footer** — lists the most important keybindings.

---

## Keybindings

| Key | Action |
|---|---|
| `Enter` | Send the current prompt |
| `Shift+Enter` | Insert a newline in the prompt (for multi-line messages) |
| `Ctrl+S` | Send the current prompt (alternative) |
| `Ctrl+T` | Cycle through available UI themes |
| `Ctrl+Up` | Navigate to the previous prompt in history |
| `Ctrl+Down` | Navigate to the next prompt in history (or back to the draft) |
| `Ctrl+P` | Open the command palette |
| `Ctrl+C` | Quit the application |

> **New to TUI keybindings?** These work in the terminal, not the browser address bar. Make sure the application window has focus. On some systems, `Ctrl+S` may be intercepted by the terminal for flow control — use `Enter` instead.

---

## Sending a message

1. Click on the prompt input at the bottom of the screen, or it will already be focused on startup.
2. Type your message.
3. Press `Enter` to send, or click the **Send** button.

While a response is in progress, the prompt input, Send button, and Attach button are all disabled. The status bar shows "Waiting for xAI response..." until the reply arrives.

> **Sending multi-line prompts:** Press `Shift+Enter` to add a newline inside the prompt. This is useful when you want to structure a question with multiple paragraphs or include a code block.

---

## Prompt history

The app keeps a history of prompts you have sent in the current session.

- Press `Ctrl+Up` to navigate backward through previous prompts.
- Press `Ctrl+Down` to navigate forward. If you are already at the most recent entry, pressing `Ctrl+Down` restores the draft you were typing before you started navigating.

Your current unsent text is saved as a draft while you browse history and restored when you return to the end.

---

## Conversation history

By default, the app sends the full conversation history with each request so Grok can answer follow-up questions in context.

To turn this off, open Settings (`Ctrl+P` → **Settings** → **Chat** tab) and disable **Use Conversation History**. With history disabled, each message is treated as a fresh, standalone request.

Use **Clear Chat** from the command palette to wipe the chat log and reset conversation history without quitting the app.

---

## Attaching files and images

You can attach files or images to your next message. The attachment is sent alongside your prompt text.

### To attach a file or folder

1. Click the **Attach File/Folder** button in the attachments bar.
2. A dialog appears. Either:
   - Type or paste a file path, folder path, or image URL directly into the input field, or
   - Click **Browse...** to open an interactive file browser.
3. Click **Attach** to confirm.

### File browser controls

- Use the file list to navigate directories. Directories are shown as `[DIR]`, supported images as `[IMG]`, and other files as `[FILE]`.
- Double-click a directory (or press `Enter` when it is highlighted) to open it.
- Click a file or directory to select it; click **Use Selected** to attach it.
- The **Home** and **CWD** buttons jump to your home directory or the current working directory.
- On Windows, a drive selector appears so you can switch between drives.

### Attaching a folder

When you attach a folder, all supported files inside it (including in subdirectories) are added as individual attachments. Unsupported files are silently skipped; the status bar reports how many were skipped.

### Attaching an image URL

Paste an image URL (starting with `http://` or `https://`) directly into the source field. Only image URLs are supported this way; for non-image remote content, download the file first.

### Supported file types

**Images (sent as inline image data):**
`.jpg`, `.jpeg`, `.png`

**Documents and code (sent as file data with filename metadata):**
`.txt`, `.md`, `.rst`, `.rtf`, `.csv`, `.tsv`, `.json`, `.jsonl`, `.yaml`, `.yml`, `.toml`, `.ini`, `.cfg`, `.conf`, `.log`, `.xml`, `.html`, `.pdf`, `.doc`, `.docx`, `.ppt`, `.pptx`, `.xls`, `.xlsx`

**Code files:**
`.py`, `.pyi`, `.js`, `.jsx`, `.mjs`, `.cjs`, `.ts`, `.tsx`, `.java`, `.c`, `.h`, `.cpp`, `.hpp`, `.cc`, `.cs`, `.go`, `.rs`, `.rb`, `.php`, `.swift`, `.kt`, `.kts`, `.scala`, `.sql`, `.sh`, `.bash`, `.zsh`, `.ps1`, `.bat`, `.cmd`

**Named files without extensions:**
`Dockerfile`, `Makefile`, `README`, `LICENSE`

Any file with a `text/*` MIME type is also accepted.

### Clearing attachments

Click **Clear Attach** to remove all queued attachments before sending. The button is disabled when there are no pending attachments.

---

## Settings

Open the settings dialog from the command palette (`Ctrl+P` → **Settings**) or by pressing `Escape` to close it without saving.

The settings dialog has five tabs:

### Chat tab

| Field | Description |
|---|---|
| Model | The Grok chat model to use. Click **Refresh Models** to fetch the current list from the xAI API. |
| System Prompt | The instruction given to the model at the start of every conversation. |
| Use Conversation History | When enabled, the full conversation history is sent with each request. |

### Tools tab

| Field | Description |
|---|---|
| Web Search | Enable the `web_search` tool so Grok can search the web. |
| X Search | Enable the `x_search` tool so Grok can search X (Twitter). |
| X Search: Image Understanding | Attach image understanding to X search results (requires X Search enabled). |
| X Search: Video Understanding | Attach video understanding to X search results (requires X Search enabled). |
| Code Interpreter | Enable the `code_interpreter` tool so Grok can execute code. |
| File Search | Enable the `file_search` tool to query vector stores. |

### File Search tab

| Field | Description |
|---|---|
| Vector Store IDs | Comma-separated list of vector store IDs (e.g., `vs_abc123,vs_def456`). Required when File Search is enabled. |
| File Search Max Results | Maximum number of results to return per query. Default: 10. |

### Image tab

| Field | Description |
|---|---|
| Enable Grok Imagine Tool | Register the image generation function tool with the model. |
| Return Image As Base64 | Return image data as base64 instead of a URL. Useful if URLs expire quickly. |
| Use Last Generated Image For Edits | Automatically pass the last generated image as the source when editing. |
| Grok Imagine Model | Which image model to use. Click **Refresh Models** in the Chat tab to update options. |
| Image Count | Number of images to generate per request (1-10). |
| Aspect Ratio | Output aspect ratio. Options: `auto`, `1:1`, `16:9`, `9:16`, `4:3`, `3:4`, `3:2`, `2:3`, `2:1`, `1:2`, `19.5:9`, `9:19.5`, `20:9`, `9:20`. |

### MCP tab

| Field | Description |
|---|---|
| Enable MCP Tools | Enable all configured MCP servers as tools. |
| Add Server / Remove Selected | Add or remove MCP server entries. |
| Server Label | Optional display name for the server. |
| Server URL | Required URL of the MCP server endpoint. |
| Server Description | Optional description sent to the model. |
| Allowed Tool Names | Comma-separated list of tool names to allow. Leave blank to allow all. |
| Authorization Header Value | Optional `Authorization` header value for authenticated servers. |
| Extra Headers JSON | Optional JSON object of additional HTTP headers (e.g., `{"X-Org": "demo"}`). |

Click **Save** to apply settings. Settings are written to `.textual-grok-settings.json` in the current working directory. Click **Cancel** or press `Escape` to discard changes.

---

## Themes

Press `Ctrl+T` to cycle through the available Textual themes. The selected theme is persisted to the settings file automatically.

The code highlighting theme in the chat log is chosen to complement the selected UI theme. Dark UI themes use dark code themes; light UI themes use light code themes.

---

## Session images

When Grok generates images (using the Grok Imagine tool or `<grok:render>` tags), they appear in the **Session Image** panel below the chat log.

- Each image shows a thumbnail, its index number, and a truncated source reference.
- Click a thumbnail or click the **Open** button beneath it to open the image in the full-screen image gallery.
- Scroll the image panel horizontally with a mouse scroll wheel while hovering over it.

Images are stored in temporary files for the duration of the session. They are deleted when you quit the app.

### Image gallery

The gallery screen shows a single image at a time and lets you navigate between all session images. Press `Escape` or `Q` to close the gallery and return to the chat.

To save an image permanently, use the export function available within the gallery.

---

## Exports

### Saving the chat transcript

Open the command palette (`Ctrl+P`) and select **Save Chat**. The transcript is saved as a Markdown file at:

```
exports/chat-YYYYMMDD-HHMMSS.md
```

The `exports/` directory is created in the current working directory if it does not exist.

### Saving generated images

Saved images are written to `exports/images/` with filenames like `image-YYYYMMDD-HHMMSS.png`.

---

## Command palette

Press `Ctrl+P` to open the command palette. Available commands:

| Command | Action |
|---|---|
| Settings | Open the settings dialog |
| Clear Chat | Clear the chat log and reset conversation history |
| Save Chat | Save the transcript to `exports/` |

Type to search commands by name.
