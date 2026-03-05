# Configuration

TextualGrok is configured in two layers:

1. **Environment variables** (`.env` file or shell environment) — read once at startup, control low-level defaults.
2. **UI settings** (`.textual-grok-settings.json`) — persisted automatically, control everything you can change in the Settings dialog.

---

## Environment variables

Set these in a `.env` file in the directory where you run the app, or export them in your shell before running.

| Variable | Required | Default | Description |
|---|---|---|---|
| `XAI_API_KEY` | Yes | — | Your xAI API key. The app refuses to start without this. |
| `XAI_MODEL` | No | `grok-4-1-fast-non-reasoning` | The chat model to use on first launch. Overridden by saved settings after the first run. |
| `XAI_SYSTEM_PROMPT` | No | `You are a concise, helpful assistant.` | The system prompt used on first launch. Overridden by saved settings after the first run. |
| `XAI_IMAGE_MODEL` | No | `grok-imagine-image` | The image generation model used on first launch. Overridden by saved settings after the first run. |

> **Note:** Environment variables only provide the initial defaults. Once you have opened Settings and clicked Save, those values are written to `.textual-grok-settings.json` and take precedence over the environment on subsequent runs.

### Example `.env` file

```dotenv
XAI_API_KEY=xai-yourkeyhere
XAI_MODEL=grok-4-1-fast-non-reasoning
XAI_SYSTEM_PROMPT=You are a concise, helpful assistant.
XAI_IMAGE_MODEL=grok-imagine-image
```

---

## Settings file

The settings file is a JSON file named `.textual-grok-settings.json`. It is created or updated in the **current working directory** every time you click **Save** in the Settings dialog, or when the app saves the theme after you use `Ctrl+T`.

> **Important:** The file is saved in whichever directory you run the app from. If you run the app from different directories, each directory gets its own settings file. Run the app consistently from the same directory if you want settings to persist reliably.

### Legacy filename

Older installations used `.textualbot_settings.json`. The app checks for this file on startup if `.textual-grok-settings.json` does not exist, and loads it as a fallback. Once you save settings through the UI, a new `.textual-grok-settings.json` is created and the legacy file is no longer read.

### File format

The settings file is plain JSON. You can edit it with any text editor if you need to make bulk changes or reset a value. The app merges the file with built-in defaults on startup, so unknown or missing keys are ignored gracefully.

### All settings fields

| Field | Type | Default | Description |
|---|---|---|---|
| `chat_model` | string | from `XAI_MODEL` | Chat model identifier. |
| `system_prompt` | string | from `XAI_SYSTEM_PROMPT` | System instruction for the model. Cannot be empty. |
| `theme` | string | `"dracula"` | Active UI theme name. |
| `include_history` | boolean | `true` | Whether to send conversation history with each request. |
| `web_search` | boolean | `true` | Enable the `web_search` tool. |
| `x_search` | boolean | `true` | Enable the `x_search` tool. |
| `x_search_image_understanding` | boolean | `false` | Enable image understanding on X search results. |
| `x_search_video_understanding` | boolean | `false` | Enable video understanding on X search results. |
| `code_interpreter` | boolean | `true` | Enable the `code_interpreter` tool. |
| `file_search` | boolean | `false` | Enable the `file_search` tool. |
| `vector_store_ids_raw` | string | `""` | Comma-separated vector store IDs. |
| `file_search_max_results_raw` | string | `"10"` | Maximum results per file search query. |
| `image_generation` | boolean | `false` | Enable the Grok Imagine `generate_image` tool. |
| `image_as_base64` | boolean | `false` | Return images as base64 data instead of URLs. |
| `image_model` | string | from `XAI_IMAGE_MODEL` | Grok Imagine model identifier. |
| `image_count_raw` | string | `"1"` | Number of images to generate per call (1-10). |
| `image_source_url_raw` | string | `""` | Optional source image URL for edits. Cleared on save via UI. |
| `image_use_last` | boolean | `true` | Automatically use the last generated image as the source for edits. |
| `image_aspect_ratio_raw` | string | `"1:1"` | Default aspect ratio for generated images. |
| `mcp_enabled` | boolean | `false` | Enable MCP tool servers. |
| `mcp_servers` | array | `[]` | List of MCP server configuration objects (see below). |

### MCP server object format

Each entry in `mcp_servers` is a JSON object with the following fields:

| Field | Required | Type | Description |
|---|---|---|---|
| `server_url` | Yes | string | The MCP server endpoint URL. |
| `server_label` | No | string | Short display name. |
| `server_description` | No | string | Description passed to the model. |
| `authorization` | No | string | Value for the `Authorization` HTTP header. |
| `allowed_tool_names` | No | array of strings | If set, only these tool names are exposed. |
| `extra_headers` | No | object | Additional HTTP headers as a string-to-string map. |

Example `mcp_servers` entry:

```json
{
  "server_url": "https://mcp.example.com/sse",
  "server_label": "example",
  "server_description": "An example MCP server",
  "authorization": "Bearer my-secret-token",
  "allowed_tool_names": ["search", "summarize"],
  "extra_headers": {
    "X-Org-ID": "my-org"
  }
}
```

### Example settings file

```json
{
  "chat_model": "grok-4-1-fast-non-reasoning",
  "system_prompt": "You are a concise, helpful assistant.",
  "theme": "dracula",
  "include_history": true,
  "web_search": true,
  "x_search": true,
  "x_search_image_understanding": false,
  "x_search_video_understanding": false,
  "code_interpreter": true,
  "file_search": false,
  "vector_store_ids_raw": "",
  "file_search_max_results_raw": "10",
  "image_generation": false,
  "image_as_base64": false,
  "image_model": "grok-imagine-image",
  "image_count_raw": "1",
  "image_source_url_raw": "",
  "image_use_last": true,
  "image_aspect_ratio_raw": "1:1",
  "mcp_enabled": false,
  "mcp_servers": []
}
```

---

## Validation

Settings are validated both when you click **Save** in the Settings dialog and immediately before sending each message. Validation errors are shown in the dialog or in the status bar. Common validation rules:

- `system_prompt` cannot be empty.
- `image_count_raw` must be a positive integer between 1 and 10.
- `file_search_max_results_raw` must be a positive integer.
- `vector_store_ids_raw` must contain at least one ID when File Search is enabled.
- `image_aspect_ratio_raw` must be one of the supported aspect ratio values or `auto`.
- Each MCP server in `mcp_servers` must have a non-empty `server_url`.

---

## API connection settings

The xAI API base URL is `https://api.x.ai/v1`. This is not configurable through settings or environment variables. The HTTP client uses a 60-second timeout for chat requests and a 180-second timeout for image generation requests.
