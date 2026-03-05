# Tools

TextualGrok exposes five categories of tools to the Grok model: web search, X search, code interpreter, file search, and remote MCP servers. Each tool is opt-in and configured from the **Tools** tab of the Settings dialog (`Ctrl+P` → Settings).

When a tool is enabled, it is registered in the API request. Grok decides autonomously whether and when to call it based on the conversation.

---

## Web Search (`web_search`)

Lets Grok search the public web to answer questions that require up-to-date information.

### Enable

Open Settings → **Tools** tab → toggle **Web Search** on.

### Behavior

When enabled, the API request includes `{"type": "web_search"}` in the tools list. Grok will call this tool when it determines web content would improve its answer. Citation URLs from search results are appended to the response automatically.

### Default

Enabled by default.

---

## X Search (`x_search`)

Lets Grok search posts on X (formerly Twitter). This is separate from general web search and targets social content.

### Enable

Open Settings → **Tools** tab → toggle **X Search** on.

### Optional filters

Two additional toggles appear under X Search:

| Toggle | Effect |
|---|---|
| X Search: Image Understanding | Enables `enable_image_understanding` on the `x_search` tool. Grok can interpret image content found in X posts. |
| X Search: Video Understanding | Enables `enable_video_understanding` on the `x_search` tool. Grok can interpret video content found in X posts. |

Enabling either filter automatically enables X Search if it is not already on.

### Default

Enabled by default. Image and video understanding are off by default.

---

## Code Interpreter (`code_interpreter`)

Lets Grok write and execute code to answer questions that benefit from computation, data analysis, or transformation.

### Enable

Open Settings → **Tools** tab → toggle **Code Interpreter** on.

### Behavior

When enabled, the API request includes `{"type": "code_interpreter"}`. Grok may run arbitrary code on xAI's infrastructure to produce its answer. Results are included in the response text.

### Default

Enabled by default.

---

## File Search (`file_search`)

Lets Grok query one or more xAI vector stores to retrieve relevant content from documents you have previously uploaded and indexed.

> **What is a vector store?** It is a database of document embeddings hosted by xAI. You upload documents to a store via the xAI API or dashboard, and the store allows semantic search. TextualGrok connects to existing stores — it does not upload files on your behalf.

### Enable

1. Open Settings → **Tools** tab → toggle **File Search** on.
2. Open the **File Search** tab.
3. Enter one or more vector store IDs in **Vector Store IDs**, separated by commas (e.g., `vs_abc123,vs_def456`).
4. Optionally adjust **File Search Max Results** (default: 10).
5. Click **Save**.

### Validation

If File Search is enabled and no vector store IDs are provided, the app will refuse to send a message and show a validation error. Fix this by adding at least one vector store ID.

### Default

Disabled by default.

---

## MCP (Model Context Protocol)

Lets Grok call tools exposed by remote MCP servers. This is the most flexible option: any service that implements the MCP protocol can be wired in.

> **What is MCP?** The Model Context Protocol is an open standard for exposing tools and context to language models. External services implement MCP endpoints; the model calls them during inference. See [modelcontextprotocol.io](https://modelcontextprotocol.io) for background.

### Enable

1. Open Settings → **MCP** tab.
2. Toggle **Enable MCP Tools** on.
3. Configure at least one server (see below).
4. Click **Save**.

### Adding a server

In the MCP tab, fill in the form fields and click **Add Server**:

| Field | Required | Description |
|---|---|---|
| Server URL | Yes | The HTTPS endpoint of the MCP server (e.g., `https://mcp.example.com/sse`). |
| Server Label | No | A short name shown in the server list and passed to the model. |
| Server Description | No | A description of what the server provides, sent to the model. |
| Allowed Tool Names | No | Comma-separated list of tool names to expose. Leave blank to allow all tools from the server. |
| Authorization Header Value | No | Value for the `Authorization` HTTP header (e.g., `Bearer my-token`). |
| Extra Headers JSON | No | A JSON object of additional HTTP headers. Example: `{"X-Org-ID": "demo"}`. |

### Managing servers

- The dropdown under **Configured MCP Servers** lists all added servers. Selecting an entry populates the form fields for editing.
- Click **Remove Selected** to delete the selected server entry.
- Changes to the form are applied when you click **Add Server** (if the URL does not match an existing entry, a new one is added; if it matches, the existing entry is updated) or when you click **Save** (the current form contents are upserted before saving).

### Validation

If MCP is enabled and no servers are configured, or if a server entry is missing a URL, the app will block the request and show an error. Fix the configuration before sending.

### Default

Disabled by default.

---

## Tool interaction during inference

When a tool-enabled request is sent, the xAI Responses API may return tool call instructions rather than a final answer. TextualGrok handles this automatically:

1. Grok returns a function call (e.g., `generate_image`, or a built-in tool call from `web_search`).
2. The app executes the tool and sends the result back to the API.
3. This loop repeats up to 8 times per request.
4. The final assistant text response is rendered in the chat log.

You do not need to do anything to drive this loop. It runs in a background thread and the UI remains responsive while it proceeds.

---

## Tool defaults summary

| Tool | Default state |
|---|---|
| Web Search | On |
| X Search | On |
| X Search Image Understanding | Off |
| X Search Video Understanding | Off |
| Code Interpreter | On |
| File Search | Off |
| MCP | Off |
