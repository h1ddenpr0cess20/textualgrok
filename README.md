# TextualGrok

A terminal-first chatbot UI built with [Textual](https://textual.textualize.io/) that connects to [xAI's Responses API](https://docs.x.ai/api) — with tool use, file attachments, image generation, and MCP server support.

TextualGrok runs as a full-screen TUI in your terminal. It supports web search, X search, code interpreter, file search, and remote MCP tools. You can attach local files and folders or image URLs to your messages, generate and edit images with Grok Imagine, and export transcripts and images to disk.

## Table of Contents

- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Quick Start

Requires Python 3.10+ and an [xAI API key](https://x.ai).

**macOS / Linux**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then set XAI_API_KEY in .env
python textualgrok.py
```

**Windows (PowerShell)**

```powershell
python -m venv .venv; . .venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env   # then set XAI_API_KEY in .env
python textualgrok.py
```

## Documentation

Full documentation lives in the [docs/](docs/) folder:

- [docs/index.md](docs/index.md) — documentation home and table of contents
- [docs/getting-started.md](docs/getting-started.md) — installation, setup, and first run
- [docs/usage.md](docs/usage.md) — UI controls, keybindings, attachments, settings, exports
- [docs/tools.md](docs/tools.md) — web search, X search, code interpreter, file search, MCP
- [docs/image-generation.md](docs/image-generation.md) — Grok Imagine, settings, grok:render tags
- [docs/configuration.md](docs/configuration.md) — environment variables, settings file reference
- [docs/architecture.md](docs/architecture.md) — module map, data flow, extension points
- [docs/contributing.md](docs/contributing.md) — dev setup, coding conventions, testing

## Contributing

Contributions are welcome. See [docs/contributing.md](docs/contributing.md) for setup instructions, coding conventions, and pull request guidelines.

## License

See [LICENSE](LICENSE).
