# Getting Started

## TL;DR

Install dependencies, add your xAI API key to a `.env` file, and run `python textualgrok.py`.

---

## Requirements

- Python 3.10 or later
- An xAI API key (get one at [x.ai](https://x.ai))
- A terminal that supports full-screen TUI applications

> **New to terminal applications?** TextualGrok is a TUI (Text User Interface) — a full-screen app that runs inside your terminal. It is not a web page and does not require a browser. Open any terminal application (Terminal on macOS, Windows Terminal on Windows, any terminal emulator on Linux) and follow the steps below.

---

## Step 1 — Clone or download the project

```bash
git clone https://github.com/your-org/textualgrok.git
cd textualgrok
```

If you downloaded a zip archive, unzip it and `cd` into the resulting folder.

---

## Step 2 — Create a virtual environment

Using a virtual environment keeps TextualGrok's dependencies isolated from your system Python installation.

**macOS / Linux (bash or zsh)**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
```

**Windows (Command Prompt)**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

> **What is a virtual environment?** It is an isolated copy of Python that lives inside the project folder. Packages installed here do not affect your system-wide Python installation, and vice versa. You must activate it every time you open a new terminal session before running the app.

---

## Step 3 — Install dependencies

With the virtual environment active:

```bash
pip install -r requirements.txt
```

This installs:

| Package | Version | Purpose |
|---|---|---|
| `httpx` | 0.27.0+ | HTTP client for xAI API calls |
| `textual` | 6.0.0+ | TUI framework |
| `textual-serve` | any | Browser serving mode |
| `python-dotenv` | 1.0.1+ | Loads `.env` files |
| `textual-image` | 0.8.3+ | Inline image rendering in the terminal |

---

## Step 4 — Configure your API key

Create a `.env` file in the project root. You can copy the provided example:

**macOS / Linux**

```bash
cp .env.example .env
```

**Windows (PowerShell)**

```powershell
Copy-Item .env.example .env
```

**Windows (Command Prompt)**

```cmd
copy .env.example .env
```

Open `.env` in any text editor and set your xAI API key:

```dotenv
XAI_API_KEY=your_key_here
```

The other variables in `.env` are optional and have sensible defaults. See [configuration.md](configuration.md) for the full list.

> **Do not commit your `.env` file to version control.** It contains your API key, which should be kept private. The project's `.gitignore` should already exclude it.

---

## Step 5 — Run the app

```bash
python textualgrok.py
```

The application opens in your terminal. You will see a chat log area, a prompt input at the bottom, and a footer showing available keybindings.

To quit, press `Ctrl+C`.

---

## Serving in a browser (optional)

TextualGrok can also run as a web app using Textual Serve. This is useful for sharing the app over a network or running it in an environment where you cannot use a full terminal.

```bash
python textualgrokserve.py
```

Open the URL printed to the terminal in any modern browser.

> **Note:** Textual Serve proxies the same terminal application through a browser. The experience is nearly identical to the native terminal version, but inline image rendering quality may vary by browser.

---

## First run checklist

- [ ] Virtual environment created and activated
- [ ] `pip install -r requirements.txt` completed with no errors
- [ ] `.env` file exists with `XAI_API_KEY` set
- [ ] `python textualgrok.py` opens the chat UI
- [ ] Typing a message and pressing `Enter` returns a response from Grok

If you see an error about `XAI_API_KEY`, double-check that your `.env` file is in the same directory where you run the command, and that the key is not surrounded by extra quotes.

---

## Next steps

- Learn the keyboard controls and UI: [usage.md](usage.md)
- Enable tools like web search or code interpreter: [tools.md](tools.md)
- Generate images with Grok Imagine: [image-generation.md](image-generation.md)
- Review all configuration options: [configuration.md](configuration.md)
