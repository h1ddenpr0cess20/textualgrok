# Contributing

## TL;DR

Fork the repo, create a virtual environment, install dependencies, make your changes, and open a pull request. There is no build step. Run the app to test changes manually.

---

## Development setup

1. Fork and clone the repository.

2. Create and activate a virtual environment:

   **macOS / Linux**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   **Windows (PowerShell)**

   ```powershell
   python -m venv .venv
   . .venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API key:

   ```bash
   cp .env.example .env
   # Edit .env and set XAI_API_KEY
   ```

5. Run the app to confirm everything works:

   ```bash
   python textualgrok.py
   ```

---

## Project layout

See [architecture.md](architecture.md) for a full description of each module. The key principle is that each module has a single responsibility:

- UI logic lives in `chat_app.py`, modal screens, and `chat_log.py`.
- API calls are isolated in `xai_client.py`.
- Response parsing is isolated in `response_parser.py`.
- Data types are defined in `ui_types.py` and `models.py`.
- Configuration lives in `config.py` (environment) and `settings_persistence.py` (file).

When adding a feature, keep this separation. Do not put API calls in UI code or UI logic in the client.

---

## Coding conventions

- **Python version:** 3.10+. Use `match`/`case`, `X | Y` union types, and `from __future__ import annotations` where needed.
- **Type hints:** All public functions and methods should have type annotations. Use `dict`, `list`, `tuple` (lowercase) rather than `Dict`, `List`, `Tuple` from `typing`.
- **Imports:** Standard library first, then third-party, then local. Separate groups with a blank line.
- **Error handling:** Raise `ValueError` for user-visible configuration errors. Raise `RuntimeError` for API and network errors. Catch specific exception types; avoid bare `except:`.
- **No global state:** Do not use module-level mutable state. Pass dependencies explicitly.
- **Docstrings:** Write a one-line docstring for non-trivial public functions. Use Google-style docstrings (Args/Returns/Raises) for functions with complex signatures.
- **Line length:** 100 characters. There is no linter configuration; apply this as a guideline.

---

## Testing

The project includes a test suite for pure-logic modules. Tests live alongside the source (check the repository for the test directory structure).

To run tests:

```bash
python -m pytest
```

The test suite covers:

- `response_parser.py` — text extraction, function call extraction, image extraction, citation extraction.
- `render_tags.py` — tag parsing, attribute parsing, body parsing, end-to-end render handling.
- `options.py` — tool list assembly, validation errors.
- `conversation.py` — history assembly, commit, clear.
- `settings_persistence.py` — load/save round-trips, legacy fallback.
- `attachments.py` — file type detection.

When adding a new pure-logic module, add corresponding tests. UI modules (`chat_app.py`, screens) are not currently covered by automated tests and are tested manually.

---

## Adding a feature — checklist

1. Understand where the feature fits in the architecture (see [architecture.md](architecture.md)).
2. Add any new data fields to `UISettings` in `ui_types.py` with a sensible default.
3. Implement the logic in the appropriate module.
4. If the feature is user-configurable, add UI controls to the relevant tab in `settings_screen.py` and wire them up in `_collect_settings()`.
5. Update `build_request_options()` in `options.py` if the feature affects the API tools list.
6. Write tests for any pure-logic code.
7. Test the feature manually by running the app.
8. Update the relevant documentation file in `docs/`.

---

## Pull request guidelines

- Open one pull request per logical change.
- Write a clear description of what the PR does and why.
- If the PR fixes a bug, include steps to reproduce the bug before the fix.
- Keep PRs small and focused. Large refactors that mix multiple concerns are harder to review.
- Do not include unrelated formatting or whitespace changes.

---

## What not to change

- Do not move `.textual-grok-settings.json` to a different path without a migration plan — existing users would lose their settings.
- Do not change the xAI API base URL unless you are testing against a different environment.
- Do not add dependencies without discussing first. The dependency list is intentionally minimal.
