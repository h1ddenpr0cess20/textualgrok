"""Browser serving entrypoint for the Textual chatbot."""

from pathlib import Path
import sys

from textual_serve.server import Server

from textualbot_app.app import ChatApp
from textualbot_app.config import AppConfig


app = ChatApp(AppConfig.from_env())

_app_path = Path(__file__).with_name("textualbot.py").resolve()
server = Server(f'"{sys.executable}" "{_app_path}"')


if __name__ == "__main__":
    server.serve()
