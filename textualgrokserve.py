"""Browser serving entrypoint for the Textual chatbot."""

from pathlib import Path
import sys

from textual_serve.server import Server

from textualgrok.app import ChatApp
from textualgrok.config import AppConfig


app = ChatApp(AppConfig.from_env())

_app_path = Path(__file__).with_name("textualgrok.py").resolve()
server = Server(f'"{sys.executable}" "{_app_path}"')


if __name__ == "__main__":
    server.serve()
