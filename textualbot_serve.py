from textualbot_app.app import ChatApp
from textualbot_app.config import AppConfig


app = ChatApp(AppConfig.from_env())
