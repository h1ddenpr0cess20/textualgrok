from textualbot_app.app import ChatApp
from textualbot_app.config import AppConfig


def main() -> None:
    config = AppConfig.from_env()
    ChatApp(config).run()


if __name__ == "__main__":
    main()
