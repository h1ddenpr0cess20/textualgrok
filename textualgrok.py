from textualgrok.app import ChatApp
from textualgrok.config import AppConfig


def main() -> None:
    config = AppConfig.from_env()
    ChatApp(config).run()


if __name__ == "__main__":
    main()
