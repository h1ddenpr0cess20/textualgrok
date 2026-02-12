from textualbot_app.models import ChatMessage


class ConversationState:
    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt
        self.history: list[ChatMessage] = []

    def build_request(self, user_prompt: str, *, include_history: bool = True) -> list[ChatMessage]:
        history: list[ChatMessage] = self.history if include_history else []
        return [
            {"role": "system", "content": self.system_prompt},
            *history,
            {"role": "user", "content": user_prompt},
        ]

    def commit_turn(self, user_prompt: str, assistant_text: str) -> None:
        self.history.append({"role": "user", "content": user_prompt})
        self.history.append({"role": "assistant", "content": assistant_text})

    def clear_history(self) -> None:
        self.history.clear()
