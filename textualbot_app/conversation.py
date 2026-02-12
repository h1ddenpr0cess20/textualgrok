from textualbot_app.models import ChatMessage


class ConversationState:
    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt
        self.history: list[ChatMessage] = []

    def build_request(
        self,
        user_prompt: str,
        *,
        include_history: bool = True,
        user_content_parts: list[dict[str, str]] | None = None,
    ) -> list[ChatMessage]:
        history: list[ChatMessage] = self.history if include_history else []
        user_content: object
        if user_content_parts:
            user_content = [*user_content_parts, {"type": "input_text", "text": user_prompt}]
        else:
            user_content = user_prompt
        return [
            {"role": "system", "content": self.system_prompt},
            *history,
            {"role": "user", "content": user_content},
        ]

    def commit_turn(self, user_content: object, assistant_text: str) -> None:
        self.history.append({"role": "user", "content": user_content})
        self.history.append({"role": "assistant", "content": assistant_text})

    def clear_history(self) -> None:
        self.history.clear()
