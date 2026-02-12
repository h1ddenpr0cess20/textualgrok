from dataclasses import dataclass
from typing import Any, Optional, TypedDict


class ChatMessage(TypedDict):
    role: str
    content: Any


@dataclass
class ChatResult:
    text: str
    response_id: Optional[str]
    image_urls: list[str] | None = None
    image_b64: list[str] | None = None
