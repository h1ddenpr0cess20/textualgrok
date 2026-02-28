import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    api_key: str
    model: str
    system_prompt: str
    image_model: str

    @classmethod
    def from_env(cls) -> "AppConfig":
        load_dotenv()

        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set XAI_API_KEY before running this app.")

        model = os.getenv("XAI_MODEL", "grok-4-1-fast-non-reasoning")
        system_prompt = os.getenv(
            "XAI_SYSTEM_PROMPT",
            "You are a concise, helpful assistant.",
        )
        image_model = os.getenv("XAI_IMAGE_MODEL", "grok-imagine-image")
        return cls(
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            image_model=image_model,
        )
