import json
from typing import Optional
from urllib.request import Request, urlopen

from .config import settings


class GroqService:
    """Groq is used only for final answer generation."""

    def __init__(self) -> None:
        settings.validate()

    def generate_answer(
        self,
        question: str,
        context: str,
        model: Optional[str] = None,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a retrieval-augmented generation assistant. "
                    "Answer using only the retrieved document context. "
                    "Do not invent facts. If the context is insufficient, "
                    'say: "I could not find enough information in the '
                    'supplied documents." Mention source filenames when '
                    "useful."
                ),
            },
            {
                "role": "user",
                "content": (
                    "RETRIEVED CONTEXT\n"
                    "-----------------\n"
                    f"{context}\n\n"
                    "USER QUESTION\n"
                    "-------------\n"
                    f"{question}"
                ),
            },
        ]

        payload = {
            "model": model or settings.groq_model,
            "messages": messages,
            "temperature": 0.1,
        }

        request = Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {settings.groq_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))

        return payload["choices"][0]["message"]["content"] or ""
