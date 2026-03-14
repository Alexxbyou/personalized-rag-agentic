from __future__ import annotations

import os

import dspy
from dotenv import load_dotenv
from openai import OpenAI


class OpenAIService:
    def __init__(
        self,
        api_key: str | None = None,
        chat_model: str = "gpt-5-mini",
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.client = OpenAI(api_key=self.api_key)
        self._dspy_configured = False

    def call_llm(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        kwargs = {"model": model or self.chat_model, "messages": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def get_embedding(self, text: str, model: str | None = None) -> list[float]:
        response = self.client.embeddings.create(
            input=text,
            model=model or self.embedding_model,
        )
        return response.data[0].embedding

    def configure_dspy(self, model: str | None = None) -> None:
        if self._dspy_configured:
            return
        lm = dspy.LM(f"openai/{model or self.chat_model}", api_key=self.api_key)
        dspy.configure(lm=lm)
        self._dspy_configured = True
