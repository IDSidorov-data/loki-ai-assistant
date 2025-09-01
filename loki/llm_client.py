# loki/llm_client.py
import httpx
import logging
import os
import json
from typing import AsyncGenerator
from .prompts import SYSTEM_PROMPT


class OllamaLLMClient:
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "my-llama3")
        self.async_client = httpx.AsyncClient(timeout=120.0)
        self.system_prompt = SYSTEM_PROMPT
        self._load_model_on_startup()

    def _load_model_on_startup(self):
        try:
            with httpx.Client(timeout=300.0) as client:
                client.post(f"{self.base_url}/api/show", json={"name": self.model})
        except Exception:
            pass

    async def stream_response(self, user_prompt: str) -> AsyncGenerator[str, None]:
        try:
            async with self.async_client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "system": self.system_prompt,
                    "prompt": user_prompt,
                    "stream": True,
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            token = data.get("response", "")
                            if token:
                                yield token
                            if data.get("done"):
                                break
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            yield "Произошла ошибка."

    def close(self):
        self.async_client.close()
