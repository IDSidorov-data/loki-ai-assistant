import httpx
import logging
import os
import json
from typing import Optional, AsyncGenerator
from .prompts import SYSTEM_PROMPT
from loki import config


class OllamaLLMClient:
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", config.DEFAULT_OLLAMA_BASE_URL)
        self.model = os.getenv("OLLAMA_MODEL", config.DEFAULT_OLLAMA_MODEL)
        self.async_client = httpx.AsyncClient(timeout=120.0)
        logging.info(
            f"OllamaLLMClient initialized for model '{self.model}' at {self.base_url}"
        )
        self._load_model_on_startup()

    def _load_model_on_startup(self):
        logging.info(f"Pre-loading model '{self.model}' onto the GPU...")
        try:
            with httpx.Client(timeout=300.0) as client:
                response = client.post(
                    f"{self.base_url}/api/show", json={"name": self.model}
                )
                response.raise_for_status()
            logging.info(f"Model '{self.model}' is pre-loaded and ready on the GPU.")
        except Exception as e:
            logging.error(
                f"An error occurred while pre-loading the model: {e}", exc_info=True
            )

    async def stream_response(self, user_prompt: str) -> AsyncGenerator[str, None]:
        """
        Отправляет запрос к Ollama и получает ответ в виде потока токенов.
        """
        try:
            # Открываем асинхронный стриминговый запрос
            async with self.async_client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "system": SYSTEM_PROMPT,
                    "prompt": user_prompt,
                    "stream": True,
                },
            ) as response:
                response.raise_for_status()
                # Асинхронно итерируемся по строкам ответа
                async for line in response.aiter_lines():
                    if line:
                        try:
                            # Каждая строка - это JSON-объект
                            data = json.loads(line)
                            # Извлекаем сам токен (часть слова)
                            token = data.get("response", "")
                            if token:
                                yield token
                            # Если модель закончила генерацию, выходим
                            if data.get("done"):
                                break
                        except json.JSONDecodeError:
                            logging.warning(f"Failed to decode JSON line: {line}")
        except httpx.RequestError as e:
            logging.error(f"Could not connect to Ollama server: {e}")
            yield "Произошла ошибка при подключении к языковой модели."
        except Exception as e:
            logging.error(
                f"An unexpected error occurred in LLM client stream: {e}", exc_info=True
            )
            yield "Произошла непредвиденная ошибка."

    async def close(self):
        await self.async_client.aclose()
