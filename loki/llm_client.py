import httpx
import logging
import os
from typing import Optional
from .prompts import SYSTEM_PROMPT


class OllamaLLMClient:
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "llama3:8b-instruct-q4_k_m")
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

    async def get_response(self, user_prompt: str) -> Optional[str]:
        try:
            response = await self.async_client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "system": SYSTEM_PROMPT,
                    "prompt": user_prompt,
                    "stream": False,
                    "options": {"num_gpu": 99},
                },
            )
            response.raise_for_status()
            response_data = response.json()
            return response_data.get("response")
        except httpx.RequestError as e:
            logging.error(f"Could not connect to Ollama server: {e}")
            return "Произошла ошибка при подключении к языковой модели. Пожалуйста, проверьте, запущен ли сервер Ollama."
        except Exception as e:
            logging.error(
                f"An unexpected error occurred in LLM client: {e}", exc_info=True
            )
            return "Произошла непредвиденная ошибка при обработке вашего запроса."

    async def close(self):
        await self.async_client.aclose()
