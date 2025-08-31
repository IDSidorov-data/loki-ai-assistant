# loki/llm_client.py

import httpx
import logging
import os
import json
from typing import AsyncGenerator, Dict
from enum import Enum, auto

from .prompts import SYSTEM_PROMPT
from loki import config


class LlmParseState(Enum):
    """Представляет текущее состояние парсера потока от LLM."""

    SEARCHING = auto()
    IN_THOUGHTS = auto()
    IN_ANSWER = auto()


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

    async def stream_response(
        self, user_prompt: str
    ) -> AsyncGenerator[Dict[str, str], None]:
        """
        Отправляет запрос к Ollama и отдает поток типизированных данных
        (`thought` или `answer`), парся теги на лету с помощью конечного автомата.
        """
        state = LlmParseState.SEARCHING
        buffer = ""
        tags_found = False
        # Окно поиска для предотвращения разрезания тегов, приходящих в разных пакетах
        TAG_SEARCH_WINDOW = 20

        try:
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
                async for line in response.aiter_lines():
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if not token:
                            continue

                        buffer += token

                        # Обработка буфера конечным автоматом
                        while True:
                            original_buffer = buffer

                            if state == LlmParseState.SEARCHING:
                                if "<мысли>" in buffer:
                                    tags_found = True
                                    parts = buffer.split("<мысли>", 1)
                                    # Все, что было до тега (если было) - это мусор, игнорируем
                                    buffer = parts[1]
                                    state = LlmParseState.IN_THOUGHTS
                                elif "<ответ>" in buffer:
                                    tags_found = True
                                    parts = buffer.split("<ответ>", 1)
                                    buffer = parts[1]
                                    state = LlmParseState.IN_ANSWER

                            elif state == LlmParseState.IN_THOUGHTS:
                                if "</мысли>" in buffer:
                                    parts = buffer.split("</мысли>", 1)
                                    if parts[0]:
                                        yield {"type": "thought", "content": parts[0]}
                                    buffer = parts[1]
                                    state = LlmParseState.SEARCHING
                                else:
                                    # Отдаем безопасную часть, оставляя хвост для поиска закрывающего тега
                                    safe_point = max(0, len(buffer) - TAG_SEARCH_WINDOW)
                                    if safe_point > 0:
                                        yield {
                                            "type": "thought",
                                            "content": buffer[:safe_point],
                                        }
                                        buffer = buffer[safe_point:]
                                    break  # Ждем еще токенов

                            elif state == LlmParseState.IN_ANSWER:
                                if "</ответ>" in buffer:
                                    parts = buffer.split("</ответ>", 1)
                                    if parts[0]:
                                        yield {"type": "answer", "content": parts[0]}
                                    buffer = parts[1]
                                    state = LlmParseState.SEARCHING
                                else:
                                    safe_point = max(0, len(buffer) - TAG_SEARCH_WINDOW)
                                    if safe_point > 0:
                                        yield {
                                            "type": "answer",
                                            "content": buffer[:safe_point],
                                        }
                                        buffer = buffer[safe_point:]
                                    break  # Ждем еще токенов

                            # Если буфер не изменился за итерацию, выходим из while
                            if buffer == original_buffer:
                                break

                    except json.JSONDecodeError:
                        logging.warning(f"Failed to decode JSON line: {line}")

        except httpx.RequestError as e:
            logging.error(f"Could not connect to Ollama server: {e}")
            yield {
                "type": "answer",
                "content": "Произошла ошибка при подключении к языковой модели.",
            }
        except Exception as e:
            logging.error(
                f"An unexpected error occurred in LLM client stream: {e}", exc_info=True
            )
            yield {"type": "answer", "content": "Произошла непредвиденная ошибка."}
        finally:
            # Отдаем все, что осталось в буфере после завершения
            if buffer:
                final_type = "answer" if state == LlmParseState.IN_ANSWER else "thought"
                yield {"type": final_type, "content": buffer}

            # Fallback: если модель нарушила протокол и не прислала теги,
            # считаем весь ответ "ответом".
            if not tags_found:
                # Собираем весь "сырой" ответ из буфера, который накопился
                full_raw_response = buffer
                if full_raw_response:
                    yield {"type": "answer", "content": full_raw_response}

    async def close(self):
        await self.async_client.aclose()
