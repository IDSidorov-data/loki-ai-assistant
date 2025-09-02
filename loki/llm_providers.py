# loki/llm_providers.py
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any
import httpx
import logging
import os
import json
import google.generativeai as genai


class LLMProvider(ABC):
    """Абстрактный базовый класс для всех провайдеров языковых моделей."""

    @abstractmethod
    async def stream_response(
        self, user_prompt: str, system_prompt: str
    ) -> AsyncGenerator[str, None]:
        """Отправляет запрос к LLM и асинхронно возвращает ответ в виде потока токенов."""
        pass

    def close(self):
        """Закрывает соединения, если это необходимо. Может быть переопределен."""
        pass


class OllamaProvider(LLMProvider):
    """Провайдер для работы с локальными моделями через Ollama."""

    def __init__(self):
        """Инициализирует клиент Ollama."""
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv(
            "OLLAMA_MODEL", "mistral"
        )  # Значение по умолчанию, если в .env нет
        self.async_client = httpx.AsyncClient(timeout=120.0)
        logging.info(
            f"Ollama Provider initialized with model: {self.model} at {self.base_url}"
        )

    async def stream_response(
        self, user_prompt: str, system_prompt: str
    ) -> AsyncGenerator[str, None]:
        """
        Отправляет запрос к LLM и асинхронно возвращает ответ в виде потока токенов.
        """
        try:
            async with self.async_client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "system": system_prompt,
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
            logging.error(f"LLM stream error: {e}")
            yield "Произошла ошибка при работе с локальным сервисом."

    def close(self):
        """Закрывает сессию HTTP-клиента."""
        # Используем try-except на случай, если клиент уже закрыт
        try:
            if not self.async_client.is_closed:
                self.async_client.close()
        except Exception as e:
            logging.warning(f"Error closing httpx client: {e}")


class GoogleAIProvider(LLMProvider):
    """Провайдер для работы с моделями Google через Gemini API."""

    def __init__(self):
        """
        Инициализирует клиент Google AI, загружая все настройки
        из переменных окружения (.env файла).
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY не найден в .env файле. Получите его в Google AI Studio."
            )
        genai.configure(api_key=api_key)

        # 1. Загружаем имя модели из .env, с "gemini-1.5-flash" в качестве значения по умолчанию
        self.model_name = os.getenv("GOOGLE_MODEL_NAME", "gemini-2.5-flash")

        # 2. Загружаем параметры генерации из .env, преобразуя их в нужные типы (float, int)
        try:
            temperature = float(os.getenv("GOOGLE_TEMPERATURE", 0))
            top_p = int(os.getenv("GOOGLE_TOP_P", 1))
            top_k = int(os.getenv("GOOGLE_TOP_K", 1))
            max_output_tokens = int(os.getenv("GOOGLE_MAX_TOKENS", 2048))
        except (ValueError, TypeError) as e:
            logging.error(f"Ошибка при чтении настроек генерации из .env: {e}")
            # Устанавливаем безопасные значения по умолчанию в случае ошибки
            temperature = 0
            top_p = 1
            top_k = 1
            max_output_tokens = 2048

        # 3. Собираем конфигурацию в словарь, который будем использовать при каждом запросе
        self.generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output_tokens": max_output_tokens,
        }

        # 4. Инициализируем модель с именем, загруженным из .env
        self.model = genai.GenerativeModel(self.model_name)

        logging.info(f"Google AI Provider initialized with model: {self.model_name}")
        logging.info(f"Generation config: {self.generation_config}")

    async def stream_response(
        self, user_prompt: str, system_prompt: str
    ) -> AsyncGenerator[str, None]:
        """Отправляет запрос к Gemini API и возвращает потоковый ответ."""
        try:
            # Для Gemini системный промпт лучше передавать отдельно, если API это поддерживает,
            # но для универсальности объединяем его с пользовательским.
            full_prompt = f"{system_prompt}\n\n---ЗАПРОС ПОЛЬЗОВАТЕЛЯ---\n{user_prompt}"

            # Передаем заранее собранную конфигурацию в каждый запрос
            response = await self.model.generate_content_async(
                full_prompt, stream=True, generation_config=self.generation_config
            )
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logging.error(f"Ошибка при работе с Google AI API: {e}")
            yield "Произошла ошибка при обращении к облачному сервису."


def get_llm_provider() -> LLMProvider:
    """
    Фабричная функция, которая создает и возвращает экземпляр LLM-провайдера
    на основе переменной окружения LLM_PROVIDER.
    """
    provider_type = os.getenv("LLM_PROVIDER", "local").lower()

    if provider_type == "google":
        logging.info("Используется облачный провайдер: Google AI.")
        return GoogleAIProvider()
    elif provider_type == "local":
        logging.info("Используется локальный провайдер: Ollama.")
        return OllamaProvider()
    else:
        raise ValueError(
            f"Неизвестный тип провайдера '{provider_type}'. Допустимые значения: 'local', 'google'."
        )
