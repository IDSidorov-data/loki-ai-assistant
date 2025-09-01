# loki/llm_client.py
"""
Клиент для взаимодействия с API Ollama.

Этот модуль предоставляет класс `OllamaLLMClient`, который инкапсулирует
логику отправки запросов к языковой модели, работающей через Ollama.
Он поддерживает потоковое получение ответа для минимизации задержек.
"""
import httpx
import logging
import os
import json
from typing import AsyncGenerator

class OllamaLLMClient:
    """
    Асинхронный клиент для потокового получения ответов от LLM через Ollama.

    Атрибуты:
        base_url (str): URL-адрес сервера Ollama.
        model (str): Название модели, которая будет использоваться для генерации.
        async_client (httpx.AsyncClient): Асинхронный HTTP-клиент для выполнения запросов.
    """
    def __init__(self):
        """Инициализирует клиент Ollama, загружая конфигурацию из переменных окружения."""
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "my-llama3")
        self.async_client = httpx.AsyncClient(timeout=120.0)
        self._load_model_on_startup()

    def _load_model_on_startup(self):
        """
        Отправляет запрос к Ollama, чтобы убедиться, что модель загружена в память.
        Это помогает избежать задержки при первом реальном запросе после запуска.
        Использует синхронный клиент, так как выполняется один раз при инициализации.
        """
        try:
            # Используем стандартный httpx.Client для одноразового синхронного вызова
            with httpx.Client(timeout=300.0) as client:
                # /api/show - это endpoint Ollama для получения информации о модели.
                # Сам факт запроса заставляет Ollama подгрузить модель, если она не в памяти.
                client.post(f"{self.base_url}/api/show", json={"name": self.model})
        except Exception:
            # Игнорируем ошибки, так как это не критичная для запуска операция.
            # Если Ollama не доступен, основная логика обработает это позже.
            pass

    async def stream_response(self, user_prompt: str, system_prompt: str) -> AsyncGenerator[str, None]:
        """
        Отправляет запрос к LLM и асинхронно возвращает ответ в виде потока токенов.

        Args:
            user_prompt (str): Запрос от пользователя.
            system_prompt (str): Системный промпт, определяющий поведение модели.

        Yields:
            str: Отдельные токены (слова или части слов) по мере их генерации моделью.
        """
        try:
            # Используем асинхронный менеджер контекста для потокового запроса
            async with self.async_client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "system": system_prompt,
                    "prompt": user_prompt,
                    "stream": True  # Ключевой параметр для включения потоковой передачи
                }
            ) as response:
                response.raise_for_status()  # Вызовет исключение для кодов ошибок 4xx/5xx
                # Асинхронно итерируемся по строкам ответа
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            token = data.get("response", "")
                            if token:
                                yield token
                            # Ollama сигнализирует о конце генерации полем "done": true
                            if data.get("done"):
                                break
                        except json.JSONDecodeError:
                            # Игнорируем строки, которые не являются валидным JSON
                            pass
        except Exception as e:
            logging.error(f"LLM stream error: {e}")
            yield "Произошла ошибка."

    def close(self):
        """Закрывает сессию HTTP-клиента. Должен вызываться при завершении работы."""
        self.async_client.close()