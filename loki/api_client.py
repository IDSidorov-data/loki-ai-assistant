# loki/api_client.py

import os
import requests
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Загружаем переменные из .env файла в окружение.
# Это нужно сделать до того, как мы попытаемся получить доступ к переменным.
load_dotenv()


class OllamaClient:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.getenv(
            "OLLAMA_API_URL", "http://localhost:8080/api"
        )

        # Получаем токен из переменных окружения
        self.token = os.getenv("LOKI_OLLAMA_TOKEN")
        if not self.token:
            # Если токен не найден, программа не сможет работать.
            # Вызываем исключение с понятным сообщением об ошибке.
            raise ValueError(
                "LOKI_OLLAMA_TOKEN не найден в переменных окружения. "
                "Пожалуйста, создайте .env файл в корне проекта и добавьте его."
            )

        self.last_processed_message_id: Optional[str] = None
        logging.info(f"OllamaClient initialized for base URL: {self.base_url}")

    def get_latest_user_message(self) -> Optional[Dict[str, Any]]:
        try:
            # Используем токен, сохраненный в self.token при инициализации
            headers = {
                "Cookie": f"token={self.token}",
                "Authorization": f"Bearer {self.token}",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36 Edg/139.0.0.0",
            }

            # Этап 1: Получаем список чатов (GET)
            list_chats_url = f"{self.base_url}/v1/chats/"
            response = requests.get(list_chats_url, headers=headers, timeout=5)
            response.raise_for_status()
            chats_summary = response.json()

            if not chats_summary:
                return None

            # Этап 2: Получаем детали последнего чата (GET)
            latest_chat_id = chats_summary[0].get("id")
            if not latest_chat_id:
                return None

            chat_details_url = f"{self.base_url}/v1/chats/{latest_chat_id}"
            details_response = requests.get(
                chat_details_url, headers=headers, timeout=5
            )
            details_response.raise_for_status()

            latest_chat_details = details_response.json()
            messages = latest_chat_details.get("chat", {}).get("messages", [])

            if not messages:
                return None

            # Этап 3: Ищем новое сообщение
            latest_user_message = None
            for message in reversed(messages):
                if message.get("role") == "user":
                    latest_user_message = message
                    break

            if not latest_user_message:
                return None

            message_id = latest_user_message.get("id")
            if message_id != self.last_processed_message_id:
                self.last_processed_message_id = message_id
                logging.info(f"New user message detected (ID: {message_id})")
                return latest_user_message
            else:
                return None

        except Exception as e:
            logging.error(f"An error occurred in OllamaClient: {e}", exc_info=True)
            return None
