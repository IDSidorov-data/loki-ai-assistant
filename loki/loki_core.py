# loki/loki_core.py

import os
import time
import requests
import logging
from loki.api_client import OllamaClient
from loki.command_parser import find_commands

# --- Конфигурация ---
# Получаем параметры из переменных окружения с разумными значениями по умолчанию
POLLING_INTERVAL_SECONDS = int(os.getenv("LOKI_POLLING_INTERVAL", "2"))
VISUAL_SERVER_URL = os.getenv(
    "LOKI_VISUAL_SERVER_URL", "http://127.0.0.1:8000/set_state"
)
LOG_LEVEL = os.getenv("LOKI_LOG_LEVEL", "INFO").upper()

# Настройка логирования
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")


def update_visual_state(status: str):
    """Отправляет команду на обновление состояния в visual_server."""
    try:
        response = requests.post(VISUAL_SERVER_URL, json={"status": status}, timeout=3)
        if response.status_code == 200:
            logging.info(f"State updated successfully to: {status}")
        else:
            logging.warning(
                f"Failed to update state. Server responded with {response.status_code}: {response.text}"
            )
    except requests.exceptions.RequestException as e:
        logging.error(f"Could not connect to visual_server: {e}")


def main_loop():
    """Главный цикл оркестратора LOKI."""
    client = OllamaClient()
    logging.info("LOKI Core started. Initializing state to 'idle'.")

    update_visual_state("idle")

    while True:
        try:
            logging.debug("Polling API for new messages...")

            message = client.get_latest_user_message()

            if message and "content" in message:
                logging.info(f"Processing message content: \"{message['content']}\"")

                commands = find_commands(message["content"])

                if commands:
                    new_state = commands[0]
                    logging.info(f"Command found. New state to set: '{new_state}'")
                    update_visual_state(new_state)
                else:
                    logging.info("No commands found in the message.")

            time.sleep(POLLING_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logging.info("Shutdown signal received. Exiting LOKI Core.")
            break
        except Exception as e:
            logging.error(
                f"An unexpected error occurred in the main loop: {e}", exc_info=True
            )
            time.sleep(POLLING_INTERVAL_SECONDS * 2)


if __name__ == "__main__":
    main_loop()
