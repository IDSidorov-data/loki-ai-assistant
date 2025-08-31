# loki/command_parser.py

import re
import json
import logging
from typing import Optional, Dict, Any, Tuple

COMMAND_JSON_PATTERN = re.compile(r"\[CMD\](.*?)\[/CMD\]", re.DOTALL)
RESPONSE_PATTERN = re.compile(r"<ответ>(.*?)</ответ>", re.DOTALL)


def extract_clean_response(raw_text: str) -> str:
    """
    Извлекает чистый текстовый ответ из тегов <ответ>.
    Это основной метод для очистки ответа от "мыслей" модели.
    """
    if not isinstance(raw_text, str):
        return ""

    match = RESPONSE_PATTERN.search(raw_text)
    if match:
        return match.group(1).strip()

    # Fallback: если модель нарушила формат и не выдала теги <ответ>,
    # пытаемся найти хотя бы теги <мысли> и убрать их.
    # Если и их нет, возвращаем исходный текст.
    if "<мысли>" in raw_text:
        return raw_text.split("</мысли>")[-1].strip()

    return raw_text.strip()


def parse_llm_response(clean_text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Парсит чистый текст ответа, ищет в нем [CMD] блок и извлекает JSON.
    """
    if not isinstance(clean_text, str):
        return "", None

    command_json = None
    match = COMMAND_JSON_PATTERN.search(clean_text)

    if match:
        json_str = match.group(1).strip()
        try:
            # Этап "самоисцеления" JSON перед парсингом

            # 1. Заменяем одинарные кавычки на двойные
            json_str_healed = json_str.replace("'", '"')

            # 2. Добавляем кавычки к ключам без них (например, {key: "value"})
            # Это сложное регулярное выражение ищет ключи без кавычек и добавляет их
            json_str_healed = re.sub(
                r"([{,]\s*)(\w+)(\s*:)", r'\1"\2"\3', json_str_healed
            )

            command_json = json.loads(json_str_healed)
            logging.info(f"Successfully parsed command JSON: {command_json}")

        except json.JSONDecodeError:
            logging.error(
                f"Failed to parse command JSON even after healing: {json_str}"
            )
            command_json = None

    # Убираем сам [CMD] блок из текста, который будет озвучен
    text_to_speak = COMMAND_JSON_PATTERN.sub("", clean_text).strip()

    return text_to_speak, command_json
