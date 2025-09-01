# loki/command_parser.py
"""
Парсер ответа от языковой модели (LLM).

Основная задача - извлечь из текстового ответа LLM специальный блок [CMD]...[/CMD],
содержащий JSON-объект с командой для выполнения, и отделить его от текста,
предназначенного для озвучки пользователю.
"""
import re
import json
import logging
from typing import Optional, Dict, Any, Tuple

# Регулярное выражение для поиска блока [CMD]...[/CMD] и захвата его содержимого
COMMAND_JSON_PATTERN = re.compile(r"\\[CMD\\](.*?)\\[/CMD\\]", re.DOTALL)


def parse_llm_response(text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Ищет в тексте блок [CMD], извлекает из него JSON и очищает текст.

    Args:
        text (str): Сырой текстовый ответ от LLM.

    Returns:
        Tuple[str, Optional[Dict[str, Any]]]:
        - Первым элементом идет очищенный текст для озвучки.
        - Вторым элементом идет распарсенный JSON-объект команды или None,
          если команда не найдена или произошла ошибка парсинга.
    """
    if not isinstance(text, str):
        return "", None

    command_json = None
    match = COMMAND_JSON_PATTERN.search(text)

    if match:
        json_str = match.group(1).strip()
        try:
            # Простое "самоисцеление" JSON: LLM иногда генерирует JSON с ключами без кавычек.
            # Эта замена добавляет кавычки к ключам, делая парсер более устойчивым.
            # Пример: {tool_name: "set_status"} -> {"tool_name": "set_status"}
            json_str_healed = re.sub(
                r"([,{]\s*)(\w+)(\s*:)", r'\1"\2"\3', json_str.replace("'", '"')
            )
            command_json = json.loads(json_str_healed)
        except json.JSONDecodeError:
            logging.error(f"Failed to parse command JSON: {json_str}")
            command_json = None

    # Удаляем блок [CMD] из текста, который будет озвучен
    text_to_speak = COMMAND_JSON_PATTERN.sub("", text).strip()
    return text_to_speak, command_json
