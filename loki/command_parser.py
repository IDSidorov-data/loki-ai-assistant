# loki/command_parser.py

import re
import json
import logging
from typing import Optional, Dict, Any, Tuple

COMMAND_JSON_PATTERN = re.compile(r"\[CMD\](.*?)\[/CMD\]", re.DOTALL)


def parse_llm_response(text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Ищет в тексте блок [CMD] и извлекает из него JSON.
    Возвращает текст без этого блока и сам JSON-объект.
    """
    if not isinstance(text, str):
        return "", None

    command_json = None
    match = COMMAND_JSON_PATTERN.search(text)

    if match:
        json_str = match.group(1).strip()
        try:
            # Простое "самоисцеление" JSON
            json_str_healed = re.sub(
                r"([{,]\s*)(\w+)(\s*:)", r'\1"\2"\3', json_str.replace("'", '"')
            )
            command_json = json.loads(json_str_healed)
        except json.JSONDecodeError:
            logging.error(f"Failed to parse command JSON: {json_str}")
            command_json = None

    text_to_speak = COMMAND_JSON_PATTERN.sub("", text).strip()
    return text_to_speak, command_json
