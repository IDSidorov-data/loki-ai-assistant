# loki/command_parser.py

import re
import json
import logging
from typing import Optional, Dict, Any, Tuple

COMMAND_JSON_PATTERN = re.compile(r"\[CMD\](.*?)\[/CMD\]", re.DOTALL)


def parse_llm_response(text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    if not isinstance(text, str):
        return "", None

    command_json = None
    match = COMMAND_JSON_PATTERN.search(text)

    if match:
        json_str = match.group(1).strip()
        try:
            command_json = json.loads(json_str)
        except json.JSONDecodeError:
            logging.error(f"Failed to parse command JSON: {json_str}")
            command_json = None

    clean_text = COMMAND_JSON_PATTERN.sub("", text).strip()

    return clean_text, command_json
