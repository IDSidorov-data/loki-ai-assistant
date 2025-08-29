# loki/command_parser.py

import re
from typing import List

# Паттерн обновлен: \s+ соответствует одному или нескольким пробельным символам (пробел, таб, и т.д.)
# Это делает парсер устойчивым к лишним пробелам.
COMMAND_PATTERN = re.compile(r"set\s+status\s+to\s+(\w+)", re.IGNORECASE)


def find_commands(text: str) -> List[str]:
    """
    Ищет предопределенные команды в тексте с помощью скомпилированного
    регулярного выражения. Устойчив к множественным пробелам.
    Пример: "LOKI, set status to processing" -> вернет ["processing"]
    """
    if not isinstance(text, str):
        return []

    return COMMAND_PATTERN.findall(text)
