# tests/test_command_parser.py

import pytest
from loki.command_parser import find_commands


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        # Стандартный случай
        ("LOKI, please set status to processing", ["processing"]),
        # Другой регистр
        ("loki, SET STATUS TO idle now", ["idle"]),
        # Команда отсутствует
        ("LOKI, what is the current status?", []),
        # Несколько команд
        (
            "First, set status to listening, then later set status to speaking.",
            ["listening", "speaking"],
        ),
        # Пустая строка
        ("", []),
        # Команда в начале строки
        ("set status to idle", ["idle"]),
        # Тест на лишние пробелы (теперь должен проходить)
        ("  set   status   to   processing  ", ["processing"]),
    ],
)
def test_find_commands_valid_strings(input_text, expected_output):
    """
    Проверяет функцию find_commands с различными валидными строковыми входными данными.
    """
    assert find_commands(input_text) == expected_output


@pytest.mark.parametrize(
    "invalid_input",
    [
        None,
        12345,
        ["list", "of", "strings"],
        {"a": "dict"},
    ],
)
def test_find_commands_invalid_input_types(invalid_input):
    """
    Проверяет, что функция корректно обрабатывает нестроковые входные данные,
    возвращая пустой список без вызова исключений.
    """
    assert find_commands(invalid_input) == []
