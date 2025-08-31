# tests/test_command_parser.py

import pytest
from loki.command_parser import parse_llm_response


def test_parse_with_valid_command():
    """
    Тест: Ответ LLM содержит валидный JSON-блок [CMD].
    Ожидание: Текст очищен, JSON успешно распарсен.
    """
    text = 'Конечно, я сделаю это. [CMD]{"tool_name": "set_status", "parameters": {"status": "speaking"}}[/CMD]'
    clean_text, command = parse_llm_response(text)
    assert clean_text == "Конечно, я сделаю это."
    assert command == {"tool_name": "set_status", "parameters": {"status": "speaking"}}


def test_parse_without_command():
    """
    Тест: Ответ LLM не содержит блок [CMD].
    Ожидание: Текст не изменен, команда отсутствует (None).
    """
    text = "Я не могу выполнить эту команду, но я постараюсь помочь."
    clean_text, command = parse_llm_response(text)
    assert clean_text == "Я не могу выполнить эту команду, но я постараюсь помочь."
    assert command is None


def test_parse_with_malformed_json():
    """
    Тест: Ответ LLM содержит блок [CMD] с некорректным JSON.
    Ожидание: Текст очищен, команда отсутствует (None) из-за ошибки парсинга.
    """
    text = 'Хорошо. [CMD]{"tool_name": "set_status", "parameters": {"status": "speaking}}[/CMD]'
    clean_text, command = parse_llm_response(text)
    assert clean_text == "Хорошо."
    assert command is None


def test_parse_with_empty_cmd_block():
    """
    Тест: Ответ LLM содержит пустой блок [CMD].
    Ожидание: Текст очищен, команда отсутствует (None).
    """
    text = "Сделано. [CMD][/CMD]"
    clean_text, command = parse_llm_response(text)
    assert clean_text == "Сделано."
    assert command is None


def test_parse_empty_string_input():
    """
    Тест: На вход подана пустая строка.
    Ожидание: Возвращается пустая строка и None.
    """
    text = ""
    clean_text, command = parse_llm_response(text)
    assert clean_text == ""
    assert command is None


def test_parse_none_input():
    """
    Тест: На вход подан None.
    Ожидание: Возвращается пустая строка и None для защиты от ошибок.
    """
    text = None
    clean_text, command = parse_llm_response(text)
    assert clean_text == ""
    assert command is None


def test_text_with_multiple_cmd_blocks():
    """

    Тест: Ответ LLM содержит несколько блоков [CMD].
    Ожидание: Regex по своей природе "жадный" и найдет только первый блок.
               Текст будет очищен от всех вхождений.
    """
    text = 'Первая команда. [CMD]{"key": "val1"}[/CMD] И вторая. [CMD]{"key": "val2"}[/CMD]'
    clean_text, command = parse_llm_response(text)
    assert clean_text == "Первая команда.  И вторая."
    assert command == {"key": "val1"}  # re.search находит первое вхождение
