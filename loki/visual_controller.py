# loki/visual_controller.py
"""
Контроллер визуального состояния LOKI.

Этот модуль отвечает за отправку команд для изменения внешнего вида ассистента.
В данной реализации он управляет свойствами живых обоев в Wallpaper Engine,
вызывая его исполняемый файл с определенными параметрами командной строки.
"""
import os
import asyncio
import logging
from typing import Dict, Any

from loki import config


async def set_loki_visual_state_async(status: str):
    """
    Асинхронно запускает процесс Wallpaper Engine для установки свойства.

    Формирует и выполняет команду в командной строке для изменения
    свойства `loki_state` в активных обоях. Выполняется асинхронно,
    чтобы не блокировать основной поток приложения.

    Args:
        status (str): Новое значение статуса (e.g., "idle", "listening", "speaking").
    """
    wallpaper_engine_path = os.getenv(
        "WALLPAPER_ENGINE_PATH", config.DEFAULT_WALLPAPER_ENGINE_PATH
    )
    if not os.path.exists(wallpaper_engine_path):
        logging.warning(
            f"Wallpaper Engine not found at: {wallpaper_engine_path}. Visuals disabled."
        )
        return

    command = [
        wallpaper_engine_path,
        "-control",
        "setProperty",
        "-property",
        "loki_state",
        "-value",
        str(status),
    ]
    try:
        # Запускаем внешний процесс, не блокируя основной event loop
        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        # Ожидаем завершения процесса и получаем его вывод
        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            logging.info(f"Successfully set loki_state to '{status}'.")
        else:
            logging.error(f"Command failed with exit code {process.returncode}.")
            if stderr:
                logging.error(f"Stderr: {stderr.decode().strip()}")
    except Exception as e:
        logging.error(
            f"An unexpected error occurred while running subprocess: {e}", exc_info=True
        )


def handle_visual_command(command: Dict[str, Any]):
    """
    Обрабатывает команду и, если это команда смены статуса, запускает ее выполнение.

    Эта функция является диспетчером: она проверяет, является ли команда
    командой `set_status`, и если да, создает фоновую задачу для ее асинхронного
    выполнения. Это позволяет основному циклу продолжать работу без ожидания.

    Args:
        command (Dict[str, Any]): Распарсенный JSON-объект команды от LLM.
    """
    if not command or command.get("tool_name") != "set_status":
        return

    parameters = command.get("parameters", {})
    status = parameters.get("status")

    if status:
        logging.info(f"Scheduling visual state update to '{status}'.")
        # asyncio.create_task() запускает корутину в фоновом режиме,
        # не блокируя текущий поток выполнения.
        asyncio.create_task(set_loki_visual_state_async(status))
    else:
        logging.warning("Command 'set_status' received without a 'status' parameter.")
