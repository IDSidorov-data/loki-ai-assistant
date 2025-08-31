# loki/visual_controller.py

import os
import asyncio
import logging
from typing import Dict, Any

# Значение по умолчанию остается здесь
DEFAULT_PATH = (
    "C:/Program Files (x86)/Steam/steamapps/common/wallpaper_engine/wallpaper32.exe"
)


async def set_loki_visual_state_async(status: str):
    """
    АСИНХРОННО отправляет команду в Wallpaper Engine.
    """
    wallpaper_engine_path = os.getenv("WALLPAPER_ENGINE_PATH", DEFAULT_PATH)

    if not os.path.exists(wallpaper_engine_path):
        # Используем f-string для вывода правильного пути в логе
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
        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
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
    if not command or command.get("tool_name") != "set_status":
        return

    parameters = command.get("parameters", {})
    status = parameters.get("status")

    if status:
        logging.info(f"Scheduling visual state update to '{status}'.")
        asyncio.create_task(set_loki_visual_state_async(status))
    else:
        logging.warning("Command 'set_status' received without a 'status' parameter.")
