# loki/stt_handler.py
"""
Обработчик Speech-to-Text (STT).

Предоставляет класс-обертку над библиотекой `openai-whisper` для
преобразования аудиофайлов в текст.
"""
import whisper
import logging
import os
from typing import Optional
from .utils import time_it


class WhisperSTT:
    """
    Класс для транскрибации аудио с использованием модели Whisper.
    """

    def __init__(self, model_name: str = "base"):
        """
        Инициализирует и загружает указанную модель Whisper.

        Args:
            model_name (str): Название модели Whisper для загрузки (e.g., "base", "small").
        """
        logging.info(f"Loading Whisper STT model '{model_name}'...")
        # Принудительно используем CPU. На Windows с AMD GPU запуск на GPU
        # требует сложной настройки ROCm, которая часто нестабильна.
        # CPU-вариант надежнее для данного проекта.
        device = "cpu"
        logging.info(f"Whisper will use CPU for stability.")
        self.model = whisper.load_model(model_name, device=device)
        logging.info("Whisper STT model loaded successfully.")

    @time_it
    def transcribe(self, audio_path: str) -> Optional[str]:
        """
        Транскрибирует аудиофайл в текст.

        После транскрибации временный аудиофайл автоматически удаляется.

        Args:
            audio_path (str): Путь к аудиофайлу для транскрибации.

        Returns:
            Optional[str]: Распознанный текст или None в случае ошибки.
        """
        if not os.path.exists(audio_path):
            logging.error(f"Audio file not found at: {audio_path}")
            return None
        try:
            logging.info(f"Transcribing audio file: {audio_path}")
            # fp16=False необходимо для работы на CPU.
            result = self.model.transcribe(audio_path, fp16=False)
            text = result.get("text", "").strip()
            logging.info(f"Transcription result: '{text}'")
            return text
        except Exception as e:
            logging.error(f"An error occurred during transcription: {e}", exc_info=True)
            return None
        finally:
            # Блок finally гарантирует, что временный файл будет удален,
            # даже если во время транскрибации произошла ошибка.
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception as e:
                    logging.error(
                        f"Failed to remove temporary audio file {audio_path}: {e}"
                    )
