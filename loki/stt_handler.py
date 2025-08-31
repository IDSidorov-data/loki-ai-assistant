# loki/stt_handler.py

import whisper
import logging
import os
from typing import Optional


class WhisperSTT:
    def __init__(self, model_name: str = "base"):
        """
        Инициализирует движок Whisper STT для работы на CPU.
        Модель 'base' выбрана как оптимальный баланс скорости и точности для русского языка.
        """
        logging.info(f"Loading Whisper STT model '{model_name}'...")

        # Принудительно запускаем Whisper на CPU, так как это самый стабильный вариант.
        device = "cpu"
        logging.info(f"Whisper will use CPU.")

        self.model = whisper.load_model(model_name, device=device)
        logging.info("Whisper STT model loaded successfully.")

    def transcribe(self, audio_path: str) -> Optional[str]:
        """
        Распознает речь из аудиофайла.
        """
        if not os.path.exists(audio_path):
            logging.error(f"Audio file not found at: {audio_path}")
            return None

        try:
            logging.info(f"Transcribing audio file: {audio_path}")
            # fp16=False обязательно для стабильной работы на CPU
            result = self.model.transcribe(audio_path, fp16=False)
            text = result.get("text", "").strip()
            logging.info(f"Transcription result: '{text}'")
            return text
        except Exception as e:
            logging.error(f"An error occurred during transcription: {e}", exc_info=True)
            return None
        finally:
            # Этот блок гарантирует, что временный файл будет удален, даже если произошла ошибка
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception as e:
                    logging.error(
                        f"Failed to remove temporary audio file {audio_path}: {e}"
                    )
