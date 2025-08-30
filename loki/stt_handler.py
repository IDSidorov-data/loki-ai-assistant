# loki/stt_handler.py

import whisper
import logging
import os
from typing import Optional

# Убираем импорт torch_directml, он здесь больше не нужен


class WhisperSTT:
    def __init__(
        self, model_name: str = "small"
    ):  # Возвращаемся на small, это лучший баланс для CPU
        logging.info(f"Loading Whisper STT model '{model_name}'...")

        # --- ФИНАЛЬНОЕ ИСПРАВЛЕНИЕ ---
        # Принудительно запускаем Whisper на CPU.
        # Это избегает всех багов несовместимости с DirectML.
        device = "cpu"
        logging.info(f"Whisper will use CPU.")
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

        self.model = whisper.load_model(model_name, device=device)
        logging.info("Whisper STT model loaded successfully.")

    def transcribe(self, audio_path: str) -> Optional[str]:
        if not os.path.exists(audio_path):
            logging.error(f"Audio file not found at: {audio_path}")
            return None

        try:
            logging.info(f"Transcribing audio file: {audio_path}")
            # fp16=False обязательно для работы на CPU
            result = self.model.transcribe(audio_path, fp16=False)
            text = result.get("text")
            logging.info(f"Transcription result: '{text}'")
            return text
        except Exception as e:
            logging.error(f"An error occurred during transcription: {e}", exc_info=True)
            return None
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)
