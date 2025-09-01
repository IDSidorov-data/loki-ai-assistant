# loki/stt_handler.py
import whisper
import logging
import os
from typing import Optional
from .utils import time_it  # <<< Убедитесь, что этот импорт есть


class WhisperSTT:
    def __init__(self, model_name: str = "base"):
        # ... (код __init__ без изменений)
        logging.info(f"Loading Whisper STT model '{model_name}'...")
        device = "cpu"
        logging.info(f"Whisper will use CPU.")
        self.model = whisper.load_model(model_name, device=device)
        logging.info("Whisper STT model loaded successfully.")

    @time_it  # <<< Убедитесь, что этот декоратор здесь
    def transcribe(self, audio_path: str) -> Optional[str]:
        # ... (код transcribe без изменений)
        if not os.path.exists(audio_path):
            logging.error(f"Audio file not found at: {audio_path}")
            return None
        try:
            logging.info(f"Transcribing audio file: {audio_path}")
            result = self.model.transcribe(audio_path, fp16=False)
            text = result.get("text", "").strip()
            logging.info(f"Transcription result: '{text}'")
            return text
        except Exception as e:
            logging.error(f"An error occurred during transcription: {e}", exc_info=True)
            return None
        finally:
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception as e:
                    logging.error(
                        f"Failed to remove temporary audio file {audio_path}: {e}"
                    )
