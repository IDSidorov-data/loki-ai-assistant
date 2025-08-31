# loki/tts_handler.py

import logging
from piper.voice import PiperVoice
import os
from typing import AsyncGenerator


class Piper_Engine:
    def __init__(self, model_path: str):
        logging.info("Loading Piper TTS model...")
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Piper model file not found at: {model_path}")

        self.voice = PiperVoice.load(model_path)
        self.sample_rate = self.voice.config.sample_rate
        logging.info(
            f"Piper TTS model loaded successfully. Sample rate: {self.sample_rate} Hz"
        )

    async def stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Синтезирует речь и отдает аудиоданные по частям (чанками).
        Это позволяет начать воспроизведение до того, как вся фраза будет сгенерирована.
        """
        try:
            logging.info(f"Streaming speech for: '{text}'")
            # audio_generator - это ленивый итератор, который генерирует чанки по требованию
            audio_generator = self.voice.synthesize(text)

            for chunk in audio_generator:
                # Мы не копим чанки, а сразу отдаем их наружу
                yield chunk.audio_int16_bytes

        except Exception as e:
            logging.error(
                f"An error occurred during Piper TTS synthesis stream: {e}",
                exc_info=True,
            )
