# loki/tts_handler.py

import logging
import sounddevice as sd
import numpy as np
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

    def speak(self, text: str):
        """
        Синтезирует и воспроизводит речь (блокирующий метод).
        Используется для надежного воспроизведения после полной генерации ответа.
        """
        try:
            logging.info(f"Synthesizing speech for: '{text}'")
            audio_generator = self.voice.synthesize(text)
            audio_chunks = [chunk.audio_int16_bytes for chunk in audio_generator]
            audio_bytes = b"".join(audio_chunks)

            if not audio_bytes:
                logging.warning("Synthesis resulted in empty audio. Nothing to play.")
                return

            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            logging.info("Playing synthesized audio...")
            sd.play(audio_array, samplerate=self.sample_rate)
            sd.wait()
            logging.info("Playback finished.")
        except Exception as e:
            logging.error(
                f"An error occurred during Piper TTS synthesis or playback: {e}",
                exc_info=True,
            )

    async def stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Синтезирует речь и отдает аудиоданные по частям (чанками).
        (Временно не используется в пользу более надежного метода speak)
        """
        try:
            logging.info(f"Streaming speech for: '{text}'")
            audio_generator = self.voice.synthesize(text)

            for chunk in audio_generator:
                yield chunk.audio_int16_bytes

        except Exception as e:
            logging.error(
                f"An error occurred during Piper TTS synthesis stream: {e}",
                exc_info=True,
            )
