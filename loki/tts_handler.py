# loki/tts_handler.py
"""
Обработчик Text-to-Speech (TTS).

Предоставляет класс-обертку над библиотекой `piper-tts` для синтеза речи из текста.
Поддерживает как блокирующее, так и потоковое воспроизведение для максимальной
гибкости и отзывчивости.
"""
import logging
import sounddevice as sd
import numpy as np
from piper.voice import PiperVoice
import os
from typing import AsyncGenerator


class Piper_Engine:
    """
    Класс для синтеза речи с использованием движка Piper TTS.
    """

    def __init__(self, model_path: str):
        """
        Инициализирует и загружает голосовую модель Piper.

        Args:
            model_path (str): Путь к файлу голосовой модели `.onnx`.

        Raises:
            FileNotFoundError: Если файл модели по указанному пути не найден.
        """
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

        Этот метод сначала полностью синтезирует аудио, а затем воспроизводит его.
        Идеально подходит для коротких, критически важных фраз (например, сообщений
        об ошибках), где прерывание нежелательно.

        Args:
            text (str): Текст для озвучки.
        """
        try:
            logging.info(f"Synthesizing speech for: '{text}'")
            audio_generator = self.voice.synthesize(text)
            # Собираем все аудио-чанки в один байтовый массив
            audio_bytes = b"".join(chunk.audio_int16_bytes for chunk in audio_generator)

            if not audio_bytes:
                logging.warning("Synthesis resulted in empty audio. Nothing to play.")
                return

            # Преобразуем байты в массив numpy для воспроизведения
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            logging.info("Playing synthesized audio...")
            sd.play(audio_array, samplerate=self.sample_rate)
            sd.wait()  # Блокируем выполнение до окончания воспроизведения
            logging.info("Playback finished.")
        except Exception as e:
            logging.error(
                f"An error occurred during Piper TTS synthesis or playback: {e}",
                exc_info=True,
            )

    async def stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Асинхронно синтезирует речь и отдает аудиоданные по частям (чанками).

        Этот метод является асинхронным генератором, который позволяет начать
        воспроизведение аудио до того, как вся фраза будет синтезирована.
        Идеально для длинных ответов LLM, чтобы минимизировать задержку
        и улучшить воспринимаемую отзывчивость ассистента.

        Args:
            text (str): Текст для синтеза.

        Yields:
            bytes: Сырые аудиоданные (чанки) в формате int16.
        """
        try:
            audio_generator = self.voice.synthesize(text)
            for chunk in audio_generator:
                if chunk.audio_int16_bytes:
                    yield chunk.audio_int16_bytes
        except Exception as e:
            logging.error(
                f"An error occurred during Piper TTS synthesis stream: {e}",
                exc_info=True,
            )
