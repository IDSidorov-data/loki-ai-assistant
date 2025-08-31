# loki/loki_core.py

import sys
import os
import re
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import logging
import struct
import pvporcupine
import pyaudio
import asyncio
from dotenv import load_dotenv
import sounddevice as sd

load_dotenv()

from loki.audio_handler import record_command_vad
from loki.stt_handler import WhisperSTT
from loki.tts_handler import Piper_Engine
from loki.llm_client import OllamaLLMClient
from loki.command_parser import parse_llm_response, extract_clean_response
from loki.visual_controller import handle_visual_command

LOG_LEVEL = os.getenv("LOKI_LOG_LEVEL", "INFO").upper()
PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")
WAKE_WORD = os.getenv("LOKI_WAKE_WORD", "jarvis")
PIPER_VOICE_PATH = os.getenv("LOKI_PIPER_VOICE_PATH")
CUSTOM_WAKE_WORD_PATH = os.getenv("LOKI_CUSTOM_WAKE_WORD_PATH")

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")


class LokiOrchestrator:
    def __init__(self):
        self.stt_engine = WhisperSTT(model_name="base")
        self.tts_engine = Piper_Engine(model_path=PIPER_VOICE_PATH)
        self.llm_client = OllamaLLMClient()
        self.porcupine = None
        self.pa = None
        self.audio_stream = None
        self.interrupt_event = asyncio.Event()
        self.current_command_task = None

    async def initialize_resources_async(self):
        if not PICOVOICE_ACCESS_KEY:
            raise ValueError("PICOVOICE_ACCESS_KEY не найден.")
        if not PIPER_VOICE_PATH or not os.path.exists(PIPER_VOICE_PATH):
            raise ValueError(
                "LOKI_PIPER_VOICE_PATH не найден или указан неверный путь."
            )
        try:
            keywords = [WAKE_WORD]
            keyword_paths = None
            if CUSTOM_WAKE_WORD_PATH and os.path.exists(CUSTOM_WAKE_WORD_PATH):
                keywords = None
                keyword_paths = [CUSTOM_WAKE_WORD_PATH]
                logging.info(f"Using custom wake word from: {CUSTOM_WAKE_WORD_PATH}")
            else:
                logging.info(f"Using built-in wake word: '{WAKE_WORD}'")
            self.porcupine = pvporcupine.create(
                access_key=PICOVOICE_ACCESS_KEY,
                keywords=keywords,
                keyword_paths=keyword_paths,
            )
            self.pa = pyaudio.PyAudio()
            while True:
                try:
                    self.audio_stream = self.pa.open(
                        rate=self.porcupine.sample_rate,
                        channels=1,
                        format=pyaudio.paInt16,
                        input=True,
                        frames_per_buffer=self.porcupine.frame_length,
                    )
                    logging.info("Audio stream opened successfully.")
                    break
                except IOError as e:
                    logging.error(
                        f"Failed to open audio stream: {e}. Retrying in 5 seconds..."
                    )
                    await asyncio.sleep(5)
            logging.info(f"LOKI initialized. Listening for wake word...")
        except Exception as e:
            logging.error(f"Failed to initialize resources: {e}")
            await self.cleanup()
            raise

    async def _listen_for_wake_word_async(self):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._blocking_listen)

    def _blocking_listen(self):
        while True:
            pcm = self.audio_stream.read(
                self.porcupine.frame_length, exception_on_overflow=False
            )
            pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
            result = self.porcupine.process(pcm)
            if result >= 0:
                logging.info("Wake word detected!")
                self.interrupt_event.set()
                return

    async def run_async(self):
        handle_visual_command(
            {"tool_name": "set_status", "parameters": {"status": "idle"}}
        )
        while True:
            try:
                await self._listen_for_wake_word_async()

                if self.current_command_task and not self.current_command_task.done():
                    logging.info("Interrupting previous command.")
                    self.current_command_task.cancel()
                    try:
                        await self.current_command_task
                    except asyncio.CancelledError:
                        logging.info("Previous command task cancelled.")

                self.interrupt_event.clear()

                handle_visual_command(
                    {"tool_name": "set_status", "parameters": {"status": "listening"}}
                )
                command_audio_path = record_command_vad()

                self.current_command_task = asyncio.create_task(
                    self.handle_command_async(command_audio_path)
                )

            except KeyboardInterrupt:
                logging.info("Shutdown signal received.")
                break
            except Exception as e:
                logging.error(f"An error occurred in the main loop: {e}", exc_info=True)
                await asyncio.sleep(1)

    async def handle_command_async(self, audio_path: str):
        audio_playback_stream = None
        try:
            loop = asyncio.get_running_loop()
            user_command_text = await loop.run_in_executor(
                None, self.stt_engine.transcribe, audio_path
            )
            if not user_command_text:
                return

            thoughts_content = ""
            answer_content = ""
            answer_text_for_tts_buffer = ""
            answer_started = False

            # Основной цикл, который одновременно получает токены от LLM и отправляет аудио в TTS
            async for chunk in self.llm_client.stream_response(user_command_text):
                if self.interrupt_event.is_set():
                    logging.info("LLM stream processing interrupted.")
                    break

                content = chunk.get("content", "")

                if chunk["type"] == "thought":
                    thoughts_content += content
                elif chunk["type"] == "answer":
                    answer_content += content

                    if not answer_started:
                        # Инициализируем аудиопоток только тогда, когда есть что говорить
                        answer_started = True
                        handle_visual_command(
                            {
                                "tool_name": "set_status",
                                "parameters": {"status": "speaking"},
                            }
                        )
                        audio_playback_stream = sd.RawOutputStream(
                            samplerate=self.tts_engine.sample_rate,
                            channels=1,
                            dtype="int16",
                        )
                        audio_playback_stream.start()
                        logging.info("Audio playback stream started (on-demand).")

                    answer_text_for_tts_buffer += content

                    # Улучшенная логика разбиения на предложения
                    sentence_terminators = re.compile(r"([.!?]|,\s|\.\.\.|\n)")
                    while True:
                        match = sentence_terminators.search(answer_text_for_tts_buffer)
                        if not match:
                            break

                        end_index = match.end()
                        sentence = answer_text_for_tts_buffer[:end_index].strip()
                        answer_text_for_tts_buffer = answer_text_for_tts_buffer[
                            end_index:
                        ]

                        if sentence:
                            logging.info(f"Streaming sentence to TTS: '{sentence}'")
                            async for audio_chunk in self.tts_engine.stream(sentence):
                                if self.interrupt_event.is_set():
                                    break
                                if (
                                    audio_playback_stream
                                    and not audio_playback_stream.closed
                                ):
                                    audio_playback_stream.write(audio_chunk)
                        if self.interrupt_event.is_set():
                            break
                if self.interrupt_event.is_set():
                    break

            # Озвучиваем остаток из буфера
            if (
                answer_text_for_tts_buffer.strip()
                and not self.interrupt_event.is_set()
                and audio_playback_stream
            ):
                logging.info(
                    f"Streaming final sentence to TTS: '{answer_text_for_tts_buffer.strip()}'"
                )
                async for audio_chunk in self.tts_engine.stream(
                    answer_text_for_tts_buffer.strip()
                ):
                    if audio_playback_stream and not audio_playback_stream.closed:
                        audio_playback_stream.write(audio_chunk)

            # "Холодный" парсер
            if not self.interrupt_event.is_set():
                full_structured_response = (
                    f"<мысли>{thoughts_content}</мысли><ответ>{answer_content}</ответ>"
                )
                clean_response = extract_clean_response(full_structured_response)
                _, command_json = parse_llm_response(clean_response)
                if command_json:
                    handle_visual_command(command_json)

            if not answer_started and not self.interrupt_event.is_set():
                logging.warning("LLM response did not contain an <answer> block.")
                await loop.run_in_executor(
                    None,
                    self.tts_engine.speak,
                    "Произошла ошибка при формировании ответа.",
                )

        except asyncio.CancelledError:
            logging.info("Command handling was cancelled.")
            raise
        finally:
            if audio_playback_stream:
                audio_playback_stream.stop()
                audio_playback_stream.close()
                logging.info("Audio playback stream closed.")
            if not self.interrupt_event.is_set():
                handle_visual_command(
                    {"tool_name": "set_status", "parameters": {"status": "idle"}}
                )

    async def cleanup(self):
        logging.info("Cleaning up resources...")
        if self.current_command_task:
            self.current_command_task.cancel()
        if self.audio_stream:
            self.audio_stream.close()
        if self.pa:
            self.pa.terminate()
        if self.porcupine:
            self.porcupine.delete()
        if self.llm_client:
            await self.llm_client.close()


async def main():
    orchestrator = LokiOrchestrator()
    try:
        await orchestrator.initialize_resources_async()
        await orchestrator.run_async()
    finally:
        if orchestrator:
            await orchestrator.cleanup()
    logging.info("LOKI Core has shut down.")


if __name__ == "__main__":
    asyncio.run(main())
