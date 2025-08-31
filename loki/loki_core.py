import sys
import os

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
        try:
            loop = asyncio.get_running_loop()
            user_command_text = await loop.run_in_executor(
                None, self.stt_engine.transcribe, audio_path
            )
            if user_command_text:

                # 1. Собираем полный ответ от LLM, проверяя на прерывание
                full_llm_response = ""
                async for token in self.llm_client.stream_response(user_command_text):
                    if self.interrupt_event.is_set():
                        logging.info("LLM stream interrupted.")
                        return  # Немедленно выходим
                    full_llm_response += token

                # 2. Извлекаем чистый ответ из тегов <ответ>
                clean_response = extract_clean_response(full_llm_response)

                # 3. Парсим чистый ответ на текст для озвучки и команду
                text_to_speak, command_json = parse_llm_response(clean_response)

                # 4. Озвучиваем текст целиком (непотоково, но надежно)
                if text_to_speak and not self.interrupt_event.is_set():
                    handle_visual_command(
                        {
                            "tool_name": "set_status",
                            "parameters": {"status": "speaking"},
                        }
                    )
                    await loop.run_in_executor(
                        None, self.tts_engine.speak, text_to_speak
                    )

                # 5. Выполняем команду, если она была
                if command_json and not self.interrupt_event.is_set():
                    handle_visual_command(command_json)

        except asyncio.CancelledError:
            logging.info("Command handling was cancelled.")
            raise
        finally:
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
