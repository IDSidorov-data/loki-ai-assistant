# loki/loki_core.py
import sys
import os
import re
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
from loki.command_parser import parse_llm_response
from loki.visual_controller import handle_visual_command
# Импортируем оба промпта
from loki.prompts import DEFAULT_PROMPT, COMMAND_PROMPT

LOG_LEVEL = os.getenv("LOKI_LOG_LEVEL", "INFO").upper()
PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")
WAKE_WORD = os.getenv("LOKI_WAKE_WORD", "jarvis")
PIPER_VOICE_PATH = os.getenv("LOKI_PIPER_VOICE_PATH")
COMMAND_KEYWORDS = ["статус", "режим", "переключись", "состояние"]

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")

class LokiOrchestrator:
    def __init__(self):
        self.stt_engine = WhisperSTT(model_name="base")
        self.tts_engine = Piper_Engine(model_path=PIPER_VOICE_PATH)
        self.llm_client = OllamaLLMClient()
        self.porcupine = None; self.pa = None; self.audio_stream = None
        self.interrupt_event = asyncio.Event(); self.current_command_task = None

    async def initialize_resources_async(self):
        if not PICOVOICE_ACCESS_KEY: raise ValueError("PICOVOICE_ACCESS_KEY не найден.")
        if not PIPER_VOICE_PATH or not os.path.exists(PIPER_VOICE_PATH): raise ValueError("LOKI_PIPER_VOICE_PATH не найден или указан неверный путь.")
        try:
            keywords = [WAKE_WORD]
            keyword_paths = None
            if os.getenv("LOKI_CUSTOM_WAKE_WORD_PATH"):
                keyword_paths = [os.getenv("LOKI_CUSTOM_WAKE_WORD_PATH")]
                keywords = None
            self.porcupine = pvporcupine.create(access_key=PICOVOICE_ACCESS_KEY, keywords=keywords, keyword_paths=keyword_paths)
            self.pa = pyaudio.PyAudio()
            while True:
                try:
                    self.audio_stream = self.pa.open(rate=self.porcupine.sample_rate, channels=1, format=pyaudio.paInt16, input=True, frames_per_buffer=self.porcupine.frame_length)
                    break
                except IOError: await asyncio.sleep(5)
            logging.info("LOKI initialized.")

            # <<< ДОБАВЛЕНО: Запускаем "разогрев" LLM в фоновом режиме
            logging.info("Warming up LLM engine...")
            asyncio.create_task(self._warm_up_llm())

        except Exception as e: await self.cleanup(); raise

    # <<< ДОБАВЛЕНО: Новый метод для "разогрева"
    async def _warm_up_llm(self):
        """Отправляет фиктивный запрос к LLM, чтобы устранить "холодный старт"."""
        try:
            # Используем async for, чтобы "прочитать" весь ответ, но ничего с ним не делаем.
            # ВАЖНО: передаем DEFAULT_PROMPT, так как метод stream_response теперь требует два аргумента.
            async for _ in self.llm_client.stream_response("Привет", system_prompt=DEFAULT_PROMPT):
                pass
            logging.info("LLM engine is warm and ready.")
        except Exception as e:
            logging.warning(f"LLM warm-up failed: {e}")

    async def _listen_for_wake_word_async(self):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._blocking_listen)

    def _blocking_listen(self):
        while True:
            pcm = self.audio_stream.read(self.porcupine.frame_length, exception_on_overflow=False)
            pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
            if self.porcupine.process(pcm) >= 0: self.interrupt_event.set(); return

    async def run_async(self):
        handle_visual_command({"tool_name": "set_status", "parameters": {"status": "idle"}})
        while True:
            try:
                await self._listen_for_wake_word_async()
                if self.current_command_task and not self.current_command_task.done():
                    self.current_command_task.cancel()
                    try: await self.current_command_task
                    except asyncio.CancelledError: pass
                self.interrupt_event.clear()
                handle_visual_command({"tool_name": "set_status", "parameters": {"status": "listening"}})
                command_audio_path = record_command_vad()
                self.current_command_task = asyncio.create_task(self.handle_command_async(command_audio_path))
            except KeyboardInterrupt: break
            except Exception: await asyncio.sleep(1)

    def _is_command_request(self, text: str) -> bool:
        """Проверяет, содержит ли текст ключевые слова для команд."""
        return any(keyword in text.lower() for keyword in COMMAND_KEYWORDS)

    async def handle_command_async(self, audio_path: str):
        audio_playback_stream = None
        try:
            loop = asyncio.get_running_loop()
            user_command_text = await loop.run_in_executor(None, self.stt_engine.transcribe, audio_path)
            if not user_command_text: return

            # Логика выбора промпта
            system_prompt = DEFAULT_PROMPT
            if self._is_command_request(user_command_text):
                logging.info("Command keywords detected, switching to COMMAND_PROMPT.")
                system_prompt = COMMAND_PROMPT
            
            full_response = ""
            answer_started = False
            tts_buffer = ""

            # Передаем выбранный промпт в клиент
            async for token in self.llm_client.stream_response(user_command_text, system_prompt):
                if self.interrupt_event.is_set(): break
                
                if not answer_started:
                    answer_started = True
                    handle_visual_command({"tool_name": "set_status", "parameters": {"status": "speaking"}})
                    audio_playback_stream = sd.RawOutputStream(samplerate=self.tts_engine.sample_rate, channels=1, dtype='int16')
                    audio_playback_stream.start()
                
                full_response += token
                tts_buffer += token

                sentence_terminators = re.compile(r'([.!?]|,\s|\.\.\.|\n)')
                while True:
                    match = sentence_terminators.search(tts_buffer)
                    if not match: break
                    
                    end_index = match.end()
                    sentence, tts_buffer = tts_buffer[:end_index].strip(), tts_buffer[end_index:]
                    
                    if sentence:
                        async for audio_chunk in self.tts_engine.stream(sentence):
                            if self.interrupt_event.is_set(): break
                            if audio_playback_stream and not audio_playback_stream.closed:
                                audio_playback_stream.write(audio_chunk)
                    if self.interrupt_event.is_set(): break
            
            if tts_buffer.strip() and not self.interrupt_event.is_set() and audio_playback_stream:
                async for audio_chunk in self.tts_engine.stream(tts_buffer.strip()):
                    if audio_playback_stream and not audio_playback_stream.closed:
                        audio_playback_stream.write(audio_chunk)

            if not self.interrupt_event.is_set() and full_response:
                logging.info(f"Full LLM response received: '{full_response}'")
                text_to_speak, command_json = parse_llm_response(full_response)
                if command_json: handle_visual_command(command_json)

        except asyncio.CancelledError: raise
        finally:
            if audio_playback_stream: audio_playback_stream.stop(); audio_playback_stream.close()
            if not self.interrupt_event.is_set():
                handle_visual_command({"tool_name": "set_status", "parameters": {"status": "idle"}})

    async def cleanup(self):
        if self.current_command_task: self.current_command_task.cancel()
        if self.audio_stream: self.audio_stream.close()
        if self.pa: self.pa.terminate()
        if self.porcupine: self.porcupine.delete()
        if self.llm_client: self.llm_client.close()

async def main():
    orchestrator = LokiOrchestrator()
    try:
        await orchestrator.initialize_resources_async()
        await orchestrator.run_async()
    finally:
        if orchestrator: orchestrator.cleanup()

if __name__ == "__main__":
    asyncio.run(main())