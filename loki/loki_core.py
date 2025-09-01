# loki/loki_core.py
"""
Главный оркестратор голосового ассистента LOKI.

Этот модуль связывает все компоненты системы воедино:
- Распознавание ключевого слова (wake word) с помощью Picovoice Porcupine.
- Запись голоса и определение конца фразы (VAD).
- Преобразование речи в текст (STT) с помощью Whisper.
- Взаимодействие с языковой моделью (LLM) через Ollama.
- Синтез речи (TTS) с помощью Piper.
- Управление состоянием (визуализация) ассистента.
"""
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

# Загрузка переменных окружения из .env файла
load_dotenv()

# Импорт локальных модулей проекта
from loki.audio_handler import record_command_vad
from loki.stt_handler import WhisperSTT
from loki.tts_handler import Piper_Engine
from loki.llm_client import OllamaLLMClient
from loki.command_parser import parse_llm_response
from loki.visual_controller import handle_visual_command
from loki.prompts import DEFAULT_PROMPT, COMMAND_PROMPT

# Конфигурация на основе переменных окружения
LOG_LEVEL = os.getenv("LOKI_LOG_LEVEL", "INFO").upper()
PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")
WAKE_WORD = os.getenv("LOKI_WAKE_WORD", "jarvis")
PIPER_VOICE_PATH = os.getenv("LOKI_PIPER_VOICE_PATH")
COMMAND_KEYWORDS = ["статус", "режим", "переключись", "состояние"]

# Настройка логирования
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")

class LokiOrchestrator:
    """
    Управляет основным циклом работы ассистента LOKI.

    Инициализирует все необходимые движки (STT, TTS, LLM, Wake Word),
    управляет асинхронным циклом прослушивания и обработки команд,
    а также корректно освобождает ресурсы при завершении работы.
    """
    def __init__(self):
        """Инициализирует экземпляры всех необходимых сервисов."""
        self.stt_engine = WhisperSTT(model_name="base")
        self.tts_engine = Piper_Engine(model_path=PIPER_VOICE_PATH)
        self.llm_client = OllamaLLMClient()
        self.porcupine = None
        self.pa = None
        self.audio_stream = None
        # Событие для прерывания длительных операций (например, TTS) при активации wake word
        self.interrupt_event = asyncio.Event()
        self.current_command_task = None

    async def initialize_resources_async(self):
        """
        Асинхронно инициализирует аудио-ресурсы (Porcupine, PyAudio).
        Эти операции могут быть блокирующими, поэтому выполняются асинхронно.
        Также запускает фоновую задачу для "разогрева" LLM.
        """
        if not PICOVOICE_ACCESS_KEY: raise ValueError("PICOVOICE_ACCESS_KEY не найден.")
        if not PIPER_VOICE_PATH or not os.path.exists(PIPER_VOICE_PATH): raise ValueError("LOKI_PIPER_VOICE_PATH не найден или указан неверный путь.")
        try:
            # Настройка Porcupine для стандартного или кастомного wake word
            keywords = [WAKE_WORD]
            keyword_paths = None
            if os.getenv("LOKI_CUSTOM_WAKE_WORD_PATH"):
                keyword_paths = [os.getenv("LOKI_CUSTOM_WAKE_WORD_PATH")]
                keywords = None
            
            self.porcupine = pvporcupine.create(access_key=PICOVOICE_ACCESS_KEY, keywords=keywords, keyword_paths=keyword_paths)
            self.pa = pyaudio.PyAudio()
            
            # Попытка открыть аудиопоток с повторами в случае ошибки
            while True:
                try:
                    self.audio_stream = self.pa.open(rate=self.porcupine.sample_rate, channels=1, format=pyaudio.paInt16, input=True, frames_per_buffer=self.porcupine.frame_length)
                    break
                except IOError:
                    logging.warning("Не удалось открыть аудиопоток, повтор через 5 секунд...")
                    await asyncio.sleep(5)
            
            logging.info("LOKI initialized.")
            
            # Запускаем "разогрев" LLM в фоновой задаче, не блокируя старт
            logging.info("Warming up LLM engine...")
            asyncio.create_task(self._warm_up_llm())
            
        except Exception as e:
            await self.cleanup()
            raise

    async def _warm_up_llm(self):
        """
        Отправляет фиктивный запрос к LLM для устранения "холодного старта".
        Первый запрос к модели может быть долгим, так как модель загружается в память.
        Этот метод выполняет его заранее, во время инициализации.
        """
        try:
            # Мы полностью "прочитываем" потоковый ответ, чтобы убедиться,
            # что генерация действительно произошла, но игнорируем сами токены.
            async for _ in self.llm_client.stream_response("Привет", system_prompt=DEFAULT_PROMPT):
                pass
            logging.info("LLM engine is warm and ready.")
        except Exception as e:
            logging.warning(f"LLM warm-up failed: {e}")

    async def _listen_for_wake_word_async(self):
        """Асинхронная обертка для блокирующего метода прослушивания."""
        loop = asyncio.get_running_loop()
        # Запускаем блокирующий код в отдельном потоке, чтобы не заморозить event loop
        await loop.run_in_executor(None, self._blocking_listen)

    def _blocking_listen(self):
        """
        Блокирующий метод, который непрерывно слушает аудиопоток в поиске wake word.
        Работает в отдельном потоке.
        """
        while True:
            pcm = self.audio_stream.read(self.porcupine.frame_length, exception_on_overflow=False)
            pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
            # porcupine.process возвращает индекс ключевого слова (0 в нашем случае), если оно найдено
            if self.porcupine.process(pcm) >= 0:
                # Устанавливаем событие, чтобы основной цикл мог среагировать
                self.interrupt_event.set()
                return

    async def run_async(self):
        """Основной асинхронный цикл работы ассистента."""
        handle_visual_command({"tool_name": "set_status", "parameters": {"status": "idle"}})
        while True:
            try:
                # 1. Ждем произнесения wake word
                await self._listen_for_wake_word_async()
                
                # Если предыдущая задача обработки команды еще выполняется, отменяем ее
                if self.current_command_task and not self.current_command_task.done():
                    self.current_command_task.cancel()
                    try:
                        await self.current_command_task
                    except asyncio.CancelledError:
                        pass # Ожидаемое исключение при отмене
                
                self.interrupt_event.clear() # Сбрасываем событие прерывания
                
                # 2. Переключаемся в режим прослушивания и записываем команду
                handle_visual_command({"tool_name": "set_status", "parameters": {"status": "listening"}})
                command_audio_path = record_command_vad()
                
                # 3. Создаем асинхронную задачу для обработки записанной команды
                self.current_command_task = asyncio.create_task(self.handle_command_async(command_audio_path))
                
            except KeyboardInterrupt:
                logging.info("Завершение работы по команде пользователя.")
                break
            except Exception as e:
                logging.error(f"Критическая ошибка в основном цикле: {e}")
                await asyncio.sleep(1)

    def _is_command_request(self, text: str) -> bool:
        """
        Проверяет, содержит ли текст ключевые слова для команд.
        Это простой, но эффективный способ для выбора нужного системного промпта.

        Args:
            text (str): Распознанный текст от пользователя.

        Returns:
            bool: True, если текст является командой, иначе False.
        """
        return any(keyword in text.lower() for keyword in COMMAND_KEYWORDS)

    async def _speak_text(self, text: str):
        """
        Озвучивает переданный текст с помощью TTS движка.

        Args:
            text (str): Текст для озвучки.
        """
        audio_playback_stream = sd.RawOutputStream(samplerate=self.tts_engine.sample_rate, channels=1, dtype='int16')
        audio_playback_stream.start()
        try:
            async for audio_chunk in self.tts_engine.stream(text):
                if self.interrupt_event.is_set():
                    break # Прерываем озвучку, если снова услышали wake word
                if not audio_playback_stream.closed:
                    audio_playback_stream.write(audio_chunk)
        finally:
            audio_playback_stream.stop()
            audio_playback_stream.close()

    async def handle_command_async(self, audio_path: str):
        """
        Полный цикл обработки одной команды: STT -> LLM -> TTS/Command.
        Эта функция реализует ключевую логику "сначала парсинг, потом озвучка".

        Args:
            audio_path (str): Путь к временному аудиофайлу с записанной командой.
        """
        command_executed = False
        try:
            # Шаг 1: Преобразование речи в текст
            loop = asyncio.get_running_loop()
            user_command_text = await loop.run_in_executor(None, self.stt_engine.transcribe, audio_path)
            if not user_command_text or self.interrupt_event.is_set():
                return

            # Шаг 2: Выбор системного промпта на основе распознанного текста
            system_prompt = DEFAULT_PROMPT
            if self._is_command_request(user_command_text):
                logging.info("Command keywords detected, switching to COMMAND_PROMPT.")
                system_prompt = COMMAND_PROMPT

            # Шаг 3: Получение ПОЛНОГО ответа от LLM.
            # Мы накапливаем все токены в одну строку перед дальнейшей обработкой.
            full_response = ""
            async for token in self.llm_client.stream_response(user_command_text, system_prompt):
                if self.interrupt_event.is_set(): break
                full_response += token
            
            if self.interrupt_event.is_set() or not full_response:
                return

            logging.info(f"Full LLM response received: '{full_response}'")

            # Шаг 4: Парсинг ответа. Извлекаем текст для озвучки и JSON для выполнения.
            text_to_speak, command_json = parse_llm_response(full_response)

            # Шаг 5: Выполнение команды, если она была найдена
            if command_json:
                handle_visual_command(command_json)
                # Устанавливаем флаг, если команда меняет состояние (чтобы не сбросить его в finally)
                if command_json.get("tool_name") == "set_status":
                    command_executed = True

            # Шаг 6: Озвучка чистого текста, если он есть
            if text_to_speak:
                handle_visual_command({"tool_name": "set_status", "parameters": {"status": "speaking"}})
                await self._speak_text(text_to_speak)

        except asyncio.CancelledError:
            logging.info("Задача обработки команды была отменена.")
            raise
        finally:
            # Шаг 7: Возврат в состояние ожидания, но только если не была выполнена
            # команда, которая устанавливает постоянный статус (например, "processing").
            if not self.interrupt_event.is_set() and not command_executed:
                handle_visual_command({"tool_name": "set_status", "parameters": {"status": "idle"}})

    async def cleanup(self):
        """Корректно освобождает все занятые ресурсы."""
        logging.info("Освобождение ресурсов...")
        if self.current_command_task:
            self.current_command_task.cancel()
        if self.audio_stream:
            self.audio_stream.close()
        if self.pa:
            self.pa.terminate()
        if self.porcupine:
            self.porcupine.delete()
        if self.llm_client:
            self.llm_client.close()
        logging.info("Ресурсы освобождены.")

async def main():
    """Основная точка входа в приложение."""
    orchestrator = LokiOrchestrator()
    try:
        await orchestrator.initialize_resources_async()
        await orchestrator.run_async()
    finally:
        if orchestrator:
            await orchestrator.cleanup()

if __name__ == "__main__":
    asyncio.run(main())