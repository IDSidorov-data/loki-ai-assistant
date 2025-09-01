# loki/config.py
"""
Центральный файл конфигурации для LOKI.

Этот модуль содержит статические параметры и значения по умолчанию, используемые
в различных частях приложения. Централизация настроек позволяет легко изменять
поведение системы, не внося правок в основной код.

Значения из этого файла могут быть переопределены переменными окружения
в файле `.env`.
"""

# --- Audio Handler Configuration ---
# Параметры для записи звука и VAD (Voice Activity Detection)

AUDIO_FORMAT = "int16"  # Формат семплов (соответствует pyaudio.paInt16)
AUDIO_CHANNELS = 1  # Моно-звук для совместимости с моделями
AUDIO_RATE = (
    16000  # Частота дискретизации, стандарт для большинства STT моделей (Whisper, etc.)
)
CHUNK_DURATION_MS = (
    30  # Длительность одного чанка в мс (рекомендация webrtcvad: 10, 20 или 30)
)
# Размер чанка в семплах: 16000 Гц * 30 мс / 1000 = 480 семплов
CHUNK_SIZE = int(AUDIO_RATE * CHUNK_DURATION_MS / 1000)
# Уровень агрессивности VAD (от 0 до 3). 3 - самый "агрессивный",
# требует наименьшей громкости для срабатывания детектора речи.
VAD_AGGRESSIVENESS = 3

# --- LLM Client Configuration ---
# Значения по умолчанию для подключения к локальному серверу Ollama
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3:8b-instruct-q4_k_m"

# --- Visual Controller Configuration ---
# Путь по умолчанию к исполняемому файлу Wallpaper Engine.
# Может быть переопределен через переменную окружения WALLPAPER_ENGINE_PATH.
DEFAULT_WALLPAPER_ENGINE_PATH = (
    "C:/Program Files (x86)/Steam/steamapps/common/wallpaper_engine/wallpaper32.exe"
)
