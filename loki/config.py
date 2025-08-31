# loki/config.py

# --- Audio Handler Configuration ---
# Параметры для записи звука и VAD (Voice Activity Detection)
AUDIO_FORMAT = "int16"  # Формат семплов (paInt16)
AUDIO_CHANNELS = 1
AUDIO_RATE = 16000  # Частота дискретизации, стандарт для STT моделей
CHUNK_DURATION_MS = 30  # Длительность одного чанка в миллисекундах
# Размер чанка в семплах: 16000 * 30 / 1000 = 480
CHUNK_SIZE = int(AUDIO_RATE * CHUNK_DURATION_MS / 1000)
VAD_AGGRESSIVENESS = 3  # Уровень агрессивности VAD (от 0 до 3)

# --- LLM Client Configuration ---
# Значения по умолчанию для подключения к Ollama
# Эти значения могут быть переопределены через переменные окружения .env
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3:8b-instruct-q4_k_m"

# --- Visual Controller Configuration ---
# Путь по умолчанию к исполняемому файлу Wallpaper Engine
# Может быть переопределен через переменную окружения WALLPAPER_ENGINE_PATH
DEFAULT_WALLPAPER_ENGINE_PATH = (
    "C:/Program Files (x86)/Steam/steamapps/common/wallpaper_engine/wallpaper32.exe"
)
