# loki/audio_handler.py
"""
Обработчик аудиоввода.

Отвечает за запись звука с микрофона и использование Voice Activity Detection (VAD)
для автоматического определения момента, когда пользователь закончил говорить.
"""
import pyaudio
import wave
import logging
import webrtcvad
import collections
import tempfile

from loki import config

# Преобразуем строковый формат из конфига в константу PyAudio
FORMAT = getattr(pyaudio, f"pa{config.AUDIO_FORMAT.capitalize()}")


def record_command_vad() -> str:
    """
    Записывает аудио с микрофона до тех пор, пока не будет обнаружена тишина.

    Использует VAD (Voice Activity Detection) для определения наличия речи.
    Запись начинается после первого обнаружения голоса и заканчивается после
    некоторого периода тишины.

    Returns:
        str: Путь к временному WAV-файлу с записанной командой.
    """
    vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=config.AUDIO_CHANNELS,
        rate=config.AUDIO_RATE,
        input=True,
        frames_per_buffer=config.CHUNK_SIZE,
    )
    logging.info(">>> Recording started. Speak your command.")

    # Кольцевой буфер для хранения аудио перед началом речи, чтобы не обрезать начало фразы
    ring_buffer = collections.deque(maxlen=10)
    frames = []
    triggered = False  # Флаг, который становится True после обнаружения речи
    silent_chunks = 0

    while True:
        chunk = stream.read(config.CHUNK_SIZE)
        is_speech = vad.is_speech(chunk, config.AUDIO_RATE)

        if not triggered:
            ring_buffer.append(chunk)
            if is_speech:
                # Речь обнаружена, переключаем триггер и добавляем пред-записанные чанки
                logging.info("Voice activity detected.")
                triggered = True
                frames.extend(list(ring_buffer))
                ring_buffer.clear()
        else:
            # Речь уже идет, просто добавляем чанки
            frames.append(chunk)
            if not is_speech:
                # Если наступила тишина, начинаем считать "тихие" чанки
                silent_chunks += 1
                if silent_chunks > config.VAD_SILENCE_PADDING_CHUNKS:
                    # Если тишина длится достаточно долго, прекращаем запись
                    break
            else:
                # Если речь возобновилась, сбрасываем счетчик тишины
                silent_chunks = 0

    logging.info(">>> Recording finished.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Сохраняем записанные аудиоданные во временный WAV-файл
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        output_filename = tf.name
        with wave.open(output_filename, "wb") as wf:
            wf.setnchannels(config.AUDIO_CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(config.AUDIO_RATE)
            wf.writeframes(b"".join(frames))

    return output_filename
