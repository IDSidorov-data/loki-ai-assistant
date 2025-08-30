# loki/audio_handler.py

import pyaudio
import wave
import logging
import webrtcvad
import collections
import tempfile

# --- Константы для записи аудио ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_DURATION_MS = 30
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)
VAD_AGGRESSIVENESS = 3


def record_command_vad():
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )

    logging.info(">>> Recording started. Speak your command.")

    ring_buffer = collections.deque(maxlen=10)
    frames = []
    triggered = False
    silent_chunks = 0
    num_padding_chunks = 15

    while True:
        chunk = stream.read(CHUNK_SIZE)
        is_speech = vad.is_speech(chunk, RATE)

        if not triggered:
            ring_buffer.append(chunk)
            if is_speech:
                logging.info("Voice activity detected.")
                triggered = True
                frames.extend(list(ring_buffer))
                ring_buffer.clear()
        else:
            frames.append(chunk)
            if not is_speech:
                silent_chunks += 1
                if silent_chunks > num_padding_chunks:
                    break
            else:
                silent_chunks = 0

    logging.info(">>> Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        output_filename = tf.name
        with wave.open(output_filename, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(frames))

    return output_filename
