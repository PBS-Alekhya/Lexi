# src/tts.py
import pyttsx3
import time
from typing import Any
import queue

def run_tts_stream(text: str, chunk_size: int, out_queue: "queue.Queue[str]") -> None:
    """
    Background function: speaks text in successive chunks and pushes progressive text
    into out_queue so the main thread can update the UI.
    - text: full text to speak
    - chunk_size: number of characters to reveal per update
    - out_queue: queue where partial texts are put; final sentinel is None
    """
    # initialize engine inside the thread (safer)
    engine = pyttsx3.init()
    engine.setProperty("rate", 160)
    engine.setProperty("volume", 0.9)

    # speak in increments, push progressive text to queue
    try:
        n = len(text)
        for start in range(0, n, chunk_size):
            # progressive text to show in UI (from beginning to current chunk end)
            progressive_text = text[: start + chunk_size ]
            out_queue.put(progressive_text)

            # speak only the current chunk (slice)
            chunk_to_speak = text[start : start + chunk_size]
            if chunk_to_speak.strip():
                engine.say(chunk_to_speak)
                engine.runAndWait()
            # small yield so main thread can poll often
            time.sleep(0.05)
    except Exception:
        # if TTS fails, still send final text so UI shows answer
        out_queue.put(text)
    finally:
        # signal completion
        out_queue.put(None)
