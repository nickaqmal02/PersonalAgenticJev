"""
Jev Voice — Phase 5 (Voice)

Speak to Jev, Jev speaks back.

Pipeline:
    1. Press ENTER → record audio from mic (silence detection stops)
    2. Transcribe with faster-whisper (local, no API key)
    3. Feed transcript to jev_brain.think() → DeepSeek + Jev pipeline
    4. Speak the reply with edge-tts (free, no API key)

Requirements (already installed in .venv):
    - faster-whisper (local STT, model downloads on first run ~150MB)
    - edge-tts (free TTS via Microsoft Edge)
    - sounddevice + numpy (mic recording)
    - ffmpeg (already installed via brew)
"""

import os
import sys
import wave
import tempfile
import asyncio
import struct

# --- .env loader ---
def load_env(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

load_env(os.path.join(os.path.dirname(__file__), ".env"))

import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import edge_tts


# ─── Config ───────────────────────────────────────────────────────
SAMPLE_RATE = 16000          # Whisper wants 16kHz
CHANNELS = 1
SILENCE_THRESHOLD = 0.01    # RMS below this = silence
SILENCE_DURATION = 2.0       # seconds of silence before stopping
MAX_RECORDING_SECONDS = 15   # safety cap
WHISPER_MODEL = "base"       # tiny|base|small|medium|large|turbo (base = good + fast)
TTS_VOICE = "en-US-GuyNeural"  # male voice (change to en-US-AriaNeural for female)
TTS_RATE = "+0%"             # speed: -10% slower, +10% faster

# ─── STT: Record + Transcribe ──────────────────────────────────────

def record_audio() -> str:
    """Record from mic until silence detected. Return path to WAV file."""
    print("🎤 Listening... (speak now)")

    frames = []
    silence_start = None
    recording_start = None

    def callback(indata, frame_count, time_info, status):
        nonlocal silence_start, recording_start
        if recording_start is None:
            recording_start = time_info.inputBufferAdcTime

        frames.append(indata.copy())

        # Check for silence
        rms = np.sqrt(np.mean(indata**2))
        if rms < SILENCE_THRESHOLD:
            if silence_start is None:
                silence_start = time_info.inputBufferAdcTime
            elif time_info.inputBufferAdcTime - silence_start > SILENCE_DURATION:
                raise sd.CallbackStop
        else:
            silence_start = None

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            callback=callback,
            blocksize=1024,
        ) as stream:
            stream.start()
            # Timeout safety
            import time
            start = time.time()
            while stream.active and (time.time() - start) < MAX_RECORDING_SECONDS:
                time.sleep(0.1)
    except sd.CallbackStop:
        pass  # normal — silence detected
    except Exception as e:
        print(f"  Mic error: {e}")
        return ""

    if not frames:
        return ""

    # Concatenate all frames
    audio = np.concatenate(frames, axis=0)

    # Save as WAV
    wav_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    with wave.open(wav_path, "w") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        # Convert float32 to int16
        audio_int = (audio * 32767).astype(np.int16)
        wf.writeframes(audio_int.tobytes())

    return wav_path


def transcribe(wav_path: str, model: WhisperModel) -> str:
    """Transcribe WAV file to text using faster-whisper.

    faster-whisper can accept a numpy array directly, avoiding the av library
    bug. We read the WAV with the stdlib wave module + numpy.
    """
    import wave
    import numpy as np

    with wave.open(wav_path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        sample_rate = wf.getframerate()

    # Convert bytes to float32 numpy array (what faster-whisper expects)
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    # faster-whisper accepts numpy array directly
    segments, _ = model.transcribe(audio, language="en")
    text = " ".join(segment.text for segment in segments).strip()
    return text


# ─── TTS: Speak the reply ─────────────────────────────────────────

async def speak(text: str) -> None:
    """Convert text to speech and play it."""
    # Clean text for TTS (remove emojis, markdown)
    clean = text
    for char in "✅❌⚠️🤔🧠🎵":
        clean = clean.replace(char, "")
    clean = clean.strip()
    if not clean:
        return

    # Generate audio
    mp3_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    communicate = edge_tts.Communicate(clean, TTS_VOICE, rate=TTS_RATE)
    await communicate.save(mp3_path)

    # Play with afplay (macOS built-in)
    os.system(f"afplay {mp3_path}")

    # Clean up
    os.unlink(mp3_path)


# ─── Main Voice Loop ──────────────────────────────────────────────

def main():
    # Load Whisper model (downloads on first run ~150MB)
    print(f"Loading Whisper model '{WHISPER_MODEL}'...")
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    print("Whisper ready.\n")

    # Import jev_brain (the two-model pipeline)
    sys.path.insert(0, os.path.dirname(__file__))
    from jev_brain import think

    print("=" * 50)
    print("  Jev Voice v0.1")
    print("  Press ENTER to speak, type 'quit' to exit")
    print("=" * 50)

    while True:
        try:
            user_input = input("\njev> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        if user_input == "":
            # Voice mode — record + transcribe
            wav_path = record_audio()
            if not wav_path:
                print("  (no audio captured)")
                continue

            transcript = transcribe(wav_path, model)
            os.unlink(wav_path)  # clean up

            if not transcript:
                print("  (didn't catch that)")
                continue

            print(f"  📝 You said: \"{transcript}\"")

            if transcript.lower() in ("quit", "exit", "stop", "goodbye"):
                print("Bye!")
                break

            user_input = transcript

        # Run through the two-model pipeline
        result = think(user_input)
        print(f"  {result}")

        # Speak the reply
        print("  🔊 Speaking...")
        asyncio.run(speak(result))


if __name__ == "__main__":
    main()
