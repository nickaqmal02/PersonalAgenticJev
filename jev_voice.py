"""
Jev Voice v0.3 — Always-on wake word mode

Flow:
    🟡 Idle: always listening with Whisper "tiny" (39MB, fast)
       ↓ detects "hey jev"
    🟢 Awake: record command with Whisper "base" (better accuracy)
       ↓ 2s silence → stop
    ⚡ Process: transcribe → DeepSeek + Jev → execute
    🔊 Speak reply
       ↓
    🟡 Back to idle

Usage:
    python3 jev_voice.py
    → just talk: "hey jev, open spotify"
    → or type: text commands still work
    → say "hey jev, goodbye" or press Ctrl+C to exit
"""

import os
import sys
import wave
import tempfile
import asyncio
import time
import numpy as np

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
from faster_whisper import WhisperModel
import edge_tts


# ─── Config ───────────────────────────────────────────────────────
SAMPLE_RATE = 16000
CHANNELS = 1
GAIN_BOOST = 10.0

# Wake word detection (tiny model, always on)
WAKE_MODEL = "tiny"              # 39MB, fast — for "hey jev" detection
WAKE_CHUNK_SECONDS = 2.0        # listen in 2-second chunks
WAKE_KEYWORDS = ["hey jev", "hey jeff", "hey jeff", "hey jef", "a jev", "hey dev", "hey jev"]
# (include variations — Whisper tiny can mishear "jev" as "jeff" etc.)

# Command recording (base model, better accuracy)
COMMAND_MODEL = "base"
SILENCE_THRESHOLD = 0.015
SILENCE_DURATION = 2.0
MAX_COMMAND_SECONDS = 15

# TTS
TTS_VOICE = "en-US-GuyNeural"
TTS_RATE = "+0%"


# ─── Models ───────────────────────────────────────────────────────

_wake_model = None
_command_model = None

def get_wake_model() -> WhisperModel:
    global _wake_model
    if _wake_model is None:
        print(f"  Loading wake word model '{WAKE_MODEL}'...")
        _wake_model = WhisperModel(WAKE_MODEL, device="cpu", compute_type="int8")
    return _wake_model

def get_command_model() -> WhisperModel:
    global _command_model
    if _command_model is None:
        print(f"  Loading command model '{COMMAND_MODEL}'...")
        _command_model = WhisperModel(COMMAND_MODEL, device="cpu", compute_type="int8")
    return _command_model


# ─── Audio helpers ───────────────────────────────────────────────

def save_wav(audio: np.ndarray) -> str:
    wav_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    with wave.open(wav_path, "w") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        audio_int = (audio * 32767).astype(np.int16)
        wf.writeframes(audio_int.tobytes())
    return wav_path


def transcribe_audio(audio: np.ndarray, model: WhisperModel) -> str:
    """Transcribe numpy audio array → text."""
    segments, _ = model.transcribe(audio, language="en", beam_size=3)
    text = " ".join(segment.text for segment in segments).strip()
    return text.lower()


# ─── Wake word detection ─────────────────────────────────────────

def listen_for_wake_word() -> bool:
    """
    Listen in 2-second chunks. Transcribe each chunk.
    Return True if wake word detected.
    """
    model = get_wake_model()
    chunk_samples = int(WAKE_CHUNK_SECONDS * SAMPLE_RATE)

    print("\r🟡 Listening for 'hey jev'...     ", end="", flush=True)

    frames = []

    def callback(indata, frame_count, time_info, status):
        boosted = np.clip(indata.copy() * GAIN_BOOST, -1.0, 1.0)
        frames.append(boosted)

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32", callback=callback, blocksize=1024) as stream:
            stream.start()
            start = time.time()
            while stream.active and (time.time() - start) < WAKE_CHUNK_SECONDS:
                time.sleep(0.05)
    except Exception:
        return False

    if not frames:
        return False

    audio = np.concatenate(frames, axis=0)
    rms = np.sqrt(np.mean(audio**2))

    # Skip if too quiet (saves CPU — no point transcribing silence)
    if rms < 0.003:
        return False

    # Transcribe and check for wake word
    text = transcribe_audio(audio, model)

    if text:
        # Check if any wake keyword matches
        for kw in WAKE_KEYWORDS:
            if kw in text:
                return True

    return False


# ─── Command recording ────────────────────────────────────────────

def record_command() -> str:
    """Record command after wake word. Stop on silence."""
    print("\r🟢 Yes? I'm listening...           ", flush=True)

    frames = []
    silence_start = None
    heard_speech = False

    def callback(indata, frame_count, time_info, status):
        nonlocal silence_start, heard_speech
        boosted = np.clip(indata.copy() * GAIN_BOOST, -1.0, 1.0)
        frames.append(boosted)
        rms = np.sqrt(np.mean(boosted**2))
        if rms > SILENCE_THRESHOLD:
            heard_speech = True
            silence_start = None
        elif heard_speech:
            if silence_start is None:
                silence_start = time_info.inputBufferAdcTime
            elif time_info.inputBufferAdcTime - silence_start > SILENCE_DURATION:
                raise sd.CallbackStop

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32", callback=callback, blocksize=1024) as stream:
            stream.start()
            start = time.time()
            while stream.active and (time.time() - start) < MAX_COMMAND_SECONDS:
                time.sleep(0.05)
    except sd.CallbackStop:
        pass
    except Exception as e:
        print(f"  Mic error: {e}")
        return ""

    if not frames or not heard_speech:
        print("  (no speech detected)")
        return ""

    audio = np.concatenate(frames, axis=0)

    # VAD trim
    window = 1600
    speech_idx = []
    for i in range(0, len(audio) - window, window):
        chunk = audio[i:i+window]
        if np.sqrt(np.mean(chunk**2)) > 0.005:
            speech_idx.append(i)
    if speech_idx:
        start_s = max(0, speech_idx[0] - window)
        end_s = min(len(audio), speech_idx[-1] + window * 2)
        audio = audio[start_s:end_s]

    model = get_command_model()
    text = transcribe_audio(audio, model)
    return text


# ─── TTS ─────────────────────────────────────────────────────────

async def speak(text: str) -> None:
    clean = text
    for char in "✅❌⚠️🤔🧠🎵":
        clean = clean.replace(char, "")
    clean = clean.strip()
    if not clean:
        return
    mp3_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    communicate = edge_tts.Communicate(clean, TTS_VOICE, rate=TTS_RATE)
    await communicate.save(mp3_path)
    os.system(f"afplay {mp3_path}")
    os.unlink(mp3_path)


# ─── Main loop ────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  Jev Voice v0.3 — Wake Word Mode")
    print("  Say 'hey jev' to activate, then speak your command")
    print("  Press Ctrl+C to exit")
    print("=" * 50)

    # Preload both models
    wake_model = get_wake_model()
    command_model = get_command_model()
    print("  Both models loaded. Listening...\n")

    # Import the brain
    sys.path.insert(0, os.path.dirname(__file__))
    from jev_brain import think

    try:
        while True:
            # Phase 1: Listen for wake word
            wake_detected = listen_for_wake_word()

            if not wake_detected:
                continue

            # Optional: play a small acknowledgment sound
            print("\r⚡  Hey Jev detected!              ", flush=True)

            # Phase 2: Record the command
            transcript = record_command()

            if not transcript:
                print("  (didn't catch that)")
                continue

            print(f'  📝 You said: "{transcript}"')

            # Check for exit command
            if any(kw in transcript for kw in ["goodbye", "exit", "shut down", "go to sleep"]):
                print("  Jev: Going to sleep. Talk later! 👋")
                asyncio.run(speak("Going to sleep. Talk later!"))
                break

            # Phase 3: Process through the two-model pipeline
            result = think(transcript)
            print(f"  {result}")

            # Phase 4: Speak the reply
            print("  🔊 Speaking...")
            asyncio.run(speak(result))

    except KeyboardInterrupt:
        print("\n\n  Jev: Shutting down. Bye! 👋")


if __name__ == "__main__":
    main()
