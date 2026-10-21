"""
Jev Voice — Phase 5 (Voice)

Speak to Jev, Jev speaks back.

Two modes:
  1. Push-to-talk (default): press ENTER → speak → press ENTER again → transcribe
  2. Auto silence: record until 2s of silence detected

Pipeline:
    1. Record audio from mic
    2. Transcribe with faster-whisper (local, no API key)
    3. Feed transcript to jev_brain.think() → DeepSeek + Jev pipeline
    4. Speak the reply with edge-tts (free, no API key)
"""

import os
import sys
import wave
import tempfile
import asyncio
import threading

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
SAMPLE_RATE = 16000
CHANNELS = 1
GAIN_BOOST = 10.0            # amplify mic input (macOS built-in mic is quiet)
SILENCE_THRESHOLD = 0.015
SILENCE_DURATION = 2.0
MAX_RECORDING_SECONDS = 15
WHISPER_MODEL = "base"
TTS_VOICE = "en-US-GuyNeural"
TTS_RATE = "+0%"
USE_PUSH_TO_TALK = True      # True = press ENTER to stop, False = auto silence


# ─── Recording ───────────────────────────────────────────────────

def record_audio_push_to_talk() -> str:
    """Push-to-talk: press ENTER to start, press ENTER to stop."""
    input("  Press ENTER to start recording, then speak...")
    print("  🎤 Recording... (press ENTER to stop)")

    frames = []
    stop_flag = threading.Event()

    # Listen for ENTER in a background thread
    def wait_for_enter():
        input()
        stop_flag.set()

    listener = threading.Thread(target=wait_for_enter, daemon=True)
    listener.start()

    def callback(indata, frame_count, time_info, status):
        boosted = indata.copy() * GAIN_BOOST
        boosted = np.clip(boosted, -1.0, 1.0)
        frames.append(boosted)

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            callback=callback,
            blocksize=1024,
        ) as stream:
            stream.start()
            import time
            start = time.time()
            while stream.active and not stop_flag.is_set() and (time.time() - start) < MAX_RECORDING_SECONDS:
                time.sleep(0.05)
    except Exception as e:
        print(f"  Mic error: {e}")
        return ""

    if not frames:
        print("  (no audio captured)")
        return ""

    audio = np.concatenate(frames, axis=0)
    rms = np.sqrt(np.mean(audio**2))
    print(f"  📊 Audio level: {rms:.4f} ({'good' if rms > 0.02 else 'quiet'})")

    return save_wav(audio)


def record_audio_auto() -> str:
    """Auto: record until silence detected."""
    print("  🎤 Listening... (speak now)")

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
            import time
            start = time.time()
            while stream.active and (time.time() - start) < MAX_RECORDING_SECONDS:
                time.sleep(0.1)
    except sd.CallbackStop:
        pass
    except Exception as e:
        print(f"  Mic error: {e}")
        return ""

    if not frames or not heard_speech:
        print("  (no speech detected)")
        return ""

    audio = np.concatenate(frames, axis=0)
    return save_wav(audio)


def save_wav(audio: np.ndarray) -> str:
    """Save numpy audio to WAV file."""
    wav_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    with wave.open(wav_path, "w") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        audio_int = (audio * 32767).astype(np.int16)
        wf.writeframes(audio_int.tobytes())
    return wav_path


def record_audio() -> str:
    """Record audio — push-to-talk or auto mode."""
    if USE_PUSH_TO_TALK:
        return record_audio_push_to_talk()
    return record_audio_auto()


# ─── Transcription ────────────────────────────────────────────────

def transcribe(wav_path: str, model: WhisperModel) -> str:
    """Transcribe WAV → text with faster-whisper (numpy array, bypass av bug)."""
    with wave.open(wav_path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    # VAD: trim leading/trailing silence
    # Find where speech starts and ends
    window = 1600  # 100ms windows
    speech_windows = []
    for i in range(0, len(audio) - window, window):
        chunk = audio[i:i+window]
        rms = np.sqrt(np.mean(chunk**2))
        if rms > 0.005:  # lower threshold for VAD
            speech_windows.append(i)

    if speech_windows:
        start = max(0, speech_windows[0] - window)
        end = min(len(audio), speech_windows[-1] + window * 2)
        audio = audio[start:end]
        print(f"  ✂️ Trimmed to speech: {start/16000:.1f}s - {end/16000:.1f}s")

    segments, _ = model.transcribe(audio, language="en", beam_size=5, best_of=3)
    text = " ".join(segment.text for segment in segments).strip()
    return text


# ─── TTS ─────────────────────────────────────────────────────────

async def speak(text: str) -> None:
    """Convert text to speech and play it."""
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


# ─── Main ─────────────────────────────────────────────────────────

def main():
    print(f"Loading Whisper model '{WHISPER_MODEL}'...")
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    print("Whisper ready.\n")

    sys.path.insert(0, os.path.dirname(__file__))
    from jev_brain import think

    print("=" * 50)
    print("  Jev Voice v0.2")
    if USE_PUSH_TO_TALK:
        print("  Press ENTER to start, ENTER to stop recording")
    else:
        print("  Press ENTER to start (auto-stops on silence)")
    print("  Type 'quit' to exit")
    print("=" * 50)

    while True:
        try:
            user_input = input("\njev> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            break

        if user_input:
            # Text input — skip recording
            pass
        else:
            # Voice mode
            wav_path = record_audio()
            if not wav_path:
                continue

            transcript = transcribe(wav_path, model)
            os.unlink(wav_path)

            if not transcript:
                print("  (didn't catch that — try again, speak louder)")
                continue

            print(f'  📝 You said: "{transcript}"')

            if transcript.lower() in ("quit", "exit", "stop", "goodbye"):
                break

            user_input = transcript

        result = think(user_input)
        print(f"  {result}")
        print("  🔊 Speaking...")
        asyncio.run(speak(result))


if __name__ == "__main__":
    main()
