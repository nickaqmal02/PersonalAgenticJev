"""
Debug 2 — record with ffmpeg (bypass sounddevice), save WAV, play it back, transcribe
"""
import subprocess
import numpy as np
import wave
from faster_whisper import WhisperModel

WAV_PATH = "/tmp/jev_test_input.wav"

print("=" * 50)
print("  This will record 4 seconds from your EarPods mic")
print("  SAY 'hey jev, open spotify' WHEN YOU SEE THE PROMPT")
print("=" * 50)

input("\nPress ENTER when ready...")

print("\n🎤 RECORDING NOW — SPEAK!")
# Use ffmpeg to record — it handles macOS audio devices properly
subprocess.run([
    "ffmpeg", "-f", "avfoundation", "-i", ":1",
    "-t", "4", "-ar", "16000", "-ac", "1",
    WAV_PATH, "-y"
], capture_output=True)

print("✅ Recording done!\n")

# Load the WAV
with wave.open(WAV_PATH, "rb") as wf:
    frames = wf.readframes(wf.getnframes())
    sr = wf.getframerate()
    channels = wf.getnchannels()

audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
rms = np.sqrt(np.mean(audio**2))
peak = np.max(np.abs(audio))
duration = len(audio) / sr

print(f"Duration: {duration:.1f}s")
print(f"Sample rate: {sr}")
print(f"RMS: {rms:.6f}")
print(f"Peak: {peak:.6f}")

# Play it back so you can HEAR what was recorded
print("\n🔊 Playing back what was recorded...")
subprocess.run(["afplay", WAV_PATH])

# Now transcribe with base model
print("\n📝 Transcribing with Whisper base...")
model = WhisperModel("base", device="cpu", compute_type="int8")

# Try multiple settings
for label, kwargs in [
    ("default", {}),
    ("no_condition", {"condition_on_previous_text": False}),
    ("with_prompt", {"condition_on_previous_text": False, "initial_prompt": "hey jev open spotify battery"}),
    ("auto_lang", {"language": None, "condition_on_previous_text": False}),
]:
    segments, _ = model.transcribe(audio, beam_size=5, **kwargs)
    text = " ".join(s.text for s in segments).strip()
    print(f"  {label:15s}: \"{text}\"")

# Also try with tiny model
print("\n📝 Trying tiny model...")
model_tiny = WhisperModel("tiny", device="cpu", compute_type="int8")
segments, _ = model_tiny.transcribe(audio, beam_size=1, condition_on_previous_text=False)
text = " ".join(s.text for s in segments).strip()
print(f"  tiny           : \"{text}\"")
