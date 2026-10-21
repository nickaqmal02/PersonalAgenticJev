"""
Debug script — test what Whisper hears when you say 'hey jev'
"""
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
GAIN_BOOST = 5.0

print("Loading Whisper base model...")
model = WhisperModel("base", device="cpu", compute_type="int8")
print("Ready!\n")

print("Listening in 3-second chunks. Say 'hey jev' clearly.")
print("Press Ctrl+C to stop.\n")

chunk_samples = int(3.0 * SAMPLE_RATE)

for attempt in range(1, 20):
    print(f"\r🟡 Chunk {attempt}: listening...     ", end="", flush=True)
    
    frames = []
    
    def callback(indata, frame_count, time_info, status):
        boosted = np.clip(indata.copy() * GAIN_BOOST, -1.0, 1.0)
        frames.append(boosted)
    
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                           callback=callback, blocksize=1024, device=1) as stream:
            stream.start()
            import time
            start = time.time()
            while stream.active and (time.time() - start) < 3.0:
                time.sleep(0.05)
    except Exception as e:
        print(f" Error: {e}")
        continue
    
    if not frames:
        print(" (no audio)")
        continue
    
    audio = np.concatenate(frames, axis=0)
    rms = np.sqrt(np.mean(audio**2))
    
    # Skip silence
    if rms < 0.003:
        print(f"\r⚪ Chunk {attempt}: silence (RMS {rms:.4f})     ", end="", flush=True)
        continue
    
    # Transcribe — condition_on_previous_text=False fixes empty transcripts
    segments, _ = model.transcribe(audio, language="en", beam_size=5,
                                   condition_on_previous_text=False,
                                   initial_prompt="hey jev open spotify battery volume")
    text = " ".join(s.text for s in segments).strip()
    
    print(f"\r🔊 Chunk {attempt}: RMS={rms:.4f} | Transcript: \"{text}\"")
