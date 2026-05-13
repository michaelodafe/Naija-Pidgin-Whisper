"""
Live mic → Nigerian Pidgin transcription. Wispr-Flow-style:
speak, pause, see the transcript. Ctrl-C to quit.

Run `python infer/01_merge_and_convert.py` first.
"""
import queue
import sys
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from silero_vad import VADIterator, load_silero_vad

from decode import transcribe as decode_transcribe

CT2_DIR = Path(__file__).parent / "whisper-pidgin-ct2"
SR = 16000
WIN = 512                  # samples per VAD window (32 ms @ 16 kHz)
PRE_ROLL_BLOCKS = 10       # ~320 ms of audio kept ahead of detected speech start
MIN_UTTER_SAMPLES = SR // 4   # ignore <250 ms blips

if not CT2_DIR.exists():
    sys.exit(f"Missing {CT2_DIR}. Run infer/01_merge_and_convert.py first.")

print("Loading ASR (faster-whisper, int8)...")
asr = WhisperModel(str(CT2_DIR), device="cpu", compute_type="int8")
print("Loading VAD (Silero)...")
vad_iter = VADIterator(load_silero_vad(), sampling_rate=SR)

audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
def on_audio(indata, frames, time_info, status):
    audio_q.put(indata[:, 0].copy())


def transcribe(audio: np.ndarray) -> None:
    if len(audio) < MIN_UTTER_SAMPLES:
        return
    text = decode_transcribe(asr, audio, use_hotwords=True, use_postprocess=True)
    if text:
        print(f"> {text}", flush=True)


pre_roll: deque = deque(maxlen=PRE_ROLL_BLOCKS)
speech_buf: list = []
in_speech = False

print("\nListening. Speak Pidgin (Ctrl-C to quit).\n")
with sd.InputStream(channels=1, samplerate=SR, blocksize=WIN,
                    dtype="float32", callback=on_audio):
    try:
        while True:
            chunk = audio_q.get()
            if len(chunk) != WIN:
                continue
            event = vad_iter(torch.from_numpy(chunk), return_seconds=False)
            if in_speech:
                speech_buf.append(chunk)
            else:
                pre_roll.append(chunk)
            if event and "start" in event and not in_speech:
                in_speech = True
                speech_buf = list(pre_roll)
                pre_roll.clear()
            elif event and "end" in event and in_speech:
                in_speech = False
                audio = np.concatenate(speech_buf)
                speech_buf = []
                transcribe(audio)
    except KeyboardInterrupt:
        print("\nBye.")
