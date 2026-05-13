"""
HuggingFace Inference Endpoints custom handler for Pidgin Whisper v1.

Loads the CT2 int8/fp16 model from ./ct2/ and serves transcription via
faster-whisper. Includes Path A pipeline (initial-prompt hotwords +
postprocess).

Endpoint contract (JSON in/out):
  POST /  body: { "inputs": "<base64-encoded WAV/MP3/Opus audio>" }
        ret:  { "text": "<transcript>" }
"""
import base64
import io
import os
import re
from typing import Any, Dict

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel


INITIAL_PROMPT = (
    "dis na bbc news pidgin tori about buhari tinubu atiku saraki "
    "tony nwoye femi otedola akinwunmi ambode oseloka henry obaze "
    "zainab balogun jimoh moshood and aisha for nigeria politics "
    "for states like lagos anambra delta kogi niger abuja kano rivers "
    "edo ogun salford and offa with organizations like apc pdp nema "
    "jamb frsc brt jp morgan and wikipedia pipo dey tok say di dey na "
    "wey pikin tori sabi hapun sometin anytin becos redi alredi neva "
    "dem una abi oga chillax snakebite"
)

_DIGIT_PAIR = re.compile(r"(\d) (\d)")
_PUNCT = re.compile(r"[.,!?;:\"]")
_INTRA_NUM_COMMA = re.compile(r"(\d),(\d)")


def postprocess(text: str) -> str:
    while True:
        new = _INTRA_NUM_COMMA.sub(r"\1\2", text)
        if new == text:
            break
        text = new
    text = _PUNCT.sub("", text)
    while True:
        new = _DIGIT_PAIR.sub(r"\1\2", text)
        if new == text:
            break
        text = new
    return re.sub(r" +", " ", text).strip()


class EndpointHandler:
    def __init__(self, path: str = ""):
        ct2_dir = os.path.join(path, "ct2") if os.path.isdir(os.path.join(path, "ct2")) else path
        if torch.cuda.is_available():
            device, compute_type = "cuda", "float16"
        else:
            device, compute_type = "cpu", "int8"
        self.asr = WhisperModel(ct2_dir, device=device, compute_type=compute_type)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Accept either raw bytes or {"inputs": <base64>} / {"audio": <base64>}
        if isinstance(data, (bytes, bytearray)):
            audio_bytes = bytes(data)
        else:
            payload = data.get("inputs", data.get("audio"))
            if isinstance(payload, str):
                audio_bytes = base64.b64decode(payload)
            elif isinstance(payload, (bytes, bytearray)):
                audio_bytes = bytes(payload)
            else:
                return {"error": "Send audio bytes or base64 in 'inputs'"}

        try:
            audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        except Exception as e:
            return {"error": f"Could not decode audio: {e}"}

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            except ImportError:
                return {"error": "Server cannot resample; send 16kHz audio"}

        segments, _ = self.asr.transcribe(
            audio.astype(np.float32),
            language="en",
            task="transcribe",
            beam_size=1,
            initial_prompt=INITIAL_PROMPT,
        )
        text = "".join(s.text for s in segments).strip().lower()
        return {"text": postprocess(text)}
