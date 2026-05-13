"""
Pidgin Whisper — HuggingFace Space demo.

Loads `openai/whisper-large-v3-turbo` + the LoRA adapter from
`michaelodafe/whisper-pidgin-v1`, merges in-memory, and transcribes
audio uploaded or recorded by the user.

Targets free-CPU Spaces; latency is ~5–15 s per clip on CPU.
"""
import os
import re

import gradio as gr
import librosa
import numpy as np
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

BASE = "openai/whisper-large-v3-turbo"
ADAPTER = "michaelodafe/whisper-pidgin-v1"

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


print("Loading processor + base model + adapter...")
processor = WhisperProcessor.from_pretrained(BASE, language="english", task="transcribe")
base = WhisperForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.float32)
peft_model = PeftModel.from_pretrained(base, ADAPTER)
model = peft_model.merge_and_unload().eval()
model.generation_config.language = "english"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.generation_config.suppress_tokens = []
prompt_ids = processor.get_prompt_ids(INITIAL_PROMPT, return_tensors="pt")
print("Ready.")


@torch.no_grad()
def transcribe(audio):
    if audio is None:
        return ""
    sr, arr = audio
    if arr.dtype.kind == "i":
        arr = arr.astype("float32") / np.iinfo(arr.dtype).max
    else:
        arr = arr.astype("float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != 16000:
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
    inputs = processor(arr, sampling_rate=16000, return_tensors="pt")
    out = model.generate(
        inputs.input_features,
        max_length=225,
        prompt_ids=prompt_ids,
    )
    text = processor.batch_decode(out, skip_special_tokens=True)[0]
    return postprocess(text.strip().lower())


with gr.Blocks(title="Pidgin Whisper") as demo:
    gr.Markdown("# Pidgin Whisper")
    gr.Markdown(
        "Nigerian Pidgin English speech-to-text. Upload an audio clip "
        "or record one from your mic, then press Transcribe.\n\n"
        "Trained on ~8.6 h of Pidgin audio with LoRA on top of "
        "`openai/whisper-large-v3-turbo`. **21.37% WER** on the held-out test "
        "set vs the published 29.6% baseline. "
        "[Model](https://huggingface.co/michaelodafe/whisper-pidgin-v1) · "
        "[Source](https://github.com/michaelodafe/Naija-Pidgin-Whisper)"
    )
    with gr.Row():
        audio_in = gr.Audio(label="Audio", sources=["microphone", "upload"])
    with gr.Row():
        btn = gr.Button("Transcribe", variant="primary")
    out = gr.Textbox(label="Transcription", lines=4)
    btn.click(transcribe, inputs=audio_in, outputs=out)
    gr.Examples(
        examples=[],
        inputs=audio_in,
    )

demo.launch()
