---
title: Pidgin Whisper
emoji: 🇳🇬
colorFrom: green
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: Nigerian Pidgin English speech-to-text
---

# Pidgin Whisper — HF Space

Browser demo for [`michaelodafe/whisper-pidgin-v1`](https://huggingface.co/michaelodafe/whisper-pidgin-v1).

Records or uploads audio, transcribes via the LoRA-finetuned Whisper.

This Space loads on free CPU hardware. Latency is ~5–15 s per clip;
for production use, see the
[full project repository](https://github.com/michaelodafe/Naija-Pidgin-Whisper)
for the faster `faster-whisper` deployment path.
