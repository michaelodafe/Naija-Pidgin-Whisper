# Pidgin Whisper

Open-source Nigerian Pidgin English speech-to-text. Fine-tunes
[`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo)
with LoRA on ~8.6 hours of curated Pidgin audio. Trained on a free
Kaggle T4 in under 4 hours.

## Result

| Metric | Pidgin Whisper v1 | Wav2Vec2-XLSR-53 baseline |
|---|---|---|
| Test WER | **21.37%** | 29.6% |
| Test CER | **9.90%** | — |

That's an **8.2 percentage-point absolute improvement (28% relative)**
over the strongest published Pidgin ASR result on the same dataset,
using a free GPU and a single fine-tuning run.

## Links

- 🤗 **Model:** [michaelodafe/whisper-pidgin-v1](https://huggingface.co/michaelodafe/whisper-pidgin-v1) (LoRA adapter, 26 MB)
- 🎤 **Demo Space:** [try it in your browser](https://huggingface.co/spaces/michaelodafe/pidgin-whisper)
- 📦 **Dataset:** [michaelodafe/pidgin-asr-combined](https://huggingface.co/datasets/michaelodafe/pidgin-asr-combined) (~8.6 h, combined from public Pidgin sources)
- 📖 **Full design notes:** [documentation.md](documentation.md)

## Quick start — run locally

```bash
git clone https://github.com/michaelodafe/Naija-Pidgin-Whisper.git
cd pidgin-whisper
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: paste your HF_TOKEN

# one-time: pull base model + adapter, merge, convert to int8 CTranslate2 (~5 min)
HF_HUB_DISABLE_XET=1 python infer/01_merge_and_convert.py

# live mic → Pidgin transcription
python infer/02_streaming_demo.py
```

Speak into your mic; transcripts print after each utterance. Ctrl-C to quit.

On macOS, grant microphone permission to Terminal/iTerm on first run.

## Reproducing v1

End-to-end training pipeline — about 1 hour of your time + 4 hours of
free Kaggle GPU.

1. **Build the dataset** — `python scripts/01_fetch_data.py` pulls and
   normalizes [`asr-nigerian-pidgin/nigerian-pidgin-1.0`](https://huggingface.co/datasets/asr-nigerian-pidgin/nigerian-pidgin-1.0)
   and [`Rexe/nigerian-pidgin-speech`](https://huggingface.co/datasets/Rexe/nigerian-pidgin-speech),
   pushes the combined dataset to your HF account.
2. **Inspect** — `python scripts/02_inspect.py` writes 20 random sample
   clips to `data/samples/` so you can listen.
3. **Train on Kaggle** — paste cells from
   [`notebooks/03_finetune_kaggle.py`](notebooks/03_finetune_kaggle.py)
   into a Kaggle GPU notebook. Add `HF_TOKEN` to Kaggle Secrets. Set
   accelerator to T4 x1, internet on.
4. **Inference locally** — see Quick start above.

Full step-by-step in [documentation.md](documentation.md).

## How it works

- **Base model:** `openai/whisper-large-v3-turbo` (809M params).
- **Fine-tune:** LoRA `r=32, alpha=64`, target modules `q_proj, v_proj`.
  Effective batch 16 (4 per device × 4 grad-accum), LR 1e-4, 5 epochs.
- **Inference:** `faster-whisper` (CTranslate2 `int8_float16`) — 4–6×
  faster than transformers, 781 MB on disk after quantization.
- **Streaming:** Silero VAD detects utterance boundaries, model
  transcribes after each pause. ~200–600 ms latency on a Mac CPU.
- **Decode-time enhancements** (Path A in `infer/decode.py`):
  - `initial_prompt` hotwords (Nigerian proper nouns + Pidgin function
    words) bias the decoder toward correct vocabulary.
  - Postprocess strips punctuation and merges digit groups to match
    training-label conventions.

## Repo layout

```
pidgin-whisper/
├── scripts/
│   ├── 01_fetch_data.py          build + push combined dataset to HF
│   ├── 02_inspect.py             sample audio + stats
│   └── 05_eval_pathA.py          A/B/C/D eval of decode-time enhancements
├── notebooks/
│   └── 03_finetune_kaggle.py     LoRA training notebook (Kaggle T4)
├── infer/
│   ├── 01_merge_and_convert.py   merge LoRA + export CTranslate2 int8
│   ├── 02_streaming_demo.py      live mic → Pidgin transcript
│   ├── 03_test_on_samples.py     batch transcribe data/samples/
│   ├── 05_create_endpoint_repo.py push deployment repo to HF
│   ├── decode.py                 Path A hotwords + postprocess
│   └── endpoint/
│       ├── handler.py            HF Inference Endpoint custom handler
│       ├── requirements.txt
│       └── test_endpoint.py      smoke-test client
├── space/                        HF Spaces demo (Gradio)
├── documentation.md              full design notes + post-v1 work log
└── requirements.txt
```

## Deployment

Two paths supported:

**HuggingFace Inference Endpoint** (cloud, ~$10–15/month for low traffic)
— see [`infer/endpoint/`](infer/endpoint/) and `infer/05_create_endpoint_repo.py`.
T4 GPU, scale-to-zero, ~150–400 ms warm latency.

**HF Space** (free CPU demo for browser use) — see [`space/`](space/).

## Limitations

- Trained on **read-style news Pidgin** (BBC News Pidgin register). Casual
  conversational Pidgin will have higher WER.
- Code-switching to English mid-utterance is handled, but not heavy
  code-switching with Yoruba / Igbo / Hausa — those weren't in training.
- The model normalizes some Pidgin orthographic variants (`hin` ↔ `him`,
  `kain` ↔ `kind`) — partially a label-inconsistency artifact in the
  source dataset.
- Audio over 30 s gets silently truncated to Whisper's 30-s window.

## License

MIT for the code in this repo.
The trained model adapter inherits Whisper's MIT license, but is fine-tuned
on `asr-nigerian-pidgin/nigerian-pidgin-1.0` (CC-BY-4.0); attribution
required.

## Acknowledgments

- The [`asr-nigerian-pidgin/nigerian-pidgin-1.0`](https://huggingface.co/datasets/asr-nigerian-pidgin/nigerian-pidgin-1.0)
  dataset team for the only sizeable open Pidgin ASR corpus.
- OpenAI for Whisper.
- HuggingFace for the model + dataset hosting and `transformers` /
  `datasets` / `peft` libraries.
- SYSTRAN for `faster-whisper` and CTranslate2.
- Silero team for the VAD.
- Kaggle for free GPU compute.
