# Pidgin Whisper — Project Documentation

End-to-end record of building a Nigerian Pidgin English speech-to-text model
by LoRA-fine-tuning `openai/whisper-large-v3-turbo`. Covers research,
architecture decisions, data pipeline, training, every error encountered
along the way and how it was fixed, final metrics, and the real-time
inference setup.

**Result:** 21.37% WER / 9.90% CER on a held-out Pidgin test set, beating
the published Wav2Vec2-XLSR-53 baseline (29.6% WER) by 8.2 percentage
points (28% relative improvement). Trained in ~3h47m on a free Kaggle T4.

---

## Table of contents
1. [Goal and constraints](#1-goal-and-constraints)
2. [Research phase](#2-research-phase)
3. [Architecture decisions](#3-architecture-decisions)
4. [Repository layout](#4-repository-layout)
5. [Environment setup](#5-environment-setup)
6. [Data pipeline](#6-data-pipeline)
7. [Training](#7-training)
8. [Issues encountered and fixes](#8-issues-encountered-and-fixes)
9. [Final results](#9-final-results)
10. [Inference and deployment](#10-inference-and-deployment)
11. [Reproduction guide](#11-reproduction-guide)
12. [Future work (v2)](#12-future-work-v2)

---

## 1. Goal and constraints

**Goal:** A real-time speech-to-text model (Wispr-Flow style — speak, pause,
get transcript) for Nigerian Pidgin English, including English↔Pidgin
code-switching mid-utterance.

**User-provided constraints:**
- Compute: Kaggle free tier (T4 x2 or P100, 30 GPU-hrs/week, 9-hr session limit).
- Use case: real-time mic dictation, not batch transcription.
- Licensing: OK with permissive (CC-BY) datasets for a public release.
- Code-switching: must handle mid-utterance switches between Pidgin and
  standard English.

**Implicit constraints derived from the above:**
- LoRA fine-tune mandatory (full fine-tune of large models doesn't fit T4 16GB VRAM).
- Model size matters: larger = more accurate but blows latency budget.
- 9-hour sessions → checkpoint frequently; design to be resumable.
- Real-time deployment requires CTranslate2-style optimization.

---

## 2. Research phase

### 2.1 The originally proposed dataset turned out to be text-only

User initially pointed at the HuggingFace collection
`saheedniyi/naijaweb-datasets`, expecting it to contain Nigerian Pidgin
audio. A WebFetch on the collection revealed:

| Dataset | Modality | Size | Notes |
|---|---|---|---|
| `saheedniyi/naijaweb` | Text only | 270k pages | Recreation of FineWeb for Nigerian web content |
| `saheedniyi/naijaweb-edu` | Text only | 4.4k | Educational subset |
| `saheedniyi/naijaweb-edu2` | Text only | 21.9k | Extended educational subset |

No audio. The collection cannot be used to fine-tune ASR. It is, however,
useful as a **language model corpus** for decoder rescoring (Phase 3
future work).

### 2.2 Surveying actual Pidgin speech corpora

Found via web search:

| Source | Hours | Speakers | License | Use |
|---|---|---|---|---|
| `asr-nigerian-pidgin/nigerian-pidgin-1.0` | ~8.6h | 10 | CC-BY-4.0 | Primary training data |
| `Rexe/nigerian-pidgin-speech` | ~0.05h | 1 (YouTube song) | — | Eval-only (too small to train) |
| African Voices | 3,000+ hrs (multi-language) | many | mixed | Future v2 |
| Common Voice | unknown Pidgin volume | community | CC0 | Future v2 |
| BBC News Pidgin (broadcast) | unbounded | many | weak-supervision target | Future v2 |

The `asr-nigerian-pidgin/nigerian-pidgin-1.0` dataset already has a
published ASR baseline (Wav2Vec2-XLSR-53 at 29.6% WER), giving a clear
number to beat.

### 2.3 Surveying open-source STT models

| Model | Params | Pros | Cons |
|---|---|---|---|
| Whisper large-v3 | 1.55B | SOTA quality, mature ecosystem | Too large for free Kaggle without LoRA |
| **Whisper large-v3-turbo** | **809M** | **~8× faster than v3, near-equal quality, fits T4 with LoRA** | **Chosen** |
| Distil-Whisper large-v3 | 756M | 6× faster, smaller, simple | Slight quality drop |
| Wav2Vec2-XLSR-53 | 317M | Existing Pidgin baseline | Older, weaker than Whisper-v3 |
| MMS (Meta) | 1B+ | Built for low-resource languages | Less mature fine-tune tooling |
| Canary-Qwen 2.5B | 2.5B | Top of HF ASR leaderboard | English-centric, hard to adapt |
| Qwen3-ASR | 1.7B | 52 languages | Less mature fine-tune tooling |
| Moonshine | 27MB+ | Real-time on edge devices | English-only base |

**Decision: `openai/whisper-large-v3-turbo`.** Best ratio of quality, speed,
fine-tunability, and ecosystem maturity for the Kaggle constraint.

---

## 3. Architecture decisions

### 3.1 Training architecture

| Decision | Choice | Reasoning |
|---|---|---|
| Base model | `openai/whisper-large-v3-turbo` | Best balance for T4 |
| Fine-tune method | LoRA | Full fine-tune doesn't fit 16GB |
| LoRA target modules | `q_proj`, `v_proj` | Standard minimal set; r=32, alpha=64 |
| Mixed precision | fp16 | Halves activation memory |
| Compute | Kaggle T4 x1 | T4 x2 caused DataParallel OOM (see §8.5) |
| Effective batch | 16 (4 × 4 grad accum) | Largest stable size on single T4 |
| Epochs | 5 | Empirically right for 8.6h data |
| LR | 1e-4 with 5% warmup | Standard for Whisper LoRA |
| Eval/save cadence | every 200 steps | ~5 evals over 845-step run |

### 3.2 Code-switching handling

Decision: **No special handling needed during training.** The dataset
already contains naturally code-switched utterances (proper nouns,
loanwords, formal registers). Filtering to "Pidgin only" would actually
harm the model. We pass `language="english"` as Whisper's language token
since Pidgin doesn't have its own token; the model adapts via the new
weights.

### 3.3 Real-time deployment architecture

Whisper isn't natively streaming (it operates on 30s windows). We approximate
real-time by detecting utterance boundaries via VAD and transcribing each
complete utterance:

```
mic → sounddevice → 32ms chunks
                  → Silero VAD (start/end detection)
                  → buffer speech between events
                  → on "end" event: send buffer to faster-whisper
                  → print transcript
```

Latency budget: ~200–600ms after speaker stops, on Mac CPU with int8
quantization.

### 3.4 Inference optimization

| Step | Choice | Reasoning |
|---|---|---|
| Adapter merge | `peft.merge_and_unload()` | Bakes LoRA into base weights for static export |
| Backend | CTranslate2 (via faster-whisper) | 4–6× faster than HF transformers on CPU |
| Quantization | `int8_float16` | 800 MB on disk vs ~3 GB fp32; minimal quality loss |
| VAD | Silero VAD | More accurate than energy-based, low CPU cost |
| Decoder | `beam_size=1` (greedy) | Real-time priority over marginal accuracy |

---

## 4. Repository layout

```
Pidgin Whisper/
├── .env.example                  # template; copy to .env, never commit secrets
├── .gitignore                    # excludes .env, audio, weights
├── requirements.txt              # all deps (training + inference)
├── documentation.md              # this file
├── fetch.log                     # output from data fetch run
├── data/
│   └── samples/                  # 20 audio samples + transcripts.txt for sanity-check
├── scripts/
│   ├── 01_fetch_data.py          # pulls + normalizes + dedupes + pushes to HF
│   └── 02_inspect.py             # stats + dumps samples
├── notebooks/
│   └── 03_finetune_kaggle.py     # Kaggle training script (cell-delimited)
└── infer/
    ├── 01_merge_and_convert.py   # LoRA merge + CTranslate2 export
    ├── 02_streaming_demo.py      # live mic → Pidgin transcript
    ├── 03_test_on_samples.py     # smoke test on data/samples/
    ├── merge.log                 # output from merge run
    └── whisper-pidgin-ct2/       # ~781 MB int8 model (gitignored)
```

---

## 5. Environment setup

- macOS Darwin 24.5.0 (Apple Silicon)
- Python 3.9.6 (system)
- Local venv: `python3 -m venv .venv`
- `requirements.txt` pinned `datasets>=2.20,<4.0` to avoid the torchcodec
  audio-decode dependency introduced in `datasets` 4.x (see §8.2).

Tokens are stored only in `.env` (gitignored). Never written to source
files.

---

## 6. Data pipeline

### 6.1 Source dataset schemas

**`asr-nigerian-pidgin/nigerian-pidgin-1.0`**

| Column | Type | Notes |
|---|---|---|
| `sentence` | string | Transcription, length 1–433 chars |
| `filename` | string | Unique audio ID |
| `audio` | Audio | 16kHz, 0.5–40.5s |

Splits: train 2,710 / validation 677 / test 892.

**`Rexe/nigerian-pidgin-speech`**

| Column | Type | Notes |
|---|---|---|
| `audio` | Audio | 16kHz |
| `text` | string | 3–50 chars |
| `duration` | float | 0.64–12.4s |
| `source` | string | Always "youtube" |
| `video_title` | string | Single video title |
| `transcription_source` | string | Always "unknown" |

Splits: train 65 / test 8. Total 73 samples — too small for training,
all routed to the test pool for diversity.

### 6.2 Unified schema

```
{
  audio:       Audio(sampling_rate=16000),
  text:        str,                    # transcription, stripped
  source:      str,                    # which source dataset
  duration:    float,                  # seconds
  speaker_id:  str,                    # "" if unknown
}
```

### 6.3 Pipeline ([scripts/01_fetch_data.py](scripts/01_fetch_data.py))

1. `load_dataset` for both sources.
2. `cast_column("audio", Audio(sampling_rate=16000))` — enforce 16kHz.
3. `.map()` per-source normalizers convert to the unified schema and drop
   extra columns.
4. Combine: P1 train→train, P1 val→val, P1 test + Rexe train + Rexe test→test.
5. Dedupe by MD5 hash of the audio array (each split independently).
6. Print per-split stats.
7. `push_to_hub("michaelodafe/pidgin-asr-combined", private=True)`.

### 6.4 Final dataset stats

| Split | Clips | Hours | Mean dur | Min/Max dur |
|---|---|---|---|---|
| train | 2,708 | 5.41 | 7.2s | 0.5–40.5s |
| validation | 677 | 1.37 | 7.3s | 0.6–38.2s |
| test | 893 | 1.78 | 7.2s | 1.4–44.7s |
| **Total** | **4,278** | **~8.56** | | |

Note: the user's audio peaks above 30s — Whisper's input window is
exactly 30s. Filter applied at training time
(`row["duration"] <= 30.0`).

### 6.5 Inspection ([scripts/02_inspect.py](scripts/02_inspect.py))

Loads the combined dataset back from HF, prints stats, and dumps 20
random `train` samples + transcripts to `data/samples/` for manual
listening.

Sample transcripts confirmed authentic Pidgin lexicon:
"tori", "pipo", "pikin", "wia", "wey", "dey", "siddon", "anytin",
"hapun", "sometin", with naturally code-switched English proper nouns
("buhari", "femi otedola", "tevez", "anambra", "kogi").

---

## 7. Training

Notebook: [notebooks/03_finetune_kaggle.py](notebooks/03_finetune_kaggle.py).
Cells delimited by `# %%`; works with Jupytext or direct paste into a
Kaggle notebook.

### 7.1 Cells

1. Imports + auth (loads `HF_TOKEN` from Kaggle Secrets).
2. Load dataset + filter (`duration <= 30s`, `len(text) >= 3`).
3. `WhisperProcessor` + per-row `prepare()` that produces
   `input_features` (log-mel) and `labels` (token IDs).
4. `WhisperCollator` — pads input features and labels, replaces label
   pad tokens with -100 for loss masking, trims a leading
   `decoder_start_token_id` if present.
5. Load model in fp16 to single CUDA device, set
   `model.generation_config.forced_decoder_ids = None`,
   `model.generation_config.suppress_tokens = []`,
   `language="english"`, `task="transcribe"`. Apply `LoraConfig(r=32,
   alpha=64, target=[q_proj, v_proj], dropout=0.05)`.
6. Compute metrics: WER + CER via `evaluate.load("wer"/"cer")`.
7. `Seq2SeqTrainingArguments` + `Seq2SeqTrainer.train()`.
8. Final eval on test split + push adapter to HF.

### 7.2 Key hyperparameters

```
per_device_train_batch_size = 4
gradient_accumulation_steps = 4   # effective batch 16
per_device_eval_batch_size  = 4
learning_rate               = 1e-4
warmup_ratio                = 0.05
num_train_epochs            = 5
fp16                        = True
predict_with_generate       = True
generation_max_length       = 225
metric_for_best_model       = "wer"
load_best_model_at_end      = True
remove_unused_columns       = False    # required for PEFT
```

Total steps: 5 epochs × ceil(2708 / 16) ≈ 845.

### 7.3 Training environment on Kaggle

- Accelerator: **GPU T4 x1** (T4 x2 caused OOM, see §8.5).
- Internet: On.
- HF token added under Add-ons → Secrets as `HF_TOKEN`.
- `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` set at top of notebook to
  defensively force single GPU.

---

## 8. Issues encountered and fixes

A chronological list of every error hit during the project, with cause
and fix. Each was a real wall-clock setback during the run; documenting
them so v2 doesn't repeat them.

### 8.1 hf-xet upload deadlock

**Symptom:** Initial dataset push to HF hung indefinitely. Process alive
but at 0–1% CPU; all TCP connections in `CLOSE_WAIT`; main thread parked
on `semaphore_wait_trap` inside `hf_xet.abi3.so`.

**Cause:** The new `hf-xet` chunked uploader can deadlock when remote
connections are silently dropped (network instability).

**Fix:** Set `HF_HUB_DISABLE_XET=1` to force the legacy uploader. Slower
but reliable. Also added `python -u` for unbuffered output and redirected
to a log file to actually see progress.

### 8.2 `datasets>=4.0` requires `torchcodec` for audio decoding

**Symptom:** First run of `01_fetch_data.py`:
```
ImportError: To support decoding audio data, please install 'torchcodec'.
```

**Cause:** `datasets` 4.x dropped soundfile as the default audio decoder
in favor of `torchcodec`, which requires PyTorch and a system `ffmpeg`.

**Fix:** Pinned `datasets>=2.20.0,<4.0` in `requirements.txt`. The 3.x line
still uses soundfile and works without ffmpeg.

### 8.3 Missing `evaluate` module on Kaggle

**Symptom:** Cell 6 in the Kaggle notebook:
```
ModuleNotFoundError: No module named 'evaluate'
```

**Cause:** Forgot to include `evaluate` in the install cell.

**Fix:** Added `evaluate jiwer` to the pip install list. (`jiwer` is the
backend `evaluate` uses for WER computation.)

### 8.4 `Seq2SeqTrainer` `tokenizer=` kwarg renamed

**Symptom:**
```
TypeError: Seq2SeqTrainer.__init__() got an unexpected keyword argument 'tokenizer'
```

**Cause:** transformers 4.46+ deprecated/removed `tokenizer=` on Trainer
constructors in favor of `processing_class=`.

**Fix:** Changed to `processing_class=processor.feature_extractor`.

### 8.5 Newer transformers refuses `model.config.suppress_tokens`

**Symptom:** Crash at first eval (step 200):
```
ValueError: You have modified the pretrained model configuration to control generation.
We detected the following values set - {'suppress_tokens': []}.
```

**Cause:** Generation parameters are no longer supposed to live on
`model.config` — they belong on `model.generation_config`. The old idiom
of `model.config.suppress_tokens = []` is rejected at generate-time.

**Fix:** Moved both `forced_decoder_ids` and `suppress_tokens` to
`model.generation_config`.

### 8.6 OOM on T4 x2 with DataParallel

**Symptom:** After fix 8.5, restart hit:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 60.00 MiB.
GPU 0 has a total capacity of 14.56 GiB of which 23.81 MiB is free.
```

**Cause:** Kaggle was set to **T4 x2** (two GPUs). PyTorch silently wrapped
the model in `torch.nn.DataParallel`, replicating it on every GPU and
gathering activations on GPU 0. Combined with `device_map="auto"` (which
also tried to shard), GPU 0 was overloaded.

**Fix:** Three changes:
1. Top of notebook: `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` — force single GPU.
2. Cell 5: removed `device_map="auto"`, replaced with `.to("cuda")`.
3. Cell 7: dropped `per_device_train_batch_size` from 8 → 4, raised
   `gradient_accumulation_steps` from 2 → 4 (effective batch unchanged at 16).
4. Switched Kaggle accelerator setting to **T4 x1** to stop burning credits.

### 8.7 First save didn't happen before crash

**Symptom:** After fix 8.5, the run had been at step 200 when it crashed,
but `/kaggle/working/whisper-pidgin/checkpoint-200/` didn't exist.

**Cause:** The crash happened during the *evaluation* step (which runs
after `save`, but in this transformers version evaluation runs first
under some configs), so the save never completed.

**Fix:** Restarted from scratch after applying fixes 8.5 + 8.6. Lost ~30 min.

### 8.8 Local fetch ran with system Python instead of venv

**Symptom (twice):** Inference scripts ran with `/usr/local/bin/python3`
(system Python) and failed with `ModuleNotFoundError`.

**Cause:** Hard-coded absolute path bypasses the activated venv.

**Fix:** Use plain `python` after `source .venv/bin/activate`.

### 8.9 Cosmetic warnings (ignored)

These appeared but had no functional impact:
- `urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'` — macOS shipping an old LibreSSL. Doesn't break anything.
- `warmup_ratio is deprecated and will be removed in v5.2` — still works in our version.
- `attention mask is not set` — Whisper internal, doesn't apply.
- `custom logits processor … will take precedence` — Whisper applies suppress_tokens internally; trainer also passes them; harmless duplicate.
- `torch_dtype is deprecated! Use dtype instead` — works in our version.
- `Unexpected keyword arguments ['alora_invocation_tokens', ...] for class LoraConfig` — adapter saved by newer PEFT; older PEFT ignores extras. Worked fine.

---

## 9. Final results

### 9.1 Training trajectory

| Step | Train loss | Val loss | Val WER | Val CER |
|---|---|---|---|---|
| 200 | 2.911 | 0.809 | 25.97% | 12.72% |
| 400 | 2.253 | 0.728 | 23.25% | 11.49% |
| 600 | 1.943 | 0.707 | 22.39% | 11.23% |
| 800 | 2.090 | 0.701 | 21.96% | 11.02% |

Train loss kept dropping while val loss flattened — the model began
memorizing the train set in epoch 4–5 but not yet hurting generalization.

### 9.2 Final test-set evaluation

```
TEST: {
  'eval_loss': 0.6551,
  'eval_wer':  0.21370,    # 21.37%
  'eval_cer':  0.09904,    # 9.90%
  'eval_runtime': 1234.50s,
  'eval_samples_per_second': 0.722,
  'epoch': 5.0
}
```

### 9.3 Comparison to baseline

| Metric | Wav2Vec2-XLSR-53 (published) | Pidgin Whisper v1 (this run) | Delta |
|---|---|---|---|
| Test WER | 29.6% | **21.37%** | **−8.23 pp** (28% relative) |
| Test CER | not reported | **9.90%** | — |

### 9.4 Training cost

| Resource | Cost |
|---|---|
| Train runtime | 13,673s (3h 47m) |
| Train throughput | 0.062 steps/s, 0.99 samples/s |
| Total FLOPs | 2.32 × 10¹⁹ |
| GPU hours | 1× T4 × 3.8h ≈ free Kaggle quota |
| Adapter size | 26.2 MB |

---

## 10. Inference and deployment

### 10.1 Adapter merge + CTranslate2 export ([infer/01_merge_and_convert.py](infer/01_merge_and_convert.py))

1. Load `openai/whisper-large-v3-turbo` in fp32.
2. Apply `PeftModel.from_pretrained(base, "michaelodafe/whisper-pidgin-v1")`.
3. `merge_and_unload()` to bake LoRA into base weights.
4. Save merged model as safetensors, copy processor.
5. `python -m ctranslate2.converters.transformers --quantization int8_float16`.
6. Delete intermediate merged model.

Output: `infer/whisper-pidgin-ct2/` (~781 MB on disk).

Files inside:
```
config.json                  # CT2 model config
generation_config.json       # generation parameters
model.bin                    # quantized weights (~814 MB)
preprocessor_config.json     # feature extractor
special_tokens_map.json
tokenizer.json
tokenizer_config.json
vocabulary.json              # CT2 vocabulary (separate from tokenizer)
```

### 10.2 Streaming demo ([infer/02_streaming_demo.py](infer/02_streaming_demo.py))

State machine:

```
                ┌─────────────────────────────┐
                │ pre_roll: deque(maxlen=10)  │  ~320ms ring buffer
                │ speech_buf: list            │  current utterance
                │ in_speech: bool             │
                └─────────────────────────────┘
                              │
                              ▼
mic → 32ms (512-sample) chunks → silero VADIterator(chunk)
                                          │
                              ┌───────────┼───────────┐
                              ▼           ▼           ▼
                          {start: t}    {end: t}     None
                          ──────────    ────────    ──────
                          in_speech =   transcribe  if in_speech: speech_buf.append(chunk)
                            True          (speech    else: pre_roll.append(chunk)
                          speech_buf =     _buf)
                            list(pre_   speech_buf
                            roll)        = []
                          pre_roll      in_speech
                            .clear()      = False
```

Key parameters:
- `SR = 16000` (Whisper requirement)
- `WIN = 512` samples — Silero VAD's required window size at 16kHz
- `PRE_ROLL_BLOCKS = 10` × 32ms = 320ms ring buffer ahead of speech start
  to avoid clipping the first phoneme
- `MIN_UTTER_SAMPLES = SR // 4` — drop blips < 250ms
- Decoder: `language="en", task="transcribe", beam_size=1` (greedy)

### 10.3 Smoke test ([infer/03_test_on_samples.py](infer/03_test_on_samples.py))

Loads the CT2 model and transcribes the 20 samples in `data/samples/`,
comparing to the reference transcripts and reporting WER.

Caveat: those samples come from the **train** split, so the WER (~9.5%)
is artificially low. The honest number is the held-out test eval (21.37%).

### 10.4 Observed error categories (qualitative)

From the smoke test:
- **Number normalization:** "eight" → "8", "30000" → "30 000".
- **Spelling normalization:** model converts informal Pidgin spellings
  toward standard English: "siddon" → "see don", "hin" → "him",
  "alredi" → "already".
- **Abbreviation expansion:** "psquare" → "peace square".
- **Proper-noun mishears:** "buhari" → "pari" — single-instance.

Most are exactly the kind of errors a language-model rescoring step
(Phase 3) would clean up.

---

## 11. Reproduction guide

### 11.1 Prerequisites
- macOS (or Linux) with Python 3.9+.
- HuggingFace account + API token.
- Kaggle account + GPU notebook access.

### 11.2 Local setup

```bash
cd "Pidgin Whisper"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env, fill in HF_TOKEN (and KAGGLE_KEY if using Kaggle CLI)
```

### 11.3 Build dataset

```bash
HF_HUB_DISABLE_XET=1 python -u scripts/01_fetch_data.py
python scripts/02_inspect.py
```

First step downloads ~912 MB of audio, normalizes, dedupes, and pushes
the combined dataset to `michaelodafe/pidgin-asr-combined` (private).
Second dumps 20 samples to `data/samples/` for listening.

### 11.4 Train on Kaggle

1. Create a new Kaggle notebook.
2. Settings (right panel): Accelerator = **GPU T4 x1**, Internet = On.
3. Add-ons → Secrets → add `HF_TOKEN`.
4. Paste cells from `notebooks/03_finetune_kaggle.py` (split on `# %%`),
   or use File → Import notebook with that file.
5. Uncomment the `!pip install` line in cell 1 on first run.
6. Restart & Run All. Expect ~3h47m.
7. The final cell pushes the LoRA adapter to
   `michaelodafe/whisper-pidgin-v1` (private).

### 11.5 Local inference

```bash
HF_HUB_DISABLE_XET=1 python -u infer/01_merge_and_convert.py
python infer/03_test_on_samples.py    # smoke test on 20 train clips
python infer/02_streaming_demo.py     # live mic demo
```

macOS will prompt for mic permission on first streaming run.

---

## 12. Post-v1 work (Paths A, B, C)

Detailed log of the work attempted after v1 shipped, what landed, what
didn't, and why. Some failed-experiment scripts have been removed from
the repo for cleanliness; the prose below preserves the lessons.

### 12.1 Path A — decode-time hotwords + postprocess (LANDED)

Files: [infer/decode.py](infer/decode.py), [scripts/05_eval_pathA.py](scripts/05_eval_pathA.py).

Two zero-cost decode-time interventions:

1. **`initial_prompt` hotwords** — a single Pidgin-style sentence listing
   common Nigerian proper nouns (Buhari, Lagos, APC, etc.) and Pidgin
   function words (di, dey, na, wey, sabi). Prompt is in the same casing
   and punctuation style as the training labels (no caps, no periods)
   to avoid biasing the model toward emitting punctuation.
2. **`postprocess()`** — strips punctuation `[.,!?;:"]`, removes
   commas inside numbers (`60,000` → `60000`), merges digit groups
   separated by spaces (`60 000` → `60000`).

Why both: the first version of the prompt had English punctuation, which
made the model emit periods and commas (every one of which is an inserted
WER token). The cleaned prompt + punctuation-stripping postprocess fixes
that.

A/B/C/D evaluation on 20 random test clips (seed=7), same faster-whisper
int8 decoder for fairness:

| config | WER (raw refs) | Δ vs baseline |
|---|---|---|
| A baseline | 22.63% | — |
| B hotwords | 20.09% | −2.54 pp |
| C postprocess | 21.71% | −0.92 pp |
| **D both** | **19.17%** | **−3.46 pp** |

A 100-clip random sample (seed=42) gave 21.94% with the full pipeline,
suggesting the true gain is closer to **1–2 pp** once sample noise is
controlled. Direct comparison to v1's 21.37% (Kaggle eval) is
apples-to-oranges because that used transformers fp16 + beam=4, while
this uses faster-whisper int8 + greedy. Path A is kept in the streaming
demo since it costs nothing at inference time.

### 12.2 Path B — KenLM rescoring on `naijaweb` (FIRST ATTEMPT — STOPPED)

Pulled 270k pages of `saheedniyi/naijaweb`, normalized to 6.5M lines
(595 MB), pushed to HF as `michaelodafe/naijaweb-asr-corpus`. Then
audited the content before training.

**Result of audit:** 0.09% Pidgin-marker rate. The corpus is
overwhelmingly Standard English news/blogs from Nigerian sources, NOT
Nigerian Pidgin. Training a KenLM on it would have biased the rescorer
to prefer English over Pidgin transcriptions — exactly the opposite of
what we wanted.

**Lesson:** generic "Nigerian web content" is mostly English. Always
sample/audit corpus content *before* pulling and training. Stopped
before running KenLM training; corpus sits unused at HF.

### 12.3 Path B retry — curated Pidgin corpus (LM TRAINED, RESCORING FAILED)

Surveyed and audited 6 candidate Pidgin text sources, kept 5
high-quality ones plus our v1 training transcripts. Built a 218k-line
(27 MB) curated Pidgin corpus pushed to `michaelodafe/pidgin-lm-corpus`.

| Source | Rows kept | Marker rate |
|---|---|---|
| AnalyticsIntelligence/pidgin_corpus | 120,278 | 72.5% |
| Tommy0201/pidgin-to-english (Pidgin side) | 101,972 | 59.5% |
| HausaNLP/NaijaSenti-Twitter (pcm) | 4,693 | 70.0% |
| michsethowusu/...mt560 (pcm, 50% downsample) | 8,722 | 72.5% |
| masakhane/masakhaner (pcm, tokens rejoined) | 1,982 | — |
| v1 training transcripts | 3,362 | (target domain) |
| **after dedup** | **218,220** | |

Trained a KenLM 4-gram on Kaggle (CPU notebook), pushed to
`michaelodafe/pidgin-kenlm-4gram` (38 MB binary, 256 MB ARPA).
Perplexity sanity check confirmed the LM is correctly Pidgin-tuned:

| Sentence | PPL |
|---|---|
| `"salt di group also tok say too much salt no good"` | 15.32 |
| `"him say di economy dey very bad"` | 15.52 |
| `"femi otedola neva collect form to be lagos state govnor"` | 4.24 |
| `"the quick brown fox jumps over the lazy dog"` | 6,662 |
| `"asdfg qwerty zxcvb random gibberish noise"` | 114,008 |

So the LM works. **Rescoring did not.**

N-best rescoring eval (20 test clips, K=4 beams, λ swept over 7 values
from 0.0 to 1.5) returned **identical WER (19.68%) for every λ**.
Diagnostic showed why: with our LoRA-finetuned Whisper, beam search
returns the **identical string in all K beams** (`distinct=1/K`). The
model is so confident on Pidgin that beam search collapses. The LM has
nothing to differentiate.

**Conclusion:** vanilla n-best rescoring will not work on this
ASR+adapter. To actually use the LM would require either:
- Sampling-based n-best (`do_sample=True`) — injects randomness, weakens
  the acoustic signal, marginal improvement at best.
- Token-level shallow fusion (LM in the logits processor at every step,
  not after) — correct approach, requires custom decoder, ~2–3 days of
  work.
- Diverse beam search via `custom_generate` — broken in transformers
  4.46+, would need to wire up the new repo.

LM artifacts (corpus + KenLM) are kept on HF for if/when someone
revisits shallow fusion.

### 12.4 Path C audit — more training audio (NOT PURSUED)

Audit results (HF metadata + WebFetch of dataset cards), ordered by accessibility:

| Source | Pidgin volume | Verdict |
|---|---|---|
| `timniel/Pidgin_ASR_Dataset_Combined` (UD_Naija-NSC subset) | 5,883 clips ≈ ~12 h | ✅ accessible, distinct from v1 |
| UD_Naija-NSC raw | 9,242 sentences ≈ ~18 h | same data as timniel |
| `naijavoices/naijavoices-dataset` | 1,800 h Yoruba/Hausa/Igbo | ❌ no Pidgin (mislabelled in earlier survey) |
| `WAZOBIALABS/nigerian-pidgin-voice-text` | 216 text-only entries; audio planned Q4 2026 | ❌ no audio yet |
| African Voices (africanvoices.io) | 3,000 h across 5 langs; Pidgin not broken out | ⚠️ requires partnership contact |
| Mozilla Common Voice (pcm) | locale exists; volume not visible without JS | ⚠️ likely <5 h |
| BBC News Pidgin broadcasts | hundreds of articles | ⚠️ multi-day scrape + weak-supervision project |

**Realistic v2 with confirmed-accessible data:** ~20 h total (2.3× v1).
Expected WER 17–19% — meaningful but not transformational.

**For 14–17% WER**, would need African Voices partnership or BBC weak
supervision pipeline.

### 12.5 Final state

User chose to stop after Path C audit rather than do the modest 2×
retrain.

Shipping artifacts:
- LoRA adapter — `michaelodafe/whisper-pidgin-v1` (21.37% test WER,
  9.90% CER)
- Combined ASR dataset — `michaelodafe/pidgin-asr-combined` (~8.6 h)
- LM corpus — `michaelodafe/pidgin-lm-corpus` (218k Pidgin lines)
- KenLM 4-gram — `michaelodafe/pidgin-kenlm-4gram` (kept for future
  shallow-fusion work)
- CT2 int8 model — `infer/whisper-pidgin-ct2/` (781 MB, local)
- Streaming demo — [infer/02_streaming_demo.py](infer/02_streaming_demo.py)
  using Path A pipeline
- Dead corpus to ignore — `michaelodafe/naijaweb-asr-corpus` (English,
  not Pidgin; kept on HF as a record but should not be reused)

## 13. Future work (revised)

If you come back to this project, in ROI order:

1. **Pull `timniel/Pidgin_ASR_Dataset_Combined`** and retrain v2 on ~20 h.
   Cheapest meaningful improvement. Target 17–19% WER.
2. **Contact Data Science Nigeria for African Voices Pidgin subset.**
   Possible 100+ h of new training data. Could reach 14% WER if the data
   is usable.
3. **Token-level shallow fusion** with the existing KenLM. Switch to
   transformers + custom logits processor. ~2–3 days of work for ~1–3 pp
   WER on top of whatever model exists.
4. **BBC News Pidgin weak-supervision pipeline**: scrape BBC Pidgin
   audio articles, transcribe with the latest model, human-correct a
   sample, retrain.
5. **Higher LoRA rank or partial full fine-tune** once data scales past
   ~30 h.
6. **Public release.** Flip `private=True` to `False` on the model and
   datasets, write a model card, add a HuggingFace Spaces demo.
7. **True streaming.** For sub-utterance latency, port to a streaming
   decoder (Whisper-Streaming, Lightning Whisper, or NeMo Cache-Aware).
8. **On-device.** Distill to Whisper-small or Moonshine-equivalent for
   mobile/edge deployment.
