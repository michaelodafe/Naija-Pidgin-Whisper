"""
LoRA fine-tune of Whisper-large-v3-turbo on Nigerian Pidgin English.
Designed to run on Kaggle (single T4 GPU, ~16GB VRAM, 9-hr session limit).

Upload this file as a Kaggle notebook (or paste cell-by-cell).
Cells are delimited by '# %%' for compatibility with Jupytext / VS Code.

Before running on Kaggle:
  1. Add HF_TOKEN to Kaggle "Add-ons → Secrets" with label `HF_TOKEN`.
  2. Enable GPU T4 x1 in the notebook settings.
  3. Enable Internet.
"""

# %% [markdown]
# ## 1. Install + auth

# %%
# !pip install -q "transformers>=4.45" "datasets>=2.20,<4.0" "peft>=0.13" \
#   "accelerate>=0.34" "evaluate>=0.4" "jiwer>=3.0" "librosa>=0.10.2" "soundfile>=0.12"

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # single GPU; DataParallel + device_map=auto OOMs

import torch
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
login(token=HF_TOKEN)
os.environ["HF_HUB_DISABLE_XET"] = "1"

DATASET_REPO = "michaelodafe/pidgin-asr-combined"
BASE_MODEL = "openai/whisper-large-v3-turbo"
OUTPUT_REPO = "michaelodafe/whisper-pidgin-v1"
LANG_TOKEN = "english"   # closest Whisper language to Nigerian Pidgin
TASK = "transcribe"

assert torch.cuda.is_available(), "Enable GPU T4 in Kaggle notebook settings."
print("GPU:", torch.cuda.get_device_name(0))

# %% [markdown]
# ## 2. Load + filter dataset

# %%
from datasets import Audio, load_dataset

ds = load_dataset(DATASET_REPO)
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

MAX_DUR = 30.0   # Whisper's input window
MIN_TEXT_LEN = 3

def keep(row):
    return MIN_TEXT_LEN <= len(row["text"]) and row["duration"] <= MAX_DUR

ds = ds.filter(keep)
for split in ds:
    print(f"{split}: {len(ds[split])} clips ({sum(ds[split]['duration'])/3600:.2f} h)")

# %% [markdown]
# ## 3. Processor + feature extraction

# %%
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained(BASE_MODEL, language=LANG_TOKEN, task=TASK)

def prepare(row):
    audio = row["audio"]
    feats = processor.feature_extractor(audio["array"], sampling_rate=16000).input_features[0]
    labels = processor.tokenizer(row["text"]).input_ids
    return {"input_features": feats, "labels": labels}

ds = ds.map(prepare, remove_columns=ds["train"].column_names, num_proc=2)

# %% [markdown]
# ## 4. Data collator

# %%
from dataclasses import dataclass
from typing import Any
import torch

@dataclass
class WhisperCollator:
    processor: Any

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # Whisper prepends decoder_start_token_id at training time; trim if already present
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

collator = WhisperCollator(processor=processor)

# %% [markdown]
# ## 5. Load base model + apply LoRA

# %%
from transformers import WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model = WhisperForConditionalGeneration.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16
).to("cuda")
model.generation_config.forced_decoder_ids = None
model.generation_config.suppress_tokens = []
model.generation_config.language = LANG_TOKEN
model.generation_config.task = TASK
model = prepare_model_for_kbit_training(model)

lora = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

# %% [markdown]
# ## 6. Metrics — WER + CER

# %%
import evaluate
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer, "cer": cer}

# %% [markdown]
# ## 7. Training

# %%
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="/kaggle/working/whisper-pidgin",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,        # effective batch 16
    per_device_eval_batch_size=4,
    learning_rate=1e-4,
    warmup_ratio=0.05,
    num_train_epochs=5,
    fp16=True,
    eval_strategy="steps",
    eval_steps=200,
    save_steps=200,
    save_total_limit=2,
    logging_steps=25,
    report_to="none",
    predict_with_generate=True,
    generation_max_length=225,
    metric_for_best_model="wer",
    greater_is_better=False,
    load_best_model_at_end=True,
    remove_unused_columns=False,          # PEFT needs this off
    label_names=["labels"],
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    data_collator=collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)
trainer.train()

# %% [markdown]
# ## 8. Final eval on test split + push

# %%
test_metrics = trainer.evaluate(ds["test"])
print("TEST:", test_metrics)

model.push_to_hub(OUTPUT_REPO, private=True)
processor.push_to_hub(OUTPUT_REPO, private=True)
print(f"Pushed adapter to hf://{OUTPUT_REPO}")
