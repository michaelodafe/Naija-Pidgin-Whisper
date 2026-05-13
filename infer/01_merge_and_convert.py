"""
One-time setup for fast inference:
  1. Pull base Whisper + your LoRA adapter from HF
  2. Merge LoRA into base weights
  3. Convert to CTranslate2 (int8_float16) for use with faster-whisper

Run once. Produces ./infer/whisper-pidgin-ct2/.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

load_dotenv()
login(token=os.environ["HF_TOKEN"])

BASE = os.environ.get("BASE_MODEL", "openai/whisper-large-v3-turbo")
ADAPTER = os.environ.get("MODEL_OUTPUT_REPO", "michaelodafe/whisper-pidgin-v1")
HERE = Path(__file__).parent
MERGED = HERE / "_merged"
CT2 = HERE / "whisper-pidgin-ct2"

print(f"[1/3] Loading base model: {BASE}")
base = WhisperForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.float32)

print(f"[2/3] Applying adapter: {ADAPTER}")
peft_model = PeftModel.from_pretrained(base, ADAPTER)
merged = peft_model.merge_and_unload()
merged.save_pretrained(MERGED, safe_serialization=True)
WhisperProcessor.from_pretrained(BASE).save_pretrained(MERGED)

print(f"[3/3] Converting to CTranslate2 → {CT2}")
if CT2.exists():
    shutil.rmtree(CT2)
subprocess.run(
    [
        sys.executable, "-m", "ctranslate2.converters.transformers",
        "--model", str(MERGED),
        "--output_dir", str(CT2),
        "--copy_files", "tokenizer.json", "preprocessor_config.json",
                        "generation_config.json", "special_tokens_map.json",
                        "tokenizer_config.json",
        "--quantization", "int8_float16",
    ],
    check=True,
)
shutil.rmtree(MERGED)
print(f"Done. Model ready at: {CT2}")
