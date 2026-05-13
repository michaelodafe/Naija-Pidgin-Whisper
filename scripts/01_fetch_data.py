"""
Fetch Nigerian Pidgin ASR datasets, normalize to a unified schema, dedupe,
and push the combined dataset to the user's HF account.

Unified schema:
  - audio: Audio (16kHz)
  - text: str         (transcription)
  - source: str       (which source dataset)
  - duration: float   (seconds)
  - speaker_id: str   ("" if unknown)
"""
import hashlib
import os
from pathlib import Path

from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

HF_TOKEN = os.environ["HF_TOKEN"]
COMBINED_REPO = os.environ["COMBINED_DATASET_REPO"]

login(token=HF_TOKEN)

TARGET_SR = 16000


def normalize_pidgin_v1(ds):
    # asr-nigerian-pidgin/nigerian-pidgin-1.0: sentence, filename, audio
    def fix(row):
        audio = row["audio"]
        return {
            "audio": audio,
            "text": row["sentence"].strip(),
            "source": "asr-nigerian-pidgin-1.0",
            "duration": len(audio["array"]) / audio["sampling_rate"],
            "speaker_id": row["filename"].split("_")[0] if "_" in row["filename"] else "",
        }

    keep = ["audio", "text", "source", "duration", "speaker_id"]
    return ds.map(fix, remove_columns=[c for c in ds.column_names if c not in keep])


def normalize_rexe(ds):
    # Rexe/nigerian-pidgin-speech: audio, text, duration, source, video_title, transcription_source
    def fix(row):
        return {
            "audio": row["audio"],
            "text": row["text"].strip(),
            "source": f"rexe-youtube-{row.get('video_title', 'unknown')[:30]}",
            "duration": float(row["duration"]),
            "speaker_id": "",
        }

    keep = ["audio", "text", "source", "duration", "speaker_id"]
    return ds.map(fix, remove_columns=[c for c in ds.column_names if c not in keep])


def audio_hash(row):
    arr = row["audio"]["array"]
    h = hashlib.md5(arr.tobytes()).hexdigest()
    return {"_hash": h}


def dedupe(ds):
    ds = ds.map(audio_hash)
    seen = set()
    keep_idx = []
    for i, h in enumerate(ds["_hash"]):
        if h not in seen:
            seen.add(h)
            keep_idx.append(i)
    ds = ds.select(keep_idx)
    return ds.remove_columns(["_hash"])


def main():
    print("Loading asr-nigerian-pidgin/nigerian-pidgin-1.0 ...")
    p1 = load_dataset("asr-nigerian-pidgin/nigerian-pidgin-1.0")
    p1 = DatasetDict({k: p1[k].cast_column("audio", Audio(sampling_rate=TARGET_SR)) for k in p1})
    p1 = DatasetDict({k: normalize_pidgin_v1(v) for k, v in p1.items()})

    print("Loading Rexe/nigerian-pidgin-speech ...")
    rx = load_dataset("Rexe/nigerian-pidgin-speech")
    rx = DatasetDict({k: rx[k].cast_column("audio", Audio(sampling_rate=TARGET_SR)) for k in rx})
    rx = DatasetDict({k: normalize_rexe(v) for k, v in rx.items()})

    # Rexe is tiny (73 samples, single source) — route entirely to test/eval pool.
    combined = DatasetDict(
        {
            "train": p1["train"],
            "validation": p1["validation"],
            "test": concatenate_datasets([p1["test"], rx["train"], rx["test"]]),
        }
    )

    print("Deduplicating by audio hash ...")
    combined = DatasetDict({k: dedupe(v) for k, v in combined.items()})

    for split, ds in combined.items():
        total_hours = sum(ds["duration"]) / 3600
        print(f"  {split}: {len(ds)} clips, {total_hours:.2f} hours")

    print(f"\nPushing to hf://{COMBINED_REPO} ...")
    combined.push_to_hub(COMBINED_REPO, private=True)
    print("Done.")


if __name__ == "__main__":
    main()
