"""
Inspect the combined Pidgin ASR dataset: print stats and dump 20 sample
clips + transcripts to data/samples/ for manual listening.
"""
import os
import random
from pathlib import Path

import soundfile as sf
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

COMBINED_REPO = os.environ["COMBINED_DATASET_REPO"]
SAMPLES_DIR = Path(__file__).parent.parent / "data" / "samples"
N_SAMPLES = 20


def main():
    ds = load_dataset(COMBINED_REPO)

    for split in ds:
        d = ds[split]
        durations = d["duration"]
        texts = d["text"]
        sources = d["source"]
        print(f"\n[{split}] {len(d)} clips | {sum(durations) / 3600:.2f} h")
        print(f"  duration: min={min(durations):.1f}s  max={max(durations):.1f}s  "
              f"mean={sum(durations) / len(durations):.1f}s")
        print(f"  text len:  min={min(len(t) for t in texts)}  max={max(len(t) for t in texts)}")
        src_counts = {}
        for s in sources:
            src_counts[s] = src_counts.get(s, 0) + 1
        for s, c in sorted(src_counts.items(), key=lambda x: -x[1]):
            print(f"    source: {s}: {c}")

    # Dump samples from train for listening
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    train = ds["train"].shuffle(seed=42).select(range(N_SAMPLES))
    transcripts_path = SAMPLES_DIR / "transcripts.txt"
    with transcripts_path.open("w") as f:
        for i, row in enumerate(train):
            wav_path = SAMPLES_DIR / f"sample_{i:02d}.wav"
            sf.write(wav_path, row["audio"]["array"], row["audio"]["sampling_rate"])
            f.write(f"sample_{i:02d}.wav\t{row['text']}\n")
    print(f"\nWrote {N_SAMPLES} samples to {SAMPLES_DIR}")


if __name__ == "__main__":
    main()
