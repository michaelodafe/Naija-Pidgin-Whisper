"""
Smoke test: transcribe data/samples/*.wav with the merged Pidgin model
and compare to the reference transcripts.
"""
from pathlib import Path

import jiwer
from faster_whisper import WhisperModel

ROOT = Path(__file__).parent.parent
CT2 = Path(__file__).parent / "whisper-pidgin-ct2"
SAMPLES = ROOT / "data" / "samples"
REF_FILE = SAMPLES / "transcripts.txt"

print("Loading model...")
model = WhisperModel(str(CT2), device="cpu", compute_type="int8")

refs, hyps = [], []
for line in REF_FILE.read_text().strip().splitlines():
    fname, ref = line.split("\t", 1)
    segments, _ = model.transcribe(
        str(SAMPLES / fname), language="en", task="transcribe", beam_size=1
    )
    hyp = "".join(s.text for s in segments).strip()
    refs.append(ref)
    hyps.append(hyp)
    print(f"\n[{fname}]")
    print(f"  REF: {ref}")
    print(f"  HYP: {hyp}")

print(f"\n--- WER on 20 samples: {jiwer.wer(refs, hyps):.4f} ---")
