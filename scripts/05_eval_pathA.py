"""
A/B/C/D evaluation of Path A interventions on a held-out test sample.

Configurations:
  A) baseline:        no hotwords, no postprocess (matches v1 behavior)
  B) +hotwords only
  C) +postprocess only
  D) hotwords + postprocess  (final v2 pipeline)

Uses the same 20-clip seed as scripts/04_inspect_test_predictions.py
so numbers are directly comparable.
"""
import os
import sys
from pathlib import Path

import jiwer
from datasets import load_dataset
from dotenv import load_dotenv
from faster_whisper import WhisperModel

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from infer.decode import transcribe, postprocess  # noqa: E402

load_dotenv()
COMBINED = os.environ["COMBINED_DATASET_REPO"]
CT2 = ROOT / "infer" / "whisper-pidgin-ct2"
REPORT = ROOT / "data" / "test_samples" / "pathA_eval.md"
N = 20

print("Loading model...")
asr = WhisperModel(str(CT2), device="cpu", compute_type="int8")

print("Loading test split...")
test = load_dataset(COMBINED, split="test").shuffle(seed=7).select(range(N))

refs = [ex["text"].strip() for ex in test]
results = {"A_baseline": [], "B_hotwords": [], "C_postprocess": [], "D_both": []}

for i, ex in enumerate(test):
    audio = ex["audio"]["array"]
    results["A_baseline"].append(transcribe(asr, audio, use_hotwords=False, use_postprocess=False))
    results["B_hotwords"].append(transcribe(asr, audio, use_hotwords=True,  use_postprocess=False))
    results["C_postprocess"].append(transcribe(asr, audio, use_hotwords=False, use_postprocess=True))
    results["D_both"].append(transcribe(asr, audio, use_hotwords=True,  use_postprocess=True))
    print(f"[{i:02d}/{N}] done")

# Also fair to apply postprocess to refs since their format is inconsistent
refs_norm = [postprocess(r) for r in refs]

print("\n=== WER ===")
print(f"{'config':<15} {'raw refs':<12} {'norm refs':<12}")
for k, hyps in results.items():
    raw = jiwer.wer(refs, hyps)
    norm = jiwer.wer(refs_norm, hyps)
    print(f"{k:<15} {raw:<12.4f} {norm:<12.4f}")

with REPORT.open("w") as f:
    f.write(f"# Path A evaluation\n\n")
    f.write(f"Same {N} test clips as scripts/04. seed=7.\n\n")
    f.write(f"| config | WER (raw refs) | WER (postprocessed refs) |\n")
    f.write(f"|---|---|---|\n")
    for k, hyps in results.items():
        raw = jiwer.wer(refs, hyps)
        norm = jiwer.wer(refs_norm, hyps)
        f.write(f"| {k} | {raw:.4f} | {norm:.4f} |\n")
    f.write(f"\n## Per-clip diff (config A vs config D)\n\n")
    for i in range(N):
        a = results["A_baseline"][i]
        d = results["D_both"][i]
        if a != d:
            wa = jiwer.wer(refs[i], a)
            wd = jiwer.wer(refs[i], d)
            arrow = "↓" if wd < wa else ("↑" if wd > wa else "=")
            f.write(f"### clip {i:02d}  (WER {wa:.2f} → {wd:.2f} {arrow})\n\n")
            f.write(f"- **REF:** {refs[i]}\n")
            f.write(f"- **A:**   {a}\n")
            f.write(f"- **D:**   {d}\n\n")

print(f"\nReport: {REPORT}")
