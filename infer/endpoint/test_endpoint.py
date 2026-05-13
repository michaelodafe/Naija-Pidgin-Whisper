"""
Smoke-test a deployed HF Inference Endpoint.

Usage:
  python infer/endpoint/test_endpoint.py <endpoint_url> [audio_file.wav]

If no audio file is given, uses data/samples/sample_00.wav.
"""
import base64
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

if len(sys.argv) < 2:
    print("Usage: test_endpoint.py <endpoint_url> [audio_file]")
    sys.exit(1)

ENDPOINT = sys.argv[1].rstrip("/")
ROOT = Path(__file__).parent.parent.parent
audio_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else ROOT / "data" / "samples" / "sample_00.wav"

assert audio_path.exists(), f"Audio file not found: {audio_path}"
TOKEN = os.environ["HF_TOKEN"]

with audio_path.open("rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

t0 = time.time()
r = requests.post(
    ENDPOINT,
    headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
    json={"inputs": audio_b64},
    timeout=180,
)
elapsed = time.time() - t0
print(f"Status:  {r.status_code}")
print(f"Latency: {elapsed:.2f}s")
print(f"Body:    {r.text[:500]}")
