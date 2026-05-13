"""
Build the HF model repo that backs the Inference Endpoint:
  - Uploads CT2 model files (the merged adapter + base, int8_float16).
  - Uploads handler.py + requirements.txt at the repo root.

Run once. After this, deploy via https://ui.endpoints.huggingface.co/ —
see infer/endpoint/DEPLOY.md (or the conversation that produced this).
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, login

load_dotenv()
login(token=os.environ["HF_TOKEN"])

REPO = "michaelodafe/whisper-pidgin-endpoint"
ROOT = Path(__file__).parent.parent
CT2 = ROOT / "infer" / "whisper-pidgin-ct2"
ENDPOINT_DIR = ROOT / "infer" / "endpoint"

assert CT2.exists(), f"Missing {CT2}. Run infer/01_merge_and_convert.py first."
assert (ENDPOINT_DIR / "handler.py").exists()
assert (ENDPOINT_DIR / "requirements.txt").exists()

api = HfApi()
api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)

print("Uploading handler.py + requirements.txt ...")
for f in ["handler.py", "requirements.txt"]:
    api.upload_file(
        path_or_fileobj=str(ENDPOINT_DIR / f),
        path_in_repo=f,
        repo_id=REPO,
        repo_type="model",
    )

print(f"Uploading CT2 model files (~781 MB) under ct2/ ...")
api.upload_folder(
    folder_path=str(CT2),
    path_in_repo="ct2",
    repo_id=REPO,
    repo_type="model",
)

print(f"\nDone. Repo: hf://{REPO}")
print("Next: deploy at https://ui.endpoints.huggingface.co/new")
