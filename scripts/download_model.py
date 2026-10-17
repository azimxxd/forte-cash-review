"""Explicit preparation step: downloads public weights only, never reads user videos.
Run on the deployment host before copying private data; offline inference is separate.
"""

import argparse
from pathlib import Path

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--destination", type=Path, default=Path("models/Qwen3-VL-4B-Instruct")
    )
    p.add_argument(
        "--revision",
        required=True,
        help="Reviewed Hugging Face commit SHA for reproducible weights",
    )
    a = p.parse_args()
    if len(a.revision) != 40 or any(c not in "0123456789abcdef" for c in a.revision):
        p.error("--revision must be a full lowercase 40-character commit SHA")
    from huggingface_hub import snapshot_download

    snapshot_download(
        "Qwen/Qwen3-VL-4B-Instruct", revision=a.revision, local_dir=a.destination
    )
    (a.destination / "DEPLOYED_REVISION").write_text(a.revision + "\n")
