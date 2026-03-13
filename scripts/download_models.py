#!/usr/bin/env python3
"""
Download official RWKV-7 G1 models from Hugging Face.

Downloads only the LATEST file for the chosen size (sorted by filename,
which encodes the release date as YYYYMMDD).  Uses huggingface-cli under
the hood so downloads are resumable — if interrupted, re-running the
script continues where it left off.

Example::

    python3 scripts/download_models.py --size 2.9b

Set HF_TOKEN in the environment (or pass --token) for faster, authenticated
downloads:

    HF_TOKEN=hf_xxx python3 scripts/download_models.py --size 2.9b

Requires the ``huggingface_hub`` library::

    pip install huggingface_hub
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    from huggingface_hub import list_repo_files
except ImportError as exc:
    raise SystemExit(
        "huggingface_hub is required. Install with 'pip install huggingface_hub'."
    ) from exc


REPO_ID = "BlinkDL/rwkv7-g1"
DEFAULT_PATTERN = (".pth", ".onnx")


def pick_latest(files: list) -> str:
    """Return the lexicographically last filename — filenames encode YYYYMMDD."""
    return sorted(files)[-1]


def download_model(
    repo=REPO_ID,
    dest_dir="models",
    pattern=DEFAULT_PATTERN,
    size=None,
    token=None,
    latest_only=True,
):
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    hf_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    print(f"Listing files in repository {repo}…")
    list_kwargs = {}
    if hf_token:
        list_kwargs["token"] = hf_token
    all_files = list(list_repo_files(repo, **list_kwargs))

    candidates = [
        f for f in all_files
        if Path(f).suffix.lower() in pattern
        and (not size or size.lower() in Path(f).name.lower())
    ]

    if not candidates:
        msg = "No model files found"
        if size:
            msg += f" matching size '{size}'"
        print(msg + ".")
        sys.exit(1)

    to_download = [pick_latest(candidates)] if latest_only else candidates

    print(f"{'Latest file' if latest_only else 'Files'} to download:")
    for f in to_download:
        print(f"  {f}")

    if not hf_token:
        print()
        print("  Tip: set HF_TOKEN for faster authenticated downloads.")
        print("  huggingface.co → Settings → Access Tokens → New token (Read)")
        print()

    for filename in to_download:
        dest_file = dest_path / Path(filename).name
        if dest_file.exists():
            print(f"Already exists, skipping: {dest_file.name}")
            continue

        print(f"Downloading {filename}…")
        cmd = [
            sys.executable, "-m", "huggingface_hub.commands.huggingface_cli",
            "download",
            repo,
            filename,
            "--local-dir", str(dest_path),
            "--local-dir-use-symlinks", "False",
        ]
        if hf_token:
            cmd += ["--token", hf_token]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print("CLI download failed, falling back to hf_hub_download…")
            from huggingface_hub import hf_hub_download
            kwargs = dict(repo_id=repo, filename=filename, local_dir=str(dest_path))
            if hf_token:
                kwargs["token"] = hf_token
            hf_hub_download(**kwargs)

        if dest_file.exists():
            print(f"Saved: {dest_file.name}")
        else:
            print(f"Warning: expected file not found at {dest_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Download RWKV-7 G1 model weights from Hugging Face"
    )
    parser.add_argument("--repo", default=REPO_ID)
    parser.add_argument("--dest", default="models")
    parser.add_argument("--pattern", nargs="*", default=list(DEFAULT_PATTERN))
    parser.add_argument("--size", default=None,
        help="Size filter e.g. '2.9b'. Downloads latest matching file only.")
    parser.add_argument("--token", default=None,
        help="HuggingFace access token (or set HF_TOKEN env var)")
    parser.add_argument("--all", dest="all_versions", action="store_true",
        help="Download all versions, not just the latest")
    args = parser.parse_args()

    download_model(
        repo=args.repo,
        dest_dir=args.dest,
        pattern=tuple(args.pattern),
        size=args.size,
        token=args.token,
        latest_only=not args.all_versions,
    )


if __name__ == "__main__":
    main()
