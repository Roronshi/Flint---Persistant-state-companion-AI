#!/usr/bin/env python3
"""
Download official RWKV‑7 G1 models from Hugging Face.

This script uses the huggingface_hub API to fetch all available model
weights from the repository ``BlinkDL/rwkv7-g1``.  By default the files
are downloaded into the ``models`` directory within the Flint project.
Only files ending in ``.pth`` or ``.onnx`` are downloaded.  Existing files
will be skipped unless the ``--force`` flag is provided.

Run this script before starting Flint to populate the models directory.

Example::

    python3 scripts/download_models.py

Requires the ``huggingface_hub`` library.  Install with::

    pip install huggingface_hub

If you are downloading large models you may want to set the
``HUGGINGFACE_HUB_TOKEN`` environment variable to your personal access
token for authenticated downloads.
"""
import argparse
import os
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except ImportError as exc:
    raise SystemExit(
        "huggingface_hub is required. Install with 'pip install huggingface_hub'."
    ) from exc


REPO_ID = "BlinkDL/rwkv7-g1"
DEFAULT_PATTERN = (".pth", ".onnx")


def download_all_models(
    repo: str = REPO_ID,
    dest_dir: str = "models",
    pattern: tuple[str, ...] = DEFAULT_PATTERN,
    force: bool = False,
    size: str | None = None,
) -> None:
    """Download model files matching ``pattern`` (and optionally ``size``) from ``repo``.

    Parameters
    ----------
    repo: str
        The Hugging Face repository to download from.
    dest_dir: str
        Destination directory where files will be saved.
    pattern: tuple[str, ...]
        File extensions to download (default: ``(".pth", ".onnx")``).
    force: bool
        If True, download and overwrite existing files.
    size: str | None
        If given, only download files whose filename contains this substring
        (case-insensitive).  E.g. ``"1.5b"`` downloads only the 1.5B model.
        Without this argument every matching file in the repo is downloaded.
    """
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    print(f"Listing files in repository {repo}…")
    files = list_repo_files(repo)
    model_files = [f for f in files if f.lower().endswith(pattern)]
    if size:
        model_files = [f for f in model_files if size.lower() in Path(f).name.lower()]
    if not model_files:
        msg = f"No model files found matching pattern"
        if size:
            msg += f" and size '{size}'"
        print(msg + ".")
        return
    print(f"Found {len(model_files)} model file(s):")
    for f in model_files:
        print(f"  {f}")
    for f in model_files:
        dest_file = dest_path / Path(f).name
        if dest_file.exists() and not force:
            print(f"Skipping existing file {dest_file}")
            continue
        print(f"Downloading {f}…")
        try:
            hf_hub_download(repo_id=repo, filename=f, cache_dir=str(dest_path), local_dir=str(dest_path), force_download=force)
        except Exception as exc:
            print(f"Failed to download {f}: {exc}")
        else:
            print(f"Saved {dest_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download RWKV-7 G1 model weights from Hugging Face")
    parser.add_argument("--repo", default=REPO_ID, help="Hugging Face repo id (default: BlinkDL/rwkv7-g1)")
    parser.add_argument("--dest", default="models", help="Destination directory for model files (default: models)")
    parser.add_argument(
        "--pattern",
        nargs="*",
        default=DEFAULT_PATTERN,
        help="File extensions to download (e.g. .pth .onnx).  Defaults to both .pth and .onnx",
    )
    parser.add_argument(
        "--size",
        default=None,
        help="Only download files whose name contains this string (e.g. '1.5b', '7.2b').  "
             "Omit to download all matching files.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()
    download_all_models(repo=args.repo, dest_dir=args.dest, pattern=tuple(args.pattern), force=args.force, size=args.size)


if __name__ == "__main__":
    main()