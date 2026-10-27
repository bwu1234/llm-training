"""Pinned model files: explicit download with SHA-256 checks, and verification before loading.

Downloading is the only network step; every loader calls `verify_pinned` first, so a training
or inference run never fetches anything.
"""

import hashlib
import urllib.request
from pathlib import Path


def sha256_file(path):
    with open(path, "rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def download_pinned(base_url, files, path):
    """Fetch each `files` entry ({name: sha256}) missing or mismatched in `path`."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for name, expected in files.items():
        target = path / name
        if target.is_file() and sha256_file(target) == expected:
            continue
        partial = target.with_name(target.name + ".partial")
        digest = hashlib.sha256()
        print(f"Downloading {base_url}/{name}", flush=True)
        # Stream in 1 MiB chunks, hashing as we write, so a multi-GB file never sits in memory.
        with urllib.request.urlopen(f"{base_url}/{name}", timeout=60) as response:
            with partial.open("wb") as out:
                while chunk := response.read(1 << 20):
                    digest.update(chunk)
                    out.write(chunk)
        if digest.hexdigest() != expected:
            partial.unlink()
            raise ValueError(f"Downloaded {name} SHA256 mismatch")
        partial.replace(target)


def verify_pinned(files, path, prepare_hint):
    for name, expected in files.items():
        target = Path(path) / name
        if not target.is_file() or sha256_file(target) != expected:
            raise ValueError(
                f"{target} is missing or differs from the pinned file; run {prepare_hint}"
            )
