"""Token corpora: explicit download, streaming `uint16` preparation, and verification.

`prepare` is the only path that downloads. It writes one flat token file per split plus a
manifest; training memory-maps those files (`token_dataset.py`), so memory does not grow with
corpus size and nothing is re-tokenized on start or resume.

Document policy: split the raw text on `<|endoftext|>`, strip surrounding whitespace, drop
empty documents, and prepend the end-of-text token to every document (as build-nanogpt's
`fineweb.py` does). Windows are cut from the concatenated stream, so attention crosses
document boundaries, separated by that token.
"""

import hashlib
import json
import urllib.request
from itertools import islice
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
SEPARATOR = "<|endoftext|>"
SPLITS = ("train", "val")

CORPORA = {
    "tinystories": {
        "description": "TinyStories (Eldan and Li, 2023, arXiv:2305.07759), original release",
        "license": "CDLA-Sharing-1.0",
        "repository": "https://huggingface.co/datasets/roneneldan/TinyStories",
        "revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
        # split: (file, sha256 = the Hub's LFS object id, bytes)
        "files": {
            "train": (
                "TinyStories-train.txt",
                "c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f",
                1924281556,
            ),
            "val": (
                "TinyStories-valid.txt",
                "94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4",
                19447282,
            ),
        },
    },
}


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def download_corpus(name, raw_dir: Path):
    """Fetch the pinned raw files, verifying size and SHA-256; reuse verified local copies."""
    spec = CORPORA[name]
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, (filename, expected, size) in spec["files"].items():
        path = raw_dir / filename
        paths[split] = path
        if path.is_file() and path.stat().st_size == size and file_sha256(path) == expected:
            continue
        url = f"{spec['repository']}/resolve/{spec['revision']}/{filename}"
        temporary = path.with_suffix(path.suffix + ".part")
        digest, received = hashlib.sha256(), 0
        print(f"Downloading {url} ({size / 1e6:.0f} MB)", flush=True)
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as out:
            while chunk := response.read(1 << 22):
                digest.update(chunk)
                out.write(chunk)
                received += len(chunk)
        if received != size or digest.hexdigest() != expected:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"Downloaded {filename} does not match its pinned size/SHA-256")
        temporary.replace(path)
    return paths


def iter_documents(path: Path, chunk_chars=1 << 22):
    """Yield documents separated by `<|endoftext|>`, streaming the file in chunks."""
    buffer = ""
    with path.open(encoding="utf-8") as stream:
        while chunk := stream.read(chunk_chars):
            # Keep the unterminated tail: a separator may straddle two chunks.
            *complete, buffer = (buffer + chunk).split(SEPARATOR)
            for document in complete:
                if document := document.strip():
                    yield document
    if buffer := buffer.strip():
        yield buffer


def build_token_corpus(
    sources, out_dir: Path, tokenizer, tokenizer_manifest, provenance, batch_documents=4096
):
    """Tokenize each split's text file into `<split>.bin` (`uint16`) and write the manifest.

    The manifest is written last, so its presence marks a complete corpus.
    """
    eot = tokenizer.eot_token
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").unlink(missing_ok=True)
    splits = {}
    for split in SPLITS:
        target = out_dir / f"{split}.bin"
        temporary = target.with_suffix(".bin.tmp")
        digest, tokens, documents = hashlib.sha256(), 0, 0
        stream = iter_documents(Path(sources[split]))
        with temporary.open("wb") as out:
            while batch := list(islice(stream, batch_documents)):
                encoded = tokenizer.encode_ordinary_batch(batch)
                count = sum(len(ids) for ids in encoded) + len(encoded)
                array = np.fromiter(
                    (t for ids in encoded for t in (eot, *ids)), dtype=np.int64, count=count
                )
                if array.min() < 0 or array.max() > np.iinfo(np.uint16).max:
                    raise ValueError("Token ids do not fit in uint16")
                data = array.astype(np.uint16).tobytes()
                out.write(data)
                digest.update(data)
                tokens += count
                documents += len(batch)
        if not documents:
            temporary.unlink()
            raise ValueError(f"No documents in the {split} source")
        temporary.replace(target)
        splits[split] = {
            "file": target.name,
            "sha256": digest.hexdigest(),
            "tokens": tokens,
            "documents": documents,
            "source_sha256": file_sha256(Path(sources[split])),
        }
        print(f"Prepared {split}: {documents} documents, {tokens} tokens", flush=True)
    manifest = {
        "format_version": FORMAT_VERSION,
        "dtype": "uint16",
        "document_policy": "split on <|endoftext|>; strip whitespace; drop empty documents",
        "eot_policy": f"prepended: every document starts with token {eot}",
        "cross_document_attention": True,
        "tokenizer_manifest": tokenizer_manifest,
        "provenance": provenance,
        "splits": splits,
    }
    temporary = out_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(out_dir / "manifest.json")
    return manifest


def prepare_corpus(name, out_dir: Path, tokenizer_cache: Path):
    from ..model.gpt2.tokenizer import load_tokenizer

    manifest_path = out_dir / "manifest.json"
    tokenizer_manifest = json.loads((tokenizer_cache / "manifest.json").read_text())
    if manifest_path.is_file():
        try:
            load_token_corpus(out_dir, name, tokenizer_manifest)
            print(f"Token corpus already prepared: {out_dir}")
            return
        except ValueError as exc:
            print(f"Rebuilding token corpus ({exc})", flush=True)
    sources = download_corpus(name, out_dir / "raw")
    spec = CORPORA[name]
    provenance = {
        "corpus": name,
        "description": spec["description"],
        "license": spec["license"],
        "repository": spec["repository"],
        "revision": spec["revision"],
        "files": {split: spec["files"][split][0] for split in SPLITS},
    }
    build_token_corpus(
        sources, out_dir, load_tokenizer(tokenizer_cache), tokenizer_manifest, provenance
    )


def load_token_corpus(path: Path, name, tokenizer_manifest):
    """Verify a prepared corpus; return its manifest and the manifest's SHA-256.

    The manifest records every token file's hash, so its own hash identifies the data.
    """
    try:
        raw = (path / "manifest.json").read_bytes()
        manifest = json.loads(raw)
        if manifest["format_version"] != FORMAT_VERSION or manifest["dtype"] != "uint16":
            raise ValueError("Unsupported token corpus format")
        if manifest["provenance"]["corpus"] != name:
            raise ValueError(f"Token corpus is {manifest['provenance']['corpus']!r}, not {name!r}")
        if manifest["tokenizer_manifest"] != tokenizer_manifest:
            raise ValueError("Token corpus was built with a different tokenizer")
        for split in SPLITS:
            entry = manifest["splits"][split]
            file = path / entry["file"]
            if file.stat().st_size != 2 * entry["tokens"] or file_sha256(file) != entry["sha256"]:
                raise ValueError(f"Token file {file.name} does not match its manifest")
    except (OSError, KeyError, TypeError) as exc:
        raise ValueError(
            f"Token corpus missing/corrupt at {path}; run `llm-training prepare` first"
        ) from exc
    return manifest, hashlib.sha256(raw).hexdigest()
