"""Explicit GPT-2 asset preparation; normal training never downloads assets."""

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def _cache_dir(path: Path | str):
    previous = os.environ.get("TIKTOKEN_CACHE_DIR")
    os.environ["TIKTOKEN_CACHE_DIR"] = str(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TIKTOKEN_CACHE_DIR", None)
        else:
            os.environ["TIKTOKEN_CACHE_DIR"] = previous


def prepare_tokenizer(path: Path):
    # Preparation is the only code path allowed to acquire tokenizer assets.
    import tiktoken
    from tiktoken_ext.openai_public import gpt2

    path.mkdir(parents=True, exist_ok=True)
    with _cache_dir(path):
        # Construct directly: get_encoding caches globally and could skip
        # populating a second preparation directory in the same process.
        tiktoken.Encoding(**gpt2())
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in path.iterdir()
        if p.is_file() and p.name != "manifest.json"
    }
    if len(files) != 2:
        raise ValueError("Expected the two GPT-2 tokenizer assets in a dedicated cache directory")
    manifest = {"encoding": "gpt2", "files": files}
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_tokenizer(path: Path):
    import tiktoken

    try:
        manifest = json.loads((path / "manifest.json").read_text())
        expected_names = {
            hashlib.sha1(url.encode()).hexdigest()
            for url in (public_vocab_url(), public_encoder_url())
        }
        if manifest["encoding"] != "gpt2" or set(manifest["files"]) != expected_names:
            raise ValueError("Invalid tokenizer manifest")
        for name, expected in manifest["files"].items():
            if (
                Path(name).name != name
                or hashlib.sha256((path / name).read_bytes()).hexdigest() != expected
            ):
                raise ValueError("Tokenizer asset hash mismatch")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Tokenizer assets missing/corrupt; run `llm-training prepare` first"
        ) from exc
    # Avoid the network-capable get_encoding cache loader altogether. Construct
    # from the verified local assets through the same tiktoken BPE parsers.
    import tiktoken_ext.openai_public as public
    from tiktoken.load import data_gym_to_mergeable_bpe_ranks

    # Empty cache disables tiktoken's secondary cache for these local paths.
    with _cache_dir(""):
        ranks = data_gym_to_mergeable_bpe_ranks(
            vocab_bpe_file=str(path / hashlib.sha1(public_vocab_url().encode()).hexdigest()),
            encoder_json_file=str(path / hashlib.sha1(public_encoder_url().encode()).hexdigest()),
        )
    return tiktoken.Encoding(
        name="gpt2",
        explicit_n_vocab=50257,
        pat_str=public.r50k_pat_str,
        mergeable_ranks=ranks,
        special_tokens={"<|endoftext|>": 50256},
    )


def public_vocab_url():
    return "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe"


def public_encoder_url():
    return "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json"
