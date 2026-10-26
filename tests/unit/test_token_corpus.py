"""Token corpus preparation, verification, and memory-mapped windows (offline, stub tokenizer)."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from llm_training.data import corpus
from llm_training.data.corpus import build_token_corpus, iter_documents, load_token_corpus
from llm_training.data.dataset import SlidingWindowTokenDataset
from llm_training.data.token_dataset import TokenWindowDataset

EOT = 50256
TOKENIZER_MANIFEST = {"encoding": "gpt2", "files": {"a": "0" * 64}}


class CharTokenizer:
    """One token per character; enough to check framing, dtype, and manifests."""

    eot_token = EOT

    def __init__(self, offset=0):
        self.offset = offset

    def encode_ordinary_batch(self, texts):
        return [[ord(c) + self.offset for c in text] for text in texts]


def ids(text):
    return [ord(c) for c in text]


def build(tmp_path, train_text, val_text="v", tokenizer=None, name="tinystories"):
    sources = {}
    tmp_path.mkdir(parents=True, exist_ok=True)
    for split, text in (("train", train_text), ("val", val_text)):
        sources[split] = tmp_path / f"{split}.txt"
        sources[split].write_text(text)
    out = tmp_path / "corpus"
    manifest = build_token_corpus(
        sources, out, tokenizer or CharTokenizer(), TOKENIZER_MANIFEST, {"corpus": name}
    )
    return out, manifest


def test_documents_split_on_separator_even_across_read_chunks(tmp_path):
    path = tmp_path / "raw.txt"
    path.write_text("ab\n<|endoftext|>\n  cd \n<|endoftext|><|endoftext|>\nef\n")
    for chunk_chars in (1, 3, 7, 1 << 20):
        assert list(iter_documents(path, chunk_chars)) == ["ab", "cd", "ef"]


def test_build_prepends_end_of_text_and_records_counts(tmp_path):
    out, manifest = build(tmp_path, "ab<|endoftext|>c", "xyz")
    train = np.fromfile(out / "train.bin", dtype=np.uint16).tolist()
    val = np.fromfile(out / "val.bin", dtype=np.uint16).tolist()
    assert train == [EOT, *ids("ab"), EOT, *ids("c")]
    assert val == [EOT, *ids("xyz")]
    assert manifest["splits"]["train"] | {"sha256": None, "source_sha256": None} == {
        "file": "train.bin",
        "sha256": None,
        "tokens": 5,
        "documents": 2,
        "source_sha256": None,
    }
    assert (
        manifest["splits"]["train"]["sha256"]
        == hashlib.sha256((out / "train.bin").read_bytes()).hexdigest()
    )
    loaded, digest = load_token_corpus(out, "tinystories", TOKENIZER_MANIFEST)
    assert loaded == manifest
    assert digest == hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest()


def test_small_batches_give_identical_files(tmp_path):
    text = "<|endoftext|>".join(f"story {i}" for i in range(50))
    whole, _ = build(tmp_path / "a", text)
    sources = {"train": tmp_path / "a/train.txt", "val": tmp_path / "a/val.txt"}
    batched = build_token_corpus(
        sources,
        tmp_path / "b",
        CharTokenizer(),
        TOKENIZER_MANIFEST,
        {"corpus": "tinystories"},
        batch_documents=3,
    )
    assert batched["splits"] == json.loads((whole / "manifest.json").read_text())["splits"]


def test_ids_that_do_not_fit_uint16_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="uint16"):
        build(tmp_path, "a", tokenizer=CharTokenizer(offset=70000))
    assert not (tmp_path / "corpus/manifest.json").exists()


def test_verification_rejects_corruption_tokenizer_and_corpus_mismatch(tmp_path):
    out, _ = build(tmp_path, "abcdef")
    with pytest.raises(ValueError, match="not 'other'"):
        load_token_corpus(out, "other", TOKENIZER_MANIFEST)
    with pytest.raises(ValueError, match="different tokenizer"):
        load_token_corpus(out, "tinystories", TOKENIZER_MANIFEST | {"encoding": "x"})
    data = bytearray((out / "train.bin").read_bytes())
    data[0] ^= 1
    (out / "train.bin").write_bytes(bytes(data))
    with pytest.raises(ValueError, match="does not match"):
        load_token_corpus(out, "tinystories", TOKENIZER_MANIFEST)
    with pytest.raises(ValueError, match="prepare"):
        load_token_corpus(tmp_path / "missing", "tinystories", TOKENIZER_MANIFEST)


class ListTokenizer:
    def __init__(self, tokens):
        self.tokens = tokens

    def encode(self, text, allowed_special=None):
        return self.tokens


@pytest.mark.parametrize("length,stride", [(4, 4), (4, 1), (5, 3), (8, 8)])
def test_memmap_windows_equal_the_book_sliding_windows(tmp_path, length, stride):
    tokens = torch.randint(0, 50257, (103,), generator=torch.Generator().manual_seed(0)).tolist()
    path = tmp_path / "tokens.bin"
    np.array(tokens, dtype=np.uint16).tofile(path)
    expected = SlidingWindowTokenDataset("", ListTokenizer(tokens), length, stride)
    actual = TokenWindowDataset(path, length, stride)
    assert len(actual) == len(expected) > 0
    for i in range(len(expected)):
        for a, b in zip(actual[i], expected[i]):
            assert a.dtype == torch.int64
            assert torch.equal(a, b)


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_download_verifies_pinned_hash_and_reuses_verified_files(tmp_path, monkeypatch):
    payload = b"story<|endoftext|>"
    good = hashlib.sha256(payload).hexdigest()
    spec = {
        "repository": "https://example.invalid/data",
        "revision": "rev",
        "files": {"train": ("t.txt", good, len(payload)), "val": ("v.txt", good, len(payload))},
    }
    monkeypatch.setitem(corpus.CORPORA, "fake", spec)
    urls = []

    def urlopen(url, timeout):
        urls.append(url)
        return Response(payload)

    monkeypatch.setattr(corpus.urllib.request, "urlopen", urlopen)
    paths = corpus.download_corpus("fake", tmp_path)
    assert urls == ["https://example.invalid/data/resolve/rev/t.txt", f"{urls[0][:-5]}v.txt"]
    assert paths["train"].read_bytes() == payload
    corpus.download_corpus("fake", tmp_path)
    assert len(urls) == 2  # verified local copies are reused

    paths["train"].unlink()
    spec["files"]["train"] = ("t.txt", "0" * 64, len(payload))
    with pytest.raises(ValueError, match="pinned"):
        corpus.download_corpus("fake", tmp_path)
    assert not Path(paths["train"]).exists()
    assert not list(tmp_path.glob("*.part"))
