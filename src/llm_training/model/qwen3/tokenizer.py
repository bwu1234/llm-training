# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSES/reasoning-from-scratch.txt).
# Source: https://github.com/rasbt/reasoning-from-scratch (reasoning_from_scratch/qwen3.py)
# Modified: selected symbols, reorganized imports/modules and formatting for llm-training.

import re
from pathlib import Path


class Qwen3Tokenizer:
    _SPECIALS = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|object_ref_start|>",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|box_end|>",
        "<|quad_start|>",
        "<|quad_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]
    _SPLIT_RE = re.compile(r"(<\|[^>]+?\|>)")

    def __init__(
        self,
        tokenizer_file_path="tokenizer-base.json",
        apply_chat_template=False,
        add_generation_prompt=False,
        add_thinking=False,
    ):
        from tokenizers import Tokenizer

        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking

        tok_path = Path(tokenizer_file_path)
        if not tok_path.is_file():
            raise FileNotFoundError(
                f"Tokenizer file '{tok_path}' not found. Please ensure it's available."
            )

        self._tok = Tokenizer.from_file(str(tok_path))
        self._special_to_id = {t: self._tok.token_to_id(t) for t in self._SPECIALS}

        self.pad_token = "<|endoftext|>"
        self.pad_token_id = self._special_to_id.get(self.pad_token)

        # Match HF behavior: chat model → <|im_end|>, base model → <|endoftext|>
        fname = tok_path.name.lower()
        if "base" in fname and "reasoning" not in fname:
            self.eos_token = "<|endoftext|>"
        else:
            self.eos_token = "<|im_end|>"
        self.eos_token_id = self._special_to_id.get(self.eos_token)

    def encode(self, prompt, chat_wrapped=None):
        if chat_wrapped is None:
            chat_wrapped = self.apply_chat_template

        stripped = prompt.strip()
        if stripped in self._special_to_id and "\n" not in stripped:
            return [self._special_to_id[stripped]]

        if chat_wrapped:
            prompt = self._wrap_chat(prompt)

        ids = []
        for part in filter(None, self._SPLIT_RE.split(prompt)):
            if part in self._special_to_id:
                ids.append(self._special_to_id[part])
            else:
                ids.extend(self._tok.encode(part).ids)
        return ids

    def decode(self, token_ids):
        return self._tok.decode(token_ids, skip_special_tokens=False)

    def _wrap_chat(self, user_msg):
        s = f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        if self.add_generation_prompt:
            s += "<|im_start|>assistant"
            if self.add_thinking:
                s += "\n"  # insert no <think> tag, just a new line
            else:
                s += "\n<think>\n\n</think>\n\n"
        return s


def _bytes_to_unicode():
    """GPT-2's byte-level BPE alphabet: each byte maps to one printable character."""
    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1)]
    printable += range(ord("®"), ord("ÿ") + 1)
    chars, extra = {}, 0
    for byte in range(256):
        if byte in printable:
            chars[byte] = chr(byte)
        else:
            chars[byte] = chr(256 + extra)
            extra += 1
    return chars


class Qwen3TextTokenizer:
    """The subset of tiktoken's `Encoding` that the next-token trainer uses, for Qwen3.

    Not part of the source: it lets the corpus builder, trainer and `predict` treat the GPT-2
    and Qwen3 tokenizers alike. Raw text is encoded literally, as tiktoken's `encode_ordinary`
    does: `<|im_start|>` inside a document becomes ordinary tokens, not the control token,
    unless `encode` is told to allow it.
    """

    name = "qwen3"

    def __init__(self, tokenizer_file_path):
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(str(tokenizer_file_path))
        self._tok.encode_special_tokens = True
        self._added = {i: t.content for i, t in self._tok.get_added_tokens_decoder().items()}
        self._special = {t: i for i, t in self._added.items() if t in Qwen3Tokenizer._SPECIALS}
        self.eot_token = self._special["<|endoftext|>"]
        self.n_vocab = self._tok.get_vocab_size()
        self._byte_values = {c: b for b, c in _bytes_to_unicode().items()}

    def encode_ordinary(self, text):
        return self._tok.encode(text, add_special_tokens=False).ids

    def encode_ordinary_batch(self, texts):
        return [e.ids for e in self._tok.encode_batch(texts, add_special_tokens=False)]

    def encode(self, text, allowed_special=frozenset()):
        allowed = set(self._special) if allowed_special == "all" else set(allowed_special)
        if unknown := allowed - set(self._special):
            raise ValueError(f"Not Qwen3 special tokens: {sorted(unknown)}")
        if not allowed:
            return self.encode_ordinary(text)
        pattern = "(" + "|".join(map(re.escape, sorted(allowed))) + ")"
        ids = []
        for part in filter(None, re.split(pattern, text)):
            ids += [self._special[part]] if part in allowed else self.encode_ordinary(part)
        return ids

    def decode(self, token_ids):
        return self._tok.decode(token_ids, skip_special_tokens=False)

    def decode_single_token_bytes(self, token_id):
        if token_id in self._added:
            return self._added[token_id].encode()
        token = self._tok.id_to_token(token_id)
        if token is None:
            # The model's embedding has 151,936 rows; ids past the tokenizer's 151,665 are
            # padding with no text.
            return b""
        return bytes(self._byte_values[c] for c in token)
