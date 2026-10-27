"""Inspect the full-vocabulary next-token distribution along a greedy rollout."""

import hashlib
import html
import json
from pathlib import Path

import torch

from .model.registry import check_model_file, family, family_name


def _token(tokenizer, token_id):
    raw = tokenizer.decode_single_token_bytes(token_id)
    try:
        text = raw.decode("utf-8")
        complete = True
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
        complete = False
    return {"token_id": token_id, "text": text, "bytes_hex": raw.hex(), "complete_utf8": complete}


def trace_next_tokens(
    model, tokenizer, prompt, *, context_size, max_new_tokens=8, top_k=5, device="cpu"
):
    """Report raw softmax probabilities, showing top_k without renormalizing.

    Always append the highest-logit token (lowest ID breaks exact ties), matching
    generate_greedy. Each new distribution conditions on the selected prefix.
    """
    for name, value in (
        ("context_size", context_size),
        ("max_new_tokens", max_new_tokens),
        ("top_k", top_k),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("Prompt must contain at least one token")
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    steps = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for step in range(1, max_new_tokens + 1):
                context = ids[:, -context_size:]
                logits = model(context)[0, -1, :]
                if top_k > logits.numel():
                    raise ValueError(f"top_k must not exceed vocabulary size {logits.numel()}")
                if not torch.isfinite(logits).all():
                    raise ValueError("Model produced nonfinite next-token logits")
                probabilities = torch.softmax(logits, dim=-1)
                # Sorting logits keeps argmax and displayed order consistent even
                # when softmax rounds two different logits to the same probability.
                candidates = torch.argsort(logits, descending=True, stable=True)[:top_k]
                selected_id = int(torch.argmax(logits).item())
                rows = [
                    _token(tokenizer, int(token_id))
                    | {
                        "rank": rank,
                        "probability": float(probabilities[token_id].item()),
                        "selected": int(token_id) == selected_id,
                    }
                    for rank, token_id in enumerate(candidates.tolist(), start=1)
                ]
                mass = sum(row["probability"] for row in rows)
                context_ids = context[0].tolist()
                steps.append(
                    {
                        "step": step,
                        "context_token_ids": context_ids,
                        "context_text": tokenizer.decode(context_ids),
                        "context_tokens_dropped": ids.shape[1] - context.shape[1],
                        "candidates": rows,
                        "selected_token_id": selected_id,
                        "displayed_probability_mass": mass,
                        "other_probability_mass": max(0.0, 1.0 - mass),
                    }
                )
                ids = torch.cat((ids, torch.tensor([[selected_id]], device=device)), dim=1)
    finally:
        model.train(was_training)
    all_ids = ids[0].tolist()
    generated = all_ids[len(prompt_ids) :]
    return {
        "format_version": 1,
        "prompt": prompt,
        "prompt_token_ids": prompt_ids,
        "selection": "greedy argmax; lowest token ID breaks exact logit ties",
        "probability_definition": "softmax over the full vocabulary; temperature=1; no filtering",
        "top_k_display_only": top_k,
        "context_size": context_size,
        "max_new_tokens": max_new_tokens,
        "steps": steps,
        "generated_token_ids": generated,
        "generated_text": tokenizer.decode(generated),
        "all_token_ids": all_ids,
        "text": tokenizer.decode(all_ids),
    }


def predict_checkpoint(
    checkpoint, tokenizer_cache, prompt, *, max_new_tokens=8, top_k=5, device="cpu", num_threads=1
):
    """Load inference weights or a full checkpoint without restoring training state."""
    checkpoint = Path(checkpoint).resolve()
    model, tokenizer, config, optimizer_steps, manifest = load_inference_model(
        checkpoint, tokenizer_cache, device, num_threads
    )
    report = trace_next_tokens(
        model,
        tokenizer,
        prompt,
        context_size=config["context_length"],
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        device=device,
    )
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return report | {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest.hexdigest(),
        "optimizer_steps": optimizer_steps,
        "model_config": config,
        "tokenizer_manifest": manifest,
        "device": device,
        "num_threads": num_threads,
        "torch": str(torch.__version__),
    }


def generate_from_checkpoint(
    checkpoint,
    tokenizer_cache,
    prompt,
    *,
    max_new_tokens=200,
    temperature=0.8,
    top_k=None,
    seed=0,
    device="cpu",
    num_threads=1,
):
    """Sample a continuation; stops early at `<|endoftext|>`. Temperature 0 is greedy."""
    from .lm.generation import generate_sampled

    for name, value in (("max_new_tokens", max_new_tokens), ("top_k", top_k)):
        if (value is not None or name != "top_k") and (type(value) is not int or value <= 0):
            raise ValueError(f"{name} must be a positive integer")
    if type(temperature) not in (int, float) or not temperature >= 0:
        raise ValueError("temperature must be a non-negative number")
    model, tokenizer, config, optimizer_steps, _ = load_inference_model(
        checkpoint, tokenizer_cache, device, num_threads
    )
    prompt_ids = tokenizer.encode(prompt) or [tokenizer.eot_token]
    model.eval()
    ids = generate_sampled(
        model,
        torch.tensor([prompt_ids], device=device),
        max_new_tokens,
        config["context_length"],
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.eot_token,
        generator=torch.Generator().manual_seed(seed),
    )[0].tolist()
    return {
        "prompt": prompt,
        "text": tokenizer.decode(ids),
        "generated_text": tokenizer.decode(ids[len(prompt_ids) :]),
        "token_ids": ids,
        "temperature": temperature,
        "top_k": top_k,
        "seed": seed,
        "checkpoint": str(Path(checkpoint).resolve()),
        "optimizer_steps": optimizer_steps,
    }


def read_saved_model(checkpoint):
    """Return (model config, state dict, optimizer steps, tokenizer manifest) from a project
    model.pt or a full training checkpoint, told apart by their keys."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if isinstance(saved, dict) and "model_config" in saved:
        saved = check_model_file(saved, checkpoint)
        return (
            saved["model_config"],
            saved["model_state_dict"],
            saved.get("optimizer_steps"),
            saved.get("tokenizer"),
        )
    if not isinstance(saved, dict) or not isinstance(saved.get("config", {}).get("model"), dict):
        raise ValueError("Expected a project model.pt or full training checkpoint")
    if "model_state_dict" not in saved:
        raise ValueError("Checkpoint is missing model configuration or weights")
    return (
        saved["config"]["model"],
        saved["model_state_dict"],
        saved.get("progress", {}).get("optimizer_steps"),
        saved.get("data_identity", {}).get("tokenizer_manifest"),
    )


def checkpoint_arch(checkpoint):
    """The model family of a model.pt or full checkpoint, read without building the model."""
    return family_name(read_saved_model(checkpoint)[0])


def load_inference_model(checkpoint, tokenizer_cache=None, device="cpu", num_threads=1):
    """Return (model, tokenizer, model config, optimizer steps, tokenizer manifest)."""
    if type(num_threads) is not int or num_threads <= 0:
        raise ValueError("num_threads must be a positive integer")
    if device not in ("cpu", "mps", "cuda"):
        raise ValueError("device must be cpu, mps, or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    torch.set_num_threads(num_threads)
    checkpoint = Path(checkpoint).resolve()
    config, state, optimizer_steps, expected = read_saved_model(checkpoint)
    model_family = family(family_name(config))
    if tokenizer_cache is None:
        tokenizer_cache = model_family.default_tokenizer_cache
    tokenizer, manifest = model_family.load_tokenizer(Path(tokenizer_cache).resolve())
    if expected is not None and expected != manifest:
        raise ValueError("Tokenizer identity differs from the checkpoint")
    model = model_family.build(config | {"arch": model_family.name})
    model_family.load_weights(model, state)
    del state  # Full checkpoints also hold optimizer tensors, unneeded for inference.
    model.to(device)
    return model, tokenizer, config, optimizer_steps, manifest


def _code(text):
    escaped = html.escape(json.dumps(text, ensure_ascii=False), quote=False).replace("|", "&#124;")
    return f"<code>{escaped}</code>"


def render_markdown(report):
    """Escaped token strings make whitespace, punctuation and UTF-8 fragments visible."""
    lines = ["# Next-token predictions", "", f"Prompt: {_code(report['prompt'])}", ""]
    if "checkpoint" in report:
        lines += [
            f"Checkpoint: {_code(report['checkpoint'])}",
            "",
            f"Training update: {report['optimizer_steps']}; device: {report['device']}.",
            "",
        ]
    lines += [
        "Probabilities use the full-vocabulary softmax at temperature 1. Top-k only limits",
        "the displayed candidates; it does not renormalize probabilities or change generation.",
        "Each step appends the greedy winner. Quoted token strings expose leading spaces",
        "and escapes such as `\\n`. A token can be part of a word or a UTF-8 character;",
        "the JSON report preserves exact token bytes. Percentages are rounded for display.",
        "",
    ]
    for step in report["steps"]:
        lines += [f"## Step {step['step']}", "", f"Context: {_code(step['context_text'])}", ""]
        if step["context_tokens_dropped"]:
            lines += [
                f"Context window dropped {step['context_tokens_dropped']} earlier token(s).",
                "",
            ]
        lines += [
            "| Rank | Token ID | Token text | Probability | Selected |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in step["candidates"]:
            text = _code(row["text"])
            if not row["complete_utf8"]:
                text += f" (bytes: {row['bytes_hex']})"
            lines.append(
                f"| {row['rank']} | {row['token_id']} | {text} | "
                f"{row['probability']:.4%} | {'yes' if row['selected'] else ''} |"
            )
        lines += ["", f"Other tokens: {step['other_probability_mass']:.4%} probability mass.", ""]
    lines += ["## Generated text", "", _code(report["text"]), ""]
    return "\n".join(lines)
