"""Commands for explicit preparation, bounded training, and hardware inspection."""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from .config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    training = commands.add_parser("train")
    inputs = training.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--config", type=Path)
    inputs.add_argument("--resume", type=Path)
    training.add_argument(
        "--stop-after", type=int, help="Stop safely at this absolute completed optimizer-step count"
    )
    training.add_argument(
        "--allow-code-drift",
        action="store_true",
        help="Resume even though training code differs from the checkpoint's (recorded)",
    )
    commands.add_parser("inspect")
    prediction = commands.add_parser("predict", help="Show next-token candidates and probabilities")
    prediction.add_argument("--checkpoint", type=Path, required=True)
    prediction.add_argument("--prompt", required=True)
    prediction.add_argument("--tokenizer-cache", type=Path, default=Path("data/tokenizer"))
    prediction.add_argument("--max-new-tokens", type=int, default=8)
    prediction.add_argument("--top-k", type=int, default=5, help="Number of candidates to display")
    prediction.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    prediction.add_argument("--num-threads", type=int, default=1)
    prediction.add_argument("--output", type=Path, help="Save the full prediction trace as JSON")
    generation = commands.add_parser("generate", help="Sample a continuation from a checkpoint")
    generation.add_argument("--checkpoint", type=Path, required=True)
    generation.add_argument("--prompt", default="Once upon a time")
    generation.add_argument("--tokenizer-cache", type=Path, default=Path("data/tokenizer"))
    generation.add_argument("--max-new-tokens", type=int, default=200)
    generation.add_argument("--temperature", type=float, default=0.8, help="0 decodes greedily")
    generation.add_argument("--top-k", type=int, help="Sample only from the k likeliest tokens")
    generation.add_argument("--seed", type=int, default=0)
    generation.add_argument("--num-samples", type=int, default=1, help="Seeds seed..seed+n-1")
    generation.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    generation.add_argument("--num-threads", type=int, default=1)
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            from .training.trainer import collect_runtime_metadata

            print(json.dumps(collect_runtime_metadata(), indent=2))
            return
        if args.command == "predict":
            from .prediction import predict_checkpoint, render_markdown

            report = predict_checkpoint(
                args.checkpoint,
                args.tokenizer_cache,
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                top_k=args.top_k,
                device=args.device,
                num_threads=args.num_threads,
            )
            if args.output:
                if args.output.resolve() == args.checkpoint.resolve():
                    raise ValueError("Prediction output must not overwrite the model checkpoint")
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(render_markdown(report))
            return
        if args.command == "generate":
            from .prediction import generate_from_checkpoint

            if args.num_samples <= 0:
                raise ValueError("--num-samples must be positive")
            for seed in range(args.seed, args.seed + args.num_samples):
                result = generate_from_checkpoint(
                    args.checkpoint,
                    args.tokenizer_cache,
                    args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    seed=seed,
                    device=args.device,
                    num_threads=args.num_threads,
                )
                print(f"--- seed {seed}, update {result['optimizer_steps']}\n{result['text']}\n")
            return
        cfg = load_config(args.config) if args.config else None
        if args.command == "prepare" and cfg.data.format == "tokens":
            from .data.corpus import prepare_corpus
            from .data.tokenizer import prepare_tokenizer

            prepare_tokenizer(Path(cfg.data.tokenizer_cache))
            prepare_corpus(cfg.data.corpus, Path(cfg.data.path), Path(cfg.data.tokenizer_cache))
            print(
                f"Prepared {cfg.data.corpus} tokens in {cfg.data.path} and GPT-2 tokenizer assets"
            )
        elif args.command == "prepare":
            from .data.tokenizer import prepare_tokenizer

            path = Path(cfg.data.path)
            if not path.exists():
                if cfg.data.source_url is None:
                    raise ValueError(f"Missing local dataset: {path}")
                with urllib.request.urlopen(cfg.data.source_url, timeout=30) as response:
                    data = response.read()
                if hashlib.sha256(data).hexdigest() != cfg.data.sha256:
                    raise ValueError("Downloaded data SHA256 mismatch")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            if cfg.data.sha256 and hashlib.sha256(path.read_bytes()).hexdigest() != cfg.data.sha256:
                raise ValueError("Existing data SHA256 mismatch")
            prepare_tokenizer(Path(cfg.data.tokenizer_cache))
            print(f"Prepared {path} and GPT-2 tokenizer assets")
        else:
            from .training.trainer import train

            train(
                cfg,
                resume=args.resume,
                stop_after=args.stop_after,
                allow_code_drift=args.allow_code_drift,
            )
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
