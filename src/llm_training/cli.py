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
    grpo_prepare = commands.add_parser(
        "grpo-prepare", help="Fetch the pinned Qwen3-0.6B base files and MATH problem sets"
    )
    grpo_prepare.add_argument("--config", type=Path, required=True)
    grpo_training = commands.add_parser(
        "grpo-train", help="GRPO fine-tuning of Qwen3-0.6B base on MATH (verifiable rewards)"
    )
    grpo_inputs = grpo_training.add_mutually_exclusive_group(required=True)
    grpo_inputs.add_argument("--config", type=Path)
    grpo_inputs.add_argument("--resume", type=Path)
    grpo_training.add_argument(
        "--stop-after", type=int, help="Stop safely at this absolute completed step count"
    )
    grpo_training.add_argument(
        "--allow-code-drift",
        action="store_true",
        help="Resume even though GRPO code differs from the checkpoint's (recorded)",
    )
    math_eval = commands.add_parser(
        "math-eval", help="MATH-500 accuracy of the base model or a GRPO snapshot"
    )
    evaluated = math_eval.add_mutually_exclusive_group(required=True)
    evaluated.add_argument("--snapshot", type=Path, help="A GRPO snapshots/stepNNNNN.pt file")
    evaluated.add_argument("--base", action="store_true", help="The untrained base model")
    math_eval.add_argument("--weights", type=Path, default=Path("data/qwen3-0.6b-base"))
    math_eval.add_argument("--data", type=Path, default=Path("data/math"))
    math_eval.add_argument("--limit", type=int, default=500, help="Score the first N problems")
    math_eval.add_argument("--max-new-tokens", type=int, default=2048)
    math_eval.add_argument("--grade-timeout", type=float, default=30.0)
    math_eval.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    math_eval.add_argument(
        "--output",
        type=Path,
        help="JSONL records; reruns skip recorded problems (default: beside the snapshot)",
    )
    args = parser.parse_args()
    try:
        if args.command == "grpo-prepare":
            from .model.qwen3.pretrained import prepare_qwen3_base
            from .reasoning.config import load_grpo_config
            from .reasoning.math_data import prepare_math_data

            cfg = load_grpo_config(args.config)
            prepare_qwen3_base(cfg.model.weights)
            prepare_math_data(cfg.data.path)
            print(f"Prepared Qwen3-0.6B base in {cfg.model.weights} and MATH in {cfg.data.path}")
            return
        if args.command == "grpo-train":
            from .reasoning.config import load_grpo_config
            from .reasoning.trainer import train_grpo

            train_grpo(
                load_grpo_config(args.config) if args.config else None,
                resume=args.resume,
                stop_after=args.stop_after,
                allow_code_drift=args.allow_code_drift,
            )
            return
        if args.command == "math-eval":
            _math_eval(args)
            return
        if args.command == "inspect":
            from .training.provenance import collect_runtime_metadata

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
            from .model.gpt2.tokenizer import prepare_tokenizer

            prepare_tokenizer(Path(cfg.data.tokenizer_cache))
            prepare_corpus(cfg.data.corpus, Path(cfg.data.path), Path(cfg.data.tokenizer_cache))
            print(
                f"Prepared {cfg.data.corpus} tokens in {cfg.data.path} and GPT-2 tokenizer assets"
            )
        elif args.command == "prepare":
            from .model.gpt2.tokenizer import prepare_tokenizer

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


def _math_eval(args):
    from .model.qwen3.pretrained import load_qwen3_base
    from .reasoning.evaluate import evaluate_math500, load_snapshot
    from .reasoning.grading import TimeLimitedGrader
    from .reasoning.math_data import load_math_data

    if not 1 <= args.limit <= 500:
        raise ValueError("--limit must be between 1 and 500")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    problems = load_math_data(args.data, "math500")[: args.limit]
    if args.snapshot:
        model, tokenizer, meta = load_snapshot(args.snapshot, args.weights, args.device)
        output = args.output or args.snapshot.with_name(
            f"{args.snapshot.stem}.math500-first{args.limit}.jsonl"
        )
        evaluated = {"snapshot": str(args.snapshot.resolve()), "step": meta["step"]}
    else:
        if args.output is None:
            raise ValueError("--base needs --output")
        model, tokenizer = load_qwen3_base(args.weights, device=args.device, attn_impl="sdpa")
        output = args.output
        evaluated = {"base": str(args.weights.resolve())}
    with TimeLimitedGrader(args.grade_timeout) as grader:
        summary = evaluate_math500(
            model, tokenizer, problems, grader, args.device, output, args.max_new_tokens
        )
    summary = evaluated | summary
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
