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
    prediction.add_argument(
        "--tokenizer-cache",
        type=Path,
        help="Tokenizer assets (default: data/tokenizer for GPT-2, data/qwen3-0.6b-base for Qwen3)",
    )
    prediction.add_argument("--max-new-tokens", type=int, default=8)
    prediction.add_argument("--top-k", type=int, default=5, help="Number of candidates to display")
    prediction.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    prediction.add_argument("--num-threads", type=int, default=1)
    prediction.add_argument("--output", type=Path, help="Save the full prediction trace as JSON")
    generation = commands.add_parser("generate", help="Sample a continuation from a checkpoint")
    generation.add_argument("--checkpoint", type=Path, required=True)
    generation.add_argument("--prompt", default="Once upon a time")
    generation.add_argument(
        "--tokenizer-cache",
        type=Path,
        help="Tokenizer assets (default: data/tokenizer for GPT-2, data/qwen3-0.6b-base for Qwen3)",
    )
    generation.add_argument("--max-new-tokens", type=int, default=200)
    generation.add_argument("--temperature", type=float, default=0.8, help="0 decodes greedily")
    generation.add_argument("--top-k", type=int, help="Sample only from the k likeliest tokens")
    generation.add_argument("--seed", type=int, default=0)
    generation.add_argument("--num-samples", type=int, default=1, help="Seeds seed..seed+n-1")
    generation.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    generation.add_argument("--num-threads", type=int, default=1)
    responding = commands.add_parser(
        "respond", help="Greedy responses of a fine-tuned model to held-out instructions"
    )
    responding.add_argument("--checkpoint", type=Path, required=True)
    responding.add_argument(
        "--config", type=Path, required=True, help="The run's recipe (config.resolved.yaml)"
    )
    responding.add_argument("--split", default="test", help="test (chapter 7) or validation")
    responding.add_argument("--limit", type=int, help="Respond to the first N examples")
    responding.add_argument("--max-new-tokens", type=int, default=256)
    responding.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    responding.add_argument(
        "--output", type=Path, help="JSONL; reruns skip recorded examples (default: beside it)"
    )
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
    evaluated.add_argument("--model", type=Path, help="A Qwen3 model.pt, e.g. from SFT")
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
        if args.command == "respond":
            from .respond import respond

            if args.limit is not None and args.limit <= 0:
                raise ValueError("--limit must be positive")
            if args.max_new_tokens <= 0:
                raise ValueError("--max-new-tokens must be positive")
            output = args.output or args.checkpoint.with_name(
                f"{args.checkpoint.stem}.{args.split}-responses.jsonl"
            )
            respond(
                args.checkpoint,
                args.config,
                split=args.split,
                limit=args.limit,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
                output=output,
            )
            print(f"Responses: {output}")
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
        if args.command == "prepare":
            _prepare(cfg)
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


def _prepare(cfg):
    """Fetch everything a recipe reads: tokenizer, starting weights, and data."""
    from .model.registry import PRETRAINED, prepare_pretrained, prepare_tokenizer

    prepare_tokenizer(cfg.model.arch, Path(cfg.data.tokenizer_cache))
    if cfg.model.init_from in PRETRAINED:
        prepare_pretrained(cfg.model.init_from, cfg.model.weights_dir)
        print(f"Prepared {cfg.model.init_from} weights in {cfg.model.weights_dir}")
    if cfg.data.format == "tokens":
        from .data.corpus import prepare_corpus

        prepare_corpus(
            cfg.data.corpus, Path(cfg.data.path), cfg.model.arch, Path(cfg.data.tokenizer_cache)
        )
        print(f"Prepared {cfg.data.corpus} {cfg.model.arch} tokens in {cfg.data.path}")
        return
    if cfg.data.format == "instructions":
        from .data.instructions import prepare_instructions

        prepare_instructions(cfg.data.dataset, cfg.data.path)
        print(f"Prepared {cfg.data.dataset} in {cfg.data.path}")
        return
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
    print(f"Prepared {path} and {cfg.model.arch} tokenizer assets")


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
    if args.model:
        import torch

        from .model.assets import verify_pinned
        from .model.qwen3.pretrained import FILES, TOKENIZER
        from .model.qwen3.tokenizer import Qwen3Tokenizer
        from .prediction import checkpoint_arch, load_inference_model

        if checkpoint_arch(args.model) != "qwen3":
            raise ValueError("math-eval scores Qwen3 models")
        # Only the tokenizer comes from --weights; the model is the file being scored.
        verify_pinned({TOKENIZER: FILES[TOKENIZER]}, args.weights, "`llm-training prepare`")
        model, _, _, steps, _ = load_inference_model(
            args.model, args.weights, args.device, torch.get_num_threads()
        )
        model.eval()
        tokenizer = Qwen3Tokenizer(str(args.weights / TOKENIZER))
        output = args.output or args.model.with_name(
            f"{args.model.stem}.math500-first{args.limit}.jsonl"
        )
        evaluated = {"model": str(args.model.resolve()), "optimizer_steps": steps}
    elif args.snapshot:
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
