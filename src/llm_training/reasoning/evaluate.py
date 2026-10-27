"""MATH-500 accuracy as chapter 3 of *Build a Reasoning Model (From Scratch)* measures it.

Greedy decoding up to `max_new_tokens` (the book evaluates with 2048), the book's prompt, the
answer taken from the last `\\boxed{}` or else the last number, and SymPy-equivalence grading.
Records append to a JSONL file one problem at a time, and a rerun skips problems already
recorded there, so a multi-hour evaluation survives interruption.
"""

import json
import time
from pathlib import Path

import torch

from llm_training.model.qwen3.model import QWEN3_0_6B, Qwen3Model
from llm_training.model.qwen3.pretrained import TOKENIZER, load_qwen3_state, verify_qwen3_base
from llm_training.model.qwen3.tokenizer import Qwen3Tokenizer

from .grading import extract_final_candidate, render_prompt
from .grpo import generate_greedy_ids

SNAPSHOT_FORMAT = 1


def load_snapshot(path, weights, device, attn_impl="sdpa"):
    """Return (model, tokenizer, snapshot metadata) for a GRPO weights snapshot.

    The tokenizer comes from the pinned base files in `weights`, which GRPO does not change.
    """
    snapshot = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(snapshot, dict) or snapshot.get("format_version") != SNAPSHOT_FORMAT:
        raise ValueError(f"{path} is not a GRPO weights snapshot")
    verify_qwen3_base(weights)
    tokenizer = Qwen3Tokenizer(str(Path(weights) / TOKENIZER))
    model = Qwen3Model(QWEN3_0_6B | snapshot["model_config"] | {"attn_impl": attn_impl})
    load_qwen3_state(model, snapshot["model_state_dict"])
    meta = {k: v for k, v in snapshot.items() if k != "model_state_dict"}
    return model.to(device).eval(), tokenizer, meta


def evaluate_math500(
    model: Qwen3Model,
    tokenizer: Qwen3Tokenizer,
    problems,
    grader,
    device,
    out_path: Path,
    max_new_tokens=2048,
    progress=True,
):
    """Score `problems` (MATH-500 rows) and return the summary over every recorded row."""
    out_path = Path(out_path)
    done = {}
    if out_path.is_file():
        for line in out_path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                done[record["index"]] = record
    for index, record in done.items():
        if not 1 <= index <= len(problems) or record["problem"] != problems[index - 1]["problem"]:
            raise ValueError(f"{out_path} holds records for a different problem list")
        if record["max_new_tokens"] != max_new_tokens:
            raise ValueError(
                f"{out_path} was recorded with max_new_tokens={record['max_new_tokens']}"
            )
    started = time.perf_counter()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as out:
        for index, row in enumerate(problems, start=1):
            if index in done:
                continue
            generated = generate_greedy_ids(
                model, tokenizer, render_prompt(row["problem"]), device, max_new_tokens
            )
            text = tokenizer.decode(generated)
            extracted = extract_final_candidate(text)
            correct, issue = grader.grade(extracted, row["answer"])
            record = {
                "index": index,
                "problem": row["problem"],
                "gtruth_answer": row["answer"],
                "generated_text": text,
                "extracted": extracted,
                "correct": correct,
                "gen_tokens": len(generated),
                "hit_token_limit": len(generated) == max_new_tokens,
                "max_new_tokens": max_new_tokens,
                "grader_issue": issue,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            done[index] = record
            if progress:
                correct_so_far = sum(r["correct"] for r in done.values())
                elapsed = time.perf_counter() - started
                print(
                    f"MATH-500 {len(done)}/{len(problems)}  correct {correct_so_far}  "
                    f"last {len(generated)} tok  {elapsed / 60:.1f} min",
                    flush=True,
                )
    records = [done[i] for i in sorted(done)]
    return {
        "problems": len(records),
        "correct": sum(r["correct"] for r in records),
        "accuracy": sum(r["correct"] for r in records) / len(records) if records else 0.0,
        "mean_gen_tokens": sum(r["gen_tokens"] for r in records) / len(records) if records else 0.0,
        "hit_token_limit": sum(r["hit_token_limit"] for r in records),
        "grader_issues": sum(r["grader_issue"] is not None for r in records),
        "max_new_tokens": max_new_tokens,
        "records": str(out_path),
    }
