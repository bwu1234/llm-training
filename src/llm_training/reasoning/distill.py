# Copyright (c) Sebastian Raschka under Apache License 2.0 (see model/qwen3/LICENSE.txt).
# Source: https://github.com/rasbt/reasoning-from-scratch
# (ch08/04_train_with_distillation/distill.py)
# Modified: selected symbols and formatting for llm-training.

"""Chapter 8's answer format for distilling teacher reasoning traces into the base model.

`format_distilled_answer` is the training target without think tokens, the base-model default
of the chapter's `distill.py`: the teacher's reasoning, a blank line, then its final answer.
"""


def strip_think_tags(text):
    return text.replace("<think>", "").replace("</think>", "").strip()


def format_distilled_answer(entry, use_think_tokens=False):
    content = str(entry["message_content"]).strip()
    if not content:
        raise ValueError("Missing non-empty 'message_content' field.")

    content = strip_think_tags(content)

    if "message_thinking" in entry:
        thinking = str(entry["message_thinking"]).strip()
    else:
        thinking = ""
    thinking = strip_think_tags(thinking)

    if use_think_tokens:
        return f"<think>{thinking}</think>\n\n{content}"

    if thinking:
        return f"{thinking}\n\n{content}"

    return content
