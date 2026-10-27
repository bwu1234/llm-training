# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSES/reasoning-from-scratch.txt).
# Source: https://github.com/rasbt/reasoning-from-scratch (reasoning_from_scratch/ch02.py,
# ch04.py, ch06.py, and ch06/02_rlvr_grpo_scripts_intro/rlvr_grpo_original_no_kl.py)
# Modified: selected symbols, reorganized imports/modules and formatting for llm-training.
# Also modified:
# - Sampling draws from an explicit `generator` (the training loop checkpoints its state) and
#   samples on the model's device, as the bonus script does, instead of copying to the CPU.
# - Grading goes through a `TimeLimitedGrader`.
# - `grpo_step` (upstream `compute_grpo_loss` plus the backward pass) backpropagates each
#   rollout as soon as its log-probability is computed, so only one rollout's activations are
#   alive at a time. The accumulated gradient equals upstream's `mean` loss gradient.
# - When every rollout earns the same reward, the advantages are exactly zero, so the
#   gradient is zero; `grpo_step` then sets zero gradients without the forward/backward pass.

import torch

from llm_training.model.qwen3.model import KVCache

from .grading import extract_final_candidate, render_prompt


def top_p_filter(probas, top_p):
    if top_p is None or top_p >= 1.0:
        return probas

    # Step 4.1: Sort by descending probability
    sorted_probas, sorted_idx = torch.sort(probas, dim=1, descending=True)

    # Step 4.2: Cumulative sum
    cumprobas = torch.cumsum(sorted_probas, dim=1)

    # Step 4.3.1: Keep tokens where prefix cumulative mass (before token) is < top_ps
    # Example: [0.5, 0.41, 0.09] with top_p=0.9 should keep the first two tokens
    prefix = cumprobas - sorted_probas  # cumulative mass before each token
    keep = prefix < top_p
    # Always keep at least one token (fallback for very small/non-positive top_p)
    keep[:, 0] = True

    # Step 4.3.2: Zero out beyond cutoff
    kept_sorted = torch.where(keep, sorted_probas, torch.zeros_like(sorted_probas))
    # Step 4.3.3: Map back to original order
    filtered = torch.zeros_like(probas).scatter(1, sorted_idx, kept_sorted)

    # Step 4.4: Renormalize to sum to 1
    denom = torch.sum(filtered, dim=1, keepdim=True).clamp_min(1e-12)
    # keepdim=True is technically not necessary but it makes the code work in batched cases
    return filtered / denom


@torch.no_grad()
def sample_response(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9,
    generator=None,
):
    input_ids = torch.tensor(tokenizer.encode(prompt), device=device)

    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    logits = model(input_ids.unsqueeze(0), cache=cache)[:, -1]

    generated = []
    for step in range(max_new_tokens):
        if temperature and temperature != 1.0:
            logits = logits / temperature

        probas = torch.softmax(logits, dim=-1)
        probas = top_p_filter(probas, top_p)
        next_token = torch.multinomial(probas, num_samples=1, generator=generator)

        token_id = next_token.item()
        generated.append(token_id)

        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            break
        if step + 1 < max_new_tokens:  # Skips the forward pass after the last token is generated
            logits = model(next_token, cache=cache)[:, -1]

    full_token_ids = torch.cat(
        [
            input_ids,
            torch.tensor(generated, device=device, dtype=input_ids.dtype),
        ]
    )
    return full_token_ids, input_ids.numel(), tokenizer.decode(generated)


@torch.no_grad()
def generate_greedy_ids(model, tokenizer, prompt, device, max_new_tokens):
    """Greedy KV-cache decoding as in upstream `generate_text_basic_stream_cache` (ch02):
    stops before emitting EOS. Returns the generated token IDs."""
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    token_ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
    out = model(token_ids, cache=cache)[:, -1]
    generated = []
    for step in range(max_new_tokens):
        next_token = torch.argmax(out, dim=-1, keepdim=True)
        if tokenizer.eos_token_id is not None and torch.all(next_token == tokenizer.eos_token_id):
            break
        generated.append(next_token.item())
        if step + 1 < max_new_tokens:  # Skips the forward pass after the last token is generated
            out = model(next_token, cache=cache)[:, -1]
    return generated


def reward_rlvr(answer_text, ground_truth, grader):
    """Return (reward, grader issue or None)."""
    extracted = extract_final_candidate(
        answer_text,
        fallback=None,  # Require \boxed{}
    )
    if not extracted:
        return 0.0, None
    correct, issue = grader.grade(extracted, ground_truth)
    return float(correct), issue


def sequence_logprob(model, token_ids, prompt_len):
    logits = model(token_ids.unsqueeze(0)).squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=-1)
    selected = logprobs[:-1].gather(1, token_ids[1:].unsqueeze(-1)).squeeze(-1)
    return torch.sum(selected[prompt_len - 1 :])


def grpo_step(
    model,
    tokenizer,
    example,
    grader,
    device,
    num_rollouts=8,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9,
    generator=None,
):
    """Sample a group of rollouts, reward them, and accumulate the GRPO policy-gradient
    gradient into `.grad`. The caller clips and steps.

    Returns the statistics and per-rollout samples for logging.
    """
    if num_rollouts < 2:
        raise ValueError("GRPO needs at least two rollouts per problem")
    prompt = render_prompt(example["problem"])

    was_training = model.training
    model.eval()
    rollouts, rewards, samples = [], [], []
    for _ in range(num_rollouts):
        # Stage 1: generate rollouts
        token_ids, prompt_len, text = sample_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            generator=generator,
        )
        # Stage 2: compute rewards
        reward, issue = reward_rlvr(text, example["answer"], grader)
        rollouts.append((token_ids, prompt_len))
        rewards.append(reward)
        samples.append(
            {
                "text": text,
                "reward": reward,
                "gen_len": token_ids.numel() - prompt_len,
                "grader_issue": issue,
            }
        )
    model.train(was_training)

    # Stage 3: compute advantages
    reward_tensor = torch.tensor(rewards, device=device)
    advantages = (reward_tensor - reward_tensor.mean()) / (reward_tensor.std() + 1e-4)
    zero_advantage = len(set(rewards)) == 1

    # Stages 4-5: sequence log-probabilities and the policy-gradient loss
    # -mean(A_i * logp_i), accumulated one rollout at a time.
    pg_loss = 0.0
    if zero_advantage:
        for param in model.parameters():
            param.grad = torch.zeros_like(param)
    else:
        for (token_ids, prompt_len), advantage in zip(rollouts, advantages):
            logp = sequence_logprob(model, token_ids, prompt_len)
            loss = -(advantage.detach() * logp) / num_rollouts
            loss.backward()
            pg_loss += loss.item()

    return {
        "loss": pg_loss,
        "rewards": rewards,
        "advantages": advantages.cpu().tolist(),
        "zero_advantage": zero_advantage,
        "samples": samples,
    }
