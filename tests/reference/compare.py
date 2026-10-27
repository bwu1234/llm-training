"""Run a small numerical comparison against the frozen upstream definitions."""

import argparse
import json
from pathlib import Path

import torch

from llm_training.lm.loss import next_token_cross_entropy
from llm_training.model.gpt2.model import GPTModel
from tests.reference import baseline

ATOL, RTOL = 1e-6, 1e-5  # Declared before measuring.
TINY_CONFIG = dict(
    vocab_size=64,
    context_length=8,
    emb_dim=16,
    n_heads=4,
    n_layers=2,
    drop_rate=0.1,
    qkv_bias=False,
)


def compare():
    torch.set_num_threads(1)
    torch.manual_seed(123)
    reference = baseline.GPTModel(TINY_CONFIG)
    port = GPTModel(TINY_CONFIG)
    port.load_state_dict(reference.state_dict())
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    results = []
    for model, loss_fn in [(reference, baseline.calc_loss_batch), (port, next_token_cross_entropy)]:
        # Includes identical nonzero dropout draws.
        torch.manual_seed(456)
        logits = model(x)
        loss = loss_fn(x, y, model, "cpu")
        loss.backward()
        gradients = {n: p.grad.clone() for n, p in model.named_parameters()}
        torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1).step()
        results.append((logits.detach(), loss.detach(), gradients, model.state_dict()))
    a, b = results
    differences = {}
    for label, left, right in [("logits", a[0], b[0]), ("loss", a[1], b[1])]:
        torch.testing.assert_close(left, right, atol=ATOL, rtol=RTOL)
        differences[label] = (left - right).abs().max().item()
    for label, left, right in [("gradients", a[2], b[2]), ("updated_state", a[3], b[3])]:
        assert left.keys() == right.keys()
        for name in left:
            torch.testing.assert_close(left[name], right[name], atol=ATOL, rtol=RTOL)
        differences[label] = max((left[n] - right[n]).abs().max().item() for n in left)
    return {
        "scope": "tiny CPU FP32; identical weights/batches/dropout; one AdamW update",
        "python": __import__("platform").python_version(),
        "torch": str(torch.__version__),
        "atol": ATOL,
        "rtol": RTOL,
        "max_absolute_differences": differences,
        "passed": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(compare(), indent=2) + "\n"
    if args.output:
        args.output.write_text(report)
    print(report)
