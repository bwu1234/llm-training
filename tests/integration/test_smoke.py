"""Offline CLI smoke; explicitly supply prepared tokenizer assets to enable it."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


def test_offline_training_and_fresh_process_reload(tmp_path):
    cache = os.environ.get("LLM_TEST_TOKENIZER_CACHE")
    if not cache:
        pytest.skip("Set LLM_TEST_TOKENIZER_CACHE to a prepared GPT-2 cache")
    raw = yaml.safe_load(Path("configs/smoke.yaml").read_text())
    raw["data"]["path"] = str(Path("tests/fixtures/smoke.txt").resolve())
    raw["data"]["tokenizer_cache"] = str(Path(cache).resolve())
    raw["output"]["root"] = str(tmp_path / "runs")
    config = tmp_path / "smoke.yaml"
    config.write_text(yaml.safe_dump(raw))
    # Block network connection attempts in the training process, including
    # accidental tokenizer downloads. Use a fresh process for both operations.
    script = """
import runpy, socket, sys
def forbidden(*args, **kwargs):
    raise AssertionError("Network access during offline training")
socket.socket.connect = forbidden
sys.argv = ["llm-training", "train", "--config", sys.argv[1]]
runpy.run_module("llm_training", run_name="__main__")
"""
    subprocess.run([sys.executable, "-c", script, str(config)], check=True, timeout=60)
    (run,) = (tmp_path / "runs").iterdir()
    status = json.loads((run / "status.json").read_text())
    assert status["status"] == "completed"
    assert status["optimizer_steps"] == 4
    assert status["tokens_seen"] == 128
    assert len((run / "metrics.jsonl").read_text().splitlines()) == 4
    reload_script = """
import json, socket, sys, torch
from pathlib import Path
from llm_training.model.gpt2.tokenizer import load_tokenizer
from llm_training.generation import generate_greedy
from llm_training.model.gpt2.model import GPTModel
def forbidden(*args, **kwargs):
    raise AssertionError("Network access during offline reload")
socket.socket.connect = forbidden
torch.set_num_threads(1)
run = Path(sys.argv[1])
saved = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
model = GPTModel(saved["model_config"])
model.load_state_dict(saved["model_state_dict"])
model.eval()
tokenizer = load_tokenizer(Path(sys.argv[2]))
sample = json.loads((run / "sample.json").read_text())
ids = generate_greedy(model, torch.tensor([tokenizer.encode(sample["prompt"])]),
                          20, saved["model_config"]["context_length"])
assert ids[0].tolist() == sample["token_ids"]
"""
    subprocess.run([sys.executable, "-c", reload_script, str(run), cache], check=True, timeout=60)
    prediction_script = """
import runpy, socket, sys
def forbidden(*args, **kwargs):
    raise AssertionError("Network access during prediction")
socket.socket.connect = forbidden
sys.argv = ["llm-training", "predict", *sys.argv[1:]]
runpy.run_module("llm_training", run_name="__main__")
"""
    sample = json.loads((run / "sample.json").read_text())
    for index, checkpoint in enumerate((run / "model.pt", run / "checkpoints/last.pt")):
        output = tmp_path / f"predictions-{index}.json"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                prediction_script,
                "--checkpoint",
                str(checkpoint),
                "--tokenizer-cache",
                cache,
                "--prompt",
                sample["prompt"],
                "--max-new-tokens",
                "3",
                "--top-k",
                "3",
                "--output",
                str(output),
            ],
            check=True,
            timeout=60,
            capture_output=True,
            text=True,
        )
        report = json.loads(output.read_text())
        prefix = len(report["prompt_token_ids"])
        assert report["generated_token_ids"] == sample["token_ids"][prefix : prefix + 3]
        assert report["optimizer_steps"] == 4
        assert "## Step 3" in result.stdout
        assert "Other tokens:" in result.stdout
        for step in report["steps"]:
            assert len(step["candidates"]) == 3
            assert step["candidates"][0]["token_id"] == step["selected_token_id"]
            total = sum(c["probability"] for c in step["candidates"])
            assert total + step["other_probability_mass"] == pytest.approx(1)
