---
paths:
  - "LICENSES/**"
  - "src/llm_training/model/gpt2/**"
  - "src/llm_training/model/qwen3/**"
  - "src/llm_training/data/dataset.py"
  - "src/llm_training/data/instructions.py"
  - "src/llm_training/lm/generation.py"
  - "src/llm_training/lm/loss.py"
  - "src/llm_training/lm/monitoring.py"
  - "src/llm_training/reasoning/distill.py"
  - "src/llm_training/reasoning/grading.py"
  - "src/llm_training/reasoning/grpo.py"
  - "src/llm_training/reasoning/trainer.py"
  - "src/llm_training/upstream-provenance.json"
  - "docs/upstream-provenance.md"
  - "tests/reference/**"
---

# Upstream provenance

- Derived files keep Raschka's headers and are marked as modified.
- When derived code or its mapping changes, update `docs/upstream-provenance.md` and
  `src/llm_training/upstream-provenance.json`. The JSON ships in the wheel and is copied into
  each run's provenance.
- Each upstream repository's license is in `LICENSES/`, unchanged:
  `LLMs-from-scratch.txt` (Apache 2.0 with the books excluded from "Source") and
  `reasoning-from-scratch.txt` (plain Apache 2.0). A derived file's header names its file, as
  `(see LICENSES/<repository>.txt)`. A new upstream repository adds its license there.
