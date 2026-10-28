# Add a model family

The next-token trainer, `prepare`, `predict`, `generate` and `respond` never branch on a family
name. Each family describes itself once, as a `Family` (`model/family.py`) in
`model/<name>/family.py`: its recipe settings and their validation, how to build the model and
load weights into it, its tokenizer, its pinned releases, and any extra package versions resume
must match. To add one, write `model/<name>/` with that file, then add the name to `FAMILIES`
in `model/registry.py` and its releases to `PRETRAINED`. `tests/unit/test_families.py` checks
the declaration, and `tests/unit/test_fingerprint_scope.py` checks that the resume fingerprint
covers the family's code and nothing from another family. GRPO (`reasoning/`) is built on
Qwen3 directly and is not covered by this. To show the new family in the
[architecture explorer](../architecture-explorer.md), add a preset and notes for its module
classes to `docs/architecture/explainer.json`.
