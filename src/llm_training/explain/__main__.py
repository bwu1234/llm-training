"""`python -m llm_training.explain`: regenerate docs/architecture/spec.js."""

from .spec import REPO, SPEC, write_spec

spec = write_spec()
for preset in spec["presets"]:
    print(f"{preset['id']:<24} {preset['total']:>13,} parameters")
print(f"wrote {SPEC.relative_to(REPO)}")
