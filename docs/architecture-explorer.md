# Architecture explorer

[`docs/architecture/index.html`](architecture/index.html) puts two presets side by side,
such as GPT-2 small and Qwen3-0.6B. Each row is something that runs in the forward pass, in the
order it runs, with its output shape (`B` batch, `T` tokens):

- a module, with its parameter count;
- a call to one of the package's functions, such as RoPE's `apply_rope`;
- a tensor operation that the repo's own code performs: residual additions, the head split,
  the causal mask, softmax, the SwiGLU product, each step of RoPE and of the norms.
  Consecutive operations that do one thing share a row, which opens to them: LayerNorm's
  "Normalize" is `sub`, `add`, `sqrt` and `div`.

Every row is named for what it does: "QK-norm, keys", "Residual connection", "Split queries
into heads", "Rotate each channel pair". Buffers such as RoPE's `cos` and `sin` tables are
listed under the module that stores them, with links to the steps that read them. Selecting a
row shows what it does and its source, with the line it runs highlighted, and its data flow:
where each input came from and what reads its output, marked in the tree. That shows, for
example, that Qwen3 adds nothing to its token embedding: the causal mask and the RoPE tables
go to every block as separate arguments. A tensor that a preset builds or stores but never
computes with is tagged "not read here": under `attn_impl: sdpa`, `is_causal=True` masks
inside the fused kernel, so GPT-2's `mask` buffer and Qwen3's built mask go unread. A
selected module also highlights the module at the same path in the other model. The
`attn_impl: explicit` presets show attention's scores, mask, scaling and softmax as separate
steps; with `sdpa` they are one fused call.

It is published at <https://bwu1234.github.io/llm-training/>. To view a local copy, open the file in a browser; it needs
no server. Locally its source links open the files in your checkout. On the published page
they open the files on GitHub at the commit it was built from, so line numbers match.
The `pages` job in `.github/workflows/ci.yml` deploys it after every push to `main` once the
checks pass, so a stale `spec.js` can't be published. It publishes `index.html`,
`spec.js` and `LICENSES/`, plus a one-line `site.js` that sets the GitHub link root. Nothing
else from `docs/` goes online.

The page's data, `docs/architecture/spec.js`, is generated from the models themselves:

```bash
uv run python -m llm_training.explain
```

`llm_training.explain` builds each preset listed in `docs/architecture/explainer.json` on
PyTorch's `meta` device (shapes without storage, so Qwen3 needs no 2.4 GB allocation). It runs
one forward pass and records, with input and output shapes:

- module calls, with module hooks;
- calls to the package's own functions, with Python's `sys.monitoring`;
- every tensor operation the package's code performs, with PyTorch's `TorchFunctionMode`.
  Operations inside a PyTorch module such as `nn.Linear` belong to that module's row;
- for each of those, which earlier node produced each tensor it reads.

A `TorchDispatchMode` sees every kernel that runs and fails the generation if one ran outside
both a recorded operation and a PyTorch module, so a step can't be dropped silently. A second
pass with a different batch and length names the dimensions that changed. The prose for each
module class, function, operation and buffer, and the name of each module and source line,
live in `explainer.json`, so `src/` holds no family-specific text. `tests/unit/test_explain.py` fails when:

- `spec.js` is stale, for example after an edit to a model's `forward`;
- a documented parameter count changes;
- a module class, function, operation or buffer has no note, a module or step has no name,
  or a note or name refers to something that no longer runs;
- folding the repeated blocks loses or adds a step.

The explainer is outside every resume fingerprint, so changing it never affects
`train --resume`.
