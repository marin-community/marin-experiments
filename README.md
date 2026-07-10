# marin-experiments

Copy-paste templates for running [Marin](https://github.com/marin-community/marin) pipelines as standalone experiments. Each template is a self-contained directory — marin is pulled in as a library via PyPI (`marin-core`, `marin-levanter`, … as nightly `0.2.x.dev` wheels), no submodule, no vendoring.

## Getting started

### 1. Pick a template

| Template | Input | Pipeline |
| --- | --- | --- |
| [`tiny-stories/`](tiny-stories/) | HF text dataset | download → tokenize → train |
| [`speech-asr/`](speech-asr/) | HF audio dataset | download → Mimi-encode → train BPE → tokenize → train |

Start with `tiny-stories/` if your data is text. Start with `speech-asr/` if you need a pre-tokenization stage (audio, images, anything that needs to become discrete tokens before training).

### 2. Copy the directory

```
cp -r tiny-stories my-experiment
cd my-experiment
```

Each template has its own `pyproject.toml` and virtual environment — nothing cross-references the source directory.

### 3. Adapt

Every template is driven by one `launch.py` that wires `ExecutorStep`s together. The per-template README walks through each stage and calls out what to change:

- **Data**: swap the HF dataset ID + revision at the top of `launch.py`.
- **Model**: resize `TINY_MODEL` / `SPEECH_MODEL` (`hidden_dim`, `num_layers`, `num_heads`, `max_seq_len`).
- **Tokenizer**: swap `MARIN_TOKENIZER`, or (for speech-asr) change the BPE vocab size / special tokens.

### 4. Run locally on CPU

Every template supports a CPU smoke test that exercises the full pipeline end-to-end on a tiny subset — enough to confirm download → tokenize → train → checkpoint works before committing compute.

```
ACCELERATOR=cpu MARIN_PREFIX=/tmp/marin uv run python launch.py
```

Finishes in under a minute for `tiny-stories`, ~3 min for `speech-asr` (Mimi on CPU dominates).

This same smoke test runs nightly in CI against the latest nightly wheels
([`canary-e2e.yml`](.github/workflows/canary-e2e.yml)); when upstream runtime
drift breaks a template, the workflow files a GitHub issue. Templates that
cannot run accelerator-free opt out with `[tool.marin-experiments]
e2e_canary = false` in their `pyproject.toml`.

### 5. Scale up on the shared marin cluster

Once the smoke test passes, submit the same `launch.py` to the shared marin TPU cluster via [iris](https://github.com/marin-community/iris):

```
uv run iris --cluster=marin job run python launch.py --region=europe-west4
```

`--cluster=marin` targets the shared coordinator. `--region` is required because TPU availability is region-scoped and the default `us-central1` has no `v6e-4` capacity.

### BYO cluster

If you don't have access to the shared marin cluster, you can run your own iris cluster — see the [iris docs](https://github.com/marin-community/iris) for setup.

## Troubleshooting

### `uv` fails to resolve or download a `marin-*` wheel

The `marin-*` packages (`marin-core`, `marin-levanter`, etc.) are published to
PyPI as nightly `0.2.x.dev` wheels. A committed `uv.lock` can go stale if new
releases have been published since it was last updated. Repin against the latest:

```
uv lock --upgrade
```

A scheduled workflow ([`repin-lockfiles.yml`](.github/workflows/repin-lockfiles.yml))
keeps the locks in this repo fresh, but if you copied a template into your own
repo a while ago you'll need to repin yourself.

## Repo layout

```
README.md            # this file
AGENTS.md            # repo-level guidance for Claude / other agents
tiny-stories/        # text template
speech-asr/          # audio template
delayed-gradient-pp/ # pipeline-parallel staleness experiment (accelerator-only)
.github/             # nightly automation: repin locks, dry-run import fixer, e2e canary
```
