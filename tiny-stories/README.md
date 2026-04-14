# tiny-stories

A minimal, end-to-end marin experiment: download [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) from HuggingFace, tokenize it, and train a ~30M-param Grug decoder-only transformer. It exercises the full marin pipeline (HF download → tokenize → train) in one file.

**Copy this directory as the skeleton for your own experiment.** Marin is pulled in as a library via `find-links` wheels in `pyproject.toml` — no submodule, no vendoring.

## Run on the shared marin cluster (TPU)

```
uv run iris --cluster=marin job run python launch.py --region=europe-west4
```

`--cluster=marin` targets the shared marin coordinator. `--region` is required: TPU availability is region-scoped, and the child job otherwise inherits the coordinator's region (`us-central1`), which has no `v6e-4` capacity. The default slice is `v6e-4` in `europe-west4`, 2000 steps, `bfloat16` compute.

## Local smoke test (CPU)

Start a local iris cluster in one terminal:

```
iris --cluster=local cluster start --local
```

Submit the pipeline:

```
ACCELERATOR=cpu MARIN_PREFIX=/tmp/marin \
    uv run iris --config=submodules/marin/lib/iris/examples/local.yaml \
    job run -- python launch.py
```

Tokenizes the first ~1k records per shard and trains for 1 step — enough to prove download → tokenize → train → checkpoint works. ~30s once wheels are cached.

You can bypass iris entirely and just run the executor:

```
ACCELERATOR=cpu MARIN_PREFIX=/tmp/marin uv run python launch.py
```

## Pipeline stages

`launch.py` wires three `ExecutorStep`s. Each step's config is hashed (via `versioned(...)`) so outputs cache on content; `output_path_of(...)` and `this_output_path()` thread paths between steps without hardcoding them.

### 1. `tinystories_download` — raw HF parquet

```python
tinystories_download = download_hf_step(
    "raw/tinystories",
    hf_dataset_id=TINYSTORIES_HF_ID,
    revision=TINYSTORIES_HF_REVISION,
    hf_urls_glob=["data/*.parquet"],
).as_executor_step()
```

Pinned to an HF revision so caching is deterministic. Swap `TINYSTORIES_HF_ID` / `TINYSTORIES_HF_REVISION` for a different dataset; refresh the revision with `curl -s https://huggingface.co/api/datasets/<id> | jq -r .sha`.

### 2. `tinystories_tokenized` — Levanter cache

```python
tinystories_tokenized = ExecutorStep(
    name="tokenized/tinystories",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[output_path_of(tinystories_download, "data/train-*.parquet")],
        validation_paths=[output_path_of(tinystories_download, "data/validation-*.parquet")],
        cache_path=this_output_path(),
        tokenizer=versioned(MARIN_TOKENIZER),
        format=TextLmDatasetFormat(),
        sample_count=versioned(1_000 if _accelerator() == "cpu" else None),
    ),
)
```

Uses `marin-community/marin-tokenizer`. CPU caps `sample_count=1000` per shard; TPU/GPU tokenize everything. Swap the tokenizer by changing `MARIN_TOKENIZER`.

### 3. `tiny_stories_trial` — training

`run_tiny_stories_trial` wraps Levanter's `TrainerConfig` with a `GrugRunConfig` (model, data, resources, optimizer) and calls `run_grug`. Resize the model via `TINY_MODEL` (`hidden_dim`, `num_layers`, `num_heads`, `max_seq_len`). Change accelerator/resources via `_resolve_resources()`.

## Environment variables

| Var | Default | Meaning |
| --- | --- | --- |
| `ACCELERATOR` | `tpu` | `tpu` / `gpu` / `cpu` — picks resources, batch size, precision, sample cap. |
| `MARIN_PREFIX` | `/tmp/marin` | Pipeline output root (GCS on cluster, local dir otherwise). |
| `TINY_STORIES_STEPS` | `2000` (`1` on CPU) | Training step count. |
| `GRUG_RUN_ID` | `tiny-stories-30m-$ACCELERATOR` | W&B / run ID. |
| `WANDB_API_KEY` | unset | Optional. W&B is disabled on CPU regardless. |

## Files

- `launch.py` — pipeline definition, `TINY_MODEL`, `executor_main` entrypoint.
- `model.py` — `GrugModelConfig`.
- `train.py` — `run_grug`, `GrugRunConfig`, `GrugTrainerConfig`, `GrugEvalConfig`.
- `pyproject.toml` — marin wheel pins and `find-links` URLs.
