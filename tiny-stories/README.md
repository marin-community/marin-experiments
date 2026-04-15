# tiny-stories

Download [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), tokenize it with `marin-community/marin-tokenizer`, train a ~30M-param Grug decoder-only transformer.

See the [repo root README](../README.md) for the getting-started workflow (copy, adapt, smoke-test, scale up). This file documents what's specific to the tiny-stories template.

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

CPU caps `sample_count=1000` per shard; TPU/GPU tokenize everything. Swap the tokenizer by changing `MARIN_TOKENIZER`.

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
