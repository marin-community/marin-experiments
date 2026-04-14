# speech-asr

A minimal, end-to-end marin experiment: download [LibriSpeech](https://huggingface.co/datasets/openslr/librispeech_asr), encode audio with [kyutai/mimi](https://huggingface.co/kyutai/mimi), train a BPE tokenizer over the token stream, tokenize, and train a ~30M-param Grug decoder-only LM on `<audio_tokens> <|sep|> <transcript>` sequences. Five stages wired from one `launch.py`.

**Copy this directory as the skeleton for your own experiment.** Marin is pulled in as a library via `find-links` wheels in `pyproject.toml` — no submodule, no vendoring.

## Run on the shared marin cluster (TPU)

```
uv run iris --cluster=marin job run python launch.py --region=europe-west4
```

`--cluster=marin` targets the shared marin coordinator. `--region` is required: TPU availability is region-scoped, and the child job otherwise inherits `us-central1`, which has no `v6e-4` capacity. Mimi has no TPU kernels, so stage 2 falls back to CPU on TPU jobs; stage 5 runs on `v6e-4` in `europe-west4` at `bfloat16` compute for 2000 steps.

## Local smoke test (CPU)

Start a local iris cluster in one terminal:

```
iris --cluster=local cluster start --local
```

Submit the pipeline:

```
ACCELERATOR=cpu MARIN_PREFIX=/tmp/marin-speech \
    uv run iris --config=submodules/marin/lib/iris/examples/local.yaml \
    job run -- python launch.py
```

Caps Mimi encoding at 20 clips per shard so the PyTorch codec finishes on CPU in ~3 min, then trains for 1 step — enough to prove download → encode → BPE → tokenize → train works end-to-end.

You can bypass iris entirely and just run the executor:

```
ACCELERATOR=cpu MARIN_PREFIX=/tmp/marin-speech uv run python launch.py
```

## Pipeline stages

`launch.py` wires five `ExecutorStep`s. Each step's config is hashed (via `versioned(...)`) so outputs cache on content; `output_path_of(...)` and `this_output_path()` thread paths between steps without hardcoding them.

### 1. `librispeech_download` — raw HF parquet

```python
librispeech_download = download_hf_step(
    "raw/librispeech",
    hf_dataset_id=LIBRISPEECH_HF_ID,
    revision=LIBRISPEECH_HF_REVISION,
    hf_urls_glob=_resolve_librispeech_globs(),
).as_executor_step()
```

CPU pulls only `clean/validation/*.parquet` (~300 MB); TPU/GPU also pulls `clean/train.100`. Swap `LIBRISPEECH_HF_ID` / `LIBRISPEECH_HF_REVISION` for a different dataset.

### 2. `librispeech_audio_tokens` — Mimi neural codec

```python
librispeech_audio_tokens = ExecutorStep(
    name="audio-tokens/librispeech-mimi",
    fn=run_mimi_encode,
    config=MimiEncodeConfig(
        input_glob=output_path_of(librispeech_download, "clean/*/*.parquet"),
        ...
        max_samples=versioned(20 if _accelerator() == "cpu" else None),
        mimi_model_id=versioned(MIMI_MODEL_ID),
        num_codebooks=versioned(MIMI_NUM_CODEBOOKS),
    ),
)
```

Zephyr pipeline that runs `kyutai/mimi` over each audio shard and writes `{id, mimi_tokens, text}` parquet rows. Mimi emits 32 codebooks; we keep the first 8 via `num_quantizers=`. Token ordering is time-major: `[t0_cb0, ..., t0_cb7, t1_cb0, ...]`.

### 3. `librispeech_bpe` — corpus build + BPE training

```python
librispeech_bpe = ExecutorStep(
    name="bpe/librispeech-mimi",
    fn=run_bpe_training,
    config=BpeTrainConfig(
        input_glob=output_path_of(librispeech_audio_tokens, "mimi-*.parquet"),
        ...
        vocab_size=versioned(_BPE_VOCAB),
        audio_token_prefix=versioned("A_"),
        sep_token=versioned("<|sep|>"),
        bos_token=versioned("<|bos|>"),
        eos_token=versioned("<|eos|>"),
    ),
)
```

Two phases: a zephyr job rewrites each row as `"A_123 A_456 ... <|sep|> transcript"` parquet, then the orchestrator trains a `ByteLevelBPETokenizer` over those shards and `save_pretrained`s an HF-loadable tokenizer. `<|bos|>` / `<|eos|>` are registered specials but are **not** written into corpus lines — marin's tokenize step auto-prepends/appends them at doc boundaries. Resize via `SPEECH_BPE_VOCAB`.

### 4. `librispeech_tokenized` — Levanter cache

```python
librispeech_tokenized = ExecutorStep(
    name="tokenized/librispeech-speech",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[output_path_of(librispeech_bpe, "corpus-*.parquet")],
        validation_paths=[output_path_of(librispeech_bpe, "corpus-*.parquet")],
        cache_path=this_output_path(),
        tokenizer=output_path_of(librispeech_bpe),
        format=TextLmDatasetFormat(),
        sample_count=versioned(1_000 if _accelerator() == "cpu" else None),
    ),
)
```

Train and val point at the same glob — demo shortcut, since the CPU smoke corpus is ~20 clips. Swap in a disjoint val glob for real runs.

### 5. `speech_trial` — training

`run_speech_asr_trial` wraps Levanter's `TrainerConfig` with a `GrugRunConfig` and calls `run_grug`. `SPEECH_MODEL` is the same ~30M-param Grugformer as tiny-stories but with `max_seq_len=4096` (audio tokens dominate: ~1000 per 10s clip plus transcript) and `vocab_size` rounded up from the BPE vocab to the next multiple of 128.

## Environment variables

| Var | Default | Meaning |
| --- | --- | --- |
| `ACCELERATOR` | `tpu` | `tpu` / `gpu` / `cpu` — picks resources, batch size, precision, sample caps. |
| `MARIN_PREFIX` | `/tmp/marin` | Pipeline output root (GCS on cluster, local dir otherwise). |
| `SPEECH_ASR_STEPS` | `2000` (`1` on CPU) | Training step count. |
| `SPEECH_BPE_VOCAB` | `16384` (`2048` on CPU) | BPE vocab size; model vocab is rounded up to a multiple of 128. |
| `GRUG_RUN_ID` | `speech-asr-30m-$ACCELERATOR` | W&B / run ID. |
| `WANDB_API_KEY` | unset | Optional. W&B is disabled on CPU regardless. |

## Files

- `launch.py` — pipeline definition, `SPEECH_MODEL`, `executor_main` entrypoint.
- `audio_tokens.py` — `MimiEncodeConfig`, `run_mimi_encode` (zephyr + kyutai/mimi).
- `bpe.py` — `BpeTrainConfig`, `run_bpe_training` (corpus build + ByteLevelBPE).
- `model.py` — `GrugModelConfig`.
- `train.py` — `run_grug`, `GrugRunConfig`, `GrugTrainerConfig`, `GrugEvalConfig`.
- `pyproject.toml` — marin wheel pins and `find-links` URLs.
