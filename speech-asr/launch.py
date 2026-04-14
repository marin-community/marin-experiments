# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""speech-asr: an e2e Grugformer run on LibriSpeech audio tokens.

Pipeline (5 stages, wired incrementally):
  1. Download raw LibriSpeech audio from HuggingFace.
  2. Encode audio to discrete tokens with zephyr's Mimi neural codec.
  3. Train a BPE tokenizer over the Mimi token stream.
  4. Tokenize the encoded corpus into a Levanter cache.
  5. Train a ~30M-param Grug transformer on the tokenized cache.

Accelerator is selected via the ``ACCELERATOR`` env var (``tpu``/``gpu``/``cpu``,
default ``tpu``). Storage root is ``MARIN_PREFIX`` (falls back to ``/tmp/marin``).

Launch on the shared marin cluster:

    uv run iris --cluster=marin job run python launch.py --region=europe-west4

Launch on a local iris cluster (CPU smoke test):

    iris --cluster=local cluster start --local
    MARIN_PREFIX=/tmp/marin ACCELERATOR=cpu \\
        uv run iris --config=submodules/marin/lib/iris/examples/local.yaml \\
        job run -- python launch.py
"""

import dataclasses
import os
from dataclasses import dataclass, field
from datetime import timedelta

import jmp
from fray.cluster import CpuConfig, GpuConfig, ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDataConfig, TextLmDatasetFormat
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.datakit.download.huggingface import download_hf_step
from marin.execution.executor import (
    ExecutorStep,
    executor_main,
    output_path_of,
    this_output_path,
    versioned,
)
from marin.processing.tokenize import TokenizeConfig, lm_data_config, tokenize

from audio_tokens import MimiEncodeConfig, run_mimi_encode
from bpe import BpeTrainConfig, run_bpe_training
from model import GrugModelConfig
from train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug

# openslr/librispeech_asr carries audio bytes inline in parquet shards under
# clean/{train.100,train.360,validation,test}/*.parquet — exactly the layout
# we need (no external tar/URL fetching). The `clean/validation/0000.parquet`
# shard is small (~300 MB) and ideal for CPU smoke-tests.
LIBRISPEECH_HF_ID = "openslr/librispeech_asr"
# Pin to a specific HF revision so caching is deterministic. Refresh by
# running: curl -s https://huggingface.co/api/datasets/<id> | jq -r .sha
LIBRISPEECH_HF_REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
MIMI_MODEL_ID = "kyutai/mimi"
MIMI_NUM_CODEBOOKS = 8


def _accelerator() -> str:
    return os.environ.get("ACCELERATOR", "tpu").lower()


def _resolve_audio_resources() -> ResourceConfig:
    """Resources for stage 2 (Mimi encoding).

    Mimi is a PyTorch codec with no TPU kernels — on TPU jobs we still encode
    audio on CPU. GPU is the fast path (Mimi is ~50× realtime on an L4).
    """
    kind = _accelerator()
    if kind == "gpu":
        return ResourceConfig(cpu=4, ram="32g", disk="40g", device=GpuConfig(variant="l4", count=1))
    return ResourceConfig(cpu=4, ram="16g", disk="20g", device=CpuConfig())


def _resolve_bpe_resources() -> ResourceConfig:
    """Resources for stage 3 (BPE corpus + training).

    BPE training is pure CPU — tokenizers' Rust core doesn't benefit from
    accelerator allocations regardless of ACCELERATOR.
    """
    return ResourceConfig.with_cpu(cpu=4, ram="16g", disk="20g")


def _resolve_train_resources() -> ResourceConfig:
    """Resources for stage 5 (Grug training)."""
    kind = _accelerator()
    if kind == "tpu":
        # v6e-4 on europe-west4-a / us-east1-d / us-east5-b; must pin regions
        # explicitly because iris inherits us-central1 from the parent coordinator
        # and us-central1 has no v6e-4 groups.
        return ResourceConfig.with_tpu(
            "v6e-4",
            slice_count=1,
            cpu=4,
            ram="32g",
            disk="20g",
            regions=["europe-west4"],
        )
    if kind == "gpu":
        return ResourceConfig.with_gpu("a100", count=1, cpu=4, ram="32g", disk="40g")
    if kind == "cpu":
        return ResourceConfig.with_cpu(cpu=4, ram="16g", disk="20g")
    raise ValueError(f"Unknown ACCELERATOR={kind!r}; expected tpu|gpu|cpu")


def _resolve_vocab_size() -> int:
    # Tiny vocab on CPU so the smoke corpus (20 clips) trains in seconds;
    # full-size vocab on real runs.
    return int(os.environ.get("SPEECH_BPE_VOCAB", "2048" if _accelerator() == "cpu" else "16384"))


def _resolve_steps() -> int:
    return int(os.environ.get("SPEECH_ASR_STEPS", "1" if _accelerator() == "cpu" else "2000"))


def _resolve_batch_size() -> int:
    return 8 if _accelerator() == "cpu" else 128


def _resolve_run_id(default: str) -> str:
    return os.environ.get("GRUG_RUN_ID", default)


def _resolve_tracker(tracker: TrackerConfig, run_id: str) -> TrackerConfig:
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(tracker, name=run_id)
    return tracker


# Round the BPE vocab up to the next multiple of 128 for accelerator efficiency.
# This is baked into the model config at module import time — the BPE tokenizer's
# actual vocab is fixed by _resolve_vocab_size() (same env var) so the two agree.
_BPE_VOCAB = _resolve_vocab_size()
_MODEL_VOCAB = ((_BPE_VOCAB + 127) // 128) * 128


# ~30M-parameter Grugformer, same shape as TINY_MODEL in tiny-stories. Goal is
# pipeline exercise, not SOTA speech. max_seq_len is bumped to 4096 because audio
# tokens dominate: ~1000 Mimi tokens per 10s clip + transcript + specials.
SPEECH_MODEL = GrugModelConfig(
    vocab_size=_MODEL_VOCAB,
    hidden_dim=128,
    intermediate_dim=448,
    num_layers=4,
    num_heads=2,
    num_kv_heads=2,
    max_seq_len=4096,
    head_dim=None,
)


@dataclass(frozen=True)
class SpeechAsrLaunchConfig:
    """Last-mile run config for the speech-asr trial."""

    model: GrugModelConfig
    data: LmDataConfig
    output_path: str
    run_id: str
    resources: ResourceConfig
    steps: int
    batch_size: int
    seed: int
    mp: str  # jmp policy string.
    tracker: TrackerConfig
    optimizer: OptimizerConfig
    require_accelerator: bool
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = None


# -- Pipeline steps ----------------------------------------------------------

# 1) Raw HF download. On CPU we pull only the validation shard (~300 MB); on
#    TPU/GPU we pull train.clean.100 + validation so stage 2 has real training
#    data. The glob is selected at module import time, baked into the step's
#    config, and pinned via the revision sha above.
_LIBRISPEECH_GLOBS_SMOKE = ["clean/validation/*.parquet"]
_LIBRISPEECH_GLOBS_FULL = ["clean/train.100/*.parquet", "clean/validation/*.parquet"]


def _resolve_librispeech_globs() -> list[str]:
    return _LIBRISPEECH_GLOBS_SMOKE if _accelerator() == "cpu" else _LIBRISPEECH_GLOBS_FULL


librispeech_download = download_hf_step(
    "raw/librispeech",
    hf_dataset_id=LIBRISPEECH_HF_ID,
    revision=LIBRISPEECH_HF_REVISION,
    hf_urls_glob=_resolve_librispeech_globs(),
).as_executor_step()


# 2) Mimi-encode audio shards into discrete tokens (parquet).
librispeech_audio_tokens = ExecutorStep(
    name="audio-tokens/librispeech-mimi",
    fn=run_mimi_encode,
    config=MimiEncodeConfig(
        input_glob=output_path_of(librispeech_download, "clean/*/*.parquet"),
        output_path=this_output_path(),
        resources=versioned(_resolve_audio_resources()),
        max_workers=versioned(1 if _accelerator() == "cpu" else 4),
        batch_size=versioned(2 if _accelerator() == "cpu" else 8),
        max_samples=versioned(20 if _accelerator() == "cpu" else None),
        audio_column=versioned("audio"),
        text_column=versioned("text"),
        mimi_model_id=versioned(MIMI_MODEL_ID),
        num_codebooks=versioned(MIMI_NUM_CODEBOOKS),
    ),
)


# 3) Train a BPE tokenizer on the Mimi token + transcript corpus.
librispeech_bpe = ExecutorStep(
    name="bpe/librispeech-mimi",
    fn=run_bpe_training,
    config=BpeTrainConfig(
        input_glob=output_path_of(librispeech_audio_tokens, "mimi-*.parquet"),
        output_path=this_output_path(),
        resources=versioned(_resolve_bpe_resources()),
        max_workers=versioned(1 if _accelerator() == "cpu" else 4),
        vocab_size=versioned(_BPE_VOCAB),
        min_frequency=versioned(2),
        audio_token_prefix=versioned("A_"),
        sep_token=versioned("<|sep|>"),
        pad_token=versioned("<|pad|>"),
        bos_token=versioned("<|bos|>"),
        eos_token=versioned("<|eos|>"),
    ),
)


# 4) Tokenize the corpus into a Levanter cache. Stage 3's corpus lines do NOT
#    carry explicit <|bos|>/<|eos|> — marin's tokenize prepends/appends the
#    tokenizer's registered bos/eos automatically, and our BPE tokenizer has
#    them registered as single-id special tokens. Train and val point at the
#    same glob: demo simplification (the CPU smoke corpus is only ~20 clips).
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


# -- Training ---------------------------------------------------------------


def run_speech_asr_trial(config: SpeechAsrLaunchConfig) -> None:
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        profiler=ProfilerConfig(enabled=False, start_step=5, num_steps=100, perfetto_link=False),
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id),
        use_explicit_mesh_axes=True,
        require_accelerator=config.require_accelerator,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=os.path.join(config.output_path, "checkpoints"),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=10),
            keep=[{"every": 1000}],
        ),
    )
    grug_trainer = dataclasses.replace(config.grug_trainer, trainer=trainer)
    run_config = GrugRunConfig(
        model=config.model,
        data=config.data,
        resources=config.resources,
        optimizer=config.optimizer,
        trainer=grug_trainer,
        eval=config.eval,
    )
    run_grug(run_config)


RESOLVED_RUN_ID = _resolve_run_id(f"speech-asr-30m-{_accelerator()}")


speech_trial = ExecutorStep(
    name="speech-asr/librispeech-30m",
    fn=run_speech_asr_trial,
    config=SpeechAsrLaunchConfig(
        model=versioned(SPEECH_MODEL),
        data=lm_data_config(librispeech_tokenized),
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(_resolve_train_resources()),
        steps=versioned(_resolve_steps()),
        batch_size=versioned(_resolve_batch_size()),
        seed=versioned(0),
        mp=versioned(
            "params=float32,compute=float32,output=float32"
            if _accelerator() == "cpu"
            else "params=float32,compute=bfloat16,output=bfloat16"
        ),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "speech-asr", "librispeech", _accelerator()],
            group="speech-asr-librispeech",
            name=None,
            replicate_path=this_output_path(),
            mode="disabled" if _accelerator() == "cpu" else "online",
        ),
        optimizer=versioned(
            AdamConfig(
                learning_rate=6e-4,
                weight_decay=0.1,
                lr_schedule="cosine",
                decay=0.2,
                min_lr_ratio=0.1,
                warmup=200,
            )
        ),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=1e-4,
                ema_beta=None,
                log_every=1,
            )
        ),
        require_accelerator=versioned(_accelerator() != "cpu"),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[librispeech_download, librispeech_audio_tokens, librispeech_bpe, librispeech_tokenized, speech_trial],
        description=f"speech-asr: LibriSpeech speech-token training ({_accelerator()}).",
    )
