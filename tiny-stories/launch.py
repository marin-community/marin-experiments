# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""tiny-stories: an e2e Grugformer run on TinyStories.

Pipeline:
  1. Download raw TinyStories parquet from HuggingFace.
  2. Tokenize with the marin tokenizer into a Levanter cache.
  3. Train a ~30M-param Grug transformer on the tokenized cache.

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
from fray.cluster import ResourceConfig
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

from model import GrugModelConfig
from train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug

MARIN_TOKENIZER = "marin-community/marin-tokenizer"
TINYSTORIES_HF_ID = "roneneldan/TinyStories"
# Pin to a specific HF revision so caching is deterministic. Refresh by
# running: curl -s https://huggingface.co/api/datasets/<id> | jq -r .sha
TINYSTORIES_HF_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"


def _accelerator() -> str:
    return os.environ.get("ACCELERATOR", "tpu").lower()


def _resolve_resources() -> ResourceConfig:
    kind = _accelerator()
    if kind == "tpu":
        # v6e-4: smallest v6e pod, plentiful on hai-gcp-models in europe-west4-a
        # / us-east1-d / us-east5-b. An explicit regions= list is required
        # because the iris client auto-inherits the parent coordinator's region
        # (us-central1) unless the child sets its own region constraint —
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


def _resolve_steps() -> int:
    # CPU is a smoke test — a couple of dozen steps is enough to prove the pipe works.
    return int(os.environ.get("TINY_STORIES_STEPS", "1" if _accelerator() == "cpu" else "2000"))


def _resolve_batch_size() -> int:
    return 8 if _accelerator() == "cpu" else 128


@dataclass(frozen=True)
class TinyStoriesLaunchConfig:
    """Last-mile run config for the tiny-stories trial."""

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


# ~30M parameter Grugformer, mirrors llama_30m from marin's experiments/llama.py.
TINY_MODEL = GrugModelConfig(
    vocab_size=128_256,
    hidden_dim=128,
    intermediate_dim=448,
    num_layers=4,
    num_heads=2,
    num_kv_heads=2,
    max_seq_len=1024,
    head_dim=None,
)


# -- Pipeline steps ----------------------------------------------------------

# 1) Raw HF download — preserves the original parquet layout.
tinystories_download = download_hf_step(
    "raw/tinystories",
    hf_dataset_id=TINYSTORIES_HF_ID,
    revision=TINYSTORIES_HF_REVISION,
    hf_urls_glob=["data/*.parquet"],
).as_executor_step()


# 2) Tokenize — reads parquet from (1), writes Levanter cache shards.
tinystories_tokenized = ExecutorStep(
    name="tokenized/tinystories",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[output_path_of(tinystories_download, "data/train-*.parquet")],
        validation_paths=[output_path_of(tinystories_download, "data/validation-*.parquet")],
        cache_path=this_output_path(),
        tokenizer=versioned(MARIN_TOKENIZER),
        format=TextLmDatasetFormat(),
        # CPU smoke test caps at 1k records per shard. Full dataset on TPU/GPU.
        sample_count=versioned(1_000 if _accelerator() == "cpu" else None),
    ),
)


# -- Training ---------------------------------------------------------------


def _resolve_run_id(default: str) -> str:
    return os.environ.get("GRUG_RUN_ID", default)


def _resolve_tracker(tracker: TrackerConfig, run_id: str) -> TrackerConfig:
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(tracker, name=run_id)
    return tracker


def run_tiny_stories_trial(config: TinyStoriesLaunchConfig) -> None:
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


RESOLVED_RUN_ID = _resolve_run_id(f"tiny-stories-30m-{_accelerator()}")


tiny_stories_trial = ExecutorStep(
    name="tiny-stories/tinystories-30m",
    fn=run_tiny_stories_trial,
    config=TinyStoriesLaunchConfig(
        model=versioned(TINY_MODEL),
        data=lm_data_config(tinystories_tokenized),
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(_resolve_resources()),
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
            tags=["grug", "tiny-stories", "tinystories", _accelerator()],
            group="tiny-stories-tinystories",
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
        steps=[tinystories_download, tinystories_tokenized, tiny_stories_trial],
        description=f"tiny-stories: ~30M Grugformer on TinyStories ({_accelerator()}).",
    )
