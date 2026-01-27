# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""tiny-tpu: a tiny Grugformer run on TinyStories.

Adapted from submodules/marin/experiments/grug/base/launch.py. The grug
template owns the model and training loop; this file is just the last-mile
run config for the tiny-tpu experiment. It depends on marin as a package
(not via the marin monorepo's experiments/ directory).

Launched via:

    iris --cluster=marin job run --tpu v5litepod-16 \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -- uv run python launch.py
"""

import dataclasses
import os
from dataclasses import dataclass, field
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import (
    DatasetComponent,
    HfDatasetSourceConfig,
    LmDataConfig,
    TextLmDatasetFormat,
)
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

# Local (sibling) modules.
from model import GrugModelConfig
from train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug

MARIN_TOKENIZER = "marin-community/marin-tokenizer"


@dataclass(frozen=True)
class TinyTpuLaunchConfig:
    """Last-mile run config for the tiny-tpu trial.

    Mirrors GrugBaseLaunchConfig but kept local to this file so tiny-tpu has no
    cross-variant dataclass dependency (grug-inline: each variant owns its
    own launch config).
    """

    model: GrugModelConfig
    data: LmDataConfig
    output_path: str
    run_id: str
    resources: ResourceConfig
    steps: int
    batch_size: int
    seed: int
    mp: str  # jmp policy string, e.g. "params=float32,compute=bfloat16,output=bfloat16".
    tracker: TrackerConfig
    optimizer: OptimizerConfig
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = None  # No validation for a tiny smoke run.


# ~30M parameter Grugformer. Dimensions mirror llama_30m from
# submodules/marin/experiments/llama.py:158-165 so runs can be compared against
# train_tiny_model_tpu.py outputs.
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


# TinyStories as a single direct HF source. Levanter builds the tokenized cache
# on first access, so no separate marin tokenize step is needed — the grug
# trainer consumes LmDataConfig directly.
TINYSTORIES_DATA = LmDataConfig(
    tokenizer=MARIN_TOKENIZER,
    cache_dir="cache/tinystories",
    shuffle=True,
    components={
        "tinystories": DatasetComponent(
            source=HfDatasetSourceConfig(
                id="roneneldan/TinyStories",
                format=TextLmDatasetFormat(),
            ),
            format=TextLmDatasetFormat(),
        ),
    },
)


def _resolve_run_id(default: str) -> str:
    return os.environ.get("GRUG_RUN_ID", default)


def _resolve_tracker(tracker: TrackerConfig, run_id: str) -> TrackerConfig:
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(tracker, name=run_id)
    return tracker


def run_tiny_tpu_trial(config: TinyTpuLaunchConfig) -> None:
    # Map template launch knobs onto a full Levanter TrainerConfig, then hand
    # off to grug's run_grug (which dispatches through fray to TPU workers).
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        profiler=ProfilerConfig(enabled=False, start_step=5, num_steps=100, perfetto_link=False),
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id),
        use_explicit_mesh_axes=True,
        require_accelerator=True,
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


RESOLVED_RUN_ID = _resolve_run_id("tiny-tpu-tinystories-30m")


tiny_tpu_trial = ExecutorStep(
    name="tiny-tpu/tinystories-30m",
    fn=run_tiny_tpu_trial,
    config=TinyTpuLaunchConfig(
        model=versioned(TINY_MODEL),
        data=TINYSTORIES_DATA,
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        # v6e-4: smallest v6e pod, plentiful on hai-gcp-models in europe-west4-a
        # / us-east1-d / us-east5-b. An explicit regions= list is required
        # because the iris client auto-inherits the parent coordinator's region
        # (us-central1) unless the child sets its own region constraint —
        # and us-central1 has no v6e-4 groups.
        resources=versioned(
            ResourceConfig.with_tpu(
                "v6e-4",
                slice_count=1,
                cpu=4,
                ram="16g",
                disk="20g",
                regions=["europe-west4"],
            )
        ),
        steps=versioned(2_000),
        batch_size=versioned(128),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "tiny-tpu", "tinystories"],
            group="tiny-tpu-tinystories",
            name=None,  # filled from run_id in _resolve_tracker
            replicate_path=this_output_path(),
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
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[tiny_tpu_trial],
        description="tiny-tpu: ~30M Grugformer on TinyStories for 2000 steps.",
    )
