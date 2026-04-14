# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Zephyr pipeline that encodes LibriSpeech audio into Mimi discrete tokens.

Input:  parquet shards containing an ``audio`` column (``{bytes, path}``) and a
        ``text`` column (the transcript).
Output: parquet shards with rows ``{id: str, mimi_tokens: list[int], text: str}``.

Token ordering is time-major: ``[t0_cb0, t0_cb1, ..., t0_cb7, t1_cb0, ...]``.
PR #2686 (marin) does not itself run Mimi — it consumes a pre-published token
dataset — so there is no upstream convention to match. Time-major keeps all
codebook IDs for a given frame contiguous, which we expect the downstream BPE
stage to exploit.

Mimi is a PyTorch model and cannot run on TPU, so we fall back to CPU whenever
``ACCELERATOR=tpu``. The launch script selects ``CpuConfig`` / ``GpuConfig``
via ``_resolve_audio_resources()``.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from zephyr import Dataset, ShardInfo, ZephyrContext, zephyr_worker_ctx
from zephyr.readers import load_parquet

MIMI_SAMPLE_RATE = 24000
_MIMI_SHARED_KEY = "mimi_model_id"


@dataclass(frozen=True)
class MimiEncodeConfig:
    input_glob: str
    output_path: str
    resources: ResourceConfig
    max_workers: int
    batch_size: int
    max_samples: int | None
    audio_column: str
    text_column: str
    mimi_model_id: str
    num_codebooks: int


def run_mimi_encode(config: MimiEncodeConfig) -> None:
    ctx = ZephyrContext(
        max_workers=config.max_workers,
        resources=config.resources,
        name="mimi-encode",
    )
    ctx.put(_MIMI_SHARED_KEY, config.mimi_model_id)
    ctx.put("audio_column", config.audio_column)
    ctx.put("text_column", config.text_column)
    ctx.put("num_codebooks", config.num_codebooks)

    ds = Dataset.from_files(config.input_glob).flat_map(load_parquet)
    if config.max_samples is not None:
        ds = ds.take_per_shard(config.max_samples)
    pipeline = ds.window(config.batch_size).map_shard(_encode_batches).write_parquet(
        f"{config.output_path}/mimi-{{shard:05d}}-of-{{total:05d}}.parquet",
    )
    ctx.execute(pipeline)


def _encode_batches(batches: Iterator[Sequence[dict]], _shard: ShardInfo) -> Iterator[dict]:
    """Encode a shard's rows through Mimi, one windowed batch at a time.

    Heavy deps (torch/transformers/soundfile/numpy) are imported locally so the
    orchestrator process never loads them — only workers pay the cost.
    """
    import io

    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoFeatureExtractor, MimiModel

    worker = zephyr_worker_ctx()
    model_id: str = worker.get_shared(_MIMI_SHARED_KEY)
    audio_col: str = worker.get_shared("audio_column")
    text_col: str = worker.get_shared("text_column")
    num_codebooks: int = worker.get_shared("num_codebooks")

    model = MimiModel.from_pretrained(model_id).eval()
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    target_sr = getattr(feature_extractor, "sampling_rate", MIMI_SAMPLE_RATE)

    for batch in batches:
        waveforms: list[np.ndarray] = []
        metas: list[tuple[str, str]] = []
        for row in batch:
            audio = row[audio_col]
            samples, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
            if samples.ndim == 2:
                samples = samples.mean(axis=1)
            if sr != target_sr:
                samples = _resample_linear(samples, sr, target_sr)
            waveforms.append(samples)
            row_id = row.get("id") or audio.get("path") or ""
            metas.append((str(row_id), row.get(text_col, "") or ""))

        inputs = feature_extractor(
            raw_audio=waveforms,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            encoded = model.encode(
                inputs["input_values"],
                padding_mask=inputs.get("padding_mask"),
                num_quantizers=num_codebooks,
            )
        # codes: [B, num_codebooks, T]
        codes = encoded.audio_codes.to(torch.int64).cpu().numpy()

        for (row_id, text), row_codes in zip(metas, codes, strict=True):
            # Time-major flatten: iterate time outermost, codebooks innermost.
            flat = row_codes.transpose(1, 0).reshape(-1).tolist()
            yield {"id": row_id, "mimi_tokens": flat, "text": text}


def _resample_linear(samples, src_sr: int, dst_sr: int):
    """Cheap linear resampler — avoids pulling in torchaudio/librosa for one call.

    Quality is fine for Mimi at LibriSpeech's 16 kHz → 24 kHz upsample: Mimi's
    own frontend smooths further. Swap in torchaudio.functional.resample if
    fidelity ever matters.
    """
    import numpy as np

    if src_sr == dst_sr:
        return samples
    duration = samples.shape[0] / src_sr
    dst_len = int(round(duration * dst_sr))
    src_t = np.linspace(0.0, duration, num=samples.shape[0], endpoint=False, dtype=np.float64)
    dst_t = np.linspace(0.0, duration, num=dst_len, endpoint=False, dtype=np.float64)
    return np.interp(dst_t, src_t, samples).astype(samples.dtype)
