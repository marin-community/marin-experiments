# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage 3: train a byte-level BPE tokenizer over the Mimi-encoded corpus.

Two-phase step:

  (1) Zephyr corpus build. We re-read the stage-2 parquet shards and rewrite
      them as a flat ``text`` column whose values look like::

          <|bos|> A_123 A_456 ... A_789 <|sep|> the quick brown fox <|eos|>

      Materializing this text now lets stage 4 consume the corpus with a
      stock ``marin.processing.tokenize`` step using ``TextLmDatasetFormat``.

  (2) In-process BPE training. The orchestrator iterates the corpus shards
      it just wrote and trains a ``ByteLevelBPETokenizer`` over them, then
      wraps the result as a ``PreTrainedTokenizerFast`` and ``save_pretrained``s
      to ``output_path`` so ``AutoTokenizer.from_pretrained(<path>)`` works.

The audio-token prefix (``A_`` by default) keeps the vocabulary of audio IDs
disjoint from the transcript's byte-level BPE merges — ByteLevelBPE will treat
``A_123`` as just another token string and the merges it learns there are
independent of English subwords.
"""

import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass

import fsspec
import pyarrow.parquet as pq
from fray.cluster import ResourceConfig
from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from zephyr import Dataset, ZephyrContext
from zephyr.readers import load_parquet

# Corpus rows omit explicit <|bos|>/<|eos|> — marin's tokenize step auto-prepends
# the tokenizer's registered bos_token and appends eos_token at doc boundaries,
# and we register both below as single-id special tokens. <|sep|> is inserted
# explicitly because it's a mid-sequence separator, not a doc boundary.

_PREFIX_KEY = "audio_token_prefix"
_SEP_KEY = "sep_token"


@dataclass(frozen=True)
class BpeTrainConfig:
    input_glob: str
    output_path: str
    resources: ResourceConfig
    max_workers: int
    vocab_size: int
    min_frequency: int
    audio_token_prefix: str
    sep_token: str
    pad_token: str
    bos_token: str
    eos_token: str


def run_bpe_training(config: BpeTrainConfig) -> None:
    """Build the BPE corpus via zephyr, then train the tokenizer in-process."""
    _build_corpus(config)
    _train_and_save_tokenizer(config)


def _build_corpus(config: BpeTrainConfig) -> None:
    ctx = ZephyrContext(
        max_workers=config.max_workers,
        resources=config.resources,
        name="bpe-corpus",
    )
    ctx.put(_PREFIX_KEY, config.audio_token_prefix)
    ctx.put(_SEP_KEY, config.sep_token)

    ds = (
        Dataset.from_files(config.input_glob)
        .flat_map(load_parquet)
        .map(_format_corpus_line)
        .write_parquet(f"{config.output_path}/corpus-{{shard:05d}}-of-{{total:05d}}.parquet")
    )
    ctx.execute(ds)


def _format_corpus_line(row: dict) -> dict:
    """Turn a stage-2 row into a single ``text`` string.

    Heavy state (worker-shared prefixes/special tokens) is fetched lazily per
    row — this is cheap because ``zephyr_worker_ctx().get_shared`` is memoized.
    """
    from zephyr import zephyr_worker_ctx

    worker = zephyr_worker_ctx()
    prefix: str = worker.get_shared(_PREFIX_KEY)
    sep: str = worker.get_shared(_SEP_KEY)

    audio_part = " ".join(f"{prefix}{t}" for t in row["mimi_tokens"])
    transcript = (row.get("text") or "").strip().lower()
    return {"text": f"{audio_part} {sep} {transcript}"}


def _train_and_save_tokenizer(config: BpeTrainConfig) -> None:
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        _iter_corpus_text(config.output_path),
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=[config.pad_token, config.bos_token, config.eos_token, config.sep_token],
    )

    # The ByteLevelBPETokenizer → PreTrainedTokenizerFast bridge via
    # ``tokenizer_object=`` occasionally drops the byte-level pre-tokenizer
    # metadata on round-trip. Going through a temp ``tokenizer.json`` is the
    # HF-recommended path and is stable across tokenizers>=0.20 / transformers>=4.47.
    with tempfile.TemporaryDirectory() as tmp:
        tokenizer_file = os.path.join(tmp, "tokenizer.json")
        tokenizer.save(tokenizer_file)
        hf_tok = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_file,
            pad_token=config.pad_token,
            bos_token=config.bos_token,
            eos_token=config.eos_token,
            sep_token=config.sep_token,
        )

    hf_tok.save_pretrained(config.output_path)
    _verify_roundtrip(config)


def _iter_corpus_text(output_path: str) -> Iterator[str]:
    """Stream the ``text`` column of every corpus parquet shard.

    Uses ``fsspec`` so both local paths and ``gs://`` URIs work; zephyr already
    declares gcsfs as a transitive dep.
    """
    fs, fs_path = fsspec.core.url_to_fs(output_path)
    shard_paths = sorted(fs.glob(f"{fs_path.rstrip('/')}/corpus-*.parquet"))
    if not shard_paths:
        raise RuntimeError(f"No corpus shards found under {output_path}; stage 1 of BPE produced nothing.")
    for path in shard_paths:
        with fs.open(path, "rb") as handle:
            pqf = pq.ParquetFile(handle)
            for group_idx in range(pqf.num_row_groups):
                table = pqf.read_row_group(group_idx, columns=["text"])
                for value in table.column("text").to_pylist():
                    if value:
                        yield value


def _verify_roundtrip(config: BpeTrainConfig) -> None:
    """Load the saved tokenizer back via AutoTokenizer and encode a sample line.

    We deliberately use ``AutoTokenizer`` (not the ``hf_tok`` instance we just
    saved) so a broken ``tokenizer_config.json`` or missing special-tokens map
    surfaces here rather than at stage-4 launch time.
    """
    loaded = AutoTokenizer.from_pretrained(config.output_path)
    sample = f"{config.audio_token_prefix}1 {config.audio_token_prefix}2 {config.sep_token} hello world"
    ids = loaded.encode(sample)
    if not ids:
        raise RuntimeError("Saved tokenizer produced empty encoding for sample line.")
    decoded = loaded.decode(ids)
    if not decoded.strip():
        raise RuntimeError("Saved tokenizer produced empty decoding for sample line.")
