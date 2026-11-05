"""MC9 missing tasks: OpenBookQA and SocialIQa.

These two MCQ benchmarks are part of the FlexOlmo MC9 aggregate but are not
available in the installed inspect_evals version.  They follow the same
pattern as the upstream inspect_evals MCQ tasks (zero-shot, ``multiple_choice``
solver, ``choice`` scorer).
"""

from __future__ import annotations

import random
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from inspect_evals.utils.huggingface import hf_dataset, load_dataset

from ._sharding import shard_sequence

# ---------------------------------------------------------------------------
# OpenBookQA
# ---------------------------------------------------------------------------

OPENBOOKQA_DATASET_PATH = "allenai/openbookqa"
OPENBOOKQA_REVISION = "388097ea7776314e93a529163e0fea805b8a6454"


@task(name="openbookqa")
def openbookqa(shuffle: bool = True, num_shards: int = 1, shard_index: int = 0) -> Task:
    """OpenBookQA elementary science MCQ benchmark (500 test samples)."""
    dataset = hf_dataset(
        path=OPENBOOKQA_DATASET_PATH,
        name="main",
        split="test",
        sample_fields=_openbookqa_record_to_sample,
        shuffle=shuffle,
        revision=OPENBOOKQA_REVISION,
    )
    if num_shards > 1:
        dataset = MemoryDataset(
            samples=shard_sequence(dataset, num_shards=num_shards, shard_index=shard_index),
            name=f"openbookqa-shard-{shard_index}-of-{num_shards}",
            location=OPENBOOKQA_DATASET_PATH,
        )
    return Task(
        dataset=dataset,
        solver=multiple_choice(),
        scorer=choice(),
    )


def _openbookqa_record_to_sample(record: dict[str, Any]) -> Sample:
    choices = record["choices"]
    labels = choices["label"]
    texts = choices["text"]
    answer_key = record["answerKey"].strip()
    target_index = labels.index(answer_key)
    target = chr(ord("A") + target_index)
    return Sample(
        id=record["id"],
        input=record["question_stem"],
        choices=texts,
        target=target,
    )


# ---------------------------------------------------------------------------
# SocialIQa
# ---------------------------------------------------------------------------

SIQA_DATASET_PATH = "lighteval/siqa"
SIQA_REVISION = "54c6a1f8cb6daf4f5abf24a601852612fb35eb25"


@task(name="socialiqa")
def socialiqa(shuffle: bool = True, num_shards: int = 1, shard_index: int = 0) -> Task:
    """SocialIQa social commonsense reasoning MCQ benchmark (1,954 validation samples).

    Uses ``lighteval/siqa`` (parquet conversion) because ``allenai/social_i_qa``
    ships a deprecated dataset-loading script that is incompatible with modern
    ``datasets`` versions.
    """
    raw = load_dataset(SIQA_DATASET_PATH, split="validation", revision=SIQA_REVISION)
    samples: list[Sample] = []
    for idx, record in enumerate(raw):
        samples.append(_siqa_record_to_sample(record, idx))
    if shuffle:
        random.Random(4242).shuffle(samples)
    if num_shards > 1:
        samples = shard_sequence(samples, num_shards=num_shards, shard_index=shard_index)
    dataset = MemoryDataset(
        samples=samples,
        name=f"socialiqa-shard-{shard_index}-of-{num_shards}",
        location=SIQA_DATASET_PATH,
        shuffled=shuffle,
    )
    return Task(
        dataset=dataset,
        solver=multiple_choice(),
        scorer=choice(),
    )


def _siqa_record_to_sample(record: dict[str, Any], idx: int = 0) -> Sample:
    context = record["context"]
    question = record["question"]
    choices = [record["answerA"], record["answerB"], record["answerC"]]
    label = int(record["label"])
    target = chr(ord("A") + label - 1)
    prompt = f"Q: {context} {question}\nA:"
    return Sample(
        id=f"siqa_{idx}",
        input=prompt,
        choices=choices,
        target=target,
    )
