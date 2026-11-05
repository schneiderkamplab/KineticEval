from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

from inspect_ai.dataset import MemoryDataset, Sample

T = TypeVar("T")


def shard_sequence(items: Iterable[T], *, num_shards: int = 1, shard_index: int = 0) -> list[T]:
    if num_shards < 1:
        raise ValueError("`num_shards` must be >= 1.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("`shard_index` must satisfy 0 <= shard_index < num_shards.")

    materialized = list(items)
    if num_shards == 1:
        return materialized
    return [
        item
        for index, item in enumerate(materialized)
        if index % num_shards == shard_index
    ]


def shard_samples(
    samples: Iterable[Sample],
    *,
    name: str,
    location: str,
    num_shards: int = 1,
    shard_index: int = 0,
    shuffled: bool = False,
) -> MemoryDataset:
    shard = shard_sequence(samples, num_shards=num_shards, shard_index=shard_index)
    dataset_name = name if num_shards == 1 else f"{name}-shard-{shard_index}-of-{num_shards}"
    return MemoryDataset(
        samples=shard,
        name=dataset_name,
        location=location,
        shuffled=shuffled,
    )
