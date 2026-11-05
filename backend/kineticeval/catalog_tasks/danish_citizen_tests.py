from __future__ import annotations

import math
import os
import random
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    accuracy,
    metric,
    scorer,
)
from inspect_ai.solver import TaskState, generate
from ._sharding import shard_samples

DEFAULT_DATASET_ID = "alexandrainst/danish-citizen-tests-updated"
DEFAULT_SPLIT = "test"
DEFAULT_MAX_GEN_TOKS = 8
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TEST_SIZE = 512
DEFAULT_VAL_SIZE = 64
DEFAULT_PROMPT_TEMPLATE = """Spørgsmål: {text}

Besvar ovenstående spørgsmål ved at svare med {choice_labels}, og intet andet."""

SplitName = Literal["train", "val", "test"]

_RE_FIRST_CHOICE = re.compile(r"^\s*([A-Da-d])\b")
_RE_ANY_CHOICE = re.compile(r"\b([A-Da-d])\b")
_CHOICE_LETTERS = "abcd"
_CITIZENSHIP_TEST = "indfødsretsprøven"
_PERMANENT_RESIDENCE_TEST = "medborgerskabsprøven"
_INVALID_PREDICTION_LABEL = "__invalid__"


@task(name="danish-citizen-tests")
def danish_citizen_tests(
    dataset_id: str = DEFAULT_DATASET_ID,
    split: str = DEFAULT_SPLIT,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    max_gen_toks: int = DEFAULT_MAX_GEN_TOKS,
    temperature: float = DEFAULT_TEMPERATURE,
    shuffle: bool = False,
    seed: int | None = None,
    limit: int | None = None,
    preferred_metric: str | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
) -> Task:
    # Exporters can read this from recorded task_args to override display defaults.
    _ = preferred_metric

    if max_gen_toks < 1:
        raise ValueError("`max_gen_toks` must be >= 1.")

    normalized_split = _normalize_split_name(split)
    records = _load_records(dataset_id)
    split_records = _select_split_records(records=records, split=normalized_split)
    samples = [record_to_sample(record=record, prompt_template=prompt_template) for record in split_records]

    if shuffle:
        random.Random(seed).shuffle(samples)
    if limit is not None:
        samples = samples[:limit]

    return Task(
        dataset=shard_samples(
            samples=samples,
            name="Danish Citizen Tests",
            location=dataset_id,
            num_shards=num_shards,
            shard_index=shard_index,
            shuffled=shuffle,
        ),
        solver=[generate(max_tokens=max_gen_toks, temperature=temperature)],
        scorer=danish_citizen_tests_scorer(),
    )


def _normalize_split_name(split: str) -> SplitName:
    normalized = split.strip().lower()
    match normalized:
        case "train" | "training":
            return "train"
        case "val" | "valid" | "validation" | "dev":
            return "val"
        case "test":
            return "test"
        case "":
            raise ValueError("`split` must be a non-empty string.")
        case _:
            raise ValueError(
                f"Unsupported split '{split}'. Supported values: "
                "['dev', 'test', 'train', 'training', 'val', 'valid', 'validation']"
            )


def _load_records(dataset_id: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, split="train", token=_resolve_hf_token())
    records = [_normalize_record(dict(record)) for record in dataset]
    records = _deduplicate_records(records)
    return records


def _resolve_hf_token() -> str | bool:
    if (token := os.getenv("HF_TOKEN")):
        return token
    if (token := os.getenv("HUGGINGFACE_API_KEY")):
        return token
    return False


def _normalize_record(raw_record: Mapping[str, Any]) -> dict[str, Any]:
    question = _require_string(raw_record, "question")
    answer = _normalize_answer_label(_require_string(raw_record, "answer"))

    raw_options = raw_record.get("options")
    if raw_options is None:
        raw_options = [
            raw_record.get("option_a"),
            raw_record.get("option_b"),
            raw_record.get("option_c"),
            raw_record.get("option_d"),
        ]
    if not isinstance(raw_options, Sequence) or isinstance(raw_options, (str, bytes)):
        raise ValueError("Record field 'options' must be a sequence of option strings.")

    options = [option.strip() for option in raw_options if isinstance(option, str) and option.strip()]
    if len(options) < 2:
        raise ValueError("Each record must have at least two non-empty options.")
    if len(options) > len(_CHOICE_LETTERS):
        raise ValueError(
            f"Each record may have at most {len(_CHOICE_LETTERS)} options."
        )
    if answer not in _CHOICE_LETTERS[: len(options)]:
        raise ValueError(
            f"Answer label '{answer}' does not match the number of options ({len(options)})."
        )

    test_type = _require_string(raw_record, "test_type")
    year = raw_record.get("year")
    if not isinstance(year, int):
        raise ValueError("Record field 'year' must be an integer.")

    version = _require_string(raw_record, "version")

    return {
        "question": question,
        "options": options,
        "answer": answer,
        "test_type": test_type,
        "year": year,
        "version": version,
    }


def _require_string(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Record field '{field}' must be a non-empty string.")
    return value.strip()


def _normalize_answer_label(value: str) -> str:
    cleaned = value.strip().lower()
    if cleaned not in _CHOICE_LETTERS:
        raise ValueError(
            f"Unsupported answer label '{value}'. Supported labels: {_CHOICE_LETTERS}."
        )
    return cleaned


def _clean_text(text: str) -> str:
    return text.replace("\n", " ").strip()


def _deduplicate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for record in records:
        key = (
            record["question"],
            tuple(record["options"]),
            record["answer"],
            record["test_type"],
            record["year"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _select_split_records(
    *, records: list[dict[str, Any]], split: SplitName
) -> list[dict[str, Any]]:
    citizenship_records = [
        record for record in records if record["test_type"] == _CITIZENSHIP_TEST
    ]
    permanent_records = [
        record for record in records if record["test_type"] == _PERMANENT_RESIDENCE_TEST
    ]

    remaining_permanent = list(permanent_records)
    test_records = list(citizenship_records)
    for year in sorted({record["year"] for record in remaining_permanent}, reverse=True):
        year_records = [record for record in remaining_permanent if record["year"] == year]
        test_records.extend(year_records)
        if len(test_records) >= DEFAULT_TEST_SIZE:
            break
    test_keys = {_record_key(record) for record in test_records}
    remaining_permanent = [
        record for record in remaining_permanent if _record_key(record) not in test_keys
    ]

    val_records: list[dict[str, Any]] = []
    for year in sorted({record["year"] for record in remaining_permanent}, reverse=True):
        year_records = [record for record in remaining_permanent if record["year"] == year]
        val_records.extend(year_records)
        if len(val_records) >= DEFAULT_VAL_SIZE:
            break
    val_keys = {_record_key(record) for record in val_records}
    train_records = [
        record for record in remaining_permanent if _record_key(record) not in val_keys
    ]

    split_map = {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }
    return split_map[split]


def _record_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        record["question"],
        tuple(record["options"]),
        record["answer"],
        record["test_type"],
        record["year"],
        record["version"],
    )


def record_to_sample(*, record: Mapping[str, Any], prompt_template: str) -> Sample:
    options = list(record["options"])
    labels = _CHOICE_LETTERS[: len(options)]
    option_lookup = {label: option for label, option in zip(labels, options, strict=True)}
    text = _clean_text(str(record["question"])) + "\nSvarmuligheder:\n" + "\n".join(
        f"{label}. {_clean_text(option_lookup[label])}" for label in labels
    )

    return Sample(
        input=prompt_template.format(
            text=text,
            choice_labels=", ".join(labels),
        ),
        target=record["answer"],
        metadata={
            "question": record["question"],
            "text": text,
            "options": option_lookup,
            "test_type": record["test_type"],
            "year": record["year"],
            "version": record["version"],
        },
    )


def _extract_choice(text: str, options: Mapping[str, str]) -> str | None:
    if not text:
        return None

    for pattern in (_RE_FIRST_CHOICE, _RE_ANY_CHOICE):
        match = pattern.search(text)
        if match is not None:
            choice = match.group(1).lower()
            if choice in options:
                return choice

    normalized = _normalize_for_match(text)
    matched_labels = [
        label
        for label, option in options.items()
        if option and _normalize_for_match(option) in normalized
    ]
    if len(matched_labels) == 1:
        return matched_labels[0]

    return None


def _normalize_for_match(text: str) -> str:
    return " ".join(text.lower().split())


@metric(name="mcc")
def danish_citizen_tests_mcc() -> Metric:
    def metric(scores: list[SampleScore]) -> float:
        pairs = _label_pairs(scores)
        labels = sorted(
            {
                target
                for target, _prediction in pairs
            }
            | {
                prediction
                for _target, prediction in pairs
                if prediction is not None and prediction != _INVALID_PREDICTION_LABEL
            }
        )
        return _mcc_from_pairs(pairs, labels=labels)

    return metric


@scorer(metrics=[accuracy(), danish_citizen_tests_mcc()], name="knowledge")
def danish_citizen_tests_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        options = state.metadata.get("options")
        if not isinstance(options, Mapping):
            raise ValueError("Missing option metadata for danish-citizen-tests sample.")

        option_map = {
            str(label): str(option)
            for label, option in options.items()
            if isinstance(label, str) and isinstance(option, str)
        }
        predicted = _extract_choice(state.output.completion, option_map)
        expected = _normalize_answer_label(target.text)

        return Score(
            value=CORRECT if predicted == expected else INCORRECT,
            answer=predicted or "",
            explanation=f"predicted={predicted!r}, expected={expected!r}",
            metadata={
                "prediction": predicted or _INVALID_PREDICTION_LABEL,
                "target": expected,
            },
        )

    return score


def _label_pairs(scores: list[SampleScore]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for sample_score in scores:
        metadata = sample_score.score.metadata or {}
        target = _normalize_answer_label(str(metadata.get("target") or ""))
        prediction = str(metadata.get("prediction") or _INVALID_PREDICTION_LABEL)
        pairs.append((target, prediction))
    return pairs


def _mcc_from_pairs(pairs: list[tuple[str, str]], *, labels: Sequence[str]) -> float:
    if not pairs or not labels:
        return 0.0

    full_labels = list(labels)
    if any(prediction == _INVALID_PREDICTION_LABEL for _target, prediction in pairs):
        if _INVALID_PREDICTION_LABEL not in full_labels:
            full_labels.append(_INVALID_PREDICTION_LABEL)

    index = {label: idx for idx, label in enumerate(full_labels)}
    matrix = [[0 for _ in full_labels] for _ in full_labels]

    for target, prediction in pairs:
        matrix[index[target]][index[prediction]] += 1

    total = sum(sum(row) for row in matrix)
    if total == 0:
        return 0.0

    correct = sum(matrix[i][i] for i in range(len(full_labels)))
    true_totals = [sum(row) for row in matrix]
    predicted_totals = [
        sum(matrix[row_idx][col_idx] for row_idx in range(len(full_labels)))
        for col_idx in range(len(full_labels))
    ]

    covariance = correct * total - sum(
        predicted * actual
        for predicted, actual in zip(predicted_totals, true_totals, strict=True)
    )
    pred_norm = total * total - sum(count * count for count in predicted_totals)
    true_norm = total * total - sum(count * count for count in true_totals)
    denominator = math.sqrt(pred_norm * true_norm)
    if denominator == 0.0:
        return 0.0

    return covariance / denominator
