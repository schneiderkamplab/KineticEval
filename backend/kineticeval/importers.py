"""Materialize data at import; runtime execution never re-fetches datasets."""
import json
from .schemas import Definition, Example


def from_rows(name, rows, mapping=None, source=None):
    mapping = mapping or {'input': 'input', 'target': 'target', 'id': 'id'}
    examples = []
    for i, row in enumerate(rows):
        if mapping['input'] not in row or mapping['target'] not in row:
            raise ValueError('Input or target field missing from dataset row')
        target = str(row[mapping['target']])
        if mapping.get('target_transform') == 'gsm8k':
            if '####' not in target:
                raise ValueError('GSM8K reference missing final-answer delimiter')
            target = target.rsplit('####', 1)[1].strip()
        examples.append(Example(id=str(row.get(mapping.get('id', 'id'), i)), input=str(row[mapping['input']]),
            target=target, subset=str(row.get(mapping.get('subset', 'subset'), 'default')),
            choices=row.get(mapping.get('choices', 'choices'), []), metadata={'source_row': row}))
    return Definition(name=name, examples=examples, source=source or {'kind': 'inline'})


def from_jsonl(name, content, mapping=None):
    return from_rows(name, [json.loads(line) for line in content.splitlines() if line.strip()], mapping, {'kind': 'jsonl'})


def from_huggingface(path, name=None, split='test', revision=None, limit=20, mapping=None):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    sha = HfApi().dataset_info(path, revision=revision).sha
    dataset = load_dataset(path, name, split=split, revision=sha)
    rows = [dataset[i] for i in range(min(limit, len(dataset)))]
    return from_rows(path, rows, mapping, {'kind': 'huggingface', 'path': path, 'config': name, 'split': split,
        'revision': sha, 'limit': limit, 'fingerprint': dataset._fingerprint, 'mapping': mapping})
