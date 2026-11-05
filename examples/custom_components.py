"""Example installed extensions; add this directory to PYTHONPATH when using them."""

def numeric_score(payload, config):
    """Accept numerically equal short answers, preserving parse failures as zero."""
    try:
        value = float(payload['response'].strip()) == float(payload['sample']['target'])
    except ValueError:
        value = False
    return {'value': float(value), 'explanation': 'Numeric short-answer comparison'}


def answered(payload, config):
    """Only assess a dimension supported by the available evidence."""
    return {'semantic_correct': None, 'format_compliance': None,
            'explanation': 'Nonempty text alone cannot establish correctness.',
            'evidence': [payload['response']], 'confidence': 'low'}


def mean_score(scores, config):
    return sum(row['value'] for row in scores) / len(scores)
