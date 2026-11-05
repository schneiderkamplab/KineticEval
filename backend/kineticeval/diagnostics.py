"""Conservative answer extraction; unresolved is distinct from wrong."""
import re


def diagnose(response, target):
    text = response.strip()
    normalized = text.rstrip('.!').strip().casefold()
    expected = target.strip().casefold()
    extracted = None
    if normalized in {'a', 'b', 'c', 'd', 'yes', 'no'}:
        extracted = normalized
    match = re.fullmatch(r'(?:the answer is|det rigtige svar er)\s+([ABCD])[.!]?', text, re.I)
    if match:
        extracted = match[1].casefold()
    semantic = None if extracted is None else extracted == expected
    exact = response == target
    return {'original_score': int(exact), 'semantic_correct': semantic,
            'extracted_answer': extracted, 'format_compliance': None,
            'instruction_adequacy': None,
            'label': 'success' if exact else 'scoring too harsh' if semantic and normalized == expected else 'semantically correct' if semantic else 'wrong answer' if semantic is False else 'other',
            'heuristic_version': '1'}
