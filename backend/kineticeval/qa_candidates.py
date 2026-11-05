"""Reviewable heuristic candidates derived from observed QA failure modes.

These are installed candidates, not part of the default active heuristic set.
"""
import re
from .components import qa, normalize


def qa_with_final_marker(payload, config):
    finding=qa(payload, config)
    if finding['semantic_correct'] is not None:
        return finding
    response=payload['response']
    markers=re.findall(r'(?im)^\s*(?:\*\*)?ANSWER:\s*([A-Z])(?:\*\*)?\s*$',response)
    final=re.search(r'(?i)(?:^|\n)\s*(?:\*\*)?ANSWER:\s*([A-Z])(?:\*\*)?\s*\Z',response)
    if not final or len({v.upper() for v in markers})!=1:
        return finding
    letter=final[1].upper()
    choices=payload['sample'].get('choices',[])
    if choices and ord(letter)-65 >= len(choices):
        return finding
    correct=normalize(letter)==normalize(payload['sample']['target'])
    finding.update(semantic_correct=correct, explanation='Unambiguous final ANSWER marker; conflicting explicit markers remain unresolved.',
        evidence=[final[0].strip()],confidence='high')
    finding['labels']=[label for label in finding['labels'] if label!='other']
    if not correct: finding['labels'].append('wrong answer')
    elif payload['score']['value']<1: finding['labels'].append('semantically correct')
    return finding
