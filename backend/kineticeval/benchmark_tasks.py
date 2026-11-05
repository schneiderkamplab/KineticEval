"""Generated-answer QA benchmark definitions derived from Inspect Evals.

All three use zero-shot prompting, no requested chain of thought, and a uniform
64-token output budget. MMLU's model-dependent token-budget wrapper is replaced
with the equivalent built-in multiple_choice solver at that fixed budget.
"""


def winogrande_qa():
    from inspect_evals.winogrande import winogrande
    return winogrande(fewshot=0, shuffle=False)


def hellaswag_qa():
    from inspect_evals.hellaswag import hellaswag
    from inspect_ai.model import GenerateConfig
    task = hellaswag(shuffle=False, split='validation')
    task.config = GenerateConfig(temperature=0, max_tokens=64)
    return task


def mmlu_qa():
    from inspect_evals.mmlu import mmlu_0_shot
    from inspect_ai.solver import multiple_choice
    task = mmlu_0_shot(shuffle=False, cot=False, max_non_cot_tokens=64)
    task.solver = multiple_choice(cot=False, max_tokens=64)
    return task
