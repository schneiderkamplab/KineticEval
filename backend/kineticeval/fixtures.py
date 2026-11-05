"""Small deterministic fixture; never presented as a real model benchmark."""
def demo_definition():
    return {
        'name': 'QA diagnostic laboratory',
        'description': 'Small fixture for testing punctuation, semantic answers, and genuine errors.',
        'expected_format': 'letter',
        'prompt_template': '{input}\n{choices}',
        'examples': [
            {'id': 'capital', 'input': 'Which city is the capital of France?', 'target': 'A', 'choices': ['Paris', 'Berlin', 'Rome', 'Madrid'], 'subset': 'geography'},
            {'id': 'sum', 'input': 'What is 2 + 2?', 'target': 'A', 'choices': ['4', '3', '5', '6'], 'subset': 'arithmetic'},
            {'id': 'planet', 'input': 'Which planet is known as the Red Planet?', 'target': 'B', 'choices': ['Venus', 'Mars', 'Earth', 'Jupiter'], 'subset': 'science'},
        ],
    }


def inspect_fixture():
    from inspect_ai import Task
    from inspect_ai.dataset import Sample
    from inspect_ai.solver import generate, prompt_template
    from inspect_ai.scorer import exact
    return Task(dataset=[Sample(id='one', input='What is 2 + 2?', target='4')],
        solver=[prompt_template('{prompt}\nAnswer briefly.'), generate()], scorer=exact())


def inspect_mcq_fixture():
    from inspect_ai import Task
    from inspect_ai.dataset import Sample
    from inspect_ai.solver import multiple_choice
    from inspect_ai.scorer import choice
    return Task(dataset=[Sample(id='mcq', input='Which is two?', choices=['one', 'two'], target='B')],
        solver=multiple_choice(), scorer=choice())
