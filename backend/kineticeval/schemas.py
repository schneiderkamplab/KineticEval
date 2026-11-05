"""Public configuration contracts. Revisions contain snapshots, never secrets."""
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator


class Example(BaseModel):
    id: str
    input: str
    target: str
    subset: str = 'default'
    choices: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    messages: list[dict[str, str]] | None = None


class Component(BaseModel):
    name: str
    version: str = '1'
    config: dict[str, Any] = Field(default_factory=dict)


class Sampler(BaseModel):
    seed: int = 42
    limit: int | None = Field(default=None, ge=1)
    subsets: list[str] = Field(default_factory=list)


class InspectorConfig(BaseModel):
    enabled: bool = False
    provider: Literal['ucloud', 'fake'] = 'ucloud'
    model: str = ''
    version: str = '1'
    audit_fraction: float = Field(default=.1, ge=0, le=1)
    audit_min: int = Field(default=10, ge=0)
    audit_max: int = Field(default=100, ge=0)
    failure_inspector_limit: int | None = Field(default=None, ge=0)
    seed: int = 42
    max_tokens: int = Field(default=2048, ge=128)
    endpoint: str | None = None

    @model_validator(mode='after')
    def valid(self):
        if self.audit_min > self.audit_max:
            raise ValueError('audit_min must not exceed audit_max')
        if self.enabled and self.provider == 'ucloud' and not self.model:
            raise ValueError('Select a UCloud inspector model')
        return self


class Definition(BaseModel):
    name: str = Field(min_length=1)
    description: str = ''
    source: dict[str, Any] = Field(default_factory=lambda: {'kind': 'inline'})
    examples: list[Example] = Field(min_length=1)
    overlays: dict[str, dict[str, Any]] = Field(default_factory=dict)
    prompt_template: str = '{input}'
    system_prompt: str = ''
    expected_format: Literal['none', 'letter', 'answer_letter', 'yes_no', 'short_answer'] = 'none'
    sampler: Sampler = Field(default_factory=Sampler)
    generation_defaults: dict[str, Any] = Field(default_factory=dict)
    scorer: Component = Field(default_factory=lambda: Component(name='exact'))
    metrics: list[Component] = Field(default_factory=lambda: [Component(name='accuracy')])
    heuristics: list[Component] = Field(default_factory=lambda: [Component(name='qa')])
    inspector: InspectorConfig = Field(default_factory=InspectorConfig)

    @model_validator(mode='after')
    def validate_examples(self):
        if set(self.generation_defaults) - {'max_tokens','temperature','top_p','seed'}:
            raise ValueError('Unsupported generation default')
        ModelSpec(name='defaults', backend='fake', parameters=self.generation_defaults)
        ids = [e.id for e in self.examples]
        if len(ids) != len(set(ids)):
            raise ValueError('Example identifiers must be unique')
        for key, patch in self.overlays.items():
            if key not in ids:
                raise ValueError(f'Unknown overlay example: {key}')
            if set(patch) - {'input', 'target', 'choices', 'subset', 'metadata', 'messages'}:
                raise ValueError('Overlay may only change example data')
        for example in self.examples:
            Example.model_validate({**example.model_dump(), **self.overlays.get(example.id, {})})
        # Templates deliberately exclude the reference answer.
        import string
        fields = [field for _, field, _, _ in string.Formatter().parse(self.prompt_template) if field]
        if set(fields) - {'input', 'choices'}:
            raise ValueError('Template supports only {input} and {choices}')
        return self


class ModelSpec(BaseModel):
    model_identifier: str | None = None
    configuration_id: str | None = None
    name: str = Field(min_length=1)
    backend: Literal['inspect', 'fake', 'ucloud', 'hf'] = 'inspect'
    parameters: dict[str, Any] = Field(default_factory=dict)
    endpoint: str | None = None

    revision: str | None = None
    local_path: str | None = None
    artifact_hash: str | None = None
    dtype: Literal['bfloat16', 'float16', 'float32'] = 'bfloat16'
    context_limit: int = Field(default=4096, ge=128)
    chat_template_kwargs: dict[str, Any] = Field(default_factory=lambda: {'enable_thinking': False})

    @model_validator(mode='after')
    def no_secrets(self):
        allowed = {'temperature', 'max_tokens', 'top_p', 'seed', 'response', 'responses', 'style'}
        if set(self.parameters) - allowed:
            raise ValueError(f'Unsupported generation parameters; allowed: {sorted(allowed)}')
        if self.backend != 'fake' and set(self.parameters) & {'response', 'responses', 'style'}:
            raise ValueError('Fixture parameters are only available for fake models')
        if self.endpoint:
            from urllib.parse import urlparse
            endpoint = urlparse(self.endpoint)
            if endpoint.scheme not in {'http','https'} or not endpoint.hostname or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
                raise ValueError('Endpoint must be an HTTP(S) URL without credentials, query, or fragment')
        p = self.parameters
        if 'seed' in p and type(p['seed']) is not int:
            raise ValueError('seed must be an integer')
        if 'max_tokens' in p and (type(p['max_tokens']) is not int or p['max_tokens'] < 1):
            raise ValueError('max_tokens must be a positive integer')
        if 'temperature' in p and (type(p['temperature']) not in (int,float) or not 0 <= p['temperature'] <= 2):
            raise ValueError('temperature must be between 0 and 2')
        if 'top_p' in p and (type(p['top_p']) not in (int,float) or not 0 < p['top_p'] <= 1):
            raise ValueError('top_p must be greater than zero and at most one')
        if self.backend == 'hf' and p.get('max_tokens',1) >= self.context_limit:
            raise ValueError('Output tokens must leave room for the prompt')
        return self


class RunRequest(BaseModel):
    analysis_overrides: dict[str, Any] = Field(default_factory=dict)
    revision_id: str
    models: list[ModelSpec] = Field(min_length=1)
    example_ids: list[str] | None = None
    subsets: list[str] | None = None
    repeats: int = Field(default=1, ge=1, le=100)
    fresh: bool = False
    parent_id: str | None = None


class Finding(BaseModel):
    semantic_correct: bool | None = None
    format_compliance: bool | None = None
    instruction_adequacy: bool | None = None
    scorer_correct: bool | None = None
    explanation: str = ''
    evidence: list[str] = Field(default_factory=list)
    confidence: Literal['high', 'medium', 'low'] = 'low'
    labels: list[str] = Field(default_factory=list)


class InspectorResult(Finding):
    pass


class Suggestion(BaseModel):
    kind: Literal['overlay', 'prompt', 'scorer', 'heuristic', 'experiment']
    title: str
    rationale: str
    patch: dict[str, Any]
    example_ids: list[str] = Field(default_factory=list)


class ReportResult(BaseModel):
    summary: str
    findings: list[str] = Field(default_factory=list)
    suggestions: list[Suggestion] = Field(default_factory=list)
