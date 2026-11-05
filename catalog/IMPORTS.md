# HRM comparison catalog

The catalog now includes ten additional generated-answer evaluations from the HRM comparison inventory: BoolQ, ARC-Easy, ARC-Challenge, CommonsenseQA, PIQA, GSM8K, MMLU-Pro, OpenBookQA, SocialIQa, and PIQA-da. Their pinned Inspect task factories, arguments, scoring functions, metrics, and example hashes are recorded in the dataset and revision files. GSM8K uses the upstream default ten-shot protocol; these entries do not claim to reproduce every historical HRM run or its alternative prompts. Each new evaluation has a named `full` revision and a named `smoke` revision selecting exactly ten examples with seed 42, matching the original three evaluations. Full task examples remain stored in both; the sampler defines the selected inputs.

The model library includes comparison models from the HRM-test manifests and HRM-Text evaluation README. `comparison-model-sources.json` maps added configurations to their source. HF revisions are pinned when metadata is available. Raw comparison templates, the HRM CoT template, and thinking configurations are separate model cards. Generation budgets are editable suggestions; importing a model does not certify that every runtime supports its architecture. Local Mimir export configurations require the corresponding files on the execution server. No import launches inference.

## Compatibility gaps

- BBH's custom solver is outside the current single-generation importer.
- DROP and several Gen5 tasks use multiple references or scorers, requiring an extended scoring contract.
- Code benchmarks and agentic tasks require sandbox/tool support.
- DALA uses nonnumeric score labels, requiring an explicit metric adapter.
- Danish citizen tests could not be downloaded with the available HF access.
- Local Danish HellaSwag and other source-specific prompt variants still need portable dataset recipes and compatible task adapters; local repository paths must not become build dependencies.

These tasks are not silently replaced by approximate protocols. Further imports should preserve the original scoring behavior, including known weaknesses, so KineticEval's heuristics and inspector can assess them with traceable evidence.
