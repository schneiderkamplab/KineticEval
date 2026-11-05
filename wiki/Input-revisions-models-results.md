# Input revisions, Models, and Results

This supersedes the revision and dashboard design in OKF v0.2 where scoring and inspection settings were part of revision identity.

## Evaluation inputs and defaults

An input revision identifies data provenance, examples and corrected references, sampling, and prompt construction. Editing scoring, metrics, heuristic components, expected-format analysis, the LLM inspector, or generation suggestions does not create a new input revision. These suggestions are attached to a revision and frozen into each run. Names and descriptions are display metadata. Every revision displays its creation date.

The inspector investigates responses and proposes improvements; “judge” is reserved for future judged scoring. Its assessments do not overwrite original scores. Proposed input corrections require approval; analysis proposals change future-run suggestions. Ongoing inspection-budget adjustments remain separately recorded and do not reset the selected cohort on retry.

## Models and launches

Models are named configurations of physical models, with identifiers or local exports, weight revisions, precision, context limits, generation defaults, and chat-template options. Configurations can represent multiple variants of the same physical model. Display names do not invalidate identical generation cache entries.

The shared launch dialog selects evaluation revisions and model configurations, supports overrides, and groups one run per evaluation into a Result. Defaults resolve from application/import fallbacks, revision suggestions, model configuration, and explicit model overrides. Runs pin effective settings. Batch launch is atomic and idempotent; partial reruns and approved experiment arms retain provenance and Result membership.

## Results

Results are editable collections, including collections assembled from existing runs. Removing a membership or collection does not delete runs. A model-by-evaluation matrix and per-evaluation tables show metrics, coverage, and status. Best values are bold; the second-best distinct values are underlined, including ties.

Ranking normally compares matching revisions, realized samples, repeats, scorers, and metrics. Users may explicitly override mismatch restrictions, with a visible warning. Unknown metric directions require a selection. Repeated runs stay separate, and there is no implicit overall score across unrelated evaluations. Metric bars and correct/incorrect/unscored response stacks accompany the tables; inspector coverage is separate.

## Presentation

Supported workflows have structured controls. JSON is optional under Advanced; original prompts, responses, and historical settings remain read-only evidence. Imports support file upload, field mapping, and a readable preview. Experiment arms, regression cases, proposals, and extension settings have field editors or readable summaries. Custom component settings survive editing of unrelated fields. Dialogs support keyboard navigation and local error messages.

## Migration and preservation

The live database was backed up before migration. All revision IDs were replaced and references rewritten without aliases. Eleven legacy revisions became five input revisions: one MMLU, one HellaSwag, and three Winogrande (original/full, amended prompt, and a subsequently created ten-example sample). All seventeen historical runs were retained. Five historical model configurations were seeded.

On the validation copy, all 77,045 successful generation artifacts, 77,045 scoring artifacts, and 77,045 diagnostic artifacts matched the originals. All 9,447 pinned inspection protocols validated after the terminology migration, and all seventeen runs remained readable. Protocol text and fingerprints are retained, including historical schema names. Migration markers make both migrations idempotent.

The historical Result **Original local benchmark campaign · full splits** groups the original full Winogrande, HellaSwag, and MMLU runs across Mimir, Gemma, and Qwen. Creating it queues no inference.

## Verification map

- `test_revision_identity.py`, `test_run_settings.py`: input identity, reference changes, default edits, frozen runs, precedence, and single-run collections.
- `test_input_migration.py`: consolidation, rewritten references, labels, snapshots, stable jobs, inspector storage conversion, and protocol preservation.
- `test_library.py`: named variants, shared generation cache, atomic batch launch, idempotency, and independent collection lifetime.
- `test_revision_deletion.py`: shared jobs, stale previews, in-flight claim fencing, and Result membership cleanup.
- `test_inspection_budget.py`: retained assessments and fixed selection through retries.
- Frontend unit tests: matrix ranking, ties, unknown direction, typed editing, extension preservation, and JSON round trips.
- Browser workflows: three-by-three launch without JSON, JSONL upload, model forms, suggestions, inspection, input corrections, partial reruns, experiment approval, deletion previews, dates/names, and keyboard focus.

Verification on 5 November 2026: 103 backend tests passed (one optional integration skipped), 16 frontend unit tests passed, and 12 browser workflows passed. The live historical Result rendered three model rows and three evaluation columns without JavaScript errors. No new inference was used for live verification.

## Workload preview and selection

Evaluation cards support multi-selection and launching the selected revisions together. Models use the same card layout, with generation defaults and selection controls. The launch dialog's **Preview workload** action resolves model revisions and reads the cache without creating runs or queue jobs. It distinguishes reusable, new, queued, running, and previously failed generation jobs, deduplicating shared generation work across the whole launch. Scoring and heuristic reuse are checked where cached evidence permits it.

When all scoring and diagnostic evidence is cached, the framework computes the actual audit/failure selection and checks matching inspection jobs. Otherwise it shows a bounded estimate, respecting the failure limit and successful-response audits. Potential pattern investigations and reports are displayed separately; repair attempts may add HTTP requests. Editing launch settings invalidates the preview. Launching from a preview uses its resolved model and analysis settings.

Results show per-model execution state for unfinished scores: running, queued, or waiting for earlier stages. Active runs also expose their current phase and stage counts, refreshed automatically.

## Current smoke campaign diagnostics

The ten-example campaign reused earlier Transformers responses for all WinoGrande configurations. Gemma’s 64-token configuration also reused HellaSwag and MMLU responses; its 256-token configuration needed new generations. The vLLM loader exposed an incomplete local snapshot: weights and tokenizer existed but processor metadata was missing. Metadata completion now uses the pinned revision, and only the twenty affected generation jobs were retried. This does not change prompts or generation parameters.

HRM’s HellaSwag requests contain separate system and user messages, while its tokenizer has no chat template. Those requests remain explicit failures: silently joining messages would change prompt rendering. Inspector investigation/report failures include provider timeouts and missing text responses. Original generated responses and scores remain available independently of those analysis failures.

Metrics now schedule as soon as each model finishes scoring and diagnostics, without waiting for other models. Result cells show cache reuse, per-model running/queued work, and expandable execution failures. Successful regression tests do not establish that a live GPU job has completed; live campaign recovery remains separately tracked.

The read-only live workload preview for the five-model WinoGrande smoke configuration found all fifty responses reusable in 3.04 seconds, with no run created. The live five-model Result rendered without JavaScript errors. vLLM initialization is now explicitly text-only for this text QA backend, avoiding unrelated multimodal encoder profiling on future engine loads. Provider failures now retain safe finish-reason and output-token diagnostics, without exposing reasoning content or automatically increasing token budgets.

### Remaining operational verification

- Gemma recovery produced all ten HellaSwag responses. Eight MMLU jobs encountered a separate Torch Inductor initialization error on one worker (`@jit functions should be defined in a Python file`); that failed worker was stopped and only those eight jobs retried, with other resident Gemma workers available.
- Eight HRM MMLU requests remained in generation. GPU process monitoring confirmed compute use; completion and runtime remain unverified. Do not restart a live worker solely because polling times out.
- HRM HellaSwag requires an explicit supported rendering choice for its system/user messages before a new attempt. Preserve the failed attempts and input provenance.
- GitHub push is blocked by missing checkout authentication. The local implementation commit is `fb2a2bb`; later diagnostic changes are separate.

Prompt preflight now validates each request's supported rendering and context budget before GPU model allocation. Invalid requests fail individually; they do not prevent valid requests from the same model batch from running. Tests cover mixed valid/invalid requests and preservation of the exact valid prompt. Existing model processes are not interrupted to install this change; newly started workers use it. The user will perform the GitHub push.
