# OKF v0.2 — KineticEval

**Status:** working design and implementation record. **Snapshot:** 4 November 2026, 20:53 UTC. This is document version 0.2; the Python package remains version 0.1.0.

KineticEval makes evaluations inspectable and improvable over time. It preserves conventional benchmark scores while explaining what those scores miss: correct answers rejected by scorers, unclear answer instructions, format problems, truncation, genuinely incorrect answers, and unresolved cases. Evaluations are living definitions with versioned data corrections, prompts, scoring, and diagnostic knowledge.

The first implementation focuses on question answering. Judged evaluations, BERTScore-based evaluations, arena comparisons, and fluid evaluations belong to the longer-term vision; their evaluation-specific workflows are not yet implemented.

## 1. Governing principles

1. **KineticEval owns evaluation intelligence.** The framework detects symptoms, selects evidence, invokes the LLM judge, stores findings, and creates proposals. A coding agent implements and tests that machinery. It must not act as the evaluator, manually commission a model-specific investigation, or present its own recommendations as judge output.
2. **Retain the original score.** Semantic assessments and diagnostic labels supplement benchmark scoring; they do not silently replace it.
3. **Use heuristics first, judges where needed.** Judge unresolved dimensions of failed responses, and audit samples of both successful and failed responses to detect errors in scorers and heuristics.
4. **Keep uncertainty explicit.** An incomplete response or ambiguous answer is not automatically wrong. Requested settings are not proof of effective model behavior.
5. **Changes require provenance.** Data overlays, prompt edits, scorers, heuristics, and generation-setting experiments create distinct revisions or runs. Keep their parents and baseline comparisons.
6. **Recommendations are not activation.** A judge may propose changes; it does not silently modify the current evaluation. New heuristic behavior requires positive regression examples and counterexamples.
7. **Reuse expensive work safely.** Compatible generations survive rescoring, diagnostic changes, retries, and partial reruns. Changed inputs or model settings change cache identity.
8. **Separate evaluation models from the judge.** In this campaign, evaluated HF models run locally. The UCloud API is used only for GLM-5.3 judging.

The contributor-facing version of these principles lives in [AGENTS.md](../AGENTS.md).

## 2. Diagnostic model

The framework distinguishes dimensions before applying explanatory labels. A response can be semantically correct while violating an explicit format instruction, or be scored incorrectly because a scorer extracted the wrong letter.

| Dimension | Question |
|---|---|
| Semantic correctness | Does the response express the correct answer in the task language? |
| Format compliance | Does it follow the answer format actually requested? |
| Instruction adequacy | Did the prompt specify the expectations used by scoring? |
| Scorer correctness | Did the scorer interpret and grade this response appropriately? |

Required diagnostic cases include:

- **Scoring too harsh:** `yes.` rejected when `yes` was expected, or a clear equivalent such as “affirmative.”
- **Semantically correct answer in another form:** “The answer is A.” when a scorer expects a bare letter.
- **Insufficient instructions:** a scorer expects A–D although the rendered prompt did not establish that contract.
- **Wrong answer:** an unambiguous D when A is correct. Do not confuse this with Danish “Det rigtige svar er A.” by extracting the initial D.
- **Other or unresolved:** conflicting answers, truncation, ambiguous references, unsupported formats, or evidence insufficient for a reliable assessment.

These categories can overlap. Partial heuristic coverage is useful: a format heuristic can resolve format compliance while leaving semantics for the judge. Judge disagreements remain visible alongside the original heuristic finding.

## 3. Architecture

### Evaluation definitions and revisions

An evaluation has tasks/examples, optional subsets, dataset provenance, a sampler, prompt templates, scoring functions, metrics, and heuristics. Runs pin immutable revisions and materialized sample manifests.

An upstream dataset remains identifiable when a user corrects an example. Store a correction overlay keyed by example ID rather than rewriting the upstream dataset. A reference-only correction permits generation reuse; changing the rendered prompt requires a new generation for the affected example.

The initial task format is single-generation QA. Inspect imports currently support text inputs, one scalar scorer, and a restricted set of supported solvers. Unsupported setup, sandbox, multi-turn, or custom solver behavior must be rejected explicitly rather than approximated silently.

### Inspect AI has three independent roles

- **Definition source:** lift supported existing evaluation definitions, materialize their examples and prompts, and retain component provenance.
- **Scoring and metrics provider:** apply Inspect scoring functions and metrics to stored generations; extend these with KineticEval or installed custom components.
- **Generation backend:** execute pending generations for models and tasks supported by Inspect.

Inspect is an adapter, not the identity of an evaluation. Local Transformers generation and other backends can share evaluation definitions and analysis stages.

### Durable generation cache and work queue

The generation cache records completed results and pending work, grouped by model for efficient execution. Identity includes rendered messages, model/backend configuration, generation parameters, repeat identity, and an optional fresh-generation nonce. HF revisions or local export hashes pin weights.

Generation, scoring, diagnostics, metrics, individual judge assessments, investigations, and final reports are separate jobs with durable attempts. Workers claim jobs using leases and tokens. Expired claims can be recovered; stale worker writes are rejected. External requests are not guaranteed exactly once after an interrupted connection.

### Storage, workers, and interface

The prototype uses FastAPI, SQLAlchemy, and SQLite, with a React/TypeScript/Vite dashboard. GPU workers claim model-grouped batches. CPU analysis workers process scoring and diagnostic stages. Judge workers use an OpenAI-compatible UCloud endpoint and validate structured results.

The dashboard supports evaluation creation, revision inspection, run control, model comparison, example drill-down, and proposal review. Drill-down retains sampler/source information, rendered messages, generations and metadata, scoring, heuristic assessments, and judge evidence. It is a local application without shared-hosting authentication or permissions.

## 4. Autonomous judge workflow

### Individual assessment and audits

Failed generations with unresolved dimensions are routed to the judge. Successful and failed responses are also sampled separately for each model. Current defaults are 10% per stratum, minimum 10 and maximum 100, bounded by available examples. Membership, seeds, and inclusion probabilities are recorded.

The UCloud judge is `zai-org/GLM-5.3` at `https://ai.cloud.sdu.dk/v1`. Credentials stay outside definitions and repository files. Evidence is treated as untrusted input. Structured output is validated, and cited response excerpts must actually occur in the response. Invalid structured responses receive bounded correction retries; failure remains inspectable.

### Automatic pattern investigations

The new generic scheduler detects a pattern when at least three responses and at least 20% of a model's completed responses either hit the token limit or are unresolved scored failures. Current scheduling occurs after the run's base scoring, diagnostics, and metrics are ready, before all individual judge audits must finish.

It selects up to 12 affected responses and 12 controls deterministically. The evidence includes population and sample counts, example data, prompt templates, rendered prompts, model configuration, requested generation settings, stop reasons, token counts, scores, and heuristic findings. A lack of controls is recorded rather than concealed.

The scheduler detects symptoms; **the LLM judge supplies the diagnosis and recommendations**. It may suggest experiments involving prompts, token budgets, thinking settings, scorers, or heuristics, but must distinguish observed behavior from hypotheses. A flag such as `enable_thinking=False` alone does not establish actual thinking behavior.

Investigation results appear in the dashboard, feed the final report, and create pending proposals linked to their originating job. They do not launch follow-up experiments automatically. Experiment proposals are currently advisory; a separate run must be configured to test them.

### Final report

The final report summarizes results, coverage, audit findings, investigations, and proposed changes. Current detailed report evidence is capped at the first 200 rows with an explicit truncation note. This is a known limitation: replace positional selection with representative stratified evidence and aggregate failure summaries before treating reports as comprehensive.

## 5. Partial reruns and resume semantics

| Action | Intended behavior |
|---|---|
| Resume | Continue the same run and sample manifest from stored checkpoints. |
| Retry failed stages | Preserve successful work and retry failed jobs. |
| Partial rerun | Create a linked run for selected examples, subsets, models, or repeats; show its denominator. |
| Reference correction | Reuse unchanged generations; invalidate dependent scoring and analysis. |
| Prompt or generation-setting change | Regenerate affected inputs; preserve prior results. |
| Scorer or heuristic change | Reuse compatible generations and recompute dependent stages. |
| Fresh generation | Explicitly bypass cache reuse without erasing the baseline. |
| Cancel | Stop scheduling for that run; in-flight or shared jobs may still finish. |

Acceptance criteria include stable sample identity, explicit partial coverage, no duplicate accepted writes after recovery, and reproducible lineage from each finding back to its inputs and component versions.

## 6. Current campaign and protocol

Three local models are evaluated on three full splits, preceded by seeded 100-example pilots. Full runs reuse compatible pilot generations.

| Model | Pinned source |
|---|---|
| Mimir v1.8 | Approved local step-3,800,000 EMA export at `/work/dfm/euroeval-runs/mimir-v1.8-3800000-validation/model` |
| Gemma 4 E2B | `google/gemma-4-E2B-it`, revision `4adf15748de9e1a0097b6274cee9c15836c41c37` |
| Qwen 3.5 2B | `Qwen/Qwen3.5-2B`, revision `15852e8c16360a2fea060d615a32b45270f8a8fc` |

Mimir's local content digest is `98a1a3bfe5f35f59b2e46d3fcaf3f4b3e55e0e5f2925c6a2c2cb8e00b430c6c8`.

| Benchmark | Split and size | Dataset revision |
|---|---|---|
| WinoGrande | Validation, 1,267 | `01e74176c63542e6b0bcb004dcdea22d94fb67b5` |
| HellaSwag | Validation, 10,042 | `218ec52e09a7e7462a5400043bb9a69a41d06b76` |
| MMLU | Deduplicated test, 13,937 across 57 subjects | `c30699e8356da336a370243923dbaf21066bb9fe` |

Inspect's MMLU preparation removed 105 rows from 14,042. The wrapper uses an explicit non-CoT multiple-choice solver. This adaptation and deduplication belong to the protocol, not to a claim of exact leaderboard replication.

Generation is zero-shot, greedy, BF16, with native chat templates, requested `enable_thinking=False`, a 4,096-token context limit, and 64 maximum new tokens. These are **generated-answer scores, not likelihood-ranking leaderboard scores**. GPU workers use authorized spare memory alongside the existing training job.

## 7. Experimental results

### Completed full-split strict scores

| Model | WinoGrande | HellaSwag | MMLU |
|---|---:|---:|---|
| Mimir v1.8 | 975/1,267 — 76.95% | 8,040/10,042 — 80.06% | Scoring in progress |
| Gemma 4 E2B | 0/1,267 — 0.00% | 5,611/10,042 — 55.88% | Scoring in progress |
| Qwen 3.5 2B | 645/1,267 — 50.91% | 5,450/10,042 — 54.27% | Scoring in progress |

Gemma reached the token limit on all 1,267 WinoGrande responses. Its strict zero is therefore not a reliable standalone capability ranking. This is an observed failure pattern for the framework's judge to investigate; no completed judge diagnosis is claimed in this snapshot.

Other observed token-limit counts are Gemma HellaSwag 461, Qwen WinoGrande 22, and Mimir zero on both completed benchmarks. Heuristics flagged 37 Qwen HellaSwag responses as scoring too harsh. These are heuristic findings, not independently confirmed judge conclusions.

### Completed 100-example pilots

| Model | WinoGrande | HellaSwag | MMLU |
|---|---:|---:|---:|
| Mimir v1.8 | 79% | 71% | 66% |
| Gemma 4 E2B | 0% | 66% | 16% |
| Qwen 3.5 2B | 51% | 49% | 40% |

Pilots are subsets of the full runs, not independent replications.

### Earlier Gemma diagnostic experiments

On the same 20 selected WinoGrande examples:

| Variant | Strict correct | Token-limited responses |
|---|---:|---:|
| Original prompt, 64 tokens | 0/20 | 20/20 |
| Original prompt, 256 tokens | 11/20 | 7/20 |
| Clearer answer-only prompt, 64 tokens | 11/20 | 0/20 |

These experiments were initiated by the coding agent before the ownership principle was clarified. Preserve them as historical experimental evidence; do not label them judge-proposed, automatically initiated by KineticEval, or representative full-benchmark results. No new follow-up variant should be selected by the coding agent in place of the judge.

The existing candidate final-answer-marker heuristic and prompt-clarification proposal also remain separate from LLM-generated recommendations. They have not replaced the baseline.

## 8. Implementation state at this snapshot

- All 75,738 baseline generations are complete: 25,246 per model. An additional 40 Gemma diagnostic generations are retained separately.
- Full WinoGrande and HellaSwag scoring are complete. MMLU generation is complete; scoring and subsequent diagnostics are still running. Partial MMLU scores are intentionally not presented as final results here.
- The first judge-enabled WinoGrande pilot has 95 accepted individual assessments and 84 pending assessments. This is not completed campaign-wide judging.
- KineticEval automatically queued two investigations: Gemma's 100/100 pilot truncation and unresolved failures, and Qwen's 49/100 unresolved pilot failures. Both were running at snapshot time.
- Investigation recommendations and a completed final judge report were not yet available at snapshot time.
- Mimir encountered an allocation-limit OOM and was resumed with smaller batches. Gemma encountered SQLite lock contention and was resumed. Completed outputs were retained and affected work retried.
- Queue improvements include atomic batch completion, longer lock waits, and moving model pinning outside the run-creation write lock. Large SQLite scheduling transactions remain a performance concern.
- Validation of the latest changes: 17 workflow tests passed; 10 investigation/integration tests passed with one opt-in live test skipped; frontend production build passed; browser workflow passed. These counts describe targeted checks, not a new claim about the entire test suite.

## 9. Plan and TODOs

### Immediate completion

- [ ] Finish MMLU scoring, diagnostics, and metrics; export only complete full-split results as final scores.
- [ ] Let KineticEval complete the active investigations and persist judge-authored findings and proposals.
- [ ] Complete the three pilot audits, then full-run judging and reports; surface failures and remaining coverage explicitly.
- [ ] Review the judge's recommendations and choose controlled, matched-example experiments through the framework.
- [ ] Preserve distinctions among baseline results, heuristic estimates, sampled judge findings, and follow-up results in exports and the dashboard.

### Framework improvements

- [ ] Make investigation thresholds, sample budgets, and cost/concurrency limits configurable and revisioned rather than fixed defaults.
- [ ] Version the judge prompt/protocol in cache identity; changing worker prompt code must not leave provenance ambiguous.
- [ ] Add first-class experiment plans, approvals, run creation, and paired comparisons for generation-setting recommendations. Current experiment proposals are advisory only.
- [ ] Improve report sampling and aggregate coverage across models, subsets, failure categories, and judge disagreements.
- [ ] Extend the dashboard's review workflow so suggested experiments can be configured without hand-editing requests.
- [ ] Add queue fairness, transient-provider retry/backoff, clearer resource/cost budgets, and bulk database operations that avoid long write locks.
- [ ] Improve preservation of raw generated token/channel information where needed to verify thinking behavior. Decoded text currently skips special tokens.
- [ ] Expand regression tests for multilingual answers, conflicting markers, negation, truncation, insufficient instructions, and partial heuristic coverage.
- [ ] Broaden Inspect task support through explicit adapter capabilities, without silently weakening unsupported semantics.
- [ ] Validate proposed heuristics on held-out examples and counterexamples before activation; monitor subsequent disagreements.

### Longer-term design

- [ ] Extend evaluation contracts beyond single-generation QA to judged tasks, semantic similarity/BERTScore, arena/pairwise comparisons, and fluid evaluations.
- [ ] Support richer dataset overlays and review history, evolving subsets, and reusable domain-specific scoring/heuristic libraries.
- [ ] Add shared deployment authentication, access controls, and a production storage/queue strategy before multi-user hosting.

## 10. Navigation and reproducibility

- [Contributor principles](../AGENTS.md)
- [Setup, API, adapters, and worker commands](../README.md)
- [Automatic investigation scheduler](../backend/kineticeval/investigations.py)
- [Evaluation lifecycle and dependency scheduling](../backend/kineticeval/service.py)
- [Judge provider and validation](../backend/kineticeval/providers.py)
- [Investigation lifecycle tests](../backend/tests/test_investigations.py)
- [Campaign manifest](../artifacts/campaign.json), [comparison CSV](../artifacts/comparison.csv), and [comparison JSON](../artifacts/comparison.json)
- [Earlier diagnostic run manifest](../artifacts/gemma-diagnostics.json)

The `artifacts/` links refer to local ignored files and change as work progresses; they are not a portable archival release. Run lineage and complete evidence remain in `kineticeval.db`. Refresh exported status with `python scripts/export_comparison.py`. The dashboard is served at `http://127.0.0.1:8000` while the API process is running.

Full baseline run IDs: WinoGrande `de0e971e31c94415994ea8776bf804f4`; HellaSwag `1ea545f4b0204469ac46b7bdaa1da57d`; MMLU `1d52d1fd98324a548480443ae30a99d0`. The first judged pilot is `d6f73fb33d0d47e9832ef8ca3cc0c3b5`. IDs are retained here to make the snapshot auditable against subsequent progress.
