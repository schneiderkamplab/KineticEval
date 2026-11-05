# KineticEval

[Project wiki](wiki/README.md) · [OKF v0.2 design and status record](wiki/OKF-v0.2.md)

A local workspace for living question-answering evaluations. Preserve benchmark scores, investigate failures, audit successes, and improve datasets and prompts without losing previous results.

## Run locally

Python 3.11+ and Node.js 20+ are required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,inspect,data]'
npm ci --prefix frontend
npm run build --prefix frontend
```

Start the API/dashboard and worker in separate terminals using the same environment and database path:

```bash
python -m uvicorn kineticeval.api:create_app --factory --host 127.0.0.1 --port 8000
python -m kineticeval.worker
```

Open **http://127.0.0.1:8000**. Choose **Start with QA fixture**, then **Start evaluation**. For a demo, choose **Additional options → Use demo models**; these are deterministic fixtures. Evaluated models use local `hf` weights or an `inspect` adapter. UCloud is reserved for inspectors in this workspace. Jobs remain pending until the worker runs. Use `python -m kineticeval.worker --concurrency 4` for concurrent handlers, or `--once` to drain available work and exit.

For frontend development, run `npm run dev --prefix frontend`; its API proxy points to port 8000. Set `KINETICEVAL_DB` in both processes to use a different database. SQLite persists definitions, sampled rows, rendered prompts, attempts, and analysis artifacts. Keep the service bound to localhost; there are no accounts or shared-hosting permissions.

## Inspect AI's three roles

1. **Import definitions:** choose Import → Inspect AI and supply an installed task factory, for example `{"factory":"kineticeval.fixtures:inspect_fixture","limit":20}`. Dataset examples and rendered prompts are materialized, generation defaults are inherited, and callable provenance is pinned.
2. **Scorers and metrics:** imported tasks retain their Inspect scorer and metrics. KineticEval reconstructs task state around stored generations so scoring can run independently.
3. **Generation backend:** use `{"name":"provider/model","backend":"inspect","parameters":{"temperature":0}}` to execute pending model requests through Inspect.

V1 imports support one scorer, scalar scores, text inputs, and exactly one generation per example. Supported solvers are `generate`, `prompt_template`, `system_message`, and unshuffled `multiple_choice`, including chains. Unsupported setup, sandbox, multi-turn, or custom solver behavior is rejected explicitly. Inspect private registry access is isolated in `inspect_adapter.py` and covered by integration tests. Changing installed task code requires importing a new revision.

## Data and revisions

Import JSONL with explicit field mappings, or a capped Hugging Face dataset. A GSM8K configuration is prefilled in the dashboard; `examples/gsm8k-snapshot.json` is a verified three-example snapshot with its upstream revision and original rows. GSM8K's `####` final answer is extracted into the reference while retaining full source reasoning.

Each run pins an immutable definition and a seeded, materialized sample manifest. Corrections create overlays keyed by example ID. The dashboard previews prompt and reference changes before saving a revision. Historical runs retain their original references and scores.

## Resume, retry, and partial reruns

- **Cancel** stops new scheduling for that run. In-flight calls can finish; completed work is retained. Shared jobs may continue for another active run.
- **Resume** continues the same run and sample manifest from persisted checkpoints.
- **Retry failed stages** retries failed generation or analysis jobs. Successful generations remain untouched.
- **Rerun selected** creates a linked run for selected example IDs and chosen models. Compatible artifacts are reused. Partial denominators remain visible.
- **Fresh generations** bypasses reuse with a new sample identity. Repeats have separate identities.

Cache identity includes rendered messages, backend/model configuration, generation parameters, and repeat identity. Reference-only corrections reuse generations; prompts and model inputs invalidate affected generations. Scoring, heuristics, inspection, and reports have separate dependency keys. Metric-only changes reuse underlying scores. Aggregate metric outputs are persisted, so historical summaries remain readable after installed components change. Lease recovery rejects stale worker writes. A network interruption can leave a provider request's outcome unknown; retries do not promise exactly-once external execution.

## Inspectors and improvement proposals

Set `UCLOUD_BASE_URL` to the inference server's OpenAI-compatible base URL **including `/v1`**, and `UCLOUD_API_KEY` in the API and worker environment. Do not put credentials in definitions or files committed to source control. The address is supplied by the UCloud model playground; it is intentionally not guessed.

`GET /api/providers/ucloud/models` discovers available model IDs. Enable `inspector` in a configuration revision with `provider: "ucloud"` and a discovered `model`. The worker uses chat completions, validates structured output, and rejects fabricated evidence excerpts. Failures remain inspectable and retryable. Live integration checks are opt-in; normal tests use a mocked transport or an explicitly labeled fake inspector.

Failed responses with unresolved dimensions receive inspector assessment. Successes and failures are also sampled separately per model for auditing (10%, minimum 10, maximum 100, bounded by stratum size). Seeds, sample membership, and inclusion probabilities are recorded. Inspector findings never silently overwrite heuristic findings or benchmark scores.

Reports summarize evidence and propose data, prompt, scorer, or heuristic changes. Reports include at most 200 detailed results and explicitly disclose truncation. Users approve proposals into new revisions. Heuristic proposals require positive cases and counterexamples to pass regression validation. No generated Python is executed.

## Custom components

Built-in scorers: `exact`, `normalized`, `mcq`. Built-in heuristics: conservative `qa` and reviewed `aliases` mappings. A finding may cover only one dimension; unresolved semantic correctness is not treated as wrong.

Installed Python extensions use `python:module:function` with an explicit version and configuration. Source fingerprints are pinned at revision creation. Scorers receive `(payload, config)` and return a dictionary with numeric `value` (async supported); heuristics return the `Finding` schema; metrics receive `(scores, config)` and return a numeric value. See `examples/custom_components.py`. Reinstall changed code and create a new revision before rerunning.

## Verification

```bash
python -m pytest -q
npm test --prefix frontend
npm run build --prefix frontend
npx --prefix frontend playwright install chromium
npm run test:e2e --prefix frontend
```

For an explicit live inspector check, set `KINETICEVAL_LIVE_TESTS=1` and `UCLOUD_INSPECTOR_MODEL` in addition to the UCloud endpoint and key, then run `pytest -k live_ucloud`. This makes one small inference request.

Browser tests launch their own disposable API, worker, and database. API documentation is available at `/docs`. Tests cover cache reuse, corrections, partial runs, stale leases, analysis retries, proposal validation, Inspect integration, UCloud's transport contract, and the dashboard workflow.

Reference documentation: [Inspect tasks](https://inspect.aisi.org.uk/tasks.html), [Inspect models](https://inspect.aisi.org.uk/models.html), [UCloud integration](https://docs.cloud.sdu.dk/inference/model-integration.html).


## Local Hugging Face evaluation

The `hf` backend runs pinned Hugging Face weights locally. The fleet prefers vLLM for supported architectures, with compiled Transformers as the single-GPU fallback. Run the fleet from the CUDA environment containing both engines, and the analysis worker from the API environment:

```bash
python -m kineticeval.gpu_fleet --db kineticeval.db --gpus 0,1,2,3,4,5,6,7 --cap-gib 80
python -m kineticeval.worker --analysis-only --concurrency 4
```

The fleet targets 90% of currently free VRAM, capped at 80 GiB per GPU. It estimates weight size plus runtime reserve, selects the smallest fitting TP group (1, 2, 4, or 8), and starts independent TP=1 replicas when the model fits. Tensor-parallel execution uses vLLM; unsupported large-model architectures fail explicitly rather than silently substituting pipeline parallelism. It retains resident weights across compatible requests and releases idle groups when another model needs them. Models with continuous pending work can delay other models. Other applications' allocations are left intact.

vLLM manages KV-cache capacity and continuous batching within its VRAM budget. Transformers uses `torch.compile`, starts with small batches, then adjusts from observed peak memory; CUDA OOM reduces the batch and retries the same requests. The memory target is a budget, not a guarantee of exactly 90% utilization for every workload. Compilation and model loading add startup latency. Actual engine, TP, compilation state, pinned weights, and generation settings are recorded with responses.

For manual debugging, `python -m kineticeval.local_worker --watch --engine auto --adaptive` runs one worker; optional `--model` restricts it. Production workers are managed by the fleet. The analysis-only worker schedules scoring as responses arrive and handles checks, metrics, configured inspectors, and reports across all active runs. Supply its UCloud environment for inspection. The dashboard polls run status and progress automatically.

Model configurations include `revision` for HF weights, or `local_path` for an existing export. Local files receive a content digest; remote revisions resolve to commit IDs. Native chat templates, template arguments, dtype, context limits, runtime versions, token counts, and stop reasons are recorded. Inputs exceeding the context budget fail explicitly rather than being truncated. HRM prefix-attention masks are preserved when the model configuration enables them.

The current comparison uses zero-shot Inspect Evals definitions for WinoGrande validation, HellaSwag validation, and deduplicated English MMLU test. It uses greedy BF16 generation, native chat templates with `enable_thinking=False` requested, and 64 output tokens. These are generated-answer scores, not likelihood-ranking leaderboard scores. `scripts/create_campaign.py` queues a seeded 100-example pilot and the full split; full runs reuse pilot generations. `scripts/analyze_campaign.py` processes finished runs by stage. Credential-free manifests and results are stored under ignored `artifacts/` and the local database.

UCloud is used separately for the inspector. It requires a **provider inference token** generated from a model page in the UCloud AI playground. A general platform API token for `cloud.sdu.dk` does not authenticate against the inference endpoint. Always use the server address displayed with the inference token.

## Automatic failure investigations

With inspection enabled, KineticEval schedules a model-specific investigation when at least three completed responses and at least 20% of a model's responses hit the token limit or remain unresolved scored failures. This runs before all individual audits finish. The framework samples up to 12 affected responses and 12 controls deterministically, retaining population counts and the sampling policy. Evidence includes the source examples, rendered prompts, requested model settings, generation metadata, scores, and heuristic findings.

The LLM inspector supplies the diagnosis and recommendations; the scheduler only detects symptoms. Investigations appear in run results and feed the final report. Recommendations retain their originating job and remain pending. Experiment recommendations require an explicit approval preview: they do not alter settings, rerun models, or rewrite scores before approval. A requested thinking flag is evidence of configuration, not proof of effective behavior.

## Approving traceable follow-up experiments

Open **Experiments**, choose a inspector recommendation, select a baseline model and matched example IDs, and edit named arms. Supported overrides are generation parameters (`max_tokens`, `temperature`, `top_p`, `seed`), `enable_thinking`, and the prompt template. Unsupported transformations such as choice reordering are not silently translated into executable plans.

**Save approval preview** persists an immutable plan with exact settings, sample scope, rendered prompt examples, cache reuse counts, and output-token ceilings. It creates no run or generation jobs. **Approve and queue follow-up runs** atomically creates linked runs for all arms, recording the source proposal, inspector job, baseline, and approval time. Repeated approval requests return the same runs. Prompt changes create new definition revisions; weights, references, scoring, and baseline repeats remain fixed.

The comparison panel matches example ID and repeat for the selected model. It reports coverage, original and follow-up scores on scored pairs, paired score differences, and truncation counts. Incomplete pairs do not contribute to score changes. New local generations also retain raw decoded output with special tokens and token IDs; older cached generations may lack these fields.

Workers still execute queued work. A local GPU worker can wait with `--watch` and reuse loaded weights across parameter/template-argument arms. Run `python scripts/analyze_experiments.py` in the pinned CPU environment to process approved runs and their configured inspectors; it requests the inference credential without storing it in files. The GPU fleet watches all eight authorized GPUs and allocates only the available headroom. No follow-up inference occurs until approval.

## Experiment validation and inspector cache provenance

Experiment preparation and approval both reject unsupported operations, prose-only step lists, and plans that make no effective change to model inputs or generation settings. A baseline arm is allowed alongside changed arms; a fully unchanged rerun is not an experiment. Cached generations from a previously executed changed arm remain reusable. Unsupported proposals stay visible with a blocking reason; they are not converted into default no-op arms.

New inspector, investigation, and report jobs persist their exact system instructions, output schema, evidence-validation policy, and correction policy, together with a SHA-256 identity. These are part of the job cache key. New runs cannot reuse a assessment from a different protocol. Resuming an existing run keeps its original pinned protocol, and workers send that stored prompt rather than rebuilding it from current code. Pending pre-versioning jobs use the frozen legacy protocol; historical unversioned results remain inspectable but cannot satisfy new versioned cache entries. Schema or integrity mismatches are rejected before contacting the provider.

## Evaluation overview and editing

Evaluation cards and detail pages show example counts, actual corrections, evaluation type, and the scorer in use. Detail pages add selected sample counts, aggregate metrics, a correction indicator, and subset distribution bars. Inspect scorers are identified by their underlying scorer, rather than only the adapter name.

Use **Edit prompts**, **Edit sampling**, **Edit scoring & checks**, or **Edit inspector** to open only the relevant settings. Their saved patches are limited to that section. Use **Edit evaluation** for labeled fields covering name, description, prompts, answer format, sampling, scoring, diagnostic checks, and the inspector. **Preview changes** summarizes affected prompts and references before saving a new revision. Existing custom/Inspect component configuration is preserved when unrelated fields change. Example corrections and common import options also use fields; imported choice changes that require coordinated prompt edits remain an advanced operation. **View/edit configuration (JSON)** explicitly opens the advanced editor.

### Progress and cancellation

Progress separates completed, running, queued-for-worker, and dependency-waiting work. Heuristic coverage counts every checked response; semantic verdicts separately report correct, incorrect, and unresolved cases. Analysis workers share inspector capacity between active runs, prioritizing an investigation or report within each run before further item inspection. Scoring and heuristic checks stream as generations arrive.

Cancellation applies to the selected run, not automatically to its parent or other related runs. The result view exposes the parent link. Inspector workers check for an active consumer before each provider request, including repair attempts. Already-sent requests may incur cost; completed evidence is retained. Canceled claims awaiting a provider request return to the queue for a deliberate resume, but cannot be claimed while every linked run is inactive.

### Input revisions, model configurations, and Results

Evaluation revisions identify data (including reference answers), sampling, and prompt construction. Editing suggested scoring, metrics, heuristics, or LLM inspector settings does not create an input revision. Every run freezes its effective settings; subsequent changes to suggestions do not rewrite historical results. Revision names are optional and creation timestamps are displayed in local time.

The **Models** library stores named configurations, including multiple variants of the same physical model. The shared **Create runs** dialog selects evaluation revisions and model configurations and permits overrides before launch. One evaluation across several models is one run; several evaluations create several runs grouped into a **Result**. Submission retries use an idempotency key. Matching generations remain reusable across display-name changes.

**Results** can also group existing runs. Removing membership or deleting a collection does not delete its runs. Comparisons rank fully scored entries within matching input/scoring groups; an explicit mismatch override shows a warning. Unknown metric direction requires a choice before ranking. Response outcome charts distinguish correct, incorrect, and unscored responses; inspection coverage is separate.

Scoring and inspection are different roles: the **LLM inspector** investigates diagnostic dimensions, audits responses, and proposes changes. The word judge is reserved for future judged scoring. Pinned historical inspection prompts and schemas remain intact for reproducibility.

For an older database, stop the API and all workers, take a consistent SQLite backup, then run:

```sh
python -m kineticeval.migrate_inspector kineticeval.db
python -m kineticeval.migrate_inputs kineticeval.db
```

These migrations are transactional and idempotent. Input consolidation assigns new revision IDs and rewrites references without aliases; run and job IDs remain stable. Restart the API and workers only after both commands complete. Never run old workers against the migrated schema. Roll back by restoring the backup together with its matching code version.
