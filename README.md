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

Open **http://127.0.0.1:8000**. Choose **Start with QA fixture**, then **Start evaluation**. The two default models are deterministic fixtures, clearly labeled as such. Real models use `hf` (local GPU), `inspect`, or `ucloud` backends. Jobs remain pending until the worker runs. Use `python -m kineticeval.worker --concurrency 4` for concurrent handlers, or `--once` to drain available work and exit.

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

Cache identity includes rendered messages, backend/model configuration, generation parameters, and repeat identity. Reference-only corrections reuse generations; prompts and model inputs invalidate affected generations. Scoring, heuristics, judging, and reports have separate dependency keys. Metric-only changes reuse underlying scores. Aggregate metric outputs are persisted, so historical summaries remain readable after installed components change. Lease recovery rejects stale worker writes. A network interruption can leave a provider request's outcome unknown; retries do not promise exactly-once external execution.

## Judges and improvement proposals

Set `UCLOUD_BASE_URL` to the inference server's OpenAI-compatible base URL **including `/v1`**, and `UCLOUD_API_KEY` in the API and worker environment. Do not put credentials in definitions or files committed to source control. The address is supplied by the UCloud model playground; it is intentionally not guessed.

`GET /api/providers/ucloud/models` discovers available model IDs. Enable `judge` in a configuration revision with `provider: "ucloud"` and a discovered `model`. The worker uses chat completions, validates structured output, and rejects fabricated evidence excerpts. Failures remain inspectable and retryable. Live integration checks are opt-in; normal tests use a mocked transport or an explicitly labeled fake judge.

Failed responses with unresolved dimensions receive judge assessment. Successes and failures are also sampled separately per model for auditing (10%, minimum 10, maximum 100, bounded by stratum size). Seeds, sample membership, and inclusion probabilities are recorded. Judge findings never silently overwrite heuristic findings or benchmark scores.

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

For an explicit live judge check, set `KINETICEVAL_LIVE_TESTS=1` and `UCLOUD_JUDGE_MODEL` in addition to the UCloud endpoint and key, then run `pytest -k live_ucloud`. This makes one small inference request.

Browser tests launch their own disposable API, worker, and database. API documentation is available at `/docs`. Tests cover cache reuse, corrections, partial runs, stale leases, analysis retries, proposal validation, Inspect integration, UCloud's transport contract, and the dashboard workflow.

Reference documentation: [Inspect tasks](https://inspect.aisi.org.uk/tasks.html), [Inspect models](https://inspect.aisi.org.uk/models.html), [UCloud integration](https://docs.cloud.sdu.dk/inference/model-integration.html).


## Local Hugging Face evaluation

The `hf` backend runs model weights on this server; it does not send generations to a hosted API. Use a CUDA-enabled Transformers environment with support for the selected architecture. Set `CUDA_VISIBLE_DEVICES` to an authorized GPU and run one model-grouped worker per model:

```bash
CUDA_VISIBLE_DEVICES=5 python -m kineticeval.local_worker --db kineticeval.db \
  --model danish-foundation-models/DFM-Mimir-v1.8 --batch-size 32 --memory-fraction 0.20
```

Model configurations include `revision` for HF weights, or `local_path` for an existing export. Local files receive a content digest; remote revisions resolve to commit IDs. Native chat templates, template arguments, dtype, context limits, runtime versions, token counts, and stop reasons are recorded. Inputs exceeding the context budget fail explicitly rather than being truncated. HRM prefix-attention masks are preserved when the model configuration enables them.

The current comparison uses zero-shot Inspect Evals definitions for WinoGrande validation, HellaSwag validation, and deduplicated English MMLU test. It uses greedy BF16 generation, native chat templates with `enable_thinking=False` requested, and 64 output tokens. These are generated-answer scores, not likelihood-ranking leaderboard scores. `scripts/create_campaign.py` queues a seeded 100-example pilot and the full split; full runs reuse pilot generations. `scripts/analyze_campaign.py` processes finished runs by stage. Credential-free manifests and results are stored under ignored `artifacts/` and the local database.

UCloud is used separately for the judge. It requires a **provider inference token** generated from a model page in the UCloud AI playground. A general platform API token for `cloud.sdu.dk` does not authenticate against the inference endpoint. Always use the server address displayed with the inference token.

## Automatic failure investigations

With judging enabled, KineticEval schedules a model-specific investigation when at least three completed responses and at least 20% of a model's responses hit the token limit or remain unresolved scored failures. This runs before all individual audits finish. The framework samples up to 12 affected responses and 12 controls deterministically, retaining population counts and the sampling policy. Evidence includes the source examples, rendered prompts, requested model settings, generation metadata, scores, and heuristic findings.

The LLM judge supplies the diagnosis and recommendations; the scheduler only detects symptoms. Investigations appear in run results and feed the final report. Recommendations retain their originating job and remain pending. Experiment recommendations require an explicit approval preview: they do not alter settings, rerun models, or rewrite scores before approval. A requested thinking flag is evidence of configuration, not proof of effective behavior.

## Approving traceable follow-up experiments

Open **Experiments**, choose a judge recommendation, select a baseline model and matched example IDs, and edit named arms. Supported overrides are generation parameters (`max_tokens`, `temperature`, `top_p`, `seed`), `enable_thinking`, and the prompt template. Unsupported transformations such as choice reordering are not silently translated into executable plans.

**Save approval preview** persists an immutable plan with exact settings, sample scope, rendered prompt examples, cache reuse counts, and output-token ceilings. It creates no run or generation jobs. **Approve and queue follow-up runs** atomically creates linked runs for all arms, recording the source proposal, judge job, baseline, and approval time. Repeated approval requests return the same runs. Prompt changes create new definition revisions; weights, references, scoring, and baseline repeats remain fixed.

The comparison panel matches example ID and repeat for the selected model. It reports coverage, original and follow-up scores on scored pairs, paired score differences, and truncation counts. Incomplete pairs do not contribute to score changes. New local generations also retain raw decoded output with special tokens and token IDs; older cached generations may lack these fields.

Workers still execute queued work. A local GPU worker can wait with `--watch` and reuse loaded weights across parameter/template-argument arms. Run `python scripts/analyze_experiments.py` in the pinned CPU environment to process approved runs and their configured judges; it requests the inference credential without storing it in files. The current Gemma worker waits on GPU 6 with batch size 8 and a 0.25 memory fraction. No follow-up inference occurs until approval.

## Experiment validation and judge cache provenance

Experiment preparation and approval both reject unsupported operations, prose-only step lists, and plans that make no effective change to model inputs or generation settings. A baseline arm is allowed alongside changed arms; a fully unchanged rerun is not an experiment. Cached generations from a previously executed changed arm remain reusable. Unsupported proposals stay visible with a blocking reason; they are not converted into default no-op arms.

New judge, investigation, and report jobs persist their exact system instructions, output schema, evidence-validation policy, and correction policy, together with a SHA-256 identity. These are part of the job cache key. New runs cannot reuse a judgement from a different protocol. Resuming an existing run keeps its original pinned protocol, and workers send that stored prompt rather than rebuilding it from current code. Pending pre-versioning jobs use the frozen legacy protocol; historical unversioned results remain inspectable but cannot satisfy new versioned cache entries. Schema or integrity mismatches are rejected before contacting the provider.
