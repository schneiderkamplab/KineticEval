# Follow-up approval workflow

Implementation update, 4 November 2026. This supplements the earlier [OKF v0.2 snapshot](OKF-v0.2.md).

KineticEval now converts experiment recommendations into editable, immutable approval previews in the dashboard. Preparing a preview creates no runs. Explicit approval creates linked runs atomically and records the original judge proposal, originating judge job, baseline, exact arms, matched example IDs, and approval timestamp. Repeat approval requests cannot create duplicate runs. Unsupported data transformations remain outside the executable configuration surface.

The comparison view uses matched example IDs and repeats, displaying scored-pair coverage and paired differences rather than comparing incomplete denominators. Prompt arms create separate definition revisions; generation-only arms reuse the evaluation definition. Existing compatible generations are reused. Newly generated outputs retain raw decoded text and generated token IDs for channel inspection; old cache entries may not contain them.

## Prepared Gemma experiment

KineticEval's GLM-5.3 investigation proposed six arms on the same 100 WinoGrande pilot examples: the 64-token baseline, caps of 128, 256, and 512 tokens, a terminal prompt reminder at 256 tokens, and thinking requested at 256 tokens. The mechanism behind the observed failure remains a judge hypothesis to test. No arm is an adopted fix.

Preview `1b7667b88edb42aa9efa9419c41826b1` is awaiting user approval. It can reuse 120 generations and requires 480 new generations. The source proposal is `870734dd12c2910df4eebcc8453439aeb2a782433a90e5107bd743a83d94fd20`; the source investigation is `961d51d3acb0624a2e3e921736d1eb9cab5ece0f93cc3d644a4ffeb65fa48f7b`.

GPU and analysis workers are waiting for approved work. The baseline remains unchanged. The dashboard approval action, not a coding agent decision, authorizes the queued follow-up runs.

## Updated baseline completion

MMLU scoring has completed under the original protocol: Mimir 65.78%, Gemma 23.73%, Qwen 46.45%. Gemma's MMLU generations hit the token limit in 9,382 of 13,937 examples; interpret the strict score alongside that coverage problem. Campaign-wide judge auditing and reporting remain in progress.

## Validation and remaining work

Backend tests cover preview isolation, invalid configurations, partial scopes, prompt revisions, idempotent approval, transaction rollback, rejected-proposal handling, paired comparison, API routing, and local-worker parameter-arm switching. Browser checks cover explicit approval, invalidating the displayed preview after edits, and the existing evaluation workflow.

Remaining work includes automated experiment decision rules, configurable data transformations, richer comparison metrics, full raw-output backfill policy, cost estimates, and multi-user approval identity. Token ceilings in the preview are bounds on requested output tokens, not billing estimates.
