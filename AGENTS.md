# Repository Guidelines

## Core Principles
KineticEval is a dynamic evaluation framework, not just a benchmark score reporter. Preserve original scores while investigating failures, scorer mistakes, ambiguous instructions, and uncertain semantic correctness.

**KineticEval itself must invoke the LLM judge.** Coding agents implement and verify the framework; they must not substitute their own diagnosis, manually prompt the judge about particular failures, or hand-author recommendations presented as judge findings. Recurring failure patterns must automatically trigger framework-managed investigations with traceable evidence and persisted recommendations.

Use reliable heuristics first. Ask the judge about unresolved dimensions and audit samples of both successes and failures. Keep semantic correctness, format compliance, instruction adequacy, and scorer correctness separate. Represent uncertainty explicitly; never infer an answer from an incidental letter.

Judge recommendations are proposals, not automatic changes. Test suggested heuristics against positive examples and counterexamples. Evaluate prompt, scoring, data, and generation-setting changes as separate, traceable revisions or runs. Retain the original evidence and comparison baseline.

## Architecture and Provenance
Evaluations are living definitions: retain dataset provenance, subsets, samplers, templates, overlays, scorers, metrics, and heuristics. Record rendered prompts, model configuration, generation metadata, scores, diagnostic evidence, and judge findings.

Inspect AI can supply initial evaluation definitions, scoring functions, and metrics, and serve as one generation backend. Keep these roles independent and extensible.

Cache generations by effective task and model configuration. Support model-grouped pending work, partial reruns, and durable resumes. Reuse generations for scoring-only changes; changed prompts or generation settings must change cache identity. Preserve attempts and reject stale worker writes.

## Project Structure and Commands
Python source lives in `backend/kineticeval/`, pytest tests in `backend/tests/`, and the React dashboard in `frontend/`. Operational scripts live in `scripts/`; local results belong in ignored `artifacts/`.

Install with `pip install -e '.[dev,inspect]'`. Run `pytest` from the repository root. In `frontend/`, use `npm run dev`, `npm run build`, and `npm run test:e2e` for development, compilation, and browser checks.

## Implementation and Configuration
Use four-space Python indentation, snake_case functions, and focused modules. Test lifecycle invariants and meaningful behavior; external API tests must be opt-in.

For the current campaign, evaluated HF models run locally within authorized GPU headroom; UCloud serves only the judge. Never commit credentials or expose them in logs. There is no established Git history convention; use imperative commit subjects and describe changes, validation, and limitations in pull requests.
