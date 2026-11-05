# KineticEval Wiki

- [OKF v0.2 — KineticEval](OKF-v0.2.md): design goals, governing principles, architecture, implementation snapshot, experimental results, and TODOs, recorded on 4 November 2026.

See [AGENTS.md](../AGENTS.md) for contributor principles and [README.md](../README.md) for setup and operational commands. Versioned wiki records distinguish measured results, inspector findings, and planned work; later progress should not be silently backdated into an earlier snapshot.

- [Follow-up approval workflow](Follow-up-approval.md): subsequent implementation update, prepared Gemma approval, and completed MMLU baseline scores.

- Inspector protocol fingerprinting and rejection of unsupported/no-op experiments were added on 5 November 2026; see the corresponding section in the [operational README](../README.md#experiment-validation-and-inspector-cache-provenance). The previously approved Mimir shuffling follow-up is marked blocked, with historical outputs retained.

- [Input revisions, Models, and Results](Input-revisions-models-results.md): current revision model, migration evidence, structured dashboard workflows, comparison semantics, and verification map.

- [Storage and reproducible catalog](Storage-and-catalog.md): two SQLite databases, versioned definitions, full HF example materialization, rebuild verification, and full/smoke revisions.
- [HRM comparison import coverage](../catalog/IMPORTS.md): added evaluation and model definitions, provenance, and explicit compatibility gaps.
