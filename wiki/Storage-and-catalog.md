# Storage and reproducible catalog

The execution database retains runs, Results, generation cache, jobs, scores, inspector evidence, and attempts. The definitions database contains evaluation revisions with full examples, revision labels, and model configurations. Input and model IDs remain stable across rebuilds.

The Git catalog stores readable metadata, dataset recipes, revision settings and overlays, and model templates. Bulk HF examples are downloaded by the single `scripts/build_definitions.py` entrypoint. Counts and content hashes reject dataset drift. Pinned Inspect packages and committed adapters reproduce task-specific prompts, filtering, and deduplication. Rebuilding neither generates answers nor invokes the inspector.

The initial catalog covers three evaluations, seven revisions, six model configurations, and 25,246 distinct materialized examples across its three dataset recipes. These are complete task datasets, not the samples selected by smoke runs. A full rebuild matched the stored example hashes.

Migration explicitly moves the four catalog tables, preserving IDs. Both attached SQLite databases use rollback journals for atomic cross-database transactions. Legacy one-file databases remain readable until explicitly migrated. The execution database retains frozen historical run configuration and per-item evidence; those snapshots do not replace the editable catalog.

See [build, export, migration, and backup instructions](../catalog/README.md). Dashboard edits must be exported and committed to make a subsequent checkout reproduce them. Export currently rejects inline datasets without a versioned recipe.

Further catalog expansion from HRM source repositories is under compatibility review: supported task imports can be added with their original scoring/protocol, while unsupported custom solvers, multiple-reference targets, sandboxes, and inaccessible datasets require explicit follow-up. No imported comparison should start generation automatically.
