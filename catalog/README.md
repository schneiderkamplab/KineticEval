# Rebuilding the definitions database

`catalog/` is the reviewable source for KineticEval's definitions database. It contains evaluation metadata, input revisions, revision names and dates, analysis defaults, model settings, chat templates, and dataset recipes. It does not contain generated responses, run history, credentials, model weights, or bulk Hugging Face examples.

From the repository root, install the pinned catalog dependencies and run the single builder:

```bash
python -m pip install -e '.[catalog]'
python scripts/build_definitions.py --output kineticeval.definitions.db
```

The builder downloads datasets through Hugging Face (using its normal local cache), reconstructs the examples with the versioned task adapters, verifies their counts and content hashes, and populates SQLite. The database contains the complete examples selected by each dataset recipe, independently of a run's smoke/full sampler. Task-defined filtering and deduplication still apply: the current MMLU adapter produces 13,937 examples, HellaSwag 10,042, and WinoGrande 1,267. It does not download model weights or invoke an LLM.

The manifest `catalog.json` lists the active files. `datasets/` holds reproducible loaders and expected hashes; `evaluations/` holds metadata and revisions; `models/` holds model configurations; `templates/` contains editable Jinja templates. Corrections live in revision overlays, preserving the downloaded baseline. Task adapters and custom scoring code must be committed in the Python package, or supplied by pinned dependencies; external working directories are not build dependencies. Local model paths remain machine-specific and must be configured before generation on another server.

A failed build leaves the destination untouched. Existing output is refused unless `--replace` is explicit. Stop services before replacing a database they use. Gated datasets require your own Hugging Face access credentials in the environment; never commit these.

## Two databases

`kineticeval.db` holds execution history, Results, jobs, generations, scores, inspections, and attempts. `kineticeval.definitions.db` holds evaluations, revisions, labels, model definitions, and materialized examples. Set `KINETICEVAL_DEFINITIONS_DB` to use a different catalog path. Stable IDs allow a rebuilt catalog to reconnect to retained execution history.

For an existing single-file installation, stop all API and worker processes, back up the database, then run:

```bash
python -m kineticeval.storage kineticeval.db
```

The explicit migration preserves IDs and moves definition tables transactionally. Older installations remain readable until migrated. Back up both databases together while services are stopped; restoring only the catalog does not restore execution history.

After changing definitions in the dashboard, export them before committing:

```bash
python scripts/build_definitions.py --export-from kineticeval.definitions.db
```

Before migration, export from `kineticeval.db` instead. Review the Git diff and rebuild to a separate output file before publishing. Inline datasets require an explicit versioned recipe; export fails instead of silently embedding their data. Rebuilding preserves catalog IDs, names, dates, overlays, templates, and suggested defaults; it does not invent new revisions.
