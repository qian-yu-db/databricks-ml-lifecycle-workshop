# Databricks Asset Bundle — Lapse MLOps Pipeline (Module 4 demo)

This bundle packages the interactive workshop notebooks into an **automated, version-controlled
pipeline** — the "ML engineer / CI-CD" story. It defines one Lakeflow Job that chains:

```
train (03) → register (04) → batch_score (05)
```

on a daily schedule (paused in `dev`). This is the same work you did by hand in the notebooks, now
reproducible and deployable from a terminal or IDE.

## Demo commands

```bash
cd bundle
databricks bundle validate --strict --target dev --profile <profile>
databricks bundle deploy    --target dev --profile <profile>
databricks bundle run lapse_mlops_pipeline --target dev --profile <profile>
```

`validate` checks the config, `deploy` creates the job (and syncs the notebooks) in the workspace,
`run` triggers it on-demand.

## Notebook path note

The job tasks reference the notebooks in `../notebooks` (one level up from this bundle). The
`sync.paths: [../notebooks]` entry in `databricks.yml` brings them into the bundle.

**If `bundle validate` on your CLI version reports the notebooks are outside the sync root**, run
the bundle from the **repo root** instead: move `databricks.yml` and `resources/` up to the repo
root (so the bundle root contains `notebooks/`), change the task paths from `../../notebooks/…` to
`../notebooks/…`, and add `sync.exclude` entries for `.venv`, `.superpowers`, `tests`, `__pycache__`.
Everything else stays the same.

## Prerequisites

- Authenticated profile `<profile>` (`databricks auth login --profile <profile>`)
- Serverless jobs enabled in the workspace
- `00_setup` has been run once (creates the catalog/schema/volume and bronze data), and `02` has
  created the feature table + label spine the training task reads.
