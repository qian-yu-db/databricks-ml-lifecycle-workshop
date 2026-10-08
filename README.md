# Databricks ML Lifecycle Workshop (Insurance / Policy Lapse)

A 2-hour, instructor-led walkthrough of the end-to-end ML lifecycle on Databricks, aimed at a
data-science / ML-engineering audience that is new to the platform. The running use case is
**policy-lapse prediction** (which policyholders are likely not to renew), on a **synthetic**
dataset the workshop generates for you — so no data setup is required.

## Audience & goal

Experienced ML practitioners, new to Databricks. By the end they've seen a complete, governed path:
**data → features → train → register → deploy (batch + real-time) → monitor**, plus how to work from
the **CLI/IDE** and ship with a **Databricks Asset Bundle**.

## Run order & timing (maps to the agenda)

| Notebook | Module | Time |
|---|---|---|
| `notebooks/00_setup.py` | shared setup (run first) | — |
| `notebooks/01_platform_orientation.py` | Module 0 — Orientation + CLI/IDE | 15–20 min |
| `notebooks/02_feature_engineering.py` | Module 1 — Data & Feature Store | 20–25 min |
| `notebooks/03_model_training.py` | Module 2 — Training & MLflow | 25–30 min |
| `notebooks/04_model_registration.py` | Module 3 — Registration | 15 min |
| `notebooks/05_batch_inference.py` | Module 4a — Batch scoring | part of 25 min |
| `notebooks/06_model_serving.py` | Module 4b — Real-time serving | part of 25 min |
| `bundle/` | Module 4c — DAB intro | part of 25 min |
| `notebooks/07_model_monitoring.py` | Module 5 — Monitoring | 10 min |

## Outside-the-workspace moments

- **CLI / IDE** (Module 0): `docs/cli_quickstart.md` — a 5-minute terminal + VS Code demo.
- **DAB** (Module 4): `bundle/` — deploy the train→register→score pipeline as a scheduled job.

## Configuration

This repo ships with placeholders so you can point it at your own workspace.

- **Profile**: nothing is hardcoded. Pass `--profile <your-profile>` on CLI/bundle commands (or set
  `DATABRICKS_CONFIG_PROFILE`). To pin one in the bundle, add a `workspace.profile` under the target in
  `bundle/databricks.yml`.
- **Unity Catalog location**: edit the top of `notebooks/00_setup.py`:
  ```python
  catalog = "main"                    # a catalog you can create objects in
  db      = "ml_lifecycle_workshop"   # schema (created if missing)
  volume  = "datasets"                # volume for raw data (created if missing)
  ```
- **Workspace requirements**: serverless compute; UC permissions to create a schema, volume, and tables;
  Model Serving + a Lakebase online store enabled (for Module 4b real-time serving).

## Self-bootstrapping data

`00_setup.py` **generates** a synthetic ~7,000-row policy book, writes it to the volume, and loads a
`workshop_lapse_bronze` Delta table — but only if that table doesn't already exist (idempotent). Any
notebook that `%run`s `00_setup` (that's `02`–`07`) triggers it the first time; `01` is a markdown-only
orientation and runs no code. So there is **no manual data upload** — just run the notebooks.

## Pre-session warm-up (recommended)

The online store and serving endpoint take **minutes** to provision. Before a live session:

1. Run `00_setup`, then `02`, `03`, `04` (creates data, feature table, trains, registers Champion).
2. Run `06_model_serving.py` once so the online store + endpoint reach **READY**.
3. (Optional) pre-run `07`'s monitor creation so the dashboard exists.

## Running the bundle (Module 4)

```bash
cd bundle
databricks bundle validate --strict --target dev --profile <your-profile>
databricks bundle deploy    --target dev --profile <your-profile>
databricks bundle run lapse_mlops_pipeline --target dev --profile <your-profile>
```

The job uses **serverless** compute pinned to **environment version 6** (Standard base).

## Teardown

The last cell of `notebooks/07_model_monitoring.py` has a commented cleanup block that removes the
monitor, serving endpoint, online store, and workshop tables so you can re-run from scratch.
