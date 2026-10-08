# Databricks CLI & IDE — 5-Minute Demo (Module 0)

**Message:** the same Databricks ML platform you use in the browser is fully drivable from your
laptop — terminal or IDE. That's what makes automation, reproducibility, and CI/CD possible.

All commands target the workshop profile `<profile>`. Swap it for your own.

## 1. Install the unified CLI

```bash
# macOS / Linux
brew install databricks
# or the official installer:
curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh

databricks --version      # expect v0.2xx / v1.x (the unified CLI, NOT the old `databricks-cli` pip package)
```

> The old `pip install databricks-cli` package is **deprecated**. Use the standalone unified CLI above.

## 2. Authenticate (once)

```bash
databricks auth login --host <workspace-url> --profile <profile>
databricks auth profiles          # list configured profiles + their hosts
databricks current-user me --profile <profile>
```

## 3. Browse Unity Catalog from the terminal

```bash
# NOTE: UC subcommands take POSITIONAL arguments (not --flags)
databricks catalogs list --profile <profile>
databricks schemas list main --profile <profile>
databricks tables list main ml_lifecycle_workshop --profile <profile>
databricks tables get main.ml_lifecycle_workshop.workshop_lapse_bronze --profile <profile>
```

## 4. ML surfaces from the terminal

```bash
databricks experiments list --profile <profile>
databricks serving-endpoints list --profile <profile>
```

## 5. Move notebooks between laptop and workspace

```bash
# push a local notebook to the workspace
databricks workspace import /Workspace/Users/<you>/demo \
  --file ./notebooks/00_setup.py --format SOURCE --language PYTHON --overwrite \
  --profile <profile>

# pull it back
databricks workspace export /Workspace/Users/<you>/demo --format SOURCE \
  --profile <profile>
```

## 6. VS Code extension (teaser)

Install **Databricks** from the VS Code Marketplace. It lets you:

- Sync your local project to the workspace and **run notebooks on Databricks compute** from the editor
- **Debug** Python with breakpoints against remote compute
- Browse workspace files, clusters, jobs, and bundles
- Scaffold and deploy **Databricks Asset Bundles** (see `bundle/`) without leaving the IDE

**Takeaway:** notebook, terminal, and IDE are three doors into one governed platform — pick the one
that fits the task. Automation and CI/CD live behind the terminal and IDE doors.
