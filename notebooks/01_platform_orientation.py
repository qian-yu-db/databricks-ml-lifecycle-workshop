# Databricks notebook source
# DBTITLE 1,Module 0: Platform Orientation
# MAGIC %md
# MAGIC # Navigating Databricks for ML
# MAGIC
# MAGIC This notebook is a **guided walkthrough** — the instructor will navigate the UI while participants follow along. No code execution required.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Workshop Overview
# MAGIC
# MAGIC We are building a **policy-lapse prediction** system end-to-end on Databricks — predicting which policyholders are likely **not to renew**, so retention teams can act early.
# MAGIC
# MAGIC | Module | What you'll learn | Time |
# MAGIC |------|-------------------|------|
# MAGIC | **0. Platform Orientation** | Navigate the workspace for ML; CLI & IDE | 15–20 min |
# MAGIC | **1. Data & Feature Engineering** | Load data, EDA, build a Feature Store table | 20–25 min |
# MAGIC | **2. Model Training** | Train a model, track with MLflow | 25–30 min |
# MAGIC | **3. Model Registration** | Register models in Unity Catalog | 15 min |
# MAGIC | **4. Inference / MLOps** | Batch scoring, real-time serving, deploy with a DAB | 25 min |
# MAGIC | **5. Model Monitoring** | Set up Lakehouse Monitoring | 10 min |

# COMMAND ----------

# DBTITLE 1,Workspace Layout
# MAGIC %md
# MAGIC ## 1. Workspace Layout
# MAGIC
# MAGIC ### Left sidebar — key areas for ML practitioners
# MAGIC
# MAGIC | Icon | Area | What it does |
# MAGIC |------|------|-------------|
# MAGIC | **Workspace** | `/Workspace/Users/<you>/` | Your notebooks, files, and repos |
# MAGIC | **Catalog** | Unity Catalog Explorer | Browse tables, models, volumes, functions |
# MAGIC | **Compute** | Clusters & SQL Warehouses | Machines that run your code |
# MAGIC | **Experiments** | MLflow Experiments | Track and compare ML runs |
# MAGIC | **Serving** | Model Serving | Real-time REST endpoints for models |
# MAGIC | **Jobs & Pipelines** | Lakeflow Jobs | Schedule and orchestrate notebooks |
# MAGIC
# MAGIC > **✨ Try it now**: Click on **Catalog** in the left sidebar. Browse to a catalog → schema → table to see columns, sample data, and lineage.

# COMMAND ----------

# DBTITLE 1,Compute
# MAGIC %md
# MAGIC ## 2. Compute: What Runs Your Code?
# MAGIC
# MAGIC | Compute Type | Best for | Key traits |
# MAGIC |---|---|---|
# MAGIC | **All-Purpose Cluster** | Interactive development in notebooks | You control the size, runtime, libraries |
# MAGIC | **Serverless Compute** | Quick, on-demand execution | No cluster management — Databricks handles it |
# MAGIC | **SQL Warehouse** | BI dashboards & SQL queries | Optimized for SQL, auto-scales |
# MAGIC | **Job Cluster** | Scheduled production workloads | Ephemeral — spins up per job, shuts down after |
# MAGIC
# MAGIC For this workshop we'll use **Serverless Compute** (or an all-purpose cluster your instructor set up) — attach and go.
# MAGIC
# MAGIC > **Key concept**: *Runtime versions* come with pre-installed ML libraries (PyTorch, TensorFlow, scikit-learn, XGBoost, MLflow). Choose a **ML Runtime** when creating clusters for data science work.

# COMMAND ----------

# DBTITLE 1,Unity Catalog
# MAGIC %md
# MAGIC ## 3. Unity Catalog — One Place for Everything
# MAGIC
# MAGIC Unity Catalog is Databricks' governance layer. It organizes **all** data and AI assets:
# MAGIC
# MAGIC ```
# MAGIC Catalog (e.g. main)
# MAGIC   └── Schema (e.g. ml_lifecycle_workshop)
# MAGIC         ├── Tables      → Delta tables with your data
# MAGIC         ├── Models      → ML models registered via MLflow
# MAGIC         ├── Volumes     → Unstructured / raw files (CSVs, PDFs, images)
# MAGIC         └── Functions   → UDFs and Feature Functions
# MAGIC ```
# MAGIC
# MAGIC **Why this matters for ML:**
# MAGIC - **Models live alongside the data** they were trained on → full lineage
# MAGIC - **Access control** (who can read data, deploy models) is managed centrally
# MAGIC - **Versioning** for both tables and models is built in
# MAGIC
# MAGIC > This workshop's raw policy data lands in the volume `/Volumes/main/ml_lifecycle_workshop/datasets/`, then gets loaded into a Delta table — you'll see this in Module 1.
# MAGIC
# MAGIC > **✨ Try it now**: In the Catalog Explorer, navigate to your catalog → schema. You'll see tables, volumes, and (after we train) models here.

# COMMAND ----------

# DBTITLE 1,Notebooks
# MAGIC %md
# MAGIC ## 4. Notebooks — Your ML Workbench
# MAGIC
# MAGIC **Key features for data scientists:**
# MAGIC - **Multi-language**: Python, SQL, Scala, R — mix in the same notebook with `%sql`, `%python` magic commands
# MAGIC - **Built-in visualizations**: Click the **+** on any table result to add charts
# MAGIC - **Collaboration**: Real-time co-editing, comments, version history
# MAGIC - **Variables explorer**: See DataFrames and variables in the right panel
# MAGIC - **Databricks Assistant**: AI help built into the editor
# MAGIC
# MAGIC > **Tip**: Use `display(df)` to show any Spark or pandas DataFrame with the rich table viewer.

# COMMAND ----------

# DBTITLE 1,Working from the CLI and IDE
# MAGIC %md
# MAGIC ## 5. Beyond the Browser — CLI & IDE
# MAGIC
# MAGIC You can drive the same platform from your **laptop**, not just the browser. This matters for automation, reproducibility, and CI/CD.
# MAGIC
# MAGIC ### Databricks CLI (unified)
# MAGIC ```bash
# MAGIC # macOS/Linux install (Homebrew or the official script)
# MAGIC brew install databricks
# MAGIC # or: curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
# MAGIC
# MAGIC # Authenticate once, then every command targets that profile
# MAGIC databricks auth login --host <workspace-url> --profile <profile>
# MAGIC
# MAGIC databricks catalogs list --profile <profile>
# MAGIC databricks experiments list --profile <profile>
# MAGIC databricks serving-endpoints list --profile <profile>
# MAGIC ```
# MAGIC > See **`docs/cli_quickstart.md`** in this project for the full 5-minute demo script.
# MAGIC
# MAGIC ### VS Code Extension
# MAGIC The **Databricks extension for VS Code** lets you:
# MAGIC - Develop notebooks locally, sync and run them on Databricks compute
# MAGIC - Debug Python with breakpoints
# MAGIC - Browse workspace files, clusters, and jobs
# MAGIC
# MAGIC ### And it all packages up
# MAGIC The same project can be deployed as a **Databricks Asset Bundle (DAB)** — a versioned, CI/CD-friendly definition of jobs and resources. We'll demo this in **Module 4**.

# COMMAND ----------

# DBTITLE 1,Let's get started!
# MAGIC %md
# MAGIC ## Ready? Let's Build!
# MAGIC
# MAGIC Now that you know your way around, let's start building our policy-lapse prediction model.
# MAGIC
# MAGIC **Next step**: [02_feature_engineering]($./02_feature_engineering) → Load data, explore, and engineer features