# Databricks notebook source
# DBTITLE 1,Module 3: Model Registration
# MAGIC %md
# MAGIC # Model Registration in Unity Catalog (15 min)
# MAGIC
# MAGIC In this notebook we will:
# MAGIC 1. Find the **best run** from our experiment programmatically
# MAGIC 2. **Register** the model to Unity Catalog
# MAGIC 3. Add **descriptions** and **tags** for governance
# MAGIC 4. Set a **Champion alias** to mark it as production-ready
# MAGIC
# MAGIC **Why register models?**
# MAGIC - Central discovery: anyone can find and use approved models
# MAGIC - Governance: track who trained what, with which data
# MAGIC - Versioning: roll back to any previous version
# MAGIC - Lineage: see the path from feature table → model → predictions
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/mlops-uc-end2end-3-v2.png?raw=true" width="1200">

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install --quiet "databricks-feature-engineering>=0.16.0" --upgrade
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Run shared setup
# MAGIC %run ./00_setup

# COMMAND ----------

# DBTITLE 1,Find best run intro
# MAGIC %md
# MAGIC ## Step 1: Find the Best Run Programmatically
# MAGIC
# MAGIC We trained a model in the previous notebook. Now we'll use `mlflow.search_runs()` to find the best run by validation F1 score.
# MAGIC
# MAGIC This is how production MLOps works — automated pipelines select the best model, not humans clicking through the UI.

# COMMAND ----------

# DBTITLE 1,Search best run
import mlflow

experiment_name = f"{xp_path}/{xp_name}"
mlflow.set_experiment(experiment_name)

best_run = mlflow.search_runs(
    order_by=["metrics.val_f1_score DESC"],
    max_results=1,
    filter_string="status = 'FINISHED'",
)

run_id = best_run.iloc[0]["run_id"]
val_f1 = best_run.iloc[0].get("metrics.val_f1_score", float("nan"))
print(f"Best run ID: {run_id}")
print(f"Validation F1: {val_f1}")
best_run[["run_id", "metrics.val_f1_score", "metrics.val_accuracy_score"]]

# COMMAND ----------

# DBTITLE 1,Register model intro
# MAGIC %md
# MAGIC ## Step 2: Register the Model to Unity Catalog
# MAGIC
# MAGIC Registering a model is like "committing" it to the catalog. It gets:
# MAGIC - A **3-level name**: `catalog.schema.model_name`
# MAGIC - **Version numbers**: auto-incremented with each registration
# MAGIC - **Full lineage**: links back to the experiment run, the feature table, and the data

# COMMAND ----------

# DBTITLE 1,Register model
# Register the best model from our experiment to Unity Catalog.
# The model was logged with fe.log_model, so its feature lineage travels with it.
print(f"Registering model to {model_name}...")
model_details = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=model_name)
print(f"\n✅ Registered as {model_name} version {model_details.version}")

# COMMAND ----------

# DBTITLE 1,Add description intro
# MAGIC %md
# MAGIC ## Step 3: Document the Model
# MAGIC
# MAGIC Good governance means documenting what a model does, how well it performs, and who owns it. We'll add descriptions at both the **model level** and the **version level**.

# COMMAND ----------

# DBTITLE 1,Add descriptions and tags
from mlflow import MlflowClient

client = MlflowClient()

client.update_registered_model(
    name=model_name,
    description="Predicts policy lapse (non-renewal) for the insurance book of business. "
                "Trained on the workshop_lapse_features Feature Store table (FeatureLookup) using LightGBM.",
)

client.update_model_version(
    name=model_name,
    version=model_details.version,
    description=f"LightGBM model with validation F1={val_f1}. Trained during the ML lifecycle workshop.",
)

print("Model and version descriptions updated!")

# COMMAND ----------

# DBTITLE 1,Set alias intro
# MAGIC %md
# MAGIC ## Step 4: Set the Champion Alias
# MAGIC
# MAGIC **Aliases** are human-readable labels that point to a specific model version. Common pattern:
# MAGIC - `Champion` → the model currently serving production traffic
# MAGIC - `Challenger` → a new candidate being validated against the Champion
# MAGIC
# MAGIC Instead of hardcoding version numbers in your code, you reference `@Champion` — and updating production is just re-pointing the alias.

# COMMAND ----------

# DBTITLE 1,Set Champion alias
client.set_registered_model_alias(name=model_name, alias="Champion", version=model_details.version)

print(f"✅ Alias 'Champion' now points to {model_name} version {model_details.version}")
print(f"\nLoad it anywhere with: models:/{model_name}@Champion")

# COMMAND ----------

# DBTITLE 1,Explore in Catalog Explorer
# MAGIC %md
# MAGIC ## Step 5: Explore in Catalog Explorer
# MAGIC
# MAGIC > **✨ Try it now**:
# MAGIC > 1. Open **Catalog** in the left sidebar
# MAGIC > 2. Navigate to your catalog → schema → **Models** section
# MAGIC > 3. Click on `workshop_lapse_model`
# MAGIC > 4. You'll see:
# MAGIC >    - Version history with descriptions
# MAGIC >    - The `Champion` alias badge
# MAGIC >    - **Lineage tab** showing the feature table and experiment connected to this model
# MAGIC >    - **Schema tab** showing input/output signature
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## What Comes Next in Production?
# MAGIC
# MAGIC In a real MLOps workflow, you would also:
# MAGIC - **Validate the Challenger**: run automated tests comparing Challenger vs Champion
# MAGIC - **Approval gates**: require sign-off before promoting a model
# MAGIC - **CI/CD**: trigger registration from a Git-based pipeline (Databricks Asset Bundles — demoed in Module 4)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Next step**: [05_batch_inference]($./05_batch_inference) → Use the Champion model to score new policies