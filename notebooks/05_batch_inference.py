# Databricks notebook source
# DBTITLE 1,Module 4: Inference & MLOps — Part A: Batch Scoring
# MAGIC %md
# MAGIC # Inference & MLOps — Part A: Batch Scoring
# MAGIC
# MAGIC In this notebook we will:
# MAGIC 1. Build a keys-only scoring input (new policies to score)
# MAGIC 2. Score them with **`fe.score_batch`** — features are auto-joined from the Feature Store
# MAGIC 3. Save predictions to a Delta table
# MAGIC 4. Discuss how to schedule this as a recurring job
# MAGIC
# MAGIC **Batch inference** is the most common deployment pattern — score new data on a schedule (hourly, daily) and write results to a table for downstream dashboards and applications.
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/mlops-uc-end2end-5-v2.png?raw=true" width="1200">

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install --quiet "databricks-feature-engineering>=0.16.0" lightgbm --upgrade
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Run shared setup
# MAGIC %run ./00_setup

# COMMAND ----------

# DBTITLE 1,Prepare inference keys intro
# MAGIC %md
# MAGIC ## Step 1: Prepare Inference Input (keys only)
# MAGIC
# MAGIC Because the model was logged with the Feature Store, we only need the **primary keys** (`policy_id`) to score — `fe.score_batch` auto-joins the features. We simulate "new policies" with our held-out test split.

# COMMAND ----------

# DBTITLE 1,Build the scoring input
inference_keys = (
    spark.table("workshop_lapse_labels")
    .filter("split = 'test'")
    .select("policy_id")
)
print(f"Policies to score: {inference_keys.count()}")
display(inference_keys.limit(5))

# COMMAND ----------

# DBTITLE 1,Score with fe.score_batch intro
# MAGIC %md
# MAGIC ## Step 2: Batch Score with the Champion Model
# MAGIC
# MAGIC We reference the model by its **alias** (`@Champion`), not a version number — so promoting a new model is just re-pointing the alias, with zero code changes. `fe.score_batch` loads the model, auto-joins the features registered in its lineage, and scores at scale on Spark.

# COMMAND ----------

# DBTITLE 1,Score and save predictions
from databricks.feature_engineering import FeatureEngineeringClient
import pyspark.sql.functions as F

fe = FeatureEngineeringClient(model_registry_uri="databricks-uc")
champion_version = client.get_model_version_by_alias(model_name, "Champion").version

scored = (
    fe.score_batch(model_uri=f"models:/{model_name}@Champion", df=inference_keys, result_type="string")
    .withColumnRenamed("prediction", "predicted_lapsed")
    .withColumn("inference_timestamp", F.current_timestamp())
    .withColumn("model_version", F.lit(champion_version))
)

predictions_table = "workshop_lapse_predictions"
scored.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(predictions_table)

print(f"✅ Predictions saved to {catalog}.{db}.{predictions_table}: {spark.table(predictions_table).count()} rows")
display(
    spark.table(predictions_table)
    .select("policy_id", "predicted_lapsed", "inference_timestamp", "model_version")
    .limit(10)
)

# COMMAND ----------

# DBTITLE 1,Scheduling
# MAGIC %md
# MAGIC ## Step 3: (Conceptual) Schedule as a Recurring Job
# MAGIC
# MAGIC In production, you'd schedule this notebook to run on a cadence:
# MAGIC
# MAGIC 1. **Lakeflow Jobs**: Click **Schedule** in the top-right of this notebook to create a job
# MAGIC 2. Set a **trigger**: cron schedule (e.g., daily at 6 AM) or file arrival
# MAGIC 3. **Job clusters / serverless**: ephemeral compute that spins up, runs, and shuts down automatically
# MAGIC 4. **Alerts**: get notified on failure
# MAGIC
# MAGIC > **In Part C of this module** we'll make this real with a **Databricks Asset Bundle** that packages train → register → score as a scheduled, version-controlled pipeline.

# COMMAND ----------

# DBTITLE 1,Summary and next step
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC **What we did:**
# MAGIC - Built a keys-only scoring input (`policy_id`)
# MAGIC - Scored with `fe.score_batch` — features auto-joined from the Feature Store
# MAGIC - Saved predictions to a Delta table with timestamp + model version
# MAGIC
# MAGIC > **Key takeaway**: With a Feature Store-backed model you score by **keys only** and reference the model by **alias** (`@Champion`) — the platform handles the feature join and lets you swap models without code changes.
# MAGIC
# MAGIC **Next step**: [06_model_serving]($./06_model_serving) → Serve the Champion model in real time