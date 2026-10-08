# Databricks notebook source
# DBTITLE 1,Module 5: Model Monitoring
# MAGIC %md
# MAGIC # Module 5: Model Monitoring (10 min)
# MAGIC
# MAGIC In this notebook we will:
# MAGIC 1. Prepare an **inference table** with predictions and ground truth labels
# MAGIC 2. Create a **Lakehouse Monitor** to track model quality and data drift
# MAGIC 3. Explore the auto-generated monitoring **dashboard**
# MAGIC 4. Discuss alerting and retraining triggers
# MAGIC
# MAGIC **Why monitor?** Models degrade over time as data distributions shift. Monitoring catches problems before they impact the business.
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/advanced/banners/mlflow-uc-end-to-end-advanced-7-v2.png?raw=true" width="1200">

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install --quiet databricks-sdk --upgrade
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Run shared setup
# MAGIC %run ./00_setup

# COMMAND ----------

# DBTITLE 1,Inference table intro
# MAGIC %md
# MAGIC ## Step 1: Prepare the Inference Table
# MAGIC
# MAGIC Lakehouse Monitoring needs a table with:
# MAGIC - **Predictions** from the model
# MAGIC - **Timestamps** for each inference
# MAGIC - **Ground truth labels** (when available) for quality metrics
# MAGIC - **Model version** to compare across versions
# MAGIC
# MAGIC We'll join our predictions with the original labels to create this table.

# COMMAND ----------

# DBTITLE 1,Create inference table for monitoring
import pyspark.sql.functions as F

predictions = spark.table("workshop_lapse_predictions")
labels = (
    spark.table("workshop_lapse_labels")
    .filter("split = 'test'")
    .select("policy_id", F.col("lapsed").alias("lapsed_label"))
)

monitoring_df = predictions.join(labels, on="policy_id", how="left")

(monitoring_df.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workshop_lapse_monitoring"))

# Enable Change Data Feed (required by Lakehouse Monitoring)
spark.sql("ALTER TABLE workshop_lapse_monitoring SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

print(f"Monitoring table created: {catalog}.{db}.workshop_lapse_monitoring")
display(spark.table("workshop_lapse_monitoring").select(
    "policy_id", "predicted_lapsed", "lapsed_label", "inference_timestamp", "model_version"
).limit(5))

# COMMAND ----------

# DBTITLE 1,Create monitor intro
# MAGIC %md
# MAGIC ## Step 2: Create a Lakehouse Monitor
# MAGIC
# MAGIC Databricks **Lakehouse Monitoring** lets you attach a monitor to any Delta table. For ML inference tables, it automatically computes:
# MAGIC - **Model quality metrics**: accuracy, precision, recall, F1 (when labels are available)
# MAGIC - **Data drift**: statistical tests comparing current vs baseline distributions
# MAGIC - **Data quality**: null rates, outliers, schema changes
# MAGIC
# MAGIC It generates a **DBSQL dashboard** automatically — no configuration needed!

# COMMAND ----------

# DBTITLE 1,Create the monitor
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    MonitorInferenceLog, MonitorInferenceLogProblemType
)

w = WorkspaceClient()
monitoring_table = f"{catalog}.{db}.workshop_lapse_monitoring"

try:
    info = w.quality_monitors.create(
        table_name=monitoring_table,
        inference_log=MonitorInferenceLog(
            problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
            prediction_col="predicted_lapsed",
            timestamp_col="inference_timestamp",
            granularities=["1 day"],
            model_id_col="model_version",
            label_col="lapsed_label",  # optional but enables quality metrics
        ),
        output_schema_name=f"{catalog}.{db}",
        assets_dir=f"/Workspace/Users/{current_user}/workshop_lapse_monitoring",
    )
    print(f"✅ Monitor created for {monitoring_table}")

except Exception as e:
    if "already exist" in str(e).lower():
        print(f"Monitor already exists for {monitoring_table}, retrieving...")
        info = w.quality_monitors.get(table_name=monitoring_table)
    else:
        raise

print(f"Monitor status: {info.status}")

# COMMAND ----------

# DBTITLE 1,Wait for refresh
# MAGIC %md
# MAGIC ## Step 3: Wait for the Initial Refresh
# MAGIC
# MAGIC When a monitor is first created, it triggers an initial refresh to compute metrics. This may take a few minutes.
# MAGIC
# MAGIC > **Note**: In a real workshop, the instructor may show this step pre-computed. The refresh generates two output tables:
# MAGIC > - `workshop_lapse_monitoring_profile_metrics` — statistical profiles per time window
# MAGIC > - `workshop_lapse_monitoring_drift_metrics` — drift scores vs baseline

# COMMAND ----------

# DBTITLE 1,Wait for monitor refresh
import time
from databricks.sdk.service.catalog import MonitorInfoStatus, MonitorRefreshInfoState

# Wait for monitor to become active
while info.status == MonitorInfoStatus.MONITOR_STATUS_PENDING:
    info = w.quality_monitors.get(table_name=monitoring_table)
    time.sleep(10)

print(f"Monitor status: {info.status}")

refreshes = w.quality_monitors.list_refreshes(table_name=monitoring_table).refreshes
if refreshes:
    latest = refreshes[0]
    print(f"Latest refresh state: {latest.state}")
    if latest.state in (MonitorRefreshInfoState.PENDING, MonitorRefreshInfoState.RUNNING):
        print("Refresh is running... (this can take 3-5 minutes)")
        print("You can proceed to the dashboard exploration while it completes.")
else:
    print("Triggering initial refresh...")
    w.quality_monitors.run_refresh(table_name=monitoring_table)
    print("Refresh triggered. This can take 3-5 minutes.")

# COMMAND ----------

# DBTITLE 1,Explore dashboard
# MAGIC %md
# MAGIC ## Step 4: Explore the Monitoring Dashboard
# MAGIC
# MAGIC > **✨ Try it now**:
# MAGIC > 1. Open **Catalog** in the left sidebar
# MAGIC > 2. Navigate to the `workshop_lapse_monitoring` table
# MAGIC > 3. Click the **Quality** tab
# MAGIC > 4. Click **View Dashboard** to open the auto-generated DBSQL dashboard
# MAGIC
# MAGIC The dashboard shows:
# MAGIC - **Inference volume** over time
# MAGIC - **Model performance** (accuracy, F1) per time window
# MAGIC - **Prediction drift** (confusion matrix, prediction distributions)
# MAGIC - **Feature drift** for each input column
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Step 5: (Conceptual) Alerting & Retraining
# MAGIC
# MAGIC In production, you'd set up:
# MAGIC 1. **Alerts**: SQL alerts on the drift/profile metrics tables (e.g., "alert me if F1 drops below 0.5")
# MAGIC 2. **Scheduled monitoring**: cron job to refresh the monitor daily
# MAGIC 3. **Retraining trigger**: when drift exceeds a threshold, kick off a retraining job
# MAGIC
# MAGIC This closes the ML feedback loop: **train → deploy → monitor → retrain**.

# COMMAND ----------

# DBTITLE 1,Cleanup helper (optional)
# === OPTIONAL CLEANUP ===
# Uncomment the lines below to remove the monitor, endpoint, online store, and workshop tables.
# Useful if you want to re-run the workshop from scratch.

# w.quality_monitors.delete(table_name=monitoring_table, purge_artifacts=True)
# print(f"Monitor deleted for {monitoring_table}")

# import mlflow.deployments
# from databricks.feature_engineering import FeatureEngineeringClient
# mlflow.deployments.get_deploy_client("databricks").delete_endpoint(endpoint_name)
# FeatureEngineeringClient().delete_online_store(name=online_store)

# for t in ["workshop_lapse_monitoring", "workshop_lapse_predictions",
#           "workshop_lapse_labels", "workshop_lapse_features",
#           "workshop_lapse_bronze"]:
#     spark.sql(f"DROP TABLE IF EXISTS {catalog}.{db}.{t}")
# spark.sql(f"DROP TABLE IF EXISTS {feature_table}_online")
# print("Workshop tables cleaned up!")

# COMMAND ----------

# DBTITLE 1,Workshop Wrap-Up
# MAGIC %md
# MAGIC ## 🎉 Workshop Complete!
# MAGIC
# MAGIC ### What we covered in 2 hours:
# MAGIC
# MAGIC | Module | Topic | Key Databricks Feature |
# MAGIC |--------|-------|------------------------|
# MAGIC | 0 | Platform Orientation | Workspace, Catalog, Compute, CLI/IDE |
# MAGIC | 1 | Feature Engineering | Delta, pandas-on-Spark, Feature Store |
# MAGIC | 2 | Model Training | MLflow Experiments, autolog, FeatureLookup lineage |
# MAGIC | 3 | Model Registration | Models in Unity Catalog, aliases |
# MAGIC | 4 | Inference / MLOps | fe.score_batch, Model Serving, Asset Bundles |
# MAGIC | 5 | Model Monitoring | Lakehouse Monitoring, auto-dashboards |
# MAGIC
# MAGIC ### Where to go next:
# MAGIC - **Challenger validation**: Automate model comparison and promotion pipelines
# MAGIC - **CI/CD**: Use Databricks Asset Bundles (DABs) for Git-based MLOps (see `bundle/`)
# MAGIC - **Advanced monitoring**: Custom metrics, drift detection with automated retraining
# MAGIC
# MAGIC ### Resources:
# MAGIC - [Databricks ML Documentation](https://docs.databricks.com/en/machine-learning/index.html)
# MAGIC - [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
# MAGIC - [dbdemos.ai](https://www.dbdemos.ai/) — full end-to-end demos you can install in your workspace