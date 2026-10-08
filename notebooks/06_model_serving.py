# Databricks notebook source
# DBTITLE 1,Module 4: Inference & MLOps — Part B: Real-Time Serving
# MAGIC %md
# MAGIC # Module 4: Inference & MLOps — Part B: Real-Time Serving (part of 25 min)
# MAGIC
# MAGIC In this notebook we will:
# MAGIC 1. Publish the Feature Store table to an **online store** (Lakebase) for low-latency lookups
# MAGIC 2. Deploy the Champion model to a **Model Serving endpoint**
# MAGIC 3. Send a **test request** — the endpoint auto-looks-up features by `policy_id`
# MAGIC
# MAGIC **Real-time serving** powers decisions at the moment of a user action — e.g., a retention offer surfaced while a policyholder is in the renewal flow.
# MAGIC
# MAGIC > **⏱️ Instructor note**: the online store and endpoint take **minutes to provision** on first creation. **Pre-warm them before the session** (run this notebook once beforehand) so only the test request runs live. See the project `README.md` for pre-session steps.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install --quiet "databricks-feature-engineering>=0.16.0" --upgrade
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Run shared setup
# MAGIC %run ./00_setup

# COMMAND ----------

# DBTITLE 1,Publish prerequisites intro
# MAGIC %md
# MAGIC ## Step 1: Prepare the Feature Table for Online Publishing
# MAGIC
# MAGIC Continuous online sync requires **Change Data Feed** enabled and the primary key marked **NOT NULL**.

# COMMAND ----------

# DBTITLE 1,Enable CDF + NOT NULL
spark.sql(f"ALTER TABLE {feature_table} SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')")
spark.sql(f"ALTER TABLE {feature_table} ALTER COLUMN policy_id SET NOT NULL")
print("CDF enabled and policy_id set NOT NULL.")

# COMMAND ----------

# DBTITLE 1,Create online store + publish intro
# MAGIC %md
# MAGIC ## Step 2: Create the Online Store and Publish
# MAGIC
# MAGIC The **online store** (backed by Databricks Lakebase) serves feature values with sub-millisecond latency. We publish the offline Feature Store table to it in `CONTINUOUS` mode so it stays in sync on every Delta commit.
# MAGIC
# MAGIC > The online store name must be **DNS-compliant** (hyphens, no underscores) — set in `00_setup` as `ml-lifecycle-workshop-online-store`.

# COMMAND ----------

# DBTITLE 1,Create online store and publish table
from databricks.feature_engineering import FeatureEngineeringClient
import time

fe = FeatureEngineeringClient(model_registry_uri="databricks-uc")

# Guard: online store names must be DNS-compliant (hyphens only).
assert "_" not in online_store, f"online store name must use hyphens, not underscores: {online_store}"

# get_online_store returns None (does NOT raise) when the store is absent.
store = fe.get_online_store(name=online_store)
if store is None:
    print("Creating online store (~5 min first time)...")
    fe.create_online_store(name=online_store, capacity="CU_1")
    time.sleep(300)
    store = fe.get_online_store(name=online_store)

fe.publish_table(
    online_store=store,
    source_table_name=feature_table,
    online_table_name=f"{feature_table}_online",
    publish_mode="CONTINUOUS",
)
print(f"✅ Published {feature_table} → {feature_table}_online")

# COMMAND ----------

# DBTITLE 1,Create serving endpoint intro
# MAGIC %md
# MAGIC ## Step 3: Deploy the Model Serving Endpoint
# MAGIC
# MAGIC We create a Model Serving endpoint for the Champion version. Because the model carries feature lineage and the feature table is online, the endpoint performs **automatic feature lookup** — the caller passes only `policy_id`.

# COMMAND ----------

# DBTITLE 1,Create the endpoint
import mlflow.deployments

deploy = mlflow.deployments.get_deploy_client("databricks")
champion_version = client.get_model_version_by_alias(model_name, "Champion").version

try:
    deploy.create_endpoint(
        name=endpoint_name,
        config={
            "served_entities": [{
                "entity_name": model_name,
                "entity_version": champion_version,
                "workload_size": "Small",
                "scale_to_zero_enabled": True,
            }]
        },
    )
    print(f"Creating endpoint {endpoint_name} (first deploy ~5–10 min)...")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"Endpoint {endpoint_name} already exists.")
    else:
        raise

# COMMAND ----------

# DBTITLE 1,Readiness check intro
# MAGIC %md
# MAGIC ## Step 4: Wait for Readiness, Then Query
# MAGIC
# MAGIC A serving endpoint reports two independent fields: **`state.ready`** (can it serve?) and **`state.config_update`** (is a config change in flight?). We wait for `state.ready == "READY"` before querying.
# MAGIC
# MAGIC The request body carries **only `policy_id`** — the endpoint auto-joins features from the online store.

# COMMAND ----------

# DBTITLE 1,Wait for READY
import time

for _ in range(60):  # up to ~10 min
    status = deploy.get_endpoint(endpoint_name)
    ready = status.get("state", {}).get("ready")
    print(f"state.ready = {ready}")
    if ready == "READY":
        break
    time.sleep(10)

# COMMAND ----------

# DBTITLE 1,Send a test request
resp = deploy.predict(
    endpoint=endpoint_name,
    inputs={"dataframe_records": [{"policy_id": "POL-000001"}, {"policy_id": "POL-000002"}]},
)

print(resp)

# COMMAND ----------

# DBTITLE 1,Summary and next step
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC **What we did:**
# MAGIC - Enabled CDF + NOT NULL and published the feature table to a Lakebase **online store**
# MAGIC - Deployed the Champion model to a **Model Serving endpoint**
# MAGIC - Queried it with **keys only** — features auto-joined online
# MAGIC
# MAGIC > **Key takeaway**: The *same* feature definitions used in training and batch scoring resolve the endpoint's features in real time — one definition, no training/serving skew.
# MAGIC
# MAGIC **Next step**: [07_model_monitoring]($./07_model_monitoring) → Monitor predictions for drift and quality