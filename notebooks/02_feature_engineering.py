# Databricks notebook source
# DBTITLE 1,Module 1: Feature Engineering
# MAGIC %md
# MAGIC # Data Exploration & Feature Engineering (20 - 25 min)
# MAGIC
# MAGIC In this notebook we will:
# MAGIC 1. Load raw policy data from Unity Catalog
# MAGIC 2. Explore the data using SQL and built-in visualizations
# MAGIC 3. Engineer features using **pandas-on-Spark**
# MAGIC 4. Register a **Feature Store** table with `FeatureEngineeringClient`
# MAGIC 5. Save a label spine table with train/test splits
# MAGIC
# MAGIC **Use case**: Predict **policy lapse** (non-renewal) for an insurance book of business.
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/mlops-uc-end2end-1-v2.png?raw=true" width="1200">

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install --quiet "databricks-feature-engineering>=0.16.0" --upgrade
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Run shared setup
# MAGIC %run ./00_setup

# COMMAND ----------

# DBTITLE 1,Step 1: Load the Raw Data
# MAGIC %md
# MAGIC ## Step 1: Load the Raw Policy Data
# MAGIC
# MAGIC Our data engineering team landed raw policy records in a Volume, which `00_setup` loaded into a Bronze table. Let's load it and take a look.
# MAGIC
# MAGIC > **💡 Tip**: `display()` gives you a rich table viewer. Click the **+** tab to create visualizations!

# COMMAND ----------

# DBTITLE 1,Load Bronze table
policy_df = spark.table("workshop_lapse_bronze")
display(policy_df)

# COMMAND ----------

# DBTITLE 1,Quick EDA with SQL
# MAGIC %md
# MAGIC ## Step 2: Quick Exploratory Data Analysis
# MAGIC
# MAGIC Let's understand the data before building features. Use SQL for fast exploration — you can mix Python and SQL in the same notebook!

# COMMAND ----------

# DBTITLE 1,Lapse distribution
# MAGIC %sql
# MAGIC -- How balanced is our target variable?
# MAGIC SELECT lapsed, COUNT(*) AS count
# MAGIC FROM workshop_lapse_bronze
# MAGIC GROUP BY lapsed

# COMMAND ----------

# DBTITLE 1,Lapse by contract term
# MAGIC %sql
# MAGIC -- Lapse rate by contract term
# MAGIC -- After running, click "+" to create a bar chart!
# MAGIC SELECT contract_term, lapsed, COUNT(*) AS count
# MAGIC FROM workshop_lapse_bronze
# MAGIC GROUP BY contract_term, lapsed
# MAGIC ORDER BY contract_term

# COMMAND ----------

# DBTITLE 1,Lapse by payment method
# MAGIC %sql
# MAGIC -- Do customers without auto-pay lapse more?
# MAGIC SELECT payment_method, lapsed, COUNT(*) AS count
# MAGIC FROM workshop_lapse_bronze
# MAGIC GROUP BY payment_method, lapsed
# MAGIC ORDER BY payment_method

# COMMAND ----------

# DBTITLE 1,EDA with pandas-on-Spark
# MAGIC %md
# MAGIC ## Step 3: Visualize with pandas-on-Spark
# MAGIC
# MAGIC pandas-on-Spark lets you write familiar **pandas** code that runs distributed on Spark under the hood. No need to learn a new API — just call `.pandas_api()` on a Spark DataFrame.

# COMMAND ----------

# DBTITLE 1,Pandas-on-Spark pie chart
# pandas-on-Spark: write pandas code, Spark executes it at scale
import pyspark.pandas as ps

ps.set_option("plotting.backend", "matplotlib")
policy_psdf = policy_df.pandas_api()
policy_psdf["policy_type"].value_counts().plot.pie(title="Policy Type Distribution")

# COMMAND ----------

# DBTITLE 1,Feature engineering intro
# MAGIC %md
# MAGIC ## Step 4: Feature Engineering
# MAGIC
# MAGIC We'll create a cleaning and featurization function:
# MAGIC 1. **Impute** missing numerical values
# MAGIC 2. **Derive** `premium_per_policy` (premium adjusted for bundling)
# MAGIC 3. **Derive** `claims_per_year` (claims normalized by tenure)

# COMMAND ----------

# DBTITLE 1,Define featurization function
import pyspark.sql.functions as F
from pyspark.sql import DataFrame


def build_lapse_features(dataDF: DataFrame) -> DataFrame:
    """Clean + featurize raw policy data using pandas-on-Spark."""
    psdf = dataDF.pandas_api()

    # 1. Impute numeric nulls
    psdf = psdf.fillna({
        "tenure_months": 0, "monthly_premium": 0.0, "annual_premium": 0.0,
        "late_payments_12m": 0, "prior_claims_count": 0, "claims_last_year": 0,
    })

    # 2. Premium per policy (bundling-adjusted)
    psdf["premium_per_policy"] = psdf["annual_premium"] / psdf["num_policies"].clip(lower=1)

    # 3. Claims rate over tenure (in years)
    psdf["claims_per_year"] = psdf["prior_claims_count"] / (psdf["tenure_months"].clip(lower=1) / 12.0)

    return psdf.to_spark()

# COMMAND ----------

# DBTITLE 1,Apply features and display
features_df = build_lapse_features(spark.table("workshop_lapse_bronze")).drop("lapsed")
display(features_df)

# COMMAND ----------

# DBTITLE 1,Feature Store intro
# MAGIC %md
# MAGIC ## Step 5: Register a Feature Store Table
# MAGIC
# MAGIC Instead of a plain Delta table, we'll register our features with the **`FeatureEngineeringClient`**. This gives us:
# MAGIC - **Lineage**: Unity Catalog tracks which models consume which features
# MAGIC - **Consistency**: the same feature definitions are used at training, batch scoring, and real-time serving (no training/serving skew)
# MAGIC - **Reuse**: other teams can discover and reuse these features
# MAGIC
# MAGIC The table is keyed on `policy_id`. (This is an *intro* — we'll use the table's `FeatureLookup` in the next module.)

# COMMAND ----------

# DBTITLE 1,Create the feature table
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient(model_registry_uri="databricks-uc")

try:
    fe.create_table(
        name=feature_table,
        primary_keys=["policy_id"],
        df=features_df,
        description="Policy-level features for lapse prediction (ML lifecycle workshop).",
    )
    print(f"Created feature table {feature_table}")
except Exception as e:
    if "already exists" in str(e).lower():
        fe.write_table(name=feature_table, df=features_df, mode="merge")
        print(f"Feature table exists — upserted rows into {feature_table}")
    else:
        raise

# COMMAND ----------

# DBTITLE 1,Create the label spine (with train/test split)
# MAGIC %md
# MAGIC ## Step 6: Save the Label Spine
# MAGIC
# MAGIC The **label** (`lapsed`) lives separately from the features. We create a small "spine" table of `policy_id` + label, with an 80/20 train/test split. In Module 2 we'll join features onto this spine via `FeatureLookup`.

# COMMAND ----------

# DBTITLE 1,Write the label spine
labels = (
    spark.table("workshop_lapse_bronze")
    .select("policy_id", "lapsed")
    .withColumn("random", F.rand(seed=42))
    .withColumn("split", F.when(F.col("random") < 0.8, "train").otherwise("test"))
    .drop("random")
)
labels.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("workshop_lapse_labels")
print("Label spine saved to workshop_lapse_labels")
display(spark.table("workshop_lapse_labels").groupBy("split", "lapsed").count().orderBy("split", "lapsed"))

# COMMAND ----------

# DBTITLE 1,Summary and next step
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC **What we did:**
# MAGIC - Loaded raw policy data from a Unity Catalog table (sourced from a Volume)
# MAGIC - Explored data with SQL and pandas-on-Spark visualizations
# MAGIC - Engineered features (imputation, premium-per-policy, claims-per-year)
# MAGIC - Registered a **Feature Store table** keyed on `policy_id`
# MAGIC - Saved a label spine with train/test splits
# MAGIC
# MAGIC > **✨ Try it now**: In Catalog Explorer, open your schema → **Features** tab. You'll see `workshop_lapse_features` with its primary key and (soon) its model lineage.
# MAGIC
# MAGIC **Next step**: [03_model_training]($./03_model_training) → Build a training set with FeatureLookup and track training with MLflow