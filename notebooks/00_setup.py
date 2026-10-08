# Databricks notebook source
# DBTITLE 1,Workshop Setup — Configuration
# ============================================================
# ML Lifecycle Workshop — Shared Setup
# ============================================================
# TO-DO: 
#   change catalog/schema below to match your environment.
#   Each users can also override these before running.
# ============================================================

# --- Configuration (EDIT THESE) --------------------------------
catalog = "main"                    # Unity Catalog catalog
db      = "ml_lifecycle_workshop"   # Schema inside the catalog
volume  = "datasets"                # Volume for raw data files
# ---------------------------------------------------------------

import re, warnings, logging
import mlflow
from mlflow import MlflowClient

warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)

# Current user (for experiment paths)
current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

# Set UC as the model registry
mlflow.set_registry_uri("databricks-uc")

# Ensure schema + volume exist, then select the catalog/schema
spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{db}`")
spark.sql(f"CREATE VOLUME IF NOT EXISTS `{catalog}`.`{db}`.`{volume}`")
spark.sql(f"USE CATALOG `{catalog}`")
spark.sql(f"USE SCHEMA `{db}`")

volume_path = f"/Volumes/{catalog}/{db}/{volume}"
print(f"Using {catalog}.{db} | Volume: {volume_path} | User: {current_user}")

# COMMAND ----------

# DBTITLE 1,Synthetic Data Generator (insurance policy lapse)
import numpy as np
import pandas as pd

GENDERS = ["Male", "Female"]
MARITAL = ["Single", "Married", "Divorced"]
REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
POLICY_TYPES = ["Auto", "Home", "Life", "Renters"]
CONTRACT_TERMS = ["Monthly", "Annual", "Biennial"]
COVERAGE_LEVELS = ["Basic", "Standard", "Premium"]
PAYMENT_METHODS = ["Auto-pay", "Credit card", "Check", "Bank transfer"]


def generate_lapse_data(n_rows: int = 7000, seed: int = 42) -> pd.DataFrame:
    """Synthetic insurance policy book with a lapse label.

    Lapse propensity is correlated with monthly term, no auto-pay, late
    payments, low tenure, single policy, and prior claims — so the model
    learns believable (not perfect) signal. Deterministic given the seed.
    (Mirrors scripts/generate_lapse_data.py, which is unit-tested.)
    """
    rng = np.random.default_rng(seed)

    contract_term = rng.choice(CONTRACT_TERMS, n_rows, p=[0.50, 0.40, 0.10])
    payment_method = rng.choice(PAYMENT_METHODS, n_rows, p=[0.40, 0.30, 0.15, 0.15])
    auto_pay = np.where(
        payment_method == "Auto-pay", "Yes",
        rng.choice(["Yes", "No"], n_rows, p=[0.30, 0.70]),
    )
    tenure_months = rng.integers(1, 120, n_rows)
    num_policies = rng.integers(1, 5, n_rows)
    num_optional_coverages = rng.integers(0, 6, n_rows)
    late_payments_12m = rng.poisson(0.7, n_rows)
    prior_claims_count = rng.poisson(1.0, n_rows)
    monthly_premium = np.round(rng.uniform(40, 400, n_rows), 2)

    logit = (
        -1.3
        + 0.95 * (contract_term == "Monthly")
        - 0.70 * (contract_term == "Biennial")
        + 0.85 * (auto_pay == "No")
        + 0.25 * late_payments_12m
        - 0.015 * tenure_months
        - 0.45 * (num_policies >= 2)
        + 0.15 * prior_claims_count
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    lapsed = np.where(rng.uniform(0, 1, n_rows) < prob, "Yes", "No")

    return pd.DataFrame({
        "policy_id": [f"POL-{i:06d}" for i in range(1, n_rows + 1)],
        "age": rng.integers(18, 85, n_rows),
        "gender": rng.choice(GENDERS, n_rows),
        "marital_status": rng.choice(MARITAL, n_rows, p=[0.35, 0.50, 0.15]),
        "region": rng.choice(REGIONS, n_rows),
        "policy_type": rng.choice(POLICY_TYPES, n_rows, p=[0.45, 0.30, 0.15, 0.10]),
        "contract_term": contract_term,
        "tenure_months": tenure_months,
        "monthly_premium": monthly_premium,
        "annual_premium": np.round(monthly_premium * 12 * rng.uniform(0.90, 1.0, n_rows), 2),
        "deductible": rng.choice([250, 500, 1000, 2000], n_rows),
        "coverage_level": rng.choice(COVERAGE_LEVELS, n_rows, p=[0.30, 0.50, 0.20]),
        "num_policies": num_policies,
        "num_optional_coverages": num_optional_coverages,
        "payment_method": payment_method,
        "auto_pay": auto_pay,
        "paperless_billing": rng.choice(["Yes", "No"], n_rows, p=[0.60, 0.40]),
        "prior_claims_count": prior_claims_count,
        "claims_last_year": np.minimum(prior_claims_count, rng.integers(0, 4, n_rows)),
        "late_payments_12m": late_payments_12m,
        "has_telematics": rng.choice(["Yes", "No"], n_rows, p=[0.30, 0.70]),
        "lapsed": lapsed,
    })

# COMMAND ----------

# DBTITLE 1,Generate raw data → Volume → Bronze table
bronze_table = "workshop_lapse_bronze"
raw_path = f"{volume_path}/policies_raw.csv"

if not spark.catalog.tableExists(bronze_table):
    print("Generating synthetic insurance policy data...")
    pdf = generate_lapse_data(n_rows=7000, seed=42)

    # Write the raw file to the UC volume (mirrors a real ingestion landing zone)
    pdf.to_csv(raw_path, index=False)
    print(f"Wrote raw file to {raw_path}")

    # Read the raw file back with Spark into the bronze Delta table
    bronze = (spark.read.option("header", True).option("inferSchema", True).csv(raw_path))
    bronze.write.mode("overwrite").saveAsTable(bronze_table)
    print(f"Created {catalog}.{db}.{bronze_table} ({bronze.count()} rows)")
else:
    print(f"Table {catalog}.{db}.{bronze_table} already exists — skipping generation.")

# COMMAND ----------

# DBTITLE 1,Shared variables for downstream notebooks
# Variables available to all notebooks that %run this setup
feature_table = f"{catalog}.{db}.workshop_lapse_features"
model_name    = f"{catalog}.{db}.workshop_lapse_model"
xp_name       = "workshop_lapse_experiment"
xp_path       = f"/Users/{current_user}/mlflow_experiments"
endpoint_name = "ml-lifecycle-workshop-lapse"
online_store  = "ml-lifecycle-workshop-online-store"   # DNS-compliant: hyphens only, no underscores

client = MlflowClient()
print(f"Model will be registered as: {model_name}")
print(f"Feature table: {feature_table}")
print(f"Experiment path: {xp_path}/{xp_name}")
print(f"Serving endpoint: {endpoint_name} | Online store: {online_store}")