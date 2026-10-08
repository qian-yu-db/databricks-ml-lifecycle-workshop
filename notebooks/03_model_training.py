# Databricks notebook source
# DBTITLE 1,Module 2: Model Training
# MAGIC %md
# MAGIC # Model Training with MLflow (25–30 min)
# MAGIC
# MAGIC In this notebook we will:
# MAGIC 1. Build a **training set** from our Feature Store table using `FeatureLookup`
# MAGIC 2. Build a **scikit-learn preprocessing pipeline**
# MAGIC 3. Train a **LightGBM** classifier
# MAGIC 4. Track everything with **MLflow** (parameters, metrics, artifacts) and capture **feature lineage**
# MAGIC 5. Evaluate the model and view results in the Experiments UI
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/mlops-uc-end2end-2-v2.png?raw=true" width="1200">

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install --quiet "databricks-feature-engineering>=0.16.0" lightgbm optuna --upgrade
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Run shared setup
# MAGIC %run ./00_setup

# COMMAND ----------

# DBTITLE 1,Create MLflow Experiment
# MAGIC %md
# MAGIC ## Step 1: Set Up the MLflow Experiment
# MAGIC
# MAGIC **MLflow** is Databricks' open-source platform for the ML lifecycle. An **experiment** groups multiple training runs so you can compare them.
# MAGIC
# MAGIC > **Key concepts:**
# MAGIC > - **Experiment** = a named container for runs (like a folder for your training attempts)
# MAGIC > - **Run** = one execution of training code (logs params, metrics, model artifacts)
# MAGIC > - **Autolog** = automatic logging of everything — one line of code!

# COMMAND ----------

# DBTITLE 1,Set MLflow experiment
import mlflow

experiment_name = f"{xp_path}/{xp_name}"

try:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
except Exception:
    print(f"Creating experiment: {experiment_name}")
    experiment_id = mlflow.create_experiment(name=experiment_name)

mlflow.set_experiment(experiment_name)
print(f"Experiment: {experiment_name} (ID: {experiment_id})")

# COMMAND ----------

# DBTITLE 1,Build training set with FeatureLookup
# MAGIC %md
# MAGIC ## Step 2: Build the Training Set with FeatureLookup
# MAGIC
# MAGIC We start from the **label spine** (`policy_id` + `lapsed`) and let `FeatureLookup` join the features from our Feature Store table. This is the key to **no training/serving skew** — the same lookup definition is reused at scoring and serving time, and it captures **feature lineage** on the model.

# COMMAND ----------

# DBTITLE 1,Create the training set
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

fe = FeatureEngineeringClient(model_registry_uri="databricks-uc")

train_spine = (
    spark.table("workshop_lapse_labels")
    .filter("split = 'train'")
    .select("policy_id", "lapsed")
)

training_set = fe.create_training_set(
    df=train_spine,
    feature_lookups=[FeatureLookup(table_name=feature_table, lookup_key="policy_id")],
    label="lapsed",
    exclude_columns=["policy_id"],
)
training_pd = training_set.load_df().toPandas()
print(f"Training rows: {len(training_pd)} | columns: {list(training_pd.columns)}")

# COMMAND ----------

# DBTITLE 1,Preprocessing intro
# MAGIC %md
# MAGIC ## Step 3: Build a Preprocessing Pipeline
# MAGIC
# MAGIC We use scikit-learn's `ColumnTransformer`, building the column lists **dynamically** from the training data so they can't drift:
# MAGIC - **Numerical columns** → impute missing values + standardize
# MAGIC - **Categorical columns** → one-hot encode

# COMMAND ----------

# DBTITLE 1,Define preprocessing + model pipeline
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier

label_col = "lapsed"
X = training_pd.drop(columns=[label_col])
# Keep the label as "Yes"/"No" strings: predictions stay human-readable and match the
# ground-truth labels used downstream in batch scoring (05) and monitoring (07).
y = training_pd[label_col]

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
print(f"Numeric: {num_cols}")
print(f"Categorical: {cat_cols}")

preprocessor = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="mean")), ("scale", StandardScaler())]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
], remainder="drop")

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LGBMClassifier(
        n_estimators=250, max_depth=8, learning_rate=0.07, num_leaves=64, random_state=42,
    )),
])
print("Preprocessing + LightGBM pipeline defined!")

# COMMAND ----------

# DBTITLE 1,Training intro
# MAGIC %md
# MAGIC ## Step 4: Train with MLflow Tracking + Feature Lineage
# MAGIC
# MAGIC We'll:
# MAGIC 1. Use `mlflow.sklearn.autolog()` to automatically log parameters and metrics
# MAGIC 2. Log the model with **`fe.log_model(training_set=...)`** — this binds feature lineage so batch/real-time scoring can auto-join features
# MAGIC 3. Use `mlflow.evaluate()` to get classification metrics + plots (confusion matrix, ROC, PR curve)

# COMMAND ----------

# DBTITLE 1,Train and log with feature lineage
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Enable autolog but let fe.log_model own the model artifact (lineage).
mlflow.sklearn.autolog(log_models=False, silent=True)

with mlflow.start_run(run_name="lightgbm_lapse") as run:
    model.fit(X_tr, y_tr)

    # fe.log_model captures feature lineage; required so fe.score_batch can auto-join features.
    fe.log_model(
        model=model,
        artifact_path="model",
        flavor=mlflow.sklearn,
        training_set=training_set,
        registered_model_name=None,   # we register in Module 3
    )

    # The FE-logged model is consumed via fe.score_batch / serving (it needs the lookup
    # key to auto-join features), so it can't be loaded as a plain pyfunc here. Evaluate
    # the in-memory pipeline on a static prediction set instead of the runs:/ URI.
    val_df = X_val.assign(**{label_col: y_val})
    val_df["prediction"] = model.predict(X_val)
    val_results = mlflow.evaluate(
        data=val_df,
        predictions="prediction",
        targets=label_col,
        model_type="classifier",
        evaluator_config={"metric_prefix": "val_", "pos_label": "Yes"},
    )

    print(f"\n✅ Training complete!")
    print(f"Run ID: {run.info.run_id}")
    print(f"Validation F1: {val_results.metrics.get('val_f1_score', 'N/A')}")
    print(f"Validation Accuracy: {val_results.metrics.get('val_accuracy_score', 'N/A')}")

# COMMAND ----------

# DBTITLE 1,Explore experiment UI
# MAGIC %md
# MAGIC ## Step 5: Explore the Experiments UI
# MAGIC
# MAGIC MLflow logged everything automatically. Let's explore:
# MAGIC
# MAGIC > **✨ Try it now**:
# MAGIC > 1. Click the **beaker icon** (🧪) in the right sidebar → opens the Experiment panel
# MAGIC > 2. Click on the run name (`lightgbm_lapse`) to see details
# MAGIC > 3. Check out the **Metrics** tab — accuracy, F1, precision, recall
# MAGIC > 4. Check **Artifacts** — the model, confusion matrix, ROC curve, and PR curve are all saved
# MAGIC
# MAGIC You can also view artifacts programmatically:

# COMMAND ----------

# DBTITLE 1,Display evaluation artifacts
import os, tempfile
from IPython.display import Image

temp_dir = tempfile.mkdtemp()
eval_path = mlflow.artifacts.download_artifacts(run_id=run.info.run_id, dst_path=temp_dir)

confusion_path = os.path.join(eval_path, "val_confusion_matrix.png")
if os.path.exists(confusion_path):
    print("Confusion Matrix:")
    display(Image(filename=confusion_path))

# COMMAND ----------

# DBTITLE 1,Show ROC curve
roc_path = os.path.join(eval_path, "training_roc_curve.png")
if os.path.exists(roc_path):
    print("ROC Curve (training, from autolog):")
    display(Image(filename=roc_path))
else:
    print("ROC curve artifact not found. Available artifacts:")
    print([f for f in os.listdir(eval_path) if f.endswith('.png')])

# COMMAND ----------

# DBTITLE 1,Optuna intro
# MAGIC %md
# MAGIC ## Step 6: Hyperparameter Tuning with Optuna
# MAGIC
# MAGIC Instead of guessing hyperparameters, let **Optuna** search for us. Optuna uses smart sampling (TPE by default) to explore the space efficiently — far better than manual grid search.
# MAGIC
# MAGIC We'll:
# MAGIC 1. Define a **small search space** over key LightGBM hyperparameters (demo-sized!)
# MAGIC 2. Run **10 trials** — each trial trains a model and logs to MLflow as a **nested run**
# MAGIC 3. Retrain the **best model** and log it with `fe.log_model` so feature lineage is preserved

# COMMAND ----------

# DBTITLE 1,Optuna hyperparameter search
import optuna
from sklearn.metrics import f1_score
from sklearn.base import clone

# Disable autolog so each trial logs only what we explicitly specify.
mlflow.sklearn.autolog(disable=True)

def objective(trial):
    """Small search space for demo purposes."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=50),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 64, step=16),
    }
    pipe = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("classifier", LGBMClassifier(**params, random_state=42, verbose=-1)),
    ])
    with mlflow.start_run(nested=True):
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_val)
        f1 = f1_score(y_val, preds, pos_label="Yes")
        mlflow.log_params(params)
        mlflow.log_metric("val_f1", f1)
    return f1

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize", study_name="lightgbm_tuning_demo")

with mlflow.start_run(run_name="optuna_parent") as parent_run:
    study.optimize(objective, n_trials=10)
    mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    mlflow.log_metric("best_val_f1", study.best_value)

best_params = study.best_params
print(f"\n✅ Optuna complete! ({len(study.trials)} trials)")
print(f"Best validation F1: {study.best_value:.4f}")
print(f"Best params: {best_params}")

# Retrain the best model on the training split and log with feature lineage.
best_model = Pipeline([
    ("preprocessor", clone(preprocessor)),
    ("classifier", LGBMClassifier(**best_params, random_state=42, verbose=-1)),
])

with mlflow.start_run(run_name="lightgbm_best_optuna") as best_run:
    best_model.fit(X_tr, y_tr)

    fe.log_model(
        model=best_model,
        artifact_path="model",
        flavor=mlflow.sklearn,
        training_set=training_set,
        registered_model_name=None,
    )

    best_f1 = f1_score(y_val, best_model.predict(X_val), pos_label="Yes")
    mlflow.log_metric("val_f1", best_f1)

    print(f"\n✅ Best model logged with feature lineage!")
    print(f"Run ID: {best_run.info.run_id}")
    print(f"Validation F1: {best_f1:.4f}")

# COMMAND ----------

# DBTITLE 1,Summary and next step
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC **What we did:**
# MAGIC - Built a training set from the Feature Store with `FeatureLookup` (lineage captured)
# MAGIC - Built a preprocessing + LightGBM pipeline
# MAGIC - Used `mlflow.sklearn.autolog()` for zero-effort tracking
# MAGIC - Logged the model with `fe.log_model()` so scoring can auto-join features
# MAGIC - Used `mlflow.evaluate()` for classification metrics and plots
# MAGIC - Tuned hyperparameters with **Optuna** and logged the best model with feature lineage
# MAGIC
# MAGIC > **Key takeaway**: MLflow tracks your entire training process automatically, and the Feature Store binds feature lineage to the model.
# MAGIC
# MAGIC **Next step**: [04_model_registration]($./04_model_registration) → Register the model in Unity Catalog for governance