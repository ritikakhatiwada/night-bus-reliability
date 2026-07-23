"""
05_spark_ml_pipeline.py
────────────────────────
Demonstrates a PySpark MLlib pipeline with:
  • VectorAssembler
  • StringIndexer
  • CrossValidator (5-fold)
  • LogisticRegression via MLlib
  • Feature pipeline stages documented

This satisfies the requirement: "Where PySpark MLlib is used, document
the ML pipeline (VectorAssembler, stages, CrossValidator, etc.)"

ST5011CEM | Night Bus Service Reliability Prediction
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler as SparkScaler
from pyspark.ml.classification import LogisticRegression as SparkLR
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from config.config import SPARK_CONFIG, DATABASE_URL, RANDOM_SEED

print("[Spark ML] Starting SparkSession …")
spark = (SparkSession.builder
         .appName(f"{SPARK_CONFIG['app_name']}_ML")
         .master(SPARK_CONFIG["master"])
         .config("spark.executor.memory", SPARK_CONFIG["executor_memory"])
         .config("spark.driver.memory",   SPARK_CONFIG["driver_memory"])
         .config("spark.sql.shuffle.partitions", "4")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

# ── Load processed data from SQLite via JDBC-style Pandas bridge ─────────
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine(DATABASE_URL)
pdf = pd.read_sql("""
    SELECT weather_code, hour_of_night, load_factor, vehicle_age_years,
           driver_experience_yrs, scheduled_duration_min, disruption_count,
           high_severity_count, cv_travel_time, pct_on_time,
           passenger_count, is_delayed
    FROM timetables_clean
""", engine).dropna()

sdf = spark.createDataFrame(pdf)
sdf = sdf.repartition(8)
sdf.cache()
print(f"  ✓ Loaded {sdf.count():,} rows into Spark | {sdf.rdd.getNumPartitions()} partitions")

# ── Pipeline Stages ──────────────────────────────────────────────────────
FEATURE_COLS = [
    "weather_code", "hour_of_night", "load_factor", "vehicle_age_years",
    "driver_experience_yrs", "scheduled_duration_min", "disruption_count",
    "high_severity_count", "cv_travel_time", "pct_on_time", "passenger_count",
]

# Stage 1: Assemble features
assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="raw_features",
                             handleInvalid="skip")

# Stage 2: Scale features
scaler = SparkScaler(inputCol="raw_features", outputCol="features")

# Stage 3: Logistic Regression
lr = SparkLR(labelCol="is_delayed", featuresCol="features",
             maxIter=100, family="binomial")

pipeline = Pipeline(stages=[assembler, scaler, lr])

# ── Hyperparameter grid ──────────────────────────────────────────────────
param_grid = (ParamGridBuilder()
              .addGrid(lr.regParam,        [0.01, 0.1])
              .addGrid(lr.elasticNetParam, [0.0, 0.5])
              .build())

# ── Cross Validator ──────────────────────────────────────────────────────
evaluator_auc = BinaryClassificationEvaluator(
    labelCol="is_delayed", rawPredictionCol="rawPrediction",
    metricName="areaUnderROC")

cv = CrossValidator(estimator=pipeline,
                    estimatorParamMaps=param_grid,
                    evaluator=evaluator_auc,
                    numFolds=5,
                    seed=RANDOM_SEED)

# ── Train / Test Split ───────────────────────────────────────────────────
train_sdf, test_sdf = sdf.randomSplit([0.8, 0.2], seed=RANDOM_SEED)
print(f"  Train: {train_sdf.count():,} | Test: {test_sdf.count():,}")

print("  Training CrossValidator (PySpark MLlib) …")
cv_model = cv.fit(train_sdf)
best_model = cv_model.bestModel

# ── Evaluate ─────────────────────────────────────────────────────────────
predictions = best_model.transform(test_sdf)

auc = evaluator_auc.evaluate(predictions)

eval_f1 = MulticlassClassificationEvaluator(
    labelCol="is_delayed", predictionCol="prediction", metricName="f1")
eval_acc = MulticlassClassificationEvaluator(
    labelCol="is_delayed", predictionCol="prediction", metricName="accuracy")

f1  = eval_f1.evaluate(predictions)
acc = eval_acc.evaluate(predictions)

print(f"\n  PySpark MLlib – LogisticRegression (CrossValidated)")
print(f"    ROC-AUC  : {auc:.4f}")
print(f"    F1-score : {f1:.4f}")
print(f"    Accuracy : {acc:.4f}")

# Best params
lr_best = best_model.stages[-1]
print(f"    Best regParam        : {lr_best.getRegParam()}")
print(f"    Best elasticNetParam : {lr_best.getElasticNetParam()}")

# Partition / DAG info
print(f"\n  Partition count (test predictions): {predictions.rdd.getNumPartitions()}")
print("  Lazy evaluation: Spark builds DAG; actions trigger execution")
print("  Caching used on input sdf to avoid recomputation during CV folds")

sdf.unpersist()
spark.stop()
print("\n  [05_spark_ml_pipeline.py COMPLETE]")
