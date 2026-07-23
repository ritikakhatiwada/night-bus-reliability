"""
02_spark_ingestion.py
─────────────────────
PySpark data ingestion, cleaning, repartitioning, caching, and
persistence to SQLite.  Satisfies:
  • PySpark DataFrames for large-scale transforms
  • ≥ 4 partitions (configured as 8)
  • Caching + repartition demonstration
  • Parameterised SQL writes (SQLAlchemy; no string concatenation)
  • Multi-dataset join by service_code / trip_id

ST5011CEM | Night Bus Service Reliability Prediction
"""

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType
import pandas as pd
from sqlalchemy import create_engine, text
from config.config import (SPARK_CONFIG, DATA_DIR, DATABASE_URL,
                           N_PARTITIONS, ON_TIME_URBAN_MINUTES)

# ── 1. SparkSession ──────────────────────────────────────────────────────
print("\n[1] Starting SparkSession …")
spark = (SparkSession.builder
         .appName(SPARK_CONFIG["app_name"])
         .master(SPARK_CONFIG["master"])
         .config("spark.executor.memory", SPARK_CONFIG["executor_memory"])
         .config("spark.driver.memory",   SPARK_CONFIG["driver_memory"])
         .config("spark.sql.shuffle.partitions", SPARK_CONFIG["sql_shuffle_partitions"])
         .config("spark.default.parallelism",    SPARK_CONFIG["default_parallelism"])
         .getOrCreate())
spark.sparkContext.setLogLevel(SPARK_CONFIG["log_level"])
print(f"  ✓ SparkSession active | UI: {spark.sparkContext.uiWebUrl}")

# ── 2. Load CSVs ─────────────────────────────────────────────────────────
print("\n[2] Loading CSVs into PySpark DataFrames …")
t0 = time.time()

df_tt  = spark.read.csv(os.path.join(DATA_DIR, "timetables.csv"),
                         header=True, inferSchema=True)
df_dis = spark.read.csv(os.path.join(DATA_DIR, "disruptions.csv"),
                         header=True, inferSchema=True)
df_loc = spark.read.csv(os.path.join(DATA_DIR, "locations.csv"),
                         header=True, inferSchema=True)

print(f"  ✓ timetables  : {df_tt.count():>8,} rows | {df_tt.rdd.getNumPartitions()} partitions")
print(f"  ✓ disruptions : {df_dis.count():>8,} rows | {df_dis.rdd.getNumPartitions()} partitions")
print(f"  ✓ locations   : {df_loc.count():>8,} rows | {df_loc.rdd.getNumPartitions()} partitions")

# ── 3. Repartition to N_PARTITIONS ───────────────────────────────────────
print(f"\n[3] Repartitioning to {N_PARTITIONS} partitions …")
df_tt  = df_tt.repartition(N_PARTITIONS,  "service_code")
df_dis = df_dis.repartition(N_PARTITIONS, "service_code")
df_loc = df_loc.repartition(N_PARTITIONS, "trip_id")

# Cache hot DataFrame (timetables used repeatedly downstream)
df_tt.cache()
print(f"  ✓ timetables cached | partitions = {df_tt.rdd.getNumPartitions()}")

# ── 4. Data Cleaning ─────────────────────────────────────────────────────
print("\n[4] Cleaning & feature engineering …")

# Drop nulls in critical columns
df_tt = df_tt.dropna(subset=["trip_id","service_code","delay_minutes","is_delayed"])

# Cast & clip delay
df_tt = (df_tt
         .withColumn("delay_minutes",
                     F.col("delay_minutes").cast(DoubleType()))
         .withColumn("delay_minutes_clipped",
                     F.least(F.greatest(F.col("delay_minutes"),
                                        F.lit(-30.0)), F.lit(60.0)))
         .withColumn("is_delayed", F.col("is_delayed").cast(IntegerType()))
         .withColumn("load_factor",
                     F.col("load_factor").cast(DoubleType()))
         )

# Headway: time gap between consecutive trips on same route (proxy)
window_spec = (
    __import__("pyspark.sql.window", fromlist=["Window"])
    .Window.partitionBy("service_code")
    .orderBy("scheduled_depart")
)
df_tt = df_tt.withColumn(
    "prev_depart",
    F.lag("scheduled_depart").over(window_spec)
)
df_tt = df_tt.withColumn(
    "headway_min",
    (F.unix_timestamp("scheduled_depart") -
     F.unix_timestamp("prev_depart")) / 60.0
)

# Travel Time CV per route (coefficient of variation)
route_stats = (df_tt
               .groupBy("service_code")
               .agg(
                   F.mean("actual_duration_min").alias("mean_dur"),
                   F.stddev("actual_duration_min").alias("std_dur"),
                   F.count("*").alias("trip_count"),
               )
               .withColumn("cv_travel_time",
                           F.col("std_dur") / F.col("mean_dur"))
               )

df_tt = df_tt.join(F.broadcast(route_stats), on="service_code", how="left")

# Service reliability per route (% on-time)
route_reliability = (df_tt
                     .groupBy("service_code")
                     .agg(
                         (F.sum("service_reliability_flag") /
                          F.count("*")).alias("pct_on_time")
                     ))
df_tt = df_tt.join(F.broadcast(route_reliability), on="service_code", how="left")

# Disruption count per route (join disruptions)
dis_agg = (df_dis
           .groupBy("service_code")
           .agg(F.count("*").alias("disruption_count"),
                F.sum(F.when(F.col("severity") == "High", 1).otherwise(0))
                 .alias("high_severity_count"))
           )

df_tt = df_tt.join(F.broadcast(dis_agg), on="service_code", how="left")
df_tt = df_tt.fillna({"disruption_count": 0, "high_severity_count": 0})

# Weather encoding
weather_map = {"Clear": 0, "Rain": 1, "Fog": 2, "Snow": 3, "Windy": 4}
mapping_expr = F.create_map([F.lit(x)
                              for k, v in weather_map.items()
                              for x in (k, v)])
df_tt = df_tt.withColumn("weather_code", mapping_expr[F.col("weather")])

print(f"  ✓ Final timetable rows after cleaning: {df_tt.count():,}")
print(f"  ✓ Columns: {len(df_tt.columns)}")

# ── 5. PySpark SQL EDA ───────────────────────────────────────────────────
print("\n[5] PySpark SQL – EDA queries …")
df_tt.createOrReplaceTempView("timetables")
df_dis.createOrReplaceTempView("disruptions")

# Delay stats
delay_stats = spark.sql("""
    SELECT
        service_code,
        ROUND(AVG(delay_minutes), 2)     AS avg_delay_min,
        ROUND(STDDEV(delay_minutes), 2)  AS std_delay_min,
        ROUND(MIN(delay_minutes), 2)     AS min_delay_min,
        ROUND(MAX(delay_minutes), 2)     AS max_delay_min,
        COUNT(*)                          AS trips,
        ROUND(AVG(is_delayed)*100, 1)    AS pct_delayed
    FROM timetables
    GROUP BY service_code
    ORDER BY avg_delay_min DESC
    LIMIT 10
""")
print("\n  Top 10 routes by avg delay:")
delay_stats.show(truncate=False)

# Disruption breakdown
dis_summary = spark.sql("""
    SELECT disruption_type,
           COUNT(*) AS count,
           ROUND(AVG(duration_min),1) AS avg_duration_min,
           SUM(CASE WHEN severity='High' THEN 1 ELSE 0 END) AS high_count
    FROM disruptions
    GROUP BY disruption_type
    ORDER BY count DESC
""")
print("  Disruption breakdown:")
dis_summary.show(truncate=False)

# ── 6. Persist to SQLite ─────────────────────────────────────────────────
print("\n[6] Writing processed data to SQLite …")
engine = create_engine(DATABASE_URL, echo=False)

def spark_to_sqlite(sdf, table_name, engine, chunksize=5000):
    """Convert Spark DF → Pandas → SQLite; parameterised via SQLAlchemy."""
    pdf = sdf.toPandas()
    pdf.to_sql(table_name, con=engine, if_exists="replace",
               index=False, chunksize=chunksize, method="multi")
    print(f"  ✓ {table_name}: {len(pdf):,} rows written")

# Write key tables
spark_to_sqlite(df_tt.select(
    "trip_id","service_code","operator","scheduled_depart","actual_depart",
    "delay_minutes","delay_minutes_clipped","is_delayed","weather","weather_code",
    "scheduled_duration_min","actual_duration_min","passenger_count","capacity",
    "load_factor","vehicle_age_years","driver_experience_yrs","hour_of_night",
    "day_of_week","disruption_count","high_severity_count",
    "mean_dur","std_dur","cv_travel_time","pct_on_time","headway_min"
), "timetables_clean", engine)

spark_to_sqlite(df_dis, "disruptions_clean", engine)
spark_to_sqlite(
    df_loc.join(
        df_tt.select("trip_id","service_code","operator","is_delayed"),
        on="trip_id", how="left"
    ).dropna(subset=["service_code"]),
    "locations_enriched", engine
)

# Demonstrate parameterised query (prevent SQL injection)
with engine.connect() as conn:
    result = conn.execute(
        text("SELECT COUNT(*) AS total FROM timetables_clean WHERE is_delayed = :flag"),
        {"flag": 1}
    ).fetchone()
    print(f"\n  Parameterised query → delayed trips: {result[0]:,}")

# ── 7. Unpersist cache ───────────────────────────────────────────────────
df_tt.unpersist()
spark.stop()

elapsed = time.time() - t0
print(f"\n  Pipeline wall-clock time: {elapsed:.1f}s")
print("  [02_spark_ingestion.py COMPLETE]")
