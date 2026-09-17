"""PySpark/Delta reproduction of the local DuckDB gold builds.

Mirrors sql/gold/gold_zone_performance.sql and
sql/gold/gold_daily_business_metrics.sql exactly, but as genuine PySpark
running on Databricks rather than DuckDB running locally. This is evidence
that the same gold semantics hold up on a distributed engine, not a second
independent implementation -- so every rule below is copied from the SQL
comments, not reinvented.

STATUS: written, NOT executed in this environment. There is no JDK and no
Databricks account configured on this machine, so this script has never
run. See databricks/README.md for the manual steps to run it on Databricks
Free Edition and reconcile its output against the local DuckDB gold parquet.

Semantics carried over exactly (see the SQL files for the full reasoning):
  - gmv = SUM(item_amount) for completed orders only, never + delivery_fee.
  - the calendar spine (every date x zone in the fixed window) is
    load-bearing: a zone-day with zero completed orders must still emit a
    row of zeros, or a degrading zone's worst days would silently vanish
    from the series instead of reading as 0.0.
  - counts and rates guard to 0.0 on a zero-activity day; the two delivery
    duration averages stay NULL -- an average of nothing is not "zero
    minutes", and a fabricated 0 would assert instant delivery.
  - on_time_rate reuses silver's own is_on_time column rather than
    restating the +5 minute grace rule here.

Scope note: no MERGE, OPTIMIZE, or Z-ORDER. This build is a full overwrite
of a batch gold table -- there is no incremental/upsert use case here, so
MERGE would be unused ceremony, and OPTIMIZE/Z-ORDER tune query patterns
this one-shot evidence layer doesn't have. Left out rather than added as
buzzwords (spec section 8).
"""
from datetime import date, timedelta

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (
    BooleanType, DoubleType, LongType, StructField, StructType, TimestampType,
)

# Mirrors src/pulse/config.py's locked project window (START_DATE, N_DAYS).
# Databricks runs in a separate workspace without the local `pulse` package
# installed, so the window is restated here rather than imported -- keep the
# two in sync by hand if the project window ever moves.
START_DATE = date(2026, 3, 15)
N_DAYS = 180
END_DATE = START_DATE + timedelta(days=N_DAYS - 1)   # 2026-09-10

# Where the silver parquet files were uploaded (see databricks/README.md).
SILVER = "/Volumes/pulse/silver"
# Where the reconciliation CSV is written for a human to download.
CSV_EXPORT = "/Volumes/pulse/silver/_export"

spark = SparkSession.builder.appName("pulse-silver-to-gold").getOrCreate()
spark.sql("CREATE SCHEMA IF NOT EXISTS pulse")

# ---------------------------------------------------------------------------
# Explicit schemas, enforced on read -- schema inference is never used.
# Column sets match what these two gold builds actually consume; types were
# checked directly against data/silver/*.parquet (pandas dtypes -> Spark
# types), including columns later tasks added (entered_dispatch and
# courier_minutes on orders are not needed here; session_day on sessions is
# a BIGINT day offset, not a timestamp).
# ---------------------------------------------------------------------------
ORDERS_SCHEMA = StructType([
    StructField("order_id", LongType(), False),
    StructField("customer_id", LongType(), False),
    StructField("zone_id", LongType(), False),
    StructField("order_date", TimestampType(), False),
    StructField("item_amount", DoubleType(), False),
    StructField("discount_amount", DoubleType(), False),
    StructField("contribution_margin", DoubleType(), False),
    StructField("is_completed", BooleanType(), False),
    StructField("is_cancelled", BooleanType(), False),
])

SESSIONS_SCHEMA = StructType([
    # session_day is a BIGINT day offset from START_DATE, not a timestamp.
    StructField("session_day", LongType(), False),
    StructField("zone_id", LongType(), False),
])

DELIVERIES_SCHEMA = StructType([
    StructField("order_id", LongType(), False),
    StructField("promised_eta_minutes", LongType(), False),
    StructField("actual_delivery_minutes", DoubleType(), True),
    StructField("is_on_time", BooleanType(), True),
])

ZONES_SCHEMA = StructType([
    StructField("zone_id", LongType(), False),
])

orders = spark.read.schema(ORDERS_SCHEMA).parquet(f"{SILVER}/orders.parquet")
sessions = spark.read.schema(SESSIONS_SCHEMA).parquet(f"{SILVER}/sessions.parquet")
deliveries = spark.read.schema(DELIVERIES_SCHEMA).parquet(f"{SILVER}/deliveries.parquet")
zones = spark.read.schema(ZONES_SCHEMA).parquet(f"{SILVER}/zones.parquet")

orders = orders.withColumn("metric_date", F.to_date("order_date"))


def _safe_ratio(num: str, denom: str):
    """COALESCE(num / NULLIF(denom, 0), 0.0), matching the SQL guard exactly."""
    return (F.when(F.col(denom).isNull() | (F.col(denom) == 0), F.lit(0.0))
             .otherwise(F.col(num) / F.col(denom)))


# Calendar spine: every (metric_date, zone_id) in the fixed window, so a
# zone-day with zero activity still emits a row instead of disappearing.
dates_df = spark.createDataFrame(
    [(START_DATE + timedelta(days=i),) for i in range(N_DAYS)], ["metric_date"])
zone_ids = zones.select("zone_id").distinct()
zone_spine = dates_df.crossJoin(zone_ids)

# --- per-zone-day aggregates -------------------------------------------------
sess_by_zone = (sessions
    .withColumn("metric_date", F.date_add(F.lit(START_DATE), F.col("session_day").cast("int")))
    .groupBy("metric_date", "zone_id")
    .agg(F.count("*").alias("sessions")))

ord_by_zone = (orders
    .groupBy("metric_date", "zone_id")
    .agg(
        F.count("*").alias("orders_placed"),
        F.sum(F.col("is_completed").cast("int")).alias("orders_completed"),
        F.sum(F.col("is_cancelled").cast("int")).alias("orders_cancelled"),
        F.sum(F.when(F.col("is_completed"), F.col("item_amount"))).alias("gmv"),
        F.sum(F.when(F.col("is_completed"), F.col("contribution_margin")))
            .alias("contribution_margin"),
    ))

# Inner join to orders for dispatch context only -- correct here (these
# averages are over deliveries that exist), but this join must never decide
# which zone-days exist. That is the spine's job alone.
dlv_with_zone = deliveries.join(
    orders.select("order_id", "metric_date", "zone_id"), "order_id")

dlv_by_zone = (dlv_with_zone
    .groupBy("metric_date", "zone_id")
    .agg(
        F.avg("promised_eta_minutes").alias("avg_promised_eta_minutes"),
        F.avg("actual_delivery_minutes").alias("avg_actual_delivery_minutes"),
        (F.sum(F.col("is_on_time").cast("int"))
         / F.sum(F.when(F.col("is_on_time").isNotNull(), 1))).alias("on_time_rate"),
    ))

zone_perf = (zone_spine
    .join(sess_by_zone, ["metric_date", "zone_id"], "left")
    .join(ord_by_zone, ["metric_date", "zone_id"], "left")
    .join(dlv_by_zone, ["metric_date", "zone_id"], "left")
    .withColumn("sessions", F.coalesce("sessions", F.lit(0)))
    .withColumn("orders_placed", F.coalesce("orders_placed", F.lit(0)))
    .withColumn("orders_completed", F.coalesce("orders_completed", F.lit(0)))
    .withColumn("orders_cancelled", F.coalesce("orders_cancelled", F.lit(0)))
    .withColumn("gmv", F.coalesce("gmv", F.lit(0.0)))
    .withColumn("contribution_margin", F.coalesce("contribution_margin", F.lit(0.0)))
    .withColumn("order_conversion", _safe_ratio("orders_placed", "sessions"))
    .withColumn("completion_rate", _safe_ratio("orders_completed", "orders_placed"))
    .withColumn("cancellation_rate", _safe_ratio("orders_cancelled", "orders_placed"))
    .withColumn("aov", _safe_ratio("gmv", "orders_completed"))
    # avg_promised_eta_minutes / avg_actual_delivery_minutes stay NULL here --
    # no coalesce. on_time_rate is a rate, so it takes the 0.0 guard.
    .withColumn("on_time_rate", F.coalesce("on_time_rate", F.lit(0.0)))
    .select("metric_date", "zone_id", "sessions", "orders_placed", "orders_completed",
            "orders_cancelled", "gmv", "contribution_margin", "order_conversion",
            "completion_rate", "cancellation_rate", "aov", "avg_promised_eta_minutes",
            "avg_actual_delivery_minutes", "on_time_rate")
    .orderBy("metric_date", "zone_id"))

(zone_perf.write.format("delta").mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("pulse.gold_zone_performance"))

# Single-file CSV for the local reconciliation test -- download this to
# databricks/output/gold_zone_performance_spark.csv (see databricks/README.md).
(zone_perf.coalesce(1).write.mode("overwrite").option("header", "true")
    .csv(f"{CSV_EXPORT}/gold_zone_performance_spark"))

# ---------------------------------------------------------------------------
# gold_daily_business_metrics: same shape, company-wide (no zone dimension),
# plus active_customers and discount_amount. See
# sql/gold/gold_daily_business_metrics.sql.
# ---------------------------------------------------------------------------
sess_by_day = (sessions
    .withColumn("metric_date", F.date_add(F.lit(START_DATE), F.col("session_day").cast("int")))
    .groupBy("metric_date")
    .agg(F.count("*").alias("sessions")))

ord_by_day = (orders
    .groupBy("metric_date")
    .agg(
        F.count("*").alias("orders_placed"),
        F.sum(F.col("is_completed").cast("int")).alias("orders_completed"),
        F.sum(F.col("is_cancelled").cast("int")).alias("orders_cancelled"),
        F.sum(F.when(F.col("is_completed"), F.col("item_amount"))).alias("gmv"),
        F.sum(F.when(F.col("is_completed"), F.col("contribution_margin")))
            .alias("contribution_margin"),
        F.sum(F.when(F.col("is_completed"), F.col("discount_amount")))
            .alias("discount_amount"),
        F.countDistinct(F.when(F.col("is_completed"), F.col("customer_id")))
            .alias("active_customers"),
    ))

dlv_with_day = deliveries.join(orders.select("order_id", "metric_date"), "order_id")

dlv_by_day = (dlv_with_day
    .groupBy("metric_date")
    .agg(
        F.avg("promised_eta_minutes").alias("avg_promised_eta_minutes"),
        F.avg("actual_delivery_minutes").alias("avg_actual_delivery_minutes"),
        (F.sum(F.col("is_on_time").cast("int"))
         / F.sum(F.when(F.col("is_on_time").isNotNull(), 1))).alias("on_time_rate"),
    ))

daily_metrics = (dates_df
    .join(sess_by_day, "metric_date", "left")
    .join(ord_by_day, "metric_date", "left")
    .join(dlv_by_day, "metric_date", "left")
    .withColumn("sessions", F.coalesce("sessions", F.lit(0)))
    .withColumn("orders_placed", F.coalesce("orders_placed", F.lit(0)))
    .withColumn("orders_completed", F.coalesce("orders_completed", F.lit(0)))
    .withColumn("orders_cancelled", F.coalesce("orders_cancelled", F.lit(0)))
    .withColumn("active_customers", F.coalesce("active_customers", F.lit(0)))
    .withColumn("gmv", F.coalesce("gmv", F.lit(0.0)))
    .withColumn("contribution_margin", F.coalesce("contribution_margin", F.lit(0.0)))
    .withColumn("discount_amount", F.coalesce("discount_amount", F.lit(0.0)))
    .withColumn("order_conversion", _safe_ratio("orders_placed", "sessions"))
    .withColumn("completion_rate", _safe_ratio("orders_completed", "orders_placed"))
    .withColumn("cancellation_rate", _safe_ratio("orders_cancelled", "orders_placed"))
    .withColumn("aov", _safe_ratio("gmv", "orders_completed"))
    .withColumn("on_time_rate", F.coalesce("on_time_rate", F.lit(0.0)))
    .select("metric_date", "sessions", "orders_placed", "orders_completed",
            "orders_cancelled", "active_customers", "gmv", "contribution_margin",
            "discount_amount", "order_conversion", "completion_rate",
            "cancellation_rate", "aov", "avg_promised_eta_minutes",
            "avg_actual_delivery_minutes", "on_time_rate")
    .orderBy("metric_date"))

(daily_metrics.write.format("delta").mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("pulse.gold_daily_business_metrics"))
