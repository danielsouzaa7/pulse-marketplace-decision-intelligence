"""Reconciliation between the Databricks/PySpark gold build and the local
DuckDB gold build (Task 23).

There is no JDK and no Databricks account configured in this environment,
so databricks/silver_to_gold_spark.py has never been run here. The
comparison test below is skipped until a human runs it on Databricks Free
Edition and exports the CSV described in databricks/README.md -- a skip is
the honest outcome, not a failure to route around.
"""
from pathlib import Path

import pandas as pd
import pytest

from pulse import io

SPARK_SCRIPT = Path(__file__).resolve().parents[1] / "databricks" / "silver_to_gold_spark.py"
SPARK_OUT = Path(__file__).resolve().parents[1] / "databricks" / "output" / "gold_zone_performance_spark.csv"


def test_spark_script_declares_explicit_schema():
    """Cheap static check, runnable with no Spark installed: the script must
    enforce schema on read (StructType) rather than infer it."""
    source = SPARK_SCRIPT.read_text()
    assert "StructType(" in source, "expected explicit StructType schema(s)"
    assert source.count("StructType(") >= 4, "expected one StructType per silver table read"
    assert ".read.schema(" in source, "expected reads to attach an explicit schema"
    assert "inferSchema" not in source, "schema must be enforced, not inferred"


@pytest.mark.skipif(
    not SPARK_OUT.exists(),
    reason=(
        "Databricks output not yet exported: run databricks/silver_to_gold_spark.py "
        "on a Databricks workspace and download the exported CSV to "
        "databricks/output/gold_zone_performance_spark.csv (see databricks/README.md)"
    ),
)
def test_spark_and_duckdb_gold_agree():
    spark = pd.read_csv(SPARK_OUT, parse_dates=["metric_date"])
    local = io.read_gold("gold_zone_performance")
    spark["metric_date"] = pd.to_datetime(spark["metric_date"]).dt.normalize()
    local["metric_date"] = pd.to_datetime(local["metric_date"]).dt.normalize()

    keys = ["metric_date", "zone_id"]
    merged = spark.merge(local, on=keys, suffixes=("_spark", "_local"))
    assert len(merged) == len(local), "row count mismatch between engines"
    for col in ("gmv", "orders_completed", "completion_rate"):
        delta = (merged[f"{col}_spark"] - merged[f"{col}_local"]).abs().max()
        assert delta < 0.01, f"{col} diverges by {delta}"
