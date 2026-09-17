# Databricks / PySpark evidence layer

**Status: WRITTEN, NOT EXECUTED.** `silver_to_gold_spark.py` has never been run.
This machine has no JDK installed and no Databricks account configured, so
PySpark cannot start locally and there is no workspace to submit the job to.
Everything below is the exact manual procedure a human runs, on their own
Databricks account, to actually execute it and produce the CSV that
`tests/test_spark_reconciliation.py::test_spark_and_duckdb_gold_agree` checks
against. Until that happens, that one test SKIPS (by design -- see the test
file) and every other test in the suite runs and passes normally.

## What this reproduces

`silver_to_gold_spark.py` rebuilds `gold_zone_performance` (1,440 rows: 180
days x 8 zones) and `gold_daily_business_metrics` (180 rows) in PySpark,
mirroring `sql/gold/gold_zone_performance.sql` and
`sql/gold/gold_daily_business_metrics.sql` column-for-column and rule-for-rule
-- same calendar spine, same GMV definition (completed-order `item_amount`
only, never `delivery_fee`), same zero-guards on counts/rates, same NULL
duration averages on a zero-activity day. It is a second engine computing the
same thing, not a second design.

Deliberately out of scope: `MERGE`, `OPTIMIZE`, `Z-ORDER`. The build is a
full overwrite of a batch table with no incremental/upsert use case, so
`MERGE` would be unused ceremony, and the other two tune query patterns this
one-shot evidence layer never exercises. Left out rather than bolted on as
buzzwords.

## Manual steps

1. **Create a Databricks Free Edition workspace** at
   https://www.databricks.com/try-databricks (no credit card required for
   the Free Edition tier). Sign in and open a new workspace.

2. **Create a Unity Catalog volume** to hold the uploaded silver files, e.g.
   `pulse.silver` under the workspace's default catalog (Catalog Explorer ->
   Create -> Volume, or `CREATE VOLUME IF NOT EXISTS pulse.silver` after
   `CREATE SCHEMA IF NOT EXISTS pulse`). The script reads from
   `/Volumes/pulse/silver/...`.

3. **Upload the local silver parquet files** from this repo's
   `data/silver/` directory to that volume, keeping the filenames:
   - `data/silver/orders.parquet`       -> `/Volumes/pulse/silver/orders.parquet`
   - `data/silver/sessions.parquet`     -> `/Volumes/pulse/silver/sessions.parquet`
   - `data/silver/deliveries.parquet`   -> `/Volumes/pulse/silver/deliveries.parquet`
   - `data/silver/zones.parquet`        -> `/Volumes/pulse/silver/zones.parquet`

   (Catalog Explorer's "Upload to volume" UI, or `databricks fs cp` from the
   Databricks CLI if it's installed and authenticated.)

4. **Create a notebook (or job) from `silver_to_gold_spark.py`.** Either
   import the file directly as a notebook (Workspace -> Import -> file) or
   paste its contents into a new Python notebook cell. Attach it to a
   cluster (any Free Edition serverless/compute works; Delta support is
   built in, no extra library install needed).

5. **Run it.** It creates the `pulse` schema if missing, reads the four
   silver tables with explicit `StructType` schemas (no inference), builds
   both gold tables, writes each as a Delta table
   (`pulse.gold_zone_performance`, `pulse.gold_daily_business_metrics`,
   `overwrite` + `overwriteSchema=true`), and additionally writes
   `gold_zone_performance` as a single-file CSV to
   `/Volumes/pulse/silver/_export/gold_zone_performance_spark/` for step 6.

6. **Export `gold_zone_performance` to this repo.** The run in step 5
   already wrote a `part-*.csv` under
   `/Volumes/pulse/silver/_export/gold_zone_performance_spark/` in the
   volume. Download that one CSV file (Catalog Explorer -> the volume path
   -> download, or `databricks fs cp`) and save it locally, renamed, at:

   ```
   databricks/output/gold_zone_performance_spark.csv
   ```

   (create the `databricks/output/` directory if it doesn't exist yet).

7. **Run the reconciliation test** from the repo root:

   ```
   uv run pytest tests/test_spark_reconciliation.py -v
   ```

   With the CSV in place, `test_spark_and_duckdb_gold_agree` compares it
   against the local DuckDB `gold_zone_performance` parquet on
   `(metric_date, zone_id)` and asserts row-count equality plus per-column
   tolerance (< 0.01) on `gmv`, `orders_completed`, and `completion_rate`.
   `test_spark_script_declares_explicit_schema` (static, no Spark required)
   already passes today -- it just checks the script declares `StructType`
   schemas and never calls `inferSchema`.

## Why pyspark/delta-spark are not project dependencies

`silver_to_gold_spark.py` is not imported by any local code path -- it only
runs inside a Databricks cluster, which already provides PySpark and Delta.
Adding `pyspark`/`delta-spark` to `pyproject.toml` would slow down every
local `uv sync` for a dependency nothing local uses.
