-- Quarantine: orders whose merchant_id has no row in the merchant dimension.
--
-- An unmatched foreign key is evidence, not noise. Dropping these rows inside
-- 01_orders.sql would make 199 orders vanish with nothing to point at, so they
-- are written to data/silver/_rejected/orders.parquet instead and surfaced on
-- the Data Quality page.
--
-- The de-duplication below is byte-for-byte the same rule as 01_orders.sql, so
-- the two outputs partition the same set of orders exactly:
--     rows_in (deduped) = rows_out (silver) + rows_rejected (this file).
WITH raw AS (
    SELECT *
    FROM read_parquet('{bronze}/orders.parquet', file_row_number = true)
),
deduped AS (
    SELECT * EXCLUDE (file_row_number)
    FROM raw
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY order_id
        ORDER BY order_ts, file_row_number
    ) = 1
)
SELECT
    d.*,
    'orphan_merchant_fk' AS reject_reason
FROM deduped d
ANTI JOIN read_parquet('{bronze}/merchants.parquet') m USING (merchant_id);
