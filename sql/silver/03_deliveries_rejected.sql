-- Quarantine: deliveries whose timestamps break causality.
--
-- Same predicate as 03_deliveries.sql, negated, so the two outputs partition
-- bronze deliveries exactly:
--     rows_in = rows_out (silver) + rows_rejected (this file).
WITH raw AS (
    SELECT * FROM read_parquet('{bronze}/deliveries.parquet')
),
order_times AS (
    SELECT order_id, min(order_ts) AS order_ts
    FROM read_parquet('{bronze}/orders.parquet')
    GROUP BY order_id
),
flagged AS (
    SELECT
        r.*,
        COALESCE(r.delivered_ts < r.assigned_ts, FALSE)
            OR COALESCE(r.delivered_ts < o.order_ts, FALSE) AS timestamp_violation
    FROM raw r
    LEFT JOIN order_times o USING (order_id)
)
SELECT
    f.* EXCLUDE (timestamp_violation),
    'timestamp_violation' AS reject_reason
FROM flagged f
WHERE f.timestamp_violation;
