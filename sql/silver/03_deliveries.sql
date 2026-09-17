-- Silver: deliveries.
--
-- Causality invariant: a delivery cannot be delivered before it was assigned
-- to a courier, nor before the order it fulfils was placed. Bronze carries
-- rows that break it, with an associated clock offset on delivered_ts.
--
-- The primary predicate is delivered_ts < assigned_ts, because that is the
-- invariant the data actually violates structurally. delivered_ts < order_ts
-- is checked as well but is the weaker of the two: whether it fires at all
-- depends on courier-assignment timing, so keying only on it would silently
-- stop catching these rows if that calibration ever changed.
--
-- Violations are quarantined by 03_deliveries_rejected.sql, never repaired:
-- with two timestamps in conflict there is no defensible way to decide which
-- one is wrong, and clipping one to the other would fabricate a duration that
-- then flows straight into on-time rate.
--
-- A NULL delivered_ts is NOT a violation. It is a post-dispatch cancellation
-- (assigned, never delivered) -- a different data-quality story -- so those
-- rows stay in Silver with NULL delivery metrics.
WITH raw AS (
    SELECT * FROM read_parquet('{bronze}/deliveries.parquet')
),
order_times AS (
    -- Bronze, deliberately: this join supplies a TIMESTAMP for the causality
    -- test, and an order's placement time is the same fact whether or not that
    -- order survived cleaning. Row MEMBERSHIP is a separate question and is not
    -- settled here -- quality.FOREIGN_KEYS re-checks deliveries against SILVER
    -- orders after every table is clean, so a delivery whose order was
    -- quarantined follows it out. Reading silver orders here instead would
    -- only cover this one relationship and would make the file unrunnable
    -- before 01_orders.sql.
    --
    -- min() collapses bronze's duplicate order rows; the copies share an
    -- order_ts, so it is exact and the join below cannot fan out.
    SELECT order_id, min(order_ts) AS order_ts
    FROM read_parquet('{bronze}/orders.parquet')
    GROUP BY order_id
),
flagged AS (
    SELECT
        r.*,
        -- COALESCE, because a NULL delivered_ts must read as "no violation",
        -- not as an unknown that swallows the whole predicate.
        COALESCE(r.delivered_ts < r.assigned_ts, FALSE)
            OR COALESCE(r.delivered_ts < o.order_ts, FALSE) AS timestamp_violation
    FROM raw r
    LEFT JOIN order_times o USING (order_id)
)
SELECT
    f.* EXCLUDE (timestamp_violation),
    f.actual_delivery_minutes - f.promised_eta_minutes      AS delivery_delay_minutes,
    -- +5 minutes of grace, locked project-wide. NULL for never-delivered rows.
    f.actual_delivery_minutes <= f.promised_eta_minutes + 5 AS is_on_time
FROM flagged f
WHERE NOT f.timestamp_violation;
