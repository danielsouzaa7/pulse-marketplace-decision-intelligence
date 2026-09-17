-- Silver: orders.
--
-- Bronze covers the fixed project window 2026-03-15 .. 2026-09-10
-- (config.START_DATE / config.END_DATE). Nothing here re-scopes it.
--
-- Cleaning order matters: de-duplicate first (so every later count is per
-- order, not per copy), then normalise categories, then impute the numeric
-- measure, then enforce referential integrity last. Orphan-FK rows are NOT
-- silently discarded -- 01_orders_rejected.sql captures exactly the rows this
-- file's SEMI JOIN removes, into data/silver/_rejected/orders.parquet.
--
-- Metric semantics locked project-wide: GMV is item_amount alone for completed
-- orders and never includes delivery_fee, so no gmv column is materialised
-- here; Gold aggregates item_amount under is_completed.
WITH raw AS (
    SELECT *
    FROM read_parquet('{bronze}/orders.parquet', file_row_number = true)
),
deduped AS (
    -- ROW_NUMBER + QUALIFY rather than DISTINCT: a repeated order_id has to
    -- survive as ONE whole record picked deterministically, not as a
    -- column-wise collapse. Bronze's copies are not always byte-identical
    -- (defect injection landed on one copy of a pair), so the tie on order_ts
    -- is broken by physical position: the first record written for an order
    -- wins, reproducibly, run to run.
    SELECT * EXCLUDE (file_row_number)
    FROM raw
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY order_id
        ORDER BY order_ts, file_row_number
    ) = 1
),
normalised AS (
    -- Casing and spacing drift ('CREDIT_CARD', 'Credit Card', 'cc') folded
    -- back onto the canonical vocabulary, so a group-by cannot split one
    -- payment method into four.
    SELECT
        * EXCLUDE (payment_method),
        CASE replace(lower(trim(payment_method)), ' ', '_')
            WHEN 'cc' THEN 'credit_card'
            ELSE replace(lower(trim(payment_method)), ' ', '_')
        END AS payment_method
    FROM deduped
),
zone_median_fee AS (
    -- delivery_fee is a numeric measure, so a zone-level median is a
    -- defensible fill. Categorical identifiers never get this treatment.
    SELECT zone_id, median(delivery_fee) AS median_fee
    FROM normalised
    WHERE delivery_fee IS NOT NULL
    GROUP BY zone_id
),
imputed AS (
    SELECT
        n.* EXCLUDE (delivery_fee),
        COALESCE(n.delivery_fee, z.median_fee) AS delivery_fee,
        -- Every imputed value is flagged, so downstream analysis can exclude
        -- filled rows instead of trusting them blind.
        n.delivery_fee IS NULL                 AS delivery_fee_is_imputed
    FROM normalised n
    LEFT JOIN zone_median_fee z USING (zone_id)
),
valid AS (
    -- SEMI JOIN keeps orders whose merchant exists in the dimension and pulls
    -- none of the merchant columns in (merchants carries its own zone_id and
    -- commission_rate, which must not shadow the order's).
    SELECT i.*
    FROM imputed i
    SEMI JOIN read_parquet('{bronze}/merchants.parquet') m USING (merchant_id)
)
SELECT
    v.*,
    CAST(v.order_ts AS DATE)                              AS order_date,
    v.item_amount + v.delivery_fee                        AS customer_gross_value,
    v.item_amount + v.delivery_fee - v.discount_amount    AS net_revenue,
    v.item_amount * v.commission_rate + v.delivery_fee
        - v.delivery_cost - v.discount_amount             AS contribution_margin,
    v.status = 'completed'                                AS is_completed,
    v.status = 'cancelled'                                AS is_cancelled,
    v.order_hour IN {peak_hours}                          AS is_peak,
    dayofweek(v.order_ts) IN (0, 6)                       AS is_weekend,
    -- LAG over the customer's own order history: the repeat-purchase gap that
    -- Gold's retention cohorts need. NULL on a customer's first order.
    date_diff('day',
              LAG(v.order_ts) OVER (PARTITION BY v.customer_id ORDER BY v.order_ts),
              v.order_ts)                                 AS days_since_prev_order
FROM valid v;
