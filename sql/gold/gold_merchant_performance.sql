-- Gold: one row per (metric_date, merchant_id) over the fixed project window
-- config.START_DATE .. config.END_DATE, injected as start_date / end_date.
-- Under the locked window that is 180 days x 150 merchants = 27,000 rows.
--
-- WHY THIS TABLE EXISTS: it is where "customers did not want to buy" separates
-- from "customers wanted to buy but supply was not there".
--
-- Those two produce the SAME headline symptom -- zone GMV down, order_conversion
-- down -- and no amount of staring at the orders table tells them apart, because
-- a merchant that is shut emits no orders. Absence is the signal, and a
-- transaction fact cannot carry absence. availability_rate below therefore comes
-- exclusively from the merchant_availability PERIODIC SNAPSHOT, which emits a
-- row per merchant per operating hour whether or not anything happened.
--
-- It is NOT inferred from the merchants dimension (which knows nothing about
-- time) and NOT inferred from missing orders (which would define availability as
-- "sold something", making the metric a restatement of demand and useless for
-- separating supply from demand -- a merchant open all day that simply sold
-- nothing would read as 0% available).
--
--   availability_rate = available_hours / scheduled_open_hours
--
-- SCHEDULED-OPEN IS THE DENOMINATOR, NOT 24, AND NOT THE SNAPSHOT ROW COUNT.
-- A merchant closed on its weekly rest day is not unavailable, it is shut: that
-- is a business decision, not an outage, and counting it as downtime would bury
-- every real outage under a permanent -1/7 baseline. Both the numerator and the
-- denominator are carried as columns so the ratio can be audited, and so a
-- weighted roll-up to zone or company is a SUM/SUM rather than an average of
-- averages (which would weight a merchant open two hours the same as one open
-- fourteen).
--
-- A merchant-day with nothing scheduled has NO availability_rate -- it is NULL,
-- not 0.0. This is the one place this project departs from the
-- COALESCE(ratio, 0.0) convention that gold_zone_performance uses, and
-- deliberately: there, a zero denominator means real activity that measured
-- zero; here it means the quantity was never measured at all. A fabricated 0.0
-- would assert a total outage on every merchant's day off, which is both false
-- and, at ~1/7 of all merchant-days, loud enough to drown the incident this
-- column exists to expose. The anomaly detector skips NaN and does not skip 0.0.
--
-- GRAIN AND DETECTABILITY. date x merchant is a daily series per merchant, which
-- is directly scannable by the anomaly detector, and zone_id is carried so the
-- same rows roll up to a zone-level availability series without a second table
-- or a join back to the dimension. Nothing here keys on a zone id, a merchant id
-- or a date range: a supply collapse is found by reading availability_rate, and
-- the zone it happens to be in falls out of the group-by.
--
-- Money columns follow the project-wide lock: gmv is SUM(item_amount) over
-- COMPLETED orders only, never item_amount + delivery_fee.
--
-- THE SPINE IS LOAD-BEARING (same argument as gold_zone_performance): a plain
-- GROUP BY emits no row for a merchant-day with no orders, so the further a
-- merchant falls the more of its bad days vanish from the series. The spine
-- forces every merchant-day to materialise; a dead trading day reads as an
-- explicit 0.0, which is a detectable value.
WITH spine AS (
    SELECT d.metric_date, m.merchant_id, m.zone_id
    FROM (SELECT unnest(generate_series(DATE '{start_date}',
                                        DATE '{end_date}',
                                        INTERVAL 1 DAY))::DATE AS metric_date) d
    CROSS JOIN (SELECT merchant_id, zone_id
                FROM read_parquet('{silver}/merchants.parquet')) m
),
avail AS (
    -- One row per merchant per operating hour per day, emitted unconditionally.
    -- is_available is already defined as a subset of scheduled_open upstream;
    -- the AND is restated so the numerator can never exceed the denominator even
    -- if that upstream invariant is ever relaxed.
    SELECT CAST(snapshot_ts AS DATE)                                AS metric_date,
           merchant_id,
           COUNT(*) FILTER (WHERE scheduled_open)                   AS scheduled_open_hours,
           COUNT(*) FILTER (WHERE scheduled_open AND is_available)  AS available_hours
    FROM read_parquet('{silver}/merchant_availability.parquet')
    GROUP BY 1, 2
),
ord AS (
    SELECT CAST(order_date AS DATE)                          AS metric_date,
           merchant_id,
           COUNT(*)                                          AS orders_placed,
           COUNT(*) FILTER (WHERE is_completed)              AS orders_completed,
           COUNT(*) FILTER (WHERE is_cancelled)              AS orders_cancelled,
           COUNT(DISTINCT customer_id)                       AS active_customers,
           SUM(item_amount) FILTER (WHERE is_completed)      AS gmv,
           SUM(contribution_margin)
               FILTER (WHERE is_completed)                   AS contribution_margin,
           SUM(discount_amount) FILTER (WHERE is_completed)  AS discount_amount
    FROM read_parquet('{silver}/orders.parquet')
    GROUP BY 1, 2
)
SELECT
    s.metric_date,
    s.merchant_id,
    s.zone_id,
    COALESCE(a.scheduled_open_hours, 0)                          AS scheduled_open_hours,
    COALESCE(a.available_hours, 0)                               AS available_hours,
    -- NULL when nothing was scheduled: see the header. Not COALESCEd.
    a.available_hours::DOUBLE
        / NULLIF(a.scheduled_open_hours, 0)                      AS availability_rate,
    COALESCE(o.orders_placed, 0)                                 AS orders_placed,
    COALESCE(o.orders_completed, 0)                              AS orders_completed,
    COALESCE(o.orders_cancelled, 0)                              AS orders_cancelled,
    COALESCE(o.active_customers, 0)                              AS active_customers,
    COALESCE(o.gmv, 0.0)                                         AS gmv,
    COALESCE(o.contribution_margin, 0.0)                         AS contribution_margin,
    COALESCE(o.discount_amount, 0.0)                             AS discount_amount,
    COALESCE(o.orders_completed::DOUBLE
             / NULLIF(o.orders_placed, 0), 0.0)                  AS completion_rate,
    COALESCE(o.orders_cancelled::DOUBLE
             / NULLIF(o.orders_placed, 0), 0.0)                  AS cancellation_rate,
    COALESCE(o.gmv / NULLIF(o.orders_completed, 0), 0.0)         AS aov
FROM spine s
LEFT JOIN avail a USING (metric_date, merchant_id)
LEFT JOIN ord   o USING (metric_date, merchant_id)
ORDER BY s.metric_date, s.merchant_id;
