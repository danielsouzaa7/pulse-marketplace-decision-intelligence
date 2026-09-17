-- Gold: one row per date over the fixed project window config.START_DATE ..
-- config.END_DATE, injected as the start_date / end_date parameters. Under the
-- project's locked window that is 180 rows. Company scope.
--
-- The window is PASSED IN, never retyped here. Hardcoded date literals
-- duplicating config would have no enforced link back to it: move the window in
-- config and this file would keep building the old range while every test that
-- derived its expectation from the same literal kept passing. sql_runner exists
-- to close exactly that gap -- the same way 01_orders.sql takes its peak-hours
-- tuple as a parameter -- so config is the single source of truth for the
-- window and this file has none of its own.
--
-- Same shape as gold_zone_performance.sql with the zone cross-join dropped,
-- plus active_customers and discount_amount. The calendar spine is kept for the
-- same reason it is kept there: a day with no activity must materialise as an
-- explicit 0.0 rather than disappear, because a vanished row reads downstream as
-- a shorter series instead of a worse one -- an outage would partially hide
-- itself from the very detector meant to catch it.
--
-- This dataset carries RATE metrics, not just volume. At company scope the
-- volume metrics are diluted across eight zones, so a single-zone incident may
-- not clear an anomaly threshold on gmv or orders_completed while it clearly
-- moves completion_rate and cancellation_rate. A gold layer carrying only
-- volume would leave that company-level signal invisible.
--
-- Metric semantics locked project-wide:
--   gmv               = SUM(item_amount) for COMPLETED orders only, never
--                       item_amount + delivery_fee.
--   order_conversion  = orders_placed / sessions      (pre-checkout)
--   completion_rate   = orders_completed / orders_placed  (post-checkout)
--   cancellation_rate = orders_cancelled / orders_placed
--   aov               = gmv / orders_completed
--
-- Those four terms are the funnel decomposition
--     gmv = sessions x order_conversion x completion_rate x aov
-- (Task 11's primary diagnostic). It holds only while each ratio is built from
-- the same numerator and denominator counted at THIS grain, so sessions comes
-- straight from silver sessions rather than from a join that could fan out.
--
-- Every ratio is guarded with COALESCE(x / NULLIF(y, 0), 0.0): a zero-activity
-- spine day yields 0.0, never NULL and never a division error.
WITH spine AS (
    SELECT unnest(generate_series(DATE '{start_date}',
                                  DATE '{end_date}',
                                  INTERVAL 1 DAY))::DATE AS metric_date
),
sess AS (
    -- silver sessions carries session_day (0-based offset from START_DATE), not
    -- a timestamp; the calendar date is reconstructed from the same anchor the
    -- spine uses so the two sides of the join agree by construction.
    SELECT DATE '{start_date}' + CAST(session_day AS INTEGER) AS metric_date,
           COUNT(*)                                           AS sessions
    FROM read_parquet('{silver}/sessions.parquet')
    GROUP BY 1
),
ord AS (
    SELECT order_date AS metric_date,
           COUNT(*)                                             AS orders_placed,
           COUNT(*) FILTER (WHERE is_completed)                 AS orders_completed,
           COUNT(*) FILTER (WHERE is_cancelled)                 AS orders_cancelled,
           SUM(item_amount) FILTER (WHERE is_completed)         AS gmv,
           SUM(contribution_margin) FILTER (WHERE is_completed) AS contribution_margin,
           -- Discount is reported on the same population as gmv and margin, so
           -- the three can be read against one another without a footnote.
           SUM(discount_amount) FILTER (WHERE is_completed)     AS discount_amount,
           -- Customers who actually transacted that day, not merely browsed.
           COUNT(DISTINCT customer_id) FILTER (WHERE is_completed) AS active_customers
    FROM read_parquet('{silver}/orders.parquet')
    GROUP BY 1
),
dlv AS (
    -- Silver holds 95,332 deliveries against 99,457 orders: some orders never
    -- entered dispatch, and quarantined rows left for cause. An INNER join is
    -- correct HERE -- these averages are over deliveries that exist -- but it
    -- must not decide which days exist at all; that is the spine's job.
    --
    -- order_id is unique in silver orders and unique in silver deliveries, so
    -- this join cannot fan out and cannot inflate any count derived from it.
    SELECT o.order_date AS metric_date,
           AVG(d.promised_eta_minutes)    AS avg_promised_eta_minutes,
           -- AVG skips NULLs: a post-dispatch cancellation was assigned but
           -- never delivered, and has no duration to average.
           AVG(d.actual_delivery_minutes) AS avg_actual_delivery_minutes,
           -- is_on_time is silver's own column (actual <= promised + 5, the
           -- grace period locked there); restating the rule here would let the
           -- two definitions drift apart.
           COUNT(*) FILTER (WHERE d.is_on_time)::DOUBLE
               / NULLIF(COUNT(*) FILTER (WHERE d.is_on_time IS NOT NULL), 0)
                                          AS on_time_rate
    FROM read_parquet('{silver}/deliveries.parquet') d
    JOIN read_parquet('{silver}/orders.parquet') o USING (order_id)
    GROUP BY 1
)
SELECT
    s.metric_date,
    COALESCE(se.sessions, 0)                                        AS sessions,
    COALESCE(o.orders_placed, 0)                                    AS orders_placed,
    COALESCE(o.orders_completed, 0)                                 AS orders_completed,
    COALESCE(o.orders_cancelled, 0)                                 AS orders_cancelled,
    COALESCE(o.active_customers, 0)                                 AS active_customers,
    COALESCE(o.gmv, 0.0)                                            AS gmv,
    COALESCE(o.contribution_margin, 0.0)                            AS contribution_margin,
    COALESCE(o.discount_amount, 0.0)                                AS discount_amount,
    COALESCE(o.orders_placed::DOUBLE / NULLIF(se.sessions, 0), 0.0) AS order_conversion,
    COALESCE(o.orders_completed::DOUBLE
             / NULLIF(o.orders_placed, 0), 0.0)                     AS completion_rate,
    COALESCE(o.orders_cancelled::DOUBLE
             / NULLIF(o.orders_placed, 0), 0.0)                     AS cancellation_rate,
    COALESCE(o.gmv / NULLIF(o.orders_completed, 0), 0.0)            AS aov,
    -- The two averages stay NULL on a day with no deliveries: an average of
    -- nothing is not "zero minutes", and fabricating a 0 would drag any baseline
    -- built over these columns toward a duration that never happened. on_time_rate
    -- is a rate, so it takes the 0.0 guard like every other ratio above.
    d.avg_promised_eta_minutes,
    d.avg_actual_delivery_minutes,
    COALESCE(d.on_time_rate, 0.0)                                   AS on_time_rate
FROM spine s
LEFT JOIN sess se USING (metric_date)
LEFT JOIN ord  o  USING (metric_date)
LEFT JOIN dlv  d  USING (metric_date)
ORDER BY s.metric_date;
