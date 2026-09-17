-- Gold: one row per (metric_date, zone_id) over the fixed project window
-- config.START_DATE .. config.END_DATE, injected as the start_date / end_date
-- parameters. Under the project's locked window that is 180 days x 8 zones,
-- so 1440 rows.
--
-- The window is PASSED IN, never retyped here. Hardcoded date literals
-- duplicating config would have no enforced link back to it: move the window in
-- config and this file would keep building the old range while every test that
-- derived its expectation from the same literal kept passing. sql_runner exists
-- to close exactly that gap -- the same way 01_orders.sql takes its peak-hours
-- tuple as a parameter -- so config is the single source of truth for the
-- window and this file has none of its own.
--
-- THE CALENDAR SPINE IS LOAD-BEARING, NOT BOILERPLATE.
-- A plain GROUP BY metric_date, zone_id emits NO ROW for a zone-day with no
-- activity. That failure mode is causally perverse: the worse a zone gets, the
-- more of its bad days silently vanish, so an anomaly detector downstream would
-- see a SHORTER series rather than a WORSE one -- an incident would partially
-- hide itself. The spine below (generate_series CROSS JOIN the zone dimension,
-- LEFT JOINed to the facts) forces every zone-day to materialise, so a dead day
-- reads as an explicit 0.0, which is a detectable value.
--
-- Metric semantics locked project-wide:
--   gmv               = SUM(item_amount) for COMPLETED orders only. Never
--                       item_amount + delivery_fee -- the fee is not merchandise
--                       value, and mixing it in would make gold's GMV disagree
--                       with silver's.
--   order_conversion  = orders_placed / sessions      (pre-checkout)
--   completion_rate   = orders_completed / orders_placed  (post-checkout)
--   cancellation_rate = orders_cancelled / orders_placed
--   aov               = gmv / orders_completed
--
-- Those four are exactly the terms of the funnel decomposition
--     gmv = sessions x order_conversion x completion_rate x aov
-- which Task 11 uses as its primary diagnostic. The identity survives only
-- while each ratio is built from the SAME numerator and denominator counted at
-- THIS grain, so sessions is aggregated straight from silver sessions rather
-- than from any join that could fan the session count out.
--
-- Every ratio is guarded with COALESCE(x / NULLIF(y, 0), 0.0): a zero-activity
-- spine day must produce 0.0, never NULL and never a division error, because a
-- NULL would break the day-of-week baseline arithmetic downstream.
--
-- AVAILABILITY IS CARRIED HERE BECAUSE THIS IS THE GRAIN IT IS DETECTABLE AT.
-- gold_merchant_performance owns the per-merchant detail, and a zone series can
-- be rebuilt from it -- but a per-MERCHANT daily availability series cannot be
-- scanned by the anomaly detector at all: every merchant is shut one fixed
-- weekday, so one whole weekday of its series is NULL, its day-of-week baseline
-- for that weekday does not exist, and the detector correctly refuses the
-- series. Measured, not assumed: a supply collapse that moves a zone's
-- availability by -5.7% (z = -6.6) at this grain fires on zero of that zone's
-- merchants at merchant grain. Supply failure is a zone-level operational event
-- and this is where it has to be visible.
--
-- The roll-up is SUM(available) / SUM(scheduled), never AVG of the per-merchant
-- rates: an average of averages weights a merchant open two hours the same as
-- one open fourteen. Zones always have several merchants scheduled, so unlike
-- the merchant grain this ratio has no zero-denominator case in practice; it is
-- still NULLIF-guarded, and stays NULL rather than 0.0 if one ever occurs,
-- because "nothing was scheduled" is not "nothing was available".
WITH spine AS (
    SELECT d.metric_date, z.zone_id
    FROM (SELECT unnest(generate_series(DATE '{start_date}',
                                        DATE '{end_date}',
                                        INTERVAL 1 DAY))::DATE AS metric_date) d
    CROSS JOIN (SELECT zone_id FROM read_parquet('{silver}/zones.parquet')) z
),
sess AS (
    -- silver sessions carries session_day (0-based offset from START_DATE), not
    -- a timestamp; the calendar date is reconstructed from the same anchor the
    -- spine uses so the two sides of the join agree by construction.
    SELECT DATE '{start_date}' + CAST(session_day AS INTEGER) AS metric_date,
           zone_id,
           COUNT(*)                                           AS sessions
    FROM read_parquet('{silver}/sessions.parquet')
    GROUP BY 1, 2
),
ord AS (
    SELECT order_date AS metric_date,
           zone_id,
           COUNT(*)                                             AS orders_placed,
           COUNT(*) FILTER (WHERE is_completed)                 AS orders_completed,
           COUNT(*) FILTER (WHERE is_cancelled)                 AS orders_cancelled,
           SUM(item_amount) FILTER (WHERE is_completed)         AS gmv,
           SUM(contribution_margin) FILTER (WHERE is_completed) AS contribution_margin
    FROM read_parquet('{silver}/orders.parquet')
    GROUP BY 1, 2
),
avail AS (
    -- The periodic snapshot, rolled to the zone through the merchant dimension.
    -- merchant_id is unique in silver merchants, so this join cannot fan out.
    -- is_available is already a subset of scheduled_open upstream; the AND is
    -- restated so the numerator can never exceed the denominator.
    SELECT CAST(a.snapshot_ts AS DATE)                                AS metric_date,
           m.zone_id,
           COUNT(*) FILTER (WHERE a.scheduled_open)                   AS scheduled_open_hours,
           COUNT(*) FILTER (WHERE a.scheduled_open AND a.is_available) AS available_hours
    FROM read_parquet('{silver}/merchant_availability.parquet') a
    JOIN read_parquet('{silver}/merchants.parquet') m USING (merchant_id)
    GROUP BY 1, 2
),
dlv AS (
    -- Silver holds 95,332 deliveries against 99,457 orders: some orders never
    -- entered dispatch, and quarantined rows left for cause. An INNER join is
    -- therefore correct HERE -- these averages are over deliveries that exist --
    -- but it must not decide which zone-days exist at all. That is the spine's
    -- job: a zone-day with orders and no deliveries still gets its row, with
    -- NULL delivery timings.
    --
    -- order_id is unique in silver orders and unique in silver deliveries, so
    -- this join cannot fan out and cannot inflate any count derived from it.
    SELECT o.order_date AS metric_date,
           o.zone_id,
           AVG(d.promised_eta_minutes)    AS avg_promised_eta_minutes,
           -- AVG skips NULLs: a post-dispatch cancellation was assigned but
           -- never delivered, and has no duration to average.
           AVG(d.actual_delivery_minutes) AS avg_actual_delivery_minutes,
           -- is_on_time is silver's own column (actual <= promised + 5, the
           -- grace period locked there); restating the rule here would let the
           -- two definitions drift apart. NULL for never-delivered rows, so the
           -- denominator counts only deliveries that actually landed.
           COUNT(*) FILTER (WHERE d.is_on_time)::DOUBLE
               / NULLIF(COUNT(*) FILTER (WHERE d.is_on_time IS NOT NULL), 0)
                                          AS on_time_rate
    FROM read_parquet('{silver}/deliveries.parquet') d
    JOIN read_parquet('{silver}/orders.parquet') o USING (order_id)
    GROUP BY 1, 2
)
SELECT
    s.metric_date,
    s.zone_id,
    COALESCE(se.sessions, 0)                                        AS sessions,
    COALESCE(o.orders_placed, 0)                                    AS orders_placed,
    COALESCE(o.orders_completed, 0)                                 AS orders_completed,
    COALESCE(o.orders_cancelled, 0)                                 AS orders_cancelled,
    COALESCE(o.gmv, 0.0)                                            AS gmv,
    COALESCE(o.contribution_margin, 0.0)                            AS contribution_margin,
    COALESCE(o.orders_placed::DOUBLE / NULLIF(se.sessions, 0), 0.0) AS order_conversion,
    COALESCE(o.orders_completed::DOUBLE
             / NULLIF(o.orders_placed, 0), 0.0)                     AS completion_rate,
    COALESCE(o.orders_cancelled::DOUBLE
             / NULLIF(o.orders_placed, 0), 0.0)                     AS cancellation_rate,
    COALESCE(o.gmv / NULLIF(o.orders_completed, 0), 0.0)            AS aov,
    -- The two averages stay NULL on a zone-day with no deliveries: an average of
    -- nothing is not "zero minutes", and fabricating a 0 would drag any baseline
    -- built over these columns toward a duration that never happened. on_time_rate
    -- is a rate, so it takes the 0.0 guard like every other ratio above.
    d.avg_promised_eta_minutes,
    d.avg_actual_delivery_minutes,
    COALESCE(d.on_time_rate, 0.0)                                   AS on_time_rate,
    COALESCE(av.scheduled_open_hours, 0)                            AS scheduled_open_hours,
    COALESCE(av.available_hours, 0)                                 AS available_hours,
    av.available_hours::DOUBLE
        / NULLIF(av.scheduled_open_hours, 0)                        AS availability_rate
FROM spine s
LEFT JOIN sess se USING (metric_date, zone_id)
LEFT JOIN ord  o  USING (metric_date, zone_id)
LEFT JOIN dlv  d  USING (metric_date, zone_id)
LEFT JOIN avail av USING (metric_date, zone_id)
ORDER BY s.metric_date, s.zone_id;
