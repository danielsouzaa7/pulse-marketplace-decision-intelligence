-- Gold: one row per (metric_date, promotion_id).
--
-- Separates the four quantities a promotion decision actually turns on, so that
-- none of them can hide inside another:
--   gmv                  SUM(item_amount) over COMPLETED promoted orders. The
--                        project-wide lock: never item_amount + delivery_fee.
--   discount_amount      the subsidy those orders carried.
--   orders_with_promo    volume -- a campaign that "grew orders" and one that
--                        "grew margin" are different claims.
--   contribution_margin  what was left afterwards.
-- A single blended "promo ROI" number would let a campaign that bought volume by
-- giving away more than it earned look identical to one that did not.
--
-- HOW THE ECONOMICS ARE MEANT TO BE READ.
-- The two derived ratios are the whole point: discount_rate
-- (discount_amount / gmv) says how hard the subsidy was pushed, and
-- margin_per_completed_order says what it cost. A campaign whose orders and
-- conversion rise while its margin per order falls is the failure mode this
-- dataset exists to expose, and it is visible from those columns alone -- no
-- promotion id, promo_code or date range is special-cased anywhere below.
--
-- THE promotion_id = 0 ROW IS A COMPARISON GROUP, NOT A CONTROL GROUP.
-- Campaigns run on disjoint date ranges, so comparing one campaign's margin to
-- another's compares different weeks of the year -- different seasonality, and
-- in this dataset a concurrent zone-level operational incident that erodes
-- margin per order all by itself. To make a SAME-DAY comparison possible, orders
-- carrying no promotion are aggregated under the sentinel promotion_id 0
-- (promo_code 'NO_PROMOTION'), which is the value the upstream generator already
-- uses for "no campaign". It spans the whole window, so every campaign day has a
-- non-promoted counterpart on the same date.
--
-- That comparison is OBSERVATIONAL. Campaign exposure is not randomised: which
-- sessions a campaign reached is a property of the sessions, and promoted and
-- non-promoted orders differ in ways nobody measured. A difference between
-- promotion 0 and a campaign is an ASSOCIATION and must be reported as one. The
-- only randomised comparison in this project lives in gold_experiment_results.
--
-- Rows 0 and the real campaigns partition the orders table (an order carries at
-- most one promotion_id), so nothing is double counted by summing across them.
--
-- THE SPINE IS LOAD-BEARING. A campaign day on which nothing sold must still
-- emit a row -- a promotion whose orders collapse to zero would otherwise
-- shorten its own series rather than show a zero. Each campaign's spine is
-- generated from ITS OWN start_date/end_date in the promotions dimension, so a
-- campaign is present exactly on the days it was live and absent on days it was
-- not; a single window-wide spine would fabricate zero-order days for campaigns
-- that were simply not running.
WITH promo AS (
    SELECT promotion_id,
           promo_code,
           promo_type,
           funded_by,
           CAST(start_date AS DATE) AS start_date,
           CAST(end_date AS DATE)   AS end_date
    FROM read_parquet('{silver}/promotions.parquet')
    UNION ALL
    SELECT 0, 'NO_PROMOTION', 'none', 'none',
           DATE '{start_date}', DATE '{end_date}'
),
spine AS (
    SELECT p.promotion_id, p.promo_code, p.promo_type, p.funded_by,
           unnest(generate_series(p.start_date, p.end_date,
                                  INTERVAL 1 DAY))::DATE AS metric_date
    FROM promo p
),
sess AS (
    -- silver sessions carries session_day (0-based offset from START_DATE);
    -- the calendar date is reconstructed from the same anchor the spine uses.
    -- Sessions are the denominator of promo_conversion, so they are counted
    -- straight off the sessions table and never through a join that could fan
    -- the count out.
    SELECT DATE '{start_date}' + CAST(session_day AS INTEGER) AS metric_date,
           COALESCE(promotion_id, 0)                          AS promotion_id,
           COUNT(*)                                           AS sessions
    FROM read_parquet('{silver}/sessions.parquet')
    GROUP BY 1, 2
),
ord AS (
    SELECT CAST(order_date AS DATE)                          AS metric_date,
           COALESCE(promotion_id, 0)                         AS promotion_id,
           COUNT(*)                                          AS orders_with_promo,
           COUNT(*) FILTER (WHERE is_completed)              AS orders_completed,
           COUNT(*) FILTER (WHERE is_cancelled)              AS orders_cancelled,
           COUNT(DISTINCT customer_id)                       AS active_customers,
           SUM(item_amount) FILTER (WHERE is_completed)      AS gmv,
           SUM(discount_amount) FILTER (WHERE is_completed)  AS discount_amount,
           SUM(contribution_margin)
               FILTER (WHERE is_completed)                   AS contribution_margin
    FROM read_parquet('{silver}/orders.parquet')
    GROUP BY 1, 2
)
SELECT
    s.metric_date,
    s.promotion_id,
    s.promo_code,
    s.promo_type,
    -- Who paid for the subsidy. A merchant-funded discount is not a marketplace
    -- cost even though it appears in discount_amount, so the split has to travel
    -- with the numbers rather than be looked up later.
    s.funded_by,
    COALESCE(o.orders_with_promo, 0)                                AS orders_with_promo,
    COALESCE(o.orders_completed, 0)                                 AS orders_completed,
    COALESCE(o.orders_cancelled, 0)                                 AS orders_cancelled,
    COALESCE(o.active_customers, 0)                                 AS active_customers,
    COALESCE(se.sessions, 0)                                        AS sessions,
    COALESCE(o.gmv, 0.0)                                            AS gmv,
    COALESCE(o.discount_amount, 0.0)                                AS discount_amount,
    COALESCE(o.contribution_margin, 0.0)                            AS contribution_margin,
    COALESCE(o.orders_with_promo::DOUBLE
             / NULLIF(se.sessions, 0), 0.0)                         AS promo_conversion,
    COALESCE(o.discount_amount / NULLIF(o.gmv, 0), 0.0)             AS discount_rate,
    COALESCE(o.discount_amount
             / NULLIF(o.orders_completed, 0), 0.0)                  AS discount_per_completed_order,
    -- The incident metric. Margin is only earned on completed orders, so the
    -- denominator is completed orders and not everything placed.
    COALESCE(o.contribution_margin
             / NULLIF(o.orders_completed, 0), 0.0)                  AS margin_per_completed_order,
    COALESCE(o.gmv / NULLIF(o.orders_completed, 0), 0.0)            AS aov
FROM spine s
LEFT JOIN ord  o  USING (metric_date, promotion_id)
LEFT JOIN sess se USING (metric_date, promotion_id)
WHERE s.metric_date BETWEEN DATE '{start_date}' AND DATE '{end_date}'
ORDER BY s.metric_date, s.promotion_id;
