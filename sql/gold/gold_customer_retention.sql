-- Gold: one row per (cohort_month, acquisition_channel, period_index).
--
-- A customer's cohort is the CALENDAR MONTH OF THEIR FIRST COMPLETED ORDER, and
-- period_index is the number of calendar months between that first order and a
-- later one. retention_rate is the share of the cohort that ordered again in
-- that period.
--
-- WHY FIRST ORDER AND NOT SIGNUP DAY.
-- customers.signup_day exists and is tempting, but it guarantees nothing about
-- ordering in this dataset: session dates are deliberately NOT gated on signup
-- (gating them would create a 180-day volume ramp that fires a permanent false
-- anomaly), so first_order_date >= signup_day does NOT hold -- Task 16 proved it
-- by test. Subtracting signup from an order date here would therefore produce
-- negative tenures and cohorts that do not mean what their name says.
-- acquisition_channel is used as a DIMENSION LABEL only: it says how the
-- customer was acquired, never when.
--
-- FIVE WAYS THIS QUERY COULD BE QUIETLY WRONG, AND WHAT STOPS EACH.
--
-- 1. Counting customer-days as customers. A customer who ordered 12 times in a
--    month is one retained customer, not twelve. Every count below is
--    COUNT(DISTINCT customer_id) -- never COUNT(*) -- so a heavy month cannot
--    inflate the numerator. A wrong denominator still yields a number between 0
--    and 1, so no range assertion can catch this; tests/test_contracts.py proves
--    it on a fixture where one customer orders three times in one period and the
--    hand-computed answer is known.
--
-- 2. Assigning the cohort from a LATER order. first_order below is
--    MIN(order_date) GROUP BY customer_id over completed orders, computed once,
--    before anything joins to it. Every row a customer contributes carries that
--    same first_order_date, so the cohort cannot drift with the order being
--    counted.
--
-- 3. Cohort joins fanning out. Three joins, each provably 1:1 on its key:
--    first_order is one row per customer (GROUP BY customer_id); silver
--    customers is one row per customer_id; sizes is one row per
--    (cohort_month, acquisition_channel) (GROUP BY those two). So `cohort` is
--    exactly one row per customer, and no join multiplies the population.
--
-- 4. Period 0 not being exactly 1.0. Every customer's first completed order is
--    by definition in their own cohort month, so period 0's DISTINCT count is
--    the cohort membership itself. Crucially, cohort_size is computed
--    INDEPENDENTLY from the `cohort` relation rather than lifted out of period 0
--    with a MAX() window: deriving the denominator from the numerator would make
--    "period 0 == 1.0" true by circularity and untestable.
--
-- 5. A cohort/channel pair with no history. A customer's own first order always
--    lands in period 0, so no (cohort_month, acquisition_channel) pair can exist
--    with cohort_size 0 -- the denominator is never zero and needs no guard.
--
-- NO CALENDAR SPINE HERE, unlike the date-grained gold tables. A missing
-- (cohort, period) row means that period lies beyond the observation window, not
-- that retention was zero: the 2026-09 cohort has no period 1 because September
-- 2026 is where the data stops. Materialising those as 0.0 would invent a
-- retention collapse at the right edge of every cohort. For the same reason the
-- last period of a cohort is usually TRUNCATED by the window -- the 2026-08
-- cohort's period 1 covers only 2026-09-01..09-10 -- and is not comparable
-- like-for-like with a whole month. period_days_observed / period_days says how
-- much of the period the window actually saw, so a consumer can drop or weight
-- the partial rows instead of having to rediscover which ones they are.
WITH completed AS (
    -- Cancelled orders are not a purchase and never define or extend a cohort.
    SELECT customer_id, CAST(order_date AS DATE) AS order_date
    FROM read_parquet('{silver}/orders.parquet')
    WHERE is_completed
),
first_order AS (
    SELECT customer_id, MIN(order_date) AS first_order_date
    FROM completed
    GROUP BY customer_id
),
cohort AS (
    -- Exactly one row per customer who has ever completed an order.
    SELECT f.customer_id,
           f.first_order_date,
           date_trunc('month', f.first_order_date) AS cohort_start,
           c.acquisition_channel
    FROM first_order f
    JOIN read_parquet('{silver}/customers.parquet') c USING (customer_id)
),
sizes AS (
    -- The denominator, counted from the cohort population itself.
    SELECT cohort_start,
           acquisition_channel,
           COUNT(DISTINCT customer_id) AS cohort_size
    FROM cohort
    GROUP BY 1, 2
),
activity AS (
    -- One row per (customer, completed order). date_diff('month', ...) counts
    -- calendar-month boundaries crossed, which is the same arithmetic that
    -- produced cohort_start, so period_index 0 means exactly "same month as the
    -- first order" and cannot disagree with the cohort label.
    SELECT k.cohort_start,
           k.acquisition_channel,
           date_diff('month', k.first_order_date, o.order_date) AS period_index,
           o.customer_id
    FROM cohort k
    JOIN completed o USING (customer_id)
),
agg AS (
    SELECT a.cohort_start,
           a.acquisition_channel,
           a.period_index,
           s.cohort_size,
           COUNT(DISTINCT a.customer_id) AS retained_customers
    FROM activity a
    JOIN sizes s USING (cohort_start, acquisition_channel)
    GROUP BY 1, 2, 3, 4
)
SELECT
    strftime(cohort_start, '%Y-%m')                    AS cohort_month,
    acquisition_channel,
    period_index,
    cohort_size,
    retained_customers,
    retained_customers::DOUBLE / cohort_size           AS retention_rate,
    period_start                                       AS period_start_date,
    date_diff('day', period_start, period_end) + 1     AS period_days,
    GREATEST(0, date_diff('day', period_start,
                          LEAST(period_end, DATE '{end_date}')) + 1)
                                                       AS period_days_observed
FROM (
    SELECT *,
           (cohort_start + to_months(CAST(period_index AS INTEGER)))::DATE AS period_start,
           (cohort_start + to_months(CAST(period_index AS INTEGER) + 1)
                         - INTERVAL 1 DAY)::DATE                          AS period_end
    FROM agg
)
ORDER BY 1, 2, 3;
