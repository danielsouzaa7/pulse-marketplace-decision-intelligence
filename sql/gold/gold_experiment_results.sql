-- Gold: one row per (experiment_id, variant, metric_date).
--
-- Clean aggregates at the grain an A/B analysis needs. The statistics -- effect
-- sizes, confidence intervals, sequential-testing corrections, the call on
-- whether to ship -- are Task 19's job and are deliberately not attempted here.
-- This file's only responsibility is that the numbers Task 19 divides are
-- counted at the right grain and over the right population.
--
-- THE DENOMINATOR IS CUSTOMERS, NOT ORDERS.
-- Randomisation is per CUSTOMER: a customer is assigned to a variant once, and
-- every order they place inherits that assignment. Orders are therefore
-- correlated within a customer and are not independent draws, so a rate built as
-- converted_orders / total_orders has no valid standard error -- the effective
-- sample size is the number of customers, not the number of orders. Every
-- denominator this table carries is a DISTINCT CUSTOMER COUNT:
--   assigned_customers  the randomisation unit, COUNT(DISTINCT customer_id) over
--                       the assignment table -- the population, counted whether
--                       or not they ever came back. This is the denominator; it
--                       is constant per variant per day by construction, and is
--                       repeated on every row so no downstream join is needed to
--                       recover it.
--   converted_customers COUNT(DISTINCT customer_id) among them who COMPLETED an
--                       order that day. A customer ordering three times in one
--                       day counts once.
-- Order counts are carried too, but as descriptive volume, never as a
-- denominator for an inference.
--
-- BALANCE IS VISIBLE, NOT ASSUMED. assigned_customers is emitted per variant
-- rather than as one experiment-level total, so a broken split (a sample-ratio
-- mismatch) shows up as unequal denominators instead of being averaged away.
--
-- ORDERS ARE SCOPED TO THE ASSIGNMENT. Only orders placed on or after a
-- customer's assigned_date and no later than the experiment's end_date count:
-- pre-assignment behaviour cannot have been caused by a treatment that had not
-- happened yet, and letting it in would dilute both arms with identical history
-- and shrink any real effect toward zero.
--
-- THE SPINE IS LOAD-BEARING. Every (variant, day) inside the experiment window
-- materialises even when a variant recorded nothing that day. A variant whose
-- activity collapses must show zeros, not a shorter series -- and a daily series
-- with holes cannot be cumulated or differenced correctly downstream.
WITH exp AS (
    SELECT experiment_id,
           CAST(start_date AS DATE) AS start_date,
           CAST(end_date AS DATE)   AS end_date
    FROM read_parquet('{silver}/experiments.parquet')
),
assign AS (
    -- One row per (experiment_id, customer_id).
    SELECT experiment_id,
           customer_id,
           variant,
           CAST(assigned_date AS DATE) AS assigned_date
    FROM read_parquet('{silver}/experiment_assignments.parquet')
),
denom AS (
    SELECT experiment_id, variant, COUNT(DISTINCT customer_id) AS assigned_customers
    FROM assign
    GROUP BY 1, 2
),
spine AS (
    SELECT d.experiment_id,
           d.variant,
           d.assigned_customers,
           unnest(generate_series(e.start_date, e.end_date,
                                  INTERVAL 1 DAY))::DATE AS metric_date
    FROM denom d
    JOIN exp e USING (experiment_id)
),
activity AS (
    -- assign is one row per (experiment, customer), so joining it to orders
    -- fans out to one row per order -- which is what the FILTERs below count --
    -- and can never duplicate a customer within a (variant, day) group, because
    -- the DISTINCT counts collapse them.
    SELECT a.experiment_id,
           a.variant,
           CAST(o.order_date AS DATE)                            AS metric_date,
           COUNT(*)                                              AS orders_placed,
           COUNT(*) FILTER (WHERE o.is_completed)                AS orders_completed,
           COUNT(DISTINCT o.customer_id)
               FILTER (WHERE o.is_completed)                     AS converted_customers,
           COUNT(DISTINCT o.customer_id)                         AS active_customers,
           SUM(o.item_amount) FILTER (WHERE o.is_completed)      AS gmv,
           SUM(o.contribution_margin)
               FILTER (WHERE o.is_completed)                     AS contribution_margin,
           -- The incentive is the subsidy those orders carried. Named for what
           -- it is to the marketplace (a cost), not for the column it came from.
           SUM(o.discount_amount) FILTER (WHERE o.is_completed)  AS incentive_cost
    FROM assign a
    JOIN read_parquet('{silver}/orders.parquet') o USING (customer_id)
    JOIN exp e ON e.experiment_id = a.experiment_id
    WHERE CAST(o.order_date AS DATE) BETWEEN a.assigned_date AND e.end_date
    GROUP BY 1, 2, 3
)
SELECT
    s.experiment_id,
    s.variant,
    s.metric_date,
    s.assigned_customers,
    COALESCE(a.converted_customers, 0)                    AS converted_customers,
    COALESCE(a.active_customers, 0)                       AS active_customers,
    COALESCE(a.orders_placed, 0)                          AS orders_placed,
    COALESCE(a.orders_completed, 0)                       AS orders_completed,
    COALESCE(a.gmv, 0.0)                                  AS gmv,
    COALESCE(a.contribution_margin, 0.0)                  AS contribution_margin,
    COALESCE(a.incentive_cost, 0.0)                       AS incentive_cost,
    -- Per-customer, per-day. Task 19 pools these over the window itself rather
    -- than averaging a daily average, which would weight a quiet day equally
    -- with a busy one.
    COALESCE(a.converted_customers::DOUBLE
             / NULLIF(s.assigned_customers, 0), 0.0)      AS daily_conversion_rate,
    COALESCE(a.gmv / NULLIF(s.assigned_customers, 0), 0.0)
                                                          AS gmv_per_assigned_customer,
    COALESCE(a.contribution_margin
             / NULLIF(s.assigned_customers, 0), 0.0)      AS margin_per_assigned_customer
FROM spine s
LEFT JOIN activity a USING (experiment_id, variant, metric_date)
ORDER BY s.experiment_id, s.variant, s.metric_date;
