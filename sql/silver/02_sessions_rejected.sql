-- Quarantine: sessions whose zone_id is missing and cannot be resolved from
-- the customer's home zone.
--
-- This is the deliberate dead end of the categorical rule: when the only
-- deterministic relationship fails, the row leaves the dataset rather than
-- being filled with a plausible-looking zone.
--
-- Under the current generator every session carries a customer_id and every
-- customer carries a home_zone_id, so this file is expected to be EMPTY. That
-- is the rule working, not the rule missing: the quarantine exists so that the
-- day a session arrives without a resolvable customer, it lands here instead
-- of being quietly attributed to the wrong zone.
WITH raw AS (
    SELECT * FROM read_parquet('{bronze}/sessions.parquet')
),
customers AS (
    SELECT customer_id, home_zone_id
    FROM read_parquet('{bronze}/customers.parquet')
)
SELECT
    r.*,
    'null_zone_id_unresolvable' AS reject_reason
FROM raw r
LEFT JOIN customers c USING (customer_id)
WHERE r.zone_id IS NULL AND c.home_zone_id IS NULL;
