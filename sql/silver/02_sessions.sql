-- Silver: sessions.
--
-- zone_id is a categorical identifier, so it is NEVER imputed. A missing zone
-- is resolved only from a deterministic valid relationship -- the session's
-- customer's home zone, which is a fact about that customer, not a guess.
-- Filling it from a mode, a median or "the most common zone" would invent an
-- attribution that later feeds zone-level GMV and conversion, so anything the
-- customer join cannot resolve is quarantined by 02_sessions_rejected.sql.
WITH raw AS (
    SELECT * FROM read_parquet('{bronze}/sessions.parquet')
),
customers AS (
    SELECT customer_id, home_zone_id
    FROM read_parquet('{bronze}/customers.parquet')
),
resolved AS (
    SELECT
        r.* EXCLUDE (zone_id),
        -- Bronze types zone_id as DOUBLE only because the injected nulls
        -- widened it; cast back to the identifier type the dimension uses.
        CAST(COALESCE(r.zone_id, c.home_zone_id) AS BIGINT)  AS zone_id,
        -- Lineage, not imputation: marks a zone_id that came from the customer
        -- dimension rather than from the session event itself.
        r.zone_id IS NULL AND c.home_zone_id IS NOT NULL     AS zone_id_was_resolved
    FROM raw r
    LEFT JOIN customers c USING (customer_id)
)
SELECT * FROM resolved WHERE zone_id IS NOT NULL;
