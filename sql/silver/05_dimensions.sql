-- Silver: merchant dimension.
--
-- Same treatment as payment_method in 01_orders.sql, for the same reason:
-- bronze carries casing and trailing-space drift ('pizza', 'PIZZA ') that
-- would split one category into three the moment anything groups by it, and
-- merchant_category is a slice dimension across the whole product.
--
-- Title-case fold rather than a hand-written lookup, so a category added to
-- the generator later normalises without editing this file.
--
-- zones and customers arrive clean from bronze and are copied into Silver
-- unchanged by build_silver(), so Gold never has to reach back into bronze.
SELECT
    * EXCLUDE (category),
    upper(substr(trim(category), 1, 1)) || lower(substr(trim(category), 2)) AS category
FROM read_parquet('{bronze}/merchants.parquet');
