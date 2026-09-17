from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class GoldContract:
    grain: str
    primary_key: tuple[str, ...]
    required_columns: tuple[str, ...]


GOLD_CONTRACTS: dict[str, GoldContract] = {
    "gold_daily_business_metrics": GoldContract(
        grain="one row per date",
        primary_key=("metric_date",),
        required_columns=("metric_date", "gmv", "orders_placed", "orders_completed",
                          "sessions", "order_conversion", "completion_rate",
                          "cancellation_rate", "aov", "active_customers",
                          "avg_promised_eta_minutes", "avg_actual_delivery_minutes",
                          "on_time_rate", "contribution_margin", "discount_amount"),
    ),
    "gold_zone_performance": GoldContract(
        grain="one row per date x zone",
        primary_key=("metric_date", "zone_id"),
        required_columns=("metric_date", "zone_id", "gmv", "orders_placed",
                          "orders_completed", "sessions", "order_conversion",
                          "completion_rate", "cancellation_rate", "aov",
                          "avg_promised_eta_minutes", "avg_actual_delivery_minutes",
                          "on_time_rate", "contribution_margin"),
    ),
    "gold_merchant_performance": GoldContract(
        grain="one row per date x merchant",
        primary_key=("metric_date", "merchant_id"),
        required_columns=("metric_date", "merchant_id", "zone_id", "gmv",
                          "orders_completed", "availability_rate", "contribution_margin"),
    ),
    "gold_customer_retention": GoldContract(
        grain="one row per cohort_month x acquisition_channel x period_index",
        primary_key=("cohort_month", "acquisition_channel", "period_index"),
        required_columns=("cohort_month", "acquisition_channel", "period_index",
                          "cohort_size", "retained_customers", "retention_rate"),
    ),
    "gold_promotion_performance": GoldContract(
        grain="one row per date x promotion",
        primary_key=("metric_date", "promotion_id"),
        required_columns=("metric_date", "promotion_id", "orders_with_promo", "gmv",
                          "discount_amount", "contribution_margin"),
    ),
    "gold_experiment_results": GoldContract(
        grain="one row per experiment x variant x date",
        primary_key=("experiment_id", "variant", "metric_date"),
        required_columns=("experiment_id", "variant", "metric_date", "assigned_customers",
                          "converted_customers", "gmv", "contribution_margin",
                          "incentive_cost"),
    ),
}


class GoldContractError(ValueError):
    """A gold table broke its declared grain. Raised, never asserted: `python -O`
    strips assert statements, and a data gate that disappears under an
    optimisation flag lets a duplicated row double a metric in silence."""


# VALUE DOMAINS. Grain alone let impossible values through to a complete,
# plausible-looking ranking: negative GMV on one zone-7 day moved the company
# priority from 61,59 to 42,58 with no signal. Margins are the only quantities
# that may legitimately be negative (a discount can exceed commission). NaN is
# allowed only in a ratio whose denominator is zero -- "not measured", like a
# merchant not scheduled to open -- and infinity never is.
_MAY_BE_NEGATIVE = ("contribution_margin", "margin_per_completed_order",
                    "margin_per_assigned_customer")
_ORDERED_PAIRS = (("orders_completed", "orders_placed"),
                  ("retained_customers", "cohort_size"),
                  ("available_hours", "scheduled_open_hours"))


# A ratio is "not measured" only when there was nothing to measure. on_time_rate
# and the delivery-minute columns are divided by deliveries, which gold does
# not carry, so their NaN cannot be checked here and stays allowed.
_RATE_DENOMINATORS = {
    "completion_rate": "orders_placed",
    "cancellation_rate": "orders_placed",
    "order_conversion": "sessions",
    "availability_rate": "scheduled_open_hours",
    "aov": "orders_completed",
}


def _is_rate(column: str) -> bool:
    return column.endswith("_rate") or column.endswith("conversion")


def _may_be_unmeasured(column: str) -> bool:
    """A ratio can be undefined (no scheduled hours, no deliveries); an additive
    quantity cannot -- a zero-activity day is 0, never missing."""
    return (_is_rate(column) or column.endswith("_minutes") or "_per_" in column
            or column == "aov")


def _check_domain(df: pd.DataFrame, name: str) -> None:
    if df.empty:
        raise GoldContractError(f"{name}: table is empty")
    numeric = df.select_dtypes("number")
    missing = [c for c in numeric.columns
               if not _may_be_unmeasured(c) and numeric[c].isna().any()]
    if missing:
        raise GoldContractError(f"{name}: missing values in additive columns {missing}")
    for rate, denominator in _RATE_DENOMINATORS.items():
        if rate in numeric.columns and denominator in numeric.columns and (
            numeric[rate].isna() & (numeric[denominator] > 0)
        ).any():
            raise GoldContractError(
                f"{name}: {rate} missing on rows where {denominator} is above zero")
    infinite = [c for c in numeric.columns if numeric[c].abs().eq(float("inf")).any()]
    if infinite:
        raise GoldContractError(f"{name}: infinite values in {infinite}")
    negative = [c for c in numeric.columns
                if c not in _MAY_BE_NEGATIVE and (numeric[c] < 0).any()]
    if negative:
        raise GoldContractError(f"{name}: negative values in {negative}")
    out_of_range = [c for c in numeric.columns
                    if _is_rate(c) and (numeric[c] > 1).any()]
    if out_of_range:
        raise GoldContractError(f"{name}: rates above 1 in {out_of_range}")
    for part, whole in _ORDERED_PAIRS:
        if part in numeric.columns and whole in numeric.columns and (
                numeric[part] > numeric[whole]).any():
            raise GoldContractError(f"{name}: {part} exceeds {whole}")


def assert_contract(df: pd.DataFrame, name: str) -> None:
    c = GOLD_CONTRACTS[name]
    missing = set(c.required_columns) - set(df.columns)
    if missing:
        raise GoldContractError(f"{name}: missing required columns {sorted(missing)}")
    pk = list(c.primary_key)
    if df[pk].isna().any().any():
        raise GoldContractError(f"{name}: null values in primary key {pk}")
    dupes = df.duplicated(subset=pk).sum()
    if dupes:
        raise GoldContractError(f"{name}: {dupes} duplicate primary key rows on {pk}")
    _check_domain(df, name)
