# tests/test_metrics.py
from datetime import date

import numpy as np
import pandas as pd
import pytest

from pulse.types import AnalysisParams
from pulse.metrics import (
    DAILY_AVERAGE,
    METRIC_REGISTER,
    METRIC_SEMANTICS,
    PER_DELIVERY_AVERAGE,
    PER_ORDER_AVERAGE,
    RATE,
    GoldTables,
    MetricFrame,
    MissingGoldTableError,
    compute_metrics,
    load_gold,
)

P = AnalysisParams(as_of=date(2026, 9, 10))


def test_company_series_covers_every_day():
    mf = compute_metrics(load_gold(), P)
    s = mf.series("gmv", "company", "all")
    assert len(s) == 180
    assert s.index.is_monotonic_increasing


def test_zone_series_are_available_per_zone():
    mf = compute_metrics(load_gold(), P)
    s = mf.series("completion_rate", "zone", "7")
    assert len(s) == 180
    assert s.between(0, 1).all()


def test_headline_kpis_carry_recent_baseline_and_delta():
    mf = compute_metrics(load_gold(), P)
    head = mf.headline(P)
    assert 7 <= len(head) <= 12
    gmv = next(k for k in head if k["metric"] == "gmv")
    assert {"metric", "recent", "baseline", "delta_pct", "unit"}.issubset(gmv)
    assert gmv["delta_pct"] < 0      # company GMV is down in the recent window


def _synthetic_gold(nan_date: pd.Timestamp) -> GoldTables:
    """A calendar-spined 180-day company + zone gold pair, matching real
    gold's shape, with the two duration columns forced to NaN on exactly
    one date -- simulating a day with no deliveries. Every other value is
    a constant (32.0 for the durations, 1.0 for everything else) so that a
    mean computed over a window is exactly that constant if and only if the
    NaN day was skipped rather than coerced to 0.
    """
    dates = pd.date_range("2026-03-15", periods=180, freq="D")
    company = pd.DataFrame({"metric_date": dates})
    for m in METRIC_REGISTER:
        company[m] = (
            32.0 if m in ("avg_actual_delivery_minutes", "avg_promised_eta_minutes")
            else 1.0
        )
    company.loc[
        company["metric_date"] == nan_date,
        ["avg_actual_delivery_minutes", "avg_promised_eta_minutes"],
    ] = float("nan")

    zone = company.drop(columns=["active_customers"]).copy()
    zone["zone_id"] = 7

    return GoldTables({"daily_business_metrics": company, "zone_performance": zone})


def test_nan_duration_survives_series_and_is_skipped_by_headline_mean():
    # RULING: a day with no deliveries must stay NaN ("missing"), never a
    # fabricated 0.0 ("instant delivery"). avg_actual_delivery_minutes is
    # Zone 7's top associated driver in Task 11, so a silent .fillna(0) here
    # would corrupt that diagnosis without failing anything else.
    nan_date = pd.Timestamp("2026-09-01")  # inside P's 14-day recent window
    gold = _synthetic_gold(nan_date)
    mf = compute_metrics(gold, P)

    s = mf.series("avg_actual_delivery_minutes", "company", "all")
    assert np.isnan(s.loc[nan_date])                       # not 0, not -1, not any sentinel
    assert s.loc[nan_date - pd.Timedelta(days=1)] == 32.0  # neighbours untouched

    zs = mf.series("avg_actual_delivery_minutes", "zone", "7")
    assert np.isnan(zs.loc[nan_date])

    head = mf.headline(P)
    row = next(h for h in head if h["metric"] == "avg_actual_delivery_minutes")
    # pandas .mean() skips NaN by default: the recent window's mean over one
    # NaN day and thirteen days of 32.0 is exactly 32.0, not pulled toward 0.
    assert row["recent"] == 32.0
    assert row["baseline"] == 32.0


def test_load_gold_reads_whatever_is_on_disk():
    """RULING R5: load_gold() tolerates a partial gold layer and never asserts
    that all six datasets exist. Task 17 built the other four, so all six are
    readable now -- what is asserted is that it returns what is present, not a
    fixed count."""
    gold = load_gold()
    for attr in ("daily_business_metrics", "zone_performance",
                 "merchant_performance", "customer_retention",
                 "promotion_performance", "experiment_results"):
        assert isinstance(getattr(gold, attr), pd.DataFrame)


def test_missing_gold_table_names_the_table_and_its_build_step():
    """Proved against a deliberately partial container rather than against
    whatever happens to be on disk. Asserting "these four are absent" was true
    only between Task 8 and Task 17: once Task 17 built them the test started
    failing for the one reason that is not a defect, and until then it passed
    without the error path ever being reachable on a complete lake."""
    gold = GoldTables({"daily_business_metrics": pd.DataFrame()})

    # Present tables still resolve -- selectivity, not a container that fails
    # on everything.
    assert isinstance(gold.daily_business_metrics, pd.DataFrame)

    for attr, gold_name in [
        ("merchant_performance", "gold_merchant_performance"),
        ("customer_retention", "gold_customer_retention"),
        ("promotion_performance", "gold_promotion_performance"),
        ("experiment_results", "gold_experiment_results"),
    ]:
        with pytest.raises(MissingGoldTableError) as exc:
            getattr(gold, attr)
        assert gold_name in str(exc.value)
        assert "Task 17" in str(exc.value)

    # An attribute that is not a gold table at all is still a plain AttributeError.
    with pytest.raises(AttributeError):
        gold.not_a_gold_table


def test_series_asserts_no_calendar_gaps():
    # A deliberately gapped long frame (one date dropped) must trip the
    # completeness assertion in .series() rather than silently returning a
    # shorter series -- the anomaly detector's day-of-week baseline math
    # depends on that guard actually firing, not just existing.
    dates = pd.date_range("2026-03-15", periods=180, freq="D").delete(5)
    long = pd.DataFrame({
        "metric": "gmv",
        "scope": "company",
        "scope_value": "all",
        "metric_date": dates,
        "value": 1.0,
    })
    mf = MetricFrame(long)
    with pytest.raises(ValueError, match="gap in daily series"):
        mf.series("gmv", "company", "all")


# --- C1: the semantic contract ---------------------------------------------
#
# .headline() takes a flat window MEAN. For a per-day flow that is a DAILY
# AVERAGE and not the window's total, and the two differ by a factor of the
# window length -- company GMV is R$ 29,977.82 per day against R$ 419,689.55
# over the 14 days. Nothing about the float says which, so METRIC_SEMANTICS says
# it and every row carries the answer.
#
# These tests assert the ARITHMETIC each class claims, not merely that a key
# exists: a daily-average row is checked to equal the window mean AND to differ
# from the window sum by the day count, and a per-order average is checked
# against gmv/orders rather than against anything per-day.


def _window(mf, metric, p, scope="company", scope_value="all"):
    series = mf.series(metric, scope, scope_value)
    end = pd.Timestamp(p.as_of)
    start = end - pd.Timedelta(days=p.comparison_window_days - 1)
    return series.loc[start:end]


def test_every_registered_metric_is_classified():
    # An unclassified metric would render through whichever default a formatter
    # picked, which is the silent-wrong-unit failure the register exists to close.
    assert set(METRIC_SEMANTICS) == set(METRIC_REGISTER)
    assert set(METRIC_SEMANTICS.values()) <= {
        DAILY_AVERAGE, RATE, PER_ORDER_AVERAGE, PER_DELIVERY_AVERAGE
    }


def test_every_headline_row_carries_its_semantic():
    head = compute_metrics(load_gold(), P).headline(P)
    for row in head:
        assert row["semantic"] == METRIC_SEMANTICS[row["metric"]], row["metric"]


@pytest.mark.parametrize(
    "metric",
    ["gmv", "sessions", "orders_placed", "orders_completed", "contribution_margin"],
)
def test_flow_headlines_are_daily_averages_and_not_window_totals(metric):
    """The heart of C1, asserted as arithmetic rather than as a label.

    `recent` must equal the window MEAN, and must equal the window SUM divided by
    the number of days -- which is the same statement twice and is exactly the
    distinction that was missing. A test that only checked the number existed
    would have passed on the broken version.
    """
    mf = compute_metrics(load_gold(), P)
    row = next(r for r in mf.headline(P) if r["metric"] == metric)
    window = _window(mf, metric, P)

    assert row["semantic"] == DAILY_AVERAGE
    assert row["recent"] == pytest.approx(window.mean(), rel=1e-12)
    assert row["recent"] == pytest.approx(
        window.sum() / P.comparison_window_days, rel=1e-12
    )
    # And it is NOT the window total -- the two are a factor of 14 apart, so a
    # surface that showed one where it meant the other is off by that factor.
    assert row["recent"] != pytest.approx(window.sum(), rel=1e-6)
    assert window.sum() == pytest.approx(
        row["recent"] * P.comparison_window_days, rel=1e-12
    )


def test_active_customers_is_a_daily_distinct_average_with_no_window_total():
    """Daily average like the other flows, and the one where no window total
    exists at all: summing a daily-distinct column yields customer-DAYS."""
    mf = compute_metrics(load_gold(), P)
    row = next(r for r in mf.headline(P) if r["metric"] == "active_customers")
    window = _window(mf, "active_customers", P)
    assert row["semantic"] == DAILY_AVERAGE
    assert row["recent"] == pytest.approx(window.mean(), rel=1e-12)
    # The sum is customer-days and is ~6x the real distinct population, which is
    # why prioritization counts customers out of silver instead.
    assert window.sum() > row["recent"] * 2


@pytest.mark.parametrize(
    "metric",
    ["order_conversion", "completion_rate", "cancellation_rate", "on_time_rate"],
)
def test_rate_headlines_stay_rates(metric):
    """A rate is dimensionless at any window length: it is never per-day, and
    its window aggregate is the mean of the daily rates, bounded by them."""
    mf = compute_metrics(load_gold(), P)
    row = next(r for r in mf.headline(P) if r["metric"] == metric)
    window = _window(mf, metric, P)
    assert row["semantic"] == RATE
    assert row["unit"] == "ratio"
    assert 0.0 <= row["recent"] <= 1.0
    assert window.min() <= row["recent"] <= window.max()


def test_aov_is_an_average_per_order_and_never_per_day():
    """aov = gmv / orders_completed. It is already an average over a denominator
    that is not a day, so "/dia" on it would be a new false statement."""
    mf = compute_metrics(load_gold(), P)
    row = next(r for r in mf.headline(P) if r["metric"] == "aov")
    assert row["semantic"] == PER_ORDER_AVERAGE
    assert row["semantic"] != DAILY_AVERAGE
    gmv = _window(mf, "gmv", P).sum()
    orders = _window(mf, "orders_completed", P).sum()
    # The window's money per order, which the daily mean of a daily ratio
    # approximates to within a couple of per cent -- the point being that the
    # figure lives on a per-ORDER scale, not a per-day one.
    assert row["recent"] == pytest.approx(gmv / orders, rel=0.05)


def test_duration_headlines_are_per_delivery_averages():
    mf = compute_metrics(load_gold(), P)
    row = next(
        r for r in mf.headline(P) if r["metric"] == "avg_actual_delivery_minutes"
    )
    assert row["semantic"] == PER_DELIVERY_AVERAGE
    assert row["unit"] == "minutes"


def test_segment_headlines_carry_the_same_semantics_as_company_ones():
    """The semantic is a property of the METRIC, not of the scope it is measured
    at, so a zone card and a company card cannot disagree about what they show.
    """
    mf = compute_metrics(load_gold(), P)
    for row in mf.headline(P, "zone", "7"):
        assert row["semantic"] == METRIC_SEMANTICS[row["metric"]]
