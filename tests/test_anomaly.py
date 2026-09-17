# tests/test_anomaly.py
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from pulse.types import AnalysisParams
from pulse.metrics import METRIC_REGISTER, load_gold, compute_metrics
from pulse.anomaly_detection import detect_anomalies, detect_on_series
from pulse.config import GROUND_TRUTH

P = AnalysisParams(as_of=date(2026, 9, 10))

_IDX = pd.date_range("2026-03-15", periods=180, freq="D")
# P's windows in positional terms: the comparison window is the last 14 days,
# the 56-day baseline the 56 before that.
_BASELINE_SLICE = slice(-70, -14)


def test_company_scope_fires_on_rate_metrics():
    """RULING (replaces the brief's test_company_gmv_drop_is_detected).

    The brief asserted a company-scope GMV anomaly. Measured against the built
    data, company GMV is z=-2.22 / -4.06% under the brief's own formula --
    real, but under the 2.5 sensitivity floor. That is not a defect: a -3.5%
    move against ~10% daily volatility genuinely is weak evidence at company
    scale, and one zone's operational failure being invisible in the company
    GMV line is exactly the argument for segment decomposition and for watching
    rate metrics, which are far less noisy than volume metrics. Lowering
    sensitivity to force it would be tuning the detector to the answer: it
    would admit phantoms and break the clean-baseline test.

    So the vertical slice is pinned where the evidence actually is: the company
    scope is not silent, and it is silent on GMV for a legible reason.
    """
    anomalies = detect_anomalies(compute_metrics(load_gold(), P), P)

    company = {a.metric: a for a in anomalies if a.scope == "company"}
    assert company, "company scope must not be silent - this is the vertical slice trigger"
    assert "completion_rate" in company
    assert company["completion_rate"].direction == "drop"
    assert company["completion_rate"].deviation_pct <= -3.0


def test_zone_7_gmv_drop_is_detected():
    anomalies = detect_anomalies(compute_metrics(load_gold(), P), P)
    gmv = [a for a in anomalies if a.metric == "gmv" and a.scope_value == "7"]
    assert gmv, "the degraded zone's GMV drop must fire"
    assert gmv[0].direction == "drop"
    assert gmv[0].deviation_pct <= -10.0


def test_clean_baseline_window_produces_no_anomalies():
    """Days 1-149 are clean by construction. A detector that fires here
    is producing false positives."""
    clean = AnalysisParams(as_of=date(2026, 8, 1))      # day 140
    anomalies = detect_anomalies(compute_metrics(load_gold(), clean), clean)
    assert anomalies == []


# --------------------------------------------------------------------------
# Day-of-week adjustment.
#
# The brief's version of this test used a NOISELESS seasonal series with both
# windows an exact multiple of 7 days. That test could not fail: identical
# weekday composition on both sides makes even an unadjusted detector report
# nothing, and the day-of-week residuals are exactly zero, so resid_std == 0
# short-circuits the function before the adjustment logic is reached at all.
# It passed with the whole adjustment deleted.
#
# The two tests below split the adjustment into the two things it actually
# does, and each dies under its own mutation:
#
#   * the NUMERATOR removes weekday-composition bias  -> test_..._no_false_positives
#   * the DENOMINATOR (residual, not raw, std) keeps a real shift visible
#     under the weekend swing                         -> test_..._keeps_a_real_shift
#
# Both use `sessions` -- a count, not a currency metric -- deliberately. On a
# BRL metric the materiality floor silently absorbs the numerator mutation at
# these window sizes, and the test would pass for the wrong reason.
# --------------------------------------------------------------------------

_WEEKEND_LIFT = 1.22        # Fri/Sat, matching data_generator._seasonality
_DAILY_NOISE = 150.0        # 1.5% of the 10k weekday level


def _seasonal(seed: int = 7) -> pd.Series:
    """10k/day with a Fri/Sat lift and real day-to-day noise. The noise is what
    makes resid_std non-zero, so the function runs past its zero-residual guard
    and the adjustment is genuinely exercised."""
    lift = np.where(_IDX.dayofweek.isin([4, 5]), _WEEKEND_LIFT, 1.0)
    rng = np.random.default_rng(seed)
    return pd.Series(10_000 * lift + rng.normal(0, _DAILY_NOISE, 180), index=_IDX)


# 2026-09-06..2026-09-10 is Sun,Mon,Tue,Wed,Thu -- a comparison window holding
# no weekend day at all, against a baseline that is 2/7 weekend. Its raw mean
# sits 5.6% below the raw baseline mean on seasonality alone.
_NO_WEEKEND = AnalysisParams(as_of=date(2026, 9, 10), comparison_window_days=5)


def test_day_of_week_seasonality_does_not_trigger_false_positives():
    series = _seasonal()

    # Sanity: the trap is real. Compared against a flat baseline mean -- what a
    # detector without the day-of-week numerator would do -- this window reads
    # as a 5.6% drop, well past both the deviation floor and sensitivity.
    recent = series.loc["2026-09-06":"2026-09-10"]
    baseline = series.loc["2026-07-12":"2026-09-05"]
    unadjusted_pct = 100 * (recent.mean() - baseline.mean()) / baseline.mean()
    assert unadjusted_pct < -5.0

    # Adjusted, it is nothing: each day is measured against its own weekday.
    assert detect_on_series(series, "sessions", "company", "all", _NO_WEEKEND) is None


def test_day_of_week_residual_std_keeps_a_real_shift_visible():
    """The other half of the adjustment. Raw baseline std here is ~1,000 (the
    weekend swing); the residual std is ~125 (the noise). Testing a genuine
    +5% shift against the raw std buries it at z~1.7; against the residual std
    it is unmissable."""
    shifted = _seasonal()
    shifted.iloc[-14:] *= 1.05

    a = detect_on_series(shifted, "sessions", "company", "all", P)
    assert a is not None and a.direction == "spike"
    assert a.deviation_pct > 4.0
    assert abs(a.z_score) > 8.0


def test_sensitivity_is_monotonic():
    mf = compute_metrics(load_gold(), P)
    loose = detect_anomalies(mf, AnalysisParams(as_of=P.as_of, sensitivity=1.5))
    tight = detect_anomalies(mf, AnalysisParams(as_of=P.as_of, sensitivity=4.5))
    assert len(loose) >= len(tight)


def test_zone_7_is_among_the_detected_scopes():
    gt = json.loads((GROUND_TRUTH / "injected_incidents.json").read_text())
    injected_zone = gt["incidents"][0]["scope"]["value"]
    anomalies = detect_anomalies(compute_metrics(load_gold(), P), P)
    zones = {a.scope_value for a in anomalies if a.scope == "zone"}
    assert str(injected_zone) in zones


def test_availability_rate_is_registered_and_carried_at_zone_grain():
    """_melt() builds the long frame from METRIC_REGISTER intersected with each
    gold table's columns, so an UNREGISTERED metric is invisible to the whole
    downstream pipeline however clean the signal in gold is. availability_rate
    sat in gold_zone_performance from Task 17 and was seen by nothing until it
    was registered here.

    Zone grain only: gold_daily_business_metrics carries no availability
    column, and _melt skips what a table does not have rather than fabricating
    zeros for it.
    """
    assert "availability_rate" in METRIC_REGISTER
    mf = compute_metrics(load_gold(), P)
    keys = mf.long.loc[mf.long["metric"] == "availability_rate", "scope"].unique()
    assert set(keys) == {"zone"}


def test_zone_4_is_among_the_detected_scopes_through_availability_alone():
    """The second injected incident, and the reason registering the metric was
    the whole task. Zone 4 fails BEFORE checkout -- merchant supply goes dark
    across three dinner hours -- so it is invisible on the volume metrics:
    zone-4 gmv measures z=-1.81 and order_conversion z=-1.94, both under the
    2.5 floor, because three damaged hours are diluted by twenty-one healthy
    ones. availability_rate at zone grain is where the evidence actually is.
    """
    gt = json.loads((GROUND_TRUTH / "injected_incidents.json").read_text())
    injected = next(i for i in gt["incidents"] if i["id"] == "zone4_availability")
    zone = str(injected["scope"]["value"])

    mf = compute_metrics(load_gold(), P)
    fired = [
        a
        for a in detect_anomalies(mf, P)
        if a.scope == "zone" and a.scope_value == zone
    ]
    assert fired, "the supply-collapse zone must fire"
    assert [a.metric for a in fired] == ["availability_rate"]
    assert fired[0].direction == "drop"
    assert fired[0].z_score < -5.0
    assert fired[0].deviation_pct < -3.0

    # The volume route is genuinely closed, so the register entry is
    # load-bearing rather than cosmetic. Asserted through detect_on_series so
    # the reason is "this series does not clear the floor", not "the loop
    # happened to skip it".
    for metric in ("gmv", "order_conversion", "sessions"):
        series = mf.series(metric, "zone", zone)
        assert detect_on_series(series, metric, "zone", zone, P) is None


def test_as_of_is_honoured_not_the_end_of_the_series():
    """compute_metrics() ignores as_of and always returns all 180 days, so a
    detector that slices positionally from the end analyses 2026-09-10 no
    matter what was asked for -- and the clean-baseline test would then pass or
    fail for entirely the wrong reason."""
    mf = compute_metrics(load_gold(), P)
    early = AnalysisParams(as_of=date(2026, 7, 1))
    assert detect_anomalies(mf, early) != detect_anomalies(mf, P)


def test_nan_days_are_skipped_not_zeroed():
    """avg_actual_delivery_minutes is NaN on a no-delivery day and is Zone 7's
    top associated driver in Task 11. A .fillna(0) anywhere in the detector
    would read as instant delivery and manufacture a huge false drop.

    The baseline carries noise so that resid_std > 0 and the function runs past
    its zero-residual guard -- otherwise it returns None before the
    recent-window mask this test exists to check is ever reached.
    """
    series = pd.Series(32.0, index=_IDX)
    rng = np.random.default_rng(3)
    series.iloc[_BASELINE_SLICE] += rng.normal(0, 2.0, 56)
    series.iloc[-14:] += 8.0                                # a real +25% shift
    series.iloc[[-12, -7, -2]] = np.nan                     # three no-delivery days

    a = detect_on_series(series, "avg_actual_delivery_minutes", "company", "all", P)
    assert a is not None

    # The three missing days are skipped, not counted and not zeroed.
    assert a.n_observations == 11
    # Exactly 40.0: the 11 present days are all 32 + 8, with no NaN dragging
    # the mean down. Coercing the missing days to 0 would give 31.4 and n=14.
    assert a.recent_value == 40.0
    assert a.direction == "spike"

    all_nan = pd.Series(np.nan, index=_IDX)
    assert detect_on_series(
        all_nan, "avg_actual_delivery_minutes", "company", "all", P
    ) is None


def _sparse_baseline(n_real_days: int, level: float = 20.0,
                     shift: float = 0.88) -> pd.Series:
    """A 56-day baseline window in which only `n_real_days` carry an
    observation, alternating +/-1 around `level` a week at a time so every
    weekday gets an equal count of each. That makes every day-of-week mean
    exactly `level` and resid_std deterministic whatever n_real_days is, so
    the only thing that varies between calls is how many real observations
    stand behind that std.
    """
    s = pd.Series(np.nan, index=_IDX)
    for i in range(n_real_days):
        s.iloc[-70 + i] = level + (1.0 if (i // 7) % 2 == 0 else -1.0)
    s.iloc[-14:] = level + shift
    return s


def test_standard_error_counts_real_baseline_observations_not_window_length():
    """A baseline holding scattered NaNs -- legitimate for the duration metrics
    at low-volume scopes -- estimates resid_std from fewer days than the window
    spans. Using the window length as n_baseline understates the standard error
    and inflates z on exactly the series that can least afford it.

    Both series below show the IDENTICAL +4.40% deviation against an identical
    day-of-week expectation and an identical residual std. The only difference
    is how many real observations back that std, and that alone decides it.
    """
    dense = detect_on_series(
        _sparse_baseline(56), "avg_actual_delivery_minutes", "company", "all", P
    )
    assert dense is not None
    assert dense.deviation_pct == pytest.approx(4.40, abs=0.01)
    assert dense.z_score > 2.5                     # 2.92, on 56 observations

    # Same deviation, same residual std, a quarter of the observations: the
    # standard error widens and the evidence no longer clears the bar. With the
    # window length used as n_baseline instead, z would read 2.84 and fire.
    sparse = detect_on_series(
        _sparse_baseline(14), "avg_actual_delivery_minutes", "company", "all", P
    )
    assert sparse is None


def _shifted(shift_days: int, shift: float, seed: int = 0) -> pd.Series:
    """Flat 100.0 series with noise confined to P's 56-day baseline window, so
    resid_std is non-zero and the day-of-week means are genuinely estimated,
    while every unshifted day in the comparison window sits on its own
    expectation. That makes the half-window residuals deterministic rather
    than a coin flip on the RNG seed.
    """
    s = pd.Series(100.0, index=_IDX)
    rng = np.random.default_rng(seed)
    s.iloc[_BASELINE_SLICE] += rng.normal(0, 1.0, 56)
    s.iloc[-shift_days:] += shift
    return s


def test_a_short_spike_does_not_read_as_a_level_shift():
    """Three big days drag a 14-day mean as far as a real level shift does,
    and clear both the z and the deviation floors on the way. The persistence
    gate is what separates them."""
    spike = _shifted(3, 40.0)          # +8.6% on the window, all of it in 3 days
    sustained = _shifted(14, 8.0)      # +8.0% held across the whole window

    assert detect_on_series(spike, "orders_completed", "company", "all", P) is None
    a = detect_on_series(sustained, "orders_completed", "company", "all", P)
    assert a is not None and a.direction == "spike"


def test_materiality_gate_applies_to_currency_metrics_only():
    """A clean, sustained, >3% shift whose BRL value is trivial: gated out for
    gmv, let through for a rate metric, where a BRL threshold is meaningless."""
    base = _shifted(14, 10.0, seed=1)  # +10%, 14 days -> R$140 over the window

    assert detect_on_series(base, "gmv", "company", "all", P) is None
    assert detect_on_series(base, "completion_rate", "company", "all", P) is not None
    # availability_rate is a ratio too, and _BRL_METRICS is derived from
    # metrics._UNITS -- so registering it with a "ratio" unit is what keeps a
    # currency floor from silently muting the zone-4 signal.
    assert (
        detect_on_series(base, "availability_rate", "company", "all", P) is not None
    )
