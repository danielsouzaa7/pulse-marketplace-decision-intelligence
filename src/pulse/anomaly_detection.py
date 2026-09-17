# Anomaly detection: a day-of-week-adjusted, two-sample z-scan over the whole
# metric register at every scope the MetricFrame carries.
#
# Pure functions, no I/O. This module must never read data/ground_truth/ --
# the whole point of the exercise is that the engine rediscovers the injected
# incident from the evidence alone. Tests may read ground truth to check that
# it did; src/ may not.
#
# Everything reported here is an *association* in the data, never a cause.
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from pulse.metrics import _UNITS, MetricFrame
from pulse.types import AnalysisParams, Anomaly

# min_materiality_brl is a currency threshold, so it is only meaningful for
# currency-denominated metrics. Derived from metrics._UNITS rather than
# hardcoded so that a BRL metric added to the register in Task 17 is gated
# automatically instead of silently escaping the floor. Rates, counts and
# durations bypass it: "is this 3.7pp cancellation-rate move worth more than
# R$5,000?" is not a question a BRL gate can answer, and applying it there
# would mute every rate signal. Turning a deviation into money is Task 12.
_BRL_METRICS = frozenset(m for m, unit in _UNITS.items() if unit == "BRL")

# Relative floor, applied to every metric: a move that is statistically clean
# but only 1% wide is not worth a decision-maker's attention.
_MIN_DEVIATION_PCT = 3.0

# Below this many baseline days the day-of-week means are one observation per
# weekday or worse, and the residual std is not an estimate of anything.
_MIN_BASELINE_OBSERVATIONS = 14


def _windows(series: pd.Series, p: AnalysisParams) -> tuple[pd.Series, pd.Series]:
    """Slice the recent and baseline windows by calendar date anchored on
    p.as_of -- NOT by position from the end of the series. compute_metrics()
    ignores as_of and always returns the full 180 days, so a positional slice
    would analyse 2026-09-10 no matter what as_of asked for.
    """
    recent_end = pd.Timestamp(p.as_of)
    recent_start = recent_end - timedelta(days=p.comparison_window_days - 1)
    baseline_end = recent_start - timedelta(days=1)
    baseline_start = baseline_end - timedelta(days=p.baseline_window_days - 1)
    return series.loc[recent_start:recent_end], series.loc[baseline_start:baseline_end]


def _z(deviation: float, resid_std: float, n_recent: int, n_baseline: int) -> float:
    """Two-sample standard error for (recent mean - day-of-week expectation).

    The expectation is not a known constant: it is built from day-of-week means
    that are themselves estimated from n_baseline days, and that estimation
    error belongs in the denominator. Dividing by resid_std/sqrt(n_recent)
    alone treats the baseline as exact and overstates z by ~12% at the default
    14/56 windows -- enough to turn borderline noise into a page.

    Both counts are counts of real observations, never nominal window lengths:
    a baseline holding scattered NaNs (legitimate for the duration metrics at
    low-volume scopes) estimates resid_std from fewer days than the window
    spans, and passing the window length would understate the standard error
    and inflate z on exactly the series that can least afford it.
    """
    return deviation / (resid_std * np.sqrt(1.0 / n_recent + 1.0 / n_baseline))


def detect_on_series(
    series: pd.Series,
    metric: str,
    scope: str,
    scope_value: str,
    p: AnalysisParams,
) -> Anomaly | None:
    """Test one date-indexed daily series for a day-of-week-adjusted level
    shift over the comparison window. Returns None when the series is too
    short, too flat, too quiet, too brief or too small to be worth reporting.

    Directly callable on a synthetic series -- detect_anomalies() is only the
    register-wide loop around this.
    """
    recent, baseline = _windows(series, p)
    if len(recent) == 0 or len(baseline) < _MIN_BASELINE_OBSERVATIONS:
        return None

    # Day-of-week adjustment. Weekend lift on this marketplace is large enough
    # that a comparison window whose weekday composition differs from the
    # baseline's shows a multi-percent deviation on seasonality alone -- and a
    # window that is not a whole number of weeks always does. Each recent day
    # is compared against its own weekday's baseline mean, and the dispersion
    # tested against is the residual std AFTER removing that weekday effect --
    # not the raw std, which is inflated by exactly the seasonality just
    # modelled (here: ~65% of baseline GMV variance) and would bury a real
    # shift under the weekend swing.
    baseline_dow = baseline.index.dayofweek
    dow_mean = baseline.groupby(baseline_dow).mean()          # NaN-skipping
    if dow_mean.empty or not np.isfinite(dow_mean.to_numpy()).all():
        return None

    expected = pd.Series(
        np.asarray(recent.index.dayofweek.map(dow_mean), dtype=float),
        index=recent.index,
    )
    resid = baseline - np.asarray(baseline_dow.map(dow_mean), dtype=float)
    resid_std = resid.std()                                   # NaN-skipping
    if not np.isfinite(resid_std) or resid_std == 0:
        return None
    # Real observations behind resid_std, not the nominal window length.
    n_baseline = int(baseline.notna().sum())

    # NaN means "missing", never zero: avg_actual_delivery_minutes is NaN on a
    # no-delivery day, and a fabricated 0.0 there would assert instant
    # delivery. Compare only the days present on both sides so recent and
    # expected are averaged over the same calendar days.
    usable = recent.notna() & expected.notna()
    n_observations = int(usable.sum())
    if n_observations == 0:
        return None
    recent_u, expected_u = recent[usable], expected[usable]

    recent_value = float(recent_u.mean())
    baseline_value = float(expected_u.mean())
    if not np.isfinite(baseline_value) or baseline_value == 0:
        return None

    deviation_abs = recent_value - baseline_value
    deviation_pct = 100.0 * deviation_abs / abs(baseline_value)
    z_score = _z(deviation_abs, resid_std, n_observations, n_baseline)

    if abs(z_score) < p.sensitivity or abs(deviation_pct) < _MIN_DEVIATION_PCT:
        return None

    # Persistence. An incident is a sustained level shift; three lucky days can
    # drag a 14-day mean just as far. Split the window in half and require both
    # halves to move the same way and each to carry at least half the
    # sensitivity on its own (smaller, so wider) standard error. Derived from
    # p.sensitivity rather than a separate knob, so the two gates loosen and
    # tighten together. On the clean baseline this roughly halves the scan's
    # false-positive rate; it costs nothing on Zone 7, whose every recent day
    # is inside the degradation.
    half = n_observations // 2
    if half >= 1:
        halves = [
            _z(
                float(recent_u.iloc[a:b].mean() - expected_u.iloc[a:b].mean()),
                resid_std,
                b - a,
                n_baseline,
            )
            for a, b in ((0, half), (half, n_observations))
        ]
        if np.sign(halves[0]) != np.sign(halves[1]):
            return None
        if np.sign(halves[0]) != np.sign(deviation_abs):
            return None
        if min(abs(halves[0]), abs(halves[1])) < p.sensitivity / 2:
            return None

    # Materiality: the deviation is a daily rate, so scale it across the
    # comparison window before comparing it to a window-sized BRL floor.
    # Exact for the daily flows (gmv, contribution_margin: R$/day x days = R$).
    # For aov -- a per-ORDER average, not a daily flow -- delta x days is
    # dimensionally approximate; the honest scaling is delta x orders. It is
    # used anyway because it is conservative at this scale and this is a floor,
    # not an impact estimate. Turning a deviation into money is Task 12.
    if metric in _BRL_METRICS:
        if abs(deviation_abs) * p.comparison_window_days < p.min_materiality_brl:
            return None

    # First day in the window whose own residual moves at least one residual
    # std in the anomaly's direction -- roughly "when the evidence starts",
    # bounded by the window, since that is all this function looked at.
    signed = (recent_u - expected_u) * np.sign(deviation_abs)
    onset = signed[signed >= resid_std]
    first_detected = (onset.index[0] if len(onset) else recent_u.index[0]).date()

    return Anomaly(
        metric=metric,
        scope=scope,
        scope_value=str(scope_value),
        recent_value=recent_value,
        baseline_value=baseline_value,
        deviation_abs=deviation_abs,
        deviation_pct=deviation_pct,
        z_score=float(z_score),
        direction="drop" if deviation_abs < 0 else "spike",
        first_detected_date=first_detected,
        n_observations=n_observations,
    )


def detect_anomalies(metric_frame: MetricFrame, p: AnalysisParams) -> list[Anomaly]:
    """Scan every (metric, scope, scope_value) the frame carries.

    Deliberately register-wide rather than GMV-only: a zone-local operational
    failure can be diluted below the sensitivity floor on company volume
    metrics while moving the company rate metrics hard. A volume-only scan
    would leave the company level silent on an incident that is plainly there.
    """
    keys = (
        metric_frame.long[["metric", "scope", "scope_value"]]
        .drop_duplicates()
        .sort_values(["scope", "metric", "scope_value"])
    )
    found = [
        anomaly
        for k in keys.itertuples(index=False)
        if (
            anomaly := detect_on_series(
                metric_frame.series(k.metric, k.scope, k.scope_value),
                k.metric,
                k.scope,
                k.scope_value,
                p,
            )
        )
        is not None
    ]
    # Strongest evidence first; ranking by business impact is Task 12.
    return sorted(
        found, key=lambda a: (-abs(a.z_score), a.scope, a.metric, a.scope_value)
    )
