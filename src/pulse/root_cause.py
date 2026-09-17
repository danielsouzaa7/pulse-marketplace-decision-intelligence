# Diagnosis: given an anomaly, explain it from measured evidence.
#
# Four steps, each computed and none asserted:
#   1. segment_contributions  -- where in the business the deviation sits
#   2. funnel_decomposition   -- which stage of the GMV identity moved
#   3. associated_drivers     -- which operational series moved with it
#   4. classify_pattern       -- (stage, driver) -> playbook key, via a table
# plus compute_confidence(), which scores how much evidence the above found.
#
# Pure functions over the gold tables. This module must never read
# data/ground_truth/ -- the engine has to rediscover the injected incident from
# the evidence alone. Tests may read ground truth to check that it did.
#
# EVERYTHING HERE IS ASSOCIATION, NEVER CAUSATION. A contribution is a share of
# a measured deviation; a driver is a series whose movement is correlated with
# the target's. Neither establishes that one produced the other. The wording in
# comments, docstrings and returned strings is "associated driver",
# "contribution", "correlated deterioration", "evidence consistent with" --
# never "caused by", "because of" or "proves". Causal claims need the
# Experiment Lab, not a correlation coefficient.
from __future__ import annotations

import math
from functools import lru_cache

import pandas as pd

from pulse.anomaly_detection import _windows
from pulse.metrics import _UNITS, GoldTables, MetricFrame, compute_metrics
from pulse.types import (
    AnalysisParams,
    Anomaly,
    AssociatedDriver,
    Diagnosis,
    FunnelStage,
    SegmentContribution,
)

# GMV = sessions x order_conversion x completion_rate x aov. Exact in gold
# (verified daily to 1e-12), which is what makes the log decomposition below a
# decomposition rather than a regression.
FUNNEL_STAGES: tuple[str, ...] = (
    "sessions",
    "order_conversion",
    "completion_rate",
    "aov",
)

# Driver candidates -- and the exclusions are the analytically load-bearing
# part of this module, so they are stated as a rule rather than a list.
#
# RULE: a driver candidate must be plausibly OPERATIONALLY UPSTREAM of the
# broken funnel stage. Delivery time, promised ETA and on-time rate describe
# fulfilment performance, which is upstream of whether a placed order
# completes. Correlating an anomaly against its own arithmetic factors or its
# own downstream consequences produces high coefficients that explain nothing.
#
# Consequences of the rule, i.e. why every other register metric is out:
#   * sessions, order_conversion, completion_rate, aov -- the four funnel
#     identity terms. GMV is their product, so they correlate with it by
#     construction. Circular.
#   * cancellation_rate -- the arithmetic complement of completion_rate (every
#     order either completes or cancels). Measured on zone 7 the two come out
#     at +0.478 and -0.478 against GMV: exactly mirrored, which is the
#     signature of one quantity with a sign flip, not two signals.
#   * gmv, orders_placed, orders_completed -- volume restatements of the same
#     identity.
#   * contribution_margin -- downstream, not upstream. It is computed FROM
#     completed orders (commission on item amount, plus fees, less delivery
#     cost and discounts) and shares GMV's dominant term, so it measures
#     +0.880 against GMV on zone 7. It would rank first and classify the
#     incident as a margin pattern, when it is a consequence of the fulfilment
#     deterioration rather than an input to it.
#   * active_customers -- daily-distinct, not carried at zone scope.
#   * availability_rate (registered in Task 18) -- genuinely upstream of
#     order_conversion, so the rule does not exclude it, and it is left out on
#     evidence instead: no anomaly on this dataset has it as a plausible
#     driver. Zone 4 is the only scope where availability moved, and there it
#     is the TARGET (nothing else in zone 4 clears the floor: gmv z = -1.81,
#     order_conversion z = -1.94), where a candidate is skipped anyway because
#     a series correlates with itself at 1.0. Adding it would therefore buy no
#     classification that is reachable today while putting a fourth candidate
#     into the zone-7 ranking, whose top three already cluster inside 0.02.
#     If a future scope fires on conversion WITH availability moving, this is
#     the one-line change, and PATTERNS gets an ("order_conversion",
#     "availability_rate") row at the same time -- not before.
DRIVER_CANDIDATES: tuple[str, ...] = (
    "avg_actual_delivery_minutes",
    "avg_promised_eta_minutes",
    "on_time_rate",
)

# (funnel_break_stage, classification driver) -> playbook key. DATA, not
# branches: adding a pattern is a row here plus a matching entry in
# playbook.yml, and classification never looks at a zone id, a scope value or
# any other identifier.
#
# TWO RULES GOVERN WHAT MAY BE ADDED, both learned the hard way:
#
#   1. A ROW MUST BE REACHABLE. A key no diagnosis can produce is unverifiable
#      theatre. Task 18 deleted three such rows -- ("order_conversion",
#      "merchant_availability"), ("contribution_margin", "discount_rate") and
#      ("repeat_rate", "cohort_quality") -- none of which named a metric this
#      engine measures or a stage it can report. See the report for the
#      measurements that retired them.
#   2. MAP THE WHOLE FAMILY, NEVER ONE MEMBER. All three fulfilment drivers map
#      to the same key because they describe one operational signature and rank
#      within 0.02 of each other; a classification that flips between runs is
#      worse than a coarser one that holds.
#
# The None driver is not "no evidence" -- it is the signature of a metric that
# IS the operational statement, with no upstream candidate co-moving with it.
# availability_rate is the case: merchant supply going dark is not something
# fulfilment performance explains, and on zone 4 it does not (|r| 0.13-0.16,
# all three weak). See _classification_driver().
PATTERNS: dict[tuple[str, str | None], str] = {
    ("completion_rate", "avg_actual_delivery_minutes"): "fulfillment_eta_degradation",
    ("completion_rate", "avg_promised_eta_minutes"): "fulfillment_eta_degradation",
    ("completion_rate", "on_time_rate"): "fulfillment_eta_degradation",
    ("availability_rate", None): "supply_availability_gap",
}
UNKNOWN_PATTERN = "unknown_pattern"

# Segment shares only mean something for metrics whose segment values sum to
# the company value. Derived from metrics._UNITS rather than hardcoded so a
# metric added to the register in Task 17/18 is classified automatically. Rates
# and durations are averages: summing a zone's completion_rate across zones is
# not the company completion_rate, so they get no decomposition rather than a
# wrong one.
_ADDITIVE_UNITS = frozenset({"BRL", "orders", "sessions"})

# Correlation thresholds. Chosen for daily marketplace operations series, where
# real associations are diluted by weekday seasonality and day-to-day noise:
# 0.60+ is a strong co-movement, 0.35-0.60 a moderate one, below that the
# series are not moving together in any way worth reporting as evidence.
_STRONG, _MODERATE = 0.60, 0.35

# Below this many pairwise-complete days a correlation is a coincidence, not a
# measurement: |r| = 0.9 on 3 points is noise. Strength is forced to "weak"
# regardless of the coefficient.
_MIN_CORRELATION_OBSERVATIONS = 8

# Lag search range for temporal alignment, in days. Deliberately small: over a
# 14-day comparison window a larger range spends observations on shifts that
# could not be read as alignment anyway.
_MAX_LAG_DAYS = 3


@lru_cache(maxsize=2)
def _frame(gold: GoldTables, p: AnalysisParams) -> MetricFrame:
    """compute_metrics() is a pure melt of the gold tables; cached so one
    diagnose() call does not rebuild it once per step.
    """
    return compute_metrics(gold, p)


def _window_means(series: pd.Series, p: AnalysisParams) -> tuple[float, float]:
    """(recent mean, baseline mean) over the detector's own calendar windows.

    _windows() is imported from anomaly_detection on purpose rather than
    reimplemented: a diagnosis has to describe exactly the window that fired,
    and two copies of the slicing arithmetic would drift.

    BASELINE CONTAMINATION, and it is not corrected here: the 56-day baseline
    ends 2026-08-27 while the zone 7 degradation starts 2026-08-11, so 17 of
    the 56 baseline days are already degraded. Every deviation this module
    reports is therefore an UNDERSTATEMENT of the true incident. That is the
    honest behaviour of a trailing-baseline detector and is left in place;
    Task 12 should word impact conservatively for the same reason.
    """
    recent, baseline = _windows(series, p)
    return float(recent.mean()), float(baseline.mean())  # both NaN-skipping


def _analysis_window(series: pd.Series, p: AnalysisParams) -> pd.Series:
    """Baseline followed by comparison window -- the span the diagnosis reasons
    over (70 days at default params), used for driver correlations.
    """
    recent, baseline = _windows(series, p)
    return pd.concat([baseline, recent])


def _deviation_pct(recent: float, baseline: float) -> float:
    if not math.isfinite(baseline) or baseline == 0:
        return float("nan")
    return 100.0 * (recent - baseline) / abs(baseline)


def segment_contributions(
    anomaly: Anomaly, gold: GoldTables, p: AnalysisParams
) -> tuple[SegmentContribution, ...]:
    """Decompose the COMPANY-level deviation of the anomaly's metric across the
    segments of one dimension, signed, ranked by magnitude.

    Signed so offsetting segments stay visible: a share of -40% means that
    segment moved against the company deviation and is masking it. Shares are
    taken against the sum of the segment deviations, so they total 100% by
    construction.

    Note this runs whether or not the company-level metric itself fired as an
    anomaly. Computing a decomposition and firing an alert are different
    operations -- on this dataset the company GMV line does not clear the
    sensitivity floor (z = -2.22 against 2.5) precisely because one zone's
    deterioration is diluted by seven healthy ones, which is the argument for
    decomposing it rather than a reason not to.

    Returns () when the metric is not additive across segments, or when the
    segment deviations cancel to ~0 and shares would be meaningless.
    """
    if _UNITS.get(anomaly.metric) not in _ADDITIVE_UNITS:
        return ()

    frame = _frame(gold, p)
    # The dimension the anomaly is scoped to; company-scoped anomalies fall
    # back to the structural dimension gold carries (zone). Task 17's extra
    # dimensions arrive as scopes on the same frame.
    dimension = anomaly.scope if anomaly.scope != "company" else "zone"
    long = frame.long
    segments = sorted(
        long.loc[
            (long["scope"] == dimension) & (long["metric"] == anomaly.metric),
            "scope_value",
        ].unique()
    )
    if not segments:
        return ()

    stats = {
        s: _window_means(frame.series(anomaly.metric, dimension, s), p)
        for s in segments
    }
    deviations = {s: recent - base for s, (recent, base) in stats.items()}
    total = sum(deviations.values())
    scale = sum(abs(base) for _, base in stats.values())
    # Near-zero total: segments cancelled out, and dividing by that would print
    # shares like +3000% / -2900%. No decomposition is better than a fake one.
    if not math.isfinite(total) or abs(total) <= 1e-6 * scale:
        return ()

    ranked = sorted(segments, key=lambda s: (-abs(deviations[s]), s))
    return tuple(
        SegmentContribution(
            dimension=dimension,
            segment=s,
            segment_deviation_abs=deviations[s],
            contribution_pct=100.0 * deviations[s] / total,
            rank=i + 1,
        )
        for i, s in enumerate(ranked)
    )


def funnel_decomposition(
    gold: GoldTables, scope: str, scope_value: str, p: AnalysisParams
) -> tuple[FunnelStage, ...]:
    """GMV = sessions x order_conversion x completion_rate x aov.

    Taking logs makes the percentage changes additive --
    dln(GMV) = dln(sessions) + dln(conv) + dln(completion) + dln(aov) -- so the
    largest-magnitude term IS the broken stage. Computed, not asserted: no
    stage is privileged, and a demand-side incident would surface as sessions
    through exactly the same code path.
    """
    frame = _frame(gold, p)
    rows = []
    for stage in FUNNEL_STAGES:
        try:
            series = frame.series(stage, scope, scope_value)
        except KeyError:
            # Stage not carried at this scope. Not an error -- associated_drivers
            # already treats a missing series that way -- and not a movement
            # either: a stage nobody measured cannot be the broken one. Raising
            # here would turn a scope with partial coverage into a traceback.
            recent = baseline = float("nan")
        else:
            recent, baseline = _window_means(series, p)
        usable = (
            math.isfinite(recent)
            and math.isfinite(baseline)
            and recent > 0
            and baseline > 0
        )
        log_contribution = math.log(recent / baseline) if usable else 0.0
        rows.append(
            (stage, recent, baseline, _deviation_pct(recent, baseline), log_contribution)
        )

    # A funnel in which nothing measurably moved has no break to report. Without
    # this, max() hands is_primary_break to whichever stage happens to sort
    # first and the diagnosis asserts that demand collapsed on the strength of
    # no evidence at all. diagnose() then falls back to naming the metric.
    worst = max(rows, key=lambda r: abs(r[4]))
    if worst[4] == 0.0:
        return tuple(FunnelStage(*r, is_primary_break=False) for r in rows)
    return tuple(FunnelStage(*r, is_primary_break=(r is worst)) for r in rows)


def driver_correlation(
    target: pd.Series, driver: pd.Series, max_lag: int = _MAX_LAG_DAYS
) -> tuple[float, int, int]:
    """(correlation, temporal_alignment_days, n_observations) for one candidate.

    Pearson correlation on PAIRWISE-COMPLETE observations. Duration metrics are
    NaN on a no-delivery day and NaN means "not measured", never zero --
    .fillna(0) here would assert instant delivery on those days and drag the
    coefficient toward a number nobody observed. pandas' .corr() already drops
    pairs where either side is missing; n_observations reports how many days
    actually survived that, which is what the evidence strength is judged on.

    temporal_alignment_days is the lag maximising |correlation| over
    -max_lag..+max_lag; 0 means the two series move together contemporaneously.
    Positive means the driver moved FIRST by that many days. The reported
    correlation stays the contemporaneous one -- the lag is reported as
    alignment evidence, not substituted for the measurement, and a lagged fit
    is computed on fewer overlapping days.

    Alignment is not a direction of causation, and on a sustained level shift
    it is weak evidence even of ordering: every lag inside the window overlaps
    the same degraded regime, so the lag surface is close to flat (on zone 7 it
    moves only between |r| 0.44 and 0.49 across -7..+7 days). Report it; do not
    lean on it.
    """
    n_observations = int(pd.concat([target, driver], axis=1).dropna().shape[0])
    correlation = float(target.corr(driver)) if n_observations >= 2 else float("nan")
    if not math.isfinite(correlation):
        return float("nan"), 0, n_observations

    # A correlation already too thin to count as evidence gets no alignment
    # search either: shifting it leaves even fewer overlapping days.
    if n_observations < _MIN_CORRELATION_OBSERVATIONS:
        return correlation, 0, n_observations

    best_lag, best_abs = 0, abs(correlation)
    for lag in range(-max_lag, max_lag + 1):
        lagged = target.corr(driver.shift(lag))
        if pd.notna(lagged) and abs(lagged) > best_abs + 1e-12:
            best_lag, best_abs = lag, abs(lagged)
    return correlation, best_lag, n_observations


def _evidence_strength(correlation: float, n_observations: int) -> str:
    if n_observations < _MIN_CORRELATION_OBSERVATIONS or not math.isfinite(correlation):
        return "weak"
    magnitude = abs(correlation)
    if magnitude >= _STRONG:
        return "strong"
    return "moderate" if magnitude >= _MODERATE else "weak"


def associated_drivers(
    gold: GoldTables,
    scope: str,
    scope_value: str,
    target_metric: str,
    p: AnalysisParams,
) -> tuple[AssociatedDriver, ...]:
    """Operational metrics whose daily series moved with the target's, ranked by
    |correlation| over the analysis window.

    Candidates come from DRIVER_CANDIDATES -- see the rule stated there. These
    are associations: a high coefficient says the two series deteriorated
    together, not that one produced the other.

    Ranking is deterministic (|correlation|, then metric name) because the
    candidates cluster within ~0.02 of each other on this dataset and an
    unstable sort would let the downstream pattern key flip between runs.
    """
    frame = _frame(gold, p)
    try:
        target = _analysis_window(frame.series(target_metric, scope, scope_value), p)
    except KeyError:
        # No target series at this scope, so there is nothing for a candidate to
        # be correlated against. An empty driver list is the honest answer and
        # diagnose() already renders it (the pattern falls back to
        # unknown_pattern); raising would take a whole decision cycle down over
        # one scope gold happens not to carry.
        return ()

    drivers = []
    for metric in DRIVER_CANDIDATES:
        if metric == target_metric:
            continue  # a series correlates with itself at 1.0 and says nothing
        try:
            series = frame.series(metric, scope, scope_value)
        except KeyError:
            continue  # not carried at this scope; not an error
        correlation, alignment, n_observations = driver_correlation(
            target, _analysis_window(series, p)
        )
        if not math.isfinite(correlation):
            continue  # no measurable association
        recent, baseline = _window_means(series, p)
        drivers.append(
            AssociatedDriver(
                metric=metric,
                recent=recent,
                baseline=baseline,
                deviation_pct=_deviation_pct(recent, baseline),
                correlation_with_target=correlation,
                temporal_alignment_days=alignment,
                evidence_strength=_evidence_strength(correlation, n_observations),
            )
        )
    return tuple(
        sorted(drivers, key=lambda d: (-abs(d.correlation_with_target), d.metric))
    )


def _classification_driver(drivers: tuple[AssociatedDriver, ...]) -> str | None:
    """The driver half of the PATTERNS key: the best-correlated driver whose
    evidence is NOT weak, or None when no candidate reached the threshold.

    Drivers arrive ranked by |correlation|, so this is the strongest one that
    counts as evidence at all.

    WHY WEAK DRIVERS ARE EXCLUDED, and why this is not a tuning knob. A weak
    correlation already contributes exactly zero to compute_confidence() below;
    letting the same coefficient pick a PLAYBOOK -- an action a human is asked
    to approve -- would be the one place in this module where evidence the
    engine has declared worthless still decides an outcome. It is also the
    ordering trap at its worst: zone 4's three candidates measure |r| 0.164,
    0.148 and 0.131 against availability_rate, a spread far too narrow to rank,
    so whichever "won" would be a coin flip deciding the recommendation.

    Returning None is a positive finding, not a gap: "this metric moved and no
    upstream operational candidate moved with it". PATTERNS keys on it.
    """
    return next((d.metric for d in drivers if d.evidence_strength != "weak"), None)


def classify_pattern(funnel_break_stage: str, top_driver_metric: str | None) -> str:
    """(stage, driver) -> playbook key through PATTERNS. Unmatched signatures
    return "unknown_pattern" explicitly, so the engine never returns None and
    Task 13 always has a key to look up.

    A None driver is looked up like any other half-key rather than short-
    circuiting to unknown: PATTERNS carries a row for the signature "this stage
    moved and nothing upstream moved with it" (availability_rate), and the
    stages that have no such row -- every other one -- still fall back.
    """
    return PATTERNS.get((funnel_break_stage, top_driver_metric), UNKNOWN_PATTERN)


def compute_confidence(
    anomaly: Anomaly,
    primary_segment: SegmentContribution | None,
    drivers: tuple[AssociatedDriver, ...],
    p: AnalysisParams,
) -> float:
    """The documented formula, verbatim:

        confidence = 0.25 * min(|z| / 5, 1)
                   + 0.25 * top_contribution_pct / 100
                   + 0.30 * max|driver_correlation|
                   + 0.20 * consecutive_anomalous_days / comparison_window_days

    clipped to [0, 1]. Each term is evidence ACTUALLY FOUND, so missing
    evidence subtracts: no decomposition contributes 0 to the second term, no
    drivers (or only weak ones, including correlations measured on too few
    pairwise-complete days) contribute 0 to the third. Nothing defaults to a
    neutral 0.5 -- a thinly-evidenced diagnosis must not score like a
    well-evidenced one.
    """
    statistical = min(abs(anomaly.z_score) / 5.0, 1.0)

    # Shares can exceed 100% when other segments offset; cap the term rather
    # than let an offsetting decomposition inflate confidence past its weight.
    concentration = (
        min(abs(primary_segment.contribution_pct) / 100.0, 1.0)
        if primary_segment is not None
        else 0.0
    )

    # Weak drivers contribute nothing: a correlation below the reporting
    # threshold, or measured on too few days, is not evidence.
    association = max(
        (
            abs(d.correlation_with_target)
            for d in drivers
            if d.evidence_strength != "weak"
        ),
        default=0.0,
    )

    # Days of evidence inside the comparison window, from the detector's own
    # onset date ("when the evidence starts") through as_of.
    days = (p.as_of - anomaly.first_detected_date).days + 1
    persistence = min(max(days, 0) / p.comparison_window_days, 1.0)

    score = (
        0.25 * statistical
        + 0.25 * concentration
        + 0.30 * association
        + 0.20 * persistence
    )
    return min(max(score, 0.0), 1.0)


def diagnose(anomaly: Anomaly, gold: GoldTables, p: AnalysisParams) -> Diagnosis:
    """Explain one anomaly from measured evidence: where it concentrates, which
    funnel stage moved, which operational series moved with it, and which
    playbook signature that combination matches.

    Every field is derived from the gold tables. Nothing here reads ground
    truth, branches on a zone id, or special-cases a metric name.
    """
    contributions = segment_contributions(anomaly, gold, p)
    # The anomaly's own segment if the decomposition covers it (that is the
    # share the memo's CONCENTRATION field reports), else the largest
    # contributor. No identifier is hardcoded -- this is a lookup by the
    # anomaly's own scope_value.
    primary_segment = next(
        (c for c in contributions if c.segment == anomaly.scope_value),
        contributions[0] if contributions else None,
    )

    # The funnel identity is about GMV, so it is only meaningful for a GMV
    # anomaly. For any other metric the broken "stage" is the metric itself.
    funnel = (
        funnel_decomposition(gold, anomaly.scope, anomaly.scope_value, p)
        if anomaly.metric == "gmv"
        else ()
    )
    funnel_break_stage = next(
        (f.stage for f in funnel if f.is_primary_break), anomaly.metric
    )

    drivers = associated_drivers(
        gold, anomaly.scope, anomaly.scope_value, anomaly.metric, p
    )

    return Diagnosis(
        anomaly=anomaly,
        contributions=contributions,
        primary_segment=primary_segment,
        funnel=funnel,
        funnel_break_stage=funnel_break_stage,
        drivers=drivers,
        pattern=classify_pattern(
            funnel_break_stage, _classification_driver(drivers)
        ),
        confidence=compute_confidence(anomaly, primary_segment, drivers, p),
    )
