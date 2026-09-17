# tests/test_root_cause.py
import dataclasses
import json
import math
from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from pulse.anomaly_detection import detect_anomalies
from pulse.config import GROUND_TRUTH
from pulse.metrics import METRIC_REGISTER, compute_metrics, load_gold
from pulse.root_cause import (
    DRIVER_CANDIDATES,
    FUNNEL_STAGES,
    PATTERNS,
    _analysis_window,
    _classification_driver,
    _frame,
    associated_drivers,
    classify_pattern,
    compute_confidence,
    diagnose,
    driver_correlation,
    funnel_decomposition,
)
from pulse.types import AnalysisParams, AssociatedDriver

P = AnalysisParams(as_of=date(2026, 9, 10))


@lru_cache(maxsize=1)
def zone7_gmv_diagnosis():
    """RULING (replaces the brief's company_gmv_diagnosis()).

    The brief selected a company-scope GMV anomaly. No such anomaly exists:
    company GMV measures z=-2.22 against a 2.5 sensitivity floor and does not
    fire -- deliberately, per the same ruling pinned in test_anomaly.py. One
    zone's operational failure being invisible in the company GMV line is the
    product's central argument for segment decomposition, not a defect to tune
    away. The diagnosis is therefore anchored on the zone/7 GMV anomaly
    (z=-2.88, -13.45%), which is where the evidence actually is. Every other
    assertion the brief made still applies.
    """
    gold = load_gold()
    anomaly = next(
        a
        for a in detect_anomalies(compute_metrics(gold, P), P)
        if a.metric == "gmv" and a.scope == "zone" and a.scope_value == "7"
    )
    return diagnose(anomaly, gold, P)


@lru_cache(maxsize=1)
def zone4_availability_diagnosis():
    """Zone 4's detection route, measured rather than assumed.

    Zone 4 carries the injected supply incident and does NOT clear the
    sensitivity floor on gmv (z=-1.81) or on order_conversion (z=-1.94): the
    conversion damage is confined to three dinner hours and is diluted by the
    rest of an otherwise healthy day. What does clear it, at z=-6.58 / -5.69%,
    is availability_rate at ZONE grain -- a column gold has carried since Task
    17 but which nothing downstream could see until Task 18 registered it,
    since _melt() only picks up registered columns.

    Note this deliberately selects zone 4 by SCOPE, not by metric: if zone 4
    ever starts firing on something else the selection still finds it rather
    than silently skipping to a StopIteration.
    """
    gold = load_gold()
    anomaly = next(
        a
        for a in detect_anomalies(compute_metrics(gold, P), P)
        if a.scope == "zone" and a.scope_value == "4"
    )
    return diagnose(anomaly, gold, P)


# --- segment contributions ---------------------------------------------------


def test_contributions_sum_to_one_hundred_percent():
    total = sum(c.contribution_pct for c in zone7_gmv_diagnosis().contributions)
    assert abs(total - 100.0) < 1.0


def test_contributions_are_signed_so_offsetting_segments_stay_visible():
    """Zones that moved AGAINST the company deviation must show a negative
    share, not be folded into a magnitude-only total that hides them.
    """
    contributions = zone7_gmv_diagnosis().contributions
    assert any(c.contribution_pct < 0 for c in contributions)
    # And a signed share is consistent with the sign of its own deviation.
    total = sum(c.segment_deviation_abs for c in contributions)
    for c in contributions:
        assert math.copysign(1, c.contribution_pct) == math.copysign(
            1, c.segment_deviation_abs * total
        )


def test_engine_recovers_the_injected_zone_without_reading_ground_truth():
    """The RECOVERY test. Ground truth is read here, in the test, only to
    check the engine independently reached the same answer."""
    gt = json.loads((GROUND_TRUTH / "injected_incidents.json").read_text())
    injected = str(gt["incidents"][0]["scope"]["value"])
    d = zone7_gmv_diagnosis()
    assert d.primary_segment.segment == injected
    assert d.primary_segment.rank == 1
    assert d.primary_segment.contribution_pct >= 55.0


def test_rate_metrics_get_no_segment_decomposition():
    """Zone completion_rates do not sum to the company completion_rate, so
    decomposing one would produce a confident wrong number. No decomposition
    is the correct answer, and confidence drops accordingly."""
    gold = load_gold()
    anomaly = next(
        a
        for a in detect_anomalies(compute_metrics(gold, P), P)
        if a.metric == "completion_rate" and a.scope == "zone"
    )
    d = diagnose(anomaly, gold, P)
    assert d.contributions == ()
    assert d.primary_segment is None


# --- funnel decomposition ----------------------------------------------------


def test_zone7_breaks_at_completion_rate_not_conversion():
    d = zone7_gmv_diagnosis()
    assert d.funnel_break_stage == "completion_rate"
    breaks = [f for f in d.funnel if f.is_primary_break]
    assert len(breaks) == 1 and breaks[0].stage == "completion_rate"


def test_funnel_log_contributions_reconstruct_the_gmv_change():
    d = zone7_gmv_diagnosis()
    total = sum(f.log_contribution for f in d.funnel)
    expected = math.log(d.anomaly.recent_value / d.anomaly.baseline_value)
    assert abs(total - expected) < 0.02


def test_stable_demand_is_not_reported_as_the_break():
    """Sessions into zone 7 barely moved; a decomposition that named demand as
    the break would be reading an incident that is not in the data."""
    d = zone7_gmv_diagnosis()
    assert d.funnel_break_stage != "sessions"
    sessions = next(f for f in d.funnel if f.stage == "sessions")
    breaker = next(f for f in d.funnel if f.is_primary_break)
    assert abs(sessions.log_contribution) < 0.05
    assert abs(sessions.log_contribution) < abs(breaker.log_contribution) / 3
    assert not sessions.is_primary_break


def test_funnel_only_runs_for_gmv_anomalies():
    """The identity is about GMV. For any other metric the broken stage is the
    metric itself, and no funnel is fabricated for it."""
    gold = load_gold()
    anomaly = next(
        a
        for a in detect_anomalies(compute_metrics(gold, P), P)
        if a.metric == "on_time_rate"
    )
    d = diagnose(anomaly, gold, P)
    assert d.funnel == ()
    assert d.funnel_break_stage == "on_time_rate"


# --- associated drivers ------------------------------------------------------


def test_delivery_performance_is_the_associated_driver():
    d = zone7_gmv_diagnosis()
    assert d.drivers[0].metric in (
        "avg_actual_delivery_minutes",
        "avg_promised_eta_minutes",
        "on_time_rate",
    )
    assert d.drivers[0].metric == "avg_actual_delivery_minutes"
    assert d.drivers[0].evidence_strength in ("strong", "moderate")
    assert d.drivers[0].correlation_with_target < 0  # slower deliveries, lower GMV


def test_identity_terms_and_downstream_metrics_are_never_drivers():
    """The tautology guard. GMV's own arithmetic factors correlate with it by
    construction; cancellation_rate is completion_rate with a sign flip; and
    contribution_margin is computed FROM completed orders (it measures +0.880
    against zone 7 GMV and would otherwise rank first and misclassify the
    pattern). None of them are evidence about GMV.
    """
    names = {dr.metric for dr in zone7_gmv_diagnosis().drivers}
    assert names.isdisjoint(set(FUNNEL_STAGES))
    assert names.isdisjoint(
        {
            "gmv",
            "orders_placed",
            "orders_completed",
            "cancellation_rate",
            "contribution_margin",
            "active_customers",
        }
    )
    assert names == set(DRIVER_CANDIDATES)


def test_correlation_uses_pairwise_complete_observations_not_zero_fill():
    """avg_actual_delivery_minutes is NaN on a no-delivery day. Filling those
    with 0.0 would assert instant delivery and corrupt the coefficient, so the
    correlation must drop the incomplete pairs and report the n it actually
    used -- not the window length.
    """
    index = pd.date_range("2026-01-01", periods=20, freq="D")
    target = pd.Series(np.arange(20.0), index=index)
    driver = pd.Series(np.arange(20.0) * 3.0 + 5.0, index=index)
    driver.iloc[3] = np.nan
    driver.iloc[11] = np.nan

    correlation, _lag, n_observations = driver_correlation(target, driver)

    assert n_observations == 18
    assert n_observations != len(target)
    # Perfectly collinear on the days that exist. A fillna(0) would have
    # injected two (x, 0) outliers and dragged this well below 1.0.
    assert correlation == pytest.approx(1.0)
    assert driver_correlation(target, driver.fillna(0.0))[0] < 0.99


def test_correlation_on_too_few_observations_is_weak_however_high():
    """|r| = 1.0 on 3 points is noise, and must not be reported as evidence."""
    index = pd.date_range("2026-01-01", periods=20, freq="D")
    target = pd.Series(np.arange(20.0), index=index)
    driver = pd.Series(np.nan, index=index)
    driver.iloc[:3] = [1.0, 2.0, 3.0]

    correlation, _lag, n_observations = driver_correlation(target, driver)
    assert n_observations == 3
    assert correlation == pytest.approx(1.0)

    from pulse.root_cause import _evidence_strength

    assert _evidence_strength(correlation, n_observations) == "weak"
    assert _evidence_strength(correlation, 70) == "strong"


def test_driver_ranking_follows_the_data_not_a_declared_order():
    """Shuffling the winning driver's series destroys its co-movement with GMV
    and must cost it first place. If the ranking were baked in, it would not."""
    frame = _frame(load_gold(), P)
    target = _analysis_window(frame.series("gmv", "zone", "7"), P)
    series = {
        m: _analysis_window(frame.series(m, "zone", "7"), P) for m in DRIVER_CANDIDATES
    }

    def ranked(candidates):
        return sorted(
            candidates,
            key=lambda m: (-abs(driver_correlation(target, candidates[m])[0]), m),
        )

    top = ranked(series)[0]
    assert top == zone7_gmv_diagnosis().drivers[0].metric

    shuffled = dict(series)
    values = series[top].to_numpy().copy()
    np.random.default_rng(7).shuffle(values)
    shuffled[top] = pd.Series(values, index=series[top].index)

    assert ranked(shuffled)[0] != top
    assert abs(driver_correlation(target, shuffled[top])[0]) < abs(
        driver_correlation(target, series[top])[0]
    )


def test_temporal_alignment_is_contemporaneous_when_nothing_leads():
    index = pd.date_range("2026-01-01", periods=30, freq="D")
    wave = np.sin(np.arange(30) / 2.0)
    target = pd.Series(wave, index=index)
    assert driver_correlation(target, pd.Series(wave, index=index))[1] == 0
    # Driver shifted three days EARLIER leads the target by three days.
    lead = pd.Series(np.roll(wave, -3), index=index)
    assert driver_correlation(target, lead)[1] == 3


# --- pattern classification --------------------------------------------------


def test_pattern_classifies_to_the_fulfillment_playbook_key():
    assert zone7_gmv_diagnosis().pattern == "fulfillment_eta_degradation"


def test_unmatched_signatures_fall_back_explicitly():
    assert classify_pattern("sessions", "on_time_rate") == "unknown_pattern"
    assert classify_pattern("completion_rate", None) == "unknown_pattern"
    assert classify_pattern("availability_rate", None) == "supply_availability_gap"


def test_every_pattern_row_names_a_stage_and_driver_the_engine_can_produce():
    """Rule 1 of the PATTERNS table: a row no diagnosis can reach is
    unverifiable theatre. The stage half is either a funnel stage or a
    registered metric (diagnose() falls back to the metric name for anything
    that is not a GMV anomaly); the driver half is either a declared candidate
    or None. Task 18 retired three rows that satisfied neither half --
    ("order_conversion", "merchant_availability"), ("contribution_margin",
    "discount_rate") and ("repeat_rate", "cohort_quality") -- none of which
    named a metric this engine measures.
    """
    stages = set(FUNNEL_STAGES) | set(METRIC_REGISTER)
    drivers = set(DRIVER_CANDIDATES) | {None}
    for stage, driver in PATTERNS:
        assert stage in stages, f"unreachable stage in PATTERNS: {stage}"
        assert driver in drivers, f"unreachable driver in PATTERNS: {driver}"


def _driver(metric: str, correlation: float, strength: str) -> AssociatedDriver:
    return AssociatedDriver(
        metric=metric,
        recent=1.0,
        baseline=1.0,
        deviation_pct=0.0,
        correlation_with_target=correlation,
        temporal_alignment_days=0,
        evidence_strength=strength,
    )


def test_classification_driver_ignores_evidence_the_engine_calls_weak():
    """A weak correlation contributes exactly zero to compute_confidence. It
    must not be allowed to pick a PLAYBOOK either -- that would be the one
    place where evidence the engine has already declared worthless still
    decides what a human is asked to approve.
    """
    weak = (
        _driver("avg_actual_delivery_minutes", 0.164, "weak"),
        _driver("avg_promised_eta_minutes", 0.148, "weak"),
    )
    assert _classification_driver(weak) is None
    assert _classification_driver(()) is None

    # The strongest NON-weak driver wins, not merely the first in the list.
    mixed = (
        _driver("on_time_rate", 0.90, "weak"),          # thin n, however large
        _driver("avg_actual_delivery_minutes", 0.50, "moderate"),
    )
    assert _classification_driver(mixed) == "avg_actual_delivery_minutes"


# --- zone 4: the supply route, registered in Task 18 -------------------------


def test_availability_rate_is_registered_so_downstream_can_see_it():
    """_melt() only picks up registered columns, so an unregistered metric is
    invisible to the detector, the diagnosis and the playbook however clean the
    signal in gold is. availability_rate sat unseen in gold_zone_performance
    until this entry existed."""
    assert "availability_rate" in METRIC_REGISTER
    frame = compute_metrics(load_gold(), P)
    series = frame.series("availability_rate", "zone", "4")
    assert len(series) == 180
    assert series.between(0, 1).all()
    # Zone grain only: gold_daily_business_metrics carries no such column, and
    # _melt skips what a table does not have rather than fabricating zeros.
    with pytest.raises(KeyError):
        frame.series("availability_rate", "company", "all")


def test_zone4_is_diagnosed_through_availability_not_through_gmv():
    d = zone4_availability_diagnosis()
    assert d.anomaly.metric == "availability_rate"
    assert d.anomaly.direction == "drop"
    assert abs(d.anomaly.z_score) > 5.0
    # Not a GMV anomaly, so no funnel is fabricated: the broken "stage" is the
    # metric itself, which is what the PATTERNS row is keyed on.
    assert d.funnel == ()
    assert d.funnel_break_stage == "availability_rate"
    assert d.pattern == "supply_availability_gap"
    # A rate, so no segment decomposition -- zone availability rates do not sum
    # to a company availability rate.
    assert d.contributions == ()
    assert d.primary_segment is None


def test_zone4_supply_pattern_does_not_hinge_on_which_weak_driver_ranks_first():
    """The driver-ordering trap, in its worst form. All three fulfilment
    candidates measure |r| 0.13-0.16 against zone 4's availability -- a spread
    far too narrow to rank, and weak by the module's own threshold. The
    classification must not depend on which of them happens to win, and it does
    not: none of them can reach a pattern at all, and the signature the engine
    actually keys on is "nothing upstream moved with it".
    """
    d = zone4_availability_diagnosis()
    assert d.drivers, "the candidates were measured, not skipped"
    assert all(dr.evidence_strength == "weak" for dr in d.drivers)

    magnitudes = sorted(abs(dr.correlation_with_target) for dr in d.drivers)
    assert magnitudes[-1] - magnitudes[0] < 0.05, "spread wide enough to rank"

    assert _classification_driver(d.drivers) is None
    for candidate in DRIVER_CANDIDATES:
        assert classify_pattern("availability_rate", candidate) == "unknown_pattern"


# --- the whole ground-truth firewall -----------------------------------------


def test_every_injected_incident_on_a_carried_dimension_is_rediscovered():
    """THE recovery test, and the payoff of the ground-truth firewall: src/
    never reads data/ground_truth/, so anything the engine lands on it reached
    from the evidence alone. This test reads it, once, to check.

    The two incidents whose dimension the MetricFrame does not carry are
    asserted to be out of reach for a STATED, structural reason rather than
    quietly dropped -- so that wiring either dimension in without wiring in its
    detection fails here instead of passing silently.
    """
    gt = json.loads((GROUND_TRUTH / "injected_incidents.json").read_text())
    assert len(gt["incidents"]) == 4

    gold = load_gold()
    frame = compute_metrics(gold, P)
    anomalies = detect_anomalies(frame, P)
    detected_scopes = {(a.scope, a.scope_value) for a in anomalies}
    carried_dimensions = set(frame.long["scope"].unique())

    in_reach, out_of_reach = {}, {}
    for incident in gt["incidents"]:
        dimension = incident["scope"]["dimension"]
        target = in_reach if dimension in carried_dimensions else out_of_reach
        target[incident["id"]] = (dimension, str(incident["scope"]["value"]))

    assert set(in_reach) == {"zone7_degradation", "zone4_availability"}
    for incident_id, scope in in_reach.items():
        assert scope in detected_scopes, f"not rediscovered: {incident_id}"

    # Both in-reach incidents are also CLASSIFIED, not merely detected.
    patterns = {diagnose(a, gold, P).pattern for a in anomalies}
    assert {"fulfillment_eta_degradation", "supply_availability_gap"} <= patterns

    # The other two, and why the detector cannot see them. These are
    # structural facts about the gold layer, not calibration:
    assert set(out_of_reach) == {"promo_margin_erosion", "paid_social_retention"}

    # A campaign's whole life is shorter than one baseline + comparison window,
    # so a trailing-baseline detector has no uncontaminated window to compare
    # against -- the incident is inside its own baseline.
    promo = gold.promotion_performance
    freeship = promo.loc[promo["promo_code"] == "FREESHIP_WINTER", "metric_date"]
    span_days = (freeship.max() - freeship.min()).days + 1
    assert span_days < P.baseline_window_days + P.comparison_window_days

    # Retention has no daily grain at all: the table is (cohort_month,
    # acquisition_channel, period_index), so there is no series to scan.
    assert "metric_date" not in gold.customer_retention.columns


# --- confidence --------------------------------------------------------------


def test_confidence_is_bounded_and_documented():
    d = zone7_gmv_diagnosis()
    assert 0.0 <= d.confidence <= 1.0
    assert d.confidence > 0.5


def test_confidence_matches_the_documented_formula_term_by_term():
    d = zone7_gmv_diagnosis()
    expected = (
        0.25 * min(abs(d.anomaly.z_score) / 5.0, 1.0)
        + 0.25 * min(abs(d.primary_segment.contribution_pct) / 100.0, 1.0)
        + 0.30 * max(abs(dr.correlation_with_target) for dr in d.drivers)
        + 0.20
        * min(
            ((P.as_of - d.anomaly.first_detected_date).days + 1)
            / P.comparison_window_days,
            1.0,
        )
    )
    assert d.confidence == pytest.approx(expected)


def test_confidence_falls_when_evidence_is_removed():
    """Missing evidence must subtract. A neutral default would make a
    thinly-evidenced diagnosis look as trustworthy as a well-evidenced one."""
    d = zone7_gmv_diagnosis()
    full = compute_confidence(d.anomaly, d.primary_segment, d.drivers, P)
    assert full == pytest.approx(d.confidence)

    assert compute_confidence(d.anomaly, d.primary_segment, (), P) < full
    assert compute_confidence(d.anomaly, None, d.drivers, P) < full
    assert compute_confidence(d.anomaly, None, (), P) < full

    # A correlation measured on too few days is not evidence either, however
    # large the coefficient: a weak driver contributes exactly what none does.
    thin = dataclasses.replace(d.drivers[0], evidence_strength="weak")
    assert compute_confidence(d.anomaly, d.primary_segment, (thin,), P) == (
        pytest.approx(compute_confidence(d.anomaly, d.primary_segment, (), P))
    )


# --- end to end --------------------------------------------------------------


def test_every_detected_anomaly_diagnoses_without_special_casing():
    """No metric or scope in the register may crash the diagnostic path, and
    none may return a None pattern."""
    gold = load_gold()
    for anomaly in detect_anomalies(compute_metrics(gold, P), P):
        d = diagnose(anomaly, gold, P)
        assert d.pattern
        assert 0.0 <= d.confidence <= 1.0
        assert all(
            dr.evidence_strength in ("strong", "moderate", "weak") for dr in d.drivers
        )
        assert not any(dr.metric == anomaly.metric for dr in d.drivers)


def test_drivers_are_reported_for_zone_scope_metrics_only_where_carried():
    gold = load_gold()
    assert associated_drivers(gold, "company", "all", "gmv", P)
    assert associated_drivers(gold, "zone", "7", "gmv", P)


def test_a_scope_gold_does_not_carry_degrades_instead_of_raising():
    """Robustness, not tuning.

    Every downstream caller assumes the zone-7 GMV anomaly exists. It does --
    but it clears the sensitivity floor at z = -2.9 and is one calibration
    nudge from not clearing it, so the modules underneath must degrade rather
    than traceback when a scope has no series.

    Two failure modes are covered: a KeyError out of the funnel and the
    driver scan, and -- worse, because it is silent -- a funnel with nothing
    measurable in it naming `sessions` as the primary break and asserting that
    demand collapsed on the strength of no evidence at all.
    """
    gold = load_gold()

    funnel = funnel_decomposition(gold, "zone", "does-not-exist", P)
    assert len(funnel) == len(FUNNEL_STAGES)
    assert not any(f.is_primary_break for f in funnel)
    assert all(f.log_contribution == 0.0 for f in funnel)

    assert associated_drivers(gold, "zone", "does-not-exist", "gmv", P) == ()

    anomaly = dataclasses.replace(
        next(
            a
            for a in detect_anomalies(compute_metrics(gold, P), P)
            if a.metric == "gmv"
        ),
        scope="zone",
        scope_value="does-not-exist",
    )
    d = diagnose(anomaly, gold, P)
    assert d.funnel_break_stage == "gmv"        # the metric, not an invented stage
    assert d.drivers == ()
    assert d.pattern == "unknown_pattern"       # labelled, not guessed
