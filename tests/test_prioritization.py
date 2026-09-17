# tests/test_prioritization.py
import dataclasses
from datetime import date, timedelta
from functools import lru_cache

import pandas as pd
import pytest

from pulse.anomaly_detection import detect_anomalies
from pulse.io import read_silver
from pulse.metrics import compute_metrics, customers_label, load_gold
from pulse.prioritization import (
    COMPANY_SCOPE,
    PROJECTION_DAYS,
    WEIGHTS,
    UnmappedScopeError,
    _net_of_nested,
    _norm,
    distinct_customers,
    estimate_impact,
    group_by_scope,
    group_confidence,
    prioritize,
    representative,
)
from pulse.root_cause import compute_confidence, diagnose
from pulse.types import AnalysisParams, Anomaly, Diagnosis, Impact

P = AnalysisParams(as_of=date(2026, 9, 10))

_COMPONENTS = ("gmv", "orders", "customers", "confidence")


@lru_cache(maxsize=1)
def build_pairs():
    gold = load_gold()
    anomalies = detect_anomalies(compute_metrics(gold, P), P)
    return [(d := diagnose(a, gold, P), estimate_impact(d, gold, P)) for a in anomalies]


def recompute(breakdown: dict) -> float:
    """The score, rebuilt by hand from nothing but the returned breakdown."""
    return 100.0 * sum(
        breakdown[f"w_{name}"] * breakdown[f"n_{name}"] for name in _COMPONENTS
    )


# --------------------------------------------------------------------------
# The brief's tests, unmodified
# --------------------------------------------------------------------------


def test_impact_is_positive_and_projects_thirty_days():
    _, impact = build_pairs()[0]
    assert impact.gmv_at_risk_brl > 0
    assert impact.orders_lost > 0
    assert abs(impact.projected_30d_brl - impact.daily_run_rate_brl * 30) < 1.0


def test_ranking_is_dense_and_ordered():
    prios = prioritize(build_pairs(), P)
    assert [p.rank for p in prios] == list(range(1, len(prios) + 1))
    scores = [p.impact_score for p in prios]
    assert scores == sorted(scores, reverse=True)


def test_score_breakdown_weights_sum_to_one_and_reproduce_the_score():
    p0 = prioritize(build_pairs(), P)[0]
    w = p0.score_breakdown
    assert abs(sum(w[f"w_{name}"] for name in _COMPONENTS) - 1.0) < 1e-9
    assert abs(recompute(w) - p0.impact_score) < 0.01


def test_zone_7_ranks_first():
    assert prioritize(build_pairs(), P)[0].diagnosis.primary_segment.segment == "7"


# --------------------------------------------------------------------------
# Representative selection: the member that EXPLAINS most speaks for the group
# --------------------------------------------------------------------------


def test_representative_of_a_group_with_a_gmv_anomaly_is_that_gmv_anomaly():
    """The GMV anomaly is the only member carrying a segment decomposition and a
    funnel split -- the funnel identity is about GMV and segment shares are only
    defined for additive metrics. Whenever one is present it must represent its
    group, or the memo renders with its two most explanatory fields empty.
    """
    groups = group_by_scope(build_pairs())
    with_gmv = {
        key: group
        for key, group in groups.items()
        if any(d.anomaly.metric == "gmv" for d, _ in group)
    }
    assert with_gmv, "fixture changed: no group contains a GMV anomaly"
    for key, group in with_gmv.items():
        assert representative(group)[0].anomaly.metric == "gmv", key


def test_priority_one_exposes_the_concentration_and_the_funnel_break():
    """Task 14's memo has a CONCENTRATION field and a funnel-break field. They
    must come off the Priority, not a side call.
    """
    top = prioritize(build_pairs(), P)[0]
    assert (top.diagnosis.anomaly.scope, top.diagnosis.anomaly.scope_value) == (
        "zone",
        "7",
    )
    assert top.diagnosis.primary_segment is not None
    assert top.diagnosis.primary_segment.segment == "7"
    # Threshold, not a literal. The share is a ratio of two noisy window means
    # (standard error ~180 BRL/day against a ~-700 BRL/day deviation), so its
    # point value moves several percentage points on any change to the rng
    # stream -- including one that injects no incident at all. ">= 55% at rank
    # 1" is the calibration target spec section 5 actually states; do not
    # re-tighten this to whatever a particular run happens to print.
    assert top.diagnosis.primary_segment.rank == 1
    assert top.diagnosis.primary_segment.contribution_pct >= 55.0
    assert top.diagnosis.funnel_break_stage == "completion_rate"
    assert len(top.diagnosis.funnel) == 4
    assert top.diagnosis.pattern == "fulfillment_eta_degradation"


def test_representative_prefers_a_real_pattern_before_anything_else():
    """Explanatory payload never outranks having a playbook signature: an
    unknown_pattern member gives Task 13 nothing to act on.
    """
    group = group_by_scope(build_pairs())[("zone", "7")]
    chosen = representative(group)[0]
    assert chosen.pattern != "unknown_pattern"

    # And among the patterned members it is the one with the payload, even
    # though a sibling has nearly twice its |z| (5.62 vs 2.87).
    patterned = [d for d, _ in group if d.pattern != "unknown_pattern"]
    assert max(abs(d.anomaly.z_score) for d in patterned) > abs(
        chosen.anomaly.z_score
    ), "fixture changed: payload preference is no longer load-bearing here"
    assert chosen.primary_segment is not None and chosen.funnel


def test_group_by_scope_still_exposes_the_members_the_representative_is_not():
    """The other six zone-7 anomalies remain evidence for the memo even though
    they do not represent the group.
    """
    members = group_by_scope(build_pairs())[("zone", "7")]
    metrics = {d.anomaly.metric for d, _ in members}
    assert len(metrics) == 7
    assert {"completion_rate", "cancellation_rate", "on_time_rate"} <= metrics


# --------------------------------------------------------------------------
# Confidence is the group's evidence, not the display choice
# --------------------------------------------------------------------------


def test_group_confidence_is_the_best_evidenced_member_not_the_representative():
    """Choosing the better EXPLANATION must not be scored as if evidence had
    disappeared. The zone-7 GMV anomaly explains most (56.15% concentration,
    funnel break) but scores 0.6238; siblings measure higher on |z| and driver
    correlation. The group is as confident as its best-evidenced member.
    """
    group = group_by_scope(build_pairs())[("zone", "7")]
    rep = representative(group)[0]
    best = max(d.confidence for d, _ in group)

    assert group_confidence(group) == best
    assert best > rep.confidence, "fixture changed: decoupling no longer matters"

    top = prioritize(build_pairs(), P)[0]
    assert top.score_breakdown["n_confidence"] == best
    assert top.score_breakdown["representative_confidence"] == rep.confidence
    assert top.diagnosis.confidence == rep.confidence


def test_group_confidence_never_rises_when_a_thin_member_is_added():
    """Monotone in evidence: a badly-evidenced extra anomaly cannot lift a group,
    and a group can never be more confident than its best member.
    """
    group = group_by_scope(build_pairs())[("zone", "7")]
    before = group_confidence(group)
    thin = (synthetic("7", (), None, z_score=-0.1), group[0][1])

    after = group_confidence(group + [thin])
    assert after == before
    assert after >= thin[0].confidence


# --------------------------------------------------------------------------
# Scope dedup: one priority per scope, impact claimed once
# --------------------------------------------------------------------------


def test_one_priority_per_scope_not_one_per_anomaly():
    pairs = build_pairs()
    prios = prioritize(pairs, P)
    scopes = [
        (p.diagnosis.anomaly.scope, p.diagnosis.anomaly.scope_value) for p in prios
    ]

    assert len(pairs) > len(prios), "dedup did nothing: one priority per anomaly"
    assert len(scopes) == len(set(scopes)), "same scope claimed by two priorities"

    # WHICH scopes fire is a property of the data, not of this test. The
    # ("zone", "4") entry this used to hardcode came from a zone-4
    # cancellation_rate anomaly in a run where zone 4 carried no incident at
    # all -- the set was pinning noise. The invariant is that the priorities
    # partition exactly the scopes that fired and lose no anomaly on the way.
    assert set(scopes) == set(group_by_scope(pairs))
    assert sum(p.score_breakdown["supporting_anomalies"] for p in prios) == len(pairs)


def test_zone_seven_group_records_seven_supporting_anomalies():
    prios = prioritize(build_pairs(), P)
    zone7 = next(p for p in prios if p.diagnosis.anomaly.scope_value == "7")
    assert zone7.score_breakdown["supporting_anomalies"] == 7.0


def test_aggregate_does_not_claim_money_its_segments_already_claimed():
    """Company GMV is the sum of the zone GMVs. If the company priority reported
    its gross deviation, the ranked list would claim zone 7's R$8.7k twice.
    """
    prios = prioritize(build_pairs(), P)
    company = next(p for p in prios if p.diagnosis.anomaly.scope == COMPANY_SCOPE)
    zones = [p for p in prios if p.diagnosis.anomaly.scope == "zone"]

    gross = estimate_impact(company.diagnosis, load_gold(), P)
    assert company.impact.gmv_at_risk_brl < gross.gmv_at_risk_brl
    assert company.impact.orders_lost < gross.orders_lost
    assert company.impact.margin_impact_brl < gross.margin_impact_brl
    # The customer footprint is netted too, not left gross: 5,699 - 1,016 - 767.
    assert company.impact.customers_affected < gross.customers_affected
    assert company.impact.customers_affected == gross.customers_affected - sum(
        z.impact.customers_affected for z in zones
    )
    assert company.score_breakdown["nested_groups_netted"] == float(len(zones))

    # The identity holds on the SIGNED run rates -- company GMV is the sum of
    # the zone GMVs, so the residual is exactly what the zones do not claim.
    # (It does not hold on the clamped at-risk magnitudes, and must not be
    # asserted there: zone 4's GMV rose, so it claims 0 while still adding its
    # favourable move back into the residual.)
    reconstructed = company.impact.daily_run_rate_brl + sum(
        z.impact.daily_run_rate_brl for z in zones
    )
    assert abs(reconstructed - gross.daily_run_rate_brl) < 1e-6

    # And the netted impact still satisfies the 30-day projection identity.
    assert (
        abs(company.impact.projected_30d_brl - company.impact.daily_run_rate_brl * 30)
        < 1.0
    )


# --------------------------------------------------------------------------
# Impact: the numbers, and their units
# --------------------------------------------------------------------------


def zone7_impact() -> Impact:
    return group_by_scope(build_pairs())[("zone", "7")][0][1]


def window_means_from_gold(metric: str, scope: str, scope_value: str) -> tuple[float, float]:
    """(recent mean, baseline mean) sliced straight out of gold by calendar date.

    Deliberately NOT pulse.root_cause._window_means: the point is to check that
    Impact reports the deviation the gold tables carry, and reusing the module's
    own slicing arithmetic would make that assertion circular.
    """
    series = compute_metrics(load_gold(), P).series(metric, scope, scope_value)
    recent_end = pd.Timestamp(P.as_of)
    recent_start = recent_end - timedelta(days=P.comparison_window_days - 1)
    baseline_end = recent_start - timedelta(days=1)
    baseline_start = baseline_end - timedelta(days=P.baseline_window_days - 1)
    return (float(series.loc[recent_start:recent_end].mean()),
            float(series.loc[baseline_start:baseline_end].mean()))


def test_zone_seven_impact_reconciles_with_the_measured_deviation():
    """Impact must report the deviation gold carries, in the right units and
    with the right signs.

    This used to assert six literals (-625.31 +/- 0.5, 8754.28 +/- 5.0, ...).
    They were one realization's point values on a statistic whose standard
    error is roughly 180 BRL/day, and they broke on any change to the rng
    stream -- including one that injects no incident whatsoever. Recomputing
    both sides makes the test assert the reconciliation it was named for, and
    survive the next generator change. Do not re-pin this to literals.
    """
    impact = zone7_impact()
    days = P.comparison_window_days

    gmv_recent, gmv_baseline = window_means_from_gold("gmv", "zone", "7")
    gmv_per_day = gmv_recent - gmv_baseline
    assert gmv_per_day < 0, "fixture changed: zone 7 GMV is no longer below baseline"

    # Signed run-rate fields.
    assert impact.daily_run_rate_brl == pytest.approx(gmv_per_day)
    assert impact.projected_30d_brl == pytest.approx(gmv_per_day * PROJECTION_DAYS)
    # Magnitude fields, scaled across the comparison window.
    assert impact.gmv_at_risk_brl == pytest.approx(-gmv_per_day * days)

    orders_recent, orders_baseline = window_means_from_gold(
        "orders_completed", "zone", "7")
    assert impact.orders_lost == int(round(-(orders_recent - orders_baseline) * days))

    margin_recent, margin_baseline = window_means_from_gold(
        "contribution_margin", "zone", "7")
    assert impact.margin_impact_brl == pytest.approx(
        -(margin_recent - margin_baseline) * days)

    assert impact.gmv_at_risk_brl > 0 and impact.margin_impact_brl > 0
    assert impact.orders_lost > 0
    assert 0 < impact.customers_affected <= 25_000


def test_impact_is_scope_denominated_not_metric_denominated():
    """Every zone-7 anomaly -- GMV, a rate, a duration -- reports the same money,
    because the money is the zone's, not the fired metric's.
    """
    members = group_by_scope(build_pairs())[("zone", "7")]
    assert len({impact for _, impact in members}) == 1

    # A duration metric fired in this group; its impact is still in BRL, not
    # minutes. 6.0 minutes of delivery deviation must not reach a BRL field.
    duration = next(
        impact
        for d, impact in members
        if d.anomaly.metric == "avg_actual_delivery_minutes"
    )
    assert duration.gmv_at_risk_brl > 1000.0


def test_units_are_brl_counts_and_ratios():
    for _, impact in build_pairs():
        assert isinstance(impact.orders_lost, int)
        assert isinstance(impact.customers_affected, int)
        assert isinstance(impact.gmv_at_risk_brl, float)
        assert isinstance(impact.margin_impact_brl, float)
        assert impact.gmv_at_risk_brl >= 0.0 and impact.margin_impact_brl >= 0.0
        assert impact.orders_lost >= 0 and impact.customers_affected >= 0
        # BRL magnitudes are the daily run rate scaled across the window -- a
        # percentage entering here would not satisfy this identity.
        assert (
            abs(
                impact.gmv_at_risk_brl
                - max(0.0, -impact.daily_run_rate_brl) * P.comparison_window_days
            )
            < 1e-6
        )
        assert (
            abs(impact.projected_30d_brl - impact.daily_run_rate_brl * PROJECTION_DAYS)
            < 1e-6
        )

    for priority in prioritize(build_pairs(), P):
        # Rates are ratios on [0, 1], never percentages.
        assert 0.0 <= priority.diagnosis.confidence <= 1.0
        for name in _COMPONENTS:
            assert 0.0 <= priority.score_breakdown[f"n_{name}"] <= 1.0
        assert 0.0 <= priority.impact_score <= 100.0


def test_customers_affected_is_distinct_over_the_window_not_customer_days():
    """Summing a daily-distinct count gives customer-DAYS. The honest figure is
    strictly smaller whenever anyone orders twice in the window -- here 1,016
    customers against 1,367 customer-days.
    """
    end = pd.Timestamp(P.as_of)
    start = end - timedelta(days=P.comparison_window_days - 1)
    orders = read_silver("orders")
    window = orders[
        (orders["order_date"] >= start)
        & (orders["order_date"] <= end)
        & (orders["zone_id"] == 7)
    ]
    customer_days = int(window.groupby("order_date")["customer_id"].nunique().sum())

    affected = zone7_impact().customers_affected
    assert affected == window["customer_id"].nunique()
    assert affected < customer_days
    assert customer_days > 1.25 * affected, "window too quiet to distinguish the two"


def test_unmapped_scope_raises_rather_than_counting_zero_customers():
    with pytest.raises(UnmappedScopeError):
        distinct_customers(
            "merchant_category",
            "pizza",
            pd.Timestamp("2026-08-28"),
            pd.Timestamp(P.as_of),
        )


def test_none_primary_segment_is_handled_without_inventing_one():
    """Rate and duration metrics get no segment decomposition by design, so most
    diagnoses carry primary_segment=None. Impact must not crash on that, and
    must not invent a segment to fill the hole.
    """
    pairs = build_pairs()
    unsegmented = [(d, i) for d, i in pairs if d.primary_segment is None]
    # Nine of twelve: three company rate/duration anomalies, five zone-7
    # rate/duration anomalies, and zone 4's availability_rate (registered in
    # Task 18). Only zone 7's gmv and orders_completed and company
    # contribution_margin are denominated in additive units, so only those
    # three carry a segment decomposition -- see root_cause._ADDITIVE_UNITS.
    segmented = [d for d, _ in pairs if d.primary_segment is not None]
    assert {d.anomaly.metric for d in segmented} == {
        "gmv",
        "orders_completed",
        "contribution_margin",
    }
    assert len(unsegmented) == len(pairs) - len(segmented) == 9
    for _, impact in unsegmented:
        assert impact.customers_affected > 0

    ranked = prioritize(unsegmented, P)
    assert ranked, "prioritize crashed on segment-less diagnoses"
    assert all(p.diagnosis.primary_segment is None for p in ranked)


# --------------------------------------------------------------------------
# Scoring: explainability, missing evidence, degenerate sets
# --------------------------------------------------------------------------


def test_every_breakdown_reconstructs_its_own_score():
    for priority in prioritize(build_pairs(), P):
        assert abs(recompute(priority.score_breakdown) - priority.impact_score) < 1e-9


def synthetic(scope_value: str, drivers, primary_segment, z_score=-4.0) -> Diagnosis:
    anomaly = Anomaly(
        metric="gmv",
        scope="zone",
        scope_value=scope_value,
        recent_value=900.0,
        baseline_value=1000.0,
        deviation_abs=-100.0,
        deviation_pct=-10.0,
        z_score=z_score,
        direction="drop",
        first_detected_date=P.as_of - timedelta(days=P.comparison_window_days - 1),
        n_observations=P.comparison_window_days,
    )
    return Diagnosis(
        anomaly=anomaly,
        contributions=(),
        primary_segment=primary_segment,
        funnel=(),
        funnel_break_stage="completion_rate",
        drivers=drivers,
        pattern="fulfillment_eta_degradation",
        confidence=compute_confidence(anomaly, primary_segment, drivers, P),
    )


IMPACT = Impact(
    gmv_at_risk_brl=1400.0,
    orders_lost=20,
    customers_affected=50,
    margin_impact_brl=400.0,
    daily_run_rate_brl=-100.0,
    projected_30d_brl=-3000.0,
)


def test_removing_drivers_lowers_the_score_it_never_raises_it():
    """Missing evidence must subtract. Two single-member groups with identical
    impact, so the only thing that can move the score is the evidence found.
    """
    evidenced = representative(group_by_scope(build_pairs())[("zone", "7")])[0]
    assert evidenced.drivers, "fixture changed: zone 7 diagnosis has no drivers"

    with_drivers = synthetic("A", evidenced.drivers, evidenced.primary_segment)
    without_drivers = synthetic("B", (), None)

    assert without_drivers.confidence < with_drivers.confidence

    ranked = prioritize([(with_drivers, IMPACT), (without_drivers, IMPACT)], P)
    assert ranked[0].diagnosis.anomaly.scope_value == "A"
    assert (
        ranked[1].score_breakdown["n_confidence"]
        < ranked[0].score_breakdown["n_confidence"]
    )
    # Impact terms are identical, so the whole gap is the evidence gap.
    gap = 100.0 * WEIGHTS["w_confidence"] * (
        with_drivers.confidence - without_drivers.confidence
    )
    assert abs((ranked[0].impact_score - ranked[1].impact_score) - gap) < 1e-9


def test_empty_candidate_set_returns_empty_without_dividing_by_zero():
    assert prioritize([], P) == []


def test_all_equal_candidates_do_not_divide_by_zero():
    a = synthetic("A", (), None)
    b = synthetic("B", (), None)
    ranked = prioritize([(a, IMPACT), (b, IMPACT)], P)

    assert [p.rank for p in ranked] == [1, 2]
    assert ranked[0].impact_score == ranked[1].impact_score
    for priority in ranked:
        # Degenerate case, documented in _norm: identical non-zero candidates
        # all normalise to 1.0 rather than 0/0.
        assert priority.score_breakdown["n_gmv"] == 1.0
        assert priority.score_breakdown["n_orders"] == 1.0
        assert priority.score_breakdown["n_customers"] == 1.0


def test_all_zero_candidates_earn_no_impact_weight():
    nothing = dataclasses.replace(
        IMPACT,
        gmv_at_risk_brl=0.0,
        orders_lost=0,
        customers_affected=0,
        daily_run_rate_brl=0.0,
        projected_30d_brl=0.0,
    )
    ranked = prioritize([(synthetic("A", (), None), nothing)], P)
    assert ranked[0].score_breakdown["n_gmv"] == 0.0
    assert _norm(0.0, 0.0) == 0.0


def test_ranking_is_deterministic_under_reordered_input():
    pairs = build_pairs()

    def signature(prios):
        return [
            (
                p.rank,
                p.diagnosis.anomaly.scope,
                p.diagnosis.anomaly.scope_value,
                p.diagnosis.anomaly.metric,
                round(p.impact_score, 12),
            )
            for p in prios
        ]

    assert signature(prioritize(pairs, P)) == signature(
        prioritize(list(reversed(pairs)), P)
    )


def test_ranking_never_branches_on_a_scope_value():
    """Zone 7 leads on measured impact, not identity.

    This used to be asserted through `n_orders == 1.0` -- that zone 7 topped
    every normalised component. It no longer tops the orders component (the
    company completion_rate group legitimately claims more orders lost once
    three further incidents are in the data), which says nothing at all about
    whether the ranking branches on an identifier. The property is asserted
    directly instead: relabel every scope value and the ranking must not move.
    """
    pairs = build_pairs()
    prios = prioritize(pairs, P)
    top = prios[0]
    assert top.score_breakdown["n_gmv"] == 1.0
    assert top.impact.gmv_at_risk_brl == max(p.impact.gmv_at_risk_brl for p in prios)
    assert recompute(top.score_breakdown) == max(
        recompute(p.score_breakdown) for p in prios
    )

    # Same measurements, different identifiers. Ranks, scores and the mapped
    # identity of every rank must come out unchanged; anything reading a zone
    # id would move here. (Scores are distinct in this run, so the documented
    # (scope, scope_value) tie-break cannot reorder them either way.)
    alias = {
        value: f"scope-{i}"
        for i, value in enumerate(sorted({d.anomaly.scope_value for d, _ in pairs}))
    }
    renamed = [
        (
            dataclasses.replace(
                d,
                anomaly=dataclasses.replace(
                    d.anomaly, scope_value=alias[d.anomaly.scope_value]
                ),
            ),
            impact,
        )
        for d, impact in pairs
    ]
    after = prioritize(renamed, P)

    assert [p.rank for p in after] == [p.rank for p in prios]
    assert [p.impact_score for p in after] == [p.impact_score for p in prios]
    assert [p.diagnosis.anomaly.scope_value for p in after] == [
        alias[p.diagnosis.anomaly.scope_value] for p in prios
    ]


# --------------------------------------------------------------------------
# I1: the netted customer figure is a residual, and says so
# --------------------------------------------------------------------------


def test_netting_marks_its_customer_figure_as_a_residual():
    """The two quantities that shared one field, with the flag that tells them
    apart.

    A segment scope's customers_affected is a measured distinct count. An
    aggregate's is that count less what the nested segment groups already claim,
    which exists so the ranking does not weight the same customers twice. It is
    smaller than the measured population and is not a count of anybody.
    """
    aggregate = Impact(
        gmv_at_risk_brl=15600.0, orders_lost=260, customers_affected=5632,
        margin_impact_brl=5000.0, daily_run_rate_brl=-1114.29,
        projected_30d_brl=-33428.7,
    )
    nested = [
        Impact(
            gmv_at_risk_brl=10350.87, orders_lost=139, customers_affected=1030,
            margin_impact_brl=3920.96, daily_run_rate_brl=-739.35,
            projected_30d_brl=-22180.44,
        ),
        Impact(
            gmv_at_risk_brl=3040.0, orders_lost=25, customers_affected=706,
            margin_impact_brl=900.0, daily_run_rate_brl=-217.26,
            projected_30d_brl=-6517.91,
        ),
    ]
    assert aggregate.customers_are_residual is False

    residual = _net_of_nested(aggregate, nested, days=14)
    assert residual.customers_are_residual is True
    assert residual.customers_affected == 5632 - 1030 - 706
    assert residual.customers_affected != aggregate.customers_affected

    # The measured population and the prioritisation quantity stay distinct, and
    # the label follows the flag rather than the scope name.
    assert "clientes que pediram" in customers_label(aggregate).lower()
    assert "clientes que pediram" not in customers_label(residual).lower()
    assert "residual" in customers_label(residual).lower()


def test_an_unnetted_impact_keeps_the_measured_wording():
    """The fix is a distinction, not a blanket hedge: a measured count must still
    be describable as one, or the honest case loses its wording too."""
    measured = Impact(
        gmv_at_risk_brl=1.0, orders_lost=1, customers_affected=1030,
        margin_impact_brl=1.0, daily_run_rate_brl=-1.0, projected_30d_brl=-30.0,
    )
    assert _net_of_nested(measured, [], days=14) is measured
    assert "clientes que pediram" in customers_label(measured).lower()


def test_the_residual_is_never_described_as_a_measured_count():
    """Against the live run, where the two figures genuinely differ."""
    priorities = prioritize(build_pairs(), P)
    company = next(
        (p for p in priorities if p.diagnosis.anomaly.scope == "company"), None
    )
    assert company is not None, "no aggregate priority in this run"

    measured = distinct_customers(
        "company", "all",
        pd.Timestamp(P.as_of) - pd.Timedelta(days=P.comparison_window_days - 1),
        pd.Timestamp(P.as_of),
    )
    assert company.impact.customers_affected < measured
    assert company.impact.customers_are_residual is True

    label = customers_label(company.impact).lower()
    for banned in ("clientes que pediram", "clientes do escopo",
                   "clientes afetados medidos"):
        assert banned not in label, label


# --------------------------------------------------------------------------
# I4: the gap between ranks is a number the engine returns
# --------------------------------------------------------------------------


def test_the_score_gap_to_the_next_rank_is_returned():
    """app/** may not derive a business figure, and a gap between two scores is
    one -- so prioritize() returns it rather than leaving a page to subtract.
    """
    priorities = prioritize(build_pairs(), P)
    assert len(priorities) >= 2
    for first, second in zip(priorities, priorities[1:]):
        assert first.score_breakdown["score_gap_to_next"] == pytest.approx(
            first.impact_score - second.impact_score, rel=1e-12
        )
        assert first.score_breakdown["score_gap_to_next"] >= 0.0
    # The last candidate has no next one, and a gap of nothing is not a small gap.
    assert priorities[-1].score_breakdown["score_gap_to_next"] == 0.0


def test_the_score_gap_is_zero_for_a_single_priority():
    priorities = prioritize([build_pairs()[0]], P)
    assert len(priorities) == 1
    assert priorities[0].score_breakdown["score_gap_to_next"] == 0.0
