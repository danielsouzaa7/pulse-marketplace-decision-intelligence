# Impact estimation and explainable prioritisation.
#
#   estimate_impact(diagnosis, gold, params) -> Impact
#   prioritize(pairs, params)                -> list[Priority]
#
# Two rules do the analytical work here, and both exist to stop the same bug:
# claiming the same money more than once.
#
#   1. GROUP BY SCOPE. Seven zone-7 metrics fire on this dataset (gmv,
#      orders_completed, completion_rate, cancellation_rate, on_time_rate and
#      both duration metrics). They are seven symptoms of ONE incident, not
#      seven incidents. Ranking them independently would produce seven
#      near-identical entries each claiming the same R$8.7k. So anomalies are
#      grouped by (scope, scope_value), one Priority per group, impact computed
#      once per group.
#
#   2. NET AN AGGREGATE AGAINST ITS SEGMENTS. Company GMV is the sum of the
#      zone GMVs, so a company-scope group and a zone-scope group in the same
#      run overlap by construction -- on this dataset zone 7 IS 56.15% of the
#      company GMV deviation. Letting both claim their gross deviation is rule
#      1's bug one level up: the company priority would claim R$15.6k of window
#      deviation of which R$8.8k is zone 7's, already claimed. The company group
#      therefore claims only the residual: its own deviation less what the
#      segment-scope groups in the same run already claim. This is structural
#      (an aggregate contains its segments) and never branches on a scope VALUE.
#
# A third rule keeps presentation from moving the ranking: the member chosen to
# REPRESENT a group is chosen for what it explains, while the confidence that
# enters the SCORE is the group's best-evidenced member. See representative()
# and prioritize() for why those must not be the same choice.
#
# EVERYTHING HERE IS AN ASSOCIATED DEVIATION, NEVER A COST. The numbers are
# "the GMV deviation observed in this scope over the comparison window", not
# "what the incident cost". Nothing in this module establishes that the
# diagnosed pattern produced the deviation.
#
# And every figure is an UNDERSTATEMENT, deliberately. The 56-day baseline ends
# 2026-08-27 while the zone 7 degradation starts 2026-08-11, so 17 baseline
# days are already degraded and the baseline itself is depressed. That is the
# honest behaviour of a trailing-baseline detector (see
# root_cause._window_means) and is NOT corrected for: the estimate is a floor,
# so "at least" is the only defensible wording for it.
from __future__ import annotations

from collections import Counter
from functools import lru_cache

import pandas as pd

from pulse.io import read_silver
from pulse.root_cause import _frame, _window_means
from pulse.types import AnalysisParams, Diagnosis, Impact, Priority

# The documented score. Named constants summing to 1.0 (asserted below and in
# the tests) so the weighting is a data change, not an edit to arithmetic.
W_GMV = 0.45
W_ORDERS = 0.20
W_CUSTOMERS = 0.15
W_CONFIDENCE = 0.20
WEIGHTS: dict[str, float] = {
    "w_gmv": W_GMV,
    "w_orders": W_ORDERS,
    "w_customers": W_CUSTOMERS,
    "w_confidence": W_CONFIDENCE,
}
if abs(sum(WEIGHTS.values()) - 1.0) >= 1e-12:
    raise ValueError(f"score weights must sum to 1.0, got {sum(WEIGHTS.values())}")

# Run-rate projection horizon, in days. A month of exposure if nothing changes
# -- long enough to be a decision input, short enough that projecting a 14-day
# observation across it is not fantasy.
PROJECTION_DAYS = 30

# The scope that aggregates every other scope. Rule 2 above.
COMPANY_SCOPE = "company"

# scope -> the silver.orders column that carries it. Used only for the DISTINCT
# customer count, which cannot come from gold: gold's active_customers column is
# daily-distinct, so summing it over a window yields customer-DAYS, roughly 6x
# the real figure at 14 days. Task 17's extra dimensions are a row here.
_SCOPE_COLUMN: dict[str, str | None] = {
    COMPANY_SCOPE: None,
    "zone": "zone_id",
}

# The three series every impact is denominated in, whatever metric tripped the
# alarm. A zone's cancellation-rate spike and a zone's GMV drop are the same
# business problem measured twice; the money involved is the same money.
_GMV = "gmv"
_ORDERS = "orders_completed"
_MARGIN = "contribution_margin"

_UNKNOWN_PATTERN = "unknown_pattern"


class UnmappedScopeError(KeyError):
    """Raised when a scope has no silver.orders column in _SCOPE_COLUMN.

    Explicit rather than a silent 0: a scope whose customer footprint cannot be
    counted must not be given a footprint of zero, which would quietly rank it
    below every scope that can be counted. Adding a dimension in Task 17 means
    adding a row to _SCOPE_COLUMN.
    """


@lru_cache(maxsize=1)
def _orders() -> pd.DataFrame:
    """silver.orders, narrowed to the columns the customer count needs.

    Cached: estimate_impact() runs once per anomaly (ten times on this dataset)
    and this is a 6 MB parquet read.
    """
    columns = ["customer_id", "order_date"] + [
        c for c in _SCOPE_COLUMN.values() if c is not None
    ]
    return read_silver("orders")[columns]


@lru_cache(maxsize=64)
def distinct_customers(
    scope: str, scope_value: str, start: pd.Timestamp, end: pd.Timestamp
) -> int:
    """Customers who placed at least one order in `scope` between `start` and
    `end` inclusive, counted DISTINCT over the whole window.

    Distinct-over-window, never a sum of daily counts. A customer who orders on
    eight of the fourteen days is one customer, not eight; summing gold's
    daily-distinct active_customers column would report customer-DAYS instead,
    roughly 6x too many at this window length. That is why this reads silver
    rather than gold, and it is the one function here that touches disk: a
    windowed distinct count is something the long-format MetricFrame cannot
    express, because days cannot be added up into customers.
    """
    if scope not in _SCOPE_COLUMN:
        raise UnmappedScopeError(
            f"scope {scope!r} has no silver.orders column in _SCOPE_COLUMN; "
            f"mapped scopes: {sorted(_SCOPE_COLUMN)}"
        )
    orders = _orders()
    mask = (orders["order_date"] >= start) & (orders["order_date"] <= end)
    column = _SCOPE_COLUMN[scope]
    if column is not None:
        mask &= orders[column].astype(str) == str(scope_value)
    return int(orders.loc[mask, "customer_id"].nunique())


def _deviation_per_day(
    metric: str, scope: str, scope_value: str, gold, p: AnalysisParams
) -> float:
    """Signed (recent mean - baseline mean) per day for one metric at one scope.

    Negative means the recent window sits below the baseline. Uses the
    diagnosis module's own window means so impact is measured over exactly the
    days the detector and the diagnosis looked at -- a third copy of the
    slicing arithmetic would drift away from the other two.
    """
    try:
        series = _frame(gold, p).series(metric, scope, scope_value)
    except KeyError:
        return 0.0  # metric not carried at this scope; not an error
    recent, baseline = _window_means(series, p)
    deviation = recent - baseline
    return float(deviation) if pd.notna(deviation) else 0.0


def _adverse(deviation_per_day: float, days: int) -> float:
    """The adverse part of a deviation, scaled across `days`, as a magnitude.

    Clamped at zero on purpose: a scope whose GMV ROSE has nothing at risk, and
    reporting the magnitude of a favourable move as "at risk" would rank good
    news as a problem. Clamping is also the conservative direction -- it can
    only shrink a claim, never inflate one.
    """
    return max(0.0, -deviation_per_day) * days


def estimate_impact(diagnosis: Diagnosis, gold, p: AnalysisParams) -> Impact:
    """Estimate the business deviation associated with one diagnosis.

    Denominated in the SCOPE's own GMV, orders and margin, never in the fired
    metric's units. A zone's cancellation-rate spike and that zone's GMV drop
    describe one problem; the money involved is the zone's money either way,
    and "13.2pp of cancellation rate" is not a number anyone can act on.

    Units -- asserted in the tests, because a percentage silently entering a
    BRL field is the classic version of this bug:
        gmv_at_risk_brl     BRL, magnitude over the comparison window
        margin_impact_brl   BRL, magnitude over the comparison window
        orders_lost         whole orders, magnitude over the comparison window
        customers_affected  distinct customers over the comparison window
        daily_run_rate_brl  BRL/day, SIGNED (negative = below baseline)
        projected_30d_brl   BRL, SIGNED, = daily_run_rate_brl * PROJECTION_DAYS

    The at-risk/lost/affected fields are magnitudes ("how much is exposed") and
    the run-rate fields are signed ("which way is it moving"). Everything here
    is an ASSOCIATED DEVIATION observed alongside the diagnosed pattern, and --
    because the baseline is itself contaminated by the incident -- a floor: the
    true effect is AT LEAST this large.
    """
    anomaly = diagnosis.anomaly
    scope, scope_value = anomaly.scope, anomaly.scope_value
    days = p.comparison_window_days

    gmv_per_day = _deviation_per_day(_GMV, scope, scope_value, gold, p)
    orders_per_day = _deviation_per_day(_ORDERS, scope, scope_value, gold, p)
    margin_per_day = _deviation_per_day(_MARGIN, scope, scope_value, gold, p)

    # The comparison window's actual calendar span, taken from the series the
    # deviation was measured on rather than recomputed, so the customer count
    # covers exactly the days the money was measured over.
    end = pd.Timestamp(p.as_of)
    recent = (
        _frame(gold, p)
        .series(_GMV, scope, scope_value)
        .loc[end - pd.Timedelta(days=days - 1) : end]
    )

    return Impact(
        gmv_at_risk_brl=_adverse(gmv_per_day, days),
        orders_lost=int(round(_adverse(orders_per_day, days))),
        customers_affected=distinct_customers(
            scope, scope_value, recent.index.min(), recent.index.max()
        ),
        margin_impact_brl=_adverse(margin_per_day, days),
        daily_run_rate_brl=gmv_per_day,
        projected_30d_brl=gmv_per_day * PROJECTION_DAYS,
    )


Pair = tuple[Diagnosis, Impact]


def group_by_scope(pairs: list[Pair]) -> dict[tuple[str, str], list[Pair]]:
    """(scope, scope_value) -> the pairs that fired there, insertion-ordered.

    Public because Task 14's memo needs the members, not just the count: "seven
    independent metrics deteriorated in this zone" is evidence, and the metric
    names live on the grouped anomalies.
    """
    groups: dict[tuple[str, str], list[Pair]] = {}
    for diagnosis, impact in pairs:
        key = (diagnosis.anomaly.scope, diagnosis.anomaly.scope_value)
        groups.setdefault(key, []).append((diagnosis, impact))
    return groups


def _explanatory_payload(diagnosis: Diagnosis) -> bool:
    """Does this diagnosis carry the two pieces a memo cannot reconstruct?

    A segment decomposition (WHERE the deviation sits) and a funnel split
    (WHICH stage moved). In practice that means the scope's GMV anomaly: the
    funnel identity GMV = sessions x conversion x completion x aov is about
    GMV, and segment shares are only defined for additive metrics. A rate or
    duration anomaly gets neither, by design, not by omission.
    """
    return diagnosis.primary_segment is not None and bool(diagnosis.funnel)


def representative(group: list[Pair]) -> Pair:
    """The pair that speaks for its group, in this order:

        1. matched a playbook pattern (an "unknown_pattern" member carries no
           signature for Task 13 to act on)
        2. carries the explanatory payload -- a segment decomposition AND a
           funnel split
        3. strongest evidence (largest |z|)
        4. metric name, so the choice is identical between runs

    Rung 2 is a general principle, not a tune to this dataset: for ANY scope
    and ANY incident, the GMV anomaly is the member that carries the segment
    decomposition and the funnel split, because the funnel identity is about
    GMV and segment shares are only defined for additive metrics. Preferring
    the member that explains most is correct everywhere. Without it the choice
    falls to |z|, and a rate metric -- which by design gets no decomposition --
    wins, leaving the most important priority in the product rendering with its
    two most explanatory fields empty.

    The impact is the same for every member of a group by construction (it is
    denominated in the scope, not in the fired metric), so choosing a
    representative IS computing impact once per group. Confidence is NOT taken
    from the representative; see prioritize().
    """
    return min(
        group,
        key=lambda pair: (
            pair[0].pattern == _UNKNOWN_PATTERN,
            not _explanatory_payload(pair[0]),
            -abs(pair[0].anomaly.z_score),
            pair[0].anomaly.metric,
        ),
    )


def group_confidence(group: list[Pair]) -> float:
    """The strongest evidence anywhere in the group.

    Deliberately NOT the representative's own confidence. The representative is
    chosen for what it EXPLAINS (rung 2 above); confidence measures how much
    evidence the engine FOUND for the incident, and it found all of it, across
    every member. Coupling the two would let a presentation choice move the
    ranking: on this dataset the zone-7 GMV anomaly explains most (56.15%
    concentration, funnel break on completion_rate) but scores 0.6238, while
    its completion_rate sibling explains nothing structural and scores 0.7298
    on a maxed |z| term and a 0.93 driver correlation. Picking the better
    explanation must not be punished as if evidence had disappeared.

    Still monotone in evidence: every member's confidence comes from
    root_cause.compute_confidence, where missing evidence contributes zero, so
    a group of thinly-evidenced diagnoses still scores low. A group can only be
    as confident as its best-evidenced member, never more.
    """
    return max(diagnosis.confidence for diagnosis, _ in group)


def _net_of_nested(impact: Impact, nested: list[Impact], days: int) -> Impact:
    """Subtract the claims of nested (segment-scope) groups from an aggregate.

    Rule 2 in the module header. The GMV run rate is netted SIGNED, so a
    segment that moved favourably correctly adds back into the residual rather
    than being subtracted from it. Orders, margin and customers are netted on
    their clamped magnitudes -- Impact does not carry their signed dailies --
    which can only under-claim the residual, the conservative direction. The
    customer subtraction is exact while segments partition customers (they do
    here: 1,016 + 767 + 3,916 = 5,699) and conservative when they do not, since
    a customer counted in two segments is removed twice.

    Everything is floored at zero: a residual cannot be negative, and an
    aggregate whose segments already account for more than all of it has
    nothing of its own left to claim.

    WHAT LEAVES HERE IS NOT A MEASUREMENT. Every field on the returned Impact is
    a residual claim, and customers_affected is the one a reader will silently
    misread as a population: 3,896 on this dataset against 5,632 customers who
    actually ordered company-wide. The returned Impact carries
    customers_are_residual=True so no label can present it as the latter.
    """
    if not nested:
        return impact

    run_rate = impact.daily_run_rate_brl - sum(n.daily_run_rate_brl for n in nested)
    return Impact(
        gmv_at_risk_brl=_adverse(run_rate, days),
        orders_lost=max(0, impact.orders_lost - sum(n.orders_lost for n in nested)),
        customers_affected=max(
            0, impact.customers_affected - sum(n.customers_affected for n in nested)
        ),
        margin_impact_brl=max(
            0.0, impact.margin_impact_brl - sum(n.margin_impact_brl for n in nested)
        ),
        daily_run_rate_brl=run_rate,
        projected_30d_brl=run_rate * PROJECTION_DAYS,
        # The customer figure that leaves this function is a residual, not a
        # count, and every rendering surface has to be able to tell. See
        # Impact.customers_are_residual.
        customers_are_residual=True,
    )


def _norm(value: float, highest: float) -> float:
    """Min-max normalisation across the run's candidate set, with the floor
    pinned at zero rather than at the smallest candidate.

    The floor is zero because these are non-negative at-risk magnitudes with a
    real absolute zero ("nothing at risk"), not arbitrary scores. Taking the
    smallest candidate as the floor would force whichever candidate happens to
    be last to score 0 on EVERY impact term no matter how material it is, and
    would make one candidate's score depend on how small an unrelated candidate
    happens to be. Anchored at zero, the component reads as "share of the
    largest at-risk magnitude in this run", which is what the memo claims it is.

    Degenerate cases, both of which would otherwise divide by zero:
      * every candidate identical and non-zero -> every candidate gets 1.0. The
        component stops discriminating, which is correct: it carries no
        information about their relative severity.
      * every candidate zero -> every candidate gets 0.0. Nothing is at risk,
        so nothing earns impact weight.
    """
    return value / highest if highest > 0 else 0.0


def _at_risk_30d(impact: Impact) -> float:
    """The adverse part of the 30-day projection, as a magnitude. A scope whose
    GMV rose projects no GMV at risk.
    """
    return max(0.0, -impact.projected_30d_brl)


def prioritize(pairs: list[Pair], p: AnalysisParams) -> list[Priority]:
    """Rank scope groups by the documented, hand-recomputable impact score:

        impact_score = 100 * ( 0.45 * norm(projected_30d_gmv_at_risk)
                             + 0.20 * norm(orders_lost)
                             + 0.15 * norm(customers_affected)
                             + 0.20 * confidence )

    One Priority per (scope, scope_value) group, not per anomaly. The returned
    score_breakdown carries both the weights and the normalised components, so
    the score can be reconstructed by hand from the returned object alone --
    that is the difference between an explainable ranking and a number. It also
    carries supporting_anomalies (how many independent metrics fired in that
    scope), nested_groups_netted (how many segment claims were removed),
    representative_confidence (the displayed diagnosis's own, which may be
    lower than the group's -- see group_confidence()) and score_gap_to_next
    (points between this candidate and the next, so a rendering surface can
    state the margin as a number instead of characterising it in a word).

    Confidence enters raw (it is already on [0, 1]) and is the GROUP's, not the
    representative's, so a thinly-evidenced group -- no segment decomposition,
    no drivers above the reporting threshold -- scores strictly lower than a
    well-evidenced one instead of defaulting to anything neutral, while the
    choice of which member to display cannot move the ranking.

    Ranks are dense 1..n and the sort is explicitly tie-broken on
    (scope, scope_value), so equal scores cannot reorder between runs.
    """
    if not pairs:
        return []

    groups = group_by_scope(pairs)
    chosen = {key: representative(group) for key, group in groups.items()}

    # Rule 2: net the company aggregate against the segment groups in this run.
    # Picking ONE segment dimension matters once Task 17 adds a second: two
    # dimensions each partition the company independently, and subtracting both
    # would remove the same money twice. The dimension with the most groups
    # wins, ties broken by name.
    # ponytail: one dimension is netted; a run needing two overlapping
    # dimensions netted jointly needs a real attribution model, not subtraction.
    dimensions = Counter(scope for scope, _ in groups if scope != COMPANY_SCOPE)
    nested_dimension = (
        min(dimensions, key=lambda s: (-dimensions[s], s)) if dimensions else None
    )
    nested = [
        impact for (scope, _), (_, impact) in chosen.items() if scope == nested_dimension
    ]

    candidates = [
        (
            key,
            chosen[key][0],
            _net_of_nested(chosen[key][1], nested, p.comparison_window_days)
            if key[0] == COMPANY_SCOPE
            else chosen[key][1],
            group,
        )
        for key, group in groups.items()
    ]

    highest = {
        "gmv": max(_at_risk_30d(i) for _, _, i, _ in candidates),
        "orders": max(float(i.orders_lost) for _, _, i, _ in candidates),
        "customers": max(float(i.customers_affected) for _, _, i, _ in candidates),
    }

    scored = []
    for key, diagnosis, impact, group in candidates:
        components = {
            "n_gmv": _norm(_at_risk_30d(impact), highest["gmv"]),
            "n_orders": _norm(float(impact.orders_lost), highest["orders"]),
            "n_customers": _norm(float(impact.customers_affected), highest["customers"]),
            "n_confidence": group_confidence(group),
        }
        score = 100.0 * sum(
            WEIGHTS[f"w_{name}"] * components[f"n_{name}"]
            for name in ("gmv", "orders", "customers", "confidence")
        )
        breakdown = {
            **WEIGHTS,
            **components,
            # Evidence and provenance, not arithmetic.
            "supporting_anomalies": float(len(group)),
            "nested_groups_netted": float(len(nested) if key[0] == COMPANY_SCOPE else 0),
            "representative_confidence": diagnosis.confidence,
        }
        scored.append((score, key, diagnosis, impact, breakdown))

    scored.sort(key=lambda row: (-row[0], row[1]))

    # The distance to the next-ranked candidate, computed HERE because it is a
    # figure about the ranking and app/** may not derive one. A rendering surface
    # that subtracted two displayed scores itself would be computing a business
    # figure in the presentation layer -- and the claim it feeds ("how far ahead
    # is first place?") has been overstated as a qualitative word before, which
    # is exactly the kind of claim that has to come off a number.
    #
    # 0.0 on the last candidate: there is no next one, and a gap of nothing is
    # not a narrow gap. Callers check rank or len() before describing it.
    for index, (score, _key, _d, _i, breakdown) in enumerate(scored):
        nxt = scored[index + 1][0] if index + 1 < len(scored) else score
        breakdown["score_gap_to_next"] = score - nxt

    return [
        Priority(
            diagnosis=diagnosis,
            impact=impact,
            impact_score=score,
            rank=rank,
            score_breakdown=breakdown,
        )
        for rank, (score, _key, diagnosis, impact, breakdown) in enumerate(
            scored, start=1
        )
    ]
