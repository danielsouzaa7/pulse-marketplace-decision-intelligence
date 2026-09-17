# Tests for the deterministic recommendation playbook (Task 13, extended in
# Task 18): diagnosis -> recommendation, no LLM involved in the policy
# decision.
#
# Beyond "does it render", these tests pin the five things that would make
# this module fail its actual purpose even while all the happy-path
# assertions pass: a leaked "{placeholder}", a recommendation that secretly
# depends on scope/metric instead of pattern, a crash on the majority
# primary_segment=None case, causal/overconfident language creeping into
# analyst-facing text, and -- the Task 18 addition -- an entry for a pattern
# the engine cannot actually produce.
from __future__ import annotations

from datetime import date

import pytest

from pulse.anomaly_detection import detect_anomalies
from pulse.metrics import compute_metrics, load_gold
from pulse.playbook import PATTERNS, PLAYBOOK, _num, recommend, render_context
from pulse.root_cause import diagnose
from pulse.types import (
    AnalysisParams,
    Anomaly,
    AssociatedDriver,
    Diagnosis,
    Impact,
    SegmentContribution,
)

P = AnalysisParams(as_of=date(2026, 9, 10))

# Substrings that would smuggle causation, false precision or autonomous
# action into text a human analyst reads and acts on.
# BOTH LANGUAGES, deliberately. The text a human reads is Brazilian Portuguese
# now, so a guarantee scanned only in English would be a guarantee enforced in
# a language nobody in the audience reads. The English entries stay: fixtures,
# docstrings and engine-adjacent strings still carry English, and a scan that
# stopped checking it would quietly stop protecting it.
_BANNED_CAUSAL = (
    "caused by",
    "the cause is",
    "proves that",
    "proves it",
    "because of",
    "due to",
    "results from",
    "is responsible for",
    "causado por",
    "causada por",
    "por causa de",
    "devido a",
    "provou",
    "prova que",
    "a causa é",
    "é a causa",
    "resulta de",
    "responsável pelo",
    "responsável pela",
)
_BANNED_OVERCONFIDENT = (
    "exact loss",
    "the incident cost",
    "perda exata",
    "custo exato",
    "o incidente custou",
)
_BANNED_AUTONOMOUS = (
    "will automatically",
    "the system will",
    "has been applied",
    "o sistema irá",
    "o sistema vai",
    "foi aplicado",
    "será aplicado",
    "automaticamente pelo sistema",
)

_TEMPLATED_FIELDS = ("action", "rationale", "expected_effect", "validation_method")


# --- fixtures: no conftest.py exists yet, so these are local to this file ---


def _make_anomaly(**overrides) -> Anomaly:
    base = dict(
        metric="gmv",
        scope="zone",
        scope_value="7",
        recent_value=886_000.0,
        baseline_value=1_001_000.0,
        deviation_abs=-115_000.0,
        deviation_pct=-11.46,
        z_score=-2.87,
        direction="drop",
        first_detected_date=date(2026, 8, 11),
        n_observations=70,
    )
    base.update(overrides)
    return Anomaly(**base)


def _make_driver(**overrides) -> AssociatedDriver:
    base = dict(
        metric="avg_actual_delivery_minutes",
        recent=42.0,
        baseline=35.0,
        deviation_pct=20.0,
        correlation_with_target=-0.4659,
        temporal_alignment_days=0,
        evidence_strength="moderate",
    )
    base.update(overrides)
    return AssociatedDriver(**base)


def _make_segment(**overrides) -> SegmentContribution:
    base = dict(
        dimension="zone",
        segment="7",
        segment_deviation_abs=-115_000.0,
        contribution_pct=56.15,
        rank=1,
    )
    base.update(overrides)
    return SegmentContribution(**base)


_UNSET = object()


@pytest.fixture
def fake_diagnosis():
    def _build(
        pattern="fulfillment_eta_degradation",
        *,
        anomaly=None,
        primary_segment=_UNSET,
        drivers=None,
        confidence=0.62,
        funnel_break_stage="completion_rate",
    ):
        anom = anomaly or _make_anomaly()
        seg = _make_segment() if primary_segment is _UNSET else primary_segment
        drv = (_make_driver(),) if drivers is None else drivers
        return Diagnosis(
            anomaly=anom,
            contributions=(seg,) if seg is not None else (),
            primary_segment=seg,
            funnel=(),
            funnel_break_stage=funnel_break_stage,
            drivers=drv,
            pattern=pattern,
            confidence=confidence,
        )

    return _build


@pytest.fixture
def fake_impact():
    def _build(**overrides):
        base = dict(
            gmv_at_risk_brl=184_200.0,
            orders_lost=1_200,
            customers_affected=900,
            margin_impact_brl=40_000.0,
            daily_run_rate_brl=13_157.0,
            projected_30d_brl=184_200.0,
        )
        base.update(overrides)
        return Impact(**base)

    return _build


def _diagnoses_from_the_real_engine() -> list[Diagnosis]:
    gold = load_gold()
    return [
        diagnose(a, gold, P) for a in detect_anomalies(compute_metrics(gold, P), P)
    ]


def _zone7_gmv_diagnosis() -> Diagnosis:
    """The real, engine-computed zone-7 diagnosis -- same lookup as
    test_root_cause.zone7_gmv_diagnosis(), reproduced here so this file does
    not depend on another test module's private helper.
    """
    return next(
        d
        for d in _diagnoses_from_the_real_engine()
        if d.anomaly.metric == "gmv"
        and d.anomaly.scope == "zone"
        and d.anomaly.scope_value == "7"
    )


def _zone4_availability_diagnosis() -> Diagnosis:
    return next(
        d
        for d in _diagnoses_from_the_real_engine()
        if d.anomaly.scope == "zone" and d.anomaly.scope_value == "4"
    )


# --- structure: every entry must be reachable, nothing speculative -----------


def test_every_pattern_has_a_playbook_entry_including_the_fallback():
    assert "unknown_pattern" in PLAYBOOK
    for pattern in PATTERNS:
        assert pattern in PLAYBOOK
        entry = PLAYBOOK[pattern]
        for field in (*_TEMPLATED_FIELDS, "owner_function", "effort"):
            assert entry.get(field), f"{pattern} missing {field}"


def test_only_patterns_the_real_engine_produces_are_registered():
    """The governing rule of this file: an entry for a pattern no diagnostic
    code can emit is unverifiable theatre, so the playbook's non-fallback keys
    must be exactly the set the engine actually reaches on the real gold
    tables -- no more (a speculative entry) and no fewer (a pattern that
    classifies with nowhere to land, which would silently fall back to
    unknown_pattern and lose its recommendation).

    Task 18 earned exactly one new entry this way. promo_margin_erosion and
    acquisition_quality_decay were investigated and rejected: neither scope
    reaches the detector at all -- see the reachability assertions in
    tests/test_root_cause.py.
    """
    reached = {d.pattern for d in _diagnoses_from_the_real_engine()}
    assert "supply_availability_gap" in reached
    assert set(PATTERNS) == reached - {"unknown_pattern"}
    assert set(PATTERNS) == {"fulfillment_eta_degradation", "supply_availability_gap"}
    for unreachable in ("promo_margin_erosion", "acquisition_quality_decay"):
        assert unreachable not in PLAYBOOK


def test_patterns_tuple_is_derived_from_the_yaml_not_hand_maintained():
    """Adding a pattern must be a YAML edit, not a code change: PATTERNS is
    computed from PLAYBOOK's own keys, so this holds by construction -- but
    pin it so a future refactor can't quietly reintroduce a hardcoded list
    that drifts from the YAML."""
    assert PATTERNS == tuple(k for k in PLAYBOOK if k != "unknown_pattern")


# --- rendering: whitelisted context, missing key raises, nothing leaks ------


def test_every_template_renders_against_a_realistic_context():
    ctx = {
        "segment": "Zona 7",
        "metric": "GMV",
        "deviation_pct": "-11.5%",
        "concentration": "56.2% do desvio em nível de empresa",
        "drivers_list": "actual delivery time, promised ETA e on-time rate",
        "driver_note": (
            "Séries associadas que se moveram junto com ela nesta janela: "
            "on-time rate."
        ),
        "confidence_pct": "62%",
        "gmv_at_risk": "R$ 184,200",
    }
    for entry in PLAYBOOK.values():
        for field in _TEMPLATED_FIELDS:
            entry[field].format(**ctx)  # raises KeyError on an unknown token


def test_missing_context_key_raises_rather_than_producing_a_placeholder():
    incomplete_ctx = {"segment": "Zona 7"}
    with pytest.raises(KeyError):
        PLAYBOOK["fulfillment_eta_degradation"]["rationale"].format(**incomplete_ctx)


def test_no_unrendered_placeholder_survives_recommend(fake_diagnosis, fake_impact):
    impact = fake_impact()
    for pattern in (*PATTERNS, "unknown_pattern", "a_pattern_nobody_taught_it"):
        rec = recommend(fake_diagnosis(pattern=pattern), impact)
        for field in (
            rec.action,
            rec.rationale,
            rec.expected_effect,
            rec.validation_method,
        ):
            assert "{" not in field and "}" not in field


def test_render_context_handles_primary_segment_none_without_crashing(
    fake_diagnosis, fake_impact
):
    """Most diagnoses carry primary_segment=None (rate/duration metrics get no
    segment decomposition, by design). This must render an honest scope
    description, never crash and never fabricate a segment."""
    d = fake_diagnosis(primary_segment=None, anomaly=_make_anomaly(scope="zone", scope_value="3"))
    ctx = render_context(d, fake_impact())
    assert ctx["segment"] == "Zona 3"  # the anomaly's own scope, not invented
    rec = recommend(d, fake_impact())
    assert rec.action and rec.rationale
    assert "{" not in rec.rationale

    company_scoped = fake_diagnosis(
        primary_segment=None, anomaly=_make_anomaly(scope="company", scope_value="all")
    )
    assert render_context(company_scoped, fake_impact())["segment"] == "Toda a empresa"


def test_driver_note_reports_only_evidence_the_engine_counts(
    fake_diagnosis, fake_impact
):
    """A weak correlation contributes zero to confidence and cannot pick a
    pattern. Naming it as an "associated series that moved with it" in the
    sentence a human reads would reintroduce, as prose, the evidence the
    engine has already judged to be none.
    """
    all_weak = fake_diagnosis(
        drivers=(
            _make_driver(correlation_with_target=0.164, evidence_strength="weak"),
            _make_driver(
                metric="on_time_rate",
                correlation_with_target=-0.131,
                evidence_strength="weak",
            ),
        )
    )
    note = render_context(all_weak, fake_impact())["driver_note"]
    assert "nenhum fator candidato atingiu" in note.lower()
    assert "actual delivery time" not in note

    mixed = fake_diagnosis(
        drivers=(
            _make_driver(evidence_strength="moderate"),
            _make_driver(
                metric="on_time_rate",
                correlation_with_target=-0.11,
                evidence_strength="weak",
            ),
        )
    )
    note = render_context(mixed, fake_impact())["driver_note"]
    assert "actual delivery time" in note
    assert "on-time rate" not in note


# --- selection is keyed on pattern alone -------------------------------------


def test_selection_never_branches_on_scope_or_metric(fake_diagnosis, fake_impact):
    """The hard requirement of this task: a `if zone == 7` anywhere is a
    failure. Two diagnoses with the SAME pattern but different scope AND
    different underlying metric must produce the identical playbook_id and
    the identical action text -- action deliberately carries no
    diagnosis-specific token, so this holds only if selection truly never
    looks past `.pattern`. Checked for EVERY registered pattern, so a pattern
    added later cannot quietly opt out of the property."""
    impact = fake_impact()
    for pattern in (*PATTERNS, "unknown_pattern"):
        zone_seven = fake_diagnosis(
            pattern=pattern,
            anomaly=_make_anomaly(scope="zone", scope_value="7", metric="gmv"),
            primary_segment=_make_segment(dimension="zone", segment="7"),
        )
        merchant_category = fake_diagnosis(
            pattern=pattern,
            anomaly=_make_anomaly(
                scope="merchant_category",
                scope_value="electronics",
                metric="availability_rate",
            ),
            primary_segment=_make_segment(
                dimension="merchant_category",
                segment="electronics",
                contribution_pct=80.0,
            ),
        )
        rec_a = recommend(zone_seven, impact)
        rec_b = recommend(merchant_category, impact)
        assert rec_a.playbook_id == rec_b.playbook_id == pattern
        assert rec_a.action == rec_b.action
        assert rec_a.owner_function == rec_b.owner_function
        assert rec_a.effort == rec_b.effort

    unknown_zone = fake_diagnosis(
        pattern="unknown_pattern",
        anomaly=_make_anomaly(scope="zone", scope_value="1", metric="cancellation_rate"),
        primary_segment=None,
    )
    unknown_channel = fake_diagnosis(
        pattern="unknown_pattern",
        anomaly=_make_anomaly(
            scope="acquisition_channel", scope_value="paid_social", metric="on_time_rate"
        ),
        primary_segment=None,
    )
    rec_c = recommend(unknown_zone, impact)
    rec_d = recommend(unknown_channel, impact)
    assert rec_c.playbook_id == rec_d.playbook_id == "unknown_pattern"
    assert rec_c.action == rec_d.action


def test_unknown_pattern_is_the_explicit_fallback_for_any_unrecognised_key(
    fake_diagnosis, fake_impact
):
    rec = recommend(fake_diagnosis(pattern="something_new"), fake_impact())
    assert rec.playbook_id == "unknown_pattern"
    assert "investig" in rec.action.lower()
    assert rec.action != ""


def test_unknown_pattern_action_is_substantive_not_a_stub(fake_diagnosis, fake_impact):
    """unknown_pattern is the MAJORITY case on this dataset (8 of 12), so
    every one of its four fields has to be real analytical guidance -- what
    was observed, and what to do next to classify it -- rather than a stub
    apology. Checked on all four, not just the action.
    """
    rec = recommend(fake_diagnosis(pattern="never_seen_before"), fake_impact())
    assert rec.playbook_id == "unknown_pattern"
    assert len(rec.action) >= 80
    assert len(rec.rationale) >= 150
    assert len(rec.expected_effect) >= 120
    assert len(rec.validation_method) >= 150
    fields = (
        rec.action,
        rec.rationale,
        rec.expected_effect,
        rec.validation_method,
    )
    for text in fields:
        for placeholder_marker in ("todo", "tbd", "a definir", "em breve"):
            assert placeholder_marker not in text.lower()
    # Guidance, not an apology: it says what to measure next.
    assert "métrica primária" in rec.validation_method.lower()
    assert "no mínimo" in rec.expected_effect.lower()


# --- language rules -----------------------------------------------------------


def test_no_causal_language_in_playbook_yaml():
    blob = " ".join(
        str(v) for entry in PLAYBOOK.values() for v in entry.values()
    ).lower()
    for phrase in _BANNED_CAUSAL:
        assert phrase not in blob


def test_no_causal_language_in_rendered_recommendations(fake_diagnosis, fake_impact):
    impact = fake_impact()
    for pattern in (*PATTERNS, "unknown_pattern", "never_before_seen"):
        rec = recommend(fake_diagnosis(pattern=pattern), impact)
        blob = " ".join(
            [rec.action, rec.rationale, rec.expected_effect, rec.validation_method]
        ).lower()
        for phrase in _BANNED_CAUSAL:
            assert phrase not in blob


def test_impact_wording_is_conservative_never_exact():
    """Every entry, not a hand-listed pair: the 56-day baseline is itself
    contaminated by the incident, so EVERY estimate this playbook renders is a
    floor and has to be worded like one."""
    for key, entry in PLAYBOOK.items():
        text = entry["expected_effect"].lower()
        for phrase in _BANNED_OVERCONFIDENT:
            assert phrase not in text
        assert "estimad" in text, f"{key} does not hedge its estimate"
        assert "no mínimo" in text, f"{key} does not word its impact as a floor"
        assert "piso" in text, f"{key} does not call its figure a floor"


def test_gmv_at_risk_is_a_magnitude_not_a_signed_deviation(fake_diagnosis, fake_impact):
    """Regression: projected_30d_brl is a signed deviation (negative for a
    drop). Rendering it raw produced "at least R$ -18,759" -- wrong on its
    face for a recoverable/at-risk amount, and backwards as a floor (a
    bigger loss would print as a smaller number). The token must be an
    absolute magnitude; direction stays carried by the surrounding words
    ("recoverable", "at risk"), not by the number's sign.

    A test that only checks for unrendered {placeholders} cannot catch a
    correctly-rendered wrong number, so this checks the actual rendered
    text against a negative (drop-direction) impact.
    """
    drop_impact = fake_impact(projected_30d_brl=-18_759.17)
    # Brazilian notation, written out rather than derived from the formatter:
    # "." groups the thousands, and the magnitude carries no sign.
    expected_magnitude = "R$ 18.759"

    for pattern in (*PATTERNS, "unknown_pattern"):
        rec = recommend(fake_diagnosis(pattern=pattern), drop_impact)
        for field in (
            rec.action,
            rec.rationale,
            rec.expected_effect,
            rec.validation_method,
        ):
            assert "R$ -" not in field
        assert expected_magnitude in rec.expected_effect


def test_a_favourable_projection_is_never_rendered_as_money_at_risk(
    fake_diagnosis, fake_impact
):
    """Review #3 class C5, in the playbook. abs() of a SIGNED projection turned a
    scope whose GMV ROSE into "GMV recuperável ... no mínimo R$ 4.619" -- reached
    at the sidebar's sensitivity 1,5 / window 28 on the company priority. A rise
    has nothing to recover, so the floor it states is zero, never the rise.
    """
    rise = fake_impact(projected_30d_brl=4_619.0, daily_run_rate_brl=153.97)
    for pattern in (*PATTERNS, "unknown_pattern"):
        rec = recommend(fake_diagnosis(pattern=pattern), rise)
        assert "R$ 4.619" not in rec.expected_effect, pattern
        assert "no mínimo R$ 0" in rec.expected_effect, pattern


def test_no_single_driver_is_named_as_the_sole_cause():
    """The three fulfilment drivers cluster at |r| 0.401-0.448 -- too narrow
    a spread to rank with confidence. The rationale must describe them as a
    cluster, never single one out as "the driver"."""
    text = PLAYBOOK["fulfillment_eta_degradation"]["rationale"].lower()
    assert "{drivers_list}" in PLAYBOOK["fulfillment_eta_degradation"]["rationale"]
    for phrase in (
        "the driver is",
        "the top driver",
        "o fator é",
        "o principal fator",
        "o fator determinante",
        "domina",
    ):
        assert phrase not in text, phrase
    # The family moved together; the ordering inside it is not robust.
    assert "em conjunto" in text


def test_no_entry_names_a_driver_metric_literally():
    """Every rationale reaches its drivers through {drivers_list} or
    {driver_note}, never by typing one in. A hardcoded metric name would be a
    claim about which series moved that the evidence has to make instead."""
    for key, entry in PLAYBOOK.items():
        blob = " ".join(entry[f] for f in _TEMPLATED_FIELDS).lower()
        for label in ("actual delivery time", "promised eta", "on-time rate"):
            assert label not in blob, f"{key} hardcodes the driver {label!r}"


def test_recommendations_are_advisory_never_autonomous():
    for entry in PLAYBOOK.values():
        text = entry["action"].lower()
        for phrase in _BANNED_AUTONOMOUS:
            assert phrase not in text


def test_validation_method_names_a_metric_a_comparison_and_a_window():
    for entry in PLAYBOOK.values():
        text = entry["validation_method"].lower()
        assert "métrica" in text
        assert any(
            word in text for word in ("controle", "comparação", "contra")
        )
        assert "dia" in text  # a stated window, e.g. "14 dias"
        assert "primária" in text  # a NAMED primary metric, not just "a metric"


# --- the two diagnoses this task must serve well -----------------------------


def test_zone7_diagnosis_produces_a_well_formed_fulfillment_recommendation(fake_impact):
    d = _zone7_gmv_diagnosis()
    assert d.pattern == "fulfillment_eta_degradation"

    rec = recommend(d, fake_impact())
    assert rec.playbook_id == "fulfillment_eta_degradation"
    assert rec.owner_function == "Operations / Logistics"
    assert rec.effort == "medium"

    assert "Zona 7" in rec.rationale
    assert f"{_num(d.confidence * 100)}%" in rec.rationale
    assert f"{_num(d.primary_segment.contribution_pct, 1)}%" in rec.rationale
    for label in ("actual delivery time", "promised ETA", "on-time rate"):
        assert label in rec.rationale

    for field in (rec.action, rec.rationale, rec.expected_effect, rec.validation_method):
        assert "{" not in field and "}" not in field
    blob = " ".join(
        [rec.action, rec.rationale, rec.expected_effect, rec.validation_method]
    ).lower()
    for phrase in (*_BANNED_CAUSAL, *_BANNED_OVERCONFIDENT, *_BANNED_AUTONOMOUS):
        assert phrase not in blob


def test_zone4_diagnosis_reaches_the_supply_recommendation_end_to_end(fake_impact):
    """Not a synthetic diagnosis built to hit the key: this is the real
    engine's zone-4 output, from gold through detection and diagnosis into the
    playbook. The pattern is only reachable at all because Task 18 registered
    availability_rate.
    """
    d = _zone4_availability_diagnosis()
    assert d.anomaly.metric == "availability_rate"
    assert d.pattern == "supply_availability_gap"

    rec = recommend(d, fake_impact())
    assert rec.playbook_id == "supply_availability_gap"
    assert rec.owner_function == "Marketplace Supply / Merchant Operations"
    assert rec.effort == "medium"

    assert "Zona 4" in rec.rationale
    assert "Merchant availability" in rec.rationale
    assert f"{_num(d.anomaly.deviation_pct, 1, plus=True)}%" in rec.rationale
    assert f"{_num(d.confidence * 100)}%" in rec.rationale
    # All three fulfilment candidates are weak here, so the rationale must say
    # nothing moved with it rather than list them as associated evidence.
    assert "nenhum fator candidato atingiu" in rec.rationale.lower()
    for label in ("actual delivery time", "promised ETA", "on-time rate"):
        assert label not in rec.rationale

    for field in (rec.action, rec.rationale, rec.expected_effect, rec.validation_method):
        assert "{" not in field and "}" not in field
    blob = " ".join(
        [rec.action, rec.rationale, rec.expected_effect, rec.validation_method]
    ).lower()
    for phrase in (*_BANNED_CAUSAL, *_BANNED_OVERCONFIDENT, *_BANNED_AUTONOMOUS):
        assert phrase not in blob


def test_rate_metric_anomaly_has_no_primary_segment_and_still_renders(fake_impact):
    """Real end-to-end check of the majority case: a zone-scoped
    completion_rate anomaly gets no segment decomposition (rates aren't
    additive across zones), so primary_segment is None on the actual engine
    output, not just in a synthetic fixture."""
    gold = load_gold()
    anomaly = next(
        a
        for a in detect_anomalies(compute_metrics(gold, P), P)
        if a.metric == "completion_rate" and a.scope == "zone"
    )
    d = diagnose(anomaly, gold, P)
    assert d.primary_segment is None

    rec = recommend(d, fake_impact())
    assert rec.action
    assert "{" not in rec.rationale and "}" not in rec.rationale
