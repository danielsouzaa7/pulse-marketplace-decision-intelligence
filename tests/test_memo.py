# Tests for the Decision Memo (Task 14).
#
# The headline test in this file is
# test_memo_is_identical_with_and_without_a_narrator: mechanical proof that the
# LLM owns no facts and no policy. Everything else here pins the properties
# that would let this memo mislead the person approving it while every
# happy-path assertion still passed:
#
#   * a rendered "{placeholder}" or a "R$ -" on a recoverable amount
#   * causal language ("caused by", "proves that") over correlational evidence
#   * first_detected_date rendered as an incident start date, which it is not
#   * a concentration share presented as the whole
#   * one of several tightly-clustered drivers named as THE cause
#   * a lead/lag story built on temporal_alignment_days, whose surface is flat
#   * a crash, or a fabricated segment, when primary_segment is None
#   * a default decision status that reads like an approval
from __future__ import annotations

import re
from dataclasses import replace
from datetime import date, datetime

import pytest

from pulse.decision_memo import (
    DECISION_STATUSES,
    DEFAULT_DECISION_STATUS,
    EVIDENCE_ANCHOR_IMPACT_SCORE,
    NullNarrator,
    build_memo,
    limitations,
    memo_to_dict,
    memo_to_markdown,
    scope_of,
)
from pulse.engine import run_decision_cycle
from pulse.metrics import load_gold
from pulse.playbook import _num
from pulse.types import (
    AnalysisParams,
    Anomaly,
    AssociatedDriver,
    Diagnosis,
    Impact,
    Priority,
    Recommendation,
    SegmentContribution,
)

P = AnalysisParams(as_of=date(2026, 9, 10))

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
    "the exact impact",
    "perda exata",
    "custo exato",
    "o impacto exato",
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
_BANNED_SOLE_DRIVER = (
    "the driver is",
    "the top driver",
    "the main driver",
    "o fator é",
    "o principal fator",
    "o fator determinante",
    "o fator dominante",
    "domina",
)
_REQUIRED_HEADINGS = (
    "## Incidente",
    "## Escopo",
    "## Período analisado",
    "## Impacto estimado",
    "## Concentração",
    "## Fatores associados",
    "## Evidências",
    "## Prioridade",
    "## Ação recomendada",
    "## Resultado potencial",
    "## Método de validação",
    "## Confiança",
    "## Limitações",
    "## Status da decisão humana",
)


@pytest.fixture(scope="module")
def gold():
    """Loaded at test time, never at import time: tests/test_contracts.py
    rebuilds data/gold mid-suite, so a snapshot taken at collection time is not
    necessarily the gold the rest of the run sees.
    """
    return load_gold()


@pytest.fixture(scope="module")
def cycle(gold):
    return run_decision_cycle(gold, P)


@pytest.fixture(scope="module")
def memos(cycle):
    return cycle.memos


@pytest.fixture(scope="module")
def markdowns(memos):
    return [memo_to_markdown(m) for m in memos]


# --- synthetic priorities, so BOTH primary_segment branches are covered
# whatever the real data happens to produce on a given run -------------------

_SEGMENT = SegmentContribution(
    dimension="zone",
    segment="7",
    segment_deviation_abs=-115_000.0,
    contribution_pct=56.15,
    rank=1,
)


def _priority(*, primary_segment, rank=1, pattern="fulfillment_eta_degradation"):
    anomaly = Anomaly(
        metric="gmv",
        scope="zone",
        scope_value="7",
        recent_value=886_000.0,
        baseline_value=1_001_000.0,
        deviation_abs=-115_000.0,
        deviation_pct=-11.46,
        z_score=-2.87,
        direction="drop",
        first_detected_date=date(2026, 8, 28),
        n_observations=14,
    )
    diagnosis = Diagnosis(
        anomaly=anomaly,
        contributions=(primary_segment,) if primary_segment is not None else (),
        primary_segment=primary_segment,
        funnel=(),
        funnel_break_stage="completion_rate",
        drivers=(
            AssociatedDriver(
                metric="avg_actual_delivery_minutes",
                recent=42.0,
                baseline=35.0,
                deviation_pct=20.0,
                correlation_with_target=-0.4659,
                temporal_alignment_days=0,
                evidence_strength="moderate",
            ),
            AssociatedDriver(
                metric="on_time_rate",
                recent=0.71,
                baseline=0.88,
                deviation_pct=-19.3,
                correlation_with_target=0.4464,
                temporal_alignment_days=2,
                evidence_strength="moderate",
            ),
        ),
        pattern=pattern,
        confidence=0.62,
    )
    impact = Impact(
        gmv_at_risk_brl=8_754.28,
        orders_lost=135,
        customers_affected=1_016,
        margin_impact_brl=3_229.39,
        daily_run_rate_brl=-625.31,
        projected_30d_brl=-18_759.17,
    )
    return Priority(
        diagnosis=diagnosis,
        impact=impact,
        impact_score=83.49,
        rank=rank,
        score_breakdown={
            "w_gmv": 0.45,
            "w_orders": 0.20,
            "w_customers": 0.15,
            "w_confidence": 0.20,
            "n_gmv": 1.0,
            "n_orders": 1.0,
            "n_customers": 0.259,
            "n_confidence": 0.62,
            "supporting_anomalies": 7.0,
            "nested_groups_netted": 0.0,
        },
    )


_RECOMMENDATION = Recommendation(
    action="Encaminhar para Operações/Logística para avaliação.",
    rationale="A taxa de conclusão é o estágio do funil que se moveu.",
    expected_effect=(
        "O GMV recuperável estimado é de no mínimo R$ 18,759 em 30 dias."
    ),
    validation_method=(
        "Diferenças em diferenças em nível de zona, janela de 14 dias."
    ),
    owner_function="Operations / Logistics",
    playbook_id="fulfillment_eta_degradation",
    effort="medium",
)


# --- the brief's four tests -------------------------------------------------


def test_memo_has_every_required_section_populated(memos):
    d = memo_to_dict(memos[0])
    for section in (
        "incident",
        "impact",
        "concentration",
        "associated_drivers",
        "evidence",
        "priority",
        "recommended_action",
        "potential_result",
        "validation_method",
        "confidence",
    ):
        assert d.get(section) not in (None, "", [], {}), f"empty section: {section}"


def test_memo_is_identical_with_and_without_a_narrator(gold):
    """THE headline test: mechanical proof the LLM owns no facts and no policy.

    A narrator returning deliberately different prose must change exactly one
    field. If any number, recommendation, status or evidence line differs, the
    narration boundary is decorative rather than structural.
    """

    class StubNarrator:
        def narrate(self, bundle):
            return "Completely different prose."

    plain = run_decision_cycle(gold, P, narrator=None).memos[0]
    narrated = run_decision_cycle(gold, P, narrator=StubNarrator()).memos[0]

    a, b = memo_to_dict(plain), memo_to_dict(narrated)
    a.pop("narrative")
    b.pop("narrative")
    assert a == b
    assert narrated.narrative == "Completely different prose."
    assert plain.narrative is None


def test_markdown_renders_without_placeholders(markdowns):
    for md in markdowns:
        assert "{" not in md and "}" not in md
        assert "TBD" not in md and "TODO" not in md
        assert "R$" in md


def test_memo_uses_association_language(markdowns):
    for md in markdowns:
        lowered = md.lower()
        assert "fatores associados" in lowered
        assert "associad" in lowered
        for banned in _BANNED_CAUSAL:
            assert banned not in lowered, banned


# --- narration boundary, the rest of it -------------------------------------


def test_null_narrator_is_the_offline_default_and_adds_nothing():
    """A complete memo needs no ANTHROPIC_API_KEY. NullNarrator is the correct
    offline behaviour, not a stub: its memo must equal the no-narrator memo in
    every field, narrative included.
    """
    plain = build_memo(_priority(primary_segment=_SEGMENT), _RECOMMENDATION, P)
    nulled = build_memo(
        _priority(primary_segment=_SEGMENT),
        _RECOMMENDATION,
        P,
        narrator=NullNarrator(),
    )
    assert memo_to_dict(plain) == memo_to_dict(nulled)
    assert nulled.narrative is None
    assert memo_to_markdown(nulled) == memo_to_markdown(plain)


def test_narrator_receives_a_memo_whose_facts_are_already_decided():
    """The boundary is structural: the narrator is handed a FINISHED memo, so
    there is no field left for it to influence even in principle.
    """
    seen = {}

    class Recorder:
        def narrate(self, bundle):
            seen["memo"] = bundle
            return "prose"

    built = build_memo(
        _priority(primary_segment=_SEGMENT), _RECOMMENDATION, P, narrator=Recorder()
    )
    handed = seen["memo"]
    assert handed.narrative is None
    assert handed.impact == built.impact
    assert handed.evidence == built.evidence
    assert handed.recommended_action == built.recommended_action
    assert built == replace(handed, narrative="prose")


def test_narrative_is_rendered_as_clearly_labelled_and_non_authoritative():
    class Stub:
        def narrate(self, memo):
            return "Some prose."

    narrated = build_memo(
        _priority(primary_segment=_SEGMENT), _RECOMMENDATION, P, narrator=Stub()
    )
    md = memo_to_markdown(narrated)
    assert "## Narrativa (escrita por IA)" in md
    assert "não carrega fatos próprios" in md
    assert "Some prose." in md


# --- structure ---------------------------------------------------------------


def test_every_required_section_is_rendered_as_a_heading(markdowns):
    assert markdowns
    for md in markdowns:
        assert len(md) > 1_500
        for heading in _REQUIRED_HEADINGS:
            assert heading in md, heading


def test_memo_numbers_are_the_priority_numbers(cycle):
    for memo, priority in zip(cycle.memos, cycle.priorities):
        assert memo.priority == priority.rank
        assert memo.impact == priority.impact
        assert memo.confidence == priority.diagnosis.confidence
        assert memo.associated_drivers == priority.diagnosis.drivers
        assert scope_of(memo) == (
            priority.diagnosis.anomaly.scope,
            priority.diagnosis.anomaly.scope_value,
        )


def test_memo_id_is_deterministic_and_carries_its_scope(memos):
    first = memos[0]
    scope, scope_value = scope_of(first)
    assert first.memo_id == f"PULSE-20260910-P1-{scope}-{scope_value}"
    assert [m.memo_id for m in memos] == [
        f"PULSE-20260910-P{m.priority}-{'-'.join(scope_of(m))}" for m in memos
    ]


def test_generated_at_is_derived_from_as_of_not_the_wall_clock(gold, memos):
    """Two runs of the same analysis must produce byte-identical artefacts --
    otherwise the CLI/library comparison compares timestamps and artifacts/
    churns on every run. generated_at is the analysis instant, not now().
    """
    for memo in memos:
        assert memo.generated_at == datetime(2026, 9, 10, 0, 0, 0)
    again = run_decision_cycle(gold, P).memos
    assert [memo_to_markdown(m) for m in again] == [memo_to_markdown(m) for m in memos]


# --- language rules ----------------------------------------------------------


def test_no_causal_overconfident_or_autonomous_language_anywhere(markdowns):
    for md in markdowns:
        lowered = md.lower()
        for phrase in (*_BANNED_CAUSAL, *_BANNED_OVERCONFIDENT, *_BANNED_AUTONOMOUS):
            assert phrase not in lowered, phrase


def test_impact_is_worded_as_a_conservative_floor(markdowns):
    for md in markdowns:
        lowered = md.lower()
        assert "no mínimo" in lowered
        assert "estimad" in lowered
        assert "piso" in lowered
        assert "subestim" in lowered


def test_numbers_are_rendered_in_brazilian_notation(markdowns):
    """Pinned by literal notation, not by agreement with the formatter.

    "." groups thousands and "," is the decimal mark, everywhere a figure
    reaches a reader. The app renders these same bytes next to its own
    pt-BR figures: a memo showing "R$ 18,759" beside a page showing
    "R$ 18.759,17" is one product speaking two number languages at a reader
    who then has to work out which one this number is in.
    """
    for md in markdowns:
        assert re.search(r"R\$ \d{1,3}(?:\.\d{3})+", md), "no grouped currency"
        assert not re.search(r"R\$ \d{1,3}(?:,\d{3})+", md), "en-US currency"
        assert re.search(r"\d,\d", md), "no comma decimal anywhere"
        assert not re.search(r"\d\.\d%", md), "en-US decimal in a percentage"
        assert not re.search(r"z = [-+]?\d+\.\d", md), "en-US decimal in a z"


def test_no_negative_currency_is_ever_rendered(markdowns):
    """projected_30d_brl and daily_run_rate_brl are SIGNED. Rendering one raw
    produces "R$ -18,759" inside an "at least" sentence -- backwards as a
    floor, since a bigger loss prints as a smaller number. Direction belongs in
    the words, never in the sign.
    """
    for md in markdowns:
        assert "R$ -" not in md
        assert "-R$" not in md


def test_first_flagged_date_is_never_rendered_as_an_incident_start(cycle, markdowns):
    """first_detected_date is bounded by the comparison window -- the detector
    never looks outside it -- so on this dataset the true onset precedes the
    window entirely. Rendering it as a start date would be a false claim.
    """
    for priority, md in zip(cycle.priorities, markdowns):
        flagged = priority.diagnosis.anomaly.first_detected_date.isoformat()
        lowered = md.lower()
        assert flagged in md
        assert "primeiro sinal identificado dentro da janela analisada" in lowered
        for phrase in (
            f"began on {flagged}",
            f"started on {flagged}",
            f"start date {flagged}",
            f"onset {flagged}",
            f"since {flagged}",
            "incident started",
            "incident onset",
            f"começou em {flagged}",
            f"iniciou em {flagged}",
            f"desde {flagged}",
            f"a partir de {flagged}",
            "início do incidente",
            "incidente começou",
            "data de início do",
        ):
            assert phrase not in lowered, phrase


def test_concentration_is_reported_as_a_share_never_as_the_whole(cycle):
    """Zone 7 carries ~56% of the company GMV deviation on this dataset, not
    100%: other zones contributed and some partly offset it.
    """
    for memo, priority in zip(cycle.memos, cycle.priorities):
        text = memo.concentration.lower()
        for phrase in (
            "entirely",
            "100% of",
            "all of the company",
            "the whole of",
            "integralmente",
            "100% do",
            "todo o desvio",
            "a totalidade",
        ):
            assert phrase not in text, phrase
        if priority.diagnosis.primary_segment is not None:
            assert "uma parcela, não o total" in text


def test_no_single_driver_is_named_as_the_cause(markdowns):
    """The fulfilment candidates cluster tightly in |r| -- too narrow a spread
    to rank with confidence -- so the memo describes them as moving together.
    """
    for md in markdowns:
        lowered = md.lower()
        for phrase in _BANNED_SOLE_DRIVER:
            assert phrase not in lowered, phrase
        if "| Série |" in md:
            assert "em conjunto" in lowered or "nenhuma é identificada" in lowered


def test_no_lead_lag_narrative_is_built_on_temporal_alignment(markdowns):
    """The lag surface on a sustained level shift is nearly flat (|r| moves
    only between 0.44 and 0.49 across -7..+7 days on zone 7), so an ordering
    story built on it would be a story about noise. The field stays in the
    data; it never becomes prose or a table column.
    """
    for md in markdowns:
        lowered = md.lower()
        assert "temporal_alignment" not in lowered
        assert " lag " not in lowered
        assert "lead/lag" not in lowered
        for phrase in (
            "days ahead of",
            "moved first",
            "preceded the",
            "led the",
            "dias antes de",
            "moveu-se primeiro",
            "se moveu primeiro",
            "precedeu",
            "antecedeu",
            "defasagem",
        ):
            assert phrase not in lowered, phrase


# --- primary_segment=None, the majority case --------------------------------


def test_none_primary_segment_renders_without_crashing_or_inventing_a_segment():
    """Most diagnoses carry primary_segment=None: rates and durations are not
    additive across segments, so no share of a company deviation exists for
    them. The memo must say so rather than crash or make one up.
    """
    memo = build_memo(_priority(primary_segment=None), _RECOMMENDATION, P)
    md = memo_to_markdown(memo)
    assert "{" not in md
    assert (
        "nenhuma decomposição por segmento está disponível"
        in memo.concentration.lower()
    )
    assert "Zona 7" in memo.concentration  # the anomaly's own scope, not invented
    assert "%" not in memo.concentration  # no fabricated share
    for heading in _REQUIRED_HEADINGS:
        assert heading in md


def test_real_primary_segment_renders_the_measured_share():
    memo = build_memo(_priority(primary_segment=_SEGMENT), _RECOMMENDATION, P)
    share = f"{_num(_SEGMENT.contribution_pct, 1)}% do desvio"
    balance = f"{_num(100.0 - _SEGMENT.contribution_pct, 1, plus=True)}%"
    assert share in memo.concentration
    assert "uma parcela, não o total" in memo.concentration.lower()
    assert balance in memo.concentration  # the balance, stated explicitly


def test_every_real_memo_this_run_renders_fully(markdowns, memos):
    assert len(markdowns) == len(memos)
    for md in markdowns:
        assert md.startswith("# Resumo da decisão")
        assert md.endswith("\n")


# --- limitations and human decision status ----------------------------------


def test_limitations_are_specific_to_how_pulse_actually_works():
    text = " ".join(limitations(P)).lower()
    assert "piso" in text and "baseline de 56 dias" in text
    assert "associação" in text and "não estabelece" in text
    assert "limitada pela janela" in text
    assert f"cerca de {P.comparison_window_days // 2} dias" in text
    assert "sintétic" in text
    assert "subestimam" in text
    assert len(limitations(P)) >= 4


def test_limitations_appear_in_every_rendered_memo(markdowns):
    for md in markdowns:
        section = md.split("## Limitações", 1)[1]
        for marker in (
            "piso",
            "associação",
            "sintétic",
            "limitada pela janela",
        ):
            assert marker in section.lower(), marker


def test_human_decision_status_defaults_to_investigate_and_approves_nothing(markdowns):
    assert DEFAULT_DECISION_STATUS == "INVESTIGATE"
    assert DECISION_STATUSES[0] == "INVESTIGATE"
    for md in markdowns:
        section = md.split("## Status da decisão humana", 1)[1]
        lowered = section.lower()
        assert "INVESTIGATE" in section
        assert "nenhum humano revisou isto ainda" in lowered
        assert "nada neste memorando entra em vigor automaticamente" in lowered
        assert "aprovar" in lowered
        for phrase in _BANNED_AUTONOMOUS:
            assert phrase not in lowered, phrase


def test_derived_sections_are_carried_in_the_dict(memos):
    for memo in memos:
        d = memo_to_dict(memo)
        assert d["human_decision_status"] == "INVESTIGATE"
        assert d["limitations"] == list(limitations(P))
        assert d["period"].startswith(
            "Janela de comparação de 2026-08-28 a 2026-09-10"
        )
        assert (d["scope"], d["scope_value"]) == scope_of(memo)


def test_score_line_explains_a_confidence_that_differs_from_the_memos_own(cycle):
    """prioritization scores a scope group on its BEST-evidenced member, which
    need not be the diagnosis the memo is written about. Two different
    confidence numbers in one document, with no explanation of why, is the kind
    of detail that makes a reader distrust everything else in it.
    """
    for memo, priority in zip(cycle.memos, cycle.priorities):
        score_line = next(
            e for e in memo.evidence if e.startswith(EVIDENCE_ANCHOR_IMPACT_SCORE)
        )
        assert _num(priority.impact_score, 2) in score_line
        group_confidence = priority.score_breakdown["n_confidence"]
        if abs(group_confidence - priority.diagnosis.confidence) > 1e-12:
            assert _num(group_confidence, 3) in score_line
            assert _num(priority.diagnosis.confidence, 3) in score_line
            assert (
                "ver a linha de score na seção Evidências"
                in memo_to_markdown(memo)
            )


def test_uniformly_weak_drivers_are_not_narrated_as_deteriorating_together():
    """Zone 4's candidates correlate at |r| 0.06-0.12 on series that moved
    0.6%. Describing those as "deteriorated together" -- which a spread-only
    cluster rule would -- reads as evidence where there is none. Where every
    candidate is weak, the memo has to say the association is unusable.
    """
    base = _priority(primary_segment=_SEGMENT)
    weak = tuple(
        replace(d, correlation_with_target=r, evidence_strength="weak")
        for d, r in zip(base.diagnosis.drivers, (-0.12, -0.09))
    )
    priority = replace(base, diagnosis=replace(base.diagnosis, drivers=weak))
    md = memo_to_markdown(build_memo(priority, _RECOMMENDATION, P))
    assert "nenhuma é identificada como fator" in md
    assert "se deterioraram em conjunto" not in md
    assert "não porque sustentam o diagnóstico" in md


# --- C1: the memo names the unit of time on every figure ---------------------


def test_a_daily_average_metric_value_is_marked_per_day():
    """_value() renders a WINDOW AGGREGATE, and for a per-day flow that is a
    daily average.

    Without the marker the memo said "GMV is R$ 2.731,66" about a fourteen-day
    window whose GMV was R$ 38.243,24, two sections above an Impact table quoting
    the accumulated figure -- the same document contradicting itself in the
    reader's own arithmetic.
    """
    from pulse.decision_memo import _value

    assert _value("gmv", 4756.225) == "R$ 4.756/dia"
    assert _value("contribution_margin", 3358.4776) == "R$ 3.358/dia"
    assert _value("orders_completed", 484.1429) == "484/dia"
    assert _value("sessions", 3059.2857) == "3.059/dia"
    assert _value("active_customers", 473.0) == "473/dia"


def test_a_rate_a_ticket_and_a_duration_are_never_marked_per_day():
    """The most tempting wrong fix is to suffix everything "/dia". A rate is
    dimensionless, a ticket is per ORDER and a duration is per DELIVERY.
    """
    from pulse.decision_memo import _value

    assert _value("completion_rate", 0.7345) == "73,5%"
    assert _value("aov", 61.922) == "R$ 62"
    assert _value("avg_actual_delivery_minutes", 31.7463) == "31,7 min"
    for metric, value in (
        ("completion_rate", 0.7345), ("aov", 61.922),
        ("avg_actual_delivery_minutes", 31.7463),
    ):
        assert "/dia" not in _value(metric, value)


def test_the_incident_sentence_marks_its_own_figures(memos):
    """The memo's first sentence quotes recent against baseline. Both are daily
    averages for a money metric, and both carry the marker.
    """
    money = [m for m in memos if "gmv" in m.incident.lower()
             or "GMV" in m.incident]
    assert money, "no memo in this run is written about a money metric"
    for memo in money:
        assert "/dia" in memo.incident, memo.incident


def test_the_impact_table_separates_accumulated_from_daily(markdowns):
    """Four rows accumulate over the window, two are per day, and they differ by
    a factor of the window length. Every row names which it is.
    """
    for markdown in markdowns:
        assert "acumulado na janela de 14 dias" in markdown
        assert "Desvio do GMV médio diário (ritmo)" in markdown
        assert "Ritmo diário projetado para 30 dias, acumulado" in markdown


def test_the_impact_sentence_states_the_multiplication_between_the_two(memos):
    """A daily deviation and a window deviation are the same measurement at two
    scales. The sentence says so, so the two cannot read as disagreeing figures.
    """
    for memo in memos:
        sentence = next(
            (line for line in memo.evidence if "GMV MÉDIO DIÁRIO" in line
             or "GMV médio diário" in line),
            None,
        )
        assert sentence is not None, memo.memo_id
        if "ACUMULADO" in sentence:
            assert "multiplicada pelo número de dias" in sentence
            assert "não uma segunda cifra" in sentence


def test_the_impact_sentence_arithmetic_actually_holds(cycle):
    """The claim the sentence makes, checked: the accumulated figure IS the daily
    deviation times the window length.
    """
    for priority in cycle.priorities[:3]:
        impact = priority.impact
        if impact.gmv_at_risk_brl <= 0:
            continue
        assert impact.gmv_at_risk_brl == pytest.approx(
            abs(impact.daily_run_rate_brl) * cycle.params.comparison_window_days,
            rel=1e-9,
        )


# --- I1: the memo's customer row follows the Impact, not the scope name ------


def test_the_memo_never_calls_a_netted_residual_a_measured_customer_count(cycle):
    """The aggregate priority's customer figure is a residual; its memo says so.

    3,896 where 5,632 customers actually ordered. The number alone cannot say
    which it is, which is why Impact carries the flag and one function turns it
    into words.
    """
    company = [
        m for m in cycle.memos
        if m.impact.customers_are_residual
    ]
    assert company, "no netted aggregate memo in this run"
    for memo in company:
        markdown = memo_to_markdown(memo)
        assert "Clientes considerados no impacto residual" in markdown
        assert "Clientes que pediram neste escopo" not in markdown

    segments = [m for m in cycle.memos if not m.impact.customers_are_residual]
    assert segments, "no segment-scope memo in this run"
    for memo in segments:
        markdown = memo_to_markdown(memo)
        assert "Clientes que pediram neste escopo" in markdown
        assert "residual" not in markdown.lower()
