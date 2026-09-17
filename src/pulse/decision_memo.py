# The Decision Memo: one Priority + its Recommendation, assembled into the
# artefact a human actually reads and approves.
#
#   build_memo(priority, recommendation, params, narrator=None) -> DecisionMemo
#   memo_to_dict(memo)                                          -> dict
#   memo_to_markdown(memo)                                      -> str
#
# THE LLM BOUNDARY LIVES HERE, and it is structural rather than asserted.
# build_memo assembles the whole memo from computed objects with
# narrative=None, and only then -- once every fact and every policy field is
# already fixed -- offers the finished memo to the narrator and puts whatever
# comes back into the single `narrative` field. A narrator therefore cannot
# reach a number, a recommendation or a status even in principle: it is handed
# a memo that is already complete. tests/test_memo.py proves the memo dict is
# identical with a narrator and without one.
#
# There is no Anthropic client in this module and no API call. `Narrator` is a
# Protocol and `NullNarrator` is the offline default; live narration is a later
# task. PULSE generates a complete memo with no API key.
#
# LANGUAGE. The memo is written in Brazilian Portuguese, because a Brazilian
# analyst reads it. What is NOT translated is what the engine keys on: memo_id,
# DECISION_STATUSES, the dataclass fields, the JSON keys, metric identifiers
# and the metric labels those identifiers render into ("GMV", "completion
# rate"). The scope DIMENSION identifier stays English too (`zone`, the same
# token memo_id and the gold tables carry), but the prose word built around it
# is Portuguese -- "Zona 7", not "Zone 7" -- rendered by metrics._scope_phrase,
# the single place that word is written; pulse.playbook and pulse.copilot
# render the same word through the same function, so a memo and the playbook
# rationale it quotes can never disagree on what to call a scope. Numbers keep
# the engine's own formatting so that the Copilot's numeric validator, which
# checks every figure in an answer against the evidence bundle, still
# recognises them.
#
# LANGUAGE RULES, enforced by tests in tests/test_memo.py in both languages,
# each one a property of THIS dataset rather than a style preference:
#
#   * Association, never causation. Every driver is a series that moved with
#     the target; nothing here establishes that one produced the other.
#     "associado a", "consistente com" -- never "causado por" / "devido a".
#   * Impact is a FLOOR. The baseline window ends the day before the comparison
#     window, so a deterioration that began inside it is already part of the
#     baseline being measured against. "estimado", "no mínimo", "piso
#     conservador" -- never "perda exata" ou "o incidente custou".
#   * first_detected_date is NOT the incident onset. The detector only looks
#     inside the comparison window, so the date it reports is bounded by that
#     window and on this dataset the real onset precedes it entirely. It is
#     rendered as "primeiro sinal identificado dentro da janela analisada",
#     never as a start date.
#   * Concentration is a SHARE, never the whole. One segment carrying 56% of a
#     company deviation means the other 44% sits elsewhere, and a negative
#     share means a segment moved favourably and masked part of the total.
#   * No single driver is named as THE cause. The fulfilment candidates cluster
#     tightly in correlation strength, so which one ranks first is not robust:
#     "as métricas de entrega se deterioraram em conjunto", never "o tempo de
#     entrega é a causa". And where every candidate is weak, the memo says the
#     association is not evidence rather than describing three flat series as
#     moving together.
#   * temporal_alignment_days is NOT narrated. The lag surface on a sustained
#     level shift is close to flat -- every lag inside the window overlaps the
#     same degraded regime -- so a lead/lag story built on it would be a story
#     about noise. The field is carried in the data, never in the prose.
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, time, timedelta
from typing import Protocol, runtime_checkable

from pulse.metrics import (
    DAILY_AVERAGE,
    METRIC_SEMANTICS,
    _UNITS,
    _scope_phrase,
)
from pulse.metrics import customers_label as _customers_label
from pulse.playbook import _brl, _label, _num, _sentence_case
from pulse.prioritization import PROJECTION_DAYS
from pulse.types import (
    AnalysisParams,
    AssociatedDriver,
    DecisionMemo,
    Diagnosis,
    Impact,
    Priority,
    Recommendation,
)

# The human-in-the-loop states. INVESTIGATE is the default every memo is born
# in: it records that no human has looked at this yet. It is deliberately not
# an approval, and nothing in PULSE acts on any of these -- Task 15 stores the
# state in session state and the engine never reads it back. The status VALUES
# are engine vocabulary and stay in English; the sentence around them does not.
DEFAULT_DECISION_STATUS = "INVESTIGATE"
DECISION_STATUSES: tuple[str, ...] = ("INVESTIGATE", "APPROVED", "REJECTED")

_DECISION_STATUS_NOTE = (
    "Este memorando é uma recomendação para um humano nomeado aprovar, "
    "rejeitar ou investigar mais a fundo. O PULSE escreve arquivos e renderiza "
    "páginas; ele não aciona nenhum sistema do marketplace e nada neste "
    "memorando entra em vigor automaticamente."
)

# Selector prefixes for the two evidence lines this module writes whose
# opening words app/streamlit_app.py matches (str.startswith) to pull them
# back out of memo.evidence -- tests/test_app.py asserts the same match.
# Defined once, here, where the sentences themselves are written, rather than
# as three separate literals in decision_memo.py, streamlit_app.py and
# test_app.py: a startswith() selector that stops matching produces an empty
# panel, not an error, so the three copies drifting apart would fail silently
# on screen rather than in a test.
EVIDENCE_ANCHOR_IMPACT_SCORE = "Score de impacto"
EVIDENCE_ANCHOR_PLAYBOOK_ENTRY = "Entrada do playbook"

# Absolute-correlation spread below which the drivers are described as one
# cluster rather than ranked. 0.15 is wide enough to cover both shapes this
# dataset produces (the three fulfilment candidates land within 0.02 of each
# other against zone-7 GMV, and within 0.11 against zone-7 completion rate).
_CLUSTER_SPREAD = 0.15


@runtime_checkable
class Narrator(Protocol):
    """Anything that can turn a finished memo into prose.

    It receives a DecisionMemo whose every fact, number and policy field is
    already decided, and returns text for the `narrative` field or None. It has
    no other channel into the memo.
    """

    def narrate(self, memo: DecisionMemo) -> str | None: ...


class NullNarrator:
    """The offline default: no narration at all.

    Not a stub waiting to be filled in -- it is the correct behaviour with no
    API key, which is the state this project must work in. A memo built with
    NullNarrator is complete.
    """

    def narrate(self, memo: DecisionMemo) -> str | None:
        del memo
        return None


# --- formatting ------------------------------------------------------------
#
# Every currency helper renders a MAGNITUDE and carries direction in words.
# Formatting a signed deviation raw produced "R$ -18,759" in a "no mínimo"
# sentence, which is backwards as a floor: a bigger loss prints as a smaller
# number. Direction belongs in the sentence, never in the sign.
#
# Numbers are rendered in Brazilian notation ("R$ 18.759", "-11,5%") by
# playbook._num / playbook._brl, which this module imports rather than
# duplicating. The Copilot's numeric validator parses that same notation, so
# the two move together and every figure in a memo still verifies against the
# evidence bundle.


def _direction_word(value: float) -> str:
    """Direction as words, agreeing with a following "baseline"."""
    if value < 0:
        return "abaixo da"
    return "acima da" if value > 0 else "em linha com a"


def _signed_brl(value: float, per: str = "") -> str:
    """e.g. "R$ 625 por dia abaixo da baseline" / "R$ 384 acima da baseline"."""
    suffix = f" {per}" if per else ""
    return f"{_brl(value)}{suffix} {_direction_word(value)} baseline"


def _value(metric: str, value: float) -> str:
    """Render a WINDOW AGGREGATE of one metric, in its own units and with its own
    aggregation marked.

    Every value this is handed is a window mean -- an anomaly's recent_value and
    baseline_value, a driver's recent and baseline, all from
    root_cause._window_means -- so for a metric METRIC_SEMANTICS classifies as a
    per-day flow the figure is a DAILY AVERAGE and is suffixed "/dia". Without
    that suffix this memo said "GMV is R$ 2.731,66" about a fourteen-day window
    whose GMV was R$ 38.243,24, and said it two sections above an Impact table
    quoting the accumulated figure -- the same document contradicting itself in
    the reader's own arithmetic.

    The unit and the semantic are both the engine's, read from metrics rather
    than decided here.
    """
    unit = _UNITS.get(metric, "")
    per_day = "/dia" if METRIC_SEMANTICS.get(metric) == DAILY_AVERAGE else ""
    if unit == "BRL":
        return f"{_brl(value)}{per_day}"
    if unit == "ratio":
        return f"{_num(value * 100, 1)}%"
    if unit == "minutes":
        return f"{_num(value, 1)} min"
    if unit in ("orders", "sessions", "customers"):
        return f"{_num(value)}{per_day}"
    return f"{_num(value, 1)}{per_day}"


def memo_id_for(rank: int, scope: str, scope_value: str, p: AnalysisParams) -> str:
    """The memo's stable identifier, carrying the scope it was written about.

    Deterministic on purpose -- no uuid, no wall clock -- so the same analysis
    produces the same memo ids and the artefacts diff cleanly between runs. It
    is also the memo's only machine-readable scope carrier: DecisionMemo is a
    fixed dataclass with no scope field, and memo_to_markdown/memo_to_dict read
    the scope back out of here rather than re-deriving it from prose.
    """
    return f"PULSE-{p.as_of:%Y%m%d}-P{rank}-{scope}-{scope_value}"


def scope_of(memo: DecisionMemo) -> tuple[str, str]:
    """(scope, scope_value) recovered from memo_id_for()'s format."""
    parts = memo.memo_id.split("-")
    return parts[3], "-".join(parts[4:])


# --- windows, limitations, status ------------------------------------------


def period(p: AnalysisParams) -> str:
    recent_start = p.as_of - timedelta(days=p.comparison_window_days - 1)
    baseline_end = recent_start - timedelta(days=1)
    baseline_start = baseline_end - timedelta(days=p.baseline_window_days - 1)
    return (
        f"Janela de comparação de {recent_start} a {p.as_of} "
        f"({p.comparison_window_days} dias), medida contra uma baseline "
        f"ajustada por dia da semana de {baseline_start} a {baseline_end} "
        f"({p.baseline_window_days} dias)."
    )


def limitations(p: AnalysisParams) -> tuple[str, ...]:
    """What this memo cannot tell you. Specific to how PULSE actually works --
    a generic disclaimer would be worth nothing to the person approving it.
    """
    recent_start = p.as_of - timedelta(days=p.comparison_window_days - 1)
    baseline_end = recent_start - timedelta(days=1)
    baseline_start = baseline_end - timedelta(days=p.baseline_window_days - 1)
    return (
        f"Todo número aqui é um piso, não uma estimativa do efeito completo. A "
        f"baseline de {p.baseline_window_days} dias ({baseline_start} a "
        f"{baseline_end}) é o período imediatamente anterior à janela de "
        f"comparação, então uma deterioração que já havia começado dentro dela "
        f"faz parte da baseline contra a qual está sendo medida. Isso deprime "
        f"a baseline e encolhe o desvio medido. Os números deste memorando "
        f'portanto subestimam o desvio associado, e por isso são escritos como '
        f'"no mínimo".',
        "Todo fator neste memorando é uma associação. Uma correlação diz que "
        "duas séries se moveram juntas na mesma janela; ela não estabelece que "
        "uma produziu a outra, e nenhum número aqui deve ser lido como um "
        "custo atribuível ao padrão diagnosticado. Afirmações causais exigem "
        "um experimento controlado, que é exatamente o que a seção Método de "
        "validação pede.",
        f"A data que este memorando reporta é onde a evidência começa dentro "
        f"da janela de comparação, não quando o incidente teve início. O "
        f"detector olha apenas os {p.comparison_window_days} dias encerrados "
        f"em {p.as_of}, então qualquer origem anterior a {recent_start} está "
        f"fora do que ele pode enxergar e a data reportada é limitada pela "
        f"janela, e não pelos dados.",
        f"A detecção é deliberadamente lenta diante de um incidente novo. O "
        f"detector exige que as duas metades da janela de comparação se movam "
        f"no mesmo sentido antes de disparar, o que suprime falsos positivos "
        f"de três dias de sorte mas significa que um incidente genuíno "
        f"começando hoje permanece invisível por cerca de "
        f"{p.comparison_window_days // 2} dias no comprimento de janela atual. "
        f"Ausência de anomalia não é evidência de saúde.",
        "Todos os dados aqui são sintéticos e gerados com semente fixa. Eles "
        "demonstram o método, a estatística e o caminho de decisão; não são "
        "uma afirmação sobre nenhum marketplace real.",
    )


# --- memo sections ---------------------------------------------------------


def _incident(d: Diagnosis, p: AnalysisParams) -> str:
    a = d.anomaly
    return (
        f"{_scope_phrase(a.scope, a.scope_value)}: {_label(a.metric)} está "
        f"{_num(abs(a.deviation_pct), 1)}% {_direction_word(a.deviation_abs)} "
        f"baseline ajustada por dia da semana nos {p.comparison_window_days} "
        f"dias até {p.as_of} ({_value(a.metric, a.recent_value)} contra "
        f"{_value(a.metric, a.baseline_value)}, z = "
        f"{_num(a.z_score, 2, plus=True)}, "
        f"{a.n_observations} dias observados). Primeiro sinal identificado "
        f"dentro da janela analisada: {a.first_detected_date}; é onde a "
        f"evidência começa dentro da janela, não uma afirmação sobre quando o "
        f"incidente teve início."
    )


def _concentration(d: Diagnosis) -> str:
    """Where the deviation sits, as a SHARE.

    Both branches matter. A share only exists for metrics whose segment values
    sum to the company value, and many diagnoses carry primary_segment=None --
    rates and durations get no decomposition by design, and a decomposition
    whose segments cancel out gets none either. The no-decomposition branch
    says so plainly rather than inventing a share or borrowing one, and carries
    no "%" at all: a percent sign in that sentence is a fabricated share.
    """
    a = d.anomaly
    segment = d.primary_segment
    if segment is None:
        return (
            f"Nenhuma decomposição por segmento está disponível para "
            f"{_label(a.metric)} aqui. Uma parcela de um desvio em nível de "
            f"empresa só faz sentido quando os valores dos segmentos somam o "
            f"valor da empresa, e nenhuma foi calculada para esta métrica "
            f"nesta janela, então este memorando não reporta concentração em "
            f"vez de inventar uma. O que está medido é o escopo em que a "
            f"anomalia disparou: {_scope_phrase(a.scope, a.scope_value)}."
        )

    others = [c for c in d.contributions if c.segment != segment.segment][:2]
    tail = (
        (
            " Próximos maiores: "
            + ", ".join(
                f"{_scope_phrase(c.dimension, c.segment)} "
                f"({_num(c.contribution_pct, 1, plus=True)}%)"
                for c in others
            )
            + "."
        )
        if others
        else ""
    )
    return (
        f"{_scope_phrase(segment.dimension, segment.segment)} responde por "
        f"{_num(segment.contribution_pct, 1)}% do desvio de {_label(a.metric)} "
        f"em nível de empresa nesta janela. Isso é uma parcela, não o total: "
        f"os {_num(100.0 - segment.contribution_pct, 1, plus=True)}% restantes "
        f"estão em outros "
        f"segmentos, e uma parcela negativa ali significa que um segmento se "
        f"moveu favoravelmente e mascarou parte do total.{tail}"
    )


def _driver_sentence(drivers: tuple[AssociatedDriver, ...]) -> str:
    """The associated series, phrased for what the coefficients actually are.

    Three branches, because one sentence cannot honestly cover all three cases:
    nothing measurable, a set of weak candidates that must NOT be described as
    deteriorating together (r = 0.12 on series that moved 0.6% is not evidence
    of anything), and a genuine cluster too tight to rank.
    """
    if not drivers:
        return (
            "Nenhum fator candidato atingiu uma correlação mensurável com esta "
            "métrica na janela analisada, então este memorando não nomeia "
            "nenhuma série associada, em vez de reportar uma série fraca como "
            "evidência."
        )
    listed = ", ".join(
        f"{_label(x.metric)} (r = {_num(x.correlation_with_target, 2, plus=True)}, "
        f"{x.evidence_strength})"
        for x in drivers
    )
    magnitudes = [abs(x.correlation_with_target) for x in drivers]
    spread = max(magnitudes) - min(magnitudes)

    if all(x.evidence_strength == "weak" for x in drivers):
        closing = (
            "Todas são fracas pelo limiar de reporte, portanto nenhuma é "
            "identificada como fator: estão listadas por completude, e não "
            "porque sustentam o diagnóstico. Esta assinatura ainda não tem "
            "associação utilizável por trás dela."
        )
    elif len(drivers) > 1 and spread <= _CLUSTER_SPREAD:
        closing = (
            f"Elas ficam a menos de {_num(spread, 2)} umas das outras em "
            f"correlação "
            f"absoluta -- faixa estreita demais para ranquear com qualquer "
            f"confiança -- portanto essas séries se deterioraram em conjunto e "
            f"nenhuma delas é identificada como o fator."
        )
    else:
        closing = (
            "Cada uma é uma série que se moveu junto com o alvo na mesma "
            "janela. Nenhuma é identificada como o fator: co-movimento é "
            "associação, e associação não estabelece direção."
        )
    return f"Séries associadas que se moveram com ela: {listed}. {closing}"


def _impact_sentence(impact: Impact, p: AnalysisParams) -> str:
    """The deviation, with the two units it is measured in kept apart.

    TWO DIFFERENT QUANTITIES, and they differ by a factor of the window length.
    daily_run_rate_brl is a per-DAY gap between the comparison window's daily
    average and the baseline's; gmv_at_risk_brl is that gap ACCUMULATED over the
    window's days. Writing them side by side without naming which is which is
    how a R$ 739 run rate and a R$ 10.351 window deviation read as a
    contradiction rather than as the same finding at two scales, so the sentence
    names the average, names the accumulation, and states the multiplication
    that connects them.
    """
    if impact.gmv_at_risk_brl > 0:
        return (
            f"O GMV MÉDIO DIÁRIO deste escopo ficou "
            f"{_signed_brl(impact.daily_run_rate_brl, 'por dia')} na janela de "
            f"comparação. ACUMULADO sobre os {p.comparison_window_days} dias da "
            f"janela, isso é um desvio estimado de no mínimo "
            f"{_brl(impact.gmv_at_risk_brl)} — a mesma medição multiplicada pelo "
            f"número de dias, e não uma segunda cifra — e projeta "
            f"{_brl(impact.projected_30d_brl)} "
            f"{_direction_word(impact.projected_30d_brl)} baseline em "
            f"{PROJECTION_DAYS} dias se o ritmo persistir. Um desvio observado, "
            f"não um custo, e um piso em vez da estimativa completa."
        )
    return (
        f"Nenhum GMV abaixo da baseline neste escopo na janela de comparação: o "
        f"GMV médio diário aqui corre a "
        f"{_signed_brl(impact.daily_run_rate_brl, 'por dia')}, portanto não há "
        f"desvio acumulado a reportar na janela. A anomalia disparou em uma "
        f"métrica cuja deterioração ainda não apareceu como dinheiro neste "
        f"escopo, e é por isso que ela ocupa a posição que ocupa."
    )


def _score_sentence(priority: Priority) -> str:
    """The impact score, written so it can be rebuilt by hand from this line.

    The opening words are EVIDENCE_ANCHOR_IMPACT_SCORE: app/streamlit_app.py
    selects this exact sentence out of memo.evidence by matching that same
    module-level constant, and the score is the engine's own named quantity.

    The confidence term is the GROUP's best-evidenced member, which is not
    always the confidence of the diagnosis this memo is written about: the
    member chosen to explain an incident and the member carrying the strongest
    evidence for it need not be the same one. Where the two differ, both are
    named -- a reader who spotted two different confidence numbers in one memo
    and found no explanation would be right to distrust the whole document.
    """
    b = priority.score_breakdown
    terms = " + ".join(
        f"{_num(b['w_' + name], 2)} x {_num(b['n_' + name], 3)} {name}"
        for name in ("gmv", "orders", "customers", "confidence")
    )
    note = ""
    if abs(b["n_confidence"] - priority.diagnosis.confidence) > 1e-12:
        note = (
            f" O termo de confiança é a evidência mais forte de todo este "
            f"grupo de escopo ({_num(b['n_confidence'], 3)}), enquanto a seção "
            f"Confiança reporta a do próprio diagnóstico "
            f"({_num(priority.diagnosis.confidence, 3)}). Eles diferem quando o "
            f"membro que melhor EXPLICA o incidente não é o membro com a "
            f"evidência estatística mais forte, e escolher a melhor explicação "
            f"não pode parecer evidência desaparecendo."
        )
    return (
        f"{EVIDENCE_ANCHOR_IMPACT_SCORE} {_num(priority.impact_score, 2)} = "
        f"100 x ({terms}), em que "
        f"cada componente de impacto é normalizado contra o maior candidato "
        f"desta execução e a confiança entra como medida. O score é "
        f"reconstruível à mão a partir desta linha.{note}"
    )


def _supporting_sentence(count: int) -> str:
    if count <= 1:
        return (
            "Uma métrica neste escopo disparou nesta janela. O impacto é "
            "calculado uma vez por escopo, e não uma vez por métrica, então um "
            "escopo que dispara em várias métricas não reivindica o mesmo "
            "dinheiro duas vezes."
        )
    return (
        f"{count} métricas independentes neste escopo dispararam na mesma "
        f"janela. Elas são tratadas como um único incidente com uma única "
        f"cifra de impacto, e não como {count} problemas separados "
        f"reivindicando o mesmo dinheiro."
    )


def _evidence(
    priority: Priority, rec: Recommendation, p: AnalysisParams
) -> tuple[str, ...]:
    """Everything the ranking and the recommendation actually rest on, in
    reading order. Computed from the Priority alone -- no figure in this memo
    is written by hand.
    """
    d = priority.diagnosis
    a = d.anomaly
    b = priority.score_breakdown
    lines = [
        f"{_sentence_case(_label(a.metric))} em "
        f"{_scope_phrase(a.scope, a.scope_value)}: "
        f"{_value(a.metric, a.recent_value)} contra uma baseline de "
        f"{_value(a.metric, a.baseline_value)} "
        f"({_num(a.deviation_pct, 1, plus=True)}%, "
        f"z = {_num(a.z_score, 2, plus=True)}) em {a.n_observations} dias "
        f"observados. As duas "
        f"metades da janela de comparação se movem no mesmo sentido, que é o "
        f"que o detector exige antes de disparar.",
        _supporting_sentence(int(b["supporting_anomalies"])),
    ]

    if d.funnel:
        broken = next((f for f in d.funnel if f.is_primary_break), None)
        stages = ", ".join(
            f"{_label(f.stage)} {_num(f.deviation_pct, 1, plus=True)}%"
            for f in d.funnel
        )
        lines.append(
            f"O GMV se decompõe exatamente como sessions x order_conversion x "
            f"completion_rate x aov. Nesta janela: {stages}. A maior "
            f"log-contribuição é {_label(broken.stage)}, portanto esse é o "
            f"estágio que se moveu -- calculado a partir da identidade, não "
            f"assumido."
        )
    else:
        lines.append(
            f"A identidade do funil decompõe o GMV, portanto não se aplica a "
            f"{_label(a.metric)}. O estágio levado para a classificação de "
            f"padrão é a própria métrica ({d.funnel_break_stage})."
        )

    lines.append(_driver_sentence(d.drivers))
    lines.append(_impact_sentence(priority.impact, p))

    if b.get("nested_groups_netted", 0):
        lines.append(
            f"Este é um escopo agregado, portanto sua reivindicação é "
            f"compensada contra as {int(b['nested_groups_netted'])} "
            f"reivindicações de escopo de segmento da mesma execução. Um total "
            f"de empresa contém seus segmentos por construção; sem essa "
            f"compensação o mesmo desvio seria contado nos dois níveis."
        )

    lines.append(_score_sentence(priority))
    # The opening words are EVIDENCE_ANCHOR_PLAYBOOK_ENTRY: app/streamlit_app.py
    # selects this exact sentence out of memo.evidence the same way it selects
    # the score sentence above -- see EVIDENCE_ANCHOR_PLAYBOOK_ENTRY's own
    # comment near the top of this module.
    lines.append(
        f"{EVIDENCE_ANCHOR_PLAYBOOK_ENTRY} `{rec.playbook_id}` (responsável: "
        f"{rec.owner_function}, "
        f"esforço: {rec.effort}) foi selecionada apenas pelo padrão "
        f"diagnosticado -- nenhuma zona, métrica ou valor de escopo participa "
        f"da busca. Sua justificativa: {rec.rationale}"
    )
    return tuple(lines)


# --- assembly --------------------------------------------------------------


def build_memo(
    priority: Priority,
    recommendation: Recommendation,
    params: AnalysisParams,
    narrator: Narrator | None = None,
) -> DecisionMemo:
    """One Priority + its Recommendation -> the memo a human approves.

    Assembled entirely from computed objects. The narrator, if any, is offered
    the FINISHED memo and may only fill `narrative`; it cannot reach any other
    field, which is what makes "the engine decides, the LLM narrates" a
    structural property rather than a promise.

    generated_at is derived from params.as_of rather than the wall clock. Two
    runs of the same analysis must produce byte-identical artefacts -- that is
    what lets the CLI output and a library call be compared at all, and what
    keeps artifacts/ diffable. The memo is a statement about an analysis date,
    not about the minute it happened to be rendered.
    """
    d = priority.diagnosis
    a = d.anomaly
    memo = DecisionMemo(
        memo_id=memo_id_for(priority.rank, a.scope, a.scope_value, params),
        incident=_incident(d, params),
        impact=priority.impact,
        concentration=_concentration(d),
        associated_drivers=d.drivers,
        evidence=_evidence(priority, recommendation, params),
        priority=priority.rank,
        recommended_action=recommendation.action,
        potential_result=recommendation.expected_effect,
        validation_method=recommendation.validation_method,
        confidence=d.confidence,
        generated_at=datetime.combine(params.as_of, time.min),
        params=params,
        narrative=None,
    )
    if narrator is None:
        return memo
    return replace(memo, narrative=narrator.narrate(memo))


def memo_to_dict(memo: DecisionMemo) -> dict:
    """The memo as plain data, plus the sections DecisionMemo has no field for.

    scope/scope_value/period/limitations/human_decision_status are DERIVED --
    deterministic functions of fields already on the memo -- so the dict stays
    identical between two runs of the same analysis, narrator or not. Dates and
    datetimes are left as date objects; json_default() renders them for the
    JSON artefact.
    """
    scope, scope_value = scope_of(memo)
    return {
        **asdict(memo),
        "scope": scope,
        "scope_value": scope_value,
        "period": period(memo.params),
        "limitations": list(limitations(memo.params)),
        "human_decision_status": DEFAULT_DECISION_STATUS,
    }


def json_default(obj):
    """`default=` for json.dumps over engine output.

    Two real leaks, both of which raise "Object of type X is not JSON
    serializable" without this: dataclasses.asdict(Anomaly) keeps
    first_detected_date as a datetime.date, and MetricFrame.headline() returns
    numpy scalars straight out of pandas. Everything else in the engine's
    output is already a Python builtin.
    """
    if hasattr(obj, "isoformat"):          # date / datetime
        return obj.isoformat()
    if hasattr(obj, "item"):               # numpy scalar
        return obj.item()
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


# --- markdown --------------------------------------------------------------


def _driver_table(drivers: tuple[AssociatedDriver, ...]) -> str:
    """Recent, baseline, deviation, correlation, strength.

    temporal_alignment_days is deliberately absent: on a sustained level shift
    the lag surface is nearly flat, so a lead/lag column would invite a reader
    to build an ordering story out of noise. The field stays in the data for
    anyone who wants it; it does not get a column here.
    """
    if not drivers:
        return (
            "_Nenhum fator candidato atingiu uma correlação mensurável nesta "
            "janela._"
        )
    header = (
        "| Série | Recente | Baseline | Desvio | Correlação | Evidência |\n"
        "|---|---|---|---|---|---|\n"
    )
    rows = "\n".join(
        f"| {_label(x.metric)} | {_value(x.metric, x.recent)} | "
        f"{_value(x.metric, x.baseline)} | "
        f"{_num(x.deviation_pct, 1, plus=True)}% | "
        f"{_num(x.correlation_with_target, 3, plus=True)} | "
        f"{x.evidence_strength} |"
        for x in drivers
    )
    return header + rows


def _impact_table(impact: Impact, p: AnalysisParams) -> str:
    """The impact fields, each row naming its own unit of time.

    Four of these are ACCUMULATED over the comparison window and two are PER
    DAY, and the rows that differ by a factor of the window length sit four
    lines apart. Every accumulated row therefore says "acumulado" and the
    run-rate row says "médio diário" -- the same distinction
    _impact_sentence makes in prose, made again here because a table is read
    without the prose.

    The customer row's LABEL is chosen by _customers_label(), not fixed: at an
    aggregate scope the figure is a residual left after the nested segment
    claims are removed, and it is not the scope's measured customer count.
    """
    window = f"acumulado na janela de {p.comparison_window_days} dias"
    rows = [
        ("Desvio de GMV, " + window, f"no mínimo {_brl(impact.gmv_at_risk_brl)}"),
        ("Pedidos abaixo da baseline, " + window, _num(impact.orders_lost)),
        (
            f"{_customers_label(impact)}, janela de "
            f"{p.comparison_window_days} dias",
            _num(impact.customers_affected),
        ),
        (
            "Desvio de margem de contribuição, " + window,
            f"no mínimo {_brl(impact.margin_impact_brl)}",
        ),
        (
            "Desvio do GMV médio diário (ritmo)",
            _signed_brl(impact.daily_run_rate_brl, "por dia"),
        ),
        (
            f"Ritmo diário projetado para {PROJECTION_DAYS} dias, acumulado",
            _signed_brl(impact.projected_30d_brl),
        ),
    ]
    return "| Medida | Estimativa |\n|---|---|\n" + "\n".join(
        f"| {name} | {value} |" for name, value in rows
    )


def _priority_sentence(rank: int) -> str:
    if rank == 1:
        return (
            "Ocupa a primeira posição desta execução de análise: a maior "
            "prioridade no cenário atual, por impacto medido e evidência "
            "disponível, não por identidade do escopo."
        )
    return f"Ocupa a posição {rank} desta execução de análise."


def memo_to_markdown(memo: DecisionMemo) -> str:
    """The memo as the document a human reads. Every figure comes off the memo
    object; nothing here is typed in.
    """
    scope, scope_value = scope_of(memo)
    sections = [
        f"# Resumo da decisão — Prioridade {memo.priority} — "
        f"{_scope_phrase(scope, scope_value)}",
        f"`{memo.memo_id}` · data da análise {memo.params.as_of} · "
        f"gerado em {memo.generated_at:%Y-%m-%d}",
        "## Incidente",
        memo.incident,
        "## Escopo",
        f"**{_scope_phrase(scope, scope_value)}** (dimensão `{scope}`, valor "
        f"`{scope_value}`). Impacto, evidências e recomendação abaixo estão "
        f"todos denominados neste escopo.",
        "## Período analisado",
        period(memo.params),
        "## Impacto estimado",
        _impact_table(memo.impact, memo.params),
        "Estimado e conservador: estes são desvios observados junto ao padrão "
        "diagnosticado, expressos como piso. Ver Limitações.",
        "## Concentração",
        memo.concentration,
        "## Fatores associados",
        _driver_table(memo.associated_drivers),
        _driver_sentence(memo.associated_drivers),
        "## Evidências",
        "\n".join(f"{i}. {line}" for i, line in enumerate(memo.evidence, start=1)),
        "## Prioridade",
        _priority_sentence(memo.priority),
        "## Ação recomendada",
        memo.recommended_action,
        "## Resultado potencial",
        memo.potential_result,
        "## Método de validação",
        memo.validation_method,
        "## Confiança",
        f"{_num(memo.confidence * 100)}% para o diagnóstico sobre o qual este "
        f"memorando "
        f"foi escrito — a parcela da evidência disponível que ele de fato "
        f"encontrou (força estatística, concentração por segmento, força de "
        f"associação e persistência). Evidência ausente subtrai; nada assume um "
        f"valor neutro por padrão. O termo de confiança dentro do impact score "
        f"pode ser maior: ver a linha de score na seção Evidências.",
        "## Limitações",
        "\n".join(f"- {line}" for line in limitations(memo.params)),
        "## Status da decisão humana",
        f"**{DEFAULT_DECISION_STATUS}** — nenhum humano revisou isto ainda. "
        f"{_DECISION_STATUS_NOTE} Estados disponíveis: "
        + ", ".join(DECISION_STATUSES)
        + ".",
    ]
    if memo.narrative:
        sections += [
            "## Narrativa (escrita por IA)",
            "> Escrita por um modelo de linguagem a partir do memorando "
            "finalizado acima. Ela não carrega fatos próprios: cada número, "
            "recomendação e status deste documento foi calculado antes de ela "
            "rodar.",
            memo.narrative,
        ]
    return "\n\n".join(sections).strip() + "\n"
