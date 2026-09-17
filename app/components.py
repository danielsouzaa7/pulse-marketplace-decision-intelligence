# Presentation only.
#
# THE RULE THIS FILE EXISTS TO KEEP: nothing here computes a business figure.
# Every number that reaches the screen arrives already computed on an engine
# object (a KPI row, an Anomaly, a Diagnosis, an Impact, a Priority, a memo) or
# in a gold column, and is only formatted, filtered, sorted, pivoted or drawn.
# There is no mean, no standard deviation, no group-by, no re-derivation of a
# deviation, a share, a contribution or a score anywhere in app/. `pulse decide`
# and these pages read the same objects, so they cannot disagree.
#
# What IS allowed here: choosing a colour, choosing an order, turning 0.745
# into "74,5%", and knowing that a rising cancellation rate should be drawn in
# red. That last one is the only judgement in this file, it lives in one
# frozenset, and it changes no number.
#
# EVERY USER-FACING WORD COMES FROM app/ui_text.py. The labels, the vocabulary
# maps and the Brazilian number formatters live there so there is one copy of
# each, and this module composes them into cards, tables and charts.
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# The engine's own unit register and decision states, imported rather than
# restated. A second copy would be a second source of truth for what a metric is
# measured in and what a human may decide. Only the RENDERING of a state is
# translated (ui_text.DECISION_STATUS_LABELS); the values stay English.
from pulse.decision_memo import DECISION_STATUSES, DEFAULT_DECISION_STATUS
from pulse.metrics import _UNITS as METRIC_UNITS
from pulse.metrics import (
    LOWER_IS_BETTER,
    METRIC_SEMANTICS,
    SCORE_CUSTOMERS_LABEL,
    customers_label,
)

from app import ui_text as T
from app.theme import ACCENT, CRITICAL, LINE, NEUTRAL, POSITIVE, TEXT

# Re-exported so every page reaches the one register through `ui.`, and adding a
# second literal anywhere is a diff a reviewer can see.
from app.ui_text import (  # noqa: F401
    COLUMN_LABELS,
    NOT_AVAILABLE,
    channel_label,
    decision_status_label,
    evidence_strength_label,
    fmt_brl,
    fmt_date,
    fmt_float,
    fmt_int,
    fmt_pct,
    fmt_pp,
    fmt_ratio,
    fmt_value,
    funded_by_label,
    headline_kpi_label,
    headline_label,
    metric_label,
    pattern_label,
    promo_type_label,
    scope_label,
)

# Plotly formats its own numbers inside hover boxes and on axis ticks. This is
# the one switch that makes those Brazilian too: decimal comma, thousands dot.
_BR_SEPARATORS = ",."

# The only judgement in this module: which direction of travel is bad news.
# It picks a colour and nothing else -- no ranking, no filtering, no number.
# The registered metrics come from the engine's register; the three discount
# columns are gold columns the register does not carry.
_LOWER_IS_BETTER = LOWER_IS_BETTER | frozenset(
    {
        "discount_rate",
        "discount_amount",
        "discount_per_completed_order",
    }
)


def unit_of(metric: str) -> str:
    """The engine's unit if it registers one, else the display unit."""
    return METRIC_UNITS.get(metric) or T.EXTRA_UNITS.get(metric, "")


def semantic_of(metric: str) -> str | None:
    """The engine's window-aggregate semantic, or None for a gold column the
    engine's register does not carry (a retention rate, a promo discount) --
    those are rendered as the single figure they are, never as a daily rate.
    """
    return METRIC_SEMANTICS.get(metric)


def fmt_metric_value(metric: str, value: float) -> str:
    """A WINDOW AGGREGATE of one metric, in its own unit and with its own
    aggregation marked.

    Every caller hands this a window mean -- a driver's recent/baseline come
    from the same _window_means the anomaly did -- so the engine's semantic is
    passed through and a per-day flow renders "/dia". Without it a driver table
    shows R$ 2.731,66 for a fourteen-day column and the reader supplies the
    missing word themselves, usually the wrong one.
    """
    return fmt_value(value, unit_of(metric), semantic_of(metric))


def tone(metric: str, delta_pct: float) -> str:
    """"pos" | "crit" | "flat" -- a CSS class, on sign alone.

    No magnitude threshold: inventing a "material change" band here would be a
    business judgement made in the presentation layer, which is exactly what
    min_materiality_brl already does in the engine, with a number a reviewer
    can see and change.
    """
    if delta_pct != delta_pct or delta_pct == 0:
        return "flat"
    improving = delta_pct < 0 if metric in _LOWER_IS_BETTER else delta_pct > 0
    return "pos" if improving else "crit"


def tone_colour(metric: str, delta_pct: float) -> str:
    return {"pos": POSITIVE, "crit": CRITICAL}.get(tone(metric, delta_pct), NEUTRAL)


# --- KPI rows --------------------------------------------------------------


def kpi_index(rows: tuple[dict, ...] | list[dict], scope: str, scope_value: str) -> dict:
    """metric -> KPI row, for one scope. A filter over already-computed rows."""
    return {
        row["metric"]: row
        for row in rows
        if row.get("scope", "company") == scope
        and str(row.get("scope_value", "all")) == str(scope_value)
    }


def kpi_card_html(row: dict, hero: bool = False) -> str:
    """One .headline() row as a card.

    Both the label and the value are rendered through the row's own `semantic`,
    so a card cannot show a daily average under a bare metric name. The engine
    put that key on the row; this reads it rather than deciding for itself.
    """
    delta = row["delta_pct"]
    semantic = row.get("semantic")
    return (
        f'<div class="pulse-kpi{" hero" if hero else ""}">'
        f'<div class="label">{headline_kpi_label(row)}</div>'
        f'<div class="value">{fmt_value(row["recent"], row["unit"], semantic)}</div>'
        f'<div class="delta tone-{tone(row["metric"], delta)}">{fmt_pct(delta)}'
        f'<span class="foot">vs. '
        f'{fmt_value(row["baseline"], row["unit"], semantic)} '
        f"na linha de base</span></div></div>"
    )


def fact_card_html(label: str, value: str, foot: str = "", tone_class: str = "flat",
                   hero: bool = False) -> str:
    """A KPI card for a figure that is not a KPI row -- an experiment's effect
    size, a quarantine count. Same card, same grid; the caller supplies an
    already-formatted string, so nothing is computed on the way in.
    """
    return (
        f'<div class="pulse-kpi{" hero" if hero else ""}">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="delta tone-{tone_class}">'
        f'<span class="foot">{foot}</span></div></div>'
    )


def kpi_row(rows: list[dict], per_row: int = 6, hero: bool = False) -> None:
    for start in range(0, len(rows), per_row):
        chunk = rows[start : start + per_row]
        columns = st.columns(per_row, gap="small")
        for column, row in zip(columns, chunk):
            column.markdown(kpi_card_html(row, hero=hero), unsafe_allow_html=True)


def card_row(cards: list[str], per_row: int = 4) -> None:
    """Pre-rendered cards laid out on the same grid as kpi_row."""
    for start in range(0, len(cards), per_row):
        chunk = cards[start : start + per_row]
        columns = st.columns(per_row, gap="small")
        for column, html in zip(columns, chunk):
            column.markdown(html, unsafe_allow_html=True)


# --- page chrome -----------------------------------------------------------


def section(eyebrow: str, title: str, note: str | None = None) -> None:
    st.markdown(
        f'<div style="margin-top:34px">'
        f'<div class="pulse-eyebrow">{eyebrow}</div>'
        f'<div class="pulse-h2">{title}</div>'
        + (f'<p class="pulse-note">{note}</p>' if note else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def page_head(group: str, question: str, note: str) -> None:
    """Every page leads with the question it answers, not with a theme.

    A page titled "Operações" invites browsing; a page titled "Onde a entrega
    está falhando, e quanto isso está custando?" can be answered or not.
    """
    st.markdown(
        f'<div class="pulse-eyebrow">{group}</div>'
        f'<div class="pulse-question">{question}</div>'
        f'<p class="pulse-note">{note}</p>',
        unsafe_allow_html=True,
    )
    notices()


def callout(text: str, quiet: bool = False) -> None:
    st.markdown(
        f'<div class="pulse-callout{" quiet" if quiet else ""}">{text}</div>',
        unsafe_allow_html=True,
    )


def notices() -> None:
    """The two standing notices. They are not decoration: PULSE recommends and
    a human decides, and every figure on the page comes from generated data.
    """
    st.markdown(
        '<div style="margin-top:10px">'
        '<span class="pulse-notice"><span class="dot"></span>'
        f"{T.NOTICE_DECISION_SUPPORT}</span>"
        '<span class="pulse-notice"><span class="dot syn"></span>'
        f"{T.NOTICE_SYNTHETIC_DATA}</span></div>",
        unsafe_allow_html=True,
    )


def status_pill(label: str, state: str = "info") -> str:
    """A small labelled pill, returned as HTML so a caller can render a row of
    them in one markdown call. `state` is one of ok | warn | crit | info and
    picks a colour; it changes no number and decides nothing.
    """
    kind = state if state in ("ok", "warn", "crit", "info") else "info"
    return f'<span class="pulse-pill {kind}"><span class="dot"></span>{label}</span>'


def validation_pills(validation) -> tuple[str, ...]:
    """A Copilot answer's verification state, as pills.

    NO STATE IS GREEN. The validator reads Portuguese with patterns: when it
    finds a divergence that is a fact worth a red pill, but when it finds none
    that establishes nothing about wording it has no pattern for -- three review
    rounds each found new phrasings of a false claim that passed. A green "all
    figures verified" turned every one of those into a false success, so a clean
    run is reported as what it is: no divergence detected. Nothing checked is
    said as such, and a causal claim gets its own red pill whatever the figures
    did.
    """
    unverified = len(validation.unverified_figures)
    checked = validation.figures_checked
    if unverified:
        out = [status_pill(
            f"{fmt_int(unverified)} de {fmt_int(checked)} figuras NÃO ligadas à "
            f"afirmação em que aparecem", "crit")]
    elif checked == 0:
        out = [status_pill("Nenhuma figura numérica para verificar", "info")]
    else:
        out = [status_pill(
            f"Nenhuma divergência detectada nas {fmt_int(checked)} figuras "
            f"(verificação por padrões, não é prova)", "info")]
    if validation.causal_claims:
        count = len(validation.causal_claims)
        words = ("afirmação causal não sustentada" if count == 1
                 else "afirmações causais não sustentadas")
        out.append(status_pill(f"{fmt_int(count)} {words} pela evidência", "crit"))
    return tuple(out)


def pills(*html: str) -> None:
    st.markdown("".join(html), unsafe_allow_html=True)


# --- human in the loop ------------------------------------------------------


def decision_state_control(memo_id: str) -> str:
    """The human decision on one memo. Records; executes nothing.

    Writes a state and an audit line into st.session_state and returns the
    state. There is no request, no process, no file and no marketplace call
    anywhere on this path -- PULSE recommends, a named human decides, and the
    decision lives in this browser session until someone acts on it elsewhere.

    The state STORED is the engine's own English value; only the button label is
    Portuguese, so what a reader approves and what a memo records are one thing.
    """
    key = f"decision_status::{memo_id}"
    if key not in st.session_state:
        st.session_state[key] = DEFAULT_DECISION_STATUS
    st.segmented_control(
        "Status da decisão",
        options=list(DECISION_STATUSES),
        format_func=decision_status_label,
        key=key,
    )
    status = st.session_state[key] or DEFAULT_DECISION_STATUS

    audit: list[dict] = st.session_state.setdefault("decision_audit", [])
    if not audit or (audit[-1]["memo_id"], audit[-1]["state"]) != (memo_id, status):
        audit.append(
            {
                "memo_id": memo_id,
                "state": status,
                "at": datetime.now().isoformat(timespec="seconds"),
            }
        )
    st.caption(
        f"Trilha de auditoria da sessão: {len(audit)} mudança(s) de estado "
        f"registrada(s). Última — {audit[-1]['memo_id']} definido como "
        f"{decision_status_label(audit[-1]['state'])} em {audit[-1]['at']}."
    )
    return status


# --- priorities ------------------------------------------------------------


def priority_card_html(priority: Any) -> str:
    anomaly = priority.diagnosis.anomaly
    breakdown = priority.score_breakdown
    supporting = int(breakdown["supporting_anomalies"])
    exposure = priority.impact.projected_30d_brl
    return (
        f'<div class="pulse-prio{" lead" if priority.rank == 1 else ""}">'
        f'<div class="rank">Prioridade #{priority.rank}</div>'
        f'<div class="scope">{scope_label(anomaly.scope, anomaly.scope_value)}</div>'
        f'<div class="pattern">{pattern_label(priority.diagnosis.pattern)}</div>'
        f'<div class="row"><span>Pontuação de impacto</span>'
        f"<b>{fmt_float(priority.impact_score)}</b></div>"
        f'<div class="row"><span>Exposição estimada em 30 dias</span>'
        f"<b>{fmt_brl(exposure)} "
        f"{'abaixo' if exposure < 0 else 'acima'}</b></div>"
        f'<div class="row"><span>Anomalias que sustentam</span>'
        f"<b>{supporting}</b></div></div>"
    )


def priority_label(priority: Any) -> str:
    anomaly = priority.diagnosis.anomaly
    return (
        f"#{priority.rank}  {scope_label(anomaly.scope, anomaly.scope_value)}"
        f"  ·  {pattern_label(priority.diagnosis.pattern)}"
    )


def priority_rows(priorities: tuple[Any, ...] | list[Any]) -> list[dict]:
    """The ranked list as a table, in the engine's own order.

    Whatever the engine ranked first is first here. No scope is named in this
    module, in a heading, or in a template anywhere in app/ -- the ranking is
    close enough between runs that a hard-coded scope would eventually be a lie.
    """
    return [
        {
            "Posição": p.rank,
            "Escopo": scope_label(
                p.diagnosis.anomaly.scope, p.diagnosis.anomaly.scope_value
            ),
            "Métrica": metric_label(p.diagnosis.anomaly.metric),
            "Padrão": pattern_label(p.diagnosis.pattern),
            "Pontuação de impacto": fmt_float(p.impact_score),
            "Exposição em 30 dias": (
                f"{fmt_brl(p.impact.projected_30d_brl)} "
                f"{'abaixo' if p.impact.projected_30d_brl < 0 else 'acima'}"
            ),
            "Confiança": fmt_float(p.diagnosis.confidence, 3),
        }
        for p in priorities
    ]


def ranking_note(priorities: tuple[Any, ...] | list[Any]) -> str:
    """How far apart the top two actually are, as the engine's own number.

    NO WORD FOR THE SIZE OF THE GAP. This sentence used to call the margin
    "estreita" from a fixed template, which was written when the top two sat a
    fraction of a point apart and was still being printed beside a 22-point
    gap -- a claim that had stopped being true and that nothing would have
    caught, because no policy anywhere in PULSE defines how wide a gap has to be.
    The distance is stated in points and the reader judges it. Any qualitative
    adjective for the size of a gap would need a tested classification policy
    before it could be said, and there is no reason to introduce one.

    The gap itself comes off score_breakdown["score_gap_to_next"], computed in
    pulse.prioritization: subtracting two displayed scores here would be this
    module deriving a business figure, which is the one thing it may not do.
    """
    if len(priorities) < 2:
        return (
            "Apenas uma prioridade superou os limiares nesta execução, portanto "
            "não há diferença de ordenação a relatar."
        )
    lead, second = priorities[0], priorities[1]
    where = scope_label(
        lead.diagnosis.anomaly.scope, lead.diagnosis.anomaly.scope_value
    )
    gap = lead.score_breakdown["score_gap_to_next"]
    return (
        f"<b>{where}</b> ocupa a primeira posição nesta execução, com "
        f"{fmt_float(lead.impact_score)} de 100 contra "
        f"{fmt_float(second.impact_score)} do item seguinte — uma distância de "
        f"<b>{fmt_float(gap)} pontos</b>. A classificação é uma ordenação de "
        f"onde olhar primeiro, não uma afirmação de que o restante é ruído: ele "
        f"tem a maior prioridade no cenário atual, e nada além disso."
    )


# Patterns whose diagnosed signature is itself a post-checkout operational
# story, and the ONLY ones for which a page may say so. Keyed on the engine's
# pattern identifier, so a pattern added to the playbook gets no explanatory
# sentence until it is added here deliberately.
#
# WHY THIS DICT EXISTS. The Decision Intelligence page used to render one
# hard-coded fulfilment mechanism for EVERY priority, whatever the engine had
# diagnosed -- a sentence asserting that a worse post-checkout experience is
# more expensive to serve and that margin therefore falls faster than revenue.
# On a supply-availability priority whose own driver table on the same page
# showed fulfilment times IMPROVING, the page asserted it anyway. A mechanism is
# a claim about WHY, and PULSE's whole discipline is that it does not have one
# outside the randomised experiment.
#
# What is allowed here is a statement of CONSISTENCY with the diagnosed
# signature -- "consistent with operational deterioration after checkout" -- and
# nothing that names a cause or an effect.
_PATTERN_CONSISTENCY: dict[str, str] = {
    "fulfillment_eta_degradation": (
        "Os sinais são consistentes com deterioração operacional após o "
        "checkout, que é o padrão que o motor diagnosticou aqui."
    ),
}


def margin_gmv_note(margin_row: dict | None, gmv_row: dict | None,
                    pattern: str) -> str | None:
    """What the margin and GMV rows OBSERVABLY did, and nothing about why.

    Returns None when either row is absent, so the caller renders nothing rather
    than a sentence with a gap in it.

    The first half is arithmetic-free: which of two already-computed deviations
    is the lower one, stated as the observation it is. The second half is added
    only for a pattern _PATTERN_CONSISTENCY lists, and even then says
    "consistent with", never "caused".
    """
    if not margin_row or not gmv_row:
        return None
    margin_pct, gmv_pct = margin_row["delta_pct"], gmv_row["delta_pct"]
    if margin_pct < gmv_pct:
        observed = (
            f"<b>A margem de contribuição caiu mais do que o GMV neste "
            f"recorte.</b> A margem fechou em {fmt_pct(margin_pct)} contra "
            f"{fmt_pct(gmv_pct)} do GMV na mesma janela. Isso é o que as duas "
            f"séries fizeram; nada aqui diz por quê."
        )
    else:
        observed = (
            f"<b>A margem de contribuição não caiu mais do que o GMV neste "
            f"recorte.</b> A margem fechou em {fmt_pct(margin_pct)} contra "
            f"{fmt_pct(gmv_pct)} do GMV na mesma janela."
        )
    consistency = _PATTERN_CONSISTENCY.get(pattern)
    return f"{observed} {consistency}" if consistency else observed


def segment_line(primary_segment: Any) -> str:
    """Where the deviation sits -- or a straight statement that this anomaly
    has no segment decomposition.

    primary_segment is None for most diagnoses on this dataset (segment shares
    are only defined for additive metrics, so a rate or duration anomaly gets
    none by design). Saying so is the correct rendering; inventing a segment or
    printing "None" are the two ways to get it wrong.
    """
    if primary_segment is None:
        return (
            "Sem decomposição por segmento para esta anomalia. A participação de "
            "um segmento só é definida para métricas aditivas (moeda, pedidos, "
            "sessões) — uma anomalia de taxa ou de duração não recebe nenhuma, em "
            "vez de receber uma errada."
        )
    where = scope_label(primary_segment.dimension, primary_segment.segment)
    return (
        f"<b>{where}</b> concentra a maior contribuição observada: "
        f"<b>{fmt_float(primary_segment.contribution_pct)}%</b> do desvio "
        f"apurado no nível da empresa nesta janela. Isso é uma participação, não "
        f"o total — o restante está em outros segmentos, e uma participação "
        f"negativa ali significa que um segmento se moveu favoravelmente e "
        f"mascarou parte do total."
    )


def confidence_note(priority: Any) -> str:
    """Both confidence figures, named. They legitimately differ and an
    unexplained divergence reads as a bug.
    """
    breakdown = priority.score_breakdown
    return (
        f"Duas medidas de confiança se aplicam e ambas estão corretas. A "
        f"pontuação usa a evidência mais forte do <b>grupo</b>, "
        f"<b>{fmt_float(breakdown['n_confidence'], 3)}</b> — a mais bem "
        f"evidenciada entre as {int(breakdown['supporting_anomalies'])} anomalias "
        f"que dispararam neste escopo. O diagnóstico mostrado abaixo é o membro "
        f"que <i>explica</i> mais, e a confiança dele é "
        f"<b>{fmt_float(breakdown['representative_confidence'], 3)}</b>. As duas "
        f"divergem quando a melhor explicação não é a estatística isolada mais "
        f"forte; escolher a melhor explicação não pode ser lido como evidência "
        f"desaparecendo."
    )


# --- tables ----------------------------------------------------------------


def driver_rows(drivers: tuple[Any, ...]) -> list[dict]:
    """Associated drivers as display rows, in the engine's own ranked order.

    "Direcionador associado" and "Defasagem", never "cause" and "delay before
    the effect": the coefficient is a correlation and the alignment is the lag
    that maximises it, which is evidence and not a direction of causation.
    """
    return [
        {
            "Direcionador associado": metric_label(driver.metric),
            "Recente": fmt_metric_value(driver.metric, driver.recent),
            "Linha de base": fmt_metric_value(driver.metric, driver.baseline),
            "Variação": fmt_pct(driver.deviation_pct, places=1),
            "Correlação r": fmt_float(driver.correlation_with_target, 4, signed=True),
            "Defasagem": f"{driver.temporal_alignment_days:+d} d",
            "Evidência": evidence_strength_label(driver.evidence_strength),
        }
        for driver in drivers
    ]


_SCORE_COMPONENTS = (
    ("n_gmv", "w_gmv", "GMV em risco (ritmo diário projetado em 30 dias)"),
    ("n_orders", "w_orders", "Pedidos abaixo da baseline na janela"),
    # Not "clientes afetados": on an aggregate scope this term is the residual
    # left after the nested segment claims are netted out, and "afetados" reads
    # as a measured population. SCORE_CUSTOMERS_LABEL is the engine's own wording
    # for the distinction -- see metrics.customers_label.
    ("n_customers", "w_customers", SCORE_CUSTOMERS_LABEL),
    ("n_confidence", "w_confidence", "Confiança da evidência (grupo)"),
)


def score_rows(breakdown: dict) -> list[dict]:
    """The score's inputs as the engine returned them: each weight beside the
    normalised component it multiplies. Deliberately NOT multiplied out here --
    the engine already wrote that arithmetic into the memo's evidence line, and
    doing it again on the page would be a second implementation of the score.
    """
    return [
        {
            "Componente": label,
            "Peso": fmt_float(breakdown[weight_key]),
            "Normalizado (0-1)": fmt_float(breakdown[value_key], 3),
        }
        for value_key, weight_key, label in _SCORE_COMPONENTS
    ]


def impact_rows(impact: Any, comparison_window_days: int) -> list[dict]:
    """Estimated, conservative, and labelled as both: every row is worded "no
    mínimo", because the baseline the gap is measured against already contains
    part of the deterioration.

    TWO UNITS OF TIME, NAMED. Four rows are ACCUMULATED over the comparison
    window and two are PER DAY, and they differ by a factor of the window
    length: a R$ 739 daily gap and a R$ 10.351 window deviation are the same
    measurement, and a table that labels neither invites the reader to treat them
    as two findings that disagree.

    The customer row's label comes from metrics.customers_label(impact), not from
    a literal here: at an aggregate scope that figure is a residual left after
    the nested segment claims are netted out, not the scope's measured customer
    count, and only the Impact knows which it is.
    """
    direction = "abaixo" if impact.daily_run_rate_brl < 0 else "acima"
    window = f"acumulado na janela de {comparison_window_days} dias"
    return [
        {
            "Estimativa": f"Desvio de GMV, {window}",
            "Valor": f"no mínimo {fmt_brl(impact.gmv_at_risk_brl)}",
        },
        {
            "Estimativa": f"Pedidos abaixo da baseline, {window}",
            "Valor": f"no mínimo {fmt_int(impact.orders_lost)}",
        },
        {
            "Estimativa": f"{customers_label(impact)}, janela de "
                          f"{comparison_window_days} dias",
            "Valor": fmt_int(impact.customers_affected),
        },
        {
            "Estimativa": f"Desvio de margem de contribuição, {window}",
            "Valor": f"no mínimo {fmt_brl(impact.margin_impact_brl)}",
        },
        {
            "Estimativa": "Desvio do GMV médio diário (ritmo)",
            "Valor": f"{fmt_brl(impact.daily_run_rate_brl)} por dia {direction} "
                     f"da linha de base",
        },
        {
            "Estimativa": "Ritmo diário projetado para 30 dias, acumulado",
            "Valor": (
                f"{fmt_brl(impact.projected_30d_brl)} "
                f"{'abaixo' if impact.projected_30d_brl < 0 else 'acima'} "
                f"da linha de base"
            ),
        },
    ]


def localise_dates(frame: pd.DataFrame) -> pd.DataFrame:
    """A COPY of the frame with its date columns written 10/09/2026.

    A presentation copy, always: an analytical frame is never localised in
    place, because a date turned into a string is a date no longer orderable by
    anything that reads it afterwards.
    """
    dated = [
        column for column in frame.columns
        if pd.api.types.is_datetime64_any_dtype(frame[column])
    ]
    if not dated:
        return frame
    out = frame.copy()
    for column in dated:
        out[column] = out[column].dt.strftime("%d/%m/%Y")
    return out


def table(rows: list[dict] | pd.DataFrame, empty_note: str = "") -> None:
    empty = rows.empty if isinstance(rows, pd.DataFrame) else not rows
    if empty:
        if empty_note:
            st.markdown(f'<p class="pulse-note">{empty_note}</p>',
                        unsafe_allow_html=True)
        return
    st.dataframe(
        localise_dates(rows) if isinstance(rows, pd.DataFrame) else rows,
        hide_index=True,
    )


def as_embedded_markdown(markdown: str) -> str:
    """The memo, adjusted for living inside a page rather than being the page.

    Headings drop two levels so the memo's own H1 does not outrank the section
    titles around it, and dollars are escaped (see escape_dollars). Display
    only: the download button hands over the engine's bytes untouched, so the
    file matches artifacts/decision-memo.md.
    """
    demoted = [
        "##" + line if line.startswith("#") else line
        for line in markdown.splitlines()
    ]
    return escape_dollars("\n".join(demoted))


def escape_dollars(markdown: str) -> str:
    """Streamlit renders $...$ as LaTeX, which swallows "R$ 4.829,10 contra
    R$ 5.454,00" into a maths span. Escaping is a display concern only: the
    download button hands over the engine's exact bytes, unescaped, so the
    file matches artifacts/decision-memo.md character for character.
    """
    return markdown.replace("$", r"\$")


# --- charts ----------------------------------------------------------------
#
# Each one carries an argument the text cannot make as quickly. Every value
# plotted is read straight off an engine object or off a gold column; the only
# transformations are selection, ordering, pivoting and unit conversion.
#
# Localisation happens through Plotly's own presentation hooks -- `labels=`,
# `title=`, `hovertemplate`, axis titles -- and through the `separators` set in
# chart(), which is what turns 18759.17 into 18.759,17 inside a hover box. No
# canonical frame is renamed for display.

# "2026-03": a month-only string, the exact shape gold_customer_retention's
# cohort_month carries (strftime('%Y-%m', ...) in SQL -- never a real date
# column). Plotly Express still auto-detects a column shaped like this as a
# date axis and, with no explicit tickformat, falls back to its own default
# ("Mar 2026") -- English, on an otherwise Portuguese page. _localise_date_axis
# below checks for this shape as well as for a true datetime64 column, so
# every chart that routes through it gets a Portuguese-safe axis regardless of
# which of the two shapes gold happens to store the date-like column in.
_MONTH_STRING = re.compile(r"^\d{4}-\d{2}$")


def _localise_date_axis(figure: go.Figure, values: Any) -> None:
    """dd/mm on the ticks and dd/mm/yyyy in the hover, when the axis is a real
    date; mm/yyyy on both, when the axis is a month-only string ("2026-03")
    that Plotly auto-parses into a date axis of its own accord. Neither branch
    touches `values` -- only the figure's display format, never the frame the
    caller pivoted or plotted from.

    The month check does not gate on dtype: pandas' own string dtype varies by
    version (plain "object" on older pandas, a dedicated StringDtype on newer
    ones), so asking "is every value a two-part YYYY-MM string" directly is
    both simpler and version-proof.
    """
    index = pd.Index(values)
    if pd.api.types.is_datetime64_any_dtype(index):
        figure.update_xaxes(tickformat="%d/%m", hoverformat="%d/%m/%Y")
    elif len(index) and all(
        isinstance(v, str) and _MONTH_STRING.match(v) for v in index
    ):
        figure.update_xaxes(tickformat="%m/%Y", hoverformat="%m/%Y")


def scope_delta_chart(
    index: dict, metrics: tuple[str, ...], title: str
) -> go.Figure | None:
    """Horizontal bars of each metric's own % change at one scope.

    This is the placed-vs-completed picture: order placement and order
    completion side by side, in the same scope, over the same window.
    """
    present = [m for m in metrics if m in index]
    if not present:
        return None
    rows = [index[m] for m in present]
    figure = go.Figure(
        go.Bar(
            x=[row["delta_pct"] for row in rows],
            y=[metric_label(row["metric"]) for row in rows],
            orientation="h",
            marker=dict(
                color=[tone_colour(row["metric"], row["delta_pct"]) for row in rows]
            ),
            text=[fmt_pct(row["delta_pct"], places=1) for row in rows],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:+.2f}% vs. linha de base<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title="variação % vs. linha de base",
        yaxis=dict(autorange="reversed", automargin=True),
        height=64 + 38 * len(rows),
        margin=dict(l=8, r=78, t=44, b=34),
    )
    return figure


def funnel_waterfall(funnel: tuple[Any, ...]) -> go.Figure | None:
    """The log decomposition of the GMV change, stage by stage.

    GMV = sessions x conversion x completion rate x AOV, so in logs the four
    terms add up to the GMV change and the largest-magnitude bar IS the stage
    that moved. The bars are the engine's log_contribution values; the labels
    are each stage's own percentage change.
    """
    if not funnel:
        return None
    figure = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["relative"] * len(funnel) + ["total"],
            x=[metric_label(stage.stage) for stage in funnel]
              + ["GMV (variação em log)"],
            y=[stage.log_contribution for stage in funnel] + [0.0],
            text=[fmt_pct(stage.deviation_pct, places=1) for stage in funnel] + [""],
            textposition="outside",
            connector=dict(line=dict(color=LINE, width=1)),
            increasing=dict(marker=dict(color=POSITIVE)),
            decreasing=dict(marker=dict(color=CRITICAL)),
            totals=dict(marker=dict(color=ACCENT)),
            hovertemplate="%{x}: contribuição em log %{y:+.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Decomposição do funil — contribuição aditiva em log para a "
              "variação do GMV",
        yaxis=dict(title="contribuição em log", automargin=True),
        xaxis=dict(automargin=True),
        height=400,
        margin=dict(l=8, r=8, t=44, b=60),
    )
    return figure


def contribution_chart(contributions: tuple[Any, ...]) -> go.Figure | None:
    """Each segment's share of the company-level deviation, in rank order.

    Drawn as shares precisely so the leading segment cannot be read as the
    whole of it: the other bars are on the same axis, and the negative ones are
    segments that moved favourably and offset part of the total.
    """
    if not contributions:
        return None
    ordered = sorted(contributions, key=lambda c: c.rank)
    colours = [
        ACCENT
        if c.rank == 1
        else (POSITIVE if c.contribution_pct < 0 else "#33507F")
        for c in ordered
    ]
    figure = go.Figure(
        go.Bar(
            x=[c.contribution_pct for c in ordered],
            # The share rides on the category label rather than on an outside
            # data label: an offsetting segment's bar is a couple of pixels
            # wide, and its label lands on top of the axis text.
            y=[
                f"{scope_label(c.dimension, c.segment)}  "
                f"{fmt_float(c.contribution_pct, 1)}%"
                for c in ordered
            ],
            orientation="h",
            marker=dict(color=colours),
            customdata=[c.segment_deviation_abs for c in ordered],
            hovertemplate=(
                "%{y}: %{x:.2f}% do desvio da empresa "
                "(R$ %{customdata:,.2f})<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title="Participação no desvio de GMV da empresa, por segmento",
        xaxis_title="% do desvio da empresa (negativo = compensando)",
        yaxis=dict(autorange="reversed", automargin=True),
        height=64 + 32 * len(ordered),
        margin=dict(l=8, r=78, t=44, b=34),
    )
    return figure


def heatmap_chart(
    pivoted: pd.DataFrame, metric: str, title: str, height: int | None = None
) -> go.Figure | None:
    """An already-pivoted grid, drawn.

    The caller pivots (a reshape of rows gold already carries, never an
    aggregation) and this draws it. The colour scale comes from the registered
    template; the only decision made here is `reversescale`, which reads off
    _LOWER_IS_BETTER so a metric whose high end is the bad end is not drawn
    green at its worst.
    """
    if pivoted is None or pivoted.empty:
        return None
    ratio = unit_of(metric) == "ratio"
    figure = go.Figure(
        go.Heatmap(
            z=pivoted.values,
            x=list(pivoted.columns),
            y=[str(value) for value in pivoted.index],
            reversescale=metric in _LOWER_IS_BETTER,
            hovertemplate=(
                "%{y} · %{x}<br>"
                + metric_label(metric)
                + (": %{z:.1%}" if ratio else ": %{z:,.2f}")
                + "<extra></extra>"
            ),
            colorbar=dict(title=dict(text=metric_label(metric), side="right")),
            hoverongaps=False,
        )
    )
    figure.update_layout(
        title=title,
        height=height or (140 + 22 * len(pivoted.index)),
        yaxis=dict(autorange="reversed", automargin=True, type="category"),
        xaxis=dict(automargin=True),
        margin=dict(l=8, r=8, t=44, b=40),
    )
    _localise_date_axis(figure, pivoted.columns)
    return figure


def timeline_chart(
    frame: pd.DataFrame, x: str, y: str, colour: str, title: str,
    y_title: str | None = None, height: int = 380,
) -> go.Figure | None:
    """One line per value of `colour`, plotted as the rows stand.

    No aggregation: every point is a row of a gold table. If two rows shared an
    (x, colour) pair the line would zig-zag between them, which is the correct
    way for a duplicated grain to look.
    """
    if frame is None or frame.empty:
        return None
    figure = px.line(
        frame, x=x, y=y, color=colour, markers=len(frame) < 200,
        labels=COLUMN_LABELS,
    )
    figure.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title=y_title or metric_label(y),
        legend_title_text="",
        height=height,
        margin=dict(l=8, r=8, t=56, b=40),
    )
    _localise_date_axis(figure, frame[x])
    return figure


def scatter_chart(
    frame: pd.DataFrame, x: str, y: str, colour: str, title: str,
    size: str | None = None, hover: tuple[str, ...] = (),
    x_title: str | None = None, y_title: str | None = None, height: int = 460,
) -> go.Figure | None:
    """Each point is one row of a gold table. Nothing is binned or averaged."""
    if frame is None or frame.empty:
        return None
    figure = px.scatter(
        frame, x=x, y=y, color=colour, size=size, hover_data=list(hover),
        labels=COLUMN_LABELS, size_max=20, opacity=0.82,
    )
    figure.update_layout(
        title=title,
        xaxis_title=x_title or metric_label(x),
        yaxis_title=y_title or metric_label(y),
        legend_title_text="",
        height=height,
        margin=dict(l=8, r=8, t=56, b=40),
    )
    return figure


def interval_chart(
    point: float, low: float, high: float, title: str, x_title: str
) -> go.Figure:
    """One effect estimate and its confidence interval, against zero.

    The whole read of a null result is visual: does the interval cross the zero
    line. The three values are drawn exactly as the analyser returned them,
    scaled from fractions to points the same way fmt_pp scales them for text.
    """
    figure = go.Figure()
    figure.add_shape(
        type="line", x0=0, x1=0, y0=-0.5, y1=0.5,
        line=dict(color=CRITICAL, width=1.5, dash="dash"),
    )
    figure.add_trace(
        go.Scatter(
            x=[low * 100.0, high * 100.0], y=[0, 0], mode="lines",
            line=dict(color=ACCENT, width=6), name="Intervalo de confiança de 95%",
            hovertemplate="Limite do intervalo: %{x:+.2f} p.p.<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[point * 100.0], y=[0], mode="markers",
            marker=dict(color=TEXT, size=15, line=dict(color=ACCENT, width=2)),
            name="Efeito observado",
            hovertemplate="Efeito observado: %{x:+.2f} p.p.<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis=dict(visible=False, range=[-0.5, 0.5]),
        height=230,
        showlegend=True,
        margin=dict(l=8, r=8, t=56, b=40),
    )
    return figure


def chart(figure: go.Figure | None, empty_note: str = "") -> None:
    """Render with the app's own template and Brazilian number separators.

    theme=None is load-bearing: Streamlit's default theme="streamlit" overrides
    the registered template. `separators` is set here rather than per chart so
    no figure anywhere can be drawn with a foreign decimal point.
    """
    if figure is None:
        if empty_note:
            st.markdown(f'<p class="pulse-note">{empty_note}</p>', unsafe_allow_html=True)
        return
    figure.update_layout(separators=_BR_SEPARATORS)
    st.plotly_chart(figure, theme=None)
