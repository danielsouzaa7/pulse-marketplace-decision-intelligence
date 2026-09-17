# Every label, vocabulary map and display formatter the PULSE surface uses, in
# one module.
#
# THE BOUNDARY THIS FILE IS: the engine speaks in identifiers -- `completion_rate`,
# `zone`, `INVESTIGATE`, `fulfillment_eta_degradation` -- and the product speaks
# Brazilian Portuguese. The mapping between the two lives here and nowhere else,
# with one deliberate exception: SCOPE_LABELS below is imported from
# pulse.metrics rather than authored here, because the engine's own prose
# (decision_memo's evidence lines, the playbook's rationale, the Copilot's
# offline answer) needs the identical Portuguese word for a scope, not a
# second copy that could drift from this page's. Nothing upstream of this
# module is renamed: a metric key, a scope name, a decision state and a gold
# column keep the names the engine, the SQL and the tests use, and only their
# *rendering* is translated.
#
# NOTHING HERE COMPUTES. The formatters convert units for display (0.745 ->
# "74,5%", 0.0105 -> "+1,05 p.p.") and switch a decimal comma for a decimal
# point. They are called at render time, on a value the engine already produced,
# and they never write back into a frame an analysis will read.
#
# This is deliberately NOT an i18n framework. There is one language, so there is
# one module of plain dicts and plain functions -- no catalogue loader, no
# locale negotiation, no message ids.
from __future__ import annotations

from typing import Any

from pulse.metrics import DAILY_AVERAGE
from pulse.metrics import _METRIC_LABELS_PT as ENGINE_METRIC_LABELS
from pulse.metrics import _SCOPE_LABELS as SCOPE_LABELS
from pulse.metrics import semantic_of

# --- the register ------------------------------------------------------------
#
# Every metric the engine publishes, every gold column a page puts on screen,
# and the experiment's own primary metric. A key missing from here renders as
# its identifier, which is a visible bug rather than a silent one.

METRIC_LABELS: dict[str, str] = {
    # The engine's own register, imported rather than retyped: pulse.copilot's
    # claim validator has to recognise which metric a Portuguese sentence is
    # making a claim about, so the engine needs these words too and a second
    # copy here is how the page and the validator come to disagree about what
    # "Taxa de conclusão" refers to.
    **ENGINE_METRIC_LABELS,
    # gold columns the engine's register does not carry
    "orders_cancelled": "Pedidos cancelados",
    "available_hours": "Horas disponíveis",
    "scheduled_open_hours": "Horas programadas de funcionamento",
    "retention_rate": "Taxa de retenção",
    "cohort_size": "Tamanho da coorte",
    "retained_customers": "Clientes retidos",
    "discount_rate": "Taxa de desconto",
    "discount_amount": "Valor de desconto",
    "discount_per_completed_order": "Desconto por pedido concluído",
    "margin_per_completed_order": "Margem por pedido concluído",
    "orders_with_promo": "Pedidos com promoção",
    "promo_conversion": "Conversão da promoção",
    "repeat_rate": "Taxa de recompra",
    # identifier and calendar columns that reach a table header
    "metric_date": "Data",
    "zone_id": "Zona",
    "merchant_id": "Parceiro",
    "promotion_id": "ID da promoção",
    "promo_code": "Campanha",
    "promo_type": "Tipo",
    "funded_by": "Financiado por",
    "cohort_month": "Coorte",
    "acquisition_channel": "Canal de aquisição",
    "period_index": "Período",
    "period_start_date": "Início do período",
    "period_days": "Duração do período (dias)",
    "period_days_observed": "Dias observados",
    "experiment_id": "Experimento",
    "variant": "Variante",
}

# The same register, used as a column-rename map. A table selects the gold
# columns it needs and renames through this; pandas ignores the keys that are
# not present, so one dict serves every page and a column can only be named in
# one place.
COLUMN_LABELS: dict[str, str] = METRIC_LABELS

# Gold columns the ENGINE's unit register does not carry, and what they are
# measured in. A display register and nothing else: it decides whether a number
# is drawn as "58,1%" or "0,58", never what the number is.
EXTRA_UNITS: dict[str, str] = {
    "retention_rate": "ratio",
    "discount_rate": "ratio",
    "promo_conversion": "ratio",
    "repeat_rate": "ratio",
    "margin_per_completed_order": "BRL",
    "discount_per_completed_order": "BRL",
    "discount_amount": "BRL",
}

# --- the window-aggregate vocabulary -----------------------------------------
#
# METRIC_LABELS above names the METRIC. These name what .headline() actually
# reports about it, which for a per-day flow is a DAILY AVERAGE and not the
# window's total -- R$ 29.977,82 per day, not the R$ 419.689,55 the 14 days add
# up to. A card headed "GMV" showing the first number is read as the second, and
# no formatter can fix that because the float is right; only the label is wrong.
#
# Written out per metric rather than derived by appending a suffix: Portuguese
# agreement differs by gender and number ("GMV médio diário", "Sessões por
# dia", "Margem de contribuição média diária"), and a generated label would be
# grammatically wrong on half the register.
#
# Only DAILY_AVERAGE metrics appear here. A rate is a rate at any window length
# and a per-order average is not per day, so both keep their plain label --
# labelling the ticket médio "/dia" would be a new false statement, not a fix.
HEADLINE_LABELS: dict[str, str] = {
    "gmv": "GMV médio diário",
    "orders_placed": "Pedidos realizados por dia",
    "orders_completed": "Pedidos concluídos por dia",
    "sessions": "Sessões por dia",
    "contribution_margin": "Margem de contribuição média diária",
    "active_customers": "Clientes ativos por dia",
}

# The suffix on the VALUE, carrying the same fact as the label so a figure
# quoted on its own -- copied out of a card, read aloud in a meeting -- still
# says what it is.
PER_DAY_SUFFIX = "/dia"


# --- engine vocabularies -----------------------------------------------------
#
# Values the engine emits as identifiers and the product shows as words. The
# engine's own values are never changed: `DECISION_STATUSES` stays
# ("INVESTIGATE", "APPROVED", "REJECTED") and only its rendering is translated,
# so a memo written by `pulse decide` and a decision recorded on screen refer to
# the same state.
#
# SCOPE_LABELS is imported above (pulse.metrics._SCOPE_LABELS) rather than
# declared here like the vocabularies below it: decision_memo, playbook and
# copilot all render the same scope word through pulse.metrics._scope_phrase,
# so this page reads the one dict that word actually comes from instead of
# keeping a copy that could say something different for the same scope.

PATTERN_LABELS: dict[str, str] = {
    "fulfillment_eta_degradation": "degradação do prazo de entrega",
    "supply_availability_gap": "lacuna de disponibilidade da oferta",
    "unknown_pattern": "padrão não classificado",
}

# Display only. The engine's DECISION_STATUSES values stay English.
DECISION_STATUS_LABELS: dict[str, str] = {
    "INVESTIGATE": "Investigar",
    "APPROVED": "Aprovado",
    "REJECTED": "Rejeitado",
}

EVIDENCE_STRENGTH_LABELS: dict[str, str] = {
    "strong": "forte",
    "moderate": "moderada",
    "weak": "fraca",
}

STATISTICAL_VERDICT_LABELS: dict[str, str] = {
    "significant": "significativo",
    "not_significant": "não significativo",
}

# "unavailable" is not a fourth direction. It is the state the Experiment Lab is
# actually in: the dataset carries no identifiable treatment-specific incremental
# cost, so no economic verdict exists to render. Folding it into "marginal" would
# show a measured-looking result that happens to be small.
# pulse.experiments.ECONOMIC_STATUS_* rendered. "illustrative_only" is the state
# EXP-001 is in: the incremental arithmetic is shown for the method it
# demonstrates and the number it produces measures nothing.
ECONOMIC_STATUS_LABELS: dict[str, str] = {
    "measured": "medida",
    "illustrative_only": "apenas ilustrativa",
    "unavailable": "indisponível",
}

BUSINESS_VERDICT_LABELS: dict[str, str] = {
    "positive": "positivo",
    "marginal": "marginal",
    "negative": "negativo",
    "unavailable": "indisponível",
}

CHANNEL_LABELS: dict[str, str] = {
    "organic": "Orgânico",
    "paid_search": "Busca paga",
    "paid_social": "Mídia social paga",
    "referral": "Indicação",
}

PROMO_TYPE_LABELS: dict[str, str] = {
    "free_delivery": "Entrega grátis",
    "percentage_off": "Desconto percentual",
    "none": "Sem promoção",
}

FUNDED_BY_LABELS: dict[str, str] = {
    "marketplace": "Marketplace",
    "merchant": "Parceiro",
    "none": "Sem financiamento",
}

# --- navigation --------------------------------------------------------------
#
# The group keys are internal identifiers that also order the sidebar; the
# labels are what a reader sees. PULSE stays uppercase everywhere.

GROUP_LABELS: dict[str, str] = {
    "Command": "Comando",
    "Analytics": "Análises",
    "Platform": "Plataforma",
}

PAGE_TITLES: dict[str, str] = {
    "decision": "Inteligência para Decisão",
    "copilot.py": "PULSE Copilot",
    "operations.py": "Operações e Regiões",
    "merchants.py": "Parceiros",
    "customers.py": "Clientes e Retenção",
    "promotions.py": "Promoções",
    "experiment_lab.py": "Laboratório de Experimentos",
    "data_quality.py": "Qualidade de Dados e Arquitetura",
}

APP_TITLE = "PULSE — Inteligência para Decisão"
APP_TAGLINE = "Inteligência para decisão em operações de marketplace"

# --- formatting --------------------------------------------------------------
#
# Brazilian conventions, applied at render time and only at render time. An
# analytical frame is never written back to in a localised form: these functions
# take a float and return a string, and the float they were handed is unchanged.

NOT_AVAILABLE = "n/d"

# "18,759.17" -> "18.759,17". Both separators swap in a single pass, which is
# why str.translate is used rather than two chained replaces.
_SEPARATORS = str.maketrans({",": ".", ".": ","})


def _br(value: float, places: int, signed: bool = False) -> str:
    """One number in Brazilian notation: "." for thousands, "," for decimals."""
    return format(value, f"{'+' if signed else ''},.{places}f").translate(_SEPARATORS)


def fmt_int(value: float) -> str:
    """A count, grouped: 12345 -> "12.345"."""
    if value != value:  # NaN
        return NOT_AVAILABLE
    return _br(value, 0)


def fmt_float(value: float, places: int = 2, signed: bool = False) -> str:
    """A bare number in Brazilian notation. `signed` for figures whose sign is
    part of the reading -- a correlation coefficient, a temporal alignment.
    """
    if value != value:
        return NOT_AVAILABLE
    return _br(value, places, signed=signed)


def fmt_brl(value: float) -> str:
    """A magnitude, never a signed currency string. Direction belongs in the
    surrounding words -- the engine's memo helpers take the same line, because
    "R$ -18.759,17" in a "no mínimo" sentence reads as a smaller loss.
    """
    if value != value:
        return NOT_AVAILABLE
    return f"R$ {_br(abs(value), 2)}"


def fmt_pct(delta_pct: float, places: int = 2) -> str:
    """A percentage change, signed, because a delta's direction is its meaning.

    The input is already in percent (the engine's deviation_pct), so nothing is
    scaled here.
    """
    if delta_pct != delta_pct:
        return NOT_AVAILABLE
    return f"{_br(delta_pct, places, signed=True)}%"


def fmt_ratio(value: float, places: int = 1, signed: bool = False) -> str:
    """A fraction rendered as a percentage: 0.0340 -> "+3,4%".

    Unit conversion for display, and distinct from fmt_pp: this is a *relative*
    figure, a ratio expressed in percent.
    """
    if value != value:
        return NOT_AVAILABLE
    return f"{_br(value * 100.0, places, signed=signed)}%"


def fmt_pp(fraction: float, places: int = 2) -> str:
    """A fraction rendered as signed percentage POINTS: 0.0105 -> "+1,05 p.p.".

    The distinction from fmt_pct is load-bearing and must survive translation:
    an absolute difference between two rates is in points, and writing it as
    "%" is exactly how a +1,05 p.p. move gets read as a +3,4% move. Both figures
    are correct and they are not the same figure.
    """
    if fraction != fraction:
        return NOT_AVAILABLE
    return f"{_br(fraction * 100.0, places, signed=True)} p.p."


def fmt_date(value: Any) -> str:
    """A date as Brazil writes it: 10/09/2026. Accepts date, datetime or
    pandas Timestamp; anything else is returned as it came.
    """
    strftime = getattr(value, "strftime", None)
    return strftime("%d/%m/%Y") if strftime is not None else str(value)


def fmt_value(value: float, unit: str, semantic: str | None = None) -> str:
    """One metric value in its own unit. Unit conversion for display
    (0.745 -> "74,5%"), not a calculation.

    `semantic` is pulse.metrics.METRIC_SEMANTICS for the metric the value came
    from, and is what separates a single day's figure from a WINDOW AGGREGATE of
    it. Pass DAILY_AVERAGE and the figure is suffixed "/dia", because a window
    mean of a per-day flow is a rate per day and reads as a window total without
    it. Omit it (the default) for a value that is already one day's, one order's
    or one delivery's -- a gold row, a chart point -- where the suffix would say
    nothing the axis does not.
    """
    if value != value:  # NaN
        return NOT_AVAILABLE
    per_day = PER_DAY_SUFFIX if semantic == DAILY_AVERAGE else ""
    if unit == "BRL":
        return f"{fmt_brl(value)}{per_day}"
    if unit == "ratio":
        # A rate is never per-day: it is dimensionless at any window length, and
        # METRIC_SEMANTICS classifies every ratio as RATE, so this branch cannot
        # be reached with DAILY_AVERAGE.
        return f"{_br(value * 100.0, 1)}%"
    if unit == "minutes":
        return f"{_br(value, 1)} min"
    if unit in ("orders", "sessions", "customers"):
        return f"{_br(value, 0)}{per_day}"
    return f"{_br(value, 1)}{per_day}"


# --- label lookups -----------------------------------------------------------


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").capitalize())


def headline_label(metric: str, semantic: str | None = None) -> str:
    """The label for a WINDOW AGGREGATE of `metric`, not for the metric itself.

    Falls back to metric_label() for everything that is not a daily average,
    which is the correct answer for a rate (dimensionless) and for an average
    per order or per delivery (already averaged over something that is not a
    day). `semantic` defaults to the engine's own classification, so a caller
    holding only a metric name still gets the right label.
    """
    if semantic is None:
        semantic = semantic_of(metric) if metric in ENGINE_METRIC_LABELS else None
    if semantic == DAILY_AVERAGE:
        return HEADLINE_LABELS.get(metric, metric_label(metric))
    return metric_label(metric)


def headline_kpi_label(row: dict) -> str:
    """A .headline() row's label, read off the row's own semantic."""
    return headline_label(row["metric"], row.get("semantic"))


def scope_label(scope: str, scope_value: Any) -> str:
    """("zone", <value>) -> "Zona <value>"; ("company", "all") ->
    "Toda a empresa".

    The scope VALUE is passed through untouched: it is engine output, and the
    only way a specific zone may ever appear on screen. The mapping itself
    (SCOPE_LABELS) is imported from pulse.metrics, so this and
    pulse.metrics._scope_phrase -- which decision_memo, playbook and copilot
    all render through -- can never disagree on the word for a scope.
    """
    if scope == "company":
        return SCOPE_LABELS["company"]
    prefix = SCOPE_LABELS.get(scope, scope.replace("_", " ").capitalize())
    return f"{prefix} {scope_value}"


def pattern_label(pattern: str) -> str:
    return PATTERN_LABELS.get(pattern, pattern.replace("_", " "))


def decision_status_label(status: str) -> str:
    return DECISION_STATUS_LABELS.get(status, status)


def evidence_strength_label(strength: str) -> str:
    return EVIDENCE_STRENGTH_LABELS.get(strength, strength)


def statistical_verdict_label(verdict: str) -> str:
    return STATISTICAL_VERDICT_LABELS.get(verdict, verdict.replace("_", " "))


def business_verdict_label(verdict: str) -> str:
    return BUSINESS_VERDICT_LABELS.get(verdict, verdict)


def economic_status_label(status: str) -> str:
    return ECONOMIC_STATUS_LABELS.get(status, status.replace("_", " "))


def channel_label(channel: str) -> str:
    return CHANNEL_LABELS.get(channel, channel.replace("_", " ").capitalize())


def promo_type_label(promo_type: str) -> str:
    return PROMO_TYPE_LABELS.get(promo_type, promo_type.replace("_", " ").capitalize())


def funded_by_label(funder: str) -> str:
    return FUNDED_BY_LABELS.get(funder, funder.capitalize())


# --- the standing wording ----------------------------------------------------
#
# Sentences that appear on more than one page, or that carry a guarantee the
# product must not quietly strengthen. They live here so there is exactly one
# copy of each to review.

NOTICE_DECISION_SUPPORT = "Apoio à decisão — nenhuma execução operacional automática"
NOTICE_SYNTHETIC_DATA = "Todos os dados são sintéticos"

# "No mínimo", never "está custando". Every impact figure is a deviation
# OBSERVED ALONGSIDE the diagnosed pattern, never a cost attributed to it.
IMPACT_FLOOR_NOTE = (
    "<b>Duas unidades de tempo nesta tabela.</b> As linhas marcadas "
    "\"acumulado na janela\" somam o desvio sobre todos os dias da janela de "
    "comparação; a linha de <b>ritmo</b> é o desvio da <b>média diária</b>. As "
    "duas diferem por um fator igual ao número de dias da janela e descrevem a "
    "mesma medição. "
    "Cada número aqui é um desvio <b>estimado</b>, observado junto ao padrão "
    "diagnosticado, e nunca um custo atribuído a ele. A janela de linha de base "
    "termina no dia anterior ao início da janela de comparação, de modo que uma "
    "deterioração já em curso faz parte da própria base contra a qual se compara "
    "— o que rebaixa a base e encurta a diferença medida. Estes números, "
    "portanto, subestimam, e estão redigidos como \"no mínimo\"."
)

NO_ANOMALY_NOTE = (
    "Nenhuma anomalia superou a sensibilidade e o piso de materialidade atuais. "
    "A ausência de anomalia não é evidência de saúde — reduza a sensibilidade na "
    "barra lateral para ampliar a rede."
)

# Association, never causation. The single exception in PULSE is the randomised
# experiment, and even there the causal wording comes from pulse.experiments.
ASSOCIATION_NOTE = (
    "Os direcionadores são séries que se moveram <i>junto</i> a esta — uma "
    "associação relatada como evidência, nunca uma causa demonstrada."
)

# first_detected_date is where the evidence STARTS INSIDE THE WINDOW. It is not
# the date the incident began, and the two are routinely confused.
FIRST_SIGNAL_NOTE = (
    "Primeiro sinal identificado dentro da janela analisada: <b>{date}</b>. "
    "É onde a evidência começa dentro da janela, não uma afirmação sobre quando "
    "o incidente teve início."
)

# The statistics are measured. There is NO measured ROI. That contrast is the
# point, and it is stronger than the one this constant used to make: the earlier
# wording called the ROI an "estimativa ilustrativa", which still reads as a
# weaker measurement of the same thing. It is not a measurement at all. The cost
# side is the ordinary promotional discount, both arms carry it, and the CONTROL
# arm carries more -- so there is no treatment cost to divide by and no ROI to
# report. The engine says the same thing in one place
# (pulse.experiments.NO_MEASURED_ROI_NOTE) and this page quotes that rather than
# keeping a second wording of the same absence.
ROI_UNAVAILABLE_HEADING = "<b>Nenhum ROI de tratamento medido é reportado.</b>"
ROI_PROXY_WARNING = (
    "O resultado estatístico das seções 02 e 03 não é proxy — aquele é medido e "
    "se sustenta. A aritmética abaixo é exibida pelo método que demonstra, não "
    "pelo número que produz."
)

QUARANTINE_EMPTY_REASON = "— (nenhuma linha em quarentena)"
