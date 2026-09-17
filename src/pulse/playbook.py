# Deterministic recommendation playbook: diagnosis -> Recommendation, with no
# LLM involvement in the policy decision. This module is why the Decision
# Memo works with no API key, and why "the engine decides, the LLM narrates"
# is a claim the project can defend rather than assert.
#
# playbook.yml is DATA, loaded once at import. Adding a pattern (Task 16,
# Task 18) is a YAML edit -- a new top-level key there, and a matching entry
# in root_cause.PATTERNS -- never a branch here. PATTERNS below is derived
# from the YAML file's own keys, so a new entry extends it with zero code
# change.
#
# recommend() selects a template by diagnosis.pattern ALONE. It never
# branches on a scope value, a zone id or a metric name -- doing so would be
# a hard failure of the point of this module: one deterministic, auditable
# lookup table, not a pile of if-this-zone special cases.
#
# Templates render through str.format() against the whitelisted context dict
# built by render_context(). A template referencing a key that dict does not
# carry raises KeyError at render time -- str.format's own behaviour, not
# extra code -- rather than leaking a literal "{token}" into an
# analyst-facing memo.
#
# LANGUAGE. The rendered prose is Brazilian Portuguese, because an analyst
# reads it. The lookup keys and the metric identifiers are not translated:
# they are what the engine keys on and what the gold tables are named, and a
# label that drifts from its identifier is a label nobody can trace back to a
# column. The metric LABELS below (_METRIC_LABELS) are the same story and stay
# English for the same reason. The scope WORD a reader sees inline ("Zona 7",
# "Toda a empresa") is different: it is prose built around the untranslated
# scope identifier, not the identifier itself, so it is translated -- see
# _scope_label below, which renders it via metrics._scope_phrase rather than
# spelling it out here, so this module's rationale text and decision_memo's
# evidence text can never disagree on the word for the same scope.
from __future__ import annotations

from pathlib import Path

import yaml

from pulse.metrics import _scope_phrase
from pulse.types import Diagnosis, Impact, Recommendation

PLAYBOOK: dict[str, dict[str, str]] = yaml.safe_load(
    (Path(__file__).parent / "playbook.yml").read_text(encoding="utf-8")
)

_FALLBACK = "unknown_pattern"

# Patterns the playbook has a REAL (non-fallback) entry for -- i.e. PLAYBOOK's
# keys minus the fallback itself, since the fallback is not something a
# diagnosis classifies AS, it's what recommend() uses when classification
# didn't match anything. Derived from PLAYBOOK's own keys (not a
# hand-maintained list) so a pattern added to the YAML in Task 16/18 extends
# this tuple automatically.
PATTERNS: tuple[str, ...] = tuple(k for k in PLAYBOOK if k != _FALLBACK)

# Human-readable labels for metric names that appear in rendered text. Falls
# back to a lightly cleaned version of the raw metric name for anything not
# listed, so a metric added to metrics.METRIC_REGISTER in Task 17/18 is still
# readable without a code change here.
#
# NOT TRANSLATED, deliberately. These are the metric identifiers made
# readable, and they are the same strings the glossary, the gold columns and
# the KPI rows carry. Translating them would put a Portuguese label in front
# of a reader and leave nothing that points back at the column it came from.
_METRIC_LABELS: dict[str, str] = {
    "gmv": "GMV",
    "orders_placed": "orders placed",
    "orders_completed": "orders completed",
    "sessions": "sessions",
    "order_conversion": "order conversion",
    "completion_rate": "completion rate",
    "cancellation_rate": "cancellation rate",
    "aov": "average order value",
    "avg_actual_delivery_minutes": "actual delivery time",
    "avg_promised_eta_minutes": "promised ETA",
    "on_time_rate": "on-time rate",
    "contribution_margin": "contribution margin",
    "active_customers": "active customers",
    # The raw name would clean up to "availability rate", which does not say
    # whose availability. It is merchant supply, measured as the share of
    # scheduled-open merchant hours that were orderable.
    "availability_rate": "merchant availability",
}


def _label(metric: str) -> str:
    return _METRIC_LABELS.get(metric, metric.replace("_", " "))


def _sentence_case(text: str) -> str:
    return text if not text or not text[0].islower() else text[0].upper() + text[1:]


def _num(value: float, places: int = 0, plus: bool = False) -> str:
    """A number in Brazilian notation: "." groups thousands, "," is decimal.

    Python formats en-US, so the two separators are swapped in a single pass
    through a sentinel. PRESENTATION ONLY: the rounding is the rounding the
    engine already did, and no computed value moves. It lives here rather
    than in decision_memo because playbook.py is the module both the memo
    and the Copilot already import their shared text helpers from.

    The Copilot's numeric validator parses this exact notation
    (copilot._to_float). The two must move together: a figure rendered in
    one notation and checked in another reads as unverified.
    """
    text = f"{value:{'+' if plus else ''},.{places}f}"
    return text.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def _brl(value: float) -> str:
    """A MAGNITUDE in BRL, e.g. "R$ 18.759". Direction lives in the words."""
    return f"R$ {_num(abs(value))}"


def _scope_label(d: Diagnosis) -> str:
    """Describe where the diagnosis is scoped -- never a crash, never a
    fabricated segment, for primary_segment=None (8 of 12 current diagnoses:
    rate and duration metrics get no segment decomposition by design, so they
    carry no primary_segment even when the anomaly itself is zone-scoped).

    Falls back to the anomaly's OWN scope, which is honest evidence (it is
    literally where the anomaly fired) even when no decomposition covers it.

    The word is rendered through metrics._scope_phrase, the one place that
    renders a scope into prose -- decision_memo and copilot use the same
    function, so "Zona 7" here and "Zona 7" in the memo's own evidence can
    never say two different things about the same scope.
    """
    if d.primary_segment is not None:
        return _scope_phrase(d.primary_segment.dimension, d.primary_segment.segment)
    return _scope_phrase(d.anomaly.scope, d.anomaly.scope_value)


def _concentration(d: Diagnosis) -> str:
    if d.primary_segment is not None:
        return (
            f"{_num(d.primary_segment.contribution_pct, 1)}% do desvio "
            "em nível de empresa"
        )
    return (
        "nenhuma decomposição por segmento disponível para esta métrica "
        "(taxas e durações não são aditivas entre segmentos)"
    )


def _join_labels(metrics: list[str]) -> str:
    labels = [_label(m) for m in metrics]
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " e " + labels[-1]


def _drivers_list(d: Diagnosis) -> str:
    if not d.drivers:
        return "nenhum fator associado"
    return _join_labels([dr.metric for dr in d.drivers])


def _driver_note(d: Diagnosis) -> str:
    """One sentence on what, if anything, moved with the anomaly.

    Reports only drivers whose evidence is NOT weak. A correlation below the
    reporting threshold contributes zero to root_cause.compute_confidence and
    is excluded from pattern classification there; naming it here as an
    "associated series that moved with it" would reintroduce, in the sentence
    a human actually reads, the evidence the engine has already judged to be
    none. On zone 4 all three fulfilment candidates measure |r| 0.13-0.16
    against merchant availability, and "nothing upstream moved with it" is the
    finding, not an omission.
    """
    reported = [dr.metric for dr in d.drivers if dr.evidence_strength != "weak"]
    if not reported:
        return (
            "Nenhum fator candidato atingiu o limiar de correlação reportável "
            "nesta janela."
        )
    return (
        "Séries associadas que se moveram junto com ela nesta janela: "
        f"{_join_labels(reported)}."
    )


def render_context(diagnosis: Diagnosis, impact: Impact) -> dict[str, str]:
    """The whitelisted rendering context for playbook.yml templates.

    Every template field may reference ONLY these keys -- str.format raises
    KeyError on anything else, which is the point: a typo in a template
    surfaces as a test failure, never a literal "{placeholder}" in output a
    human is meant to read and act on.

    Handles diagnosis.primary_segment=None (see _scope_label/_concentration)
    without crashing and without inventing a segment that was never observed.
    """
    return {
        "segment": _scope_label(diagnosis),
        "metric": _sentence_case(_label(diagnosis.anomaly.metric)),
        "deviation_pct": f"{_num(diagnosis.anomaly.deviation_pct, 1, plus=True)}%",
        "concentration": _concentration(diagnosis),
        "drivers_list": _drivers_list(diagnosis),
        "driver_note": _driver_note(diagnosis),
        "confidence_pct": f"{_num(diagnosis.confidence * 100)}%",
        # Magnitude, not signed deviation: both templates that consume this
        # token phrase it as an amount at risk / recoverable ("no mínimo R$
        # X"), where direction is already carried by the surrounding words.
        # projected_30d_brl is negative for a drop, so rendering it raw
        # produced "no mínimo R$ -18,759" -- backwards as a floor, since a
        # bigger loss would print as a smaller (more negative) number.
        # Only the ADVERSE part, as prioritization._at_risk_30d scores it: abs()
        # alone turned a scope whose GMV ROSE into recoverable money.
        "gmv_at_risk": _brl(max(0.0, -impact.projected_30d_brl)),
    }


def recommend(diagnosis: Diagnosis, impact: Impact) -> Recommendation:
    """diagnosis, impact -> deterministic Recommendation. No LLM involved.

    Selection is keyed on diagnosis.pattern alone. unknown_pattern is a real
    entry in PLAYBOOK (not a conditionally-built stub), so any pattern
    PLAYBOOK does not carry -- including one root_cause hasn't been taught
    yet -- falls back to it explicitly. recommend() therefore never returns
    None and never raises for an unrecognised pattern.
    """
    key = diagnosis.pattern if diagnosis.pattern in PLAYBOOK else _FALLBACK
    entry = PLAYBOOK[key]
    ctx = render_context(diagnosis, impact)
    return Recommendation(
        action=entry["action"].format(**ctx).strip(),
        rationale=entry["rationale"].format(**ctx).strip(),
        expected_effect=entry["expected_effect"].format(**ctx).strip(),
        validation_method=entry["validation_method"].format(**ctx).strip(),
        owner_function=entry["owner_function"],
        playbook_id=key,
        effort=entry["effort"],
    )
