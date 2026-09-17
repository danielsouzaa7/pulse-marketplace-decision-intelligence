# Semantic grounding for the Copilot — the adversarial matrix and an INDEPENDENT
# oracle.
#
# WHY THIS FILE EXISTS SEPARATELY FROM tests/test_copilot.py.
#
# Code review #2 found that the previous round's headline benchmark was
# circular: its truth set was `copilot._build_claims(...)` filtered by the
# validator's own acceptance predicate at the validator's own tolerance, so
# "0.0% false acceptance" was true by construction and could not fail for the
# reason it claimed to test. The published "61.2% -> 0.0%" was not reproducible.
#
# Everything in this file is therefore built from the ENGINE's canonical
# objects — DecisionCycleResult, metrics.METRIC_SEMANTICS, metrics._UNITS,
# ExperimentReport — and NEVER from anything in copilot.py that participates in
# validation. The one import guard below is asserted, not just intended.
#
# THE CONTRACT UNDER TEST. A numeric statement is certified only when the
# system can support all five of:
#
#     WHAT   concept/metric it refers to
#     WHICH  scope it refers to
#     WHAT   value it states
#     WHAT   unit it is in
#     WHAT   it MEANS — daily average vs window total, measured vs residual vs
#            illustrative
#
# A number appearing somewhere in the bundle is not enough. A substring of valid
# engine prose is not enough. A correct value with the wrong meaning is not
# enough. Ambiguous context fails closed.
from __future__ import annotations

import ast
import pathlib
import re

import pytest

from pulse.config import AS_OF
from pulse.copilot import build_evidence_bundle, validate_response
from pulse.engine import run_decision_cycle
from pulse.experiments import analyse_experiment
from pulse.metrics import (
    DAILY_AVERAGE,
    METRIC_SEMANTICS,
    PER_DELIVERY_AVERAGE,
    PER_ORDER_AVERAGE,
    RATE,
    _METRIC_LABELS_PT,
    _UNITS,
    load_gold,
)
from pulse.types import AnalysisParams

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = AnalysisParams(as_of=AS_OF)

# The oracle's OWN reading of a pt-BR figure. Written here rather than imported:
# the validator's tokenizer and parser are part of what is under test.
_ORACLE_NUMBER_RE = re.compile(r"-?\d(?:[\d.]*\d)?(?:,\d+)?")


def _parse_br(token: str) -> float:
    return float(token.replace(".", "").replace(",", "."))


@pytest.fixture(scope="module")
def gold():
    return load_gold()


@pytest.fixture(scope="module")
def result(gold):
    return run_decision_cycle(gold, P)


@pytest.fixture(scope="module")
def report(gold):
    return analyse_experiment(gold, "EXP-001")


@pytest.fixture(scope="module")
def bundle(result):
    return build_evidence_bundle(result)


# --- the oracle's independence, asserted --------------------------------------


# The only names this file may take from pulse.copilot: the entry points a
# CALLER of the Copilot uses. An allowlist, because the denylist it replaces went
# stale -- 8 of its 20 names no longer existed and ~13 current acceptance
# internals were not on it (review #3). A validator internal added after this
# line was written is forbidden by default.
_COPILOT_PUBLIC = frozenset({
    "build_evidence_bundle", "validate_response", "experiment_evidence", "ask",
    "NullNarrator", "MAX_BUNDLE_CHARS", "EvidenceBundleTooLargeError",
    "ForbiddenEvidenceKeyError", "SYSTEM_PROMPT",
})


def test_this_module_takes_only_the_copilots_public_entry_points():
    """The circularity guard, enforced on this file's own AST.

    Review #2's finding was that the benchmark's truth set came out of the
    validator's own index, filtered by the validator's own predicate -- a
    benchmark that cannot fail. Review #3 found the guard against that was a
    stale denylist that getattr() walked straight past.
    """
    tree = ast.parse(pathlib.Path(__file__).read_text(encoding="utf-8"))
    taken: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "pulse.copilot":
                taken.update(alias.name for alias in node.names)
            if node.module == "pulse":
                # The module object itself would allow copilot.<anything>.
                assert all(alias.name != "copilot" for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(("pulse", "importlib")), alias.name
    assert taken <= _COPILOT_PUBLIC, (
        f"the oracle took validator internals: {taken - _COPILOT_PUBLIC}"
    )
    assert "validate_response" in taken

    # No dynamic route around the import check.
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    dynamic = {"getattr", "vars", "globals", "locals", "__import__", "eval",
               "__globals__", "__builtins__",
               "exec", "import_module", "__dict__", "modules", "importlib"}
    assert not names & dynamic, f"dynamic access in the oracle: {names & dynamic}"


# --- the oracle ----------------------------------------------------------------
#
# (concept, scope, value, unit_kind, semantic) tuples derived from engine
# objects. Deliberately a different shape and a different derivation from
# whatever copilot.py builds internally: this says what is TRUE, not what the
# validator happens to index.

ORACLE_CURRENCY = "currency"
ORACLE_PERCENT = "percent"
ORACLE_POINTS = "points"
ORACLE_RATIO = "ratio"
ORACLE_COUNT = "count"
ORACLE_DURATION = "duration"
ORACLE_SCORE = "score"

_ORACLE_UNIT = {
    "BRL": ORACLE_CURRENCY,
    "ratio": ORACLE_RATIO,
    "minutes": ORACLE_DURATION,
    "orders": ORACLE_COUNT,
    "sessions": ORACLE_COUNT,
    "customers": ORACLE_COUNT,
}


def _oracle_facts(result) -> set[tuple]:
    """Every (concept, scope, |value|, kind, semantic) the engine actually holds.

    Read straight off the result objects. No copilot import participates.
    """
    facts: set[tuple] = set()

    def add(concept, scope, value, kind, semantic=None):
        if value is None or value != value:
            return
        facts.add((concept, scope, round(abs(float(value)), 6), kind, semantic))

    for row in tuple(result.headline_kpis) + tuple(result.segment_kpis):
        scope = (row["scope"], str(row["scope_value"]))
        metric = row["metric"]
        semantic = METRIC_SEMANTICS[metric]
        kind = _ORACLE_UNIT[_UNITS[metric]]
        for field in ("recent", "baseline"):
            add(metric, scope, row[field], kind, semantic)
            if kind == ORACLE_RATIO:
                add(metric, scope, row[field] * 100.0, ORACLE_PERCENT, semantic)
        add(metric, scope, row["delta_pct"], ORACLE_PERCENT, "relative_change")

    for anomaly in result.anomalies:
        scope = (anomaly.scope, str(anomaly.scope_value))
        metric = anomaly.metric
        semantic = METRIC_SEMANTICS[metric]
        kind = _ORACLE_UNIT[_UNITS[metric]]
        for value in (anomaly.recent_value, anomaly.baseline_value):
            add(metric, scope, value, kind, semantic)
            if kind == ORACLE_RATIO:
                add(metric, scope, value * 100.0, ORACLE_PERCENT, semantic)
        add(metric, scope, anomaly.deviation_pct, ORACLE_PERCENT, "relative_change")
        add(metric, scope, anomaly.z_score, ORACLE_SCORE)
        add(metric, scope, anomaly.n_observations, ORACLE_COUNT)

    for priority in result.priorities:
        anomaly = priority.diagnosis.anomaly
        scope = (anomaly.scope, str(anomaly.scope_value))
        impact = priority.impact
        # Whose money: estimate_impact denominates impact in the scope's GMV,
        # completed orders and margin, never in the metric that fired.
        add("gmv", scope, impact.gmv_at_risk_brl, ORACLE_CURRENCY, "window_total")
        add("contribution_margin", scope, impact.margin_impact_brl,
            ORACLE_CURRENCY, "window_total")
        add("orders_completed", scope, impact.orders_lost, ORACLE_COUNT,
            "window_total")
        add("gmv", scope, impact.daily_run_rate_brl, ORACLE_CURRENCY, DAILY_AVERAGE)
        add("gmv", scope, impact.projected_30d_brl, ORACLE_CURRENCY, "window_total")
        add(
            None, scope, impact.customers_affected, ORACLE_COUNT,
            "residual_customers" if impact.customers_are_residual
            else "measured_customers",
        )
        add(None, scope, priority.impact_score, ORACLE_SCORE)
        add(None, scope, priority.diagnosis.confidence, ORACLE_SCORE)
    return facts


def _oracle_percent_values(result, metric, scope) -> set[float]:
    """Every percent-kind magnitude legitimately sayable about (metric, scope).

    Only facts ABOUT that metric. Accepting concept-free facts here encoded loose
    scope-level attribution as truth (review #3), which put the classes that
    were still open outside the benchmark by construction.
    """
    out = set()
    for concept, fact_scope, value, kind, _semantic in _oracle_facts(result):
        if kind == ORACLE_PERCENT and fact_scope == scope and concept == metric:
            out.add(value)
    return out


def _oracle_levels(result, metric, scope) -> dict[str, set]:
    """{"recent": {...}, "baseline": {...}}: every rendered level of (metric,
    scope) per window, straight off the engine objects."""
    out: dict[str, set] = {"recent": set(), "baseline": set()}
    rows = [(r["metric"], (r["scope"], str(r["scope_value"])), r["recent"], r["baseline"])
            for r in tuple(result.headline_kpis) + tuple(result.segment_kpis)]
    rows += [(a.metric, (a.scope, str(a.scope_value)), a.recent_value, a.baseline_value)
             for a in result.anomalies]
    for priority in result.priorities:
        anomaly = priority.diagnosis.anomaly
        rows += [(d.metric, (anomaly.scope, str(anomaly.scope_value)), d.recent, d.baseline)
                 for d in priority.diagnosis.drivers]
    for row_metric, row_scope, recent, baseline in rows:
        if row_metric == metric and row_scope == scope:
            out["recent"].add(_level(metric, recent))
            out["baseline"].add(_level(metric, baseline))
    return out


def _oracle_signed_changes(result, metric, scope) -> set[float]:
    out = {r["delta_pct"] for r in tuple(result.headline_kpis) + tuple(result.segment_kpis)
           if r["metric"] == metric and (r["scope"], str(r["scope_value"])) == scope}
    out |= {a.deviation_pct for a in result.anomalies
            if a.metric == metric and (a.scope, str(a.scope_value)) == scope}
    return {v for v in out if v == v}


def _oracle_count_values(result, scope) -> set[float]:
    out = set()
    for _concept, fact_scope, value, kind, _semantic in _oracle_facts(result):
        if kind == ORACLE_COUNT and fact_scope == scope:
            out.add(value)
    return out


def _scope_phrase(scope) -> str:
    return "Toda a empresa" if scope[0] == "company" else f"Zona {scope[1]}"


def _pt(metric: str) -> str:
    return _METRIC_LABELS_PT[metric]


def _num(value: float, places: int = 2) -> str:
    return f"{value:,.{places}f}".translate(str.maketrans({",": ".", ".": ","}))


# --- C1: a correct value with the wrong semantic must fail ---------------------


def test_a_daily_average_may_be_stated_as_a_daily_average(result, bundle):
    """The true reading passes. Without this the whole contract could be
    satisfied by rejecting everything."""
    row = next(r for r in result.headline_kpis if r["metric"] == "gmv")
    text = (
        f"O GMV médio diário de Toda a empresa foi de R$ {_num(row['recent'])}."
    )
    assert validate_response(text, bundle).all_verified, text


def test_a_daily_average_may_not_be_stated_as_a_window_total(result, bundle):
    """CRITICAL B, as a test. The value is real to the last centavo; the meaning
    is wrong by a factor of the window length, and no arithmetic check can catch
    it because nothing about the float says which quantity it is.
    """
    row = next(r for r in result.headline_kpis if r["metric"] == "gmv")
    daily = _num(row["recent"])
    for claim in (
        f"O GMV total de Toda a empresa nos 14 dias foi de R$ {daily}.",
        f"O GMV acumulado de Toda a empresa na janela foi de R$ {daily}.",
        f"Toda a empresa somou R$ {daily} de GMV na janela de comparação.",
    ):
        report = validate_response(claim, bundle)
        assert not report.all_verified, claim


@pytest.mark.parametrize("metric", ["gmv", "sessions", "orders_completed"])
def test_no_flow_metrics_daily_average_can_be_called_a_total(
    result, bundle, metric
):
    row = next(r for r in result.headline_kpis if r["metric"] == metric)
    value = _num(row["recent"], 0 if metric != "gmv" else 2)
    claim = (
        f"O total de {_pt(metric)} de Toda a empresa nos 14 dias foi "
        f"{value}."
    )
    assert not validate_response(claim, bundle).all_verified, claim


def test_active_customers_daily_average_is_not_the_customer_population(
    result, bundle
):
    """active_customers is daily-DISTINCT: no window total exists for it at all,
    so calling the daily average a population is a category error as well as a
    magnitude one."""
    row = next(r for r in result.headline_kpis if r["metric"] == "active_customers")
    value = _num(row["recent"], 0)
    assert not validate_response(
        f"{value} clientes distintos pediram em Toda a empresa na janela de "
        f"comparação.", bundle
    ).all_verified


def test_an_accumulated_window_figure_passes_when_it_is_structured(
    result, bundle
):
    """The other half of the contract: a genuine window total, present as
    structured evidence, may be stated as one."""
    priority = result.priorities[0]
    scope = _scope_phrase(
        (priority.diagnosis.anomaly.scope, priority.diagnosis.anomaly.scope_value)
    )
    claim = (
        f"Em {scope}, o desvio de GMV acumulado na janela de 14 dias é de no "
        f"mínimo R$ {_num(priority.impact.gmv_at_risk_brl)}."
    )
    assert validate_response(claim, bundle).all_verified, claim


# --- I1: residual customers may not be called measured customers --------------


def test_a_residual_customer_count_may_not_be_called_a_measured_one(
    result, bundle
):
    """I1 reaching the Copilot. 3,896 is what is left of an aggregate's customer
    count after the nested segment claims are netted out. It is not a count of
    anybody, and the measured company population is 5,632."""
    residual = next(
        (p for p in result.priorities if p.impact.customers_are_residual), None
    )
    assert residual is not None, "no netted aggregate priority in this run"
    value = _num(residual.impact.customers_affected, 0)
    scope = _scope_phrase(
        (residual.diagnosis.anomaly.scope, residual.diagnosis.anomaly.scope_value)
    )
    for claim in (
        f"Em {scope}, {value} clientes pediram na janela.",
        f"Em {scope}, {value} clientes do escopo foram medidos na janela.",
        f"O número medido de clientes em {scope} é {value}.",
    ):
        assert not validate_response(claim, bundle).all_verified, claim


def test_a_residual_customer_count_may_be_stated_as_a_residual(result, bundle):
    residual = next(
        p for p in result.priorities if p.impact.customers_are_residual
    )
    value = _num(residual.impact.customers_affected, 0)
    scope = _scope_phrase(
        (residual.diagnosis.anomaly.scope, residual.diagnosis.anomaly.scope_value)
    )
    claim = (
        f"Em {scope}, os clientes considerados no impacto residual somam "
        f"{value}."
    )
    assert validate_response(claim, bundle).all_verified, claim


def test_a_measured_customer_count_keeps_the_measured_wording(result, bundle):
    """The fix must be a distinction, not a blanket hedge: a segment scope's
    count IS measured and must remain sayable as one."""
    measured = next(
        p for p in result.priorities if not p.impact.customers_are_residual
    )
    value = _num(measured.impact.customers_affected, 0)
    scope = _scope_phrase(
        (measured.diagnosis.anomaly.scope, measured.diagnosis.anomaly.scope_value)
    )
    claim = f"Em {scope}, {value} clientes pediram na janela de comparação."
    assert validate_response(claim, bundle).all_verified, claim


# --- I3: an illustrative ratio may not be called a measured ROI ---------------


def test_a_proxy_ratio_may_not_be_called_a_measured_roi(result, report):
    """I3 reaching the Copilot. measured_roi is None; the cost side is a discount
    the CONTROL arm carries more of. Every phrasing that asserts measurement
    must fail."""
    from pulse.copilot import experiment_evidence

    bundle = build_evidence_bundle(
        result, experiment_summary=experiment_evidence(report)
    )
    value = _num(abs(report.economics.roi) * 100.0, 1)
    for claim in (
        f"O ROI medido do tratamento foi de -{value}%.",
        f"O retorno medido do experimento foi -{value}%.",
        f"O ROI real do incentivo é -{value}%.",
        f"O ROI observado do tratamento foi -{value}%.",
        f"Então o tratamento teve ROI de -{value}%, certo?",
    ):
        assert not validate_response(claim, bundle).all_verified, claim


def test_a_proxy_ratio_may_be_stated_as_the_illustration_it_is(result, report):
    from pulse.copilot import experiment_evidence

    bundle = build_evidence_bundle(
        result, experiment_summary=experiment_evidence(report)
    )
    value = _num(abs(report.economics.roi) * 100.0, 1)
    claim = (
        f"Na demonstração metodológica com custo proxy, a razão calculada foi "
        f"de -{value}%."
    )
    assert validate_response(claim, bundle).all_verified, claim


def test_the_bundle_never_carries_a_measured_roi(result, report):
    from pulse.copilot import experiment_evidence

    evidence = experiment_evidence(report)
    assert evidence["measured_roi"] is None
    assert evidence["economic_evaluation_status"] == "illustrative_only"
    # The proxy ratio is not indexed under a generic "roi" name.
    assert "roi" not in {k for k in evidence if k == "roi"}


# --- the quotation class of exploit -------------------------------------------


def test_no_substring_of_engine_prose_bypasses_numeric_grounding(bundle):
    """CRITICAL A. The exemption accepted any >=24-char substring of bundle
    prose, so a fragment beginning INSIDE a numeral matched and carried a
    magnitude that exists nowhere: "8,2%" for a real 88,2%.
    """
    prose = [
        value for value in _all_bundle_strings(bundle.to_dict())
        if len(value) > 60
    ]
    assert prose, "no bundle prose to attack"

    known = set(bundle.known_numbers())
    fabricated = []
    for statement in prose:
        for start in range(1, min(len(statement), 400)):
            fragment = statement[start:]
            if len(fragment) < 24:
                break
            # A fragment that begins mid-numeral states a truncated magnitude.
            if not (statement[start - 1].isdigit() and fragment[0].isdigit()):
                continue
            if not validate_response(fragment, bundle).all_verified:
                continue
            # THE PROPERTY THAT MATTERS: was a magnitude the bundle does not
            # contain certified? The original exploit truncated 88,2 to 8,2 and
            # 8,2 existed nowhere among the bundle's magnitudes. A truncation
            # whose leading figure happens to equal a real run-level count (a
            # "0" or a "4" from a data-quality row) certifies nothing the engine
            # did not compute, so it is not the defect -- and asserting on it
            # would be asserting on an artefact of where the cut landed.
            leading = _ORACLE_NUMBER_RE.match(fragment)
            value = abs(_parse_br(leading.group(0)))
            if not any(abs(value - k) <= 1e-9 for k in known):
                fabricated.append(fragment[:70])
    assert not fabricated, (
        f"{len(fabricated)} truncated prose fragments certified a magnitude "
        f"absent from the bundle: {fabricated[:3]}"
    )


def test_changing_one_digit_inside_engine_prose_is_rejected(bundle):
    statement = next(
        value for value in _all_bundle_strings(bundle.to_dict())
        if "R$" in value and len(value) > 80
    )
    tampered = re.sub(r"(\d)", lambda m: str((int(m.group(1)) + 1) % 10), statement,
                      count=1)
    assert tampered != statement
    assert not validate_response(tampered, bundle).all_verified


def test_a_valid_engine_statement_reattached_to_another_scope_is_rejected(
    result, bundle
):
    """Quoting a real sentence about one scope while asserting another is the
    attack the substring exemption could not see."""
    priority = result.priorities[0]
    lead = (priority.diagnosis.anomaly.scope, priority.diagnosis.anomaly.scope_value)
    other = next(
        (p.diagnosis.anomaly.scope, p.diagnosis.anomaly.scope_value)
        for p in result.priorities
        if (p.diagnosis.anomaly.scope, p.diagnosis.anomaly.scope_value) != lead
    )
    claim = (
        f"Em {_scope_phrase(other)}, o desvio de GMV acumulado na janela de 14 "
        f"dias é de no mínimo R$ {_num(priority.impact.gmv_at_risk_brl)}."
    )
    assert not validate_response(claim, bundle).all_verified, claim


def test_two_valid_figures_may_not_be_recombined_into_a_false_claim(
    result, bundle
):
    """Each number is real; the sentence pairs them with the wrong metrics."""
    anomalies = {
        (a.metric, a.scope, a.scope_value): a.deviation_pct
        for a in result.anomalies
    }
    gmv = next(
        (k, v) for k, v in anomalies.items()
        if k[0] == "gmv" and k[1] == "zone"
    )
    completion = next(
        (k, v) for k, v in anomalies.items()
        if k[0] == "completion_rate" and k[1] == "zone" and k[2] == gmv[0][2]
    )
    scope = _scope_phrase(("zone", gmv[0][2]))
    swapped = (
        f"Em {scope}, o GMV caiu {_num(abs(completion[1]), 1)}% e a "
        f"{_pt('completion_rate')} caiu {_num(abs(gmv[1]), 1)}%."
    )
    assert not validate_response(swapped, bundle).all_verified, swapped

    correct = (
        f"Em {scope}, o GMV caiu {_num(abs(gmv[1]), 1)}% e a "
        f"{_pt('completion_rate')} caiu {_num(abs(completion[1]), 1)}%."
    )
    assert validate_response(correct, bundle).all_verified, correct


def test_one_valid_and_one_fabricated_figure_fails_the_whole_response(
    result, bundle
):
    row = next(r for r in result.headline_kpis if r["metric"] == "gmv")
    claim = (
        f"O GMV médio diário de Toda a empresa foi de R$ {_num(row['recent'])}, "
        f"afetando 87 clientes."
    )
    report = validate_response(claim, bundle)
    assert not report.all_verified
    assert "87" in report.unverified_figures


def _all_bundle_strings(node) -> list[str]:
    out: list[str] = []
    if isinstance(node, str):
        out.append(node)
    elif isinstance(node, dict):
        for value in node.values():
            out.extend(_all_bundle_strings(value))
    elif isinstance(node, (list, tuple)):
        for value in node:
            out.extend(_all_bundle_strings(value))
    return out


# --- the three frozen pre-fix premises ---------------------------------------
#
# Recorded as failing premises before the validator was touched, and asserted
# here so each stays closed. Every one of the three was certified
# `all_verified=True` at the previous HEAD.


def test_premise_the_truncation_exploit_is_closed(bundle):
    """8,2 existed nowhere among the bundle's magnitudes; the real completion
    rate is 88,2%. A >=24-char substring of bundle prose was accepted without
    per-figure grounding, and substrings can begin inside a numeral."""
    report = validate_response(
        "8,2% contra 91,2%, z = -5,40, 14 dias observados).", bundle
    )
    assert not report.all_verified
    assert "8,2" in report.unverified_figures


def test_premise_a_daily_mean_is_no_longer_certified_as_a_window_total(
    result, bundle
):
    """The value is the company GMV daily mean to the centavo. The 14-day total
    is R$ 419.689,55 -- fourteen times larger."""
    row = next(r for r in result.headline_kpis if r["metric"] == "gmv")
    report = validate_response(
        f"O GMV total da empresa nos 14 dias foi de R$ {_num(row['recent'])}.",
        bundle,
    )
    assert not report.all_verified


def test_premise_the_residual_count_is_no_longer_certified_as_measured(
    result, bundle
):
    """3.896 is the aggregate's customer count after nested netting. 5.632
    customers actually ordered."""
    residual = next(
        p for p in result.priorities if p.impact.customers_are_residual
    )
    report = validate_response(
        f"Em toda a empresa, {_num(residual.impact.customers_affected, 0)} "
        f"clientes pediram na janela de comparação.",
        bundle,
    )
    assert not report.all_verified


# --- sentence segmentation ----------------------------------------------------


@pytest.mark.parametrize(
    "template",
    [
        "A {label} caiu {value} p.p. na {scope}.",
        "A {label} caiu {value} p.p. aprox. na {scope}.",
        "A {label} de aprox. {value} p.p. foi medida na {scope}.",
    ],
)
def test_an_abbreviation_does_not_strand_a_figure_from_its_scope(
    result, bundle, template
):
    """`p.p.` and `aprox.` end in a full stop. A naive [.!?]\\s+ splitter cut the
    sentence there, orphaning the figure from the scope it named, and the orphan
    then defaulted to company-wide — so a COMPANY figure grounded a claim that
    explicitly said Zona 7. That is the original I5 failure mode, reachable with
    the engine's own notation and no adversarial intent.
    """
    company = next(
        r for r in result.headline_kpis if r["metric"] == "completion_rate"
    )
    claim = template.format(
        label=_pt("completion_rate"),
        value=_num(abs(company["delta_pct"]), 1),
        scope="Zona 7",
    )
    assert not validate_response(claim, bundle).all_verified, claim


def test_a_scope_survives_an_abbreviation_when_the_claim_is_true(
    result, bundle
):
    """The same segmentation must not reject a TRUE claim written with the same
    notation — otherwise the fix is just a stricter reject-everything."""
    zone7 = next(
        r for r in result.segment_kpis
        if r["metric"] == "completion_rate" and r["scope_value"] == "7"
    )
    claim = (
        f"A {_pt('completion_rate')} caiu aprox. "
        f"{_num(abs(zone7['delta_pct']), 1)}% na Zona 7."
    )
    assert validate_response(claim, bundle).all_verified, claim


# --- unknown concept must fail closed -----------------------------------------


@pytest.mark.parametrize(
    "claim",
    [
        "O NPS em Zona 7 é 62,4%.",
        "A retenção em Zona 7 está em no mínimo R$ 10.351.",
        "Fraude em Zona 4 custou R$ 6.517,91.",
        "A taxa de retenção caiu 13,5% na Zona 7.",
        "O churn em Toda a empresa foi de 88,2%.",
    ],
)
def test_an_unrecognised_concept_fails_closed(bundle, claim):
    """An unknown metric used to produce an EMPTY concept set, indistinguishable
    from "no metric named", so the sentence inherited every scope-level fact and
    real numbers licensed claims about concepts the engine has never heard of.
    Unknown scopes failed closed; unknown concepts failed open."""
    report = validate_response(claim, bundle)
    assert not report.all_verified, claim


def test_a_sentence_with_no_concept_at_all_may_still_use_scope_context(
    result, bundle
):
    """NO_CONCEPT_MENTIONED and UNKNOWN_CONCEPT are different states, and only
    the first may inherit context."""
    priority = result.priorities[0]
    scope = _scope_phrase(
        (priority.diagnosis.anomaly.scope, priority.diagnosis.anomaly.scope_value)
    )
    claim = (
        f"Em {scope}, o desvio acumulado na janela de 14 dias é de no mínimo "
        f"R$ {_num(priority.impact.gmv_at_risk_brl)}."
    )
    assert validate_response(claim, bundle).all_verified, claim


# --- percent vs percentage points ---------------------------------------------


def test_a_relative_percentage_may_not_ground_a_percentage_point_claim(
    result, bundle
):
    """A relative change and an absolute difference between two rates are
    different quantities. Folding p.p. into percent let one ground the other."""
    row = next(
        r for r in result.headline_kpis if r["metric"] == "completion_rate"
    )
    relative = _num(abs(row["delta_pct"]), 1)
    assert validate_response(
        f"A {_pt('completion_rate')} de Toda a empresa caiu {relative}%.", bundle
    ).all_verified
    assert not validate_response(
        f"A {_pt('completion_rate')} de Toda a empresa caiu {relative} p.p.",
        bundle,
    ).all_verified


def test_an_experiment_effect_in_points_may_not_be_stated_as_a_percentage(
    result, report
):
    from pulse.copilot import experiment_evidence

    bundle = build_evidence_bundle(
        result, experiment_summary=experiment_evidence(report)
    )
    points = _num(report.test.absolute_diff * 100.0, 2)
    relative = _num(report.test.relative_uplift * 100.0, 1)
    assert validate_response(
        f"O efeito absoluto do experimento foi de +{points} p.p.", bundle
    ).all_verified
    assert validate_response(
        f"O efeito relativo do experimento foi de +{relative}%.", bundle
    ).all_verified
    # ... and neither may borrow the other's unit.
    assert not validate_response(
        f"O efeito absoluto do experimento foi de +{points}%.", bundle
    ).all_verified
    assert not validate_response(
        f"O efeito relativo do experimento foi de +{relative} p.p.", bundle
    ).all_verified


def test_a_dimensionless_score_does_not_license_a_percentage(result, bundle):
    """A correlation of -0,7456 and a confidence of 0,6238 are not 74,56% and
    62,38%. An unconditional x100 twin made every such field usable as a
    percentage, which is where the residual false acceptances lived."""
    priority = result.priorities[0]
    confidence = priority.diagnosis.confidence
    claim = f"A confiança do diagnóstico é {_num(confidence * 100.0, 1)}%."
    assert not validate_response(claim, bundle).all_verified, claim
    driver = priority.diagnosis.drivers[0]
    correlation = abs(driver.correlation_with_target)
    assert not validate_response(
        f"A {_pt(driver.metric)} caiu {_num(correlation * 100.0, 1)}% na Zona 7.",
        bundle,
    ).all_verified


# --- currency / count / notation ----------------------------------------------


def test_a_count_may_not_be_written_as_currency_or_the_reverse(result, bundle):
    priority = result.priorities[0]
    scope = _scope_phrase(
        (priority.diagnosis.anomaly.scope, priority.diagnosis.anomaly.scope_value)
    )
    orders = _num(priority.impact.orders_lost, 0)
    assert not validate_response(
        f"Em {scope}, o desvio acumulado na janela de 14 dias foi de R$ {orders}.",
        bundle,
    ).all_verified
    money = _num(priority.impact.gmv_at_risk_brl, 0)
    assert not validate_response(
        f"Em {scope}, {money} pedidos ficaram abaixo da baseline na janela.",
        bundle,
    ).all_verified


def test_en_us_notation_is_rejected_under_the_pt_br_contract(result, bundle):
    row = next(r for r in result.headline_kpis if r["metric"] == "gmv")
    en_us = f"{row['recent']:,.2f}"
    assert not validate_response(
        f"O GMV médio diário de Toda a empresa foi de R$ {en_us}.", bundle
    ).all_verified


@pytest.mark.parametrize(
    "rendered", ["2026-09-10", "10/09/2026"]
)
def test_a_date_the_bundle_carries_is_not_a_business_claim(bundle, rendered):
    """Both formats the product renders. The app writes dd/mm/yyyy
    (ui_text.fmt_date); the engine writes ISO. A model echoing either must not be
    flagged, and neither may a date's digits ground a count."""
    report = validate_response(f"A data da análise é {rendered}.", bundle)
    assert report.all_verified, report.unverified_figures


def test_a_date_the_bundle_does_not_carry_is_reported(bundle):
    for absent in ("2019-01-01", "01/01/2019"):
        report = validate_response(f"A data da análise é {absent}.", bundle)
        assert not report.all_verified, absent


def test_a_bare_year_does_not_ground_a_count(bundle):
    assert not validate_response(
        "Em Toda a empresa, 2026 clientes pediram na janela.", bundle
    ).all_verified


def test_a_claim_with_no_numbers_is_not_failed_for_having_none(bundle):
    report = validate_response(
        "A margem de contribuição caiu mais do que o GMV neste recorte.", bundle
    )
    assert report.all_verified
    assert report.figures_checked == 0


# --- the offline answer must still verify -------------------------------------


def test_the_deterministic_offline_answer_is_fully_grounded(bundle):
    """The anti-over-fix guard. The offline answer is a pure function of the
    bundle, so every figure in it must ground STRUCTURALLY — not through an
    exemption. If this fails, the answer's prose has to become more explicit,
    never the validator more permissive.
    """
    from pulse.copilot import NullNarrator, ask

    answer = ask("O que aconteceu?", bundle, NullNarrator())
    assert answer.validation.all_verified, answer.validation.unverified_figures
    assert answer.validation.figures_checked > 10


# --- review #3: the classes the numeric contract still let through -----------
#
# Every figure below is read off engine objects. Each class is asserted both
# ways: the false claim is rejected AND the true claim of the same shape passes,
# so a validator that rejects everything cannot satisfy this block.

# The oracle's own polarity, written from the metric definitions rather than
# imported: a rise in these is bad news.
_ORACLE_LOWER_IS_BETTER = frozenset(
    {"cancellation_rate", "avg_actual_delivery_minutes", "avg_promised_eta_minutes"}
)


def _where(anomaly) -> str:
    return _scope_phrase((anomaly.scope, str(anomaly.scope_value)))


def _level(metric: str, value: float) -> str | None:
    """A window level rendered in its unit, or None where this block does not
    render the metric (per-day flows need the daily marker, tested elsewhere)."""
    unit = _UNITS[metric]
    if unit == "ratio":
        return f"{_num(value * 100.0, 1)}%"
    if unit == "minutes":
        return f"{_num(value, 1)} min"
    return None


@pytest.mark.parametrize("up, down", [("subiu", "caiu"), ("aumentou", "diminuiu"),
                                      ("cresceu", "recuou")])
def test_c5_a_movement_may_not_be_narrated_in_the_opposite_direction(
    result, bundle, up, down
):
    """C5. abs() stripped the sign from every fact, so "subiu 5,7%" certified
    against a 5,7% DROP -- 31/31 inversions in review #3."""
    checked = 0
    for anomaly in result.anomalies:
        size = _num(abs(anomaly.deviation_pct), 1)
        true_verb, false_verb = (up, down) if anomaly.deviation_pct > 0 else (down, up)
        subject = f"A {_pt(anomaly.metric)} em {_where(anomaly)}"
        good = f"{subject} {true_verb} {size}%."
        bad = f"{subject} {false_verb} {size}%."
        assert validate_response(good, bundle).all_verified, good
        assert not validate_response(bad, bundle).all_verified, bad
        checked += 1
    assert checked >= 10


def test_c5_an_explicit_sign_may_not_contradict_the_fact(result, bundle):
    for anomaly in result.anomalies:
        size = _num(abs(anomaly.deviation_pct), 1)
        real, fake = ("+", "-") if anomaly.deviation_pct > 0 else ("-", "+")
        subject = f"A {_pt(anomaly.metric)} em {_where(anomaly)} variou"
        assert validate_response(f"{subject} {real}{size}%.", bundle).all_verified
        assert not validate_response(f"{subject} {fake}{size}%.", bundle).all_verified


def test_c5_improvement_and_deterioration_follow_the_metrics_polarity(
    result, bundle
):
    """A cancellation rate that ROSE got worse; an on-time rate that FELL got
    worse. "melhorou" about either is an inversion even with the right number."""
    for anomaly in result.anomalies:
        size = _num(abs(anomaly.deviation_pct), 1)
        improving = (anomaly.deviation_pct < 0) == (
            anomaly.metric in _ORACLE_LOWER_IS_BETTER
        )
        true_word, false_word = (
            ("melhorou", "piorou") if improving else ("piorou", "melhorou")
        )
        subject = f"A {_pt(anomaly.metric)} em {_where(anomaly)}"
        good, bad = f"{subject} {true_word} {size}%.", f"{subject} {false_word} {size}%."
        assert validate_response(good, bundle).all_verified, good
        assert not validate_response(bad, bundle).all_verified, bad


def test_c5_scope_money_keeps_its_direction(result, bundle):
    for priority in result.priorities[:3]:
        anomaly = priority.diagnosis.anomaly
        run_rate = priority.impact.daily_run_rate_brl
        real, fake = ("acima", "abaixo") if run_rate > 0 else ("abaixo", "acima")
        stem = f"Em {_where(anomaly)}, o GMV médio diário ficou R$ {_num(abs(run_rate), 0)} por dia"
        assert validate_response(f"{stem} {real} da baseline.", bundle).all_verified
        assert not validate_response(f"{stem} {fake} da baseline.", bundle).all_verified


def test_c4_scope_money_belongs_to_the_series_it_was_measured_on(result, bundle):
    """C4 and Important 1. estimate_impact denominates money in the SCOPE's GMV
    (and margin in its margin), never in the metric that fired. A clause naming
    another metric may not quote it: "A Conversão em pedidos em Zona 7 custou
    R$ 10.350,87" certified while conversion had moved +6,76%."""
    from pulse.playbook import _label

    for priority in result.priorities[:3]:
        anomaly = priority.diagnosis.anomaly
        where = _where(anomaly)
        gmv = f"R$ {_num(priority.impact.gmv_at_risk_brl, 0)}"
        margin = f"R$ {_num(priority.impact.margin_impact_brl, 0)}"
        run_rate = f"R$ {_num(abs(priority.impact.daily_run_rate_brl), 0)}"
        for good in (
            f"Em {where}, o desvio de GMV acumulado na janela é de no mínimo {gmv}.",
            f"Em {where}, o desvio acumulado na janela é de no mínimo {gmv}.",
            f"Em {where}, o desvio de margem de contribuição acumulado na janela "
            f"é de no mínimo {margin}.",
        ):
            assert validate_response(good, bundle).all_verified, good
        assert not validate_response(
            f"Em {where}, o desvio de GMV acumulado na janela é de no mínimo {margin}.",
            bundle,
        ).all_verified
        for other in METRIC_SEMANTICS:
            if other == "gmv":
                continue
            for name in (_pt(other), _label(other)):
                bad = (f"Em {where}, o desvio de {name} acumulado na janela é de "
                       f"no mínimo {gmv}.")
                assert not validate_response(bad, bundle).all_verified, bad
                c4 = (f"Em {where}, o {name} médio diário ficou {run_rate} por dia "
                      f"abaixo da baseline.")
                assert not validate_response(c4, bundle).all_verified, c4


def test_important3_a_deviation_may_not_be_stated_as_a_level(result, bundle):
    """Same metric, same scope, same window, 6,4x apart: the accumulated
    DEVIATION stated as the window's GMV, the orders BELOW baseline stated as
    orders completed, a relative change stated as the rate itself."""
    for priority in result.priorities[:3]:
        where = _where(priority.diagnosis.anomaly)
        gmv = _num(priority.impact.gmv_at_risk_brl, 0)
        orders = _num(priority.impact.orders_lost, 0)
        for bad in (
            f"Em {where}, o GMV total acumulado na janela foi de R$ {gmv}.",
            f"Em {where}, o GMV acumulado na janela somou R$ {gmv}.",
            f"Em {where}, foram concluídos {orders} pedidos na janela.",
            f"Em {where}, o total de pedidos concluídos na janela foi {orders}.",
        ):
            assert not validate_response(bad, bundle).all_verified, bad
    for anomaly in result.anomalies:
        level = _level(anomaly.metric, anomaly.recent_value)
        if level is None or _UNITS[anomaly.metric] != "ratio":
            continue
        subject = f"A {_pt(anomaly.metric)} em {_where(anomaly)} é"
        assert validate_response(f"{subject} {level}.", bundle).all_verified
        bad = f"{subject} {_num(abs(anomaly.deviation_pct), 1)}%."
        assert not validate_response(bad, bundle).all_verified, bad


def test_important4_recent_and_baseline_are_not_interchangeable(result, bundle):
    """26/28 in review #3: the baseline stated as the current value certified."""
    recent_values: dict[tuple, set] = {}
    for anomaly in result.anomalies:
        key = (anomaly.metric, anomaly.scope, str(anomaly.scope_value))
        recent_values.setdefault(key, set()).add(_level(anomaly.metric, anomaly.recent_value))
    for row in result.headline_kpis:
        key = (row["metric"], row["scope"], str(row["scope_value"]))
        recent_values.setdefault(key, set()).add(_level(row["metric"], row["recent"]))

    checked = 0
    for anomaly in result.anomalies:
        recent = _level(anomaly.metric, anomaly.recent_value)
        base = _level(anomaly.metric, anomaly.baseline_value)
        key = (anomaly.metric, anomaly.scope, str(anomaly.scope_value))
        if recent is None or recent == base or base in recent_values[key]:
            continue
        verb = "subiu" if anomaly.deviation_pct > 0 else "caiu"
        name, where = _pt(anomaly.metric), _where(anomaly)
        assert validate_response(f"A {name} atual em {where} é {recent}.", bundle).all_verified
        assert not validate_response(f"A {name} atual em {where} é {base}.", bundle).all_verified
        assert validate_response(f"A {name} em {where} era {base} na baseline.", bundle).all_verified
        assert not validate_response(f"A {name} em {where} era {recent} na baseline.", bundle).all_verified
        good = f"A {name} em {where} {verb} de {base} para {recent}."
        bad = f"A {name} em {where} {verb} de {recent} para {base}."
        assert validate_response(good, bundle).all_verified, good
        assert not validate_response(bad, bundle).all_verified, bad
        checked += 1
    assert checked >= 5


@pytest.mark.parametrize("separator", [" — ", " / ", " vs ", " ao passo que ", "; já "])
def test_important5_two_metrics_do_not_pool_across_any_separator(
    result, bundle, separator
):
    anomalies = {(a.metric, a.scope, str(a.scope_value)): a for a in result.anomalies}
    gmv = next(a for k, a in anomalies.items() if k[0] == "gmv" and k[1] == "zone")
    completion = anomalies[("completion_rate", "zone", str(gmv.scope_value))]
    g, c = _num(abs(gmv.deviation_pct), 1), _num(abs(completion.deviation_pct), 1)
    where = _where(gmv)
    good = f"Em {where}, o GMV caiu {g}%{separator}a {_pt('completion_rate')} caiu {c}%."
    bad = f"Em {where}, o GMV caiu {c}%{separator}a {_pt('completion_rate')} caiu {g}%."
    assert validate_response(good, bundle).all_verified, good
    assert not validate_response(bad, bundle).all_verified, bad


@pytest.mark.parametrize("period", ["por hora", "por semana", "por mês", "por minuto",
                                    "/hora", "ao mês"])
def test_important6_a_window_figure_may_not_be_restated_on_another_time_basis(
    result, bundle, period
):
    row = next(r for r in result.headline_kpis if r["metric"] == "gmv")
    bad_daily = f"O GMV médio de Toda a empresa foi de R$ {_num(row['recent'])} {period}."
    assert not validate_response(bad_daily, bundle).all_verified, bad_daily
    priority = result.priorities[0]
    bad_total = (
        f"Em {_where(priority.diagnosis.anomaly)}, o desvio de GMV acumulado é de "
        f"no mínimo R$ {_num(priority.impact.gmv_at_risk_brl, 0)} {period}."
    )
    assert not validate_response(bad_total, bundle).all_verified, bad_total


def test_important7_a_figure_true_of_one_scope_is_not_true_of_two(result, bundle):
    gmv = next(a for a in result.anomalies if a.metric == "gmv" and a.scope == "zone")
    other_zone = next(
        str(a.scope_value) for a in result.anomalies
        if a.scope == "zone" and str(a.scope_value) != str(gmv.scope_value)
    )
    company = next(r for r in result.headline_kpis if r["metric"] == "gmv")
    g, co = _num(abs(gmv.deviation_pct), 1), _num(abs(company["delta_pct"]), 1)
    zone, other = _where(gmv), f"Zona {other_zone}"
    for bad in (
        f"Em {other} e {zone}, o GMV caiu {g}%.",
        f"O GMV caiu {g}% em {zone} e em {other}.",
        f"O GMV caiu {g}% em Toda a empresa e na {zone}.",
        f"O GMV caiu {co}% em {zone} e {g}% em Toda a empresa.",
    ):
        assert not validate_response(bad, bundle).all_verified, bad
    good = f"O GMV caiu {g}% em {zone} e {co}% em Toda a empresa."
    assert validate_response(good, bundle).all_verified, good


@pytest.mark.parametrize(
    "template",
    [
        "O GMV de fraude em {where} caiu {g}%.",
        "A Taxa de conclusão de entregas expressas em {where} caiu {c}%.",
        "- NPS em {where}: R$ {money}",
        "NPS de {where}: {g}%",
        "Churn — {where}: {c}%",
    ],
)
def test_important2_an_unknown_concept_does_not_inherit_a_known_ones_numbers(
    result, bundle, template
):
    anomalies = {(a.metric, a.scope, str(a.scope_value)): a for a in result.anomalies}
    gmv = next(a for k, a in anomalies.items() if k[0] == "gmv" and k[1] == "zone")
    completion = anomalies[("completion_rate", "zone", str(gmv.scope_value))]
    priority = next(
        p for p in result.priorities
        if p.diagnosis.anomaly.scope_value == gmv.scope_value
    )
    claim = template.format(
        where=_where(gmv),
        g=_num(abs(gmv.deviation_pct), 1),
        c=_num(abs(completion.deviation_pct), 1),
        money=_num(priority.impact.gmv_at_risk_brl, 0),
    )
    assert not validate_response(claim, bundle).all_verified, claim


def test_important2_a_scope_qualifier_is_not_an_unknown_concept(result, bundle):
    gmv = next(a for a in result.anomalies if a.metric == "gmv" and a.scope == "zone")
    claim = f"O GMV da {_where(gmv)} caiu {_num(abs(gmv.deviation_pct), 1)}%."
    assert validate_response(claim, bundle).all_verified, claim


def test_important9_the_engines_own_residual_sentence_grounds(result, bundle):
    """"não uma contagem medida" is a NEGATION of measurement, and the measured
    cue matched it, so the engine's own residual sentence rejected itself."""
    residual = next(p for p in result.priorities if p.impact.customers_are_residual)
    claim = (
        f"Em {_where(residual.diagnosis.anomaly)}, os clientes considerados no "
        f"impacto residual — líquido dos escopos aninhados, e portanto não uma "
        f"contagem medida — somam {_num(residual.impact.customers_affected, 0)}."
    )
    assert validate_response(claim, bundle).all_verified, claim


@pytest.mark.parametrize(
    "claim",
    [
        "A queda do GMV em Zona 7 foi causada pelo atraso nas entregas.",
        "O atraso nas entregas causou a queda da taxa de conclusão.",
        "A taxa de conclusão caiu por causa do tempo de entrega.",
        "O GMV caiu devido ao aumento de cancelamentos.",
        "Os cancelamentos são responsáveis pela perda de GMV.",
        "O tempo de entrega levou à queda de pedidos concluídos.",
    ],
)
def test_a_causal_claim_is_reported_even_with_no_figure_in_it(bundle, claim):
    """Association, never causation, is rule 3 of the system prompt -- and until
    now nothing checked it. A narration that promotes an association to a cause
    must not come back under a clean badge."""
    report = validate_response(claim, bundle)
    assert not report.all_verified, claim
    assert report.causal_claims, claim


@pytest.mark.parametrize(
    "claim",
    [
        "A queda do GMV em Zona 7 está associada ao atraso nas entregas.",
        "Todo fator listado é uma associação, não uma causa.",
        "Nada aqui estabelece que uma série causou a outra.",
        "Nenhum experimento randomizado está anexado, portanto nada aqui sustenta "
        "uma afirmação causal.",
    ],
)
def test_association_and_negated_causation_are_not_causal_claims(bundle, claim):
    report = validate_response(claim, bundle)
    assert report.all_verified, (claim, report.causal_claims)


def test_a_masked_scope_digit_or_date_is_not_counted_as_a_checked_figure(bundle):
    """figures_checked counted the 7 in "Zona 7", so a badge could read "Todas as
    N figuras ligadas" where nothing was bound at all."""
    report = validate_response("A Zona 7 foi analisada em 2026-09-10.", bundle)
    assert report.all_verified
    assert report.figures_checked == 1  # the date, checked as a date


# --- review #4: plain wording, not the benchmark's templates ------------------
#
# The code review of round 3 wrote sentences the way a person or a model would,
# without looking at the validator's word lists, and found false claims certified
# in plain Portuguese and true ones rejected. Both lists are built from engine
# objects; each FALSE sentence says what the engine actually holds.


def _natural_facts(result):
    by = {(a.metric, a.scope, str(a.scope_value)): a for a in result.anomalies}
    gmv = next(a for k, a in by.items() if k[0] == "gmv" and k[1] == "zone")
    zone = str(gmv.scope_value)
    top = next(p for p in result.priorities
               if (p.diagnosis.anomaly.scope, str(p.diagnosis.anomaly.scope_value)) == ("zone", zone))
    company_gmv = next(r for r in result.headline_kpis if r["metric"] == "gmv")
    return {
        "where": f"Zona {zone}",
        "g": _num(abs(gmv.deviation_pct), 1),
        "g_co": _num(abs(company_gmv["delta_pct"]), 1),
        "c": _num(abs(by[("completion_rate", "zone", zone)].deviation_pct), 1),
        "c_recent": _num(by[("completion_rate", "zone", zone)].recent_value * 100, 1),
        "c_base": _num(by[("completion_rate", "zone", zone)].baseline_value * 100, 1),
        "c_co": _num(abs(by[("completion_rate", "company", "all")].deviation_pct), 1),
        "c_co_recent": _num(by[("completion_rate", "company", "all")].recent_value * 100, 1),
        "x_co_recent": _num(by[("cancellation_rate", "company", "all")].recent_value * 100, 1),
        "x_recent": _num(by[("cancellation_rate", "zone", zone)].recent_value * 100, 1),
        "x_base": _num(by[("cancellation_rate", "zone", zone)].baseline_value * 100, 1),
        "o": _num(abs(by[("on_time_rate", "zone", zone)].deviation_pct), 1),
        "money": _num(top.impact.gmv_at_risk_brl, 0),
        "materiality": _num(P.min_materiality_brl, 0),
        "confidence": _num(top.diagnosis.confidence, 3),
        "score": _num(top.impact_score, 2),
        "customers": _num(top.impact.customers_affected, 0),
        "projection": _num(abs(top.impact.projected_30d_brl), 0),
        "gmv_recent": _num(gmv.recent_value, 0),
        "co_score": _num(next(p for p in result.priorities
                              if p.diagnosis.anomaly.scope == "company").impact_score, 2),
    }


_NATURAL_FALSE = (
    # a threshold, a score, a confidence and a customer count quoted as a metric
    "O GMV médio diário em {where} foi de R$ {materiality}.",
    "O GMV médio diário em {where} foi de R$ {materiality} por hora.",
    "O GMV em {where} é R$ {materiality} por semana.",
    "A taxa de conclusão em {where} é {confidence}.",
    "O GMV em {where} é {score}.",
    "Os pedidos concluídos em {where} foram {customers}.",
    "Em {where}, {customers} pedidos foram perdidos, acumulado na janela.",
    # direction without a verb from the list
    "O GMV em {where} teve um desvio positivo de {g}%.",
    "O GMV em {where} se recuperou {g}%.",
    "O GMV em {where} não caiu {g}%.",
    # a change stated as a level, with the change word elsewhere in the sentence
    "Em {where}, o GMV caiu: o total acumulado na janela foi de R$ {money}.",
    "Em {where}, onde o GMV caiu, o GMV acumulado na janela somou R$ {money}.",
    "A taxa de conclusão em {where} caiu e é {c}%.",
    # the baseline stated as the comparison window's value
    "A taxa de conclusão em {where} na janela de comparação foi de {c_base}%.",
    "A taxa de conclusão em {where} é {c_base}%.",
    # one figure pooled across scopes or metrics through the clause splitter
    "Em Zona 4 e Zona 7, o GMV caiu {g}% e a taxa de conclusão da empresa caiu {c_co}%.",
    "Em {where}, a margem de contribuição e o GMV acumulam desvio de no mínimo R$ {money}.",
    # a narrower concept than the metric named
    "O GMV de clientes novos em {where} caiu {g}%.",
    "O GMV de pedidos cancelados em {where} caiu {g}%.",
    # causal wording the expression list missed
    "O aumento no tempo de entrega causou a queda do GMV em {where}.",
    "Sem dúvida, o atraso nas entregas causou a queda do GMV.",
    "A queda do GMV em {where} se deve ao atraso nas entregas.",
    # review #5: a LEVEL written as a change
    "A taxa de conclusão em {where} caiu {c_recent}%.",
    "A taxa de conclusão em {where} caiu {c_base}%.",
    "Na {where}, a taxa de conclusão teve uma queda de {c_base}% em relação à baseline.",
    "O GMV médio diário em {where} caiu R$ {gmv_recent} por dia.",
    # review #5: the 30-day projection and the measured window deviation swapped
    "Em {where}, o desvio de GMV acumulado na janela de comparação é de no mínimo R$ {projection}.",
    "Em {where}, o GMV projetado para 30 dias acumula no mínimo R$ {money} abaixo da baseline.",
    # review #5: a scope named once, then a company figure under a pronoun
    "Na {where}, a taxa de conclusão caiu {c}%. Nessa zona, o GMV caiu {g_co}%.",
    # review #5: an integer cannot stand for a 0..1 score, nor a denial for a cause
    "A confiança do diagnóstico em {where} é 1.",
    "Não há dúvida de que o atraso nas entregas causou a queda do GMV em {where}.",
    # review #6: a level as a change, in the forms the adjacent-verb rule missed
    "A queda da taxa de conclusão em {where} foi de {c_recent}%.",
    "A variação da taxa de conclusão em {where} foi de {c_recent}%.",
    "O aumento da taxa de cancelamento em {where} foi de {x_recent}%.",
    "A queda do GMV médio diário em {where} foi de R$ {gmv_recent} por dia.",
    "A taxa de conclusão em {where} caiu quase {c_recent}%.",
    "A taxa de conclusão em {where} caiu mais de {c_recent}%.",
    "A taxa de conclusão em {where} registrou uma retração de {c_recent}%.",
    "A taxa de conclusão em {where} caiu {c_recent}% para o menor nível do período.",
    # review #6: a zone figure under a company-wide phrase after a zone was named
    "Na {where}, a taxa de conclusão caiu {c}%. No geral, o GMV caiu {g}%.",
    "Na {where}, a taxa de conclusão caiu {c}%. Globalmente, o GMV caiu {g}%.",
    # review #6: "porque", and period lengths quoted as if they were the window
    "O GMV em {where} caiu {g}% porque as entregas atrasaram.",
    "O GMV em {where} caiu {g}% em função do atraso nas entregas.",
    "O GMV em {where} caiu {g}% nos últimos 30 dias.",
    "O GMV em {where} caiu {g}% nos últimos 56 dias.",
    "Em {where}, o GMV projetado para os próximos 14 dias acumula no mínimo R$ {projection} abaixo da baseline.",
    "Em {where}, a projeção de perda de GMV na janela de comparação é de R$ {projection}.",
    "O desvio acumulado do projeto em {where} é de R$ {projection} abaixo da baseline.",
)

_NATURAL_TRUE = (
    "O GMV em {where} variou {g}%.",
    "Em {where}, o GMV caiu {g}% nas últimas duas semanas.",
    "O GMV da empresa caiu {g_co}% nas últimas duas semanas.",
    "O GMV caiu {g}% em {where}, contra {g_co}% em Toda a empresa.",
    "A taxa de cancelamento da empresa ({x_co_recent}%) está abaixo da de {where} ({x_recent}%).",
    "A taxa de conclusão da empresa está em {c_co_recent}%, acima da de {where}, que está em {c_recent}%.",
    "O on-time rate de entregas em {where} caiu {o}%.",
    "Em {where}, o GMV e a taxa de conclusão caíram, respectivamente, {g}% e {c}%.",
    "A taxa de conclusão em {where} passou de {c_base}% para {c_recent}%.",
    "Em {where} a taxa de conclusão é {c_recent}%, abaixo dos {c_base}% da baseline.",
    "A taxa de cancelamento em {where} praticamente dobrou, de {x_base}% para {x_recent}%.",
    "A taxa de conclusão em {where} ficou {c}% abaixo da baseline.",
    "Em {where}, o desvio de GMV acumulado na janela é de no mínimo R$ {money}.",
    "Em {where}, 1 prioridade e {customers} clientes estão neste escopo.",
    "Em {where}, a pontuação de impacto é {score} e a confiança do diagnóstico é {confidence}.",
    # review #5 counterparts
    "A taxa de conclusão em {where} caiu de {c_base}% para {c_recent}%.",
    "A taxa de conclusão em {where} caiu para {c_recent}%.",
    "Em {where}, o GMV projetado para 30 dias acumula no mínimo R$ {projection} abaixo da baseline.",
    "Na {where}, o desvio de GMV acumulado na janela é de no mínimo R$ {money}, e o GMV projetado para 30 dias acumula R$ {projection} abaixo da baseline.",
    "Na {where}, a taxa de conclusão caiu {c}%. Nessa zona, o GMV caiu {g}%.",
    # review #6 counterparts
    "A taxa de conclusão em {where} caiu de {c_base}% na baseline para {c_recent}%.",
    "A taxa de cancelamento em {where} subiu de {x_base}% (baseline) para {x_recent}%.",
    "Em {where}, a queda do GMV foi de {g}%.",
    "A taxa de conclusão em {where} caiu quase {c}%.",
    "O GMV da empresa caiu {g_co}% nos 14 dias da janela de comparação.",
    "Na {where}, a taxa de conclusão caiu {c}%. Em Toda a empresa, o GMV caiu {g_co}%.",
)


@pytest.mark.parametrize("template", _NATURAL_FALSE)
def test_review4_a_false_claim_in_plain_wording_is_not_certified(result, template):
    from pulse.copilot import experiment_evidence

    claim = template.format(**_natural_facts(result))
    bundle = build_evidence_bundle(result)
    assert not validate_response(claim, bundle).all_verified, claim


@pytest.mark.parametrize("template", _NATURAL_TRUE)
def test_review4_a_true_claim_in_plain_wording_is_certified(result, bundle, template):
    claim = template.format(**_natural_facts(result))
    report = validate_response(claim, bundle)
    assert report.all_verified, (claim, report.unverified_figures, report.causal_claims)


def test_review4_the_experiment_exception_covers_only_the_experiment(result, report):
    """With an experiment attached, any sentence that merely MENTIONED it was
    exempt from the causal check."""
    from pulse.copilot import experiment_evidence

    bundle = build_evidence_bundle(result, experiment_summary=experiment_evidence(report))
    assert report.causal_language_licensed
    flagged = validate_response(
        "O atraso nas entregas causou a queda do GMV em Zona 7, ao contrário do "
        "que o experimento testou.", bundle)
    assert flagged.causal_claims
    licensed = validate_response(
        "O tratamento do experimento causou um aumento na taxa de recompra.", bundle)
    assert not licensed.causal_claims


# --- the independent benchmark -------------------------------------------------


def _mutations(result):
    """Fabricated claims built by MUTATING engine truth, with the oracle saying
    which mutations are genuinely false.

    Seven mutation families, each attacking one leg of the contract: wrong
    scope, wrong metric, wrong unit, wrong semantic, wrong value, unknown
    concept, swapped values.
    """
    cases: list[tuple[str, str]] = []
    kpis = [r for r in tuple(result.headline_kpis) + tuple(result.segment_kpis)]
    scopes = sorted({(r["scope"], str(r["scope_value"])) for r in kpis})

    for row in kpis:
        metric, scope = row["metric"], (row["scope"], str(row["scope_value"]))
        semantic = METRIC_SEMANTICS[metric]
        delta = abs(row["delta_pct"])
        truthy = _oracle_percent_values(result, metric, scope)

        # wrong value: an integer percent no fact at this (metric, scope) holds
        for n in range(0, 101):
            if any(abs(n - v) <= 0.5 for v in truthy):
                continue
            cases.append(
                ("wrong_value",
                 f"O {_pt(metric)} caiu {n}% em {_scope_phrase(scope)}.")
            )

        # wrong scope: this metric's real delta asserted about another scope
        for other in scopes:
            if other == scope:
                continue
            if any(
                abs(delta - v) <= 0.05
                for v in _oracle_percent_values(result, metric, other)
            ):
                continue
            cases.append(
                ("wrong_scope",
                 f"O {_pt(metric)} caiu {_num(delta, 1)}% em "
                 f"{_scope_phrase(other)}.")
            )

        # wrong metric: this metric's real delta asserted about another metric
        for other_metric in METRIC_SEMANTICS:
            if other_metric == metric:
                continue
            if any(
                abs(delta - v) <= 0.05
                for v in _oracle_percent_values(result, other_metric, scope)
            ):
                continue
            cases.append(
                ("wrong_metric",
                 f"O {_pt(other_metric)} caiu {_num(delta, 1)}% em "
                 f"{_scope_phrase(scope)}.")
            )

        # wrong semantic: a daily average called a window total
        if semantic == DAILY_AVERAGE and _UNITS[metric] == "BRL":
            cases.append(
                ("wrong_semantic",
                 f"O {_pt(metric)} total de {_scope_phrase(scope)} nos 14 dias "
                 f"foi de R$ {_num(row['recent'])}.")
            )
        # wrong unit: a rate's percentage written as currency
        if _UNITS[metric] == "ratio":
            cases.append(
                ("wrong_unit",
                 f"O {_pt(metric)} de {_scope_phrase(scope)} foi de R$ "
                 f"{_num(row['recent'] * 100.0, 1)}.")
            )

    # unknown concept carrying a real magnitude
    for row in kpis[:12]:
        scope = (row["scope"], str(row["scope_value"]))
        for unknown in ("NPS", "churn", "taxa de retenção", "fraude"):
            cases.append(
                ("unknown_concept",
                 f"O {unknown} de {_scope_phrase(scope)} foi "
                 f"{_num(abs(row['delta_pct']), 1)}%.")
            )

    # counts asserted about the wrong scope
    for priority in result.priorities:
        anomaly = priority.diagnosis.anomaly
        scope = (anomaly.scope, str(anomaly.scope_value))
        count = priority.impact.customers_affected
        for other in scopes:
            if other == scope or count in _oracle_count_values(result, other):
                continue
            cases.append(
                ("wrong_scope_count",
                 f"Em {_scope_phrase(other)}, {_num(count, 0)} clientes "
                 f"pediram na janela de comparação.")
            )

    # --- review #3 families: the classes that were outside this set ----------
    zones = sorted({str(a.scope_value) for a in result.anomalies if a.scope == "zone"})
    for anomaly in result.anomalies:
        metric, scope = anomaly.metric, (anomaly.scope, str(anomaly.scope_value))
        where, name = _scope_phrase(scope), _pt(metric)
        size = _num(abs(anomaly.deviation_pct), 1)
        up = anomaly.deviation_pct > 0
        verb = "subiu" if up else "caiu"
        if not any(abs(-anomaly.deviation_pct - v) <= 0.05
                   for v in _oracle_signed_changes(result, metric, scope)):
            cases.append(("direction_inversion",
                          f"A {name} em {where} {'caiu' if up else 'subiu'} {size}%."))
            improving = (not up) == (metric in _ORACLE_LOWER_IS_BETTER)
            cases.append(("polarity_inversion",
                          f"A {name} em {where} {'piorou' if improving else 'melhorou'} "
                          f"{size}%."))
        if _UNITS[metric] == "ratio" and not any(
            abs(abs(anomaly.deviation_pct) - v) <= 0.05
            for v in (anomaly.recent_value * 100, anomaly.baseline_value * 100)
        ):
            cases.append(("deviation_as_level", f"A {name} em {where} é {size}%."))
        recent = _level(metric, anomaly.recent_value)
        base = _level(metric, anomaly.baseline_value)
        levels = _oracle_levels(result, metric, scope)
        if recent and recent != base:
            if base not in levels["recent"]:
                cases.append(("recent_baseline_swap", f"A {name} atual em {where} é {base}."))
                cases.append(("recent_baseline_swap",
                              f"A {name} em {where} {verb} de {recent} para {base}."))
            if recent not in levels["baseline"]:
                cases.append(("recent_baseline_swap",
                              f"A {name} em {where} era {recent} na baseline."))
        for qualifier in ("de fraude", "de entregas expressas", "do app"):
            cases.append(("unknown_concept_phrase",
                          f"A {name} {qualifier} em {where} {verb} {size}%."))
        for other_zone in zones:
            other = ("zone", other_zone)
            if other == scope or any(abs(abs(anomaly.deviation_pct) - v) <= 0.05
                                     for v in _oracle_percent_values(result, metric, other)):
                continue
            cases.append(("conflicting_scope",
                          f"Em Zona {other_zone} e {where}, a {name} {verb} {size}%."))
            cases.append(("conflicting_scope",
                          f"A {name} {verb} {size}% em {where} e em Zona {other_zone}."))

    for priority in result.priorities[:3]:
        anomaly = priority.diagnosis.anomaly
        scope = (anomaly.scope, str(anomaly.scope_value))
        where, impact = _scope_phrase(scope), priority.impact
        if impact.gmv_at_risk_brl <= 0:
            continue
        money = _num(impact.gmv_at_risk_brl, 0)
        for other in METRIC_SEMANTICS:
            if other != "gmv":
                cases.append(("loose_attribution",
                              f"Em {where}, o desvio de {_pt(other)} acumulado na janela "
                              f"é de no mínimo R$ {money}."))
        cases.append(("deviation_as_level",
                      f"Em {where}, o GMV total acumulado na janela foi de R$ {money}."))
        cases.append(("deviation_as_level",
                      f"Em {where}, foram concluídos {_num(impact.orders_lost, 0)} pedidos "
                      f"na janela."))
        for period in ("por hora", "por semana", "por mês"):
            cases.append(("period_synonym",
                          f"Em {where}, o desvio de GMV acumulado é de no mínimo R$ {money} "
                          f"{period}."))
        rate = _num(abs(impact.daily_run_rate_brl), 0)
        wrong_way = "acima" if impact.daily_run_rate_brl < 0 else "abaixo"
        cases.append(("direction_inversion",
                      f"Em {where}, o GMV médio diário ficou R$ {rate} por dia {wrong_way} "
                      f"da baseline."))

    for gmv in (a for a in result.anomalies if a.metric == "gmv"):
        scope = (gmv.scope, str(gmv.scope_value))
        for sibling in result.anomalies:
            if ((sibling.scope, str(sibling.scope_value)) != scope or sibling is gmv
                    or abs(abs(sibling.deviation_pct) - abs(gmv.deviation_pct)) <= 0.05):
                continue
            g_verb = "subiu" if gmv.deviation_pct > 0 else "caiu"
            s_verb = "subiu" if sibling.deviation_pct > 0 else "caiu"
            for separator in (" — ", " / ", " vs ", " ao passo que ", ", enquanto "):
                cases.append(("cartesian_separator",
                              f"Em {_scope_phrase(scope)}, o GMV {g_verb} "
                              f"{_num(abs(sibling.deviation_pct), 1)}%{separator}a "
                              f"{_pt(sibling.metric)} {s_verb} "
                              f"{_num(abs(gmv.deviation_pct), 1)}%."))
    # --- review #5 families ---------------------------------------------------
    for anomaly in result.anomalies:
        metric, scope = anomaly.metric, (anomaly.scope, str(anomaly.scope_value))
        if _UNITS[metric] != "ratio":
            continue
        where, name = _scope_phrase(scope), _pt(metric)
        verb = "subiu" if anomaly.deviation_pct > 0 else "caiu"
        # Only the CHANGES the oracle holds -- a level's own x100 rendering is a
        # percent fact too, and would exclude every case.
        changes = {v for concept, fact_scope, v, kind, semantic in _oracle_facts(result)
                   if concept == metric and fact_scope == scope
                   and kind == ORACLE_PERCENT and semantic == "relative_change"}
        noun = "O aumento" if anomaly.deviation_pct > 0 else "A queda"
        for value in (anomaly.recent_value, anomaly.baseline_value):
            if any(abs(value * 100 - v) <= 0.05 for v in changes):
                continue
            level = _num(value * 100, 1)
            cases.append(("level_as_change", f"A {name} em {where} {verb} {level}%."))
            cases.append(("level_as_change",
                          f"{noun} da {name} em {where} foi de {level}%."))
            cases.append(("level_as_change", f"A {name} em {where} {verb} quase {level}%."))
    company_changes = {
        (r["metric"]): r["delta_pct"] for r in result.headline_kpis}
    for anomaly in result.anomalies:
        if anomaly.scope != "zone" or anomaly.metric not in company_changes:
            continue
        if abs(abs(anomaly.deviation_pct) - abs(company_changes[anomaly.metric])) <= 0.05:
            continue
        verb = "subiu" if anomaly.deviation_pct > 0 else "caiu"
        for phrase in ("No geral", "Globalmente", "No marketplace como um todo"):
            cases.append(("scope_carried_to_company",
                          f"Na {_scope_phrase((anomaly.scope, str(anomaly.scope_value)))}, "
                          f"a análise terminou. {phrase}, a {_pt(anomaly.metric)} {verb} "
                          f"{_num(abs(anomaly.deviation_pct), 1)}%."))
        for days in (P.baseline_window_days, 30):
            cases.append(("day_count_as_window",
                          f"A {_pt(anomaly.metric)} em "
                          f"{_scope_phrase((anomaly.scope, str(anomaly.scope_value)))} {verb} "
                          f"{_num(abs(anomaly.deviation_pct), 1)}% nos últimos {days} dias."))
    for priority in result.priorities[:3]:
        anomaly = priority.diagnosis.anomaly
        where, impact = _scope_phrase((anomaly.scope, str(anomaly.scope_value))), priority.impact
        if impact.gmv_at_risk_brl <= 0:
            continue
        cases.append(("projection_vs_window",
                      f"Em {where}, o desvio de GMV acumulado na janela de comparação é de "
                      f"no mínimo R$ {_num(abs(impact.projected_30d_brl), 0)}."))
        cases.append(("projection_vs_window",
                      f"Em {where}, o GMV projetado para 30 dias acumula no mínimo "
                      f"R$ {_num(impact.gmv_at_risk_brl, 0)} abaixo da baseline."))
    return cases


def _true_claims(result) -> list[str]:
    """The same shapes, stated TRULY. A validator that rejects any of these is
    buying its false-acceptance rate with false rejections."""
    claims: list[str] = []
    for anomaly in result.anomalies:
        metric, scope = anomaly.metric, (anomaly.scope, str(anomaly.scope_value))
        where, name = _scope_phrase(scope), _pt(metric)
        size = _num(abs(anomaly.deviation_pct), 1)
        up = anomaly.deviation_pct > 0
        verb = "subiu" if up else "caiu"
        improving = (not up) == (metric in _ORACLE_LOWER_IS_BETTER)
        claims.append(f"A {name} em {where} {verb} {size}%.")
        claims.append(f"A {name} em {where} {'melhorou' if improving else 'piorou'} {size}%.")
        claims.append(f"A {name} em {where} variou {'+' if up else '-'}{size}%.")
        recent = _level(metric, anomaly.recent_value)
        base = _level(metric, anomaly.baseline_value)
        if recent:
            claims.append(f"A {name} atual em {where} é {recent}.")
            claims.append(f"A {name} em {where} era {base} na baseline.")
            claims.append(f"A {name} em {where} {verb} de {base} para {recent}.")
    for priority in result.priorities[:3]:
        anomaly = priority.diagnosis.anomaly
        where, impact = _scope_phrase((anomaly.scope, str(anomaly.scope_value))), priority.impact
        rate = _num(abs(impact.daily_run_rate_brl), 0)
        way = "abaixo" if impact.daily_run_rate_brl < 0 else "acima"
        claims.append(f"Em {where}, o GMV médio diário ficou R$ {rate} por dia {way} da baseline.")
        if impact.gmv_at_risk_brl > 0:
            claims.append(f"Em {where}, o desvio de GMV acumulado na janela é de no mínimo "
                          f"R$ {_num(impact.gmv_at_risk_brl, 0)}.")
            claims.append(f"Em {where}, o desvio acumulado na janela é de no mínimo "
                          f"R$ {_num(impact.gmv_at_risk_brl, 0)}.")
        claims.append(f"Em {where}, o desvio de margem de contribuição acumulado na janela "
                      f"é de no mínimo R$ {_num(impact.margin_impact_brl, 0)}.")
        if impact.projected_30d_brl < 0:
            claims.append(f"Em {where}, o GMV projetado para 30 dias acumula no mínimo "
                          f"R$ {_num(abs(impact.projected_30d_brl), 0)} abaixo da baseline.")
    return claims


def test_the_independent_benchmark_reports_a_real_false_acceptance_rate(
    result, bundle, capsys
):
    """THE BENCHMARK. Truth comes from `_oracle_facts(result)` — engine objects
    only — and the mutations are constructed against it. Nothing here consults
    the validator's index, its aliases or its tolerance, so the rate this
    measures can move when the validator changes, which is the whole point.
    """
    cases = _mutations(result)
    assert len(cases) > 1500, f"only {len(cases)} mutations generated"

    known = bundle.known_numbers()

    def flat_guard_accepts(text: str) -> bool:
        """The ORIGINAL guard's rule, reproduced here as the measured baseline.

        "Does this magnitude exist anywhere in the bundle?" -- no concept, no
        scope, no unit, no meaning. Kept so the report carries a real BEFORE
        figure on the SAME mutation set, rather than a number produced by a
        different method on a different set.
        """
        for token in _ORACLE_NUMBER_RE.findall(text):
            value = abs(_parse_br(token))
            places = len(token.partition(",")[2])
            tolerance = 0.5 * (10.0**-places) + 1e-9
            if not any(abs(value - k) <= tolerance for k in known):
                return False
        return True

    accepted: dict[str, int] = {}
    totals: dict[str, int] = {}
    baseline_accepted = 0
    for family, claim in cases:
        totals[family] = totals.get(family, 0) + 1
        if flat_guard_accepts(claim):
            baseline_accepted += 1
        if validate_response(claim, bundle).all_verified:
            accepted[family] = accepted.get(family, 0) + 1

    total = sum(totals.values())
    total_accepted = sum(accepted.values())
    rate = total_accepted / total

    lines = [f"{'family':22s} {'n':>6s} {'accepted':>9s} {'rate':>7s}"]
    for family in sorted(totals):
        n, a = totals[family], accepted.get(family, 0)
        lines.append(f"{family:22s} {n:6d} {a:9d} {100 * a / n:6.1f}%")
    lines.append(f"{'TOTAL':22s} {total:6d} {total_accepted:9d} {100 * rate:6.1f}%")
    lines.append(
        f"{'(flat-guard baseline)':22s} {total:6d} {baseline_accepted:9d} "
        f"{100 * baseline_accepted / total:6.1f}%"
    )
    with capsys.disabled():
        print("\n" + "\n".join(lines))

    # The baseline has to stay high, or the mutation set has stopped being
    # adversarial and the 0% below would mean nothing.
    assert baseline_accepted / total > 0.5, (
        f"the flat-guard baseline collapsed to "
        f"{100 * baseline_accepted / total:.1f}%, so these mutations are no "
        f"longer a meaningful adversarial set"
    )
    assert rate <= 0.005, (
        f"independent false-acceptance rate {100 * rate:.2f}% over {total} "
        f"mutations; per family: {accepted}"
    )
    # Every review #3 family is in the set, and none may be accepted at all.
    review3 = {"direction_inversion", "polarity_inversion", "deviation_as_level",
               "recent_baseline_swap", "unknown_concept_phrase", "conflicting_scope",
               "loose_attribution", "period_synonym", "cartesian_separator"}
    assert review3 <= set(totals), review3 - set(totals)
    assert not {family: accepted[family] for family in review3 if family in accepted}
    review5 = {"level_as_change", "projection_vs_window", "scope_carried_to_company",
               "day_count_as_window"}
    assert review5 <= set(totals), review5 - set(totals)
    assert not {family: accepted[family] for family in review5 if family in accepted}

    trues = _true_claims(result)
    rejected = [claim for claim in trues if not validate_response(claim, bundle).all_verified]
    with capsys.disabled():
        print(f"{'(true claims rejected)':22s} {len(trues):6d} {len(rejected):9d}")
    assert len(trues) >= 60
    assert not rejected, rejected[:5]

    # The README quotes this run; the figures it quotes are this run's.
    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    review3_total = sum(totals[family] for family in review3)
    assert f"On the benchmark's {total:,} templated mutations the validator accepts **0**" in readme
    assert f"{len(_NATURAL_FALSE)} false sentences" in readme
    assert f"{len(_NATURAL_TRUE)} true ones" in readme
    assert f"accepts **{round(100 * baseline_accepted / total)}%**" in readme
    assert f"their {review3_total} mutations" in readme
    review5_total = sum(totals[family] for family in review5)
    assert f"of the {review5_total} mutations in the round-5 and round-6 families" in readme
    assert f"It also runs {len(trues)} true claims" in readme


# --- the bundle's own limits ----------------------------------------------------


def test_the_real_experiment_summary_fits_under_the_cap(result, report):
    """The documented injection point must work with the real current payload.
    `asdict(report)` was 4,544 chars and pushed the bundle to 31,831 over a
    30,000 cap — so the advertised path raised, and the suggested Copilot
    question about the experiment could not be answered."""
    from pulse.copilot import MAX_BUNDLE_CHARS, experiment_evidence

    bundle = build_evidence_bundle(
        result, experiment_summary=experiment_evidence(report)
    )
    size = len(bundle.to_json())
    assert size <= MAX_BUNDLE_CHARS, size
    # And with real headroom, not by a hair.
    assert size < MAX_BUNDLE_CHARS * 0.95, size


def test_an_oversized_bundle_raises_an_exception_not_an_assertion(result):
    """`assert` is stripped under `python -O`. A size contract and a
    row-identifier leak guard must not evaporate with an optimisation flag."""
    from pulse.copilot import EvidenceBundleTooLargeError

    with pytest.raises(EvidenceBundleTooLargeError):
        build_evidence_bundle(
            result, experiment_summary={"padding": "x" * 40_000}
        )


def test_a_row_identifier_in_an_injected_summary_raises(result):
    from pulse.copilot import ForbiddenEvidenceKeyError

    with pytest.raises(ForbiddenEvidenceKeyError):
        build_evidence_bundle(
            result, experiment_summary={"customer_id": 4242}
        )


def test_the_validator_has_exactly_one_definition_of_each_helper():
    """Two shadowed copies of a tolerance rule shipped in the previous round.
    One of them was dead, and editing it would have had no effect."""
    source = (ROOT / "src" / "pulse" / "copilot.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = [
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    duplicates = {name for name in names if names.count(name) > 1}
    assert not duplicates, f"shadowed definitions: {duplicates}"


# --- the bundle's evidence contract -------------------------------------------


def test_every_kpi_and_anomaly_row_in_the_bundle_carries_its_semantic(bundle):
    """The gap the C1 regression lived in.

    The round-1 fix put `semantic` on every `.headline()` row, and
    `copilot._kpi_row` then whitelisted fields and dropped it. The only guard
    asserted the SYSTEM PROMPT contained the word "daily_average" — it never
    checked the bundle, so the prompt instructed the model to read a field that
    was not there and nothing failed. This asserts the contents.
    """
    assert bundle.headline_kpis, "no KPI rows to check"
    for row in bundle.headline_kpis:
        assert row.get("semantic") == METRIC_SEMANTICS[row["metric"]], row["metric"]
    assert bundle.detected_anomalies, "no anomaly rows to check"
    for row in bundle.detected_anomalies:
        assert row.get("semantic") == METRIC_SEMANTICS[row["metric"]], row["metric"]
    # And the instruction that depends on it is only honest if the field is
    # present, so assert the pair together rather than either alone.
    from pulse.copilot import SYSTEM_PROMPT

    assert "daily_average" in SYSTEM_PROMPT
    assert '"semantic"' in bundle.to_json()


def test_the_bundle_no_longer_carries_unvalidatable_memo_prose(bundle):
    """The `evidence` prose stated figures that existed nowhere in the bundle as
    structured values, which is why a text-matching exemption had to exist to
    wave them through. It is replaced by the structure it was built from."""
    for entry in bundle.priorities:
        assert "evidence" not in entry
        # ... and the structure that replaced it is there.
        assert "score_breakdown" in entry
        assert isinstance(entry.get("funnel"), list)
        assert isinstance(entry.get("segment_shares"), list)


def test_the_priority_entry_carries_the_residual_flag(bundle):
    """Provenance cannot be derived from a count, so it has to travel."""
    for entry in bundle.priorities:
        assert "customers_are_residual" in entry["impact"]
