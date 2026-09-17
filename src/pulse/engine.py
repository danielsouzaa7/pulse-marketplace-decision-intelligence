# THE orchestrator. There is exactly one, and this is it.
#
#   run_decision_cycle(gold, params, narrator=None) -> DecisionCycleResult
#
# `pulse decide` calls this and serialises what comes back. Task 15's Streamlit
# calls this and renders what comes back. Neither of them may recompute a
# metric, a deviation, a contribution, a score or a memo line -- that is what
# makes "the CLI and the UI cannot diverge" a structural fact rather than a
# claim, and tests/test_engine.py proves it by comparing a CLI-written artefact
# against a direct library call at full float precision.
#
# This module composes; it does not compute. Every step below already exists
# and is tested in its own module, and nothing here reimplements any part of
# one. If a number is wrong, it is wrong in the module that owns it.
from __future__ import annotations

from pulse.anomaly_detection import detect_anomalies
from pulse.decision_memo import Narrator, build_memo
from pulse.metrics import GoldTables, compute_metrics
from pulse.playbook import recommend
from pulse.prioritization import estimate_impact, prioritize
from pulse.root_cause import diagnose
from pulse.types import AnalysisParams, DecisionCycleResult

# How many priorities get a full memo. A decision surface that hands a human
# ten memos has not prioritised anything; three is what the product promises
# and what the Streamlit page renders. The full ranked list is returned
# regardless, so nothing is hidden -- only the memo writing is capped.
MEMO_LIMIT = 3


def run_decision_cycle(
    gold: GoldTables,
    params: AnalysisParams,
    narrator: Narrator | None = None,
) -> DecisionCycleResult:
    """Gold tables in, ranked priorities and Decision Memos out.

    metrics -> anomalies -> diagnosis -> impact -> priority -> recommendation
    -> memo, in that order, each step calling the module that owns it.

    `narrator` is threaded through to build_memo and can only ever fill the
    memo's `narrative` field; passing one changes no fact, no number and no
    recommendation anywhere in the result. Default None means the whole cycle
    runs offline with no API key, which is the state this project must work in.
    """
    metric_frame = compute_metrics(gold, params)
    anomalies = detect_anomalies(metric_frame, params)

    pairs = [
        (diagnosis, estimate_impact(diagnosis, gold, params))
        for diagnosis in (diagnose(anomaly, gold, params) for anomaly in anomalies)
    ]
    priorities = prioritize(pairs, params)

    memos = tuple(
        build_memo(
            priority,
            recommend(priority.diagnosis, priority.impact),
            params,
            narrator=narrator,
        )
        for priority in priorities[:MEMO_LIMIT]
    )

    # KPI rows at each memo'd priority's own scope, deduplicated and in rank
    # order. run_decision_cycle is the only place allowed to ask for these: a
    # page that called MetricFrame.headline() itself would be a second engine.
    scopes: list[tuple[str, str]] = []
    for priority in priorities[:MEMO_LIMIT]:
        anomaly = priority.diagnosis.anomaly
        if (anomaly.scope, anomaly.scope_value) not in scopes:
            scopes.append((anomaly.scope, anomaly.scope_value))

    return DecisionCycleResult(
        as_of=params.as_of,
        params=params,
        anomalies=tuple(anomalies),
        priorities=tuple(priorities),
        memos=memos,
        headline_kpis=tuple(metric_frame.headline(params)),
        segment_kpis=tuple(
            row
            for scope, scope_value in scopes
            for row in metric_frame.headline(params, scope, scope_value)
        ),
    )
