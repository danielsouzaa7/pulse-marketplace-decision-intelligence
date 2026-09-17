# Tests for the single orchestrator and the CLI that serialises it (Task 14).
#
# ONE ENGINE ONLY is the point of this file. The load-bearing test is
# test_cli_and_library_produce_the_same_result: it runs the CLI in a separate
# process, reads the artefact it wrote, and compares the numbers against a
# direct run_decision_cycle() call at FULL float precision. Rounding before
# comparing would hide exactly the drift the test exists to catch, so nothing
# is rounded.
#
# The other traps pinned here, each one verified rather than hypothetical:
#   * json.dumps raises on engine output as-is -- asdict(Anomaly) keeps a
#     datetime.date, and headline_kpis carries numpy scalars out of pandas. The
#     artefact must round-trip through json.load.
#   * the CLI must contain serialisation only. An import of any analysis module
#     in cli.py is a second engine being born, and is asserted absent from the
#     AST rather than from a string search, because the module comment in
#     cli.py names those functions deliberately.
#   * ground truth must stay verification-only, which is checked by REMOVING it
#     and re-running the engine rather than by searching source text.
from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pytest

from pulse import cli
from pulse.config import GROUND_TRUTH
from pulse.engine import MEMO_LIMIT, run_decision_cycle
from pulse.metrics import load_gold
from pulse.types import AnalysisParams, DecisionCycleResult

P = AnalysisParams(as_of=date(2026, 9, 10))

ROOT = Path(cli.ROOT)
DECISIONS = ROOT / "artifacts" / "decisions.json"
MEMO_MD = ROOT / "artifacts" / "decision-memo.md"

# Modules that compute. The CLI may import none of them: its job is to load
# gold, call the orchestrator and write files.
_ANALYSIS_MODULES = {
    "pulse.anomaly_detection",
    "pulse.root_cause",
    "pulse.prioritization",
    "pulse.playbook",
}


def signature(result: DecisionCycleResult) -> list[tuple]:
    """What the run decided, in a form where anything moving shows up.

    Rank, scope value, impact score at full precision and pattern: a change to
    the ranking, to the scopes selected, to any input of the score, or to the
    classification all land here. Scores are unrounded on purpose -- a leak
    that shifted a score by 1e-9 would still be a leak.
    """
    return [
        (p.rank, p.diagnosis.anomaly.scope_value, p.impact_score, p.diagnosis.pattern)
        for p in result.priorities
    ]


@pytest.fixture(scope="module")
def gold():
    """Loaded at test time, never at import time.

    tests/test_contracts.py rebuilds data/gold mid-suite, and DuckDB's
    aggregation is not bit-identical across rebuilds. A gold snapshot taken at
    collection time and the gold a CLI subprocess reads during the run can
    therefore differ in the last ULP -- which would make the one-engine
    comparison below fail on a difference that has nothing to do with the
    engine. Loading here means the library call and the CLI read the same
    files, which is the only thing that comparison is about.
    """
    return load_gold()


@pytest.fixture(scope="module")
def cli_payload():
    """Run `python -m pulse.cli decide` once and read what it wrote."""
    completed = subprocess.run(
        [sys.executable, "-m", "pulse.cli", "decide"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "wrote" in completed.stdout
    return json.loads(DECISIONS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def result(gold, cli_payload):
    # Depends on cli_payload so the library call and the CLI run back to back
    # against the same on-disk gold.
    del cli_payload
    return run_decision_cycle(gold, P)


# --- the brief's three tests -------------------------------------------------


def test_cycle_returns_ranked_priorities_and_matching_memos(result):
    assert result.anomalies and result.priorities and result.memos
    assert len(result.memos) == min(MEMO_LIMIT, len(result.priorities))
    assert [p.rank for p in result.priorities] == list(
        range(1, len(result.priorities) + 1)
    )
    assert [m.priority for m in result.memos] == [
        p.rank for p in result.priorities[: len(result.memos)]
    ]
    assert result.as_of == P.as_of
    assert result.headline_kpis


def test_cli_decide_writes_both_artifacts(cli_payload):
    assert DECISIONS.exists() and MEMO_MD.exists()
    assert cli_payload["priorities"][0]["rank"] == 1
    assert cli_payload["memos"][0]["priority"] == 1
    assert MEMO_MD.read_text(encoding="utf-8").startswith("# Resumo da decisão")


def test_cli_and_library_produce_the_same_result(cli_payload, result):
    """ONE ENGINE ONLY -- the CLI must not have its own logic.

    Compared at full precision, not rounded: a second implementation that
    agreed to six decimal places would still be a second implementation.
    """
    assert len(cli_payload["priorities"]) == len(result.priorities)
    for from_cli, from_lib in zip(cli_payload["priorities"], result.priorities):
        assert from_cli["impact_score"] == from_lib.impact_score
        assert from_cli["rank"] == from_lib.rank
        assert from_cli["impact"] == asdict(from_lib.impact)
        assert from_cli["diagnosis"]["confidence"] == from_lib.diagnosis.confidence
        assert from_cli["diagnosis"]["pattern"] == from_lib.diagnosis.pattern
        assert from_cli["score_breakdown"] == from_lib.score_breakdown

    assert len(cli_payload["anomalies"]) == len(result.anomalies)
    for from_cli, from_lib in zip(cli_payload["anomalies"], result.anomalies):
        assert from_cli["z_score"] == from_lib.z_score
        assert from_cli["deviation_abs"] == from_lib.deviation_abs
        assert (
            from_cli["first_detected_date"] == from_lib.first_detected_date.isoformat()
        )

    assert [m["memo_id"] for m in cli_payload["memos"]] == [
        m.memo_id for m in result.memos
    ]
    assert [m["evidence"] for m in cli_payload["memos"]] == [
        list(m.evidence) for m in result.memos
    ]


# --- serialisation ------------------------------------------------------------


def test_decisions_json_round_trips_through_json_load(cli_payload):
    """json.dumps raises on engine output without an encoder: asdict(Anomaly)
    keeps first_detected_date as a datetime.date, and headline_kpis holds numpy
    scalars. Both are covered here because both are actually in the artefact.
    """
    reloaded = json.loads(DECISIONS.read_text(encoding="utf-8"))
    assert reloaded["as_of"] == "2026-09-10"
    assert reloaded["params"]["as_of"] == "2026-09-10"
    assert reloaded.keys() == cli_payload.keys()
    for anomaly in reloaded["anomalies"]:
        assert date.fromisoformat(anomaly["first_detected_date"])
    for kpi in reloaded["headline_kpis"]:
        assert isinstance(kpi["recent"], float)
        assert isinstance(kpi["delta_pct"], float)
    for memo in reloaded["memos"]:
        assert isinstance(memo["generated_at"], str)
        assert memo["params"]["as_of"] == "2026-09-10"
        assert memo["human_decision_status"] == "INVESTIGATE"


def test_plain_json_dumps_would_have_failed(result):
    """Regression guard for the encoder itself: without json_default this
    raises, so the encoder is load-bearing rather than decorative.
    """
    with pytest.raises(TypeError):
        json.dumps(asdict(result.anomalies[0]))


def test_artifacts_are_generated_from_engine_output_not_hand_authored(
    cli_payload, result
):
    """Every figure in the markdown must be reachable from the engine objects,
    which is what the JSON is a mechanical dump of.
    """
    from pulse.playbook import _num  # the engine's own pt-BR formatter

    md = MEMO_MD.read_text(encoding="utf-8")
    top = result.priorities[0]
    assert _num(top.impact_score, 2) in md
    assert top.diagnosis.anomaly.first_detected_date.isoformat() in md
    assert cli_payload["memos"][0]["memo_id"] in md
    assert md.count("# Resumo da decisão") == len(result.memos)
    assert "{" not in md and "}" not in md


# --- the CLI owns no analysis -------------------------------------------------


def test_cli_imports_no_analysis_module():
    tree = ast.parse(Path(cli.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert not (imported & _ANALYSIS_MODULES), imported & _ANALYSIS_MODULES
    assert "pulse.engine" in imported


def test_engine_is_the_only_place_the_pipeline_is_composed():
    """Nothing outside engine.py may chain detection -> diagnosis -> impact ->
    prioritisation. A second composition is a second engine even when it reuses
    the same functions.
    """
    composers = []
    for path in (ROOT / "src" / "pulse").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        if {"pulse.anomaly_detection", "pulse.root_cause"} <= names:
            composers.append(path.name)
    assert composers == ["engine.py"], composers


# --- ground truth is verification-only ---------------------------------------


def test_engine_produces_identical_output_with_ground_truth_removed(gold):
    """data/ground_truth/ records what the generator deliberately injected. It
    exists so tests can check the engine independently rediscovered it, and no
    module under src/pulse/ may read it.

    Checked BEHAVIOURALLY, by taking the directory away and re-running: if any
    production module reads it, removing it changes the answer. The text search
    this replaces cannot tell apart defining the path (config.py), writing the
    files (incidents.py, quality.py, data_generator.py), naming the rule in a
    comment (anomaly_detection.py, root_cause.py) and actually reading the
    data -- it needed a whitelist entry for each of the first four and still
    produced a false positive on the fifth. This test cannot be satisfied by a
    comment, and it has been mutation-checked: making engine.py read the
    injected zone and re-rank on it makes this test fail.

    Two details keep it from being vacuous. The directory is asserted GONE
    before the second run, so a move that silently failed proves nothing. And
    the restore is in a finally, so a failure here cannot leave the repository
    broken for every test that runs afterwards.
    """
    assert GROUND_TRUTH.exists(), "premise: ground truth must be present to remove it"
    before = signature(run_decision_cycle(gold, P))

    hidden = GROUND_TRUTH.parent / "_gt_hidden_for_test"
    shutil.move(GROUND_TRUTH, hidden)
    try:
        assert not GROUND_TRUTH.exists()  # prove the premise
        after = signature(run_decision_cycle(load_gold(), P))
    finally:
        shutil.move(hidden, GROUND_TRUTH)

    assert before == after
    assert GROUND_TRUTH.exists()  # restored
    assert not hidden.exists()


# --- narration changes nothing at the cycle level -----------------------------


def test_a_narrator_changes_no_number_in_the_whole_cycle(gold, result):
    class StubNarrator:
        def narrate(self, memo):
            return "Entirely different prose about the same facts."

    narrated = run_decision_cycle(gold, P, narrator=StubNarrator())
    assert narrated.anomalies == result.anomalies
    assert narrated.priorities == result.priorities
    assert [m.narrative for m in narrated.memos] == [
        "Entirely different prose about the same facts."
    ] * len(narrated.memos)
    assert all(m.narrative is None for m in result.memos)
    for a, b in zip(narrated.memos, result.memos):
        assert a.impact == b.impact
        assert a.evidence == b.evidence
        assert a.recommended_action == b.recommended_action
        assert a.confidence == b.confidence


def test_cycle_runs_offline_with_no_api_key(monkeypatch, gold, result):
    """The memo must be fully generated with no ANTHROPIC_API_KEY present.
    Nothing in the engine builds a client, so this is a property, not a mock.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert run_decision_cycle(gold, P).memos[0].evidence == result.memos[0].evidence
