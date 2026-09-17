# pulse generate | build | decide | run
#
# SERIALISATION ONLY. This module computes nothing. It loads gold, calls
# run_decision_cycle() -- the single orchestrator -- and writes what comes back
# to disk. There is no analysis here to drift from the library's, which is the
# whole point: tests/test_engine.py compares the numbers in
# artifacts/decisions.json against a direct run_decision_cycle() call at full
# float precision, and that test can only pass while this file stays a writer.
#
# A deliberate consequence: the CLI imports neither detect_anomalies, diagnose,
# estimate_impact, prioritize nor recommend. If one of those names ever appears
# in this file, a second engine has been born. A test asserts they are absent.
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from pulse.config import ARTIFACTS, AS_OF, ROOT
from pulse.decision_memo import json_default, memo_to_dict, memo_to_markdown
from pulse.engine import run_decision_cycle
from pulse.metrics import load_gold
from pulse.types import AnalysisParams, DecisionCycleResult

DECISIONS_JSON = ARTIFACTS / "decisions.json"
DECISION_MEMO_MD = ARTIFACTS / "decision-memo.md"
STREAMLIT_APP = ROOT / "app" / "streamlit_app.py"


def result_to_dict(result: DecisionCycleResult) -> dict:
    """The whole cycle as plain data. A mechanical asdict() of every returned
    object rather than a hand-picked subset: choosing which fields to publish
    would be a judgement call living in the CLI, and the artefact is meant to
    be the engine's output, not a summary of it.
    """
    return {
        "as_of": result.as_of,
        "params": asdict(result.params),
        "headline_kpis": [dict(kpi) for kpi in result.headline_kpis],
        "segment_kpis": [dict(kpi) for kpi in result.segment_kpis],
        "anomalies": [asdict(a) for a in result.anomalies],
        "priorities": [asdict(p) for p in result.priorities],
        "memos": [memo_to_dict(m) for m in result.memos],
    }


def write_artifacts(result: DecisionCycleResult) -> tuple[Path, Path]:
    """Write decisions.json and decision-memo.md. Returns both paths.

    json_default() is not optional: dataclasses.asdict() keeps
    Anomaly.first_detected_date as a datetime.date and headline_kpis arrives
    holding numpy scalars straight out of pandas, and plain json.dumps raises
    on both. A round-trip through json.load is asserted in the tests.
    """
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    DECISIONS_JSON.write_text(
        json.dumps(result_to_dict(result), indent=2, default=json_default),
        encoding="utf-8",
    )
    DECISION_MEMO_MD.write_text(
        "\n\n---\n\n".join(memo_to_markdown(m) for m in result.memos),
        encoding="utf-8",
    )
    return DECISIONS_JSON, DECISION_MEMO_MD


def cmd_generate(_args) -> int:
    """Seeded synthetic generation -> data/bronze/*.parquet."""
    from pulse.data_generator import generate_all

    tables = generate_all()
    for name, df in sorted(tables.items()):
        print(f"bronze {name}: {len(df):,} rows")
    return 0


def cmd_build(_args) -> int:
    """DuckDB over sql/*.sql -> data/silver, data/gold."""
    from pulse.quality import build_gold, build_silver

    report = build_silver()
    print(f"silver: {len(report)} quality checks, "
          f"{int(report['rows_rejected'].sum()):,} rows quarantined")
    build_gold()
    print("gold: built")
    return 0


def cmd_decide(_args) -> int:
    """The decision cycle -> artifacts/decisions.json, decision-memo.md."""
    params = AnalysisParams(as_of=AS_OF)
    result = run_decision_cycle(load_gold(), params)
    decisions, memo = write_artifacts(result)
    print(f"as of {result.as_of}: {len(result.anomalies)} anomalies, "
          f"{len(result.priorities)} priorities, {len(result.memos)} memos")
    for p in result.priorities:
        a = p.diagnosis.anomaly
        print(f"  #{p.rank} {a.scope}/{a.scope_value} {p.diagnosis.pattern} "
              f"score {p.impact_score:.2f} confidence {p.diagnosis.confidence:.0%}")
    print(f"wrote {decisions}")
    print(f"wrote {memo}")
    return 0


def cmd_run(_args) -> int:
    """Launch the Streamlit app against the same engine."""
    if not STREAMLIT_APP.exists():
        print(f"no Streamlit app at {STREAMLIT_APP} yet", file=sys.stderr)
        return 1
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(STREAMLIT_APP)], cwd=ROOT
    )


COMMANDS = {
    "generate": cmd_generate,
    "build": cmd_build,
    "decide": cmd_decide,
    "run": cmd_run,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pulse", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, handler in COMMANDS.items():
        sub.add_parser(name, help=(handler.__doc__ or "").strip().splitlines()[0])
    args = parser.parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
