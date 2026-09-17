# Shared session state for every PULSE page: the cached loaders, the analysis
# parameters, and the page setup call.
#
# It exists because a multipage app can load the same data eight times and
# recompute the same decision cycle on every page switch. Two Streamlit traps
# are handled here once, for all pages:
#
#   1. GOLD IS NOT BIT-STABLE ACROSS A REBUILD. DuckDB's aggregation order
#      moves some floats by a couple of ULPs between builds, so gold is loaded
#      ONCE into a cached resource and passed through. Re-reading it per widget
#      interaction would let the numbers drift mid-session, and drift ACROSS
#      pages would be worse: two pages of one app disagreeing.
#   2. THE CACHE MUST INVALIDATE WHEN A PARAMETER MOVES. cycle() is keyed on
#      the AnalysisParams VALUES, not on a params object, so the cache key is
#      exactly the set of knobs the sidebar exposes and moving the sensitivity
#      slider recomputes. A stale result behind a changed parameter would make
#      the live-parameter demo a lie.
#
# NOTHING HERE COMPUTES. It loads, it caches, and it hands back what the engine
# and the gold tables already contain. The parameter NAMES stay English because
# they are the engine's own AnalysisParams fields; only the labels a reader sees
# are Portuguese, and those come from this module's widget calls.
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # `streamlit run app/streamlit_app.py` puts app/ on sys.path, not the repo
    # root, so `from app.components import ...` needs this.
    sys.path.insert(0, str(ROOT))

from pulse.config import AS_OF, BRONZE, GOLD, SILVER  # noqa: E402
from pulse.engine import run_decision_cycle  # noqa: E402
from pulse.experiments import analyse_experiment  # noqa: E402
from pulse.io import read_silver  # noqa: E402
from pulse.metrics import load_gold  # noqa: E402
from pulse.types import AnalysisParams  # noqa: E402

from app.theme import inject_css, register_template  # noqa: E402

# The sidebar's defaults, and the values a page uses when it is rendered
# outside the navigation host (a test harness, say) and no sidebar ever ran.
DEFAULTS: dict = {
    "as_of": AS_OF,
    "comparison_window_days": 14,
    "baseline_window_days": 56,
    "sensitivity": 2.5,
    "min_materiality_brl": 5_000.0,
}

_PARAMS_KEY = "pulse_params"


def page_setup() -> None:
    """Template and stylesheet. Idempotent, and called once per page render so
    a page still looks right when it is run on its own.
    """
    register_template()
    inject_css()


@st.cache_resource(show_spinner="Carregando as tabelas Gold…")
def gold():
    """Loaded once per session. See trap 1 in the module header."""
    return load_gold()


@st.cache_data(show_spinner="Executando o ciclo de decisão…")
def cycle(
    as_of: date,
    comparison_window_days: int,
    baseline_window_days: int,
    sensitivity: float,
    min_materiality_brl: float,
):
    """The whole product, behind one call.

    Keyed on the AnalysisParams values rather than on a params object, so the
    cache key is exactly the set of knobs the sidebar exposes (see trap 2).
    """
    params = AnalysisParams(
        as_of=as_of,
        comparison_window_days=comparison_window_days,
        baseline_window_days=baseline_window_days,
        sensitivity=sensitivity,
        min_materiality_brl=min_materiality_brl,
    )
    return run_decision_cycle(gold(), params)


@st.cache_data(show_spinner="Analisando o experimento…")
def experiment(experiment_id: str):
    """pulse.experiments IS the analyser. The Laboratório de Experimentos page
    renders what it returns and reimplements none of it.
    """
    return analyse_experiment(gold(), experiment_id)


@st.cache_data(show_spinner="Lendo o relatório de qualidade da camada Silver…")
def quality_report() -> pd.DataFrame:
    """The real artifact the Silver build wrote. Never a hand-written number."""
    return read_silver("_quality_report")


@st.cache_data(show_spinner="Lendo a quarentena…")
def quarantine(table: str) -> pd.DataFrame:
    """One quarantine file, or an empty frame if the pipeline never wrote one.

    A file with zero rows is not the same as a missing file: the first says the
    check ran and rejected nothing, the second says the check did not run.
    """
    path = SILVER / "_rejected" / f"{table}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _stems(directory: Path) -> tuple[str, ...]:
    if not directory.exists():
        return ()
    return tuple(sorted(p.stem for p in directory.glob("*.parquet")))


def quarantine_tables() -> tuple[str, ...]:
    return _stems(SILVER / "_rejected")


def bronze_tables() -> tuple[str, ...]:
    return _stems(BRONZE)


def gold_tables() -> tuple[str, ...]:
    return _stems(GOLD)


def sidebar() -> dict:
    """The analysis parameters, rendered once by the navigation host.

    Every page reads the result through params(), so the knobs are global to
    the session and a page switch does not reset them.
    """
    with st.sidebar:
        st.markdown('<div class="pulse-eyebrow">Parâmetros da análise</div>',
                    unsafe_allow_html=True)
        sensitivity = st.slider(
            "Sensibilidade de anomalia (z)", 1.5, 4.5, 2.5, 0.1,
            key="pulse_sensitivity",
            help="|z| mínimo para que uma métrica seja reportada como anomalia. "
                 "Quanto maior, mais rigoroso.",
        )
        window = st.select_slider(
            "Janela de comparação (dias)", options=[7, 14, 28], value=14,
            key="pulse_window",
            help="A janela recente sobre a qual todo desvio é medido.",
        )
        materiality = st.number_input(
            "Materialidade mínima (R$)", 0, 100_000, 5_000, 1_000,
            key="pulse_materiality",
            help="Piso monetário aplicado às métricas aditivas: um desvio menor "
                 "que este na janela não justifica uma decisão.",
        )
        st.divider()
        st.markdown(
            '<p class="pulse-note">Todo número de decisão no PULSE é produzido '
            "por <code>run_decision_cycle()</code>, a mesma chamada por trás de "
            "<code>pulse decide</code>. Mover um parâmetro reexecuta o motor.</p>",
            unsafe_allow_html=True,
        )
    knobs = {
        "as_of": AS_OF,
        "comparison_window_days": int(window),
        "baseline_window_days": 56,
        "sensitivity": float(sensitivity),
        "min_materiality_brl": float(materiality),
    }
    st.session_state[_PARAMS_KEY] = knobs
    return knobs


def params() -> dict:
    """The knobs the sidebar last set, or the defaults if it never ran."""
    return dict(st.session_state.get(_PARAMS_KEY, DEFAULTS))


def span(knobs: dict) -> tuple[date, date, date]:
    """(analysis_start, recent_start, as_of) — calendar arithmetic on the
    parameters, which is what the engine's own windows are cut from. It
    measures nothing; it only says which dates a chart should show.
    """
    as_of = knobs["as_of"]
    recent_start = as_of - timedelta(days=knobs["comparison_window_days"] - 1)
    analysis_start = recent_start - timedelta(days=knobs["baseline_window_days"])
    return analysis_start, recent_start, as_of


def window_slice(frame: pd.DataFrame, start: date, end: date,
                 column: str = "metric_date") -> pd.DataFrame:
    """Rows inside a date range. A filter, nothing more."""
    dates = pd.to_datetime(frame[column])
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return frame.loc[mask]


def leading_zone(result, zone_ids: list) -> str | None:
    """The zone the engine ranked highest this run, or None if it ranked none.

    NOTHING in this app assumes which zone that is. The membership of the ranked
    list has changed between builds and the priority margin between first and
    second is whatever score_gap_to_next returns on the run, so every page
    renders whatever comes back and no zone is named in any source file.
    """
    known = {str(z) for z in zone_ids}
    for priority in result.priorities:
        anomaly = priority.diagnosis.anomaly
        if anomaly.scope == "zone" and str(anomaly.scope_value) in known:
            return str(anomaly.scope_value)
    return None
