# Tests for the decision surface (Tasks 15, 21, 22, 26).
#
# The load-bearing one is test_app_computes_no_analytics: if a business figure
# can be computed in app/, a page and `pulse decide` can disagree, and the
# product's central claim ("the CLI and the UI cannot diverge") stops being a
# structural fact. It scans every source under app/, so a new page is covered
# the moment it exists -- and test_every_page_script_is_scanned asserts exactly
# that, so the multipage split cannot quietly create an unscanned surface.
# Reshaping for display -- a pivot, a sort, a dict(), a filter -- is explicitly
# still allowed, which is asserted here too so the scanner cannot become a ban
# on presentation code.
#
# The second load-bearing group is the rendering smoke test. Every page in the
# navigation is rendered through Streamlit's own AppTest harness and asserted
# to raise nothing, including the pages that hit the None-segment and
# empty-funnel cases. A page that has never been rendered has not been tested.
#
# The third group is the localisation boundary (Task 26). The product speaks
# Brazilian Portuguese; the engine speaks identifiers. Both halves are asserted:
# app/ui_text.py covers every metric the UI exposes, the formatters produce
# Brazilian output without touching the underlying value, and the honesty
# guarantees -- association not causation, "ocupa a primeira posição" not
# "domina", a null result that stays a null result -- are checked in BOTH
# languages, because a translation is exactly where a claim gets quietly
# strengthened.
from __future__ import annotations

import ast
import re
from datetime import date
from pathlib import Path

import pytest

from pulse.decision_memo import (
    EVIDENCE_ANCHOR_IMPACT_SCORE,
    EVIDENCE_ANCHOR_PLAYBOOK_ENTRY,
)
from pulse.metrics import DAILY_AVERAGE
from pulse.types import (
    AnalysisParams,
    Anomaly,
    AssociatedDriver,
    Diagnosis,
    Impact,
    Priority,
)

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
PAGES = APP / "views"

# Computing any of these in a rendering layer means a second implementation of
# a number the engine already owns.
BANNED = (
    ".mean()",
    ".std()",
    ".median()",
    ".sum()",
    ".max()",
    ".min()",
    ".agg(",
    ".corr(",
    ".rolling(",
    ".cumsum(",
    ".quantile(",
    "value_counts(",
    "groupby(",
    "pivot_table(",          # .pivot() reshapes; pivot_table AGGREGATES
    "detect_anomalies",
    "diagnose(",
    "prioritize(",
    "estimate_impact",
    # The near miss no other token catches: compute_metrics(gold, params)
    # .headline(scope, value) would let a page derive its own KPI rows at any
    # scope, which is a second engine wearing a rendering layer's clothes. The
    # engine returns segment_kpis precisely so the page never needs to.
    "compute_metrics",
)

# Modules that compute. app/ may import none of them -- the same rule
# tests/test_engine.py holds the CLI to, for the same reason.
ANALYSIS_MODULES = {
    "pulse.anomaly_detection",
    "pulse.root_cause",
    "pulse.prioritization",
    "pulse.playbook",
}

# A specific zone, written into a source file, in either language. The engine's
# ranking is decided by a fraction of a point and the membership of the ranked
# list has changed between builds, so a page that names a zone is a page that
# will eventually be wrong while looking confident. "Zona 7" is legitimate on
# screen only as a value rendered from engine output.
ZONE_LITERAL = re.compile(r"\b(?:zone|zona|zonas|regi[aã]o)[\s_-]*\d", re.IGNORECASE)

# Overstatement, in two flavours and two languages. The first overstates a
# ranking that is decided by a fraction of a point; the second turns an
# association into a cause. The one place a causal claim is licensed is the
# Experiment Lab, and only because randomisation was checked -- and even there
# the wording comes from pulse.experiments, not from a template in app/.
OVERSTATEMENT = re.compile(
    r"\b("
    r"dominates|dominate|dominant lead|clearly leads|caused by|due to|proves|"
    r"proven"
    r"|domina\w*|lideran[çc]a|lidera\w*"
    r"|causad[oa]s? (?:por|pel[oa]s?)|provocad[oa]s? (?:por|pel[oa]s?)"
    r"|devido [aà]o?s?|por causa d[aeo]|prova(?:m|ram)? que|provou que"
    r"|comprova\w*"
    r")\b",
    re.IGNORECASE,
)

# English UI text that must not survive the translation. Deliberately NOT a
# ban on English words: GMV, ROI, SLA, A/B, SQL, BI, API, JSON, CSV, KPI,
# PySpark, Delta, DuckDB, Streamlit, Databricks, Bronze/Silver/Gold and
# "PULSE Copilot" all legitimately stay, and so does every engine identifier.
# These are the specific labels and sentences the product used to show.
ENGLISH_UI = (
    "Orders placed", "Orders completed", "Orders cancelled", "Completion rate",
    "Cancellation rate", "Contribution margin", "Active customers",
    "Retention rate", "Discount rate", "On-time rate", "Average order value",
    "Merchant availability", "Promised ETA", "Actual delivery time",
    "Available hours", "Scheduled open hours", "Cohort size",
    "Retained customers", "Acquisition channel", "Days observed",
    "Decision status", "Download this memo", "Ask about this analysis",
    "Evidence behind this answer", "Analysis parameters",
    "Anomaly sensitivity", "Comparison window", "Minimum materiality",
    "Decision Intelligence", "Operations & Zones", "Customers & Retention",
    "Experiment Lab", "Data Quality & Architecture", "Pulse Copilot",
    "at least ", "vs baseline", "No segment decomposition", "null result",
    "not a measured return", "demonstration of the method", "Reject reason",
    "Quarantine file", "Relationship enforced", "Rule enforced",
    "Priority #", "Estimated 30-day exposure", "Supporting anomalies",
    "Model-narrated", "not AI output",
)

# The evidence-line selector app/streamlit_app.py uses to pull two sentences
# out of memo.evidence by startswith(), imported from decision_memo.py -- the
# module that actually writes them -- rather than retyped here a third time.
# A selector that has to agree across a module boundary and gets typed out
# separately in three files is exactly how it drifts silently: startswith()
# matching nothing renders an empty panel, not an error. Both anchors are
# Portuguese now, the same as everything else the product shows, so there is
# no longer an English exception to carve out of the scan below.
ENGINE_EVIDENCE_ANCHORS = (EVIDENCE_ANCHOR_IMPACT_SCORE, EVIDENCE_ANCHOR_PLAYBOOK_ENTRY)

# What the product must be called, in the language it is delivered in.
EXPECTED_PAGE_TITLES = (
    "Inteligência para Decisão",
    "PULSE Copilot",
    "Operações e Regiões",
    "Parceiros",
    "Clientes e Retenção",
    "Promoções",
    "Laboratório de Experimentos",
    "Qualidade de Dados e Arquitetura",
)

# The metric register, as the brief specifies it. Extended below to everything
# the UI actually exposes, which is asserted separately.
REQUIRED_METRIC_LABELS = {
    "sessions": "Sessões",
    "orders_placed": "Pedidos realizados",
    "orders_completed": "Pedidos concluídos",
    "order_conversion": "Conversão em pedidos",
    "completion_rate": "Taxa de conclusão",
    "cancellation_rate": "Taxa de cancelamento",
    "avg_actual_delivery_minutes": "Tempo médio real de entrega",
    "avg_promised_eta_minutes": "ETA médio prometido",
    "on_time_rate": "Taxa de entregas no prazo",
    "gmv": "GMV",
    "contribution_margin": "Margem de contribuição",
    "active_customers": "Clientes ativos",
    "availability_rate": "Taxa de disponibilidade",
    "retention_rate": "Taxa de retenção",
    "discount_rate": "Taxa de desconto",
    "margin_per_completed_order": "Margem por pedido concluído",
}


def _analytics_tokens(text: str) -> list[str]:
    return [token for token in BANNED if token in text]


def _app_sources() -> list[Path]:
    return sorted(APP.rglob("*.py"))


def _page_scripts() -> list[Path]:
    return sorted(PAGES.glob("*.py"))


def _string_literals(path: Path) -> list[str]:
    """Every string literal in a source file, minus the docstrings.

    Comments and docstrings are developer documentation and stay English --
    they are internal code, not product. Only what can reach a screen is
    scanned, which is what makes the English check specific enough to be
    useful rather than a blanket ban on English words.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)) or not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            docstrings.add(id(first.value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


# --- I4: a magnitude word for a score gap, as a class ------------------------
#
# PULSE defines no threshold policy for when a score gap counts as meaningful,
# so no surface may characterise one in a word. The ban cannot be a list of
# phrases: three literals were banned once and the claim returned in a fourth
# wording. So it is a magnitude adjective found within _GAP_WINDOW characters
# of score-gap vocabulary -- which leaves "uma amostra pequena" alone, because
# nothing about a gap is nearby.

_GAP_MAGNITUDE = re.compile(
    r"(?<![\w-])(?:"
    r"estreit[oa]s?|ampl[oa]s?|pequen[oa]s?|grandes?|clar[oa]s?|claramente"
    r"|enormes?|folgad[oa]s?|confort[aá]ve(?:l|is)|esmagador[ae]s?|dominantes?"
    r"|narrow|wide|slim|commanding|decisive|dominant|clearly"
    r")(?![\w-])",
    re.IGNORECASE,
)
# What makes the adjective a claim about the RANKING rather than about a sample,
# an effect or a grid. Deliberately excludes "margem" (contribution margin is
# all over this codebase) and "ranquear" (the driver-correlation spread sentence
# in decision_memo.py is policy-backed, gated on _CLUSTER_SPREAD, and is not
# about the priority-score gap).
_GAP_SUBJECT = re.compile(
    r"(?<![\w-])(?:"
    r"vantage(?:m|ns)|dist[aâ]ncias?|diferen[cç]as?|lideran[cç]as?"
    r"|dianteiras?|gaps?|leads?|advantages?"
    r"|classifica\w*|ordena[cç]\w*|prioridades?|pontua[cç]\w*|score_gap"
    r"|ranking|primeiro lugar|primeira posi[cç][aã]o"
    r")(?![\w-])",
    re.IGNORECASE,
)
_GAP_WINDOW = 70


def _screen_chunks(path: Path) -> list[str]:
    """The text this file can put on a screen, in renderable pieces.

    Markdown is scanned whole. For Python, comments and docstrings are excluded
    for the same reason the English scanner excludes them -- they are developer
    documentation, and app/components.py legitimately quotes the removed word
    while explaining why it was removed. An f-string's literal parts are joined
    into one chunk, because a sentence split across source lines is one sentence
    on the screen.
    """
    if path.suffix != ".py":
        return [path.read_text(encoding="utf-8")]
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)) or not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            docstrings.add(id(first.value))
    grouped: set[int] = set()
    chunks: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        parts = []
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                grouped.add(id(sub))
                parts.append(sub.value)
        chunks.append(" ".join(parts))
    chunks += [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
        and id(node) not in grouped
    ]
    return chunks


def _gap_magnitude_in(text: str) -> list[str]:
    """Every window in `text` where a magnitude word sits beside gap vocabulary."""
    flat = re.sub(r"\s+", " ", text)
    found = []
    for match in _GAP_MAGNITUDE.finditer(flat):
        window = flat[max(0, match.start() - _GAP_WINDOW) : match.end() + _GAP_WINDOW]
        if _GAP_SUBJECT.search(window):
            found.append(window)
    return found


def _gap_magnitude_hits(path: Path) -> list[str]:
    return [hit for chunk in _screen_chunks(path) for hit in _gap_magnitude_in(chunk)]


def _gap_scanner_sources() -> list[Path]:
    return (
        sorted(APP.rglob("*.py"))
        + sorted((ROOT / "src" / "pulse").rglob("*.py"))
        + [ROOT / "README.md"]
    )


# --- the rule ----------------------------------------------------------------


def test_app_computes_no_analytics():
    offenders = {
        str(path): _analytics_tokens(path.read_text(encoding="utf-8"))
        for path in _app_sources()
    }
    assert _app_sources(), "no app sources found to scan"
    assert {p: t for p, t in offenders.items() if t} == {}


def test_reshaping_for_display_is_still_allowed():
    """The scanner bans recomputation, not presentation."""
    for allowed in (
        "frame.pivot(index='metric', columns='scope')",
        "sorted(contributions, key=lambda c: c.rank)",
        "[dict(row) for row in rows]",
        "priority.diagnosis.drivers",
        "run_decision_cycle(gold, params)",
        "frame.sort_values('availability_rate').head(15)",
        "report.drop_duplicates(subset='table', keep='first')",
        "frame.loc[mask].rename(columns={'gmv': 'GMV'})",
        "len(frame.loc[frame['reject_reason'] == reason])",
        "min(test.control_n, test.treatment_n)",
        # Localisation is presentation: a relabel and a strftime on a copy.
        "frame.rename(columns=COLUMN_LABELS)",
        "out[column] = out[column].dt.strftime('%d/%m/%Y')",
        "f['promo_type'].map(ui.promo_type_label)",
    ):
        assert _analytics_tokens(allowed) == [], allowed
    assert _analytics_tokens("series.mean()") == [".mean()"]
    assert _analytics_tokens("frame.groupby('zone')") == ["groupby("]
    assert _analytics_tokens("frame.pivot_table(aggfunc='mean')") == ["pivot_table("]


def test_app_imports_no_analysis_module():
    """Asserted from the AST rather than from a string search: an import is the
    one way a page could reach an analysis function without naming it inline.
    """
    for path in _app_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        assert not (imported & ANALYSIS_MODULES), (path, imported & ANALYSIS_MODULES)


def test_app_never_reads_ground_truth():
    """Ground truth exists to verify recovery. A page that can read it can
    display the generator's answer as analysis.
    """
    for path in _app_sources():
        assert "ground_truth" not in path.read_text(encoding="utf-8").lower(), path


def test_app_module_imports_cleanly():
    """Importing must not run the page: Streamlit executes the entry script as
    "__main__", so the navigation host is guarded and an import is
    side-effect-free.
    """
    import app.streamlit_app as page

    assert callable(page.main)
    assert page.EVIDENCE_METRICS[:2] == ("orders_placed", "orders_completed")


# --- the multipage surface ---------------------------------------------------


def test_no_directory_beside_the_entry_point_is_auto_discovered_as_pages():
    """Streamlit auto-discovers a directory named `pages` next to the entry
    script. A browser refresh or deep link on /copilot then ran that script
    DIRECTLY: st.navigation never ran, the sidebar listed raw English filenames
    and the analysis parameters were gone (review #3 -- three README screenshots
    were captured in exactly that state). Page scripts live under another name.
    """
    assert not (APP / "pages").exists()
    assert PAGES.is_dir() and PAGES.name != "pages"


def test_every_page_script_is_registered():
    """A script in app/views/ that no navigation group lists is a page nobody
    can reach and nobody is rendering in the smoke test below.
    """
    import app.streamlit_app as page

    assert set(page.page_files()) == set(_page_scripts())
    assert len(_page_scripts()) == 7


def test_every_page_script_is_scanned_by_the_analytics_rule():
    """The scanner globs app/**; this asserts the pages actually land in it, so
    the multipage split cannot create a surface the rule does not reach.
    """
    scanned = set(_app_sources())
    for script in _page_scripts():
        assert script in scanned, script


def test_navigation_groups_keep_their_english_identifiers():
    """The group KEY is an identifier and orders the sidebar; the group LABEL is
    product text. Keeping them apart is why a rename is a translation and not a
    restructuring.
    """
    import app.streamlit_app as page
    from app.ui_text import GROUP_LABELS

    assert tuple(page.PAGE_GROUPS) == ("Command", "Analytics", "Platform")
    assert set(page.PAGE_GROUPS) == set(GROUP_LABELS)


def test_navigation_is_brazilian_portuguese():
    """The eight pages, named as the product names them. PULSE stays uppercase."""
    import app.streamlit_app as page
    from app.ui_text import GROUP_LABELS, PAGE_TITLES

    assert tuple(PAGE_TITLES.values()) == EXPECTED_PAGE_TITLES
    assert tuple(GROUP_LABELS.values()) == ("Comando", "Análises", "Plataforma")

    # Every registered script has a title, and the default page has one too.
    for filenames in page.PAGE_GROUPS.values():
        for filename in filenames:
            assert filename in PAGE_TITLES, filename
    assert "decision" in PAGE_TITLES
    assert "PULSE Copilot" in PAGE_TITLES.values()
    assert "Pulse Copilot" not in PAGE_TITLES.values()


def test_no_page_names_a_specific_zone():
    """Nothing may assume which zone ranks first, in either language.

    The margin between the top two priorities is a fraction of a point and the
    ranked list's membership has changed between Phase 2 builds. Every page
    renders whatever the engine ranks first; a zone written into a heading, a
    template or a default would survive the run that makes it false.
    """
    offenders = {
        str(path): ZONE_LITERAL.findall(path.read_text(encoding="utf-8"))
        for path in _app_sources()
    }
    assert {p: hits for p, hits in offenders.items() if hits} == {}
    # The regex has to actually catch both languages, or it is decoration.
    assert ZONE_LITERAL.search("Zone 7") and ZONE_LITERAL.search("Zona 7")
    assert not ZONE_LITERAL.search("zone_id") and not ZONE_LITERAL.search("Zona {z}")


def test_no_page_overstates_the_ranking_or_the_evidence():
    offenders = {
        str(path): OVERSTATEMENT.findall(path.read_text(encoding="utf-8"))
        for path in _app_sources()
    }
    assert {p: hits for p, hits in offenders.items() if hits} == {}


def test_the_overstatement_scanner_catches_both_languages():
    """A translated claim is exactly where a hedge gets dropped, so the scanner
    that guards the English wording has to guard the Portuguese one too.
    """
    for banned in (
        "Zone A dominates the ranking",
        "the drop was caused by the ETA",
        "GMV fell due to cancellations",
        "esta zona domina a classificação",
        "a queda foi causada pelo ETA",
        "o GMV caiu devido aos cancelamentos",
        "o teste provou que o incentivo funciona",
        "o experimento comprova o efeito",
        "este escopo lidera com ampla vantagem",
    ):
        assert OVERSTATEMENT.search(banned), banned
    for allowed in (
        "ocupa a primeira posição",
        "tem a maior prioridade no cenário atual",
        "uma associação relatada como evidência",
        "observado junto ao padrão diagnosticado",
        "consistente com a degradação do prazo de entrega",
        "concentrado em um escopo",
        "uma leitura causal do tratamento",
        "nunca uma causa demonstrada",
    ):
        assert not OVERSTATEMENT.search(allowed), allowed


def test_no_english_ui_label_survives_in_app_sources():
    """The product is delivered in Brazilian Portuguese.

    Scanned from the AST so comments and docstrings -- which are internal code
    documentation and stay English -- are excluded, and with a deliberate
    allowlist so the terms that legitimately remain (GMV, ROI, SQL, JSON,
    DuckDB, Streamlit, Bronze/Silver/Gold, PULSE Copilot, and every engine
    identifier) are not flagged. The two evidence-line anchors app/ matches by
    startswith() (ENGINE_EVIDENCE_ANCHORS) are themselves Portuguese now, so
    unlike before there is no English exception left to carve out here --
    test_the_evidence_anchors_are_shared_not_retyped below covers them.
    """
    offenders: dict[str, list[str]] = {}
    for path in _app_sources():
        for literal in _string_literals(path):
            hits = [phrase for phrase in ENGLISH_UI if phrase in literal]
            if hits:
                offenders.setdefault(str(path), []).extend(hits)
    assert offenders == {}


def test_ai_is_written_ia_in_user_text():
    """"AI" is an English abbreviation; the product says IA. The engine's own
    identifiers are untouched -- this scans what a reader can see.
    """
    offenders = {
        str(path): [s for s in _string_literals(path) if re.search(r"\bAI\b", s)]
        for path in _app_sources()
    }
    assert {p: hits for p, hits in offenders.items() if hits} == {}


def test_the_evidence_anchors_are_shared_not_retyped():
    """The prefix each evidence sentence opens with used to be a bare string
    literal typed out separately in decision_memo.py, streamlit_app.py and
    tests/test_app.py -- three places a selector that has to agree across a
    module boundary could quietly drift apart, and a startswith() selector
    that stops matching fails silently (an empty panel, not an exception).
    All three now import EVIDENCE_ANCHOR_IMPACT_SCORE /
    EVIDENCE_ANCHOR_PLAYBOOK_ENTRY from decision_memo.py, the module that
    writes the sentences, instead of retyping the text.
    """
    host = (APP / "streamlit_app.py").read_text(encoding="utf-8")
    for name in ("EVIDENCE_ANCHOR_IMPACT_SCORE", "EVIDENCE_ANCHOR_PLAYBOOK_ENTRY"):
        assert name in host, name
    assert "from pulse.decision_memo import" in host
    # And the retired English literals do not survive anywhere under app/.
    for path in _app_sources():
        text = path.read_text(encoding="utf-8")
        assert '"Impact score' not in text, path
        assert '"Playbook entry' not in text, path


def test_decision_states_are_session_only_and_execute_nothing():
    text = (APP / "components.py").read_text(encoding="utf-8")
    assert "status_pill" in text
    assert "decision_state_control" in text
    assert "st.session_state" in text
    for banned in ("requests.post", "httpx.post", "subprocess", "os.system"):
        assert banned not in text, "human-in-the-loop must not execute actions"


def test_decision_states_reuse_the_engines_own_state_machine():
    """The states a human may choose are the engine's, imported, not a second
    list that could drift out of step with the memo's own wording.
    """
    from pulse.decision_memo import DECISION_STATUSES

    assert DECISION_STATUSES == ("INVESTIGATE", "APPROVED", "REJECTED")
    text = (APP / "components.py").read_text(encoding="utf-8")
    assert "from pulse.decision_memo import DECISION_STATUSES" in text


def test_decision_status_labels_are_display_only():
    """INVESTIGATE -> Investigar on screen, INVESTIGATE in the memo.

    Translating the VALUE would mean a memo written by `pulse decide` and a
    decision recorded in the browser no longer refer to the same state.
    """
    from pulse.decision_memo import DECISION_STATUSES

    from app.components import decision_state_control, decision_status_label
    from app.ui_text import DECISION_STATUS_LABELS

    assert DECISION_STATUS_LABELS == {
        "INVESTIGATE": "Investigar",
        "APPROVED": "Aprovado",
        "REJECTED": "Rejeitado",
    }
    assert tuple(DECISION_STATUS_LABELS) == DECISION_STATUSES
    for status in DECISION_STATUSES:
        assert decision_status_label(status) != status
    # The control stores the engine's value and only formats the button.
    source = (APP / "components.py").read_text(encoding="utf-8")
    assert "options=list(DECISION_STATUSES)" in source
    assert "format_func=decision_status_label" in source
    assert callable(decision_state_control)


def test_experiment_lab_reports_no_measured_roi_at_all():
    """The statistics are a measured result. There is no measured ROI.

    The earlier wording called the ROI an "estimativa ilustrativa", which still
    reads as a weaker measurement of the same quantity. It is not a measurement:
    the dataset carries no incentive-value column, the only cost-shaped quantity
    in it is the ordinary discount both arms carry, and the CONTROL arm carries
    more of it. So no treatment-specific incremental cost exists to divide by.
    The engine owns the sentence; the page quotes it.
    """
    from pulse.experiments import NO_MEASURED_ROI_NOTE, PROXY_METHOD_NOTE

    from app.ui_text import ROI_PROXY_WARNING, ROI_UNAVAILABLE_HEADING

    assert "Nenhum ROI de tratamento medido é reportado" in ROI_UNAVAILABLE_HEADING
    assert "não é possível calcular um ROI de tratamento medido" in NO_MEASURED_ROI_NOTE
    assert "CONTROLE carregando mais" in NO_MEASURED_ROI_NOTE
    assert "não representa ROI medido" in PROXY_METHOD_NOTE
    # The contrast is the point: the statistics are NOT a proxy.
    assert "não é proxy" in ROI_PROXY_WARNING
    page = (PAGES / "experiment_lab.py").read_text(encoding="utf-8")
    assert "ROI_UNAVAILABLE_HEADING" in page
    assert "NO_MEASURED_ROI_NOTE" in page
    # And no surface claims an illustrative ROI is a weaker measured one.
    for text in (ROI_PROXY_WARNING, ROI_UNAVAILABLE_HEADING, PROXY_METHOD_NOTE):
        assert "Estimativa ilustrativa de ROI" not in text


# --- the localisation register -----------------------------------------------


def test_metric_labels_match_the_specified_register():
    from app.ui_text import METRIC_LABELS

    for metric, label in REQUIRED_METRIC_LABELS.items():
        assert METRIC_LABELS[metric] == label, metric


def test_metric_label_map_covers_every_metric_the_ui_exposes():
    """Audited against the engine's register and the gold tables themselves, so
    a metric added upstream cannot reach a screen as a bare identifier.

    gold_experiment_results is excluded deliberately: its columns feed
    pulse.experiments and reach the page only through the analyser's own
    report, never as a table header.
    """
    from pulse.metrics import _UNITS, load_gold

    from app.ui_text import METRIC_LABELS

    gold = load_gold()
    exposed = set(_UNITS) | {"repeat_rate"}
    for name in ("daily_business_metrics", "zone_performance",
                 "merchant_performance", "customer_retention",
                 "promotion_performance"):
        exposed |= set(getattr(gold, name).columns)
    missing = sorted(exposed - set(METRIC_LABELS))
    assert missing == [], missing


def test_every_label_is_reached_through_one_module():
    """components re-exports ui_text's helpers rather than restating them: two
    copies of "Taxa de conclusão" is two places to fix a typo and one place to
    miss it.
    """
    from app import components as ui
    from app import ui_text

    for name in ("metric_label", "scope_label", "pattern_label", "fmt_brl",
                 "fmt_pct", "fmt_pp", "fmt_value", "fmt_date",
                 "decision_status_label", "channel_label"):
        assert getattr(ui, name) is getattr(ui_text, name), name
    assert ui.COLUMN_LABELS is ui_text.METRIC_LABELS


def test_engine_vocabularies_render_as_words():
    from app.ui_text import (
        business_verdict_label,
        evidence_strength_label,
        pattern_label,
        scope_label,
        statistical_verdict_label,
    )

    assert scope_label("company", "all") == "Toda a empresa"
    assert scope_label("zone", "3") == "Zona 3"
    assert pattern_label("fulfillment_eta_degradation") == (
        "degradação do prazo de entrega"
    )
    assert pattern_label("supply_availability_gap") == (
        "lacuna de disponibilidade da oferta"
    )
    assert evidence_strength_label("strong") == "forte"
    assert statistical_verdict_label("not_significant") == "não significativo"
    assert business_verdict_label("negative") == "negativo"
    # An unregistered value degrades to something readable rather than raising.
    assert pattern_label("brand_new_pattern") == "brand new pattern"


# --- formatting --------------------------------------------------------------


def test_currency_is_rendered_as_a_magnitude_in_brazilian_notation():
    """Direction belongs in the words, never in the sign: "R$ -18.759,17" inside
    a "no mínimo" sentence reads as a smaller loss, which is backwards.
    """
    from app.components import fmt_brl

    assert fmt_brl(-18759.171428) == "R$ 18.759,17"
    assert fmt_brl(18759.171428) == "R$ 18.759,17"
    assert fmt_brl(0.0) == "R$ 0,00"
    assert fmt_brl(1234567.891) == "R$ 1.234.567,89"


def test_percentages_carry_their_direction_and_a_decimal_comma():
    from app.components import fmt_pct

    assert fmt_pct(4.086) == "+4,09%"
    assert fmt_pct(-11.464620649617483) == "-11,46%"
    assert fmt_pct(-11.4646, places=1) == "-11,5%"
    assert fmt_pct(0.0) == "+0,00%"
    assert fmt_pct(float("nan")) == "n/d"


def test_absolute_effects_are_rendered_in_points_not_percent():
    """A difference between two rates is in percentage POINTS. Writing +0.0105
    as "+1,05%" is how a one-point move gets read as a one-percent move, which
    is the classic way a null is dressed up. The two formatters are separate
    functions precisely so the two units cannot be confused.
    """
    from app.components import fmt_pct, fmt_pp, fmt_ratio

    assert fmt_pp(0.010547281553821453) == "+1,05 p.p."
    assert fmt_pp(-0.006017120165790809) == "-0,60 p.p."
    assert fmt_pp(0.027111683273433714) == "+2,71 p.p."
    assert fmt_pp(float("nan")) == "n/d"

    # The same experiment's absolute and relative figures, side by side: a
    # +1,05 p.p. move is a +3,4% move, and neither may be printed as the other.
    assert fmt_pp(0.010547281553821453) == "+1,05 p.p."
    assert fmt_ratio(0.0344, 1, signed=True) == "+3,4%"
    assert fmt_pct(3.44, places=1) == "+3,4%"
    assert "p.p." not in fmt_pct(3.44)


def test_values_render_in_their_own_units():
    from app.components import fmt_value

    assert fmt_value(29924.765, "BRL") == "R$ 29.924,76"
    assert fmt_value(0.7450695, "ratio") == "74,5%"
    assert fmt_value(31.5618, "minutes") == "31,6 min"
    assert fmt_value(100.0714, "orders") == "100"
    assert fmt_value(12345.6, "sessions") == "12.346"
    # The "/dia" comes from the metric's SEMANTIC now, not from a unit string
    # that spelled it out for one metric only -- see METRIC_SEMANTICS.
    assert fmt_value(473.142857, "customers") == "473"
    assert fmt_value(473.142857, "customers", DAILY_AVERAGE) == "473/dia"
    assert fmt_value(29924.765, "BRL", DAILY_AVERAGE) == "R$ 29.924,76/dia"
    assert fmt_value(float("nan"), "BRL") == "n/d"


def test_dates_are_rendered_the_way_brazil_writes_them():
    from app.components import fmt_date

    assert fmt_date(date(2026, 9, 10)) == "10/09/2026"
    assert fmt_date(date(2026, 1, 2)) == "02/01/2026"


def test_formatting_never_alters_the_underlying_value():
    """Display only. A localised string is produced at render time from a float
    the engine computed; the float itself is never rewritten, which is why an
    analytical frame can never be handed a comma-decimal number.
    """
    from app.components import fmt_brl, fmt_pct, fmt_pp, localise_dates

    raw = -18759.171428
    assert fmt_brl(raw) == "R$ 18.759,17"
    assert raw == -18759.171428                      # untouched by formatting

    # Reading the rendered string back gives the value the engine produced,
    # to the precision the string claims.
    rendered = fmt_pct(-11.464620649617483).rstrip("%").replace(".", "").replace(",", ".")
    assert float(rendered) == pytest.approx(-11.46)
    points = fmt_pp(0.010547281553821453).removesuffix(" p.p.").replace(",", ".")
    assert float(points) == pytest.approx(1.05)

    # localise_dates hands back a COPY; the caller's frame keeps its datetimes.
    import pandas as pd

    frame = pd.DataFrame({"metric_date": pd.to_datetime(["2026-09-10"]), "gmv": [1.5]})
    out = localise_dates(frame)
    assert out["metric_date"].iloc[0] == "10/09/2026"
    assert pd.api.types.is_datetime64_any_dtype(frame["metric_date"])
    assert frame["gmv"].iloc[0] == 1.5


def test_gold_columns_outside_the_engine_register_still_have_a_display_unit():
    """gold carries columns the engine's metric register does not, and a
    retention rate drawn as "0,58" is a retention rate nobody reads.
    """
    from app.components import unit_of

    assert unit_of("gmv") == "BRL"                       # the engine's register
    assert unit_of("retention_rate") == "ratio"          # display-only
    assert unit_of("margin_per_completed_order") == "BRL"
    assert unit_of("merchant_id") == ""


def test_currency_is_escaped_for_streamlit_markdown():
    """Streamlit renders $...$ as LaTeX, which swallows "R$ 4.829,10 contra
    R$ 5.454,00" into a maths span. Display-only: the download button hands
    over the engine's exact bytes so the file matches
    artifacts/decision-memo.md.
    """
    from app.components import escape_dollars

    raw = "R$ 4.829,10 contra R$ 5.454,00"
    assert escape_dollars(raw) == raw.replace("$", chr(92) + "$")
    assert escape_dollars("nada a escapar") == "nada a escapar"


def test_embedded_memo_headings_are_demoted():
    """The memo's own H1 must not outrank the page section it sits inside."""
    from app.components import as_embedded_markdown

    lines = ["# Decision Memo", "text", "## Impact", "more"]
    out = as_embedded_markdown("\n".join(lines))
    assert out.splitlines() == ["### Decision Memo", "text", "#### Impact", "more"]


def test_tone_knows_which_direction_is_bad_news():
    from app.components import tone

    assert tone("orders_placed", 4.086) == "pos"
    assert tone("orders_completed", -11.45) == "crit"
    assert tone("cancellation_rate", 106.628) == "crit"   # up is bad
    assert tone("cancellation_rate", -12.0) == "pos"
    assert tone("avg_actual_delivery_minutes", 23.56) == "crit"
    assert tone("on_time_rate", -10.63) == "crit"
    assert tone("discount_rate", 40.0) == "crit"
    assert tone("gmv", 0.0) == "flat"
    assert tone("gmv", float("nan")) == "flat"


# --- the empty-diagnosis case ------------------------------------------------
#
# primary_segment is None for most anomalies on this dataset and funnel is
# empty for the same ones, because segment shares and the funnel identity are
# only defined for additive metrics. The lower-ranked priorities hit it.


def _bare_priority() -> Priority:
    """A rate anomaly: no segment decomposition, no funnel, one driver."""
    anomaly = Anomaly(
        metric="completion_rate",
        scope="company",
        scope_value="all",
        recent_value=0.879938494802188,
        baseline_value=0.913422818418438,
        deviation_abs=-0.03348432361625,
        deviation_pct=-3.665807656768089,
        z_score=-6.332098094031548,
        direction="drop",
        first_detected_date=date(2026, 8, 28),
        n_observations=14,
    )
    diagnosis = Diagnosis(
        anomaly=anomaly,
        contributions=(),
        primary_segment=None,
        funnel=(),
        funnel_break_stage="completion_rate",
        drivers=(
            AssociatedDriver(
                metric="avg_promised_eta_minutes",
                recent=35.987010360478045,
                baseline=35.5378662179465,
                deviation_pct=1.263846680543606,
                correlation_with_target=-0.7002508770102124,
                temporal_alignment_days=1,
                evidence_strength="strong",
            ),
        ),
        pattern="fulfillment_eta_degradation",
        confidence=0.6600752631030637,
    )
    impact = Impact(
        gmv_at_risk_brl=7015.9575,
        orders_lost=124,
        customers_affected=3916,
        margin_impact_brl=638.79668125,
        daily_run_rate_brl=-501.1398214285573,
        projected_30d_brl=-15034.194642856719,
    )
    return Priority(
        diagnosis=diagnosis,
        impact=impact,
        impact_score=82.63630211867502,
        rank=2,
        score_breakdown={
            "w_gmv": 0.45,
            "w_orders": 0.20,
            "w_customers": 0.15,
            "w_confidence": 0.20,
            "n_gmv": 0.8014316996942972,
            "n_orders": 0.9185185185185185,
            "n_customers": 1.0,
            "n_confidence": 0.6600752631030637,
            "supporting_anomalies": 2.0,
            "nested_groups_netted": 2.0,
            "representative_confidence": 0.6600752631030637,
        },
    )


def test_missing_segment_renders_as_a_statement_not_as_none():
    from app.components import segment_line

    line = segment_line(None)
    assert "None" not in line
    assert "sem decomposição por segmento" in line.lower()


def test_present_segment_is_rendered_as_a_share():
    from pulse.types import SegmentContribution

    from app.components import segment_line

    line = segment_line(
        SegmentContribution(
            dimension="zone",
            segment="7",
            segment_deviation_abs=-625.3057142857151,
            contribution_pct=56.14987959334022,
            rank=1,
        )
    )
    assert "56,15%" in line
    assert "participação, não o total" in line
    assert not OVERSTATEMENT.search(line)


def test_the_ranking_note_never_claims_more_than_an_ordering():
    """"Ocupa a primeira posição" is the strongest claim the arithmetic supports.

    The gap itself is asserted separately, on two genuinely different scores --
    see test_the_ranking_note_states_the_actual_gap_between_two_real_scores.
    """
    from app.components import ranking_note

    note = ranking_note([
        _scored_priority(83.76, rank=1, gap_to_next=22.17),
        _scored_priority(61.59, rank=2, gap_to_next=0.0),
    ])
    assert "ocupa a primeira posição" in note
    assert "maior prioridade no cenário atual" in note
    assert not OVERSTATEMENT.search(note)
    assert "não há diferença de ordenação" in ranking_note(
        [_scored_priority(83.76, rank=1, gap_to_next=0.0)]
    )


def test_empty_diagnosis_renders_through_every_helper():
    from app import components as ui

    priority = _bare_priority()
    assert ui.funnel_waterfall(priority.diagnosis.funnel) is None
    assert ui.contribution_chart(priority.diagnosis.contributions) is None
    assert "None" not in ui.priority_card_html(priority)
    assert ui.driver_rows(priority.diagnosis.drivers)
    assert len(ui.score_rows(priority.score_breakdown)) == 4
    assert ui.impact_rows(priority.impact, 14)
    assert ui.confidence_note(priority)
    assert ui.scope_delta_chart({}, ("gmv",), "vazio") is None
    assert ui.priority_rows([priority])[0]["Posição"] == 2
    assert ui.priority_rows([priority])[0]["Escopo"] == "Toda a empresa"


def test_driver_rows_are_labelled_as_association_not_cause():
    from app.components import driver_rows

    row = driver_rows(_bare_priority().diagnosis.drivers)[0]
    assert set(row) == {
        "Direcionador associado", "Recente", "Linha de base", "Variação",
        "Correlação r", "Defasagem", "Evidência",
    }
    assert row["Direcionador associado"] == "ETA médio prometido"
    assert row["Correlação r"] == "-0,7003"
    assert row["Evidência"] == "forte"


def test_empty_frames_render_as_a_note_rather_than_an_empty_chart():
    import pandas as pd

    from app import components as ui

    empty = pd.DataFrame()
    assert ui.heatmap_chart(empty, "on_time_rate", "t") is None
    assert ui.timeline_chart(empty, "x", "y", "c", "t") is None
    assert ui.scatter_chart(empty, "x", "y", "c", "t") is None


def test_both_confidence_figures_are_named():
    """Two legitimate numbers per priority -- the group's and the
    representative diagnosis's. An unexplained divergence reads as a bug.
    """
    from app.components import confidence_note

    note = confidence_note(_bare_priority())
    assert "grupo" in note.lower()
    assert "0,660" in note


def test_impact_is_worded_as_a_floor():
    from app.components import impact_rows

    rendered = " ".join(row["Valor"] for row in impact_rows(_bare_priority().impact, 14))
    assert "no mínimo" in rendered
    assert "R$ -" not in rendered          # magnitudes, never signed currency
    assert "abaixo da linha de base" in rendered


def test_the_first_signal_wording_is_not_an_incident_start_date():
    """first_detected_date is where the evidence starts INSIDE the window. A
    page that calls it the start of the incident has invented a fact.
    """
    from app.ui_text import FIRST_SIGNAL_NOTE

    filled = FIRST_SIGNAL_NOTE.format(date="28/08/2026")
    assert "primeiro sinal identificado dentro da janela analisada" in filled.lower()
    assert "início do incidente" not in filled.lower()
    assert "não uma afirmação sobre quando o incidente teve início" in filled


# --- rendering ---------------------------------------------------------------
#
# Streamlit's own harness, on every page in the navigation. The entry script
# renders the Decision Intelligence page (it is the default page), and each
# page script is rendered standalone.

RENDER_TARGETS = [APP / "streamlit_app.py"] + _page_scripts()


@pytest.fixture(scope="module")
def rendered():
    """page path -> rendered AppTest, one render each, shared by the tests below.

    The engine's caches are process-global, so the first render pays for the
    gold load and the decision cycle and the rest are nearly free.
    """
    from streamlit.testing.v1 import AppTest

    out = {}
    for path in RENDER_TARGETS:
        harness = AppTest.from_file(str(path), default_timeout=300)
        harness.run()
        out[path.name] = harness
    return out


@pytest.mark.parametrize("name", [p.name for p in RENDER_TARGETS])
def test_every_page_renders_without_exception(rendered, name):
    harness = rendered[name]
    assert not harness.exception, [
        (e.type, e.value) for e in harness.exception
    ]
    assert harness.markdown, f"{name} rendered nothing"


def test_every_page_renders_in_portuguese(rendered):
    """The standing notices are on every page, so their wording is the cheapest
    honest check that a page actually went through the translated chrome.
    """
    from app.ui_text import NOTICE_DECISION_SUPPORT, NOTICE_SYNTHETIC_DATA

    for name, harness in rendered.items():
        blob = "\n".join(m.value for m in harness.markdown)
        assert NOTICE_DECISION_SUPPORT in blob, name
        assert NOTICE_SYNTHETIC_DATA in blob, name


def test_the_decision_page_renders_the_none_segment_case(rendered):
    """primary_segment is None and funnel is empty for most diagnoses on this
    dataset. The page must say so rather than crash or print "None".
    """
    harness = rendered["streamlit_app.py"]
    blob = "\n".join(m.value for m in harness.markdown)
    assert ("Sem decomposição por segmento" in blob
            or "concentra a maior contribuição observada" in blob)
    assert ">None<" not in blob


def test_the_decision_page_dates_the_first_signal_without_overclaiming(rendered):
    blob = "\n".join(m.value for m in rendered["streamlit_app.py"].markdown)
    assert "Primeiro sinal identificado dentro da janela analisada" in blob
    assert "não uma afirmação sobre quando o incidente teve início" in blob


def test_the_decision_page_actually_renders_both_evidence_anchors(rendered):
    """_evidence_line() selects a sentence out of memo.evidence by
    startswith(); if the anchor the page looks for ever drifted even one
    character from the prefix decision_memo.py actually writes, the callout
    and the caption below would both render EMPTY rather than raise, and
    test_every_page_renders_without_exception would not catch it. Rendering
    the real page and checking both anchors are actually on it is the only
    check here that would fail on that regression.
    """
    harness = rendered["streamlit_app.py"]
    markdown_blob = "\n".join(m.value for m in harness.markdown)
    caption_blob = "\n".join(c.value for c in harness.caption)
    # Section 06: the score sentence renders inside a callout (st.markdown).
    assert EVIDENCE_ANCHOR_IMPACT_SCORE in markdown_blob
    # Section 07: the playbook sentence renders via st.caption (it carries
    # backticks the engine wrote; see the comment in streamlit_app.py).
    assert EVIDENCE_ANCHOR_PLAYBOOK_ENTRY in caption_blob


def test_the_copilot_page_labels_an_answer_it_did_not_get_from_a_model(rendered):
    from pulse.copilot import AnthropicNarrator

    if AnthropicNarrator.available():
        pytest.skip("an API key is set; the offline label is not the state under test")
    blob = "\n".join(m.value for m in rendered["copilot.py"].markdown)
    assert "não é saída de IA" in blob
    assert "Nenhum modelo de linguagem está sendo chamado" in blob
    # The badge wording follows what the guard actually guarantees: it DETECTS
    # divergence by pattern and certifies nothing, so a clean run says so.
    assert "Nenhuma divergência detectada" in blob
    assert "ligadas a um fato estruturado" not in blob
    assert "fatos estruturados" in blob


def test_the_copilot_transcript_starts_again_when_a_parameter_moves():
    """Review #5: turns validated against one bundle sat above another bundle's
    JSON after the sidebar moved, looking current."""
    from streamlit.testing.v1 import AppTest

    from app.state import DEFAULTS

    harness = AppTest.from_file(str(PAGES / "copilot.py"), default_timeout=300)
    harness.run()
    (_question, first), = harness.session_state["copilot_history"]
    assert "(14 dias)" in first.period

    harness.session_state["pulse_params"] = {**DEFAULTS, "comparison_window_days": 7}
    harness.run()
    assert not harness.exception
    history = harness.session_state["copilot_history"]
    assert len(history) == 1
    assert "(7 dias)" in history[0][1].period


def test_the_navigation_host_renders_no_raw_gold_table(rendered):
    """A bare expression at the top level of a Streamlit script is DISPLAYED
    ("magic"). The data-gate check once touched the gold tables that way and
    printed both of them above every page -- caught in a screenshot, not a test.
    """
    harness = rendered["streamlit_app.py"]
    for frame in harness.dataframe:
        columns = set(getattr(frame.value, "columns", ()))
        assert not {"metric_date", "sessions", "orders_placed"} <= columns, columns


def test_a_failed_gold_gate_stops_the_app_with_words_not_a_traceback(monkeypatch):
    """Review #4: GoldContractError reached the user as a raw traceback."""
    from streamlit.testing.v1 import AppTest

    from pulse.contracts import GoldContractError

    import app.state as state

    def broken():
        raise GoldContractError("gold_zone_performance: negative values in ['gmv']")

    state.gold.clear()
    monkeypatch.setattr(state, "load_gold", broken)
    try:
        harness = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=120)
        harness.run()
        assert not harness.exception, [(e.type, e.value) for e in harness.exception]
        errors = " ".join(e.value for e in harness.error)
        assert "não passaram na verificação" in errors
        assert "negative values" in errors
    finally:
        state.gold.clear()


def test_the_copilot_badge_never_turns_green_over_nothing_checked():
    """"Todas as 0 figuras ligadas a um fato estruturado" rendered green over an
    empty narration: absence of evidence shown as success. And a causal claim
    has its own red pill, whatever the figures did."""
    from pulse.copilot import ValidationReport

    from app.components import validation_pills

    empty = validation_pills(ValidationReport(True, (), 0))
    assert len(empty) == 1 and "pulse-pill ok" not in empty[0]
    assert "Nenhuma figura numérica" in empty[0]

    # No state of the badge is green. A pattern-based check that found nothing
    # has not proven anything, so it says what it did and not that the answer
    # is right (the product decision after three review rounds kept finding
    # wordings the patterns missed).
    clean = validation_pills(ValidationReport(True, (), 4))
    assert "pulse-pill info" in clean[0]
    assert "Nenhuma divergência detectada nas 4 figuras" in clean[0]
    assert "não é prova" in clean[0]
    for report in (ValidationReport(True, (), 0), ValidationReport(True, (), 4),
                   ValidationReport(False, ("47,3",), 4),
                   ValidationReport(False, (), 4, ("O atraso causou a queda.",))):
        assert not any("pulse-pill ok" in pill for pill in validation_pills(report))

    causal = validation_pills(
        ValidationReport(False, (), 4, ("O atraso causou a queda.",))
    )
    assert any("pulse-pill crit" in pill and "causal" in pill for pill in causal)

    bad = validation_pills(ValidationReport(False, ("47,3",), 4))
    assert "pulse-pill crit" in bad[0] and "1 de 4" in bad[0]

    from pulse.config import AS_OF
    from pulse.copilot import build_evidence_bundle, validate_response
    from pulse.engine import run_decision_cycle
    from pulse.metrics import load_gold
    from pulse.types import AnalysisParams

    bundle = build_evidence_bundle(run_decision_cycle(load_gold(), AnalysisParams(as_of=AS_OF)))
    report = validate_response("Na Zona 9, o GMV caiu 13,5%.", bundle)
    # Never "2 de 1 figuras": everything reported was counted.
    assert len(report.unverified_figures) <= report.figures_checked


def test_the_experiment_page_presents_the_null_as_the_finding(rendered):
    blob = "\n".join(m.value for m in rendered["experiment_lab.py"].markdown)
    assert "Não foi detectado efeito estatisticamente significativo" in blob
    # No evidence of an effect is not evidence of no effect, and the power
    # analysis is what makes the null informative rather than inconclusive.
    assert "ausência de evidência de efeito, e não evidência de ausência" in blob
    assert "resultado nulo com poder estatístico adequado" in blob
    # I3: no measured ROI exists, so the page says that rather than labelling a
    # proxy as a weaker measurement of the same thing.
    assert "Nenhum ROI de tratamento medido é reportado" in blob
    assert "não representa ROI medido" in blob
    assert "não é possível calcular um ROI de tratamento medido" in blob
    # Absolute effect and relative effect are both on the page, in their own
    # units: relative alone is how a small effect gets oversold.
    assert "Efeito absoluto" in blob and "Efeito relativo" in blob
    assert "p.p." in blob


def _section_blob(harness, eyebrow: str, next_eyebrow: str) -> str:
    """Everything a page rendered between one section eyebrow and the next.

    A caveat two sections below the claim it qualifies is not a caveat, so the
    tests that care about where a qualifier sits need to read one section at a
    time rather than the whole page as one blob.
    """
    values = [m.value for m in harness.markdown]
    start = next(i for i, v in enumerate(values) if eyebrow in v)
    end = next(i for i, v in enumerate(values) if next_eyebrow in v)
    assert start < end, (eyebrow, next_eyebrow)
    return "\n".join(values[start:end])


def test_the_power_section_qualifies_the_proxy_breakeven_where_it_states_it(rendered):
    """Statistical power is measured. The break-even threshold is not.

    breakeven_absolute_lift divides by economics.incentive_brl, which is a PROXY
    cost: the dataset has no incentive-value column and the only cost-shaped
    quantity in it is the ordinary discount BOTH arms carry, with the CONTROL
    arm carrying more. The engine's conclusion carries that caveat and so does
    the README; this section used to drop it exactly where it made the claim and
    recover it two sections later, under "Economia incremental".

    So: the MDE may stay a measured figure, and every figure derived from the
    proxy break-even -- the threshold, the required n, the powered-for-the-
    decision verdict -- must be labelled where it appears.
    """
    from pulse.experiments import ECONOMIC_STATUS_ILLUSTRATIVE, PROXY_METHOD_NOTE

    from app.ui_text import economic_status_label, fmt_int, fmt_pp

    from app import state

    report = state.experiment("EXP-001")
    assert report.economic_evaluation_status == ECONOMIC_STATUS_ILLUSTRATIVE
    section = _section_blob(rendered["experiment_lab.py"], "03 · ", "04 · ")

    # (a) the qualifier is in the SAME section as the figure it qualifies.
    assert fmt_pp(report.economics.breakeven_absolute_lift) in section
    assert PROXY_METHOD_NOTE in section
    assert "custo proxy" in section
    # and it is the engine's status field talking, not a hard-coded adjective.
    assert economic_status_label(ECONOMIC_STATUS_ILLUSTRATIVE) in section
    assert "economic_evaluation_status" in (
        PAGES / "experiment_lab.py"
    ).read_text(encoding="utf-8")

    # (b) the unqualified absolute claim is gone. It asserted a measured
    # treatment-cost break-even, which no figure on this page supports.
    assert "grande o bastante para pagar" not in section
    assert "para pagar o incentivo teria sido visível" not in section
    assert "pequeno demais para importar comercialmente" not in section

    # The measured half survives, in the units that make it a power statement:
    # the design could resolve the calculated MDE and did not find an effect.
    assert fmt_pp(report.mde_at_80_power) in section
    assert fmt_int(min(report.test.control_n, report.test.treatment_n)) in section
    assert "80% de poder" in section


def test_the_data_quality_page_groups_the_quarantine_by_reason(rendered):
    """One table can be quarantined by several checks. A per-table total would
    report two different failures with two different fixes as one number.
    """
    frames = [d.value for d in rendered["data_quality.py"].dataframe]
    quarantine = next(f for f in frames if "Motivo da rejeição" in f.columns)
    per_file = quarantine.groupby("Arquivo de quarentena")["Motivo da rejeição"].count()
    assert per_file.max() > 1, "no quarantine file shows more than one reason"

    foreign_keys = next(f for f in frames if "Relação imposta" in f.columns)
    assert (foreign_keys["Rejeitadas"] == 0).any(), (
        "a declared relationship that rejected zero rows must still be shown: "
        "'0 violações encontradas' and 'nunca verificado' must not look the same"
    )


def test_tables_render_dates_in_brazilian_notation(rendered):
    """Every date a table shows goes through the one display copy in
    components.table(), so no page can leak an ISO date into the product.
    """
    frames = [d.value for d in rendered["merchants.py"].dataframe]
    dated = next(f for f in frames if "Data" in f.columns)
    assert re.fullmatch(r"\d{2}/\d{2}/\d{4}", str(dated["Data"].iloc[0]))


# --- the engine contract the page depends on ---------------------------------


@pytest.fixture(scope="module")
def result():
    from pulse.engine import run_decision_cycle
    from pulse.metrics import load_gold

    return run_decision_cycle(load_gold(), AnalysisParams(as_of=date(2026, 9, 10)))


def test_segment_kpis_carry_the_placed_versus_completed_evidence(result):
    """The page's strongest evidence is orders PLACED against orders COMPLETED
    in one scope. It has to come from the engine, because the page may not
    compute it -- so the engine has to keep returning it.
    """
    from app.components import kpi_index

    top = result.priorities[0].diagnosis.anomaly
    index = kpi_index(result.segment_kpis, top.scope, top.scope_value)
    assert {"orders_placed", "orders_completed", "gmv"} <= set(index)
    for row in index.values():
        assert row["scope"] == top.scope
        assert str(row["scope_value"]) == str(top.scope_value)
        assert {"recent", "baseline", "delta_pct", "unit"} <= set(row)

    # Demand up, fulfilment down: the divergence the page is built around.
    assert index["orders_placed"]["delta_pct"] > index["orders_completed"]["delta_pct"]


def test_every_memo_has_the_evidence_lines_the_page_selects(result):
    import app.streamlit_app as page

    for memo in result.memos:
        assert page._evidence_line(memo, EVIDENCE_ANCHOR_IMPACT_SCORE)
        assert page._evidence_line(memo, EVIDENCE_ANCHOR_PLAYBOOK_ENTRY)
        assert page._memo_for(result, result.priorities[memo.priority - 1]) is memo


def _scored_priority(score: float, rank: int, gap_to_next: float) -> Priority:
    """A Priority with a real score and a real gap to the next one.

    The fixture the old I4 test used built two IDENTICAL priorities, so the gap
    it was guarding was zero by construction and any claim about the margin was
    vacuously true. This one takes both numbers, which is what makes a gap
    assertion mean something.
    """
    from dataclasses import replace as _replace

    base = _bare_priority()
    breakdown = dict(base.score_breakdown)
    breakdown["score_gap_to_next"] = gap_to_next
    return _replace(base, impact_score=score, rank=rank, score_breakdown=breakdown)


# --- C1: the presentation side of the semantic contract --------------------


def _headline_row(metric: str, **over) -> dict:
    """One .headline() row, shaped exactly as the engine returns it."""
    from pulse.metrics import METRIC_SEMANTICS, _UNITS

    row = {
        "metric": metric,
        "scope": "company",
        "scope_value": "all",
        "recent": 1.0,
        "baseline": 1.0,
        "delta_pct": 0.0,
        "unit": _UNITS[metric],
        "semantic": METRIC_SEMANTICS[metric],
    }
    row.update(over)
    return row


@pytest.mark.parametrize(
    "metric, recent, expected_value, expected_label",
    [
        ("gmv", 29977.825, "R$ 29.977,83/dia", "GMV médio diário"),
        ("orders_completed", 484.1429, "484/dia", "Pedidos concluídos por dia"),
        ("orders_placed", 549.2143, "549/dia", "Pedidos realizados por dia"),
        ("sessions", 3059.2857, "3.059/dia", "Sessões por dia"),
        (
            "contribution_margin",
            3358.4776,
            "R$ 3.358,48/dia",
            "Margem de contribuição média diária",
        ),
        ("active_customers", 473.0, "473/dia", "Clientes ativos por dia"),
    ],
)
def test_a_daily_average_card_says_so_in_both_the_label_and_the_value(
    metric, recent, expected_value, expected_label
):
    """C1's regression, on the surface a reader actually sees.

    Both halves are asserted. A value suffixed "/dia" under a card headed "GMV"
    is still ambiguous, and a label reading "GMV médio diário" over a bare
    R$ 29.977,83 is still quotable out of context. The card carries the fact
    twice because a figure gets copied without its label.
    """
    from app.components import fmt_value, headline_kpi_label, kpi_card_html

    row = _headline_row(metric, recent=recent)
    assert fmt_value(row["recent"], row["unit"], row["semantic"]) == expected_value
    assert headline_kpi_label(row) == expected_label
    card = kpi_card_html(row)
    assert expected_label in card
    assert expected_value in card


@pytest.mark.parametrize(
    "metric, recent, expected",
    [
        ("completion_rate", 0.8819, "88,2%"),
        ("cancellation_rate", 0.1181, "11,8%"),
        ("order_conversion", 0.1798, "18,0%"),
        ("on_time_rate", 0.9286, "92,9%"),
    ],
)
def test_a_rate_card_stays_a_percentage_and_is_never_per_day(metric, recent, expected):
    from app.components import fmt_value, headline_kpi_label

    row = _headline_row(metric, recent=recent)
    rendered = fmt_value(row["recent"], row["unit"], row["semantic"])
    assert rendered == expected
    assert "/dia" not in rendered
    assert "/dia" not in headline_kpi_label(row)
    assert "diári" not in headline_kpi_label(row)


def test_aov_is_an_average_per_order_and_is_never_labelled_per_day():
    """The most tempting wrong fix for C1 is to suffix every currency figure
    "/dia". aov is BRL per ORDER, so that would be a new false statement."""
    from app.components import fmt_value, headline_kpi_label

    row = _headline_row("aov", recent=61.922)
    rendered = fmt_value(row["recent"], row["unit"], row["semantic"])
    assert rendered == "R$ 61,92"
    assert "/dia" not in rendered
    assert headline_kpi_label(row) == "Ticket médio"
    assert "/dia" not in headline_kpi_label(row)
    assert "diári" not in headline_kpi_label(row)


def test_a_duration_card_is_minutes_per_delivery_not_per_day():
    from app.components import fmt_value, headline_kpi_label

    row = _headline_row("avg_actual_delivery_minutes", recent=27.7324)
    assert fmt_value(row["recent"], row["unit"], row["semantic"]) == "27,7 min"
    assert "/dia" not in headline_kpi_label(row)


def test_every_real_headline_card_renders_its_own_semantic(result):
    """Against live engine output rather than a fixture: every flow card carries
    "/dia" and no rate or per-order card does.
    """
    from app.components import kpi_card_html
    from pulse.metrics import DAILY_AVERAGE

    for row in result.headline_kpis:
        card = kpi_card_html(row)
        if row["semantic"] == DAILY_AVERAGE:
            assert "/dia" in card, row["metric"]
        else:
            assert "/dia" not in card, row["metric"]


def test_the_impact_table_separates_the_daily_rate_from_the_window_total(result):
    """The memo/impact distinction, on the page.

    Four rows accumulate over the window and two are per day; they differ by a
    factor of the window length. Every accumulated row says "acumulado" and the
    run-rate row says "médio diário", because a R$ 739 run rate beside a
    R$ 10.351 window figure reads as two findings that disagree otherwise.
    """
    from app.components import impact_rows

    priority = result.priorities[0]
    rows = impact_rows(priority.impact, result.params.comparison_window_days)
    labels = [row["Estimativa"] for row in rows]

    accumulated = [label for label in labels if "acumulado" in label]
    assert len(accumulated) >= 3, labels
    assert any("médio diário" in label for label in labels), labels

    gmv_window = next(row for row in rows if row["Estimativa"].startswith("Desvio de GMV"))
    run_rate = next(row for row in rows if "médio diário" in row["Estimativa"])
    assert "acumulado na janela de 14 dias" in gmv_window["Estimativa"]
    assert "por dia" in run_rate["Valor"]
    # And the two really are a window apart, so the labels are load-bearing.
    assert priority.impact.gmv_at_risk_brl == pytest.approx(
        abs(priority.impact.daily_run_rate_brl) * result.params.comparison_window_days,
        rel=1e-9,
    )


def test_the_memo_differentiates_a_daily_run_rate_from_an_accumulated_window(result):
    """The same distinction inside the engine's own prose, which the page embeds.

    The memo used to put a R$ 739/day deviation and a R$ 10.351 window deviation
    three sections apart with nothing naming the difference, so the document
    contradicted itself in the reader's own arithmetic.
    """
    from pulse.decision_memo import memo_to_markdown

    markdown = memo_to_markdown(result.memos[0])
    impact_sentence = next(
        line for line in result.memos[0].evidence if "GMV MÉDIO DIÁRIO" in line
    )
    assert "ACUMULADO" in impact_sentence
    assert "multiplicada pelo número de dias" in impact_sentence
    assert "acumulado na janela de 14 dias" in markdown
    assert "Desvio do GMV médio diário (ritmo)" in markdown


# --- C2: no unconditional causal mechanism ---------------------------------


def _kpi(metric: str, delta_pct: float) -> dict:
    return _headline_row(metric, delta_pct=delta_pct)


def test_a_delivery_mechanism_is_only_offered_for_a_fulfilment_pattern():
    """The fix for C2, stated as the two cases that were previously identical.

    The page rendered ONE delivery-cost mechanism for every priority. A
    supply-availability priority got told that a degraded delivery experience
    costs more to serve while its own driver table on the same page showed
    delivery time improving.
    """
    from app.components import margin_gmv_note

    margin, gmv = _kpi("contribution_margin", -7.92), _kpi("gmv", -3.59)

    fulfilment = margin_gmv_note(margin, gmv, "fulfillment_eta_degradation")
    supply = margin_gmv_note(margin, gmv, "supply_availability_gap")
    unknown = margin_gmv_note(margin, gmv, "unknown_pattern")

    # A: the diagnosed pattern IS post-checkout, so consistency may be stated.
    assert "após o checkout" in fulfilment
    assert "consistentes com" in fulfilment

    # B: it is not, so no delivery-side explanation appears at all.
    for text in (supply, unknown):
        for banned in ("checkout", "entrega", "atender", "delivery"):
            assert banned not in text.lower(), (text, banned)

    # Every branch states only what the two series did.
    for text in (fulfilment, supply, unknown):
        assert "caiu mais do que o GMV" in text
        assert not OVERSTATEMENT.search(text)


def test_the_margin_note_states_the_comparison_that_actually_holds():
    from app.components import margin_gmv_note

    fell_more = margin_gmv_note(
        _kpi("contribution_margin", -7.92), _kpi("gmv", -3.59), "unknown_pattern"
    )
    did_not = margin_gmv_note(
        _kpi("contribution_margin", -1.10), _kpi("gmv", -3.59), "unknown_pattern"
    )
    assert "A margem de contribuição caiu mais do que o GMV" in fell_more
    assert "não caiu mais do que o GMV" in did_not
    assert margin_gmv_note(None, _kpi("gmv", -3.59), "unknown_pattern") is None
    assert margin_gmv_note(_kpi("gmv", -3.59), None, "unknown_pattern") is None


def test_no_page_asserts_a_delivery_cost_mechanism():
    """The specific sentence C2 was about, banned in app/ in both languages."""
    banned = re.compile(
        r"custa mais para atender|experiência de entrega degradada"
        r"|costs more to serve",
        re.IGNORECASE,
    )
    for path in _app_sources():
        assert not banned.search(path.read_text(encoding="utf-8")), path


def test_every_non_fulfilment_priority_renders_without_a_delivery_story(result):
    """Rendered, not inspected: the pattern that is NOT post-checkout must not
    produce delivery-side prose anywhere in its margin note.
    """
    from app.components import kpi_index, margin_gmv_note

    others = [
        p for p in result.priorities
        if p.diagnosis.pattern != "fulfillment_eta_degradation"
    ]
    assert others, "no non-fulfilment priority in this run to check"
    for priority in others:
        anomaly = priority.diagnosis.anomaly
        index = kpi_index(result.segment_kpis, anomaly.scope, anomaly.scope_value)
        note = margin_gmv_note(
            index.get("contribution_margin"), index.get("gmv"),
            priority.diagnosis.pattern,
        )
        if note is None:
            continue
        assert "checkout" not in note.lower()
        assert "entrega" not in note.lower()


# --- I1: a netted residual is never called a measured customer count -------


def test_a_netted_customer_residual_is_never_labelled_as_a_measured_count():
    """The two quantities that shared one label, rendered side by side.

    The company priority's figure is a residual left after the nested segment
    claims are netted out: 3,896 where 5,632 customers actually ordered. It is
    not a count of anybody and may not be described as one.
    """
    from app.components import impact_rows
    from pulse.metrics import customers_label

    measured = Impact(
        gmv_at_risk_brl=10350.87, orders_lost=139, customers_affected=5632,
        margin_impact_brl=3920.96, daily_run_rate_brl=-739.35,
        projected_30d_brl=-22180.44,
    )
    residual = Impact(
        gmv_at_risk_brl=4655.0, orders_lost=96, customers_affected=3896,
        margin_impact_brl=1200.0, daily_run_rate_brl=-332.5,
        projected_30d_brl=-9975.0, customers_are_residual=True,
    )
    assert measured.customers_affected != residual.customers_affected

    banned = ("clientes que pediram", "clientes do escopo", "clientes afetados")
    label = customers_label(residual)
    for phrase in banned:
        assert phrase not in label.lower(), label
    assert "residual" in label.lower()

    # The measured one keeps the measured wording -- the fix is a distinction,
    # not a blanket hedge that loses a true statement.
    assert "clientes que pediram" in customers_label(measured).lower()

    rendered = " ".join(row["Estimativa"] for row in impact_rows(residual, 14))
    for phrase in banned:
        assert phrase not in rendered.lower(), rendered
    assert "residual" in rendered.lower()


def test_the_live_aggregate_priority_renders_its_customers_as_a_residual(result):
    """Against the real run: the company group is netted, the zone groups are not,
    and the label follows the Impact rather than the scope name.
    """
    from pulse.metrics import customers_label

    company = next(
        (p for p in result.priorities if p.diagnosis.anomaly.scope == "company"), None
    )
    assert company is not None, "no aggregate priority in this run"
    assert company.impact.customers_are_residual is True
    assert "clientes que pediram" not in customers_label(company.impact).lower()

    for priority in result.priorities:
        if priority.diagnosis.anomaly.scope != "company":
            assert priority.impact.customers_are_residual is False
            assert "clientes que pediram" in customers_label(priority.impact).lower()


def test_the_score_table_does_not_call_the_customer_term_affected_customers():
    from app.components import score_rows

    labels = [row["Componente"] for row in score_rows(_bare_priority().score_breakdown)]
    assert not any("afetados" in label.lower() for label in labels), labels
    assert any("considerados no impacto" in label for label in labels), labels


# --- I4: the ranking gap is a number, never a word -------------------------


def test_the_ranking_note_states_the_actual_gap_between_two_real_scores():
    """Replaces a test that built two IDENTICAL priorities, so the margin it was
    guarding was zero by construction and "uma vantagem estreita" was vacuously
    true. Two genuinely different scores, and the displayed gap is asserted.
    """
    from app.components import ranking_note

    lead = _scored_priority(83.76, rank=1, gap_to_next=83.76 - 61.59)
    second = _scored_priority(61.59, rank=2, gap_to_next=0.0)

    note = ranking_note([lead, second])
    assert "83,76" in note
    assert "61,59" in note
    assert "22,17 pontos" in note
    assert "ocupa a primeira posição" in note
    assert not OVERSTATEMENT.search(note)


def test_the_ranking_note_follows_the_actual_order_when_it_reverses():
    """Reverse the two scores and the presentation follows the ranking rather
    than a remembered winner."""
    from app.components import ranking_note

    note = ranking_note([
        _scored_priority(61.59, rank=1, gap_to_next=61.59 - 33.00),
        _scored_priority(33.00, rank=2, gap_to_next=0.0),
    ])
    assert "61,59" in note
    assert "33,00" in note
    assert "28,59 pontos" in note
    assert "83,76" not in note


def test_no_surface_characterises_the_size_of_the_ranking_gap():
    """No qualitative word for the SIZE of a score gap anywhere a reader sees,
    in either language, because no policy in PULSE defines one.

    This replaces a scanner that banned three LITERAL phrases ("vantagem
    estreita", "vantagem ampla", "narrow lead"). Those three were removed and
    the claim came straight back in a fourth wording -- "a diferença entre o
    primeiro e o segundo é pequena o bastante" -- which a literal list could
    not see. So the ban is on the CLASS: a magnitude adjective inside a
    proximity window of score-gap vocabulary. "pequena" is not banned on its
    own; a small sample is a legitimate thing to call small.
    """
    offenders = {
        str(path): _gap_magnitude_hits(path) for path in _gap_scanner_sources()
    }
    assert _gap_scanner_sources(), "no sources found to scan"
    assert {p: hits for p, hits in offenders.items() if hits} == {}


def test_the_gap_magnitude_scanner_catches_the_class_and_spares_benign_uses():
    """The previous scanner's real failure was that nobody checked it caught
    what it was meant to catch, so a reworded claim walked straight past it.
    Both halves are asserted here: every shape of the banned class, and the
    benign uses of the same adjectives that must stay legal.
    """
    for banned in (
        "O primeiro tem uma vantagem estreita sobre o segundo.",
        "a diferença entre o primeiro e o segundo é pequena o bastante para que "
        "a ordenação seja uma orientação",
        "O gap pequeno entre os dois não decide nada.",
        "O escopo abriu ampla vantagem na classificação.",
        "Há uma grande distância entre o primeiro e o segundo na classificação.",
        "A liderança clara do primeiro item na classificação.",
        "A prioridade que ficou em primeiro lugar é dominante.",
        "Zone 7 clearly leads the ranking.",
        "Zone 7 holds a narrow lead over the runner-up.",
        "Zone 7 holds a wide lead in the ranking.",
    ):
        assert _gap_magnitude_in(banned), banned
    for allowed in (
        # The adjective is about a sample, an effect or a grid -- not a gap.
        "uma amostra pequena não sustenta o efeito",
        "o efeito absoluto não pode ser inflado por uma base pequena",
        "a média da empresa é tirada sobre uma grade de oito zonas",
        "Desvio de margem de contribuição, no mínimo",
        # The one policy-backed narrowness claim in PULSE: it is about DRIVER
        # CORRELATION spread, gated on _CLUSTER_SPREAD, states the measured
        # spread, and is not about the priority-score gap at all.
        "Elas ficam a menos de 0,15 umas das outras em correlação absoluta -- "
        "faixa estreita demais para ranquear com qualquer confiança -- portanto "
        "essas séries se deterioraram em conjunto",
    ):
        assert not _gap_magnitude_in(allowed), allowed


def test_the_readme_states_the_gap_the_engine_and_the_app_actually_print(
    result, rendered
):
    """One rounding policy, stated once, and asserted so it cannot drift again.

    The engine rounds the FULL-PRECISION difference, so the printed gap is not
    the difference of the two rounded scores printed beside it: 83.76 - 61.59
    reads 22.17, while score_gap_to_next rounds to 22.16. README claimed 22.17
    and the app printed 22,16 -- the documentation disagreed with the product by
    a hundredth of a point. The expected figure is derived from the engine here;
    hard-coding it is exactly how the two drifted apart.
    """
    from app.ui_text import fmt_float

    gap = result.priorities[0].score_breakdown["score_gap_to_next"]
    printed = fmt_float(gap)  # "22,16" -- Brazilian notation, the app's own
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    # README writes numbers in English notation, but it has to be the same figure.
    assert f"**{printed.replace(',', '.')} points**" in readme
    # And the app prints it through that same formatter.
    blob = "\n".join(m.value for m in rendered["operations.py"].markdown)
    assert f"{printed} pontos" in blob
    # The naive subtraction of the two displayed scores is NOT the printed gap,
    # and README states the policy rather than leaving a reader to discover it.
    naive = fmt_float(
        round(result.priorities[0].impact_score, 2)
        - round(result.priorities[1].impact_score, 2)
    )
    assert naive != printed
    assert f"**{naive.replace(',', '.')} points**" not in readme
    assert "not the difference of the two rounded scores" in re.sub(
        r"\s+", " ", readme
    )


def test_the_engine_returns_the_gap_so_the_page_does_not_compute_it(result):
    """app/ may not derive a business figure, and a gap between two scores is
    one. prioritize() returns it; ranking_note reads it.
    """
    priorities = result.priorities
    assert len(priorities) >= 2
    for first, second in zip(priorities, priorities[1:]):
        assert first.score_breakdown["score_gap_to_next"] == pytest.approx(
            first.impact_score - second.impact_score, rel=1e-12
        )
    assert priorities[-1].score_breakdown["score_gap_to_next"] == 0.0
    assert "ranking_note" in (ROOT / "app" / "streamlit_app.py").read_text(
        encoding="utf-8"
    )


def test_the_readme_reports_the_suite_size_it_actually_has():
    """A stale count survived a whole fix round: the README said 355 tests /
    354 passed while the suite had grown to 466. Collected at test time rather
    than typed, so the figure cannot drift from the suite again.

    Counts COLLECTED tests, which is what a reader would get from `pytest`, and
    asserts the module count separately because the tree diagram states it too.
    """
    import subprocess
    import sys

    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         str(ROOT / "tests")],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert collected.returncode == 0, collected.stderr[-2000:]
    # This pytest reports "tests/test_x.py: N" per module under -q, so the
    # total is summed rather than read off a summary line.
    total = 0
    for line in collected.stdout.splitlines():
        name, _, count = line.partition(": ")
        if name.startswith("tests") and name.endswith(".py") and count.strip().isdigit():
            total += int(count.strip())
    assert total > 0, collected.stdout[-2000:]

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"{total} tests" in readme, (
        f"README does not state the real suite size of {total}"
    )
    modules = len(list((ROOT / "tests").glob("test_*.py")))
    assert f"{modules} modules" in readme, (
        f"README does not state the real module count of {modules}"
    )
