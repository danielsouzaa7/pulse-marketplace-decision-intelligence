# PULSE — the navigation host, and the Decision Intelligence page itself.
#
# ONE ENGINE. main() below calls run_decision_cycle(gold, params) and renders
# what comes back. It computes nothing: no mean, no standard deviation, no
# group-by, no deviation, share, contribution or score recomputed here or
# anywhere in app/. Every figure is read off an engine object, which is why
# `pulse decide` and this page cannot show different numbers.
#
# The app is a multipage app built with st.navigation. The Gate #1 decision
# flow — company signal, ranked priorities, evidence, root cause, impact,
# score, recommendation, memo, human decision — is unchanged and is registered
# as the first page, because it is the product. The analytics and platform
# pages are separate scripts under app/views/, each answering one decision
# question; none of them re-tells this flow and none of them recomputes it.
#
# The cached loaders and the sidebar live in app/state.py so every page shares
# one gold load and one decision-cycle cache. The two Streamlit traps they
# handle (gold is not bit-stable; the cache must key on the parameter VALUES)
# are documented there.
#
# The group keys and the page filenames below are identifiers; the titles a
# reader sees are looked up in app/ui_text.py, which is the only place a
# user-facing name is written.
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # `streamlit run app/streamlit_app.py` puts app/ on sys.path, not the repo
    # root, so `from app.components import ...` needs this.
    sys.path.insert(0, str(ROOT))

from pulse.decision_memo import (  # noqa: E402
    _DECISION_STATUS_NOTE,
    EVIDENCE_ANCHOR_IMPACT_SCORE,
    EVIDENCE_ANCHOR_PLAYBOOK_ENTRY,
    memo_to_markdown,
)

from pulse.contracts import GoldContractError  # noqa: E402
from pulse.metrics import MissingGoldTableError  # noqa: E402

from app import components as ui  # noqa: E402
from app import state  # noqa: E402
from app import ui_text as T  # noqa: E402

# NOT "pages": Streamlit auto-discovers that name beside the entry script and a
# refresh on a page URL would then run the page directly, bypassing st.navigation.
PAGES_DIR = Path(__file__).resolve().parent / "views"

# The scope-level story, in the order a reader needs it: demand first, then
# what happened to that demand after checkout, then what it cost.
EVIDENCE_METRICS: tuple[str, ...] = (
    "orders_placed",
    "orders_completed",
    "completion_rate",
    "cancellation_rate",
    "avg_actual_delivery_minutes",
    "gmv",
    "contribution_margin",
)

# Every page in the app, grouped by the job it does rather than by the data it
# happens to show. Command is where a decision is made; Analytics is where the
# question behind it is interrogated; Platform is where the method is checked.
PAGE_GROUPS: dict[str, tuple[str, ...]] = {
    "Command": ("copilot.py",),
    "Analytics": (
        "operations.py",
        "merchants.py",
        "customers.py",
        "promotions.py",
    ),
    "Platform": (
        "experiment_lab.py",
        "data_quality.py",
    ),
}


def page_files() -> tuple[Path, ...]:
    """Every registered page script. Pure: no Streamlit call, so a test can ask
    the app which pages it claims to have without running it.
    """
    return tuple(
        PAGES_DIR / filename
        for pages in PAGE_GROUPS.values()
        for filename in pages
    )


def _masthead(result) -> None:
    params = result.params
    recent_start = params.as_of - timedelta(days=params.comparison_window_days - 1)
    st.markdown(
        '<div class="pulse-mast">'
        '<div class="name"><span class="mark">◆</span>PULSE</div>'
        f'<div class="tag">{T.APP_TAGLINE}</div>'
        "</div>"
        f'<p class="pulse-note" style="margin-top:6px">'
        f"Janela de comparação <b>{ui.fmt_date(recent_start)} → "
        f"{ui.fmt_date(params.as_of)}</b> "
        f"({params.comparison_window_days} dias), contra uma linha de base "
        f"ajustada por dia da semana de {params.baseline_window_days} dias "
        f"&nbsp;·&nbsp; data de referência <b>{ui.fmt_date(result.as_of)}</b> "
        f"&nbsp;·&nbsp; sensibilidade z ≥ {ui.fmt_float(params.sensitivity, 1)}</p>",
        unsafe_allow_html=True,
    )
    ui.notices()


def _company_section(result) -> None:
    ui.section(
        "01 · Sinal da empresa",
        "O painel agregado, onde a falha de um segmento se esconde",
        "Esta é a visão de onde a maioria dos times parte. Nada aqui exige "
        "atenção: o GMV está alguns por cento abaixo e os pedidos realizados "
        "estão estáveis. A visão da empresa é uma média, e uma média sobre oito "
        "zonas é exatamente onde uma zona com falha desaparece. Cada cartão de "
        "fluxo abaixo é uma <b>média diária</b> da janela de comparação, não o "
        "total da janela — as duas diferem por um fator igual ao número de dias, "
        "e os rótulos dizem qual é qual.",
    )
    ui.kpi_row(list(result.headline_kpis), per_row=4)


def _priorities_section(result):
    ui.section(
        "02 · O que o motor classificou",
        "Três prioridades, uma por grupo de escopo",
        "As anomalias são agrupadas por escopo antes da classificação, de modo "
        "que sete sintomas de um mesmo incidente permanecem um incidente "
        "reivindicando um único valor de impacto. A reivindicação do grupo da "
        "empresa é compensada contra os grupos de segmento que ela contém, "
        "então o mesmo dinheiro nunca é contado duas vezes.",
    )
    columns = st.columns(len(result.priorities), gap="small")
    for column, priority in zip(columns, result.priorities):
        column.markdown(ui.priority_card_html(priority), unsafe_allow_html=True)

    # ui.ranking_note() is the one implementation of this sentence -- the
    # Operations page renders the same one. There used to be a second copy here
    # that characterised the size of the gap in a fixed adjective, whatever gap
    # the run actually produced: a claim no policy in PULSE defines, and one that
    # was off by 22 points on the current data. The note now states the distance
    # as the engine's own number and leaves the judgement to the reader.
    st.markdown(
        f'<p class="pulse-note" style="margin-top:12px">'
        f"{ui.ranking_note(result.priorities)}</p>",
        unsafe_allow_html=True,
    )

    st.markdown('<div style="margin-top:18px"></div>', unsafe_allow_html=True)
    labels = {ui.priority_label(p): p for p in result.priorities}
    chosen = st.segmented_control(
        "Examinar prioridade",
        options=list(labels),
        default=next(iter(labels)),
        key="selected_priority",
    )
    return labels.get(chosen, result.priorities[0])


def _evidence_section(result, priority) -> None:
    anomaly = priority.diagnosis.anomaly
    diagnosis_pattern = priority.diagnosis.pattern
    where = ui.scope_label(anomaly.scope, anomaly.scope_value)
    index = ui.kpi_index(result.segment_kpis, anomaly.scope, anomaly.scope_value)

    ui.section(
        "03 · Evidência",
        f"{where}: o que aconteceu antes do checkout, e depois",
        "O mesmo registro de métricas da visão da empresa, medido sobre a série "
        "do próprio escopo na mesma janela.",
    )

    placed, completed = index.get("orders_placed"), index.get("orders_completed")
    if placed and completed:
        ui.kpi_row([placed, completed], per_row=2, hero=True)
        diverging = placed["delta_pct"] > completed["delta_pct"]
        st.markdown('<div style="margin-top:14px"></div>', unsafe_allow_html=True)
        if diverging:
            # The superlative belongs to the top-ranked scope only. Attaching
            # it to whichever priority happens to be selected would overstate
            # a two-point gap on a lower-ranked one.
            emphasis = (
                " Nesta execução, é a evidência isolada mais forte que o motor "
                "encontrou."
                if priority.rank == 1
                else ""
            )
            ui.callout(
                f"<b>Demanda e entrega se moveram em direções opostas.</b> Em "
                f"{where}, os pedidos <b>realizados</b> fecharam em "
                f"{ui.fmt_pct(placed['delta_pct'])} enquanto os pedidos "
                f"<b>concluídos</b> fecharam em "
                f"{ui.fmt_pct(completed['delta_pct'])} na mesma janela. Os "
                f"clientes continuaram chegando e continuaram pedindo; uma "
                f"parcela crescente desses pedidos nunca se concluiu. Nenhuma "
                f"explicação pelo lado da demanda produz essa divergência."
                f"{emphasis}"
            )
        else:
            ui.callout(
                f"Em {where}, os pedidos realizados fecharam em "
                f"{ui.fmt_pct(placed['delta_pct'])} e os pedidos concluídos em "
                f"{ui.fmt_pct(completed['delta_pct'])} na mesma janela."
            )

    ui.chart(
        ui.scope_delta_chart(
            index, EVIDENCE_METRICS,
            f"{where} — variação vs. linha de base, por métrica",
        ),
        empty_note="Nenhuma linha de KPI no nível do escopo foi retornada para "
                   "esta prioridade.",
    )

    st.markdown(
        f'<p class="pulse-note" style="margin-top:6px">'
        f"{T.FIRST_SIGNAL_NOTE.format(date=ui.fmt_date(anomaly.first_detected_date))}"
        f"</p>",
        unsafe_allow_html=True,
    )

    # OBSERVATION, KEYED ON THE DIAGNOSED PATTERN. This block used to render one
    # fulfilment-cost MECHANISM unconditionally, for every priority -- so a
    # supply-availability priority whose own driver table two sections below
    # showed fulfilment times improving was still handed that explanation.
    # ui.margin_gmv_note() states what the two series did and adds a consistency
    # sentence only for the pattern whose signature is itself post-checkout.
    note = ui.margin_gmv_note(
        index.get("contribution_margin"), index.get("gmv"), diagnosis_pattern
    )
    if note:
        ui.callout(note, quiet=True)


def _root_cause_section(priority) -> None:
    diagnosis = priority.diagnosis
    ui.section(
        "04 · Causa raiz",
        "Qual etapa se moveu, e o que se moveu junto",
        "O GMV se decompõe exatamente como sessões × conversão × taxa de "
        "conclusão × ticket médio. Em logaritmos esses quatro termos são "
        "aditivos, de modo que a barra de maior magnitude <i>é</i> a etapa que "
        "se moveu — calculada a partir da identidade, não presumida. Tudo à "
        "direita é uma associação: séries que se moveram juntas, nunca uma "
        "causa demonstrada.",
    )

    left, right = st.columns([1.05, 1], gap="medium")
    with left:
        ui.chart(
            ui.funnel_waterfall(diagnosis.funnel),
            empty_note=(
                "Sem decomposição de funil para esta anomalia. A identidade do "
                "funil trata do GMV, então uma anomalia de taxa ou de duração "
                "não recebe nenhuma, em vez de receber uma aplicada de forma "
                "indevida."
            ),
        )
        if diagnosis.funnel:
            st.markdown(
                f'<p class="pulse-note">Ruptura principal: '
                f"<b>{ui.metric_label(diagnosis.funnel_break_stage)}</b>. As "
                f"barras são contribuições em log; o rótulo de cada barra é a "
                f"variação percentual da própria etapa.</p>",
                unsafe_allow_html=True,
            )
    with right:
        # No empty_note here: segment_line() below already states the absent
        # case, and saying it twice reads as a bug.
        ui.chart(ui.contribution_chart(diagnosis.contributions))
        st.markdown(
            f'<p class="pulse-note">{ui.segment_line(diagnosis.primary_segment)}</p>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="margin-top:20px"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pulse-eyebrow">Direcionadores associados, ordenados por |r|'
        "</div>",
        unsafe_allow_html=True,
    )
    if diagnosis.drivers:
        ui.table(ui.driver_rows(diagnosis.drivers))
        st.markdown(
            '<p class="pulse-note">Correlação ao longo da janela de análise. A '
            "defasagem é o atraso que maximiza |r| dentro de ±3 dias e é "
            "reportada como evidência, não como direção de causalidade. Padrão "
            f"diagnosticado: <b>{ui.pattern_label(diagnosis.pattern)}</b>.</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p class="pulse-note">Nenhuma série candidata superou o limiar de '
            "reporte para esta anomalia.</p>",
            unsafe_allow_html=True,
        )


def _impact_section(priority, params) -> None:
    ui.section(
        "05 · Impacto estimado",
        "Um piso, não uma previsão — e duas unidades de tempo, rotuladas",
        T.IMPACT_FLOOR_NOTE,
    )
    ui.table(ui.impact_rows(priority.impact, params.comparison_window_days))


def _score_section(priority, memo) -> None:
    ui.section(
        "06 · Pontuação de prioridade",
        f"{ui.fmt_float(priority.impact_score)} de 100, reconstruível à mão",
        "Cada componente de impacto é normalizado contra o maior candidato "
        "desta execução; a confiança entra como foi medida. Os pesos e os "
        "componentes normalizados são mostrados em separado, exatamente como o "
        "motor os retornou.",
    )
    left, right = st.columns([1, 1.15], gap="medium")
    with left:
        ui.table(ui.score_rows(priority.score_breakdown))
        breakdown = priority.score_breakdown
        st.markdown(
            f'<p class="pulse-note">{int(breakdown["supporting_anomalies"])} '
            f"anomalias sustentam este grupo de escopo &nbsp;·&nbsp; "
            f"{int(breakdown['nested_groups_netted'])} reivindicações de "
            f"segmento aninhadas foram compensadas nesta.</p>",
            unsafe_allow_html=True,
        )
    with right:
        line = _evidence_line(memo, EVIDENCE_ANCHOR_IMPACT_SCORE)
        if line:
            ui.callout(line, quiet=True)
        st.markdown('<div style="margin-top:10px"></div>', unsafe_allow_html=True)
        ui.callout(ui.confidence_note(priority))


def _recommendation_section(memo) -> None:
    ui.section(
        "07 · Recomendação",
        "Selecionada apenas a partir do padrão diagnosticado",
        "O playbook é indexado pelo padrão — nenhuma zona, métrica ou valor de "
        "escopo participa da busca — de modo que a mesma assinatura em qualquer "
        "ponto do negócio produz a mesma recomendação.",
    )
    ui.callout(f"<b>Ação.</b> {memo.recommended_action}")
    st.markdown('<div style="margin-top:10px"></div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="medium")
    with left:
        ui.callout(f"<b>Resultado potencial.</b> {memo.potential_result}", quiet=True)
    with right:
        ui.callout(f"<b>Método de validação.</b> {memo.validation_method}", quiet=True)
    playbook = _evidence_line(memo, EVIDENCE_ANCHOR_PLAYBOOK_ENTRY)
    if playbook:
        # st.caption rather than a styled <p>: the sentence carries inline
        # backticks the engine wrote, and raw HTML would print them literally.
        st.caption(playbook)


def _memo_section(memo) -> None:
    ui.section(
        "08 · Memorando de decisão",
        memo.memo_id,
        "O artefato que uma pessoa nomeada lê e aprova. Renderizado a partir do "
        "mesmo objeto de memorando que <code>pulse decide</code> grava em "
        "<code>artifacts/decision-memo.md</code>.",
    )
    markdown = memo_to_markdown(memo)
    with st.container(border=True):
        st.markdown(ui.as_embedded_markdown(markdown))
    st.download_button(
        "Baixar este memorando (Markdown)",
        data=markdown,
        file_name=f"{memo.memo_id}.md",
        mime="text/markdown",
    )


def _decision_section(memo) -> None:
    ui.section(
        "09 · Decisão humana",
        "Registrada aqui, executada em lugar nenhum",
        _DECISION_STATUS_NOTE,
    )
    status = ui.decision_state_control(memo.memo_id)
    ui.callout(
        f"<b>{memo.memo_id} — {ui.decision_status_label(status)}</b>. Registrado "
        f"apenas nesta sessão do navegador. O PULSE grava arquivos e renderiza "
        f"páginas; ele não chama nenhum sistema do marketplace, e nada nesta "
        f"página entra em vigor automaticamente.",
        quiet=True,
    )


def _evidence_line(memo, anchor: str) -> str | None:
    """One of the memo's own evidence sentences, selected by its opening words.

    `anchor` is one of the EVIDENCE_ANCHOR_* constants decision_memo.py
    exports -- the same constant that module used to open the sentence in the
    first place, imported rather than retyped, so this selector cannot drift
    from what it is selecting. Selecting an engine-authored string is not
    authoring one: the arithmetic in the score sentence was written by
    prioritization.py, which is the only place that may write it.
    """
    return next((line for line in memo.evidence if line.startswith(anchor)), None)


def _memo_for(result, priority):
    return next((m for m in result.memos if m.priority == priority.rank), None)


def main() -> None:
    """The Gate #1 decision flow, unchanged. Registered as the first page."""
    state.page_setup()
    result = state.cycle(**state.params())

    _masthead(result)
    _company_section(result)

    if not result.priorities:
        ui.section(
            "02 · O que o motor classificou",
            "Nenhuma anomalia superou os limiares atuais",
            T.NO_ANOMALY_NOTE,
        )
        return

    priority = _priorities_section(result)
    _evidence_section(result, priority)
    _root_cause_section(priority)
    _impact_section(priority, result.params)

    memo = _memo_for(result, priority)
    if memo is None:
        st.markdown(
            '<p class="pulse-note">Nenhum memorando de decisão foi escrito para '
            "esta prioridade — os memorandos são limitados às três primeiras.</p>",
            unsafe_allow_html=True,
        )
        return

    _score_section(priority, memo)
    _recommendation_section(memo)
    _memo_section(memo)
    _decision_section(memo)


def _navigation():
    pages = {
        T.GROUP_LABELS["Command"]: [
            st.Page(main, title=T.PAGE_TITLES["decision"], url_path="decision",
                    default=True)
        ]
    }
    for group, filenames in PAGE_GROUPS.items():
        pages.setdefault(T.GROUP_LABELS[group], []).extend(
            st.Page(PAGES_DIR / filename, title=T.PAGE_TITLES[filename])
            for filename in filenames
        )
    return st.navigation(pages)


if __name__ == "__main__":
    # Streamlit executes the entry script as "__main__", so this runs under
    # `streamlit run` while keeping `import app.streamlit_app` side-effect-free
    # for the tests. The sidebar renders here, before the selected page runs,
    # so one set of analysis parameters is shared by every page.
    st.set_page_config(page_title=T.APP_TITLE, page_icon="◆", layout="wide")
    state.page_setup()
    try:
        gold = state.gold()
        # Assigned, never a bare expression: Streamlit DISPLAYS a bare
        # expression at script level, and this one would print both tables.
        _required = (gold.daily_business_metrics, gold.zone_performance)
    except (GoldContractError, MissingGoldTableError) as exc:
        # A data gate that failed is a stop, stated in words, never a
        # traceback -- and never a page rendered from whatever did load.
        st.error(
            "Os dados Gold não passaram na verificação, e nada foi analisado. "
            f"Motivo: {exc}. Reconstrua as camadas com `pulse build`."
        )
        st.stop()
    state.sidebar()
    _navigation().run()
