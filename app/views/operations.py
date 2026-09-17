# Operations & Zones — where is fulfilment failing, and how much is it costing?
#
# Two sources, neither of them recomputed here: the engine's ranked priorities
# (which scope, which pattern, what it is worth) and gold_zone_performance
# (the shape of the failure across the grid). The heatmap is a pivot of columns
# the gold table already carries; every currency figure is read off an Impact
# the engine produced.
#
# No zone is named anywhere in this file. The ranking is recomputed from the
# current parameters and the membership of the ranked list has changed between
# builds, so the page renders whatever comes back first. The margin between
# first and second is whatever score_gap_to_next returns on the run, and nothing
# here assumes anything about its size -- an earlier version of this comment
# asserted "a fraction of a point" and was still saying it at 22 points.
from __future__ import annotations

import streamlit as st

from app import components as ui
from app import state
from app import ui_text as T

# Fulfilment, before and after checkout. The selector defaults to on-time rate
# because that is the one an operations lead is accountable for.
GRID_METRICS: tuple[str, ...] = (
    "on_time_rate",
    "avg_actual_delivery_minutes",
    "cancellation_rate",
    "completion_rate",
    "availability_rate",
    "order_conversion",
)

SCOPE_METRICS: tuple[str, ...] = (
    "orders_placed",
    "orders_completed",
    "completion_rate",
    "cancellation_rate",
    "on_time_rate",
    "avg_actual_delivery_minutes",
    "gmv",
    "contribution_margin",
)


def render() -> None:
    state.page_setup()
    knobs = state.params()
    result = state.cycle(**knobs)
    analysis_start, recent_start, as_of = state.span(knobs)

    ui.page_head(
        "Análises · Operações",
        "Onde a entrega está falhando, e quanto isso está custando?",
        "A falha de entrega é invisível no escopo da empresa: uma média sobre "
        "oito zonas absorve uma zona com falha. Esta página mostra a grade sobre "
        "a qual a média é tirada e coloca ao lado dela a estimativa de custo do "
        "próprio motor.",
    )

    ui.section(
        "01 · O que o motor classificou em primeiro lugar",
        "O que voltou primeiro nesta execução — não uma resposta fixa",
        "A classificação é recalculada a partir da sensibilidade e da "
        "materialidade atuais sempre que um parâmetro se move. Nada nesta página "
        "presume qual escopo vence.",
    )
    if not result.priorities:
        ui.callout(T.NO_ANOMALY_NOTE)
    else:
        ui.table(ui.priority_rows(result.priorities))
        st.markdown(
            f'<p class="pulse-note" style="margin-top:8px">'
            f"{ui.ranking_note(result.priorities)}</p>",
            unsafe_allow_html=True,
        )

    ui.section(
        "02 · A grade sobre a qual a média da empresa é tirada",
        f"Zona a zona, {ui.fmt_date(analysis_start)} → {ui.fmt_date(as_of)}",
        "O período completo que o motor mede: uma linha de base de "
        f"{knobs['baseline_window_days']} dias seguida pela janela de comparação "
        f"de {knobs['comparison_window_days']} dias. Uma faixa que escurece na "
        "borda direita é uma deterioração dentro da janela de comparação; uma "
        "que já estava escura faz parte da base contra a qual se compara, que é "
        "o motivo pelo qual os números de impacto do motor subestimam.",
    )
    metric = st.selectbox(
        "Métrica operacional",
        options=GRID_METRICS,
        format_func=ui.metric_label,
        key="ops_grid_metric",
    )
    zones = state.window_slice(state.gold().zone_performance, analysis_start, as_of)
    grid = zones.pivot(index="zone_id", columns="metric_date", values=metric)
    ui.chart(
        ui.heatmap_chart(
            grid, metric,
            f"{ui.metric_label(metric)} por zona e por dia — gold_zone_performance",
            height=120 + 34 * len(grid.index),
        ),
        empty_note="gold_zone_performance não carrega linhas neste período.",
    )
    st.markdown(
        '<p class="pulse-note">Cada célula é um par zona-dia exatamente como a '
        "tabela Gold o armazena. Uma célula em branco é uma medição ausente, não "
        "um zero — a disponibilidade do parceiro é indefinida num dia sem horas "
        "programadas de funcionamento, e fica em branco em vez de ser desenhada "
        "como uma falha.</p>",
        unsafe_allow_html=True,
    )

    if not result.priorities:
        return

    top = result.priorities[0]
    anomaly = top.diagnosis.anomaly
    where = ui.scope_label(anomaly.scope, anomaly.scope_value)

    ui.section(
        "03 · O escopo que ficou em primeiro, métrica a métrica",
        f"{where} contra a sua própria linha de base ajustada por dia da semana",
        "O mesmo registro de métricas da visão da empresa, medido sobre a série "
        "do próprio escopo na mesma janela. Cada barra é uma linha de KPI do "
        "motor; nenhuma delas é calculada nesta página.",
    )
    ui.chart(
        ui.scope_delta_chart(
            ui.kpi_index(result.segment_kpis, anomaly.scope, anomaly.scope_value),
            SCOPE_METRICS,
            f"{where} — variação vs. linha de base, por métrica",
        ),
        empty_note=(
            "O motor não retornou linhas de KPI no nível do escopo para esta "
            "prioridade. Os KPIs de escopo são produzidos para os escopos que "
            "receberam um memorando."
        ),
    )
    st.markdown(
        f'<p class="pulse-note">Padrão diagnosticado: '
        f"<b>{ui.pattern_label(top.diagnosis.pattern)}</b>. {T.ASSOCIATION_NOTE}</p>",
        unsafe_allow_html=True,
    )

    ui.section(
        "04 · Quanto isso está custando",
        "Um piso, não uma previsão",
        T.IMPACT_FLOOR_NOTE,
    )
    ui.table(ui.impact_rows(top.impact, knobs["comparison_window_days"]))
    st.markdown(
        f'<p class="pulse-note">Janela de comparação '
        f"<b>{ui.fmt_date(recent_start)} → {ui.fmt_date(as_of)}</b>. Leia a lista "
        f"classificada completa acima antes de agir sobre uma linha desta tabela: "
        f"a ordenação é uma orientação sobre onde olhar, não um veredito.</p>",
        unsafe_allow_html=True,
    )


render()
