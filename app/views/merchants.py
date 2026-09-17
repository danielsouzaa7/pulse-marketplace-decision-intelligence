# Merchants (Parceiros) — where is supply constraining demand, rather than
# demand being absent?
#
# The question matters because the two look identical in a GMV chart. A zone
# whose merchants are shut has demand arriving and finding nothing to buy; a
# zone nobody is shopping in has no demand to serve. gold_merchant_performance
# carries availability_rate (available_hours / scheduled_open_hours) beside
# orders on the same merchant-day row, which separates them without computing
# anything.
#
# "Parceiro" rather than "comerciante": this synthetic marketplace is a
# food-delivery business, and a restaurant on a delivery platform is a partner
# in Brazilian usage. The gold column stays `merchant_id` -- only its label
# changed.
#
# The zone selector defaults to whichever zone the engine ranked highest this
# run. No zone is named in this file.
from __future__ import annotations

import streamlit as st

from app import components as ui
from app import state

WORST_N = 15


def render() -> None:
    state.page_setup()
    knobs = state.params()
    result = state.cycle(**knobs)
    analysis_start, recent_start, as_of = state.span(knobs)

    ui.page_head(
        "Análises · Parceiros",
        "Onde a oferta está restringindo a demanda, em vez de a demanda estar "
        "ausente?",
        "Os dois casos aparecem como queda de GMV. Só um deles se resolve "
        "ligando para os parceiros. A disponibilidade do parceiro é a parcela "
        "das horas programadas de funcionamento efetivamente atendida, e ela "
        "está na mesma linha Gold dos pedidos que essas horas receberam ou "
        "deixaram de receber.",
    )

    merchants = state.gold().merchant_performance
    zone_ids = sorted(merchants["zone_id"].unique())
    leading = state.leading_zone(result, zone_ids)
    default_index = zone_ids.index(int(leading)) if leading is not None else 0

    ui.section(
        "01 · Disponibilidade dentro de uma zona",
        "Parceiro a parceiro, ao longo do período de análise",
        "O seletor abre na zona que o motor classificou em primeiro nesta "
        "execução — seja qual for — e todas as outras estão a um clique. Uma "
        "linha de células escuras é um parceiro que deixou de atender as "
        "próprias horas programadas; uma célula em branco é um dia em que ele "
        "não estava programado para abrir, o que não é uma falha e não é "
        "desenhado como tal.",
    )
    zone = st.selectbox(
        "Zona", options=zone_ids, index=default_index, key="merch_zone",
        format_func=lambda z: ui.scope_label("zone", z),
    )
    span_rows = state.window_slice(merchants, analysis_start, as_of)
    in_zone = span_rows.loc[span_rows["zone_id"] == zone]
    grid = in_zone.pivot(
        index="merchant_id", columns="metric_date", values="availability_rate"
    )
    ui.chart(
        ui.heatmap_chart(
            grid, "availability_rate",
            f"{ui.scope_label('zone', zone)} — disponibilidade dos parceiros por dia",
            height=140 + 20 * len(grid.index),
        ),
        empty_note="gold_merchant_performance não carrega linhas para esta zona "
                   "neste período.",
    )

    ui.section(
        "02 · Restrição ou ausência",
        f"Cada par parceiro-dia na janela de comparação, "
        f"{ui.fmt_date(recent_start)} → {ui.fmt_date(as_of)}",
        "Um ponto por parceiro por dia, em todas as zonas. Pontos baixos nos "
        "dois eixos estão restritos pela oferta — as horas não foram atendidas e "
        "os pedidos não chegaram. Pontos altos em disponibilidade e baixos em "
        "pedidos são um diagnóstico inteiramente diferente: o parceiro estava "
        "aberto e a demanda não veio. Essa distinção é todo o propósito da "
        "página, e nenhuma das duas leituras é uma afirmação causal.",
    )
    window_rows = state.window_slice(merchants, recent_start, as_of)
    points = window_rows.dropna(subset=["availability_rate"]).assign(
        zone=lambda f: f["zone_id"].astype(str)
    )
    ui.chart(
        ui.scatter_chart(
            points,
            x="availability_rate",
            y="orders_placed",
            colour="zone",
            title="Disponibilidade do parceiro contra pedidos realizados — um "
                  "ponto por parceiro-dia",
            hover=("merchant_id", "metric_date", "gmv", "completion_rate"),
            x_title="Disponibilidade do parceiro (horas disponíveis ÷ horas "
                    "programadas)",
            y_title="Pedidos realizados",
        ),
        empty_note="Nenhum par parceiro-dia com taxa de disponibilidade definida "
                   "nesta janela.",
    )

    ui.section(
        "03 · Os pares parceiro-dia que menos atenderam o que haviam programado",
        f"As {WORST_N} menores taxas de disponibilidade na janela de comparação",
        "Ordenado por uma coluna que a camada Gold já carrega. As horas "
        "programadas e as disponíveis aparecem ao lado da taxa para que o "
        "denominador fique visível: uma taxa de 50% sobre duas horas programadas "
        "é uma conversa diferente de uma taxa de 50% sobre quatorze.",
    )
    worst = (
        points.sort_values("availability_rate")
        .head(WORST_N)
        .loc[
            :,
            [
                "metric_date", "merchant_id", "zone_id", "scheduled_open_hours",
                "available_hours", "availability_rate", "orders_placed",
                "orders_completed", "gmv",
            ],
        ]
        .rename(columns={**ui.COLUMN_LABELS,
                         "availability_rate": "Disponibilidade"})
    )
    ui.table(worst, empty_note="Nenhum par parceiro-dia a ordenar nesta janela.")
    st.markdown(
        '<p class="pulse-note">A disponibilidade é detectável no grão de '
        "<b>zona</b> e não no grão de parceiro, e é por isso que o motor varre "
        "as zonas em busca dela. Cada parceiro aqui fecha num dia fixo da "
        "semana, então um dia da semana inteiro da série de cada parceiro fica "
        "indefinido, a linha de base daquele dia da semana não existe, e o "
        "detector de anomalias corretamente recusa a série. As linhas acima são, "
        "portanto, evidência a ser lida, não anomalias que o motor disparou.</p>",
        unsafe_allow_html=True,
    )


render()
