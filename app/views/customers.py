# Customers & Retention — which cohorts and channels are deteriorating?
#
# gold_customer_retention stores one row per (cohort month, acquisition
# channel, period index) with the retention rate already computed, alongside
# the two columns that decide whether that rate can be read at all:
# period_days and period_days_observed. A period observed for ten of its thirty
# days has a retention rate that is mechanically low, and reading it as a
# deterioration is the most common way a cohort chart lies.
#
# This page pivots, filters and relabels. It computes no rate, no average and
# no trend. The channel column is relabelled on a DISPLAY COPY only -- the gold
# frame keeps its own values, because a chart legend is not a place to rewrite
# data.
from __future__ import annotations

import streamlit as st

from app import components as ui
from app import state


def render() -> None:
    state.page_setup()

    ui.page_head(
        "Análises · Clientes",
        "Quais coortes e canais estão se deteriorando?",
        "Uma tabela de coortes responde a uma pergunta que uma contagem diária "
        "de clientes ativos não responde: se os clientes que chegam agora voltam "
        "como voltavam os que chegaram antes deles. A retenção é medida por "
        "canal de aquisição, porque um canal é uma decisão que alguém toma sobre "
        "investimento.",
    )

    retention = state.gold().customer_retention
    channels = sorted(retention["acquisition_channel"].unique())
    periods = sorted(int(p) for p in retention["period_index"].unique())

    ui.section(
        "01 · Retenção por coorte, um canal de cada vez",
        "Mês da coorte por período, como a camada Gold armazena",
        "Cada linha é uma coorte de aquisição, cada coluna é o número de "
        "períodos de 30 dias desde a aquisição. O período 0 é o próprio período "
        "de aquisição e é 100% por construção, então o sinal está nas colunas à "
        "direita dele. Leia uma coluna de cima a baixo, não uma linha da "
        "esquerda para a direita: comparar o período 1 entre coortes é comparar "
        "coisas equivalentes.",
    )
    channel = st.selectbox(
        "Canal de aquisição", options=channels, key="cust_channel",
        format_func=ui.channel_label,
    )
    chosen = retention.loc[retention["acquisition_channel"] == channel]
    grid = chosen.pivot(
        index="cohort_month", columns="period_index", values="retention_rate"
    )
    ui.chart(
        ui.heatmap_chart(
            grid, "retention_rate",
            f"{ui.channel_label(channel)} — taxa de retenção por coorte e período",
            height=150 + 34 * len(grid.index),
        ),
        empty_note="gold_customer_retention não carrega linhas para este canal.",
    )

    ui.section(
        "02 · O mesmo período, todos os canais",
        "Onde os canais se separam",
        "Uma linha por canal de aquisição em um único índice de período. Canais "
        "que acompanham uns aos outros e depois se separam são o sinal legível; "
        "uma célula baixa isolada não é. Qualquer separação aqui é uma "
        "associação entre um canal e uma taxa de retenção — não está demonstrado "
        "que o canal a produziu, e a composição da coorte se move junto com "
        "investimento, segmentação e sazonalidade ao mesmo tempo.",
    )
    period = st.select_slider(
        "Períodos desde a aquisição", options=periods,
        value=periods[1] if len(periods) > 1 else periods[0],
        key="cust_period",
    )
    # A display copy: the channel identifiers stay in gold, the words go on the
    # legend and in the table.
    at_period = (
        retention.loc[retention["period_index"] == period]
        .sort_values("cohort_month")
        .assign(
            acquisition_channel=lambda f: f["acquisition_channel"].map(
                ui.channel_label
            )
        )
    )
    ui.chart(
        ui.timeline_chart(
            at_period,
            x="cohort_month",
            y="retention_rate",
            colour="acquisition_channel",
            title=f"Retenção no período {period} por coorte de aquisição",
            y_title="Taxa de retenção",
        ),
        empty_note="Nenhuma coorte alcançou este período.",
    )
    ui.table(
        at_period.loc[
            :,
            ["cohort_month", "acquisition_channel", "cohort_size",
             "retained_customers", "retention_rate", "period_days",
             "period_days_observed"],
        ].rename(columns={**ui.COLUMN_LABELS,
                          "acquisition_channel": "Canal",
                          "retained_customers": "Retidos"}),
        empty_note="Nenhuma linha neste período.",
    )

    ui.section(
        "03 · Quais células ainda não se completaram",
        "Períodos parcialmente observados, listados em vez de ocultados",
        "Um período cuja janela avança além da data da análise foi observado por "
        "menos dias do que a sua duração, então a taxa de retenção dele só pode "
        "subir. Essas células são mecanicamente baixas e <b>não</b> são "
        "evidência de coorte em deterioração. Elas estão listadas aqui em vez de "
        "descartadas, porque uma tabela de coortes que esconde silenciosamente a "
        "própria borda direita é como um artefato de censura vira um achado.",
    )
    partial = (
        retention.loc[retention["period_days_observed"] < retention["period_days"]]
        .sort_values(["cohort_month", "acquisition_channel", "period_index"])
        .assign(
            acquisition_channel=lambda f: f["acquisition_channel"].map(
                ui.channel_label
            )
        )
    )
    ui.table(
        partial.loc[
            :,
            ["cohort_month", "acquisition_channel", "period_index",
             "period_start_date", "period_days", "period_days_observed",
             "retention_rate"],
        ].rename(columns={**ui.COLUMN_LABELS,
                          "acquisition_channel": "Canal",
                          "retention_rate": "Taxa de retenção (até agora)"}),
        empty_note="Todo período desta tabela foi observado integralmente.",
    )


render()
