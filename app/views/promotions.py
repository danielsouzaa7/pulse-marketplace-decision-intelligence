# Promotions — which campaign is buying volume while destroying economics?
#
# A promotion that lifts orders is trivially easy to call a success. The
# question worth asking is what the lift cost, and gold_promotion_performance
# already carries both halves on the same campaign-day row: discount_rate and
# discount_per_completed_order on one side, margin_per_completed_order on the
# other.
#
# Every point on this page is one row of that table. Nothing is averaged,
# ranked by a computed figure, or attributed. promo_type and funded_by are
# relabelled on a DISPLAY COPY; the gold values are untouched.
from __future__ import annotations

import streamlit as st

from app import components as ui
from app import state

WORST_N = 15

# The permanent row: every day carries a NO_PROMOTION record, which is the
# baseline every campaign day should be read against rather than against zero.
# It is a code in the data, not a word on a page, so it is not translated.
BASELINE_CODE = "NO_PROMOTION"


def render() -> None:
    state.page_setup()

    ui.page_head(
        "Análises · Promoções",
        "Qual campanha está comprando volume enquanto destrói a economia?",
        "A taxa de desconto e a margem por pedido concluído estão na mesma linha "
        "Gold, então a troca que uma campanha fez fica visível sem precisar "
        "modelá-la. Quem financiou o desconto decide se ele chega a incidir "
        "sobre a margem do marketplace, e essa coluna também está na linha.",
    )

    promos = state.gold().promotion_performance
    campaigns = sorted(promos["promo_code"].unique())

    ui.section(
        "01 · A troca que cada campanha fez",
        "Taxa de desconto contra margem por pedido concluído",
        "Um ponto por campanha-dia, dimensionado pelos pedidos que levaram a "
        "promoção. O quadrante inferior direito é o caro: taxa de desconto alta "
        "e margem baixa em cada pedido que ela comprou. A nuvem "
        f"<b>{BASELINE_CODE}</b> é a mesma medição em dias sem campanha em "
        "andamento, e é a comparação que importa.",
    )
    ui.chart(
        ui.scatter_chart(
            promos,
            x="discount_rate",
            y="margin_per_completed_order",
            colour="promo_code",
            size="orders_with_promo",
            title="Economia das campanhas — um ponto por campanha-dia",
            hover=("metric_date", "promo_type", "funded_by", "gmv",
                   "orders_completed"),
            x_title="Taxa de desconto (desconto ÷ GMV)",
            y_title="Margem de contribuição por pedido concluído (R$)",
        ),
        empty_note="gold_promotion_performance não carrega linhas.",
    )

    ui.section(
        "02 · Quando aconteceu",
        "Margem por pedido concluído, dia a dia",
        "As campanhas rodam em sequência e não em paralelo, então isto se lê "
        "como uma sucessão, com a linha de base sem campanha correndo por baixo "
        "o tempo todo. Uma campanha cujo trecho fica abaixo dessa linha de base "
        "vendeu com margem pior do que vender sem promoção alguma.",
    )
    selected = st.multiselect(
        "Campanhas", options=campaigns, default=campaigns, key="promo_codes"
    )
    chosen = promos.loc[promos["promo_code"].isin(selected)].sort_values("metric_date")
    ui.chart(
        ui.timeline_chart(
            chosen,
            x="metric_date",
            y="margin_per_completed_order",
            colour="promo_code",
            title="Margem de contribuição por pedido concluído, por campanha",
            y_title="Margem por pedido concluído (R$)",
            height=420,
        ),
        empty_note="Nenhuma campanha selecionada.",
    )
    ui.chart(
        ui.timeline_chart(
            chosen,
            x="metric_date",
            y="discount_rate",
            colour="promo_code",
            title="Taxa de desconto, por campanha",
            y_title="Desconto ÷ GMV",
            height=360,
        ),
        empty_note="Nenhuma campanha selecionada.",
    )

    ui.section(
        "03 · Os dias com maior desconto",
        f"Os {WORST_N} campanha-dias com maior taxa de desconto",
        "Ordenado por uma coluna que a camada Gold já carrega. <b>Financiado "
        "por</b> é a coluna a ler primeiro: um desconto financiado pelo parceiro "
        "é margem do parceiro, enquanto um financiado pelo marketplace incide "
        "diretamente sobre a margem de contribuição pela qual este negócio é "
        "medido.",
    )
    worst = (
        chosen.sort_values("discount_rate", ascending=False)
        .head(WORST_N)
        .loc[
            :,
            ["metric_date", "promo_code", "promo_type", "funded_by",
             "orders_with_promo", "orders_completed", "gmv", "discount_rate",
             "discount_per_completed_order", "margin_per_completed_order",
             "promo_conversion"],
        ]
        .assign(
            promo_type=lambda f: f["promo_type"].map(ui.promo_type_label),
            funded_by=lambda f: f["funded_by"].map(ui.funded_by_label),
        )
        .rename(columns={**ui.COLUMN_LABELS,
                         "discount_per_completed_order": "Desconto / pedido",
                         "margin_per_completed_order": "Margem / pedido",
                         "promo_conversion": "Conversão"})
    )
    ui.table(worst, empty_note="Nenhum campanha-dia a ordenar.")
    st.markdown(
        '<p class="pulse-note">Estas linhas são uma <b>associação</b> entre uma '
        "janela de campanha e a economia observada dentro dela. As campanhas não "
        "são aleatorizadas e não se sobrepõem, então um período de campanha "
        "difere de um período sem campanha em sazonalidade, composição de dias "
        "da semana e em tudo o mais que tenha se movido naquela quinzena. O "
        "único lugar do PULSE em que uma afirmação causal é autorizada é o "
        "Laboratório de Experimentos, e apenas porque a aleatorização foi "
        "verificada lá.</p>",
        unsafe_allow_html=True,
    )


render()
