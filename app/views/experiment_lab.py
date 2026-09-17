# Experiment Lab — did the proposed intervention work?
#
# pulse.experiments.analyse_experiment() IS the analysis: the two-proportion
# test, the randomisation checks, the power calculation and the incremental
# economics all live there and are tested there. This page renders the
# ExperimentReport it returns and reimplements none of it.
#
# Three things this page must not do, each of which is how an experiment
# write-up usually goes wrong:
#
#   1. Dress up a null. The result here is not significant, and the honest
#      presentation of a null is the null plus the power analysis that makes it
#      informative — not a softer word for "no effect". "Não foi detectado
#      efeito estatisticamente significativo" says there is no evidence of an
#      effect; it does not say there is evidence of no effect, and the two are
#      different findings.
#   2. Quote the relative effect alone. A small absolute move on a small base
#      reads large in relative terms, so the absolute effect leads (in points,
#      via fmt_pp) and the relative one sits beside it (in percent, via
#      fmt_ratio). The two formatters exist to keep those apart.
#   3. Present an ROI as a measured return. The dataset carries no
#      incentive-value column at all, and the only cost-shaped quantity in it --
#      the observed discount -- is carried by BOTH arms, with the CONTROL arm
#      carrying more. So there is no treatment-specific incremental cost to
#      divide by, no measured ROI exists, and none is shown: report.measured_roi
#      is None and report.business_verdict is "unavailable". The incremental
#      arithmetic is still rendered, labelled as the methodological example it
#      is, because the method is the part most write-ups get wrong.
#
#   4. Let a proxy-derived threshold carry a measured verdict. The break-even
#      lift divides by that same proxy cost, so the threshold, the
#      required_n_per_arm derived from it and the "powered for the decision"
#      verdict are all proxy-dependent and are labelled off
#      report.economic_evaluation_status WHERE THEY APPEAR -- not two sections
#      later, which is where the caveat used to live. The MDE is not: it needs
#      only the control rate and the arm sizes, so it stays a measured figure
#      and section 03 is worded as a statement about statistical power.
#
# This is also the ONLY page in PULSE licensed to use causal language, and only
# because the randomisation checks below actually passed.
from __future__ import annotations

import streamlit as st

from pulse.experiments import (
    ECONOMIC_STATUS_MEASURED,
    NO_MEASURED_ROI_NOTE,
    PROXY_METHOD_NOTE,
)

from app import components as ui
from app import state
from app import ui_text as T


def render() -> None:
    state.page_setup()

    ui.page_head(
        "Plataforma · Experimentos",
        "A intervenção proposta funcionou?",
        "Em todo o resto do PULSE um direcionador é uma associação. Aqui, e "
        "somente aqui, a aleatorização foi verificada antes de o efeito ser "
        "lido — é isso que torna legítima uma leitura causal do tratamento "
        "nesta página e ilegítima em todas as outras.",
    )

    results = state.gold().experiment_results
    ids = sorted(results["experiment_id"].unique())
    experiment_id = st.selectbox("Experimento", options=ids, key="exp_id")
    report = state.experiment(experiment_id)
    test, economics, checks = report.test, report.economics, report.randomisation
    metric = ui.metric_label(report.primary_metric)

    ui.section(
        "01 · A hipótese, e se a aleatorização se sustenta",
        f"{report.experiment_id} — {ui.fmt_date(report.start_date)} → "
        f"{ui.fmt_date(report.end_date)}",
        "Tudo abaixo desta seção é condicional a estas quatro verificações. Se "
        "elas falharem, o valor-p, o intervalo e sobretudo a palavra 'causal' "
        "deixam de significar o que parecem significar.",
    )
    ui.callout(f"<b>Hipótese.</b> {report.hypothesis}")
    st.markdown('<div style="margin-top:12px"></div>', unsafe_allow_html=True)
    ui.pills(
        ui.status_pill(
            f"Desequilíbrio na razão da amostra: p = "
            f"{ui.fmt_float(checks.srm_p_value, 3)}",
            "ok" if checks.srm_passed else "crit",
        ),
        ui.status_pill(
            f"{ui.fmt_int(checks.duplicate_assignments)} atribuições duplicadas",
            "ok" if checks.duplicate_assignments == 0 else "crit",
        ),
        ui.status_pill(
            f"{ui.fmt_int(checks.cross_variant_customers)} clientes nos dois braços",
            "ok" if checks.cross_variant_customers == 0 else "crit",
        ),
        ui.status_pill(
            "Linguagem causal autorizada" if report.causal_language_licensed
            else "Linguagem causal NÃO autorizada",
            "ok" if report.causal_language_licensed else "crit",
        ),
    )
    ui.table(
        [
            {
                "Verificação": "Braço de controle",
                "Valor": f"{ui.fmt_int(checks.control_n)} clientes",
            },
            {
                "Verificação": "Braço de tratamento",
                "Valor": f"{ui.fmt_int(checks.treatment_n)} clientes",
            },
            {
                "Verificação": "Participação observada do tratamento",
                "Valor": f"{ui.fmt_ratio(checks.observed_treatment_share, 4)} "
                         f"(pretendida {ui.fmt_ratio(checks.intended_treatment_share, 0)})",
            },
            {
                "Verificação": "Clientes que converteram (controle / tratamento)",
                "Valor": f"{ui.fmt_int(test.control_x)} / "
                         f"{ui.fmt_int(test.treatment_x)}",
            },
            {
                "Verificação": "Pedidos excluídos antes da atribuição",
                "Valor": ui.fmt_int(report.window.excluded_pre_assignment),
            },
            {
                "Verificação": "Pedidos excluídos após o fechamento da janela",
                "Valor": ui.fmt_int(report.window.excluded_post_window),
            },
        ]
    )

    ui.section(
        "02 · O efeito",
        f"{metric} — {T.statistical_verdict_label(report.statistical_verdict)}",
        "O efeito absoluto vem primeiro porque é o que não pode ser inflado por "
        "uma base pequena. O número relativo é o mesmo efeito dividido pela taxa "
        "do controle, e citá-lo sozinho é como um movimento deste tamanho é "
        "vendido como resultado.",
    )
    ui.card_row(
        [
            ui.fact_card_html(
                "Efeito absoluto", ui.fmt_pp(test.absolute_diff),
                f"{ui.fmt_value(test.control_rate, 'ratio')} → "
                f"{ui.fmt_value(test.treatment_rate, 'ratio')}",
                "flat", hero=True,
            ),
            ui.fact_card_html(
                "Efeito relativo", ui.fmt_ratio(test.relative_uplift, 1, signed=True),
                "o mesmo movimento, dividido pela taxa do controle", "flat",
                hero=True,
            ),
            ui.fact_card_html(
                "Intervalo de confiança de 95%",
                f"{ui.fmt_pp(test.ci_95[0])} … {ui.fmt_pp(test.ci_95[1])}",
                "cruza o zero" if test.ci_95[0] < 0 < test.ci_95[1]
                else "não cruza o zero",
                "crit" if test.ci_95[0] < 0 < test.ci_95[1] else "pos", hero=True,
            ),
            ui.fact_card_html(
                "Valor-p", ui.fmt_float(test.p_value, 4),
                f"contra α = {ui.fmt_float(test.alpha, 2)}", "flat", hero=True,
            ),
        ],
        per_row=4,
    )
    ui.chart(
        ui.interval_chart(
            test.absolute_diff, test.ci_95[0], test.ci_95[1],
            f"Efeito absoluto sobre {metric.lower()}, com o intervalo de 95%",
            "Pontos percentuais contra o controle",
        )
    )
    if test.significant:
        ui.callout(
            f"<b>O intervalo exclui o zero.</b> Com p = "
            f"{ui.fmt_float(test.p_value, 4)}, a hipótese nula de ausência de "
            f"efeito é rejeitada a α = {ui.fmt_float(test.alpha, 2)}."
        )
    else:
        ui.callout(
            f"<b>Não foi detectado efeito estatisticamente significativo, e esse "
            f"é o achado.</b> O intervalo vai de {ui.fmt_pp(test.ci_95[0])} a "
            f"{ui.fmt_pp(test.ci_95[1])} e cruza o zero, então com p = "
            f"{ui.fmt_float(test.p_value, 4)} a hipótese nula de ausência de "
            f"efeito não pode ser rejeitada. Os {ui.fmt_pp(test.absolute_diff)} "
            f"observados são o que uma amostra deste tamanho produz por acaso "
            f"com frequência suficiente para não carregar evidência de um "
            f"movimento real. Isto é ausência de evidência de efeito, e não "
            f"evidência de ausência de efeito: não é uma vitória pequena, e não "
            f"é um resultado à espera de mais dados — a análise de poder abaixo "
            f"explica por quê."
        )

    # The two halves of this section do NOT rest on the same evidence, so they
    # are not worded as if they did. The MDE needs the control rate and the arm
    # sizes; the break-even lift divides by economics.incentive_brl, a proxy
    # cost. The qualifier is read off the engine's own status field so the page
    # cannot keep saying "illustrative" if the data ever stops being a proxy.
    econ_measured = report.economic_evaluation_status == ECONOMIC_STATUS_MEASURED
    econ_label = T.economic_status_label(report.economic_evaluation_status)
    econ_foot = f"avaliação econômica {econ_label}"
    econ_tone = "flat" if econ_measured else "warn"
    econ_note = "" if econ_measured else f"{econ_foot.capitalize()}. {PROXY_METHOD_NOTE}"

    ui.section(
        "03 · O desenho tinha poder para detectar um efeito?",
        "Um resultado nulo só é informativo com o menor efeito detectável ao "
        "lado dele",
        "O menor efeito detectável é medido: precisa apenas da taxa do controle "
        "e do tamanho dos braços, e é ele que diz qual efeito este desenho "
        "conseguia resolver. O limiar de equilíbrio ao lado dele sai da economia "
        f"do incentivo, e vale o que vale o custo que o alimenta. {econ_note}",
    )
    per_arm = min(test.control_n, test.treatment_n)
    ui.card_row(
        [
            ui.fact_card_html(
                "Menor efeito detectável com 80% de poder",
                ui.fmt_pp(report.mde_at_80_power),
                f"medido, com {ui.fmt_int(per_arm)} clientes por braço",
            ),
            ui.fact_card_html(
                "Ganho de equilíbrio",
                ui.fmt_pp(economics.breakeven_absolute_lift),
                econ_foot, econ_tone,
            ),
            ui.fact_card_html(
                "Clientes por braço para esse limiar",
                ui.fmt_int(report.required_n_per_arm),
                f"contra {ui.fmt_int(per_arm)} executados · {econ_foot}",
                econ_tone,
            ),
            ui.fact_card_html(
                "Com poder para esse limiar",
                "Sim" if report.adequately_powered else "Não",
                f"com 80% de poder · {econ_foot}",
                ("ok" if report.adequately_powered else "crit") if econ_measured
                else "warn",
            ),
        ],
        per_row=4,
    )
    if report.adequately_powered and not test.significant:
        ui.callout(
            f"<b>Um resultado nulo com poder estatístico adequado ao efeito que "
            f"este desenho media, não um resultado inconclusivo.</b> Com "
            f"{ui.fmt_int(per_arm)} clientes por braço, qualquer efeito de pelo "
            f"menos {ui.fmt_pp(report.mde_at_80_power)} teria sido detectado com "
            f"80% de poder, e nenhum foi — essa é a parte medida. O limiar de "
            f"equilíbrio de {ui.fmt_pp(economics.breakeven_absolute_lift)}, e os "
            f"{ui.fmt_int(report.required_n_per_arm)} clientes por braço que ele "
            f"exigiria, dividem pelo custo proxy: se esse limiar fosse o critério "
            f"comercial, este desenho o cobriria — mas o critério não está "
            f"medido. {econ_note} Rodar por mais tempo reduziria o menor efeito "
            f"detectável e responderia a outra pergunta: se existe um efeito "
            f"abaixo de {ui.fmt_pp(report.mde_at_80_power)}, e não a esta."
        )
    elif not report.adequately_powered:
        ui.callout(
            f"<b>Sem poder para o limiar de equilíbrio.</b> Com "
            f"{ui.fmt_int(per_arm)} clientes por braço, o desenho só conseguia "
            f"resolver {ui.fmt_pp(report.mde_at_80_power)} — essa é a parte "
            f"medida — aquém do limiar de equilíbrio de "
            f"{ui.fmt_pp(economics.breakeven_absolute_lift)}. {econ_note} Quanto "
            f"a esse limiar, portanto, este resultado nulo não conclui nada."
        )

    ui.section(
        "04 · Economia incremental",
        "Um método trabalhado — e nenhum ROI medido para reportar",
        "A aritmética abaixo é a parte que a maioria dos relatórios erra, e vale "
        "mostrar por si só: o incentivo é pago em <b>cada</b> resgate do grupo "
        "tratado, enquanto a margem só é ganha nos <b>incrementais</b>. Clientes "
        "que voltariam de qualquer forma custam o subsídio mesmo assim. Lançar o "
        "incentivo apenas contra os pedidos incrementais é o erro que transforma "
        "uma promoção deficitária em uma vitória reportada.",
    )
    # The absence first, in the engine's own words, then the table. Putting the
    # arithmetic first and the caveat after is how a proxy gets read as a result.
    ui.callout(
        f"{T.ROI_UNAVAILABLE_HEADING} {NO_MEASURED_ROI_NOTE} {T.ROI_PROXY_WARNING}"
    )
    st.markdown('<div style="margin-top:12px"></div>', unsafe_allow_html=True)
    ui.card_row(
        [
            ui.fact_card_html(
                "ROI de tratamento medido", ui.NOT_AVAILABLE,
                "sem custo incremental identificável do tratamento", "crit",
            ),
            ui.fact_card_html(
                "Situação da avaliação econômica",
                T.economic_status_label(report.economic_evaluation_status),
                PROXY_METHOD_NOTE, "warn",
            ),
            ui.fact_card_html(
                "Veredito comercial",
                T.business_verdict_label(report.business_verdict),
                "nenhum veredito econômico é derivado de um custo proxy", "warn",
            ),
        ],
        per_row=3,
    )
    st.markdown('<div style="margin-top:12px"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="pulse-note"><b>{PROXY_METHOD_NOTE}</b> Cada linha abaixo é '
        f"aritmética correta sobre um custo que não é do tratamento.</p>",
        unsafe_allow_html=True,
    )
    ui.table(
        [
            {
                "Componente": "Resgates do tratamento sobre os quais o incentivo "
                              "foi pago",
                "Valor": ui.fmt_int(test.treatment_x),
            },
            {
                "Componente": "Conversões do controle, escaladas ao braço de "
                              "tratamento",
                "Valor": ui.fmt_float(economics.control_conversions, 1),
            },
            {
                "Componente": "Conversões incrementais",
                "Valor": ui.fmt_float(economics.incremental_orders, 1),
            },
            {
                "Componente": "Incentivo por resgate (proxy)",
                "Valor": ui.fmt_brl(economics.incentive_brl),
            },
            {
                "Componente": "Custo total do incentivo (proxy)",
                "Valor": ui.fmt_brl(economics.incentive_cost),
            },
            {
                "Componente": "Margem por pedido concluído (medida)",
                "Valor": ui.fmt_brl(economics.margin_per_order_brl),
            },
            {
                "Componente": "Pedidos concluídos por cliente convertido (medido)",
                "Valor": ui.fmt_float(economics.downstream_multiplier, 3),
            },
            {
                "Componente": "Margem incremental",
                "Valor": ui.fmt_brl(economics.incremental_margin),
            },
            {
                "Componente": "Razão sobre o custo proxy (não é ROI medido)",
                "Valor": ui.fmt_ratio(economics.roi, 1, signed=True),
            },
        ]
    )

    ui.section(
        "05 · O veredito, nas palavras do próprio analisador",
        f"Estatisticamente {T.statistical_verdict_label(report.statistical_verdict)} "
        f"· avaliação econômica "
        f"{T.business_verdict_label(report.business_verdict)}",
        "Escrito por pulse.experiments, não por esta página. A mesma frase "
        "aparece onde quer que o relatório seja renderizado.",
    )
    ui.callout(report.conclusion)
    if report.warnings:
        st.markdown('<div style="margin-top:12px"></div>', unsafe_allow_html=True)
        for warning in report.warnings:
            ui.callout(warning, quiet=True)


render()
