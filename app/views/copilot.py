# PULSE Copilot — ask the analysis a question, and check the answer.
#
# The boundary this page renders is the whole point of it:
#
#   TRUSTED     the evidence bundle. Engine-computed, bounded, no raw rows, no
#               row identifiers, serialised once and shown in full below.
#   UNTRUSTED   whatever the human types. It is stripped of the sentinels it
#               could otherwise forge, stripped of control characters, capped,
#               and placed in its own trailing block — never in the system
#               prompt, and never anywhere a fact is read from.
#
# pulse.copilot already implements the bundle, the sanitiser, the request
# boundary and the numeric validator, with its own test suite. This page calls
# it. It does not re-implement any part of it, and it does not add a second
# path by which a question could reach a figure.
#
# With no API key there is no model call at all: ask() renders a deterministic
# summary of the engine's own output, from templates, and says so. That state
# is labelled on every turn. A template is never presented as IA output.
from __future__ import annotations

import html

import streamlit as st

from pulse.copilot import AnthropicNarrator, ask, build_evidence_bundle

from app import components as ui
from app import state

SUGGESTED: tuple[str, ...] = (
    "Qual é a coisa mais importante desta análise?",
    "Quanto está custando a principal prioridade, e em que janela?",
    "Qual etapa do funil se moveu, e como sabemos que foi essa?",
    "O que precisaria ser verdade para este diagnóstico estar errado?",
    "Existe alguma afirmação causal nesta análise?",
    "Qual foi o ROI medido do experimento?",
    "O que devemos fazer primeiro, e como validaríamos que funcionou?",
)


def _turn(question: str, answer) -> None:
    with st.chat_message("user"):
        # Rendered as plain text, never as markdown or HTML: the question is
        # untrusted input and must not be able to style, link or hide anything
        # in the transcript it appears in.
        st.text(question)

    with st.chat_message("assistant"):
        source = (
            ui.status_pill("Narrado por IA", "info")
            if answer.is_ai_generated
            else ui.status_pill(
                "IA indisponível — resposta determinística por template, não é "
                "saída de IA", "warn"
            )
        )
        flags = tuple(
            ui.status_pill(f"Entrada sinalizada: {flag.replace('_', ' ')}", "warn")
            for flag in answer.question_flags
        )
        ui.pills(source, *ui.validation_pills(answer.validation), *flags)

        st.markdown(ui.escape_dollars(answer.answer))

        if answer.validation.causal_claims:
            ui.callout(
                "<b>Afirmação causal sinalizada.</b> Esta evidência sustenta "
                "associação, não causa — a única exceção é um experimento "
                "randomizado anexado, sobre a própria métrica primária. Frases: "
                + "; ".join(
                    f"<i>{html.escape(sentence)}</i>"
                    for sentence in answer.validation.causal_claims
                )
                + ". A detecção é por lista de expressões: uma resposta sem "
                "sinalização não está provada livre de linguagem causal."
            )

        if answer.validation.unverified_figures:
            ui.callout(
                "<b>Números não verificados são reportados, não removidos.</b> "
                "A verificação por padrões não conseguiu ligar estes números a "
                "um fato do pacote de evidências na afirmação em que aparecem — "
                "o valor pode não existir ali, ou a frase pode estar escrita de "
                "um jeito que os padrões não reconhecem: "
                + ", ".join(
                    f"<code>{f}</code>" for f in answer.validation.unverified_figures
                )
                + ". Um número apagado em silêncio é uma resposta editada em "
                "silêncio, então a divergência é exposta e quem lê decide."
            )

        with st.expander("Evidência por trás desta resposta"):
            ui.table(
                [
                    {"Campo": "Segmento", "Valor": answer.segment},
                    {"Campo": "Período", "Valor": answer.period},
                    {
                        "Campo": "Confiança",
                        "Valor": ui.fmt_float(answer.confidence, 3)
                        if answer.confidence is not None else ui.NOT_AVAILABLE,
                    },
                    {
                        "Campo": "Métricas envolvidas",
                        "Valor": ", ".join(
                            ui.metric_label(m) for m in answer.metrics_used
                        ) or "—",
                    },
                    {
                        "Campo": "Próxima ação recomendada",
                        "Valor": answer.recommended_next_action,
                    },
                ]
            )
            if answer.evidence:
                st.markdown(
                    "".join(
                        f"- {ui.escape_dollars(line)}\n" for line in answer.evidence
                    )
                )
            st.caption(
                "Segmento, período, confiança e a ação recomendada são lidos do "
                "motor depois que qualquer narração retorna, e sobrescrevem o "
                "que um modelo tenha colocado ali. Um modelo pode mudar a "
                "redação de uma resposta e nada mais."
            )


def render() -> None:
    state.page_setup()
    knobs = state.params()
    result = state.cycle(**knobs)
    bundle = build_evidence_bundle(result)
    narrator = AnthropicNarrator() if AnthropicNarrator.available() else None

    ui.page_head(
        "Comando · Copilot",
        "Pergunte à análise — e confira a resposta contra a evidência",
        "Cada figura numérica de uma resposta é verificada individualmente "
        "contra a evidência <b>estruturada</b> do motor — conceito (inclusive "
        "de qual série é o dinheiro de um escopo), escopo, unidade, "
        "significado (média diária ou total acumulado; medida, residual ou "
        "ilustrativa), direção (subiu ou caiu, melhorou ou piorou), janela "
        "(recente ou baseline) e desvio versus nível. Datas e identificadores "
        "de escopo são conferidos como datas e escopos, não como grandezas. "
        "Uma figura que não se liga à afirmação em que aparece é sinalizada, e "
        "a ligação ambígua é recusada. Frases que afirmam causa também são "
        "sinalizadas. A leitura é feita por padrões de linguagem, não por "
        "compreensão: uma afirmação falsa escrita de um jeito sem padrão conhecido "
        "pode passar sem sinalização, e o "
        "restante de uma frase em torno de uma figura ligada não é verificado.",
    )

    ui.pills(
        ui.status_pill(
            "Narração por IA disponível" if narrator is not None
            else "Sem chave de API — respostas determinísticas, sem IA",
            "info" if narrator is not None else "warn",
        ),
        ui.status_pill(
            f"Pacote de evidências: {ui.fmt_int(len(bundle.to_json()))} caracteres, "
            f"{ui.fmt_int(bundle.grounded_fact_count())} fatos estruturados",
            "info",
        ),
        ui.status_pill(
            f"{ui.fmt_int(len(bundle.priorities))} prioridades · "
            f"{ui.fmt_int(len(bundle.detected_anomalies))} anomalias no escopo",
            "info",
        ),
    )

    if narrator is None:
        ui.callout(
            "<b>Nenhum modelo de linguagem está sendo chamado.</b> "
            "<code>ANTHROPIC_API_KEY</code> não está definida, então cada "
            "resposta abaixo é um resumo determinístico da própria saída do "
            "motor, renderizado a partir de templates. Ela não é escrita para a "
            "sua pergunta específica — a pergunta nem sequer entra como insumo, "
            "que é precisamente o motivo pelo qual nenhuma pergunta, hostil ou "
            "não, consegue mover um número nela. Isso é rotulado em cada turno e "
            "nunca é apresentado como saída de IA."
        )

    ui.section(
        "01 · Comece por aqui",
        "Sete perguntas que esta análise consegue de fato responder",
        "Cada uma é respondível apenas a partir do pacote de evidências. Uma "
        "pergunta que o pacote não consegue responder recebe essa resposta, em "
        "vez de ser respondida a partir de outro lugar.",
    )
    columns = st.columns(3, gap="small")
    for index, prompt in enumerate(SUGGESTED):
        if columns[index % 3].button(prompt, key=f"copilot_prompt_{index}",
                                     use_container_width=True):
            st.session_state["copilot_pending"] = prompt

    # chat_input is called unconditionally and before the pending prompt is
    # read: a widget that only exists on some runs loses its state on the
    # others, and the box would vanish for a rerun every time a suggested
    # prompt was clicked.
    typed = st.chat_input("Pergunte sobre esta análise")
    pending = st.session_state.pop("copilot_pending", None)
    # Each turn was checked against the bundle of the parameters it was asked
    # under. When a parameter moves, those turns would sit above a different
    # bundle looking current, so the transcript starts again.
    knobs_key = tuple(sorted((k, str(v)) for k, v in knobs.items()))
    if st.session_state.get("copilot_history_knobs") != knobs_key:
        st.session_state["copilot_history"] = []
        st.session_state["copilot_history_knobs"] = knobs_key
    history: list = st.session_state.setdefault("copilot_history", [])
    question = pending or typed
    if question is None and not history:
        # Arrive with something on the page rather than an empty transcript.
        question = SUGGESTED[0]
    if question:
        history.append((question, ask(question, bundle, narrator)))

    ui.section(
        "02 · Transcrição",
        "Cada turno carrega a própria verificação",
        "As etiquetas acima de cada resposta dizem de onde vieram as palavras e "
        "se a verificação por padrões encontrou alguma divergência. Nenhuma "
        "divergência detectada não é prova de que a resposta está certa.",
    )
    for asked, answer in history:
        _turn(asked, answer)

    ui.section(
        "03 · Tudo o que o modelo tem permissão de ver",
        "O pacote de evidências, na íntegra",
        "Limitado por construção: teto em cada lista, nenhuma linha bruta, "
        "nenhum identificador de linha, nenhum DataFrame. O pacote <i>é</i> a "
        "forma serializada, não uma visão sobre objetos vivos, e o validador "
        "numérico confere, por padrões de linguagem, cada figura de uma "
        "resposta contra o fato <b>estruturado</b> deste pacote que a frase diz "
        "estar citando. Não "
        "existe atalho por correspondência de texto: nem a prosa do próprio "
        "motor é aceita por coincidir com o texto. Parte dessa prosa cita "
        "números que o pacote não carrega como fato estruturado (um segmento "
        "fora do pacote, uma projeção sem a palavra que diz que ela é "
        "acumulada) e, se reproduzida, é sinalizada. A ligação ambígua é "
        "recusada, porque uma recusa injusta custa uma conferência e um número "
        "fabricado certificado custa a decisão.",
    )
    with st.expander("Pacote de evidências (JSON)"):
        st.json(bundle.to_dict(), expanded=False)
    ui.callout(
        "<b>Como a metade não confiável é tratada.</b> A sua pergunta nunca "
        "entra no prompt de sistema e nunca alcança um número. Ela é despojada "
        "das sentinelas que poderia forjar e dos caracteres de controle que "
        "poderiam reordenar esta transcrição, tem o comprimento limitado e é "
        "colocada em um bloco próprio ao final, depois da evidência. Sinalizações "
        "de padrão são registradas e exibidas; nada se ramifica a partir delas, "
        "porque uma lista de padrões não é uma fronteira de segurança — a "
        "fronteira é que a pergunta não alcança um número, e o validador que "
        "confere os números depois.",
        quiet=True,
    )


render()
