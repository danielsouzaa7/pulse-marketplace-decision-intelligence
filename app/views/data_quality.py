# Data Quality & Architecture — can I trust this analysis?
#
# Every figure on this page is read out of data/silver/_quality_report.parquet,
# the artifact the Silver build actually wrote, plus the quarantine files it
# actually wrote beside it. Nothing here is typed in by hand, and nothing here
# is recomputed: the report already carries rows_in, rows_out, rows_rejected,
# rows_repaired and the rule text for every check that ran.
#
# Two things this page exists to get right:
#
#   1. ONE TABLE CAN HAVE SEVERAL REJECT REASONS. The deliveries quarantine
#      holds rows rejected by two different checks, and a page that assumes one
#      reason per table silently hides one of them. The quarantine is shown per
#      reason, not per table.
#   2. A CHECK THAT FOUND NOTHING IS EVIDENCE. The declared foreign keys that
#      rejected zero rows are shown with their zeros, because "0 violações
#      encontradas" and "nunca verificado" look identical on a page that hides
#      the former.
#
# The layer names (Bronze, Silver, Gold) and the table and column identifiers
# are the pipeline's own names and stay as they are; only the prose around them
# is Portuguese.
from __future__ import annotations

import streamlit as st

from app import components as ui
from app import state
from app import ui_text as T

_FK_MARKER = "references"


def _flow_html(bronze: tuple[str, ...], gold: tuple[str, ...]) -> str:
    stages = (
        ("Ingestão", "Bronze", "Bruto como recebido, com os defeitos mantidos: "
         "registros de pedido duplicados, variação de caixa, taxas de entrega "
         "ausentes, zonas irresolúveis, marcações de tempo impossíveis.",
         [f"<code>{name}</code>" for name in bronze]),
        ("Validação", "Silver", "Toda verificação abaixo roda aqui. As linhas são "
         "reparadas no lugar quando o reparo é defensável, e postas em "
         "quarentena em <code>_rejected/</code> quando não é. Nada é descartado "
         "em silêncio.",
         ["Deduplicação", "Integridade referencial", "Causalidade temporal",
          "Normalização de categorias", "Imputação numérica, sinalizada"]),
        ("Agregação", "Gold", "O grão analítico que o motor lê. Construído em "
         "SQL, com calendário completo, uma linha por entidade por dia.",
         [f"<code>{name}</code>" for name in gold]),
        ("Decisão", "Engine", "run_decision_cycle() — métricas, anomalias, "
         "diagnóstico, impacto, prioridade, memorando. O único lugar onde um "
         "número de negócio é calculado.",
         ["Cada página do PULSE renderiza esta saída e não recalcula nada dela."]),
    )
    blocks = []
    for index, (kind, name, note, items) in enumerate(stages):
        lead = " lead" if name == "Engine" else ""
        listed = "".join(f"<li>{item}</li>" for item in items)
        blocks.append(
            f'<div class="stage{lead}"><div class="kind">{kind}</div>'
            f'<div class="name">{name}</div>'
            f'<p class="pulse-note" style="margin-top:4px">{note}</p>'
            f"<ul>{listed}</ul></div>"
        )
        if index < len(stages) - 1:
            blocks.append('<div class="arrow">→</div>')
    return f'<div class="pulse-flow">{"".join(blocks)}</div>'


def render() -> None:
    state.page_setup()

    ui.page_head(
        "Plataforma · Qualidade de dados",
        "Posso confiar nesta análise?",
        "Uma análise vale o que valem as linhas por baixo dela. Tudo nesta "
        "página é lido do relatório de qualidade e dos arquivos de quarentena "
        "que a construção da camada Silver gravou — não de um resumo escrito "
        "sobre eles.",
    )

    report = state.quality_report()

    ui.section(
        "01 · Como uma linha vira uma decisão",
        "Quatro estágios, e o único lugar onde um número de negócio é calculado",
        "Cada estágio faz um trabalho, e a fronteira entre validação e análise é "
        "deliberada: a decisão de reparo é tomada uma vez, na camada Silver, "
        "onde fica registrada — nunca dentro de uma análise, onde ficaria "
        "invisível.",
    )
    st.markdown(
        _flow_html(state.bronze_tables(), state.gold_tables()),
        unsafe_allow_html=True,
    )

    ui.section(
        "02 · O que a camada Bronze trouxe, e o que a Silver deixou passar",
        "Por tabela, a partir da primeira e da última verificação do relatório",
        "O relatório é escrito na ordem do pipeline, então a primeira "
        "verificação de uma tabela carrega a contagem de linhas Bronze que ela "
        "recebeu e a última carrega a contagem de linhas Silver que ela "
        "publicou. As duas são selecionadas do relatório, não contadas aqui.",
    )
    first = report.drop_duplicates(subset="table", keep="first")
    last = report.drop_duplicates(subset="table", keep="last")
    counts = first.loc[:, ["table", "rows_in"]].merge(
        last.loc[:, ["table", "rows_out"]], on="table"
    ).rename(
        columns={"table": "Tabela", "rows_in": "Linhas de entrada (Bronze)",
                 "rows_out": "Linhas de saída (Silver)"}
    )
    ui.table(counts, empty_note="O relatório de qualidade não carrega verificações.")

    ui.section(
        "03 · Toda verificação que rodou",
        f"{len(report)} verificações, com a regra que cada uma impõe",
        "<b>rows_repaired</b> e <b>rows_rejected</b> são desfechos diferentes e "
        "são reportados em separado. Um reparo é um valor normalizado ou imputado "
        "no lugar, com a linha mantida e a imputação sinalizada; uma rejeição é "
        "uma linha movida para quarentena porque nenhum reparo era defensável.",
    )
    ui.table(
        report.loc[
            :,
            ["check_name", "table", "rows_in", "rows_out", "rows_rejected",
             "rows_repaired", "rule"],
        ].rename(
            columns={
                "check_name": "Verificação",
                "table": "Tabela",
                "rows_in": "Linhas de entrada",
                "rows_out": "Linhas de saída",
                "rows_rejected": "Rejeitadas",
                "rows_repaired": "Reparadas",
                "rule": "Regra imposta",
            }
        ),
        empty_note="O relatório de qualidade não carrega verificações.",
    )

    ui.section(
        "04 · As chaves estrangeiras declaradas, incluindo as que não acharam nada",
        "Uma verificação que reporta zero é evidência de que ela rodou",
        "A integridade referencial é imposta entre tabelas Silver, não presumida "
        "a partir da carga Bronze. Uma linha-filha cujo pai foi posto em "
        "quarentena também vai para a quarentena, de modo que nenhuma tabela "
        "limpa fica apontando para uma linha que a Silver removeu. As "
        "verificações que rejeitaram zero linhas estão listadas com os seus "
        "zeros: escondê-las tornaria \"0 violações encontradas\" "
        "indistinguível de \"nunca verificado\".",
    )
    foreign_keys = report.loc[report["check_name"].str.contains(_FK_MARKER)]
    ui.table(
        foreign_keys.loc[
            :, ["check_name", "table", "rows_in", "rows_rejected", "rule"]
        ].rename(
            columns={
                "check_name": "Relação imposta",
                "table": "Tabela-filha",
                "rows_in": "Linhas verificadas",
                "rows_rejected": "Rejeitadas",
                "rule": "Regra",
            }
        ),
        empty_note="Nenhuma verificação de integridade referencial está declarada "
                   "no relatório.",
    )

    ui.section(
        "05 · A quarentena, por motivo",
        "Linhas rejeitadas são movidas, nunca descartadas",
        "Agrupadas por <b>reject_reason</b> em vez de por tabela, porque uma "
        "mesma tabela pode ser posta em quarentena por várias verificações "
        "diferentes — uma violação de marcação de tempo e um pai órfão são "
        "falhas distintas com correções distintas, e um total por tabela as "
        "reportaria como um número só. Um arquivo de quarentena vazio também "
        "aparece: significa que a verificação rodou e não achou nada, o que não "
        "é a mesma coisa que uma verificação que nunca rodou.",
    )
    rows = []
    for table in state.quarantine_tables():
        frame = state.quarantine(table)
        if frame.empty or "reject_reason" not in frame.columns:
            rows.append(
                {
                    "Arquivo de quarentena": f"_rejected/{table}.parquet",
                    "Motivo da rejeição": T.QUARANTINE_EMPTY_REASON,
                    "Linhas": 0,
                }
            )
            continue
        for reason in sorted(frame["reject_reason"].unique()):
            rows.append(
                {
                    "Arquivo de quarentena": f"_rejected/{table}.parquet",
                    "Motivo da rejeição": str(reason),
                    # A census of the rows in a file, not a business figure:
                    # the quality report already publishes the same totals
                    # per check, and these two agree by construction.
                    "Linhas": len(frame.loc[frame["reject_reason"] == reason]),
                }
            )
    ui.table(
        rows,
        empty_note="Nenhum diretório de quarentena foi gravado pela construção "
                   "da camada Silver.",
    )
    ui.callout(
        "<b>O que isto autoriza e o que não autoriza.</b> Estas verificações "
        "estabelecem que as linhas que alimentam o motor estão deduplicadas, "
        "referencialmente íntegras e ordenadas causalmente no tempo, e que cada "
        "linha removida ainda pode ser inspecionada. Elas não estabelecem que as "
        "medições estão corretas, que a amostra é representativa, nem que "
        "qualquer relação na análise seja causal. Todos os dados do PULSE são "
        "sintéticos e gerados com defeitos injetados conhecidos, que é "
        "exatamente o motivo pelo qual estas contagens valem ser mostradas: elas "
        "são a evidência de que o pipeline pegou o que foi colocado ali para ele "
        "pegar.",
        quiet=True,
    )


render()
