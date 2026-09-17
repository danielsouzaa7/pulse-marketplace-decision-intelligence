# Every colour, font size and chart default for the PULSE surface, in one
# place. One Plotly template is registered and made the default, so no chart
# anywhere in the app sets a colour, a font or a margin of its own; one CSS
# block is injected once.
#
# The template carries the sequential colourscale too, so a heatmap on any page
# inherits it the same way a bar chart inherits the colorway. The only per-chart
# colour decision left anywhere is `reversescale`, which is driven by
# components._LOWER_IS_BETTER -- a direction-of-travel judgement, not a palette.
#
# Nothing here reads data and nothing here computes. This module is allowed to
# know what "bad" looks like in a chart, never what "bad" is in the business.
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Matches .streamlit/config.toml. Changing a colour means changing both.
BG = "#0B1220"
SURFACE = "#131C2E"
LINE = "#1E2A42"
TEXT = "#E6EDF7"
MUTED = "#8A9BB8"
ACCENT = "#4C7DFF"

# Semantic. Used only through tone() in components.py, never chosen inline.
POSITIVE = "#17B26A"
WARNING = "#F59E0B"
CRITICAL = "#F04438"
NEUTRAL = "#5B6E92"

TEMPLATE = "pulse_dark"

# Ten, because the promotions surface draws eleven campaign series and a
# six-colour cycle makes two campaigns share a colour on the same axis.
COLORWAY = [
    ACCENT, POSITIVE, WARNING, CRITICAL, "#9B8AFB",
    "#6BD6E8", "#F472B6", "#A3E635", "#FB923C", "#38BDF8",
]

# Worse -> better, through a neutral middle. A chart of a metric where LOW is
# the bad end reads directly; one where HIGH is the bad end sets reversescale.
HEAT = [
    [0.00, "#5B1A1F"],
    [0.25, CRITICAL],
    [0.50, NEUTRAL],
    [0.75, "#0F7A48"],
    [1.00, POSITIVE],
]

FONT = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, '
    "Helvetica, Arial, sans-serif"
)


def register_template() -> None:
    """Register `pulse_dark` and make it the Plotly default.

    Idempotent: Streamlit re-runs the whole script on every widget interaction,
    so this is called many times per session, and once per page on top of that.
    """
    pio.templates[TEMPLATE] = go.layout.Template(
        layout=dict(
            paper_bgcolor=BG,
            plot_bgcolor=BG,
            font=dict(family=FONT, color=TEXT, size=13),
            title=dict(font=dict(size=14, color=TEXT), x=0, xanchor="left"),
            colorway=COLORWAY,
            colorscale=dict(sequential=HEAT, sequentialminus=HEAT, diverging=HEAT),
            xaxis=dict(
                gridcolor=LINE,
                zerolinecolor=LINE,
                linecolor=LINE,
                tickfont=dict(color=MUTED, size=12),
                title=dict(font=dict(color=MUTED, size=12)),
            ),
            yaxis=dict(
                gridcolor=LINE,
                zerolinecolor=LINE,
                linecolor=LINE,
                tickfont=dict(color=MUTED, size=12),
                title=dict(font=dict(color=MUTED, size=12)),
            ),
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                font=dict(color=MUTED, size=12),
                orientation="h",
                yanchor="bottom",
                y=1.02,
                x=0,
            ),
            margin=dict(l=8, r=8, t=44, b=8),
            hoverlabel=dict(bgcolor=SURFACE, bordercolor=LINE, font=dict(color=TEXT)),
            bargap=0.34,
        )
    )
    pio.templates.default = TEMPLATE


_CSS = f"""
<style>
  :root {{
    --pulse-bg: {BG};
    --pulse-surface: {SURFACE};
    --pulse-line: {LINE};
    --pulse-text: {TEXT};
    --pulse-muted: {MUTED};
    --pulse-accent: {ACCENT};
    --pulse-pos: {POSITIVE};
    --pulse-warn: {WARNING};
    --pulse-crit: {CRITICAL};
    --pulse-neutral: {NEUTRAL};
  }}

  #MainMenu, footer {{visibility: hidden;}}
  .stAppDeployButton, .stAppHeader {{display: none;}}
  .block-container {{padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1380px;}}
  [data-testid="stSidebar"] {{border-right: 1px solid var(--pulse-line);}}

  /* --- masthead --- */
  .pulse-mast {{display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap;}}
  .pulse-mast .name {{
    font-size: 30px; font-weight: 700; letter-spacing: -0.02em;
    color: var(--pulse-text);
  }}
  .pulse-mast .mark {{color: var(--pulse-accent); margin-right: 6px;}}
  .pulse-mast .tag {{color: var(--pulse-muted); font-size: 14px;}}

  /* --- section headers --- */
  .pulse-eyebrow {{
    color: var(--pulse-muted); font-size: 11px; font-weight: 600;
    letter-spacing: .14em; text-transform: uppercase; margin-bottom: 2px;
  }}
  .pulse-h2 {{
    color: var(--pulse-text); font-size: 20px; font-weight: 650;
    letter-spacing: -0.01em; margin: 0 0 4px 0;
  }}
  .pulse-note {{color: var(--pulse-muted); font-size: 13px; line-height: 1.55; margin: 0;}}
  .pulse-note b {{color: var(--pulse-text); font-weight: 600;}}

  /* The question a page answers, set above its own title. */
  .pulse-question {{
    color: var(--pulse-text); font-size: 25px; font-weight: 660;
    letter-spacing: -0.015em; margin: 0 0 6px 0; line-height: 1.28;
  }}

  /* --- KPI card --- */
  /* Every card is the same height and nothing inside it wraps mid-number:
     a KPI grid that reflows "R$ 29,925" onto two lines is unreadable at the
     narrow column widths a six-across row produces. */
  .pulse-kpi {{
    background: var(--pulse-surface); border: 1px solid var(--pulse-line);
    border-radius: 10px; padding: 13px 15px 12px 15px;
    height: 100%; min-height: 116px;
    display: flex; flex-direction: column; overflow: hidden;
  }}
  .pulse-kpi .label {{
    color: var(--pulse-muted); font-size: 10.5px; font-weight: 600;
    letter-spacing: .08em; text-transform: uppercase; line-height: 1.35;
    min-height: 28px;
  }}
  .pulse-kpi .value {{
    color: var(--pulse-text); font-size: clamp(18px, 1.65vw, 26px);
    font-weight: 650; line-height: 1.2; margin-top: auto;
    white-space: nowrap; font-variant-numeric: tabular-nums;
  }}
  .pulse-kpi .delta {{
    font-size: 13px; font-weight: 600; margin-top: 4px; white-space: nowrap;
    font-variant-numeric: tabular-nums;
  }}
  .pulse-kpi .delta .foot {{
    display: block; color: var(--pulse-muted); font-weight: 400;
    font-size: 11.5px; margin-top: 1px;
  }}
  .pulse-kpi.hero {{border-color: var(--pulse-accent); min-height: 132px;}}
  .pulse-kpi.hero .value {{font-size: clamp(24px, 2.4vw, 34px);}}

  .tone-pos {{color: var(--pulse-pos);}}
  .tone-crit {{color: var(--pulse-crit);}}
  .tone-warn {{color: var(--pulse-warn);}}
  .tone-flat {{color: var(--pulse-muted);}}

  /* --- priority card --- */
  .pulse-prio {{
    background: var(--pulse-surface); border: 1px solid var(--pulse-line);
    border-left: 3px solid var(--pulse-neutral);
    border-radius: 10px; padding: 14px 16px; height: 100%; min-height: 196px;
  }}
  .pulse-prio.lead {{border-left-color: var(--pulse-accent);}}
  .pulse-prio .rank {{
    color: var(--pulse-muted); font-size: 11px; font-weight: 700;
    letter-spacing: .12em; text-transform: uppercase;
  }}
  .pulse-prio .scope {{
    color: var(--pulse-text); font-size: 19px; font-weight: 650; margin-top: 3px;
  }}
  .pulse-prio .pattern {{color: var(--pulse-accent); font-size: 13px; margin-top: 2px;}}
  .pulse-prio .row {{
    display: flex; justify-content: space-between; gap: 10px;
    font-size: 13px; margin-top: 7px; color: var(--pulse-muted);
    border-top: 1px solid var(--pulse-line); padding-top: 7px;
  }}
  .pulse-prio .row b {{color: var(--pulse-text); font-weight: 600;
                       font-variant-numeric: tabular-nums;}}

  /* --- standing notices, pills and callouts --- */
  .pulse-notice {{
    display: inline-flex; align-items: center; gap: 7px;
    background: var(--pulse-surface); border: 1px solid var(--pulse-line);
    border-radius: 999px; padding: 5px 13px; margin-right: 8px;
    color: var(--pulse-muted); font-size: 12px;
  }}
  .pulse-notice .dot {{
    width: 6px; height: 6px; border-radius: 50%; background: var(--pulse-warn);
  }}
  .pulse-notice .dot.syn {{background: var(--pulse-accent);}}

  .pulse-pill {{
    display: inline-flex; align-items: center; gap: 7px;
    border-radius: 999px; padding: 5px 13px; margin: 0 8px 8px 0;
    font-size: 12px; font-weight: 600; letter-spacing: .01em;
    border: 1px solid var(--pulse-line); background: var(--pulse-surface);
    color: var(--pulse-muted);
  }}
  .pulse-pill .dot {{width: 6px; height: 6px; border-radius: 50%;
                     background: currentColor;}}
  .pulse-pill.ok {{color: var(--pulse-pos); border-color: #16412F;}}
  .pulse-pill.warn {{color: var(--pulse-warn); border-color: #4A3510;}}
  .pulse-pill.crit {{color: var(--pulse-crit); border-color: #4A1B18;}}
  .pulse-pill.info {{color: var(--pulse-accent); border-color: #22345F;}}

  .pulse-callout {{
    background: var(--pulse-surface); border: 1px solid var(--pulse-line);
    border-left: 3px solid var(--pulse-accent);
    border-radius: 8px; padding: 12px 15px; color: var(--pulse-text);
    font-size: 13.5px; line-height: 1.6;
  }}
  .pulse-callout.quiet {{border-left-color: var(--pulse-neutral);}}
  .pulse-callout b {{font-weight: 650;}}

  /* --- pipeline diagram (Data Quality) --- */
  .pulse-flow {{
    display: flex; align-items: stretch; gap: 10px; flex-wrap: wrap;
    margin-top: 10px;
  }}
  .pulse-flow .stage {{
    flex: 1 1 210px; background: var(--pulse-surface);
    border: 1px solid var(--pulse-line); border-top: 3px solid var(--pulse-neutral);
    border-radius: 10px; padding: 12px 14px;
  }}
  .pulse-flow .stage.lead {{border-top-color: var(--pulse-accent);}}
  .pulse-flow .stage .name {{
    color: var(--pulse-text); font-size: 14px; font-weight: 650;
  }}
  .pulse-flow .stage .kind {{
    color: var(--pulse-muted); font-size: 10.5px; font-weight: 600;
    letter-spacing: .12em; text-transform: uppercase; margin-bottom: 2px;
  }}
  .pulse-flow .stage ul {{
    margin: 8px 0 0 0; padding-left: 16px; color: var(--pulse-muted);
    font-size: 12.5px; line-height: 1.55;
  }}
  .pulse-flow .arrow {{
    align-self: center; color: var(--pulse-accent); font-size: 18px;
  }}

  hr {{border-color: var(--pulse-line); margin: 6px 0 2px 0;}}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
