# All KPI formulas live here and nowhere else. compute_metrics() is a pure
# function (no I/O) that turns whichever gold tables are present into a long
# DataFrame (metric, scope, scope_value, metric_date, value); MetricFrame
# pivots that long frame on demand for .series() and aggregates it for
# .headline(). load_gold() is the one place that touches disk.
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import pandas as pd

from pulse.config import GOLD
from pulse.contracts import assert_contract
from pulse.io import read_gold
from pulse.types import AnalysisParams, Impact

# gold table name -> (attribute exposed on GoldTables, build step that produces it)
_GOLD_TABLES: dict[str, tuple[str, str]] = {
    "gold_daily_business_metrics": ("daily_business_metrics", "Task 8"),
    "gold_zone_performance": ("zone_performance", "Task 8"),
    "gold_merchant_performance": ("merchant_performance", "Task 17"),
    "gold_customer_retention": ("customer_retention", "Task 17"),
    "gold_promotion_performance": ("promotion_performance", "Task 17"),
    "gold_experiment_results": ("experiment_results", "Task 17"),
}
_ATTR_TO_NAME = {attr: (name, step) for name, (attr, step) in _GOLD_TABLES.items()}


class MissingGoldTableError(AttributeError):
    """Raised when code asks for a gold table whose build step hasn't run yet.

    Deliberately an AttributeError subclass (that's what makes it fire out of
    GoldTables.__getattr__ at all) -- but that also means hasattr(gold, attr)
    or getattr(gold, attr, default) will silently swallow it into "attribute
    absent" and lose this message. That's an accepted tradeoff of using
    __getattr__ for this, not a bug: Task 17 (and any UI code checking table
    availability) should catch MissingGoldTableError by name, not hasattr().
    """


class GoldTables:
    """Container exposing whichever gold parquet files are present as
    attributes (e.g. `gold.daily_business_metrics`). Only two of the six
    declared datasets exist before Task 17; accessing one of the other four
    raises MissingGoldTableError naming the build step that produces it,
    rather than a bare AttributeError/KeyError.
    """

    def __init__(self, tables: dict[str, pd.DataFrame]):
        self._tables = tables

    def __getattr__(self, attr: str) -> pd.DataFrame:
        # __getattr__ only fires when normal lookup (incl. self._tables) misses.
        if attr in self._tables:
            return self._tables[attr]
        if attr in _ATTR_TO_NAME:
            gold_name, step = _ATTR_TO_NAME[attr]
            raise MissingGoldTableError(
                f"gold table '{gold_name}' is not available yet — its build "
                f"step ({step}) has not run. Present tables: "
                f"{sorted(self._tables)}"
            )
        raise AttributeError(attr)


def load_gold() -> GoldTables:
    """Load whatever gold parquet files exist on disk. Tolerates a partial
    gold layer (RULING R5): only reads files that are present, never asserts
    all six GOLD_CONTRACTS datasets exist.
    """
    tables: dict[str, pd.DataFrame] = {}
    for gold_name, (attr, _step) in _GOLD_TABLES.items():
        if (GOLD / f"{gold_name}.parquet").exists():
            tables[attr] = read_gold(gold_name)
            # The build gate never sees a stale or hand-edited parquet; this is
            # the boundary where disk becomes engine input.
            assert_contract(tables[attr], gold_name)
    return GoldTables(tables)


# The metric register. RATE metrics (order_conversion, completion_rate,
# cancellation_rate, on_time_rate, availability_rate) are listed alongside
# volume metrics (gmv, orders_placed, orders_completed, sessions) on purpose:
# at company scope a zone-local incident can be diluted below the anomaly
# detector's sensitivity floor on volume metrics while still moving the rates
# strongly. A register carrying only volume metrics would leave that signal
# invisible.
#
# _melt() only picks up REGISTERED columns, so a metric gold carries but this
# tuple omits is invisible to every downstream module -- the detector, the
# diagnosis, the playbook. availability_rate was exactly that until Task 18:
# gold_zone_performance has carried it since Task 17, and zone 4's supply
# collapse (z = -6.58, -5.69%, on zone 4 alone) was sitting in the parquet
# unseen because nothing had registered the column.
#
# availability_rate is carried at ZONE grain only -- gold_daily_business_metrics
# has no such column, and _melt skips what a table does not have, so registering
# it adds a zone series and no company series. That asymmetry is deliberate and
# is argued in gold_zone_performance.sql: a per-MERCHANT availability series
# cannot be scanned at all (every merchant is shut one fixed weekday, so a whole
# weekday of its series is NULL, its day-of-week baseline for that weekday does
# not exist, and detect_on_series correctly refuses it -- measured: 0 of 150
# merchants fire, including 0 of zone 4's 13). Supply failure is a zone-level
# operational event and the zone roll-up is the grain it is detectable at.
METRIC_REGISTER: tuple[str, ...] = (
    "gmv",
    "orders_placed",
    "orders_completed",
    "sessions",
    "order_conversion",
    "completion_rate",
    "cancellation_rate",
    "aov",
    "avg_actual_delivery_minutes",
    "avg_promised_eta_minutes",
    "on_time_rate",
    "contribution_margin",
    "active_customers",
    "availability_rate",
)

# Subset reported by .headline() — company-level KPI overview. Both duration
# metrics would be redundant on one headline card, so only the actual (the
# one Task 11 correlates drivers against) is kept here; avg_promised_eta_minutes
# stays in the full register for zone/company .series() lookups.
#
# availability_rate is likewise register-only: it exists at zone grain and not
# at company grain, so it would render as a permanently blank row on the
# company KPI overview this tuple describes.
_HEADLINE_METRICS: tuple[str, ...] = (
    "gmv",
    "orders_placed",
    "orders_completed",
    "sessions",
    "order_conversion",
    "completion_rate",
    "cancellation_rate",
    "aov",
    "on_time_rate",
    "contribution_margin",
    "avg_actual_delivery_minutes",
    "active_customers",
)

_UNITS: dict[str, str] = {
    "gmv": "BRL",
    "orders_placed": "orders",
    "orders_completed": "orders",
    "sessions": "sessions",
    "order_conversion": "ratio",
    "completion_rate": "ratio",
    "cancellation_rate": "ratio",
    "aov": "BRL",
    "avg_actual_delivery_minutes": "minutes",
    "avg_promised_eta_minutes": "minutes",
    "on_time_rate": "ratio",
    "contribution_margin": "BRL",
    # Daily-distinct, not additive (RULING): summing gold's active_customers
    # column across days yields customer-days, not customers. Reporting the
    # window's daily average (never a sum) keeps the number honest and avoids
    # the I/O a true distinct-count-over-window would need inside a pure
    # compute_metrics/headline pipeline. The "/day (avg)" that used to be
    # spelled into this unit string now comes from METRIC_SEMANTICS below,
    # which says the same thing for every metric it is true of instead of for
    # this one alone.
    "active_customers": "customers",
    # available_hours / scheduled_open_hours. A ratio, so: no BRL materiality
    # gate in the detector (a currency floor cannot answer "is 5.7pp of
    # availability worth R$5,000?"), and no segment decomposition in the
    # diagnosis (zone availability rates do not sum to a company availability
    # rate). Both of those follow from this one entry rather than from a
    # branch anywhere downstream.
    "availability_rate": "ratio",
}

# THE SEMANTIC CONTRACT. _UNITS above says what a value is MEASURED IN; this
# says what a WINDOW AGGREGATE of it MEANS -- and those are different questions
# that were being answered by one register, which is how the most visible number
# in the product came to be presented as something it is not.
#
# .headline() takes a flat mean over the comparison window. For a per-day flow
# that mean is a DAILY AVERAGE and not the window's total: company GMV over the
# 14 days ending 2026-09-10 is R$ 419.689,55, while the headline figure is
# R$ 29.977,82 -- the same measurement, one fourteenth of the size, and the two
# are indistinguishable on a card labelled "GMV". The register below is what
# lets every rendering surface say which one it is showing, once, rather than
# each surface guessing.
#
# The four classes are not stylistic. They differ in what the denominator of the
# window mean is:
#   DAILY_AVERAGE       per DAY. The metric is a flow or a count that gold
#                       emits once per calendar day, so the window mean is
#                       "per day over the window" and a window TOTAL also
#                       exists and is a different number (days x the mean).
#   RATE                a ratio, already dimensionless. The window mean is the
#                       mean of the daily rates; there is no total to confuse
#                       it with, because rates do not add.
#   PER_ORDER_AVERAGE   per ORDER (aov = gmv / orders_completed). Already an
#                       average over a non-day denominator, so "/dia" on it
#                       would be simply false.
#   PER_DELIVERY_AVERAGE  per DELIVERY. Same argument as aov.
#
# active_customers is DAILY_AVERAGE with an extra property the others lack: it
# is daily-DISTINCT, so no window total exists for it at all (summing days
# yields customer-days). The daily average is the only honest window figure,
# which is why it is the one metric that was already labelled this way.
DAILY_AVERAGE = "daily_average"
RATE = "rate"
PER_ORDER_AVERAGE = "per_order_average"
PER_DELIVERY_AVERAGE = "per_delivery_average"

METRIC_SEMANTICS: dict[str, str] = {
    "gmv": DAILY_AVERAGE,
    "orders_placed": DAILY_AVERAGE,
    "orders_completed": DAILY_AVERAGE,
    "sessions": DAILY_AVERAGE,
    "contribution_margin": DAILY_AVERAGE,
    "active_customers": DAILY_AVERAGE,
    "order_conversion": RATE,
    "completion_rate": RATE,
    "cancellation_rate": RATE,
    "on_time_rate": RATE,
    "availability_rate": RATE,
    "aov": PER_ORDER_AVERAGE,
    "avg_actual_delivery_minutes": PER_DELIVERY_AVERAGE,
    "avg_promised_eta_minutes": PER_DELIVERY_AVERAGE,
}

# Every registered metric is classified, and nothing else is. A metric added to
# METRIC_REGISTER without a semantic would render through whichever default a
# formatter happened to pick, which is exactly the silent-wrong-unit failure
# this register exists to close -- so it is checked at import, not a .get()
# default. Raised rather than asserted so `python -O` cannot skip it.
if set(METRIC_SEMANTICS) != set(METRIC_REGISTER):
    raise ValueError(
        f"METRIC_SEMANTICS and METRIC_REGISTER disagree: "
        f"{set(METRIC_SEMANTICS) ^ set(METRIC_REGISTER)}"
    )
if set(_UNITS) != set(METRIC_REGISTER):
    raise ValueError(f"_UNITS and METRIC_REGISTER disagree: "
                     f"{set(_UNITS) ^ set(METRIC_REGISTER)}")


# Which direction of travel is BAD news for a registered metric. Every other
# registered metric is better higher. One register because two readers need the
# identical judgement: the app colours a delta with it, and pulse.copilot needs
# it to tell "melhorou" from "piorou" -- an inversion a correct number cannot
# reveal.
LOWER_IS_BETTER: frozenset[str] = frozenset(
    {"cancellation_rate", "avg_actual_delivery_minutes", "avg_promised_eta_minutes"}
)
if not LOWER_IS_BETTER <= set(METRIC_REGISTER):
    raise ValueError(f"unregistered metric in LOWER_IS_BETTER: "
                     f"{LOWER_IS_BETTER - set(METRIC_REGISTER)}")


def semantic_of(metric: str) -> str:
    """What a window aggregate of `metric` means. KeyError on an unregistered
    metric, deliberately: there is no safe default here.
    """
    return METRIC_SEMANTICS[metric]


# Portuguese labels for the engine's metric identifiers. These live here, beside
# _UNITS and _SCOPE_LABELS, for the same reason _SCOPE_LABELS does: more than
# one module needs the identical Portuguese word for a metric, and a second copy
# is how two surfaces come to call the same column two different things.
# app/ui_text.py merges this dict into its own (adding the gold columns the
# engine register does not carry), and pulse.copilot's claim validator reads it
# to recognise which metric a sentence is making a claim about.
_METRIC_LABELS_PT: dict[str, str] = {
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
    "aov": "Ticket médio",
    "contribution_margin": "Margem de contribuição",
    "active_customers": "Clientes ativos",
    "availability_rate": "Taxa de disponibilidade",
}
if set(_METRIC_LABELS_PT) != set(METRIC_REGISTER):
    raise ValueError(f"_METRIC_LABELS_PT and METRIC_REGISTER disagree: "
                     f"{set(_METRIC_LABELS_PT) ^ set(METRIC_REGISTER)}")


# Portuguese words for the scope DIMENSIONS the engine carries as identifiers
# ("company", "zone", "merchant" -- memo_id, gold columns and the app's scope
# selector all keep those exact tokens). This dict is the one place that
# translation is written down: pulse.decision_memo, pulse.playbook and
# pulse.copilot all render a scope through _scope_phrase() below rather than
# each spelling out its own "Zone"/"company-wide" text, which is how the memo
# ("Zone 7") and the playbook rationale ("company-wide") previously drifted
# into two different English forms of the same word. app/ui_text.py imports
# this dict directly (pulse.metrics carries no analysis logic, so it is not
# one of the modules app/ is barred from importing) rather than keeping a
# second copy that could drift from it again.
_SCOPE_LABELS: dict[str, str] = {
    "company": "Toda a empresa",
    "zone": "Zona",
    "merchant": "Parceiro",
}


def _scope_phrase(scope: str, scope_value) -> str:
    """"zone", "7" -> "Zona 7"; "company", "all" -> "Toda a empresa".

    The dimension IDENTIFIER ("zone", "company", ...) is untranslated
    everywhere it is one -- memo_id, gold columns, the app's scope selector.
    This is only the prose rendering: the word a reader sees inline in a
    sentence, in whichever language the surrounding text is in.
    """
    if scope == "company":
        return _SCOPE_LABELS["company"]
    prefix = _SCOPE_LABELS.get(scope, scope.replace("_", " ").capitalize())
    return f"{prefix} {scope_value}"


# How to NAME the customer figure on an Impact, in Brazilian Portuguese.
#
# Two different quantities share one field. At a segment scope
# impact.customers_affected is a DISTINCT COUNT of the customers who ordered
# there; at an aggregate scope prioritization nets the nested segment claims out
# of it and what remains is a RESIDUAL -- 3,896 against the 5,632 customers who
# actually ordered company-wide on this dataset. A residual is a quantity in a
# ranking calculation, not a set of people, and cannot be enumerated.
#
# The pair lives here, beside _UNITS and _SCOPE_LABELS, for the same reason they
# do: pulse.decision_memo's impact table and app/components.py's both render this
# field, app/** is barred from importing pulse.prioritization (where the netting
# happens), and a second copy of the wording is how one surface comes to call a
# residual a measured count again. Impact.customers_are_residual is the flag;
# this is the only place it is turned into words.
_CUSTOMERS_MEASURED_LABEL = "Clientes que pediram neste escopo"
_CUSTOMERS_RESIDUAL_LABEL = (
    "Clientes considerados no impacto residual (líquido dos escopos aninhados)"
)

# The same distinction inside the score breakdown, where the component is a
# normalised share rather than a count. Neutral on purpose: the company row's
# term is a residual and a zone row's is a measured count, and one table header
# covers both rows.
SCORE_CUSTOMERS_LABEL = "Clientes considerados no impacto"


def customers_label(impact: Impact) -> str:
    """The honest name for impact.customers_affected on this particular Impact."""
    return (
        _CUSTOMERS_RESIDUAL_LABEL
        if impact.customers_are_residual
        else _CUSTOMERS_MEASURED_LABEL
    )


def _melt(df: pd.DataFrame, scope: str, scope_value_col: str | None) -> pd.DataFrame:
    """Long-format one gold table: (metric, scope, scope_value, metric_date, value)."""
    id_cols = ["metric_date"] + ([scope_value_col] if scope_value_col else [])
    value_cols = [m for m in METRIC_REGISTER if m in df.columns]
    long = df[id_cols + value_cols].melt(
        id_vars=id_cols, value_vars=value_cols, var_name="metric", value_name="value"
    )
    long["scope"] = scope
    long["scope_value"] = (
        long[scope_value_col].astype(str) if scope_value_col else "all"
    )
    return long[["metric", "scope", "scope_value", "metric_date", "value"]]


@dataclass
class MetricFrame:
    """Long-format (metric, scope, scope_value, metric_date, value) frame.
    .series() pivots a single (metric, scope, scope_value) slice on demand,
    keeping the register declarative — adding a metric in Task 17/18 is a data
    change to METRIC_REGISTER, not a new code path here.
    """

    long: pd.DataFrame = field(repr=False)

    def series(self, metric: str, scope: str, scope_value: str) -> pd.Series:
        mask = (
            (self.long["metric"] == metric)
            & (self.long["scope"] == scope)
            & (self.long["scope_value"] == str(scope_value))
        )
        sub = self.long.loc[mask, ["metric_date", "value"]].sort_values("metric_date")
        if sub.empty:
            raise KeyError(
                f"no series for metric={metric!r} scope={scope!r} scope_value={scope_value!r}"
            )
        s = sub.set_index("metric_date")["value"]
        s.index = pd.DatetimeIndex(s.index)

        # Gold is calendar-spined, so this should already hold — check it
        # rather than assume it, since the anomaly detector's day-of-week
        # baseline arithmetic depends on a gap-free daily series. Raised, not
        # asserted: under `python -O` an assert is gone and a gapped series
        # would shift every day-of-week baseline without a word.
        full_range = pd.date_range(s.index.min(), s.index.max(), freq="D")
        if len(s) != len(full_range) or not (s.index == full_range).all():
            raise ValueError(
                f"gap in daily series for metric={metric!r} scope={scope!r} "
                f"scope_value={scope_value!r}: expected {len(full_range)} "
                f"calendar days, got {len(s)}"
            )
        return s

    def headline(
        self,
        params: AnalysisParams,
        scope: str = "company",
        scope_value: str = "all",
    ) -> list[dict]:
        # recent/baseline here is a flat window-mean over the comparison and
        # baseline windows -- a KPI-card summary, nothing more. For a
        # DAILY_AVERAGE metric that mean is a per-DAY figure and the window
        # TOTAL is a different number; the "semantic" key below carries which
        # one this is, because nothing about the float itself says so. Task 10's
        # anomaly detector computes a DIFFERENT, day-of-week-adjusted
        # baseline and z-score over this same .series() data for the actual
        # firing decision. A headline delta_pct and a fired anomaly's
        # deviation_pct for the same metric are therefore not the same
        # number computed twice -- day-of-week effects can make them diverge,
        # and that is expected, not a bug.
        recent_end = pd.Timestamp(params.as_of)
        recent_start = recent_end - timedelta(days=params.comparison_window_days - 1)
        baseline_end = recent_start - timedelta(days=1)
        baseline_start = baseline_end - timedelta(days=params.baseline_window_days - 1)

        # `scope` defaults to the company overview. Passing a segment scope
        # returns the same rows measured on that segment's own series, which is
        # what lets a rendering surface show a zone's funnel without
        # recomputing anything -- see engine.run_decision_cycle's segment_kpis.
        out = []
        for metric in _HEADLINE_METRICS:
            try:
                s = self.series(metric, scope, scope_value)
            except KeyError:
                # Not carried at this scope (active_customers is company-only).
                # Skipped rather than reported as zero, which would read as a
                # measurement of nothing rather than as an absent measurement.
                continue
            recent = s.loc[recent_start:recent_end].mean()
            baseline = s.loc[baseline_start:baseline_end].mean()
            if pd.isna(baseline) or baseline == 0:
                delta_pct = float("nan")
            else:
                delta_pct = (recent - baseline) / abs(baseline) * 100.0
            out.append(
                {
                    "metric": metric,
                    "scope": scope,
                    "scope_value": str(scope_value),
                    "recent": recent,
                    "baseline": baseline,
                    "delta_pct": delta_pct,
                    "unit": _UNITS[metric],
                    # What "recent" and "baseline" MEAN, not just what they are
                    # measured in. Carried on the row rather than looked up by
                    # each renderer, so the CLI artefact, the Copilot bundle and
                    # every page read the same classification off the same
                    # object -- and a KPI card cannot present a daily average as
                    # a window total.
                    "semantic": METRIC_SEMANTICS[metric],
                }
            )
        return out


def compute_metrics(gold: GoldTables, params: AnalysisParams) -> MetricFrame:
    """Pure function: build the long MetricFrame from whichever gold tables
    the register needs. No I/O here — load_gold() already did the reading.
    """
    del params  # window/sensitivity choices are applied downstream (.headline())
    frames = [
        _melt(gold.daily_business_metrics, scope="company", scope_value_col=None),
        _melt(gold.zone_performance, scope="zone", scope_value_col="zone_id"),
    ]
    long = pd.concat(frames, ignore_index=True)
    return MetricFrame(long)
