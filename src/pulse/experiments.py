# The Experiment Lab -- the "DID IT WORK?" half of PULSE.
#
# LANGUAGE. Every sentence this module CONSTRUCTS is Brazilian Portuguese,
# because the Experiment Lab page renders them verbatim: the randomisation
# notes, the warnings and the conclusion. What stays as it is: experiment_id,
# variant, primary_metric, every dataclass field, and the verdict TOKENS
# ("significant"/"not_significant", "positive"/"marginal"/"negative"), which
# the app keys its own labels on. Numbers are written in Brazilian notation,
# and percentage POINTS stay visibly distinct from percentages.
#
# This is the ONE module in PULSE allowed to make a causal claim, and only
# because EXP-001 carries an actual randomised design. Everywhere else in this
# codebase (root cause, drivers, promotion economics) is associational. The
# licence is not automatic: it is conditional on the randomisation surviving
# check_randomisation(), and analyse_experiment() downgrades its own language
# when those checks fail.
#
# THE UNIT OF ANALYSIS IS THE CUSTOMER, BECAUSE THE UNIT OF RANDOMISATION IS.
# A customer is assigned once and every order they place inherits that
# assignment, so orders within a customer are correlated and are not independent
# draws. A rate built as converted_orders / total_orders has no valid standard
# error: the effective sample size is the number of customers. Every denominator
# here is a distinct assigned-customer count.
#
# WHY THIS MODULE DOES NOT SUM gold_experiment_results.converted_customers.
# That column is a per-DAY distinct count (Task 17's declared grain). Summing it
# over the window yields customer-DAYS, not customers: a customer who ordered on
# six days would be counted six times, inflating the numerator while the
# denominator stayed fixed -- and the rate can then exceed 1.0. The window-level
# repeat rate is therefore recomputed as an independent COUNT(DISTINCT) over
# silver assignments x orders (window_conversions), while gold supplies the
# denominator (assigned_customers, taken ONCE per variant -- it is repeated on
# every row by construction, so summing it would multiply it by the day count)
# and the money columns, which are additive and may be summed.
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import pandas as pd
from scipy import stats

from pulse.io import read_silver
# The engine's own Brazilian number formatters. playbook.py is where the
# shared text helpers live (decision_memo and copilot import them from there
# too); it holds no analysis this module could pick up by importing it.
from pulse.playbook import _brl, _num

# SRM is a hard blocker on the whole analysis, so it is tested at a deliberately
# strict threshold: at alpha=0.05 one experiment in twenty would be thrown away
# for nothing. 0.001 is the common industry setting.
SRM_ALPHA = 0.001

# Below this |roi| the business call is "marginal" rather than a direction.
# Only ever consulted when the cost side is MEASURED; see economic_evaluation().
_ROI_MARGINAL_BAND = 0.05

# WHERE THE COST SIDE CAME FROM, as a value on the Economics object rather than
# as a caveat in prose somewhere near it.
#
# An ROI is a ratio of two quantities and it is only a MEASURED treatment ROI if
# BOTH of them are measured on the treatment. The margin side is: it is the
# contribution margin gold carries for those orders. The cost side is not, and
# cannot be on this dataset -- there is no incentive-value column anywhere in
# silver or gold, so the subsidy is taken from `incentive_cost`, which
# gold_experiment_results computes as SUM(discount_amount) over completed orders.
# That is the ordinary promotional discount every order carries, in both arms.
# Measured on EXP-001: control R$ 6.353,78 against treatment R$ 6.161,79 -- the
# CONTROL arm carries MORE of it. A quantity the control arm has more of is not a
# cost of the treatment, and a ratio built on it is not the treatment's ROI.
#
# So the cost basis is part of the contract, and a verdict is only allowed to
# read the ROI when the basis says the cost was measured on the treatment.
COST_BASIS_MEASURED = "measured_treatment_incentive"
COST_BASIS_PROXY_DISCOUNTS = "proxy_observed_discounts"

# The three states an economic evaluation can be in. "illustrative_only" is not
# a softer "measured": it means the arithmetic demonstrates the incremental
# method and the number it produces is not a measurement of anything.
ECONOMIC_STATUS_MEASURED = "measured"
ECONOMIC_STATUS_ILLUSTRATIVE = "illustrative_only"
ECONOMIC_STATUS_UNAVAILABLE = "unavailable"

# The business verdict token for "no defensible economic call can be made".
# Added to the three directional tokens rather than folding into "marginal",
# which would read as a measured result that happened to be small.
BUSINESS_VERDICT_UNAVAILABLE = "unavailable"

# The sentence a reader gets instead of an ROI. One copy, here, because the
# engine's conclusion, the Experiment Lab page and the Copilot's evidence
# bundle all have to say the same thing about the same absence.
NO_MEASURED_ROI_NOTE = (
    "O conjunto de dados atual não contém uma medida identificável do custo "
    "incremental do tratamento: não existe coluna de valor de incentivo, e o "
    "desconto observado — a única grandeza de custo disponível — é carregado "
    "pelos dois braços, com o CONTROLE carregando mais. Por isso não é possível "
    "calcular um ROI de tratamento medido de forma defensável, e nenhum é "
    "reportado."
)

PROXY_METHOD_NOTE = (
    "Exemplo metodológico usando custo proxy — não representa ROI medido."
)


class ExperimentNotFoundError(KeyError):
    """Raised when an experiment_id is absent from gold_experiment_results."""


@dataclass(frozen=True)
class TestResult:
    """A two-proportion z-test, reported so that neither effect size can be
    quoted without the other. Relative uplift alone is the classic way to dress
    up a small effect (+1pp on a 3% base is "+33%"), so absolute_diff is the
    primary field and relative_uplift is derived from it."""
    control_n: int
    control_x: int
    treatment_n: int
    treatment_x: int
    control_rate: float
    treatment_rate: float
    absolute_diff: float          # treatment_rate - control_rate, as a fraction
    relative_uplift: float        # absolute_diff / control_rate
    z_statistic: float
    p_value: float
    ci_95: tuple[float, float]    # CI on absolute_diff
    alpha: float
    significant: bool

    @property
    def absolute_diff_pp(self) -> float:
        return self.absolute_diff * 100.0


@dataclass(frozen=True)
class Economics:
    """The part most experiment write-ups get wrong.

    The incentive is paid on EVERY treatment redemption. The margin is earned
    only on the INCREMENTAL ones -- customers who would have come back anyway
    still cost the subsidy. Charging the incentive only against incremental
    orders is the error that turns a money-losing promotion into a reported win.

    That arithmetic is the point of this class and it is correct. What it cannot
    do is invent a treatment cost: see COST_BASIS_MEASURED / cost_basis.
    """
    incentive_cost: float
    incremental_orders: float
    incremental_margin: float
    # The ratio the arithmetic produces. Whether it is a MEASURED return is
    # decided by cost_basis below, not by this field: read it through
    # economic_evaluation(), which returns None for it unless the cost was
    # measured on the treatment.
    roi: float
    # Where the cost side came from. Defaults to the proxy because that is what
    # this dataset can supply; a caller with a real incentive column passes
    # COST_BASIS_MEASURED and nothing else about this module changes.
    cost_basis: str = COST_BASIS_PROXY_DISCOUNTS
    control_conversions: float = 0.0      # scaled to the treatment arm size
    treatment_conversions: float = 0.0
    incentive_brl: float = 0.0
    margin_per_order_brl: float = 0.0
    downstream_multiplier: float = 1.0
    breakeven_absolute_lift: float = 0.0  # absolute pp lift needed for roi == 0


@dataclass(frozen=True)
class RandomisationChecks:
    """Does the randomisation actually hold? Everything downstream -- the
    p-value, the CI, and above all the word 'caused' -- is conditional on it."""
    control_n: int
    treatment_n: int
    observed_treatment_share: float
    intended_treatment_share: float
    srm_p_value: float
    srm_passed: bool
    duplicate_assignments: int      # one customer assigned more than once
    cross_variant_customers: int    # one customer present in BOTH variants
    passed: bool
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class WindowCounts:
    """Distinct converting customers per variant over the experiment window,
    plus the rows the window guard threw away -- reported rather than silently
    dropped so a test can prove the guard actually bit."""
    converted: dict[str, int]
    orders_completed: dict[str, int]
    excluded_pre_assignment: int
    excluded_post_window: int


@dataclass(frozen=True)
class ExperimentReport:
    experiment_id: str
    hypothesis: str
    primary_metric: str
    start_date: date
    end_date: date
    randomisation: RandomisationChecks
    test: TestResult
    economics: Economics
    window: WindowCounts
    required_n_per_arm: int          # for the break-even effect, at 80% power
    mde_at_80_power: float           # smallest effect the realised n can resolve
    adequately_powered: bool
    statistical_verdict: str         # "significant" | "not_significant"
    # "positive" | "marginal" | "negative" | "unavailable". The last is not a
    # weaker direction: it says no defensible economic call exists, which is the
    # state this dataset is actually in.
    business_verdict: str
    causal_language_licensed: bool
    conclusion: str
    # THE TWO LAYERS, KEPT APART. The statistical result above is measured and
    # stands on its own. These two say what the economics are worth:
    #   economic_evaluation_status  "measured" | "illustrative_only" | "unavailable"
    #   measured_roi                the return, or None when there is no
    #                               identifiable treatment-specific incremental
    #                               cost to divide by -- which is None here.
    # economics.roi still carries the illustrative ratio, so the method can be
    # shown; measured_roi is the field anything resembling a decision may read.
    economic_evaluation_status: str = ECONOMIC_STATUS_UNAVAILABLE
    measured_roi: float | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def two_proportion_test(c_n: int, c_x: int, t_n: int, t_x: int,
                        alpha: float = 0.05) -> TestResult:
    """Two-sided two-proportion z-test on distinct-customer counts.

    The p-value uses the POOLED standard error (the null is p_c == p_t, so the
    variance is estimated under that null) while the confidence interval uses
    the UNPOOLED standard error (the CI is not computed under the null). Mixing
    them up is the standard textbook combination, not an inconsistency: a
    pooled-SE interval would not be a valid interval for the difference, and an
    unpooled-SE test is slightly anti-conservative.
    """
    if c_n <= 0 or t_n <= 0:
        raise ValueError("both arms need at least one assigned customer")
    if not (0 <= c_x <= c_n and 0 <= t_x <= t_n):
        raise ValueError(f"conversions outside arm size: {c_x}/{c_n}, {t_x}/{t_n}")

    p_c, p_t = c_x / c_n, t_x / t_n
    diff = p_t - p_c

    pool = (c_x + t_x) / (c_n + t_n)
    se_pooled = math.sqrt(pool * (1 - pool) * (1 / c_n + 1 / t_n))
    z = diff / se_pooled if se_pooled > 0 else 0.0
    p_value = 2 * stats.norm.sf(abs(z))

    se_unpooled = math.sqrt(p_c * (1 - p_c) / c_n + p_t * (1 - p_t) / t_n)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (diff - z_crit * se_unpooled, diff + z_crit * se_unpooled)

    return TestResult(
        control_n=c_n, control_x=c_x, treatment_n=t_n, treatment_x=t_x,
        control_rate=p_c, treatment_rate=p_t,
        absolute_diff=diff,
        relative_uplift=diff / p_c if p_c > 0 else float("nan"),
        z_statistic=z, p_value=p_value, ci_95=ci, alpha=alpha,
        significant=bool(p_value < alpha),
    )


def required_sample_size(baseline: float, mde: float,
                         alpha: float = 0.05, power: float = 0.80) -> int:
    """Customers PER ARM needed to detect an absolute lift of `mde` on a
    `baseline` proportion. Normal approximation, two-sided, equal arms."""
    if not 0 < baseline < 1:
        raise ValueError(f"baseline must be a proportion, got {baseline}")
    if mde <= 0 or not 0 < baseline + mde < 1:
        raise ValueError(f"mde {mde} takes the treated rate outside (0, 1)")

    p1, p2 = baseline, baseline + mde
    p_bar = (p1 + p2) / 2
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    n = (z_a * math.sqrt(2 * p_bar * (1 - p_bar))
         + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / mde ** 2
    return math.ceil(n)


def detectable_effect(baseline: float, n_per_arm: int,
                      alpha: float = 0.05, power: float = 0.80) -> float:
    """Inverse of required_sample_size: the smallest absolute lift the realised
    arm size can resolve. An underpowered null and a well-powered null are
    different statements, and this is the number that separates them.

    ponytail: pooled-variance normal approximation, so it round-trips with
    required_sample_size to within a few percent rather than exactly. Invert
    numerically if that ever matters.
    """
    if n_per_arm <= 0:
        raise ValueError("n_per_arm must be positive")
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return z * math.sqrt(2 * baseline * (1 - baseline) / n_per_arm)


# --------------------------------------------------------------------------
# Economics
# --------------------------------------------------------------------------

def experiment_economics(control_conversions: float,
                         treatment_conversions: float,
                         incentive_brl: float,
                         margin_per_order_brl: float,
                         downstream_multiplier: float = 1.0,
                         control_n: int | None = None,
                         treatment_n: int | None = None,
                         cost_basis: str = COST_BASIS_PROXY_DISCOUNTS) -> Economics:
    """Cost is paid on EVERY treatment redemption; margin is earned only on the
    INCREMENTAL ones.

    If the arms differ in size, pass control_n/treatment_n and the control
    conversions are rescaled to the treatment arm's population before
    differencing -- comparing raw counts across unequal arms attributes the size
    difference to the treatment.

    `incentive_brl` is the per-redemption subsidy and `cost_basis` says where it
    came from. The default is the proxy, because the default caller is this
    dataset and this dataset has no incentive column. The returned `roi` is
    arithmetic either way; economic_evaluation() is what decides whether it may
    be reported as a measured return.
    """
    if incentive_brl <= 0:
        raise ValueError("incentive_brl must be positive to form an ROI")
    if treatment_conversions < 0 or control_conversions < 0:
        raise ValueError("conversion counts cannot be negative")

    scaled_control = control_conversions
    if control_n and treatment_n:
        scaled_control = control_conversions * (treatment_n / control_n)

    incentive_cost = treatment_conversions * incentive_brl
    incremental_orders = treatment_conversions - scaled_control
    incremental_margin = (incremental_orders * margin_per_order_brl
                          * downstream_multiplier)
    roi = (incremental_margin - incentive_cost) / incentive_cost

    # Break-even lift: incremental_margin == incentive_cost, expressed as the
    # absolute lift in the converting-customer rate. Falls straight out of
    #   d * m * k == r_t * i     (per treated customer)
    margin_per_conversion = margin_per_order_brl * downstream_multiplier
    breakeven = 0.0
    if treatment_n and margin_per_conversion > 0:
        rate_t = treatment_conversions / treatment_n
        breakeven = rate_t * incentive_brl / margin_per_conversion

    return Economics(
        incentive_cost=incentive_cost,
        incremental_orders=incremental_orders,
        incremental_margin=incremental_margin,
        roi=roi,
        cost_basis=cost_basis,
        control_conversions=scaled_control,
        treatment_conversions=treatment_conversions,
        incentive_brl=incentive_brl,
        margin_per_order_brl=margin_per_order_brl,
        downstream_multiplier=downstream_multiplier,
        breakeven_absolute_lift=breakeven,
    )


def economic_evaluation(economics: Economics) -> tuple[str, float | None, str]:
    """(status, measured_roi, business_verdict) for one Economics.

    THE ONE GATE. A measured ROI requires a cost measured on the TREATMENT, and
    there is exactly one way to get one: Economics.cost_basis says so. Anything
    else returns measured_roi=None and a business verdict of "unavailable" --
    never a direction, because a direction read off a proxy is a measured-looking
    claim built on a quantity the control arm carries more of.

    This gate is why no "do not ship" can be derived from the proxy economics.
    The decision not to ship EXP-001 as designed rests on the statistical result,
    which is measured; the economics contribute nothing to it because there are
    no defensible economics to contribute.
    """
    if economics.cost_basis != COST_BASIS_MEASURED:
        return (
            ECONOMIC_STATUS_ILLUSTRATIVE,
            None,
            BUSINESS_VERDICT_UNAVAILABLE,
        )
    roi = economics.roi
    if roi < -_ROI_MARGINAL_BAND:
        verdict = "negative"
    elif roi > _ROI_MARGINAL_BAND:
        verdict = "positive"
    else:
        verdict = "marginal"
    return ECONOMIC_STATUS_MEASURED, roi, verdict


# --------------------------------------------------------------------------
# Validity guards
# --------------------------------------------------------------------------

def check_randomisation(assignments: pd.DataFrame,
                        intended_treatment_share: float = 0.5,
                        srm_alpha: float = SRM_ALPHA) -> RandomisationChecks:
    """Three ways a per-customer randomisation breaks, checked explicitly.

    1. Sample ratio mismatch -- the observed split differs from the intended one
       by more than chance. Almost always a bug in the assignment or the logging
       (a redirect that drops one arm, a filter applied to one variant), and it
       invalidates the comparison rather than merely biasing it.
    2. Duplicate assignment -- a customer assigned more than once. Their
       behaviour enters the numerator repeatedly while the denominator counts
       them once.
    3. Cross-variant contamination -- a customer present in BOTH arms. They are
       treated and control simultaneously, which pulls the two arms toward each
       other and biases any real effect toward zero.
    """
    counts = assignments.groupby("variant")["customer_id"].nunique()
    c_n = int(counts.get("control", 0))
    t_n = int(counts.get("treatment", 0))
    total = c_n + t_n

    observed_share = t_n / total if total else 0.0
    expected = [total * (1 - intended_treatment_share),
                total * intended_treatment_share]
    if total and min(expected) > 0:
        srm_p = float(stats.chisquare([c_n, t_n], f_exp=expected).pvalue)
    else:
        srm_p = float("nan")
    srm_passed = bool(srm_p == srm_p and srm_p >= srm_alpha)  # NaN fails

    per_customer = assignments.groupby("customer_id")["variant"]
    duplicates = int((assignments["customer_id"].value_counts() > 1).sum())
    cross = int((per_customer.nunique() > 1).sum())

    notes: list[str] = []
    if not srm_passed:
        notes.append(
            f"SRM (desbalanceamento na proporção da amostra): observado "
            f"{c_n}/{t_n} ({_num(observed_share * 100, 4)}% em tratamento) "
            f"contra {_num(intended_treatment_share * 100, 2)}% pretendido, "
            f"qui-quadrado p={f'{srm_p:.2e}'.replace('.', ',')} "
            f"< {_num(srm_alpha, 3)}. Os braços não são comparáveis."
        )
    if duplicates:
        notes.append(
            f"{_num(duplicates)} cliente(s) com mais de uma atribuição."
        )
    if cross:
        notes.append(
            f"{_num(cross)} cliente(s) aparecem em AMBAS as variantes."
        )

    return RandomisationChecks(
        control_n=c_n, treatment_n=t_n,
        observed_treatment_share=observed_share,
        intended_treatment_share=intended_treatment_share,
        srm_p_value=srm_p, srm_passed=srm_passed,
        duplicate_assignments=duplicates, cross_variant_customers=cross,
        passed=bool(srm_passed and not duplicates and not cross),
        notes=tuple(notes),
    )


def window_conversions(assignments: pd.DataFrame, orders: pd.DataFrame,
                       end_date) -> WindowCounts:
    """Distinct converting customers per variant over the experiment window.

    Two exclusions, both load-bearing:
      * orders BEFORE a customer's own assigned_date -- behaviour that predates
        the treatment cannot have been caused by it, and letting it in fills
        both arms with identical history and shrinks any real effect toward
        zero. (On EXP-001 this is the majority of the joined rows.)
      * orders AFTER end_date -- post-treatment leakage; the window is the
        window.
    A customer who orders six times in the window counts ONCE: repeated orders
    by the same person are not independent observations.
    """
    end = pd.Timestamp(end_date)
    joined = orders.merge(
        assignments[["customer_id", "variant", "assigned_date"]],
        on="customer_id", how="inner")
    order_date = pd.to_datetime(joined["order_date"])
    assigned = pd.to_datetime(joined["assigned_date"])

    pre = order_date < assigned
    post = order_date > end
    kept = joined[~pre & ~post & joined["is_completed"].astype(bool)]

    return WindowCounts(
        converted={v: int(d["customer_id"].nunique())
                   for v, d in kept.groupby("variant")},
        orders_completed={v: int(len(d)) for v, d in kept.groupby("variant")},
        excluded_pre_assignment=int(pre.sum()),
        excluded_post_window=int((~pre & post).sum()),
    )


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------

def _pp(x: float) -> str:
    """Percentage POINTS, and labelled as such.

    A difference between two rates is in p.p., not in %. Writing +0.0105 as
    "+1,05%" next to a relative "+3,4%" is the classic way a one-point move
    gets read as a one-percent move, and it is how a small effect gets
    oversold.
    """
    return f"{_num(x * 100, 2, plus=True)} p.p."


# The business verdict as a word in a sentence. The FIELD keeps the English
# token -- the app branches on it and so do the tests -- and this is only its
# inflection inside the conclusion.
_VERDICT_PT = {"positive": "positivo", "marginal": "marginal",
               "negative": "negativo"}


def analyse_experiment(gold, experiment_id: str,
                       assignments: pd.DataFrame | None = None,
                       orders: pd.DataFrame | None = None,
                       experiments: pd.DataFrame | None = None,
                       intended_treatment_share: float = 0.5,
                       alpha: float = 0.05) -> ExperimentReport:
    """Full read on one experiment: validity, effect, economics, verdict.

    `gold` supplies the denominator (assigned_customers, once per variant) and
    the additive money columns. The numerator is recomputed from silver because
    gold's converted_customers is a per-day count. The silver frames are
    injectable so the validity guards are testable on fixtures that deliberately
    break them.
    """
    exp_rows = gold.experiment_results
    exp_rows = exp_rows[exp_rows["experiment_id"] == experiment_id]
    if exp_rows.empty:
        available = sorted(gold.experiment_results["experiment_id"].unique())
        raise ExperimentNotFoundError(
            f"experiment '{experiment_id}' is not in gold_experiment_results; "
            f"available: {available}")

    if assignments is None:
        assignments = read_silver("experiment_assignments")
    if orders is None:
        orders = read_silver("orders")
    if experiments is None:
        experiments = read_silver("experiments")

    assignments = assignments[assignments["experiment_id"] == experiment_id]
    definition = experiments[
        experiments["experiment_id"] == experiment_id].iloc[0]
    start_date = pd.Timestamp(definition["start_date"]).date()
    end_date = pd.Timestamp(definition["end_date"]).date()

    randomisation = check_randomisation(assignments, intended_treatment_share)
    window = window_conversions(assignments, orders, end_date)

    # assigned_customers is repeated on every row of the variant -- take it ONCE.
    denom = exp_rows.groupby("variant")["assigned_customers"].first()
    c_n, t_n = int(denom["control"]), int(denom["treatment"])
    c_x = window.converted.get("control", 0)
    t_x = window.converted.get("treatment", 0)

    test = two_proportion_test(c_n=c_n, c_x=c_x, t_n=t_n, t_x=t_x, alpha=alpha)

    # The margin side is MEASURED. The money columns are additive over dates,
    # unlike converted_customers.
    money = exp_rows.groupby("variant")[
        ["contribution_margin", "incentive_cost", "orders_completed"]].sum()
    margin_per_order = (money["contribution_margin"].sum()
                        / max(money["orders_completed"].sum(), 1))
    # THE COST SIDE IS A PROXY, AND THIS IS THE LINE THAT MAKES IT ONE. There is
    # no incentive-value column in this dataset; gold's `incentive_cost` is
    # SUM(discount_amount) over completed orders, which is the ordinary
    # promotional discount both arms carry. On EXP-001 the CONTROL arm carries
    # more of it than the treatment, so this is not a treatment-specific cost.
    # The division is kept because the incremental-economics METHOD is worth
    # showing; cost_basis below is what stops it being reported as a measurement.
    incentive_brl = float(money.loc["treatment", "incentive_cost"]) / max(t_x, 1)
    # Not an LTV assumption -- the observed completed orders a converting
    # treatment customer placed inside the window.
    orders_per_converter = window.orders_completed.get("treatment", 0) / max(t_x, 1)

    economics = experiment_economics(
        control_conversions=c_x, treatment_conversions=t_x,
        incentive_brl=incentive_brl, margin_per_order_brl=float(margin_per_order),
        downstream_multiplier=orders_per_converter,
        control_n=c_n, treatment_n=t_n,
        cost_basis=COST_BASIS_PROXY_DISCOUNTS)

    economic_status, measured_roi, business_verdict = economic_evaluation(economics)

    # Powered for WHAT? The only threshold that decides anything here is the
    # lift at which the incentive pays for itself.
    required_n = required_sample_size(test.control_rate,
                                      economics.breakeven_absolute_lift, alpha)
    mde = detectable_effect(test.control_rate, min(c_n, t_n), alpha)
    adequately_powered = bool(min(c_n, t_n) >= required_n)

    statistical_verdict = "significant" if test.significant else "not_significant"

    warnings = list(randomisation.notes)
    if economic_status != ECONOMIC_STATUS_MEASURED:
        warnings.append(f"AVALIAÇÃO ECONÔMICA INDISPONÍVEL. {NO_MEASURED_ROI_NOTE}")
    if not adequately_powered:
        warnings.append(
            f"PODER ESTATÍSTICO INSUFICIENTE para o efeito de equilíbrio: "
            f"seriam necessários {_num(required_n)} clientes por braço; "
            f"{_num(min(c_n, t_n))} foram executados.")
    if window.excluded_post_window:
        warnings.append(
            f"{_num(window.excluded_post_window)} pedidos posteriores à "
            f"janela foram excluídos (proteção contra vazamento "
            f"pós-tratamento).")

    # The causal licence is CONDITIONAL, and the sentence says which way it
    # went. If SRM, the duplicate check or the cross-variant check ever fails,
    # this prose downgrades itself to an association -- the wording is not a
    # fixed claim that happens to be true on this dataset.
    causal = randomisation.passed
    if causal:
        causal_clause = (
            "A randomização se sustenta (SRM p={srm}, nenhuma atribuição "
            "duplicada, nenhum cliente em duas variantes), então esta é uma "
            "leitura causal do efeito do tratamento sobre {metric}"
        ).format(srm=_num(randomisation.srm_p_value, 3),
                 metric=definition["primary_metric"])
    else:
        causal_clause = (
            "A randomização FALHOU nas suas verificações ("
            + "; ".join(randomisation.notes)
            + "), então isto é uma associação, e não um efeito causal"
        )

    # "No effect was detected" is not "the experiment did not work", and
    # absence of evidence of an effect is not evidence of absence of one.
    # Both distinctions are carried in the sentence rather than left to the
    # reader.
    if test.significant:
        stat_clause = (
            f"p={_num(test.p_value, 4)} < {_num(alpha, 2)}, então a hipótese "
            f"nula de ausência de efeito é rejeitada")
    else:
        stat_clause = (
            f"p={_num(test.p_value, 4)} e esse intervalo cruza o zero, então "
            f"não foi detectado efeito estatisticamente significativo e a "
            f"hipótese nula de ausência de efeito não pode ser rejeitada — "
            f"isto é ausência de evidência de efeito, e não evidência de "
            f"ausência de efeito")

    # The power analysis is what separates an informative null from an
    # inconclusive one, so it is stated either way.
    # The break-even lift divides by the incentive, so it inherits the proxy's
    # status: the MDE is measured (it needs only the control rate and the arm
    # size), the threshold it is compared against is illustrative. Said out loud
    # rather than left for a reader to work out from the section below.
    breakeven_basis = (
        "" if economic_status == ECONOMIC_STATUS_MEASURED else
        " (o ganho de equilíbrio deriva do custo proxy, portanto este limiar é "
        "ilustrativo; o menor efeito detectável não é)"
    )
    power_clause = (
        f"O desenho resolvia {_pp(mde)} a 80% de poder e precisava de apenas "
        f"{_num(required_n)} por braço para enxergar o ganho de equilíbrio de "
        f"{_pp(economics.breakeven_absolute_lift)}{breakeven_basis}, portanto "
        f"este é um resultado nulo com poder estatístico adequado na pergunta "
        f"que importa, e não um resultado inconclusivo"
        if adequately_powered else
        f"Com {_num(min(c_n, t_n))} por braço o desenho só conseguia resolver "
        f"{_pp(mde)}, abaixo do ganho de equilíbrio de "
        f"{_pp(economics.breakeven_absolute_lift)}{breakeven_basis}, portanto o "
        f"resultado nulo é inconclusivo")

    # TWO LAYERS, TWO SENTENCES, AND THE RECOMMENDATION RESTS ON THE FIRST.
    #
    # The economics paragraph used to read a direction off the proxy ROI and then
    # conclude "do not ship" from it. The direction was not measurable and the
    # recommendation was therefore resting on a quantity the control arm carries
    # more of. What the economics layer can honestly say now is that it has
    # nothing to say, plus the method the arithmetic demonstrates -- and the
    # recommendation comes off the statistical result, which is measured.
    if economic_status == ECONOMIC_STATUS_MEASURED:
        economics_clause = (
            f"No plano econômico o braço é {_VERDICT_PT[business_verdict]}: o "
            f"incentivo é pago em todos os {_num(t_x)} resgates do braço de "
            f"tratamento ({_brl(economics.incentive_cost)}), enquanto apenas "
            f"{_num(economics.incremental_orders, 0)} conversões são "
            f"incrementais ({_brl(economics.incremental_margin)} de margem), "
            f"com ROI medido de {_num(measured_roi * 100, 0, plus=True)}%."
        )
    else:
        economics_clause = (
            f"No plano econômico não há veredito. {NO_MEASURED_ROI_NOTE} A "
            f"aritmética incremental segue exibida porque o MÉTODO é o que a "
            f"maioria dos relatórios erra — o incentivo é pago em todos os "
            f"{_num(t_x)} resgates do braço de tratamento, enquanto apenas "
            f"{_num(economics.incremental_orders, 0)} conversões são "
            f"incrementais — mas o número que ela produz é um "
            f"{PROXY_METHOD_NOTE[0].lower()}{PROXY_METHOD_NOTE[1:]} O resultado "
            f"estatístico acima não é proxy: aquele é medido e se sustenta."
        )

    if test.significant and economic_status == ECONOMIC_STATUS_MEASURED             and business_verdict == "positive":
        recommendation_clause = "Implantar, e seguir medindo a linha de margem."
    elif test.significant:
        recommendation_clause = (
            "Há efeito estatístico medido; a decisão de implantar depende de uma "
            "avaliação econômica que este conjunto de dados não sustenta."
        )
    else:
        recommendation_clause = (
            "Não foi detectado efeito, portanto não há evidência de efeito que "
            "sustente uma implantação deste incentivo como desenhado — e isso é "
            "ausência de evidência, e não demonstração de que nenhum efeito "
            "exista. O "
            "próximo passo que a análise sustenta é iterar na segmentação para "
            "que o subsídio alcance clientes que não teriam voltado sozinhos, e "
            "repetir o experimento REGISTRANDO o custo do incentivo, sem o qual "
            "a pergunta econômica continuará sem resposta."
        )

    conclusion = (
        f"{definition['experiment_name']} moveu "
        f"{definition['primary_metric']} de "
        f"{_num(test.control_rate * 100, 2)}% para "
        f"{_num(test.treatment_rate * 100, 2)}%: {_pp(test.absolute_diff)} "
        f"em termos absolutos, "
        f"{_num(test.relative_uplift * 100, 1, plus=True)}% em termos "
        f"relativos, IC 95% [{_pp(test.ci_95[0])}, {_pp(test.ci_95[1])}], "
        f"{stat_clause}. {causal_clause}. {power_clause}. "
        f"{economics_clause} {recommendation_clause}"
    )

    return ExperimentReport(
        experiment_id=experiment_id,
        hypothesis=str(definition["hypothesis"]),
        primary_metric=str(definition["primary_metric"]),
        start_date=start_date, end_date=end_date,
        randomisation=randomisation, test=test, economics=economics,
        window=window,
        required_n_per_arm=required_n, mde_at_80_power=mde,
        adequately_powered=adequately_powered,
        statistical_verdict=statistical_verdict,
        business_verdict=business_verdict,
        causal_language_licensed=causal,
        conclusion=conclusion,
        economic_evaluation_status=economic_status,
        measured_roi=measured_roi,
        warnings=tuple(warnings),
    )
