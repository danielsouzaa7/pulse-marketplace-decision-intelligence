# tests/test_experiments.py
#
# Statistics is where a passing test is cheapest to fake: a wrong denominator
# still yields a plausible-looking rate, and a two-proportion test still returns
# a number when handed the wrong units. So every numeric assertion here is
# against a hand-computed or scipy-reference value, and every guard test first
# proves its own premise -- it shows the bad thing is actually present in the
# fixture, and (where it matters) that the naive computation would have given a
# different, wrong answer.
import math

import pandas as pd
import pytest
from scipy import stats

from pulse.playbook import _num
from pulse.experiments import (
    _pp,
    BUSINESS_VERDICT_UNAVAILABLE,
    COST_BASIS_MEASURED,
    COST_BASIS_PROXY_DISCOUNTS,
    ECONOMIC_STATUS_ILLUSTRATIVE,
    ECONOMIC_STATUS_MEASURED,
    ExperimentNotFoundError,
    analyse_experiment,
    check_randomisation,
    detectable_effect,
    economic_evaluation,
    experiment_economics,
    required_sample_size,
    two_proportion_test,
    window_conversions,
)
from pulse.metrics import load_gold


# --------------------------------------------------------------------------
# The z-test, against references
# --------------------------------------------------------------------------

def test_two_proportion_test_matches_a_scipy_reference():
    """z^2 for a two-proportion test with pooled variance IS Pearson's chi-square
    on the 2x2 table without continuity correction. scipy computes that
    independently, so this pins the statistic, not just its type."""
    c_n, c_x, t_n, t_x = 6000, 852, 6000, 1110
    r = two_proportion_test(c_n=c_n, c_x=c_x, t_n=t_n, t_x=t_x)

    assert r.control_rate == pytest.approx(0.142)
    assert r.treatment_rate == pytest.approx(0.185)
    assert r.absolute_diff == pytest.approx(0.043)

    chi2, p_ref, _, _ = stats.chi2_contingency(
        [[c_x, c_n - c_x], [t_x, t_n - t_x]], correction=False)
    assert r.z_statistic ** 2 == pytest.approx(chi2, rel=1e-9)
    assert r.p_value == pytest.approx(p_ref, rel=1e-9)
    assert r.p_value < 0.01 and r.significant

    lo, hi = r.ci_95
    assert lo > 0 and hi > lo
    assert lo < r.absolute_diff < hi


def test_confidence_interval_matches_a_hand_computed_unpooled_interval():
    """The CI uses the UNPOOLED standard error. Computing it with the pooled SE
    instead is a real and common bug, so the test asserts the interval is the
    unpooled one AND that it differs from the pooled alternative."""
    c_n, c_x, t_n, t_x = 6000, 852, 6000, 1110
    p_c, p_t = c_x / c_n, t_x / t_n
    se_unpooled = math.sqrt(p_c * (1 - p_c) / c_n + p_t * (1 - p_t) / t_n)
    pool = (c_x + t_x) / (c_n + t_n)
    se_pooled = math.sqrt(pool * (1 - pool) * (1 / c_n + 1 / t_n))
    assert se_unpooled != pytest.approx(se_pooled, rel=1e-4)   # premise

    z = stats.norm.ppf(0.975)
    expected = (p_t - p_c - z * se_unpooled, p_t - p_c + z * se_unpooled)

    r = two_proportion_test(c_n=c_n, c_x=c_x, t_n=t_n, t_x=t_x)
    assert r.ci_95[0] == pytest.approx(expected[0], rel=1e-12)
    assert r.ci_95[1] == pytest.approx(expected[1], rel=1e-12)


def test_a_null_effect_is_reported_as_a_null_and_the_interval_covers_zero():
    """The test must be capable of NOT finding an effect."""
    r = two_proportion_test(c_n=6000, c_x=900, t_n=6000, t_x=912)
    assert not r.significant
    assert r.p_value > 0.05
    lo, hi = r.ci_95
    assert lo < 0 < hi


def test_absolute_and_relative_effects_are_both_reported_and_consistent():
    """Relative uplift alone is how a trivial effect gets dressed up: +1pp on a
    3% base reads as +33%. Both numbers must be present and must agree."""
    r = two_proportion_test(c_n=10_000, c_x=300, t_n=10_000, t_x=400)
    assert r.absolute_diff == pytest.approx(0.01)
    assert r.absolute_diff_pp == pytest.approx(1.0)
    assert r.relative_uplift == pytest.approx(1 / 3, rel=1e-9)
    assert r.relative_uplift == pytest.approx(r.absolute_diff / r.control_rate)
    # The premise of the warning: relative is 33x the absolute here.
    assert r.relative_uplift > 30 * r.absolute_diff


def test_orders_as_the_unit_of_analysis_would_change_the_answer():
    """Randomisation is per customer, so the denominator is customers. Feeding
    the same experiment at order grain (more orders than customers, correlated
    within a customer) shrinks the standard error and manufactures significance.
    This test proves the wrong unit is not merely stylistically wrong."""
    by_customer = two_proportion_test(c_n=6000, c_x=1800, t_n=6000, t_x=1890)
    # Identical rates (30.0% vs 31.5%), but 2.5 orders per customer inflates n.
    by_order = two_proportion_test(c_n=15_000, c_x=4500, t_n=15_000, t_x=4725)

    assert by_customer.control_rate == pytest.approx(by_order.control_rate)
    assert by_customer.absolute_diff == pytest.approx(by_order.absolute_diff)
    assert not by_customer.significant           # the honest answer
    assert by_order.significant                  # the fabricated one
    assert by_order.p_value < by_customer.p_value / 10


def test_test_rejects_impossible_counts():
    with pytest.raises(ValueError):
        two_proportion_test(c_n=100, c_x=101, t_n=100, t_x=10)
    with pytest.raises(ValueError):
        two_proportion_test(c_n=0, c_x=0, t_n=100, t_x=10)


# --------------------------------------------------------------------------
# Power
# --------------------------------------------------------------------------

def test_required_sample_size_is_sane():
    n = required_sample_size(baseline=0.142, mde=0.03, alpha=0.05, power=0.80)
    assert 1_500 < n < 4_000


def test_required_sample_size_matches_a_hand_computed_value():
    baseline, mde = 0.142, 0.03
    p1, p2 = baseline, baseline + mde
    p_bar = (p1 + p2) / 2
    z_a, z_b = stats.norm.ppf(0.975), stats.norm.ppf(0.80)
    expected = math.ceil(
        (z_a * math.sqrt(2 * p_bar * (1 - p_bar))
         + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / mde ** 2)
    assert required_sample_size(baseline, mde) == expected


def test_smaller_effects_and_more_power_both_cost_more_customers():
    base = required_sample_size(0.30, 0.03)
    assert required_sample_size(0.30, 0.015) > 3 * base      # ~4x for half the mde
    assert required_sample_size(0.30, 0.03, power=0.95) > base
    assert required_sample_size(0.30, 0.03, alpha=0.01) > base


def test_detectable_effect_round_trips_with_required_sample_size():
    n = required_sample_size(0.30, 0.025)
    assert detectable_effect(0.30, n) == pytest.approx(0.025, rel=0.05)


def test_required_sample_size_rejects_an_impossible_design():
    with pytest.raises(ValueError):
        required_sample_size(baseline=0.95, mde=0.10)        # rate > 1
    with pytest.raises(ValueError):
        required_sample_size(baseline=0.30, mde=0.0)


# --------------------------------------------------------------------------
# Economics
# --------------------------------------------------------------------------

def test_economics_pays_incentive_on_every_redemption_not_just_incremental():
    """The subtlety that decides the business verdict. A formula that charged the
    incentive only against the 258 incremental orders would give
    roi = (258*14 - 258*8) / (258*8) = +0.75 -- a reported win on a campaign that
    actually loses R$5,268."""
    e = experiment_economics(control_conversions=852, treatment_conversions=1110,
                             incentive_brl=8.0, margin_per_order_brl=14.0,
                             downstream_multiplier=1.0)
    assert e.incentive_cost == pytest.approx(1110 * 8.0)     # NOT 258 * 8.0
    assert e.incentive_cost == pytest.approx(8880.0)
    assert e.incremental_orders == 258
    assert e.incremental_margin == pytest.approx(258 * 14.0)
    assert e.incremental_margin == pytest.approx(3612.0)
    assert e.roi == pytest.approx((3612.0 - 8880.0) / 8880.0)
    assert e.roi == pytest.approx(-0.5932432432432432)
    assert e.roi < 0

    # The wrong formula, spelled out, so the premise cannot silently vanish.
    wrong_roi = (258 * 14.0 - 258 * 8.0) / (258 * 8.0)
    assert wrong_roi > 0 and wrong_roi != pytest.approx(e.roi)


def test_downstream_value_moves_roi_toward_breakeven():
    base = experiment_economics(852, 1110, 8.0, 14.0, downstream_multiplier=1.0)
    ltv = experiment_economics(852, 1110, 8.0, 14.0, downstream_multiplier=2.3)
    assert ltv.roi > base.roi
    assert -0.15 < ltv.roi < 0.0
    assert ltv.incentive_cost == pytest.approx(base.incentive_cost)  # cost unmoved


def test_unequal_arms_are_scaled_not_compared_raw():
    """A treatment arm 10% larger than control produces ~10% more conversions with
    no effect at all. Differencing raw counts books that as incremental."""
    e = experiment_economics(control_conversions=1000, treatment_conversions=1100,
                             incentive_brl=5.0, margin_per_order_brl=20.0,
                             control_n=10_000, treatment_n=11_000)
    assert e.control_conversions == pytest.approx(1100.0)   # 1000 * 11000/10000
    assert e.incremental_orders == pytest.approx(0.0)
    assert e.incremental_margin == pytest.approx(0.0)
    assert e.roi == pytest.approx(-1.0)                     # pure subsidy burn
    # Premise: the unscaled comparison would have claimed +100 incremental.
    naive = experiment_economics(1000, 1100, 5.0, 20.0)
    assert naive.incremental_orders == pytest.approx(100.0)
    assert naive.roi > e.roi


def test_breakeven_lift_is_the_lift_that_actually_zeroes_roi():
    """Self-consistency: feed the reported break-even lift back in as the real
    effect and the ROI must land on zero."""
    t_n = c_n = 10_000
    t_x, incentive, margin = 3000, 3.0, 10.0
    e = experiment_economics(control_conversions=2900, treatment_conversions=t_x,
                             incentive_brl=incentive, margin_per_order_brl=margin,
                             control_n=c_n, treatment_n=t_n)
    lift = e.breakeven_absolute_lift
    assert lift == pytest.approx(0.30 * 3.0 / 10.0)          # 0.09 -> 9.00pp

    control_at_breakeven = (t_x / t_n - lift) * c_n
    at_be = experiment_economics(control_conversions=control_at_breakeven,
                                 treatment_conversions=t_x,
                                 incentive_brl=incentive,
                                 margin_per_order_brl=margin,
                                 control_n=c_n, treatment_n=t_n)
    assert at_be.roi == pytest.approx(0.0, abs=1e-12)


def test_economics_rejects_a_zero_incentive():
    with pytest.raises(ValueError):
        experiment_economics(100, 120, incentive_brl=0.0, margin_per_order_brl=10.0)


# --------------------------------------------------------------------------
# Randomisation guards -- each fixture contains the defect it claims to catch
# --------------------------------------------------------------------------

def _assignments(rows):
    return pd.DataFrame(rows, columns=["customer_id", "variant"]).assign(
        experiment_id="EXP-T", assigned_date=pd.Timestamp("2026-08-17"))


def _balanced(n_control, n_treatment):
    return _assignments(
        [(i, "control") for i in range(n_control)]
        + [(1_000_000 + i, "treatment") for i in range(n_treatment)])


def test_a_balanced_split_passes_every_check():
    """Proves the guard is not always-fail, which is what makes the next three
    tests mean something."""
    checks = check_randomisation(_balanced(6031, 5969))
    assert checks.srm_passed and checks.passed
    assert checks.srm_p_value > 0.5
    assert checks.duplicate_assignments == 0
    assert checks.cross_variant_customers == 0
    assert checks.observed_treatment_share == pytest.approx(5969 / 12000)
    assert checks.notes == ()


def test_sample_ratio_mismatch_is_caught():
    """5,400 / 6,600 on an intended 50/50. Under the null that split has
    probability ~1e-32; a broken arm, not bad luck."""
    checks = check_randomisation(_balanced(5400, 6600))
    assert checks.observed_treatment_share == pytest.approx(0.55)
    assert checks.srm_p_value < 1e-20                        # premise, quantified
    assert not checks.srm_passed
    assert not checks.passed
    note = next(n for n in checks.notes if "SRM" in n)
    assert "desbalanceamento na proporção da amostra" in note.lower()
    assert "não são comparáveis" in note


def test_a_split_that_is_off_by_chance_alone_is_not_flagged():
    """The SRM check must not fire on ordinary sampling noise, or it is useless.
    6,060/5,940 is a 0.5pp imbalance -- visible, and entirely expected."""
    checks = check_randomisation(_balanced(6060, 5940))
    assert checks.observed_treatment_share != pytest.approx(0.5)   # premise
    assert 0.001 < checks.srm_p_value
    assert checks.srm_passed and checks.passed


def test_a_customer_in_both_variants_is_caught():
    df = _assignments([(1, "control"), (2, "control"), (3, "treatment"),
                       (2, "treatment")])            # customer 2 is in both
    checks = check_randomisation(df)
    both = set(df[df.variant == "control"].customer_id) & \
        set(df[df.variant == "treatment"].customer_id)
    assert both == {2}                                # premise
    assert checks.cross_variant_customers == 1
    assert not checks.passed
    assert any("AMBAS as variantes" in n for n in checks.notes)


def test_duplicate_assignment_within_one_variant_is_caught():
    """A customer assigned twice to the SAME arm passes the cross-variant check
    but is still one person counted as two observations."""
    df = _assignments([(1, "control"), (1, "control"), (2, "treatment")])
    checks = check_randomisation(df)
    assert checks.duplicate_assignments == 1
    assert checks.cross_variant_customers == 0        # the other check is silent
    assert not checks.passed


# --------------------------------------------------------------------------
# Window discipline -- pre-assignment, post-window, repeated orders
# --------------------------------------------------------------------------

_WINDOW_ORDERS = pd.DataFrame([
    # customer, order_date, is_completed
    (1, "2026-08-10", True),    # PRE-assignment: excluded
    (1, "2026-08-20", True),    # counts -> control converts
    (2, "2026-08-18", True),    # counts
    (2, "2026-08-19", True),    # same customer again -> still ONE conversion
    (2, "2026-08-20", False),   # not completed
    (3, "2026-09-15", True),    # POST-window: excluded
    (3, "2026-08-25", False),   # cancelled: not a conversion
], columns=["customer_id", "order_date", "is_completed"])

_WINDOW_ASSIGN = _assignments([(1, "control"), (2, "treatment"), (3, "treatment")])
_WINDOW_END = "2026-09-10"


def test_window_guard_drops_pre_assignment_and_post_window_orders():
    w = window_conversions(_WINDOW_ASSIGN, _WINDOW_ORDERS, _WINDOW_END)
    assert w.excluded_pre_assignment == 1      # premise: the guard actually bit
    assert w.excluded_post_window == 1
    assert w.converted == {"control": 1, "treatment": 1}
    # Without the guards the naive answer is control 1 (customer 1 twice) and
    # treatment 2 (customer 3 converts on a post-window order).
    assert w.converted["treatment"] != 2


def test_repeated_orders_by_one_customer_count_once():
    w = window_conversions(_WINDOW_ASSIGN, _WINDOW_ORDERS, _WINDOW_END)
    assert w.orders_completed["treatment"] == 2    # premise: two completed orders
    assert w.converted["treatment"] == 1           # but one customer


# --------------------------------------------------------------------------
# The real experiment, end to end
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def report():
    return analyse_experiment(load_gold(), "EXP-001")


def test_report_states_both_verdicts(report):
    """EXP-001 is a NULL result, and its economics are UNAVAILABLE rather than
    negative. Nothing here is tuned to make either come out otherwise: the effect
    is +1.05pp with a CI that covers zero, and the cost side of the economics is
    a discount the control arm carries more of.
    """
    assert report.statistical_verdict == "not_significant"
    assert report.business_verdict == BUSINESS_VERDICT_UNAVAILABLE
    assert report.measured_roi is None
    assert report.test.p_value > 0.05
    assert report.test.ci_95[0] < 0 < report.test.ci_95[1]

    text = report.conclusion.lower()
    # The null as a finding, never as a failure, and the two halves of the
    # distinction that makes it one.
    assert "não foi detectado efeito estatisticamente significativo" in text
    assert "não pode ser rejeitada" in text
    assert "ausência de evidência de efeito, e não evidência de ausência" in text
    for banned in ("o experimento não funcionou", "não teve efeito",
                   "prova que", "comprova"):
        assert banned not in text, banned
    assert "iterar na segmentação" in text
    # Percentage POINTS and percent, in their own units and never merged.
    assert _pp(report.test.absolute_diff) in report.conclusion
    assert "p.p." in report.conclusion
    assert f"{_num(report.test.relative_uplift * 100, 1, plus=True)}%" in (
        report.conclusion)
    # No ROI figure at all in the prose: a percentage beside the word ROI is read
    # as a measured return however it is hedged.
    assert f"{_num(report.economics.roi * 100, 0, plus=True)}%" not in (
        report.conclusion)
    assert "exemplo metodológico usando custo proxy" in text
    assert "o resultado estatístico acima não é proxy" in text


def test_the_denominator_is_assigned_customers_taken_once_per_variant(report):
    """assigned_customers is repeated on all 25 dates of each variant. Summing it
    would give 150,775 control 'customers' -- a denominator 25x too large and a
    conversion rate 25x too small."""
    gold = load_gold().experiment_results
    rows = gold[gold.experiment_id == "EXP-001"]
    control = rows[rows.variant == "control"]
    assert control["assigned_customers"].nunique() == 1          # premise
    assert len(control) == 25                                    # premise
    assert report.test.control_n == int(control["assigned_customers"].iloc[0])
    assert report.test.control_n != int(control["assigned_customers"].sum())


def test_the_numerator_is_not_the_sum_of_a_per_day_distinct_count(report):
    """converted_customers is per DAY. Summing it gives customer-days: 2,882
    treatment 'converters' against 5,969 assigned -- a 48% repeat rate built from
    a numerator that double-counts anyone who ordered on more than one day. The
    real distinct count is 1,888."""
    gold = load_gold().experiment_results
    rows = gold[(gold.experiment_id == "EXP-001") & (gold.variant == "treatment")]
    customer_days = int(rows["converted_customers"].sum())
    assert customer_days > report.test.treatment_x               # premise
    assert report.test.treatment_x == report.window.converted["treatment"]
    assert report.test.treatment_rate < customer_days / report.test.treatment_n


def test_pre_assignment_history_is_excluded_from_the_real_experiment(report):
    """The experiment starts on day 155 of a 180-day history, so most of the
    joined order rows predate assignment. Including them would dilute both arms
    with identical pre-treatment behaviour."""
    assert report.window.excluded_pre_assignment > 10_000
    assert report.window.excluded_post_window == 0


def test_randomisation_holds_so_causal_language_is_licensed(report):
    assert report.randomisation.passed
    assert report.randomisation.srm_p_value > 0.001
    assert report.randomisation.duplicate_assignments == 0
    assert report.randomisation.cross_variant_customers == 0
    assert report.causal_language_licensed
    assert "leitura causal do efeito do tratamento" in report.conclusion
    assert "A randomização se sustenta" in report.conclusion


def test_broken_randomisation_downgrades_the_causal_claim():
    """Same gold, same orders, but 300 control customers also appear in treatment.
    The report must stop saying 'caused'."""
    from pulse.io import read_silver

    assignments = read_silver("experiment_assignments")
    contaminated = assignments[assignments.variant == "control"].head(300).copy()
    contaminated["variant"] = "treatment"
    broken = pd.concat([assignments, contaminated], ignore_index=True)

    report = analyse_experiment(load_gold(), "EXP-001", assignments=broken)
    assert report.randomisation.cross_variant_customers == 300
    assert not report.randomisation.passed
    assert not report.causal_language_licensed
    assert "leitura causal" not in report.conclusion
    assert "uma associação, e não um efeito causal" in report.conclusion
    assert any("AMBAS as variantes" in w for w in report.warnings)


def test_the_experiment_was_powered_for_the_effect_that_would_pay_for_itself(report):
    """An underpowered null and a well-powered null are different statements.
    This one is well powered for the break-even lift and explicitly is not
    powered to resolve the tiny effect it happened to observe."""
    assert report.adequately_powered
    assert report.required_n_per_arm < min(report.test.control_n,
                                           report.test.treatment_n)
    # The break-even lift is far larger than anything observed.
    assert report.economics.breakeven_absolute_lift > 8 * report.test.absolute_diff
    # ...and the observed effect itself is below the design's resolution.
    assert report.test.absolute_diff < report.mde_at_80_power


def test_unknown_experiment_id_names_what_is_available():
    with pytest.raises(ExperimentNotFoundError) as exc:
        analyse_experiment(load_gold(), "EXP-999")
    assert "EXP-001" in str(exc.value)


# --------------------------------------------------------------------------
# I3: the statistical layer and the economic layer, kept apart
# --------------------------------------------------------------------------
#
# The statistics here are sound and are unchanged. The economics were not: the
# "incentive cost" was inferred from discount_amount, which both arms carry and
# the CONTROL arm carries MORE of, on what is structurally an A/A test with
# respect to the intervention. A direction read off that ratio was driving a
# "do not ship" verdict. These tests pin the separation.


def test_the_control_arm_carries_more_discount_than_the_treatment_arm():
    """The premise, measured, before anything is asserted about it.

    If this ever reverses, the proxy is still not a treatment cost -- but the
    sharpest single piece of evidence that it is not would be gone, and a reader
    should be told rather than left with a stale argument in a docstring.
    """
    rows = load_gold().experiment_results
    rows = rows[rows.experiment_id == "EXP-001"]
    totals = rows.groupby("variant")["incentive_cost"].sum()
    assert totals["control"] > totals["treatment"]
    assert totals["control"] == pytest.approx(6353.78, abs=0.01)
    assert totals["treatment"] == pytest.approx(6161.79, abs=0.01)


def test_control_discount_can_never_become_a_treatment_cost():
    """A measured ROI requires cost_basis == COST_BASIS_MEASURED, and nothing in
    analyse_experiment can reach that value: the cost it has is a proxy.

    Asserted as a property of the gate rather than of today's numbers -- hand
    economic_evaluation() a proxy-based Economics whose ROI is strongly positive
    and it still refuses to report a measured return.
    """
    flattering = experiment_economics(
        control_conversions=100, treatment_conversions=1000,
        incentive_brl=0.01, margin_per_order_brl=50.0,
        control_n=1000, treatment_n=1000,
    )
    assert flattering.roi > 1.0                      # the proxy looks wonderful
    assert flattering.cost_basis == COST_BASIS_PROXY_DISCOUNTS
    status, measured_roi, verdict = economic_evaluation(flattering)
    assert status == ECONOMIC_STATUS_ILLUSTRATIVE
    assert measured_roi is None
    assert verdict == BUSINESS_VERDICT_UNAVAILABLE

    # And the gate does open for a genuinely measured cost, so it is a contract
    # and not a blanket refusal.
    measured = experiment_economics(
        control_conversions=100, treatment_conversions=1000,
        incentive_brl=0.01, margin_per_order_brl=50.0,
        control_n=1000, treatment_n=1000,
        cost_basis=COST_BASIS_MEASURED,
    )
    status, measured_roi, verdict = economic_evaluation(measured)
    assert status == ECONOMIC_STATUS_MEASURED
    assert measured_roi == pytest.approx(measured.roi)
    assert verdict == "positive"


def test_no_measured_roi_is_emitted_without_a_measured_treatment_cost(report):
    assert report.economics.cost_basis == COST_BASIS_PROXY_DISCOUNTS
    assert report.economic_evaluation_status == ECONOMIC_STATUS_ILLUSTRATIVE
    assert report.measured_roi is None
    assert report.business_verdict == BUSINESS_VERDICT_UNAVAILABLE
    # The illustrative ratio is still computed, because the incremental METHOD is
    # the part worth showing. It is just not a measured return.
    assert report.economics.roi < -0.5


def test_no_ship_verdict_is_derived_from_the_proxy_economics(report):
    """The recommendation has to come off the statistical result.

    The conclusion may say there is no evidence of an effect to support shipping.
    It may not say the economics are negative, because they are not measurable.
    """
    text = report.conclusion.lower()
    assert "não há veredito" in text
    assert "não é possível calcular um ROI de tratamento medido".lower() in text
    assert "não há evidência de efeito que sustente uma implantação" in text
    for banned in ("o braço é negativo", "economicamente negativo",
                   "roi medido de", "não implantar como desenhado"):
        assert banned not in text, banned
    # And the absence is in the warnings, not only buried in the prose.
    assert any("AVALIAÇÃO ECONÔMICA INDISPONÍVEL" in w for w in report.warnings)


def test_the_statistical_result_is_unchanged_by_the_economics_fix(report):
    """Every statistical field at the values the pre-fix engine produced. The
    economics contract changed; no statistic moved.
    """
    assert report.statistical_verdict == "not_significant"
    assert report.test.control_n == 6031
    assert report.test.treatment_n == 5969
    assert report.test.control_x == 1844
    assert report.test.treatment_x == 1888
    assert report.test.control_rate == pytest.approx(0.3058, abs=5e-5)
    assert report.test.treatment_rate == pytest.approx(0.3163, abs=5e-5)
    assert report.test.absolute_diff == pytest.approx(0.010547281553821453, rel=1e-9)
    assert report.test.relative_uplift == pytest.approx(0.034496017, rel=1e-7)
    assert report.test.p_value == pytest.approx(0.2120, abs=5e-5)
    assert report.test.ci_95[0] == pytest.approx(-0.0060174, rel=1e-4)
    assert report.test.ci_95[1] == pytest.approx(0.0271120, rel=1e-4)
    assert report.mde_at_80_power == pytest.approx(0.023627112871609974, rel=1e-9)
    assert report.required_n_per_arm == 404
    assert report.adequately_powered is True
    assert report.randomisation.passed is True
    assert report.causal_language_licensed is True
    assert report.randomisation.srm_p_value == pytest.approx(0.5714, abs=5e-5)


def test_the_null_keeps_its_nuance(report):
    """Absence of evidence of an effect is not evidence of absence of one, and the
    conclusion carries both halves rather than leaving them to the reader.
    """
    text = report.conclusion.lower()
    assert "não foi detectado efeito estatisticamente significativo" in text
    assert "não pode ser rejeitada" in text
    assert "ausência de evidência de efeito, e não evidência de ausência" in text
    assert "ausência de evidência, e não demonstração de que nenhum efeito" in text
    for banned in ("o experimento não funcionou", "não teve efeito",
                   "prova que", "comprova"):
        assert banned not in text, banned


def test_the_breakeven_threshold_declares_that_it_rests_on_the_proxy(report):
    """The MDE needs only the control rate and the arm size, so it is measured.
    The break-even lift divides by the incentive, so it inherits the proxy's
    status -- and the sentence says so instead of letting a measured-looking
    "powered for the decision" claim stand unqualified.
    """
    assert "ganho de equilíbrio deriva do custo proxy" in report.conclusion
    assert "menor efeito detectável não é" in report.conclusion
