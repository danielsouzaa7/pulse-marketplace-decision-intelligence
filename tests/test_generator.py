import hashlib
import pandas as pd
import pytest
from datetime import date

from pulse.config import (AS_OF, BRONZE, END_DATE, N_DAYS, N_MERCHANTS,
                          OPERATING_HOURS, SEED, START_DATE)
from pulse.data_generator import (make_rng, generate_zones, generate_customers,
                                  generate_merchants, generate_all)

BRONZE_TABLES = {"zones", "customers", "merchants", "promotions", "experiments",
                 "experiment_assignments", "sessions", "orders", "deliveries",
                 "merchant_availability"}

PEAK_DINNER_HOURS = [18, 19, 20]


def frame_hash(df: pd.DataFrame) -> str:
    """Content hash of a DataFrame. Deliberately NOT a file-byte hash."""
    ordered = df.sort_index(axis=1).sort_values(by=sorted(df.columns)).reset_index(drop=True)
    return hashlib.sha256(
        pd.util.hash_pandas_object(ordered, index=False).values.tobytes()
    ).hexdigest()


def with_margin(orders: pd.DataFrame) -> pd.DataFrame:
    """contribution_margin per the locked definition in spec section 4. Computed
    here rather than read from a column: bronze carries the components, and
    gold owns the aggregate."""
    return orders.assign(
        contribution_margin=orders["item_amount"] * orders["commission_rate"]
        + orders["delivery_fee"] - orders["delivery_cost"] - orders["discount_amount"])


@pytest.fixture(scope="module")
def tables():
    return generate_all()


@pytest.fixture(scope="module")
def availability(tables):
    return tables["merchant_availability"].merge(
        tables["merchants"][["merchant_id", "zone_id"]], on="merchant_id")


def uptime(a: pd.DataFrame) -> float:
    """The availability KPI: available hours / scheduled-open hours."""
    return a["is_available"].sum() / a["scheduled_open"].sum()


def test_calendar_is_fixed_and_180_days():
    assert START_DATE == date(2026, 3, 15)
    assert END_DATE == date(2026, 9, 10)
    assert N_DAYS == 180
    assert AS_OF == END_DATE
    assert SEED == 42


def test_same_seed_produces_identical_content():
    a = generate_customers(make_rng())
    b = generate_customers(make_rng())
    assert frame_hash(a) == frame_hash(b)


def test_generate_all_is_reproducible_end_to_end(tables):
    """generate_all threads ONE rng through every generator in sequence, so each
    consumes draws from a shared advancing stream. Reproducibility of a single
    generator called with a fresh rng does not imply the pipeline is."""
    again = generate_all()
    assert set(again) == set(tables)
    for name, df in tables.items():
        assert frame_hash(df) == frame_hash(again[name]), name


def test_generate_all_returns_and_writes_all_ten_tables(tables):
    assert set(tables) == BRONZE_TABLES
    for name in BRONZE_TABLES:
        assert (BRONZE / f"{name}.parquet").exists(), name


def test_zone_7_is_the_largest_zone_by_demand_weight():
    zones = generate_zones(make_rng())
    assert len(zones) == 8
    assert abs(zones["demand_weight"].sum() - 1.0) < 1e-9
    top = zones.sort_values("demand_weight", ascending=False).iloc[0]
    assert int(top["zone_id"]) == 7
    assert 0.16 <= top["demand_weight"] <= 0.20


def test_customers_have_dimension_attributes_not_derived_segments():
    c = generate_customers(make_rng())
    assert len(c) == 25_000
    assert c["customer_id"].is_unique
    assert set(c["acquisition_channel"].unique()) == {
        "organic", "paid_social", "referral", "paid_search"}
    # value tier / at-risk are DERIVED in gold, never generated
    assert "segment" not in c.columns
    assert "is_at_risk" not in c.columns


def test_merchants_are_150_and_reference_valid_zones():
    rng = make_rng()
    zones = generate_zones(rng)
    m = generate_merchants(rng, zones)
    assert len(m) == 150
    assert m["merchant_id"].is_unique
    assert set(m["zone_id"]).issubset(set(zones["zone_id"]))
    assert m["commission_rate"].between(0.12, 0.28).all()


def test_row_counts_are_on_target(tables):
    assert 95_000 <= len(tables["orders"]) <= 105_000
    assert 500_000 <= len(tables["sessions"]) <= 620_000
    assert len(tables["deliveries"]) <= len(tables["orders"])


def test_orders_reference_only_converting_sessions(tables):
    s, o = tables["sessions"], tables["orders"]
    converting = set(s.loc[s["converted"], "session_id"])
    assert set(o["session_id"]).issubset(converting)
    assert len(converting) == len(o)


def test_money_columns_exist_and_delivery_cost_varies(tables):
    o = tables["orders"]
    for col in ("item_amount", "delivery_fee", "delivery_cost",
                "discount_amount", "commission_rate"):
        assert col in o.columns
    # delivery_cost MUST vary per order, else contribution margin is a
    # scalar multiple of GMV and the promo incident is undetectable.
    assert o["delivery_cost"].std() > 0.5


def test_zone7_completion_rate_collapses_in_the_incident_window(tables):
    o = tables["orders"]
    z7 = o[o["zone_id"] == 7]
    before = z7[z7["order_day"] < 150]
    after = z7[z7["order_day"] >= 153]
    comp_before = (before["status"] == "completed").mean()
    comp_after = (after["status"] == "completed").mean()
    assert comp_before > 0.90
    assert comp_after < 0.80
    assert (comp_after / comp_before) < 0.85


def test_zone7_order_placement_conversion_stays_stable(tables):
    """Correction #1: Zone 7 must NOT depress pre-checkout conversion."""
    s = tables["sessions"]
    z7 = s[s["zone_id"] == 7]
    conv_before = z7[z7["session_day"] < 150]["converted"].mean()
    conv_after = z7[z7["session_day"] >= 153]["converted"].mean()
    assert abs(conv_after - conv_before) / conv_before < 0.05


def test_zone7_delivery_time_degrades(tables):
    d = tables["deliveries"].merge(
        tables["orders"][["order_id", "zone_id", "order_day"]], on="order_id")
    z7 = d[d["zone_id"] == 7]
    before = z7[z7["order_day"] < 150]["actual_delivery_minutes"].mean()
    after = z7[z7["order_day"] >= 153]["actual_delivery_minutes"].mean()
    assert after / before > 1.25


# --------------------------------------------------------------------------
# merchant_availability: the periodic snapshot fact table
# --------------------------------------------------------------------------


def test_merchant_availability_is_hourly_within_the_operating_window(tables):
    a = tables["merchant_availability"]
    assert len(a) == N_MERCHANTS * len(OPERATING_HOURS) * N_DAYS == 378_000
    assert a["snapshot_hour"].between(10, 23).all()
    # The PK is (merchant_id, snapshot_ts): snapshot_hour alone is the hour of
    # DAY, which necessarily repeats across 180 days -- the brief's
    # (merchant_id, snapshot_hour) cannot be unique over 378,000 rows.
    assert not a.duplicated(subset=["merchant_id", "snapshot_ts"]).any()
    assert not a.duplicated(
        subset=["merchant_id", "snapshot_day", "snapshot_hour"]).any()
    assert {"is_available", "scheduled_open"}.issubset(a.columns)


def test_merchant_availability_is_emitted_regardless_of_activity(tables):
    """A periodic snapshot, not a transaction fact: every merchant has a row for
    every operating hour of every day, whether or not anything happened. That is
    the only way a closed merchant -- which produces no orders at all -- becomes
    visible to analysis."""
    a = tables["merchant_availability"]
    assert a.groupby("merchant_id").size().eq(len(OPERATING_HOURS) * N_DAYS).all()
    assert a["snapshot_day"].nunique() == N_DAYS
    assert set(a["merchant_id"]) == set(tables["merchants"]["merchant_id"])

    # Both kinds of absence are representable and distinguishable: shut by
    # schedule (not a failure) and open-but-offline (the availability KPI's
    # numerator loss).
    assert not a["scheduled_open"].all()
    assert (a["scheduled_open"] & ~a["is_available"]).any()
    assert not a.loc[~a["scheduled_open"], "is_available"].any()


def test_zone4_availability_incident_fires_in_the_snapshot_table(availability):
    """Incident #2, measured where it lands rather than at the effect function.

    Merchant downtime is an absence: if the snapshot rows were not emitted, or
    if _effects() evaluated zone4_availability against a placeholder context,
    this collapse would simply not be in the data and nothing else would
    complain.
    """
    peak = availability[availability["snapshot_hour"].isin(PEAK_DINNER_HOURS)]
    z4 = peak[peak["zone_id"] == 4]
    before = uptime(z4[z4["snapshot_day"] < 158])
    after = uptime(z4[z4["snapshot_day"] >= 161])
    assert before > 0.93
    assert after < 0.80
    assert after / before < 0.80

    # ...and nowhere else. Off-peak hours in zone 4, and dinner peak in every
    # other zone, must be untouched, or the incident is not zone-and-hour local.
    off_peak = availability[(availability["zone_id"] == 4)
                            & ~availability["snapshot_hour"].isin(PEAK_DINNER_HOURS)]
    assert abs(uptime(off_peak[off_peak["snapshot_day"] >= 161])
               / uptime(off_peak[off_peak["snapshot_day"] < 158]) - 1) < 0.02
    others = peak[peak["zone_id"] != 4]
    assert abs(uptime(others[others["snapshot_day"] >= 161])
               / uptime(others[others["snapshot_day"] < 158]) - 1) < 0.02


def test_zone4_breaks_conversion_while_completion_holds(availability, tables):
    """The Zone 7 mirror, in the generated data: supply fails BEFORE checkout,
    so conversion falls and the orders that are placed complete normally."""
    s = tables["sessions"]
    z4 = s[(s["zone_id"] == 4) & s["session_hour"].isin(PEAK_DINNER_HOURS)]
    conv_before = z4[z4["session_day"] < 158]["converted"].mean()
    conv_after = z4[z4["session_day"] >= 161]["converted"].mean()
    assert conv_after / conv_before < 0.90

    o = tables["orders"]
    z4o = o[o["zone_id"] == 4]
    comp_before = (z4o[z4o["order_day"] < 158]["status"] == "completed").mean()
    comp_after = (z4o[z4o["order_day"] >= 161]["status"] == "completed").mean()
    assert abs(comp_after - comp_before) < 0.02

    d = tables["deliveries"].merge(
        o[["order_id", "zone_id", "order_day"]], on="order_id")
    z4d = d[d["zone_id"] == 4]
    eta_before = z4d[z4d["order_day"] < 158]["actual_delivery_minutes"].mean()
    eta_after = z4d[z4d["order_day"] >= 161]["actual_delivery_minutes"].mean()
    assert abs(eta_after / eta_before - 1) < 0.03


# --------------------------------------------------------------------------
# promotions, experiments, experiment assignments
# --------------------------------------------------------------------------


def test_promotions_are_definitions_carrying_the_freeship_incident_window(tables):
    p = tables["promotions"]
    assert len(p) == 10
    assert p["promotion_id"].is_unique and p["promo_code"].is_unique
    freeship = p.loc[p["promo_code"] == "FREESHIP_WINTER"].iloc[0]
    assert (int(freeship["start_day"]), int(freeship["end_day"])) == (152, 172)
    assert freeship["funded_by"] == "marketplace"
    assert freeship["promo_type"] == "free_delivery"
    # No other campaign may overlap days 152-172: promo_margin_erosion keys on
    # "a promotion is attached", so an overlapping campaign would silently
    # inherit the incident's multipliers.
    others = p[p["promo_code"] != "FREESHIP_WINTER"]
    assert not ((others["start_day"] <= 172) & (others["end_day"] >= 152)).any()


def test_promotion_lifts_orders_while_destroying_margin(tables):
    o = with_margin(tables["orders"])
    promo = o[o["promotion_id"].notna()]
    assert len(promo) > 0
    assert promo["contribution_margin"].mean() < o["contribution_margin"].mean()


def test_promo_margin_erosion_fires_inside_the_freeship_window(tables):
    """Incident #3, measured in the data. The promotion keys on promotion_id --
    a field that is NOT part of the (day, zone, hour) key -- so an incident
    evaluated against a placeholder context would leave promoted and unpromoted
    orders economically identical and nothing else would notice."""
    o = with_margin(tables["orders"])
    window = o[o["order_day"].between(152, 172)]
    promo = window[window["promotion_id"].notna()]
    plain = window[window["promotion_id"].isna()]
    assert len(promo) > 1_000

    assert promo["discount_amount"].mean() > 2.0 * plain["discount_amount"].mean()
    assert promo["contribution_margin"].mean() < 0.80 * plain["contribution_margin"].mean()

    # The lift half of the same incident, at the top of the funnel.
    s = tables["sessions"]
    sw = s[s["session_day"].between(152, 172)]
    lift = (sw[sw["promotion_id"].notna()]["converted"].mean()
            / sw[sw["promotion_id"].isna()]["converted"].mean())
    assert lift > 1.05

    # A campaign OUTSIDE the incident window costs nothing extra: the margin
    # erosion belongs to FREESHIP_WINTER, not to carrying a promotion_id.
    early = with_margin(tables["orders"])
    early = early[early["order_day"] < 150]
    ep, en = early[early["promotion_id"].notna()], early[early["promotion_id"].isna()]
    assert len(ep) > 1_000
    assert abs(ep["contribution_margin"].mean() / en["contribution_margin"].mean() - 1) < 0.05


def test_experiments_and_assignments_are_separate_grains(tables):
    e, a = tables["experiments"], tables["experiment_assignments"]
    assert len(e) == 1
    assert e["experiment_id"].iloc[0] == "EXP-001"
    assert {"hypothesis", "primary_metric", "start_day", "end_day"}.issubset(e.columns)

    assert len(a) == 12_000
    assert not a.duplicated(subset=["experiment_id", "customer_id"]).any()
    assert set(a["variant"]) == {"control", "treatment"}
    assert set(a["customer_id"]).issubset(set(tables["customers"]["customer_id"]))
    assert set(a["experiment_id"]) == set(e["experiment_id"])
    # The definition grain is not smeared across the assignment grain.
    assert "hypothesis" not in a.columns


# --------------------------------------------------------------------------
# paid-social retention decay
# --------------------------------------------------------------------------


def test_paid_social_retention_decay_fires_in_the_order_table(tables):
    """Incident #4, measured in the data. It keys on acquisition_channel and
    signup_day -- neither of which is in the geographic memoisation key -- so an
    incident evaluated against a placeholder context would leave the affected
    cohort ordering exactly like everyone else."""
    o, c = tables["orders"], tables["customers"]
    c = c.assign(n_orders=c["customer_id"]
                 .map(o.groupby("customer_id").size()).fillna(0))

    def cohort(channel, recent):
        mask = (c["acquisition_channel"] == channel) & (
            c["signup_day"] >= 120 if recent else c["signup_day"] < 120)
        return c[mask]

    decayed, control = cohort("paid_social", True), cohort("paid_social", False)
    assert decayed["n_orders"].mean() / control["n_orders"].mean() < 0.75

    def repeat_rate(group):
        active = group[group["n_orders"] >= 1]
        return (active["n_orders"] >= 2).mean()

    assert repeat_rate(decayed) < repeat_rate(control)

    # Scoped to the CHANNEL, not to the signup date: customers acquired in the
    # same period through another channel are untouched, or the incident would
    # be a calendar effect masquerading as a cohort one.
    assert abs(cohort("organic", True)["n_orders"].mean()
               / cohort("organic", False)["n_orders"].mean() - 1) < 0.08


def test_signup_day_is_not_constrained_by_ordering_activity(tables):
    """DELIBERATE, and Task 17 must know it: signup_day is a free dimension
    attribute, never enforced against session dates. Gating sessions on it would
    ramp volume over 180 days and fire a permanent false anomaly, so
    `first_order_day >= signup_day` does NOT hold and a D30 cohort cannot be
    derived by subtracting the two.
    """
    o, c = tables["orders"], tables["customers"]
    first = o.groupby("customer_id")["order_day"].min()
    signup = c.set_index("customer_id")["signup_day"].reindex(first.index)
    assert (first < signup).any()


# --------------------------------------------------------------------------
# I2: the experiment DESCRIBED is the experiment BUILT
# --------------------------------------------------------------------------
#
# The README used to describe EXP-001 as an at-risk reactivation test ("customers
# with no order in 21+ days, excluding zones 7 and 4"), and the generator samples
# customers UNIFORMLY with no filter of any kind -- so zone 7, one of the two
# zones the exclusion claimed to remove, is in fact the LARGEST assigned group.
# The generator was not changed to make the claim true; the claims were changed
# to match the generator, and these tests are what keeps them matched.

_README = (__import__("pathlib").Path(__file__).resolve().parents[1] / "README.md")


def test_assignment_is_a_uniform_sample_with_no_eligibility_filter(tables):
    """The randomised population is EVERY customer, drawn with equal probability.

    Proved three ways rather than asserted: the sample is a strict subset of the
    whole customer table of the declared size; the assigned customers' ordering
    history spans the full range of the unassigned ones' (so no recency filter
    was applied); and every zone is represented.
    """
    from pulse.config import N_CUSTOMERS

    customers, assignments = tables["customers"], tables["experiment_assignments"]
    assert len(customers) == N_CUSTOMERS
    assert len(assignments) == 12_000
    assert set(assignments["customer_id"]).issubset(set(customers["customer_id"]))

    # No recency or activity filter: the assigned group's base_order_rate covers
    # essentially the same range as the unassigned group's. A targeted at-risk
    # cohort would be visibly skewed to the low end.
    assigned = customers[customers["customer_id"].isin(assignments["customer_id"])]
    rest = customers[~customers["customer_id"].isin(assignments["customer_id"])]
    assert assigned["base_order_rate"].mean() == pytest.approx(
        rest["base_order_rate"].mean(), rel=0.05
    )
    assert assigned["base_order_rate"].max() >= rest["base_order_rate"].max() * 0.9


def test_no_zone_is_excluded_from_the_experiment(tables):
    """All eight zones are in, and the zone carrying the flagship incident is the
    LARGEST assigned group -- the exact opposite of the exclusion the README used
    to claim.
    """
    customers, assignments = tables["customers"], tables["experiment_assignments"]
    joined = assignments.merge(
        customers[["customer_id", "home_zone_id"]], on="customer_id"
    )
    per_zone = joined.groupby("home_zone_id").size()
    assert set(per_zone.index) == set(range(1, 9)), "a zone is missing entirely"
    assert (per_zone > 0).all()
    assert per_zone.idxmax() == 7, (
        "zone 7 is no longer the largest assigned group; the README states that "
        "it is, so one of the two has to change"
    )


def test_the_split_is_a_per_customer_coin_flip_not_an_exact_half(tables):
    """Binomial around 50/50, which is what gives the sample-ratio test something
    real to check -- and why the observed treatment share is 49.74%.
    """
    counts = tables["experiment_assignments"]["variant"].value_counts()
    assert counts["control"] == 6_031
    assert counts["treatment"] == 5_969
    assert counts["control"] != counts["treatment"]


def test_no_treatment_effect_is_injected_anywhere(tables):
    """The null the Experiment Lab reports is the CORRECT answer, not a
    disappointing one: with respect to the intervention this is an A/A test.

    Asserted structurally -- generate_orders takes no assignment table, so no code
    path exists by which a customer's variant could change their behaviour -- and
    then measured on the data.
    """
    import inspect

    from pulse.data_generator import generate_orders, generate_sessions

    for function in (generate_sessions, generate_orders):
        parameters = set(inspect.signature(function).parameters)
        assert not any("assign" in p or "experiment" in p for p in parameters), (
            f"{function.__name__} can see the experiment assignment"
        )

    orders, assignments = tables["orders"], tables["experiment_assignments"]
    variant = dict(zip(assignments["customer_id"], assignments["variant"]))
    scoped = orders[orders["customer_id"].isin(variant)].copy()
    scoped["variant"] = scoped["customer_id"].map(variant)
    scoped["completed"] = scoped["status"] == "completed"
    rates = scoped.groupby("variant")["completed"].mean()
    # An injected uplift would show here before any windowing; it does not.
    assert rates["treatment"] == pytest.approx(rates["control"], abs=0.02)


def test_the_experiment_name_and_hypothesis_do_not_claim_targeting(tables):
    """The only user-facing prose in the experiment table. It described a targeted
    at-risk cohort while the assignment was uniform, so a reader comparing the two
    would have concluded the targeting had silently broken.
    """
    row = tables["experiments"].iloc[0]
    text = f"{row['experiment_name']} {row['hypothesis']}".lower()
    for banned in ("em risco", "direcionado", "at-risk", "at risk", "targeted"):
        assert banned not in text, banned
    assert "aleatória" in text, "the uniform sample is not stated"
    # The internal identifier is stable; only the prose moved.
    assert row["experiment_id"] == "EXP-001"
    assert row["primary_metric"] == "repeat_rate"


def test_the_readme_describes_the_experiment_that_exists(tables):
    """Claim integrity, in the document a reader starts from."""
    readme = _README.read_text(encoding="utf-8")
    section = readme[readme.index("EXP-001"):]
    section = section[: section.index("## ")] if "## " in section else section

    for banned in ("at-risk reactivation incentive", "21+ days",
                   "excluding zones"):
        assert banned not in readme, banned

    # And it states the design that IS built.
    assert "all 25,000 customers" in readme
    assert "uniform" in readme.lower()
    assert "Zones excluded" in readme and "**none.**" in readme
    assert "Injected treatment effect" in readme
    assert "6,031" in readme and "5,969" in readme
    counts = tables["experiment_assignments"]["variant"].value_counts()
    assert f"{counts['control']:,}" in readme
    assert f"{counts['treatment']:,}" in readme
