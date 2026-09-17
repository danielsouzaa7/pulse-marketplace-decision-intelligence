import dataclasses
import itertools
import json

import numpy as np
import pytest

from pulse.config import GROUND_TRUTH
from pulse.data_generator import _effects
from pulse.incidents import (CONTEXT_DEFAULTS, INCIDENTS, INCIDENT_KEYS, Effects,
                             IncidentContext, effects_for, paid_social_retention,
                             promo_margin_erosion, write_ground_truth,
                             zone4_availability, zone7_degradation)

_EFFECT_FIELDS = tuple(f.name for f in dataclasses.fields(Effects))
_CONTEXT_FIELDS = tuple(f.name for f in dataclasses.fields(IncidentContext))


def ctx(day, zone=7, hour=19, channel="organic", signup_day=0, promotion_id=None):
    return IncidentContext(day=day, zone_id=zone, hour=hour,
                           merchant_id=1, customer_id=1,
                           acquisition_channel=channel,
                           signup_day=signup_day, promotion_id=promotion_id)


def moved(effects: Effects) -> set[str]:
    """The multipliers an incident actually touches -- its effect signature."""
    return {f for f in _EFFECT_FIELDS if getattr(effects, f) != 1.0}


def test_zone7_is_inert_during_the_clean_baseline():
    e = effects_for(ctx(day=100))
    assert e == Effects.none()


def test_zone7_is_inert_in_other_zones():
    assert effects_for(ctx(day=170, zone=3)) == Effects.none()


def test_zone7_ramps_in_over_three_days_then_saturates():
    day150 = effects_for(ctx(day=150))
    day153 = effects_for(ctx(day=153))
    day175 = effects_for(ctx(day=175))
    assert 1.0 < day150.cancel_mult < day153.cancel_mult
    assert day153.cancel_mult == day175.cancel_mult          # saturated
    assert abs(day175.cancel_mult - 3.25) < 1e-9             # 8% -> 26%
    assert abs(day175.actual_delivery_mult - 1.35) < 1e-9
    assert abs(day175.promised_eta_mult - 1.12) < 1e-9


def test_zone7_never_touches_pre_checkout_conversion():
    """Zone 7 is a POST-checkout failure. This is what keeps it
    analytically distinct from Zone 4."""
    for day in (150, 160, 180):
        assert effects_for(ctx(day=day)).conversion_mult == 1.0


# --- incident #2: zone 4 -----------------------------------------------------


def test_zone4_suppresses_availability_only_at_peak_hours():
    off_peak = effects_for(ctx(day=170, zone=4, hour=15))
    at_peak = effects_for(ctx(day=170, zone=4, hour=19))
    assert off_peak.availability_mult == 1.0
    assert at_peak.availability_mult < 0.75


def test_zone4_breaks_conversion_not_completion():
    """The mirror image of Zone 7 - this is what the funnel must separate."""
    e = effects_for(ctx(day=170, zone=4, hour=19))
    assert e.conversion_mult < 1.0
    assert e.cancel_mult == 1.0
    assert e.actual_delivery_mult == 1.0


def test_zone4_is_inert_before_its_start_day_and_in_other_zones():
    assert effects_for(ctx(day=157, zone=4, hour=19)) == Effects.none()
    assert zone4_availability(ctx(day=170, zone=5, hour=19)) == Effects.none()


def test_zone4_ramps_in_over_three_days_then_saturates():
    d158 = zone4_availability(ctx(day=158, zone=4, hour=19))
    d161 = zone4_availability(ctx(day=161, zone=4, hour=19))
    d179 = zone4_availability(ctx(day=179, zone=4, hour=19))
    assert 1.0 > d158.availability_mult > d161.availability_mult
    assert d161 == d179                                       # saturated
    assert abs(d179.availability_mult - 0.70) < 1e-9
    assert abs(d179.conversion_mult - 0.78) < 1e-9


def test_zone4_and_zone7_effect_signatures_are_disjoint():
    """THE distinction the whole product rests on.

    Zone 4 fails before checkout and Zone 7 after it, so they must move
    DISJOINT sets of multipliers: if both moved both, `order_conversion` vs
    `completion_rate` would no longer separate them and the engine's central
    diagnostic claim would be untestable.
    """
    z4 = zone4_availability(ctx(day=170, zone=4, hour=19))
    z7 = zone7_degradation(ctx(day=170, zone=7, hour=19))

    # Zone 4 touches no fulfilment lever...
    assert z4.cancel_mult == 1.0
    assert z4.actual_delivery_mult == 1.0
    assert z4.promised_eta_mult == 1.0
    # ...and Zone 7 touches no demand-side lever.
    assert z7.conversion_mult == 1.0
    assert z7.availability_mult == 1.0

    assert moved(z4) and moved(z7), "an incident that moves nothing proves nothing"
    assert moved(z4).isdisjoint(moved(z7))


# --- incident #3: FREESHIP_WINTER -------------------------------------------


def test_promo_lifts_conversion_and_multiplies_the_discount():
    e = effects_for(ctx(day=160, zone=1, hour=12, promotion_id=10))
    assert abs(e.conversion_mult - 1.14) < 1e-9
    assert abs(e.discount_mult - 2.4) < 1e-9
    assert moved(e) == {"conversion_mult", "discount_mult"}


def test_promo_is_inert_without_a_promotion_and_outside_its_window():
    assert promo_margin_erosion(ctx(day=160, zone=1, hour=12)) == Effects.none()
    assert promo_margin_erosion(
        ctx(day=151, zone=1, hour=12, promotion_id=10)) == Effects.none()
    assert promo_margin_erosion(
        ctx(day=173, zone=1, hour=12, promotion_id=10)) == Effects.none()


# --- incident #4: paid-social retention decay --------------------------------


def test_retention_decay_applies_to_recent_paid_social_cohorts_only():
    hit = effects_for(ctx(day=100, channel="paid_social", signup_day=120))
    assert abs(hit.repeat_mult - 0.57) < 1e-9                 # D30 42% -> 24%
    assert moved(hit) == {"repeat_mult"}
    assert paid_social_retention(
        ctx(day=100, channel="paid_social", signup_day=119)) == Effects.none()
    assert paid_social_retention(
        ctx(day=100, channel="organic", signup_day=170)) == Effects.none()


def test_retention_decay_is_a_cohort_effect_not_a_calendar_one():
    """It keys on WHO was acquired, not on WHEN the session happened, so it is
    identical on day 0 and day 179."""
    early = paid_social_retention(ctx(day=0, channel="paid_social", signup_day=150))
    late = paid_social_retention(ctx(day=179, channel="paid_social", signup_day=150))
    assert early == late != Effects.none()


# --- combination -------------------------------------------------------------


def test_overlapping_incidents_combine_multiplicatively():
    """Zone 4 at peak, inside the FREESHIP window, for a decayed cohort: three
    incidents at once, each on its own multiplier, none overwriting another."""
    e = effects_for(ctx(day=170, zone=4, hour=19, promotion_id=10,
                        channel="paid_social", signup_day=150))
    assert abs(e.conversion_mult - 0.78 * 1.14) < 1e-9        # zone 4 x promo
    assert abs(e.availability_mult - 0.70) < 1e-9
    assert abs(e.discount_mult - 2.4) < 1e-9
    assert abs(e.repeat_mult - 0.57) < 1e-9
    assert e.cancel_mult == 1.0


def test_combine_multiplies_matching_fields_not_transposed_ones():
    a = Effects(actual_delivery_mult=2.0, promised_eta_mult=3.0, cancel_mult=5.0,
                conversion_mult=7.0, availability_mult=11.0, discount_mult=13.0,
                repeat_mult=17.0)
    b = Effects(actual_delivery_mult=1.5, promised_eta_mult=2.5, cancel_mult=3.5,
                conversion_mult=4.5, availability_mult=5.5, discount_mult=6.5,
                repeat_mult=7.5)
    c = a.combine(b)
    assert c.actual_delivery_mult == 2.0 * 1.5
    assert c.promised_eta_mult    == 3.0 * 2.5
    assert c.cancel_mult          == 5.0 * 3.5
    assert c.conversion_mult      == 7.0 * 4.5
    assert c.availability_mult    == 11.0 * 5.5
    assert c.discount_mult        == 13.0 * 6.5
    assert c.repeat_mult          == 17.0 * 7.5


def test_combine_with_none_is_identity():
    a = Effects(actual_delivery_mult=2.0, promised_eta_mult=3.0, cancel_mult=5.0,
                conversion_mult=7.0, availability_mult=11.0, discount_mult=13.0,
                repeat_mult=17.0)
    assert a.combine(Effects.none()) == a
    assert Effects.none().combine(a) == a


# --- the memoisation key -----------------------------------------------------


def test_every_incident_declares_a_memoisation_key():
    assert set(INCIDENT_KEYS) == set(INCIDENTS)
    for fn, key in INCIDENT_KEYS.items():
        assert key, fn.__name__
        assert set(key) <= set(_CONTEXT_FIELDS), fn.__name__


def test_context_defaults_are_inert():
    """data_generator._effects() holds every field a generator did not supply at
    CONTEXT_DEFAULTS. A default that is not inert would inject an effect into
    every partial context in the pipeline."""
    assert effects_for(IncidentContext(**CONTEXT_DEFAULTS)) == Effects.none()


def test_declared_keys_cover_every_field_an_incident_reads():
    """An incident must be INSENSITIVE to every context field it did not
    declare, because _effects() will hold those fields at a placeholder.

    This localises the failure: an incident whose key omits a field it reads
    fails here by name, rather than as data that quietly never appeared.
    """
    probes = {"day": 165, "zone_id": 4, "hour": 19, "merchant_id": 77,
              "customer_id": 99, "acquisition_channel": "paid_social",
              "signup_day": 150, "promotion_id": 10}
    for fn, key in INCIDENT_KEYS.items():
        base = {f: probes[f] if f in key else CONTEXT_DEFAULTS[f]
                for f in _CONTEXT_FIELDS}
        reference = fn(IncidentContext(**base))
        for field in _CONTEXT_FIELDS:
            if field in key:
                continue
            perturbed = fn(IncidentContext(**{**base, field: probes[field]}))
            assert perturbed == reference, (
                f"{fn.__name__} reads {field!r} but does not declare it in "
                f"INCIDENT_KEYS -- _effects() would evaluate it against a "
                f"placeholder and the incident would silently never fire")


def test_vectorised_effects_reproduce_the_scalar_incident_functions():
    """THE landmine guard.

    _effects() is fast because it memoises each incident on only the fields
    that incident declares. effects_for() always sees the whole context. Any
    disagreement means a declared key is incomplete and an incident is being
    evaluated against placeholder values -- which shows up as an incident
    missing from the data, with no error and no failing assertion anywhere
    else. Row-by-row equality over a grid that straddles every incident
    boundary is what turns that into a red test.
    """
    grid = list(itertools.product(
        (0, 100, 149, 150, 151, 152, 157, 158, 159, 161, 172, 173, 179),  # day
        (1, 4, 7),                                                        # zone
        (12, 17, 18, 19, 20, 21),                                         # hour
        (0, 3, 10),                                       # promotion_id (0=none)
        ("organic", "paid_social"),                                    # channel
        (0, 119, 120, 179),                                         # signup_day
    ))
    cols = np.array(grid, dtype=object).T
    fields = dict(
        day=cols[0].astype(np.int64), zone_id=cols[1].astype(np.int64),
        hour=cols[2].astype(np.int64), promotion_id=cols[3].astype(np.int64),
        acquisition_channel=cols[4].astype(str),
        signup_day=cols[5].astype(np.int64),
    )
    got = _effects(**fields)

    for i, (day, zone, hour, promo, channel, signup) in enumerate(grid):
        want = effects_for(ctx(day=day, zone=zone, hour=hour, channel=channel,
                               signup_day=signup,
                               promotion_id=promo or None))
        for f in _EFFECT_FIELDS:
            assert got[f][i] == pytest.approx(getattr(want, f)), (f, grid[i])

    # The grid has to actually exercise every incident, or the comparison above
    # is a comparison of ones.
    for f in ("cancel_mult", "availability_mult", "conversion_mult",
              "discount_mult", "repeat_mult"):
        assert (got[f] != 1.0).any(), f


# --- ground truth ------------------------------------------------------------


def test_ground_truth_is_written_outside_the_medallion_layers():
    write_ground_truth()
    path = GROUND_TRUTH / "injected_incidents.json"
    assert path.exists()
    payload = json.loads(path.read_text())
    z7 = next(i for i in payload["incidents"] if i["id"] == "zone7_degradation")
    assert z7["start_day"] == 150
    assert z7["scope"] == {"dimension": "zone", "value": 7}
    assert "bronze" not in str(path) and "silver" not in str(path) and "gold" not in str(path)


def test_ground_truth_records_every_injected_incident():
    """Task 18's recovery test compares what the engine found against this
    register, so an incident missing from it is an incident nobody checks."""
    write_ground_truth()
    payload = json.loads((GROUND_TRUTH / "injected_incidents.json").read_text())
    incidents = payload["incidents"]
    assert {i["id"] for i in incidents} == {fn.__name__ for fn in INCIDENTS}
    # zone 7 stays first: tests/test_root_cause.py and tests/test_anomaly.py
    # read incidents[0] as the flagship incident.
    assert incidents[0]["id"] == "zone7_degradation"

    breaks = {i["id"]: i["expected_funnel_break"] for i in incidents}
    assert breaks["zone7_degradation"] == "completion_rate"
    assert breaks["zone4_availability"] == "order_conversion"
    assert breaks["promo_margin_erosion"] == "contribution_margin"
    assert breaks["paid_social_retention"] == "repeat_rate"
    for i in incidents:
        assert i["scope"]["dimension"] and i["expected_drivers"]
