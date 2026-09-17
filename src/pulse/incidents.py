import json
from dataclasses import dataclass
from pulse.config import GROUND_TRUTH

@dataclass(frozen=True)
class Effects:
    actual_delivery_mult: float = 1.0
    promised_eta_mult: float = 1.0
    cancel_mult: float = 1.0
    conversion_mult: float = 1.0
    availability_mult: float = 1.0
    discount_mult: float = 1.0
    repeat_mult: float = 1.0

    @staticmethod
    def none() -> "Effects":
        return Effects()

    def combine(self, other: "Effects") -> "Effects":
        return Effects(*[a * b for a, b in zip(
            (self.actual_delivery_mult, self.promised_eta_mult, self.cancel_mult,
             self.conversion_mult, self.availability_mult, self.discount_mult,
             self.repeat_mult),
            (other.actual_delivery_mult, other.promised_eta_mult, other.cancel_mult,
             other.conversion_mult, other.availability_mult, other.discount_mult,
             other.repeat_mult))])

@dataclass(frozen=True)
class IncidentContext:
    day: int
    zone_id: int
    hour: int
    merchant_id: int
    customer_id: int
    acquisition_channel: str
    signup_day: int
    promotion_id: int | None

def _ramp(day: int, start: int, days: int = 3) -> float:
    return min(1.0, max(0.0, (day - start + 1) / days))

def zone7_degradation(ctx: IncidentContext) -> Effects:
    if ctx.day < 150 or ctx.zone_id != 7:
        return Effects.none()
    r = _ramp(ctx.day, 150)
    # No conversion effect: Zone 7 fails AFTER checkout.
    return Effects(
        actual_delivery_mult=1 + 0.35 * r,
        promised_eta_mult=1 + 0.12 * r,
        cancel_mult=1 + 2.25 * r,        # 8% -> 26% at saturation
    )

def zone4_availability(ctx: IncidentContext) -> Effects:
    """Zone 4: merchant supply goes dark across the dinner peak.

    The MIRROR IMAGE of zone7_degradation and the reason the funnel
    decomposition is worth computing. This one fails BEFORE checkout: sessions
    arrive as usual, but fewer of them find an open merchant, so
    order_conversion falls. Every order that IS placed is fulfilled normally.

    It therefore sets availability_mult and conversion_mult and NOTHING else --
    no cancel_mult, no delivery multipliers. The two incidents' effect
    signatures are disjoint by construction, which is what lets the engine
    report `order_conversion` here and `completion_rate` in zone 7 off an
    identical headline symptom ("zone GMV is down").
    """
    if ctx.day < 158 or ctx.zone_id != 4 or ctx.hour not in (18, 19, 20):
        return Effects.none()
    r = _ramp(ctx.day, 158)
    return Effects(availability_mult=1 - 0.30 * r, conversion_mult=1 - 0.22 * r)

def promo_margin_erosion(ctx: IncidentContext) -> Effects:
    """FREESHIP_WINTER: marketplace-funded free delivery, no basket minimum.

    Lifts conversion on the sessions it reaches and multiplies the
    marketplace-funded discount those orders carry, so orders go UP while
    contribution margin goes DOWN -- the one incident whose headline metric
    moves in the flattering direction.

    Keyed on promotion_id, which is NOT part of the (day, zone, hour) key the
    other geographic incidents use. See INCIDENT_KEYS.
    """
    if not (152 <= ctx.day <= 172) or ctx.promotion_id is None:
        return Effects.none()
    return Effects(conversion_mult=1.14, discount_mult=2.4)

def paid_social_retention(ctx: IncidentContext) -> Effects:
    """Recent paid-social cohorts come back less often: D30 repeat 42% -> 24%.

    Keyed on the CUSTOMER (acquisition_channel, signup_day), not on the
    calendar: the damage is in who was acquired, not in what happened on a
    given day, so it shows up as a cohort effect rather than a date range.
    """
    if ctx.acquisition_channel != "paid_social" or ctx.signup_day < 120:
        return Effects.none()
    return Effects(repeat_mult=0.57)      # D30 repeat 42% -> 24%

INCIDENTS = [zone7_degradation, zone4_availability,
             promo_margin_erosion, paid_social_retention]

# Which IncidentContext fields each incident actually switches on.
#
# THIS IS LOAD-BEARING, not documentation. data_generator._effects() evaluates
# every incident by calling the scalar function above -- the single source of
# truth for incident logic -- but it cannot afford one call per row, so it
# memoises on a key. One key covering every field would be
# 180 x 8 x 24 x 11 x 4 x 180 combinations and unusable; one key covering only
# (day, zone, hour) would evaluate promo_margin_erosion and
# paid_social_retention against placeholder values, so they would silently
# never fire -- no error, no failing assertion, just an incident missing from
# the data.
#
# So the key is PER INCIDENT and declared here. Each stays small
# (1440 / 20160 / ~180 / 720 combinations), every incident is still evaluated
# by calling its own scalar function, and a future incident with a new
# dependency is a row in this dict rather than a redesign.
#
# The declarations are verified, not trusted: test_incidents asserts that
# varying any field NOT in an incident's key leaves its Effects unchanged.
INCIDENT_KEYS: dict = {
    zone7_degradation: ("day", "zone_id"),
    zone4_availability: ("day", "zone_id", "hour"),
    promo_margin_erosion: ("day", "promotion_id"),
    paid_social_retention: ("acquisition_channel", "signup_day"),
}

# Placeholder context for fields a generator does not carry (merchant
# availability knows nothing about acquisition channels, say). Every value here
# MUST be inert -- effects_for(IncidentContext(**CONTEXT_DEFAULTS)) is
# Effects.none() -- or a partial context would inject an effect nobody asked
# for. test_incidents asserts exactly that.
CONTEXT_DEFAULTS: dict = {
    "day": 0, "zone_id": 1, "hour": 12, "merchant_id": 0, "customer_id": 0,
    "acquisition_channel": "organic", "signup_day": 0, "promotion_id": None,
}

def effects_for(ctx: IncidentContext) -> Effects:
    out = Effects.none()
    for fn in INCIDENTS:
        out = out.combine(fn(ctx))
    return out

GROUND_TRUTH_SPEC = [
    {"id": "zone7_degradation", "start_day": 150, "end_day": 180,
     "scope": {"dimension": "zone", "value": 7},
     "expected_funnel_break": "completion_rate",
     "expected_drivers": ["avg_actual_delivery_minutes", "cancellation_rate"]},
    {"id": "zone4_availability", "start_day": 158, "end_day": 180,
     "scope": {"dimension": "zone", "value": 4},
     "expected_funnel_break": "order_conversion",
     "expected_drivers": ["merchant_availability"]},
    {"id": "promo_margin_erosion", "start_day": 152, "end_day": 172,
     "scope": {"dimension": "promotion", "value": "FREESHIP_WINTER"},
     "expected_funnel_break": "contribution_margin",
     "expected_drivers": ["discount_rate"]},
    # start_day here is a SIGNUP-day threshold, not a calendar window: the
    # cohort is defined by when a customer was acquired, and its effect is
    # visible across the whole series rather than from a start date onwards.
    {"id": "paid_social_retention", "start_day": 120, "end_day": 180,
     "scope": {"dimension": "acquisition_channel", "value": "paid_social"},
     "expected_funnel_break": "repeat_rate",
     "expected_drivers": ["cohort_quality"]},
]

def write_ground_truth() -> None:
    GROUND_TRUTH.mkdir(parents=True, exist_ok=True)
    (GROUND_TRUTH / "injected_incidents.json").write_text(
        json.dumps({"seed": 42, "incidents": GROUND_TRUTH_SPEC}, indent=2))
