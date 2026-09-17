"""Bronze fact generation: sessions -> orders -> deliveries.

Generation ORDER is structural, not stylistic. Orders are drawn only from
converting sessions, and deliveries only from orders that entered dispatch, so
the funnel identity

    GMV = sessions x order_conversion x completion_rate x aov

holds by construction rather than by assertion.

All randomness comes from the single seeded ``Generator(PCG64(SEED))`` threaded
through every generator. The calendar is fixed and absolute: day indices 0..179
map to START_DATE .. END_DATE, never to today().
"""

import numpy as np, pandas as pd
from numpy.random import Generator, PCG64

from pulse.config import (SEED, N_CUSTOMERS, N_MERCHANTS, START_DATE, N_DAYS,
                          TARGET_ORDERS, TARGET_CONVERSION, OPERATING_HOURS,
                          PEAK_HOURS)
from pulse.incidents import (CONTEXT_DEFAULTS, INCIDENTS, INCIDENT_KEYS,
                             IncidentContext, write_ground_truth)
from pulse.io import write_bronze
from pulse.quality import inject_defects


def make_rng() -> Generator:
    return Generator(PCG64(SEED))

# Zone 7 is deliberately the largest zone: a zone-local incident must be able
# to surface as a company-level anomaly. See spec section 5.
ZONE_WEIGHTS = np.array([0.14, 0.12, 0.11, 0.13, 0.10, 0.09, 0.18, 0.13])

# --- calibration constants -------------------------------------------------
# Sessions are sized from the config funnel target so the identity above is
# satisfied at the top of the funnel too: 100k orders / 17.9% conversion.
N_SESSIONS = int(round(TARGET_ORDERS / TARGET_CONVERSION))   # 558_659
BASE_CANCEL_RATE = 0.08          # x cancel_mult (3.25 saturated) -> 26% in Zone 7
PRE_DISPATCH_CANCEL_SHARE = 0.45  # share of cancels that never reach a courier
COURIER_COST_FIXED_BRL = 2.60     # dispatch overhead per served order
COURIER_COST_PER_MIN_BRL = 0.35   # courier time is the variable cost driver
POST_DISPATCH_CANCEL_COST_SHARE = 0.55  # courier went out, order never landed
SERVICE_TIME_FACTOR = 0.80        # courier minutes vs zone base ETA (baseline)
SERVICE_TIME_SIGMA = 0.22
DISCOUNT_INCIDENCE = 0.20
BASE_MERCHANT_UPTIME = 0.97       # x availability_mult during scheduled-open hours

# Share of sessions a running campaign reaches. NOT in the spec: it is the free
# calibration parameter of incident #3, set the way spec section 5 says to set
# one -- "final multipliers are set empirically against these assertions rather
# than derived on paper". It governs how much of FREESHIP_WINTER's +14%
# conversion lift reaches the company line and therefore how much it offsets the
# two concurrent zone incidents sharing its window. Swept over 0.10-0.42; 0.15
# is where the spec's calibration targets hold (company completion rate <= -3%,
# zone 7 rank-1 at >= 55% contribution breaking at completion_rate, clean
# baseline silent, zone 7 order_conversion stable).
PROMO_EXPOSURE_SHARE = 0.15

# Hours 10..23; lunch (11-13) and dinner (18-20) peaks per config.PEAK_HOURS.
_HOURS = np.array(OPERATING_HOURS)
_HOUR_WEIGHTS = np.array([0.6, 2.0, 2.6, 2.2, 1.0, 0.8, 0.8,
                          1.1, 2.4, 3.0, 2.4, 1.5, 0.9, 0.5])
_HOUR_P = _HOUR_WEIGHTS / _HOUR_WEIGHTS.sum()

# Distance bands: fee, service-time and cost all move together with distance.
_BAND_P = np.array([0.45, 0.38, 0.17])
_BAND_FEE = np.array([6.90, 9.90, 13.90])
_BAND_DISTANCE = np.array([0.85, 1.00, 1.25])

_EFFECT_FIELDS = ("actual_delivery_mult", "promised_eta_mult", "cancel_mult",
                  "conversion_mult", "availability_mult", "discount_mult",
                  "repeat_mult")

_EPOCH = np.datetime64(START_DATE, "s")

# Promotion calendar. FREESHIP_WINTER (id 10) is incident #3 of spec section 5;
# the other nine are ordinary campaigns spread over the clean window so the
# promotion dimension has history to compare the incident against. They
# deliberately do NOT overlap days 152-172: promo_margin_erosion keys on "a
# promotion is attached", so a second campaign inside that window would inherit
# the incident's multipliers.
_PROMOTION_CALENDAR = [
    (i + 1, f"CAMPAIGN_{i + 1:02d}",
     "percentage_off" if i % 2 else "free_delivery",
     "merchant" if i % 2 else "marketplace",
     5 + 16 * i, 18 + 16 * i)
    for i in range(9)
] + [(10, "FREESHIP_WINTER", "free_delivery", "marketplace", 152, 172)]

# The single experiment of spec section 5's calendar.
#
# A DEFINITION, NOT AN EFFECT, and the name and hypothesis below say so now.
# Nothing in INCIDENTS keys on the experiment and generate_orders applies no
# treatment uplift, so assignment changes no measure: with respect to the
# intervention this is structurally an A/A test, and the null the Experiment Lab
# reports is the correct answer rather than a disappointing one.
#
# THE NAME AND HYPOTHESIS USED TO DESCRIBE A DESIGN THAT DOES NOT EXIST. They
# said "clientes em risco" and "incentivo direcionado", and
# generate_experiment_assignments below samples customers UNIFORMLY: no
# recency filter, no zone exclusion, every customer equally eligible. A reader
# comparing the two would have concluded the targeting had silently broken. The
# prose is what changed; the assignment mechanism is untouched, and so is the
# generated data's analytical signature.
#
# experiment_id, primary_metric and the day offsets are keys and stay as they
# are. A constant string draws no random numbers, so the only bytes that move
# when this prose changes are the two cells holding it.
_EXPERIMENT = {
    "experiment_id": "EXP-001",
    "experiment_name": "Incentivo de reativação — amostra aleatória da base",
    "hypothesis": "Um incentivo de reativação oferecido a uma amostra aleatória "
                  "da base de clientes aumenta a taxa de recompra em 30 dias sem "
                  "corroer a margem de contribuição.",
    "primary_metric": "repeat_rate",
    "start_day": 155,
    "end_day": 179,
}
N_EXPERIMENT_ASSIGNMENTS = 12_000


def generate_zones(rng) -> pd.DataFrame:
    return pd.DataFrame({
        "zone_id": np.arange(1, 9),
        "zone_name": [f"Zone {i}" for i in range(1, 9)],
        "demand_weight": ZONE_WEIGHTS,
        "base_eta_minutes": rng.integers(28, 42, size=8),
        "courier_density_index": np.round(rng.uniform(0.7, 1.3, size=8), 3),
    })

def generate_customers(rng) -> pd.DataFrame:
    channels = np.array(["organic", "paid_social", "referral", "paid_search"])
    return pd.DataFrame({
        "customer_id": np.arange(1, N_CUSTOMERS + 1),
        "signup_day": rng.integers(0, N_DAYS, size=N_CUSTOMERS),
        "home_zone_id": rng.choice(np.arange(1, 9), size=N_CUSTOMERS, p=ZONE_WEIGHTS),
        "acquisition_channel": rng.choice(channels, size=N_CUSTOMERS, p=[.42, .24, .14, .20]),
        "device_os": rng.choice(["ios", "android"], size=N_CUSTOMERS, p=[.45, .55]),
        "base_order_rate": np.round(rng.gamma(2.0, 0.5, size=N_CUSTOMERS), 4),
    })

def generate_merchants(rng, zones) -> pd.DataFrame:
    cats = np.array(["Pizza", "Burger", "Japanese", "Brazilian", "Healthy", "Dessert"])
    return pd.DataFrame({
        "merchant_id": np.arange(1, N_MERCHANTS + 1),
        "merchant_name": [f"Merchant {i:03d}" for i in range(1, N_MERCHANTS + 1)],
        "zone_id": rng.choice(zones["zone_id"].values, size=N_MERCHANTS, p=ZONE_WEIGHTS),
        "category": rng.choice(cats, size=N_MERCHANTS),
        "commission_rate": np.round(rng.uniform(0.12, 0.28, size=N_MERCHANTS), 4),
        "avg_ticket": np.round(rng.uniform(28, 95, size=N_MERCHANTS), 2),
        "quality_score": np.round(rng.uniform(3.2, 4.9, size=N_MERCHANTS), 2),
    })


def _dates(days) -> pd.Series:
    return pd.Timestamp(START_DATE) + pd.to_timedelta(np.asarray(days), unit="D")


def generate_promotions() -> pd.DataFrame:
    """Promotion DEFINITIONS. Deterministic: a campaign calendar is a decision
    somebody made, not a random draw, so this consumes no rng."""
    df = pd.DataFrame(_PROMOTION_CALENDAR,
                      columns=["promotion_id", "promo_code", "promo_type",
                               "funded_by", "start_day", "end_day"])
    df["start_date"] = _dates(df["start_day"])
    df["end_date"] = _dates(df["end_day"])
    return df


def generate_experiments() -> pd.DataFrame:
    """Experiment DEFINITION grain -- one row per experiment. Kept separate from
    experiment_assignments (experiment x customer) on purpose: collapsing the
    two would make the hypothesis a per-customer attribute and the variant split
    unreadable."""
    df = pd.DataFrame([_EXPERIMENT])
    df["start_date"] = _dates(df["start_day"])
    df["end_date"] = _dates(df["end_day"])
    return df


def generate_experiment_assignments(rng, customers, experiments) -> pd.DataFrame:
    """Experiment x customer grain. The assignment mechanism an A/B test's
    statistics assume, and exactly that -- nothing more.

    THE POPULATION IS EVERY CUSTOMER. rng.choice over all N_CUSTOMERS without
    replacement draws N_EXPERIMENT_ASSIGNMENTS of them with equal probability.
    There is no eligibility filter of any kind: not recency, not order count,
    not zone. Every zone is represented in proportion to ZONE_WEIGHTS, which
    means the two zones carrying live incidents are IN, and the largest zone is
    the largest assigned group.

    THE SPLIT IS A PER-CUSTOMER COIN FLIP. rng.random() < 0.5 per row, so the
    realised split is binomial around 50/50 rather than exactly equal -- which
    is what gives check_randomisation's sample-ratio test something real to
    test, and why the observed treatment share is 49.74% and not 50.00%.

    NO TREATMENT EFFECT IS APPLIED ANYWHERE. generate_orders never reads the
    assignment table, so a treated customer's behaviour is drawn from the same
    distribution as a control customer's.
    """
    exp = experiments.iloc[0]
    pos = rng.choice(len(customers), size=N_EXPERIMENT_ASSIGNMENTS, replace=False)
    return pd.DataFrame({
        "experiment_id": exp["experiment_id"],
        "customer_id": np.sort(customers["customer_id"].to_numpy()[pos]),
        "variant": np.where(rng.random(N_EXPERIMENT_ASSIGNMENTS) < 0.5,
                            "treatment", "control"),
        "assigned_day": int(exp["start_day"]),
        "assigned_date": exp["start_date"],
    })


def _seasonality(days: np.ndarray) -> np.ndarray:
    """Weekly pattern + payday lift. Baseline behaviour, not an incident."""
    dow = (days + START_DATE.weekday()) % 7
    weekly = np.where(np.isin(dow, [4, 5]), 1.22, 1.0)          # Fri/Sat lift
    payday = np.where(np.isin(days % 30, [0, 1, 2, 14, 15]), 1.08, 1.0)
    return weekly * payday


def _zone_lut(zones: pd.DataFrame, column: str) -> np.ndarray:
    """zone_id -> value lookup array, so zone attributes join by indexing."""
    lut = np.zeros(int(zones["zone_id"].max()) + 1)
    lut[zones["zone_id"].to_numpy()] = zones[column].to_numpy()
    return lut


def _promotion_by_day(promotions: pd.DataFrame) -> np.ndarray:
    """day -> running promotion_id, 0 where no campaign is live."""
    lut = np.zeros(N_DAYS, dtype=np.int64)
    for pid, start, end in zip(promotions["promotion_id"],
                               promotions["start_day"], promotions["end_day"]):
        lut[int(start):int(end) + 1] = int(pid)
    return lut


# CONTEXT_DEFAULTS with every value expressible inside a numpy array: None is
# not, so the no-promotion placeholder is the 0 sentinel promotion_id uses
# everywhere in this module. _context() maps it back to None.
_ARRAY_PLACEHOLDERS = {**CONTEXT_DEFAULTS, "promotion_id": 0}


def _context(values: dict) -> IncidentContext:
    ctx = {**CONTEXT_DEFAULTS, **values}
    pid = ctx["promotion_id"]
    ctx["promotion_id"] = None if pid is None or int(pid) == 0 else int(pid)
    return IncidentContext(**ctx)


def _effects(**fields) -> dict[str, np.ndarray]:
    """Vectorised ``effects_for``: PER-INCIDENT memoisation.

    Each incident is evaluated by calling its own scalar function -- the single
    source of truth for incident logic -- once per distinct combination of the
    context fields IT declares in incidents.INCIDENT_KEYS, and the result is
    broadcast back to rows. Effects still combine multiplicatively across
    incidents, exactly as Effects.combine does.

    One shared (day, zone, hour) key is what this replaces: it would have
    evaluated promo_margin_erosion and paid_social_retention against
    placeholder values, so those incidents would silently never have fired.

    ``fields`` are row-aligned arrays named after IncidentContext fields; any
    field not passed is held at its (inert) CONTEXT_DEFAULTS value, so a
    generator only supplies the dimensions it actually has. promotion_id uses 0
    for "no promotion" -- numpy arrays cannot carry None.
    """
    n = len(next(iter(fields.values())))
    out = {f: np.ones(n) for f in _EFFECT_FIELDS}
    for fn in INCIDENTS:
        key = INCIDENT_KEYS[fn]
        cols = [np.asarray(fields[k]) if k in fields
                else np.full(n, _ARRAY_PLACEHOLDERS[k]) for k in key]
        codes, uniques = pd.MultiIndex.from_arrays(cols).factorize()
        values = np.array([
            [getattr(fn(_context(dict(zip(key, combo)))), f) for f in _EFFECT_FIELDS]
            for combo in uniques])
        for j, f in enumerate(_EFFECT_FIELDS):
            out[f] *= values[codes, j]
    return out


def _timestamps(days: np.ndarray, hours: np.ndarray,
                seconds_into_hour: np.ndarray) -> np.ndarray:
    secs = (days.astype(np.int64) * 86_400 + hours.astype(np.int64) * 3_600
            + seconds_into_hour.astype(np.int64))
    return _EPOCH + secs.astype("timedelta64[s]")


def _nullable_promotion_id(promotion_id: np.ndarray) -> pd.arrays.IntegerArray:
    col = pd.array(promotion_id, dtype="Int64")
    col[promotion_id == 0] = pd.NA
    return col


def generate_sessions(rng, customers, zones, promotions) -> pd.DataFrame:
    """~560k sessions over the fixed 180-day window.

    Volume is shaped by zone demand_weight, the hour-of-day curve and
    _seasonality(). Conversion probability is shaped by customer propensity and
    hour only -- never by day -- so a zone's conversion is flat over time unless
    an incident sets conversion_mult. Zone 7 never does: it fails AFTER
    checkout, which is what keeps it distinguishable from a pre-checkout break.
    Zone 4 and FREESHIP_WINTER both do, in opposite directions.
    """
    n = N_SESSIONS
    days = np.arange(N_DAYS)
    day_p = _seasonality(days)
    session_day = np.repeat(days, rng.multinomial(n, day_p / day_p.sum()))
    session_hour = rng.choice(_HOURS, size=n, p=_HOUR_P)
    zone_id = rng.choice(zones["zone_id"].to_numpy(), size=n,
                         p=zones["demand_weight"].to_numpy())

    # Retention decay is a CUSTOMER-level effect: an affected cohort simply
    # comes back less often. It is applied to how often a customer is drawn
    # into a session and NOT to propensity, because "returns less" is a
    # frequency change, not a change in how well a visit converts.
    order_rate = customers["base_order_rate"].to_numpy()
    repeat_mult = _effects(
        acquisition_channel=customers["acquisition_channel"].to_numpy(),
        signup_day=customers["signup_day"].to_numpy())["repeat_mult"]
    draw_weight = order_rate * repeat_mult

    # Customer drawn within the session's zone, weighted by return frequency.
    home_zone = customers["home_zone_id"].to_numpy()
    cust_pos = np.empty(n, dtype=np.int64)
    for z in zones["zone_id"].to_numpy():
        pool = np.flatnonzero(home_zone == z)
        rows = np.flatnonzero(zone_id == z)
        cust_pos[rows] = rng.choice(pool, size=rows.size,
                                    p=draw_weight[pool] / draw_weight[pool].sum())

    # Campaign exposure: which sessions a running promotion actually reached.
    #
    # Drawn from a JUMPED substream of the same seed, not from ``rng``. PCG64's
    # .jumped() advances the state far enough that the two streams provably
    # never overlap, so this is still one seed and still reproducible -- but it
    # does not shift every draw that follows it. Taking n doubles out of the
    # main stream here would have re-randomised `converted`, and with it every
    # order in the CLEAN part of the calendar, for a campaign that touches only
    # days 152-172.
    exposure = Generator(PCG64(SEED).jumped(1)).random(n)
    promotion_id = np.where(exposure < PROMO_EXPOSURE_SHARE,
                            _promotion_by_day(promotions)[session_day], 0)

    propensity = (order_rate[cust_pos] / order_rate.mean()) ** 0.6 \
        * np.where(np.isin(session_hour, PEAK_HOURS), 1.12, 0.95)
    eff = _effects(day=session_day, zone_id=zone_id, hour=session_hour,
                   promotion_id=promotion_id)
    p_convert = np.clip(
        TARGET_CONVERSION * propensity / propensity.mean() * eff["conversion_mult"],
        0.005, 0.95)

    return pd.DataFrame({
        "session_id": np.arange(1, n + 1),
        "customer_id": customers["customer_id"].to_numpy()[cust_pos],
        "zone_id": zone_id,
        "session_day": session_day,
        "session_hour": session_hour,
        "device_os": customers["device_os"].to_numpy()[cust_pos],
        "promotion_id": _nullable_promotion_id(promotion_id),
        "converted": rng.random(n) < p_convert,
    })


def generate_orders(rng, sessions, merchants, zones) -> pd.DataFrame:
    """One order per converting session -- the funnel identity, structurally.

    The Zone 7 incident reaches this table through cancel_mult (order STATUS,
    decided after placement) and actual_delivery_mult (courier minutes, hence
    delivery_cost). It never touches whether a session converted.
    """
    s = sessions.loc[sessions["converted"]]
    n = len(s)
    zone_id = s["zone_id"].to_numpy()
    order_day = s["session_day"].to_numpy()
    order_hour = s["session_hour"].to_numpy()
    promotion_id = s["promotion_id"].fillna(0).to_numpy(dtype=np.int64)
    eff = _effects(day=order_day, zone_id=zone_id, hour=order_hour,
                   promotion_id=promotion_id)

    # Merchant drawn within the session's zone, weighted by quality_score.
    merchant_zone = merchants["zone_id"].to_numpy()
    quality = merchants["quality_score"].to_numpy()
    m_pos = np.empty(n, dtype=np.int64)
    for z in zones["zone_id"].to_numpy():
        pool = np.flatnonzero(merchant_zone == z)
        rows = np.flatnonzero(zone_id == z)
        m_pos[rows] = rng.choice(pool, size=rows.size,
                                 p=quality[pool] / quality[pool].sum())

    item_amount = np.round(
        merchants["avg_ticket"].to_numpy()[m_pos]
        * rng.lognormal(-0.5 * 0.35 ** 2, 0.35, n), 2)

    band = rng.choice(3, size=n, p=_BAND_P)
    delivery_fee = _BAND_FEE[band]

    # Courier time: the physical quantity the Zone 7 incident degrades, and the
    # basis of delivery_cost. generate_deliveries reuses this same draw, so cost
    # and delivered duration agree per order.
    base_eta = _zone_lut(zones, "base_eta_minutes")[zone_id]
    density = _zone_lut(zones, "courier_density_index")[zone_id]
    courier_minutes = (base_eta * SERVICE_TIME_FACTOR * _BAND_DISTANCE[band]
                       / density ** 0.35
                       * rng.lognormal(-0.5 * SERVICE_TIME_SIGMA ** 2,
                                       SERVICE_TIME_SIGMA, n)
                       * eff["actual_delivery_mult"])

    # Cancellation is decided AFTER placement: post-checkout fulfilment failure.
    cancelled = rng.random(n) < np.clip(BASE_CANCEL_RATE * eff["cancel_mult"], 0.0, 0.95)
    entered_dispatch = ~(cancelled & (rng.random(n) < PRE_DISPATCH_CANCEL_SHARE))

    # Per-order cost with real variance: margin must not be a scalar function of
    # GMV, or a margin-destroying promotion would be mathematically undetectable.
    served_cost = COURIER_COST_FIXED_BRL + COURIER_COST_PER_MIN_BRL * courier_minutes
    delivery_cost = np.round(np.where(
        ~entered_dispatch, 0.0,
        np.where(cancelled, POST_DISPATCH_CANCEL_COST_SHARE * served_cost,
                 served_cost)), 2)

    discount_share = rng.uniform(0.08, 0.22, n) * (rng.random(n) < DISCOUNT_INCIDENCE)
    order_ts = _timestamps(order_day, order_hour, rng.integers(0, 3600, n))

    return pd.DataFrame({
        "order_id": np.arange(1, n + 1),
        "session_id": s["session_id"].to_numpy(),
        "customer_id": s["customer_id"].to_numpy(),
        "merchant_id": merchants["merchant_id"].to_numpy()[m_pos],
        "zone_id": zone_id,
        "order_day": order_day,
        "order_hour": order_hour,
        "order_ts": order_ts,
        "status": np.where(cancelled, "cancelled", "completed"),
        "item_amount": item_amount,
        "delivery_fee": delivery_fee,
        "delivery_cost": delivery_cost,
        "discount_amount": np.round(item_amount * discount_share * eff["discount_mult"], 2),
        "commission_rate": merchants["commission_rate"].to_numpy()[m_pos],
        "promotion_id": _nullable_promotion_id(promotion_id),
        "payment_method": rng.choice(["credit_card", "pix", "debit_card", "wallet"],
                                     size=n, p=[.48, .31, .13, .08]),
        "entered_dispatch": entered_dispatch,
        "courier_minutes": np.where(entered_dispatch, courier_minutes, np.nan),
    })


def generate_deliveries(rng, orders, zones) -> pd.DataFrame:
    """One row per order that entered dispatch (pre-dispatch cancels excluded).

    A post-dispatch cancel keeps its assignment and pickup timestamps but has no
    delivered_ts and no actual_delivery_minutes -- it was never delivered.
    """
    o = orders.loc[orders["entered_dispatch"]]
    n = len(o)
    eff = _effects(day=o["order_day"].to_numpy(), zone_id=o["zone_id"].to_numpy(),
                   hour=o["order_hour"].to_numpy())

    promised = np.round(_zone_lut(zones, "base_eta_minutes")[o["zone_id"].to_numpy()]
                        * eff["promised_eta_mult"] * rng.uniform(0.95, 1.12, n))

    minutes = o["courier_minutes"].to_numpy()
    order_ts = o["order_ts"].to_numpy().astype("datetime64[s]")
    offset = (minutes * 60).astype("timedelta64[s]")
    delivered = (o["status"].to_numpy() == "completed")

    return pd.DataFrame({
        "delivery_id": np.arange(1, n + 1),
        "order_id": o["order_id"].to_numpy(),
        "assigned_ts": order_ts + (offset * 0.12).astype("timedelta64[s]"),
        "picked_up_ts": order_ts + (offset * 0.45).astype("timedelta64[s]"),
        "delivered_ts": np.where(delivered, order_ts + offset,
                                 np.datetime64("NaT", "s")),
        "promised_eta_minutes": promised.astype(np.int64),
        "actual_delivery_minutes": np.where(delivered, minutes, np.nan),
    })


def generate_merchant_availability(rng, merchants) -> pd.DataFrame:
    """PERIODIC SNAPSHOT fact: one row per merchant per operating hour per day,
    emitted whether or not anything happened -- 150 x 14 x 180 = 378,000 rows.

    A transaction fact table cannot carry this. A closed merchant produces no
    orders, so its downtime is an ABSENCE, and an absence has to be manufactured
    into rows before any analysis can see it. This is what lets Task 17 tell
    "customers did not want to buy" (conversion fell, merchants were open) from
    "customers wanted to buy but supply was gone" (merchants were shut) -- the
    zone 4 / zone 7 distinction, measured rather than assumed.

    ``scheduled_open`` is the denominator of the availability KPI (a merchant
    closed on its rest day is not unavailable, it is shut), ``is_available`` the
    numerator. Zone 4's incident reaches only the numerator.
    """
    hours = np.asarray(OPERATING_HOURS)
    n_m, n_h = len(merchants), len(hours)
    closed_dow = rng.integers(0, 7, size=n_m)                    # weekly rest day
    opens_at = rng.choice(hours[:3], size=n_m)                   # 10:00 / 11:00 / 12:00
    uptime = np.clip(rng.normal(BASE_MERCHANT_UPTIME, 0.02, n_m), 0.80, 0.999)

    m = np.repeat(np.arange(n_m), N_DAYS * n_h)
    day = np.tile(np.repeat(np.arange(N_DAYS), n_h), n_m)
    hour = np.tile(hours, n_m * N_DAYS)

    scheduled_open = (((day + START_DATE.weekday()) % 7) != closed_dow[m]) \
        & (hour >= opens_at[m])
    eff = _effects(day=day, zone_id=merchants["zone_id"].to_numpy()[m], hour=hour)
    is_available = scheduled_open & (
        rng.random(len(m)) < uptime[m] * eff["availability_mult"])

    return pd.DataFrame({
        "merchant_id": merchants["merchant_id"].to_numpy()[m],
        "snapshot_ts": _timestamps(day, hour, np.zeros(len(m), dtype=np.int64)),
        "snapshot_day": day,
        "snapshot_hour": hour,
        "scheduled_open": scheduled_open,
        "is_available": is_available,
    })


def generate_all() -> dict[str, pd.DataFrame]:
    """Full bronze pipeline. One rng threads every generator, so the whole
    pipeline -- not just each generator in isolation -- is reproducible.

    Writes the ground-truth incident register, then injects quality defects
    (pulse.quality.inject_defects) using that SAME rng stream, and writes the
    DIRTY tables to bronze parquet -- Bronze is defined to be raw-with-defects.

    The CLEAN tables are what this function RETURNS, deliberately diverging
    from what lands on disk: every Task 5 test asserts properties of this
    return value (e.g. that orders reference only converting sessions, or the
    end-to-end reproducibility hash), and duplicated/corrupted rows would break
    those outright. Silver reads bronze parquet from disk, never this return
    value, so the two are free to disagree -- and must, for Bronze to contain
    anything for Silver to clean.
    """
    rng = make_rng()
    zones = generate_zones(rng)
    customers = generate_customers(rng)
    merchants = generate_merchants(rng, zones)
    promotions = generate_promotions()
    experiments = generate_experiments()
    sessions = generate_sessions(rng, customers, zones, promotions)
    orders = generate_orders(rng, sessions, merchants, zones)
    deliveries = generate_deliveries(rng, orders, zones)
    merchant_availability = generate_merchant_availability(rng, merchants)
    experiment_assignments = generate_experiment_assignments(rng, customers,
                                                             experiments)
    tables = {"zones": zones, "customers": customers, "merchants": merchants,
              "promotions": promotions, "experiments": experiments,
              "experiment_assignments": experiment_assignments,
              "sessions": sessions, "orders": orders, "deliveries": deliveries,
              "merchant_availability": merchant_availability}
    write_ground_truth()
    dirty = inject_defects(tables, rng)
    for name, df in dirty.items():
        write_bronze(df, name)
    return tables
