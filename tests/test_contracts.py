import dataclasses
import pandas as pd
import pytest
from datetime import date, timedelta

from pulse.types import AnalysisParams, Anomaly
from pulse.contracts import GOLD_CONTRACTS, GoldContractError, assert_contract
from pulse.quality import build_gold
from pulse.sql_runner import render_sql, run_sql_file
from pulse import io
from pulse.config import N_DAYS, N_ZONES, SILVER, SQL


def test_params_are_frozen_with_documented_defaults():
    p = AnalysisParams(as_of=date(2026, 9, 10))
    assert p.comparison_window_days == 14
    assert p.baseline_window_days == 56
    assert p.sensitivity == 2.5
    assert p.min_materiality_brl == 5_000
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.sensitivity = 9.0


def test_no_runtime_validation_in_src_is_a_bare_assert():
    """`python -O` strips assert statements. Every data gate and register check
    in src/ raises instead, and this keeps it that way."""
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "pulse"
    offenders = [
        f"{path.name}:{node.lineno}"
        for path in root.glob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Assert)
    ]
    assert not offenders, offenders


def test_the_gold_contract_gate_survives_python_dash_o():
    import subprocess
    import sys

    # A duplicated primary key: the gate must raise with asserts stripped.
    probe = (
        "import pandas as pd\n"
        "from pulse.contracts import GOLD_CONTRACTS, GoldContractError, assert_contract\n"
        "cols = GOLD_CONTRACTS['gold_daily_business_metrics'].required_columns\n"
        "df = pd.DataFrame({c: ['2026-01-01', '2026-01-01'] for c in cols})\n"
        "try:\n"
        "    assert_contract(df, 'gold_daily_business_metrics')\n"
        "except GoldContractError:\n"
        "    print('RAISED')\n"
    )
    out = subprocess.run([sys.executable, "-O", "-c", probe],
                         capture_output=True, text=True)
    assert "RAISED" in out.stdout, out.stderr[-1500:]


def _corrupt(mutate):
    import numpy as np  # noqa: F401  (used by the mutations below)

    df = io.read_gold("gold_zone_performance").copy()
    return mutate(df)


@pytest.mark.parametrize(
    "label, mutate",
    [
        ("negative gmv", lambda df: df.assign(gmv=-df["gmv"])),
        ("rate above one", lambda df: df.assign(completion_rate=df["completion_rate"] + 1)),
        ("infinite value", lambda df: df.assign(gmv=df["gmv"].where(df.index != df.index[0], float("inf")))),
        ("completed above placed", lambda df: df.assign(orders_completed=df["orders_placed"] + 1)),
        ("empty table", lambda df: df.iloc[0:0]),
        # Review #4: NaN GMV on six zone-7 days removed the GMV anomaly and
        # reordered the ranking. Additive columns are never "not measured".
        ("missing gmv", lambda df: df.assign(gmv=df["gmv"].where(df.index != df.index[0]))),
        ("missing rate with orders", lambda df: df.assign(
            completion_rate=df["completion_rate"].where(df.index != df.index[0]))),
        # Review #6: NaN availability on zone 4's recent days removed priority 3.
        ("missing availability with scheduled hours", lambda df: df.assign(
            availability_rate=df["availability_rate"].where(df.index != df.index[0]))),
        ("missing aov with completed orders", lambda df: df.assign(
            aov=df["aov"].where(df.index != df.index[0]))),
    ],
)
def test_the_gold_gate_rejects_values_no_marketplace_can_produce(gold, label, mutate):
    """Found by attacking the engine directly: every one of these ran to a
    complete, plausible-looking ranking (negative GMV on one zone-7 day moved
    the company priority from 61,59 to 42,58) with no signal that the input
    was impossible."""
    with pytest.raises(GoldContractError):
        assert_contract(_corrupt(mutate), "gold_zone_performance")


def test_load_gold_validates_what_it_reads(monkeypatch, gold):
    """The engine reads gold from disk, and a stale or hand-edited parquet is
    exactly what the build-time gate never sees."""
    from pulse import metrics

    real = io.read_gold

    def corrupted(name):
        df = real(name)
        return df.assign(gmv=-df["gmv"]) if name == "gold_zone_performance" else df

    monkeypatch.setattr(metrics, "read_gold", corrupted)
    with pytest.raises(GoldContractError, match="negative"):
        metrics.load_gold()


def test_every_gold_contract_declares_grain_and_pk():
    assert len(GOLD_CONTRACTS) == 6
    for name, c in GOLD_CONTRACTS.items():
        assert c.grain, f"{name} missing grain"
        assert c.primary_key, f"{name} missing primary key"
        assert set(c.primary_key).issubset(set(c.required_columns))


def test_assert_contract_rejects_duplicate_primary_key():
    df = pd.DataFrame({
        "metric_date": ["2026-01-01", "2026-01-01"],
        "zone_id": [7, 7],
        "gmv": [1.0, 2.0],
        "orders_placed": [10, 15],
        "orders_completed": [9, 14],
        "sessions": [100, 120],
        "order_conversion": [0.09, 0.11],
        "completion_rate": [0.9, 0.93],
        "cancellation_rate": [0.01, 0.01],
        "aov": [100.0, 110.0],
        "avg_promised_eta_minutes": [30, 32],
        "avg_actual_delivery_minutes": [35, 36],
        "on_time_rate": [0.85, 0.88],
        "contribution_margin": [500.0, 600.0],
    })
    with pytest.raises(GoldContractError, match="duplicate"):
        assert_contract(df, "gold_zone_performance")


def test_assert_contract_rejects_null_primary_key():
    df = pd.DataFrame({
        "metric_date": ["2026-01-01", None],
        "zone_id": [7, 4],
        "gmv": [1.0, 2.0],
        "orders_placed": [10, 15],
        "orders_completed": [9, 14],
        "sessions": [100, 120],
        "order_conversion": [0.09, 0.11],
        "completion_rate": [0.9, 0.93],
        "cancellation_rate": [0.01, 0.01],
        "aov": [100.0, 110.0],
        "avg_promised_eta_minutes": [30, 32],
        "avg_actual_delivery_minutes": [35, 36],
        "on_time_rate": [0.85, 0.88],
        "contribution_margin": [500.0, 600.0],
    })
    with pytest.raises(GoldContractError, match="null"):
        assert_contract(df, "gold_zone_performance")


# --------------------------------------------------------------------------
# The SQL runner's substitution rule
# --------------------------------------------------------------------------


def test_braces_in_a_sql_comment_do_not_break_the_runner(tmp_path):
    """A .sql file is mostly prose, and prose mentions placeholders.

    run_sql_file used to str.format() the whole file text, comments included, so
    a comment that happened to say {peak_hours} raised KeyError from a line that
    DuckDB would never even have seen. Task 8 lost time to exactly that. The
    premise is proved below rather than assumed: the same text is fed to
    str.format() and must still blow up there.
    """
    path = tmp_path / "braces.sql"
    path.write_text(
        "-- Mentions {peak_hours}, {not_a_param} and a JSON-ish blob {a: 1}.\n"
        "-- Prose braces are not placeholders and must survive untouched.\n"
        "SELECT '{silver}' AS silver_root, DATE '{end_date}' AS d;\n",
        encoding="utf-8",
    )

    with pytest.raises(KeyError):
        path.read_text(encoding="utf-8").format(silver="x", end_date="2026-09-10")

    rendered = render_sql(path, end_date="2026-09-10")
    assert "{not_a_param}" in rendered, "an unknown brace word must be left alone"
    assert "{peak_hours}" in rendered, "a placeholder NOT passed stays literal text"

    df = run_sql_file(path, end_date="2026-09-10")
    assert df.loc[0, "silver_root"] == SILVER.as_posix()
    assert df.loc[0, "d"] == pd.Timestamp(2026, 9, 10)


# --------------------------------------------------------------------------
# Gold
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def gold() -> None:
    """Build Gold once for the whole session.

    Every gold test depends on this rather than on a sibling test having run
    first: without it, `pytest -k test_gmv_is_item_amount` reads whatever stale
    parquet happens to be on disk and reports a pass or a failure that has
    nothing to do with the current code.
    """
    build_gold()


def test_gold_tables_satisfy_their_declared_contracts(gold):
    """All six, driven by the contract dict: a seventh dataset added without a
    contract, or a contract added without a build, fails here."""
    for name in GOLD_CONTRACTS:
        assert_contract(io.read_gold(name), name)


def test_calendar_spine_emits_a_row_for_every_zone_day(gold):
    """Without a spine, a zone-day with zero completed orders vanishes -
    the worse Zone 7 gets, the more of its bad days disappear.

    Shape only. Every zone-day in the real dataset happens to have activity, so
    a spine-less GROUP BY would also pass this -- the property itself is proved
    against a synthetic fixture with real holes in it, below.
    """
    z = io.read_gold("gold_zone_performance")
    assert len(z) == N_DAYS * N_ZONES
    assert z.groupby("zone_id")["metric_date"].nunique().eq(N_DAYS).all()
    assert z["gmv"].isna().sum() == 0          # zero-activity days are 0.0, not null


def test_gmv_is_item_amount_of_completed_orders_only(gold):
    orders = io.read_silver("orders")
    expected = orders.loc[orders["is_completed"], "item_amount"].sum()
    actual = io.read_gold("gold_daily_business_metrics")["gmv"].sum()
    assert abs(actual - expected) < 1.0


def test_merchant_performance_covers_every_merchant_day(gold):
    m = io.read_gold("gold_merchant_performance")
    merchants = io.read_silver("merchants")
    assert len(m) == N_DAYS * len(merchants)
    assert m.groupby("merchant_id")["metric_date"].nunique().eq(N_DAYS).all()
    # available_hours can never exceed the scheduled-open denominator.
    assert (m["available_hours"] <= m["scheduled_open_hours"]).all()
    # ...and the rate is genuinely absent, never 0.0, where nothing was scheduled.
    nothing_scheduled = m["scheduled_open_hours"] == 0
    assert nothing_scheduled.any(), "fixture premise: some merchant-days are shut"
    assert m.loc[nothing_scheduled, "availability_rate"].isna().all()
    assert m.loc[~nothing_scheduled, "availability_rate"].notna().all()


def test_availability_is_measured_not_inferred_from_orders(gold):
    """The point of the snapshot: a merchant-day with no orders is not the same
    thing as a merchant-day with no supply. If availability were inferred from
    the orders table these two populations would be identical."""
    m = io.read_gold("gold_merchant_performance")
    silent = m[(m["orders_placed"] == 0) & (m["scheduled_open_hours"] > 0)]
    assert len(silent) > 0, "premise: some merchant-days sold nothing while open"
    assert (silent["availability_rate"] > 0).any(), (
        "a merchant that was open and available but sold nothing must not read "
        "as unavailable -- availability is being inferred from demand"
    )


def test_experiment_results_carry_customer_denominators(gold):
    """Randomisation is per customer, so the denominators Task 19 divides by
    must be customer counts, not order counts."""
    e = io.read_gold("gold_experiment_results")
    assignments = io.read_silver("experiment_assignments")
    per_variant = assignments.groupby(["experiment_id", "variant"])["customer_id"].nunique()
    for (exp, variant), n in per_variant.items():
        rows = e[(e["experiment_id"] == exp) & (e["variant"] == variant)]
        assert (rows["assigned_customers"] == n).all()
    # A converted customer is a customer, not an order: on a day with repeat
    # buyers the two differ, and converted_customers must be the smaller.
    assert (e["converted_customers"] <= e["orders_completed"]).all()
    assert (e["converted_customers"] < e["orders_completed"]).any(), (
        "premise: some day has a customer ordering more than once"
    )


# --------------------------------------------------------------------------
# The spine, proved against a fixture that actually has holes in it
# --------------------------------------------------------------------------
#
# The real dataset populates all 1440 zone-days, so no assertion over it can
# distinguish a spine from a plain GROUP BY. This fixture is built with the
# holes the production data lacks, and it is the only thing standing between a
# future refactor and a silent regression to `GROUP BY metric_date, zone_id`.
#
# Four distinct zero-denominator paths, one per hole:
#   (3, day 2)  nothing at all -- the Zone 7 scenario, a zone gone dark
#   (*, day 4)  nothing anywhere -- proves the company-grain spine too
#   (1, day 3)  sessions but zero orders  -> order_conversion / completion / aov
#   (2, day 1)  orders but zero deliveries -> on_time_rate, NULL durations
#   (3, day 3)  orders, all cancelled, delivery rows that never delivered
#               -> aov (0 completed) and on_time_rate's INNER NULLIF

FIXTURE_START = date(2026, 1, 1)
FIXTURE_DAYS = 5
FIXTURE_ZONES = (1, 2, 3)
SESSIONS_PER_ZONE_DAY = 10

# A cancelled order carries a deliberately outlandish item_amount: every money
# column is FILTERed to completed orders, so if a FILTER were ever dropped the
# numbers would not drift slightly, they would explode, and these tests say so.
COMPLETED_AMOUNT, CANCELLED_AMOUNT = 100.0, 999.0
COMPLETED_MARGIN, CANCELLED_MARGIN = 10.0, 777.0
COMPLETED_DISCOUNT, CANCELLED_DISCOUNT = 5.0, 555.0
PROMISED_ETA = 30

# (zone, day, n_completed, n_cancelled, deliveries)
#   "all"         one delivered delivery per order
#   "none"        orders exist, dispatch never saw them
#   "undelivered" delivery rows exist but nothing was ever delivered
FIXTURE_ORDERS = [
    (1, 0, 3, 1, "all"),
    (1, 1, 2, 0, "all"),
    (1, 2, 2, 1, "all"),
    # (1, 3) absent on purpose: sessions, zero orders
    (2, 0, 2, 0, "all"),
    (2, 1, 3, 0, "none"),          # orders, zero deliveries
    (2, 2, 1, 1, "all"),
    (2, 3, 2, 0, "all"),
    (3, 0, 2, 0, "all"),
    (3, 1, 1, 1, "all"),
    # (3, 2) absent on purpose: the zone went dark
    (3, 3, 0, 2, "undelivered"),   # placed, all cancelled, never delivered
    # day 4 absent for every zone: the whole company went dark
]
NO_SESSIONS = {(3, 2)} | {(z, 4) for z in FIXTURE_ZONES}

DARK_ZONE_DAY = (3, 2)
DARK_DAY = 4

# One trading merchant per zone (id 10 * zone + 1), plus merchant 12 in zone 1
# which has availability snapshots but never sells anything.
FIXTURE_MERCHANTS = [(11, 1), (12, 1), (21, 2), (31, 3)]

# (merchant_id, day, hour, scheduled_open, is_available). Hand-computed answers
# are asserted below; nothing here is generated, so a change to the aggregation
# cannot quietly change the expectation too.
FIXTURE_AVAILABILITY = [
    # m11 day 0: 4 scheduled hours, 3 of them available, plus 2 hours it was
    # never scheduled to be open. Denominator must be 4 (scheduled), not 6
    # (rows), so the rate is 0.75 and not 0.5.
    (11, 0, 10, True, True),
    (11, 0, 11, True, True),
    (11, 0, 12, True, True),
    (11, 0, 13, True, False),
    (11, 0, 14, False, False),
    # scheduled_open False yet is_available True is impossible upstream; it is
    # here so that a numerator counting is_available alone would read 4/4 = 1.0.
    (11, 0, 15, False, True),
    (11, 1, 10, True, True),
    (11, 1, 11, True, True),
    # m11 day 2: weekly rest day. Shut is not the same as unavailable, so this
    # merchant-day has NO availability_rate at all.
    (11, 2, 10, False, False),
    (11, 2, 11, False, False),
    # m12 day 0: open long hours and almost entirely unavailable, in the same
    # zone as m11. The zone roll-up must be SUM/SUM = 4/12, not the average of
    # the two rates (0.4375).
    (12, 0, 10, True, True),
    *[(12, 0, h, True, False) for h in range(11, 18)],
    (21, 0, 10, True, True),
    (21, 0, 11, True, False),
    # Zone 3 went dark on ORDERS on day 2 while its merchant was open and fully
    # available the whole time: demand, not supply. That distinction is the
    # entire reason the snapshot table exists.
    (31, 2, 10, True, True),
    (31, 2, 11, True, True),
]

ZONE1_DAY0_AVAILABILITY = 4 / 12       # SUM(available) / SUM(scheduled)
ZONE1_DAY0_MEAN_OF_RATES = (0.75 + 0.125) / 2

# Columns that must never be NULL: a NULL here propagates into the anomaly
# detector's day-of-week baseline and poisons the whole weekday.
GUARDED = ["sessions", "orders_placed", "orders_completed", "orders_cancelled",
           "gmv", "contribution_margin", "order_conversion", "completion_rate",
           "cancellation_rate", "aov", "on_time_rate"]


def _fixture_silver(root):
    """Write a tiny silver layer with deliberate holes. Returns the frames."""
    root.mkdir(parents=True, exist_ok=True)
    zones = pd.DataFrame({"zone_id": list(FIXTURE_ZONES)})
    merchants = pd.DataFrame(FIXTURE_MERCHANTS, columns=["merchant_id", "zone_id"])
    availability = pd.DataFrame(
        [{"merchant_id": m,
          "snapshot_ts": pd.Timestamp(FIXTURE_START) + timedelta(days=d, hours=h),
          "scheduled_open": s, "is_available": a}
         for m, d, h, s, a in FIXTURE_AVAILABILITY])

    sessions = []
    for z in FIXTURE_ZONES:
        for d in range(FIXTURE_DAYS):
            if (z, d) in NO_SESSIONS:
                continue
            sessions += [{"session_id": len(sessions) + i, "session_day": d,
                          "zone_id": z} for i in range(SESSIONS_PER_ZONE_DAY)]
    sessions = pd.DataFrame(sessions)

    orders, deliveries = [], []
    for z, d, n_done, n_cancel, mode in FIXTURE_ORDERS:
        for i, done in enumerate([True] * n_done + [False] * n_cancel):
            oid = len(orders)
            orders.append({
                "order_id": oid,
                "customer_id": oid % 4,
                "zone_id": z,
                "merchant_id": 10 * z + 1,
                # a real date, not a timestamp: silver orders stores order_date
                # as DATE, and the spine joins DATE to DATE.
                "order_date": FIXTURE_START + timedelta(days=d),
                "is_completed": done,
                "is_cancelled": not done,
                "item_amount": COMPLETED_AMOUNT if done else CANCELLED_AMOUNT,
                "contribution_margin": COMPLETED_MARGIN if done else CANCELLED_MARGIN,
                "discount_amount": COMPLETED_DISCOUNT if done else CANCELLED_DISCOUNT,
            })
            if mode == "none":
                continue
            # first order of each group is late, so on_time_rate is never a
            # degenerate 1.0 that a broken denominator could accidentally match
            actual = None if mode == "undelivered" else (50.0 if i == 0 else 28.0)
            deliveries.append({
                "order_id": oid,
                "promised_eta_minutes": PROMISED_ETA,
                "actual_delivery_minutes": actual,
                "is_on_time": None if actual is None else actual <= PROMISED_ETA + 5,
            })
    orders = pd.DataFrame(orders)
    deliveries = pd.DataFrame(deliveries)
    deliveries["is_on_time"] = deliveries["is_on_time"].astype("boolean")

    for name, df in (("zones", zones), ("sessions", sessions),
                     ("orders", orders), ("deliveries", deliveries),
                     ("merchants", merchants),
                     ("merchant_availability", availability)):
        df.to_parquet(root / f"{name}.parquet", index=False)
    return zones, sessions, orders, deliveries


def _run_gold(name, monkeypatch, root):
    monkeypatch.setattr("pulse.sql_runner.SILVER", root)
    return run_sql_file(
        SQL / "gold" / f"{name}.sql",
        start_date=FIXTURE_START.isoformat(),
        end_date=(FIXTURE_START + timedelta(days=FIXTURE_DAYS - 1)).isoformat(),
    )


def _day(offset):
    return pd.Timestamp(FIXTURE_START + timedelta(days=offset))


def test_spine_materialises_a_zone_day_with_no_activity(tmp_path, monkeypatch):
    """The regression guard the real-data test cannot be: a zone that went dark
    must still produce a row, or the worse it gets the more it disappears."""
    _, sessions, orders, _ = _fixture_silver(tmp_path / "silver")
    zone, day = DARK_ZONE_DAY

    # the fixture genuinely has the hole -- otherwise this test is vacuous too
    assert not ((orders["zone_id"] == zone) & (orders["order_date"] == _day(day))).any()
    assert not ((sessions["zone_id"] == zone) & (sessions["session_day"] == day)).any()

    z = _run_gold("gold_zone_performance", monkeypatch, tmp_path / "silver")

    assert len(z) == FIXTURE_DAYS * len(FIXTURE_ZONES)
    assert z[GUARDED].isna().sum().sum() == 0, "a guarded column came back NULL"

    dark = z[(z["zone_id"] == zone) & (z["metric_date"] == _day(day))]
    assert len(dark) == 1, "the dark zone-day vanished: the spine is not holding"
    assert (dark[GUARDED].iloc[0] == 0).all(), "zero activity must read as 0, not NULL"
    # an average of nothing is not zero minutes, so these two stay NULL
    assert dark["avg_promised_eta_minutes"].isna().all()
    assert dark["avg_actual_delivery_minutes"].isna().all()
    # ...and the reason this table carries availability at all: the zone sold
    # nothing while its supply was fully open. Demand, not supply.
    assert dark["availability_rate"].iloc[0] == 1.0

    # the whole company going dark materialises too, all zones at once
    blackout = z[z["metric_date"] == _day(DARK_DAY)]
    assert len(blackout) == len(FIXTURE_ZONES)
    assert (blackout[GUARDED] == 0).all().all()


def test_zone_availability_is_a_weighted_rollup_of_scheduled_hours(tmp_path, monkeypatch):
    """Zone availability must be SUM(available) / SUM(scheduled), not the mean
    of the per-merchant rates: an average of averages weights a merchant open
    two hours the same as one open fourteen."""
    _fixture_silver(tmp_path / "silver")
    z = _run_gold("gold_zone_performance", monkeypatch, tmp_path / "silver")
    row = z[(z["zone_id"] == 1) & (z["metric_date"] == _day(0))].iloc[0]

    assert row["scheduled_open_hours"] == 12      # m11's 4 + m12's 8
    assert row["available_hours"] == 4            # m11's 3 + m12's 1
    assert row["availability_rate"] == pytest.approx(ZONE1_DAY0_AVAILABILITY)
    assert row["availability_rate"] != pytest.approx(ZONE1_DAY0_MEAN_OF_RATES)

    # A zone-day with no snapshot at all has no rate -- not a fabricated 0.0.
    quiet = z[(z["zone_id"] == 2) & (z["metric_date"] == _day(3))].iloc[0]
    assert quiet["scheduled_open_hours"] == 0
    assert pd.isna(quiet["availability_rate"])


def test_merchant_availability_denominator_is_scheduled_open_hours(tmp_path, monkeypatch):
    """Hand-computed, one case per way the denominator could be wrong."""
    _fixture_silver(tmp_path / "silver")
    m = _run_gold("gold_merchant_performance", monkeypatch, tmp_path / "silver")
    assert len(m) == FIXTURE_DAYS * len(FIXTURE_MERCHANTS)
    row = lambda mid, day: m[(m["merchant_id"] == mid)
                             & (m["metric_date"] == _day(day))].iloc[0]

    # Open 4 of the 6 hours snapshotted, available for 3 of those 4.
    #   scheduled-open denominator -> 3/4  = 0.75   (correct)
    #   all-rows denominator       -> 3/6  = 0.50
    #   is_available numerator ignoring scheduled_open -> 4/4 = 1.00
    r = row(11, 0)
    assert (r["scheduled_open_hours"], r["available_hours"]) == (4, 3)
    assert r["availability_rate"] == pytest.approx(0.75)

    # Fully available on every scheduled hour.
    assert row(11, 1)["availability_rate"] == pytest.approx(1.0)

    # Weekly rest day: shut, not unavailable. No rate at all, and emphatically
    # not 0.0 -- a 0.0 here is a fabricated outage on one seventh of all
    # merchant-days, which would bury every real one.
    r = row(11, 2)
    assert r["scheduled_open_hours"] == 0
    assert pd.isna(r["availability_rate"])

    # Open all day, almost nothing available, and it sold nothing either. The
    # spine still emits the row.
    r = row(12, 0)
    assert (r["scheduled_open_hours"], r["available_hours"]) == (8, 1)
    assert r["availability_rate"] == pytest.approx(0.125)
    assert r["orders_placed"] == 0 and r["gmv"] == 0.0

    # Open and fully available on a day the zone recorded no orders at all.
    r = row(31, 2)
    assert r["availability_rate"] == pytest.approx(1.0)
    assert r["orders_placed"] == 0

    # Trading numbers still come from the orders table: zone 1 day 0 is
    # 3 completed + 1 cancelled, all on merchant 11.
    r = row(11, 0)
    assert (r["orders_placed"], r["orders_completed"]) == (4, 3)
    assert r["gmv"] == 3 * COMPLETED_AMOUNT      # the cancelled 999 stayed out
    assert r["contribution_margin"] == 3 * COMPLETED_MARGIN


def test_every_zero_denominator_guard_yields_zero_not_null(tmp_path, monkeypatch):
    """One case per division in the SQL, each with a genuinely zero denominator."""
    _fixture_silver(tmp_path / "silver")
    z = _run_gold("gold_zone_performance", monkeypatch, tmp_path / "silver")
    row = lambda zone, day: z[(z["zone_id"] == zone)
                              & (z["metric_date"] == _day(day))].iloc[0]

    # sessions > 0, orders_placed == 0 -> conversion, completion, cancellation, aov
    r = row(1, 3)
    assert r["sessions"] == SESSIONS_PER_ZONE_DAY and r["orders_placed"] == 0
    assert (r["order_conversion"], r["completion_rate"],
            r["cancellation_rate"], r["aov"], r["gmv"]) == (0.0, 0.0, 0.0, 0.0, 0.0)

    # orders placed, zero deliveries -> on_time_rate guarded, durations NULL
    r = row(2, 1)
    assert r["orders_placed"] == 3 and r["orders_completed"] == 3
    assert r["gmv"] == 3 * COMPLETED_AMOUNT
    assert r["on_time_rate"] == 0.0
    assert pd.isna(r["avg_promised_eta_minutes"])
    assert pd.isna(r["avg_actual_delivery_minutes"])

    # every order cancelled, deliveries exist but none delivered. Exercises the
    # INNER NULLIF: avg_promised is real, yet on_time_rate's denominator is 0.
    r = row(3, 3)
    assert (r["orders_placed"], r["orders_completed"], r["orders_cancelled"]) == (2, 0, 2)
    assert r["gmv"] == 0.0 and r["aov"] == 0.0        # FILTER kept the 999s out
    assert r["completion_rate"] == 0.0 and r["cancellation_rate"] == 1.0
    assert r["avg_promised_eta_minutes"] == PROMISED_ETA
    assert pd.isna(r["avg_actual_delivery_minutes"])
    assert r["on_time_rate"] == 0.0

    # an ordinary day still computes: 3 completed of 4 placed, one late delivery
    r = row(1, 0)
    assert (r["orders_placed"], r["orders_completed"]) == (4, 3)
    assert r["gmv"] == 3 * COMPLETED_AMOUNT
    assert r["completion_rate"] == 0.75 and r["cancellation_rate"] == 0.25
    assert r["order_conversion"] == 4 / SESSIONS_PER_ZONE_DAY
    assert r["aov"] == COMPLETED_AMOUNT
    assert r["on_time_rate"] == 0.75                  # 1 of 4 deliveries was late


def test_daily_spine_materialises_a_day_with_no_activity(tmp_path, monkeypatch):
    """Same property at company grain, where a dark day has no surviving zone
    to hint that it ever existed."""
    _fixture_silver(tmp_path / "silver")
    d = _run_gold("gold_daily_business_metrics", monkeypatch, tmp_path / "silver")

    assert len(d) == FIXTURE_DAYS
    guarded = GUARDED + ["active_customers", "discount_amount"]
    assert d[guarded].isna().sum().sum() == 0

    dark = d[d["metric_date"] == _day(DARK_DAY)]
    assert len(dark) == 1, "the dark day vanished: the spine is not holding"
    assert (dark[guarded].iloc[0] == 0).all()

    # day 0: 7 completed across three zones, and the cancelled order's 999
    # stayed out of every money column
    first = d[d["metric_date"] == _day(0)].iloc[0]
    assert first["orders_placed"] == 8 and first["orders_completed"] == 7
    assert first["gmv"] == 7 * COMPLETED_AMOUNT
    assert first["contribution_margin"] == 7 * COMPLETED_MARGIN
    assert first["discount_amount"] == 7 * COMPLETED_DISCOUNT
    assert first["active_customers"] == 4        # customer_id = order_id % 4


def test_gold_window_comes_from_config_not_from_sql_literals():
    """config owns the window. A date literal back in the SQL would build the
    wrong range while every test that read the same literal kept passing.

    Globbed rather than listed, so a gold file added later is covered without
    anyone remembering to add it here. gold_customer_retention takes only
    {end_date} (cohorts run from the data, not from a start date) and
    gold_experiment_results takes neither -- each experiment's window comes from
    the experiment definition -- so the placeholder half of the check names the
    calendar-spined datasets explicitly.
    """
    windowed = ("gold_zone_performance", "gold_daily_business_metrics",
                "gold_merchant_performance", "gold_promotion_performance")
    bodies = {}
    for path in sorted((SQL / "gold").glob("*.sql")):
        sql = path.read_text(encoding="utf-8")
        bodies[path.stem] = "\n".join(line for line in sql.splitlines()
                                      if not line.lstrip().startswith("--"))
    assert set(bodies) == set(GOLD_CONTRACTS), "a gold .sql file has no contract"
    for name, body in bodies.items():
        assert "2026-03-15" not in body and "2026-09-10" not in body, name
    for name in windowed:
        assert "{start_date}" in bodies[name] and "{end_date}" in bodies[name], name


# --------------------------------------------------------------------------
# Retention, proved against a fixture whose answer was computed by hand
# --------------------------------------------------------------------------
#
# A retention rate is the easiest number in this project to get wrong and the
# hardest to catch: a wrong DENOMINATOR still produces a plausible value
# between 0 and 1, and "period 0 is 1.0" is true by construction of the bug as
# well as of the fix if cohort_size is lifted out of period 0 rather than
# counted independently. So the sizes and the retained counts below are written
# out by hand and asserted exactly.
#
#   c1  organic      2026-01-05, 2026-01-20, 2026-01-25, 2026-02-10, 2026-04-02
#       -> cohort 2026-01. THREE orders in period 0 and it is still one
#          retained customer. Period 3 (April) with no period 2 in between.
#   c2  organic      2026-01-15, 2026-02-05          -> cohort 2026-01
#   c3  organic      2026-01-28 CANCELLED, 2026-02-14 -> cohort 2026-02, because
#          a cancelled order is not a purchase and must not found a cohort.
#   c4  paid_social  2026-01-09, 2026-03-03          -> cohort 2026-01, period 2
#   c5  paid_social  2026-02-20                      -> cohort 2026-02
#
# Cohort (2026-01, organic) therefore has cohort_size 2 (c1, c2) while holding
# FOUR period-0 orders: any count that is not COUNT(DISTINCT customer_id) lands
# on 4, or on 3, and never on 2.

RETENTION_END = date(2026, 4, 30)
RETENTION_CUSTOMERS = [
    (1, "organic"), (2, "organic"), (3, "organic"),
    (4, "paid_social"), (5, "paid_social"),
]
# (customer_id, order_date, is_completed)
RETENTION_ORDERS = [
    (1, date(2026, 1, 5), True),
    (1, date(2026, 1, 20), True),
    (1, date(2026, 1, 25), True),
    (1, date(2026, 2, 10), True),
    (1, date(2026, 4, 2), True),
    (2, date(2026, 1, 15), True),
    (2, date(2026, 2, 5), True),
    (3, date(2026, 1, 28), False),
    (3, date(2026, 2, 14), True),
    (4, date(2026, 1, 9), True),
    (4, date(2026, 3, 3), True),
    (5, date(2026, 2, 20), True),
]
# (cohort_month, channel, period_index) -> (cohort_size, retained_customers)
RETENTION_EXPECTED = {
    ("2026-01", "organic", 0): (2, 2),
    ("2026-01", "organic", 1): (2, 2),
    ("2026-01", "organic", 3): (2, 1),
    ("2026-01", "paid_social", 0): (1, 1),
    ("2026-01", "paid_social", 2): (1, 1),
    ("2026-02", "organic", 0): (1, 1),
    ("2026-02", "paid_social", 0): (1, 1),
}


def _retention_silver(root):
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(RETENTION_CUSTOMERS,
                 columns=["customer_id", "acquisition_channel"]
                 ).to_parquet(root / "customers.parquet", index=False)
    pd.DataFrame([{"order_id": i, "customer_id": c, "order_date": d,
                   "is_completed": done, "is_cancelled": not done}
                  for i, (c, d, done) in enumerate(RETENTION_ORDERS)]
                 ).to_parquet(root / "orders.parquet", index=False)


def _run_retention(monkeypatch, root):
    monkeypatch.setattr("pulse.sql_runner.SILVER", root)
    return run_sql_file(SQL / "gold" / "gold_customer_retention.sql",
                        start_date=date(2026, 1, 1).isoformat(),
                        end_date=RETENTION_END.isoformat())


def test_retention_cohort_sizes_and_retained_counts_are_exact(tmp_path, monkeypatch):
    _retention_silver(tmp_path / "silver")
    r = _run_retention(monkeypatch, tmp_path / "silver")

    got = {(row.cohort_month, row.acquisition_channel, row.period_index):
           (row.cohort_size, row.retained_customers)
           for row in r.itertuples(index=False)}
    assert got == RETENTION_EXPECTED

    # No period-2 row for (2026-01, organic): nobody ordered in March. That is a
    # genuine absence, not a zero -- inventing 0.0 rows would fabricate a
    # retention collapse at the edge of every cohort.
    assert ("2026-01", "organic", 2) not in got


def test_retention_denominator_is_customers_not_orders(tmp_path, monkeypatch):
    """The test the fixture exists for. A wrong denominator still yields a
    number between 0 and 1, so the premise is stated numerically: this cohort
    holds four period-0 ORDERS and two period-0 CUSTOMERS."""
    _retention_silver(tmp_path / "silver")

    cohort = [o for o in RETENTION_ORDERS
              if o[0] in (1, 2) and o[1] < date(2026, 2, 1) and o[2]]
    assert len(cohort) == 4, "premise: four orders"
    assert len({o[0] for o in cohort}) == 2, "premise: two customers"

    r = _run_retention(monkeypatch, tmp_path / "silver")
    row = r[(r.cohort_month == "2026-01") & (r.acquisition_channel == "organic")
            & (r.period_index == 0)].iloc[0]
    assert row["cohort_size"] == 2
    assert row["retained_customers"] == 2
    assert row["retention_rate"] == 1.0


def test_a_customer_ordering_three_times_in_one_period_counts_once(tmp_path, monkeypatch):
    _retention_silver(tmp_path / "silver")
    repeats = [o for o in RETENTION_ORDERS
               if o[0] == 1 and o[1] < date(2026, 2, 1)]
    assert len(repeats) == 3, "premise: c1 ordered three times in period 0"

    r = _run_retention(monkeypatch, tmp_path / "silver")
    # c4 is alone in (2026-01, paid_social), so its cell isolates one customer.
    # c1 shares its cell with c2, and the pair must count 2 -- not 4.
    assert r[(r.cohort_month == "2026-01") & (r.acquisition_channel == "organic")
             & (r.period_index == 0)]["retained_customers"].iloc[0] == 2


def test_cohort_comes_from_the_first_completed_order(tmp_path, monkeypatch):
    """c3's first order was cancelled, so it founds nothing; c1's cohort must
    come from its January order and not from its April one."""
    _retention_silver(tmp_path / "silver")
    r = _run_retention(monkeypatch, tmp_path / "silver")

    assert r[r.cohort_month == "2026-01"]["cohort_size"].max() == 2
    assert set(r[(r.cohort_month == "2026-02")
                 & (r.acquisition_channel == "organic")]["period_index"]) == {0}
    # c1 in period 3 proves its cohort stayed anchored on the FIRST order: if
    # the cohort had drifted to the latest order, April would be period 0.
    assert r[(r.cohort_month == "2026-01") & (r.acquisition_channel == "organic")
             & (r.period_index == 3)]["retained_customers"].iloc[0] == 1


def test_period_zero_is_exactly_one_by_construction(tmp_path, monkeypatch):
    _retention_silver(tmp_path / "silver")
    r = _run_retention(monkeypatch, tmp_path / "silver")
    p0 = r[r.period_index == 0]
    assert len(p0) == 4                      # every cohort x channel pair
    assert (p0["retention_rate"] == 1.0).all()
    assert (p0["retained_customers"] == p0["cohort_size"]).all()


def test_retention_marks_periods_the_window_only_partly_saw(tmp_path, monkeypatch):
    """The last period of a cohort is usually truncated by the window, and a
    truncated period is not comparable with a whole one."""
    _retention_silver(tmp_path / "silver")
    r = _run_retention(monkeypatch, tmp_path / "silver")
    assert (r["period_days_observed"] <= r["period_days"]).all()
    assert (r["period_days_observed"] > 0).all()
    # April 2026 is 30 days and the window ends on the 30th, so nothing is
    # truncated here; shortening the window must start truncating.
    monkeypatch.setattr("pulse.sql_runner.SILVER", tmp_path / "silver")
    short = run_sql_file(SQL / "gold" / "gold_customer_retention.sql",
                         start_date=date(2026, 1, 1).isoformat(),
                         end_date=date(2026, 4, 10).isoformat())
    april = short[(short.cohort_month == "2026-01")
                  & (short.acquisition_channel == "organic")
                  & (short.period_index == 3)].iloc[0]
    assert april["period_days"] == 30 and april["period_days_observed"] == 10


# --------------------------------------------------------------------------
# The incidents, as they appear in the real gold tables
# --------------------------------------------------------------------------


def test_paid_social_retention_is_below_every_other_channel_on_average(gold):
    """The retention incident keys on (acquisition_channel, signup_day), and
    signup day is NOT cohort month -- session dates are not gated on signup, so
    an affected customer's first order can land in any cohort. The incident
    therefore shows up as a CHANNEL-level gap, present in every cohort, and not
    as a trend across cohort months.

    Worth being explicit about why the obvious test is not the one written here:
    paid_social period-1 retention does fall steeply from the early cohorts to
    the late ones -- but so does every other channel's, by almost exactly the
    same factor, because late cohorts are made of customers who took months to
    place a first order (low-frequency by selection) and their period 1 is
    truncated by the end of the window. A test asserting only that decay would
    pass with the incident removed.
    """
    r = io.read_gold("gold_customer_retention")
    p1 = r[(r["period_index"] == 1) & (r["cohort_size"] >= 100)]
    assert len(p1) >= 20, "premise: enough cohort x channel cells to compare"

    wide = p1.pivot(index="cohort_month", columns="acquisition_channel",
                    values="retention_rate")
    others = wide.drop(columns="paid_social").mean(axis=1)
    gap = wide["paid_social"] / others - 1.0
    assert (gap < 0).all(), (
        f"paid_social is not below the other channels in every cohort: {gap.to_dict()}"
    )
    assert gap.mean() < -0.05, f"channel gap too small to be the incident: {gap.mean()}"


def test_promoted_orders_lose_margin_while_gaining_conversion(gold):
    """The promo incident is an ECONOMIC shape, not a promotion id: more orders,
    a much bigger subsidy, less margin on each one. It is found by comparing a
    campaign to the non-promoted orders of THE SAME DAYS (promotion_id 0), which
    holds seasonality and any concurrent operational incident fixed.

    Observational, not causal: campaign exposure is not randomised, so this is
    an association between carrying a promotion and earning less margin.
    """
    p = io.read_gold("gold_promotion_performance")
    campaigns = p[p["promotion_id"] != 0]
    assert len(campaigns) > 0

    def economics(rows):
        return (rows["contribution_margin"].sum() / rows["orders_completed"].sum(),
                rows["discount_amount"].sum() / rows["orders_completed"].sum(),
                rows["orders_with_promo"].sum() / rows["sessions"].sum())

    worst, worst_gap = None, 0.0
    for pid, rows in campaigns.groupby("promotion_id"):
        same_days = p[(p["promotion_id"] == 0)
                      & p["metric_date"].isin(rows["metric_date"])]
        margin, discount, conv = economics(rows)
        base_margin, base_discount, base_conv = economics(same_days)
        gap = margin / base_margin - 1.0
        if gap < worst_gap:
            worst, worst_gap = (pid, discount / base_discount, conv / base_conv), gap

    assert worst is not None, "no campaign earns less margin per order than baseline"
    pid, discount_x, conv_x = worst
    # The offender is found by its economics; its identity is only reported.
    code = p.loc[p["promotion_id"] == pid, "promo_code"].iloc[0]
    assert worst_gap < -0.20, f"{code}: margin gap {worst_gap:.3f}"
    assert discount_x > 2.0, f"{code}: discount per order only x{discount_x:.2f}"
    assert conv_x > 1.10, f"{code}: conversion only x{conv_x:.2f}"


def test_one_zone_loses_supply_availability_and_the_rest_do_not(gold):
    """Zone 4's incident has to be findable from the availability measurement
    itself. Nothing below names a zone: the worst zone falls out of the data,
    and the assertion is that exactly one zone separates from the pack.
    """
    from pulse.anomaly_detection import detect_on_series
    from pulse.config import AS_OF

    z = io.read_gold("gold_zone_performance").copy()
    z["metric_date"] = pd.to_datetime(z["metric_date"])
    params = AnalysisParams(as_of=AS_OF)

    fired = {}
    for zone, rows in z.groupby("zone_id"):
        series = rows.set_index("metric_date")["availability_rate"]
        anomaly = detect_on_series(series, "availability_rate", "zone", str(zone), params)
        if anomaly is not None and anomaly.direction == "drop":
            fired[zone] = anomaly

    assert len(fired) == 1, f"expected one supply-starved zone, got {sorted(fired)}"
    (zone, anomaly), = fired.items()
    assert isinstance(anomaly, Anomaly)
    assert abs(anomaly.z_score) > params.sensitivity
    assert anomaly.deviation_pct < -3.0
