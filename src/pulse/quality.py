"""Bronze quality-defect injection, the Silver build that cleans it up, and the
Gold build that aggregates Silver into the business-facing datasets.

Injection is applied AFTER clean generation (generate_all()), never during it,
so the recorded counts are exact and independent of generation logic -- the
Silver layer's cleaning tests can assert precise numbers against them.

This module WRITES data/ground_truth/quality_defects.json and must never read
it back. Ground truth exists only so tests can verify Silver independently
detected and quarantined what was injected here; reading it back here or from
build_silver() would be grading the cleaning code's own homework.

All randomness comes from the single seeded rng passed in by the caller
(``Generator(PCG64(42))`` via ``pulse.data_generator.make_rng``).
"""

import json
import numpy as np
import pandas as pd

from pulse import io
from pulse.config import END_DATE, GROUND_TRUTH, PEAK_HOURS, SILVER, SQL, START_DATE
from pulse.contracts import assert_contract
from pulse.sql_runner import run_sql_file

PAYMENT_VARIANTS = ["credit_card", "CREDIT_CARD", "Credit Card", "cc"]
CATEGORY_VARIANTS = {"Pizza": ["Pizza", "pizza", "PIZZA "]}


def _drift(rng, current: pd.Series, variants: list[str]) -> np.ndarray:
    """Redraw each row's value from ``variants``, guaranteeing the result
    differs from what was already there.

    A drift defect that silently redraws the same string it started from
    isn't an observable defect at all -- it would make the recorded count
    disagree with what a test can actually see in the data. ``variants`` is
    assumed to hold more than one distinct string, so the rejection loop
    below always terminates (typically after zero or one retry).
    """
    # A single-valued variants list would make the rejection loop below
    # unsatisfiable and hang forever, silently, with no output to diagnose.
    # An exception is a far cheaper failure than a wedged process.
    if len(set(variants)) <= 1:
        raise ValueError("drift needs at least two distinct variants")
    variants = np.asarray(variants, dtype=object)
    current = current.to_numpy()
    choice = rng.choice(variants, size=len(current))
    collide = choice == current
    while collide.any():
        choice[collide] = rng.choice(variants, size=collide.sum())
        collide = choice == current
    return choice


def inject_defects(tables, rng) -> dict[str, pd.DataFrame]:
    """Applied AFTER clean generation so ground truth stays exact.
    Records every count to data/ground_truth/quality_defects.json."""
    counts = {}
    o = tables["orders"].copy()

    # 1. exact duplicates (0.8%)
    dup_idx = rng.choice(o.index, size=int(0.008 * len(o)), replace=False)
    o = pd.concat([o, o.loc[dup_idx]], ignore_index=True)
    counts["duplicate_orders"] = len(dup_idx)

    # 2. null delivery_fee (1.5%) - NUMERIC, imputable in silver
    n_idx = rng.choice(o.index, size=int(0.015 * len(o)), replace=False)
    o.loc[n_idx, "delivery_fee"] = None
    counts["null_delivery_fee"] = len(n_idx)

    # 3. category drift on payment_method
    d_idx = rng.choice(o.index, size=int(0.12 * len(o)), replace=False)
    o.loc[d_idx, "payment_method"] = _drift(rng, o.loc[d_idx, "payment_method"],
                                            PAYMENT_VARIANTS)
    counts["category_drift_payment"] = len(d_idx)

    # 4. orphan merchant FK (0.2%)
    f_idx = rng.choice(o.index, size=int(0.002 * len(o)), replace=False)
    o.loc[f_idx, "merchant_id"] = 999_999
    counts["orphan_merchant_fk"] = len(f_idx)

    # 5. timestamp violations (0.3%) on deliveries -- delivered before assigned.
    # Restricted to rows that were actually delivered (non-null delivered_ts):
    # a post-dispatch cancel's NaT is a different data-quality story (never
    # delivered), not a causality violation, so it must stay ineligible here.
    dl = tables["deliveries"].copy()
    eligible_dl = dl.index[dl["delivered_ts"].notna()]
    t_idx = rng.choice(eligible_dl, size=int(0.003 * len(dl)), replace=False)
    dl.loc[t_idx, "delivered_ts"] = dl.loc[t_idx, "assigned_ts"] - pd.Timedelta(hours=2)
    counts["timestamp_violations"] = len(t_idx)

    # 6. null zone_id on sessions (0.5%) - CATEGORICAL, never imputed
    s = tables["sessions"].copy()
    z_idx = rng.choice(s.index, size=int(0.005 * len(s)), replace=False)
    s.loc[z_idx, "zone_id"] = None
    counts["null_session_zone_id"] = len(z_idx)

    # 7. category drift on merchants.category -- the dimension-table twin of
    # #3. Only merchants whose category has a known variant mapping are
    # eligible (today just "Pizza"); the loop below is written to cover more
    # categories the day CATEGORY_VARIANTS grows without further changes here.
    m = tables["merchants"].copy()
    eligible_m = m.index[m["category"].isin(list(CATEGORY_VARIANTS))]
    mc_idx = rng.choice(eligible_m, size=int(0.12 * len(eligible_m)), replace=False)
    for cat, variants in CATEGORY_VARIANTS.items():
        rows = mc_idx[m.loc[mc_idx, "category"].to_numpy() == cat]
        if len(rows):
            m.loc[rows, "category"] = _drift(rng, m.loc[rows, "category"], variants)
    counts["category_drift_merchant"] = len(mc_idx)

    write_quality_ground_truth(counts)
    return {**tables, "orders": o, "deliveries": dl, "sessions": s, "merchants": m}


def write_quality_ground_truth(counts: dict) -> None:
    GROUND_TRUTH.mkdir(parents=True, exist_ok=True)
    (GROUND_TRUTH / "quality_defects.json").write_text(json.dumps(counts, indent=2))


# ---------------------------------------------------------------------------
# Silver
# ---------------------------------------------------------------------------

SILVER_SQL = SQL / "silver"
REJECTED = SILVER / "_rejected"
QUALITY_REPORT = SILVER / "_quality_report.parquet"

# Child -> parent relationships that must hold across the FINISHED Silver layer,
# declared once as data.
#
# Quarantining is not a per-table act: removing an order has to remove that
# order's deliveries too, or Silver ends up with child rows pointing at parents
# that no longer exist. Fixing that inside each .sql file would only ever fix
# the relationship somebody happened to notice -- 03_deliveries.sql reads
# order_ts from BRONZE orders, so a delivery whose order was quarantined still
# sailed through. The list below is enforced generically after every per-table
# clean, so the class is closed rather than one instance of it.
#
# Ordered parent-first: a parent is finalised before anything that depends on it
# is checked, so one pass propagates a removal all the way down the chain.
#
# (orders, merchant_id) duplicates what 01_orders.sql's SEMI JOIN already did
# and will always find zero rows. That redundancy is deliberate: one declared
# list covering every relationship beats a special case per file, and a check
# that keeps reporting zero is a live signal on the Data Quality page.
FOREIGN_KEYS = [
    ("orders", "session_id", "sessions", "session_id"),
    ("orders", "merchant_id", "merchants", "merchant_id"),
    ("deliveries", "order_id", "orders", "order_id"),
]


def _drifted(after: pd.DataFrame, before: pd.DataFrame, key: str, col: str) -> int:
    """How many surviving rows hold a different ``col`` than bronze did.

    A measurement, not a re-implementation: it compares the two tables rather
    than restating the normalisation rule that only the .sql file owns.
    """
    bronze_value = before.drop_duplicates(key).set_index(key)[col]
    return int((after[col].to_numpy() != after[key].map(bronze_value).to_numpy()).sum())


def build_silver() -> pd.DataFrame:
    """Run the Silver SQL over Bronze, quarantine what fails, return the report.

    Three outputs, all real files:
      * data/silver/*.parquet              -- the cleaned tables
      * data/silver/_rejected/*.parquet    -- every row refused, with a reason
      * data/silver/_quality_report.parquet -- one row per check

    The quality report is a product surface: the Data Quality page reads this
    file and nothing else, so every number in it is counted from the data here
    and no number is ever typed into the UI by hand. Nothing in this function
    reads data/ground_truth/quality_defects.json -- the tests compare two
    independent measurements, which is only meaningful while they stay
    independent.
    """
    run_ts = pd.Timestamp.now(tz="UTC")
    checks: list[dict] = []

    def check(name, table, rows_in, rows_out, rows_rejected, rule, rows_repaired=0):
        # rows_rejected: rows this check removed from the table.
        # rows_repaired: rows it kept but changed (normalised, imputed, resolved).
        checks.append({
            "check_name": name, "table": table,
            "rows_in": int(rows_in), "rows_out": int(rows_out),
            "rows_rejected": int(rows_rejected), "rows_repaired": int(rows_repaired),
            "rule": rule, "run_ts": run_ts,
        })

    peak_hours = "(" + ", ".join(str(h) for h in PEAK_HOURS) + ")"

    # --- orders ------------------------------------------------------------
    bronze_orders = io.read_bronze("orders")
    orders = run_sql_file(SILVER_SQL / "01_orders.sql", SILVER / "orders.parquet",
                          peak_hours=peak_hours)
    orphans = run_sql_file(SILVER_SQL / "01_orders_rejected.sql",
                           REJECTED / "orders.parquet")
    deduped = len(orders) + len(orphans)

    check("orders_deduplicate", "orders", len(bronze_orders), deduped,
          len(bronze_orders) - deduped,
          "One row per order_id. The first record written for an order wins "
          "(ROW_NUMBER over order_ts, ties broken by position in the file); the "
          "re-sent copies are discarded.")
    check("orders_merchant_fk_integrity", "orders", deduped, len(orders), len(orphans),
          "Every order's merchant_id must exist in the merchant dimension. "
          "Orders with no matching merchant are quarantined to "
          "_rejected/orders.parquet, never silently dropped.")
    check("orders_payment_method_normalised", "orders", len(orders), len(orders), 0,
          "payment_method is lowercased, spaces become underscores and 'cc' maps "
          "to 'credit_card', so casing drift cannot split one payment method "
          "into several categories.",
          rows_repaired=_drifted(orders, bronze_orders, "order_id", "payment_method"))
    check("orders_delivery_fee_imputed", "orders", len(orders), len(orders), 0,
          "A missing delivery_fee is a numeric measure, so it is filled with the "
          "median fee of its zone and flagged in delivery_fee_is_imputed.",
          rows_repaired=orders["delivery_fee_is_imputed"].sum())

    # --- sessions ----------------------------------------------------------
    bronze_sessions = io.read_bronze("sessions")
    sessions = run_sql_file(SILVER_SQL / "02_sessions.sql", SILVER / "sessions.parquet")
    unresolved = run_sql_file(SILVER_SQL / "02_sessions_rejected.sql",
                              REJECTED / "sessions.parquet")

    check("sessions_zone_id_resolution", "sessions", len(bronze_sessions),
          len(sessions), len(unresolved),
          "zone_id is a categorical identifier and is never imputed. A missing "
          "zone is resolved only from the session customer's home zone; a "
          "session that cannot be resolved that way is quarantined to "
          "_rejected/sessions.parquet.",
          rows_repaired=sessions["zone_id_was_resolved"].sum())

    # --- deliveries --------------------------------------------------------
    bronze_deliveries = io.read_bronze("deliveries")
    deliveries = run_sql_file(SILVER_SQL / "03_deliveries.sql",
                              SILVER / "deliveries.parquet")
    bad_ts = run_sql_file(SILVER_SQL / "03_deliveries_rejected.sql",
                          REJECTED / "deliveries.parquet")

    check("deliveries_timestamp_causality", "deliveries", len(bronze_deliveries),
          len(deliveries), len(bad_ts),
          "A delivery cannot be delivered before it was assigned to a courier, "
          "nor before its order was placed. Rows breaking either ordering are "
          "quarantined to _rejected/deliveries.parquet rather than repaired, "
          "because there is no defensible way to tell which timestamp is wrong. "
          "A never-delivered (cancelled) row is not a violation and is kept.")

    # --- dimensions --------------------------------------------------------
    bronze_merchants = io.read_bronze("merchants")
    merchants = run_sql_file(SILVER_SQL / "05_dimensions.sql",
                             SILVER / "merchants.parquet")

    check("merchants_category_normalised", "merchants", len(bronze_merchants),
          len(merchants), 0,
          "merchant category is title-cased and trimmed, so 'pizza', 'Pizza' and "
          "'PIZZA ' collapse to one slice instead of three.",
          rows_repaired=_drifted(merchants, bronze_merchants, "merchant_id", "category"))

    # These arrive clean -- inject_defects() touches only orders, deliveries,
    # sessions and merchants -- but they are copied through anyway so that GOLD
    # READS SILVER AND NOTHING ELSE. A gold query reaching back into bronze for
    # "the clean ones" would be a second, undeclared lineage: the boundary would
    # hold for four tables and quietly not for six, and the day a defect is
    # injected into one of these, gold would keep reading the raw copy.
    for name in ("zones", "customers", "merchant_availability", "promotions",
                 "experiments", "experiment_assignments"):
        io.write_silver(io.read_bronze(name), name)

    # --- referential integrity across the finished layer --------------------
    # Runs last, when every table has been cleaned, so a row removed by any of
    # the checks above takes its dependants with it.
    tables = {"orders": orders, "sessions": sessions,
              "deliveries": deliveries, "merchants": merchants}
    rejects = {"orders": orphans, "sessions": unresolved, "deliveries": bad_ts}

    for child, fk, parent, pk in FOREIGN_KEYS:
        rows_in = len(tables[child])
        dangling = ~tables[child][fk].isin(tables[parent][pk])
        if dangling.any():
            reason = f"orphaned_by_{parent}_quarantine"
            # reindex onto the quarantine's own columns: the child's derived
            # Silver columns are not part of the refused-row record.
            moved = (tables[child][dangling]
                     .assign(reject_reason=reason)
                     .reindex(columns=rejects[child].columns))
            rejects[child] = pd.concat([rejects[child], moved], ignore_index=True)
            tables[child] = tables[child][~dangling]
            io.write_silver(tables[child], child)
            rejects[child].to_parquet(REJECTED / f"{child}.parquet", index=False)

        check(f"{child}_{fk}_references_{parent}", child,
              rows_in, len(tables[child]), rows_in - len(tables[child]),
              f"Every {child}.{fk} must still exist in Silver {parent}.{pk}. A "
              f"row whose parent was quarantined is quarantined too, as "
              f"orphaned_by_{parent}_quarantine, so no cleaned table is left "
              f"pointing at a row that Silver removed.")

    report = pd.DataFrame(checks)
    QUALITY_REPORT.parent.mkdir(parents=True, exist_ok=True)
    report.to_parquet(QUALITY_REPORT, index=False)
    return report


# ---------------------------------------------------------------------------
# Gold
# ---------------------------------------------------------------------------

GOLD_SQL = SQL / "gold"


def build_gold() -> None:
    """Run every Gold .sql over Silver and assert its contract BEFORE writing.

    Driven by the .sql files that actually exist, NOT by iterating
    contracts.GOLD_CONTRACTS. The contract dict declares six datasets; only the
    two slice datasets are built in this phase and the other four arrive later,
    so a loop over the dict would try to run SQL that isn't written yet. The
    file is the source of truth for "does this dataset exist"; the contract is
    the source of truth for "is it shaped right".

    Order matters: SQL -> DataFrame -> assert -> write. Asserting after the
    write would leave a malformed parquet on disk even though the exception
    propagates, and the next reader would pick up that file rather than the
    error. Nothing is persisted until it has passed its contract.

    The date window comes from config (START_DATE/END_DATE) and is injected into
    the SQL, so the window has one definition rather than one per file.
    """
    for path in sorted(GOLD_SQL.glob("*.sql")):
        name = path.stem
        df = run_sql_file(path,
                          start_date=START_DATE.isoformat(),
                          end_date=END_DATE.isoformat())
        assert_contract(df, name)
        io.write_gold(df, name)
