import json
import numpy as np
import pandas as pd
import pytest

from pulse.data_generator import generate_all, make_rng
from pulse.io import read_bronze
from pulse.quality import inject_defects, build_silver, FOREIGN_KEYS
from pulse import io
from pulse.config import GROUND_TRUTH, SILVER


def test_defects_are_injected_at_the_specified_rates():
    clean = generate_all()
    dirty = inject_defects(clean, make_rng())

    n_clean = len(clean["orders"])
    n_dirty = len(dirty["orders"])
    dup_rate = (n_dirty - n_clean) / n_clean
    assert 0.006 <= dup_rate <= 0.010                        # 0.8% duplicates

    assert dirty["orders"]["delivery_fee"].isna().mean() > 0.010   # 1.5% nulls
    assert dirty["sessions"]["zone_id"].isna().mean() > 0.003      # 0.5% nulls

    pm = set(dirty["orders"]["payment_method"].dropna().unique())
    assert len(pm) > 4, "category drift must be present in bronze"

    counts = json.loads((GROUND_TRUTH / "quality_defects.json").read_text())
    o, dl, s, m = dirty["orders"], dirty["deliveries"], dirty["sessions"], dirty["merchants"]

    # Every recorded count must equal the observable signature of that exact
    # defect in the returned tables -- a recorded count that drifted away from
    # what was actually mutated must fail here, not slip through on a `> 0`
    # proxy that two overlapping defects could still satisfy vacuously.

    assert counts["duplicate_orders"] > 0
    assert counts["duplicate_orders"] == len(o) - o["order_id"].nunique()

    assert counts["null_delivery_fee"] > 0
    assert counts["null_delivery_fee"] == o["delivery_fee"].isna().sum()

    assert counts["orphan_merchant_fk"] > 0
    assert counts["orphan_merchant_fk"] == (o["merchant_id"] == 999_999).sum()

    assert counts["null_session_zone_id"] > 0
    assert counts["null_session_zone_id"] == s["zone_id"].isna().sum()

    assert counts["timestamp_violations"] > 0
    delivered = dl["delivered_ts"].notna()
    assert counts["timestamp_violations"] == (
        dl.loc[delivered, "delivered_ts"] < dl.loc[delivered, "assigned_ts"]
    ).sum()

    # Drift (quality._drift) always redraws away from a row's own current
    # value, so "differs from the clean baseline for that same id" is an
    # exact signature -- robust even though a drifted string can coincide
    # with a DIFFERENT row's legitimate original value (e.g. a drifted
    # "credit_card" is also a real payment_method for some untouched order).
    baseline_payment = clean["orders"].set_index("order_id")["payment_method"]
    assert counts["category_drift_payment"] > 0
    assert counts["category_drift_payment"] == int((
        o["payment_method"].to_numpy() != o["order_id"].map(baseline_payment).to_numpy()
    ).sum())

    baseline_category = clean["merchants"].set_index("merchant_id")["category"]
    assert counts["category_drift_merchant"] > 0
    assert counts["category_drift_merchant"] == int((
        m["category"].to_numpy() != m["merchant_id"].map(baseline_category).to_numpy()
    ).sum())

    # Regression test for the bronze wiring gap: generate_all() must persist
    # the DIRTY tables to parquet, not the clean ones it returns -- otherwise
    # Silver would read pristine bronze with nothing in it to clean.
    bronze_orders = read_bronze("orders")
    assert bronze_orders["delivery_fee"].isna().sum() > 0
    assert (bronze_orders["merchant_id"] == 999_999).sum() > 0
    assert len(bronze_orders) > len(clean["orders"])          # duplicates landed too


# --------------------------------------------------------------------------
# Silver
# --------------------------------------------------------------------------

REJECTED = SILVER / "_rejected"

# The canonical vocabularies are the ones data_generator actually draws from;
# bronze then drifts their casing/spelling. Silver must fold the drift back
# onto exactly these values and invent nothing new.
CANONICAL_PAYMENT = {"credit_card", "debit_card", "pix", "wallet"}
CANONICAL_CATEGORY = {"Pizza", "Burger", "Japanese", "Brazilian", "Healthy", "Dessert"}


@pytest.fixture(scope="session")
def report() -> pd.DataFrame:
    """Build Silver once for the whole session and hand back the quality report."""
    return build_silver()


def _rejected(table: str) -> pd.DataFrame:
    return pd.read_parquet(REJECTED / f"{table}.parquet")


def test_silver_removes_duplicates_and_quarantines_bad_rows(report):
    orders = io.read_silver("orders")

    assert orders["order_id"].is_unique
    assert orders["merchant_id"].max() < 999_999          # orphans gone
    assert not report.empty
    assert {"check_name", "table", "rows_in", "rows_out",
            "rows_rejected", "rule", "run_ts"}.issubset(report.columns)
    assert report["rows_rejected"].sum() > 0
    # The Data Quality page renders `rule` verbatim, so it must be prose.
    assert report["rule"].str.len().min() > 20


def test_orphan_fk_rows_are_quarantined_not_dropped(report):
    """An unmatched FK is evidence, not noise: it lands in the quarantine."""
    rejected = _rejected("orders")
    orphans = rejected[rejected["reject_reason"] == "orphan_merchant_fk"]
    assert len(orphans) > 0
    assert (orphans["merchant_id"] == 999_999).all()

    check = report[report["check_name"] == "orders_merchant_fk_integrity"].iloc[0]
    assert check["rows_rejected"] == len(orphans)
    assert check["rows_in"] == check["rows_out"] + check["rows_rejected"]


def test_timestamp_violations_are_quarantined(report):
    """delivered_ts before assigned_ts breaks causality -- quarantine, not clip."""
    rejected = _rejected("deliveries")
    bad = rejected[rejected["reject_reason"] == "timestamp_violation"]
    assert len(bad) > 0

    deliveries = io.read_silver("deliveries")
    delivered = deliveries["delivered_ts"].notna()
    assert (deliveries.loc[delivered, "delivered_ts"]
            >= deliveries.loc[delivered, "assigned_ts"]).all()

    assert {"delivery_delay_minutes", "is_on_time"}.issubset(deliveries.columns)
    on_time = deliveries["actual_delivery_minutes"] <= deliveries["promised_eta_minutes"] + 5
    assert (deliveries["is_on_time"].fillna(False) == on_time.fillna(False)).all()


def test_declared_foreign_keys_hold_in_silver(report):
    """Quarantining cascades: no cleaned table may point at a row Silver removed.

    Asserted for EVERY declared relationship, not just the one that happens to
    have dangling rows today -- the orders->sessions edge is zero-row under seed
    42 and would hide a regression if only the deliveries edge were tested.
    """
    for child, fk, parent, pk in FOREIGN_KEYS:
        c, p = io.read_silver(child), io.read_silver(parent)
        dangling = set(c[fk]) - set(p[pk])
        assert not dangling, f"{child}.{fk} -> {parent}.{pk}: {len(dangling)} dangling"

        check = report[report["check_name"] == f"{child}_{fk}_references_{parent}"]
        assert len(check) == 1, "every declared relationship reports one check row"
        row = check.iloc[0]
        assert row["rows_in"] == row["rows_out"] + row["rows_rejected"]
        assert row["rows_out"] == len(c)


def test_cascaded_rows_land_in_the_child_quarantine(report):
    """A delivery whose order was quarantined follows its order out, with a
    reason that names why it went -- not silently retained, not silently lost."""
    rejected = _rejected("deliveries")
    cascaded = rejected[rejected["reject_reason"] == "orphaned_by_orders_quarantine"]
    assert len(cascaded) > 0

    check = report[report["check_name"] == "deliveries_order_id_references_orders"].iloc[0]
    assert check["rows_rejected"] == len(cascaded)
    # the quarantined orders are genuinely gone from silver, not just unjoined
    assert not set(cascaded["order_id"]) & set(io.read_silver("orders")["order_id"])


def test_numeric_nulls_imputed_with_a_flag_categoricals_never_imputed(report):
    orders = io.read_silver("orders")
    sessions = io.read_silver("sessions")

    # numeric measure: imputed from the zone median, and flagged as imputed
    assert orders["delivery_fee"].isna().sum() == 0
    assert "delivery_fee_is_imputed" in orders.columns
    assert orders["delivery_fee_is_imputed"].sum() > 0

    # categorical identifier: NEVER imputed. Resolved only from the customer's
    # home zone, otherwise quarantined.
    assert sessions["zone_id"].isna().sum() == 0
    assert "zone_id_is_imputed" not in sessions.columns

    # Quarantine holds only genuinely unresolvable rows. Under the current
    # generator every session has a customer and every customer has a home
    # zone, so this file is legitimately empty -- what must hold is that
    # nothing else leaked into it and that no row went missing.
    rejected = _rejected("sessions")
    assert set(rejected["reject_reason"].unique()) <= {"null_zone_id_unresolvable"}

    check = report[report["check_name"] == "sessions_zone_id_resolution"].iloc[0]
    assert check["rows_in"] == check["rows_out"] + check["rows_rejected"]
    assert check["rows_rejected"] == len(rejected)
    assert check["rows_repaired"] > 0            # the 0.5% nulls were resolved


def test_categories_are_normalised(report):
    orders = io.read_silver("orders")
    merchants = io.read_silver("merchants")

    assert set(orders["payment_method"].unique()) <= CANONICAL_PAYMENT
    assert set(merchants["category"].unique()) <= CANONICAL_CATEGORY


def test_derived_columns_follow_the_locked_metric_semantics(report):
    """Checked across every row: a formula that is right on row 0 and wrong on
    row 40,000 is exactly the bug this has to catch."""
    orders = io.read_silver("orders")

    assert np.allclose(orders["contribution_margin"],
                       orders["item_amount"] * orders["commission_rate"]
                       + orders["delivery_fee"] - orders["delivery_cost"]
                       - orders["discount_amount"])
    assert np.allclose(orders["customer_gross_value"],
                       orders["item_amount"] + orders["delivery_fee"])
    assert np.allclose(orders["net_revenue"],
                       orders["customer_gross_value"] - orders["discount_amount"])

    assert {"order_date", "order_hour", "is_peak", "is_weekend",
            "is_completed", "is_cancelled"}.issubset(orders.columns)
    assert (orders["is_peak"] == orders["order_hour"].isin([11, 12, 13, 18, 19, 20])).all()
    assert (orders["is_completed"] == (orders["status"] == "completed")).all()
    assert (orders["is_cancelled"] == (orders["status"] == "cancelled")).all()
    assert (orders["is_weekend"]
            == pd.to_datetime(orders["order_ts"]).dt.dayofweek.isin([5, 6])).all()
    assert (orders["order_date"]
            == pd.to_datetime(orders["order_ts"]).dt.date).all()


def test_quality_report_accounts_for_every_row(report):
    """The Data Quality page reads this file and nothing else, so the numbers
    in it must reconcile on their own."""
    saved = pd.read_parquet(SILVER / "_quality_report.parquet")
    assert len(saved) == len(report)
    assert report["run_ts"].nunique() == 1

    dedupe = report[report["check_name"] == "orders_deduplicate"].iloc[0]
    assert dedupe["rows_in"] == len(read_bronze("orders"))
    assert dedupe["rows_rejected"] > 0

    # Per table the checks form an unbroken chain: each one accounts for the
    # rows it removed and hands the survivors to the next, and the last one
    # lands exactly on what is on disk.
    for table, table_checks in report.groupby("table", sort=False):
        n = table_checks[["rows_in", "rows_out", "rows_rejected"]].to_numpy()
        assert (n[:, 0] == n[:, 1] + n[:, 2]).all(), f"{table}: check does not balance"
        assert (n[1:, 0] == n[:-1, 1]).all(), f"{table}: chain broken between checks"
        assert n[-1, 1] == len(io.read_silver(table)), f"{table}: report != disk"


def test_drift_refuses_a_variant_list_it_could_never_satisfy():
    """_drift() rejection-samples until every redrawn value differs from the one
    already there. With fewer than two distinct variants that condition is
    unsatisfiable and the `while collide.any()` loop spins forever -- silently,
    with no output, in the middle of a 560k-row generation. A hang is far worse
    to diagnose than an assertion, so it asserts.
    """
    from pulse.quality import _drift

    with pytest.raises(ValueError):
        _drift(make_rng(), pd.Series(["pizza", "pizza"]), ["pizza"])
    with pytest.raises(ValueError):
        _drift(make_rng(), pd.Series(["pizza"]), ["pizza", "pizza"])

    # Two distinct variants: satisfiable, and it really does redraw away.
    out = _drift(make_rng(), pd.Series(["pizza", "pizza"]), ["pizza", "PIZZA "])
    assert (out == "PIZZA ").all()
