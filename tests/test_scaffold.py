import sys

def test_python_version_is_312():
    assert sys.version_info[:2] == (3, 12)

def test_package_imports():
    import pulse
    assert pulse.__version__ == "0.1.0"


def test_every_medallion_layer_is_present_before_any_test_reads_one(data_layers):
    """The suite's own prerequisite, asserted rather than assumed.

    tests/conftest.py builds whichever of bronze/silver/gold is missing at
    import time. This is the check on that logic: if the ensure step ever stops
    covering a layer, this fails HERE with a plain message instead of surfacing
    as a MissingGoldTableError halfway through test_anomaly -- or, worse, as a
    pass that only happened because a previous run left parquet behind.
    """
    from pulse.config import BRONZE, GOLD, GROUND_TRUTH, SILVER

    assert any(BRONZE.glob("*.parquet")), "no bronze; `pulse generate` has not run"
    assert any(SILVER.glob("*.parquet")), "no silver; `pulse build` has not run"
    assert any(GOLD.glob("*.parquet")), "no gold; `pulse build` has not run"
    assert (GROUND_TRUTH / "injected_incidents.json").exists()
    assert set(data_layers) <= {"bronze", "silver", "gold"}
