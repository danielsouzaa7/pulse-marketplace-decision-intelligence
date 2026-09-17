# Prerequisites. Every layer this suite reads is established HERE, once, before
# any test module is imported -- so that `pytest` on a fresh clone with an empty
# data/ passes instead of failing on parquet nobody built.
#
# Why import time and not a session fixture. A session fixture runs after
# collection, and collection is already too late: tests/test_copilot.py binds
#
#     RESULT = run_decision_cycle(load_gold(), P)
#
# at MODULE scope, so importing that file touches gold. pytest loads conftest.py
# before it imports any test module in the directory, which makes this the only
# hook early enough. The session fixture below is the same guarantee with a name
# a test can ask for explicitly.
#
# What the absence of this file cost, measured. With data/ moved aside the suite
# did not fail cleanly -- it PARTIALLY HEALED ITSELF, in alphabetical file order,
# and reported a different set of failures on each consecutive run:
#
#   run 1  collection of test_copilot.py raises MissingGoldTableError
#          -> the whole session aborts, 0 tests run
#   run 2  (--continue-on-collection-errors) test_contracts.py's own `gold`
#          fixture calls build_gold(), which raises because no silver exists;
#          later files build bronze (test_generator's generate_all fixture) and
#          silver as a side effect of their own fixtures. 28 failures/errors
#          spread over test_anomaly, test_app, test_contracts, test_engine,
#          test_experiments, test_memo, test_prioritization, test_root_cause.
#   run 3  bronze and silver now exist from run 2, so test_contracts' build_gold
#          SUCCEEDS mid-suite and every module after "test_c..." passes. Only
#          the modules that run before it still fail: test_anomaly (8),
#          test_app (12), plus the test_copilot collection error.
#
# That is the failure mode this file exists to remove: a green suite that is
# green because of what a previous run left on disk. The build order below is
# the same one the CLI documents -- pulse generate -> pulse build -- because
# each layer genuinely reads the one under it.
#
# Only MISSING layers are built. The generator is ~1.17M rows and minutes of
# work, and it is deterministic from config.SEED, so rebuilding parquet that is
# already correct would cost every run and buy nothing.
from __future__ import annotations

import pytest

from pulse.config import BRONZE, GOLD, GROUND_TRUTH, SILVER


def _ensure_data_layers() -> list[str]:
    """Build any medallion layer that is absent. Returns the ones it built."""
    built: list[str] = []

    # generate_all() writes bronze parquet AND data/ground_truth/
    # injected_incidents.json; test_engine's ground-truth-removal test asserts
    # that directory is present as its premise, so both are the bronze check.
    if (not any(BRONZE.glob("*.parquet"))
            or not (GROUND_TRUTH / "injected_incidents.json").exists()):
        from pulse.data_generator import generate_all

        generate_all()
        built.append("bronze")

    if not any(SILVER.glob("*.parquet")):
        from pulse.quality import build_silver

        build_silver()
        built.append("silver")

    if not any(GOLD.glob("*.parquet")):
        from pulse.quality import build_gold

        build_gold()
        built.append("gold")

    return built


LAYERS_BUILT_AT_STARTUP = _ensure_data_layers()


@pytest.fixture(scope="session")
def data_layers() -> list[str]:
    """The medallion layers are on disk. Depend on this to say so out loud.

    The work already happened at conftest import (see the note at the top of
    this file); asking for this fixture documents the dependency and is free.
    """
    return LAYERS_BUILT_AT_STARTUP
