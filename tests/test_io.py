import pandas as pd
from pandas.testing import assert_frame_equal

from pulse.io import write_bronze, read_bronze

_PROBE_NAME = "_io_roundtrip_probe"


def test_bronze_roundtrip_preserves_mixed_dtype_content(tmp_path, monkeypatch):
    monkeypatch.setattr("pulse.io.BRONZE", tmp_path / "bronze")

    df = pd.DataFrame({
        "int_col": [1, 2, 3],
        "float_col": [1.5, 2.25, 3.75],
        "str_col": ["a", "b", "c"],
        "bool_col": [True, False, True],
    })
    write_bronze(df, _PROBE_NAME)
    result = read_bronze(_PROBE_NAME)
    assert_frame_equal(result, df)


def test_write_bronze_creates_missing_directory(tmp_path, monkeypatch):
    target = tmp_path / "bronze"
    monkeypatch.setattr("pulse.io.BRONZE", target)
    assert not target.exists()

    write_bronze(pd.DataFrame({"x": [1, 2, 3]}), _PROBE_NAME)

    assert target.is_dir()
    assert (target / f"{_PROBE_NAME}.parquet").exists()
