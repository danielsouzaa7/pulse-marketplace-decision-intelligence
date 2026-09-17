"""Execute a .sql file against the lake with DuckDB.

The transformation lives in the .sql file; this module is only plumbing, so
that the Silver/Gold logic stays reviewable as SQL rather than hiding inside
Python string building.

Layer roots ({bronze}, {silver}, {gold}) are substituted for every file, and
extra ``params`` let a locked constant (config.PEAK_HOURS, say) be injected
instead of retyped into each .sql file where it would silently drift.

SUBSTITUTION IS TARGETED, NOT str.format().
----------------------------------------------------------------------------
This used to be ``text.format(**params)``, which walks the WHOLE file --
comments included -- and raises KeyError on any brace-wrapped word it does not
recognise. A .sql file is mostly prose, and prose mentions placeholders: Task 8
lost time to a comment that said `{peak_hours}`, and every SQL file written
since has had to tiptoe around its own braces. The failure is also silent in
the worst way, because `{{` is format's escape and nobody writing SQL knows it.

So only the parameter names we were actually given are replaced, by name. An
unknown `{whatever}` in a comment is now just text. If a real placeholder is
misspelled it survives into the SQL and DuckDB raises a parser error naming the
line -- still loud, just no longer the *comments* raising it.

Values are layer paths and ISO dates supplied by config, substituted into a
fixed template; nothing here concatenates a caller-supplied value into a SQL
expression.
"""

from pathlib import Path

import duckdb
import pandas as pd

from pulse.config import BRONZE, SILVER, GOLD


def render_sql(path: Path, **params) -> str:
    """The text of ``path`` with {bronze}/{silver}/{gold} and ``params``
    substituted. Separate from run_sql_file so the substitution rule is
    testable without a database."""
    sql = path.read_text(encoding="utf-8")
    # Read the layer roots at call time: tests monkeypatch pulse.sql_runner.SILVER
    # to point a gold query at a synthetic fixture lake.
    for name, value in {"bronze": BRONZE.as_posix(), "silver": SILVER.as_posix(),
                        "gold": GOLD.as_posix(), **params}.items():
        sql = sql.replace("{" + name + "}", str(value))
    return sql


def run_sql_file(path: Path, out: Path | None = None, **params) -> pd.DataFrame:
    """Run ``path`` and return the result; write it to ``out`` when given."""
    with duckdb.connect() as con:
        df = con.sql(render_sql(path, **params)).df()
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
    return df
