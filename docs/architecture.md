# Architecture

The shape of PULSE, and why it is that shape. All data is synthetic.

## The constraint that set everything else

One person, one repository, a portfolio timebox, and a dataset of 1.17M rows in
13.4 MB of parquet. Every architectural decision below follows from that: at this
size, infrastructure is cost without benefit, and the interesting engineering is
in the analytical boundaries rather than in the deployment topology.

## Shape: a modular monolith

No microservices, no Kubernetes, no Kafka, no Airflow. `src/pulse/` is a package
of pure modules; `app/` renders them; `tests/` proves them.

The boundaries that do exist are real and enforced by tests rather than by
convention:

| Boundary | Enforced by |
|---|---|
| The engine never reads ground truth | A test greps `src/pulse/` and `app/` for `ground_truth`; another moves the directory away and asserts identical output |
| The CLI computes nothing | A test asserts `cli.py` imports none of `detect_anomalies`, `diagnose`, `estimate_impact`, `prioritize`, `recommend` |
| The CLI and the UI cannot diverge | A test compares CLI-written `decisions.json` against a direct library call at full float precision |
| The LLM owns no facts | A test asserts the memo dict is identical with `NullNarrator` and with a stub narrator |
| The model never sees a row | A build-time assertion forbids `order_id`, `session_id`, `customer_id`, `merchant_id` anywhere in the evidence bundle |

## Execution: one engine, two entry points

```
pulse generate  ->  data/bronze/*.parquet          seeded, once
pulse build     ->  DuckDB over sql/*.sql -> data/silver -> data/gold
                                 |
            +--------------------+---------------------+
     pulse decide                                 streamlit run
   -> artifacts/decisions.json               -> the same pure functions
      artifacts/decision-memo.md                under st.cache_data
```

ETL is materialised once because it is deterministic and slow-ish; decision
intelligence is computed live because it is parameterised and fast. Both call the
same `run_decision_cycle()`.

There is exactly one orchestrator. Divergence between what the CLI writes and
what the app renders is not a discipline the author maintains — it is a
structural impossibility with a test attached.

## The layers

### Bronze — raw, with the defects left in

The generator writes clean data, then a defect injector plants seeded quality
problems on top and records the counts to `data/ground_truth/quality_defects.json`
so the tests can check the Silver build caught exactly what was planted. Bronze
is what a real ingestion lands: duplicates, casing drift, nulls, orphan foreign
keys, impossible timestamps.

### Silver — cleaned, typed, conformed, quarantined

Rejected rows go to `data/silver/_rejected/<table>.parquet` with a
`reject_reason` column. Nothing is silently dropped, because a row that
disappears without a record is indistinguishable from a row that never existed.

Two rules carry the weight:

- **Numeric measures may be imputed; categorical identifiers never are.** A
  missing `delivery_fee` is filled from the zone median with an `is_imputed` flag.
  A null `zone_id` on a session is resolved only from a deterministic unambiguous
  relationship, otherwise the row is quarantined. Imputing an identifier invents
  a fact.
- **Every check appends to `data/silver/_quality_report.parquet`.** That file is
  the Data Quality page's only data source. There is no hand-written quality
  number in the UI.

### Gold — six business-facing datasets

`src/pulse/contracts.py` declares each table's grain, primary key and required
columns; `assert_contract()` runs after every write. A grain violation cannot
ship silently.

The gold SQL is real `.sql` on disk executed by DuckDB against parquet.
`sql_runner.py` is deliberately small: read file, bind `{bronze}` / `{silver}` /
`{gold}` path parameters, execute, `COPY (…) TO '…' (FORMAT PARQUET)`.

**The calendar spine is load-bearing, not a formatting nicety.** A plain
`GROUP BY order_date, zone_id` emits no row for a zone-day with zero completed
orders, so the worse a zone becomes the more of its bad days vanish — the
detector would see a *shorter* series rather than a *worse* one, and the incident
would partially hide itself. A `generate_series` spine crossed with the zone
dimension forces a zero row to exist. Same class of problem as the
`merchant_availability` snapshot table: in both cases the signal is an absence,
and absences must be manufactured into rows before analysis can see them.

## The decision engine

```
compute_metrics  ->  detect_anomalies  ->  diagnose  ->  estimate_impact
                 ->  prioritize        ->  recommend  ->  build_memo
```

Every function is pure — DataFrames in, frozen dataclasses out. I/O exists at
exactly two edges: `io.py` reads parquet, `cli.py` writes artifacts. The single
exception is `prioritization.distinct_customers()`, which reads silver because a
windowed distinct count is something a long-format metric frame cannot express:
days cannot be added up into customers. It is documented as such and cached.

`MetricFrame` is long-format `(metric, scope, scope_value, metric_date, value)`
and pivots a slice on demand. Adding a metric is a data change to
`METRIC_REGISTER`, not a new code path. Adding a *dimension* is a row in
`_SCOPE_COLUMN` — and if one is missed, `UnmappedScopeError` is raised rather
than a silent zero, because a scope whose customer footprint cannot be counted
must not be given a footprint of zero and quietly ranked below everything that
can be counted.

## The LLM boundary

The engine decides; the LLM narrates. The model receives an `EvidenceBundle` —
bounded JSON, capped at 30,000 characters, never a DataFrame, never a row — and
may write prose about it. Three enforcement layers, described in full in the
README: structural (the answer's non-prose fields come off the bundle, not the
model), mechanical (each figure in the answer is checked, by pattern, for a
structured fact matching its concept, scope, unit, meaning, direction, window and
change-vs-level — dates excepted, which are only checked as dates the bundle
carries — and causal wording is flagged; benchmarked, not proven complete, and it
detects rather than certifies), and
offline-first (no key means a deterministic template, labelled as such).

The user's question is untrusted input and never enters an instruction position.
On the offline path it is never interpreted at all — the offline answer is a pure
function of the bundle, so no question can move a number in it even in principle.

## The Spark layer

`databricks/silver_to_gold_spark.py` reproduces two gold tables in genuine
PySpark with explicit `StructType` schema enforcement and Delta writes. **It has
never been run**: no JDK here, no Databricks workspace. Its reconciliation test
skips, and `databricks/README.md` is the manual runbook for anyone who wants to
execute it. It exists as an evidence path, not as a pipeline that has run.

## The UI

`st.navigation` multipage over one entry point. Every page calls
`run_decision_cycle()` under `st.cache_data` and recomputes nothing. Theme lives
in `.streamlit/config.toml` plus one injected CSS block in `app/theme.py`; chart
styling lives in a registered Plotly template, so there is no per-chart
formatting anywhere.

Human-in-the-loop states (`INVESTIGATE` / `APPROVED` / `REJECTED`) live in
`st.session_state` with a timestamped audit list. They update local session state
and execute nothing — PULSE writes files and renders pages, and calls no
marketplace system. The constraint is stated in the UI on every page.

## What is deliberately absent

- No authentication, no multi-tenancy, no user model.
- No scheduler — the pipeline is three CLI commands.
- No machine learning. Anomaly detection is a day-of-week-adjusted z-scan; the
  diagnosis is a decomposition of an exact identity plus Pearson correlation.
- No `MERGE`, `OPTIMIZE` or `Z-ORDER` in the Spark script: the build is a full
  overwrite of a batch table with no incremental use case, so they would be
  unused ceremony.
- No Power BI artifact. The gold datasets are BI-ready by grain and key, but
  nothing was built on top of them.
