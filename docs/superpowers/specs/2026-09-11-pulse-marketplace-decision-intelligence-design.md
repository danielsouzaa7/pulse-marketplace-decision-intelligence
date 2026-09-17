# PULSE — Marketplace Decision Intelligence
## Design Specification

**Date:** 2026-09-11
**Status:** Approved for implementation planning

---

## 1. What PULSE is

PULSE is a decision-intelligence platform for marketplaces. It detects important business changes, investigates their associated drivers, estimates impact, prioritizes problems, recommends actions, and helps validate whether those actions worked.

Core loop:

```
WHAT HAPPENED? -> WHY? -> HOW MUCH DOES IT MATTER? -> WHAT SHOULD WE DO? -> DID IT WORK?
```

It is not a delivery app, not a dashboard, not a chatbot, and not a frontend portfolio piece. **ALL DATA IS SYNTHETIC.** The project is informed by prior experience with digital operations and marketplaces but contains no proprietary company data.

### Non-negotiable principle

Facts are calculated deterministically first. The LLM never inspects raw rows.

```
DATA -> QUALITY/TRANSFORM -> METRICS -> ANOMALY -> DIAGNOSIS
     -> IMPACT/PRIORITY -> DECISION MEMO -> EXPERIMENT -> LEARNING
```

The AI layer sits on top of structured calculated evidence. AI may explain, synthesize, prioritize, communicate and phrase recommendations. AI may **not** fabricate KPIs, silently modify facts, claim causality without evidence, or execute marketplace actions.

---

## 2. Architecture decisions

| Decision | Choice | Rationale |
|---|---|---|
| Shape | Modular monolith | No microservices/K8s/Kafka/Airflow. Portfolio scope. |
| Local engine | DuckDB + pandas over parquet | Real `.sql` files, zero setup, sub-second over 1.2M rows. |
| Spark evidence | Databricks Free Edition notebook | No JDK locally; Windows Spark setup is a time sink at 35 MB of data. Spark exists here as *evidence*, executed for real, not as a local dependency. |
| Execution model | **Approach C** | ETL materialized once via CLI; decision intelligence computed live and parameterized. |
| LLM boundary | Engine decides, LLM narrates | Memo works offline; recommendation policy is deterministic. |
| UI | Streamlit + Plotly | Single deployable surface, no separate API, no React. |
| Auth / multi-user | None | Out of scope by design. |

### Approach C — one engine, two entry points

```
pulse generate  ->  data/bronze/*.parquet              (seeded, once)
pulse build     ->  DuckDB sql/*.sql -> silver -> gold (once)
                                 |
            +--------------------+---------------------+
     pulse decide                                 streamlit run
   -> artifacts/decisions.json               -> same pure functions
      artifacts/decision-memo.md                under st.cache_data
      (pytest, README evidence)                 (live, parameterized)
```

**ONE ENGINE ONLY.** There is exactly one orchestrator, `run_decision_cycle()`. CLI and Streamlit both import and call it. No parallel implementation of metrics, detection, diagnosis, impact, prioritization, playbook, or memo generation may exist. Divergence is structurally impossible.

---

## 3. Data model and grains

Fixed window **2026-03-15 to 2026-09-10** (180 days). Dates are hard-coded, never relative to `today()`, so the dataset is reproducible indefinitely. The app's "as of" date is 2026-09-10.

| Table | Grain — one row per… | Primary key | Rows |
|---|---|---|---|
| `zones` | zone | `zone_id` | 8 |
| `customers` | customer | `customer_id` | 25,000 |
| `merchants` | merchant | `merchant_id` | 150 |
| `sessions` | marketplace session / visit | `session_id` | ~560,000 |
| `orders` | order | `order_id` | 100,000 |
| `deliveries` | order that entered dispatch | `delivery_id` (FK `order_id`, 1:1) | ~96,000 |
| `promotions` | promotion definition | `promotion_id` | 10 |
| `experiments` | experiment **definition** | `experiment_id` | 1 |
| `experiment_assignments` | experiment x customer | (`experiment_id`, `customer_id`) | ~12,000 |
| `merchant_availability` | merchant x hour (operating window) | (`merchant_id`, `snapshot_hour`) | 378,000 |

Total approximately 1.18M rows, approximately 35 MB parquet.

### Why `sessions` exists

`conversion` is a mandatory KPI and appears in the Executive Overview. Orders supply only the numerator. Without a visit grain, conversion is uncomputable and would have to be hard-coded — which the project forbids. ~560k sessions produce 100k orders = **17.9% conversion**, realistic for a high-intent logged-in delivery app and the smallest defensible option (~3,100/day, ~390 per zone-day — statistically stable at daily zone grain).

### Why `merchant_availability` exists

Merchant offline time is an **absence**. A closed merchant produces no orders, so the fact cannot be inferred from the order table. This requires a *periodic snapshot fact table* (as distinct from a transaction fact table): one row per merchant per hour, emitted regardless of activity.

Snapshots cover only the **14-hour operating window (10:00-23:59)**: `150 x 14 x 180 = 378,000`. Recording a restaurant as closed at 04:00 is noise. Availability KPI = `available_hours / scheduled_open_hours`.

### Why `experiments` is split

`experiments` holds the definition and hypothesis (experiment grain); `experiment_assignments` holds variant assignment at customer grain. Assignments are modelled explicitly rather than hidden inside an ambiguous single table.

### Customer attributes

`customers` carries `acquisition_channel` (organic / paid_social / referral / paid_search) and `signup_ts` as **dimension attributes**. Value tier and at-risk status are *derived in gold*, never generated — otherwise retention analysis is circular.

---

## 4. Metric semantics (locked)

Definitions are fixed here and used identically everywhere. No ambiguity.

```
item_amount            merchandise value of the order
delivery_fee           fee charged to the customer
delivery_cost          actual cost to fulfil (generated per order, NOT a constant)
discount_amount        marketplace-funded discount
commission_rate        merchant take rate

GMV                    = item_amount                       (completed orders only)
customer_gross_value   = item_amount + delivery_fee
net_revenue            = customer_gross_value - discount_amount
contribution_margin    = item_amount * commission_rate
                         + delivery_fee
                         - delivery_cost
                         - discount_amount
```

`delivery_cost` **must** be a generated per-order column. If it were a global constant, contribution margin would be a scalar multiple of GMV and Event #3 (promotion lifts orders while destroying margin) would be mathematically undetectable.

### Funnel identity

```
GMV = sessions x order_conversion x completion_rate x AOV

order_conversion  = orders_placed    / sessions          (pre-checkout)
completion_rate   = orders_completed / orders_placed     (post-checkout)
AOV               = GMV              / orders_completed
```

Taking logs makes percentage change additive:

```
dln(GMV) ~= dln(sessions) + dln(order_conversion) + dln(completion_rate) + dln(AOV)
```

The largest-magnitude term is the **broken funnel stage** — computed, not asserted. This is the engine's primary diagnostic discriminator.

### KPI register

| KPI | Definition |
|---|---|
| GMV | `sum(item_amount)` where status = completed |
| Orders placed | count of placed orders |
| Completed orders | count where status = completed |
| AOV | GMV / completed orders |
| Order conversion | orders placed / sessions |
| Completion rate | completed / placed |
| Cancellation rate | cancelled / placed |
| Avg promised ETA | mean `promised_eta_minutes` |
| Avg actual delivery time | mean `actual_delivery_minutes` |
| On-time / SLA | share with `actual <= promised + 5 min` |
| Active customers | distinct customers ordering in window |
| Repeat rate | customers with >=2 orders in window / active |
| Retention (D30) | cohort share ordering again within 30 days |
| Merchant availability | available hours / scheduled open hours |
| Promotional cost | `sum(discount_amount)` |
| Promotional ROI | incremental contribution margin / promotional cost |

---

## 5. Incident generation model

**Layered effect multipliers.** A clean stochastic base process generates baseline behaviour (weekly seasonality, payday lift, hour-of-day curves). Each incident is a pure function returning multipliers on *rates*, never on outputs.

```python
def zone7_degradation(ctx) -> Effects:
    if ctx.day < 150 or ctx.zone_id != 7:
        return Effects.none()
    ramp = min(1.0, (ctx.day - 149) / 3)
    return Effects(
        actual_delivery_mult = 1 + 0.35 * ramp,   # fulfilment slows
        promised_eta_mult    = 1 + 0.12 * ramp,   # partial adaptation -> SLA breaks
        cancel_mult          = 1 + 2.25 * ramp,   # 8% -> 26%
    )
    # NOTE: no conversion effect - Zone 7 is a POST-CHECKOUT failure.

INCIDENTS = [zone7_degradation, zone4_availability,
             promo_margin_erosion, paid_social_retention]
```

Seeded with `numpy.random.Generator(PCG64(42))`.

### Zone sizing and incident calibration

Zones are deliberately **unequal in size**, as real marketplaces are. **Zone 7 carries approximately 18% of baseline GMV** (the largest zone). This is not cosmetic: it is what allows a zone-local incident to surface as a company-level anomaly.

The arithmetic must close, because the funnel identity constrains it. With order placement stable:

```
completion rate    0.92 -> 0.74           = -19.6%
zone 7 GMV         -19.6%                 (AOV stable)
company GMV        -19.6% x 18%           = -3.5%
```

`-3.5%` clears the engine's own `3%` deviation floor, so the company-level GMV anomaly fires and the vertical slice has a starting signal. Had Zone 7 been an average-sized zone with a milder incident, the company anomaly would never trigger and the flagship path would be undetectable.

**Multipliers are calibration parameters, not fixed truths.** The generator is tuned until the following observable targets hold, and tests are the gate:

| Calibration target | Asserted in |
|---|---|
| Company GMV deviation <= -3.0% in the comparison window | `test_anomaly.py` |
| Zone 7 is rank-1 segment contributor, contribution >= 55% | `test_root_cause.py` |
| Zone 7 `funnel_break_stage == "completion_rate"` | `test_root_cause.py` |
| Zone 4 `funnel_break_stage == "order_conversion"` | `test_root_cause.py` |
| Clean baseline window (days 1-149) produces zero anomalies | `test_anomaly.py` |

Concurrent incidents partially offset each other — `FREESHIP_WINTER` lifts orders while Zone 7 suppresses them — so final multipliers are set empirically against these assertions rather than derived on paper.

### Reproducibility claim

Same seed produces **logically identical datasets** — identical record content, row counts, keys and values. Tests assert content hashes of sorted DataFrames, **not** binary file identity, because parquet metadata and library versions may alter file bytes without altering data.

### Incident calendar

| Days | Dates | Event |
|---|---|---|
| 1-140 | 03-15 to 08-01 | **Clean baseline.** Seasonality + noise only. |
| 141-149 | 08-02 to 08-10 | **Clean holdout** / recent reference. Still clean. |
| 150+ | 08-11 to 09-10 | **#1 Zone 7 operational degradation.** Actual delivery time +35%, promised ETA +12%, on-time rate collapses, cancellation 8% to 26%, completion rate 92% to 74% (-19.6%), zone GMV -19.6%. 3-day ramp, persists. Sessions and order_conversion **stable**. |
| 158+ | 08-19 to 09-10 | **#2 Zone 4 peak merchant availability.** ~30% of zone-4 merchants unavailable 18:00-21:00. Sessions stable, order_conversion falls. |
| 152-172 | 08-13 to 09-02 | **#3 `FREESHIP_WINTER`.** Marketplace-funded free delivery, no basket minimum. Orders +14%, contribution margin negative. |
| signups 120+ | — | **#4 `paid_social` retention decay.** D30 repeat 42% to 24% for recent paid-social cohorts. |
| 155-180 | 08-16 to 09-10 | **EXP-001** at-risk reactivation incentive. |

Events deliberately overlap in the recent window so prioritization ranks four real competitors rather than one obvious answer.

### Zone 7 vs Zone 4 — the analytic distinction

| | Zone 7 | Zone 4 |
|---|---|---|
| sessions | stable | stable |
| order_conversion | **stable** | **falls** |
| completion_rate | **falls** | stable |
| Broken funnel stage | `completion_rate` | `order_conversion` |
| Top associated driver | `avg_actual_delivery_minutes` | `merchant_availability` |
| Failure locus | post-checkout (fulfilment) | pre-checkout (supply) |

Identical headline symptom — "zone GMV is down" — mechanically distinguishable root cause. This is the design's central proof that the system diagnoses rather than drills down.

### Ground truth isolation

```
data/ground_truth/injected_incidents.json    <- written by generator
data/ground_truth/quality_defects.json       <- written by defect injector
```

Deliberately outside `bronze/silver/gold`. Only `tests/` may read them, to assert the engine independently *recovered* the injected pattern. A test greps `src/pulse/` and `app/` and fails if either references `ground_truth`. The app must never display the generator's answer as analysis.

---

## 6. Bronze / Silver / Gold contract

### BRONZE — raw-ish

Generator output plus seeded quality defects injected *after* clean generation:

| Defect | Rate |
|---|---|
| Exact duplicate order rows | 0.8% |
| Null `delivery_fee` | 1.5% |
| Null `zone_id` on sessions | 0.5% |
| Category drift (`credit_card` / `CREDIT_CARD` / `Credit Card` / `cc`; `Pizza` / `pizza` / `PIZZA `) | ~12% of rows |
| `delivered_ts < order_ts` | 0.3% |
| Orphan `merchant_id` FK | 0.2% |

Defect counts are recorded to `data/ground_truth/quality_defects.json` for tests.

### SILVER — cleaned, typed, conformed

- Deduplication on natural key via `ROW_NUMBER() ... QUALIFY`
- Category normalization through an explicit mapping table
- Typed schema enforcement
- **Null treatment rules:**
  - Numeric measures (e.g. `delivery_fee`) may be imputed from the zone median. Every imputed value sets an `is_imputed` flag column.
  - **Categorical identifiers are never imputed.** A null `zone_id` on a session is resolved only from a deterministic valid relationship (customer home zone where the relationship is unambiguous); otherwise the row is quarantined.
- Timestamp validation: `delivered_ts >= order_ts`, violations quarantined
- FK integrity: orphans quarantined
- Derived: `order_date`, `order_hour`, `is_peak`, `is_weekend`, `is_completed`, `is_cancelled`, `delivery_delay_minutes`, `is_on_time`, `customer_gross_value`, `net_revenue`, `contribution_margin`

Rejected rows are **quarantined, not dropped**: `data/silver/_rejected/<table>.parquet` with a `reject_reason` column.

Every step appends to `data/silver/_quality_report.parquet` (`check_name, table, rows_in, rows_out, rows_rejected, rule, run_ts`). This file *is* the Data Quality page's data source. No hand-written quality numbers anywhere in the UI.

### GOLD — business-facing

| Dataset | Grain |
|---|---|
| `gold_daily_business_metrics` | date |
| `gold_zone_performance` | date x zone |
| `gold_merchant_performance` | date x merchant |
| `gold_customer_retention` | cohort_month x acquisition_channel x period_index |
| `gold_promotion_performance` | date x promotion |
| `gold_experiment_results` | experiment x variant x date |

`src/pulse/contracts.py` declares each table's grain, primary key and required columns. The build **asserts PK uniqueness and non-null** after every write, so grain violations cannot ship silently.

---

## 7. SQL / DuckDB transformations

Real `.sql` files on disk, executed by DuckDB directly against parquet. Not ORM calls, not pandas dressed up as SQL.

```
sql/
├── silver/
│   ├── 01_orders.sql
│   ├── 02_sessions.sql
│   ├── 03_deliveries.sql
│   ├── 04_merchant_availability.sql
│   └── 05_dimensions.sql
├── gold/
│   ├── gold_daily_business_metrics.sql
│   ├── gold_zone_performance.sql
│   ├── gold_merchant_performance.sql
│   ├── gold_customer_retention.sql
│   ├── gold_promotion_performance.sql
│   └── gold_experiment_results.sql
└── checks/
    └── dq_checks.sql
```

`src/pulse/sql_runner.py` is small: read file, bind `{bronze}` / `{silver}` / `{gold}` path parameters, execute, `COPY (...) TO '...' (FORMAT PARQUET)`.

Techniques deliberately exercised as evidence: CTEs, `ROW_NUMBER() ... QUALIFY` for dedupe, `LAG` for period-over-period deltas, `FILTER (WHERE ...)` aggregates, cohort self-join for retention, and `generate_series` for a **calendar spine**.

### The calendar spine is load-bearing

A plain `GROUP BY order_date, zone_id` emits **no row** for a zone-day with zero completed orders. The worse Zone 7 becomes, the more of its bad days silently vanish from the series, so the detector would see a *shorter* series rather than a *worse* one — the incident partially hides itself.

Left-joining a `generate_series` spine crossed with the zone dimension forces a zero row to exist. Absence of activity becomes an explicit `0`, which is a detectable value. Same class of problem as the `merchant_availability` snapshot decision: in both cases the signal is an absence, and absences must be manufactured into rows before analysis can see them.

---

## 8. Databricks / PySpark evidence layer

Scope is deliberately minimal and must not block delivery.

**Required:**
- `databricks/silver_to_gold_spark.py` reproduces `gold_zone_performance` and `gold_daily_business_metrics` in genuine PySpark
- Explicit `StructType` schema enforcement on read
- Delta table writes
- **Real execution** on Databricks Free Edition, screenshotted
- Reconciliation test comparing Spark output against local DuckDB output on shared columns

**Explicitly optional, only if genuinely useful and supported on Free Edition:**
- `MERGE` — only if an actual incremental/upsert use case exists
- `OPTIMIZE` / `Z-ORDER`

No Spark feature is added purely as a buzzword.

---

## 9. Decision-engine interfaces

All analysis functions are **pure**: DataFrames in, frozen dataclasses out. I/O lives only at the edges (`io.py` loads parquet, `cli.py` writes artifacts).

### Parameters

```python
@dataclass(frozen=True)
class AnalysisParams:
    as_of: date
    comparison_window_days: int = 14
    baseline_window_days: int = 56
    sensitivity: float = 2.5           # z-score threshold
    min_materiality_brl: float = 5_000
```

Only these three are exposed in the UI: sensitivity, comparison window, materiality. The point is to prove recomputation, not to build an analyst configuration product.

### Core types

```python
Anomaly(metric, scope, scope_value, recent_value, baseline_value,
        deviation_abs, deviation_pct, z_score, direction,
        first_detected_date, n_observations)

SegmentContribution(dimension, segment, segment_deviation_abs,
                    contribution_pct, rank)

FunnelStage(stage, recent, baseline, deviation_pct, log_contribution,
            is_primary_break)

AssociatedDriver(metric, recent, baseline, deviation_pct,
                 correlation_with_target, temporal_alignment_days,
                 evidence_strength)

Diagnosis(anomaly, contributions, primary_segment, funnel,
          funnel_break_stage, drivers, pattern, confidence)

Impact(gmv_at_risk_brl, orders_lost, customers_affected,
       margin_impact_brl, daily_run_rate_brl, projected_30d_brl)

Priority(diagnosis, impact, impact_score, rank, score_breakdown)

Recommendation(action, rationale, expected_effect, validation_method,
               owner_function, playbook_id, effort)

DecisionMemo(incident, impact, concentration, associated_drivers, evidence,
             priority, recommended_action, potential_result,
             validation_method, confidence, generated_at, params)
```

### Module interfaces

```python
compute_metrics(gold, params)        -> MetricFrame
detect_anomalies(metric_frame, p)    -> list[Anomaly]
diagnose(anomaly, gold, p)           -> Diagnosis
estimate_impact(diagnosis, gold, p)  -> Impact
prioritize(pairs, p)                 -> list[Priority]
recommend(diagnosis, impact)         -> Recommendation
build_memo(priority, recommendation) -> DecisionMemo

run_decision_cycle(gold, params)     -> DecisionCycleResult   # THE orchestrator
```

### Anomaly detection

Day-of-week-adjusted baseline. Trailing `baseline_window_days` ending before the comparison window; per-DOW mean and standard deviation; recent = mean of last `comparison_window_days`.

```
z = (recent_mean - baseline_dow_mean) / (baseline_std / sqrt(n_recent))
```

Fires only when `|z| >= sensitivity` **and** `|deviation_pct| >= 3%` **and** estimated impact `>= min_materiality_brl`.

Day-of-week adjustment is not optional: weekend seasonality on a marketplace is large enough to trip a naive z-score every Saturday.

### Root-cause procedure

1. **Segment contribution.** For each dimension (zone, merchant category, acquisition channel), `contribution_pct = segment_deviation_abs / total_deviation_abs * 100`. Signed, so offsetting segments are visible. Guarded against near-zero total deviation.
2. **Funnel decomposition.** Log-decomposition of the funnel identity over the primary segment. Largest-magnitude term becomes `funnel_break_stage`.
3. **Associated drivers.** Operational metrics for the primary segment, ranked by Pearson correlation of daily series within the window, with temporal alignment (lag between driver shift and target shift).
4. **Pattern classification.** `(funnel_break_stage, top_driver_metric)` maps to a playbook key.

| Funnel break | Top driver | Pattern |
|---|---|---|
| `completion_rate` | `avg_actual_delivery_minutes` | `fulfillment_eta_degradation` |
| `order_conversion` | `merchant_availability` | `supply_availability_gap` |
| `contribution_margin` | `discount_rate` | `promo_margin_erosion` |
| `repeat_rate` | `cohort_quality` | `acquisition_quality_decay` |
| any | any | `unknown_pattern` (explicit fallback) |

### Documented formulas

```
confidence = 0.25 * min(|z| / 5, 1)
           + 0.25 * top_segment_contribution_pct / 100
           + 0.30 * max|driver_correlation|
           + 0.20 * consecutive_anomalous_days / comparison_window_days

impact_score = 100 * ( 0.45 * norm(projected_30d_gmv_at_risk)
                     + 0.20 * norm(orders_lost)
                     + 0.15 * norm(customers_affected)
                     + 0.20 * confidence )
```

`norm` is min-max within the current run's candidate set. All four components and their weights are surfaced in the UI as a breakdown bar. The score is explainable, never opaque.

### Causality language

The engine and all rendered output use **associated driver**, **contribution**, **correlated operational deterioration**, and **evidence consistent with**. Causal language is permitted only where the Experiment Lab establishes it. The Copilot system prompt enforces the same constraint, and the numeric/claims validator flags violations.

---

## 10. Recommendation playbook

`src/pulse/playbook.yml` — **data, not code**, keyed by pattern.

```yaml
fulfillment_eta_degradation:
  action: >
    Rebalance courier supply into {segment} during {peak_window};
    temporarily widen delivery radius caps and raise courier incentives
    until ETA returns to baseline.
  rationale: >
    Completion rate is the broken funnel stage while sessions and order
    placement are stable, and delivery time deterioration is the strongest
    associated driver.
  expected_effect: >
    Recovering delivery time to baseline would restore an estimated
    {gmv_at_risk} over 30 days.
  validation_method: >
    Zone-level difference-in-differences against unaffected zones over a
    14-day post-intervention window; primary metric completion_rate,
    guardrail delivery_cost_per_order.
  owner_function: Operations / Logistics
  effort: medium
```

Templates render via `str.format` against a whitelisted context dict. A missing key raises. A test renders every playbook entry against every pattern's real context.

`unknown_pattern` always exists, so the engine never returns `None` — it returns an explicit "investigate: pattern not in playbook" recommendation. No silent gaps.

### LLM narration boundary

The LLM receives the rendered `Recommendation` plus evidence and may produce a separate `narrative` field, displayed in a clearly labelled block. It cannot alter `action`, `validation_method`, or any numeric field.

`tests/test_memo.py` asserts the memo dictionary is identical with `NullNarrator` and with a stub narrator — mechanical proof that the LLM owns no policy and no facts.

---

## 11. Experiment Lab

**EXP-001 — at-risk reactivation incentive.**

- **Hypothesis:** a targeted discount for lapsed customers increases 30-day repurchase rate.
- **Population:** customers with no order in 21+ days as of 2026-08-16, **excluding zones 7 and 4** to avoid confounding with live incidents.
- **Unit of randomization:** customer. Deterministic hash of `customer_id`.
- **n = 12,000** (6,000 control / 6,000 treatment).
- **Treatment:** R$8 off next order.
- **Primary metric:** 30-day repurchase rate (binary).
- **Guardrail:** contribution margin per order.

### Statistics (scipy)

| | Control | Treatment |
|---|---|---|
| Repurchase (30d) | 14.2% | 18.5% |

- Absolute difference +4.3pp, relative lift +30%
- Two-proportion z-test, p < 0.01
- 95% CI on difference: approximately [2.9pp, 5.7pp]
- Ex-ante power check (alpha 0.05, power 0.8, MDE 3pp) reported so the UI can state whether the experiment was adequately powered

### Economics

```
incentive cost         1,110 redemptions x R$8   = R$ 8,880
incremental orders     1,110 - 852               =      258
incremental margin     258 x R$14                = R$ 3,612    ROI  -59%
+ 90-day downstream    258 x R$14 x 2.3 repeat   = R$ 8,308    ROI   -6%
```

The incentive is paid on **every** treatment redemption, but margin is earned only on the **incremental** ones. 1,110 customers redeem; only 258 of those orders would not have happened anyway. The remaining 852 are subsidy to customers who were returning regardless.

### Conclusion rendered

Statistical result and business result are stated together:

> The treatment produced a statistically significant increase in repurchase rate (+4.3pp, p < 0.01). First-order economics are negative (ROI -59%); including 90-day downstream value the programme is approximately break-even (ROI -6%). Recommendation: iterate on targeting precision and incentive depth before scaling.

---

## 12. Copilot grounding contract

The LLM receives an `EvidenceBundle` — bounded JSON, **never a DataFrame**.

```python
EvidenceBundle(
    as_of, params,
    headline_kpis,        # <= 12 rows: metric, recent, baseline, delta_pct
    priorities,           # <= 3, each a full memo dict
    detected_anomalies,   # <= 10 summary rows
    experiment_summary,   # dict | None
    data_quality_summary,
    glossary,             # KPI definitions
)
```

Hard caps asserted in tests: serialized bundle `<= 30,000` characters, and zero raw order rows present.

### Enforced prompt rules

1. Answer only from the bundle. If a fact is absent, say it is not in the current evidence.
2. Never invent a number. Every figure quoted must appear in the bundle.
3. Use association language. Never assert causality unless `experiment_summary` supports it.
4. Never propose executing an action; recommendations are advisory only.
5. Return structured JSON: `answer, metrics_used[], evidence[], segment, period, confidence, recommended_next_action`.

### Numeric guard

After the response returns, `validate_response()` extracts every numeral from the answer and asserts each appears in the bundle within rounding tolerance. Violations are surfaced in the UI as an "unverified figure" badge rather than rendered silently. Tested against a stub narrator that deliberately hallucinates figures.

### Offline mode

`Narrator` protocol with two implementations, selected by API key presence:

- `AnthropicNarrator` — model `claude-sonnet-5`
- `NullNarrator` — renders the same structured answer deterministically from templates, labelled **"AI narration offline — structured evidence shown."**

Offline output is never presented as AI. All analytics and all pages except live narration work with no key.

---

## 13. Streamlit information architecture

`st.navigation` multipage, entry point `app/streamlit_app.py`.

| Group | Pages |
|---|---|
| **Command** | Executive Overview, Decision Intelligence, Pulse Copilot |
| **Analytics** | Growth, Operations, Customers, Merchants, Promotions |
| **Platform** | Experiment Lab, Data Quality & Architecture |

### Visual system

`.streamlit/config.toml`: base dark, background `#0B1220`, secondary background `#131C2E`, text `#E6EDF7`, primary accent `#4C7DFF`. Semantic colours: positive `#17B26A`, warning `#F59E0B`, critical `#F04438`.

One injected CSS block in `app/theme.py` (KPI card, chrome removal, spacing, Inter font). A registered Plotly template `pulse_dark` carries all chart styling — no per-chart formatting anywhere. Shared components in `app/components.py`: `kpi_card`, `priority_card`, `evidence_table`, `status_pill`.

No cyberpunk, no neon, no large gradients, no decorative charts.

### Sidebar (global)

As-of date, the three live analysis parameters, cache state indicator, and two standing notices:

> **Decision Support — no autonomous operational execution.**
> **All data is synthetic.**

### Executive Overview

Seven KPI cards (GMV, orders, AOV, conversion, cancellation, ETA, active customers) each with value, delta versus baseline, and sparkline. Then: GMV trend with the anomaly window shaded; order trend; zone performance bar sorted with the affected zone highlighted; Top Incidents; Top 3 Priorities compact; one-line Decision Intelligence summary.

A recruiter should understand the product here in seconds.

### Decision Intelligence (signature page)

Three priority cards — problem, affected KPI, estimated impact, segment, evidence, confidence, recommended action — each expanding to the full Decision Memo.

**Root Cause Explorer:**
- Funnel waterfall of the log-decomposition (sessions / conversion / completion / AOV contributions to the GMV change)
- Segment contribution bar
- Dual-axis driver time series with incident onset marked
- Explicit chain, rendered from computed values (format shown, numbers not hard-coded):
  `GMV down -> Zone 7 (<contribution>%) -> completion_rate break -> delivery time +<x>% -> cancellations +<y>%`

Clearest visualization, not the fanciest. Every figure in the chain is read from the `Diagnosis` object; none is written into the template.

### Human-in-the-loop

Each memo carries a state in `st.session_state`: `INVESTIGATE` / `APPROVED` / `REJECTED`, with a timestamped audit list. These update local session state only and execute nothing. The constraint is stated explicitly in the UI.

### Pulse Copilot

Chat with six suggested prompt chips matching the core questions. Each response renders the answer plus a collapsible "Evidence used" panel showing `metrics_used`, segment, period, confidence, and the verification badge (all figures verified / N unverified).

---

## 14. Testing and verification strategy

TDD where it protects important logic. Visual markup is not heavily tested.

| Test module | Guards |
|---|---|
| `test_generator.py` | same seed produces identical **content** (sorted DataFrame hashes, not file bytes); row counts; PK uniqueness; FK integrity; date range; no ground-truth leakage into bronze |
| `test_quality.py` | each defect type detected and quarantined; rejected counts match `quality_defects.json`; categorical IDs never imputed; `is_imputed` present wherever numeric imputation occurred |
| `test_contracts.py` | every gold table matches declared grain/PK/columns; PK unique and non-null; calendar spine complete (no missing zone-days) |
| `test_metrics.py` | KPI formulas against hand-computed fixtures; GMV = item_amount of completed only; contribution_margin exact |
| `test_anomaly.py` | detects Zone 7 GMV drop; **zero anomalies on a clean synthetic series**; day-of-week seasonality does not cause false positives; sensitivity behaves monotonically |
| `test_root_cause.py` | contributions sum to ~100%; Zone 7 is top segment; `funnel_break_stage == "completion_rate"` for Zone 7 and `"order_conversion"` for Zone 4; drivers include delivery time for Z7, availability for Z4 |
| `test_prioritization.py` | score reproduces by hand; ranking stable; breakdown weights sum to 1 |
| `test_playbook.py` | every pattern has an entry; every template renders against real context; unknown pattern yields explicit fallback |
| `test_memo.py` | memo dict identical with `NullNarrator` and stub narrator |
| `test_experiments.py` | z-test and CI against scipy reference; ROI arithmetic; sample-size calculation |
| `test_copilot.py` | bundle <= 30k chars; bundle contains zero raw order rows; hallucinated-number stub is caught by the validator |
| `test_ground_truth_isolation.py` | `src/pulse/` and `app/` contain no reference to `ground_truth` |
| `test_no_secrets.py` | `.env` is gitignored; no key-shaped strings in tracked files |
| `test_spark_reconciliation.py` | Databricks Delta output matches local DuckDB output on shared columns |

### Recovery tests are the headline

`test_root_cause.py` asserts the engine independently recovered what `injected_incidents.json` records was injected — while the engine itself never reads that file.

### Verification ritual before claiming done

1. `pytest -q` full pass
2. `pulse run --all` clean from an empty `data/`
3. `streamlit run` and capture a screenshot of each of the 10 pages
4. Databricks notebook executed, reconciliation test green
5. `git ls-files | grep -c '\.env$'` returns 0

---

## 15. Repository structure

```
pulse/
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── .streamlit/config.toml
├── data/
│   ├── bronze/  silver/  gold/
│   └── ground_truth/            # tests only, never read by engine or app
├── artifacts/
│   ├── decisions.json
│   └── decision-memo.md
├── src/pulse/
│   ├── types.py  contracts.py  io.py  cli.py
│   ├── data_generator.py  incidents.py  quality.py
│   ├── sql_runner.py  metrics.py  anomaly_detection.py
│   ├── root_cause.py  prioritization.py  playbook.py  playbook.yml
│   ├── experiments.py  decision_memo.py  copilot.py
│   └── engine.py                # run_decision_cycle
├── sql/
│   ├── silver/  gold/  checks/
├── databricks/
│   └── silver_to_gold_spark.py
├── app/
│   ├── streamlit_app.py  theme.py  components.py
│   └── pages/
├── dashboard/screenshots/
├── docs/
└── tests/
```

---

## 16. Security

- `.env` is gitignored and never committed.
- `.env.example` contains `ANTHROPIC_API_KEY=` only.
- No API keys, tokens or credentials in tracked files, asserted by `test_no_secrets.py`.
- The application runs fully without an API key; only live Copilot narration is unavailable, and that state is labelled explicitly.

---

## 17. Assumptions recorded

| Assumption | Basis |
|---|---|
| Currency BRL (R$) | Brazilian marketplace context |
| UI and documentation in English | Spec authored in English; recruiter audience |
| Python 3.12 pinned via `uv` | Spec requirement; local default is 3.14 |
| Analysis "today" = 2026-09-10 | Fixed for reproducibility |
| 8 zones, 150 merchants, 25,000 customers | Sized so snapshots stay ~3.8x orders, not dominant |
| Session-to-order conversion ~17.9% | Realistic for high-intent logged-in delivery apps; smallest sufficient dataset |
| One experiment (EXP-001) | Spec requires one complete A/B; schema supports N |

---

## 18. Limitations (stated in README)

- All data is synthetic; results demonstrate method, not real market conditions.
- Anomaly detection is baseline/z-score based, not a forecasting model.
- Root-cause output identifies **associated drivers**, not proven causes, except where the Experiment Lab provides causal evidence.
- Single-user, no authentication, no multi-tenancy.
- Spark layer is a reproduced evidence path, not the production pipeline.

---

## 19. Definition of done

All 25 items from the original brief, condensed:

reproducible dataset · injected incidents · Bronze/Silver/Gold working · real SQL/Python/PySpark transformations · KPIs calculated · at least one incident auto-detected · decomposed into associated drivers · impact estimated · priorities ranked · Decision Memo generated · A/B experiment end-to-end · Copilot grounded in structured evidence · human-in-the-loop states · complete Streamlit app · strong Executive Overview · Decision Intelligence page · Experiment Lab · Copilot UI · Data Quality & Architecture visible · polished UI · screenshots captured · critical tests passing · accurate README · no committed secrets · no hard-coded results.
