# KPI definitions

Every formula here is implemented once, in `src/pulse/metrics.py` (register and
units) and the gold SQL under `sql/gold/`. Nothing in `app/` recomputes a KPI.
All data is synthetic.

## Money columns

These five are generated per order; nothing downstream may assume one is a fixed
multiple of another.

| Column | Meaning |
|---|---|
| `item_amount` | merchandise value of the order |
| `delivery_fee` | fee charged to the customer |
| `delivery_cost` | actual cost to fulfil — **generated per order, not a constant** |
| `discount_amount` | marketplace-funded discount |
| `commission_rate` | merchant take rate |

```
GMV                  = item_amount                        (completed orders only)
customer_gross_value = item_amount + delivery_fee
net_revenue          = customer_gross_value - discount_amount
contribution_margin  = item_amount * commission_rate
                       + delivery_fee
                       - delivery_cost
                       - discount_amount
```

**`delivery_cost` must be a per-order column.** If it were a global constant,
contribution margin would be a scalar multiple of GMV, and a promotion that lifts
orders while destroying margin would be mathematically undetectable — the two
series would move together by construction. It is also why contribution margin
can fall 37.8% in zone 7 while GMV falls 13.5%: a degraded delivery experience
costs more to serve, so what is left after cost falls faster than what comes in.

## The funnel identity

```
GMV = sessions × order_conversion × completion_rate × AOV

order_conversion = orders_placed    / sessions        (pre-checkout)
completion_rate  = orders_completed / orders_placed   (post-checkout)
AOV              = GMV              / orders_completed
```

Exact in gold, verified daily to 1e-12. Taking logs makes the percentage changes
additive:

```
dln(GMV) ≈ dln(sessions) + dln(order_conversion) + dln(completion_rate) + dln(AOV)
```

so the largest-magnitude term **is** the broken stage — computed, never asserted.
This is the engine's primary diagnostic discriminator, and it is what separates
zone 7 (`completion_rate` breaks, conversion stable) from zone 4 (conversion
falls, completion stable) despite the identical headline symptom.

## The register

`METRIC_REGISTER` in `metrics.py` is what the detector, the diagnosis and the
playbook can see. **A metric gold carries but the register omits is invisible to
every downstream module**, however clean the signal is — `availability_rate` sat
unseen in `gold_zone_performance` at z = −6.58 until it was registered.

Two registers describe each metric and they answer different questions. `_UNITS`
says what a value is **measured in**; `METRIC_SEMANTICS` says what a **window
aggregate** of it *means*, which is what `MetricFrame.headline()` returns. One
register answering both is how company GMV came to be shown as R$ 29,977.82 on a
card headed "GMV" when the 14-day window's GMV was R$ 419,689.55 — the same
measurement, one fourteenth the size, and nothing about the float says which.

| KPI | Definition | Unit | Window aggregate | Scopes |
|---|---|---|---|---|
| `gmv` | `sum(item_amount)` where status = completed | BRL | `daily_average` | company, zone |
| `orders_placed` | count of placed orders | orders | `daily_average` | company, zone |
| `orders_completed` | count where status = completed | orders | `daily_average` | company, zone |
| `sessions` | count of marketplace sessions | sessions | `daily_average` | company, zone |
| `order_conversion` | `orders_placed / sessions` | ratio | `rate` | company, zone |
| `completion_rate` | `orders_completed / orders_placed` | ratio | `rate` | company, zone |
| `cancellation_rate` | `cancelled / orders_placed` | ratio | `rate` | company, zone |
| `aov` | `gmv / orders_completed` | BRL | `per_order_average` | company, zone |
| `avg_promised_eta_minutes` | mean `promised_eta_minutes` | minutes | `per_delivery_average` | company, zone |
| `avg_actual_delivery_minutes` | mean `actual_delivery_minutes` | minutes | `per_delivery_average` | company, zone |
| `on_time_rate` | share with `actual <= promised + 5 min` | ratio | `rate` | company, zone |
| `contribution_margin` | formula above | BRL | `daily_average` | company, zone |
| `active_customers` | distinct customers ordering | customers | `daily_average` | company |
| `availability_rate` | `available_hours / scheduled_open_hours` | ratio | `rate` | **zone only** |

**What each aggregate class means.**

- `daily_average` — a per-day flow or count. The window figure is the mean of the
  daily values, a **window total also exists and is a different number** (days ×
  the mean), and the UI says "por dia" / suffixes `/dia` on both the label and the
  value. `active_customers` is the one where no window total exists at all:
  it is daily-**distinct**, so summing days yields customer-days.
- `rate` — dimensionless at any window length. No total to confuse it with,
  because rates do not add. Rendered as a percentage and never "/dia".
- `per_order_average` — `aov` is BRL **per order**. Labelling it "/dia" would be
  a new false statement, not a fix.
- `per_delivery_average` — the two duration metrics, minutes **per delivery**.

`.headline()` puts the class on every row it returns as `semantic`, so the CLI
artefact, the Copilot bundle and every page read one classification off one
object. `METRIC_SEMANTICS` is asserted to cover `METRIC_REGISTER` exactly — a
metric added without a class raises at import rather than picking a formatter's
default.

Units are not cosmetic either. `_UNITS` drives two downstream behaviours with no
branch anywhere else:

- **The BRL materiality floor applies only to currency metrics.** "Is a 5.7pp
  availability move worth more than R$5,000?" is not a question a currency gate
  can answer, and applying it to rates would mute every rate signal.
- **Segment decomposition applies only to additive units** (BRL, orders,
  sessions). Zone completion rates do not sum to a company completion rate, so a
  rate gets no decomposition rather than a wrong one.

## The traps, one per metric family

**`active_customers` is daily-distinct and must never be summed.** Summing it
over a 14-day window yields customer-*days*, roughly 6× the real figure. Where a
true windowed distinct count is needed — the impact estimate's
`customers_affected` — it is recomputed as a `COUNT(DISTINCT)` over silver
orders, which is the one place in `prioritization.py` that touches disk. Days
cannot be added up into customers.

**`gold_experiment_results.converted_customers` is a per-day count.** Summing it
over the experiment window counts a customer who ordered on six days six times,
inflating the numerator while the denominator stays fixed — the rate can then
exceed 1.0. The Experiment Lab recomputes the window-level numerator as an
independent distinct count over silver and takes `assigned_customers` **once**
per variant (it is repeated on every row by construction).

**`cancellation_rate` is the arithmetic complement of `completion_rate`.** Every
order either completes or cancels. On zone 7 they measure −0.478 and +0.478
against GMV: exactly mirrored. They are one quantity with a sign flip, not two
signals, which is why `cancellation_rate` is excluded from driver candidates.

**`avg_actual_delivery_minutes` is NaN on a zero-delivery day, not zero.** NaN
means "not measured". Filling it with 0.0 asserts instant delivery and drags both
z-scores and correlations toward a number nobody observed. Correlation runs on
pairwise-complete observations only, and the surviving day count is what the
evidence strength is judged on.

**A headline `delta_pct` and a fired anomaly's `deviation_pct` are not the same
number computed twice.** The headline is a flat window-mean comparison; the
detector uses a day-of-week-adjusted baseline. Day-of-week effects make them
diverge, and that is expected.

## Promotional metrics

```
promotional_cost = sum(discount_amount)
promotional ROI  = incremental contribution margin / promotional cost
```

The word *incremental* is the whole of it. The cost is paid on every redemption;
the margin is earned only on the redemptions that would not have happened
anyway. Charging the cost against incremental orders only is the arithmetic that
turns a money-losing programme into a reported win.

**There is no incentive-value column in this synthetic dataset.** Where the
Experiment Lab reports an ROI it derives cost from observed discounts on
treatment orders, and that figure is labelled a proxy in the README. It
demonstrates the method; it is not a measured return.

## Retention

`gold_customer_retention` is grained `(cohort_month, acquisition_channel,
period_index)` with `retention_rate = retained_customers / cohort_size`. There is
no date column, deliberately — a cohort table is not a time series. The
consequence is stated plainly in the README: the anomaly detector, which scans
daily series, cannot see this table at all.
