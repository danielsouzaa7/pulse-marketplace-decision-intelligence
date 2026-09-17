# Trade-offs

What was chosen against, and what the choice cost. Every entry names the code or
the test that makes it real. All data is synthetic.

---

## Detection

### Precision over latency

The detector requires both halves of the comparison window to move the same way,
each carrying at least half the sensitivity on its own standard error, before it
fires at all.

**Bought:** roughly half the false-positive rate on a clean baseline. It costs
nothing on zone 7, whose every recent day is inside the degradation.

**Paid:** a genuinely fresh incident stays invisible for up to about half the
comparison window — ~7 days at the default 14. Absence of an anomaly is not
evidence of health, and the memo says so.

*Where:* `anomaly_detection.detect_on_series`, persistence block.

### A trailing baseline, uncorrected

The 56-day baseline ends the day before the comparison window. On this dataset
the zone-7 degradation starts 2026-08-11 and the baseline ends 2026-08-27, so
**17 of 56 baseline days are already degraded**.

**Bought:** the detector needs no incident calendar, no change-point model and no
human telling it when normal ended. It works on any series.

**Paid:** every impact figure understates. A persistent incident erodes its own
baseline, and the deviation measured against a depressed baseline is smaller than
the true one.

**Not corrected**, on purpose: correcting it would require knowing when the
incident began, which is the thing the detector is meant to discover. Instead
every downstream figure is worded as a floor — "at least", "estimated" — and the
memo's Limitations section explains why.

*Where:* `root_cause._window_means` docstring, `prioritization` module header.

### A z-scan, not a forecast

No ARIMA, no Prophet, no learned model.

**Bought:** the whole detection path is 200 lines a reader can verify by hand,
and the day-of-week adjustment does the one piece of seasonality work this data
actually needs (the weekday effect accounts for about 65% of baseline GMV
variance in the default 56-day window).

**Paid:** trend and holiday effects are not modelled. A slow drift would be
absorbed into the baseline rather than flagged.

---

## Diagnosis

### The driver exclusion rule, over maximum correlation

A candidate must be plausibly operationally **upstream** of the broken funnel
stage. Three candidates survive; everything else in the register is out.

**Bought:** the diagnosis explains rather than restates. Without the rule,
`contribution_margin` (r = +0.880 on zone 7) would rank first and reclassify a
fulfilment failure as a margin problem, and `cancellation_rate` (∓0.478, exactly
mirrored against `completion_rate`) would double-count one quantity as two
signals.

**Paid:** the surviving candidates measure |r| 0.401–0.448 — moderate, not
strong. The honest coefficient is lower than the dishonest one. That is the
trade.

*Where:* `root_cause.DRIVER_CANDIDATES`, which states the rule and its
consequences; `test_identity_terms_and_downstream_metrics_are_never_drivers`.

### Map the whole driver family, not the winner

All three fulfilment drivers map to one pattern key.

**Bought:** classification stability. The three cluster within 0.047 of each other
and have flipped order at other calibration points; a classification that flips
between runs is worse than a coarser one that holds.

**Paid:** the playbook cannot distinguish "delivery is slow" from "the ETA
promise is wrong". Both route to the same recommendation.

*Where:* `root_cause.PATTERNS`; the memo says these series deteriorated together
and names none of them as *the* driver.

### Weak drivers cannot pick a playbook

A driver below the reporting threshold contributes zero confidence and is also
barred from classification.

**Bought:** evidence the engine has already scored as worthless cannot decide an
action a human is asked to approve. On zone 4 the three candidates measure |r|
0.164 / 0.148 / 0.131 — a spread far too narrow to rank, so whichever "won" would
be a coin flip choosing the recommendation.

**Paid:** zone 4 reaches a pattern through the signature `(availability_rate,
None)` — "this moved and nothing upstream moved with it" — which is a narrower
statement than a named driver would be.

*Where:* `root_cause._classification_driver`;
`test_zone4_supply_pattern_does_not_hinge_on_which_weak_driver_ranks_first`.

### Only reachable playbook patterns are written down

Three pattern rows from the original design were deleted after measurement
because no diagnosis this engine can produce reaches them, and entries for
promotion margin and retention were investigated and deliberately not written.

**Bought:** every playbook entry is verifiable against a real run.

**Paid:** the playbook looks smaller than the design sketch. Eight of the twelve
diagnosed anomalies fall to `unknown_pattern`, mostly because rate and duration
metrics get no segment decomposition and no funnel split by design.

*Where:* `playbook.yml` header; `test_every_pattern_row_names_a_stage_and_driver_the_engine_can_produce`.

---

## Prioritisation

### Group by scope, net aggregates against segments

Seven zone-7 metrics fired; they become one Priority. The company group claims
only what the zone groups have not already claimed.

**Bought:** the same money is never claimed twice. Without rule 1 the UI would
show seven near-identical entries each claiming the same R$10k; without rule 2
the company priority would claim a residual that is mostly zone 7's.

**Paid:** only one segment dimension is netted. Two dimensions each partition the
company independently, and subtracting both would remove the same money twice —
so the dimension with the most groups wins. A run needing two overlapping
dimensions netted jointly needs a real attribution model, not subtraction. That
is marked in the code as a known ceiling.

*Where:* `prioritization.prioritize`, `_net_of_nested`.

### Two confidence figures, both correct

The score uses the *group's* strongest evidence; the displayed memo reports the
*representative diagnosis's* own.

**Bought:** the member chosen to be displayed is chosen for what it explains (a
segment decomposition and a funnel split), and that presentation choice cannot
move the ranking. On zone 7 the GMV anomaly explains most but scores 0.624, while
its `completion_rate` sibling explains nothing structural and scores 0.740.

**Paid:** the UI has to show two numbers and explain the difference, which it
does, in two places.

*Where:* `prioritization.group_confidence` docstring.

### Min-max anchored at zero, not at the smallest candidate

**Bought:** the component reads as "share of the largest at-risk magnitude in this
run", which is what the memo claims it is. A floor at the smallest candidate would
force whichever candidate happens to be last to score 0 on every impact term
however material it is, and would make one candidate's score depend on how small
an unrelated candidate happens to be.

**Paid:** scores are not comparable across runs with different candidate sets.
They are an ordering within one run, and nothing more.

*Where:* `prioritization._norm`.

---

## The LLM

### Deterministic engine, narrating model

**Bought:** the whole product works with no API key, every number is
reproducible, and a hallucinating narrator can change wording and nothing else.

**Paid:** the model cannot surface a pattern the engine did not find. It explains
the engine's answer; it does not have one of its own.

### A pattern list for injection reporting, a structural boundary for defence

Six injection shapes are flagged and displayed, and **nothing branches on those
flags**.

**Bought:** honesty about what a pattern list is. It can always be evaded; the
boundary that cannot is that user text never reaches an instruction position and
never touches a number.

**Paid:** the flags are informational only, which looks weaker than a filter and
is stronger than one.

---

## Infrastructure

### DuckDB locally, Spark as unexecuted evidence

**Bought:** the whole pipeline runs in seconds on a laptop with `uv sync` and
three commands. Windows Spark setup for 13 MB of parquet would have been a time
sink with no analytical return.

**Paid:** `databricks/silver_to_gold_spark.py` has never run, its reconciliation
test skips, and the README says so in those words. It demonstrates that the
transformations were written for a distributed engine; it does not demonstrate
that they executed on one.

### Streamlit, no separate API

**Bought:** one deployable surface, no frontend build, no serialisation layer
between the engine and the page.

**Paid:** no REST interface, no non-Python consumer, and page state is per
browser session.

### No authentication, no multi-tenancy

Out of scope by design, stated in the spec and in the README. The human decision
states are session-local and execute nothing, which is why the absence of an
identity model is not a security gap here — there is nothing to authorise.

---

## The dataset

### Synthetic, seeded, and fixed to an absolute window

**Bought:** anyone can reproduce every figure in the README from an empty `data/`
directory, indefinitely — dates are hard-coded, never relative to `today()`. Tests
assert content hashes of sorted DataFrames rather than file bytes, because parquet
metadata changes with library versions without changing data.

**Paid:** results demonstrate a method, not a market. No claim in this repository
is a claim about any real marketplace.

### Four incidents injected, two detectable

**Bought:** the two undetectable ones are the honest finding. A 21-day promotion
sits inside its own baseline; a cohort table has no daily series to scan. Both
limits are asserted in the recovery test, so wiring in either dimension without
wiring in its detection fails loudly.

**Paid:** the headline is "two of four", not "four of four". That is the number
the evidence supports.

### Availability at zone grain, not merchant grain

Every merchant is shut one fixed weekday, so a whole weekday of each merchant's
series is NULL, its day-of-week baseline for that weekday does not exist, and the
detector correctly refuses it — measured: 0 of 150 merchants fire, including 0 of
zone 4's 13.

**Bought:** the detector refuses a series it cannot model rather than fabricating
a baseline for it.

**Paid:** supply failure is reported at zone roll-up. Which merchants went dark is
a question the Merchants page answers visually and the detector does not answer at
all.

*Where:* `metrics.METRIC_REGISTER` comment; `supply_availability_gap` rationale in
`playbook.yml` states the limitation in the analyst-facing text.
