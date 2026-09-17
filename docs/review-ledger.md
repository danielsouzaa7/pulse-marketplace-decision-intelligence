# Review ledger — findings, corrections and what is deferred

Nothing here is a claim that the repository is clean. The repository-wide
reviewer verifies that independently.

---

## Round 1 — the seven blocking findings from code review #1

| ID | Finding | Where the fix lives |
|---|---|---|
| C1 | Comparison-window MEANS displayed as if they were window TOTALS | `metrics.METRIC_SEMANTICS`, `ui_text.HEADLINE_LABELS`, `decision_memo._value` / `_impact_sentence` / `_impact_table` |
| C2 | An unconditional delivery-cost causal mechanism on the flagship page | `components.margin_gmv_note` + `_PATTERN_CONSISTENCY` |
| I1 | A netted/residual customer count labelled as a measured count | `Impact.customers_are_residual`, `metrics.customers_label` |
| I2 | README experiment targeting/exclusion claims contradicted the generator | `data_generator._EXPERIMENT`, README "Experiments" section |
| I3 | A business verdict resting on an invalid treatment-cost proxy | `experiments.economic_evaluation`, `Economics.cost_basis` |
| I4 | A qualitative priority-gap claim, guarded by a vacuous test | `prioritize()` returns `score_gap_to_next`; `components.ranking_note` |
| I5 | A numeric validator with no claim-local grounding | `copilot` claim-local guard |

Code review #2 found that **three of those seven were not actually closed**, and
that the round had introduced one new Critical. Round 2 is that repair.

---

## Round 2 — what code review #2 found, and the correction

### Two Critical-class failures

**C3 — the verbatim-quotation exemption certified fabricated magnitudes.**
`_is_quotation` accepted any fragment of ≥24 characters that was a *substring*
of the bundle's prose, with no per-figure grounding. Substrings can begin inside
a numeral, so truncating leading digits off a real figure still matched:
`"8,2% contra 91,2%, z = -5,40, 14 dias observados)."` was certified while
`8,2` existed nowhere in the bundle (the real figure is `88,2`). 42 distinct
fabricated magnitudes were reachable. It also certified a real sentence
reattached to the wrong scope, and it certified whatever prose surrounded the
quote.

**The exemption is gone.** There is no path by which text matching bypasses
numeric grounding. That was only affordable because the bundle now carries the
structured facts behind its own prose — `funnel`, `segment_shares`,
`score_breakdown`, `concentration_remainder_pct` — instead of 8,779 characters
of memo `evidence` prose that nothing could check. The bundle got **smaller**
(27,291 → 20,626 characters) while gaining strictly more checkable facts.

**C1 regression — a daily mean was certified as a 14-day total.**
`copilot._kpi_row` whitelisted fields and dropped the `semantic` key the round-1
fix had added, so the one surface whose job is to stop a figure being misread
was the only surface that could not tell a daily average from a window total.
`"O GMV total da empresa nos 14 dias foi de R$ 29.977,83."` verified against a
daily mean whose window total is `R$ 419.689,55`. Three places claimed
otherwise: the system prompt instructed the model to read `semantic`,
`docs/kpi-definitions.md` said the bundle carried it, and the only guard
asserted the *prompt* contained the word.

Now: `_kpi_row` and `_anomaly_row` both carry `semantic`, and a figure whose
fact is a daily average or a window total is rejected unless the claim says
which of the two it means.

### The rest

| Finding | Correction |
|---|---|
| I1 reachable via Copilot — `"3.896 clientes pediram"` certified | `provenance` on every grounded fact; a `PROV_RESIDUAL` figure must be named as a residual and can never be called measured |
| I3 reachable via Copilot — `"O ROI medido do tratamento foi de -88,8%"` certified | `Economics.roi` is no longer walked as a generic leaf. `experiment_evidence()` projects it as `proxy_ratio_pct` under concept `experiment_proxy_roi` with `PROV_ILLUSTRATIVE`, declared in `_EXPERIMENT_FIELDS` |
| I4 not closed — `app/pages/operations.py` still characterised the gap | qualitative characterisation removed; the scanner now bans the **class** of claim in score-gap context rather than three literal phrases |
| Sentence splitter broke on `p.p.` / `aprox.`, stranding a figure from its scope | `segment_sentences()` masks an explicit abbreviation set before splitting |
| Unknown metric failed OPEN while unknown scope failed closed | three explicit states — `CONCEPT_NONE` / `CONCEPT_KNOWN` / `CONCEPT_UNKNOWN`; only the first may use scope context |
| Intra-sentence cross-binding let two metrics pool their values | `_clauses()` splits on conjunctions when a sentence names more than one concept |
| `p.p.` folded into percent | seven distinct kinds; `KIND_POINTS` is separate from `KIND_PERCENT` |
| Unconditional ×100 twin made correlations and confidences usable as percentages | removed entirely. A ×100 twin exists only for metrics whose `_UNITS` is `ratio`. There are now **no** concept-free percent facts in the index |
| Duplicate shadowed `_matches` / `_named_identity` | gone; a test asserts `copilot.py` has no shadowed top-level definition |
| Date masking accepted ISO only while the app renders `dd/mm/yyyy` | `_date_in_prose` accepts either form of the same day |
| `assert` was the only enforcement for the size cap and the row-identifier leak guard | `EvidenceBundleTooLargeError` and `ForbiddenEvidenceKeyError`; `python -O` can no longer strip either |

---

## The false-acceptance benchmark — corrected

**The round-1 figure of "61.2% → 0.0%" was not reproducible and has been
withdrawn.** Its truth set was `copilot._build_claims(...)` filtered by the
validator's own acceptance predicate at the validator's own tolerance, so the
0.0% was true by construction and the test could not fail for the reason it
claimed to test. That test is deleted, not moved.

The replacement (`tests/test_copilot_grounding.py`) builds its truth from the
engine's canonical objects — `DecisionCycleResult`, `METRIC_SEMANTICS`,
`_UNITS`, `ExperimentReport` — and an AST check on the test file itself asserts
it imports and reaches for none of the validator's internals. Mutations attack
one leg of the contract each: wrong scope, wrong metric, wrong unit, wrong
semantic, wrong value, unknown concept, swapped values.

Measured on the same 5,324 mutations:

| guard | accepted | rate |
|---|---|---|
| original flat guard ("does this magnitude exist anywhere?") | 3,497 | **65.7 %** |
| round-1 claim-local guard | 59 | **1.11 %** |
| round-2 semantic guard | 0 | **0.00 %** |

Per family at round 2: wrong_value 0/4,572 · wrong_metric 0/582 · wrong_scope
0/92 · wrong_unit 0/16 · wrong_semantic 0/8 · unknown_concept 0/48 ·
wrong_scope_count 0/6.

The benchmark reports the flat baseline on every run and **asserts it stays
above 50%** — if the mutation set ever stops being adversarial, the 0% stops
meaning anything, and that has to fail loudly rather than quietly.

The 0% is not the whole picture, and two limits are worth stating:

- The benchmark measures **numeric** claims. A grounded figure inside an
  ungrounded causal sentence is reported as grounded; nothing here certifies the
  non-numeric part of a sentence, and the Copilot page says so.
- Scope-level facts (an impact figure, a score) carry no concept, so a clause
  naming a *known* metric may quote them. `"a taxa de conclusão em Zona 7 custou
  R$ 10.351 acumulado"` therefore grounds, because that money is denominated in
  the scope rather than in the metric that tripped the alarm. It is not a
  fabricated magnitude, but it is a loose attribution.

---

## The bundle cap — what it measures, and the disclosure round 1 owed

`MAX_BUNDLE_CHARS = 30_000` counts **characters of the serialised bundle**,
which is exactly the representation `build_request()` embeds in the prompt
(`bundle.to_json()`, `ensure_ascii=False`). It is not bytes, not tokens, and not
the whole request — the system prompt and the question sit outside it. It is an
internal conservative bound on how much evidence a reviewer has to be able to
check, and it is not a provider limit.

**What round 1 did not write down.** Two changes in that round interacted:

1. `MAX_ANOMALIES` went 10 → 16, adding 493 characters.
2. `test_bundle_is_bounded` changed from `json.dumps(to_dict())` to `to_json()`.

Change 2 was defensible on its own — the old measurement counted the
ASCII-escaped form, ~10% larger, which is never sent anywhere, and it disagreed
with the engine's own assert. But it was also **required** for the suite to stay
green after change 1: at `MAX_ANOMALIES=10` the old measurement was 29,543 ≤
30,000 and passing; at 16 it is 30,036 > 30,000 and failing. The round-1 ledger
presented it purely as a correctness realignment. Both things were true; only
one was recorded.

Current sizes, after round 2 replaced memo prose with structure:

| payload | characters | headroom |
|---|---|---|
| default bundle | 20,763 | 9,237 (31%) |
| with the real `experiment_evidence(report)` attached | 22,300 | 7,700 (26%) |

*Corrected in round 3: the round-2 figures above were 137 characters short and
did not reproduce. These are re-measured; the Copilot page prints the live size.*

Round 2 also fixed the overflow at the documented injection point:
`build_evidence_bundle(result, experiment_summary=asdict(report))` was 31,831
characters and **raised**, so the Copilot's own suggested question about the
experiment could not be answered with an experiment attached.
`experiment_evidence()` is a 1,541-character bounded projection that names each
figure's unit in the field name (`_pp`, `_pct`, `_brl`) and carries
`measured_roi: None` explicitly, because the absence is a fact the model has to
be able to state.

---

## Round 3 — what code review #3 found, and the correction

Review #3 reported 2 Critical and 13 numbered Important findings. Every one was
reproduced before it was fixed; each fix has a test that failed first.

### Critical

**C4 — the offline Copilot narrated a scope's GMV as the fired metric's.**
`offline_evidence` wrote "o completion rate médio diário ficou R$ 909 por dia" —
a rate in R$/dia — in 38 of 72 swept sidebar configurations. The money is the
scope's GMV (`estimate_impact`), so the sentence now names GMV.
*Found while reproducing it:* the same sentence hard-coded "abaixo da baseline"
for a SIGNED run rate, so a zone whose GMV rose R$ 133/dia was narrated as below
baseline (4 of 72), and `playbook.render_context` rendered `abs()` of a signed
projection as "GMV recuperável de no mínimo R$ 4.619" when GMV had risen (18
priorities across the sweep). Both now take direction from the sign; the
playbook floors the recoverable figure at zero, as `prioritize()` scores it.
Test: `test_offline_evidence_names_the_scopes_money_and_its_real_direction`
sweeps non-default knobs and asserts it reached both preconditions.

**C5 — direction was not a grounded dimension.** Facts were indexed as
magnitudes, so "subiu 5,7%" certified against a 5,7% drop. Facts now carry the
direction of their movement; a claim's direction is read from an explicit sign
or the nearest direction word, and "melhorou"/"piorou" resolve through
`metrics.LOWER_IS_BETTER` (now the single polarity register the app also reads).

### Important

| # | Finding | Correction |
|---|---|---|
| 1 | Loose scope-level attribution: any metric could quote the scope's money | impact facts carry an `owner` series; a clause naming a different metric does not ground |
| 2 | Unknown concept inside a longer phrase ("GMV de fraude") and unrecognised sentence shapes failed open | a qualifier after a metric name that is not a scope, generic noun or number makes the clause unknown; "no concept" requires a generic noun to be named |
| 3 | A deviation certified as a level | deviation facts need a change word or a sign, and never ground a figure asserted as a window's level |
| 4 | Recent and baseline interchangeable | levels carry their window; words just before a figure ("atual", "era", "para", "contra") assert one |
| 5 | Metric pairs pooled across separators outside the clause list | each figure binds to its nearest concept, whatever separates them |
| 6 | Window figures restated per hour/week/month | any other time basis rejects every fact that has a semantic |
| 7 | One figure pooled across two named scopes | adjacent scopes form a conjunctive group the figure must be true of; a scope named right after a figure binds first |
| 8 | Benchmark excluded the open classes by construction | oracle states money ownership; 9 new families (212 mutations) — the previous validator accepts 95.8% of them, the current one 0; 75 true claims, 0 rejected; README figures asserted against the run |
| 9 | Engine prose self-rejected ("não uma contagem medida" read as measured; "por dia da semana" read as per-day) and the page claimed engine prose is accepted | negation and "dia da semana" handled; page and README claims rewritten to what holds |
| 10 | Gap scanner banned set drawn from its own word list | **open** — see Deferred |
| 11 | README stated the proxy-derived power conclusion unqualified | rewritten: the MDE is measured, the break-even comparison is illustrative |
| 12 | Screenshots 02/03/07 captured outside `st.navigation` | root cause fixed: `app/pages/` renamed `app/views/`, because Streamlit auto-discovered it and a refresh or deep link ran the page directly with raw filenames and no parameters. All eight recaptured through the sidebar |
| 13 | Circularity guard a stale, `getattr`-evadable denylist | an allowlist of the Copilot's public entry points, plus a ban on dynamic access |

Also closed from the MEDIUM/LOW list: bare `assert` gone from `src/` (an AST test
keeps it so; the gold gate is verified under `python -O`); `figures_checked` no
longer counts scope digits, and the badge no longer turns green over zero
figures; "real/reais" removed from the measured cue. Causal wording in a
narration is now flagged (expression list — a floor, not a proof).

**Found by attacking the data boundary:** negative GMV, a rate above 1, an
infinite value, completed orders above placed and an empty table each ran to a
complete, plausible ranking. The gold contract now checks value domains, and
`load_gold()` applies it to what it reads. A live model call has a bounded
timeout and one retry instead of the SDK's 600 s and two (60 s after round 4).

## Round 4 — the code review of round 3's fixes

An independent review of `7c7d3af..8a78554` confirmed the engine-text fixes, the
assert work, the rename and the benchmark reproduce — and found that "closed" in
the round-3 table above was true of the benchmark's **templates** and not of plain
wording. It wrote its probes in ordinary Portuguese: all 33 false ones passed both
the old and the round-3 validator, and 9 of 39 true ones were rejected, several
of them regressions from round 3.

| # | Finding | Correction |
|---|---|---|
| C1 | Facts with no concept and no owner (the R$ 5.000 materiality floor, the score, the confidence, the customer count) backed a claim about ANY metric | every fact now has an identity: parameters, scores, confidence, ranks and customers are named concepts; a window length grounds only a figure written as days |
| C2 | "no" (em + o) and "sem dúvida" silenced the causal check; any sentence that merely mentioned the experiment was exempt | negation only within the same stretch of the sentence, affirming idioms removed, "se deve a" added; the exception needs the experiment as the subject |
| I3 | Change words read clause-wide; recent-window names not cues; "Zona 4 e Zona 7" cut by the clause splitter; "de pedidos cancelados" accepted as a qualifier | change and direction read from the figure's own stretch; "janela de comparação"/"últimos N" are recent cues; joined scopes and metrics are conjunctive groups the splitter respects; "respectivamente" pairs figures; a negated mention binds nothing |
| I4 | Regressions: "duas semanas" read as per-week, "variou" not a change, "contra" read as baseline, cross-scope comparisons read as movement | word boundaries, "variou", "contra" removed as a window cue, direction checked on a level only when the level was reached by a movement ("caiu para") |
| I5 | README and page said an unpatterned paraphrase "is not certified" | reworded: it **can pass unflagged**; known passing examples listed; the 0 % scoped to templates |
| I6 | NaN accepted in every gold column | NaN rejected in additive columns; a ratio may still be unmeasured |
| Minor | traceback on a failed gate; singular pill for plural claims; 25 s timeout could cut a full answer; `__globals__` absent from the guard | the app stops with a sentence; plural fixed; 60 s; guard extended |

**Still open, and stated in the README:** of the review's 33 false probes, 3 still
pass — an adjective narrowing a metric ("o GMV orgânico") and a window total
restated as "no último mês" / "em 30 dias". The plain-wording set asserted in
`tests/test_copilot_grounding.py` was selected after the fixes, so it protects
against regression and is not an unbiased measurement.

## Round 5 — the code review of `82f4578`

A fresh review, not seeded with any earlier verdict, confirmed the suite, the
structural boundary, fact identities, the data gate and the documentation, and
blocked on two false passes on the validator's own central patterns:

| # | Finding | Correction |
|---|---|---|
| I-1 (blocking) | A LEVEL written as a change certified: "A taxa de conclusão em Zona 7 caiu 73,5%" (the recent level; the change is -15,2%), "O GMV médio diário em Zona 7 caiu R$ 4.756 por dia" (6,4x the real R$ 739) — contradicting "change vs level" on the page and in the README | a change word directly before a figure (not followed by "para") marks it a change magnitude, which no level can ground; `level_as_change` benchmark family — the previous validator accepted 12/12, this one 0 |
| I-2 (blocking) | The 30-day projection and the measured window deviation shared one identity, so R$ 22.180 was certified as the 14-day measured loss and the two figures could be swapped | the projection has its own semantic, asserted by "projetado/projeção/próximos N dias" in the figure's own stretch; `projection_vs_window` family — previously 6/6 accepted, now 0 |
| I-3 | A sentence with no scope defaulted to company, so "Nessa zona, o GMV caiu 4,1%" certified a company figure for Zona 7 | a scope-less sentence is read against the scope the answer last named |
| Minor | `ask()` raised on wrong JSON types; "1" grounded a 0,624 confidence; "não há dúvida de que … causou" silenced the causal check; NaN rates with a non-zero denominator passed the gate; "2 de 1 figuras"; stale comments; the Copilot transcript survived a parameter change | all fixed, each with a test that failed first |

The reviewer classified the documented residuals: "o GMV orgânico" and "no último
mês" as important but acceptable with the documented limitation (a real value
under a wrong label or period description; engine-owned fields untouched), "em
30 dias" the same once I-2 was fixed, and the two fail-closed true sentences as
minor. Those five remain as stated in the README.

## Round 6 — the code review of `fce6110`

A fresh review confirmed round 5's fixes for the exact shapes they targeted and
blocked on the shapes next to them:

| # | Finding | Correction |
|---|---|---|
| I-1 (blocking) | A level as a change still certified in the commonest forms: a change noun with "foi de" ("A queda da taxa de conclusão em Zona 7 foi de 73,5%"), a modifier before the figure ("caiu quase 73,5%"), and any "para" after it ("caiu 73,5% para o menor nível") | change word within three words of the figure, or a change noun with a copula, marks a change magnitude; exempt only a figure reached ("para/até/a X") or a starting figure followed by "para <número>" |
| I-2 (blocking, a regression of round 5) | Carrying the last scope into every scope-less sentence certified a zone figure under "No geral" / "Globalmente" — and the README called it fail-closed | a scope carries only through an explicit back-reference ("Nessa zona", "Lá"); otherwise the figure must be true of the previous scope AND the company |
| I-3 (blocking) | "porque" was not a causal cue | added. "em função de" was also removed here, on the claim that it flagged the engine's own method description — **that claim was wrong** (no engine text contains the phrase; the sentence was a reviewer's example), and round 8 restored it |
| I-4 (blocking) | NaN `availability_rate` with scheduled hours passed the gate and removed priority 3; the comment said the class was closed | `availability_rate` → `scheduled_open_hours` and `aov` → `orders_completed` join the denominator checks; on-time and minutes stay allowed, stated in the comment |
| I-5 | Day counts grounded any period ("nos últimos 56 dias"), 30 grounded as a window, "projeção ... na janela de comparação" certified | each length has its own meaning: the comparison length (and observed days) never inside a projection, the baseline length only where a baseline is named, the 30-day horizon only with projection wording; a projection named in the comparison window is rejected |
| Minor | "caiu de 86,6% na baseline para 73,5%" rejected; "projeto" read as a projection | fixed |

Measured: the `fce6110` validator accepted 54 of the 72 mutations in the round-5
and round-6 benchmark families; this one accepts 0, and still rejects 0 of 78
true claims. The README residual list was broadened to the classes the review
found (adjective and time qualifiers; periods described in words) and the
scope-carry description corrected.

## Round 7 — a product decision instead of another pattern round

Reviews 4, 5 and 6 each blocked on new plain-wording phrasings of a false claim
that the validator's patterns did not cover, right next to what the previous
round had closed. A pattern-based reader will not run out of those, and the harm
in each was the same: a green pill reading "Todas as N figuras ligadas a um fato
estruturado" presented a miss as a verification.

Decision (the project owner's): **no state of the Copilot badge is green.** A run
that finds no divergence now reads "Nenhuma divergência detectada nas N figuras
(verificação por padrões, não é prova)" in the neutral colour; divergences and
causal claims stay red; an answer with no figure says so. The transcript section
says that no divergence detected is not proof the answer is right. The validator
itself is unchanged by this round; what changed is what the product claims about
it. A test asserts that no combination of results produces a green pill.

## Round 8 — the final review of `fce6110..41dabcb`

The final review confirmed the badge decision (no state certifies), round 6's
fixes for their exact shapes, the historical 54/72 figure, the gold gate and the
failure handling, and blocked on two claims the repository made that were not
true:

| # | Finding | Correction |
|---|---|---|
| I-1 (blocking) | `ask()` let the narrator replace `metrics_used` with any real bundle metric, while README, architecture doc, module header and page all said every structured field it returns is discarded | the override is deleted; the list is the engine's; the engine-owned-field test now compares it |
| I-2 (blocking, documentation) | the README's "known classes that still pass" omitted classes that pass in natural wording on checks the README names: level-as-change forms outside the patterns, period lengths in the wrong role, dates checked only for existence, a back-reference anywhere in the sentence | all listed; "binds every figure" corrected to say dates are checked only as dates; the length comment corrected |
| I-3 | "em função de" had been removed on a false premise (see round 6) | restored, ledger corrected; common causal connectives outside the list named in the README |
| I-4 | the red callout said the unbound figures have no such value in the bundle, which is false for true figures the patterns cannot bind | reworded: the patterns could not bind them, which may mean an absent value or an unrecognised wording |
| Minor | stale "16 mutation families"; screenshot 02 captioned "grounded answers"; the section-03 note without the pattern caveat | corrected |

Deferred with the reviewer's classification as minor: the experiment exception
is broader than the system prompt's rule (not reachable in the app, which never
attaches an experiment; now stated in the README); NaN `on_time_rate` and
delivery minutes are not checked by the gate; "porque" also flags true
explanations, and the restored "em função de" flags non-causal uses such as
"calculada em função da média diária" (both fail closed); `ValidationReport.all_verified`
keeps its name.

A scoped review of the four fixes (`41dabcb..a1d4829`) found them correct, each
test failing on the base, and every new README and ledger sentence true at HEAD —
clean. Its minors were applied after it: screenshot 02's caption named things the
image does not show; this paragraph did not record the "em função de" false
positive; `EvidenceBundle.metric_names()` was left without a caller and is
deleted; and the copilot.py module header and docs/architecture.md still said
every figure is bound, which dates are not.

## Deferred: MEDIUM / LOW

The reviewer reported 7 MEDIUM, 6 LOW and 10 silent bugs after review #1. **The
per-item catalogue is not in this repository**, so the table below cannot
reproduce it and does not pretend to. It records the items the two fix rounds
actually met, judged and left alone.

| # | Item | Severity | Why deferred |
|---|---|---|---|
| L1 | README §"A rendered Decision Memo" embeds an **English** rendering of a memo the engine writes in pt-BR. The figures agree; the language does not. | Low | Documentation-only; the artefact itself is correct. Retranslating a 70-line block is beyond a blocker round. |
| L2 | `components.ranking_note()` renders on two pages with identical text. | Low | One implementation, two call sites — intended. Whether the sentence belongs on both is a product question. |
| M1 | The decision page's session audit trail prints a **wall-clock** timestamp, while every engine artefact derives `generated_at` from `params.as_of` so two runs are byte-identical. | Medium | Session-only, never written to disk, never read by the engine. Divergent from the determinism convention all the same. |
| M2 | The Copilot bundle carries no segment KPI rows, so a zone-level metric *level* can only be grounded when an anomaly fired on that metric at that zone. | Medium | Now affordable — headroom went from 2,709 to 9,374 characters. Deferred because adding it is a grounding **improvement**, not a blocker repair, and it should be measured against the benchmark on its own. |
| I10 | The score-gap magnitude scanner's meta-test draws its banned set from the word list under test, and misses phrasings such as "larga vantagem" or "empate técnico". | Medium | Verified in the product: no page renders a qualitative gap claim, and `score_gap_to_next` removes the reason to write one. A word-list scanner cannot close an open vocabulary; widening it is not a release blocker. |
| M3 | `tests/test_app.py`'s analytics scanner (`BANNED`) contains no token for arithmetic, so page-level score subtraction would pass it. | Medium | The rule it guards is real and `score_gap_to_next` now removes the reason to subtract in a page. Widening a scanner to catch operators needs care to avoid false positives on unit conversion. |
| — | The remaining items from the reviewer's catalogue | — | **Not recoverable from this repository.** They must come from the reviewer's own report before they can be ledgered individually. |
