# Engineering log

Bugs and findings from the build, in the order they were hit. Each entry
states the symptom, what I assumed, what was actually wrong, the fix, and
the numbers before and after where there are numbers to give.

---

### 2026-08-19 — Mutation testing exposed 16 winners of a `SELECT FOR UPDATE`

**Symptom.** A mutation test that deleted the `FOR UPDATE` clause from the
record-locking query still passed. It shouldn't have — two concurrent
allocators racing for the same record should produce one winner, not
multiple.

**Assumed.** The row lock alone was sufficient to serialise concurrent
transitions on one record.

**Actually wrong.** The query selected 16 candidate rows and locked all 16,
but the transition logic downstream picked among them without re-checking
which had already been claimed by a concurrent transaction between the lock
and the write. `FOR UPDATE` prevents two transactions from locking the same
row simultaneously; it does not prevent 16 different rows from each being
independently claimed by 16 different callers in the same instant.

**Fix.** Narrowed the query to lock and return exactly the one target row,
keyed by id, not a candidate set.

**Before/after.** Before: 16 winners possible per contested record under
concurrent load. After: mutation test correctly fails when `FOR UPDATE` is
removed — exactly 1 winner.

---

### 2026-08-19 — Advisory lock chain fork at audit index 1

**Symptom.** A concurrency test appending two audit entries for the same
entity from two connections produced two entries both claiming `prev_hash`
of the same parent — a forked chain at index 1, not a linear one.

**Assumed.** A row-level lock on the entity's latest audit row would
serialise appenders.

**Actually wrong.** There was no latest-audit row to lock yet in the case
that triggered this — the fork happened on the *first* append for an entity,
where no row exists to contend over. A row lock cannot serialise writers
racing to create the first row.

**Fix.** Switched to `pg_advisory_xact_lock` keyed on the entity id, taken
before either reading the previous hash or writing the new entry. This
serialises even the first-append race, since the lock exists independent of
any row.

**Before/after.** Before: reproducible fork on concurrent first-append.
After: 0 forks across the same concurrency test, run repeatedly.

---

### 2026-08-20 — Guard-before-attempt-row ordering, surfaced twice

**Symptom (harness).** Every scheduled action in a comparison run was
blocked with `duplicate_attempt`, including the very first action against a
record with zero prior attempts.

**Assumed.** The duplicate-attempt rule had a logic bug.

**Actually wrong.** The attempt row was being written *before* the guard was
asked, so the guard's own duplicate-detection query found the row that had
just been written for the action it was being asked to approve. The rule
was correct; the call order fed it its own pending write as evidence against
itself.

**Fix.** Reordered to guard-first, row-on-verdict: ask the guard, then write
the attempt row only after (and reflecting) the verdict.

**Before/after.** Before: 100% blocked on `duplicate_attempt`. After: 0
false blocks from this cause.

**Symptom (executor, later, same root cause in a different module).** Same
signature — everything blocked as a duplicate — appeared again days later
in `execute_plan`, independently of the harness fix, because the executor
had its own copy of the same ordering mistake.

**Fix.** Same reorder, applied in `forbear/services/executor.py`.

---

### 2026-08-21 — Execution order: plan order vs chronological, 197/223 blocked

**Symptom.** A comparison run against a fully valid plan blocked 197 of 223
scheduled attempts on stale-notification and out-of-window errors, despite
every individual action being legal in isolation.

**Assumed.** The plan itself contained illegal actions.

**Actually wrong.** The executor processed actions in Whittle-rank order —
highest priority first — and the virtual clock only moves forward. The
top-ranked record's slot happened to fall on day 30 of the simulation.
Executing it first dragged the clock to day 30, which put every
lower-ranked record's earlier notification window — day 3, day 7, whatever
it was — in the past by the time its turn came.

**Fix.** Sort actions chronologically by scheduled slot before execution,
independent of rank order. Rank determines *what* gets budget; slot time
determines *when* it runs.

**Before/after.** Before: 197/223 blocked (88%). After: 0 blocked for this
reason on the same plan.

---

### 2026-08-23 — Allocator history lookup: nested loop at n=10,000, 84 minutes

**Symptom.** A comparison run at n=10,000 did not complete inside a 90-minute
window; profiling isolated one query — the allocator's `_success_patterns`
history lookup — consuming 84 minutes of database CPU on its own.

**Assumed.** The bottleneck would be the uplift model's `.fit()` call, per
the expectation going in.

**Actually wrong.** The query filtered with `WHERE customer_id = ANY($1::bigint[])`
against 10,000 ids, run against tables that had just been bulk-loaded and
never `ANALYZE`d. With no statistics, the planner assumed the array was
small and chose a nested loop over customers, turning an O(n) lookup into
something closer to O(n²) against an unindexed assumption.

**Fix.** Rewrote the filter as a join against `unnest($1::bigint[])`, which
gives the planner a relation it can size, plus an explicit `ANALYZE` on the
bulk-loaded tables before the allocator runs.

**Before/after.** Before: did not finish in 84+ minutes at n=10,000. After:
26.5 seconds at n=2,000 (the size first re-tested at); the uplift model fit,
the originally suspected bottleneck, takes single-digit seconds.

---

### 2026-08-23 — Advisory lock ceiling at n=10,000, `OutOfMemoryError`

**Symptom.** Following the query fix above, the same n=10,000 run failed 22
minutes in with `asyncpg.exceptions.OutOfMemoryError: out of shared memory`,
hinting `max_locks_per_transaction`.

**Assumed.** The remaining bottleneck, if any, would also be a query-planner
issue.

**Actually wrong.** `append_entry` takes one `pg_advisory_xact_lock` per
entity, held until the transaction ends (see the 2026-08-19 entry — this is
the same mechanism that fixed the chain-fork bug, now hitting its own
ceiling). The harness runs an entire comparison — three strategies × n
records — inside a single transaction, so locks accumulate: 3 × 10,000 =
30,000 requested against a default capacity of `max_locks_per_transaction (64)
× (max_connections (100) + max_prepared_transactions)` ≈ 6,400.

**Fix.** Added `one_transaction_record_limit(conn)`, which reads the live
Postgres settings and computes the safe ceiling; `test_scale.py` now checks
it and skips with the exact arithmetic in under 2 seconds instead of failing
22 minutes in. Not fixed at n=10,000 itself — that needs either
`max_locks_per_transaction` raised and Postgres restarted, or the harness
refactored to commit per chunk (see architecture doc §6). Production is
unaffected: each decision commits and releases its lock immediately, so
locks never accumulate this way outside of the harness's single-transaction
measurement design.

**Before/after.** Before: failure at 22m16s with an opaque out-of-memory
error. After: skip in ~2s naming the exact record ceiling (1,706 records on
the default configuration) and what raises it.

**Resolved, 2026-08-25.** `max_locks_per_transaction` raised to **600**. The
full n=10,000 comparison now completes in **137 seconds** — the earlier
84-minute figure was the nested-loop query above, not this. Two consequences
worth recording: the three scale assertions stopped skipping and started
genuinely failing, which is how the net-negative result became a hard test
result rather than a note; and adding the fourth and fifth strategies raised
the lock requirement to 5 × n, so the computed ceiling (9,600 on this box)
now sits just under n=10,000 and the scale tests skip again. Committing per
chunk would end the dependency on server configuration for good.

---

### 2026-08-16 — Generator fix: segments invisible to observable features

**Symptom.** The uplift model's estimated CATE for the do_not_disturb
segment came out at **+0.089** — positive, when it must be negative by
definition (contact should make outcomes worse for this segment).

**Assumed.** The model or its hyperparameters needed adjustment. (Explicitly
ruled out per project constraint: hyperparameters are not to be tuned to fix
a directional result — the constraint deliberately does not allow this
escape hatch.)

**Actually wrong.** Segments were assigned independently of every observable
covariate in the profile generator. Plan tier, tenure, and failure code were
all drawn without reference to segment, so none of the seven features the
model sees carried any segment signal. With no separating feature, both arms
of the T-Learner converged to the same population-average prediction and
CATE collapsed toward zero-with-noise for every segment, including one whose
sign is supposed to be reliably negative.

**Fix.** Conditioned the generator on segment: `SEGMENT_PLAN_WEIGHTS` skews
do_not_disturb toward higher-value, longer-tenure plans and toward different
failure-code tilts than lost_cause. This is a generator fix, not a model fix
— the constraint that ruled out hyperparameter tuning was correct, and the
actual bug was upstream of the model entirely.

**Before/after.** Before: do_not_disturb CATE = +0.089 (wrong sign). After:
negative, and stable in sign across 5 different seeds.

---

### 2026-08-18 — Outcome redefinition: recovered AND NOT churned

**Symptom.** Before this change, the allocator had no way to prefer skipping
a record where contact would recover the invoice but lose the customer — the
uplift model's target only ever measured recovery.

**Assumed.** Recovery probability alone was a sufficient training target,
with churn handled as a separate downstream adjustment.

**Actually wrong.** Training on recovery alone makes contact look
non-negative in every case: a do_not_disturb customer who pays when chased
scores a *positive* recovery outcome, and the churn event that follows is
invisible to a target that never encodes it. There is no way to bolt churn
on afterward to a model that was never shown it during fitting — the sign
flip has to come from training data, not a post-hoc weight.

**Fix.** Redefined the fit target as the conjunction `recovered AND NOT
churned`, so a customer who pays and then leaves scores as a negative
outcome of contact, not a positive one.

**Before/after.** This redefinition is what makes negative CATE for
do_not_disturb possible at all — see the 2026-08-16 entry, which fixed a
different (generator-side) bug that was masking the same underlying segment
regardless of this target fix.

---

### 2026-08-25 — The guard's notification rule was inverted

**Symptom.** An audit probe asked the guard to permit a debit at a range of
notice periods. It permitted one sent **10 seconds** earlier and refused the
one sent 25 hours earlier — the only compliant case in the set.

**Assumed.** The rule enforced NPCI's pre-debit notification requirement,
since it was named for it, had a dedicated constant, and had passing tests.

**Actually wrong.** The bound pointed the wrong way. The rule accepted
`now - 24h <= sent_at <= now`, which reads "sent within the last 24 hours."
NPCI requires the customer to be warned *before* the debit, with at least a
day to act — the notification must have **aged past** 24 hours, not stayed
under it. Both existing tests (`older_than_24h`, `exactly_24h_old`) probed
only the expiry end, so the entire lead-time requirement was untested and the
rule enforced approximately its opposite.

Worse, `plant_target` in the adversarial suite defaulted to a **1-hour-old**
notification, so the "clean, permitted" record that every attack was measured
against — and that the sixth test used to prove the guard was not simply
refusing everything — was itself illegal to charge.

**Fix.** Replaced `NOTIFICATION_VALIDITY` with two bounds pointing in
opposite directions: `NOTIFICATION_MIN_LEAD` (24h, compared strictly, since
"at least 24 hours" is not satisfied by exactly 24 hours) and
`NOTIFICATION_MAX_AGE` (7 days, so one notification cannot authorise debits
forever). The allocator's scheduling window inverted to match — an existing
notification now opens a window that may not have started yet — and the
harness's notification stub moved from 2 hours' lead to 25.

**Before/after.** Before: debit permitted at 10s, 1h, 23h, and 24h of notice;
refused at 25h. After: refused at 10s, 1h, 23h, 24h; permitted at 25h;
refused again at 8 days. Six boundary cases now pinned by parametrised test,
plus a sixth adversarial attack.

**Uncomfortable side effect, and the point of the exercise.** With the rule
corrected, the harness's own notification stub became non-compliant and the
guard blocked **223 of 223** Forbear attempts — every strategy in the family
scored zero. The harness had never complied; nothing had been able to say so.
Fixing the stub restored the numbers unchanged.

---

### 2026-08-25 — Every Qini score in the repository was in-sample

**Symptom.** Qini *fell* as data grew: 0.3610 at n=2,000 against 0.2185 at
n=8,000, averaged over 8 seeds. More data making a model look worse is
backwards.

**Assumed.** A generator problem, or noise in the smaller sample.

**Actually wrong.** `test_uplift.build_dataset` fitted on `X` and then called
`predict_cate(X)` — the same rows. Every Qini the project reported measured
memorisation, not discrimination, and gradient boosting memorises a small
sample more completely than a large one, which is why the number fell as n
rose. A grep for `train_test_split`, `holdout`, and `cross_val` across
`forbear/` and `tests/` returned nothing: no out-of-sample evaluation existed
anywhere.

**Fix.** `UpliftModel.fit_and_evaluate` — a 70/30 split stratified on the
treatment/outcome pair (treatment alone can hand the training half an arm
with one outcome class, which `fit` refuses), fitting on the majority,
scoring both halves, and returning both numbers so the gap stays visible. The
test that asserted in-sample Qini was replaced with a held-out one, and a
second test asserts the in-sample figure exceeds it, so the defect cannot
return unnoticed.

**Before/after.** Reported 0.3749 (in-sample). Actual held-out **0.0907**,
against 0.4243 in-sample on the same data — the published figure overstated
discrimination by **4.7x**. Still above a shuffled control near zero, so the
signal is real; there is simply about a fifth as much of it as claimed.

---

### 2026-08-25 — The Whittle index had never operated under a binding budget

**Symptom.** Across 12 seeds at n=500, the skip reason
`batch_budget_exhausted` fired **zero** times. Skips were 55.1%
`negative_net_value` and 44.9% `terminal_failure_class`, and nothing else.

**Assumed.** The budget was binding in normal operation; the constant was
there and one test exercised it.

**Actually wrong.** `batch_budget` defaults to `None` in both
`AllocationConfig` and `HarnessConfig`, and only one test
(`test_harness.py`, n=120) ever set it. Every reported number — n=500,
n=10,000, the sensitivity sweep, the demo — ran with no ceiling. A Whittle
index is the Lagrangian price of an activation constraint; with no
constraint it decides sign and nothing else, and the ranking never has to
choose between two records the policy wants. The published results would have
been reproduced by `CATE > 0`.

**Fix.** Two new strategies. `forbear_constrained` runs the allocator with
`batch_budget = 0.3 × n`, which makes the constraint bind.
`classifier_only` is the ablation: same allocator, same guard, same executor,
`minimum_index = -inf`, so the model influences no decision it makes and what
remains is the classifier's own rules.

**Result, at n=500, seed 42.** forbear **+46,128**; classifier_only
**−141,495**; a difference of **187,623** on a 252,350-rupee book. The
ablation recovers *more* money — 117,981 against 70,104 — and is worth far
less, because it churns 27 customers against 2. The classifier separates dead
mandates from live ones; only the uplift model separates a persuadable
customer from one who cancels when chased. Under a binding budget,
`forbear_constrained` holds 0.647 recovered per attempt against 0.655
unconstrained, on 150 attempts instead of 223, and stays net-positive at
+39,465.

The model earns its complexity. Had it not, that would have gone here too.

**And the budget turned out to matter more than expected.** At n=10,000,
plain `forbear` is net-negative (−129,440) while `forbear_constrained` clears
zero (**+86,786**) on a quarter of the attempts. The ceiling is what stops the
allocator spending on marginal records once calibrated estimates stop skipping
them — the constraint the Whittle index exists to price, absent from every
result this project reported until now. Across 10 seeds at n=500 the same
pattern holds: constrained is positive in 10/10 seeds against unconstrained's
8/10, at a quarter of the variance.

---

### 2026-08-24 — Net-negative finding at n=10,000

**Symptom / finding, not a bug.** At n=500, Forbear's net value is positive
(+46,128). At n=10,000, run after both the query-planner and advisory-lock
fixes above, net value is **−129,440** — negative, despite still beating the
fixed-schedule baseline by 59x (−7,628,738).

**Investigated whether this was a bug.** Checked for regressions in the
allocator's ranking, the Whittle index computation, and the skip logic at
scale — none found; the skip share stays proportional to the book size
(within the harness's stability tolerance), and the ranking order is
unchanged.

**Actual cause.** Not a bug — a real property of the do-not-disturb
detection accuracy (51%, see architecture doc §5). At 20x the volume, the
absolute count of false negatives in that segment scales linearly, and their
churn cost accumulates faster than the correctly-identified segments'
recovered value does. Recorded as a finding rather than patched, per the
project's standing instruction not to patch tests or numbers to make an
inconvenient result disappear — the fix belongs in detection accuracy with
better features, not in this codebase's arithmetic.

**Correction, 2026-08-25.** This entry originally claimed the skip share
"stays proportional to the book size (within the harness's stability
tolerance)". That was measured at n=1,500 and is false at n=10,000, where the
full run now completes: skip share falls from **55.4% at n=500 to 38.1%**,
and `test_the_skip_list_scales_with_the_book` fails by 17.3 points against a
15-point tolerance.

The drift is the mechanism behind the net-negative result rather than a
separate problem. Small samples produce noisier CATE estimates, more of them
land below zero, and the system skips more than better-calibrated estimates
would justify. Some of the n=500 profit was noise skipping in a profitable
direction. Both this assertion and the net-value one are now
`xfail(strict=True)` with the mechanism in the reason string.
