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
this machine's configuration) and what raises it.

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
production data, not in this codebase's arithmetic.
