# Forbear — Architecture

## 1. Problem statement

Razorpay Subscriptions retries a failed payment automatically on three fixed
attempts — day T+1, T+2, T+3 after the original due date — per its documented
"Retry Configuration" behaviour for auto-debit subscriptions. If all three
fail, the subscription's status moves to `halted` and Razorpay does not
attempt the invoice again; per Razorpay's "Managing Failed Payments" guidance,
a domestic card cannot be manually charged once the e-mandate execution has
stopped firing. The invoice is not cancelled and not written off — it simply
has no automated path back to being paid.

For a merchant this shows up as a queue that only grows. An 8,000-subscriber
book at a typical 10% monthly involuntary-churn rate produces on the order of
800 failed debits a month. At an average invoice of ₹499, that is roughly
₹4 lakh a month sitting in `halted` subscriptions, worked by one person
manually calling or emailing customers with no system telling them which
ones are worth the call.

The tools that exist for this — dunning platforms, retry schedulers — treat
the objective as maximising recovery rate per invoice. That objective is
wrong. Recovering ₹499 from a customer who is annoyed enough by the contact
to cancel their ₹5,988/year subscription is not a win; it is a net loss of
₹5,489. Any policy that scores itself on invoices recovered, without netting
out the customers it drove away, will systematically over-contact. Forbear's
premise is that the correct unit of measurement is net customer value —
recovered revenue minus the LTV of customers who churn because of contact —
not recovery rate.

## 2. Architecture

Nine components, in the order a failed debit passes through them:

1. **Webhook ingestion** — receives Razorpay's `payment.failed` /
   `subscription.halted` events, verifies the HMAC signature, and writes an
   `at_risk_record`. Does not decide anything about the record; a malformed
   or duplicate event is rejected before it becomes a row.
2. **Failure classifier** — maps Razorpay's failure code to a reason category
   (insufficient funds, mandate revoked, card expired, etc.). Does not touch
   money or scheduling — it only labels.
3. **Uplift scorer** — estimates the incremental effect of contacting this
   customer, as a CATE (conditional average treatment effect) on net
   retention. Does not estimate absolute recovery probability as its primary
   output, and never writes to the execution path.
4. **Whittle indexer** — converts each record's CATE and remaining value into
   a single priority number under the attempt-budget constraint. Does not
   itself enforce the budget; it only ranks.
5. **Allocator** — takes the ranked list, spends the capped attempt budget on
   the highest-index records, and writes a skip entry with numbers for
   everything it declines. Does not check regulatory constraints and does not
   import `core/guard.py`.
6. **Guard** — the last gate before any outbound action, re-derived from the
   database immediately before execution: mandate status, attempt cap,
   cooldown, NPCI execution window, notification validity, duplicate-attempt
   check. Does not know anything about uplift scores, ranking, or the
   allocator's plan — it only knows the record and the constraints.
7. **Executor** — walks the plan in chronological slot order, asks the guard
   immediately before each action, and calls the injected gateway function.
   Does not decide whether an action is legal; that is the guard's job, asked
   fresh every time.
8. **Audit chain** — every transition, including every skip, gets a
   hash-linked entry keyed to the previous entry for that entity. Does not
   permit two writers to fork the chain — each append takes a
   transaction-scoped advisory lock on the entity first.
9. **Measurement harness** — replays one synthetic world through three
   strategies (Forbear, fixed schedule, unconstrained) and reports net value,
   recovery rate, churn rate, and attempts per strategy. Does not feed
   anything back into the allocator; it is read-only measurement.

**The separation invariant:** the allocator plans, the guard permits, and
they share no code. The guard imports only `forbear.config.limits` and
`forbear.models.models` — never the allocator, never anything under
`forbear.services`. If the allocator's ranking, its uplift model, or its
Whittle math is wrong, the only thing standing between that bug and a
customer's bank account is the guard, and the guard re-derives every
constraint from the database rather than trusting anything it is handed. No
LLM or model output reaches the execution path — models score, deterministic
code decides and executes.

## 3. Method

**Uplift modelling.** A T-Learner: two independent `GradientBoostingClassifier`
models, one fit on the treated (contacted) arm, one on the control arm, with
CATE taken as the difference of their predicted probabilities. A T-Learner
rather than a single learner with treatment as a feature, because a single
learner is free to ignore a weak treatment signal in favour of stronger
features — and given how much smaller the effect of contact is next to the
effect of being broke, it usually does. Trained on **net retention**
(`recovered AND NOT churned`), not gross recovery — trained on recovery
alone, a do-not-disturb customer who pays when chased scores zero uplift, and
the churn they took on the way out never shows up anywhere. See the T-Learner
choice and its failure modes in the uplift-modelling literature survey [2]
and the empirical benchmark comparing learner families under weak treatment
signal [3].

**Four segments.** Sure Thing (recovers regardless of contact), Persuadable
(recovers only if contacted), Lost Cause (does not recover either way),
Do-Not-Disturb (contact makes things worse — negative CATE). This is the
standard uplift taxonomy applied to a retention decision rather than a
marketing one [6].

**Whittle index.** A closed-form approximation — one period of value with the
continuation folded in — rather than a full Bellman solve over the restless
bandit. The derivation and threshold-optimality conditions this approximation
leans on are from Mate, Madaan, Suggala, Taneja et al., "Collapsing Bandits
and Their Application to Public Health Intervention" (NeurIPS 2020) [4]. A
full solver would recompute the continuation value at every horizon instead
of approximating it in closed form, which matters most for records with a
long remaining subscription term where the one-period approximation is
furthest from the discounted infinite-horizon answer.

**Evaluation.** Qini coefficient, measured on a held-out 30% that the model
never saw. `UpliftModel.fit_and_evaluate` splits stratified on the
treatment/outcome pair, fits on the remainder, and scores the holdout.

**Held-out Qini: 0.0907.** In-sample on the same data: 0.4243.

The in-sample figure overstates discrimination by roughly **4.7x**, and an
earlier version of this document reported 0.3749 — an in-sample number —
as though it were the model's performance. It was not. Gradient boosting
memorises, so scoring on the fitted rows measures memorisation rather than
ranking quality, and the gap here is most of the number rather than a
rounding detail. The held-out score is small but real: a shuffled ranking
scores near zero, so the model is finding signal, just far less of it than
the in-sample figure claimed.

## 4. Results

### n=500, five strategies (seed 42)

| Metric | fixed_schedule | forbear | forbear_constrained | classifier_only | unconstrained |
|---|---|---|---|---|---|
| ₹ recovered | 73,255 | 70,104 | 51,453 | 117,981 | 164,802 |
| Recovery rate | 29.0% | 29.2% | 19.4% | 43.8% | 59.6% |
| Attempts consumed | 1,183 | 223 | 150 | 367 | 1,268 |
| ₹ recovered/attempt | 0.123 | 0.655 | 0.647 | 0.597 | 0.235 |
| Records skipped | 45 | 277 | 350 | 133 | 133 |
| Customers churned | 36 | 2 | 1 | 27 | 27 |
| LTV lost to churn | 364,968 | 23,976 | 11,988 | 259,476 | 259,476 |
| **Net value** | **−291,713** | **+46,128** | **+39,465** | **−141,495** | **−94,674** |

### The ablation: does the model earn its complexity?

**The ablation result: `classifier_only` recovers ₹117,981 — more than
Forbear — and is worth ₹187,623 less, because it churns 27 customers instead
of 2. The classifier separates dead mandates from live ones. Only the uplift
model separates persuadable customers from those who cancel when chased.**

`classifier_only` is the control for the whole architecture. It runs the same
allocator, the same guard and the same executor, with one thing removed: the
value threshold. The uplift estimate influences no decision it makes, so
whatever separates the two columns is what the model and the Whittle index are
worth. Both policies skip the same terminal failure classes, because that part
belongs to the classifier and the ablation keeps it.

**`forbear_constrained` is positive in 10/10 seeds with a quarter of the
variance of the unconstrained configuration. The binding budget prevents the
allocator from spending attempts on marginal records once calibrated CATEs
stop skipping them. This is the condition the Whittle index exists to price,
and it was absent from every number this project reported before the
ablation.**

Across 10 seeds at n=500: constrained +44,983 ± 10,455, positive in 10/10;
unconstrained +31,030 ± 25,542, positive in 8/10. On the single seed tabled
above it holds 0.647 recovered per attempt against the unconstrained policy's
0.655, on 150 attempts instead of 223 — the index degrades gracefully when it
actually has to allocate rather than merely filter.

### n=10,000, five strategies (seed 42, 181s)

| Metric | fixed_schedule | forbear | forbear_constrained | classifier_only | unconstrained |
|---|---|---|---|---|---|
| ₹ recovered | 1,543,114 | 1,444,300 | 888,410 | 2,441,507 | 3,528,561 |
| Recovery rate | 31.4% | 36.0% | 17.9% | 44.9% | 62.4% |
| Attempts consumed | 23,873 | 6,186 | 3,000 | 7,671 | 26,179 |
| ₹ recovered/attempt | 0.131 | 0.582 | 0.597 | 0.586 | 0.238 |
| Customers churned | 729 | 205 | 98 | 615 | 615 |
| LTV lost to churn | 9,171,852 | 1,573,740 | 801,624 | 7,667,820 | 7,667,820 |
| **Net value** | **−7,628,738** | **−129,440** | **+86,786** | **−5,226,313** | **−4,139,259** |

Held-out Qini at this size: **0.1155** (in-sample 0.2298).

**The binding budget is what carries the thesis at scale.** Plain `forbear`
is net-negative here — −129,440, a 59x smaller loss than the fixed schedule
but a loss. `forbear_constrained`, the same policy under a ceiling of 0.3 × n,
clears zero at **+86,786**.

The reason is the mechanism in §5. As estimates calibrate, fewer records fall
below the value threshold, so an unconstrained Forbear schedules more marginal
attempts and pays their churn cost. A binding budget makes that impossible:
with only 3,000 attempts for 10,000 records, the Whittle index has to rank
rather than merely filter, and the marginal records lose to the good ones. It
consumes a quarter of the attempts, churns half as many customers as plain
Forbear, and is the only selective policy that clears zero at this size.

This is the condition under which an index-based allocator is doing something
a threshold could not — and every result this project reported before the
constrained strategy existed was produced without it.

### Sensitivity sweep

Nine points across the dunning-churn-per-contact rate, from 0% to 100%.
Crossover at **0.15**: below 15% churn per contact, chasing everything beats
being selective — the extra recovered rupees outweigh the churn cost. At or
above 15%, selectivity wins. Full points in `sweep.csv`, generator in
`forbear/services/sensitivity.py`.

### Adversarial suite

5/5 illegal actions blocked, each against a record the allocator actively
wanted (₹4,999, Whittle index +2118.6): revoked mandate, expired mandate,
attempt cap exceeded, NPCI execution-window violation, duplicate webhook
replay. A sixth test confirms the same record is permitted when clean, so the
suite is not passing by way of a guard that refuses everything.

## 5. Honest limitations

**Synthetic data circularity.** Customer profiles are conditioned on their
ground-truth causal segment — plan tier, tenure, and failure code all skew by
segment — so the uplift model has feature signal to find that it would not
have in production, where segment is never observed directly. Treatment
randomisation (independent Bernoulli per record, never correlated with
segment) and outcome noise (5% of sure-thing records fail anyway, 3% of
lost-cause records recover anyway) partially mitigate this by preventing the
model from trivially inferring segment from treatment-outcome pairs, but they
do not remove the underlying circularity: the generator and the model are
built from the same conceptual segmentation.

**Ranking quality is modest, and was previously reported as several times
better than it is.** In-sample Qini overstated discrimination by approximately
**4.7x**. The held-out figure is **0.09** (0.0907 at n=500, 0.1155 at
n=10,000) — still above the shuffled control (0.03), confirming real signal,
but a fifth of what the 0.3749 in an earlier draft of this document claimed.
Across 10 seeds the held-out score is 0.0515 ± 0.0456, and on the weakest seed
it touches −0.005: on some draws the ranking is no better than random. The
policy's value comes as much from the deterministic classifier and the budget
ceiling as from the model's ordering, and the ablation in §4 is what separates
those contributions.

**Do-not-disturb detection accuracy is 51%, and more data does not fix it.**
The system gets the sign right — negative CATE — but on individual records it
is barely better than a coin flip at identifying which specific customers are
do-not-disturb.

Improving this requires **better features — payment timing, app session data,
complaint history — not more rows.** Measured across 8 seeds, detection
accuracy went from 59% at n=2,000 to 55% at n=8,000: quadrupling the data made
it slightly *worse*, not better. An earlier version of this document called
"improving detection to 70% with production data" the first priority, which
was wrong in its mechanism — volume is not the binding constraint. The
features the model currently sees (plan tier, tenure, failure code, hour,
day-of-month, attempt count, days since contact) do not separate a customer
who resents being chased from one who simply forgot, and no amount of data
makes an absent signal present.

One mitigating measurement: only 13.3% of do-not-disturb records land in the
top half of the ranking, so the ordering is better than the 51% sign-accuracy
figure implies. Accuracy counts individual signs; the allocator only needs
the ranking.

**The churn coefficient is an assumption, not a measurement.** The
sensitivity sweep shows exactly where the decision boundary sits as a
function of dunning-churn-per-contact, but nothing in this codebase validates
what that rate actually is for a real merchant's customers. The 0.15
crossover is only as good as whatever rate gets plugged in for a live
deployment.

**At n=10,000, unconstrained Forbear is net-negative.** −129,440 rupees,
against the fixed schedule's −7,628,738. The thesis holds in relative terms —
59x smaller a loss — but the policy as originally configured loses money at
this size, and that is a finding rather than a footnote.

The same policy under a binding budget (`forbear_constrained`, 0.3 × n) clears
zero at **+86,786**, so the defect is in running the allocator without a
ceiling rather than in the allocator. That does not make the unconstrained
result go away: it was the configuration every earlier number in this project
was produced under.

**The mechanism, which is more useful than the number.** Skip share collapses
from **55.4% at n=500 to 38.1% at n=10,000**, and `negative_net_value` skips
roughly halve as a share of the book. Better-calibrated CATE estimates cross
zero less often: at n=500 the estimates are noisy, more records land below the
threshold, the system skips more, and churn stays at 0.4%. At n=10,000 the
model is better calibrated, fewer records are refused, churn rises to 2.1%,
and net value goes negative.

**So the n=500 profit is partly a small-sample artefact.** At small n, noisy
CATE estimates push more records below zero, so the system skips more and
churns less. At n=10,000, estimates calibrate, fewer cross zero, skip share
collapses from 55% to 38%, and churn rises. The constrained configuration
solves this by imposing an external ceiling: it does not rely on the estimates
falling below a threshold to stop spending, so calibration cannot erode it.

The policy is sound; the discrimination underneath it is too weak to carry the
policy at volume without that ceiling. That is the same conclusion the
held-out Qini of 0.0907 reaches from the other direction, and the two agreeing
is the reason to believe either.

The three `test_scale.py` assertions that encode these claims are marked
`xfail(strict=True)` rather than retuned or deleted. They are the claims the
project makes; recording that they currently fail, with the mechanism in the
reason string, is the honest arrangement. `strict=True` means that fixing
detection turns them red and forces the finding to be rewritten rather than
silently kept.

**Advisory lock ceiling in the measurement harness — resolved.**
`append_entry` takes a transaction-scoped advisory lock per entity so two
writers cannot fork the same audit chain. The harness runs an entire
comparison inside a single transaction, so locks accumulate across the whole
run: one world per strategy × n records each. On a default PostgreSQL
configuration (`max_locks_per_transaction=64`) that capped comparisons at
roughly 1,700 records.

Raising `max_locks_per_transaction` to **600** removed the ceiling: the full
n=10,000 comparison completes in **137 seconds** for three strategies, **181
seconds** for five. `test_scale.py` still computes the limit from the live
server settings and skips with the exact arithmetic if a machine cannot
support the run, so the suite stays honest on a default configuration. Adding
the fourth and fifth strategies raised the requirement to 5 × n locks, which
puts the computed ceiling (9,600 records here) just under n=10,000 — so the
scale tests skip again on this box even at 600, and the n=10,000 tables above
were produced by running the comparison directly. Production never had this constraint — each decision
commits and releases its lock immediately; it was a property of measuring a
whole cycle atomically. Committing per chunk would remove the dependency on
server configuration entirely and remains worth doing.

**Headline numbers are one seed, and the spread is wide.** Across 12 seeds at
n=500, Forbear's net value averaged 32,003 with a standard deviation of
23,230, and **2 of 12 seeds came out net-negative**. Seed 42, the number
quoted throughout this document, is roughly 0.6σ above the mean of its own
distribution. Two claims in particular do not survive averaging: recovered
rupees are −12.7% against the fixed schedule rather than −4.3%, and recovery
rate is −0.35pp rather than +0.2pp, worse in 6 of 12 seeds. The *relative*
thesis is robust — Forbear beat the fixed schedule in 12 of 12 seeds and the
unconstrained policy in 12 of 12 — but any single absolute figure should be
read as one draw. `run_multi_seed` now reports mean ± σ, and the README table
leads with it.

**The API has no authentication.** `/stream/run` triggers a full allocation
cycle for anyone who can reach the port, CORS is `allow_origins=["*"]`, and
the app runs DDL at startup to create its own schema. This is a deliberate
choice for a single-screen local demo and an unacceptable one anywhere else.
The stream endpoint rolls its transaction back and writes nothing permanent,
which limits the blast radius to compute.

**Cold start.** The allocator's history lookup needs prior debit records to
infer a customer's likely salary-timing window. Two seeded invoices per
customer is the current workaround in the harness; a customer with zero
payment history gets no timing signal and the allocator falls back to the
population default.

## 6. What I would do with production data

1. **Measure real dunning-churn-per-contact.** One number settles the
   sensitivity sweep and replaces the assumption in §5 with a measurement.
   This is the single highest-leverage next step.
2. **Add features that carry the do-not-disturb signal**, rather than more
   rows of the ones already there — payment timing relative to salary day, app
   session activity, complaint and support-ticket history. The measurement in
   §5 is that volume does not move detection accuracy; the current feature set
   does not contain the signal, and this is the change that would.
3. **Per-chunk commits in the harness** to remove the advisory-lock ceiling
   at scale, so measurement at n=10,000+ no longer needs a lock-capacity
   workaround.
4. **A/B test the allocator against the fixed-schedule retry** on a live
   merchant cohort, measuring net value — recovered revenue minus churned
   LTV — at a 90-day horizon, long enough to see churn that a shorter window
   would miss.

## 7. Regulatory constraints

The guard enforces, from `forbear/config/limits.py`:

- **Attempt cap** — a hard ceiling of 4 attempts against any one at-risk
  record, counting every attempt row regardless of outcome, with a minimum
  24-hour cooldown between attempts.
- **Execution windows** — NPCI-permitted debit windows in IST: before 10:00,
  13:00–17:00, and after 21:30.
- **Pre-debit notification window** — a notification is valid authorisation
  for a debit for 24 hours from being sent; an attempt outside that window is
  blocked as `notification_expired`.

**These values should be independently verified against the current NPCI
e-mandate circulars before any production use.** They were sourced
secondarily during this build and may be outdated or incomplete; this
codebase is not a substitute for checking the primary regulatory text.

## References

[2] Gutierrez & Gérardy, "A Unified Survey of Treatment Effect Heterogeneity
    Modelling and Uplift Modelling," 2024.

[3] Zhao & Harinen, "Bridging the Gap: Benchmarking 15 Uplift Modeling
    Methods," 2023.

[4] Mate et al., "Collapsing Bandits and Their Application to Public Health
    Intervention," NeurIPS 2020.

[6] Chen et al., "A Dynamic Framework for Causal User Profiling in Internet
    Lending," 2023.
