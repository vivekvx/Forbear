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

**Evaluation.** Qini coefficient against the generator's ground-truth
counterfactuals (never visible to the scorer or allocator at runtime — only
to the offline evaluation code). Score: **0.3749**, against **0.0279** on the
same data with treatment labels shuffled. The model is finding real signal,
not memorising the generator's noise.

## 4. Results

### n=500

| Metric | Fixed Schedule | Forbear | Delta |
|---|---|---|---|
| ₹ recovered | 73,255 | 70,104 | −4.3% |
| Recovery rate | 29.0% | 29.2% | +0.2pp |
| Attempts consumed | 1,183 | 223 | −81% |
| ₹ recovered/attempt | 0.123 | 0.655 | +5.3x |
| Customers churned | 36 | 2 | −94% |
| LTV lost to churn | 364,968 | 23,976 | −93% |
| Net value | −291,713 | +46,128 | +337,841 |
| Records deliberately skipped | 0 | 277 | — |

### n=10,000

Forbear: net value **−129,440**. Fixed schedule: net value **−7,628,738**.
Forbear is net-negative at this scale, but loses **59x less** than the
baseline it replaces. The absolute loss traces to do-not-disturb detection
accuracy of 51% — the model gets the *direction* right (negative CATE) but
misses roughly half the individuals in that segment, so it still spends
attempt budget contacting customers it should have left alone.

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

**Do-not-disturb detection accuracy is 51%.** The system gets the sign right
— negative CATE — but on individual records it is barely better than a coin
flip at identifying which specific customers are do-not-disturb. This is the
single largest driver of the gap between n=500 (net positive) and n=10,000
(net negative): at 10x the volume, the false negatives in this segment
accumulate into real churn cost.

**The churn coefficient is an assumption, not a measurement.** The
sensitivity sweep shows exactly where the decision boundary sits as a
function of dunning-churn-per-contact, but nothing in this codebase validates
what that rate actually is for a real merchant's customers. The 0.15
crossover is only as good as whatever rate gets plugged in for a live
deployment.

**At n=10,000, Forbear is net-negative.** −129,440 rupees. The thesis holds
in relative terms — 59x smaller a loss than the fixed schedule — but absolute
profitability at this scale requires closing the do-not-disturb detection
gap with real data. This is stated plainly because it is the honest number,
not because it is comfortable.

**Advisory lock ceiling in the measurement harness.** `append_entry` takes a
transaction-scoped advisory lock per entity so two writers cannot fork the
same audit chain. The harness runs an entire comparison — three strategies,
one world each — inside a single transaction, so locks accumulate across the
whole run: three worlds × n records each. Past roughly 1,700 records on a
default PostgreSQL configuration (`max_locks_per_transaction=64`), the
harness cannot complete a comparison in one transaction. Production does not
hit this: each decision commits and releases its lock immediately. This is a
property of measuring a whole cycle atomically, not of the system being
measured — see the engineering log for the incident.

**Cold start.** The allocator's history lookup needs prior debit records to
infer a customer's likely salary-timing window. Two seeded invoices per
customer is the current workaround in the harness; a customer with zero
payment history gets no timing signal and the allocator falls back to the
population default.

## 6. What I would do with production data

1. **Measure real dunning-churn-per-contact.** One number settles the
   sensitivity sweep and replaces the assumption in §5 with a measurement.
   This is the single highest-leverage next step.
2. **Replace synthetic profiles with observed payment history.** The
   T-Learner's feature set (`forbear/scoring/uplift.py`) is already designed
   around observable signals — plan tier, tenure, prior failure codes — not
   ground-truth segment, so this is a data-source swap, not a redesign.
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
