# Forbear

A recovery decisioning engine that optimises for net customer value, not recovery rate.

## The problem

Razorpay Subscriptions retries a failed payment on a fixed T+1, T+2, T+3 daily
schedule. Once those three attempts are exhausted, the subscription halts and
the unpaid invoice is never auto-charged again — a domestic card cannot be
manually charged after the mandate stops firing. The invoice has no automated
path back to being paid. A merchant either chases it by hand, invoice by
invoice, or writes it off.

## What Forbear does

Forbear classifies each failed debit, scores it by incremental recovery value
minus expected churn risk, and allocates a capped attempt budget only to the
records where contact actually changes the outcome. Everything else is
deliberately skipped, with a reason code and an audit entry — not silently
dropped, not chased on the hope it might work. The objective is net customer
value: recovering ₹499 from someone who cancels a ₹5,988/year subscription
over being chased is not a win.

## What it is now

Two surfaces sit on top of the same decision:

- **The merchant worklist** (`GET /worklist`, served at `/`) — chase / wait /
  leave-alone, one screen a non-technical operator opens every morning. Every
  bucket is produced by `forbear/services/decisioning.py` calling the real
  `forbear/services/allocator.py::allocate()` in preview mode
  (`commit=False`) right after ingestion — same uplift model, same Whittle
  index, same skip logic the harness measures, not a separate heuristic. The
  worklist endpoint itself runs no model and no allocator: it is a plain SQL
  read of decisions already persisted, which is what keeps it under the
  sub-200ms budget. `GET /worklist/protected` names the customers left alone
  who, per the simulator's ground truth, would have churned if chased and
  stayed subscribed because nobody contacted them — the value the system
  protected, not just recovered. That endpoint only works in demo mode, since
  it needs outcome data no real production system has; against a live
  database it reports itself unavailable rather than fabricating a number.
- **The measurement harness** (`/advanced`, `GET /stream/run`) — the
  technical decision stream, comparison table, and sensitivity sweep, for the
  deep-dive.

`allocate()`'s `commit=False`/`commit=True` modes are proven to return the
same plan for the same inputs by
`tests/test_allocator.py::test_preview_and_commit_plans_are_identical_for_scheduled_records`
(and its skipped-record counterpart) — the worklist and the harness are
provably one decision path, not two that happen to agree today.

The uplift model backing both surfaces is fitted once, offline, by
`scripts/train_model.py`, and loaded from disk at startup — never refit per
webhook, never refit per request.

## Results

Mean ± σ across **10 seeds** at n=500. One seed is one draw of a synthetic
world, and two of the first twelve tried came out net-negative — so the spread
is reported rather than a single favourable run.

| Metric | Fixed Schedule | Forbear (constrained) | Forbear (unconstrained budget) | Classifier Only |
|---|---|---|---|---|
| ₹ recovered | 78,786 ± 8,483 | 49,536 ± 6,499 | 70,333 ± 11,121 | 125,687 ± 10,467 |
| Recovery rate | 30.8% ± 1.4% | 19.9% ± 1.3% | 30.5% ± 3.1% | 44.5% ± 2.4% |
| Attempts consumed | 1,198 ± 21 | 150 ± 0 | 250 ± 23 | 388 ± 9 |
| ₹ recovered/attempt | 0.129 ± 0.006 | 0.663 ± 0.042 | 0.609 ± 0.036 | 0.574 ± 0.022 |
| Customers churned | 35 ± 5 | 1 ± 1 | 5 ± 2 | 30 ± 5 |
| LTV lost to churn | 429,781 ± 87,762 | 4,553 ± 6,310 | 39,302 ± 29,196 | 381,478 ± 75,712 |
| **Net value** | **−350,995 ± 83,849** | **+44,983 ± 10,455** | **+31,030 ± 25,542** | **−255,790 ± 73,128** |
| Net value positive in | 0/10 seeds | **10/10 seeds** | 8/10 seeds | 0/10 seeds |

**Under a binding budget, Forbear is net-positive in every seed tested. The
budget ceiling stops the allocator spending on marginal records — the
condition the Whittle index exists to price.**

The unconstrained-budget column stays because it is what the allocator does
without a ceiling, and the gap between the two columns is the argument for
having one: same policy, same model, a quarter of the variance and ~14,000
more rupees once the constraint binds.

**`classifier_only` is the ablation** — the same allocator, guard and executor
with the uplift model's value threshold removed, so the model influences
nothing it does. It recovers ₹125,687, more than either Forbear
configuration, and is worth ₹300,773 less than the constrained one, because
it churns 30 customers instead of 1. The classifier separates dead mandates
from live ones. Only the uplift model separates persuadable customers from
those who cancel when chased.

A chase-everything upper bound was measured too (recovers ₹183,702, net
−197,776); the full five-strategy tables are in
[docs/ARCHITECTURE.md §4](docs/ARCHITECTURE.md#4-results).

Held-out Qini across the same 10 seeds: **0.0515 ± 0.0456**.

<sub>**Single-seed (seed=42) numbers for reproducibility.** Forbear
(unconstrained budget): recovered 70,104 · recovery rate 29.2% · 223 attempts
· 0.655 per attempt · 2 churned · 23,976 LTV lost · net value **+46,128** ·
277 records skipped. Forbear (constrained): recovered 51,453 · 150 attempts ·
0.647 per attempt · 1 churned · 11,988 LTV lost · net value **+39,465** · 350
records skipped. Seed 42 sits about 0.6σ above the mean and was the number
previously headlined here.</sub>

### At scale (n=10,000)

At n=10,000 with a binding budget, Forbear clears zero (**+86,786**). Without
the budget it is net-negative (**−129,440**). The constrained configuration is
the recommended default.

Both beat the platform: the fixed schedule loses **−7,628,738** at this size,
so even the unconstrained configuration is a 59x smaller loss — but a loss.

The mechanism: skip share collapses from 55.4% at n=500 to 38.1% at n=10,000
as CATE estimates calibrate and fewer records cross below zero. At n=500 the
noise was doing some of the skipping, and it happened to skip profitably — so
part of the small-book profit is a small-sample artefact. Detection accuracy
for do-not-disturb is 51%, and **more data does not fix it** (59% at n=2,000
→ 55% at n=8,000); the missing ingredient is features, not rows. See
[docs/ARCHITECTURE.md §5](docs/ARCHITECTURE.md#5-honest-limitations).

### Sensitivity

Below 15% dunning churn per contact, chase everything — the extra recovery is
worth the churn risk. At or above it, be selective. See
[docs/ARCHITECTURE.md §4](docs/ARCHITECTURE.md#4-results) for the full sweep.

## Quick start

```bash
export FORBEAR_ADMIN_DSN=postgres:///postgres   # or your own DSN
psql -f schema.sql your_database

python scripts/run_demo.py          # adversarial suite, comparison, sweep, scale check
pytest                              # full suite: 414 passed, 8 skipped
```

`run_demo.py` creates and drops its own throwaway database — nothing above
needs to exist first beyond a running PostgreSQL server.

### Live-pipe demo: worklist end to end

Razorpay's test-mode dashboard isn't always available. In its place, a local
emitter (`forbear/emitter/`) builds webhook payloads in Razorpay's exact
documented shape, signs them with a real HMAC-SHA256 webhook secret, and POSTs
them at Forbear's real `/webhooks/razorpay` endpoint — the same signature
verification, replay detection, classification, and state-transition code a
real Razorpay delivery would hit. Only the sender is local: this is a
real-format payload through the real ingestion path, not a live Razorpay
account, because test-mode access to one was unavailable during this build.

```bash
export FORBEAR_DSN=postgres:///forbear
export RAZORPAY_WEBHOOK_SECRET=demo_secret
psql -f schema.sql forbear   # first run only

python scripts/train_model.py            # fits the uplift model once, writes models/uplift_model.pkl
uvicorn forbear.api.main:app --reload &
python scripts/run_live_demo.py          # emits signed webhooks at the running server
```

Open `http://localhost:8000/` for the worklist once the demo has run — chase,
wait, and leave-alone populated from real decisions made at ingestion, not
computed on page load. The technical decision stream moved to
`http://localhost:8000/advanced`.

The script truncates the database, emits a batch of mixed lifecycles
(insufficient-funds-then-recovers, insufficient-funds-then-halts,
revoked-mandate) at the running server, and reports how many events were
accepted, deduped, or rejected, and how many `at_risk_records` and audit
entries came out the other side.

`forbear/emitter/scenarios.py` also has named scenarios for a duplicate
delivery (replay) and a tampered signature (rejected with 401), and
`tests/test_emitter.py` round-trips the emitter's signatures against the real
receiver.

To see the protected-customers panel, seed a demo database with ground truth
via `forbear/services/demo_seed.py` (see `tests/test_demo_seed.py` for
usage) — it generates a synthetic batch, runs each record through the real
decision path, and separately records what would have happened if contacted,
which is what `GET /worklist/protected` reads. A database populated only by
real webhooks has no `demo_ground_truth` rows, so that endpoint reports
itself unavailable rather than a number.

## Stack

Python 3.11, FastAPI, PostgreSQL, scikit-uplift. No ORM for anything touching
money or attempt counts — explicit SQL only.

## Further reading

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — problem, method, results, limitations, what's next
- [docs/ENGINEERING_LOG.md](docs/ENGINEERING_LOG.md) — the bugs, with numbers
