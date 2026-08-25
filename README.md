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

## Results

Mean ± σ across **10 seeds** at n=500. One seed is one draw of a synthetic
world, and two of the first twelve tried came out net-negative — so the spread
is reported rather than a single favourable run.

| Metric (mean ± σ) | fixed_schedule | forbear | forbear_constrained | classifier_only | unconstrained |
|---|---|---|---|---|---|
| ₹ recovered | 78,786 ± 8,483 | 70,333 ± 11,121 | 49,536 ± 6,499 | 125,687 ± 10,467 | 183,702 ± 8,813 |
| Recovery rate | 30.8% ± 1.4% | 30.5% ± 3.1% | 19.9% ± 1.3% | 44.5% ± 2.4% | 62.7% ± 1.8% |
| Attempts consumed | 1,198 ± 21 | 250 ± 23 | 150 ± 0 | 388 ± 9 | 1,328 ± 26 |
| ₹ recovered/attempt | 0.129 ± 0.006 | 0.609 ± 0.036 | 0.663 ± 0.042 | 0.574 ± 0.022 | 0.236 ± 0.005 |
| Customers churned | 35 ± 5 | 5 ± 2 | 1 ± 1 | 30 ± 5 | 30 ± 5 |
| LTV lost to churn | 429,781 ± 87,762 | 39,302 ± 29,196 | 4,553 ± 6,310 | 381,478 ± 75,712 | 381,478 ± 75,712 |
| **Net value** | **−350,995 ± 83,849** | **+31,030 ± 25,542** | **+44,983 ± 10,455** | **−255,790 ± 73,128** | **−197,776 ± 75,157** |
| Net value positive in | 0/10 seeds | 8/10 | **10/10** | 0/10 | 0/10 |

Forbear recovers slightly *less* money than the platform's fixed schedule and
is worth ~380,000 rupees more, because it churns 5 customers instead of 35.
Recovery rate is the wrong objective; that gap is the whole argument.

**`classifier_only` is the ablation** — the same allocator, guard and
executor with the uplift model's value threshold removed. It recovers the most
money of any selective policy and loses 255,790, because without the model it
cannot tell a persuadable customer from one who cancels when chased. The
difference between it and Forbear is what the model and the Whittle index are
worth: about **287,000 rupees** on a 252,000-rupee book.

**`forbear_constrained`** runs under a binding budget of 0.3 × n. It is the
only policy positive in every seed, with a quarter of the variance — the
attempt ceiling forces the ranking to do real work instead of merely deciding
sign.

Held-out Qini across the same 10 seeds: **0.0515 ± 0.0456**.

<sub>Seed 42 alone, for reproducibility: recovered 70,104 · recovery rate 29.2% ·
223 attempts · 0.655 per attempt · 2 churned · 23,976 LTV lost · net value
**+46,128** · 277 records skipped. This single seed sits about 0.6σ above the
mean and was the number previously headlined here.</sub>

### At scale (n=10,000)

Unconstrained Forbear is net-negative at this scale: **−129,440**, against
fixed schedule's **−7,628,738** — 59x smaller a loss, not a win. Under a
binding budget of 0.3 × n, the same policy clears zero at **+86,786**, on a
quarter of the attempts.

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
pytest                              # full suite
```

`run_demo.py` creates and drops its own throwaway database — nothing above
needs to exist first beyond a running PostgreSQL server.

## Stack

Python 3.11, FastAPI, PostgreSQL, scikit-uplift. No ORM for anything touching
money or attempt counts — explicit SQL only.

## Further reading

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — problem, method, results, limitations, what's next
- [docs/ENGINEERING_LOG.md](docs/ENGINEERING_LOG.md) — the bugs, with numbers
