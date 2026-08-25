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
