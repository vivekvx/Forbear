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

## Results (n=500)

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

Same recovered rupees, a fifth of the attempts, and net value flips from a
loss to a gain because it stops paying churn cost for a marginal 0.2 points of
recovery rate.

### At scale (n=10,000)

Forbear is net-negative at this scale: **−129,440**, against fixed schedule's
**−7,628,738** — 59x smaller a loss, not a win. The gap is do-not-disturb
detection accuracy, which sits at 51% on synthetic data. Improving it with
production payment history is the first priority; see
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
