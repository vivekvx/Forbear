"""Synthetic customers, defined by how they respond to contact.

Every number Forbear reports comes out of this generator, so the one thing it
must not do is bake in the assumptions the uplift model is meant to discover.
The segments below are the causal structure of the world; the model never sees
them, and treatment is assigned at random rather than by anything correlated
with them (see treatment_assignment).

Four segments, following the standard uplift taxonomy:

    sure_thing       pays on the next cycle whether or not anyone calls
    persuadable      pays only if contacted; the only segment worth spending on
    lost_cause       does not pay either way
    do_not_disturb   would have paid, and contact makes them cancel

The last one is why Forbear optimises net value rather than recovery rate. A
policy that maximises recovery contacts everyone and quietly burns the
do_not_disturbs, and the recovery-rate metric never shows the cost.

Nothing in this package may be imported by scoring/ or services/. The profiles
here are the answer key.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from forbear.models.models import MandateStatus


class Segment(str, Enum):
    SURE_THING = "sure_thing"
    PERSUADABLE = "persuadable"
    LOST_CAUSE = "lost_cause"
    DO_NOT_DISTURB = "do_not_disturb"


# The population mix. Chosen so persuadables are a minority of the book: if
# most records were persuadable, targeting would beat blanket contact by so
# much that the evaluation would flatter any policy at all.
SEGMENT_MIX: dict[Segment, float] = {
    Segment.SURE_THING: 0.40,
    Segment.PERSUADABLE: 0.25,
    Segment.LOST_CAUSE: 0.20,
    Segment.DO_NOT_DISTURB: 0.15,
}

# Indian salary credits bunch at month start, the first weekly-ish payout, mid
# month, and the pre-month-end run. A uniform salary_day would make the
# time-dependent recovery signal far easier to learn than it is in production.
SALARY_ANCHORS = (1, 7, 15, 25)
SALARY_ANCHOR_WEIGHTS = (0.38, 0.14, 0.26, 0.22)

# Monthly income in paise: lognormal, so the tail is long and the median sits
# well below the mean. sigma=0.55 puts roughly the middle half of the book
# between ~30k and ~65k rupees.
INCOME_MEDIAN_PAISE = 45_000_00
INCOME_LOG_SIGMA = 0.55
INCOME_FLOOR_PAISE = 8_000_00

# Subscription plans, in paise. Real books are a few tiers, not a continuum.
PLAN_AMOUNTS_PAISE = (149_00, 299_00, 499_00, 999_00, 1_999_00)
PLAN_WEIGHTS = (0.30, 0.30, 0.22, 0.13, 0.05)

# Plan mix and tenure, conditioned on segment.
#
# Without this the segments are statistically invisible: nothing a scoring
# model can observe would correlate with how a customer responds to contact,
# every estimated treatment effect would collapse to the population average,
# and the do_not_disturbs would score positive along with everyone else. That
# is not a hard problem, it is an impossible one, and a book where it held
# would mean recovery targeting cannot work at all - there would be no product.
#
# So the segments leave a trace in what the system can actually see. Premium,
# long-tenured customers are likelier to resent being chased; the customers who
# need a nudge are newer; the ones who are broke churn out young and cheap. The
# distributions overlap heavily on purpose. A model should be able to find the
# structure and should never be able to recover the label - that gap is what
# keeps the evaluation from being a memorisation test.
SEGMENT_PLAN_WEIGHTS: dict[str, tuple[float, ...]] = {
    "lost_cause": (0.40, 0.32, 0.18, 0.08, 0.02),
    "persuadable": (0.34, 0.32, 0.20, 0.11, 0.03),
    "sure_thing": (0.26, 0.30, 0.24, 0.15, 0.05),
    "do_not_disturb": (0.05, 0.12, 0.25, 0.33, 0.25),
}

# Geometric tenure; the parameter is 1/mean months. Long tails, so a young
# do_not_disturb and an old lost_cause both remain perfectly ordinary.
SEGMENT_TENURE_RATE: dict[str, float] = {
    "lost_cause": 0.14,
    "persuadable": 0.08,
    "sure_thing": 0.045,
    "do_not_disturb": 0.022,
}


@dataclass(frozen=True)
class CustomerProfile:
    """One synthetic customer. Frozen: a profile is an input, not state.

    contact_sensitivity is the per-contact probability that this customer
    cancels when chased. It is zero for every segment except do_not_disturb,
    which is what makes that segment's churn cost a property of the customer
    rather than a global constant.
    """

    customer_id: str
    segment: Segment
    salary_day: int  # 1-28; 29-31 excluded because February has no such day
    salary_variance_days: int  # 0-5 jitter either side of salary_day
    monthly_income: int  # paise
    monthly_spend_rate: float  # fraction of income already committed
    plan_amount: int  # paise
    mandate_status: MandateStatus
    subscription_age_months: int
    contact_sensitivity: float


def _segment_labels(n: int) -> list[Segment]:
    """Exact counts for SEGMENT_MIX, largest remainder; the caller shuffles.

    Drawing each segment independently would be more lifelike, but it also
    means a small batch lands a couple of standard deviations off the mix and a
    run's headline numbers move for a reason that has nothing to do with the
    policy. The mix is a property of the scenario, so it is fixed exactly and
    only its assignment to customers is random.
    """
    exact = {segment: n * share for segment, share in SEGMENT_MIX.items()}
    counts = {segment: int(value) for segment, value in exact.items()}

    shortfall = n - sum(counts.values())
    by_remainder = sorted(
        exact, key=lambda segment: exact[segment] - counts[segment], reverse=True
    )
    for segment in by_remainder[:shortfall]:
        counts[segment] += 1

    labels: list[Segment] = []
    for segment, count in counts.items():
        labels.extend([segment] * count)
    return labels


def _salary_days(rng: np.random.RandomState, n: int) -> np.ndarray:
    """Anchor clusters with a couple of days of spread, clipped to 1-28."""
    anchors = rng.choice(SALARY_ANCHORS, size=n, p=SALARY_ANCHOR_WEIGHTS)
    spread = rng.randint(-2, 3, size=n)
    return np.clip(anchors + spread, 1, 28)


def _salary_variance_days(rng: np.random.RandomState, n: int) -> np.ndarray:
    """How reliably the money lands on the day it is supposed to.

    Most salaried customers are 0-1 days off. The tail at 4-5 days is the gig
    and commission earners, whose recovery timing is genuinely hard to predict
    and who should stay hard to predict here too.
    """
    return rng.choice(
        [0, 1, 2, 3, 4, 5], size=n, p=[0.34, 0.26, 0.16, 0.11, 0.08, 0.05]
    )


def _spend_rates(rng: np.random.RandomState, segments: list[Segment]) -> np.ndarray:
    """Committed share of income, drawn per segment.

    Lost causes are drawn from a distribution centred above 1.0: they are not
    ignoring the invoice, they are broke, and no amount of contact creates
    money that does not exist. Everyone else is centred below 1.0. This is the
    only place a segment leaks into an observable feature, and it should: a
    model that finds it has found something real about being broke, not the
    answer key.
    """
    n = len(segments)
    is_broke = np.array([segment is Segment.LOST_CAUSE for segment in segments])
    rates = np.where(
        is_broke,
        rng.normal(1.02, 0.10, size=n),
        rng.normal(0.78, 0.14, size=n),
    )
    return np.clip(rates, 0.20, 1.45)


def _mandate_statuses(rng: np.random.RandomState, n: int) -> list[MandateStatus]:
    """Mostly live mandates, with the dead ones a real book carries."""
    codes = rng.choice(
        [
            MandateStatus.ACTIVE.value,
            MandateStatus.PAUSED.value,
            MandateStatus.EXPIRED.value,
            MandateStatus.REVOKED.value,
        ],
        size=n,
        p=[0.88, 0.04, 0.05, 0.03],
    )
    return [MandateStatus(code) for code in codes]


def generate_profiles(
    n: int, seed: int, contact_sensitivity: Optional[float] = None
) -> list[CustomerProfile]:
    """n customers with a fixed segment mix and randomised everything else.

    contact_sensitivity pins every do_not_disturb to the same churn-per-contact
    probability instead of drawing one. Only the sensitivity sweep uses it: to
    ask where the allocator's decisions flip, the churn coefficient has to be
    the one thing moving, and a distribution of tempers would blur exactly the
    boundary the sweep exists to locate.

    Reproducible by construction: one explicitly seeded RandomState, no use of
    the global numpy random state, and no dependence on set iteration order.
    The same seed must give identical profiles on every run, or nothing
    downstream of this file can be compared between runs.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if contact_sensitivity is not None and not 0.0 <= contact_sensitivity <= 1.0:
        raise ValueError(
            f"contact_sensitivity is a probability, got {contact_sensitivity}"
        )

    rng = np.random.RandomState(seed)

    segments = _segment_labels(n)
    rng.shuffle(segments)

    salary_days = _salary_days(rng, n)
    variance_days = _salary_variance_days(rng, n)
    incomes = np.maximum(
        rng.lognormal(np.log(INCOME_MEDIAN_PAISE), INCOME_LOG_SIGMA, size=n),
        INCOME_FLOOR_PAISE,
    )
    spend_rates = _spend_rates(rng, segments)
    statuses = _mandate_statuses(rng, n)
    # Plan and tenure are drawn per segment, so the segments are learnable from
    # observables without ever being readable off them. See the comment on
    # SEGMENT_PLAN_WEIGHTS: with an unconditional draw there is no signal at
    # all, and every treatment effect estimate collapses to the population mean.
    plans = np.array(
        [
            rng.choice(PLAN_AMOUNTS_PAISE, p=SEGMENT_PLAN_WEIGHTS[segment.value])
            for segment in segments
        ]
    )
    # Geometric, so most of the book is young and a thin tail is old.
    ages = np.clip(
        [rng.geometric(SEGMENT_TENURE_RATE[segment.value]) for segment in segments],
        1,
        60,
    )
    # Only do_not_disturbs have a churn response to contact; drawn per customer
    # so the segment has a spread of tempers rather than one global rate. The
    # draw happens either way, so pinning the value does not shift the random
    # stream and a swept run stays comparable to an unswept one.
    drawn = rng.uniform(0.25, 0.75, size=n)
    sensitivities = (
        drawn if contact_sensitivity is None else np.full(n, contact_sensitivity)
    )

    return [
        CustomerProfile(
            customer_id=f"cust_{index:06d}",
            segment=segment,
            salary_day=int(salary_days[index]),
            salary_variance_days=int(variance_days[index]),
            monthly_income=int(incomes[index]),
            monthly_spend_rate=round(float(spend_rates[index]), 4),
            plan_amount=int(plans[index]),
            mandate_status=statuses[index],
            subscription_age_months=int(ages[index]),
            contact_sensitivity=(
                round(float(sensitivities[index]), 4)
                if segment is Segment.DO_NOT_DISTURB
                else 0.0
            ),
        )
        for index, segment in enumerate(segments)
    ]
