"""Random assignment to contact or no-contact.

This is the smallest file in the generator and the reason any of its numbers
mean anything. If treatment were assigned by the policy, or by any feature the
policy uses, then the difference in recovery between contacted and uncontacted
records would be confounded by whatever got them contacted: the model would be
recovering its own targeting rule and reporting it as uplift.

Assignment is a coin flip per record, independent of segment, amount, decline
code, and anything the allocator would consider. The uplift model trains on
outcomes observed under this assignment; only afterwards does the policy get to
target selectively, and its value is then measured against this random
baseline.

Independent flips rather than an exact split: forcing exactly half the batch
into treatment makes assignments depend on one another, a correction the
variance estimates on the uplift number would then have to carry. A slightly
uneven split is cheaper than a subtly wrong standard error.
"""

from __future__ import annotations

import numpy as np

from forbear.generator.batch_generator import FailedDebit

DEFAULT_TREATMENT_RATE = 0.5


def assign_treatment(
    batch: list[FailedDebit],
    treatment_rate: float = DEFAULT_TREATMENT_RATE,
    seed: int = 0,
) -> dict[str, bool]:
    """Map customer_id -> contacted, by coin flip alone.

    treatment_rate is the probability of contact, not a quota. It must stay
    strictly inside (0, 1): at 0 or 1 one arm of the experiment is empty and
    there is no counterfactual left to compare against, which is a broken run
    rather than an extreme one.
    """
    if not 0.0 < treatment_rate < 1.0:
        raise ValueError(
            f"treatment_rate must be strictly between 0 and 1, got {treatment_rate}; "
            f"a one-armed experiment cannot identify uplift"
        )

    rng = np.random.RandomState(seed)
    draws = rng.random_sample(size=len(batch))
    return {
        record.customer_id: bool(draw < treatment_rate)
        for record, draw in zip(batch, draws)
    }
