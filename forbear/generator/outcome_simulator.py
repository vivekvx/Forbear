"""Collapse each record's counterfactual pair into the one observed outcome.

For every record the batch holds two futures. Treatment assignment picks which
one happened; this module writes that one down and throws the other away. What
survives is exactly what a production database could contain, which is what the
model is allowed to train on.

Then it adds noise, and the noise is not decoration. Without it the segments
separate perfectly: every sure_thing recovers, no lost_cause ever does, and any
classifier reaches an AUC that would be a red flag in a real book. Perfect
separation here would mean the dataset was proving the model can recover the
rules it was built from - circular, and worse than useless because it looks
like success. Five percent of sure_things fail anyway because life happens, and
three percent of lost_causes pay because a relative stepped in. Both rates are
independent of treatment, so they blur the segments without biasing the uplift
estimate in either direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np

from forbear.generator.batch_generator import FailedDebit
from forbear.generator.customer_profiles import Segment

# Irreducible noise. Raising these makes the problem harder; setting either to
# zero makes the dataset a memorisation test.
SURE_THING_FAILURE_RATE = 0.05
LOST_CAUSE_RECOVERY_RATE = 0.03

# When a lost cause pays anyway it is tied to neither salary day nor contact -
# that is the point of the case. Somewhere in the week after the failure.
WINDFALL_DAYS = (1, 7)


@dataclass(frozen=True)
class Outcome:
    """What actually happened. No counterfactuals, no segment.

    This is the shape the rest of the system could legitimately see. Anything
    needing to know why an outcome happened is asking for the answer key.
    """

    customer_id: str
    treated: bool
    recovered: bool
    recovered_date: Optional[date]
    churned: bool


def simulate_outcomes(
    batch: list[FailedDebit],
    treatment_assignments: dict[str, bool],
    seed: int = 0,
) -> dict[str, Outcome]:
    """Observed outcome per record, given ground truth and who was contacted.

    seed exists because the noise is random, and a run whose noise moves
    between invocations cannot be compared against itself. It defaults so that
    callers who only care about the causal structure need not think about it.

    Every record in the batch must have an assignment. A missing one means the
    batch and the assignments came from different runs, and silently treating
    it as control would drop an unassigned record into the comparison group.
    """
    rng = np.random.RandomState(seed)

    outcomes: dict[str, Outcome] = {}
    for record in batch:
        if record.customer_id not in treatment_assignments:
            raise KeyError(
                f"no treatment assignment for {record.customer_id}; batch and "
                f"assignments do not come from the same run"
            )
        treated = treatment_assignments[record.customer_id]
        truth = record.ground_truth
        segment = Segment(truth["segment"])

        if treated:
            recovered = truth["would_pay_with_contact"]
            recovered_date = truth["would_pay_with_date"]
        else:
            recovered = truth["would_pay_without_contact"]
            recovered_date = truth["would_pay_without_date"]

        # Churn only happens to someone who was actually contacted. A
        # do_not_disturb left alone is simply a customer who paid, which is why
        # blanket contact is expensive in a way recovery rate cannot show.
        churned = bool(treated and truth["would_churn_if_contacted"])

        # Noise is drawn for every record, treated or not, so the noise itself
        # carries no treatment signal. Drawn unconditionally rather than inside
        # the branches: otherwise the number of draws would depend on the
        # segment and the stream would desynchronise between runs.
        sure_thing_fails = rng.random_sample() < SURE_THING_FAILURE_RATE
        lost_cause_pays = rng.random_sample() < LOST_CAUSE_RECOVERY_RATE
        windfall_day = int(rng.randint(WINDFALL_DAYS[0], WINDFALL_DAYS[1] + 1))

        if segment is Segment.SURE_THING and sure_thing_fails:
            recovered = False
            recovered_date = None
        elif segment is Segment.LOST_CAUSE and lost_cause_pays:
            recovered = True
            recovered_date = record.timestamp.date() + timedelta(days=windfall_day)

        outcomes[record.customer_id] = Outcome(
            customer_id=record.customer_id,
            treated=treated,
            recovered=bool(recovered),
            recovered_date=recovered_date if recovered else None,
            churned=churned,
        )

    return outcomes
