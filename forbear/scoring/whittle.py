"""Priority under a shared attempt budget. Ranks records, spends nothing.

Every at-risk record is an arm of a restless bandit: its state moves whether or
not Forbear pulls it - salary lands, mandates expire, patience runs out - and
there are more arms worth pulling than there are attempts to spend. The Whittle
index is the standard answer to that problem. It prices each arm on its own, in
value per attempt, so a global budget can be spent by sorting rather than by
solving one enormous joint optimisation.

CLOSED-FORM APPROXIMATION, NOT THE FULL INDEX
---------------------------------------------
The true Whittle index is the subsidy for passivity that makes acting and not
acting equally attractive, and computing it means solving the per-arm Bellman
equation and checking the arm is indexable at all. What follows is a
closed-form approximation: one period of value, with the continuation folded in
as a survival-weighted lifetime term instead of a discounted value function.

This is deliberate. The full derivation is Mate, Madaan, Suggala, Taneja et al.,
"Collapsing Bandits and Their Application to Public Health Intervention"
(NeurIPS 2020), which also gives the threshold-optimality conditions this
approximation quietly assumes. On synthetic data a full Bellman solver would
not be more honest than this - it would inherit its transition matrix from the
same generator - and it would be considerably harder to defend line by line. An
auditable approximation whose failure modes are visible beats an exact solution
to a model nobody can check.

Nothing here decides anything. The allocator reads these numbers, the guard
re-validates whatever the allocator proposes, and this module never learns what
either of them did.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# How far ahead a retained customer is worth counting. Twelve months is a
# horizon, not a prediction: past a year the estimate is dominated by churn the
# model has no view of, and a longer horizon would inflate every index equally
# while making the negative ones look worse than the evidence supports.
DEFAULT_REMAINING_MONTHS = 12

# Probability a recovered customer is still there afterwards. Recovery is not
# retention: some customers pay the invoice and leave anyway.
DEFAULT_RECOVERY_SURVIVAL_RATE = 0.85

# What one attempt costs, all in - the debit fee, the message, the share of
# support load it generates. The index is denominated in units of this, so it
# reads directly as "value returned per attempt spent" and a record scoring
# below 1.0 is not paying for itself.
DEFAULT_ATTEMPT_COST_PAISE = 200  # 2.00 rupees


@dataclass
class RecordScore:
    """One record on its way from the scorer to the allocator.

    Mutable, unlike most of the codebase's data objects: compute_indices fills
    whittle_index in place, and a frozen version would mean rebuilding the list
    to attach a number that belongs to the record it came from.
    """

    record_id: str
    amount: int  # paise: the invoice actually at risk
    plan_amount: int  # paise per month: the recurring value behind it
    cate: float
    whittle_index: Optional[float] = None


def compute_whittle_index(
    record: RecordScore,
    remaining_months: int = DEFAULT_REMAINING_MONTHS,
    recovery_survival_rate: float = DEFAULT_RECOVERY_SURVIVAL_RATE,
    attempt_cost_paise: int = DEFAULT_ATTEMPT_COST_PAISE,
) -> float:
    """Value per attempt spent on this record. Negative means do not touch it.

    Three terms, all scaled by the estimated effect of contact:

      cate * amount
          the invoice itself, weighted by how much contact changes whether it
          is paid at all.

      cate * remaining_ltv * recovery_survival_rate
          the subscription that continues behind the invoice. A recovered
          customer is worth more than the one payment, discounted by the share
          who pay and leave regardless.

      abs(min(cate, 0)) * remaining_ltv
          the churn cost. It engages only when the effect is negative - the
          do_not_disturb case, where the contact itself is what ends the
          relationship, and the whole subscription is at stake rather than one
          invoice.

    Note the asymmetry: when the CATE is negative the middle term is negative
    too, so lifetime value is charged twice over. That is not a compensating
    trick, it is the formula as specified, and it errs toward refusing to chase
    customers the model believes contact would drive away. The direction of that
    error is the one worth having: a record already scoring negative is one the
    allocator must not spend on regardless of how negative it is, so the extra
    severity changes no decision - only the ordering among records that are all
    being skipped anyway.
    """
    if remaining_months < 0:
        raise ValueError(
            f"remaining_months must be non-negative, got {remaining_months}"
        )
    if not 0.0 <= recovery_survival_rate <= 1.0:
        raise ValueError(
            f"recovery_survival_rate is a probability, got {recovery_survival_rate}"
        )
    if attempt_cost_paise <= 0:
        raise ValueError(
            f"attempt_cost_paise must be positive, got {attempt_cost_paise}; "
            f"a free attempt would make every index infinite"
        )

    remaining_ltv = record.plan_amount * remaining_months

    invoice_value = record.cate * record.amount
    retained_value = record.cate * remaining_ltv * recovery_survival_rate
    churn_cost = abs(min(record.cate, 0.0)) * remaining_ltv

    net_value = invoice_value + retained_value - churn_cost
    return net_value / attempt_cost_paise


def compute_indices(
    records_with_cate: list[RecordScore],
    remaining_months: int = DEFAULT_REMAINING_MONTHS,
    recovery_survival_rate: float = DEFAULT_RECOVERY_SURVIVAL_RATE,
    attempt_cost_paise: int = DEFAULT_ATTEMPT_COST_PAISE,
) -> list[RecordScore]:
    """Attach an index to every record and return them highest-value first.

    Sorted descending because that is the order the budget is spent in: take
    from the top until the attempts run out. Records are not filtered here.
    Dropping the negative ones would be a decision, and this module does not
    make decisions - it is the allocator that must never spend an attempt on a
    negative index, and the guard that re-checks it did not.
    """
    for record in records_with_cate:
        record.whittle_index = compute_whittle_index(
            record,
            remaining_months=remaining_months,
            recovery_survival_rate=recovery_survival_rate,
            attempt_cost_paise=attempt_cost_paise,
        )

    # Ties broken by record_id so the order is total rather than
    # implementation-defined: a batch that reorders itself between runs makes
    # every downstream comparison unreproducible for no visible reason.
    return sorted(
        records_with_cate,
        key=lambda record: (-record.whittle_index, record.record_id),
    )
