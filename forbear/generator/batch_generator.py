"""A day's failed debits, each carrying its own answer key.

A FailedDebit holds what the system will actually see (customer, amount,
decline code, when it failed) and, separately, the counterfactual pair that no
production record could ever contain: what this customer would do if left
alone, and what they would do if contacted. Both branches are fixed here, at
generation time, before any policy or treatment assignment exists. That is what
makes the evaluation an experiment rather than a story: the outcome cannot be
influenced by the decision, because the outcome was written down first.

ground_truth is the answer key. It stays in this package. Anything under
scoring/ or services/ that could reach it would be scoring its own inputs, and
the resulting uplift number would measure nothing at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np

from forbear.generator.customer_profiles import CustomerProfile, Segment
from forbear.models.models import FailureClass, MandateStatus
from forbear.services.classifier import classify, known_codes

# Decline codes for a live mandate. Weighted hard toward INSUFFICIENT_FUNDS
# because that is what the Razorpay book actually looks like, and because it is
# the class where waiting for salary day is the whole recovery mechanism.
LIVE_MANDATE_CODE_WEIGHTS: dict[str, float] = {
    "INSUFFICIENT_FUNDS": 0.62,
    "BANK_ACCOUNT_DEBITED_ALREADY": 0.04,
    "GATEWAY_ERROR": 0.09,
    "TIMEOUT": 0.05,
    "BAD_GATEWAY": 0.03,
    "MANDATE_LIMIT_EXCEEDED": 0.06,
    "TOKEN_EXPIRED": 0.05,
    "ACCOUNT_CLOSED": 0.03,
    "CUSTOMER_DISPUTED": 0.03,
}

# Per-segment tilts on the base weights, renormalised at draw time. A customer
# who is broke is likelier to have had the account closed or to have disputed
# the charge; a customer who is merely between paycheques is not. This is the
# same reasoning as the segment-conditioned plan and tenure draws in
# customer_profiles: the decline code is observable, so it may carry signal,
# and the tilts are mild enough that no single code identifies a segment.
SEGMENT_CODE_TILTS: dict[Segment, dict[str, float]] = {
    Segment.LOST_CAUSE: {"ACCOUNT_CLOSED": 4.0, "CUSTOMER_DISPUTED": 3.0},
    Segment.SURE_THING: {"ACCOUNT_CLOSED": 0.4, "CUSTOMER_DISPUTED": 0.4},
    Segment.PERSUADABLE: {"ACCOUNT_CLOSED": 0.4, "CUSTOMER_DISPUTED": 0.4},
    Segment.DO_NOT_DISTURB: {"ACCOUNT_CLOSED": 0.4, "CUSTOMER_DISPUTED": 0.4},
}

# A dead mandate declines for the reason it is dead. Coupling these rather than
# drawing them independently keeps the batch consistent with what the guard
# would see when it re-reads the mandate before executing.
MANDATE_STATUS_CODES: dict[MandateStatus, str] = {
    MandateStatus.EXPIRED: "MANDATE_EXPIRED",
    MandateStatus.REVOKED: "MANDATE_REVOKED",
}

# Contact is assumed to go out on the day the debit failed: the allocator runs
# against the day's batch. Persuadables then pay within three days of it.
PERSUADABLE_RESPONSE_DAYS = (1, 3)

# Non-time-dependent failures have nothing to do with salary day. A rails error
# clears as soon as it is retried; re-authorisation waits on the customer.
TRANSIENT_RECOVERY_DAYS = (1, 2)
REAUTH_RECOVERY_DAYS = (2, 5)


@dataclass(frozen=True)
class FailedDebit:
    """One failed debit plus its counterfactuals.

    Everything except ground_truth is observable. ground_truth is a plain dict
    rather than a dataclass on purpose: it is data for the simulator and the
    evaluation, never something the rest of the system has a type for.
    """

    customer_id: str
    amount: int  # paise
    failure_code: str
    timestamp: datetime  # tz-aware UTC; when the debit failed
    ground_truth: dict[str, Any] = field(repr=False)


def _failure_code(
    rng: np.random.RandomState, profile: CustomerProfile, codes: list[str]
) -> str:
    """Draw a decline code, letting a dead mandate dictate its own."""
    forced = MANDATE_STATUS_CODES.get(profile.mandate_status)
    if forced is not None:
        return forced

    tilts = SEGMENT_CODE_TILTS[profile.segment]
    weights = np.array(
        [LIVE_MANDATE_CODE_WEIGHTS[code] * tilts.get(code, 1.0) for code in codes]
    )
    return str(rng.choice(codes, p=weights / weights.sum()))


def _next_salary_date(
    rng: np.random.RandomState, profile: CustomerProfile, billing_date: date
) -> date:
    """The day this customer's balance next replenishes, after the failure.

    salary_day is the scheduled credit; salary_variance_days is how far either
    side of it the money actually lands. A customer paid on the 7th with two
    days of variance replenishes somewhere between the 5th and the 9th, which
    is the timing structure a time-dependent retry has to hit.
    """
    year, month = billing_date.year, billing_date.month
    scheduled = date(year, month, profile.salary_day)
    if scheduled <= billing_date:
        # Roll to next month. salary_day is capped at 28, so this is always a
        # real date, February included.
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
        scheduled = date(year, month, profile.salary_day)

    jitter = 0
    if profile.salary_variance_days:
        jitter = int(
            rng.randint(-profile.salary_variance_days, profile.salary_variance_days + 1)
        )
    landed = scheduled + timedelta(days=jitter)

    # Jitter must never push the payment to or before the failure: the money
    # was demonstrably not there on billing_date.
    return max(landed, billing_date + timedelta(days=1))


def _days_after(
    rng: np.random.RandomState, billing_date: date, window: tuple[int, int]
) -> date:
    low, high = window
    return billing_date + timedelta(days=int(rng.randint(low, high + 1)))


def _self_heal_date(
    rng: np.random.RandomState,
    profile: CustomerProfile,
    billing_date: date,
    failure_class: FailureClass,
) -> date:
    """When a customer who was always going to pay actually pays.

    Only time-dependent failures wait on salary day; the others resolve on
    their own clocks. The segment decides whether payment happens at all, this
    decides when.
    """
    if failure_class is FailureClass.TIME_DEPENDENT:
        return _next_salary_date(rng, profile, billing_date)
    if failure_class is FailureClass.REAUTH_REQUIRED:
        return _days_after(rng, billing_date, REAUTH_RECOVERY_DAYS)
    return _days_after(rng, billing_date, TRANSIENT_RECOVERY_DAYS)


def _ground_truth(
    rng: np.random.RandomState,
    profile: CustomerProfile,
    billing_date: date,
    failure_class: FailureClass,
) -> dict[str, Any]:
    """The counterfactual pair for one record, fixed before any decision.

    The four segments differ only in which branch pays, which is the whole
    point: a model that cannot separate them will still get the marginal
    recovery rate right and the incremental one wrong.
    """
    segment = profile.segment

    without_date: Optional[date] = None
    with_date: Optional[date] = None

    if segment in (Segment.SURE_THING, Segment.DO_NOT_DISTURB):
        # Pays either way, and contact does not move the date: a phone call
        # does not put money in an account before payday.
        without_date = _self_heal_date(rng, profile, billing_date, failure_class)
        with_date = without_date
    elif segment is Segment.PERSUADABLE:
        # Nothing happens unless someone asks; then it happens quickly.
        with_date = _days_after(rng, billing_date, PERSUADABLE_RESPONSE_DAYS)

    # Only do_not_disturbs have a non-zero contact_sensitivity, so only they
    # can churn. The coin is flipped here rather than in the simulator so the
    # record's counterfactual is complete before treatment is assigned.
    would_churn = bool(rng.random_sample() < profile.contact_sensitivity)

    return {
        "segment": segment.value,
        "would_pay_without_contact": without_date is not None,
        "would_pay_without_date": without_date,
        "would_pay_with_contact": with_date is not None,
        "would_pay_with_date": with_date,
        "would_churn_if_contacted": would_churn,
    }


def generate_batch(
    profiles: list[CustomerProfile], billing_date: date, seed: int
) -> list[FailedDebit]:
    """One failed debit per profile, with both counterfactuals attached.

    A real day's batch is a subset of the book; this generates one record per
    profile because the sampling of who fails is a separate concern from what
    happens once they have. Callers who want a partial book pass fewer
    profiles.
    """
    rng = np.random.RandomState(seed)
    # Sorted so the draw order does not depend on set iteration order; the
    # classifier's table is the source of truth for which codes exist.
    codes = sorted(known_codes() & set(LIVE_MANDATE_CODE_WEIGHTS))

    batch: list[FailedDebit] = []
    for profile in profiles:
        failure_code = _failure_code(rng, profile, codes)
        failure_class = classify(failure_code)
        ground_truth = _ground_truth(rng, profile, billing_date, failure_class)

        # Razorpay's debit runs are spread across the day. The exact minute is
        # noise, but a batch where every record shared one timestamp would hide
        # any ordering bug downstream.
        moment = datetime(
            billing_date.year,
            billing_date.month,
            billing_date.day,
            int(rng.randint(0, 24)),
            int(rng.randint(0, 60)),
            tzinfo=timezone.utc,
        )

        batch.append(
            FailedDebit(
                customer_id=profile.customer_id,
                amount=profile.plan_amount,
                failure_code=failure_code,
                timestamp=moment,
                ground_truth=ground_truth,
            )
        )
    return batch


def _check_every_known_code_is_reachable() -> None:
    """Fail at import if the classifier's table and the weights drift apart.

    A code added to the classifier and not here would silently never be
    generated, and the batch would quietly stop exercising its class.
    """
    missing = (
        known_codes()
        - set(LIVE_MANDATE_CODE_WEIGHTS)
        - set(MANDATE_STATUS_CODES.values())
    )
    if missing:
        raise RuntimeError(
            f"classifier knows codes the generator never emits: {sorted(missing)}; "
            f"add them to LIVE_MANDATE_CODE_WEIGHTS"
        )


_check_every_known_code_is_reachable()
