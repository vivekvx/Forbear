"""Razorpay's fixed retry schedule, reimplemented as a comparable policy.

T+1, T+2, T+3. Three debits per failed invoice, twenty-four hours apart,
regardless of why the debit failed, how much it was for, whether the customer
has ever paid at that hour, or whether contacting them is about to cost more
than the invoice is worth. Then it stops forever and the invoice is never
auto-charged again.

This is not a straw man. It is what the platform actually does, and it is a
perfectly reasonable default for a system with no view of the customer: with no
scores, no history and no way to tell a persuadable from a lost cause, a fixed
schedule spends its attempts uniformly because it has nothing better to spend
them on. The point of reproducing it here is that Forbear's numbers mean
nothing in isolation. "Recovered 34% of failed invoices" is only interesting
next to what this policy recovers on the same book, and the difference has to
be measured rather than asserted.

Same AllocationPlan shape as the real allocator, deliberately: the measurement
harness reads both through one code path, so a difference in the comparison can
never turn out to be a difference in the plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Optional, Sequence

from forbear.core.audit import append_entry
from forbear.core.state_machine import ENTITY_TYPE, transition
from forbear.models.models import ActionKind, FailureClass, RecordStatus
from forbear.services.allocator import (
    SKIP_TERMINAL_FAILURE_CLASS,
    SKIP_UNCLASSIFIED_FAILURE_CODE,
    AllocationPlan,
    ScheduledAction,
    SkippedRecord,
)

# The published schedule: one day, two days, three days after the failure.
FIXED_OFFSETS = (timedelta(hours=24), timedelta(hours=48), timedelta(hours=72))

ACTION_SCHEDULE = "baseline:schedule"
ACTION_SKIP = "baseline:skip"

# Even the fixed schedule cannot debit against a terminal failure - the rails
# refuse it, so it is not a policy choice. reauth_required is left in on
# purpose: Razorpay retries those too, and dropping them here would quietly
# improve the baseline past what it really does.
UNCHARGEABLE_CLASSES = frozenset({FailureClass.TERMINAL})


@dataclass(frozen=True)
class BaselineConfig:
    """There is nothing to configure. The class exists so both policies take
    the same arguments and the harness needs no special case."""

    offsets: Sequence[timedelta] = FIXED_OFFSETS


async def allocate_fixed_schedule(
    conn,
    records: Sequence[int],
    config: Optional[BaselineConfig] = None,
) -> AllocationPlan:
    """Schedule every chargeable record three times. No scoring, no ranking.

    Takes bare record ids rather than scored records, which is the honest
    signature: this policy has no use for a score, and accepting one would
    suggest it did.

    The three attempts hang off the failure time, not off now(), because T+1
    means one day after the invoice failed. A record picked up late is already
    past its first slot, and the baseline's real behaviour - firing those
    attempts anyway - is exactly what makes it lose to a policy that waits for
    payday.
    """
    if not conn.is_in_transaction():
        raise RuntimeError("allocate_fixed_schedule must run inside a transaction")

    config = config or BaselineConfig()
    plan = AllocationPlan()
    scheduled_amount = 0

    record_ids = [int(record) for record in records]
    if not record_ids:
        plan.summary = _summarise(plan, considered=0, scheduled_amount=0)
        return plan

    rows = await conn.fetch(
        """
        SELECT r.id, r.amount, r.failure_class, r.created_at, s.mandate_status
        FROM at_risk_records r
        JOIN subscriptions s ON s.id = r.subscription_id
        WHERE r.id = ANY($1::bigint[])
        ORDER BY r.id
        """,
        record_ids,
    )

    found = {row["id"] for row in rows}
    missing = sorted(set(record_ids) - found)
    if missing:
        raise LookupError(f"records that do not exist: {missing}")

    for row in rows:
        record_id = row["id"]
        failure_class = (
            FailureClass(row["failure_class"])
            if row["failure_class"] is not None
            else None
        )

        # An unclassified code has no mapping, so nothing downstream may treat
        # it as recoverable - the same rule the real allocator follows, for the
        # same reason. It is the only skip here that is not about the rails.
        if failure_class is None:
            await _skip(
                conn,
                plan,
                record_id,
                SKIP_UNCLASSIFIED_FAILURE_CODE,
                {
                    "amount": row["amount"],
                    "detail": "no classifier mapping; on the exception list",
                },
            )
            continue

        if failure_class in UNCHARGEABLE_CLASSES:
            await _skip(
                conn,
                plan,
                record_id,
                SKIP_TERMINAL_FAILURE_CLASS,
                {
                    "failure_class": failure_class.value,
                    "amount": row["amount"],
                    "detail": "the rails refuse this class; not a policy choice",
                },
            )
            continue

        failed_at = row["created_at"]
        slots = [failed_at + offset for offset in config.offsets]

        await transition(conn, record_id, RecordStatus.SCHEDULED)
        await append_entry(
            conn,
            ENTITY_TYPE,
            record_id,
            ACTION_SCHEDULE,
            {
                "policy": "fixed_t1_t2_t3",
                "amount": row["amount"],
                "failure_class": failure_class.value,
                "scheduled_at": [slot.isoformat() for slot in slots],
                "detail": "fixed schedule; no scoring, no ranking, no budget",
            },
        )

        for slot in slots:
            plan.scheduled.append(ScheduledAction(record_id, ActionKind.CHARGE, slot))
        scheduled_amount += row["amount"]

    plan.summary = _summarise(
        plan, considered=len(record_ids), scheduled_amount=scheduled_amount
    )
    return plan


async def _skip(
    conn,
    plan: AllocationPlan,
    record_id: int,
    reason: str,
    details: dict[str, Any],
) -> None:
    await transition(conn, record_id, RecordStatus.SKIPPED, reason=reason)
    await append_entry(
        conn, ENTITY_TYPE, record_id, ACTION_SKIP, {"skip_reason": reason, **details}
    )
    plan.skipped.append(SkippedRecord(record_id, reason, details))


def _summarise(
    plan: AllocationPlan, considered: int, scheduled_amount: int
) -> dict[str, Any]:
    """The same shape the allocator reports, so the two can be diffed directly.

    attempts_scheduled is the number worth reading next to Forbear's: the
    baseline spends three attempts on every record it touches, including the
    ones where all three were always going to fail.
    """
    skips_by_reason: dict[str, int] = {}
    skipped_amount = 0
    for skip in plan.skipped:
        skips_by_reason[skip.skip_reason] = skips_by_reason.get(skip.skip_reason, 0) + 1
        skipped_amount += int(skip.details.get("amount", 0))

    scheduled_records = {action.record_id for action in plan.scheduled}
    return {
        "considered": considered,
        "scheduled_count": len(scheduled_records),
        "attempts_scheduled": len(plan.scheduled),
        "skipped_count": len(plan.skipped),
        "scheduled_amount": scheduled_amount,
        "skipped_amount": skipped_amount,
        "skips_by_reason": skips_by_reason,
        "batch_budget": None,
        "policy": "fixed_t1_t2_t3",
    }
