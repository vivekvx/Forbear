"""The plan Forbear would make if no rule applied. Never executed.

Strip out the NPCI debit windows, the attempt cap, and the pre-debit
notification requirement, and chase everything the rails would physically
accept, immediately. The result is not a policy - running it would breach a
regulatory cap on the first cycle - it is a measuring stick. Subtract what the
real allocator recovers from what this one could, and the difference is the
cost of compliance: the recovery the constraints actually cost, stated as a
number instead of assumed to be either nothing or ruinous.

That number is worth having in both directions. If it is small, the compliance
argument against automated recovery is weaker than people assume. If it is
large, the merchant is entitled to know what the rules cost them, and to hear
it from their own system rather than from a vendor.

Three properties keep this honest:

  * It writes nothing. No status transitions, no audit entries, no attempts.
    A module that cannot leave a trace cannot be quietly promoted into the
    execution path by someone in a hurry.
  * It relaxes only the three constraints named above. A revoked mandate is
    still skipped, because that is the rails refusing, not a regulation.
  * It ignores the net-value filter too. The objective here is raw recovery
    probability, which is exactly the objective that counts a do_not_disturb's
    cancellation as a win. Keeping that visible is the point: the gap between
    this plan and Forbear's is part compliance and part deliberate restraint,
    and the summary reports them separately so the two are never conflated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from forbear.core.audit import server_now
from forbear.models.models import ActionKind, FailureClass, MandateStatus
from forbear.services.allocator import (
    SKIP_MANDATE_STATE_INVALID,
    SKIP_TERMINAL_FAILURE_CLASS,
    SKIP_UNCLASSIFIED_FAILURE_CODE,
    AllocationPlan,
    ScheduledAction,
    ScoredRecord,
    SkippedRecord,
    ltv_at_risk,
)

ACTION_SCHEDULE = "unconstrained:schedule"

# The same two rails-level refusals the real allocator applies. Everything else
# it applies - windows, caps, notifications, net value - is dropped.
UNCHARGEABLE_CLASSES = frozenset({FailureClass.TERMINAL, FailureClass.REAUTH_REQUIRED})
DEAD_MANDATE_STATES = frozenset({MandateStatus.REVOKED, MandateStatus.EXPIRED})


@dataclass(frozen=True)
class UnconstrainedConfig:
    """Nothing to tune. Present so the harness can call all three policies the
    same way."""


async def allocate_unconstrained(
    conn,
    records_with_scores: Sequence[ScoredRecord],
    config: Optional[UnconstrainedConfig] = None,
) -> AllocationPlan:
    """Schedule everything chargeable, right now, and report what that implies.

    Every scheduled action carries the same instant, which is the honest
    representation: with no windows to respect and no notice to give, the best
    time to attempt a debit is immediately, and the fact that this is obviously
    unimplementable is what makes it an upper bound rather than a proposal.

    Deliberately has no in-transaction requirement - there is nothing to
    commit. If this function ever needs one, something has started writing, and
    that is the bug.
    """
    plan = AllocationPlan()
    scheduled_amount = 0
    negative_value_scheduled = 0
    negative_value_amount = 0

    scores = {scored.record_id: scored for scored in records_with_scores}
    if not scores:
        plan.summary = _summarise(plan, 0, 0, 0, 0)
        return plan

    rows = await conn.fetch(
        """
        SELECT r.id, r.amount, r.failure_class, s.plan_amount, s.mandate_status
        FROM at_risk_records r
        JOIN subscriptions s ON s.id = r.subscription_id
        WHERE r.id = ANY($1::bigint[])
        ORDER BY r.id
        """,
        list(scores),
    )

    missing = sorted(set(scores) - {row["id"] for row in rows})
    if missing:
        raise LookupError(f"scored records that do not exist: {missing}")

    now = await server_now(conn)

    for row in rows:
        scored = scores[row["id"]]
        failure_class = (
            FailureClass(row["failure_class"])
            if row["failure_class"] is not None
            else None
        )
        mandate_status = MandateStatus(row["mandate_status"])

        if failure_class is None:
            plan.skipped.append(
                SkippedRecord(
                    row["id"],
                    SKIP_UNCLASSIFIED_FAILURE_CODE,
                    {"amount": row["amount"], "detail": "no classifier mapping"},
                )
            )
            continue

        if failure_class in UNCHARGEABLE_CLASSES:
            plan.skipped.append(
                SkippedRecord(
                    row["id"],
                    SKIP_TERMINAL_FAILURE_CLASS,
                    {
                        "failure_class": failure_class.value,
                        "amount": row["amount"],
                        "detail": "the rails refuse this; not a constraint we chose",
                    },
                )
            )
            continue

        if mandate_status in DEAD_MANDATE_STATES:
            plan.skipped.append(
                SkippedRecord(
                    row["id"],
                    SKIP_MANDATE_STATE_INVALID,
                    {
                        "mandate_status": mandate_status.value,
                        "amount": row["amount"],
                        "detail": "no mandate to debit against",
                    },
                )
            )
            continue

        plan.scheduled.append(ScheduledAction(row["id"], ActionKind.CHARGE, now))
        scheduled_amount += row["amount"]

        # Counted, not skipped. These are the records the real allocator
        # refuses on value grounds - contacting them is estimated to destroy
        # more than it recovers - and the whole difference between the two
        # plans would read as "compliance cost" if this went unreported.
        if scored.whittle_index < 0:
            negative_value_scheduled += 1
            negative_value_amount += ltv_at_risk(row["plan_amount"])

    plan.summary = _summarise(
        plan,
        considered=len(scores),
        scheduled_amount=scheduled_amount,
        negative_value_scheduled=negative_value_scheduled,
        negative_value_ltv_at_risk=negative_value_amount,
    )
    return plan


def _summarise(
    plan: AllocationPlan,
    considered: int,
    scheduled_amount: int,
    negative_value_scheduled: int,
    negative_value_ltv_at_risk: int,
) -> dict[str, Any]:
    """Allocator-shaped, plus the two numbers only this policy can report.

    negative_value_scheduled is how many customers this plan would chase that
    Forbear deliberately leaves alone, and negative_value_ltv_at_risk is the
    subscription value it puts on the table to do it. Anyone comparing recovery
    rates between the two policies has to net that off first, or they are
    reading a win that was paid for in churn.
    """
    skips_by_reason: dict[str, int] = {}
    skipped_amount = 0
    for skip in plan.skipped:
        skips_by_reason[skip.skip_reason] = skips_by_reason.get(skip.skip_reason, 0) + 1
        skipped_amount += int(skip.details.get("amount", 0))

    return {
        "considered": considered,
        "scheduled_count": len(plan.scheduled),
        "skipped_count": len(plan.skipped),
        "scheduled_amount": scheduled_amount,
        "skipped_amount": skipped_amount,
        "skips_by_reason": skips_by_reason,
        "batch_budget": None,
        "policy": "unconstrained_upper_bound",
        "constraints_ignored": [
            "npci_debit_windows",
            "attempt_cap",
            "pre_debit_notification",
            "net_value_threshold",
        ],
        "negative_value_scheduled": negative_value_scheduled,
        "negative_value_ltv_at_risk": negative_value_ltv_at_risk,
        "executable": False,
    }
